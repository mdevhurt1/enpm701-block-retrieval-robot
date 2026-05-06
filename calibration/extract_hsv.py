# Final Project Calibration - Polygon HSV Extractor
# Interactive tool: draw polygon masks over block color regions in
# calibration/photos/auto_*.jpg to sample clean interior HSV pixels.
# Pools samples across all photos and prints a ready-to-paste COLOR_MASKS
# block for fp/vision.py. Polygon sets are cached as .polygons.json files.
# Controls: L-click=add  Enter=close  Backspace=undo  R=reset  S=skip  Q=quit
#
# Requirements: opencv-python, numpy
# Usage: python calibration/extract_hsv.py

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


COLORS_IN_ORDER = ["red", "blue", "green"]

HUE_PAD = 4
SAT_PAD = 10
VAL_PAD = 15
WRAP_GAP_THRESHOLD = 30  # red is bimodal if largest hue gap exceeds this

DISPLAY_MAX_WIDTH = 1400  # downscale large Pi photos to fit on a laptop screen

WINDOW_NAME = "extract_hsv"

# Render the on-screen polygon in each color's natural BGR.
COLOR_BGR = {
    "red":   (0,   0,   255),
    "blue":  (255, 0,   0),
    "green": (0,   255, 0),
}


def polygon_path(img_path: str) -> str:
    return img_path + ".polygons.json"


def load_polygons(img_path: str) -> Optional[Dict[str, List[Tuple[int, int]]]]:
    p = polygon_path(img_path)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        raw = json.load(f)
    return {k: [tuple(pt) for pt in v] for k, v in raw.items()}


def save_polygons(img_path: str, polygons: Dict[str, List[Tuple[int, int]]]) -> None:
    with open(polygon_path(img_path), "w") as f:
        json.dump({k: [list(pt) for pt in v] for k, v in polygons.items()},
                  f, indent=2)


def draw_polygon_ui(bgr, color_name: str
                    ) -> Tuple[str, List[Tuple[int, int]]]:
    """Open a window and let the user draw one polygon for `color_name`.

    Returns ('commit'|'skip'|'quit', points-in-original-image-coords).
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, DISPLAY_MAX_WIDTH / w)
    if scale != 1.0:
        disp_base = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    else:
        disp_base = bgr.copy()

    points: List[Tuple[int, int]] = []  # original-image coords
    cursor = [0, 0]                     # display-image coords
    state = {"action": None}            # filled by mouse/keyboard

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor[0], cursor[1] = x, y

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    bgr_overlay = COLOR_BGR[color_name]

    while state["action"] is None:
        canvas = disp_base.copy()

        if points:
            disp_pts = [(int(px * scale), int(py * scale)) for (px, py) in points]
            for p in disp_pts:
                cv2.circle(canvas, p, 4, bgr_overlay, -1)
            for i in range(1, len(disp_pts)):
                cv2.line(canvas, disp_pts[i - 1], disp_pts[i], bgr_overlay, 2)
            cv2.line(canvas, disp_pts[-1], (cursor[0], cursor[1]), bgr_overlay, 1)

        prompt = (f"{color_name.upper()}  "
                  "L-click=add  Enter=close  Backspace=undo  "
                  "R=reset  S=skip  Q=quit")
        cv2.putText(canvas, prompt, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):              # Enter
            if len(points) >= 3:
                state["action"] = "commit"
        elif key in (8, 127):            # Backspace / Delete
            if points:
                points.pop()
        elif key in (ord('r'), ord('R')):
            points.clear()
        elif key in (ord('s'), ord('S')):
            state["action"] = "skip"
        elif key in (ord('q'), ord('Q'), 27):  # Q or Esc
            state["action"] = "quit"

    return state["action"], points


def sample_polygon_pixels(hsv, polygon: List[Tuple[int, int]]):
    """Return HSV pixels inside the polygon, eroded to skip edge mixing."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel)
    return hsv[mask == 255]


def _hue_bounds(h: np.ndarray, name: str
                ) -> List[Tuple[int, int]]:
    """Return one or two (lo, hi) hue intervals covering the sample.

    For non-red colors, single contiguous interval [min - pad, max + pad].
    For red, detect bimodality (a real 0/180 wrap) by finding the largest
    gap between consecutive sorted hue values; if it exceeds the threshold,
    split into two intervals at the gap.
    """
    h_sorted = np.sort(h)
    h_min = int(h_sorted[0])
    h_max = int(h_sorted[-1])

    if name == "red" and len(h_sorted) > 1:
        diffs = np.diff(h_sorted)
        gap_idx = int(np.argmax(diffs))
        gap = int(diffs[gap_idx])
        if gap > WRAP_GAP_THRESHOLD:
            # Bimodal: cluster A = h_sorted[:gap_idx+1] (low hues),
            #          cluster B = h_sorted[gap_idx+1:] (high hues).
            low_max = int(h_sorted[gap_idx])
            high_min = int(h_sorted[gap_idx + 1])
            return [
                (max(high_min - HUE_PAD, 0), 179),
                (0, min(low_max + HUE_PAD, 179)),
            ]

    return [(max(h_min - HUE_PAD, 0), min(h_max + HUE_PAD, 179))]


def compute_range(name: str, pixels: np.ndarray
                  ) -> List[Tuple[int, int, int, int, int, int]]:
    h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    s_min, s_max = int(np.min(s)), int(np.max(s))
    v_min, v_max = int(np.min(v)), int(np.max(v))
    h_min, h_max = int(np.min(h)), int(np.max(h))

    s_lo = max(s_min - SAT_PAD, 0)
    s_hi = min(s_max + SAT_PAD, 255)
    v_lo = max(v_min - VAL_PAD, 0)
    v_hi = min(v_max + VAL_PAD, 255)

    hue_intervals = _hue_bounds(h, name)

    print(f"  {name}:  H [{h_min:>3},{h_max:>3}]  "
          f"S [{s_min:>3},{s_max:>3}]  V [{v_min:>3},{v_max:>3}]  "
          f"-> H {hue_intervals} S [{s_lo},{s_hi}] V [{v_lo},{v_hi}]")

    return [(h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)
            for (h_lo, h_hi) in hue_intervals]


def save_previews(ranges_per_color, sample_bgr, sample_hsv, out_dir):
    for name, ranges in ranges_per_color.items():
        m = np.zeros(sample_hsv.shape[:2], dtype=np.uint8)
        for (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi) in ranges:
            lo = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
            hi = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
            m = cv2.bitwise_or(m, cv2.inRange(sample_hsv, lo, hi))
        out = cv2.bitwise_and(sample_bgr, sample_bgr, mask=m)
        path = os.path.join(out_dir, f"mask_{name}.jpg")
        cv2.imwrite(path, out)
        print(f"  saved {path}")


def mask_unmasked_photos(img_paths: List[str]) -> bool:
    """Open the polygon UI for any photo missing a cached polygon set.

    Returns True if the user pressed Q to quit early.
    """
    for img_path in img_paths:
        if load_polygons(img_path) is not None:
            print(f"[skip] {os.path.basename(img_path)}: cached polygons")
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"WARNING: cannot read {img_path}")
            continue

        print(f"[mask] {os.path.basename(img_path)}")
        polygons: Dict[str, List[Tuple[int, int]]] = {}
        quit_early = False
        for color in COLORS_IN_ORDER:
            action, points = draw_polygon_ui(bgr, color)
            if action == "quit":
                quit_early = True
                break
            if action == "commit":
                polygons[color] = points

        if polygons:
            save_polygons(img_path, polygons)
            print(f"  saved {polygon_path(img_path)} "
                  f"({', '.join(polygons.keys())})")

        if quit_early:
            return True

    return False


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(script_dir, "photos")
    img_paths = sorted(glob.glob(os.path.join(photos_dir, "auto_*.jpg")))

    if not img_paths:
        print(f"ERROR: no auto_*.jpg in {photos_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(img_paths)} photo(s) in {photos_dir}")

    quit_early = mask_unmasked_photos(img_paths)
    cv2.destroyAllWindows()

    if quit_early:
        print("Quit during masking; cached polygons preserved. "
              "Re-run to continue.")

    # Pool all polygon pixels across every masked photo.
    pooled: Dict[str, List[np.ndarray]] = {n: [] for n in COLORS_IN_ORDER}
    last_bgr, last_hsv = None, None
    masked_count = 0

    for img_path in img_paths:
        polygons = load_polygons(img_path)
        if not polygons:
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        last_bgr, last_hsv = bgr, hsv
        masked_count += 1
        for color, poly in polygons.items():
            if color not in COLORS_IN_ORDER:
                continue
            pixels = sample_polygon_pixels(hsv, poly)
            if pixels.size:
                pooled[color].append(pixels)

    if masked_count == 0:
        print("ERROR: no masked photos available; nothing to compute.",
              file=sys.stderr)
        return 1

    print(f"\nPooling pixels from {masked_count} masked photo(s):")
    ranges_per_color = {}
    for name in COLORS_IN_ORDER:
        if not pooled[name]:
            print(f"  {name}: <no polygons across any photo>")
            continue
        all_px = np.concatenate(pooled[name], axis=0)
        ranges_per_color[name] = compute_range(name, all_px)

    if not ranges_per_color:
        print("ERROR: no usable polygons found.", file=sys.stderr)
        return 1

    print("\nCOLOR_MASKS = {")
    for name in COLORS_IN_ORDER:
        if name in ranges_per_color:
            tuples = ", ".join(
                f"({a}, {b}, {c}, {d}, {e}, {f})"
                for (a, b, c, d, e, f) in ranges_per_color[name]
            )
            print(f'    "{name}": [{tuples}],')
    print("}")

    if last_bgr is not None:
        print("\nMask previews (rendered using the last processed photo):")
        save_previews(ranges_per_color, last_bgr, last_hsv, photos_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
