#!/usr/bin/env python3
"""Capture a frame using vision.py, detect all blocks, and save the
annotated image plus individual color masks to the photos/ directory."""

import os
import sys
from datetime import datetime

import cv2

# Allow imports from the parent package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fp.vision import (
    init_camera,
    release_camera,
    capture_frame,
    COLOR_MASKS,
    _build_mask,
    DetectionResult,
    _best_contour,
)

CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(CALIBRATION_DIR, "photos")


def main():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Initialising camera ...")
    init_camera()

    print("Capturing frame ...")
    frame = capture_frame()

    # Run detection on the captured frame.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    annotated = frame.copy()

    for label, ranges in COLOR_MASKS.items():
        mask = _build_mask(hsv, ranges)
        result = _best_contour(mask, label)

        # Save the color mask.
        mask_path = os.path.join(PHOTOS_DIR, f"mask_{label}_{ts}.jpg")
        cv2.imwrite(mask_path, mask)
        print(f"  Saved mask: {mask_path}")

        if result is None:
            print(f"  {label}: not detected")
            continue

        contour, area, cx, cy = result
        det = DetectionResult(label=label, cx=cx, cy=cy, area=area)
        print(f"  {label}: cx={cx}  cy={cy}  area={area:.0f}"
              f"  within_reach={det.within_reach}"
              f"  too_close={det.block_too_close}")

        # Draw contour and centroid on the annotated image.
        color_bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
        cv2.drawContours(annotated, [contour], -1, color_bgr.get(label, (255, 255, 255)), 2)
        cv2.circle(annotated, (cx, cy), 6, color_bgr.get(label, (255, 255, 255)), -1)
        cv2.putText(annotated, f"{label} {area:.0f}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr.get(label, (255, 255, 255)), 2)

    # Save raw and annotated frames.
    raw_path = os.path.join(PHOTOS_DIR, f"raw_{ts}.jpg")
    ann_path = os.path.join(PHOTOS_DIR, f"detections_{ts}.jpg")
    cv2.imwrite(raw_path, frame)
    cv2.imwrite(ann_path, annotated)
    print(f"  Saved raw frame:  {raw_path}")
    print(f"  Saved detections: {ann_path}")

    release_camera()
    print("Done.")


if __name__ == "__main__":
    main()
