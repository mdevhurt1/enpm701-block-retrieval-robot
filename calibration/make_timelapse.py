#!/usr/bin/env python3
"""Build a timelapse video from time_lapse/ frames with block detections overlaid.

Reads every saved frame, runs HSV detection for all colors, annotates
contours/centroids/labels, and writes the result to calibration/photos/timelapse.mp4.
"""

import os
import sys
import glob

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from fp.vision import COLOR_MASKS, _build_mask, _best_contour, DetectionResult

CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(CALIBRATION_DIR, "photos")
TIMELAPSE_DIR = os.path.join(os.path.dirname(CALIBRATION_DIR), "time_lapse")

FPS = 10

COLOR_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}


def annotate(frame):
    """Run detection on a frame and draw results. Returns annotated copy."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = frame.copy()

    for label, ranges in COLOR_MASKS.items():
        mask = _build_mask(hsv, ranges)
        result = _best_contour(mask, label)
        if result is None:
            continue

        contour, area, cx, cy = result
        c = COLOR_BGR.get(label, (255, 255, 255))
        cv2.drawContours(out, [contour], -1, c, 2)
        cv2.circle(out, (cx, cy), 6, c, -1)
        cv2.putText(out, f"{label} {area:.0f}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    return out


def main():
    os.makedirs(PHOTOS_DIR, exist_ok=True)

    images = sorted(glob.glob(os.path.join(TIMELAPSE_DIR, "*.jpg")))
    if not images:
        print(f"No images found in {TIMELAPSE_DIR}")
        return

    print(f"Found {len(images)} frames in {TIMELAPSE_DIR}")

    # Read the first frame to get dimensions for the video writer.
    sample = cv2.imread(images[0])
    h, w = sample.shape[:2]

    out_path = os.path.join(PHOTOS_DIR, "timelapse.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))

    for i, path in enumerate(images):
        frame = cv2.imread(path)
        if frame is None:
            continue
        annotated = annotate(frame)
        writer.write(annotated)
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  Processed {i + 1}/{len(images)}")

    writer.release()
    print(f"Saved timelapse to {out_path}  ({len(images)} frames, {FPS} fps)")


if __name__ == "__main__":
    main()
