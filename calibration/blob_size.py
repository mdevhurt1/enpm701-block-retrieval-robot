# Final Project Calibration - Live Blob Area Probe
# Continuously captures frames and prints the detected blob area and centroid
# for every color tracked by fp/vision.py. Press Ctrl+C to stop.
# Applies the same HSV masks and MIN_BLOB_AREA filters as detect_all_colors().
#
# Requirements: fp.vision (Picamera2, libcamera, opencv-python)
# Usage: python calibration/blob_size.py

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_fp_parent = os.path.dirname(_here)
if _fp_parent not in sys.path:
    sys.path.insert(0, _fp_parent)

from fp.vision import init_camera, release_camera, detect_all_colors, COLOR_MASKS, MIN_BLOB_AREA

_POLL_INTERVAL_S = 0.2


def main():
    print("Initialising camera...")
    init_camera()
    print(f"Ready. Tracking: {', '.join(COLOR_MASKS)}")
    formatted = {k: f"{v:,}" for k, v in MIN_BLOB_AREA.items()}
    print(f"MIN_BLOB_AREA: {formatted}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            detections = detect_all_colors()
            parts = []
            for color, det in detections.items():
                if det is None:
                    parts.append(f"{color}: none")
                else:
                    parts.append(f"{color}: area={det.area:,.0f} cx={det.cx} cy={det.cy}")
            print("\r\033[K  " + "  |  ".join(parts), end="", flush=True)
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        release_camera()
        print("Camera released.")


if __name__ == "__main__":
    main()
