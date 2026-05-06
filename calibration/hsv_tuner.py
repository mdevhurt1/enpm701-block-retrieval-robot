# Final Project Calibration - Live HSV Trackbar Tuner
# Streams from the Pi camera while six trackbars (H/S/V min/max) control
# the color mask in real time. Press 'p' to print values for vision.py,
# 'q' to quit. Run once per color under actual arena lighting.
#
# Requirements: picamera2, opencv-python, numpy
# Usage: python calibration/hsv_tuner.py

import cv2
import numpy as np
import time
from picamera2 import Picamera2

WIDTH, HEIGHT = 640, 480
WIN_W, WIN_H = 240, 180   # display size for each window


def nothing(_):
    pass


def main():
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(2)

    cv2.namedWindow("Mask",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Preview",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    # Tile windows to fit a small VNC display (~1024x768)
    # Row 0: Preview (left) | Mask (right)
    # Row 1: Trackbars (full width, camera feed with overlay)
    cv2.resizeWindow("Preview",   WIN_W, WIN_H)
    cv2.resizeWindow("Mask",      WIN_W, WIN_H)
    cv2.resizeWindow("Trackbars", WIN_W * 2, WIN_H + 40)
    cv2.moveWindow("Preview",   0,     0)
    cv2.moveWindow("Mask",      WIN_W, 0)
    cv2.moveWindow("Trackbars", 0,     WIN_H + 30)   # +30 for title bar

    # Trackbar defaults - start with a wide-open range
    defaults = {"H_MIN": 0, "S_MIN": 0, "V_MIN": 0,
                "H_MAX": 179, "S_MAX": 255, "V_MAX": 255}
    for name, val in defaults.items():
        limit = 179 if name == "H_MAX" or name == "H_MIN" else 255
        cv2.createTrackbar(name, "Trackbars", val, limit, nothing)

    print("Adjust trackbars to isolate your color, then press 'p' to print values, 'q' to quit.")

    while True:
        bgr  = picam2.capture_array()
        bgr  = cv2.flip(bgr, 0)
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
        s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
        v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
        h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
        s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
        v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")

        lo   = np.array([h_min, s_min, v_min], dtype=np.uint8)
        hi   = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        preview = cv2.bitwise_and(bgr, bgr, mask=mask)

        # Count masked pixels and show area
        area = int(cv2.countNonZero(mask))
        cv2.putText(bgr, f"Mask area: {area} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Mask",    mask)
        cv2.imshow("Preview", preview)
        cv2.imshow("Trackbars", bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\nCurrent HSV range:")
            print(f"  lower = ({h_min}, {s_min}, {v_min})")
            print(f"  upper = ({h_max}, {s_max}, {v_max})")
            print(f"  Python tuple: ({h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max})")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
