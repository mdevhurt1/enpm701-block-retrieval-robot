"""
hsv_tuner_static.py - Interactive HSV colour-mask calibration on a static image.

Works on a JPEG instead of a live camera feed - useful for tuning on a laptop
without a Pi or camera attached.

Usage:
    python calibration/hsv_tuner_static.py [image_path]

Defaults to calibration/photos/awb_auto.jpg when no path is given.

Controls:
  - Adjust the six trackbars (H_MIN, S_MIN, V_MIN, H_MAX, S_MAX, V_MAX)
    until the target block colour shows as solid white in the "Mask" window
    and everything else is black.
  - Press 'p' to print the current values to the terminal (copy into vision.py).
  - Press 'q' to quit.
"""

import sys
import os
import cv2
import numpy as np

DEFAULT_IMAGE = os.path.join(
    os.path.dirname(__file__), "photos", "awb_auto.jpg"
)

WIN_W, WIN_H = 640, 480


def nothing(_):
    pass


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    bgr_orig = cv2.imread(image_path)
    if bgr_orig is None:
        print(f"Error: could not load image: {image_path}")
        sys.exit(1)

    bgr_orig = cv2.resize(bgr_orig, (WIN_W, WIN_H))
    hsv = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Mask",      cv2.WINDOW_NORMAL)
    cv2.namedWindow("Preview",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    half_w = WIN_W // 2
    half_h = WIN_H // 2
    # resizeWindow sets the image-content area; trackbars add height above it.
    # Pass the image at exactly these dimensions so OpenCV doesn't rescale it.
    cv2.resizeWindow("Preview",   half_w, half_h)
    cv2.resizeWindow("Mask",      half_w, half_h)
    cv2.resizeWindow("Trackbars", WIN_W,  half_h)
    cv2.moveWindow("Preview",   0,       0)
    cv2.moveWindow("Mask",      half_w,  0)
    cv2.moveWindow("Trackbars", 0,       half_h + 30)

    defaults = {"H_MIN": 0, "S_MIN": 0, "V_MIN": 0,
                "H_MAX": 179, "S_MAX": 255, "V_MAX": 255}
    for name, val in defaults.items():
        limit = 179 if "H_" in name else 255
        cv2.createTrackbar(name, "Trackbars", val, limit, nothing)

    print(f"Image: {image_path}")
    print("Adjust trackbars to isolate your colour, then press 'p' to print values, 'q' to quit.")

    while True:
        h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
        s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
        v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
        h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
        s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
        v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")

        lo   = np.array([h_min, s_min, v_min], dtype=np.uint8)
        hi   = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)

        preview = cv2.bitwise_and(bgr_orig, bgr_orig, mask=mask)

        area = int(cv2.countNonZero(mask))

        # Resize to exactly match the content area set by resizeWindow so the
        # image is never rescaled (which distorts the slider positions).
        overlay = cv2.resize(bgr_orig.copy(), (WIN_W, half_h))

        # Draw current HSV values prominently so they're readable at a glance.
        labels = [
            f"H: [{h_min:3d}, {h_max:3d}]",
            f"S: [{s_min:3d}, {s_max:3d}]",
            f"V: [{v_min:3d}, {v_max:3d}]",
            f"Area: {area} px",
        ]
        for i, text in enumerate(labels):
            cv2.putText(overlay, text, (10, 28 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),   3)
            cv2.putText(overlay, text, (10, 28 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Mask",      mask)
        cv2.imshow("Preview",   preview)
        cv2.imshow("Trackbars", overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\nCurrent HSV range:")
            print(f"  lower = ({h_min}, {s_min}, {v_min})")
            print(f"  upper = ({h_max}, {s_max}, {v_max})")
            print(f"  Python tuple: ({h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
