# Final Project Calibration - AWB Mode Image Comparison
# Captures one full-resolution frame under each libcamera AWB preset
# (Auto through Cloudy) and saves them to calibration/photos/ for
# side-by-side color-balance comparison.
#
# Requirements: picamera2, libcamera, opencv-python
# Usage: python calibration/capture_full_res.py

import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2
from libcamera import controls, Transform

FRAME_DOWNSAMPLE = 0.5
FRAME_TOP = 0.8
FRAME_BOTTOM = 0.3

# All standard libcamera AWB modes to compare.
AWB_MODES = [
    ("Auto",         controls.AwbModeEnum.Auto),
    ("Incandescent", controls.AwbModeEnum.Incandescent),
    ("Tungsten",     controls.AwbModeEnum.Tungsten),
    ("Fluorescent",  controls.AwbModeEnum.Fluorescent),
    ("Indoor",       controls.AwbModeEnum.Indoor),
    ("Daylight",     controls.AwbModeEnum.Daylight),
    ("Cloudy",       controls.AwbModeEnum.Cloudy),
]

CALIBRATION_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(CALIBRATION_DIR, "photos")


def main():
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")

    picam2 = Picamera2()
    sensor_size = picam2.sensor_resolution  # native sensor (e.g. 3280x2464)

    cfg = picam2.create_still_configuration(
        main={"size": sensor_size, "format": "RGB888"},
    )
    picam2.configure(cfg)
    picam2.start()

    os.makedirs(PHOTOS_DIR, exist_ok=True)

    # Initial settle - let AE stabilise once, before we start switching AWB.
    time.sleep(3)

    for name, mode in AWB_MODES:
        picam2.set_controls({"AwbMode": mode})
        # Give AWB time to converge to the new mode's target CCT.
        time.sleep(2)

        frame = picam2.capture_array("main")
        # Picamera2 RGB888 actually delivers bytes in BGR order already,
        # so no cv2.cvtColor conversion is needed - OpenCV can use it directly.
        # (See homework09/still.py for the established convention.)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h = frame.shape[0]
        top = int(h * (1 - FRAME_TOP))
        bot = int(h * (1 - FRAME_BOTTOM))
        frame = frame[top:bot, :]
        frame = cv2.resize(frame, None, fx=FRAME_DOWNSAMPLE, fy=FRAME_DOWNSAMPLE,
                           interpolation=cv2.INTER_AREA)

        # Save labelled image to photos/.
        path = os.path.join(PHOTOS_DIR, "awb_{}.jpg".format(name.lower()))
        cv2.imwrite(path, frame)
        print("Saved {} ({} x {})".format(path, frame.shape[1], frame.shape[0]))

        # Save a timestamped copy of the Auto capture.
        if name == "Auto":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_path = os.path.join(PHOTOS_DIR, "auto_{}.jpg".format(ts))
            cv2.imwrite(ts_path, frame)
            print("Saved {}".format(ts_path))

    picam2.stop()
    print("Done - {} images saved to {}".format(len(AWB_MODES), PHOTOS_DIR))


if __name__ == "__main__":
    main()
