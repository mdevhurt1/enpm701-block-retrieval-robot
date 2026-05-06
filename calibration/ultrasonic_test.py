# Final Project Calibration - Ultrasonic Distance Test
# Prints live HC-SR04 distance readings via the HAL, including the
# robot-center-corrected distance and wall-proximity flag.
#
# Requirements: fp.hal (pigpio), fp.localization
# Usage: python calibration/ultrasonic_test.py

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_fp_parent = os.path.dirname(_here)
if _fp_parent not in sys.path:
    sys.path.insert(0, _fp_parent)

import fp.hal as hal
from fp.localization import Localizer

_POLL_INTERVAL_S = 0.1


def main():
    hal.init()
    print("HAL initialized.")

    if not hal.start_ultrasonic():
        print("ERROR: Ultrasonic sensor did not become ready.")
        hal.cleanup()
        return

    loc = Localizer()
    print("Printing ultrasonic readings. Press Ctrl+C to stop.\n")
    try:
        while True:
            dist = hal.get_ultrasonic_distance()
            healthy = hal.is_ultrasonic_healthy()
            to_center = loc.ultrasonic_to_robot_center(dist, 0.0)
            inside = loc.ultrasonic_reading_inside_robot_radius(dist, 0.0)
            print(
                f"  distance: {dist:6.1f} cm  to_center: {to_center:6.1f} cm"
                f"  inside_radius: {inside!s:<5}  healthy: {healthy}",
                end="\r",
            )
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hal.stop_ultrasonic()
        hal.cleanup()
        print("HAL cleaned up.")


if __name__ == "__main__":
    main()
