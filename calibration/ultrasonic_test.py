"""
ultrasonic_test.py - Print live ultrasonic distance readings.

Usage (on the Raspberry Pi):
    python calibration/ultrasonic_test.py
"""

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_fp_parent = os.path.dirname(_here)
if _fp_parent not in sys.path:
    sys.path.insert(0, _fp_parent)

import fp.hal as hal

_POLL_INTERVAL_S = 0.1


def main():
    hal.init()
    print("HAL initialised.")

    if not hal.start_ultrasonic():
        print("ERROR: Ultrasonic sensor did not become ready.")
        hal.cleanup()
        return

    print("Printing ultrasonic readings. Press Ctrl+C to stop.\n")
    try:
        while True:
            dist = hal.get_ultrasonic_distance()
            healthy = hal.is_ultrasonic_healthy()
            print(f"  distance: {dist:6.1f} cm  healthy: {healthy}", end="\r")
            time.sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hal.stop_ultrasonic()
        hal.cleanup()
        print("HAL cleaned up.")


if __name__ == "__main__":
    main()
