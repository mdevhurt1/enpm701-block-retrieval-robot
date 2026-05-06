# Final Project Calibration - Encoder Revolution Test
# Spins each motor side one revolution forward and one revolution reverse,
# printing live encoder counts to verify COUNTS_PER_REVOLUTION.
#
# Requirements: fp.hal (pigpio)
# Usage: python calibration/encoder_test.py

import sys
import os
import time

_here = os.path.dirname(os.path.abspath(__file__))
_fp_parent = os.path.dirname(_here)
if _fp_parent not in sys.path:
    sys.path.insert(0, _fp_parent)

import fp.hal as hal

_COUNTS_PER_REVOLUTION = 20
_POLL_INTERVAL_S = 0.05


def _run_until_revolution(label: str, read_encoder, start_count: int, set_motors) -> None:
    """Drive motors until encoder delta reaches one revolution, printing counts live."""
    print(f"\n--- {label} ---")
    hal.stop_motors()
    set_motors()

    while True:
        count = read_encoder()
        delta = count - start_count
        print(f"  count: {count}  delta: {delta}/{_COUNTS_PER_REVOLUTION}", end="\r")
        if delta >= _COUNTS_PER_REVOLUTION:
            break
        time.sleep(_POLL_INTERVAL_S)

    hal.stop_motors()
    final = read_encoder()
    print(f"\n  Done. final={final}  delta={final - start_count}")


def main():
    hal.init()
    print("HAL initialized.")
    print(f"One revolution = {_COUNTS_PER_REVOLUTION} counts\n")

    try:
        # Left forward
        start = hal.read_front_left_encoder()
        _run_until_revolution(
            "LEFT FORWARD",
            hal.read_front_left_encoder,
            start,
            lambda: hal.set_individual_duties(+1, hal.PWM_DUTY_STRAIGHT, +1, 0),
        )
        time.sleep(0.5)

        # Left reverse
        start = hal.read_front_left_encoder()
        _run_until_revolution(
            "LEFT REVERSE",
            hal.read_front_left_encoder,
            start,
            lambda: hal.set_individual_duties(-1, hal.PWM_DUTY_STRAIGHT, +1, 0),
        )
        time.sleep(0.5)

        # Right forward
        start = hal.read_back_right_encoder()
        _run_until_revolution(
            "RIGHT FORWARD",
            hal.read_back_right_encoder,
            start,
            lambda: hal.set_individual_duties(+1, 0, +1, hal.PWM_DUTY_STRAIGHT),
        )
        time.sleep(0.5)

        # Right reverse
        start = hal.read_back_right_encoder()
        _run_until_revolution(
            "RIGHT REVERSE",
            hal.read_back_right_encoder,
            start,
            lambda: hal.set_individual_duties(+1, 0, -1, hal.PWM_DUTY_STRAIGHT),
        )

        print("\nEncoder test complete.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hal.cleanup()
        print("HAL cleaned up.")


if __name__ == "__main__":
    main()
