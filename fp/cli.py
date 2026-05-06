# Final Project - Mission CLI Entry Point
# Initializes all hardware (HAL, IMU, ultrasonic, camera), sets the robot's
# starting pose at the SW launch zone center, then runs the FSM loop.
# Saves a timestamped path plot to disk on exit.
#
# Requirements: fp.hal, fp.vision, fp.fsm, fp.localization, fp.mapping
# Usage: python main.py  (or: python -m fp)
import logging
import sys
import datetime

from fp import arena
from fp import hal
from fp import vision
from fp import fsm
from fp.localization import Localizer
from fp.mapping import PathTracker

logger = logging.getLogger(__name__)

def _mission_main() -> int:
    tracker = None
    try:
        hal.init()
        if not hal.start_imu():
            logger.info("[CLI] IMU failed to come up; aborting")
            return 1
        
        if not hal.start_ultrasonic():
            logger.info("[CLI] Ultrasonic failed to start; aborting")
            return 1

        if not vision.init_camera():
            logger.info("[CLI] Camera failed to come up; aborting")
            return 1

        loc = Localizer()
        tracker = PathTracker()
        loc.tracker = tracker
        loc.pose.heading = hal.get_heading()

        loc.set_position(arena.LAUNCH_ZONE_CENTER[0], arena.LAUNCH_ZONE_CENTER[1])
        logger.info(f"[CLI] start pose: ({loc.pose.x_cm, loc.pose.y_cm, loc.pose.heading})")

        input("Press enter to start!")

        final_state, delivered = fsm.run(loc)
        logger.info(f"[CLI] done: state {final_state.value} delivered {delivered} of {fsm.TOTAL_BLOCKS}")
        return 0
    finally:
        try:
            if tracker is not None:
                plot_path = datetime.datetime.now().strftime("run_path_%Y%m%dT%H%M%S.png")
                tracker.save_plot(plot_path)
                logger.info(f"[CLI] path plot saved to {plot_path}")
        except Exception as e:
            logger.info(f"[CLI] path plot could not be saved: {e}")
        try:
            vision.release_camera()
        except Exception:
            pass
        try:
            hal.stop_imu()
        except Exception:
            pass
        try:
            hal.stop_ultrasonic()
        except Exception:
            pass
        try:
            hal.cleanup()
        except Exception:
            pass

def main() -> int:
    return _mission_main()

if __name__ == "__main__":
    sys.exit(main())
