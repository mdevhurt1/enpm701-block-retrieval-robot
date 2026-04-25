import sys
import logging
import argparse
import os as _os

from fp import hal
from fp import vision
from fp import navigation
from fp import state_machine

# Logging setup
_log_level = getattr(logging, _os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Argument parsing

def _run_flags_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--color", metavar="COLOR",
        choices=list(vision.COLOR_MASKS),
        help="lock autonomous run to a single block colour",
    )
    p.add_argument(
        "--no-wait", action="store_true",
        help='skip the "press Enter" handshake before autonomous run',
    )
    p.add_argument(
        "--repl", action="store_true",
        help="attach an interactive REPL to the autonomous run (pause/tune/inspect)",
    )
    return p


def _build_parser() -> argparse.ArgumentParser:
    _run_flags = _run_flags_parser()
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "ENPM 701 Grand Challenge — autonomous robot.\n\n"
            "Run without subcommand for a full autonomous run.\n"
            "Use a subcommand for interactive test/teleop modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_run_flags],
    )
    parser.add_argument(
        "--testing", metavar="MODE", dest="_legacy_testing",
        help=argparse.SUPPRESS,
    )

    sub = parser.add_subparsers(dest="mode", title="test subcommands", metavar="SUBCOMMAND")

    sub.add_parser(
        "run",
        parents=[_run_flags_parser()],
        help="autonomous run (explicit form of the default no-subcommand behaviour)",
    )

    sub.add_parser("hal",                     help="interactive HAL teleop test")
    sub.add_parser("navigation",              help="interactive navigation closed-loop teleop")
    sub.add_parser("odometry",                help="odometry teleop + path plot")
    sub.add_parser("ground_truth_correction", help="GTC teleop + before/after plot")
    sub.add_parser("initial_calibration",     help="two-wall pose calibration")
    sub.add_parser("detect",                  help="continuous block detection printout")

    p_approach = sub.add_parser("approach", help="approach a block until in reach")
    p_approach.add_argument("color", choices=list(vision.COLOR_MASKS))

    p_in_reach = sub.add_parser("in_reach", help="print whether a block is in reach")
    p_in_reach.add_argument("color", choices=list(vision.COLOR_MASKS))

    p_fsm = sub.add_parser("fsm_to_home", help="find block, grab it, return home")
    p_fsm.add_argument("color", choices=list(vision.COLOR_MASKS))
    p_fsm.add_argument("--color-override", metavar="COLOR",
                       choices=list(vision.COLOR_MASKS))

    sub.add_parser("ekf",        help="interactive EKF state inspector")
    sub.add_parser("qr",         help="live QR code detection / pose decode")
    sub.add_parser("planner",    help="occupancy grid + A* visualisation")
    sub.add_parser("camera_cal", help="run offline camera calibration tool")

    return parser


def _apply_legacy_testing_compat(argv: list) -> list:
    if "--testing" not in argv:
        return argv

    idx = argv.index("--testing")
    new_argv = [a for i, a in enumerate(argv) if i not in (idx, idx + 1)]

    if idx + 1 < len(argv):
        mode = argv[idx + 1]
        new_argv.insert(1, mode)
        if idx + 2 < len(argv) and not argv[idx + 2].startswith("-"):
            color_arg = argv[idx + 2]
            new_argv = [a for a in new_argv if a != color_arg]
            new_argv.insert(2, color_arg)

    logger.warning(
        "[MAIN] --testing flag is deprecated; use subcommand form.  "
        "E.g.: python main.py %s", " ".join(new_argv[1:])
    )
    return new_argv


# Main

def main() -> None:
    effective_argv = _apply_legacy_testing_compat(sys.argv[1:])
    parser = _build_parser()
    args   = parser.parse_args(effective_argv)

    mode           = args.mode
    color_override = args.color

    from fp import teleop  # lazy import - avoids picamera2 init at module load time

    logger.info("[MAIN] Initialising hardware")
    hal.init()

    imu_ok = hal.start_imu()
    if not imu_ok:
        logger.warning("[MAIN] IMU not ready — heading control will be unreliable")

    try:
        if mode == "hal":
            logger.info("[MAIN] HAL teleop mode")
            teleop.run_hal_teleop()
            return

        if mode == "navigation":
            logger.info("[MAIN] Navigation teleop mode")
            teleop.run_navigation_teleop()
            return

        if mode == "odometry":
            logger.info("[MAIN] Odometry teleop mode")
            teleop.run_odometry_teleop()
            return

        if mode == "ground_truth_correction":
            logger.info("[MAIN] Ground truth correction teleop mode")
            teleop.run_ground_truth_teleop()
            return

        if mode == "initial_calibration":
            logger.info("[MAIN] Initial calibration mode")
            print("  Place robot in launch zone facing south, then press Enter...")
            input()
            result = navigation.calibrate_initial_pose()
            if result is None:
                logger.error("[MAIN] Calibration failed — check ultrasonic sensor")
            return

        if mode == "detect":
            logger.info("[MAIN] Vision detect test mode")
            teleop.run_detect_test()
            return

        if mode == "approach":
            logger.info("[MAIN] Vision approach test mode — target: %s", args.color)
            teleop.run_approach_test(args.color)
            return

        if mode == "in_reach":
            logger.info("[MAIN] Vision in_reach test mode — target: %s", args.color)
            teleop.run_in_reach_test(args.color)
            return

        if mode == "fsm_to_home":
            logger.info("[MAIN] FSM to-home test — target: %s", args.color)
            teleop.run_fsm_to_home_test(args.color, args.color_override)
            return

        if mode == "ekf":
            logger.info("[MAIN] EKF inspector mode")
            teleop.run_ekf_test()
            return

        if mode == "qr":
            logger.info("[MAIN] QR detection mode")
            teleop.run_qr_test()
            return

        if mode == "planner":
            logger.info("[MAIN] Planner visualisation mode (no hardware used)")
            teleop.run_planner_test()
            return

        if mode == "camera_cal":
            logger.info("[MAIN] Camera calibration mode")
            teleop.run_camera_cal()
            return

        if mode not in (None, "run"):
            logger.error("[MAIN] Unknown mode %r", mode)
            return

        if not args.no_wait:
            print("  Ready to run autonomously.  Press Enter to start (or pass --no-wait)...")
            input()

        logger.info("[MAIN] Initialising camera")
        picam2 = vision.init_camera()

        try:
            from fp.repl.tuning_diff import load_toml_if_exists
            _toml_path = "runtime.toml"
            n = load_toml_if_exists(_toml_path)
            if n:
                logger.info("[MAIN] Loaded %d tuning override(s) from %s", n, _toml_path)

            ctx = state_machine.RobotContext()
            ctx.picam2         = picam2
            ctx.color_override = color_override
            if color_override:
                logger.info("[MAIN] Color locked to: %s", color_override)

            if args.repl:
                from fp.repl.repl import start_repl
                from fp.fsm.runner import FsmControl
                logger.info("[MAIN] Starting autonomous run with REPL")
                control = FsmControl()
                start_repl(ctx, control, picam2=picam2, toml_path=_toml_path)
            else:
                logger.info("[MAIN] Starting autonomous run")
                state_machine.run(ctx)

        finally:
            hal.gripper_open()
            navigation.stop_odometry()
            vision.release_camera(picam2)

    finally:
        logger.info("[MAIN] Cleaning up")
        hal.stop_motors()
        hal.stop_imu()
        hal.cleanup()
        logger.info("[MAIN] Done")


if __name__ == "__main__":
    main()
