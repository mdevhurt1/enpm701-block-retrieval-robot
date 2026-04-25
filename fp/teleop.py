from __future__ import annotations

import time
import logging
from typing import Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


# TeleopDispatcher

class TeleopDispatcher:
    """
    Generic teleop command loop.

    ``prompt`` may be a plain string or a zero-argument callable that returns
    the current prompt string (useful for prompts that embed live state such as
    the current odometry position).

    Each handler receives the tokenised input line as ``parts: list[str]``
    where ``parts[0]`` is the command letter.  Raise ``ValueError`` or
    ``IndexError`` to emit a "bad arguments" message; raise ``StopIteration``
    to exit the loop early.
    """

    def __init__(
        self,
        prompt: Union[str, Callable[[], str]],
        help_text: str,
        handlers: Dict[str, Callable[[list], None]],
        post_command: Optional[Callable[[], None]] = None,
    ):
        self._prompt = prompt
        self._help_text = help_text
        self._handlers = handlers
        self._post_command = post_command

    def run(self) -> None:
        print(self._help_text)

        while True:
            prompt = self._prompt() if callable(self._prompt) else self._prompt
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            parts = raw.split()
            if not parts:
                continue

            cmd = parts[0].lower()

            if cmd in ("q", "quit"):
                break
            if cmd in ("?", "help"):
                print(self._help_text)
                continue

            handler = self._handlers.get(cmd)
            if handler is None:
                print(f"  unknown command '{cmd}' — type ? for help")
                continue

            try:
                handler(parts)
            except StopIteration:
                break
            except (ValueError, IndexError):
                print(f"  bad arguments for '{cmd}' — type ? for help")
            except Exception as exc:  # noqa: BLE001
                print(f"  error: {exc}")

            if self._post_command is not None:
                self._post_command()


# Utility

def _void(*_: object) -> None:
    """Accept any arguments and return None.

    Used in handler lambdas that combine two side-effecting calls with a
    tuple expression; this avoids the tuple return type that mypy rejects.
    """


# Plot helpers (shared by odometry and GTC teleop)

def _save_odometry_plot(path: list, filename: str = "odometry_testing.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if len(path) < 2:
            print("[ODO] Not enough path points to plot")
            return

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(xs, ys, "b-", linewidth=1.2, label="path")
        ax.plot(xs[0],  ys[0],  "go", markersize=10, label="start")
        ax.plot(xs[-1], ys[-1], "rs", markersize=10, label="end")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title("Odometry Path")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[ODO] Path plot saved to {filename}  ({len(path)} points)")
    except ImportError:
        print("[ODO] matplotlib not available — skipping plot")


def _save_gtc_correction_plot(
    path_before: list, path_after: list, wall: str, index: int
) -> None:
    filename = f"ground_truth_correction_{index}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Ground Truth Correction #{index}  -  {wall} wall", fontsize=13)

        def _plot(ax, path, title):
            if len(path) >= 2:
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                ax.plot(xs, ys, "b-", linewidth=1.4)
                ax.plot(xs[0],  ys[0],  "go", markersize=9, label="start")
                ax.plot(xs[-1], ys[-1], "rs", markersize=9, label="end")
                ax.legend(fontsize=9)
            elif path:
                ax.plot(path[0][0], path[0][1], "go", markersize=9)
            ax.set_title(title)
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_aspect("equal")
            ax.grid(True)

        _plot(ax_b, path_before, "Before Correction")
        _plot(ax_a, path_after,  "After Correction")

        all_pts = path_before + path_after
        if all_pts:
            xs_all = [p[0] for p in all_pts]
            ys_all = [p[1] for p in all_pts]
            margin = 10.0
            for ax in (ax_b, ax_a):
                ax.set_xlim(min(xs_all) - margin, max(xs_all) + margin)
                ax.set_ylim(min(ys_all) - margin, max(ys_all) + margin)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[GTC] Saved {filename}")
    except ImportError:
        print("[GTC] matplotlib not available — skipping plot")


# HAL teleop

_HAL_HELP = """
HAL Teleop Commands
-------------------
  w       drive forward  (open-loop, 1 s)
  s       drive reverse  (open-loop, 1 s)
  a       pivot CCW/left (open-loop, 1 s)
  d       pivot CW/right (open-loop, 1 s)
  x       stop motors
  o       gripper open
  c       gripper close
  f       read front ultrasonic (cm)
  h       read IMU heading (deg)
  ?       show this help
  q       quit
"""

def run_hal_teleop() -> None:
    from fp import hal

    DRIVE_TIME = 1.0

    def _drive(func, name):
        print(f"  -> motors {name} {hal.PWM_DUTY_STRAIGHT}% for {DRIVE_TIME} s")
        func()
        time.sleep(DRIVE_TIME)
        hal.stop_motors()
        print(f"  heading={hal.get_heading():.1f} deg")

    TeleopDispatcher(
        prompt="hal> ",
        help_text=_HAL_HELP,
        handlers={
            "w": lambda _: _drive(hal.motors_forward,   "forward"),
            "s": lambda _: _drive(hal.motors_reverse,   "reverse"),
            "a": lambda _: _drive(hal.motors_pivot_ccw, "CCW"),
            "d": lambda _: _drive(hal.motors_pivot_cw,  "CW"),
            "x": lambda _: _void(hal.stop_motors(), print("  -> motors stopped")),
            "o": lambda _: _void(hal.gripper_open(),  print("  -> gripper open")),
            "c": lambda _: _void(hal.gripper_close(), print("  -> gripper closed")),
            "f": lambda _: print(f"  front distance: {hal.distance_front()} cm"),
            "h": lambda _: print(f"  IMU heading: {hal.get_heading():.1f} deg"),
        },
    ).run()

    hal.stop_motors()
    print("[HAL] Teleop ended")


# Navigation teleop

_NAV_HELP = """
Navigation Teleop Commands
--------------------------
  f [cm]      move forward        (default 20 cm, closed-loop encoder)
  b [cm]      move reverse        (default 20 cm, closed-loop encoder)
  l [deg]     pivot left/CCW      (default 90 deg,   closed-loop IMU)
  r [deg]     pivot right/CW      (default 90 deg,   closed-loop IMU)
  h <deg>     pivot to absolute heading (closed-loop IMU)
  g <x> <y>   navigate to coordinate in cm
  u           read front ultrasonic (cm)
  p           print current odometry position and heading
  ?           show this help
  q           quit
"""

def run_navigation_teleop() -> None:
    from fp import hal
    from fp import arena
    from fp import navigation

    pos = [0.0, 0.0]

    def _hfmt():
        return f"  heading: {navigation.get_arena_heading():.1f} deg"

    def _pfmt():
        return f"  position: ({pos[0]:.1f}, {pos[1]:.1f}) cm  heading: {navigation.get_arena_heading():.1f} deg"

    def _cmd_f(parts):
        cm = float(parts[1]) if len(parts) > 1 else 20.0
        hdg = navigation.get_arena_heading()
        navigation.set_pose(pos[0], pos[1])
        navigation.move_forward(cm)
        pos[0], pos[1] = arena.apply_motion(pos[0], pos[1], cm, hdg)
        print(_pfmt())

    def _cmd_b(parts):
        cm = float(parts[1]) if len(parts) > 1 else 20.0
        hdg = navigation.get_arena_heading()
        navigation.set_pose(pos[0], pos[1])
        navigation.move_reverse(cm)
        pos[0], pos[1] = arena.apply_motion(pos[0], pos[1], -cm, hdg)
        print(_pfmt())

    def _cmd_g(parts):
        if len(parts) < 3:
            print("  usage: g <x_cm> <y_cm>")
            return
        tx, ty = float(parts[1]), float(parts[2])
        pos[0], pos[1], _ = navigation.navigate_to(tx, ty, pos[0], pos[1])
        print(_pfmt())

    print(f"  starting position: (0.0, 0.0)  heading: {navigation.get_arena_heading():.1f} deg\n")

    TeleopDispatcher(
        prompt=lambda: f"nav ({pos[0]:.1f},{pos[1]:.1f})> ",
        help_text=_NAV_HELP,
        handlers={
            "f": _cmd_f,
            "b": _cmd_b,
            "l": lambda p: _void(navigation.pivot_left(float(p[1]) if len(p) > 1 else 90.0),  print(_hfmt())),
            "r": lambda p: _void(navigation.pivot_right(float(p[1]) if len(p) > 1 else 90.0), print(_hfmt())),
            "h": lambda p: _void(navigation.pivot_to_heading(float(p[1])),                     print(_hfmt())),
            "g": _cmd_g,
            "u": lambda _: print(f"  front distance: {hal.distance_front()} cm"),
            "p": lambda _: print(_pfmt()),
        },
    ).run()

    hal.stop_motors()
    print("[NAV] Teleop ended")


# Odometry teleop

_ODO_HELP = """
Odometry Teleop Commands
------------------------
  f [cm]      move forward        (default 20 cm)
  b [cm]      move reverse        (default 20 cm)
  l [deg]     pivot left/CCW      (default 90 deg)
  r [deg]     pivot right/CW      (default 90 deg)
  h <deg>     pivot to absolute heading
  g <x> <y>   navigate to coordinate (cm)
  p           print current odometry pose
  ?           show this help
  q           quit and save odometry_testing.png
"""

def run_odometry_teleop() -> None:
    from fp import hal
    from fp import navigation

    navigation.start_odometry(rate_hz=100)

    def _pose():
        x, y, hdg = navigation.get_pose()
        return x, y, hdg

    def _print_pose():
        x, y, hdg = _pose()
        print(f"  pose: ({x:.1f}, {y:.1f}) cm  heading: {hdg:.1f} deg")

    def _cmd_g(parts):
        if len(parts) < 3:
            print("  usage: g <x_cm> <y_cm>")
            return
        cx, cy, _ = navigation.get_pose()
        navigation.navigate_to(float(parts[1]), float(parts[2]), cx, cy)

    x0, y0, h0 = _pose()
    print(f"  starting pose: ({x0:.1f}, {y0:.1f}) cm  heading: {h0:.1f} deg\n")

    try:
        TeleopDispatcher(
            prompt=lambda: (lambda x, y, _: f"odo ({x:.1f},{y:.1f})> ")(*navigation.get_pose()),
            help_text=_ODO_HELP,
            handlers={
                "f": lambda p: navigation.move_forward(float(p[1]) if len(p) > 1 else 20.0),
                "b": lambda p: navigation.move_reverse(float(p[1]) if len(p) > 1 else 20.0),
                "l": lambda p: navigation.pivot_left( float(p[1]) if len(p) > 1 else 90.0),
                "r": lambda p: navigation.pivot_right(float(p[1]) if len(p) > 1 else 90.0),
                "h": lambda p: navigation.pivot_to_heading(float(p[1])),
                "g": _cmd_g,
                "p": lambda _: None,
            },
            post_command=_print_pose,
        ).run()
    finally:
        hal.stop_motors()
        navigation.stop_odometry()
        _save_odometry_plot(navigation.get_path())
        print("[ODO] Teleop ended")


# Ground truth correction teleop

_GTC_HELP = """
Ground Truth Correction Teleop Commands
----------------------------------------
  f [cm]        move forward      (default 20 cm)
  b [cm]        move reverse      (default 20 cm)
  l [deg]       pivot left/CCW    (default 90 deg)
  r [deg]       pivot right/CW    (default 90 deg)
  h <deg>       pivot to absolute heading
  g <x> <y>     navigate to coordinate (cm)
  c <wall>      correct pose from wall  (north/south/east/west)
  p             print current pose
  ?             show this help
  q             quit
"""

def run_ground_truth_teleop() -> None:
    from fp import hal
    from fp import navigation

    print("[GTC] Running initial calibration...")
    print("  Place the robot in the launch zone facing south, then press Enter...")
    input()
    result = navigation.calibrate_initial_pose()
    if result is None:
        print("[GTC] Calibration failed — continuing with pose (0, 0)")
        init_x, init_y = 0.0, 0.0
    else:
        init_x, init_y, _ = result

    navigation.start_odometry(rate_hz=100, x=init_x, y=init_y)
    correction_count = [0]

    def _print_pose():
        x, y, hdg = navigation.get_pose()
        print(f"  pose: ({x:.1f}, {y:.1f}) cm  heading: {hdg:.1f} deg")

    def _cmd_g(parts):
        if len(parts) < 3:
            print("  usage: g <x_cm> <y_cm>")
            return
        cx, cy, _ = navigation.get_pose()
        navigation.navigate_to(float(parts[1]), float(parts[2]), cx, cy)

    def _cmd_c(parts):
        if len(parts) < 2:
            print("  usage: c <north|south|east|west>")
            return
        wall = parts[1].lower()
        path_before = navigation.get_path()
        if navigation.correct_pose_from_wall(wall):
            correction_count[0] += 1
            _save_gtc_correction_plot(
                path_before, navigation.get_path(), wall, correction_count[0]
            )

    x0, y0, h0 = navigation.get_pose()
    print(f"  starting pose: ({x0:.1f}, {y0:.1f}) cm  heading: {h0:.1f} deg\n")

    try:
        TeleopDispatcher(
            prompt=lambda: (lambda x, y, _: f"gtc ({x:.1f},{y:.1f})> ")(*navigation.get_pose()),
            help_text=_GTC_HELP,
            handlers={
                "f": lambda p: navigation.move_forward(float(p[1]) if len(p) > 1 else 20.0),
                "b": lambda p: navigation.move_reverse(float(p[1]) if len(p) > 1 else 20.0),
                "l": lambda p: navigation.pivot_left( float(p[1]) if len(p) > 1 else 90.0),
                "r": lambda p: navigation.pivot_right(float(p[1]) if len(p) > 1 else 90.0),
                "h": lambda p: navigation.pivot_to_heading(float(p[1])),
                "g": _cmd_g,
                "c": _cmd_c,
                "p": lambda _: None,
            },
            post_command=_print_pose,
        ).run()
    finally:
        hal.stop_motors()
        navigation.stop_odometry()
        print(f"[GTC] Teleop ended  ({correction_count[0]} correction(s) performed)")


# EKF inspector teleop

_EKF_HELP = """
EKF Inspector Commands
-----------------------
  f [cm]      move forward and print EKF state
  b [cm]      move reverse and print EKF state
  l [deg]     pivot left and print EKF state
  r [deg]     pivot right and print EKF state
  h <deg>     pivot to absolute heading and print EKF state
  w <wall>    force wall correction  (north/south/east/west)
  p           print EKF pose + covariance diagonal
  ?           show this help
  q           quit
"""

def run_ekf_test() -> None:
    import math
    import numpy as np
    from fp import ekf as ekf_module
    from fp import navigation
    from fp import hal
    from fp import state_machine

    print("[EKF] Running initial calibration...")
    print("  Place robot in launch zone facing south, press Enter...")
    input()
    result = navigation.calibrate_initial_pose()
    if result is None:
        print("[EKF] Calibration failed — using (0,0,0)")
        result = (0.0, 0.0, 0.0)
    init_x, init_y, init_hdg = result

    P0 = np.diag([9.0, 9.0, math.radians(2.0) ** 2])
    filter_ = ekf_module.EKF(x0=init_x, y0=init_y, heading_deg=init_hdg, P0=P0)
    navigation.init_ekf(filter_)
    navigation.start_odometry(rate_hz=100, x=init_x, y=init_y)

    def _print_ekf():
        x, y, hdg = filter_.get_pose()
        P = filter_.get_covariance()
        diag = np.diag(P)
        print(f"  EKF pose: ({x:.1f}, {y:.1f}) cm  {hdg:.1f} deg")
        print(f"  Cov diag: x={diag[0]:.2f} cm²  y={diag[1]:.2f} cm²  "
              f"θ={math.degrees(math.sqrt(max(0, diag[2]))):.2f} deg")

    def _cmd_w(parts):
        if len(parts) < 2:
            print("  usage: w <north|south|east|west>")
            return
        wall = parts[1].lower()
        navigation.pivot_to_heading(state_machine.WALL_HEADING.get(wall, 0.0))
        navigation.correct_pose_from_wall(wall)

    try:
        TeleopDispatcher(
            prompt="ekf> ",
            help_text=_EKF_HELP,
            handlers={
                "f": lambda p: navigation.move_forward(float(p[1]) if len(p) > 1 else 20.0),
                "b": lambda p: navigation.move_reverse(float(p[1]) if len(p) > 1 else 20.0),
                "l": lambda p: navigation.pivot_left( float(p[1]) if len(p) > 1 else 90.0),
                "r": lambda p: navigation.pivot_right(float(p[1]) if len(p) > 1 else 90.0),
                "h": lambda p: navigation.pivot_to_heading(float(p[1])),
                "w": _cmd_w,
                "p": lambda _: None,
            },
            post_command=_print_ekf,
        ).run()
    finally:
        hal.stop_motors()
        navigation.stop_odometry()
        print("[EKF] Done")


# FSM to-home test

def run_fsm_to_home_test(color: str, color_override: str = None) -> None:
    from fp import navigation
    from fp import planner as planner_module
    from fp import vision
    from fp import state_machine

    VALID = set(vision.COLOR_MASKS.keys())
    if color not in VALID:
        print(f"[FSM_HOME] Unknown color '{color}'. Valid: {', '.join(sorted(VALID))}")
        return

    print("[FSM_HOME] Running initial calibration...")
    print("  Place the robot in the launch zone facing south, then press Enter...")
    input()
    result = navigation.calibrate_initial_pose()
    if result is None:
        print("[FSM_HOME] Calibration failed — check ultrasonic sensor and robot placement")
        return
    init_x, init_y, init_hdg = result
    print(f"[FSM_HOME] Starting position: ({init_x:.1f}, {init_y:.1f}) cm")

    import math
    import numpy as np
    from fp import ekf as ekf_module
    P0 = np.diag([9.0, 9.0, math.radians(2.0) ** 2])
    ekf_inst = ekf_module.EKF(x0=init_x, y0=init_y, heading_deg=init_hdg, P0=P0)
    navigation.init_ekf(ekf_inst)
    navigation.start_odometry(rate_hz=100, x=init_x, y=init_y)

    print("[FSM_HOME] Initialising camera...")
    picam2 = vision.init_camera()

    from fp import hal
    try:
        ctx = state_machine.RobotContext()
        ctx.picam2           = picam2
        ctx.ekf              = ekf_inst
        ctx.next_color_index = state_machine.COLOR_ORDER.index(color)
        ctx.color_override   = color_override
        ctx.planner          = planner_module.OccupancyGrid()
        ctx.planner.log_traversed(init_x, init_y)
        state_machine.set_collection_zone(ctx, init_x, init_y)

        hooks = state_machine.FSMTestHooks(
            skip_init_calibration=True,
            stop_after_delivery=True,
        )

        print(f"[FSM_HOME] Target: {color}  Home: ({init_x:.1f}, {init_y:.1f}) cm")
        print("[FSM_HOME] Press Enter to start...")
        input()

        state_machine.run(ctx, hooks=hooks)

    finally:
        hal.stop_motors()
        navigation.stop_odometry()
        vision.release_camera(picam2)
        print("[FSM_HOME] Done")


# Vision approach test

def run_approach_test(color: str) -> None:
    from fp import hal
    from fp import navigation
    from fp import vision

    VALID = set(vision.COLOR_MASKS.keys())
    if color not in VALID:
        print(f"[APPROACH] Unknown color '{color}'. Valid: {', '.join(sorted(VALID))}")
        return

    print("[APPROACH] Initialising camera...")
    picam2 = vision.init_camera()
    print(f"[APPROACH] Targeting: {color}  - press Ctrl-C to stop\n")

    try:
        while True:
            print(f"[APPROACH] Scanning for {color} block...")
            while True:
                frame = vision.capture_frame(picam2)
                det   = vision.detect_block(frame, color)
                if det is not None:
                    print(f"[APPROACH] {color} detected  area={int(det.area)}  "
                          f"err={det.centroid_error:+d}px  - starting approach")
                    break
                time.sleep(0.2)

            result = navigation.approach_visual(picam2, target_color=color)

            if result:
                print("[APPROACH] Block in reach — gripper position reached")
            else:
                print("[APPROACH] Approach aborted (block lost or wall detected)")

            print("[APPROACH] Press Enter to scan again, or Ctrl-C to quit...")
            try:
                input()
            except EOFError:
                break

    except KeyboardInterrupt:
        print()
    finally:
        hal.stop_motors()
        vision.release_camera(picam2)
        print("[APPROACH] Done")


# Vision in_reach test

def run_in_reach_test(color: str) -> None:
    from fp import vision

    VALID = set(vision.COLOR_MASKS.keys())
    if color not in VALID:
        print(f"[IN_REACH] Unknown color '{color}'. Valid: {', '.join(sorted(VALID))}")
        return

    print("[IN_REACH] Initialising camera...")
    picam2 = vision.init_camera()
    print(f"[IN_REACH] Targeting: {color}  - press Ctrl-C to stop\n")

    try:
        while True:
            frame = vision.capture_frame(picam2)
            det   = vision.detect_block(frame, color)
            if det is None:
                print(f"in_reach: False  ({color} not detected)")
            elif det.within_reach:
                print(f"in_reach: TRUE   area={int(det.area)}  cy={det.cy}  err={det.centroid_error:+d}px")
            else:
                print(f"in_reach: False  area={int(det.area)}  cy={det.cy}  err={det.centroid_error:+d}px")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
    finally:
        vision.release_camera(picam2)
        print("[IN_REACH] Done")


# Vision detect test

def run_detect_test() -> None:
    from fp import vision

    print("[DETECT] Initialising camera...")
    picam2 = vision.init_camera()
    print("[DETECT] Camera ready.  Scanning for blocks - press Ctrl-C to stop.\n")

    try:
        while True:
            frame = vision.capture_frame(picam2)
            det   = vision.detect_block(frame)
            if det is None:
                print("detected: None")
            else:
                reach = "IN REACH" if det.within_reach else f"err={det.centroid_error:+d}px"
                print(f"detected: {det.label:<6}  area={int(det.area):>6}  cy={det.cy:>3}  {reach}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print()
    finally:
        vision.release_camera(picam2)
        print("[DETECT] Camera released")


# QR detection test

def run_qr_test() -> None:
    from fp import qr_detection
    from fp import vision

    print("[QR] Initialising camera...")
    picam2 = vision.init_camera()
    print("[QR] Camera ready.  Point at a QR code - press Ctrl-C to stop.\n")

    try:
        while True:
            frame = vision.capture_frame(picam2)
            if frame is None:
                time.sleep(0.1)
                continue
            result = qr_detection.detect_qr(frame)
            if result is None:
                print("QR: not detected")
            else:
                x, y, theta = result
                theta_str = f"{theta:.1f} deg" if theta is not None else "N/A"
                print(f"QR: x={x:.1f} cm  y={y:.1f} cm  heading={theta_str}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print()
    finally:
        vision.release_camera(picam2)
        print("[QR] Done")


# Planner visualisation test

def run_planner_test() -> None:
    import math
    from fp import planner as planner_module
    from fp import arena

    grid = planner_module.OccupancyGrid()

    start_x = arena.LAUNCH_ZONE_SIZE_CM / 2
    start_y = arena.LAUNCH_ZONE_SIZE_CM / 2
    goal_x  = arena.COLLECTION_ZONE_SIZE_CM / 2
    goal_y  = arena.ARENA_NORTH_WALL_Y_CM - arena.COLLECTION_ZONE_SIZE_CM / 2

    grid.log_traversed(start_x, start_y, radius_cm=40.0)
    grid.log_traversed(goal_x,  goal_y,  radius_cm=40.0)

    steps = 40
    for i in range(steps + 1):
        t  = i / steps
        grid.log_traversed(
            start_x + t * (goal_x - start_x),
            start_y + t * (goal_y - start_y),
            radius_cm=15.0,
        )

    print(f"[PLAN] Start: ({start_x:.1f}, {start_y:.1f}) cm")
    print(f"[PLAN] Goal:  ({goal_x:.1f}, {goal_y:.1f}) cm")
    print("[PLAN] Running A*...")

    path = grid.astar((start_x, start_y), (goal_x, goal_y))
    if path is None:
        print("[PLAN] A* returned no path (check that enough cells are FREE)")
    else:
        print(f"[PLAN] Path computed: {len(path)} waypoints")
        for i, (wx, wy) in enumerate(path):
            print(f"  waypoint {i:2d}: ({wx:.1f}, {wy:.1f}) cm")

    print("\n[PLAN] Occupancy grid (R=robot, *=path, #=wall, .=free, ?=unknown):")
    print(grid.render_ascii(robot_world=(start_x, start_y), path=path))
    print("\n[PLAN] Done")


# Camera calibration wrapper

def run_camera_cal() -> None:
    import os
    import importlib.util

    cal_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "calibration", "calibrate_camera.py",
    )
    spec = importlib.util.spec_from_file_location("calibrate_camera", cal_script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run()
