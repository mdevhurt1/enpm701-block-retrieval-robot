"""
fp/fsm/states.py - State enum, shared helpers, per-handler functions, and
per-state classes for the Grand Challenge robot FSM.

All tuning values are read from fp.tuning at call time (not at import time)
so that tuning.set_param() mutations take effect without a restart.
"""

import math
import logging
import os
import threading
import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from fp import hal
from fp import arena
from fp import navigation
from fp import vision
from fp import ekf as ekf_module
from fp import planner as planner_module
from fp import qr_detection
from fp import tuning
from fp.fsm.context import RobotContext, FSMTestHooks

logger = logging.getLogger(__name__)

# State enum

class State(Enum):
    INIT     = auto()
    DISCOVER = auto()
    APPROACH = auto()
    GRAB     = auto()
    DELIVER  = auto()
    AVOID    = auto()
    UPDATE   = auto()
    COMPLETE = auto()
    DONE     = auto()

# Design-time constants (not runtime-tunable)

COLOR_ORDER  = ["red", "green", "blue"]
TOTAL_BLOCKS = 9

SEARCH_HEADINGS_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
MAX_SEARCH_SWEEPS   = 6

WALL_HEADING = {"south": 180.0, "north": 0.0, "east": 90.0, "west": 270.0}

P0_DIAG = [9.0, 9.0, math.radians(2.0) ** 2]

# Shared helpers

def _pose() -> str:
    x, y, hdg = navigation.get_pose()
    return f"[pose ({x:.1f}, {y:.1f}) {hdg:.1f} deg]"


def _fmt_xy(point) -> str:
    if not isinstance(point, tuple) or len(point) != 2:
        return "None"
    return f"({point[0]:.1f}, {point[1]:.1f})"


def _fmt_cov_diag(ekf: Optional[ekf_module.EKF]) -> str:
    if ekf is None:
        return "None"
    diag = np.diag(ekf.get_covariance())
    return f"[{diag[0]:.2f}, {diag[1]:.2f}, {diag[2]:.4f}]"


def _pose_delta(before: tuple, after: tuple) -> tuple:
    pos_delta_cm = math.hypot(after[0] - before[0], after[1] - before[1])
    heading_delta_deg = abs((after[2] - before[2] + 180.0) % 360.0 - 180.0)
    return pos_delta_cm, heading_delta_deg


def _sync(ctx: RobotContext):
    ctx.x, ctx.y, _ = navigation.get_pose()


def _nearest_wall() -> tuple:
    x, y, _ = navigation.get_pose()
    return min(
        [
            ("south", y                            - arena.ARENA_SOUTH_WALL_Y_CM),
            ("north", arena.ARENA_NORTH_WALL_Y_CM  - y),
            ("west",  x                            - arena.ARENA_WEST_WALL_X_CM),
            ("east",  arena.ARENA_EAST_WALL_X_CM   - x),
        ],
        key=lambda t: t[1],
    )


def _obstacle_detected(picam2, target_color: str) -> bool:
    """Return True if a non-target block is at least as close as the target block."""
    frame = vision.capture_frame(picam2)
    if frame is None:
        return False

    detections = vision.detect_all_colors(frame)
    target_det = detections.get(target_color)
    if target_det is None:
        return False
    for color, det in detections.items():
        if color == target_color:
            continue
        if det is not None and det.area >= target_det.area:
            return True
    return False


def _correction_due(ctx: RobotContext) -> Optional[str]:
    if ctx.ekf is not None:
        if ctx.ekf.max_variance() <= tuning.P_CORRECTION_THRESHOLD:
            return None
    else:
        if time.monotonic() - ctx.last_update_time < tuning.UPDATE_INTERVAL_S:
            return None

    wall, dist = _nearest_wall()
    if dist > tuning.NEAR_WALL_THRESHOLD_CM:
        return None
    if time.monotonic() < ctx._wall_update_cooldown.get(wall, 0.0):
        return None
    return wall

# State handlers

def _handle_init(ctx: RobotContext) -> State:
    if ctx.hooks is None or not ctx.hooks.skip_init_calibration:
        logger.info("[INIT] Calibrating initial pose (robot must be in launch zone facing south)...")
        result = navigation.calibrate_initial_pose()
        if result is None:
            logger.warning(f"[INIT] Calibration failed - starting at (0, 0)  {_pose()}")
            init_x, init_y, init_hdg = 0.0, 0.0, 0.0
        else:
            init_x, init_y, init_hdg = result

        P0 = np.diag(P0_DIAG)
        ctx.ekf = ekf_module.EKF(x0=init_x, y0=init_y, heading_deg=init_hdg, P0=P0)
        navigation.init_ekf(ctx.ekf)
        navigation.start_odometry(rate_hz=100, x=init_x, y=init_y)

        ctx.planner = planner_module.OccupancyGrid()
        ctx.planner.log_traversed(init_x, init_y)

        ctx.zone_x = round(arena.COLLECTION_ZONE_SIZE_CM / 2, 1)
        ctx.zone_y = round(arena.ARENA_NORTH_WALL_Y_CM - arena.COLLECTION_ZONE_SIZE_CM / 2, 1)

    ctx.home_heading     = 0.0
    ctx.run_start        = time.monotonic()
    ctx.last_update_time = time.monotonic()
    _sync(ctx)
    hal.gripper_open()

    logger.info(f"[INIT] Home: {ctx.home_heading:.1f} deg  Zone: ({ctx.zone_x}, {ctx.zone_y})  "
                f"Target: {ctx.target_color}  {_pose()}")
    return State.DISCOVER


def _handle_discover(ctx: RobotContext) -> State:
    if ctx.planner is None and ctx.sweep_count >= MAX_SEARCH_SWEEPS:
        logger.info(f"[DISCOVER] Max sweeps reached - completing run  {_pose()}")
        return State.COMPLETE

    if ctx.time_expired():
        logger.info(f"[DISCOVER] Time limit - completing run  {_pose()}")
        return State.COMPLETE

    wall = _correction_due(ctx)
    if wall:
        ctx.return_state = State.DISCOVER
        return State.UPDATE

    _sync(ctx)
    cx, cy = ctx.x, ctx.y

    frame = vision.capture_frame(ctx.picam2)

    if frame is not None:
        _color_detections = vision.detect_all_colors(frame)

        if ctx.ekf is not None:
            qr = qr_detection.detect_qr(frame)
            if qr is not None:
                x_qr, y_qr, theta_qr = qr
                R_qr = np.diag([9.0, 9.0, math.radians(2.0) ** 2])
                ctx.ekf.update_qr(x_qr, y_qr, theta_qr, R_qr)
                _sync(ctx)
                ctx.qr_last_fix_time = time.monotonic()

        if ctx.planner is not None:
            for color, det in _color_detections.items():
                if det is None:
                    continue
                pose_est = vision.estimate_block_pose(frame, det,
                                                      robot_pose=(cx, cy,
                                                                  navigation.get_arena_heading()))
                if pose_est is not None:
                    bx, by = pose_est
                    ctx.planner.log_block(bx, by, color)

        det = _color_detections.get(ctx.target_color)
        if det is not None:
            logger.info("[DISCOVER] Spotted %s  area=%d  err=%+dpx  %s",
                        det.label, int(det.area), det.centroid_error, _pose())
            ctx.current_path = []
            return State.APPROACH

    if ctx.planner is not None and not ctx.current_path:
        goal = ctx.planner.get_next_goal(ctx.target_color, (cx, cy))
        if goal is not None:
            path = ctx.planner.astar((cx, cy), goal)
            plan_debug = ctx.planner.get_last_astar_debug()
            snapped_goal = plan_debug.get("planned_goal_world")
            first_waypoint = plan_debug.get("first_waypoint_world")
            goal_was_snapped = (
                isinstance(snapped_goal, tuple)
                and len(snapped_goal) == 2
                and math.hypot(snapped_goal[0] - goal[0], snapped_goal[1] - goal[1]) > 1e-6
            )
            if goal_was_snapped:
                logger.info(
                    f"[DISCOVER] A* adjusted frontier {_fmt_xy(goal)} -> goal {_fmt_xy(snapped_goal)} "
                    f"first {_fmt_xy(first_waypoint)}  {_pose()}"
                )
            if path is None:
                logger.warning(
                    f"[DISCOVER] A* found no path to ({goal[0]:.1f}, {goal[1]:.1f}) "
                    f"- marking traversed and retrying  {_pose()}"
                )
                ctx.planner.log_traversed(goal[0], goal[1])
                return State.DISCOVER
            if not path:
                reason = "snapped to current cell via A*" if goal_was_snapped else "collapsed to current pose"
                logger.warning(
                    f"[DISCOVER] Frontier {_fmt_xy(goal)} {reason} "
                    f"(goal {_fmt_xy(snapped_goal)} first {_fmt_xy(first_waypoint)}) "
                    f"- marking traversed and retrying  {_pose()}"
                )
                ctx.planner.log_traversed(goal[0], goal[1])
                return State.DISCOVER
            ctx.current_path = path
            logger.info(f"[DISCOVER] New goal ({goal[0]:.1f}, {goal[1]:.1f})  "
                        f"path_len={len(ctx.current_path)}  {_pose()}")
        else:
            logger.info(f"[DISCOVER] Map fully explored, no target found  {_pose()}")
            return State.COMPLETE

    if ctx.current_path:
        wx, wy = ctx.current_path.pop(0)
        if math.hypot(wx - cx, wy - cy) > 1.0:
            navigation.navigate_to(wx, wy, cx, cy, correct_on_arrival=False)
            _sync(ctx)
        if ctx.planner is not None:
            ctx.planner.log_traversed(ctx.x, ctx.y)
            d_front = hal.distance_front()
            if d_front is not None:
                ctx.planner.update_from_ultrasonic(
                    navigation.get_pose(), d_front,
                    navigation.get_arena_heading()
                )
        return State.DISCOVER

    target_hdg = (ctx.home_heading + SEARCH_HEADINGS_DEG[ctx.sweep_index]) % 360
    navigation.pivot_to_heading(target_hdg)
    time.sleep(tuning.SEARCH_PIVOT_DWELL_S)

    frame2 = vision.capture_frame(ctx.picam2)
    det2   = vision.detect_block(frame2, ctx.target_color) if frame2 is not None else None
    if det2 is not None:
        logger.info(f"[DISCOVER] Spotted {det2.label} after pivot  area={int(det2.area)}  {_pose()}")
        ctx.current_path = []
        return State.APPROACH

    ctx.sweep_index += 1
    if ctx.sweep_index >= len(SEARCH_HEADINGS_DEG):
        ctx.sweep_index = 0
        ctx.sweep_count += 1
        logger.info(f"[DISCOVER] Full sweep #{ctx.sweep_count} - advancing "
                    f"{tuning.SEARCH_ADVANCE_CM} cm  {_pose()}")
        navigation.pivot_to_heading(ctx.home_heading)
        navigation.move_forward(tuning.SEARCH_ADVANCE_CM)
        _sync(ctx)
        if ctx.planner is not None:
            ctx.planner.log_traversed(ctx.x, ctx.y)

    return State.DISCOVER


def _handle_approach(ctx: RobotContext) -> State:
    if _obstacle_detected(ctx.picam2, ctx.target_color):
        logger.info(f"[APPROACH] Obstacle detected - AVOID  {_pose()}")
        ctx.return_state = State.DISCOVER
        return State.AVOID

    logger.info(f"[APPROACH] Approaching {ctx.target_color}  {_pose()}")
    success = navigation.approach_visual(ctx.picam2, ctx.target_color)
    _sync(ctx)

    if success:
        ctx.approach_fail_count = 0
        logger.info(f"[APPROACH] Block reached  {_pose()}")
        return State.GRAB

    ctx.approach_fail_count += 1
    if ctx.approach_fail_count >= 3:
        logger.warning(
            f"[APPROACH] Failed {ctx.approach_fail_count} times in a row - "
            f"clearing path and forcing re-explore  {_pose()}"
        )
        ctx.approach_fail_count = 0
        ctx.current_path = []
        ctx.sweep_index = (ctx.sweep_index + 1) % len(SEARCH_HEADINGS_DEG)

    logger.warning(f"[APPROACH] Block lost - DISCOVER  {_pose()}")
    return State.DISCOVER


def _handle_grab(ctx: RobotContext) -> State:
    logger.info(f"[GRAB] Closing gripper  {_pose()}")
    hal.gripper_close()
    time.sleep(0.5)

    _sync(ctx)
    grab_x, grab_y = ctx.x, ctx.y

    navigation.move_reverse(tuning.BACKUP_AFTER_GRAB_CM)
    _sync(ctx)
    logger.info(f"[GRAB] Backed up  {_pose()}")

    frame = vision.capture_frame(ctx.picam2)
    det = vision.detect_block(frame, ctx.target_color) if frame is not None else None
    if det is None:
        logger.info(f"[GRAB] Block not detected after backup - dropped?  {_pose()}")
        hal.gripper_open()
        return State.DISCOVER

    hal.set_carrying_block(True)

    if ctx.planner is not None:
        ctx.planner.clear_block(grab_x, grab_y)

    return State.DELIVER


def _deliver_drive_to_zone(ctx: RobotContext) -> str:
    """Navigate to the delivery zone. Returns 'arrived', 'failed', or 'expired'."""
    cx, cy, _ = navigation.get_pose()
    logger.info(f"[DELIVER] Navigating to zone ({ctx.zone_x}, {ctx.zone_y})  {_pose()}")
    arrived = False

    if ctx.planner is not None:
        path = ctx.planner.astar((cx, cy), (ctx.zone_x, ctx.zone_y))
        if path:
            arrived = navigation.follow_path(path, ctx.planner, should_abort=ctx.time_expired)
        else:
            logger.info("[DELIVER] A* found no path - direct navigate_to fallback")
            _, _, outcome = navigation.navigate_to(ctx.zone_x, ctx.zone_y, cx, cy)
            arrived = outcome == 'success'
            if not arrived:
                logger.warning(f"[DELIVER] Direct navigate_to failed with {outcome}  {_pose()}")
    else:
        _, _, outcome = navigation.navigate_to(ctx.zone_x, ctx.zone_y, cx, cy)
        arrived = outcome == 'success'
        if not arrived:
            logger.warning(f"[DELIVER] Direct navigate_to failed with {outcome}  {_pose()}")

    _sync(ctx)
    if ctx.time_expired():
        logger.info(f"[DELIVER] Time expired during navigation - dropping block  {_pose()}")
        hal.stop_motors()
        hal.gripper_open()
        return "expired"
    if not arrived:
        hal.stop_motors()
        logger.warning(f"[DELIVER] Navigation to zone failed - retaining block and retrying  {_pose()}")
        return "failed"
    logger.info(f"[DELIVER] Arrived  {_pose()}")
    return "arrived"


def _deliver_align_at_zone(ctx: RobotContext) -> None:
    """EKF wall corrections at delivery zone (NW corner)."""
    if ctx.ekf is None or ctx.time_expired():
        return
    navigation.pivot_to_heading(270.0)
    if not ctx.time_expired():
        navigation.correct_pose_from_wall('west')
    if not ctx.time_expired():
        navigation.pivot_to_heading(0.0)
    if not ctx.time_expired():
        navigation.correct_pose_from_wall('north')
    _sync(ctx)
    logger.info(f"[DELIVER] Pose corrected at zone  {_pose()}")


def _deliver_bookkeeping(ctx: RobotContext) -> None:
    hal.gripper_open()
    time.sleep(0.5)

    ctx.score            += 1
    ctx.blocks_delivered += 1
    if not ctx.color_override:
        ctx.next_color_index += 1
    ctx.sweep_index  = 0
    ctx.sweep_count  = 0
    ctx.current_path = []

    if ctx.planner is not None:
        ctx.planner.log_traversed(ctx.zone_x, ctx.zone_y)

    logger.info(f"[DELIVER] Deposited - score: {ctx.score}  "
                f"next: {ctx.target_color}  {_pose()}")


def _deliver_async_image(ctx: RobotContext) -> None:
    """Capture a delivery frame and save it asynchronously on a tracked thread."""
    frame = vision.capture_frame(ctx.picam2)
    if frame is None:
        return
    img_path    = os.path.abspath(f"delivery_{ctx.blocks_delivered}.png")
    _frame_copy = frame.copy()
    t = threading.Thread(
        target=cv2.imwrite,
        args=(img_path, _frame_copy),
        daemon=False,
        name=f"img-save-{ctx.blocks_delivered}",
    )
    t.start()
    ctx._image_save_threads.append(t)
    logger.info(f"[DELIVER] Image save started: {img_path}")


def _deliver_safe_backup(ctx: RobotContext) -> None:
    """Back up a clearance-aware distance from the delivery zone."""
    bx, by, bhdg = navigation.get_pose()
    snapped = round(bhdg / 90) * 90 % 360
    if snapped == 0:
        clearance = by - arena.ARENA_SOUTH_WALL_Y_CM
    elif snapped == 180:
        clearance = arena.ARENA_NORTH_WALL_Y_CM - by
    elif snapped == 90:
        clearance = bx - arena.ARENA_WEST_WALL_X_CM
    else:
        clearance = arena.ARENA_EAST_WALL_X_CM - bx
    safe_backup = max(
        0.0,
        min(tuning.BACKUP_AFTER_DEPOSIT_CM, clearance - tuning.BACKUP_WALL_MARGIN_CM),
    )
    logger.info(f"[DELIVER] Backing up {safe_backup:.1f} cm (clearance {clearance:.1f} cm)  {_pose()}")
    navigation.move_reverse(safe_backup)
    _sync(ctx)
    ctx.last_update_time = time.monotonic()


def _handle_deliver(ctx: RobotContext) -> State:
    if ctx.time_expired():
        logger.warning(f"[DELIVER] Time expired before delivery navigation - skipping  {_pose()}")
        hal.gripper_open()
        return State.COMPLETE

    wall = _correction_due(ctx)
    if wall:
        ctx.return_state = State.DELIVER
        return State.UPDATE

    drive_status = _deliver_drive_to_zone(ctx)
    if drive_status == "expired":
        return State.COMPLETE
    if drive_status == "failed":
        return State.DELIVER

    _deliver_align_at_zone(ctx)
    _deliver_bookkeeping(ctx)
    _deliver_async_image(ctx)
    _deliver_safe_backup(ctx)

    if ctx.blocks_delivered >= TOTAL_BLOCKS:
        logger.info(f"[DELIVER] All {TOTAL_BLOCKS} blocks collected!  {_pose()}")
        return State.COMPLETE

    if ctx.hooks is not None and ctx.hooks.stop_after_delivery:
        logger.info(f"[DELIVER] Test complete  {_pose()}")
        return State.COMPLETE

    return State.DISCOVER


def _handle_avoid(ctx: RobotContext) -> State:
    _sync(ctx)
    pre_x, pre_y = ctx.x, ctx.y

    logger.info(f"[AVOID] Backing up {tuning.AVOID_BACKUP_CM} cm  {_pose()}")
    navigation.move_reverse(tuning.AVOID_BACKUP_CM)
    _sync(ctx)

    if ctx.planner is not None:
        hdg_rad = math.radians(navigation.get_arena_heading())
        obs_x = pre_x + 30.0 * math.sin(hdg_rad)
        obs_y = pre_y + 30.0 * math.cos(hdg_rad)
        ctx.planner.log_block(obs_x, obs_y, "_obstacle")

    ctx.current_path = []

    if ctx.planner is not None:
        cx, cy, _ = navigation.get_pose()
        goal = ctx.planner.get_next_goal(ctx.target_color, (cx, cy))
        if goal is not None:
            new_path = ctx.planner.astar((cx, cy), goal)
            ctx.current_path = new_path if new_path else []
    else:
        logger.info(f"[AVOID] Pivoting {tuning.AVOID_PIVOT_DEG} deg right  {_pose()}")
        navigation.pivot_right(tuning.AVOID_PIVOT_DEG)

    ctx.sweep_index = 0
    logger.info(f"[AVOID] Resuming {ctx.return_state.name}  {_pose()}")
    return ctx.return_state


def _handle_update(ctx: RobotContext) -> State:
    wall, dist = _nearest_wall()
    logger.info(
        f"[UPDATE] {wall} wall {dist:.1f} cm away - correcting  "
        f"diag(P)={_fmt_cov_diag(ctx.ekf)}  {_pose()}"
    )

    if ctx.time_expired():
        logger.warning(f"[UPDATE] Time expired - skipping correction  {_pose()}")
        return ctx.return_state

    navigation.pivot_to_heading(WALL_HEADING[wall])

    if ctx.time_expired():
        logger.warning(f"[UPDATE] Time expired after pivot - skipping wall read  {_pose()}")
        return ctx.return_state

    pre_correction_pose = navigation.get_pose()
    ok = navigation.correct_pose_from_wall(wall)

    if ok:
        post_correction_pose = navigation.get_pose()
        pos_delta_cm, heading_delta_deg = _pose_delta(pre_correction_pose, post_correction_pose)
        _sync(ctx)
        ctx.last_update_time = time.monotonic()
        if (pos_delta_cm <= tuning.WALL_SUCCESS_MAX_DELTA_CM
                and heading_delta_deg <= tuning.WALL_SUCCESS_MAX_HEADING_DEG):
            until = time.monotonic() + tuning.WALL_SUCCESS_COOLDOWN_S
            ctx._wall_update_cooldown[wall] = until
            logger.info(
                f"[UPDATE] Small correction Δpos={pos_delta_cm:.2f} cm  "
                f"Δhdg={heading_delta_deg:.2f} deg "
                f"- suppressing {wall} wall for {tuning.WALL_SUCCESS_COOLDOWN_S:.1f}s  {_pose()}"
            )
        else:
            ctx._wall_update_cooldown.pop(wall, None)
        logger.info(f"[UPDATE] Corrected  {_pose()}")
    else:
        ctx._wall_update_cooldown[wall] = time.monotonic() + tuning.WALL_COOLDOWN_S
        logger.warning(f"[UPDATE] Sensor failure - suppressing {wall} wall for "
                       f"{tuning.WALL_COOLDOWN_S:.0f}s  {_pose()}")

    if ctx.return_state not in (State.APPROACH,):
        navigation.pivot_to_heading(ctx.home_heading)
    return ctx.return_state


def _handle_complete(ctx: RobotContext) -> State:
    hal.gripper_open()
    logger.info(f"[COMPLETE] score={ctx.score}  blocks={ctx.blocks_delivered}  {_pose()}")
    return State.DONE

# Per-state classes

class _BaseState:
    def on_enter(self, ctx: RobotContext) -> None:
        pass

    def step(self, ctx: RobotContext) -> State:
        raise NotImplementedError

    def on_exit(self, ctx: RobotContext) -> None:
        pass


class InitState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_init(ctx)


class DiscoverState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_discover(ctx)


class ApproachState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_approach(ctx)


class GrabState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_grab(ctx)


class DeliverState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_deliver(ctx)


class AvoidState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_avoid(ctx)


class UpdateState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_update(ctx)


class CompleteState(_BaseState):
    def step(self, ctx: RobotContext) -> State:
        return _handle_complete(ctx)


_STATE_CLASSES: dict = {
    State.INIT:     InitState(),
    State.DISCOVER: DiscoverState(),
    State.APPROACH: ApproachState(),
    State.GRAB:     GrabState(),
    State.DELIVER:  DeliverState(),
    State.AVOID:    AvoidState(),
    State.UPDATE:   UpdateState(),
    State.COMPLETE: CompleteState(),
}
