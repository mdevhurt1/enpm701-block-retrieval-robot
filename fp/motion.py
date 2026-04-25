import math
import logging
import time
from typing import Optional, TYPE_CHECKING

from fp import hal
from fp import arena
from fp import tuning
from fp import vision
from fp import localization

if TYPE_CHECKING:
    from fp.planner import OccupancyGrid

logger = logging.getLogger(__name__)

# PID tuning constants

KP_DIST = 1.0
KD_DIST = 0.5
MAX_CORRECTION = 100.0

KP_HDG_HOLD = 1.5

KP_HDG = 0.5
KD_HDG = 0.001
MIN_TURN_DUTY = hal.PWM_DUTY_TURN

KP_VISUAL = 0.05

_WALL_ESCAPE_CM = 8.0


# Internal movement helpers

def _run_distance(direction: int, target_counts: int) -> str:
    """
    Drive forward (direction=+1) or backward (direction=-1) for target_counts
    encoder transitions, using encoder-balance PID to keep straight.

    Returns: 'success' | 'wall' | 'stall' | 'timeout'
    """
    if direction == +1:
        left_pin  = hal.LEFT_FORWARD
        right_pin = hal.RIGHT_FORWARD
    else:
        left_pin  = hal.LEFT_REVERSE
        right_pin = hal.RIGHT_REVERSE

    base_a, base_b = hal.read_encoders()

    lock_heading = None
    for i in range(5):
        lock_heading = hal.get_heading_or_none()
        if lock_heading is not None:
            break
        if i == 4:
            logger.warning("[NAV] heading returned none, aborting _run_distance")
            return 'stall'

    prev_error = 0.0
    prev_time  = time.monotonic()

    duty           = float(hal.PWM_DUTY_STRAIGHT)
    deadline       = time.monotonic() + max(5, target_counts * 0.5)
    last_enc_total = 0
    last_progress  = time.monotonic()

    outcome = 'timeout'
    try:
        while time.monotonic() < deadline:
            cur_a, cur_b = hal.read_encoders()
            counter_a = cur_a - base_a
            counter_b = cur_b - base_b

            if (counter_a + counter_b) / 2 >= target_counts:
                outcome = 'success'
                break

            if direction == +1 and hal.wall_too_close():
                logger.warning("[NAV] Wall too close - aborting forward move")
                outcome = 'wall'
                break

            cur_total = counter_a + counter_b
            if cur_total > last_enc_total:
                last_enc_total = cur_total
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > 2.0:
                logger.warning("[NAV] No encoder progress for 2s - aborting (stall?)")
                outcome = 'stall'
                break

            now = time.monotonic()
            dt  = now - prev_time
            if dt >= 0.001:
                error      = ((hal.get_heading() - lock_heading) + 180) % 360 - 180
                derivative = (error - prev_error) / dt
                pid        = KP_DIST * error + KD_DIST * derivative
                correction = max(-MAX_CORRECTION, min(MAX_CORRECTION, pid)) * direction

                left_duty  = max(0.0, min(100.0, duty - correction))
                right_duty = max(0.0, min(100.0, duty + correction))
                hal.set_individual_duties(left_pin, left_duty, right_pin, right_duty)

                prev_error = error
                prev_time  = now
                time.sleep(0.01)
    finally:
        hal.stop_motors()
    return outcome


def _run_heading(target_heading: float, direction: int) -> None:
    """
    Pivot in place until IMU heading is within hal.HEADING_TOLERANCE of target_heading.
    direction: +1 = clockwise (right), -1 = counterclockwise (left).
    """
    if direction == +1:
        left_pin  = hal.LEFT_FORWARD
        right_pin = hal.RIGHT_REVERSE
    else:
        left_pin  = hal.LEFT_REVERSE
        right_pin = hal.RIGHT_FORWARD

    prev_error = 0.0
    prev_time  = time.monotonic()
    deadline   = time.monotonic() + 10

    localization.loc.odo_pivoting = True
    try:
        while time.monotonic() < deadline:
            error = (target_heading - hal.get_heading() + 180) % 360 - 180

            if abs(error) <= hal.HEADING_TOLERANCE:
                break

            if direction * error < -hal.HEADING_TOLERANCE:
                break

            now = time.monotonic()
            dt  = now - prev_time
            if dt >= 0.001:
                derivative = (error - prev_error) / dt
                pid = KP_HDG * error + KD_HDG * derivative

                turn_duty = max(float(MIN_TURN_DUTY), min(100.0, abs(pid)))
                hal.set_individual_duties(left_pin, turn_duty, right_pin, turn_duty)

                prev_error = error
                prev_time  = now
                time.sleep(0.01)
    finally:
        hal.stop_motors()
        localization.loc.odo_pivoting = False
        # Snap the EKF to the actual post-pivot heading.  Mahalanobis gate
        # rejected heading updates during the turn; set_pose() bypasses it.
        if localization.loc.ekf is not None:
            hdg = localization.get_arena_heading_or_none()
            if hdg is not None:
                x, y, _ = localization.loc.ekf.get_pose()
                localization.loc.ekf.set_pose(x, y, hdg)


# Public movement API

def move_forward(distance_cm: float) -> str:
    """Drive forward distance_cm.  Returns 'success', 'wall', 'stall', or 'timeout'.

    On a wall-too-close abort the robot reverses _WALL_ESCAPE_CM to clear the
    inflated obstacle boundary so the next A* plan has a navigable cell.
    """
    target = max(1, int(distance_cm * hal.COUNTS_PER_CM))
    logger.info(f"[NAV] Forward {distance_cm:.1f} cm ({target} counts)")
    localization.loc.odo_direction = +1
    try:
        result = _run_distance(+1, target)
        logger.info(f"[NAV] Forward outcome: {result}")
    finally:
        localization.loc.odo_direction = 0

    if result == 'wall':
        logger.info(f"[NAV] Wall escape: reversing {_WALL_ESCAPE_CM:.1f} cm to clear inflated zone")
        move_reverse(_WALL_ESCAPE_CM)

    return result


def _clamp_reverse_distance(distance_cm: float) -> float:
    """Reduce reverse distance if odometry says robot would back into a wall."""
    x, y, hdg = localization.get_pose()
    rear_hdg_rad = math.radians((hdg + 180) % 360)
    rear_x = math.sin(rear_hdg_rad)
    rear_y = math.cos(rear_hdg_rad)

    wall_distances = []
    if rear_y > 1e-6:
        wall_distances.append((arena.ARENA_NORTH_WALL_Y_CM - y) / rear_y)
    elif rear_y < -1e-6:
        wall_distances.append((y - arena.ARENA_SOUTH_WALL_Y_CM) / (-rear_y))
    if rear_x > 1e-6:
        wall_distances.append((arena.ARENA_EAST_WALL_X_CM - x) / rear_x)
    elif rear_x < -1e-6:
        wall_distances.append((x - arena.ARENA_WEST_WALL_X_CM) / (-rear_x))

    if not wall_distances:
        return distance_cm

    clearance = min(d for d in wall_distances if d > 0) if any(d > 0 for d in wall_distances) else 0.0
    safe = max(0.0, clearance - arena.ROBOT_SENSOR_OFFSET_CM - hal.WALL_CLEARANCE_CM)
    if distance_cm > safe:
        logger.info(f"[NAV] Clamping reverse {distance_cm:.1f} -> {safe:.1f} cm (clearance {clearance:.1f} cm)")
        return safe
    return distance_cm


def move_reverse(distance_cm: float) -> None:
    distance_cm = _clamp_reverse_distance(distance_cm)
    if distance_cm < 1.0:
        logger.warning("[NAV] Reverse skipped - too close to wall")
        return
    target = max(1, int(distance_cm * hal.COUNTS_PER_CM))
    logger.info(f"[NAV] Reverse {distance_cm:.1f} cm ({target} counts)")
    localization.loc.odo_direction = -1
    try:
        _run_distance(-1, target)
    finally:
        localization.loc.odo_direction = 0


def pivot_to_heading(target_arena_deg: float) -> None:
    """Pivot to an absolute arena heading (0 deg = north / collection-zone direction)."""
    target_imu  = localization._arena_to_imu(target_arena_deg)
    current_imu = hal.get_heading()
    error = (target_imu - current_imu + 180) % 360 - 180
    if abs(error) <= hal.HEADING_TOLERANCE:
        return
    direction = +1 if error > 0 else -1
    label = "right" if direction == +1 else "left"
    logger.info(f"[NAV] Pivot {label} -> {target_arena_deg:.1f} deg arena ({target_imu:.1f} deg IMU, Δ{error:+.1f} deg)")
    _run_heading(target_imu, direction)


def pivot_left(angle_deg: float) -> None:
    target = (hal.get_heading() - angle_deg) % 360
    logger.info(f"[NAV] Pivot left {angle_deg:.1f} deg -> {target:.1f} deg")
    _run_heading(target, -1)


def pivot_right(angle_deg: float) -> None:
    target = (hal.get_heading() + angle_deg) % 360
    logger.info(f"[NAV] Pivot right {angle_deg:.1f} deg -> {target:.1f} deg")
    _run_heading(target, +1)


# Visual approach

def approach_visual(picam2, target_color: Optional[str] = None, max_steps: int = 200,
                    max_time_s: float = 30.0) -> bool:
    """
    Drive toward a detected block using camera centroid error for steering.
    Returns True when within_reach, False on block loss, wall, or timeout.
    """
    STEP_CM   = 2.0
    BASE_DUTY = float(hal.PWM_DUTY_STRAIGHT)
    MAX_LOST_FRAMES = 3

    last_det = None
    consecutive_lost = 0
    approach_deadline = time.monotonic() + max_time_s

    for _ in range(max_steps):
        if time.monotonic() >= approach_deadline:
            logger.warning(f"[NAV] approach_visual timeout after {max_time_s:.0f}s - stopping")
            return False
        frame = vision.capture_frame(picam2)
        if frame is None:
            logger.error("[NAV] Camera frame capture failed during approach - stopping")
            return False
        det = vision.detect_block(frame, target_color)

        if det is None:
            if last_det is not None and last_det.within_reach:
                logger.warning("[NAV] Block lost at close range - treating as in reach")
                return True
            consecutive_lost += 1
            if consecutive_lost < MAX_LOST_FRAMES:
                logger.warning(
                    f"[NAV] Block not detected ({consecutive_lost}/{MAX_LOST_FRAMES}) "
                    f"- retrying"
                )
                continue
            logger.warning("[NAV] Block lost during approach - stopping")
            return False

        consecutive_lost = 0
        last_det = det

        if det.within_reach:
            logger.info(f"[NAV] Block within reach (area={int(det.area)})")
            return True

        if hal.wall_too_close():
            logger.warning("[NAV] Wall too close during visual approach - backing up and stopping")
            hal.stop_motors()
            move_reverse(8.0)
            return False

        steering   = KP_VISUAL * det.centroid_error
        left_duty  = max(0.0, min(100.0, BASE_DUTY + steering))
        right_duty = max(0.0, min(100.0, BASE_DUTY - steering))

        target_counts = max(1, int(STEP_CM * hal.COUNTS_PER_CM))

        try:
            localization.loc.odo_direction = +1
            hal.set_individual_duties(
                hal.LEFT_FORWARD, left_duty,
                hal.RIGHT_FORWARD, right_duty,
            )

            base_a, base_b = hal.read_encoders()
            step_deadline  = time.monotonic() + 3.0

            while time.monotonic() < step_deadline:
                cur_a, cur_b = hal.read_encoders()
                if (cur_a - base_a + cur_b - base_b) / 2 >= target_counts:
                    break
        finally:
            hal.stop_motors()
            localization.loc.odo_direction = 0

    logger.info("[NAV] Visual approach: max steps reached without reaching block")
    return False


# Navigate to a known (x, y) coordinate using odometry + IMU

def navigate_to(target_x: float, target_y: float, pos_x: float, pos_y: float,
                correct_on_arrival: bool = True):
    """
    Drive from (pos_x, pos_y) to (target_x, target_y).
    Returns (pos_x, pos_y, outcome) where outcome is 'success', 'wall', 'stall', or 'timeout'.
    """
    dx = target_x - pos_x
    dy = target_y - pos_y
    distance = math.hypot(dx, dy)
    if distance < 1.0:
        return pos_x, pos_y, 'success'

    heading_to_target = math.degrees(math.atan2(dx, dy)) % 360
    pivot_to_heading(heading_to_target)
    outcome = move_forward(distance)

    if outcome != 'success' or not correct_on_arrival:
        x, y = localization.get_pose()[:2]
        return x, y, outcome

    dist_north = arena.ARENA_NORTH_WALL_Y_CM - target_y
    dist_south = target_y - arena.ARENA_SOUTH_WALL_Y_CM
    dist_east  = arena.ARENA_EAST_WALL_X_CM  - target_x
    dist_west  = target_x  - arena.ARENA_WEST_WALL_X_CM

    wall_name, _, wall_heading = min(
        ('north', dist_north,  0.0),
        ('south', dist_south, 180.0),
        ('east',  dist_east,   90.0),
        ('west',  dist_west,  270.0),
        key=lambda t: t[1],
    )

    expected_sensor = min(dist_north, dist_south, dist_east, dist_west) \
                      + arena.ROBOT_SENSOR_OFFSET_CM
    logger.info(f"[NAV] Arrival check - {wall_name} wall, "
          f"expected sensor distance: {expected_sensor:.1f} cm")

    pivot_to_heading(wall_heading)
    localization.correct_pose_from_wall(wall_name)

    x, y = localization.get_pose()[:2]
    return x, y, outcome


# follow_path: sequential waypoint navigation with dynamic replanning

def follow_path(waypoints: list, planner: "OccupancyGrid",
                should_abort=None) -> bool:
    """
    Navigate through a sequence of world (x, y) waypoints.

    Returns True when the final waypoint is reached, False if replanning fails
    or should_abort fires.
    """
    ARRIVAL_TOLERANCE_CM = 8.0

    if not waypoints:
        return True

    remaining  = list(waypoints)
    final_goal = remaining[-1]

    while remaining:
        if should_abort is not None and should_abort():
            logger.warning("[NAV] follow_path: abort requested - stopping")
            hal.stop_motors()
            return False

        wx, wy = remaining[0]
        cx, cy, _ = localization.get_pose()

        if math.hypot(wx - cx, wy - cy) < ARRIVAL_TOLERANCE_CM:
            remaining.pop(0)
            if planner is not None:
                planner.log_traversed(cx, cy)
            continue

        _, _, outcome = navigate_to(
            wx, wy, cx, cy, correct_on_arrival=len(remaining) == 1
        )
        cx, cy, _ = localization.get_pose()

        if planner is not None:
            planner.log_traversed(cx, cy)

        if outcome != 'success':
            logger.warning(
                f"[NAV] follow_path: waypoint ({wx:.1f}, {wy:.1f}) failed with {outcome}"
            )
            return False

        remaining.pop(0)

        if remaining and planner is not None:
            next_goal = remaining[-1] if len(remaining) == 1 else final_goal
            new_path = planner.astar((cx, cy), next_goal)
            if new_path is None:
                logger.error(f"[NAV] follow_path: cannot reach ({next_goal[0]:.1f}, {next_goal[1]:.1f}) - aborting")
                return False
            if len(new_path) != len(remaining):
                logger.info(f"[NAV] follow_path: replanned  {len(remaining)} -> {len(new_path)} waypoints")
            remaining = new_path

    return True
