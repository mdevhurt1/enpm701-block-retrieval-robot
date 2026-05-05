import logging
import math
import time
from typing import Iterable, Optional

from fp import arena
from fp import hal
from fp import vision
from fp.localization import Localizer, ROBOT_HALF_EXTENT_CM

logger = logging.getLogger(__name__)

class TargetDetected(Exception):
    """Raised when a drive is aborted because the target color is visible."""
    def __init__(self, detection: vision.DetectionResult):
        super().__init__(f"target {detection.label!r} detected")
        self.detection = detection

class ApproachVisualStuck(Exception):
    def __init__(self):
        super().__init__("[MOTION] approach_visual is stuck")


# Tuning constants
LOOP_SLEEP_S = 0.025

# Drive tuning constants
STALL_TIMEOUT_S = 1.0
DRIVE_AROUND_LATERAL_CM = 30.0
DRIVE_AROUND_MAX_DEPTH  = 2
HEADING_HOLD_KP = 1.0
MAX_HEADING_CORRECTION = 25.0
DRIVE_DEADLINE_MIN_S = 4.0
DRIVE_DEADLINE_PER_CM_S = 0.25
REVERSE_DISTANCE_CM = 10.0
REVERSE_REPLAN_ATTEMPTS = 1
# Reverses get a tighter validation envelope than forward moves: the
# ultrasonic only sees forward, so the rear has no live sensing, and
# observed odometry drift (the y-axis-stuck-south failure mode) means
# is_valid_point's odometry-based bounds can pass a reverse that
# physically lands at the wall. This extra margin is shaved off both
# sides of the safe rectangle when validating reverse endpoints.
REVERSE_EXTRA_MARGIN_CM = 10.0

# Pivot tuning constants
MAX_PIVOT_ATTEMPTS      = 3
SLOWDOWN_THRESHOLD_DEG  = 35.0
MIN_PIVOT_DUTY          = 25.0
PIVOT_DEADLINE_S        = 6.0
TIME_BETWEEN_PIVOTS     = 0.3
HEADING_TOLERANCE       = 1.0

# Localization tuning constants
MAX_LOCALIZATION_RANGE_CM = 50.0
SWEEP_RANGE_DEG           = 30
SWEEP_STEP_DEG            = 10
SWEEP_SAMPLES             = 3
MAX_VALID_SENSOR_CM       = 60.0
MIN_VALID_SENSOR_CM       = 10.0
WALL_FIT_MAX_RMS_CM       = 3.0
# Must remain <= SWEEP_RANGE_DEG and ideally well below it: a fit whose
# theta_perp lies near or beyond the geometric edge of the sweep is
# extrapolating, not interpolating, and is typically a corner-confusion
# artifact rather than a real wall measurement.
WALL_FIT_HEADING_GATE_DEG       = 15.0
HEADING_RECAL_GATE_DEG          = 10.0
MIN_TIME_BETWEEN_LOCALIZATIONS  = 20.0

# Visual approach tuning constants
APPROACH_LARGEST_STEP       = 20.0
VISUAL_APPROACH_TOLERANCE   = 1.0
VISUAL_APPROACH_STEER_GAIN = 0.85

# Internal helper functions

def _drive(localizer: Localizer, distance_cm: float, initial_heading: float, 
           direction: int, target: str = None, visual_approach: bool = False, 
           carrying_block: bool = False) -> bool:
    if direction not in (+1, -1):
        logger.info(f"[MOTION] _drive: invalid direction {direction}")
        return False
    
    logger.info(f"[MOTION] _drive {distance_cm:.1f} cm direction {direction:+d}")

    if localizer.tracker is not None:
        localizer.tracker.start_segment()
    
    target_counts = max(1, int(distance_cm * hal.COUNTS_PER_CM))
    base_left, base_right = hal.read_front_left_encoder(), hal.read_back_right_encoder()

    base_duty = float(hal.PWM_DUTY_STRAIGHT)

    deadline = time.monotonic() + max(DRIVE_DEADLINE_MIN_S, distance_cm * DRIVE_DEADLINE_PER_CM_S)
    last_total = 0
    last_average = 0.0
    wall_trigger_count = 0

    try:
        last_progress_t = time.monotonic()

        while time.monotonic() < deadline:
            current_left, current_right = hal.read_front_left_encoder(), hal.read_back_right_encoder()
            counter_left = current_left - base_left
            counter_right = current_right - base_right
            current_total = counter_left + counter_right
            average = current_total / 2

            delta_cm = (average - last_average) / hal.COUNTS_PER_CM
            last_average = average
            
            current_heading = hal.get_heading()
            localizer.set_heading(current_heading)
            if direction == +1:
                localizer.apply_forward(delta_cm, current_heading)
            else:
                localizer.apply_reverse(delta_cm, current_heading)

            if average >= target_counts:
                return True
            
            # Median of the rolling buffer instead of a raw read - one stuck-high
            # or missed-echo sample would otherwise look like a near-wall return
            # and trip the two-consecutive-readings abort below.
            ultrasonic_distance = _buffered_ultrasonic_median()

            if direction == +1:
                if ultrasonic_distance is not None and localizer.ultrasonic_reading_inside_robot_radius(ultrasonic_distance, current_heading):
                    wall_trigger_count += 1
                else:
                    wall_trigger_count = 0
                if wall_trigger_count >= 2:
                    logger.info(f"[MOTION] _drive wall too close ({wall_trigger_count} consecutive readings), reading: {ultrasonic_distance:.1f} cm at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}, {localizer.pose.heading:.1f}), aborting drive")

                    hal.stop_motors()

                    if visual_approach:
                        backoff_result = move_reverse(localizer)

                    localize(localizer, force=localizer.pose.heading)

                    if not visual_approach:
                        backoff_result = move_reverse(localizer)

                    return False

                if not arena.in_collection_zone(localizer.pose.x_cm, localizer.pose.y_cm):
                    detections = vision.detect_all_colors()
                    target_detection = detections.get(target)
                    
                    for color in list(detections):
                        if color == target:
                            if target_detection is None:
                                continue
                            
                            if carrying_block:
                                continue
                            
                            if visual_approach:
                                continue

                            if target_detection.block_too_close:
                                raise TargetDetected(target_detection)
                            
                            continue

                        detection = detections.get(color)
                        if detection is None:
                            continue

                        if target_detection is not None and detection.area < target_detection.area:
                            continue

                        if detection.block_too_close:
                            logger.info(f"[MOTION] _drive camera abort: {color} block too close, size={detection.area:.1f}")
                            hal.stop_motors()
                            backoff_result = move_reverse(localizer)
                            return False

            if current_total > last_total:
                last_total = current_total
                last_progress_t = time.monotonic()
            elif time.monotonic() - last_progress_t > STALL_TIMEOUT_S:
                logger.info(f"[MOTION] _drive stall detected: no encoder progress ({counter_left} L, {counter_right} R) for {time.monotonic() - last_progress_t:.1f} s")
                return False

            
            error = ((initial_heading - current_heading) + 180) % 360 - 180
            corr = max(-MAX_HEADING_CORRECTION,
                       min(MAX_HEADING_CORRECTION, HEADING_HOLD_KP * error))

            corr *= direction

            left_duty = max(0.0, min(100.0, base_duty + corr))
            right_duty = max(0.0, min(100.0, base_duty - corr))
            hal.set_individual_duties(direction, left_duty, direction, right_duty)
            time.sleep(LOOP_SLEEP_S)

        return False
    except TargetDetected as e:
        raise e
    finally:
        hal.stop_motors()


def _pivot_to_heading(localizer: Localizer, target_heading: float) -> bool:
    target_heading = target_heading % 360
    
    error = ((target_heading - localizer.pose.heading) + 180) % 360 - 180

    if abs(error) <= HEADING_TOLERANCE:
        return True

    direction = +1 if error > 0 else -1
    left_direction = direction
    right_direction = direction * -1
    full_duty = float(hal.PWM_DUTY_TURN)
    deadline = time.monotonic() + PIVOT_DEADLINE_S
    wall_trigger_count = 0

    try:
        while time.monotonic() < deadline:
            localizer.set_heading(hal.get_heading())
            ultrasonic_distance = _buffered_ultrasonic_median()

            if ultrasonic_distance is not None and localizer.ultrasonic_reading_inside_robot_radius(ultrasonic_distance, localizer.pose.heading):
                wall_trigger_count += 1
            else:
                wall_trigger_count = 0
            if wall_trigger_count >= 2:
                logger.info(f"[MOTION] _pivot wall too close ({wall_trigger_count} consecutive readings), reading: {ultrasonic_distance:.1f} cm at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}, {localizer.pose.heading:.1f}), aborting pivot")

                hal.stop_motors()
                move_reverse(localizer)
                localize(localizer, force=localizer.pose.heading)
                return False
                
            error = ((target_heading - localizer.pose.heading) + 180) % 360 - 180
            absolute_error = abs(error)
            if absolute_error <= HEADING_TOLERANCE or (error * direction < 0):
                return True

            if absolute_error >= SLOWDOWN_THRESHOLD_DEG:
                duty = full_duty
            else:
                fraction = absolute_error / SLOWDOWN_THRESHOLD_DEG
                duty = MIN_PIVOT_DUTY + (full_duty - MIN_PIVOT_DUTY) * fraction

            hal.set_individual_duties(left_direction, duty, right_direction, duty)
            time.sleep(LOOP_SLEEP_S)
    finally:
        hal.stop_motors()

    return False


def move_reverse(localizer: Localizer) -> bool:
    distance_cm = REVERSE_DISTANCE_CM

    for i in range(REVERSE_REPLAN_ATTEMPTS):
        x, y = arena.apply_motion(localizer.pose.x_cm, localizer.pose.y_cm, -distance_cm, localizer.pose.heading)
        # Tighter validation than forward moves: shrink the safe envelope
        # by REVERSE_EXTRA_MARGIN_CM. Reverses run blind (no rear sensor)
        # and odometry drift accumulates between localizations - without
        # the buffer, a "valid" 5 cm reverse can physically land on the
        # wall when the real pose is already nearer than the estimate.
        if localizer.is_valid_point(x, y, extra_margin=REVERSE_EXTRA_MARGIN_CM):
            break
        logger.info(f"[MOTION] move_reverse Backup {distance_cm:.1f} cm from ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}) heading {localizer.pose.heading:.1f} deg would put robot into invalid position ({x:.1f}, {y:.1f}) [margin={REVERSE_EXTRA_MARGIN_CM:.0f}cm], skipping")
        distance_cm = distance_cm - (REVERSE_DISTANCE_CM / REVERSE_REPLAN_ATTEMPTS)

    if distance_cm <= 0:
        logger.info(f"[MOTION] move_reverse: no valid backup distance, cannot reverse")
        return False

    _drive(localizer, distance_cm, localizer.pose.heading, -1)
    return True

def pivot_to_heading(localizer: Localizer, target: float) -> bool:
    result = False
    for i in range(MAX_PIVOT_ATTEMPTS):
        if abs(((target - localizer.pose.heading) + 180) % 360 - 180) <= HEADING_TOLERANCE:
            return True
        result = _pivot_to_heading(localizer, target)
        time.sleep(TIME_BETWEEN_PIVOTS)

    return result

def move_to_point(localizer: Localizer, x_cm: float, y_cm: float,
                  target : str = None, carrying_block: bool = False,
                  navigation_retries: int = DRIVE_AROUND_MAX_DEPTH) -> bool:
    if navigation_retries < 0:
        return False
    
    dx = x_cm - localizer.pose.x_cm
    dy = y_cm - localizer.pose.y_cm
    if abs(dx) <= 1e-3 and abs(dy) <= 1e-3:
        logger.info("[MOTION] move_to_point Within tolerance of point")
        return True
    heading = math.degrees(math.atan2(dx, dy)) % 360
    distance = math.hypot(dx, dy)

    result = pivot_to_heading(localizer, heading)
    if not result:
        logger.info(f"[MOTION] move_to_point pivot not ok")
        return False
    try:
        logger.info(f"[MOTION] move_to_point target ({x_cm:.1f}, {y_cm:.1f}), distance ({distance:.1f}, heading {heading:.1f})")
        result = _drive(localizer, distance, heading, +1, target=target, carrying_block=carrying_block)
    except TargetDetected as e:
        raise e

    if not result:
        move_reverse(localizer)

        if not carrying_block:
            detection = vision.detect_block(target)
            if detection is not None:
                return False

        offset = DRIVE_AROUND_LATERAL_CM
        x1, y1 = arena.apply_motion(localizer.pose.x_cm, localizer.pose.y_cm, offset, (heading + 90) % 360)
        x2, y2 = arena.apply_motion(localizer.pose.x_cm, localizer.pose.y_cm, offset, (heading - 90) % 360)
        cz = arena.COLLECTION_ZONE_CENTER
        d1 = math.hypot(x1 - cz[0], y1 - cz[1])
        d2 = math.hypot(x2 - cz[0], y2 - cz[1])
        wx, wy = (x1, y1) if d1 < d2 else (x2, y2)
        
        if not localizer.is_valid_point(x1, y1) and not localizer.is_valid_point(x2, y2):
            return False
        elif not localizer.is_valid_point(x1, y1) and localizer.is_valid_point(x2, y2):
            wx, wy = x2, y2
        elif localizer.is_valid_point(x1, y1) and not localizer.is_valid_point(x2, y2):
            wx, wy = x1, y1
        
        move_to_point(localizer, wx, wy, target=target, carrying_block=carrying_block, navigation_retries=(navigation_retries - 1))
        result = move_to_point(localizer, x_cm, y_cm, target=target, carrying_block=carrying_block, navigation_retries=(navigation_retries - 1))

    logger.info(f"[MOTION] move_to_point returning from {navigation_retries} with {result}")
    return result


def grab_block(localizer: Localizer, color: str) -> bool:
    logger.info(f"[MOTION] grab block {color}")

    hal.gripper_close()
    time.sleep(0.5)
    move_reverse(localizer)
    pivot_to_heading(localizer, 270.0)

    det = vision.detect_block(target=color)

    if det is None or not det.within_reach:
        logger.info(f"[MOTION] grap_block may have failed: {color} block not within reach after grab attempt at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f})")
        hal.gripper_open()
        return False
    
    logger.info(f"[MOTION] grab_block appears successful: {color} block within reach after grab attempt at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f})")
    return True


def release_block(localizer: Localizer) -> bool:
    logger.info("[MOTION] release block")

    hal.gripper_open()
    time.sleep(0.5)
    move_reverse(localizer)
    return True

_last_localization = 0.0
def localize(localizer: Localizer, force: float = None, bypass_cooldown: bool = False) -> bool:
    logger.info(f"[MOTION] localize: checking for nearby walls from pose ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}) bypass_cooldown={bypass_cooldown}")
    global _last_localization
    time_since_last_localization = time.monotonic() - _last_localization
    if not bypass_cooldown and time_since_last_localization < MIN_TIME_BETWEEN_LOCALIZATIONS:
        logger.info(f"[MOTION] localize: Already recently localized {time_since_last_localization:.1f} s ago")
        return True

    if force is not None:
        heading_mod = force % 360
        if heading_mod < 45 or heading_mod >= 315:
            wall = "north"
        elif heading_mod < 135:
            wall = "east"
        elif heading_mod < 225:
            wall = "south"
        else:
            wall = "west"

        # Sanity-check the wall is plausibly in sensor range. force=heading
        # is called from wall-too-close handlers in _drive / _pivot_to_heading
        # that fired on an ultrasonic ping, but the ping might have been a
        # block, not a wall. Without this guard, _get_wall_distance accepts
        # whatever clearance the (possibly mirror-fit / fallback-argmin)
        # sweep returns, and localize then snaps the position estimate to it
        # - a 115 cm y-teleport observed in the field traced back to a
        # forced south-wall localize at y=165 with the actual south wall
        # well beyond sensor range. If our own odometry says the implied
        # wall is past MAX_VALID_SENSOR_CM + ROBOT_HALF_EXTENT_CM, the
        # ping wasn't the wall - skip the snap.
        if wall == "north":
            expected_wall_distance = arena.ARENA_NORTH_WALL_Y_CM - localizer.pose.y_cm
        elif wall == "south":
            expected_wall_distance = localizer.pose.y_cm
        elif wall == "east":
            expected_wall_distance = arena.ARENA_EAST_WALL_X_CM - localizer.pose.x_cm
        else:  # west
            expected_wall_distance = localizer.pose.x_cm

        if expected_wall_distance > MAX_VALID_SENSOR_CM + ROBOT_HALF_EXTENT_CM:
            logger.info(f"[MOTION] localize: skip force={force:.0f}; {wall} wall implied at {expected_wall_distance:.0f} cm from estimated pose, beyond sensor range ({MAX_VALID_SENSOR_CM + ROBOT_HALF_EXTENT_CM:.0f} cm)")
            return False

        dist = _get_wall_distance(localizer, wall)
        if dist is None:
            return False

        if wall == "west":
            localizer.set_position(dist, localizer.pose.y_cm)
        elif wall == "east":
            localizer.set_position(arena.ARENA_EAST_WALL_X_CM - dist, localizer.pose.y_cm)
        elif wall == "south":
            localizer.set_position(localizer.pose.x_cm, dist)
        elif wall == "north":
            localizer.set_position(localizer.pose.x_cm, arena.ARENA_NORTH_WALL_Y_CM - dist)

        _last_localization = time.monotonic()
        return True

    x_wall = None
    if (localizer.pose.x_cm < MAX_LOCALIZATION_RANGE_CM):
        x_wall = "west"
    elif localizer.pose.x_cm > arena.ARENA_EAST_WALL_X_CM - MAX_LOCALIZATION_RANGE_CM:
        x_wall = "east"

    if x_wall is None:
        logger.info(f"[MOTION] localize: sensor None in x direction, skipping x localization")

    y_wall = None
    if (localizer.pose.y_cm < MAX_LOCALIZATION_RANGE_CM):
        y_wall = "south"
    elif localizer.pose.y_cm > arena.ARENA_NORTH_WALL_Y_CM - MAX_LOCALIZATION_RANGE_CM:
        y_wall = "north"

    if y_wall is None:
        logger.info(f"[MOTION] localize: sensor None in y direction, skipping y localization")

    if x_wall is None and y_wall is None:
        return False

    x = _get_wall_distance(localizer, x_wall) if x_wall else None
    y = _get_wall_distance(localizer, y_wall) if y_wall else None

    updated = False
    if x_wall == "west" and x is not None:
        localizer.set_position(x, localizer.pose.y_cm)
        updated = True
    elif x_wall == "east" and x is not None:
        localizer.set_position(arena.ARENA_EAST_WALL_X_CM - x, localizer.pose.y_cm)
        updated = True

    if y_wall == "south" and y is not None:
        localizer.set_position(localizer.pose.x_cm, y)
        updated = True
    elif y_wall == "north" and y is not None:
        localizer.set_position(localizer.pose.x_cm, arena.ARENA_NORTH_WALL_Y_CM - y)
        updated = True

    if updated:
        _last_localization = time.monotonic()
    
    return updated

def _median_in_range(samples: Iterable[float]) -> Optional[float]:
    """Median of samples that fall within (MIN_VALID_SENSOR_CM, MAX_VALID_SENSOR_CM).

    Returns None if no sample survives gating. Median (not mean) survives the
    dominant HC-SR04 failure modes (~0 cm on missed echo, ~340 cm on stuck-high
    echo) which would otherwise pull the mean by ~30 percent and blow up the
    1/r linearisation used by _fit_wall_geometry.
    """
    survivors = [d for d in samples if MIN_VALID_SENSOR_CM < d < MAX_VALID_SENSOR_CM]
    if not survivors:
        return None
    survivors.sort()
    return survivors[len(survivors) // 2]


def _read_ultrasonic_median() -> Optional[float]:
    """Trigger SWEEP_SAMPLES fresh ultrasonic reads (sleeping between each so the
    background ping thread refreshes) and return the gated median.

    Use this after a fresh pivot, where stale buffer contents would reflect the
    pre-pivot pose. For inline use during a drive, where blocking on extra
    sleeps is unwanted, use _buffered_ultrasonic_median instead.
    """
    samples = []
    for _ in range(SWEEP_SAMPLES):
        time.sleep(LOOP_SLEEP_S)
        if hal.is_ultrasonic_healthy():
            samples.append(hal.get_ultrasonic_distance())
    return _median_in_range(samples)


def _buffered_ultrasonic_median() -> Optional[float]:
    """Gated median of the rolling-buffer samples maintained by the HAL ping thread.

    Cheaper than _read_ultrasonic_median (no sleep) but only meaningful when the
    robot has been moving smoothly long enough for the buffer to reflect current
    pose. Drops the rare 0 cm / 340 cm outliers without slowing the caller's loop.
    """
    if not hal.is_ultrasonic_healthy():
        return None
    return _median_in_range(hal.get_recent_ultrasonic_readings())


def _fit_wall_geometry(thetas_deg: list, ranges_cm: list) -> Optional[tuple]:
    """Fit r(theta) = d / cos(theta - theta_perp) by linearising in 1/r.

    A flat wall is a 2-DoF object - perpendicular distance d and
    perpendicular orientation theta_perp - and the time-of-flight law
    relates them to a sensor reading at any heading by the cosine
    geometry above. Substitute u = cos(theta_perp)/d, v = sin(theta_perp)/d:

        1/r = u * cos(theta) + v * sin(theta)

    which is linear in (u, v) and solvable in closed form via the 2x2
    normal equations. Recover d = 1/sqrt(u^2 + v^2) and theta_perp =
    atan2(v, u). The residual is computed in r-space (cm) so that it is
    directly comparable to expected sensor noise.

    Returns (d_cm, theta_perp_deg in [0, 360), rms_cm) on success, or None
    if the system is singular (sweep angles not linearly independent in
    cos/sin) or the fit predicts a non-physical 1/r anywhere in the sweep.
    """
    n = len(thetas_deg)
    if n < 3:
        return None

    s_cc = s_cs = s_ss = 0.0
    s_ic = s_is = 0.0
    for t_deg, r in zip(thetas_deg, ranges_cm):
        t = math.radians(t_deg)
        c = math.cos(t)
        s = math.sin(t)
        s_cc += c * c
        s_cs += c * s
        s_ss += s * s
        s_ic += c / r
        s_is += s / r

    det = s_cc * s_ss - s_cs * s_cs
    # Singular when every sweep stop landed at (effectively) the same heading,
    # which happens if every pivot failed to move the robot. cos and sin are
    # then not linearly independent across samples and (u, v) is undetermined.
    if abs(det) < 1e-9:
        return None

    u = ( s_ss * s_ic - s_cs * s_is) / det
    v = (-s_cs * s_ic + s_cc * s_is) / det

    norm = math.hypot(u, v)
    if norm <= 0:
        return None

    d_cm = 1.0 / norm
    theta_perp = math.degrees(math.atan2(v, u)) % 360.0

    sse = 0.0
    for t_deg, r in zip(thetas_deg, ranges_cm):
        t = math.radians(t_deg)
        denom = u * math.cos(t) + v * math.sin(t)
        # denom <= 0 means the model predicts the wall is behind the sensor
        # at this angle - the geometry has been violated, refuse the fit.
        if denom <= 0:
            return None
        r_pred = 1.0 / denom
        sse += (r_pred - r) ** 2
    rms = math.sqrt(sse / n)

    return d_cm, theta_perp, rms


def _get_wall_distance(localizer: Localizer, wall: str) -> float:
    """Estimate distance to a cardinal wall via sweep + parametric fit.

    Sweeps the ultrasonic across +/- SWEEP_RANGE_DEG of the cardinal heading
    and fits the time-of-flight law r(theta) = d / cos(theta - theta_perp).
    Every ping is an independent constraint on the same (d, theta_perp), so
    every sweep sample contributes - in contrast to the previous argmin
    approach which discarded all readings except the discrete grid minimum.
    The fit yields sub-grid theta_perp (so we skip the backtrack-and-resample
    step), exposes a residual that flags non-flat-wall returns (corner
    echoes, beam slipping past a wall edge, sensor garbage), and turns
    recalibrate_heading into a principled IMU-drift correction off a rigid
    axis-aligned reference instead of stamping the IMU with whatever heading
    the robot happened to land on.

    Falls back to argmin behaviour when the fit is singular or its residual
    exceeds WALL_FIT_MAX_RMS_CM.
    """
    headings = {"north": 0, "east": 90, "south": 180, "west": 270}
    if wall not in headings:
        return None

    known_heading = float(headings[wall])

    if not pivot_to_heading(localizer, known_heading):
        logger.info(f"[MOTION] _get_wall_distance: initial pivot failed for {wall} wall")
        return None

    time.sleep(LOOP_SLEEP_S)

    sweep_center = localizer.pose.heading

    # Record (actual_heading_after_pivot, median_range) at each sweep stop.
    # The actual IMU heading is used rather than the commanded offset because
    # pivot_to_heading lands within HEADING_TOLERANCE, not exactly on target;
    # the geometric residual must be computed against the angles the sensor
    # really pointed at.
    sweep_thetas = []
    sweep_ranges = []
    for offset in range(-SWEEP_RANGE_DEG, SWEEP_RANGE_DEG + SWEEP_STEP_DEG, SWEEP_STEP_DEG):
        pivot_to_heading(localizer, (sweep_center + offset) % 360)
        median = _read_ultrasonic_median()
        if median is not None:
            sweep_thetas.append(localizer.pose.heading)
            sweep_ranges.append(median)

    if not sweep_ranges:
        logger.info(f"[MOTION] _get_wall_distance: no valid readings during sweep for {wall} wall")
        return None

    # Per-beam dump is the first thing to look at when a fit gets rejected as
    # ill-conditioned or rms-too-high - a glance tells you whether the pivots
    # actually moved (theta spread), how many readings survived gating, and
    # whether the range pattern matches the cos shape we expect.
    logger.info(f"[MOTION] _get_wall_distance: {wall} sweep ({len(sweep_ranges)} readings): "
                + ", ".join(f"{t:.0f}deg/{r:.0f}cm" for t, r in zip(sweep_thetas, sweep_ranges)))

    fit = _fit_wall_geometry(sweep_thetas, sweep_ranges)

    reject_reason = None
    if fit is None:
        reject_reason = "ill-conditioned"
    else:
        d_cm, theta_perp, rms = fit
        drift = ((theta_perp - known_heading) + 180.0) % 360.0 - 180.0
        # Mirror-fit guard: the model r(theta) = d/cos(theta - theta_perp) is
        # symmetric under theta_perp += 180, so a fit that lands on the wrong
        # half of the unit circle can match the data with low residual but
        # describes a wall in the opposite direction. The robot was just
        # pivoted to known_heading at the start of this routine, so theta_perp
        # should be within a sweep-width of it; anything farther is the mirror.
        if abs(drift) >= WALL_FIT_HEADING_GATE_DEG:
            reject_reason = (f"theta_perp={theta_perp:.1f} differs from known "
                             f"{known_heading:.1f} by {drift:+.1f} (mirror-fit)")
        elif rms > WALL_FIT_MAX_RMS_CM:
            reject_reason = f"rms={rms:.2f} cm exceeds {WALL_FIT_MAX_RMS_CM:.1f} cm"

    if reject_reason is None:
        # Pivot to the fitted perpendicular regardless. Whether we then snap
        # the IMU offset to the cardinal depends on how much we trust the
        # drift signal: small drifts are real IMU bias and worth correcting;
        # large drifts (~20+ deg) observed in the field were corner-confused
        # fits whose recalibration injected bogus offset into the IMU. Gate
        # the recalibration on |drift| < HEADING_RECAL_GATE_DEG. The returned
        # clearance is correct in either case since d is the wall-perpendicular
        # distance and stands on its own.
        pivot_to_heading(localizer, theta_perp)
        time.sleep(LOOP_SLEEP_S)

        if abs(drift) < HEADING_RECAL_GATE_DEG:
            hal.recalibrate_heading(known_heading)
            localizer.set_heading(known_heading)
            recal_note = "recal"
        else:
            recal_note = f"recal skipped (|drift|>={HEADING_RECAL_GATE_DEG:.0f})"

        clearance = localizer.ultrasonic_to_robot_center(d_cm, known_heading)
        logger.info(f"[MOTION] _get_wall_distance: {wall} wall fit d={d_cm:.1f} cm, "
              f"theta_perp={theta_perp:.1f}, drift={drift:+.1f}, "
              f"rms={rms:.2f} cm, clearance={clearance:.1f} cm, {recal_note}")
        return clearance

    logger.info(f"[MOTION] _get_wall_distance: {wall} wall fit rejected ({reject_reason}); "
          f"falling back to argmin")

    # Fallback covers the cases the fit refuses (corner echoes, sweeps where
    # cos and sin are not separable, geometrically impossible solutions, or
    # mirror-fits). The FSM still owes itself a best-effort answer, so we
    # re-aim at the smallest sweep reading and re-sample there.
    best_idx = min(range(len(sweep_ranges)), key=lambda i: sweep_ranges[i])
    perp_imu_heading = sweep_thetas[best_idx]
    fallback_drift = ((perp_imu_heading - known_heading) + 180.0) % 360.0 - 180.0
    pivot_to_heading(localizer, perp_imu_heading)
    time.sleep(LOOP_SLEEP_S)

    logger.info(f"[MOTION] _get_wall_distance: updating heading from backup: {perp_imu_heading:.1f} to {known_heading:.1f}")
    # The fallback runs precisely when fit confidence is low (corner-confused
    # geometry, ill-conditioned sweep, mirror-fit), so slamming the IMU off
    # whatever argmin angle we landed on is the same trust failure we want
    # to avoid. Gate the recalibration on |fallback_drift| symmetrically with
    # the success branch.
    if abs(fallback_drift) < HEADING_RECAL_GATE_DEG:
        hal.recalibrate_heading(known_heading)
        localizer.set_heading(known_heading)
        recal_note = "recal"
    else:
        recal_note = f"recal skipped (|drift|>={HEADING_RECAL_GATE_DEG:.0f})"

    final_reading = _read_ultrasonic_median()
    if final_reading is None:
        logger.info(f"[MOTION] _get_wall_distance: {wall} wall no valid reading at fallback perpendicular")
        return None

    clearance = localizer.ultrasonic_to_robot_center(final_reading, known_heading)
    logger.info(f"[MOTION] _get_wall_distance: {wall} wall fallback reading {final_reading:.1f} cm, "
                f"drift={fallback_drift:+.1f}, clearance={clearance:.1f} cm, {recal_note}")
    return clearance

def approach_visual(localizer: Localizer, target: str,
                    max_time_s: float = 30.0) -> bool:
    deadline = time.monotonic() + max_time_s
    failures = 0

    while time.monotonic() <= deadline:
        if arena.in_collection_zone(localizer.pose.x_cm, localizer.pose.y_cm):
            logger.info(f"[MOTION] approach_visual inside collection zone; aborting")
            return False

        target_detection = vision.detect_block(target)
        if target_detection is None:
            logger.info(f"[MOTION] approach_visual did not detect {target} block")
            return False

        steer_deg = (vision.CAMERA_HFOV_DEG  / vision.SENSOR_RESOLUTION[0]) * target_detection.centroid_error * VISUAL_APPROACH_STEER_GAIN
        steer_heading = (localizer.pose.heading + steer_deg) % 360

        if target_detection.within_reach:
            return True

        fraction_remaining = 1.0 - target_detection.area / vision.IN_REACH_TOLERANCE[target]
        distance_cm = max(APPROACH_LARGEST_STEP * fraction_remaining, 1.0)

        logger.info(f"[MOTION] approach_visual\n\tblock area {target_detection.area:.1f}, distance {distance_cm:.1f},\n\tcentroid error {target_detection.centroid_error:.1f}, steering degree {steer_deg:.1f}")
        pivot_to_heading(localizer, steer_heading)

        if target_detection.block_too_close and steer_deg >= VISUAL_APPROACH_TOLERANCE:
            continue

        drive = _drive(localizer, distance_cm, steer_heading, +1, target=target, visual_approach=True)
        if not drive:
            failures += 1
            if failures >= 3:
                raise ApproachVisualStuck()

    logger.info(f"[MOTION] approach_visual timeout {time.monotonic() - deadline:.1f} s")
    return False
