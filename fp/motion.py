import math
import sys
import time
from typing import Optional

from fp import arena
from fp import hal
from fp import vision
from fp.localization import Localizer

class TargetDetected(Exception):
    """Raised when a drive is aborted because the target color is visible."""
    def __init__(self, detection: vision.DetectionResult):
        super().__init__(f"target {detection.label!r} detected")
        self.detection = detection

class ApproachVisualStuck(Exception):
    def __init__(self):
        super().__init__("[MOTION] approach_visual is stuck")


# Tuning constants
LOOP_SLEEP_S = 0.05

# Drive tuning constants
STALL_TIMEOUT_S = 1.5
DRIVE_AROUND_LATERAL_CM = 30.0
DRIVE_AROUND_MAX_DEPTH  = 2
HEADING_HOLD_KP = 2.0
MAX_HEADING_CORRECTION = 25.0
DRIVE_DEADLINE_MIN_S = 4.0
DRIVE_DEADLINE_PER_CM_S = 0.25
REVERSE_DISTANCE_CM = 15.0
REVERSE_REPLAN_ATTEMPTS = 4

# Pivot tuning constants
MAX_PIVOT_ATTEMPTS      = 4
SLOWDOWN_THRESHOLD_DEG  = 45.0
MIN_PIVOT_DUTY          = 25.0
PIVOT_DEADLINE_S        = 6.0
TIME_BETWEEN_PIVOTS     = 0.5
HEADING_TOLERANCE       = 1.0

# Localization tuning constants
MAX_LOCALIZATION_RANGE_CM = 50.0
SWEEP_RANGE_DEG           = 20
SWEEP_STEP_DEG            = 5
SWEEP_SAMPLES             = 3
MAX_VALID_SENSOR_CM       = 60.0
MIN_VALID_SENSOR_CM       = 5.0
WALL_FIT_MAX_RMS_CM       = 3.0

# Visual approach tuning constants
APPROACH_LARGEST_STEP       = 20.0
VISUAL_APPROACH_TOLERANCE   = 2.0

# Internal helper functions

def _drive(localizer: Localizer, distance_cm: float, initial_heading: float, 
           direction: int, target: str = None, visual_approach: bool = False, 
           carrying_block: bool = False) -> bool:
    if direction not in (+1, -1):
        print(f"[MOTION] _drive: invalid direction {direction}")
        return False
    
    print(f"[MOTION] _drive {distance_cm:.1f} cm direction {direction:+d}")

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
            # print(f"  Front Left: {counter_left} Front Right: {counter_right}  total distance: {average / hal.COUNTS_PER_CM:.1f} cm heading: {current_heading:.1f}", end="\r", flush=True)
            if direction == +1:
                localizer.apply_forward(delta_cm, current_heading)
            else:
                localizer.apply_reverse(delta_cm, current_heading)

            if average >= target_counts:
                return True
            
            if hal.is_ultrasonic_healthy():
                ultrasonic_distance = hal.get_ultrasonic_distance()
            else:
                ultrasonic_distance = None

            if direction == +1:
                if ultrasonic_distance is not None and localizer.ultrasonic_reading_inside_robot_radius(ultrasonic_distance, current_heading):
                    wall_trigger_count += 1
                else:
                    wall_trigger_count = 0
                if wall_trigger_count >= 2:
                    print(f"[MOTION] _drive wall too close ({wall_trigger_count} consecutive readings), reading: {ultrasonic_distance:.1f} cm at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}, {localizer.pose.heading:.1f}), aborting drive")

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
                            
                            if target_detection.within_reach:
                                raise TargetDetected(target_detection)
                            
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
                            print(f"[MOTION] _drive camera abort: {color} block too close, size={detection.area:.1f}")
                            hal.stop_motors()
                            backoff_result = move_reverse(localizer)
                            return False

            if current_total > last_total:
                last_total = current_total
                last_progress_t = time.monotonic()
            elif time.monotonic() - last_progress_t > STALL_TIMEOUT_S:
                print(f"[MOTION] _drive stall detected: no encoder progress ({counter_left} L, {counter_right} R) for {time.monotonic() - last_progress_t:.1f} s")
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

    try:
        while time.monotonic() < deadline:
            localizer.set_heading(hal.get_heading())
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
        if localizer.is_valid_point(x, y):
            break
        print(f"[MOTION] move_reverse Backup {distance_cm:.1f} cm from ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f}) heading {localizer.pose.heading:.1f} deg would put robot into invalid position ({x:.1f}, {y:.1f}), skipping")
        distance_cm = distance_cm - (REVERSE_DISTANCE_CM / REVERSE_REPLAN_ATTEMPTS)

    if distance_cm <= 0:
        print(f"[MOTION] move_reverse: no valid backup distance, cannot reverse")
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
        print("[MOTION] move_to_point Within tolerance of point")
        return True
    heading = math.degrees(math.atan2(dx, dy)) % 360
    distance = math.hypot(dx, dy)

    result = pivot_to_heading(localizer, heading)
    if not result:
        print(f"[MOTION] move_to_point pivot not ok")
        return False
    try:
        print(f"[MOTION] move_to_point target ({x_cm:.1f}, {y_cm:.1f}), distance ({distance:.1f}, heading {heading:.1f})")
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

    print(f"[MOTION] move_to_point returning from {navigation_retries} with {result}")
    return result


def grab_block(localizer: Localizer, color: str) -> bool:
    print(f"[MOTION] grab block {color}")

    hal.gripper_close()
    time.sleep(0.5)
    move_reverse(localizer)
    pivot_to_heading(localizer, 270.0)

    det = vision.detect_block(target=color)

    if det is None or not det.within_reach:
        print(f"[MOTION] grap_block may have failed: {color} block not within reach after grab attempt at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f})")
        hal.gripper_open()
        return False
    
    print(f"[MOTION] grab_block appears successful: {color} block within reach after grab attempt at ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f})")
    return True


def release_block(localizer: Localizer) -> bool:
    print("[MOTION] release block")

    hal.gripper_open()
    time.sleep(0.5)
    move_reverse(localizer)
    return True


def localize(localizer: Localizer, force: float = None) -> bool:
    print(f"[MOTION] localize: checking for nearby walls from pose ({localizer.pose.x_cm:.1f}, {localizer.pose.y_cm:.1f})")

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

        return True

    x_wall = None
    if (localizer.pose.x_cm < MAX_LOCALIZATION_RANGE_CM):
        x_wall = "west"
    elif localizer.pose.x_cm > arena.ARENA_EAST_WALL_X_CM - MAX_LOCALIZATION_RANGE_CM:
        x_wall = "east"

    if x_wall is None:
        print(f"[MOTION] localize: sensor None in x direction, skipping x localization")

    y_wall = None
    if (localizer.pose.y_cm < MAX_LOCALIZATION_RANGE_CM):
        y_wall = "south"
    elif localizer.pose.y_cm > arena.ARENA_NORTH_WALL_Y_CM - MAX_LOCALIZATION_RANGE_CM:
        y_wall = "north"

    if y_wall is None:
        print(f"[MOTION] localize: sensor None in y direction, skipping y localization")

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

    return updated

def _read_ultrasonic_median() -> Optional[float]:
    """Median of SWEEP_SAMPLES gated readings, or None if every sample was rejected.

    Median (not mean) survives the dominant HC-SR04 failure mode: a single
    reading near 0 cm from a missed echo or near 340 cm from a stuck-high
    echo. Either would otherwise pull the mean by ~30 percent and, more
    critically, blow up the 1/r linearisation used by _fit_wall_geometry.
    """
    samples = []
    for _ in range(SWEEP_SAMPLES):
        time.sleep(LOOP_SLEEP_S)
        if not hal.is_ultrasonic_healthy():
            continue
        d = hal.get_ultrasonic_distance()
        if MIN_VALID_SENSOR_CM < d < MAX_VALID_SENSOR_CM:
            samples.append(d)
    if not samples:
        return None
    samples.sort()
    return samples[len(samples) // 2]


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
        print(f"[MOTION] _get_wall_distance: initial pivot failed for {wall} wall")
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
        print(f"[MOTION] _get_wall_distance: no valid readings during sweep for {wall} wall")
        return None

    fit = _fit_wall_geometry(sweep_thetas, sweep_ranges)

    if fit is not None:
        d_cm, theta_perp, rms = fit
        if rms <= WALL_FIT_MAX_RMS_CM:
            # Pivot to the fitted perpendicular and snap the IMU offset to the
            # cardinal. The wall is rigid and never drifts, so the difference
            # (theta_perp - known_heading) is an honest IMU-drift estimate;
            # previously this call zeroed whatever heading we happened to land on.
            pivot_to_heading(localizer, theta_perp)
            time.sleep(LOOP_SLEEP_S)

            drift = ((theta_perp - known_heading) + 180.0) % 360.0 - 180.0
            hal.recalibrate_heading(known_heading)
            localizer.set_heading(known_heading)

            clearance = localizer.ultrasonic_to_robot_center(d_cm, known_heading)
            print(f"[MOTION] _get_wall_distance: {wall} wall fit d={d_cm:.1f} cm, "
                  f"theta_perp={theta_perp:.1f}, drift={drift:+.1f}, "
                  f"rms={rms:.2f} cm, clearance={clearance:.1f} cm")
            return clearance
        print(f"[MOTION] _get_wall_distance: {wall} wall fit rms {rms:.2f} cm exceeds "
              f"{WALL_FIT_MAX_RMS_CM:.1f} cm; falling back to argmin")
    else:
        print(f"[MOTION] _get_wall_distance: {wall} wall fit ill-conditioned; falling back to argmin")

    # Fallback covers the cases the fit refuses (corner echoes, sweeps where
    # cos and sin are not separable, geometrically impossible solutions).
    # The FSM still owes itself a best-effort answer, so we re-aim at the
    # smallest sweep reading and re-sample there - same behaviour as before.
    best_idx = min(range(len(sweep_ranges)), key=lambda i: sweep_ranges[i])
    perp_imu_heading = sweep_thetas[best_idx]
    pivot_to_heading(localizer, perp_imu_heading)
    time.sleep(LOOP_SLEEP_S)

    print(f"[MOTION] _get_wall_distance: updating heading from backup: {perp_imu_heading:.1f} to {known_heading:.1f}")
    hal.recalibrate_heading(known_heading)
    localizer.set_heading(known_heading)

    final_reading = _read_ultrasonic_median()
    if final_reading is None:
        print(f"[MOTION] _get_wall_distance: {wall} wall no valid reading at fallback perpendicular")
        return None

    clearance = localizer.ultrasonic_to_robot_center(final_reading, known_heading)
    print(f"[MOTION] _get_wall_distance: {wall} wall fallback reading {final_reading:.1f} cm, clearance {clearance:.1f} cm")
    return clearance

def approach_visual(localizer: Localizer, target: str,
                    max_time_s: float = 30.0) -> bool:
    deadline = time.monotonic() + max_time_s
    failures = 0

    while time.monotonic() <= deadline:
        if arena.in_collection_zone(localizer.pose.x_cm, localizer.pose.y_cm):
            print(f"[MOTION] approach_visual inside collection zone; aborting")
            return False

        target_detection = vision.detect_block(target)
        if target_detection is None:
            print(f"[MOTION] approach_visual did not detect {target} block")
            return False

        if target_detection.within_reach:
            print(f"[MOTION] approach_visual reached {target} block")
            return True

        steer_deg = (vision.CAMERA_HFOV_DEG  / vision.SENSOR_RESOLUTION[0]) * target_detection.centroid_error
        steer_heading = (localizer.pose.heading + steer_deg) % 360
        fraction_remaining = 1.0 - target_detection.area / vision.IN_REACH_TOLERANCE[target]
        distance_cm = max(APPROACH_LARGEST_STEP * fraction_remaining, 1.0)

        print(f"[MOTION] approach_visual\n\tblock area {target_detection.area:.1f}, distance {distance_cm:.1f},\n\tcentroid error {target_detection.centroid_error:.1f}, steering degree {steer_deg:.1f}")
        pivot_to_heading(localizer, steer_heading)

        if target_detection.block_too_close and steer_deg >= VISUAL_APPROACH_TOLERANCE:
            continue

        drive = _drive(localizer, distance_cm, steer_heading, +1, target=target, visual_approach=True)
        if not drive:
            failures += 1
            if failures >= 3:
                raise ApproachVisualStuck()

    print(f"[MOTION] approach_visual timeout {time.monotonic() - deadline:.1f} s")
    return False
