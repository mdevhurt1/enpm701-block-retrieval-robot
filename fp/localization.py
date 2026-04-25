import math
import logging
import threading
import time
from typing import Optional, TYPE_CHECKING

import numpy as np
from fp import hal
from fp import arena
from fp import tuning

if TYPE_CHECKING:
    from fp.ekf import EKF
    from fp.planner import OccupancyGrid

logger = logging.getLogger(__name__)


# Localizer — holds all mutable navigation / odometry state

class Localizer:
    """Encapsulates all mutable navigation state.  Module-level functions use
    the singleton `loc`; no bare module-level mutable globals remain."""

    def __init__(self) -> None:
        self.ekf: Optional["EKF"] = None
        self.heading_offset: float = 0.0
        self.odo_fallback_x: float = 0.0
        self.odo_fallback_y: float = 0.0
        self.odo_path: list = []
        self.odo_lock = threading.Lock()
        self.odo_stop = threading.Event()
        self.odo_thread: Optional[threading.Thread] = None
        self.odo_rate_hz: int = 100
        self.odo_direction: int = 0
        self.odo_pivoting: bool = False


loc = Localizer()


# EKF injection

def init_ekf(ekf_instance: "EKF") -> None:
    """Inject an EKF instance.  After this call, get_pose() returns EKF state
    and correct_pose_from_wall() calls ekf.update_wall() instead of raw
    assignment.  Call before start_odometry()."""
    loc.ekf = ekf_instance
    logger.info("[NAV] EKF injected")


# Heading utilities

def set_heading_offset(imu_north_heading: float) -> None:
    """Store the IMU heading (degrees) that corresponds to arena north (0 deg)."""
    loc.heading_offset = imu_north_heading % 360


def get_arena_heading() -> float:
    """Return current heading in arena degrees (0 = north / collection-zone direction)."""
    return (hal.get_heading() - loc.heading_offset) % 360


def get_arena_heading_or_none() -> Optional[float]:
    """Like get_arena_heading() but returns None when the IMU reading is stale or
    unavailable."""
    raw = hal.get_heading_or_none()
    if raw is None:
        return None
    return (raw - loc.heading_offset) % 360


def _arena_to_imu(arena_deg: float) -> float:
    return (arena_deg + loc.heading_offset) % 360


# Odometry

def _odometry_worker() -> None:
    """
    Background thread: reads encoders at odo_rate_hz and dead-reckons
    position using the current IMU heading and odo_direction flag.
    When an EKF is injected via init_ekf(), calls ekf.predict() and
    ekf.update_heading() each cycle instead of raw dead-reckoning.
    """
    interval       = 1.0 / loc.odo_rate_hz
    prev_a, prev_b = hal.read_encoders()
    prev_time      = time.monotonic()

    while not loc.odo_stop.is_set():
        t_start = time.monotonic()
        dt      = t_start - prev_time
        prev_time = t_start

        cur_a, cur_b = hal.read_encoders()
        delta_a = cur_a - prev_a
        delta_b = cur_b - prev_b
        prev_a, prev_b = cur_a, cur_b

        counts = (delta_a + delta_b) / 2.0

        hdg = get_arena_heading_or_none()
        if hdg is None:
            logger.warning("[ODO] IMU heading unavailable; skipping odometry cycle")
            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            continue
        dist = (counts / hal.COUNTS_PER_CM) * loc.odo_direction if counts > 0 else 0.0

        if loc.ekf is not None:
            if dt > 0:
                loc.ekf.predict(dist, 0.0, dt)
            if not loc.odo_pivoting:
                loc.ekf.update_heading(hdg)

            ekf_x, ekf_y, _ = loc.ekf.get_pose()
            with loc.odo_lock:
                if dist != 0:
                    loc.odo_path.append((ekf_x, ekf_y))
        else:
            if counts > 0 and loc.odo_direction != 0:
                with loc.odo_lock:
                    loc.odo_fallback_x, loc.odo_fallback_y = arena.apply_motion(
                        loc.odo_fallback_x, loc.odo_fallback_y, dist, hdg
                    )
                    loc.odo_path.append((loc.odo_fallback_x, loc.odo_fallback_y))

        elapsed = time.monotonic() - t_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_odometry(rate_hz: int = 100, x: float = 0.0, y: float = 0.0) -> None:
    """
    Initialise odometry state and start the background update thread.
    Call after hal.init() and hal.start_imu().
    """
    with loc.odo_lock:
        loc.odo_fallback_x = x
        loc.odo_fallback_y = y
        loc.odo_path    = [(x, y)]

    loc.odo_rate_hz = rate_hz
    loc.odo_stop.clear()
    loc.odo_thread = threading.Thread(target=_odometry_worker, daemon=True, name='odometry')
    loc.odo_thread.start()
    logger.info(f"[ODO] Odometry started at {rate_hz} Hz  initial pose: ({x:.1f}, {y:.1f})")


def stop_odometry() -> None:
    loc.odo_stop.set()
    if loc.odo_thread is not None:
        loc.odo_thread.join(timeout=2.0)
    logger.info("[ODO] Odometry stopped")


def get_pose() -> tuple:
    """Return current pose as (x_cm, y_cm, heading_deg).
    When EKF is active, delegates to ekf.get_pose() for authoritative state."""
    if loc.ekf is not None:
        return loc.ekf.get_pose()
    with loc.odo_lock:
        hdg = get_arena_heading_or_none()
        return (loc.odo_fallback_x, loc.odo_fallback_y, hdg if hdg is not None else 0.0)


def set_pose(x: float, y: float) -> None:
    if loc.ekf is not None:
        _, _, hdg = loc.ekf.get_pose()
        loc.ekf.set_pose(x, y, hdg)
    with loc.odo_lock:
        loc.odo_fallback_x = x
        loc.odo_fallback_y = y
        loc.odo_path.append((x, y))
    logger.info(f"[ODO] Pose set to ({x:.1f}, {y:.1f})")


def get_path() -> list:
    """Return a copy of the recorded (x, y) path."""
    with loc.odo_lock:
        return list(loc.odo_path)


# Sensor distance helpers

def _median_distance(n: int = 5) -> Optional[float]:
    readings = []
    for _ in range(n):
        d = hal.distance_front()
        if d is not None:
            readings.append(d)
        time.sleep(0.05)
    if len(readings) < 3:
        return None
    readings.sort()
    return readings[len(readings) // 2]


def _trimmed_mean_distance(n: int = 5) -> Optional[float]:
    readings = []
    for _ in range(n):
        d = hal.distance_front()
        if d is not None:
            readings.append(d)
        time.sleep(0.05)
    if len(readings) < 3:
        return None
    readings.sort()
    trimmed = readings[1:-1]
    return sum(trimmed) / len(trimmed)


def _batched_trimmed_mean_distance(
    n_batches: int = 5,
    samples_per_batch: int = 5,
    log_prefix: Optional[str] = None,
    label: str = "distance",
) -> Optional[float]:
    batch_estimates = []
    for _ in range(n_batches):
        d = _trimmed_mean_distance(samples_per_batch)
        if d is not None:
            batch_estimates.append(d)
        time.sleep(0.05)

    if len(batch_estimates) < 3:
        if log_prefix is not None:
            logger.warning(f"{log_prefix} {label} batch check failed: fewer than 3 valid batches")
        return None

    batch_estimates.sort()
    trim = 1 if len(batch_estimates) >= 5 else 0
    core = batch_estimates[trim:len(batch_estimates) - trim] if trim else batch_estimates
    mean = sum(core) / len(core)

    if log_prefix is not None:
        formatted = ", ".join(f"{value:.1f}" for value in batch_estimates)
        logger.info(f"{log_prefix} {label} batches: [{formatted}] -> robust={mean:.1f} cm")

    return mean


# Wall distance helpers

def _expected_wall_sensor_distance(wall: str, pose: Optional[tuple] = None) -> float:
    """Return the expected raw front-sensor distance to a wall from the current pose."""
    wall = wall.lower()
    if pose is None:
        x_cm, y_cm, _ = get_pose()
    else:
        x_cm, y_cm, _ = pose

    if wall == 'south':
        centre_dist = y_cm - arena.ARENA_SOUTH_WALL_Y_CM
    elif wall == 'north':
        centre_dist = arena.ARENA_NORTH_WALL_Y_CM - y_cm
    elif wall == 'west':
        centre_dist = x_cm - arena.ARENA_WEST_WALL_X_CM
    elif wall == 'east':
        centre_dist = arena.ARENA_EAST_WALL_X_CM - x_cm
    else:
        raise ValueError(f"unknown wall '{wall}'")

    return max(0.0, centre_dist - arena.ROBOT_SENSOR_OFFSET_CM)


def correct_pose_from_wall(wall: str) -> bool:
    """
    Measure distance to the named wall and update the EKF pose estimate.

    Prerequisites: robot must be facing the named wall; EKF injected via init_ekf().
    wall: 'north' | 'south' | 'east' | 'west'
    Returns True on success, False if sensor could not get a reading or EKF rejects.
    """
    assert loc.ekf is not None, "correct_pose_from_wall requires EKF; call init_ekf() first"

    wall = wall.lower()
    if wall not in ('north', 'south', 'east', 'west'):
        logger.warning(f"[GTC] Unknown wall '{wall}' - use north/south/east/west")
        return False

    _WALL_ARENA_HEADING = {'south': 180.0, 'north': 0.0, 'west': 270.0, 'east': 90.0}

    expected_sensor = _expected_wall_sensor_distance(wall)
    logger.info(f"[GTC] Measuring {wall} wall distance (already perpendicular)...")
    d = _trimmed_mean_distance()
    if d is None:
        logger.error("[GTC] Could not get a valid wall reading")
        return False
    residual = d - expected_sensor
    logger.info(
        f"[GTC] Wall distance: measured={d:.1f} cm  expected={expected_sensor:.1f} cm  residual={residual:+.1f} cm"
    )

    if abs(residual) > tuning.WALL_SWEEP_TRIGGER_CM:
        logger.warning(
            f"[GTC] Residual {residual:+.1f} cm exceeds {tuning.WALL_SWEEP_TRIGGER_CM:.1f} cm "
            f"- running wall sweep"
        )
        sweep = _sweep_and_fit_perpendicular(
            hal.get_heading(),
            sweep_half_deg=tuning.WALL_SWEEP_HALF_DEG,
            n_samples=tuning.WALL_SWEEP_SAMPLES,
            log_prefix="[GTC]",
        )
        if sweep is None:
            logger.warning("[GTC] Sweep refinement failed - using direct wall reading")
        else:
            _, d = sweep
            residual = d - expected_sensor
            logger.info(
                f"[GTC] Sweep-refined wall distance: measured={d:.1f} cm  "
                f"expected={expected_sensor:.1f} cm  residual={residual:+.1f} cm"
            )

    arena_heading = _WALL_ARENA_HEADING[wall]
    loc.ekf.update_heading(arena_heading)
    logger.info(f"[GTC] EKF heading update -> {arena_heading:.1f} deg arena")

    accepted = loc.ekf.update_wall(wall, d)
    if accepted:
        ekf_x, ekf_y, _ = loc.ekf.get_pose()
        with loc.odo_lock:
            loc.odo_path.append((ekf_x, ekf_y))
        logger.info(f"[GTC] EKF wall update accepted  pose: ({ekf_x:.1f}, {ekf_y:.1f})")
    else:
        logger.warning("[GTC] EKF wall update rejected by Mahalanobis gate")
    return accepted


def _find_wall_perpendicular(center_imu: float,
                              initial_step_deg: float = 5.0,
                              max_steps: int = 30) -> Optional[tuple]:
    """Gradient descent to find the heading that minimises front ultrasonic distance."""
    from fp import motion   # lazy — breaks circular import

    def _pivot_and_measure(target: float) -> Optional[float]:
        err = (target - hal.get_heading() + 180) % 360 - 180
        motion._run_heading(target, +1 if err >= 0 else -1)
        time.sleep(0.05)
        return _trimmed_mean_distance()

    err0 = (center_imu - hal.get_heading() + 180) % 360 - 180
    motion._run_heading(center_imu, +1 if err0 >= 0 else -1)
    time.sleep(0.05)
    d_curr = _trimmed_mean_distance()
    if d_curr is None:
        return None

    best_h, best_d = center_imu, d_curr
    valid_count = 1
    step = initial_step_deg

    probe_pos_h = (center_imu + step) % 360
    probe_neg_h = (center_imu - step) % 360
    d_probe_pos = _pivot_and_measure(probe_pos_h)
    d_probe_neg = _pivot_and_measure(probe_neg_h)

    if d_probe_pos is not None:
        valid_count += 1
        if d_probe_pos < best_d:
            best_d, best_h = d_probe_pos, probe_pos_h
    if d_probe_neg is not None:
        valid_count += 1
        if d_probe_neg < best_d:
            best_d, best_h = d_probe_neg, probe_neg_h

    if d_probe_pos is not None and d_probe_neg is not None:
        if d_probe_pos <= d_probe_neg:
            direction = +1
            current_h, current_d = probe_pos_h, d_probe_pos
        else:
            direction = -1
            current_h, current_d = probe_neg_h, d_probe_neg
    elif d_probe_pos is not None:
        direction = +1 if d_probe_pos < d_curr else -1
        current_h, current_d = probe_pos_h, d_probe_pos
    elif d_probe_neg is not None:
        direction = -1 if d_probe_neg < d_curr else +1
        current_h, current_d = probe_neg_h, d_probe_neg
    else:
        return None

    for _ in range(max_steps):
        if step < 0.5:
            break

        next_h = (current_h + direction * step) % 360
        d_next  = _pivot_and_measure(next_h)

        if d_next is None:
            step /= 2.0
            continue

        valid_count += 1

        if d_next < best_d:
            best_d, best_h = d_next, next_h

        if d_next < current_d:
            current_h, current_d = next_h, d_next
        else:
            direction = -direction
            step      /= 2.0
            current_h, current_d = next_h, d_next

    if valid_count < 3:
        return None

    _pivot_and_measure(best_h)
    return (best_h, best_d)


def _sweep_and_fit_perpendicular(
        center_imu: float,
        sweep_half_deg: float = 25.0,
        n_samples: int = 11,
        log_prefix: str = "[INIT]",
) -> Optional[tuple]:
    """
    Record a full ultrasonic sweep over [center_imu ± sweep_half_deg], fit a
    parabola to the readings, and return the minimum as the perpendicular heading.
    Returns (best_imu_heading, min_dist_cm), or None on sensor failure.
    """
    from fp import motion   # lazy — breaks circular import

    if n_samples < 3:
        n_samples = 3

    step = 2 * sweep_half_deg / (n_samples - 1)
    headings = [(center_imu - sweep_half_deg + i * step) % 360
                for i in range(n_samples)]

    sweep: list = []
    for h in headings:
        err = (h - hal.get_heading() + 180) % 360 - 180
        motion._run_heading(h, +1 if err >= 0 else -1)
        time.sleep(0.05)
        d = _trimmed_mean_distance()
        if d is not None:
            sweep.append((h, d))

    if len(sweep) < 3:
        logger.warning(f"{log_prefix} Sweep failed: fewer than 3 valid readings")
        return None

    hs = [s[0] for s in sweep]
    ds = [s[1] for s in sweep]
    min_idx = ds.index(min(ds))

    if min_idx == 0 or min_idx == len(ds) - 1:
        logger.warning(f"{log_prefix} minimum at sweep boundary (idx={min_idx}) - "
              "widening sweep recommended; using endpoint as best estimate")

    hs_unwrapped = [h - 360.0 if h - center_imu > 180 else
                    h + 360.0 if center_imu - h > 180 else h
                    for h in hs]

    try:
        coeffs = np.polyfit(hs_unwrapped, ds, 2)
        if coeffs[0] > 0:
            h_vertex_uw = -coeffs[1] / (2 * coeffs[0])
            h_lo = min(hs_unwrapped); h_hi = max(hs_unwrapped)
            if h_lo < h_vertex_uw < h_hi:
                h_vertex = h_vertex_uw % 360
                err = (h_vertex - hal.get_heading() + 180) % 360 - 180
                motion._run_heading(h_vertex, +1 if err >= 0 else -1)
                time.sleep(0.05)
                d_vertex = _trimmed_mean_distance()
                if d_vertex is not None:
                    logger.info(f"{log_prefix} Parabola fit: vertex at {h_vertex:.2f} deg, "
                          f"dist={d_vertex:.1f} cm")
                    return (h_vertex, d_vertex)
    except Exception:
        pass

    best_h, best_d = sweep[min_idx]
    err = (best_h - hal.get_heading() + 180) % 360 - 180
    motion._run_heading(best_h, +1 if err >= 0 else -1)
    return (best_h, best_d)


def calibrate_initial_pose() -> Optional[tuple]:
    """
    Establish the robot's starting (x, y) pose by geometrically aligning
    with both the south and west walls using multi-angle ultrasonic sweeps.
    Returns (x_cm, y_cm, heading_deg) on success, or None on sensor failure.
    """
    from fp import motion   # lazy — breaks circular import

    h_south_approx = hal.get_heading()
    logger.info(f"[INIT] South-facing IMU: {h_south_approx:.1f} deg - sweeping ±25 deg")
    result = _sweep_and_fit_perpendicular(h_south_approx)
    if result is None:
        logger.error("[INIT] South wall sweep failed - check ultrasonic and placement")
        return None
    h_south, d_south = result
    h_north = (h_south + 180) % 360
    set_heading_offset(h_north)
    logger.info(f"[INIT] South perp at IMU {h_south:.1f} deg  dist={d_south:.1f} cm  "
          f"->  north offset={h_north:.1f} deg")

    max_south = arena.LAUNCH_ZONE_SIZE_CM + 10.0
    if d_south > max_south:
        logger.error(f"[INIT] d_south={d_south:.1f} cm > {max_south:.1f} cm limit - "
              "robot may not be in the SW launch zone. Aborting calibration.")
        return None

    y_cm = arena.ARENA_SOUTH_WALL_Y_CM + d_south + arena.ROBOT_SENSOR_OFFSET_CM
    logger.info(f"[INIT] South wall: {d_south:.1f} cm  ->  Y = {y_cm:.1f} cm")

    motion.pivot_to_heading(270.0)
    h_west_approx = hal.get_heading()
    logger.info(f"[INIT] West-facing IMU: {h_west_approx:.1f} deg - sweeping ±25 deg")
    result = _sweep_and_fit_perpendicular(h_west_approx)
    if result is None:
        logger.error("[INIT] West wall sweep failed - check ultrasonic and placement")
        return None
    h_west, d_west = result
    logger.info(f"[INIT] West perp at IMU {h_west:.1f} deg  dist={d_west:.1f} cm")

    max_west = arena.LAUNCH_ZONE_SIZE_CM + 10.0
    if d_west > max_west:
        logger.error(f"[INIT] d_west={d_west:.1f} cm > {max_west:.1f} cm limit - "
              "robot may not be in the SW launch zone. Aborting calibration.")
        return None

    x_cm = arena.ARENA_WEST_WALL_X_CM + d_west + arena.ROBOT_SENSOR_OFFSET_CM
    logger.info(f"[INIT] West wall:  {d_west:.1f} cm  ->  X = {x_cm:.1f} cm")

    motion.pivot_to_heading(0.0)
    d_north = _batched_trimmed_mean_distance(
        n_batches=tuning.INIT_NORTH_CHECK_BATCHES,
        samples_per_batch=tuning.INIT_NORTH_CHECK_SAMPLES,
        log_prefix="[INIT]",
        label="North wall consistency",
    )
    if d_north is not None:
        span = d_south + d_north + 2 * arena.ROBOT_SENSOR_OFFSET_CM
        expected = arena.ARENA_HEIGHT_CM
        if abs(span - expected) > 20.0:
            logger.warning(f"[INIT] d_south({d_south:.1f}) + d_north({d_north:.1f}) "
                  f"= {span:.1f} cm, expected ≈{expected:.1f} cm "
                  f"(delta={abs(span-expected):.1f} cm) - check arena or sensor")
        else:
            logger.info(f"[INIT] Geometry check OK: span={span:.1f} cm ≈ arena {expected:.1f} cm")

    final_heading = get_arena_heading()
    logger.info(f"[INIT] Facing north wall - arena heading: {final_heading:.1f} deg")
    logger.info(f"\n[INIT] Initial pose: x={x_cm:.1f} cm, y={y_cm:.1f} cm, "
          f"heading={final_heading:.1f} deg")
    return (x_cm, y_cm, final_heading)
