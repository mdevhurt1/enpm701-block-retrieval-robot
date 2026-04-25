import json as _json
import logging as _logging
import os as _os

_logger = _logging.getLogger(__name__)

# Navigation movement

# Minimum arrival radius before a waypoint is considered reached (cm).
# Used by follow_path() and navigate_to().
ARRIVAL_TOLERANCE_CM: float = 8.0

# Distance advanced per visual-approach control iteration (cm).
# Lower = finer centroid control; higher = faster approach but coarser steering.
STEP_CM: float = 2.0

# Obstacle avoidance

# Reverse distance on detecting a non-target obstacle (cm).
AVOID_BACKUP_CM: float = 15.0

# Pivot angle used in the fallback (planner-less) obstacle avoidance path (deg).
AVOID_PIVOT_DEG: float = 45.0

# Search / sweep

# Forward advance after one complete 360° search sweep (cm).
SEARCH_ADVANCE_CM: float = 20.0

# Dwell time after each pivot during a sweep (seconds) — lets the camera
# settle before taking a detection snapshot.
SEARCH_PIVOT_DWELL_S: float = 0.3

# Delivery

# How far to reverse after a successful grab, before evaluating whether the
# block is still in the gripper (cm).
BACKUP_AFTER_GRAB_CM: float = 10.0

# Maximum reverse distance after depositing a block at the collection zone (cm).
BACKUP_AFTER_DEPOSIT_CM: float = 20.0

# Keep at least this many cm between the robot and the nearest wall when reversing
# after a deposit (cm).  Prevents backing into the wall corner.
BACKUP_WALL_MARGIN_CM: float = 10.0

# Correction / localization triggers

# EKF covariance gate: trigger a wall-correction UPDATE when the largest
# diagonal element of P exceeds this value (cm²).  ≈ 10 cm std.
P_CORRECTION_THRESHOLD: float = 100.0

# Fallback time interval between wall corrections when EKF is not active (s).
UPDATE_INTERVAL_S: float = 30.0

# Only attempt a wall correction if the nearest wall is within this distance (cm).
NEAR_WALL_THRESHOLD_CM: float = 61.0   # ≈ 2 ft = launch zone width

# Minimum sensor distance required before facing the wall for a correction (cm).
WALL_CORRECTION_MIN_DIST_CM: float = 5.0

# After a successful correction that only nudges the pose by a tiny amount,
# suppress immediate re-correction on the same wall to avoid UPDATE loops.
WALL_SUCCESS_COOLDOWN_S: float = 5.0
WALL_SUCCESS_MAX_DELTA_CM: float = 1.0
WALL_SUCCESS_MAX_HEADING_DEG: float = 1.0

# If the wall reading differs from the EKF-predicted sensor distance by more
# than this amount, run a short ultrasonic sweep to re-find the perpendicular.
WALL_SWEEP_TRIGGER_CM: float = 10.0

# Runtime sweep geometry used only when WALL_SWEEP_TRIGGER_CM is exceeded.
WALL_SWEEP_HALF_DEG: float = 15.0
WALL_SWEEP_SAMPLES: int = 7

# Initial north-wall consistency check: collect several trimmed-mean batches
# and discard outlier batches before checking the arena span.
INIT_NORTH_CHECK_BATCHES: int = 5
INIT_NORTH_CHECK_SAMPLES: int = 5

# Run-time budget

# Cooldown after a wall correction is rejected by the EKF gate (seconds).
# Prevents a retry loop when the sensor reading consistently disagrees with
# the EKF estimate: the robot is suppressed from re-attempting the same wall
# until it has had time to move away or the EKF state settles.
WALL_COOLDOWN_S: float = 30.0

# Total allowed run duration (seconds).  Leave 30 s buffer before the
# 10-minute hard cutoff to allow for graceful COMPLETE handling.
RUN_LIMIT_S: float = 570.0


# Runtime mutation API
#
# All UPPER_CASE float/int module attributes above are automatically registered.
# Callers that access tuning.X (module-attribute lookup) see the updated value
# immediately after set_param().  Callers that copied the value at import time
# are NOT affected — see fp/state_machine.py TODO in Step 5.

# Populated once at module end by _build_type_registry().
_TYPES: "dict[str, type]" = {}


def _build_type_registry() -> None:
    for k, v in list(globals().items()):
        if k.startswith('_') or not k.isupper():
            continue
        if isinstance(v, (int, float)):
            _TYPES[k] = type(v)


_build_type_registry()


def set_param(key: str, value) -> None:
    """Set a tuning parameter at runtime.

    Validates that the key exists and coerces value to the registered type.
    Raises KeyError for unknown keys, TypeError on bad value.
    """
    if key not in _TYPES:
        raise KeyError(f"Unknown tuning parameter: {key!r}")
    expected = _TYPES[key]
    try:
        value = expected(value)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"Tuning param {key!r} expects {expected.__name__}, "
            f"got {type(value).__name__}: {exc}"
        ) from exc
    globals()[key] = value
    _logger.info(f"[TUNING] {key} = {value!r}")


def get_param(key: str):
    if key not in _TYPES:
        raise KeyError(f"Unknown tuning parameter: {key!r}")
    return globals()[key]


def list_params() -> "dict[str, tuple]":
    """Return {name: (current_value, type_name)} for all tunable parameters."""
    return {k: (globals()[k], _TYPES[k].__name__) for k in sorted(_TYPES)}


def save(path: str = "tuning_override.json") -> None:
    data = {k: globals()[k] for k in sorted(_TYPES)}
    with open(path, "w") as fh:
        _json.dump(data, fh, indent=2)
    _logger.info(f"[TUNING] Saved {len(data)} parameters to {path!r}")


def load(path: str = "tuning_override.json") -> int:
    """Load parameter overrides from a JSON file.

    Silently skips unknown or badly-typed entries with a warning.
    Returns the number of parameters successfully updated.
    """
    if not _os.path.exists(path):
        _logger.debug(f"[TUNING] No override file at {path!r} — using defaults")
        return 0
    with open(path) as fh:
        data = _json.load(fh)
    n = 0
    for k, v in data.items():
        try:
            set_param(k, v)
            n += 1
        except (KeyError, TypeError) as exc:
            _logger.warning(f"[TUNING] Skipping {k!r}: {exc}")
    _logger.info(f"[TUNING] Loaded {n}/{len(data)} overrides from {path!r}")
    return n
