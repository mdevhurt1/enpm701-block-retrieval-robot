import math
import pytest
from unittest.mock import patch, MagicMock
from fp import motion, vision
from fp.localization import Localizer, ROBOT_SENSOR_OFFSET_CM


def _loc(x=0.0, y=0.0):
    loc = Localizer()
    loc.set_position(x, y)
    return loc


@pytest.fixture(autouse=True)
def reset_motion_globals():
    motion._last_localization = 0.0
    yield


def test_grab_block_calls_gripper_close():
    loc = _loc()
    with patch("fp.motion.hal.gripper_close") as mock_close, \
         patch("fp.motion.move_reverse"), \
         patch("fp.motion.pivot_to_heading"), \
         patch("fp.motion.vision.detect_block", return_value=None):
        motion.grab_block(loc, "red")
    mock_close.assert_called_once()


def test_grab_block_returns_false_when_block_not_visible():
    loc = _loc()
    with patch("fp.motion.hal.gripper_close"), \
         patch("fp.motion.hal.gripper_open"), \
         patch("fp.motion.move_reverse"), \
         patch("fp.motion.pivot_to_heading"), \
         patch("fp.motion.vision.detect_block", return_value=None):
        assert motion.grab_block(loc, "red") is False


def test_grab_block_returns_true_when_block_within_reach():
    loc = _loc()
    det = vision.DetectionResult(label="red", cx=480, cy=360,
                                 area=vision.IN_REACH_TOLERANCE["red"] + 1)
    with patch("fp.motion.hal.gripper_close"), \
         patch("fp.motion.move_reverse"), \
         patch("fp.motion.pivot_to_heading"), \
         patch("fp.motion.vision.detect_block", return_value=det):
        assert motion.grab_block(loc, "red") is True


def test_release_block_calls_gripper_open():
    loc = _loc()
    with patch("fp.motion.hal.gripper_open") as mock_open, \
         patch("fp.motion.move_reverse"):
        motion.release_block(loc)
    mock_open.assert_called_once()


def test_get_wall_distance_returns_none_on_invalid_wall():
    loc = _loc()
    result = motion._get_wall_distance(loc, "ceiling")
    assert result is None


def _wall_distance_mocks(reading=40.0):
    """Context manager providing all mocks needed by _get_wall_distance."""
    return [
        patch("fp.motion.pivot_to_heading", return_value=True),
        patch("fp.motion._pivot_to_heading", return_value=True),
        patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True),
        patch("fp.motion.hal.get_ultrasonic_distance", return_value=reading),
        patch("fp.motion.hal.recalibrate_heading"),
        patch("fp.motion.time.sleep"),
    ]


@pytest.mark.parametrize("wall,expected_heading", [
    ("north", 0),
    ("east", 90),
    ("south", 180),
    ("west", 270),
])
def test_get_wall_distance_pivots_to_correct_heading(wall, expected_heading):
    loc = _loc()
    pivoted = []

    def _fake_pivot(l, target):
        pivoted.append(target)
        return True

    with patch("fp.motion.pivot_to_heading", side_effect=_fake_pivot), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=40.0), \
         patch("fp.motion.hal.recalibrate_heading"), \
         patch("fp.motion.time.sleep"):
        motion._get_wall_distance(loc, wall)

    assert expected_heading in pivoted


def test_get_wall_distance_north_returns_sensor_reading():
    loc = _loc()
    with patch("fp.motion.pivot_to_heading", return_value=True), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=50.0), \
         patch("fp.motion.hal.recalibrate_heading"), \
         patch("fp.motion.time.sleep"):
        result = motion._get_wall_distance(loc, "north")
    # heading 0 -> angle_of_incidence=0 -> cos=1 -> reading + sensor_offset
    assert result == pytest.approx(50.0 + ROBOT_SENSOR_OFFSET_CM)


def test_get_wall_distance_returns_none_when_no_healthy_readings():
    loc = _loc()
    with patch("fp.motion.pivot_to_heading", return_value=True), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=False), \
         patch("fp.motion.time.sleep"):
        result = motion._get_wall_distance(loc, "north")
    assert result is None


# ----- _get_wall_distance fallback drift gate --------------------------------
#
# The fallback path runs whenever _fit_wall_geometry returns None. Pivots are
# mocked to no-op, so localizer.pose.heading stays at whatever we seed it to;
# that value becomes perp_imu_heading via the argmin pick, which makes
# fallback_drift = (heading - known) wrapped.

def test_get_wall_distance_fallback_recalibrates_when_drift_below_gate():
    # |fallback_drift| = 5 < 10 -> recalibrate as before.
    loc = _loc()
    loc.pose.heading = 5.0
    with patch("fp.motion.pivot_to_heading", return_value=True), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=40.0), \
         patch("fp.motion.hal.recalibrate_heading") as mock_recal, \
         patch("fp.motion._fit_wall_geometry", return_value=None), \
         patch("fp.motion.time.sleep"):
        result = motion._get_wall_distance(loc, "north")

    assert result == pytest.approx(40.0 + ROBOT_SENSOR_OFFSET_CM)
    mock_recal.assert_called_once_with(0.0)


def test_get_wall_distance_fallback_skips_recal_when_drift_exceeds_gate():
    # |fallback_drift| = 15 >= 10 -> recal skipped, IMU left alone.
    # Clearance is still computed from the resampled reading and returned.
    loc = _loc()
    loc.pose.heading = 15.0
    with patch("fp.motion.pivot_to_heading", return_value=True), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=40.0), \
         patch("fp.motion.hal.recalibrate_heading") as mock_recal, \
         patch("fp.motion._fit_wall_geometry", return_value=None), \
         patch("fp.motion.time.sleep"):
        result = motion._get_wall_distance(loc, "north")

    assert result == pytest.approx(40.0 + ROBOT_SENSOR_OFFSET_CM)
    mock_recal.assert_not_called()


# ----- _read_ultrasonic_median() ---------------------------------------------

def test_read_ultrasonic_median_returns_median_of_three_valid_samples():
    # MIN_VALID_SENSOR_CM uses strict > exclusion, so 10.0 is gated out.
    # Survivors: [30.0, 20.0] -> sorted [20.0, 30.0] -> index 1 -> 30.0.
    readings = iter([10.0, 30.0, 20.0])
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance",
               side_effect=lambda: next(readings)), \
         patch("fp.motion.time.sleep"):
        assert motion._read_ultrasonic_median() == pytest.approx(30.0)


def test_read_ultrasonic_median_drops_stuck_high_sample():
    # 340 cm simulates a stuck-high HC-SR04 echo - upper gate must drop it,
    # leaving the median of the two remaining in-range samples.
    readings = iter([20.0, 340.0, 22.0])
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance",
               side_effect=lambda: next(readings)), \
         patch("fp.motion.time.sleep"):
        assert motion._read_ultrasonic_median() == pytest.approx(22.0)


def test_read_ultrasonic_median_drops_near_zero_sample():
    # ~0 cm simulates a missed-echo timeout - lower gate must drop it,
    # without it the 1/r linearisation in _fit_wall_geometry would explode.
    readings = iter([1.0, 25.0, 27.0])
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance",
               side_effect=lambda: next(readings)), \
         patch("fp.motion.time.sleep"):
        assert motion._read_ultrasonic_median() == pytest.approx(27.0)


def test_read_ultrasonic_median_returns_none_when_unhealthy():
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=False), \
         patch("fp.motion.time.sleep"):
        assert motion._read_ultrasonic_median() is None


def test_read_ultrasonic_median_returns_none_when_all_out_of_range():
    readings = iter([1.0, 2.0, 3.0])
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance",
               side_effect=lambda: next(readings)), \
         patch("fp.motion.time.sleep"):
        assert motion._read_ultrasonic_median() is None


# ----- _buffered_ultrasonic_median() -----------------------------------------

def test_buffered_ultrasonic_median_returns_median_of_buffer():
    # MIN_VALID_SENSOR_CM uses strict > exclusion, so 10.0 is gated out.
    # Survivors: [30, 20, 25, 15] -> sorted [15, 20, 25, 30] -> index 2 -> 25.0.
    buffered = [10.0, 30.0, 20.0, 25.0, 15.0]
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_recent_ultrasonic_readings", return_value=buffered):
        assert motion._buffered_ultrasonic_median() == pytest.approx(25.0)


def test_buffered_ultrasonic_median_drops_outliers():
    # 0.5 simulates a missed echo, 340.0 a stuck-high echo; both must be
    # gated out before the median is taken.
    buffered = [0.5, 25.0, 340.0, 27.0]
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_recent_ultrasonic_readings", return_value=buffered):
        assert motion._buffered_ultrasonic_median() == pytest.approx(27.0)


def test_buffered_ultrasonic_median_returns_none_when_unhealthy():
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=False):
        assert motion._buffered_ultrasonic_median() is None


def test_buffered_ultrasonic_median_returns_none_when_buffer_empty():
    # Until the first ping completes, hal's deque(maxlen=5) is empty even
    # though is_ultrasonic_healthy may already be True; helper must not
    # IndexError or return a stale value.
    with patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_recent_ultrasonic_readings", return_value=[]):
        assert motion._buffered_ultrasonic_median() is None


# ----- _fit_wall_geometry() --------------------------------------------------

def _synthetic_sweep(d, theta_perp_deg, span_deg=20, step_deg=5):
    """Build perfect r(theta) = d/cos(theta - theta_perp) sweep data."""
    thetas = [theta_perp_deg + offset
              for offset in range(-span_deg, span_deg + step_deg, step_deg)]
    ranges = [d / math.cos(math.radians(t - theta_perp_deg)) for t in thetas]
    return thetas, ranges


def test_fit_recovers_clean_inputs_exactly():
    thetas, ranges = _synthetic_sweep(d=50.0, theta_perp_deg=12.0)
    d, theta_perp, rms = motion._fit_wall_geometry(thetas, ranges)
    assert d == pytest.approx(50.0, abs=1e-6)
    assert theta_perp == pytest.approx(12.0, abs=1e-6)
    assert rms == pytest.approx(0.0, abs=1e-9)


def test_fit_recovers_west_wall_through_360_wrap():
    # theta_perp near 270 - confirms atan2 + mod 360 lands in [0, 360).
    thetas, ranges = _synthetic_sweep(d=35.0, theta_perp_deg=270.0)
    d, theta_perp, rms = motion._fit_wall_geometry(thetas, ranges)
    assert d == pytest.approx(35.0, abs=1e-6)
    assert theta_perp == pytest.approx(270.0, abs=1e-6)


def test_fit_tolerates_small_noise():
    thetas, ranges = _synthetic_sweep(d=50.0, theta_perp_deg=12.0)
    # Deterministic ~1 cm perturbation pattern - simulates real HC-SR04 jitter
    # without random non-determinism in the test.
    noise = [0.4, -0.7, 0.9, -0.3, 0.5, -0.6, 0.2, 0.8, -0.5]
    noisy_ranges = [r + n for r, n in zip(ranges, noise)]
    d, theta_perp, rms = motion._fit_wall_geometry(thetas, noisy_ranges)
    assert d == pytest.approx(50.0, abs=1.0)
    assert theta_perp == pytest.approx(12.0, abs=1.0)
    assert rms < 1.5


def test_fit_returns_none_when_too_few_samples():
    assert motion._fit_wall_geometry([0.0, 5.0], [50.0, 50.2]) is None
    assert motion._fit_wall_geometry([0.0], [50.0]) is None
    assert motion._fit_wall_geometry([], []) is None


def test_fit_returns_none_when_all_thetas_identical():
    # Singular: cos and sin are constants across samples, so AtA is rank 1.
    assert motion._fit_wall_geometry([42.0, 42.0, 42.0], [50.0, 50.5, 49.8]) is None


def test_fit_returns_none_when_model_predicts_non_physical_range():
    # Engineered so the LS lands on (u, v) that produces denom <= 0 at one
    # sweep angle - the geometric model is violated and the fit must refuse.
    thetas = [0.0, 45.0, 100.0]
    ranges = [50.0, 50.0, 5.0]
    assert motion._fit_wall_geometry(thetas, ranges) is None


def test_move_to_point_already_at_target_does_nothing():
    loc = _loc(x=10.0, y=10.0)
    with patch("fp.motion.pivot_to_heading") as mock_pivot, \
         patch("fp.motion._drive") as mock_drive:
        motion.move_to_point(loc, 10.0, 10.0)
    mock_pivot.assert_not_called()
    mock_drive.assert_not_called()


def test_move_to_point_north_target():
    loc = _loc(x=0.0, y=0.0)
    headings = []
    with patch("fp.motion.pivot_to_heading",
               side_effect=lambda l, h: headings.append(h) or True), \
         patch("fp.motion._drive", return_value=True):
        motion.move_to_point(loc, 0.0, 10.0)
    assert headings[0] == pytest.approx(0.0, abs=1e-6)


def test_move_to_point_east_target():
    loc = _loc(x=0.0, y=0.0)
    headings = []
    with patch("fp.motion.pivot_to_heading",
               side_effect=lambda l, h: headings.append(h) or True), \
         patch("fp.motion._drive", return_value=True):
        motion.move_to_point(loc, 10.0, 0.0)
    assert headings[0] == pytest.approx(90.0, abs=1e-6)


def test_pivot_to_heading_already_at_target_returns_true():
    loc = _loc()
    loc.pose.heading = 90.0
    with patch("fp.motion.hal.set_individual_duties") as mock_duties, \
         patch("fp.motion.hal.stop_motors"):
        result = motion._pivot_to_heading(loc, 90.0)
    assert result is True
    mock_duties.assert_not_called()


def test_pivot_to_heading_drives_motors_until_on_target():
    loc = _loc()
    call_n = [0]

    def _fake_heading():
        call_n[0] += 1
        return 0.0 if call_n[0] < 3 else 90.0

    with patch("fp.motion.hal.get_heading", side_effect=_fake_heading), \
         patch("fp.motion.hal.set_individual_duties") as mock_duties, \
         patch("fp.motion.hal.stop_motors"), \
         patch("fp.motion.time.monotonic", return_value=0.0):
        motion._pivot_to_heading(loc, 90.0)

    assert mock_duties.call_count >= 1


def test_pivot_to_heading_returns_false_on_deadline():
    loc = _loc()
    calls = [0]

    def _tick():
        calls[0] += 1
        return 0.0 if calls[0] == 1 else 1000.0

    with patch("fp.motion.hal.get_heading", return_value=45.0), \
         patch("fp.motion.hal.set_individual_duties"), \
         patch("fp.motion.hal.stop_motors"), \
         patch("fp.motion.time.monotonic", side_effect=_tick):
        result = motion._pivot_to_heading(loc, 270.0)
    assert result is False


# -- localize() ---------------------------------------------------------------

def test_localize_near_west_wall_sets_x():
    loc = _loc(x=40.0, y=100.0)
    with patch("fp.motion._get_wall_distance", return_value=40.0) as mock_gwd:
        result = motion.localize(loc)
    assert result is True
    assert loc.pose.x_cm == pytest.approx(40.0)
    assert loc.pose.y_cm == pytest.approx(100.0)


def test_localize_near_east_wall_sets_x():
    from fp import arena as _arena
    loc = _loc(x=180.0, y=100.0)
    with patch("fp.motion._get_wall_distance", return_value=33.0):
        result = motion.localize(loc)
    assert result is True
    assert loc.pose.x_cm == pytest.approx(_arena.ARENA_EAST_WALL_X_CM - 33.0)
    assert loc.pose.y_cm == pytest.approx(100.0)


def test_localize_near_south_wall_sets_y():
    loc = _loc(x=100.0, y=40.0)
    with patch("fp.motion._get_wall_distance", return_value=40.0):
        result = motion.localize(loc)
    assert result is True
    assert loc.pose.y_cm == pytest.approx(40.0)
    assert loc.pose.x_cm == pytest.approx(100.0)


def test_localize_near_north_wall_sets_y():
    from fp import arena as _arena
    loc = _loc(x=100.0, y=180.0)
    with patch("fp.motion._get_wall_distance", return_value=33.0):
        result = motion.localize(loc)
    assert result is True
    assert loc.pose.y_cm == pytest.approx(_arena.ARENA_NORTH_WALL_Y_CM - 33.0)
    assert loc.pose.x_cm == pytest.approx(100.0)


def test_localize_center_returns_false():
    loc = _loc(x=106.5, y=106.5)
    with patch("fp.motion._get_wall_distance") as mock_gwd:
        result = motion.localize(loc)
    mock_gwd.assert_not_called()
    assert result is False
    assert loc.pose.x_cm == pytest.approx(106.5)
    assert loc.pose.y_cm == pytest.approx(106.5)


def test_localize_returns_false_when_wall_distance_fails():
    loc = _loc(x=40.0, y=100.0)
    with patch("fp.motion._get_wall_distance", return_value=None):
        result = motion.localize(loc)
    assert result is False


def test_localize_skips_when_recently_localized():
    # Cooldown gate: a localize call within MIN_TIME_BETWEEN_LOCALIZATIONS
    # of the last successful localization returns True without re-running
    # the wall sweep.
    loc = _loc(x=40.0, y=100.0)
    motion._last_localization = 1000.0
    with patch("fp.motion.time.monotonic", return_value=1005.0), \
         patch("fp.motion._get_wall_distance") as mock_gwd:
        result = motion.localize(loc)
    mock_gwd.assert_not_called()
    assert result is True


def test_localize_bypass_cooldown_runs_wall_sweep():
    # bypass_cooldown=True forces a fresh wall sweep regardless of when
    # the last localize ran. This is the path the FSM uses before the
    # carry-home leg so a stale localize doesn't gate out the refresh
    # we need before the long drive across the arena.
    loc = _loc(x=40.0, y=100.0)
    motion._last_localization = 1000.0
    with patch("fp.motion.time.monotonic", return_value=1005.0), \
         patch("fp.motion._get_wall_distance", return_value=40.0) as mock_gwd:
        result = motion.localize(loc, bypass_cooldown=True)
    mock_gwd.assert_called()
    assert result is True


def test_localize_force_heading_reads_wall():
    # Pose within sensor range of the west wall so the new
    # expected-wall-distance guard lets the force path through.
    loc = _loc(x=40.0, y=100.0)
    motion._last_localization = 0.0
    pivots = []
    with patch("fp.motion.pivot_to_heading",
               side_effect=lambda l, h: pivots.append(h) or True), \
         patch("fp.motion._pivot_to_heading", return_value=True), \
         patch("fp.motion.hal.is_ultrasonic_healthy", return_value=True), \
         patch("fp.motion.hal.get_ultrasonic_distance", return_value=40.0), \
         patch("fp.motion.hal.recalibrate_heading"), \
         patch("fp.motion.time.sleep"):
        result = motion.localize(loc, force=270.0)
    assert result is True
    assert 270 in pivots


def test_localize_force_skipped_when_wall_out_of_sensor_range():
    # Pose puts the implied south wall (heading 180 -> wall=south) at
    # ~165 cm away, well beyond the ultrasonic's 60 cm + 20 cm robot
    # half-extent envelope. The force handler must reject without
    # running the sweep - the field log that prompted this guard had
    # exactly this scenario teleporting y by 115 cm.
    loc = _loc(x=169.4, y=165.2)
    motion._last_localization = 0.0
    with patch("fp.motion._get_wall_distance") as mock_gwd:
        result = motion.localize(loc, force=180.0, bypass_cooldown=True)
    assert result is False
    mock_gwd.assert_not_called()
    # Position must be untouched - the whole point of the guard.
    assert loc.pose.x_cm == pytest.approx(169.4)
    assert loc.pose.y_cm == pytest.approx(165.2)


def test_localize_force_proceeds_when_wall_in_sensor_range():
    # Symmetric case: pose at y=40 puts the south wall (y=0) at 40 cm,
    # well inside the ultrasonic envelope. Sweep must run.
    loc = _loc(x=100.0, y=40.0)
    motion._last_localization = 0.0
    with patch("fp.motion._get_wall_distance", return_value=40.0) as mock_gwd:
        result = motion.localize(loc, force=180.0, bypass_cooldown=True)
    assert result is True
    mock_gwd.assert_called_once()
    assert loc.pose.y_cm == pytest.approx(40.0)


def test_localize_force_skipped_for_each_cardinal_when_far():
    # Belt-and-suspenders: the guard fires for all four wall directions,
    # not just south. Center-of-arena pose puts every wall at ~106 cm,
    # all of which exceed 60 + 20 = 80 cm sensor envelope.
    loc = _loc(x=106.5, y=106.5)
    motion._last_localization = 0.0
    for force_heading in (0.0, 90.0, 180.0, 270.0):
        with patch("fp.motion._get_wall_distance") as mock_gwd:
            result = motion.localize(loc, force=force_heading, bypass_cooldown=True)
        assert result is False, f"force={force_heading} should be guarded"
        mock_gwd.assert_not_called()


# -- move_reverse() -----------------------------------------------------------

def test_move_reverse_calls_drive_backward():
    loc = _loc(x=100.0, y=100.0)
    with patch("fp.motion._drive", return_value=True) as mock_drive:
        motion.move_reverse(loc)
    assert mock_drive.called
    assert mock_drive.call_args[0][3] == -1


def test_move_reverse_reduces_distance_when_full_backup_is_invalid():
    # Reverse safe-y floor with REVERSE_EXTRA_MARGIN_CM=10 is
    # 0 + 15 + 20 + 10 = 45 cm. Starting at y=52 with default heading 0
    # (north), a 10 cm reverse lands at y=42 (invalid: 42 < 45).
    # REVERSE_REPLAN_ATTEMPTS=1 -> fallback = 10.0 - 10.0/1 = 0.0 cm
    # -> distance_cm <= 0 guard fires -> returns False without calling _drive.
    loc = _loc(x=100.0, y=52.0)
    with patch("fp.motion._drive") as mock_drive:
        result = motion.move_reverse(loc)
    assert result is False
    mock_drive.assert_not_called()


def test_move_reverse_rejected_when_within_extra_margin_of_wall():
    # Even though is_valid_point would pass without the extra margin
    # (e.g. reversing from y=42 to y=32 is barely inside the bare 35 cm
    # safe floor), the REVERSE_EXTRA_MARGIN_CM buffer pushes the floor
    # up to 45 cm and rejects both the full and reduced reverses. The
    # function returns False without invoking _drive - the failure mode
    # observed where odometry drift let "valid" 5 cm reverses land on
    # the north wall is what this guards.
    loc = _loc(x=100.0, y=42.0)
    with patch("fp.motion._drive") as mock_drive:
        result = motion.move_reverse(loc)
    assert result is False
    mock_drive.assert_not_called()


def test_move_reverse_far_from_walls_uses_full_distance():
    # Regression guard: nothing should change for reverses started far
    # from any wall - the new margin only kicks in near the boundary.
    loc = _loc(x=100.0, y=100.0)
    with patch("fp.motion._drive", return_value=True) as mock_drive:
        motion.move_reverse(loc)
    driven_distance = mock_drive.call_args[0][1]
    assert driven_distance == pytest.approx(motion.REVERSE_DISTANCE_CM)


# -- pivot_to_heading() public wrapper -----------------------------------------

def test_pivot_to_heading_public_already_on_target_skips_inner_pivot():
    loc = _loc()
    loc.pose.heading = 90.0
    with patch("fp.motion._pivot_to_heading") as mock_pvt:
        result = motion.pivot_to_heading(loc, 90.0)
    assert result is True
    mock_pvt.assert_not_called()


# -- approach_visual() --------------------------------------------------------

def test_approach_visual_returns_true_when_within_reach():
    loc = _loc(x=100.0, y=100.0)
    # cx aligned with the gripper axis (SENSOR_RESOLUTION[0]/2 - SENSOR_RESOLUTION[0]/10)
    # so |steer_deg| stays under APPROACH_FINAL_CENTERING_DEG and the
    # final centering branch is a no-op here.
    cx_aligned = vision.SENSOR_RESOLUTION[0] // 2 - vision.SENSOR_RESOLUTION[0] // 10
    det = vision.DetectionResult(label="red", cx=cx_aligned, cy=180,
                                 area=vision.IN_REACH_TOLERANCE["red"] + 1)
    with patch("fp.motion.vision.detect_block", return_value=det), \
         patch("fp.motion.pivot_to_heading") as mock_pivot:
        result = motion.approach_visual(loc, "red")
    assert result is True
    mock_pivot.assert_not_called()


def test_approach_visual_returns_true_when_within_reach_off_axis():
    # Final off-axis centering pivot was removed from approach_visual.
    # Once within_reach is True the function returns True immediately
    # regardless of centroid position.
    loc = _loc(x=100.0, y=100.0)
    sw = vision.SENSOR_RESOLUTION[0]
    cx_off = sw  # far right edge - centroid_error well past tolerance
    det = vision.DetectionResult(label="red", cx=cx_off, cy=180,
                                 area=vision.IN_REACH_TOLERANCE["red"] + 1)
    with patch("fp.motion.vision.detect_block", return_value=det), \
         patch("fp.motion.pivot_to_heading") as mock_pivot:
        result = motion.approach_visual(loc, "red")
    assert result is True
    mock_pivot.assert_not_called()


def test_approach_visual_returns_false_when_no_detection():
    loc = _loc(x=100.0, y=100.0)
    with patch("fp.motion.vision.detect_block", return_value=None):
        result = motion.approach_visual(loc, "red")
    assert result is False


def test_approach_visual_returns_false_on_timeout():
    loc = _loc(x=100.0, y=100.0)
    det = vision.DetectionResult(label="red", cx=240, cy=180, area=100)
    # monotonic calls: deadline calc, loop condition check, timeout print
    with patch("fp.motion.time.monotonic", side_effect=[0.0, 31.0, 31.0]), \
         patch("fp.motion.vision.detect_block", return_value=det):
        result = motion.approach_visual(loc, "red", max_time_s=30.0)
    assert result is False


# -- _drive() invalid direction -------------------------------------------------

def test_drive_invalid_direction_returns_false():
    loc = _loc()
    result = motion._drive(loc, 20.0, 0.0, 0)
    assert result is False


# -- approach_visual() additional branches -------------------------------------

def test_approach_visual_aborts_in_collection_zone():
    from fp import arena
    cx, cy = arena.COLLECTION_ZONE_CENTER
    loc = _loc(x=cx, y=cy)
    with patch("fp.motion.vision.detect_block") as mock_detect:
        result = motion.approach_visual(loc, "red")
    assert result is False
    mock_detect.assert_not_called()


def test_approach_visual_raises_stuck_after_three_drive_failures():
    loc = _loc(x=100.0, y=100.0)
    # area=100: within_reach=False, block_too_close=False -> _drive path always taken
    det = vision.DetectionResult(label="red", cx=vision.SENSOR_RESOLUTION[0] // 2,
                                 cy=180, area=100)
    # monotonic: deadline-set, then one call per while-check (3 iterations before raise)
    with patch("fp.motion.time.monotonic", side_effect=[0.0, 0.1, 0.1, 0.1]), \
         patch("fp.motion.vision.detect_block", return_value=det), \
         patch("fp.motion.pivot_to_heading"), \
         patch("fp.motion._drive", return_value=False):
        with pytest.raises(motion.ApproachVisualStuck):
            motion.approach_visual(loc, "red", max_time_s=30.0)


def test_approach_visual_skips_drive_when_block_too_close_and_off_axis():
    # When block_too_close AND steer_deg >= VISUAL_APPROACH_TOLERANCE,
    # approach_visual skips _drive and continues steering instead.
    loc = _loc(x=100.0, y=100.0)
    # SENSOR_RESOLUTION[0]=720 in tests (overridden by test_vision.py module-level mutation).
    # cx=700 -> centroid_error = 700 - 360 + 14 = 354 -> steer_deg well above 1.0 tolerance
    # area=TOO_CLOSE_TOLERANCE["red"]=70_000 -> block_too_close=True, within_reach=False
    det = vision.DetectionResult(label="red", cx=700, cy=180,
                                 area=vision.TOO_CLOSE_TOLERANCE["red"])
    # monotonic: deadline, while-enter, while-exit, timeout-logger
    with patch("fp.motion.time.monotonic", side_effect=[0.0, 0.1, 31.0, 31.0]), \
         patch("fp.motion.vision.detect_block", return_value=det), \
         patch("fp.motion.pivot_to_heading"), \
         patch("fp.motion._drive") as mock_drive:
        result = motion.approach_visual(loc, "red", max_time_s=30.0)
    assert result is False
    mock_drive.assert_not_called()
