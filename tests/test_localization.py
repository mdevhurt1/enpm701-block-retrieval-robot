import math
import pytest
from fp.localization import Localizer, ROBOT_SENSOR_OFFSET_CM
from fp import arena


def test_pose_defaults():
    loc = Localizer()
    assert loc.pose.x_cm == 0.0
    assert loc.pose.y_cm == 0.0
    assert loc.pose.heading == 0.0


def test_set_position_roundtrip():
    loc = Localizer()
    loc.set_position(50.0, 100.0)
    assert loc.pose.x_cm == 50.0
    assert loc.pose.y_cm == 100.0


def test_apply_forward_north():
    loc = Localizer()
    loc.apply_forward(10.0, 0.0)
    assert loc.pose.x_cm == pytest.approx(0.0, abs=1e-9)
    assert loc.pose.y_cm == pytest.approx(10.0, abs=1e-9)


def test_apply_forward_east():
    loc = Localizer()
    loc.apply_forward(10.0, 90.0)
    assert loc.pose.x_cm == pytest.approx(10.0, abs=1e-9)
    assert loc.pose.y_cm == pytest.approx(0.0, abs=1e-6)


def test_apply_reverse_moves_opposite():
    loc = Localizer()
    loc.set_position(0.0, 10.0)
    loc.apply_reverse(10.0, 0.0)
    assert loc.pose.y_cm == pytest.approx(0.0, abs=1e-9)


def test_apply_forward_accumulates():
    loc = Localizer()
    loc.apply_forward(5.0, 0.0)
    loc.apply_forward(5.0, 0.0)
    assert loc.pose.y_cm == pytest.approx(10.0, abs=1e-9)


def test_apply_forward_updates_heading():
    loc = Localizer()
    loc.apply_forward(10.0, 45.0)
    assert loc.pose.heading == pytest.approx(45.0)


def test_ultrasonic_to_robot_center_heading_zero():
    loc = Localizer()
    result = loc.ultrasonic_to_robot_center(50.0, 0.0)
    # Heading 0 -> angle_of_incidence=0 -> cos=1 -> reading + offset
    assert result == pytest.approx(50.0 + ROBOT_SENSOR_OFFSET_CM)


def test_ultrasonic_reading_inside_robot_radius_close():
    loc = Localizer()
    assert loc.ultrasonic_reading_inside_robot_radius(1.0, 0.0)


def test_ultrasonic_reading_inside_robot_radius_far():
    loc = Localizer()
    assert not loc.ultrasonic_reading_inside_robot_radius(200.0, 0.0)


def test_is_valid_point_center_of_arena():
    loc = Localizer()
    assert loc.is_valid_point(arena.ARENA_WIDTH_CM / 2, arena.ARENA_HEIGHT_CM / 2)


def test_is_valid_point_outside_bounds():
    loc = Localizer()
    assert not loc.is_valid_point(0.0, 0.0)
    assert not loc.is_valid_point(arena.ARENA_WIDTH_CM, arena.ARENA_HEIGHT_CM)


def test_is_valid_point_extra_margin_shrinks_envelope():
    # Without extra_margin, a point near the edge of the safe envelope
    # is valid. With extra_margin large enough to cross that edge, the
    # same point becomes invalid. This is the path move_reverse uses to
    # buffer reverses against odometry drift.
    loc = Localizer()
    # 1 cm inside the north safe boundary - valid by default.
    edge_point_y = arena.ARENA_NORTH_WALL_Y_CM - 15.0 - 20.0 - 1.0
    assert loc.is_valid_point(100.0, edge_point_y)
    # Adding 5 cm extra margin pushes the point outside the shrunken envelope.
    assert not loc.is_valid_point(100.0, edge_point_y, extra_margin=5.0)


def test_is_valid_point_extra_margin_default_is_zero():
    # Regression guard: existing call sites pass no extra_margin and must
    # see identical behaviour to before.
    loc = Localizer()
    pt_x = arena.ARENA_WIDTH_CM / 2
    pt_y = arena.ARENA_HEIGHT_CM / 2
    assert loc.is_valid_point(pt_x, pt_y) == loc.is_valid_point(pt_x, pt_y, extra_margin=0.0)


def test_apply_forward_south():
    loc = Localizer()
    loc.set_position(0.0, 10.0)
    loc.apply_forward(10.0, 180.0)
    assert loc.pose.x_cm == pytest.approx(0.0, abs=1e-6)
    assert loc.pose.y_cm == pytest.approx(0.0, abs=1e-6)


def test_apply_forward_west():
    loc = Localizer()
    loc.set_position(10.0, 0.0)
    loc.apply_forward(10.0, 270.0)
    assert loc.pose.x_cm == pytest.approx(0.0, abs=1e-6)
    assert loc.pose.y_cm == pytest.approx(0.0, abs=1e-6)


def test_ultrasonic_to_robot_center_at_45_degrees():
    loc = Localizer()
    reading = 50.0
    result = loc.ultrasonic_to_robot_center(reading, 45.0)
    # heading 45 -> angle_of_incidence=45 (not folded since 45 is not > 45)
    expected = (reading + ROBOT_SENSOR_OFFSET_CM) * math.cos(math.radians(45))
    assert result == pytest.approx(expected, rel=1e-6)


def test_ultrasonic_to_robot_center_angle_fold_symmetry():
    # Heading 44 and 46 should fold to the same angle_of_incidence (44 deg).
    loc = Localizer()
    reading = 50.0
    result_44 = loc.ultrasonic_to_robot_center(reading, 44.0)
    result_46 = loc.ultrasonic_to_robot_center(reading, 46.0)
    assert result_44 == pytest.approx(result_46, rel=1e-6)
