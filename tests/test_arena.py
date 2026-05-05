import math
import pytest
from fp import arena


def test_apply_motion_north():
    x, y = arena.apply_motion(0.0, 0.0, 10.0, 0.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(10.0, abs=1e-9)


def test_apply_motion_east():
    x, y = arena.apply_motion(0.0, 0.0, 10.0, 90.0)
    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_apply_motion_south():
    x, y = arena.apply_motion(0.0, 0.0, 10.0, 180.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-10.0, abs=1e-9)


def test_apply_motion_west():
    x, y = arena.apply_motion(0.0, 0.0, 10.0, 270.0)
    assert x == pytest.approx(-10.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_apply_motion_northeast_diagonal():
    d = math.sqrt(2)
    x, y = arena.apply_motion(0.0, 0.0, d, 45.0)
    assert x == pytest.approx(1.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)


def test_apply_motion_zero_distance():
    x, y = arena.apply_motion(5.0, 7.0, 0.0, 90.0)
    assert x == pytest.approx(5.0)
    assert y == pytest.approx(7.0)


def test_apply_motion_from_nonzero_origin():
    x, y = arena.apply_motion(10.0, 20.0, 5.0, 0.0)
    assert x == pytest.approx(10.0, abs=1e-9)
    assert y == pytest.approx(25.0, abs=1e-9)


def test_in_launch_zone_center():
    cx = arena.LAUNCH_ZONE_SIZE_CM / 2
    cy = arena.LAUNCH_ZONE_SIZE_CM / 2
    assert arena.in_launch_zone(cx, cy)


def test_in_launch_zone_sw_corner():
    assert arena.in_launch_zone(0.0, 0.0)


def test_in_launch_zone_ne_corner():
    assert arena.in_launch_zone(arena.LAUNCH_ZONE_SIZE_CM, arena.LAUNCH_ZONE_SIZE_CM)


def test_not_in_launch_zone_just_east():
    assert not arena.in_launch_zone(arena.LAUNCH_ZONE_SIZE_CM + 0.1,
                                    arena.LAUNCH_ZONE_SIZE_CM / 2)


def test_not_in_launch_zone_just_north():
    assert not arena.in_launch_zone(arena.LAUNCH_ZONE_SIZE_CM / 2,
                                    arena.LAUNCH_ZONE_SIZE_CM + 0.1)


def test_not_in_launch_zone_negative():
    assert not arena.in_launch_zone(-0.1, 10.0)


def test_in_collection_zone_center():
    cx = arena.COLLECTION_ZONE_SIZE_CM / 2
    cy = arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM / 2
    assert arena.in_collection_zone(cx, cy)


def test_in_collection_zone_nw_corner():
    assert arena.in_collection_zone(0.0, arena.ARENA_HEIGHT_CM)


def test_in_collection_zone_se_edge():
    assert arena.in_collection_zone(
        arena.COLLECTION_ZONE_SIZE_CM,
        arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM,
    )


def test_not_in_collection_zone_just_east():
    assert not arena.in_collection_zone(
        arena.COLLECTION_ZONE_SIZE_CM + 0.1,
        arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM / 2,
    )


def test_not_in_collection_zone_just_south():
    assert not arena.in_collection_zone(
        arena.COLLECTION_ZONE_SIZE_CM / 2,
        arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM - 0.1,
    )


def test_constants_arena_positive():
    assert arena.ARENA_WIDTH_CM > 0
    assert arena.ARENA_HEIGHT_CM > 0


def test_constants_zone_sizes_positive():
    assert arena.LAUNCH_ZONE_SIZE_CM > 0
    assert arena.COLLECTION_ZONE_SIZE_CM > 0


def test_constants_walls():
    assert arena.ARENA_SOUTH_WALL_Y_CM == 0.0
    assert arena.ARENA_WEST_WALL_X_CM == 0.0
    assert arena.ARENA_NORTH_WALL_Y_CM == arena.ARENA_HEIGHT_CM
    assert arena.ARENA_EAST_WALL_X_CM == arena.ARENA_WIDTH_CM


def test_launch_zone_center_inside_zone():
    assert arena.in_launch_zone(*arena.LAUNCH_ZONE_CENTER)


def test_collection_zone_center_inside_zone():
    assert arena.in_collection_zone(*arena.COLLECTION_ZONE_CENTER)


def test_in_launch_zone_east_edge():
    assert arena.in_launch_zone(arena.LAUNCH_ZONE_SIZE_CM, arena.LAUNCH_ZONE_SIZE_CM / 2)


def test_in_collection_zone_east_edge():
    assert arena.in_collection_zone(
        arena.COLLECTION_ZONE_SIZE_CM,
        arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM / 2,
    )


def test_not_in_collection_zone_negative():
    assert not arena.in_collection_zone(-0.1, arena.ARENA_HEIGHT_CM)
