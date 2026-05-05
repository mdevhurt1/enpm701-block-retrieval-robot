import pytest
from unittest.mock import patch
from fp import search, arena, vision, localization
from fp.localization import Localizer


def _loc(x=100.0, y=100.0):
    loc = Localizer()
    loc.set_position(x, y)
    return loc


def test_facing_collection_zone_nw_corner():
    # Position inside the NW collection zone - always True regardless of heading.
    loc = _loc(x=30.0, y=arena.ARENA_HEIGHT_CM - 30.0)
    assert search._facing_collection_zone(loc)


def test_not_facing_collection_zone_se_corner():
    # Far SE corner - too far from zone.
    loc = _loc(x=arena.ARENA_WIDTH_CM - 10.0, y=10.0)
    assert not search._facing_collection_zone(loc)


def test_facing_collection_zone_inside_zone():
    loc = _loc(
        x=arena.COLLECTION_ZONE_SIZE_CM / 2,
        y=arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM / 2,
    )
    assert search._facing_collection_zone(loc)


def test_not_facing_collection_zone_far_away():
    loc = _loc(x=arena.ARENA_WIDTH_CM - 10.0, y=10.0)
    assert not search._facing_collection_zone(loc)


def test_scan_360_returns_none_when_robot_in_collection_zone():
    # scan_360 bails immediately when the robot is inside the collection zone
    # (there are no blocks to collect there, and the scan would be useless).
    planner = search.SearchPlanner()
    cx, cy = arena.COLLECTION_ZONE_CENTER
    loc = _loc(x=cx, y=cy)
    with patch("fp.search.vision.detect_all_colors") as mock_detect:
        result = planner.scan_360(loc, "red")
    assert result is None
    mock_detect.assert_not_called()


def test_scan_360_marks_visited():
    planner = search.SearchPlanner()
    loc = _loc(x=100.0, y=100.0)
    _no_det = {"red": None, "blue": None, "green": None}
    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", return_value=_no_det), \
         patch("fp.search._facing_collection_zone", return_value=False):
        planner.scan_360(loc, "red", n_steps=6, skip_collection_zone=False)
    cell = planner._cell(100.0, 100.0)
    assert planner._visits.get(cell, 0) == 1


def test_scan_360_returns_detection_on_third_step():
    planner = search.SearchPlanner()
    loc = _loc(x=100.0, y=100.0)
    det = vision.DetectionResult(label="red", cx=480, cy=360, area=5000)
    call_count = [0]
    _no_det = {"red": None, "blue": None, "green": None}

    def _fake_detect_all():
        call_count[0] += 1
        return {"red": det, "blue": None, "green": None} if call_count[0] == 3 else _no_det

    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", side_effect=_fake_detect_all), \
         patch("fp.search._facing_collection_zone", return_value=False):
        result = planner.scan_360(loc, "red", n_steps=12, skip_collection_zone=False)

    assert result is det
    assert call_count[0] == 3
    cell = planner._cell(100.0, 100.0)
    assert planner._visits.get(cell, 0) == 1


def test_scan_360_skips_zone_facing_steps():
    planner = search.SearchPlanner()
    loc = _loc()
    detect_calls = [0]

    def _fake_detect_all():
        detect_calls[0] += 1
        return {"red": None, "blue": None, "green": None}

    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", side_effect=_fake_detect_all), \
         patch("fp.search._facing_collection_zone", return_value=True):
        planner.scan_360(loc, "red", n_steps=6, skip_collection_zone=True)

    assert detect_calls[0] == 0


def test_valid_cells_in_bounds():
    RES = search.GRID_RESOLUTION_CM
    margin = localization.WALL_CLEARANCE_CM + localization.ROBOT_HALF_EXTENT_CM
    x_lo = max(arena.ARENA_WEST_WALL_X_CM + margin,
               arena.COLLECTION_ZONE_SIZE_CM + localization.ROBOT_HALF_EXTENT_CM)
    x_hi = arena.ARENA_EAST_WALL_X_CM - margin
    y_lo = arena.ARENA_SOUTH_WALL_Y_CM + margin
    y_hi = arena.ARENA_NORTH_WALL_Y_CM - margin
    planner = search.SearchPlanner()
    assert len(planner._valid_cells) > 0
    for col, row in planner._valid_cells:
        cx = col * RES + RES / 2
        cy = row * RES + RES / 2
        assert x_lo <= cx <= x_hi, f"cx={cx} out of [{x_lo}, {x_hi}]"
        assert y_lo <= cy <= y_hi, f"cy={cy} out of [{y_lo}, {y_hi}]"


def test_mark_visited_increments_count():
    planner = search.SearchPlanner()
    planner.mark_visited(150.0, 100.0)
    planner.mark_visited(150.0, 100.0)
    cell = planner._cell(150.0, 100.0)
    assert planner._visits[cell] == 2


def test_next_pose_prefers_unvisited():
    planner = search.SearchPlanner()
    target_cell = planner._valid_cells[0]
    for cell in planner._valid_cells[1:]:
        planner._visits[cell] = 1
    RES = search.GRID_RESOLUTION_CM
    expected = (target_cell[0] * RES + RES / 2, target_cell[1] * RES + RES / 2)
    for _ in range(20):
        assert planner.next_pose() == expected


def test_next_pose_prefers_least_visited():
    planner = search.SearchPlanner()
    cells = planner._valid_cells
    low_cells = cells[: len(cells) // 2]
    high_cells = cells[len(cells) // 2:]
    for cell in low_cells:
        planner._visits[cell] = 1
    for cell in high_cells:
        planner._visits[cell] = 2
    RES = search.GRID_RESOLUTION_CM
    low_centers = {(c[0] * RES + RES / 2, c[1] * RES + RES / 2) for c in low_cells}
    for _ in range(30):
        assert planner.next_pose() in low_centers


def test_get_search_quality_unvisited_returns_100():
    planner = search.SearchPlanner()
    assert planner.get_search_quality(100.0, 100.0) == pytest.approx(100.0)


def test_get_search_quality_visited_cell_returns_percentage():
    planner = search.SearchPlanner()
    # Visit every valid cell once, then a specific cell twice.
    # min_count across all valid cells = 1, so quality = (1/2)*100 = 50%.
    for cell in planner._valid_cells:
        planner._visits[cell] = 1
    target_cell = planner._valid_cells[0]
    planner._visits[target_cell] = 2
    RES = search.GRID_RESOLUTION_CM
    cx = target_cell[0] * RES + RES / 2
    cy = target_cell[1] * RES + RES / 2
    quality = planner.get_search_quality(cx, cy)
    assert quality == pytest.approx(50.0)  # 100 * (1/2)


def test_get_sighting_returns_none_when_empty():
    planner = search.SearchPlanner()
    assert planner.get_sighting("red") is None


def test_get_sighting_pops_most_recent_sighting():
    planner = search.SearchPlanner()
    planner._record_sighting("red", 100.0, 100.0, 45.0)
    sighting = planner.get_sighting("red")
    assert sighting == (100.0, 100.0, 45.0)
    # Popped - second call should return None.
    assert planner.get_sighting("red") is None


def test_record_sighting_deduplicates_within_grid_resolution():
    planner = search.SearchPlanner()
    planner._record_sighting("blue", 100.0, 100.0, 0.0)
    # Second call from nearly the same spot (within one grid cell).
    planner._record_sighting("blue", 100.0 + search.GRID_RESOLUTION_CM * 0.5, 100.0, 90.0)
    assert len(planner._sightings["blue"]) == 1


def test_record_sighting_allows_distant_positions():
    planner = search.SearchPlanner()
    planner._record_sighting("blue", 50.0, 50.0, 0.0)
    far = 50.0 + search.GRID_RESOLUTION_CM + 1.0
    planner._record_sighting("blue", far, 50.0, 90.0)
    assert len(planner._sightings["blue"]) == 2


def test_has_sighting_in_cell_empty_returns_false():
    planner = search.SearchPlanner()
    assert planner.has_sighting_in_cell(50.0, 50.0, "blue") is False


def test_has_sighting_in_cell_matches_same_cell_same_color():
    planner = search.SearchPlanner()
    planner._record_sighting("blue", 50.0, 50.0, 0.0)
    # Query inside the same grid cell as the sighting (within GRID_RESOLUTION_CM).
    assert planner.has_sighting_in_cell(55.0, 55.0, "blue") is True


def test_has_sighting_in_cell_rejects_different_cell():
    planner = search.SearchPlanner()
    planner._record_sighting("blue", 50.0, 50.0, 0.0)
    far = 50.0 + 2 * search.GRID_RESOLUTION_CM
    assert planner.has_sighting_in_cell(far, 50.0, "blue") is False


def test_has_sighting_in_cell_rejects_other_color():
    # A green sighting in the cell should NOT register as a blue sighting -
    # the override should fire only when the sighting matches the active
    # target, otherwise an unrelated stale sighting would force a scan.
    planner = search.SearchPlanner()
    planner._record_sighting("green", 90.0, 90.0, 0.0)
    assert planner.has_sighting_in_cell(95.0, 95.0, "blue") is False
    assert planner.has_sighting_in_cell(95.0, 95.0, "green") is True


# ----- clear_sightings_in_cell ------------------------------------------------

def test_clear_sightings_in_cell_empty_cache_returns_zero():
    planner = search.SearchPlanner()
    assert planner.clear_sightings_in_cell(50.0, 50.0, "red") == 0


def test_clear_sightings_in_cell_removes_matching_color_in_cell():
    planner = search.SearchPlanner()
    planner._record_sighting("red", 50.0, 50.0, 0.0)
    removed = planner.clear_sightings_in_cell(55.0, 55.0, "red")
    assert removed == 1
    assert planner.has_sighting_in_cell(55.0, 55.0, "red") is False


def test_clear_sightings_in_cell_leaves_other_colors_alone():
    planner = search.SearchPlanner()
    planner._record_sighting("red", 50.0, 50.0, 0.0)
    planner._record_sighting("blue", 50.0, 50.0, 0.0)
    removed = planner.clear_sightings_in_cell(55.0, 55.0, "red")
    assert removed == 1
    assert planner.has_sighting_in_cell(55.0, 55.0, "red") is False
    assert planner.has_sighting_in_cell(55.0, 55.0, "blue") is True


def test_clear_sightings_in_cell_leaves_other_cells_alone():
    planner = search.SearchPlanner()
    planner._record_sighting("red", 50.0, 50.0, 0.0)
    far = 50.0 + 2 * search.GRID_RESOLUTION_CM
    planner._record_sighting("red", far, 50.0, 0.0)
    removed = planner.clear_sightings_in_cell(55.0, 55.0, "red")
    assert removed == 1
    assert planner.has_sighting_in_cell(55.0, 55.0, "red") is False
    assert planner.has_sighting_in_cell(far, 50.0, "red") is True


def test_scan_360_caches_nontarget_sightings():
    planner = search.SearchPlanner()
    loc = _loc(x=100.0, y=100.0)
    blue_det = vision.DetectionResult(label="blue", cx=480, cy=360, area=5000)
    _no_red = {"red": None, "blue": blue_det, "green": None}

    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", return_value=_no_red), \
         patch("fp.search._facing_collection_zone", return_value=False):
        result = planner.scan_360(loc, "red", n_steps=4, skip_collection_zone=False)

    assert result is None
    assert planner.get_sighting("blue") is not None


def test_scan_360_clears_stale_target_sightings_in_current_cell():
    # Stale sighting for the current target at the robot's cell should be
    # invalidated at the start of scan_360 - if the block was pushed away
    # since the original observation, the upcoming scan tells the truth
    # and we don't want to keep navigating back to an empty cell.
    planner = search.SearchPlanner()
    loc = _loc(x=100.0, y=100.0)
    planner._record_sighting("red", 100.0, 100.0, 0.0)
    assert planner.has_sighting_in_cell(100.0, 100.0, "red") is True

    no_detect = {"red": None, "blue": None, "green": None}
    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", return_value=no_detect), \
         patch("fp.search._facing_collection_zone", return_value=False):
        planner.scan_360(loc, "red", n_steps=1, skip_collection_zone=False)

    assert planner.has_sighting_in_cell(100.0, 100.0, "red") is False


def test_scan_360_does_not_clear_other_color_sightings_in_current_cell():
    # The clear is target-specific: a blue sighting at our current cell
    # while we scan for red should remain - it's still useful intel for
    # the next time blue becomes the target.
    planner = search.SearchPlanner()
    loc = _loc(x=100.0, y=100.0)
    planner._record_sighting("blue", 100.0, 100.0, 0.0)

    no_detect = {"red": None, "blue": None, "green": None}
    with patch("fp.search.hal.get_heading", return_value=0.0), \
         patch("fp.search.motion.pivot_to_heading"), \
         patch("fp.search.motion._pivot_to_heading"), \
         patch("fp.search.vision.detect_all_colors", return_value=no_detect), \
         patch("fp.search._facing_collection_zone", return_value=False):
        planner.scan_360(loc, "red", n_steps=1, skip_collection_zone=False)

    assert planner.has_sighting_in_cell(100.0, 100.0, "blue") is True


# -- _angle_in_arc() ---------------------------------------------------------

@pytest.mark.parametrize("angle,arc_start,arc_end,expected", [
    (90.0,  45.0,  135.0, True),   # inside arc
    (180.0, 45.0,  135.0, False),  # outside arc
    (45.0,  45.0,  135.0, True),   # at arc start (inclusive)
    (135.0, 45.0,  135.0, True),   # at arc end (inclusive)
    (350.0, 340.0, 20.0,  True),   # inside wraparound arc
    (180.0, 340.0, 20.0,  False),  # outside wraparound arc
])
def test_angle_in_arc(angle, arc_start, arc_end, expected):
    assert search._angle_in_arc(angle, arc_start, arc_end) is expected


# -- _arcs_overlap() ---------------------------------------------------------

def test_arcs_overlap_overlapping_returns_true():
    # [0 deg, 90 deg] and [45 deg, 135 deg] share [45 deg, 90 deg]
    assert search._arcs_overlap(0.0, 90.0, 45.0, 135.0) is True


def test_arcs_overlap_disjoint_returns_false():
    # [0 deg, 45 deg] and [90 deg, 135 deg] share nothing
    assert search._arcs_overlap(0.0, 45.0, 90.0, 135.0) is False


def test_arcs_overlap_wraparound_returns_true():
    # [300 deg, 60 deg] wraps through 0 and overlaps [0 deg, 30 deg]
    assert search._arcs_overlap(300.0, 60.0, 0.0, 30.0) is True
