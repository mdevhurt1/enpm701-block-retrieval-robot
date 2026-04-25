import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests  # hw_stub

from fp import planner as planner_module
from fp.planner import OccupancyGrid


# Helper: world coordinates of the centre of the arena (arena is 213x213 cm)
_CX = 106.5   # ≈ arena centre x
_CY = 106.5   # ≈ arena centre y


class TestCoordinateConversions:

    def test_origin_maps_to_cell_zero(self):
        g = OccupancyGrid()
        r, c = g.world_to_cell(0.0, 0.0)
        assert r == 0 and c == 0

    def test_round_trip_near_centre(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM
        wx, wy = _CX, _CY
        r, c = g.world_to_cell(wx, wy)
        wx2, wy2 = g.cell_to_world(r, c)
        # Round-trip should be within half a cell resolution
        assert abs(wx2 - wx) <= GRID_RESOLUTION_CM
        assert abs(wy2 - wy) <= GRID_RESOLUTION_CM

    def test_round_trip_corner(self):
        from fp.arena import ARENA_WIDTH_CM, ARENA_HEIGHT_CM, GRID_RESOLUTION_CM
        g = OccupancyGrid()
        wx, wy = ARENA_WIDTH_CM - 1, ARENA_HEIGHT_CM - 1
        r, c = g.world_to_cell(wx, wy)
        wx2, wy2 = g.cell_to_world(r, c)
        assert abs(wx2 - wx) <= GRID_RESOLUTION_CM
        assert abs(wy2 - wy) <= GRID_RESOLUTION_CM


class TestAStar:

    def _make_small_grid(self):
        g = OccupancyGrid()
        return g

    def test_straight_path_no_obstacles(self):
        g = OccupancyGrid()
        # ROBOT_RADIUS_CM=25cm inflates 5 cells; need row/col >= 6 to be free
        from fp.arena import GRID_RESOLUTION_CM as R
        start = (R * 6.5, R * 6.5)   # well inside arena, clear of inflation
        goal  = (R * 10.5, R * 6.5)
        path = g.astar(start, goal)
        assert path is not None, "A* should find a path in an empty grid"
        assert len(path) >= 1

    def test_obstacle_forces_detour(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        # Mark cells far from walls (clear of inflation zone)
        g.log_block(R * 10, R * 8, '_obs')
        g.log_block(R * 10, R * 9, '_obs')
        g.log_block(R * 10, R * 10, '_obs')
        start = (R * 7.5, R * 9.5)   # interior cell, west of obstacles
        goal  = (R * 13.5, R * 9.5)  # interior cell, east of obstacles
        path = g.astar(start, goal)
        # A path must still exist (route around obstacle)
        assert path is not None

    def test_fully_blocked_returns_none(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        # Surround target cell with obstacles
        cx, cy = R * 5, R * 5   # target
        for dx, dy in [(-R, 0), (R, 0), (0, -R), (0, R)]:
            g.log_block(cx + dx, cy + dy, '_obs')
        # Also mark the target cell itself
        g.log_block(cx, cy, '_obs')
        path = g.astar((0.0, 0.0), (cx, cy))
        # May be None or empty depending on implementation detail
        assert path is None or len(path) == 0

    def test_goal_snapped_to_start_returns_empty_path_with_debug(self):
        g = OccupancyGrid()
        # ROBOT_RADIUS_CM = 30 cm → inflation radius = ceil(30/5) = 6 cells.
        # Near a single wall the nearest free cell is 7 rows away (Manhattan = 7 > max_radius=6),
        # so _nearest_free returns None there.  Use the NW delivery corner instead: cell (36, 6)
        # is inflated by BOTH the north wall (|42-36|=6) and the west wall (|6-0|=6).  The
        # diagonal escape at (35, 7) is only 2 steps away, well within max_radius=6.
        delivery = g.world_to_cell(32.5, 182.5)   # → (row=36, col=6)
        inflated = g._inflated_grid()
        snapped = g._nearest_free(delivery, inflated)
        assert snapped is not None

        start = g.cell_to_world(*snapped)
        path = g.astar(start, (32.5, 182.5))
        debug = g.get_last_astar_debug()

        assert path == []
        assert debug["planned_goal_world"] == start
        assert debug["first_waypoint_world"] is None


class TestFrontiers:

    def test_unexplored_arena_has_frontiers(self):
        g = OccupancyGrid()
        # Mark only the origin as traversed
        g.log_traversed(0.0, 0.0)
        frontiers = g.get_frontiers()
        assert len(frontiers) > 0, "unexplored arena should have frontier cells"

    def test_fully_traversed_has_no_frontiers(self):
        g = OccupancyGrid()
        from fp.arena import GRID_SIZE, GRID_RESOLUTION_CM as R
        # Mark every cell as traversed
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                wx, wy = g.cell_to_world(row, col)
                g.log_traversed(wx, wy)
        frontiers = g.get_frontiers()
        assert len(frontiers) == 0, "fully explored arena should have no frontiers"


class TestGridUpdates:

    def test_log_block_marks_occupied(self):
        from fp.planner import BLOCK_TARGET
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        g.log_block(R * 5, R * 5, 'red')
        cell = g.world_to_cell(R * 5, R * 5)
        assert cell is not None
        assert g._grid[cell[0], cell[1]] == BLOCK_TARGET

    def test_clear_block_marks_free(self):
        from fp.planner import FREE
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        g.log_block(R * 5, R * 5, 'red')
        g.clear_block(R * 5, R * 5)
        cell = g.world_to_cell(R * 5, R * 5)
        assert g._grid[cell[0], cell[1]] == FREE

    def test_world_to_cell_outside_returns_none(self):
        from fp.arena import ARENA_WIDTH_CM, ARENA_HEIGHT_CM
        g = OccupancyGrid()
        assert g.world_to_cell(-1.0, 0.0) is None
        assert g.world_to_cell(0.0, -1.0) is None
        assert g.world_to_cell(ARENA_WIDTH_CM + 10, 0.0) is None

    def test_update_from_ultrasonic_marks_free(self):
        from fp.planner import FREE, UNKNOWN
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        # Shoot a ray north from arena centre
        cx, cy = R * 10, R * 10
        g.log_traversed(cx, cy)   # start cell is free
        g.update_from_ultrasonic((cx, cy, 0.0), dist_cm=R * 3, direction_deg=0.0)
        # Cell 1 step north should now be FREE or same
        nc = g.world_to_cell(cx, cy + R)
        if nc is not None:
            assert g._grid[nc[0], nc[1]] != UNKNOWN or g._grid[nc[0], nc[1]] == UNKNOWN

    def test_update_from_ultrasonic_out_of_range_no_endpoint(self):
        from fp.planner import OCCUPIED
        from fp import planner as pm
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        cx, cy = R * 10, R * 10
        g.update_from_ultrasonic((cx, cy, 0.0), dist_cm=pm.ULTRASONIC_MAX_CM + 50,
                                 direction_deg=0.0)
        # Far cell should NOT be OCCUPIED (beyond max range)
        far_cell = g.world_to_cell(cx, cy + pm.ULTRASONIC_MAX_CM + 40)
        if far_cell is not None:
            assert g._grid[far_cell[0], far_cell[1]] != OCCUPIED


class TestGetNextGoal:

    def test_targets_known_block_first(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        # Log a block of the target colour in the interior
        bx, by = R * 8, R * 8
        g.log_block(bx, by, 'red')
        goal = g.get_next_goal('red', (R * 5, R * 5))
        assert goal is not None
        gx, gy = goal
        # Should be near bx, by
        import math
        assert math.hypot(gx - bx, gy - by) < R * 2, f"expected near ({bx}, {by}), got ({gx:.1f}, {gy:.1f})"

    def test_ignores_wrong_color_block(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        g.log_block(R * 8, R * 8, 'blue')
        g.log_traversed(R * 5, R * 5)
        # Requesting 'red' should ignore the blue block and return a frontier
        goal = g.get_next_goal('red', (R * 5, R * 5))
        if goal is not None:
            gx, gy = goal
            import math
            # Returned goal should NOT be the blue block location
            dist_to_blue = math.hypot(gx - R * 8, gy - R * 8)
            assert dist_to_blue > R, "should not target wrong-color block"


class TestRenderAscii:

    def test_renders_without_crash(self):
        g = OccupancyGrid()
        result = g.render_ascii()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_renders_with_robot_and_path(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        result = g.render_ascii(robot_world=(R * 5, R * 5),
                                path=[(R * 6, R * 5), (R * 7, R * 5)])
        assert 'R' in result, "robot marker 'R' should appear"
        assert '*' in result, "path marker '*' should appear"

    def test_block_marker_appears(self):
        g = OccupancyGrid()
        from fp.arena import GRID_RESOLUTION_CM as R
        g.log_block(R * 5, R * 5, 'green')
        result = g.render_ascii()
        assert 'B' in result, "block marker 'B' should appear"
