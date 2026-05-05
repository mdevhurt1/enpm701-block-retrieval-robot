import logging
import math

from fp import arena
from fp import hal
from fp import motion
from fp import vision
from fp import localization
from fp.localization import Localizer

logger = logging.getLogger(__name__)

# Grid search tuning constants
GRID_RESOLUTION_CM = 35.0

# Collection zone avoidance tuning constants
_HALF_FOV_DEG = vision.CAMERA_HFOV_DEG / 2.0
_ZONE_PAD_DEG = 5.0
_MAX_DIST_CM = 60.0

# Collection zone rectangle bounds (NW corner of arena).
_ZX_MIN = 0.0
_ZX_MAX = arena.COLLECTION_ZONE_SIZE_CM
_ZY_MIN = arena.ARENA_HEIGHT_CM - arena.COLLECTION_ZONE_SIZE_CM
_ZY_MAX = arena.ARENA_HEIGHT_CM
_ZONE_CORNERS = [(_ZX_MIN, _ZY_MIN), (_ZX_MAX, _ZY_MIN),
                 (_ZX_MIN, _ZY_MAX), (_ZX_MAX, _ZY_MAX)]


def _angle_in_arc(angle: float, arc_start: float, arc_end: float) -> bool:
    return (angle - arc_start) % 360 <= (arc_end - arc_start) % 360


def _arcs_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return (_angle_in_arc(a_start, b_start, b_end) or
            _angle_in_arc(a_end, b_start, b_end) or
            _angle_in_arc(b_start, a_start, a_end) or
            _angle_in_arc(b_end, a_start, a_end))


def _facing_collection_zone(loc: Localizer) -> bool:
    rx, ry = loc.pose.x_cm, loc.pose.y_cm

    nearest_x = max(_ZX_MIN, min(rx, _ZX_MAX))
    nearest_y = max(_ZY_MIN, min(ry, _ZY_MAX))
    if math.hypot(nearest_x - rx, nearest_y - ry) > _MAX_DIST_CM:
        return False

    if _ZX_MIN <= rx <= _ZX_MAX and _ZY_MIN <= ry <= _ZY_MAX:
        logger.info("[SEARCH] Facing collection zone")
        return True

    bearings = sorted(
        math.degrees(math.atan2(cx - rx, cy - ry)) % 360
        for cx, cy in _ZONE_CORNERS
    )
    # Largest angular gap between consecutive corner bearings; zone arc is the complement.
    gaps = [(bearings[(i + 1) % 4] - bearings[i]) % 360 for i in range(4)]
    largest_gap_idx = gaps.index(max(gaps))
    zone_arc_start = (bearings[(largest_gap_idx + 1) % 4] - _ZONE_PAD_DEG) % 360
    zone_arc_end = (bearings[largest_gap_idx] + _ZONE_PAD_DEG) % 360

    heading = hal.get_heading() % 360
    loc.set_heading(heading)
    fov_start = (heading - _HALF_FOV_DEG) % 360
    fov_end = (heading + _HALF_FOV_DEG) % 360

    if _arcs_overlap(fov_start, fov_end, zone_arc_start, zone_arc_end):
        logger.info(f"[SEARCH] Facing collection zone")
        return True

    return False


def _cell_center(col: int, row: int) -> tuple[float, float]:
    """Return the (x, y) center of a grid cell in cm."""
    half = GRID_RESOLUTION_CM / 2
    return (col * GRID_RESOLUTION_CM + half, row * GRID_RESOLUTION_CM + half)


class SearchPlanner:
    def __init__(self) -> None:
        margin = localization.WALL_CLEARANCE_CM + localization.ROBOT_HALF_EXTENT_CM
        x_lo = max(arena.ARENA_WEST_WALL_X_CM + margin,
                   arena.COLLECTION_ZONE_SIZE_CM + localization.ROBOT_HALF_EXTENT_CM)
        x_hi = arena.ARENA_EAST_WALL_X_CM - margin
        y_lo = arena.ARENA_SOUTH_WALL_Y_CM + margin
        y_hi = arena.ARENA_NORTH_WALL_Y_CM - margin

        cols = int(arena.ARENA_WIDTH_CM / GRID_RESOLUTION_CM) + 1
        rows = int(arena.ARENA_HEIGHT_CM / GRID_RESOLUTION_CM) + 1

        # Keep only cells whose centers fall inside the safe bounds.
        valid: set[tuple[int, int]] = set()
        for col in range(cols):
            for row in range(rows):
                cx, cy = _cell_center(col, row)
                if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
                    valid.add((col, row))

        # Build lawnmower order: north-to-south strips in west-to-east order snaking.
        self._lawnmower_order: list[tuple[int, int]] = []
        for i, col in enumerate(sorted(set(c for c, _ in valid))):
            north_to_south = (i % 2 == 0)
            strip = sorted((r for c, r in valid if c == col), reverse=north_to_south)
            self._lawnmower_order.extend((col, row) for row in strip)

        self._valid_cells = self._lawnmower_order
        self._visits: dict[tuple[int, int], int] = {}
        self._sightings: dict[str, list[tuple[float, float, float]]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // GRID_RESOLUTION_CM), int(y // GRID_RESOLUTION_CM))

    def mark_visited(self, x: float, y: float) -> None:
        cell = self._cell(x, y)
        self._visits[cell] = self._visits.get(cell, 0) + 1

    def next_pose(self, robot_x: float = 0.0, robot_y: float = 0.0) -> tuple[float, float]:
        min_count = min(self._visits.get(c, 0) for c in self._valid_cells)

        # Among least-visited cells, pick the one nearest to the robot.
        best_cell = None
        best_dist = float('inf')
        for cell in self._valid_cells:
            if self._visits.get(cell, 0) > min_count:
                continue
            cx, cy = _cell_center(*cell)
            dist = math.hypot(cx - robot_x, cy - robot_y)
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        x, y = _cell_center(*best_cell)
        logger.info(f"[SEARCH] next_pose: cell ({best_cell[0]}, {best_cell[1]}) center ({x:.1f}, {y:.1f}) distance {best_dist:.1f} [min_count={min_count}]")
        return (x, y)

    def scan_360(self, localizer: Localizer, target: str,
                 n_steps: int = 12,
                 skip_collection_zone: bool = True
                 ) -> vision.DetectionResult:
        self.mark_visited(localizer.pose.x_cm, localizer.pose.y_cm)
        # Drop any stale target-colour sighting at the current cell before
        # scanning. The scan is about to give us ground truth for this cell,
        # so a prior sighting (which may point at a block that has since
        # been pushed elsewhere) shouldn't survive to lure us back here on
        # a future search cycle.
        self.clear_sightings_in_cell(localizer.pose.x_cm, localizer.pose.y_cm, target)

        if arena.in_collection_zone(localizer.pose.x_cm, localizer.pose.y_cm):
            return None

        n_steps = max(1, n_steps)
        step_deg = 360.0 / n_steps
        start_heading = localizer.pose.heading

        for i in range(n_steps):
            parts = []
            if skip_collection_zone and _facing_collection_zone(localizer):
                logger.info(f"[SEARCH] scan_360: step {i + 1}/{n_steps} skipped: camera facing collection zone")
                motion.pivot_to_heading(localizer, (localizer.pose.heading + step_deg) % 360.0)
                continue

            detections = vision.detect_all_colors()
            for color, det in detections.items():
                if det is None:
                    parts.append(f"{color}: none")
                else:
                    parts.append(f"{color}: area={det.area:.0f} cx={det.cx} cy={det.cy}")
            logger.info("  " + "  |  ".join(parts))
            
            target_detection = detections.get(target)
            if target_detection is not None:
                logger.info(f"[SEARCH] scan_360: spotted {target} at scan step {i + 1}/{n_steps} area {target_detection.area:.1f}")
                return target_detection

            # Cache non-target sightings for future search cycles.
            for color, other_detection in detections.items():
                if color != target and other_detection is not None:
                    self._record_sighting(color, localizer.pose.x_cm, localizer.pose.y_cm, localizer.pose.heading)

            motion.pivot_to_heading(localizer, (localizer.pose.heading + step_deg) % 360)

        motion.pivot_to_heading(localizer, start_heading)
        return None

    def get_search_quality(self, x: float, y: float) -> float:
        cell = self._cell(x, y)
        visits = self._visits.get(cell, 0)
        if visits == 0:
            return 100.0
        min_count = min(self._visits.get(c, 0) for c in self._valid_cells)
        return (min_count / visits) * 100.0

    def has_sighting_in_cell(self, x: float, y: float, color: str) -> bool:
        """True if `color`'s sighting list has an entry inside the cell containing (x, y).

        A cached sighting is intelligence that a block of this colour was
        visible from this cell during a prior scan. The FSM uses this to
        override the skip-low-quality path in _on_search: even when the
        cell would otherwise be skipped as already-searched, a sighting
        of the *current target* forces a re-scan because the block we
        previously saw may still be there. Sightings of other colours
        do not override the skip - they're irrelevant to the current
        search and the cell stays "spent" from the planner's view.
        """
        cell = self._cell(x, y)
        for sx, sy, _ in self._sightings.get(color, ()):
            if self._cell(sx, sy) == cell:
                return True
        return False

    def clear_sightings_in_cell(self, x: float, y: float, color: str) -> int:
        """Drop sightings of `color` whose recorded position falls in the cell containing (x, y).

        Returns the number of entries removed. Called by scan_360 at the
        start of a scan to invalidate any stale sighting of the current
        target at the robot's cell: if the block has been displaced since
        the sighting was recorded, the scan will tell us the truth, and
        keeping the stale entry around would let get_sighting navigate us
        back here on a future cycle even though the block is gone.
        """
        cell = self._cell(x, y)
        sightings = self._sightings.get(color)
        if not sightings:
            return 0
        before = len(sightings)
        self._sightings[color] = [s for s in sightings if self._cell(s[0], s[1]) != cell]
        return before - len(self._sightings[color])

    def _record_sighting(self, color: str, rx: float, ry: float, heading: float) -> None:
        sightings = self._sightings.setdefault(color, [])
        # Avoid duplicates from similar positions (within one grid cell).
        for sx, sy, _ in sightings:
            if math.hypot(sx - rx, sy - ry) < GRID_RESOLUTION_CM:
                return
        sightings.append((rx, ry, heading))

    def get_sighting(self, color: str) -> tuple[float, float, float]:
        sightings = self._sightings.get(color)
        if not sightings:
            return None
        return sightings.pop()
