import heapq
import math
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from fp import hal
from fp import arena

# Grid cell state constants
UNKNOWN      = 0
FREE         = 1
OCCUPIED     = 2
BLOCK_TARGET = 3

# Robot radius used for obstacle inflation in the A* planner.
# Must be >= WALL_CLEARANCE_CM + ROBOT_SENSOR_OFFSET_CM so that the planner
# never generates goals closer to a wall than the hardware's wall_too_close()
# threshold (sensor reads < WALL_CLEARANCE_CM → robot centre must stay at
# least WALL_CLEARANCE_CM + ROBOT_SENSOR_OFFSET_CM from the wall face).
ROBOT_RADIUS_CM = hal.WALL_CLEARANCE_CM + arena.ROBOT_SENSOR_OFFSET_CM  # 30 cm

# Maximum ultrasonic range to trust for mapping (cm).
ULTRASONIC_MAX_CM = 80.0

# Frontier cluster merge radius (cells).  Frontiers closer than this are
# merged into a single centroid waypoint.
FRONTIER_CLUSTER_RADIUS = 4

# Ratio used to detect spurious ultrasonic echoes: if the measured hit is
# shorter than WALL_CLEARANCE_CM while the nearest arena boundary along the
# same ray is farther than SPURIOUS_ECHO_MULTIPLIER * WALL_CLEARANCE_CM, the
# reading is treated as a chassis/cable reflection and the OCCUPIED mark is
# suppressed.  The cleared-ray FREE marks are still applied.
SPURIOUS_ECHO_MULTIPLIER = 3.0


logger = logging.getLogger(__name__)


def _arena_wall_dist(rx: float, ry: float, direction_deg: float) -> float:
    ang = math.radians(direction_deg)
    dx  = math.sin(ang)
    dy  = math.cos(ang)
    dists = []
    if dy > 1e-9:
        dists.append((arena.ARENA_NORTH_WALL_Y_CM - ry) / dy)
    elif dy < -1e-9:
        dists.append((ry - arena.ARENA_SOUTH_WALL_Y_CM) / (-dy))
    if dx > 1e-9:
        dists.append((arena.ARENA_EAST_WALL_X_CM - rx) / dx)
    elif dx < -1e-9:
        dists.append((rx - arena.ARENA_WEST_WALL_X_CM) / (-dx))
    return min((d for d in dists if d > 0), default=float('inf'))


class OccupancyGrid:
    """
    2-D occupancy grid for a rectangular arena.

    Internal storage: self._grid[row, col]
      row 0 corresponds to Y ≈ 0 (south), increasing northward.
      col 0 corresponds to X ≈ 0 (west), increasing eastward.
    """

    def __init__(self):
        self._res  = arena.GRID_RESOLUTION_CM          # cm per cell
        self._size = arena.GRID_SIZE                   # cells per axis
        self._grid = np.full((self._size, self._size), UNKNOWN, dtype=np.uint8)
        # Per-cell colour labels for BLOCK_TARGET cells.
        self._block_colors: Dict[Tuple[int, int], str] = {}

        # Mark boundary cells as OCCUPIED (arena walls).
        self._grid[0,  :]  = OCCUPIED   # south wall  (row 0)
        self._grid[-1, :]  = OCCUPIED   # north wall  (row -1)
        self._grid[:,  0]  = OCCUPIED   # west wall   (col 0)
        self._grid[:, -1]  = OCCUPIED   # east wall   (col -1)

        # Inflated-grid cache: recomputed only when _grid_dirty is True.
        self._grid_dirty: bool = True
        self._inflated_cache: Optional[np.ndarray] = None

        # Diagnostic metadata from the most recent astar() call.
        self._last_astar_debug: Dict[str, object] = {}

    # Coordinate conversions

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Convert arena world coordinates (cm) to (row, col).
        Returns None if the point is outside the grid.
        """
        if x < 0 or y < 0:
            return None
        col = int(x / self._res)
        row = int(y / self._res)
        if 0 <= row < self._size and 0 <= col < self._size:
            return (row, col)
        return None

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Return the centre of a cell in world coordinates (cm)."""
        x = (col + 0.5) * self._res
        y = (row + 0.5) * self._res
        return (x, y)

    # Grid updates

    def update_from_ultrasonic(
        self, pose: Tuple[float, float, float],
        dist_cm: float,
        direction_deg: float,
    ):
        """
        Mark cells along the ultrasonic ray.

        pose          : (x_cm, y_cm, heading_deg) - robot centre
        dist_cm       : measured distance (cm); None or > ULTRASONIC_MAX_CM -> ray only (no hit)
        direction_deg : arena heading the sensor is pointing (usually same as robot heading)

        All cells along the ray up to (but not including) the endpoint are marked FREE.
        If dist_cm is valid and ≤ ULTRASONIC_MAX_CM, the endpoint cell is marked OCCUPIED.
        """
        rx, ry, _ = pose
        ang = math.radians(direction_deg)
        dx  = math.sin(ang)
        dy  = math.cos(ang)

        if dist_cm is None or dist_cm > ULTRASONIC_MAX_CM:
            sweep = ULTRASONIC_MAX_CM
            mark_endpoint = False
        else:
            sweep = dist_cm
            mark_endpoint = True

        changed = False

        # Step along the ray in half-cell increments
        step_cm = self._res / 2.0
        n_steps = int(sweep / step_cm)
        for i in range(n_steps):
            px = rx + dx * i * step_cm
            py = ry + dy * i * step_cm
            cell = self.world_to_cell(px, py)
            if cell is None:
                break
            row, col = cell
            if self._grid[row, col] == UNKNOWN:
                self._grid[row, col] = FREE
                changed = True

        if mark_endpoint:
            # Sanity-check against arena geometry: if the reading is far shorter
            # than the actual arena boundary in this direction, it's a spurious
            # echo (chassis, cable, oblique bounce).  Suppress the OCCUPIED mark
            # so a phantom obstacle is not burned into the map.
            wall_dist = _arena_wall_dist(rx, ry, direction_deg)
            if dist_cm < hal.WALL_CLEARANCE_CM and wall_dist > SPURIOUS_ECHO_MULTIPLIER * hal.WALL_CLEARANCE_CM:
                logger.warning(
                    f"[PLAN] Ignoring spurious ultrasonic hit: "
                    f"reading={dist_cm:.1f} cm but arena wall is {wall_dist:.1f} cm away"
                )
                mark_endpoint = False

        if mark_endpoint:
            end_x = rx + dx * dist_cm
            end_y = ry + dy * dist_cm
            cell = self.world_to_cell(end_x, end_y)
            if cell is not None:
                row, col = cell
                if self._grid[row, col] != BLOCK_TARGET:
                    self._grid[row, col] = OCCUPIED
                    changed = True

        if changed:
            self._grid_dirty = True

    def log_block(self, x: float, y: float, color: str):
        cell = self.world_to_cell(x, y)
        if cell is None:
            return
        row, col = cell
        self._grid[row, col] = BLOCK_TARGET
        self._block_colors[(row, col)] = color
        self._grid_dirty = True
        logger.info(f"[PLAN] Block '{color}' logged at ({x:.1f}, {y:.1f}) -> cell ({row}, {col})")

    def clear_block(self, x: float, y: float):
        """Mark a BLOCK_TARGET cell as FREE (block has been picked up)."""
        cell = self.world_to_cell(x, y)
        if cell is None:
            return
        row, col = cell
        if self._grid[row, col] == BLOCK_TARGET:
            self._grid[row, col] = FREE
            self._block_colors.pop((row, col), None)
            self._grid_dirty = True

    def log_traversed(self, x: float, y: float, radius_cm: float = ROBOT_RADIUS_CM):
        r_cells = int(math.ceil(radius_cm / self._res))
        centre = self.world_to_cell(x, y)
        if centre is None:
            return
        cr, cc = centre
        changed = False
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                if dr * dr + dc * dc > r_cells * r_cells:
                    continue
                row, col = cr + dr, cc + dc
                if 0 < row < self._size - 1 and 0 < col < self._size - 1:
                    if self._grid[row, col] == UNKNOWN:
                        self._grid[row, col] = FREE
                        changed = True
        if changed:
            self._grid_dirty = True

    # Processed grid for planning (inflation)

    def _inflated_grid(self) -> np.ndarray:
        """
        Return a boolean obstacle mask where True = impassable.
        OCCUPIED cells and cells within ROBOT_RADIUS_CM of an OCCUPIED cell
        are marked impassable.  BLOCK_TARGET cells are passable.

        Result is cached; recomputed only when the grid has changed since
        the last call (_grid_dirty flag).
        """
        if not self._grid_dirty and self._inflated_cache is not None:
            return self._inflated_cache

        r = int(math.ceil(ROBOT_RADIUS_CM / self._res))
        obstacle = (self._grid == OCCUPIED)
        kernel_side = 2 * r + 1
        # Pure-NumPy dilation - works without scipy on the Pi.
        pad = r
        padded = np.pad(obstacle, pad, mode='constant', constant_values=False)
        inflated = np.zeros_like(obstacle, dtype=bool)
        for dr in range(kernel_side):
            for dc in range(kernel_side):
                inflated |= padded[dr: dr + self._size, dc: dc + self._size]

        self._inflated_cache = inflated
        self._grid_dirty = False
        return inflated

    # Frontier exploration

    def get_frontiers(self) -> List[Tuple[int, int, int]]:
        """
        Find frontier cells: FREE cells adjacent (8-connected) to at least
        one UNKNOWN cell.  Cluster nearby frontiers and return each cluster as
        a (row, col, size) tuple where (row, col) is the centroid and size is
        the number of cells in the cluster (used for exploration weighting).
        """
        n = self._size
        frontiers = []
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                if self._grid[r, c] != FREE:
                    continue
                # Check 8-connected neighbourhood for UNKNOWN
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if self._grid[r + dr, c + dc] == UNKNOWN:
                            frontiers.append((r, c))
                            break
                    else:
                        continue
                    break

        if not frontiers:
            return []

        # Cluster frontiers using a greedy approach
        clusters: List[List[Tuple[int, int]]] = []
        assigned = set()
        for f in frontiers:
            if f in assigned:
                continue
            cluster = [f]
            assigned.add(f)
            for g in frontiers:
                if g in assigned:
                    continue
                dr = f[0] - g[0]
                dc = f[1] - g[1]
                if math.sqrt(dr * dr + dc * dc) <= FRONTIER_CLUSTER_RADIUS:
                    cluster.append(g)
                    assigned.add(g)
            clusters.append(cluster)

        # Return centroid and size of each cluster
        centroids = []
        for cluster in clusters:
            cr = int(round(sum(f[0] for f in cluster) / len(cluster)))
            cc = int(round(sum(f[1] for f in cluster) / len(cluster)))
            centroids.append((cr, cc, len(cluster)))
        return centroids

    # A* path planning

    def get_last_astar_debug(self) -> Dict[str, object]:
        """Return a copy of diagnostic metadata from the most recent astar() call."""
        return dict(self._last_astar_debug)

    def astar(
        self,
        start_world: Tuple[float, float],
        goal_world: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a collision-free path from *start_world* to *goal_world* using A*.

        Obstacle inflation (ROBOT_RADIUS_CM) is applied before planning.
        The returned path is a list of world-coordinate waypoints (cm),
        smoothed by a line-of-sight pass to remove redundant intermediate
        points.

        Returns None if no path exists.
        """
        start = self.world_to_cell(*start_world)
        goal  = self.world_to_cell(*goal_world)
        self._last_astar_debug = {
            "requested_start_world": tuple(start_world),
            "requested_goal_world": tuple(goal_world),
            "requested_start_cell": start,
            "requested_goal_cell": goal,
            "planned_start_cell": start,
            "planned_goal_cell": goal,
            "planned_start_world": None,
            "planned_goal_world": None,
            "first_waypoint_world": None,
        }
        if start is None or goal is None:
            return None

        inflated = self._inflated_grid()
        if inflated[start[0], start[1]]:
            # Start is inside the inflated obstacle zone (e.g. robot is pressed
            # against a wall).  Snap to the nearest free cell so A* can still
            # find a path out instead of silently returning None.
            logger.warning(
                f"[PLAN] Start cell {start} is inside inflated obstacle - snapping to nearest free"
            )
            start = self._nearest_free(start, inflated)
            if start is None:
                self._last_astar_debug["planned_start_cell"] = None
                return None

        if inflated[goal[0], goal[1]]:
            # Goal is inside an obstacle - snap to nearest free cell
            goal = self._nearest_free(goal, inflated)
            if goal is None:
                self._last_astar_debug["planned_goal_cell"] = None
                return None

        self._last_astar_debug["planned_start_cell"] = start
        self._last_astar_debug["planned_goal_cell"] = goal
        self._last_astar_debug["planned_start_world"] = self.cell_to_world(*start)
        self._last_astar_debug["planned_goal_world"] = self.cell_to_world(*goal)

        # A* on the inflated grid
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score    = {start: 0.0}
        n = self._size

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                break

            r, c = current
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if inflated[nr, nc]:
                    continue
                step = 1.414 if (dr != 0 and dc != 0) else 1.0
                tentative_g = g_score[current] + step
                neighbour   = (nr, nc)
                if tentative_g < g_score.get(neighbour, float('inf')):
                    came_from[neighbour] = current
                    g_score[neighbour]   = tentative_g
                    f = tentative_g + heuristic(neighbour, goal)
                    heapq.heappush(open_heap, (f, neighbour))
        else:
            return None   # open heap exhausted -> no path

        if goal not in came_from:
            return None

        # Reconstruct cell path
        path_cells = []
        node = goal
        while node is not None:
            path_cells.append(node)
            node = came_from[node]
        path_cells.reverse()

        # Drop the start cell: the robot is already there. Keeping it causes
        # navigate_to to aim at the cell *centre* (offset from the robot's
        # actual position), triggering a spurious pivot and a tiny move before
        # any real progress is made. When the snapped goal collapses onto the
        # current cell, this intentionally produces an empty path so the caller
        # can mark the original frontier unreachable instead of reselecting it.
        if path_cells and path_cells[0] == start:
            path_cells = path_cells[1:]

        # Convert to world coordinates
        path_world = [self.cell_to_world(r, c) for r, c in path_cells]

        # Line-of-sight smoothing: keep only waypoints where the LOS changes
        path_world = self._smooth_path(path_world, inflated)
        self._last_astar_debug["first_waypoint_world"] = path_world[0] if path_world else None

        return path_world

    def _nearest_free(
        self,
        cell: Tuple[int, int],
        inflated: np.ndarray,
        max_radius: int = 6,
    ) -> Optional[Tuple[int, int]]:
        r0, c0 = cell
        visited = {cell}
        queue   = deque([cell])
        while queue:
            r, c = queue.popleft()
            if not inflated[r, c]:
                return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited:
                    continue
                if not (0 <= nr < self._size and 0 <= nc < self._size):
                    continue
                if abs(nr - r0) + abs(nc - c0) > max_radius:
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))
        return None

    def _los_clear(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        inflated: np.ndarray,
    ) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        if dist < 1e-6:
            return True
        steps = max(2, int(dist / (self._res / 2)))
        for i in range(steps + 1):
            t  = i / steps
            px = a[0] + t * (b[0] - a[0])
            py = a[1] + t * (b[1] - a[1])
            cell = self.world_to_cell(px, py)
            if cell is None:
                return False
            if inflated[cell[0], cell[1]]:
                return False
        return True

    def _smooth_path(
        self,
        path: List[Tuple[float, float]],
        inflated: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """
        Greedy line-of-sight path smoothing.
        Starting from the first point, try to connect directly to points
        further along the path; keep only the minimal set of waypoints.
        """
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Find the farthest point visible from path[i]
            j = len(path) - 1
            while j > i + 1:
                if self._los_clear(path[i], path[j], inflated):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    # Goal selection

    def get_next_goal(
        self,
        target_color: str,
        robot_world: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """
        Select the next navigation goal.

        Priority:
          1. Known BLOCK_TARGET cell for *target_color* (nearest first).
          2. Nearest frontier centroid (exploration fallback).

        Returns world coordinates (x_cm, y_cm), or None if nothing available.
        """
        rx, ry = robot_world

        # Search for a known block of the target colour
        best_block = None
        best_block_dist = float('inf')
        for (row, col), color in self._block_colors.items():
            if color != target_color:
                continue
            bx, by = self.cell_to_world(row, col)
            d = math.hypot(bx - rx, by - ry)
            if d < best_block_dist:
                best_block_dist = d
                best_block = (bx, by)

        if best_block is not None:
            logger.info(f"[PLAN] Targeting known {target_color} block at "
                  f"({best_block[0]:.1f}, {best_block[1]:.1f}) cm  "
                  f"dist={best_block_dist:.1f} cm")
            return best_block

        # Fallback: frontier scored by distance / sqrt(size).
        # Larger unexplored regions are preferred over small nearby slivers,
        # preventing the robot from greedily hugging one wall direction while
        # leaving large areas unseen.
        frontiers = self.get_frontiers()
        if not frontiers:
            return None

        def _score(rcs):
            gx, gy = self.cell_to_world(rcs[0], rcs[1])
            dist = math.hypot(gx - rx, gy - ry)
            return dist / math.sqrt(rcs[2])

        best = min(frontiers, key=_score)
        goal = self.cell_to_world(best[0], best[1])
        logger.info(f"[PLAN] Exploring frontier at ({goal[0]:.1f}, {goal[1]:.1f}) cm  "
              f"size={best[2]}  ({len(frontiers)} frontiers total)")
        return goal

    # Visualisation (ASCII for test mode)

    def render_ascii(
        self,
        robot_world: Optional[Tuple[float, float]] = None,
        path: Optional[List[Tuple[float, float]]] = None,
    ) -> str:
        """
        Render the occupancy grid as an ASCII string.
        Characters:
          '#'  OCCUPIED    '.'  FREE    '?'  UNKNOWN
          'B'  BLOCK_TARGET  'R'  robot position  '*'  path waypoint
        Rows are printed north-up (row 0 = south at the bottom).
        """
        CHAR = {UNKNOWN: '?', FREE: '.', OCCUPIED: '#', BLOCK_TARGET: 'B'}

        # Collect path cells
        path_cells: set = set()
        if path:
            for wx, wy in path:
                c = self.world_to_cell(wx, wy)
                if c:
                    path_cells.add(c)

        # Robot cell
        robot_cell = None
        if robot_world:
            robot_cell = self.world_to_cell(*robot_world)

        lines = []
        for row in range(self._size - 1, -1, -1):   # north to south for display
            row_chars = []
            for col in range(self._size):
                cell = (row, col)
                if cell == robot_cell:
                    row_chars.append('R')
                elif cell in path_cells:
                    row_chars.append('*')
                else:
                    row_chars.append(CHAR.get(int(self._grid[row, col]), '?'))
            lines.append(''.join(row_chars))
        return '\n'.join(lines)
