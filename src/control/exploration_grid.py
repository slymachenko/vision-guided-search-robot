"""
Dynamic 2D Exploration Grid for Scenario 3 (unknown arena dimensions).
Maps cells into UNKNOWN, FREE, or OCCUPIED using robot traversal and camera FOV rays.
Traversed cells are permanently locked as FREE and can never become OCCUPIED.
"""

from collections import deque
import numpy as np
from src.env.arena import ROBOT_RADIUS, ROBOT_SIZE
from src.control.open_loop import wrap
from src.control.planner import astar_search, prune_waypoints
from src.control.camera import VirtualCamera

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class _PlanningView:
    """Lightweight adapter exposing safe FREE cells to planner.py (astar_search & prune_waypoints)."""

    def __init__(self, exp_grid: "ExplorationGrid", safe_mask: np.ndarray):
        self._g = exp_grid
        self._safe = safe_mask

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return self._g.world_to_grid(x, y)

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        return self._g.grid_to_world(col, row)

    def is_free(self, col: int, row: int) -> bool:
        if 0 <= col < self._g.width and 0 <= row < self._g.height:
            return bool(self._safe[row, col])
        return False

    def line_of_sight(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist < 1e-4:
            return self.is_free(*self.world_to_grid(p1[0], p1[1]))
        n_samples = max(2, int(np.ceil(dist / (self._g.res * 0.5))))
        xs = np.linspace(p1[0], p2[0], n_samples)
        ys = np.linspace(p1[1], p2[1], n_samples)
        for x, y in zip(xs, ys):
            if not self.is_free(*self.world_to_grid(float(x), float(y))):
                return False
        return True


class ExplorationGrid:
    """
    Dynamically-expanding 2D occupancy grid without prior arena bounds.
    - UNKNOWN: unexplored space (targets for frontier exploration)
    - FREE: unobstructed space observed by FOV or physically traversed
    - OCCUPIED: obstacle/wall surface hit by camera FOV (never overwrites traversed FREE cells)
    """

    def __init__(
        self,
        resolution: float = 0.025,
        inflation_radius: float = ROBOT_RADIUS + 0.02,
        initial_margin: float = 1.25,
    ):
        self.res = float(resolution)
        self.inflation_radius = float(inflation_radius)
        self.initial_margin = float(initial_margin)

        r_cells = int(np.ceil(self.inflation_radius / self.res))
        self._dilate_offsets = [
            (dc, dr)
            for dr in range(-r_cells, r_cells + 1)
            for dc in range(-r_cells, r_cells + 1)
            if (dc != 0 or dr != 0) and np.hypot(dc * self.res, dr * self.res) <= self.inflation_radius
        ]

        self.reset()

    def reset(self) -> None:
        """Reset grid to uninitialized empty state."""
        self.x_min = self.x_max = self.y_min = self.y_max = 0.0
        self.width = 0
        self.height = 0
        self.grid = np.empty((0, 0), dtype=np.uint8)
        self.traversed = np.empty((0, 0), dtype=bool)

    def _ensure_bounds(self, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        """Initialize or dynamically expand grid bounds aligned to cell resolution."""
        if self.width == 0:
            self.x_min = float(np.floor((min_x - self.initial_margin) / self.res) * self.res)
            self.y_min = float(np.floor((min_y - self.initial_margin) / self.res) * self.res)
            self.width = int(np.ceil((max_x + self.initial_margin - self.x_min) / self.res))
            self.height = int(np.ceil((max_y + self.initial_margin - self.y_min) / self.res))
            self.x_max = self.x_min + self.width * self.res
            self.y_max = self.y_min + self.height * self.res
            self.grid = np.full((self.height, self.width), UNKNOWN, dtype=np.uint8)
            self.traversed = np.zeros((self.height, self.width), dtype=bool)
            return

        if (
            min_x >= self.x_min + self.res
            and max_x <= self.x_max - self.res
            and min_y >= self.y_min + self.res
            and max_y <= self.y_max - self.res
        ):
            return

        pad = 0.5
        dx_cols = int(np.ceil((self.x_min - (min_x - pad)) / self.res)) if min_x < self.x_min + self.res else 0
        dy_rows = int(np.ceil((self.y_min - (min_y - pad)) / self.res)) if min_y < self.y_min + self.res else 0
        extra_r = int(np.ceil((max_x + pad - self.x_max) / self.res)) if max_x > self.x_max - self.res else 0
        extra_t = int(np.ceil((max_y + pad - self.y_max) / self.res)) if max_y > self.y_max - self.res else 0

        new_w = self.width + dx_cols + extra_r
        new_h = self.height + dy_rows + extra_t
        new_grid = np.full((new_h, new_w), UNKNOWN, dtype=np.uint8)
        new_trav = np.zeros((new_h, new_w), dtype=bool)

        new_grid[dy_rows:dy_rows + self.height, dx_cols:dx_cols + self.width] = self.grid
        new_trav[dy_rows:dy_rows + self.height, dx_cols:dx_cols + self.width] = self.traversed

        self.x_min -= dx_cols * self.res
        self.y_min -= dy_rows * self.res
        self.width = new_w
        self.height = new_h
        self.x_max = self.x_min + new_w * self.res
        self.y_max = self.y_min + new_h * self.res
        self.grid = new_grid
        self.traversed = new_trav

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip((x - self.x_min) / self.res, 0, self.width - 1))
        row = int(np.clip((y - self.y_min) / self.res, 0, self.height - 1))
        return col, row

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        x = self.x_min + (col + 0.5) * self.res
        y = self.y_min + (row + 0.5) * self.res
        return float(x), float(y)

    def mark_traversed(self, q_robot: np.ndarray) -> None:
        """Mark cells under the physical robot body as permanently FREE."""
        rx, ry, theta = float(q_robot[0]), float(q_robot[1]), float(q_robot[2])
        self._ensure_bounds(rx - ROBOT_RADIUS, rx + ROBOT_RADIUS, ry - ROBOT_RADIUS, ry + ROBOT_RADIUS)

        sc, sr = self.world_to_grid(rx, ry)
        self.traversed[sr, sc] = True
        self.grid[sr, sc] = FREE

        c_min, r_min = self.world_to_grid(rx - ROBOT_RADIUS, ry - ROBOT_RADIUS)
        c_max, r_max = self.world_to_grid(rx + ROBOT_RADIUS, ry + ROBOT_RADIUS)
        hx, hy = ROBOT_SIZE[0] / 2.0, ROBOT_SIZE[1] / 2.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                wx, wy = self.grid_to_world(col, row)
                dx, dy = wx - rx, wy - ry
                dx_loc = cos_t * dx + sin_t * dy
                dy_loc = -sin_t * dx + cos_t * dy
                if abs(dx_loc) <= hx and abs(dy_loc) <= hy:
                    self.traversed[row, col] = True
                    self.grid[row, col] = FREE

    def update(self, q_robot: np.ndarray, obstacles: list[dict], camera: VirtualCamera) -> None:
        """Update map from current robot footprint and VirtualCamera FOV rays."""
        rx, ry, _ = q_robot
        margin = camera.max_range + self.res
        self._ensure_bounds(rx - margin, rx + margin, ry - margin, ry + margin)
        self.mark_traversed(q_robot)

        angles, distances, hits = camera.scan_fov(q_robot, obstacles, cell_res=self.res)
        step_size = self.res * 0.5

        for ang, d, hit in zip(angles, distances, hits):
            ca, sa = float(np.cos(ang)), float(np.sin(ang))
            d_free = max(0.0, float(d) - 0.005) if hit else float(d)
            n_s = max(2, int(np.ceil(d_free / step_size)) + 1)
            ts = np.linspace(0.0, d_free, n_s)
            xs = rx + ts * ca
            ys = ry + ts * sa

            cols = np.clip(((xs - self.x_min) / self.res).astype(int), 0, self.width - 1)
            rows = np.clip(((ys - self.y_min) / self.res).astype(int), 0, self.height - 1)
            unk_mask = self.grid[rows, cols] == UNKNOWN
            self.grid[rows[unk_mask], cols[unk_mask]] = FREE

            # Mark hit obstacle surface as OCCUPIED unless robot physically traversed the cell
            if hit:
                oc, orow = self.world_to_grid(rx + (float(d) + 0.005) * ca, ry + (float(d) + 0.005) * sa)
                if not self.traversed[orow, oc]:
                    self.grid[orow, oc] = OCCUPIED

    def _get_safe_free_mask(self, start_xy: tuple[float, float] | None = None) -> np.ndarray:
        """Return boolean mask of FREE cells outside obstacle inflation radius."""
        occ = self.grid == OCCUPIED
        inflated = occ.copy()
        for dc, dr in self._dilate_offsets:
            r_src_s, r_src_e = max(0, -dr), self.height - max(0, dr)
            c_src_s, c_src_e = max(0, -dc), self.width - max(0, dc)
            r_dst_s, r_dst_e = max(0, dr), self.height - max(0, -dr)
            c_dst_s, c_dst_e = max(0, dc), self.width - max(0, -dc)
            inflated[r_dst_s:r_dst_e, c_dst_s:c_dst_e] |= occ[r_src_s:r_src_e, c_src_s:c_src_e]

        safe = (self.grid == FREE) & (~inflated)

        # Keep robot's immediate cell / local traversed escape patch passable if grazing inflation
        if start_xy is not None and self.width > 0:
            sc, sr = self.world_to_grid(*start_xy)
            if not safe[sr, sc]:
                safe[sr, sc] = True
                r_lo, r_hi = max(0, sr - 2), min(self.height, sr + 3)
                c_lo, c_hi = max(0, sc - 2), min(self.width, sc + 3)
                local_trav = self.traversed[r_lo:r_hi, c_lo:c_hi]
                if np.any(local_trav):
                    safe[r_lo:r_hi, c_lo:c_hi] |= local_trav
                else:
                    safe[r_lo:r_hi, c_lo:c_hi] |= (self.grid[r_lo:r_hi, c_lo:c_hi] == FREE)

        return safe

    def _bfs_reachable(self, safe: np.ndarray, start_c: int, start_r: int) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass 8-connected BFS (without diagonal corner cutting) on safe FREE cells."""
        visited = np.zeros_like(safe, dtype=bool)
        dist_grid = np.full(safe.shape, np.inf, dtype=float)
        if not safe[start_r, start_c]:
            return visited, dist_grid

        visited[start_r, start_c] = True
        dist_grid[start_r, start_c] = 0.0
        queue = deque([(start_c, start_r)])
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
        ]

        while queue:
            c, r = queue.popleft()
            d_cur = dist_grid[r, c]
            for dc, dr, cost in moves:
                nc, nr = c + dc, r + dr
                if not (0 <= nc < self.width and 0 <= nr < self.height):
                    continue
                if visited[nr, nc] or not safe[nr, nc]:
                    continue
                if dc != 0 and dr != 0 and not (safe[r, nc] and safe[nr, c]):
                    continue
                visited[nr, nc] = True
                dist_grid[nr, nc] = d_cur + cost
                queue.append((nc, nr))

        return visited, dist_grid

    def select_exploration_goal(self, q_robot: np.ndarray) -> tuple[float, float] | None:
        """Select the best reachable frontier cell (safe FREE cell adjacent to UNKNOWN)."""
        if self.width == 0:
            return None
        sx, sy, stheta = float(q_robot[0]), float(q_robot[1]), float(q_robot[2])
        safe = self._get_safe_free_mask((sx, sy))
        sc, sr = self.world_to_grid(sx, sy)
        visited, dist_grid = self._bfs_reachable(safe, sc, sr)

        unk = self.grid == UNKNOWN
        has_unk_nb = np.zeros_like(unk)
        has_unk_nb[:-1, :] |= unk[1:, :]
        has_unk_nb[1:, :] |= unk[:-1, :]
        has_unk_nb[:, :-1] |= unk[:, 1:]
        has_unk_nb[:, 1:] |= unk[:, :-1]

        fr_rows, fr_cols = np.where(safe & visited & has_unk_nb)
        if len(fr_rows) == 0:
            return None

        # Precompute 5x5 UNKNOWN counts for information-gain scoring
        unk_int = unk.astype(np.int16)
        unk_5x5 = np.zeros_like(unk_int)
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r_s, r_e = max(0, -dr), self.height - max(0, dr)
                c_s, c_e = max(0, -dc), self.width - max(0, dc)
                rd_s, rd_e = max(0, dr), self.height - max(0, -dr)
                cd_s, cd_e = max(0, dc), self.width - max(0, -dc)
                unk_5x5[rd_s:rd_e, cd_s:cd_e] += unk_int[r_s:r_e, c_s:c_e]

        dists_m = dist_grid[fr_rows, fr_cols] * self.res
        valid_far = dists_m >= 0.30
        if np.any(valid_far):
            fr_rows, fr_cols, dists_m = fr_rows[valid_far], fr_cols[valid_far], dists_m[valid_far]
        else:
            valid_min = dists_m >= 0.10
            if np.any(valid_min):
                fr_rows, fr_cols, dists_m = fr_rows[valid_min], fr_cols[valid_min], dists_m[valid_min]

        best_score = float("inf")
        best_goal = None
        for r, c, path_dist in zip(fr_rows, fr_cols, dists_m):
            fx, fy = self.grid_to_world(int(c), int(r))
            turn_pen = abs(wrap(float(np.arctan2(fy - sy, fx - sx)) - stheta))
            gain = float(unk_5x5[r, c]) / 25.0
            score = float(path_dist) + 0.12 * turn_pen - 0.35 * gain
            if score < best_score:
                best_score = score
                best_goal = (fx, fy)

        return best_goal

    def is_frontier_active(self, goal_xy: tuple[float, float], radius: float = 0.22) -> bool:
        """Check whether UNKNOWN cells still exist near goal_xy and goal remains safe."""
        if self.width == 0:
            return False
        gc, gr = self.world_to_grid(*goal_xy)
        safe = self._get_safe_free_mask(goal_xy)
        if not safe[gr, gc]:
            return False
        rad = int(np.ceil(radius / self.res))
        r_lo, r_hi = max(0, gr - rad), min(self.height, gr + rad + 1)
        c_lo, c_hi = max(0, gc - rad), min(self.width, gc + rad + 1)
        return bool(np.any(self.grid[r_lo:r_hi, c_lo:c_hi] == UNKNOWN))

    def select_fallback_goal(self, q_robot: np.ndarray) -> tuple[float, float] | None:
        """Fallback: pick farthest reachable safe FREE cell if all frontiers are exhausted."""
        if self.width == 0:
            return None
        sx, sy = float(q_robot[0]), float(q_robot[1])
        safe = self._get_safe_free_mask((sx, sy))
        sc, sr = self.world_to_grid(sx, sy)
        visited, dist_grid = self._bfs_reachable(safe, sc, sr)
        rows, cols = np.where(visited & safe)
        if len(rows) <= 1:
            return None
        idx = int(np.argmax(dist_grid[rows, cols]))
        return self.grid_to_world(int(cols[idx]), int(rows[idx]))

    def plan_path(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        """Plan pruned A* path strictly through mapped safe FREE cells."""
        if self.width == 0:
            return None
        safe = self._get_safe_free_mask(start_xy)
        view = _PlanningView(self, safe)
        raw = astar_search(view, start_xy, goal_xy)
        if raw is None:
            return None
        return prune_waypoints(raw, view)

    def plan_path_to_target(
        self,
        start_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        """Plan path to visible target, falling back to nearest reachable safe cell if needed."""
        if self.width == 0:
            return [start_xy, target_xy]
        safe = self._get_safe_free_mask(start_xy)
        view = _PlanningView(self, safe)

        if view.line_of_sight(start_xy, target_xy):
            return [start_xy, target_xy]

        tc, tr = self.world_to_grid(*target_xy)
        if safe[tr, tc]:
            raw = astar_search(view, start_xy, target_xy)
            if raw is not None:
                return prune_waypoints(raw, view)

        # Target sits near an obstacle/UNKNOWN edge: route to closest reachable safe cell
        sc, sr = self.world_to_grid(*start_xy)
        visited, _ = self._bfs_reachable(safe, sc, sr)
        rows, cols = np.where(visited & safe)
        if len(rows) == 0:
            return [start_xy, target_xy]

        xs = self.x_min + (cols + 0.5) * self.res
        ys = self.y_min + (rows + 0.5) * self.res
        idx = int(np.argmin((xs - target_xy[0]) ** 2 + (ys - target_xy[1]) ** 2))
        approach_pt = (float(xs[idx]), float(ys[idx]))
        raw = astar_search(view, start_xy, approach_pt)
        if raw is None:
            return [start_xy, target_xy]
        pruned = prune_waypoints(raw, view)
        if np.hypot(pruned[-1][0] - target_xy[0], pruned[-1][1] - target_xy[1]) > 1e-3:
            pruned.append(target_xy)
        return pruned

    def is_path_safe(self, waypoints: list[tuple[float, float]], current_xy: tuple[float, float]) -> bool:
        """Check whether remaining waypoints still have safe line-of-sight on current grid."""
        if not waypoints or self.width == 0:
            return False
        safe = self._get_safe_free_mask(current_xy)
        view = _PlanningView(self, safe)
        pts = [current_xy] + list(waypoints)
        return all(view.line_of_sight(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def get_local_occupied_points(self, p: np.ndarray, max_dist: float) -> list[np.ndarray]:
        """Return at most the closest OCCUPIED cell center per quadrant around p within max_dist."""
        if self.width == 0:
            return []
        px, py = float(p[0]), float(p[1])
        c_min, r_min = self.world_to_grid(px - max_dist, py - max_dist)
        c_max, r_max = self.world_to_grid(px + max_dist, py + max_dist)

        occ_r, occ_c = np.where(self.grid[r_min:r_max + 1, c_min:c_max + 1] == OCCUPIED)
        if len(occ_r) == 0:
            return []

        best_pts: dict[tuple[bool, bool], tuple[float, np.ndarray]] = {}
        max_d2 = max_dist * max_dist
        for dr, dc in zip(occ_r, occ_c):
            wx, wy = self.grid_to_world(c_min + int(dc), r_min + int(dr))
            dx, dy = wx - px, wy - py
            d2 = dx * dx + dy * dy
            if d2 <= max_d2:
                quad = (dx >= 0.0, dy >= 0.0)
                if quad not in best_pts or d2 < best_pts[quad][0]:
                    best_pts[quad] = (d2, np.array([wx, wy], dtype=float))

        return [item[1] for item in best_pts.values()]

    def get_mapped_areas(self) -> dict[str, float]:
        """Return mapped areas [m^2] independent of any predefined arena bounds."""
        cell_area = self.res * self.res
        return {
            "free_m2": float(np.sum(self.grid == FREE)) * cell_area,
            "occupied_m2": float(np.sum(self.grid == OCCUPIED)) * cell_area,
            "traversed_m2": float(np.sum(self.traversed)) * cell_area,
        }
