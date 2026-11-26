"""
Path planning module: 2D Occupancy Grid, A* search and Line-of-Sight waypoint pruning.
"""

import heapq
import numpy as np
from visero.env.arena import ROBOT_RADIUS, dist_point_to_obstacle


class OccupancyGrid:
    """Discretizes the 2D continuous arena into an inflated occupancy grid."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (0.0, 2.0, 0.0, 2.0),
        resolution: float = 0.025,
        inflation_radius: float = ROBOT_RADIUS + 0.02,
    ):
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.res = resolution
        self.inflation_radius = inflation_radius

        self.width = int(np.ceil((self.x_max - self.x_min) / self.res))
        self.height = int(np.ceil((self.y_max - self.y_min) / self.res))
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def populate(self, obstacles: list[dict]) -> None:
        """Mark cells occupied if their distance to any obstacle is <= inflation_radius."""
        x_coords = self.x_min + (np.arange(self.width) + 0.5) * self.res
        y_coords = self.y_min + (np.arange(self.height) + 0.5) * self.res
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)

        occupied = np.zeros(len(points), dtype=bool)

        for obs in obstacles:
            if obs.get("kind") == "wall":
                cx, cy = obs["center"][0], obs["center"][1]
                hx = obs["size"][0] / 2.0 + self.inflation_radius
                hy = obs["size"][1] / 2.0 + self.inflation_radius
                in_wall = (
                    (points[:, 0] >= cx - hx)
                    & (points[:, 0] <= cx + hx)
                    & (points[:, 1] >= cy - hy)
                    & (points[:, 1] <= cy + hy)
                )
                occupied |= in_wall
            elif obs["type"] == "cylinder":
                cx, cy = obs["center"][0], obs["center"][1]
                thresh = obs["radius"] + self.inflation_radius
                dists_sq = (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2
                occupied |= dists_sq <= thresh**2
            elif obs["type"] == "box":
                # Check distances for box obstacles
                for idx, pt in enumerate(points):
                    if (
                        not occupied[idx]
                        and dist_point_to_obstacle(pt, obs) <= self.inflation_radius
                    ):
                        occupied[idx] = True

        self.grid = occupied.reshape((self.height, self.width)).astype(np.uint8)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.clip((x - self.x_min) / self.res, 0, self.width - 1))
        row = int(np.clip((y - self.y_min) / self.res, 0, self.height - 1))
        return col, row

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        x = self.x_min + (col + 0.5) * self.res
        y = self.y_min + (row + 0.5) * self.res
        return float(x), float(y)

    def is_free(self, col: int, row: int) -> bool:
        if 0 <= col < self.width and 0 <= row < self.height:
            return bool(self.grid[row, col] == 0)
        return False

    def line_of_sight(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Check if straight-line segment between p1 and p2 is collision-free."""
        dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if dist < 1e-4:
            return self.is_free(*self.world_to_grid(p1[0], p1[1]))

        n_samples = max(2, int(np.ceil(dist / (self.res * 0.5))))
        xs = np.linspace(p1[0], p2[0], n_samples)
        ys = np.linspace(p1[1], p2[1], n_samples)
        for x, y in zip(xs, ys):
            col, row = self.world_to_grid(x, y)
            if not self.is_free(col, row):
                return False
        return True


def astar_search(
    grid_map: OccupancyGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]] | None:
    """Find shortest path from start_xy to goal_xy using 8-connected A*."""
    start_c, start_r = grid_map.world_to_grid(start_xy[0], start_xy[1])
    goal_c, goal_r = grid_map.world_to_grid(goal_xy[0], goal_xy[1])

    if not grid_map.is_free(goal_c, goal_r):
        return None

    open_heap = []
    start_node = (start_c, start_r)
    goal_node = (goal_c, goal_r)

    h_start = float(np.hypot(start_c - goal_c, start_r - goal_r))
    heapq.heappush(open_heap, (h_start, 0.0, start_node))

    came_from = {}
    g_score = {start_node: 0.0}
    closed_set = set()

    moves = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)),
        (-1, -1, np.sqrt(2)),
    ]

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal_node:
            # Reconstruct path
            path_grid = [current]
            while current in came_from:
                current = came_from[current]
                path_grid.append(current)
            path_grid.reverse()

            path_world = [start_xy]
            for c, r in path_grid[1:-1]:
                path_world.append(grid_map.grid_to_world(c, r))
            path_world.append(goal_xy)
            return path_world

        c, r = current
        for dc, dr, cost in moves:
            neighbor = (c + dc, r + dr)
            if neighbor in closed_set or not grid_map.is_free(neighbor[0], neighbor[1]):
                continue

            # Prevent cutting through diagonal obstacles
            if dc != 0 and dr != 0:
                if not grid_map.is_free(c + dc, r) or not grid_map.is_free(c, r + dr):
                    continue

            tentative_g = current_g + cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = float(np.hypot(neighbor[0] - goal_c, neighbor[1] - goal_r))
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, neighbor))

    return None


def prune_waypoints(
    path: list[tuple[float, float]],
    grid_map: OccupancyGrid,
) -> list[tuple[float, float]]:
    """Prune dense A* path into sparse corner waypoints using Line-of-Sight."""
    if len(path) <= 2:
        return list(path)

    pruned = [path[0]]
    curr = 0
    while curr < len(path) - 1:
        next_idx = curr + 1
        for i in range(len(path) - 1, curr, -1):
            if grid_map.line_of_sight(path[curr], path[i]):
                next_idx = i
                break
        pruned.append(path[next_idx])
        curr = next_idx
    return pruned


def plan_path(
    q_start: np.ndarray,
    target_xy: tuple[float, float] | np.ndarray,
    obstacles: list[dict],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    High-level helper: returns (dense_astar_path, sparse_pruned_waypoints).
    """
    grid = OccupancyGrid()
    grid.populate(obstacles)

    start_pt = (float(q_start[0]), float(q_start[1]))
    goal_pt = (float(target_xy[0]), float(target_xy[1]))

    raw_path = astar_search(grid, start_pt, goal_pt)
    if raw_path is None:
        raise RuntimeError("A* could not find a path between start and target!")

    waypoints = prune_waypoints(raw_path, grid)
    return raw_path, waypoints
