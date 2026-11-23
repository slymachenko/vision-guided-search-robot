"""
Virtual camera sensor: Field of View (FOV) geometry, target detection and exact analytical line-of-sight occlusion checking.
"""

import numpy as np
from src.control.closed_loop import closest_surface_info
from src.control.open_loop import wrap


class VirtualCamera:
    """Simulates a directional monocular camera on the robot."""

    def __init__(
        self,
        fov_angle: float = np.deg2rad(70.0),   # Total horizontal field of view (70 deg = +/- 35 deg)
        max_range: float = 0.85,               # Maximum visibility distance [m]
    ):
        self.fov_angle = fov_angle
        self.max_range = max_range

    def _segment_intersects_box(self, p1: np.ndarray, p2: np.ndarray, obs: dict) -> bool:
        """
        Exact analytical 2D segment-box intersection (Slab method).
        Detects even razor-thin 1 cm walls without any discretization loops.
        """
        center = np.array(obs["center"][:2], dtype=float)
        hx = obs["size"][0] / 2.0
        hy = obs["size"][1] / 2.0
        yaw = float(obs.get("yaw", 0.0))

        c, s = np.cos(yaw), np.sin(yaw)
        rot_t = np.array([[c, s], [-s, c]])

        # Transform segment endpoints into box local coordinates
        a = rot_t @ (p1 - center)
        b = rot_t @ (p2 - center)
        d = b - a

        t_min = 0.02  # Ignore ray start point (robot body)
        t_max = 0.98  # Ignore ray arrival point (obstacle surface)

        for i, h in enumerate([hx, hy]):
            if abs(d[i]) < 1e-7:
                if abs(a[i]) > h:
                    return False
            else:
                t1 = (-h - a[i]) / d[i]
                t2 = (h - a[i]) / d[i]
                t_enter = min(t1, t2)
                t_exit = max(t1, t2)
                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
                if t_min >= t_max:
                    return False

        return t_min < t_max

    def _segment_intersects_cylinder(self, p1: np.ndarray, p2: np.ndarray, obs: dict) -> bool:
        """Exact analytical 2D segment-cylinder intersection."""
        center = np.array(obs["center"][:2], dtype=float)
        radius = float(obs["radius"])

        v = p2 - p1
        l2 = float(np.dot(v, v))
        if l2 < 1e-8:
            return float(np.linalg.norm(p1 - center)) < radius

        # Projection of cylinder center onto segment
        t = np.clip(np.dot(center - p1, v) / l2, 0.02, 0.98)
        closest_pt = p1 + t * v
        return float(np.linalg.norm(closest_pt - center)) < (radius - 1e-4)

    def has_line_of_sight(
        self,
        p_from: np.ndarray,
        p_to: np.ndarray,
        obstacles: list[dict],
        ignore_obs: dict | None = None,
    ) -> bool:
        """
        Check if ray from p_from to p_to is completely unobstructed.
        Uses exact analytical geometry (instant and perfectly accurate for thin walls).
        """
        p_from = np.asarray(p_from[:2], dtype=float)
        p_to = np.asarray(p_to[:2], dtype=float)

        for obs in obstacles:
            if obs is ignore_obs:
                continue

            if obs["type"] == "box":
                if self._segment_intersects_box(p_from, p_to, obs):
                    return False
            elif obs["type"] == "cylinder":
                if self._segment_intersects_cylinder(p_from, p_to, obs):
                    return False

        return True

    def is_target_visible(
        self,
        q_robot: np.ndarray,
        target_xy: tuple[float, float] | np.ndarray,
        obstacles: list[dict],
    ) -> bool:
        """Check if target is inside FOV cone and has unobstructed line-of-sight."""
        rx, ry, r_theta = q_robot
        tx, ty = target_xy[0], target_xy[1]

        dx = tx - rx
        dy = ty - ry
        dist = float(np.hypot(dx, dy))

        # 1. Range check
        if dist > self.max_range or dist < 1e-4:
            return False

        # 2. Angle check (relative to robot heading)
        angle_to_target = float(np.arctan2(dy, dx))
        rel_angle = wrap(angle_to_target - r_theta)
        if abs(rel_angle) > (self.fov_angle / 2.0):
            return False

        # 3. Line-of-sight check
        return self.has_line_of_sight(np.array([rx, ry]), np.array([tx, ty]), obstacles)

    def is_obstacle_visible(self, q_robot: np.ndarray, obs: dict, all_obstacles: list[dict]) -> bool:
        """
        Check if an obstacle is visible within the camera FOV cone.
        Evaluates closest surface point, center, and corners with analytical line-of-sight.
        """
        rx, ry, r_theta = q_robot
        p = np.array([rx, ry], dtype=float)
        half_fov = self.fov_angle / 2.0

        # Candidate points to check for this obstacle
        test_points = []

        # 1. Closest surface point to the robot
        d, n_hat = closest_surface_info(p, obs, r_robot=0.0)
        p_surf = p - d * n_hat
        test_points.append(p_surf)

        # 2. Obstacle center
        center = np.array(obs["center"][:2], dtype=float)
        test_points.append(center)

        # 3. Box corners (if box/wall)
        if obs["type"] == "box":
            hx, hy = obs["size"][0] / 2.0, obs["size"][1] / 2.0
            yaw = float(obs.get("yaw", 0.0))
            c, s = np.cos(yaw), np.sin(yaw)
            rot = np.array([[c, -s], [s, c]])
            for dx, dy in [(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)]:
                test_points.append(center + rot @ np.array([dx, dy]))

        # Test each point: must be inside FOV AND not occluded by another obstacle
        for pt in test_points:
            dx = pt[0] - rx
            dy = pt[1] - ry
            dist = float(np.hypot(dx, dy))

            if dist <= self.max_range and dist > 1e-4:
                angle = float(np.arctan2(dy, dx))
                rel_angle = wrap(angle - r_theta)
                if abs(rel_angle) <= half_fov:
                    if self.has_line_of_sight(p, pt, all_obstacles, ignore_obs=obs):
                        return True

        return False

    def get_fov_local_points(self, n_arc_points: int = 9) -> list[tuple[float, float]]:
        """
        Generate 2D local boundary points of the circular sector on the floor:
        - origin (robot nose)
        - left edge
        - circular arc
        - right edge
        """
        half_fov = self.fov_angle / 2.0
        # Start at robot nose (x=0.05 m in front of center)
        pts = [(0.05, 0.0)]

        # Points along the left ray
        for r in [self.max_range * 0.5, self.max_range]:
            pts.append((float(0.05 + r * np.cos(half_fov)), float(r * np.sin(half_fov))))

        # Points along the circular arc
        arc_angles = np.linspace(half_fov, -half_fov, n_arc_points)
        for a in arc_angles:
            pts.append((float(0.05 + self.max_range * np.cos(a)), float(self.max_range * np.sin(a))))

        # Point along the right ray
        pts.append((float(0.05 + self.max_range * 0.5 * np.cos(-half_fov)), float(self.max_range * 0.5 * np.sin(-half_fov))))

        return pts
