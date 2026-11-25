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
        false_negative_prob: float = 0.0,      # Optional probability of missed target detection
        seed: int | None = None,
    ):
        self.fov_angle = float(fov_angle)
        self.max_range = float(max_range)
        self.false_negative_prob = float(false_negative_prob)
        self._rng = np.random.default_rng(seed)

    def _ray_obstacle_distance(
        self,
        origin: np.ndarray,
        unit_dir: np.ndarray,
        max_dist: float,
        obs: dict,
    ) -> float | None:
        """
        Exact analytical distance along ray (origin + t * unit_dir) to obstacle surface
        within [0, max_dist], or None if no hit.
        """
        center = np.asarray(obs["center"][:2], dtype=float)

        if obs["type"] == "box":
            hx = obs["size"][0] / 2.0
            hy = obs["size"][1] / 2.0
            yaw = float(obs.get("yaw", 0.0))
            c, s = np.cos(yaw), np.sin(yaw)
            rot_t = np.array([[c, s], [-s, c]])

            a = rot_t @ (origin - center)
            d = rot_t @ unit_dir
            t_min, t_max = 0.0, max_dist

            for i, h in enumerate((hx, hy)):
                if abs(d[i]) < 1e-9:
                    if abs(a[i]) > h:
                        return None
                else:
                    t1 = (-h - a[i]) / d[i]
                    t2 = (h - a[i]) / d[i]
                    if t1 > t2:
                        t1, t2 = t2, t1
                    t_min = max(t_min, t1)
                    t_max = min(t_max, t2)
                    if t_min > t_max:
                        return None

            return float(t_min) if 0.0 <= t_min <= max_dist else None

        if obs["type"] == "cylinder":
            radius = float(obs["radius"])
            oc = origin - center
            hb = float(np.dot(oc, unit_dir))
            c_val = float(np.dot(oc, oc)) - radius * radius
            disc = hb * hb - c_val
            if disc < 0.0:
                return None

            sqrt_d = float(np.sqrt(disc))
            t1 = -hb - sqrt_d
            t2 = -hb + sqrt_d
            if 0.0 <= t1 <= max_dist:
                return float(t1)
            if t1 < 0.0 <= t2:
                return 0.0
            return None

        raise ValueError(f"Unknown obstacle type: {obs['type']}")

    def has_line_of_sight(
        self,
        p_from: np.ndarray,
        p_to: np.ndarray,
        obstacles: list[dict],
        ignore_obs: dict | None = None,
    ) -> bool:
        """
        Check if ray from p_from to p_to is completely unobstructed.
        Uses exact analytical geometry (accurate even for razor-thin 1 cm walls).
        """
        p_from = np.asarray(p_from[:2], dtype=float)
        p_to = np.asarray(p_to[:2], dtype=float)
        diff = p_to - p_from
        dist = float(np.hypot(diff[0], diff[1]))
        if dist < 1e-6:
            return True

        unit_dir = diff / dist
        for obs in obstacles:
            if obs is ignore_obs:
                continue
            hit_d = self._ray_obstacle_distance(p_from, unit_dir, dist, obs)
            if hit_d is not None and hit_d < (dist - 1e-4):
                return False

        return True

    def scan_fov(
        self,
        q_robot: np.ndarray,
        obstacles: list[dict],
        n_rays: int | None = None,
        cell_res: float = 0.025,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cast analytical rays across the FOV sector from robot pose.
        Returns (angles, hit_distances, hit_flags) for mapping FREE and OCCUPIED cells.
        """
        rx, ry, r_theta = q_robot
        origin = np.array([rx, ry], dtype=float)
        if n_rays is None:
            n_rays = max(45, int(np.ceil(self.fov_angle * self.max_range / (cell_res * 0.8))) + 1)

        half_fov = 0.5 * self.fov_angle
        angles = np.linspace(r_theta - half_fov, r_theta + half_fov, n_rays)
        distances = np.full(n_rays, self.max_range, dtype=float)
        hits = np.zeros(n_rays, dtype=bool)

        for i, ang in enumerate(angles):
            u = np.array([np.cos(ang), np.sin(ang)], dtype=float)
            best = self.max_range
            hit = False
            for obs in obstacles:
                d = self._ray_obstacle_distance(origin, u, best, obs)
                if d is not None and d <= best:
                    best = d
                    hit = True
            distances[i] = best
            hits[i] = hit

        return angles, distances, hits

    def is_target_visible(
        self,
        q_robot: np.ndarray,
        target_xy: tuple[float, float] | np.ndarray,
        obstacles: list[dict],
    ) -> bool:
        """Check if target is inside FOV cone, unobstructed, and not dropped by false negatives."""
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

        # 3. Line-of-sight check + optional false-negative simulation
        if not self.has_line_of_sight(np.array([rx, ry]), np.array([tx, ty]), obstacles):
            return False

        if self.false_negative_prob > 0.0 and self._rng.random() < self.false_negative_prob:
            return False

        return True

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
        Generate 2D local boundary points of the circular sector on the floor
        centered at the robot pose (matching scan_fov and is_target_visible).
        """
        half_fov = self.fov_angle / 2.0
        pts = [(0.0, 0.0)]

        # Points along the left ray
        for r in [self.max_range * 0.5, self.max_range]:
            pts.append((float(r * np.cos(half_fov)), float(r * np.sin(half_fov))))

        # Points along the circular arc
        arc_angles = np.linspace(half_fov, -half_fov, n_arc_points)
        for a in arc_angles:
            pts.append((float(self.max_range * np.cos(a)), float(self.max_range * np.sin(a))))

        # Point along the right ray
        pts.append((float(self.max_range * 0.5 * np.cos(-half_fov)), float(self.max_range * 0.5 * np.sin(-half_fov))))

        return pts
