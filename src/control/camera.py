"""
Virtual camera sensor: Field of View (FOV) geometry, target detection and line-of-sight occlusion checking.
"""

import numpy as np
from src.control.open_loop import wrap
from src.env.arena import dist_point_to_obstacle


class VirtualCamera:
    """Simulates a directional monocular camera on the robot."""

    def __init__(
        self,
        fov_angle: float = np.deg2rad(70.0),   # Total horizontal field of view (70 deg = +/- 35 deg)
        max_range: float = 0.80,               # Maximum visibility distance [m]
    ):
        self.fov_angle = fov_angle
        self.max_range = max_range

    def is_target_visible(
        self,
        q_robot: np.ndarray,
        target_xy: tuple[float, float] | np.ndarray,
        obstacles: list[dict],
    ) -> bool:
        """
        Check if the target is within the camera FOV cone and not occluded by obstacles.
        """
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

        # 3. Line-of-sight check (occlusion by walls or obstacles)
        # Check intermediate points between robot and target
        n_checks = max(3, int(np.ceil(dist / 0.05)))
        check_x = np.linspace(rx, tx, n_checks)[1:-1]
        check_y = np.linspace(ry, ty, n_checks)[1:-1]

        for cx, cy in zip(check_x, check_y):
            pt = np.array([cx, cy])
            for obs in obstacles:
                # If a ray point penetrates an obstacle, the target is hidden
                if dist_point_to_obstacle(pt, obs) <= 0.0:
                    return False

        return True

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
