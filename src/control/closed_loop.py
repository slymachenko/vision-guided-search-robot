"""
Closed-loop reactive controller using Tangential Artificial Potential Fields (APF).
Generates unicycle velocity commands (v, omega) based on attractive and repulsive forces.
"""

import numpy as np
from src.control.open_loop import wrap
from src.env.arena import ROBOT_RADIUS


def closest_surface_info(p: np.ndarray, obs: dict, r_robot: float = ROBOT_RADIUS) -> tuple[float,
np.ndarray]:
    """
    Compute surface clearance distance d and outward normal n_hat
    from 2D robot center p to an obstacle (cylinder or box).
    """
    center = np.array(obs["center"][:2], dtype=float)

    if obs["type"] == "cylinder":
        vec = p - center
        dist_c = float(np.linalg.norm(vec))
        if dist_c < 1e-6:
            return 1e-3, np.array([1.0, 0.0])
        n_hat = vec / dist_c
        d = max(dist_c - (obs["radius"] + r_robot), 1e-3)
        return d, n_hat

    if obs["type"] == "box":
        hx, hy = obs["size"][0] / 2.0, obs["size"][1] / 2.0
        yaw = float(obs.get("yaw", 0.0))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])

        p_loc = rot.T @ (p - center)
        q_loc = np.array([np.clip(p_loc[0], -hx, hx), np.clip(p_loc[1], -hy, hy)])
        diff = p_loc - q_loc
        dist_s = float(np.linalg.norm(diff))

        if dist_s > 1e-6:
            n_hat = rot @ (diff / dist_s)
            d = max(dist_s - r_robot, 1e-3)
            return d, n_hat

        # Fallback if center penetrated the box
        return 1e-3, rot @ np.array([1.0, 0.0])

    raise ValueError(f"Unknown obstacle type: {obs['type']}")


def compute_apf_control(
    q: np.ndarray,
    target_xy: np.ndarray,
    obstacles: list[dict],
    r_robot: float = ROBOT_RADIUS + 0.008,
    stop_dist: float = 0.085,
    k_att: float = 1.0,
    k_rep: float = 0.0030,
    d0: float = 0.050,
    alpha_tan: float = 0.70,
    v_max: float = 0.30,
    w_max: float = 1.80,
    k_h: float = 3.0,
) -> tuple[float, float, float]:
    """
    Compute unicycle (v, omega) using a tangential APF.
    Stops smoothly when distance to target center reaches stop_dist.
    Returns (v, omega, dist_to_target_center).
    """
    x, y, theta = q
    p = np.array([x, y], dtype=float)
    target = np.asarray(target_xy[:2], dtype=float)

    diff_target = target - p
    dist_target = float(np.linalg.norm(diff_target))

    # Stop condition: already at safe stopping distance
    if dist_target <= stop_dist:
        return 0.0, 0.0, dist_target

    u_target = diff_target / dist_target
    f_att = k_att * min(1.0, dist_target) * u_target

    # Repulsive + tangential forces from obstacles
    f_rep = np.zeros(2, dtype=float)
    for obs in obstacles:
        d, n_hat = closest_surface_info(p, obs, r_robot)
        if d < d0:
            # Tangential vector oriented towards the target to slide past obstacle
            t_hat = np.array([-n_hat[1], n_hat[0]])
            if np.dot(t_hat, diff_target) < 0:
                t_hat = -t_hat

            raw_mag = k_rep * (1.0 / d - 1.0 / d0) / (d ** 2)
            mag = min(raw_mag, 3.0 * k_att)
            f_rep += mag * (n_hat + alpha_tan * t_hat)

    # Convert total force vector to unicycle commands
    f_tot = f_att + f_rep
    theta_des = float(np.arctan2(f_tot[1], f_tot[0]))
    heading_err = wrap(theta_des - theta)

    w = float(np.clip(k_h * heading_err, -w_max, w_max))
    v = float(v_max * min(1.0, np.linalg.norm(f_tot)) * max(0.0, np.cos(heading_err)))

    return v, w, dist_target


class ReactiveExplorer:
    """
    Local reactive explorer with corridor-aware clearance check:
    - Drives through narrow passages without false stops.
    - Only triggers scanning when an obstacle directly blocks the frontal path.
    - Turns towards the side with greater clearance.
    """

    def __init__(
        self,
        v_cruise: float = 0.24,
        w_scan: float = 1.30,
        min_front_clearance: float = 0.045,  # Stops only when front bumper is 4.5 cm from obstacle
        dt: float = 0.05,
    ):
        self.v_cruise = v_cruise
        self.w_scan = w_scan
        self.min_front_clearance = min_front_clearance
        self.dt = dt

        self.state = "DRIVING"     # "DRIVING" or "SCANNING"
        self.scan_steps_left = 0
        self.drive_steps = 0
        self.turn_dir = 1.0        # 1.0 = left, -1.0 = right

    def _check_frontal_block(
        self,
        p: np.ndarray,
        theta: np.ndarray,
        known_obstacles: list[dict],
    ) -> tuple[bool, float, float]:
        """
        Check if an obstacle directly blocks the robot's front corridor (width = 12 cm).
        Returns: (is_blocked, clearance_left, clearance_right).
        """
        c, s = np.cos(theta), np.sin(theta)
        u_fwd = np.array([c, s])      # Forward direction
        u_lat = np.array([-s, c])     # Lateral direction (left)

        robot_half_length = 0.05
        robot_half_width = 0.04
        corridor_half_width = robot_half_width + 0.02  # 6 cm corridor width from center line

        is_blocked = False
        min_clear_left = 1.0
        min_clear_right = 1.0

        for obs in known_obstacles:
            d, n_hat = closest_surface_info(p, obs, r_robot=ROBOT_RADIUS)
            # Vector pointing from obstacle surface towards robot
            rel_vec = -n_hat * (d + ROBOT_RADIUS)
            fwd_dist = float(np.dot(rel_vec, u_fwd))
            lat_dist = float(np.dot(rel_vec, u_lat))

            # 1. Check if obstacle is directly in front of the robot's path
            if fwd_dist > 0.0 and abs(lat_dist) <= corridor_half_width:
                clearance_ahead = fwd_dist - robot_half_length
                if clearance_ahead <= self.min_front_clearance:
                    is_blocked = True

            # 2. Measure lateral clearances to decide turn direction
            if lat_dist > 0.0:
                min_clear_left = min(min_clear_left, d)
            else:
                min_clear_right = min(min_clear_right, d)

        return is_blocked, min_clear_left, min_clear_right

    def compute_step(
        self,
        q: np.ndarray,
        known_obstacles: list[dict],
        r_robot: float = ROBOT_RADIUS,
    ) -> tuple[float, float]:
        """Compute (v, omega) for the current exploration step."""
        x, y, theta = q
        p = np.array([x, y], dtype=float)

        is_blocked, clear_left, clear_right = self._check_frontal_block(p, theta, known_obstacles)

        # State transition logic
        if self.state == "DRIVING":
            self.drive_steps += 1
            # Trigger scan if path directly ahead is blocked, or periodic scan every ~4 seconds in open space
            if is_blocked or self.drive_steps > int(4.0 / self.dt):
                self.state = "SCANNING"
                # Scan duration: ~0.8s (about 60 deg rotation)
                self.scan_steps_left = int(0.8 / self.dt)
                # Turn towards the more open side
                self.turn_dir = 1.0 if clear_left >= clear_right else -1.0
                self.drive_steps = 0

        elif self.state == "SCANNING":
            self.scan_steps_left -= 1
            # Once scan is done and front is not blocked, resume driving immediately
            if self.scan_steps_left <= 0:
                if not is_blocked:
                    self.state = "DRIVING"
                else:
                    # Keep rotating if still facing an obstacle directly ahead
                    self.scan_steps_left = int(0.3 / self.dt)

        # Output velocity commands
        if self.state == "SCANNING":
            return 0.0, self.turn_dir * self.w_scan

        # In DRIVING state: drive forward with smooth APF repulsive side-forces to stay centered
        u_fwd = np.array([np.cos(theta), np.sin(theta)])
        f_rep = np.zeros(2, dtype=float)
        d0_side = 0.05  # Only gently push away if closer than 5 cm from surface

        for obs in known_obstacles:
            d, n_hat = closest_surface_info(p, obs, r_robot)
            if d < d0_side:
                mag = min(0.003 * (1.0 / d - 1.0 / d0_side) / (d ** 2), 2.0)
                t_hat = np.array([-n_hat[1], n_hat[0]])
                if np.dot(t_hat, u_fwd) < 0:
                    t_hat = -t_hat
                f_rep += mag * (n_hat + 0.5 * t_hat)

        f_tot = u_fwd + f_rep
        des_theta = float(np.arctan2(f_tot[1], f_tot[0]))
        heading_err = wrap(des_theta - theta)

        w = float(np.clip(2.5 * heading_err, -1.8, 1.8))
        v = float(self.v_cruise * max(0.0, np.cos(heading_err)))

        return v, w
