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
    Grid-guided frontier explorer and target approach controller for Scenario 3.
    - Explores UNKNOWN frontiers on ExplorationGrid using A* over safe FREE cells.
    - Rotates in place when heading error exceeds half-FOV so the robot always sees ahead.
    - Applies local tangential APF repulsion from mapped OCCUPIED cells.
    """

    def __init__(
        self,
        v_cruise: float = 0.24,
        w_max: float = 1.80,
        fov_angle: float = np.deg2rad(70.0),
        dt: float = 0.05,
    ):
        self.v_cruise = float(v_cruise)
        self.w_max = float(w_max)
        self.half_fov = float(fov_angle) / 2.0
        self.dt = float(dt)

        self.mode = "EXPLORING"
        self.goal_xy: tuple[float, float] | None = None
        self.waypoints: list[tuple[float, float]] = []
        self.wp_idx = 0
        self.goal_steps = 0
        self.scan_steps_left = 0
        self.stall_steps = 0
        self.last_progress_pos: np.ndarray | None = None
        self._did_full_scan = False

    def _drive_to_point(
        self,
        q: np.ndarray,
        goal_xy: tuple[float, float] | np.ndarray,
        grid,
        stop_dist: float = 0.0,
    ) -> tuple[float, float]:
        """Compute smooth FOV-aware unicycle (v, w) with local APF avoidance from OCCUPIED cells."""
        x, y, theta = float(q[0]), float(q[1]), float(q[2])
        p = np.array([x, y], dtype=float)
        diff = np.asarray(goal_xy[:2], dtype=float) - p
        dist = float(np.linalg.norm(diff))

        if (stop_dist > 0.0 and dist <= stop_dist) or dist < 1e-4:
            return 0.0, 0.0

        f_att = (diff / dist) * (min(1.0, dist / 0.20) if stop_dist > 0.0 else 1.0)

        # Repulsive + tangential forces from nearest mapped OCCUPIED cells per quadrant
        f_rep = np.zeros(2, dtype=float)
        d0 = 0.055
        k_rep = 0.0028
        eff_robot_r = ROBOT_RADIUS + 0.5 * grid.res
        for occ_pt in grid.get_local_occupied_points(p, eff_robot_r + d0):
            vec = p - occ_pt
            norm_v = float(np.linalg.norm(vec))
            d = max(norm_v - eff_robot_r, 1e-3)
            if d < d0 and norm_v > 1e-6:
                n_hat = vec / norm_v
                t_hat = np.array([-n_hat[1], n_hat[0]])
                if np.dot(t_hat, diff) < 0.0:
                    t_hat = -t_hat
                mag = min(k_rep * (1.0 / d - 1.0 / d0) / (d ** 2), 2.5)
                f_rep += mag * (n_hat + 0.65 * t_hat)

        f_tot = f_att + f_rep
        des_theta = float(np.arctan2(f_tot[1], f_tot[0]))
        heading_err = wrap(des_theta - theta)

        w = float(np.clip(2.8 * heading_err, -self.w_max, self.w_max))
        # Smoothly drop forward velocity to zero at half_fov so robot never drives blind outside FOV
        if abs(heading_err) >= self.half_fov:
            return 0.0, w

        speed_scale = min(1.0, dist / 0.15) if stop_dist > 0.0 else 1.0
        turn_scale = float(np.cos((0.5 * np.pi) * (abs(heading_err) / self.half_fov)))
        v = float(self.v_cruise * speed_scale * turn_scale)
        return v, w

    def _plan_exploration(self, q: np.ndarray, grid) -> None:
        """Select a reachable frontier (or fallback cell) and plan waypoints on FREE cells."""
        p = (float(q[0]), float(q[1]))
        new_goal = grid.select_exploration_goal(q)
        if new_goal is not None:
            path = grid.plan_path(p, new_goal)
            if path:
                self.goal_xy = new_goal
                self.waypoints = path
                self.wp_idx = 1 if len(path) > 1 else 0
                self.goal_steps = 0
                self._did_full_scan = False
                return

        if not self._did_full_scan:
            self.goal_xy = None
            self.waypoints = []
            self.scan_steps_left = int(np.ceil((2.0 * np.pi) / (self.w_max * 0.85) / self.dt))
            self._did_full_scan = True
            return

        fallback = grid.select_fallback_goal(q)
        if fallback is not None:
            path = grid.plan_path(p, fallback)
            if path:
                self.goal_xy = fallback
                self.waypoints = path
                self.wp_idx = 1 if len(path) > 1 else 0
                self.goal_steps = 0
                self._did_full_scan = False
                return

        self.scan_steps_left = int(0.8 / self.dt)

    def compute_step(
        self,
        q: np.ndarray,
        grid,
        target_xy: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Compute (v, omega) for exploration or target approach using ExplorationGrid."""
        p = np.array([q[0], q[1]], dtype=float)
        p_tuple = (float(p[0]), float(p[1]))

        if self.last_progress_pos is None:
            self.last_progress_pos = p.copy()

        # Phase 2: Target approaching via A* on ExplorationGrid + local APF
        if target_xy is not None:
            target_tuple = (float(target_xy[0]), float(target_xy[1]))
            need_plan = (
                self.mode != "APPROACHING"
                or not self.waypoints
                or self.wp_idx >= len(self.waypoints)
                or not grid.is_path_safe(self.waypoints[self.wp_idx:], p_tuple)
            )
            if need_plan:
                self.mode = "APPROACHING"
                self.scan_steps_left = 0
                self.waypoints = grid.plan_path_to_target(p_tuple, target_tuple) or [p_tuple, target_tuple]
                self.wp_idx = 1 if len(self.waypoints) > 1 else 0

            while self.wp_idx < len(self.waypoints) - 1:
                if np.hypot(self.waypoints[self.wp_idx][0] - p[0], self.waypoints[self.wp_idx][1] - p[1]) <= 0.085:
                    self.wp_idx += 1
                else:
                    break

            is_final = self.wp_idx == len(self.waypoints) - 1
            return self._drive_to_point(q, self.waypoints[self.wp_idx], grid, stop_dist=0.085 if is_final else 0.0)

        # Phase 1: Frontier-based exploration
        self.mode = "EXPLORING"
        if self.scan_steps_left > 0:
            self.scan_steps_left -= 1
            self.last_progress_pos = p.copy()
            self.stall_steps = 0
            return 0.0, self.w_max * 0.85

        if np.linalg.norm(p - self.last_progress_pos) > 0.015:
            self.last_progress_pos = p.copy()
            self.stall_steps = 0
        else:
            self.stall_steps += 1

        if self.stall_steps > 25:
            self.stall_steps = 0
            self.goal_xy = None
            self.waypoints = []
            self.scan_steps_left = int(0.8 / self.dt)
            return 0.0, self.w_max * 0.85

        need_replan = (
            self.goal_xy is None
            or not self.waypoints
            or self.wp_idx >= len(self.waypoints)
            or np.hypot(self.goal_xy[0] - p[0], self.goal_xy[1] - p[1]) <= 0.10
            or (not self._did_full_scan and not grid.is_frontier_active(self.goal_xy))
            or not grid.is_path_safe(self.waypoints[self.wp_idx:], p_tuple)
            or self.goal_steps > 100
        )
        if need_replan:
            self.stall_steps = 0
            self._plan_exploration(q, grid)
            if self.scan_steps_left > 0 or not self.waypoints:
                return 0.0, self.w_max * 0.85

        while self.wp_idx < len(self.waypoints) - 1:
            if np.hypot(self.waypoints[self.wp_idx][0] - p[0], self.waypoints[self.wp_idx][1] - p[1]) <= 0.09:
                self.wp_idx += 1
            else:
                break

        self.goal_steps += 1
        v, w = self._drive_to_point(q, self.waypoints[self.wp_idx], grid, stop_dist=0.0)
        # Only track stall while actively trying to move forward (not during in-place turns)
        if v <= 0.04:
            self.stall_steps = 0
            self.last_progress_pos = p.copy()
        return v, w
