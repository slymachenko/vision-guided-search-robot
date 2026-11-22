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
