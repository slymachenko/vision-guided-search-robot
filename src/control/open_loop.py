"""
Open-loop turn-drive-turn planner and actuation model.
"""

import numpy as np


def wrap(angle: float) -> float:
    """Wrap angle into [-pi, pi]"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def actuate(
    u: np.ndarray | tuple[float, float],
    k_v_err: float = 1.0,
    k_w_err: float = 1.0,
) -> np.ndarray:
    """
    Realized velocity commands.
    In Scenario 1 (ideal): k_v_err = 1.0, k_w_err = 1.0.
    In Scenario 2 (disturbed): k_v_err and k_w_err will introduce actuation error/drift.
    """
    return np.array([k_v_err * u[0], k_w_err * u[1]], dtype=float)


def plan_open_loop(
    q_start: np.ndarray | tuple[float, float, float],
    waypoints: list[tuple[float, float]],
    w_bar: float = 1.0,
    v_bar: float = 0.3,
    dt: float = 0.05,
    stop_distance: float = 0.08,
) -> list[tuple[float, float, int]]:
    """
    Plan open-loop commands [(v, w, n_steps), ...] through a sequence of waypoints.
    Each leg is planned from the assumed arrival pose of the previous one.
    """
    commands: list[tuple[float, float, int]] = []
    q_assumed = (float(q_start[0]), float(q_start[1]), float(q_start[2]))
    targets = waypoints[1:] if len(waypoints) > 1 else waypoints

    for idx, (gx, gy) in enumerate(targets):
        is_last_waypoint = (idx == len(targets) - 1)
        x0, y0, theta0 = q_assumed
        d = float(np.hypot(gx - x0, gy - y0))
        if d < 1e-4:
            continue

        # 1. Turn to face waypoint
        heading_desired = float(np.arctan2(gy - y0, gx - x0))
        alpha = wrap(heading_desired - theta0)
        n_rot = int(np.ceil(abs(alpha) / w_bar / dt)) if abs(alpha) > 1e-4 else 0

        if n_rot > 0:
            w_cmd = alpha / (n_rot * dt)
            commands.append((0.0, float(w_cmd), n_rot))

        # 2. Drive straight (if last leg, subtract stop_distance)
        travel_dist = max(0.0, d - stop_distance) if is_last_waypoint else d
        n_tra = int(np.ceil(travel_dist / v_bar / dt)) if travel_dist > 1e-4 else 0
        if n_tra > 0:
            v_cmd = travel_dist / (n_tra * dt)
            commands.append((float(v_cmd), 0.0, n_tra))

        # Assumed pose at the end of this leg
        q_assumed = (
            x0 + travel_dist * np.cos(heading_desired),
            y0 + travel_dist * np.sin(heading_desired),
            heading_desired,
        )

    return commands
