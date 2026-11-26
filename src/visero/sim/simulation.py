"""
Simulation loop supporting single and simultaneous dual-robot execution in Swift.
"""

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm

from visero.control.closed_loop import compute_apf_control
from visero.control.open_loop import actuate
from visero.env.arena import check_collision, ROBOT_RADIUS, ROBOT_SIZE
from visero.env.swift_env import update_robot_pose


def run_dual_sim(
    q_start: np.ndarray,
    commands_ol: list[tuple[float, float, int]],
    obstacles: list[dict],
    objective: dict,
    q_start_real: np.ndarray | None = None,
    dt: float = 0.05,
    speed_factor: float = 1.0,
    k_v_err_ol: float = 1.0,
    k_w_err_ol: float = 1.0,
    k_v_err_cl: float = 1.0,
    k_w_err_cl: float = 1.0,
    noise_pos_std: float = 0.0,
    stop_distance: float = 0.09,
    max_steps: int = 1500,
    swift_state: dict | None = None,
    robot_ol_vis: sg.Cuboid | None = None,
    robot_cl_vis: sg.Cuboid | None = None,
) -> dict:
    """
    Simulate both Open-Loop (orange) and Closed-Loop (blue) robots simultaneously
    in the same Swift environment.
    """
    # 1. Initialize kinematics for both robots
    real_start = q_start if q_start_real is None else q_start_real

    robot_ol = rtb.Unicycle(dt=dt)
    robot_ol.init(list(real_start))

    robot_cl = rtb.Unicycle(dt=dt)
    robot_cl.init(list(real_start))

    target_xy = np.array(objective["center"][:2], dtype=float)
    env = swift_state["env"] if swift_state is not None else None

    # Flatten open-loop commands into step-by-step array
    ol_steps = []
    for v, w, n in commands_ol:
        ol_steps.extend([[v, w]] * n)

    # Histories and collision debouncing
    q_hist_ol = [robot_ol.q.copy()]
    q_hist_cl = [robot_cl.q.copy()]
    v_hist_ol, w_hist_ol = [], []
    v_hist_cl, w_hist_cl = [], []

    col_ol, col_cl = 0, 0
    was_col_ol, was_col_cl = False, False

    render_stride = max(1, int(round(speed_factor / 2.0)))
    vis_dt = (dt * render_stride) / max(speed_factor, 1e-3)

    ol_done = False
    cl_done = False

    for step in range(max_steps):
        # --- A. Open-Loop Step ---
        if not ol_done:
            if step < len(ol_steps):
                u_ol = actuate(ol_steps[step], k_v_err=k_v_err_ol, k_w_err=k_w_err_ol)
                v_hist_ol.append(u_ol[0])
                w_hist_ol.append(u_ol[1])
                robot_ol.step(u_ol)
                q_hist_ol.append(robot_ol.q.copy())

                # Check collision OL
                is_col_ol = check_collision(
                    robot_ol.q, obstacles, objective, ROBOT_SIZE
                )
                if is_col_ol and not was_col_ol:
                    col_ol += 1
                was_col_ol = is_col_ol

            if step >= len(ol_steps) - 1:
                ol_done = True

        # --- B. Closed-Loop Step (APF) ---
        if not cl_done:
            q_measured = robot_cl.q.copy()
            if noise_pos_std > 0.0:
                q_measured[:2] += np.random.normal(0.0, noise_pos_std, size=2)

            dist_cl = float(np.linalg.norm(target_xy - q_measured[:2]))
            if dist_cl <= stop_distance:
                cl_done = True
            else:
                v_apf, w_apf, _ = compute_apf_control(
                    q_measured,
                    target_xy,
                    obstacles,
                    r_robot=ROBOT_RADIUS,
                    stop_dist=stop_distance,
                )
                u_cl = actuate([v_apf, w_apf], k_v_err=k_v_err_cl, k_w_err=k_w_err_cl)
                v_hist_cl.append(u_cl[0])
                w_hist_cl.append(u_cl[1])
                robot_cl.step(u_cl)
                q_hist_cl.append(robot_cl.q.copy())

                # Check collision CL
                is_col_cl = check_collision(
                    robot_cl.q, obstacles, objective, ROBOT_SIZE
                )
                if is_col_cl and not was_col_cl:
                    col_cl += 1
                was_col_cl = is_col_cl

        # --- C. Swift 3D Visualization ---
        if env is not None:
            # Leave breadcrumb trail every 6 steps
            if step % 6 == 0:
                if not ol_done:
                    c_ol = sg.Sphere(
                        radius=0.007,
                        pose=sm.SE3(robot_ol.q[0], robot_ol.q[1], 0.007),
                        color=(1.0, 0.5, 0.0, 0.6),
                    )
                    swift_state["dynamic_handles"].append(env.add(c_ol))
                if not cl_done:
                    c_cl = sg.Sphere(
                        radius=0.007,
                        pose=sm.SE3(robot_cl.q[0], robot_cl.q[1], 0.007),
                        color=(0.2, 0.5, 0.9, 0.6),
                    )
                    swift_state["dynamic_handles"].append(env.add(c_cl))

            if step % render_stride == 0:
                if robot_ol_vis is not None:
                    update_robot_pose(robot_ol_vis, robot_ol.q, ROBOT_SIZE)
                if robot_cl_vis is not None:
                    update_robot_pose(robot_cl_vis, robot_cl.q, ROBOT_SIZE)
                env.step(vis_dt)

        # Stop early if both robots have completed their task
        if ol_done and cl_done:
            break

    # Final visual refresh
    if env is not None:
        if robot_ol_vis is not None:
            update_robot_pose(robot_ol_vis, robot_ol.q, ROBOT_SIZE)
        if robot_cl_vis is not None:
            update_robot_pose(robot_cl_vis, robot_cl.q, ROBOT_SIZE)
        env.step(0.0)

    # Compile results
    q_arr_ol = np.array(q_hist_ol)
    q_arr_cl = np.array(q_hist_cl)
    final_dist_ol = float(np.linalg.norm(target_xy - robot_ol.q[:2]))
    final_dist_cl = float(np.linalg.norm(target_xy - robot_cl.q[:2]))

    return {
        "open_loop": {
            "reached": (final_dist_ol <= 0.11) and (col_ol == 0),
            "collisions": col_ol,
            "time": len(v_hist_ol) * dt,
            "path_length": float(
                np.sum(np.linalg.norm(np.diff(q_arr_ol[:, :2], axis=0), axis=1))
            ),
            "final_distance": final_dist_ol,
            "q_hist": q_arr_ol,
            "v_hist": np.array(v_hist_ol),
            "w_hist": np.array(w_hist_ol),
        },
        "closed_loop": {
            "reached": (final_dist_cl <= 0.11) and (col_cl == 0),
            "collisions": col_cl,
            "time": len(v_hist_cl) * dt,
            "path_length": float(
                np.sum(np.linalg.norm(np.diff(q_arr_cl[:, :2], axis=0), axis=1))
            ),
            "final_distance": final_dist_cl,
            "q_hist": q_arr_cl,
            "v_hist": np.array(v_hist_cl),
            "w_hist": np.array(w_hist_cl),
        },
    }
