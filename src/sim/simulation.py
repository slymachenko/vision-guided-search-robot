"""
Simulation loop supporting single and simultaneous dual-robot execution in Swift.
"""

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm

from src.control.closed_loop import compute_apf_control, ReactiveExplorer
from src.control.exploration_grid import ExplorationGrid
from src.control.perception import PerceptionMemory
from src.control.open_loop import actuate
from src.control.camera import VirtualCamera
from src.env.arena import check_collision, ROBOT_RADIUS, ROBOT_SIZE
from src.env.swift_env import (
    update_robot_pose,
    update_fov_visual,
    update_swift_camera,
    reset_default_camera,
    create_grid_visual,
    update_grid_visual,
)


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
        if step < len(ol_steps):
            u_ol_cmd = ol_steps[step]
        else:
            u_ol_cmd = [0.0, 0.0]
            ol_done = True

        u_ol = actuate(u_ol_cmd, k_v_err=k_v_err_ol, k_w_err=k_w_err_ol)
        v_hist_ol.append(u_ol[0])
        w_hist_ol.append(u_ol[1])
        robot_ol.step(u_ol)
        q_hist_ol.append(robot_ol.q.copy())

        # Check collision OL
        is_col_ol = check_collision(robot_ol.q, obstacles, objective, ROBOT_SIZE)
        if is_col_ol and not was_col_ol:
            col_ol += 1
        was_col_ol = is_col_ol

        # --- B. Closed-Loop Step (APF) ---
        q_measured = robot_cl.q.copy()
        if noise_pos_std > 0.0:
            q_measured[:2] += np.random.normal(0.0, noise_pos_std, size=2)

        dist_cl = float(np.linalg.norm(target_xy - q_measured[:2]))
        if dist_cl <= stop_distance + 0.005:
            u_cl_cmd = [0.0, 0.0]
            cl_done = True
        else:
            v_apf, w_apf, _ = compute_apf_control(
                q_measured, target_xy, obstacles, r_robot=ROBOT_RADIUS, stop_dist=stop_distance
            )
            u_cl_cmd = [v_apf, w_apf]

        u_cl = actuate(u_cl_cmd, k_v_err=k_v_err_cl, k_w_err=k_w_err_cl)
        v_hist_cl.append(u_cl[0])
        w_hist_cl.append(u_cl[1])
        robot_cl.step(u_cl)
        q_hist_cl.append(robot_cl.q.copy())

        # Check collision CL
        is_col_cl = check_collision(robot_cl.q, obstacles, objective, ROBOT_SIZE)
        if is_col_cl and not was_col_cl:
            col_cl += 1
        was_col_cl = is_col_cl

        # --- C. Swift 3D Visualization ---
        if env is not None:
            # Leave breadcrumb trail every 6 steps
            if step % 6 == 0:
                if not ol_done:
                    c_ol = sg.Sphere(radius=0.007, pose=sm.SE3(robot_ol.q[0], robot_ol.q[1], 0.007),
                    color=(1.0, 0.5, 0.0, 0.6))
                    swift_state["dynamic_handles"].append(env.add(c_ol))
                if not cl_done:
                    c_cl = sg.Sphere(radius=0.007, pose=sm.SE3(robot_cl.q[0], robot_cl.q[1], 0.007),
                    color=(0.2, 0.5, 0.9, 0.6))
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
            "reached": (final_dist_ol <= 0.10) and (col_ol == 0),
            "collisions": col_ol,
            "time": len(v_hist_ol) * dt,
            "path_length": float(np.sum(np.linalg.norm(np.diff(q_arr_ol[:, :2], axis=0), axis=1))),
            "final_distance": final_dist_ol,
            "q_hist": q_arr_ol,
            "v_hist": np.array(v_hist_ol),
            "w_hist": np.array(w_hist_ol),
        },
        "closed_loop": {
            "reached": (final_dist_cl <= 0.10) and (col_cl == 0),
            "collisions": col_cl,
            "time": len(v_hist_cl) * dt,
            "path_length": float(np.sum(np.linalg.norm(np.diff(q_arr_cl[:, :2], axis=0), axis=1))),
            "final_distance": final_dist_cl,
            "q_hist": q_arr_cl,
            "v_hist": np.array(v_hist_cl),
            "w_hist": np.array(w_hist_cl),
        },
    }


def run_scenario3_sim(
    q_start: np.ndarray,
    obstacles: list[dict],
    objective: dict,
    camera: VirtualCamera,
    swift_camera_mode: str = "default",
    show_grid_vis: bool = True,
    dt: float = 0.05,
    speed_factor: float = 1.0,
    max_steps: int = 1500,
    swift_state: dict | None = None,
    robot_vis: sg.Cuboid | None = None,
    fov_vis: tuple | None = None,
) -> dict:
    """
    Scenario 3 simulation loop with incremental ExplorationGrid and FOV-based search:
    - Maps traversed FREE, visible FREE, visible OCCUPIED, and UNKNOWN cells dynamically.
    - Explores UNKNOWN frontiers via A* + local APF repulsion.
    - Locks onto target when visible in FOV and approaches safely.
    """

    robot = rtb.Unicycle(dt=dt)
    robot.init(list(q_start))

    target_xy = np.array(objective["center"][:2], dtype=float)
    env = swift_state["env"] if swift_state is not None else None

    grid = ExplorationGrid()
    memory = PerceptionMemory(camera=camera, grid=grid)
    explorer = ReactiveExplorer(fov_angle=camera.fov_angle, dt=dt)

    q_hist = [robot.q.copy()]
    v_hist, w_hist, dist_hist = [], [], []
    collisions = 0
    was_colliding = False

    target_detected = False
    step_detected = None
    q_detected = None

    render_stride = max(1, int(round(speed_factor / 2.0)))
    vis_dt = (dt * render_stride) / max(speed_factor, 1e-3)
    fov_markers, fov_local_pts = fov_vis if fov_vis is not None else (None, None)
    grid_vis = create_grid_visual() if (env is not None and show_grid_vis) else None

    if env is not None:
        if fov_markers is not None:
            update_fov_visual(fov_markers, fov_local_pts, q_start)
        if swift_camera_mode != "default":
            update_swift_camera(env, q_start, mode=swift_camera_mode)
        else:
            reset_default_camera(env)
        env.step(0.0)

    for step in range(max_steps):
        current_time = step * dt
        dist_to_target = float(np.linalg.norm(target_xy - robot.q[:2]))
        dist_hist.append(dist_to_target)

        # 1. Sense: update ExplorationGrid (footprint + FOV) and check target visibility
        was_detected = target_detected
        _, target_detected = memory.update(robot.q, obstacles, target_xy)
        if target_detected and not was_detected:
            step_detected = step
            q_detected = robot.q.copy()
            print(f"-> [TARGET DETECTED!] at step {step} ({current_time:.2f}s). Switching to approaching mode.")

        # 2. Think: check completion or compute velocity commands
        if target_detected and dist_to_target <= 0.085:
            break

        v, w = explorer.compute_step(robot.q, grid, target_xy if target_detected else None)

        # 3. Act: apply unicycle motion
        v_hist.append(v)
        w_hist.append(w)
        robot.step(np.array([v, w]))
        q_hist.append(robot.q.copy())

        # Physical collision check against the actual world
        is_col = check_collision(robot.q, obstacles, objective, ROBOT_SIZE)
        if is_col and not was_colliding:
            collisions += 1
        was_colliding = is_col

        # Swift visual updates
        if env is not None:
            if step % 6 == 0:
                crumb_color = (0.2, 0.5, 0.9, 0.6) if target_detected else (0.1, 0.8, 0.3, 0.5)
                crumb = sg.Sphere(radius=0.007, pose=sm.SE3(robot.q[0], robot.q[1], 0.007), color=crumb_color)
                swift_state["dynamic_handles"].append(env.add(crumb))

            if step % render_stride == 0:
                if robot_vis is not None:
                    update_robot_pose(robot_vis, robot.q, ROBOT_SIZE)
                if fov_markers is not None:
                    update_fov_visual(fov_markers, fov_local_pts, robot.q)
                if grid_vis is not None:
                    update_grid_visual(env, swift_state, grid, grid_vis)
                if swift_camera_mode != "default":
                    update_swift_camera(env, robot.q, mode=swift_camera_mode)
                env.step(vis_dt)

    # Final visual refresh
    if env is not None and robot_vis is not None:
        update_robot_pose(robot_vis, robot.q, ROBOT_SIZE)
        if fov_markers is not None:
            update_fov_visual(fov_markers, fov_local_pts, robot.q)
        if grid_vis is not None:
            update_grid_visual(env, swift_state, grid, grid_vis)
        if swift_camera_mode != "default":
            update_swift_camera(env, robot.q, mode=swift_camera_mode)
        env.step(0.0)

    final_dist = float(np.linalg.norm(target_xy - robot.q[:2]))
    q_arr = np.array(q_hist)

    return {
        "reached": (final_dist <= 0.10) and (collisions == 0) and target_detected,
        "target_detected": target_detected,
        "step_detected": step_detected,
        "time_detected": (step_detected * dt) if step_detected is not None else None,
        "q_detected": q_detected,
        "collisions": collisions,
        "time": len(v_hist) * dt,
        "path_length": float(np.sum(np.linalg.norm(np.diff(q_arr[:, :2], axis=0), axis=1))),
        "final_distance": final_dist,
        "q_hist": q_arr,
        "v_hist": np.array(v_hist),
        "w_hist": np.array(w_hist),
        "dist_hist": np.array(dist_hist),
        "grid": grid,
        "mapped_areas": grid.get_mapped_areas(),
    }
