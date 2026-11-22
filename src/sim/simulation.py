"""
Simulation loop executing robot motion, collision checks and Swift visual rendering.
"""

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm

from src.env.arena import check_collision, ROBOT_SIZE
from src.env.swift_env import update_robot_pose


def run_open_loop_sim(
    q_start: np.ndarray,
    commands: list[tuple[float, float, int]],
    obstacles: list[dict],
    objective: dict,
    dt: float = 0.05,
    speed_factor: float = 1.0,
    k_v_err: float = 1.0,
    k_w_err: float = 1.0,
    swift_state: dict | None = None,
    robot_vis_shape: sg.Cuboid | None = None,
) -> dict:
    """
    Execute open-loop commands on a unicycle robot.
    - speed_factor > 1.0 speeds up visual playback without altering physics dt.
    - k_v_err and k_w_err simulate actuation discrepancies (Scenario 2).
    """
    from src.control.open_loop import actuate

    robot = rtb.Unicycle(dt=dt)
    robot.init(list(q_start))

    target_xy = np.array(objective["center"][:2], dtype=float)
    target_radius = float(objective.get("radius", 0.02))

    render_stride = max(1, int(round(speed_factor / 2.0)))
    vis_dt = (dt * render_stride) / max(speed_factor, 1e-3)

    q_hist = [robot.q.copy()]
    v_hist = []
    w_hist = []
    dist_hist = []
    collisions = 0
    was_colliding = False
    step = 0

    env = swift_state["env"] if swift_state is not None else None

    for v_cmd, w_cmd, n_steps in commands:
        for _ in range(n_steps):
            step += 1

            # Distance to target center
            dist = float(np.linalg.norm(target_xy - robot.q[:2]))
            dist_hist.append(dist)

            # Actuate command
            u_real = actuate([v_cmd, w_cmd], k_v_err=k_v_err, k_w_err=k_w_err)
            v_hist.append(u_real[0])
            w_hist.append(u_real[1])

            # Kinematic step
            robot.step(u_real)
            q_hist.append(robot.q.copy())

            # Collision check (both obstacles and target sphere, debounced: counts 1 per contact)
            is_colliding = check_collision(robot.q, obstacles, objective, robot_size=ROBOT_SIZE)
            if is_colliding and not was_colliding:
                collisions += 1
            was_colliding = is_colliding

            # Swift 3D rendering
            if env is not None and robot_vis_shape is not None:
                if step % 6 == 0:  # breadcrumb trail
                    crumb = sg.Sphere(radius=0.008, pose=sm.SE3(robot.q[0], robot.q[1], 0.008), color=(1.0,
                    0.5, 0.0, 0.7))
                    handle = env.add(crumb)
                    swift_state["dynamic_handles"].append(handle)

                if step % render_stride == 0:
                    update_robot_pose(robot_vis_shape, robot.q, robot_size=ROBOT_SIZE)
                    env.step(vis_dt)

    # Render final state
    if env is not None and robot_vis_shape is not None:
        update_robot_pose(robot_vis_shape, robot.q, robot_size=ROBOT_SIZE)
        env.step(0.0)

    final_dist = float(np.linalg.norm(target_xy - robot.q[:2]))
    # Stop margin: reached if close to target surface without colliding
    reached = (final_dist <= 0.10) and (collisions == 0)

    q_arr = np.array(q_hist)
    path_len = float(np.sum(np.linalg.norm(np.diff(q_arr[:, :2], axis=0), axis=1)))

    return {
        "reached": reached,
        "collisions": collisions,
        "time": len(v_hist) * dt,
        "path_length": path_len,
        "final_distance": final_dist,
        "q_hist": q_arr,
        "v_hist": np.array(v_hist),
        "w_hist": np.array(w_hist),
        "dist_hist": np.array(dist_hist),
    }
