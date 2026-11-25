"""
Main entry point for robot simulation.
Supports:
- Scenario 1: Ideal conditions (Open-Loop vs Closed-Loop)
- Scenario 2: Disturbances and Noise (Open-Loop vs Closed-Loop)
- Scenario 3: Vision Guided Search (Single Closed-Loop Robot with FOV)
"""

import time
import numpy as np
from src.env.arena import get_fixed_scenario, generate_random_scenario
from src.env.swift_env import (
    init_swift,
    load_scenario_in_swift,
    create_robot_marker,
    update_robot_pose,
    create_fov_visual,
    close_swift_env,
)
from src.control.planner import plan_path
from src.control.open_loop import plan_open_loop
from src.control.camera import VirtualCamera
from src.sim.simulation import run_dual_sim, run_scenario3_sim
from src.sim.analysis import (
    plot_trajectories_2d,
    plot_metrics_comparison,
    plot_scenario3_maps,
    plot_scenario3_metrics,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
SCENARIO = 3                  # 1: Ideal | 2: Disturbances/Noise | 3: Vision Search
USE_RANDOM = True             # False: fixed arena | True: random seeds
N_SEEDS = 5                   # Trials to run if USE_RANDOM is True
START_SEED = 0                # Starting seed for random scenarios
SPEED_FACTOR = 3.0            # Swift visual speedup (1.0 = real-time, 2.0 = 2x speed)
PAUSE_BETWEEN = 2.0           # Seconds to observe final pose between worlds
CAMERA_MODE = "default"       # (only for Scenario 3) "default" | "first_person" | "third_person"
SHOW_GRID_VIS = True          # (only for Scenario 3) Render explored FREE/OCCUPIED tiles in Swift


def main():
    print(f"RUNNING SCENARIO {SCENARIO}: " + ("DUAL ROBOT COMPARISON" if SCENARIO in (1, 2) else "VISION-GUIDED SEARCH (FOV)"))

    swift_state = init_swift()
    env = swift_state["env"]

    try:
        # 1. Build scenario list (compatible with fixed or random seeds for all scenarios)
        scenarios = []
        if not USE_RANDOM:
            q_start, obstacles, objective = get_fixed_scenario()
            scenarios.append(("fixed", q_start, obstacles, objective))
        else:
            for s in range(START_SEED, START_SEED + N_SEEDS):
                q_start, obstacles, objective = generate_random_scenario(seed=s)
                scenarios.append((f"seed_{s}", q_start, obstacles, objective))

        # =====================================================================
        # SCENARIOS 1 & 2: DUAL ROBOT (Open-Loop vs Closed-Loop)
        # =====================================================================
        if SCENARIO in (1, 2):
            robot_ol_vis = create_robot_marker(env, color=(1.0, 0.5, 0.0, 1.0))
            robot_cl_vis = create_robot_marker(env, color=(0.2, 0.5, 0.9, 1.0))

            if SCENARIO == 1:
                k_v_err_ol, k_w_err_ol = 1.0, 1.0
                k_v_err_cl, k_w_err_cl = 1.0, 1.0
                initial_offset = np.array([0.0, 0.0, 0.0])
                noise_std = 0.0
            else:  # SCENARIO == 2
                k_v_err_ol, k_w_err_ol = 0.93, 1.08
                k_v_err_cl, k_w_err_cl = 0.95, 1.05
                initial_offset = np.array([0.03, -0.02, np.deg2rad(4.0)])
                noise_std = 0.003

            batch_results = []
            for label, q_start, obstacles, objective in scenarios:
                print(f"\n--- Loading {label} ---")
                load_scenario_in_swift(swift_state, obstacles, objective)

                # Plan Open-Loop commands
                _, waypoints = plan_path(q_start, objective["center"][:2], obstacles)
                commands_ol = plan_open_loop(q_start, waypoints)

                q_start_real = q_start + initial_offset
                update_robot_pose(robot_ol_vis, q_start_real)
                update_robot_pose(robot_cl_vis, q_start_real)

                print("Simulating Open-Loop (orange) and Closed-Loop (blue) in Swift...")
                results = run_dual_sim(
                    q_start=q_start,
                    commands_ol=commands_ol,
                    obstacles=obstacles,
                    objective=objective,
                    q_start_real=q_start_real,
                    speed_factor=SPEED_FACTOR,
                    k_v_err_ol=k_v_err_ol,
                    k_w_err_ol=k_w_err_ol,
                    k_v_err_cl=k_v_err_cl,
                    k_w_err_cl=k_w_err_cl,
                    noise_pos_std=noise_std,
                    swift_state=swift_state,
                    robot_ol_vis=robot_ol_vis,
                    robot_cl_vis=robot_cl_vis,
                )

                res_ol = results["open_loop"]
                res_cl = results["closed_loop"]

                batch_results.append({
                    "label": label,
                    "q_start": q_start,
                    "obstacles": obstacles,
                    "objective": objective,
                    "waypoints": waypoints,
                    "open_loop": res_ol,
                    "closed_loop": res_cl,
                })

                print("\nRESULTS SUMMARY:")
                print(
                    f"  Open-Loop (Orange)  -> Reached: {res_ol['reached']} | "
                    f"Collisions: {res_ol['collisions']} | Time: {res_ol['time']:.2f}s | "
                    f"Path: {res_ol['path_length']:.2f}m | Final Dist: {res_ol['final_distance']:.4f}m"
                )
                print(
                    f"  Closed-Loop (Blue)  -> Reached: {res_cl['reached']} | "
                    f"Collisions: {res_cl['collisions']} | Time: {res_cl['time']:.2f}s | "
                    f"Path: {res_cl['path_length']:.2f}m | Final Dist: {res_cl['final_distance']:.4f}m"
                )

                time.sleep(PAUSE_BETWEEN)

            # Analysis and comparative plots
            prefix = f"scenario_{SCENARIO}_" + ("random_batch" if USE_RANDOM else "fixed")
            plot_trajectories_2d(batch_results, save_path=f"results/plots/{prefix}_trajectories.png")
            plot_metrics_comparison(batch_results, dt=0.05, save_path=f"results/plots/{prefix}_metrics.png")

        # =====================================================================
        # SCENARIO 3: SINGLE ROBOT WITH VIRTUAL CAMERA (FOV)
        # =====================================================================
        elif SCENARIO == 3:
            # Single robot with virtual camera
            camera = VirtualCamera(fov_angle=np.deg2rad(70.0), max_range=0.85, false_negative_prob=0.0)
            robot_vis = create_robot_marker(env, color=(0.2, 0.5, 0.9, 1.0))

            batch_results = []
            for label, q_start, obstacles, objective in scenarios:
                print(f"\n--- Loading {label} (Scenario 3: Vision Search) ---")
                load_scenario_in_swift(swift_state, obstacles, objective)

                # Create FOV green sector markers on the floor and track for cleanup
                fov_handles, fov_pts = create_fov_visual(env, camera)
                swift_state["dynamic_handles"].extend(fov_handles)

                update_robot_pose(robot_vis, q_start)

                print("Exploring and searching target with FOV camera...")
                res = run_scenario3_sim(
                    q_start=q_start,
                    obstacles=obstacles,
                    objective=objective,
                    camera=camera,
                    swift_camera_mode=CAMERA_MODE,
                    show_grid_vis=SHOW_GRID_VIS,
                    speed_factor=SPEED_FACTOR,
                    swift_state=swift_state,
                    robot_vis=robot_vis,
                    fov_vis=(fov_handles, fov_pts),
                )

                batch_results.append({
                    "label": label,
                    "q_start": q_start,
                    "obstacles": obstacles,
                    "objective": objective,
                    "result": res,
                })

                time_det_str = f"{res['time_detected']:.1f}s" if res['time_detected'] is not None else "Never"
                areas = res["mapped_areas"]
                print("\nSCENARIO 3 RESULTS:")
                print(
                    f"  Robot (Blue) -> Reached: {res['reached']} | Detected: {res['target_detected']} "
                    f"(at {time_det_str}) | Collisions: {res['collisions']} | "
                    f"Time: {res['time']:.1f}s | Path: {res['path_length']:.2f}m | "
                    f"Final Dist: {res['final_distance']:.4f}m | "
                    f"Mapped Free: {areas['free_m2']:.2f}m² | Occ: {areas['occupied_m2']:.2f}m²"
                )

                time.sleep(PAUSE_BETWEEN)

            prefix = "scenario_3_" + ("random_batch" if USE_RANDOM else "fixed")
            plot_scenario3_maps(batch_results, save_path=f"results/plots/{prefix}_maps.png")
            plot_scenario3_metrics(batch_results, dt=0.05, save_path=f"results/plots/{prefix}_metrics.png")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        close_swift_env(swift_state)


if __name__ == "__main__":
    main()
