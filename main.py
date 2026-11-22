"""
Main entry point for robot simulation.
Executes Open-Loop (orange) and Closed-Loop (blue) side-by-side in Swift.
"""

import time
from src.env.arena import get_fixed_scenario, generate_random_scenario
from src.env.swift_env import (
    init_swift,
    load_scenario_in_swift,
    create_robot_marker,
    update_robot_pose,
    close_swift_env,
)
from src.control.planner import plan_path
from src.control.open_loop import plan_open_loop
from src.sim.simulation import run_dual_sim

# =============================================================================
# CONFIGURATION
# =============================================================================
SCENARIO = 1              # 1: Ideal | 2: Disturbances/Noise | 3: Vision Search
USE_RANDOM = True         # False: fixed arena | True: random seeds
N_SEEDS = 5               # Trials to run if USE_RANDOM is True
SPEED_FACTOR = 3.0        # Swift visual speedup (1.0 = real-time, 2.0 = 2x speed)
PAUSE_BETWEEN = 2.0       # Seconds to observe final pose between worlds


def main():
    print(f"RUNNING SCENARIO {SCENARIO}: DUAL ROBOT (Open-Loop vs Closed-Loop)")

    swift_state = init_swift()
    env = swift_state["env"]

    # Visual robot markers: orange (open loop) and blue (closed loop)
    robot_ol_vis = create_robot_marker(env, color=(1.0, 0.5, 0.0, 1.0))
    robot_cl_vis = create_robot_marker(env, color=(0.2, 0.5, 0.9, 1.0))

    # Scenario parameters
    k_v_err_ol, k_w_err_ol = 1.0, 1.0
    k_v_err_cl, k_w_err_cl = 1.0, 1.0

    if SCENARIO == 2:
        # Scenario 2: imperfect actuation on open loop
        k_v_err_ol, k_w_err_ol = 0.95, 1.05

    try:
        scenarios = []
        if not USE_RANDOM:
            q_start, obstacles, objective = get_fixed_scenario()
            scenarios.append(("fixed", q_start, obstacles, objective))
        else:
            for s in range(N_SEEDS):
                q_start, obstacles, objective = generate_random_scenario(seed=s)
                scenarios.append((f"seed_{s}", q_start, obstacles, objective))

        for label, q_start, obstacles, objective in scenarios:
            print(f"\n--- Loading {label} ---")
            load_scenario_in_swift(swift_state, obstacles, objective)
            update_robot_pose(robot_ol_vis, q_start)
            update_robot_pose(robot_cl_vis, q_start)

            # 1. Plan path for Open Loop with A* and Line-of-Sight
            _, waypoints = plan_path(q_start, objective["center"][:2], obstacles)
            commands_ol = plan_open_loop(q_start, waypoints)

            # 2. Run both robots simultaneously in Swift
            print("Simulating Open-Loop (orange) and Closed-Loop (blue) in Swift...")
            results = run_dual_sim(
                q_start=q_start,
                commands_ol=commands_ol,
                obstacles=obstacles,
                objective=objective,
                speed_factor=SPEED_FACTOR,
                k_v_err_ol=k_v_err_ol,
                k_w_err_ol=k_w_err_ol,
                k_v_err_cl=k_v_err_cl,
                k_w_err_cl=k_w_err_cl,
                swift_state=swift_state,
                robot_ol_vis=robot_ol_vis,
                robot_cl_vis=robot_cl_vis,
            )

            res_ol = results["open_loop"]
            res_cl = results["closed_loop"]

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

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        close_swift_env(swift_state)


if __name__ == "__main__":
    main()
