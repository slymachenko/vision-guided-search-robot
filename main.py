"""
Main entry point for robot simulation.
Supports Scenario 1 (ideal), Scenario 2 (disturbed/noisy), Scenario 3 (vision exploration).
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
from src.sim.simulation import run_open_loop_sim

# =============================================================================
# CONFIGURATION
# =============================================================================
SCENARIO = 1              # 1: Ideal conditions | 2: Disturbances/Noise | 3: Vision Search
USE_RANDOM = True         # False: fixed arena | True: random seeds
N_SEEDS = 5               # Trials to run if USE_RANDOM is True
SPEED_FACTOR = 4.0        # Swift visual speedup (1.0 = real-time, 2.0 = 2x speed)
PAUSE_BETWEEN = 2.0       # Seconds to observe final pose between worlds


def main():
    print(f"\nRUNNING SCENARIO {SCENARIO} (Open-Loop)")

    swift_state = init_swift()
    env = swift_state["env"]

    # Visual marker for the open-loop robot (orange)
    robot_ol_vis = create_robot_marker(env, color=(1.0, 0.5, 0.0, 1.0))

    # Scenario 1 parameters: ideal conditions (no actuation errors)
    k_v_err = 1.0
    k_w_err = 1.0
    if SCENARIO == 2:
        k_v_err = 0.95
        k_w_err = 1.05

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

            # 1. Plan path with A* and prune with Line-of-Sight
            print("Planning path with A* and Line-of-Sight pruning...")
            _, waypoints = plan_path(q_start, objective["center"][:2], obstacles)
            print(f"Found path with {len(waypoints)} waypoints.")

            # 2. Plan discrete Turn-Drive-Turn commands
            commands = plan_open_loop(q_start, waypoints)
            print(f"Generated {len(commands)} open-loop command segments.")

            # 3. Execute simulation in Swift
            print("Executing Open-Loop motion in Swift...")
            results = run_open_loop_sim(
                q_start=q_start,
                commands=commands,
                obstacles=obstacles,
                objective=objective,
                speed_factor=SPEED_FACTOR,
                k_v_err=k_v_err,
                k_w_err=k_w_err,
                swift_state=swift_state,
                robot_vis_shape=robot_ol_vis,
            )

            print(
                f"[{label}] Reached: {results['reached']} | "
                f"Collisions: {results['collisions']} | "
                f"Time: {results['time']:.2f}s | "
                f"Path Length: {results['path_length']:.2f}m | "
                f"Final Distance: {results['final_distance']:.4f}m"
            )

            time.sleep(PAUSE_BETWEEN)

    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
    finally:
        close_swift_env(swift_state)


if __name__ == "__main__":
    main()
