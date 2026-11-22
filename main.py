"""
Main entry point for testing arena generation and Swift visualization.
"""

import time
from src.env.arena import get_fixed_scenario, generate_random_scenario, ROBOT_SIZE
from src.env.swift_env import (
    init_swift,
    load_scenario_in_swift,
    create_robot_marker,
    update_robot_pose,
    close_swift_env
)

# Configuration flags
USE_RANDOM = True        # False: fixed arena | True: cycle through random seeds
N_SEEDS = 5               # Number of random worlds to test if USE_RANDOM is True
PAUSE_BETWEEN = 2.0       # Seconds to observe each world


def main():
    print("Launching Swift environment...")
    swift_state = init_swift()
    env = swift_state["env"]

    # Create visual robot markers: orange (open loop) and blue (closed loop)
    robot_ol_vis = create_robot_marker(env, color=(1.0, 0.5, 0.0, 1.0))
    robot_cl_vis = create_robot_marker(env, color=(0.2, 0.5, 0.9, 1.0))

    try:
        if not USE_RANDOM:
            print("\nLoading FIXED scenario...")
            q_start, obstacles, objective = get_fixed_scenario()
            load_scenario_in_swift(swift_state, obstacles, objective)

            # Place both robots at the initial start pose
            update_robot_pose(robot_ol_vis, q_start)
            update_robot_pose(robot_cl_vis, q_start)
            env.step(0.0)

            print("Fixed scenario loaded. Press Ctrl+C in terminal to exit.")
            while True:
                env.step(0.05)
                time.sleep(0.05)

        else:
            print(f"\nCycling through {N_SEEDS} RANDOM seeds in the SAME Swift tab...")
            for seed in range(N_SEEDS):
                print(f"-> Loading Seed {seed}...")
                q_start, obstacles, objective = generate_random_scenario(seed=seed)
                load_scenario_in_swift(swift_state, obstacles, objective)

                update_robot_pose(robot_ol_vis, q_start)
                update_robot_pose(robot_cl_vis, q_start)
                env.step(0.0)

                time.sleep(PAUSE_BETWEEN)

            print("\nAll random seeds displayed successfully!")

    except KeyboardInterrupt:
        print("\nExiting simulation.")
    finally:
        close_swift_env(swift_state)

if __name__ == "__main__":
    main()
