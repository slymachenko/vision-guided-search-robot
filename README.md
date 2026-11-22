# sim2real-vision-search

## Our Project Idea
[Project Proposal: Vision Guided Search Robot](docs/project_proposal_vision_guided_search_robot.md)

## Repository Structure

```
vision-guided-search-robot/
├── README.md
├── main.py                   # Entry-point: selects Scenario 1, 2 or 3, sets parameters and runs
└── src/
    ├── control/
    │   ├── planner.py        # A* path planning on 2D occupancy grid + LOS pruning
    │   ├── open_loop.py      # Open-loop turn-drive-turn planner and actuation model
    │   └── closed_loop.py    # Closed-loop go-to-point + APF obstacle avoidance controller
    ├── env/
    │   ├── arena.py          # Arena geometry, obstacle definitions, collision detection and scenario generation
    │   └── swift_env.py      # Swift 3D visualization manager
    └── sim/
        ├── simulation.py     # Open and closed-loop simulation of the robot in the arena, simultaneously
        └── analysis.py       # Matplotlib plots and metrics
```