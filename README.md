# sim2real-vision-search

## Our Project Idea

[Project Proposal: Vision Guided Search Robot](docs/project_proposal_vision_guided_search_robot.md)

## Usage

Run the simulation using desired configuration:

```bash
# Run 10 random seeds in the noisy scenario at 5x speed
python -m scripts.run -s 2 -r true -n 10 --speed-factor 5.0

# Run on the fixed arena with a 1-second pause between worlds
python -m scripts.run -s 1 -r false -p 1.0
```

## Repository Structure

```
├── pyproject.toml
├── README.md
├── scripts
│   └── run                       # Entry point for simulation running
└── src
    └── visero
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
