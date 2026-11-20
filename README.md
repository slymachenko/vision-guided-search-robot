# sim2real-vision-search

## Our Project Idea
[Project Proposal: Vision Guided Search Robot](docs/project_proposal_vision_guided_search_robot.md)

## Repository Structure

```text
vision-guided-search-robot/
├── README.md
├── requirements.txt
├── main.py                    # Entry-point: selects Scenario 1, 2, or 3, sets parameters and runs
└── src/
    ├── environment.py         # Swift 3D: bounded arena, obstacles, robot mesh and FOV cone
    ├── robot_controller.py    # THINK: go-to-point, APF obstacle avoidance, open-loop TDT baseline
    ├── perception.py          # SENSE (Scenarios 2–3): measurement noise and virtual camera FOV
    ├── simulation.py          # Sense-Think-Act loop, exploration waypoints, and data logging
    └── analysis.py            # Matplotlib plots and metrics
```