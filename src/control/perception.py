"""
Perception module for Scenario 3.
Updates the incremental ExplorationGrid from robot traversal and VirtualCamera FOV,
and checks whether the target is visible inside the camera cone.
"""

import numpy as np
from src.control.camera import VirtualCamera
from src.control.exploration_grid import ExplorationGrid


class PerceptionMemory:
    """
    Persistent spatial perception using ExplorationGrid and VirtualCamera.
    Replaces short-term forgetting with incremental FREE / OCCUPIED / UNKNOWN mapping.
    """

    def __init__(
        self,
        camera: VirtualCamera,
        grid: ExplorationGrid | None = None,
    ):
        self.camera = camera
        self.grid = grid if grid is not None else ExplorationGrid()
        self.target_detected = False
        self.target_pos: np.ndarray | None = None

    def update(
        self,
        q_robot: np.ndarray,
        all_obstacles: list[dict],
        target_xy: tuple[float, float] | np.ndarray | None = None,
    ) -> tuple[ExplorationGrid, bool]:
        """
        Sense the environment from current pose:
        1. Update ExplorationGrid with traversed footprint and camera FOV rays.
        2. Check target visibility within the FOV cone.
        """
        self.grid.update(q_robot, all_obstacles, self.camera)

        if target_xy is not None and not self.target_detected:
            if self.camera.is_target_visible(q_robot, target_xy, all_obstacles):
                self.target_detected = True
                self.target_pos = np.asarray(target_xy[:2], dtype=float)

        return self.grid, self.target_detected

    def reset(self) -> None:
        """Clear grid and detection state at the start of a new scenario."""
        self.grid.reset()
        self.target_detected = False
        self.target_pos = None
