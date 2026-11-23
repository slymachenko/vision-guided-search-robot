"""
Perception module with decaying working memory for detected obstacles.
Only obstacles currently visible within the camera FOV are perceived and remembered.
"""

import numpy as np
from src.control.camera import VirtualCamera


class PerceptionMemory:
    """
    Local working memory for perceived obstacles with temporal decay.
    Only obstacles entering the camera's FOV are perceived.
    Obstacles not refreshed within decay_time are forgotten.
    """

    def __init__(
        self,
        camera: VirtualCamera,
        decay_time: float = 4.0,  # Seconds an obstacle is remembered after leaving FOV
    ):
        self.camera = camera
        self.decay_time = decay_time
        # Dictionary storing: {obs_id: {"obstacle": obs_dict, "last_seen": float}}
        self._memory: dict[int, dict] = {}

    def update(
        self,
        q_robot: np.ndarray,
        all_obstacles: list[dict],
        current_time: float,
    ) -> list[dict]:
        """
        Scan environment from current robot pose using the camera FOV.
        Adds newly perceived obstacles and discards expired ones.
        """
        # 1. Perception check: only obstacles visible in the camera FOV (not occluded)
        for idx, obs in enumerate(all_obstacles):
            if self.camera.is_obstacle_visible(q_robot, obs, all_obstacles):
                # Obstacle is within the green FOV cone: refresh memory
                self._memory[idx] = {
                    "obstacle": obs,
                    "last_seen": current_time,
                }

        # 2. Decay check: forget obstacles not seen recently
        expired_keys = [
            k for k, item in self._memory.items()
            if (current_time - item["last_seen"]) > self.decay_time
        ]
        for k in expired_keys:
            del self._memory[k]

        return self.get_known_obstacles()

    def get_known_obstacles(self) -> list[dict]:
        """Return the list of obstacles currently present in working memory."""
        return [item["obstacle"] for item in self._memory.values()]

    def reset(self) -> None:
        """Clear memory at the start of a new scenario."""
        self._memory.clear()
