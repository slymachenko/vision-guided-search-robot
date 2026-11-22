"""
Perception module with decaying working memory for detected obstacles.
Simulates limited sensor range: only nearby obstacles are perceived and remembered temporarily.
"""

import numpy as np
from src.control.closed_loop import closest_surface_info
from src.env.arena import ROBOT_RADIUS


class PerceptionMemory:
    """
    Local working memory for perceived obstacles with temporal decay.
    Obstacles not refreshed within decay_time are forgotten.
    """

    def __init__(
        self,
        sensor_range: float = 0.75,   # Maximum distance to detect an obstacle surface [m]
        decay_time: float = 4.0,      # Seconds an obstacle is remembered after being seen [s]
    ):
        self.sensor_range = sensor_range
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
        Scan environment from current robot pose, add newly perceived obstacles,
        and discard expired ones. Returns list of currently known obstacles.
        """
        p = np.array(q_robot[:2], dtype=float)

        # 1. Perception check: sense obstacles within sensor_range
        for idx, obs in enumerate(all_obstacles):
            d, _ = closest_surface_info(p, obs, r_robot=ROBOT_RADIUS)
            if d <= self.sensor_range:
                # Obstacle perceived: refresh timestamp
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
