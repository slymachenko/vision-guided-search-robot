"""
Arena geometry, obstacle definitions, collision detection and scenario generation.
Completely independent of the visualization engine.
"""

import numpy as np

# Physical robot dimensions: 10 cm (length) x 8 cm (width) x 5 cm (height)
ROBOT_SIZE = (0.10, 0.08, 0.05)
# Circumscribed bounding radius in the horizontal plane (~0.064 m)
ROBOT_RADIUS = float(0.5 * np.hypot(ROBOT_SIZE[0], ROBOT_SIZE[1]))

# Arena limits: 2m x 2m bounded by 20 cm high walls
ARENA_BOUNDS = (0.0, 2.0, 0.0, 2.0)  # (x_min, x_max, y_min, y_max)

DEFAULT_WALLS = [
    {"type": "box", "kind": "wall", "size": (2.01, 0.01, 0.2), "center": (1.0, 0.0, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "wall", "size": (2.01, 0.01, 0.2), "center": (1.0, 2.0, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "wall", "size": (0.01, 2.01, 0.2), "center": (0.0, 1.0, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "wall", "size": (0.01, 2.01, 0.2), "center": (2.0, 1.0, 0.1), "yaw": 0.0},
]

FIXED_OBSTACLES = [
    {"type": "box", "kind": "obstacle", "size": (0.8, 0.3, 0.2), "center": (1.3, 1.0, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "obstacle", "size": (0.2, 0.4, 0.2), "center": (0.3, 0.6, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "obstacle", "size": (0.1, 0.3, 0.2), "center": (0.7, 0.4, 0.1), "yaw": 0.0},
    {"type": "box", "kind": "obstacle", "size": (0.6, 0.05, 0.2), "center": (0.3, 1.7, 0.1), "yaw": 0.0},
    {"type": "cylinder", "kind": "obstacle", "radius": 0.1, "height": 0.2, "center": (1.2, 0.4, 0.1)},
    {"type": "cylinder", "kind": "obstacle", "radius": 0.1, "height": 0.2, "center": (0.3, 1.3, 0.1)},
    {"type": "cylinder", "kind": "obstacle", "radius": 0.2, "height": 0.2, "center": (1.3, 1.6, 0.1)},
]


def dist_point_to_obstacle(point: np.ndarray, obs: dict) -> float:
    """Compute surface clearance distance from a 2D point (x, y) to an obstacle."""
    p = np.asarray(point[:2], dtype=float)
    center = np.array(obs["center"][:2], dtype=float)

    if obs["type"] == "cylinder":
        return float(max(0.0, np.linalg.norm(p - center) - obs["radius"]))

    if obs["type"] == "box":
        yaw = float(obs.get("yaw", 0.0))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        r_inv = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        p_local = r_inv @ (p - center)
        half = np.array([obs["size"][0] / 2.0, obs["size"][1] / 2.0])
        q_local = np.clip(p_local, -half, half)
        return float(np.linalg.norm(p_local - q_local))

    raise ValueError(f"Unknown obstacle type: {obs['type']}")


def obstacle_bounding_radius(obs: dict) -> float:
    """Return the 2D bounding radius of an obstacle."""
    if obs["type"] == "cylinder":
        return float(obs["radius"])
    return float(0.5 * np.hypot(obs["size"][0], obs["size"][1]))


def check_collision(
    q: np.ndarray,
    obstacles: list[dict],
    objective: dict | None = None,
    robot_size: tuple[float, float, float] = ROBOT_SIZE,
) -> bool:
    """
    Check if the robot's rectangular footprint touches any obstacle, wall,
    or the objective target sphere (touching the target is considered a collision).
    """
    x, y, theta = q
    hx, hy = robot_size[0] / 2.0, robot_size[1] / 2.0
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])

    # 9 representative test points on the robot body
    local_pts = np.array([
        [0.0, 0.0],
        [hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy],
        [hx, 0.0], [-hx, 0.0], [0.0, hy], [0.0, -hy],
    ])
    world_pts = local_pts @ rot.T + np.array([x, y])

    # 1. Collision with obstacles and walls
    for pt in world_pts:
        for obs in obstacles:
            if dist_point_to_obstacle(pt, obs) <= 0.0:
                return True

    # 2. Collision with target (robot must stop before touching it)
    if objective is not None:
        target_center = np.array(objective["center"][:2], dtype=float)
        target_radius = float(objective.get("radius", 0.02))
        for pt in world_pts:
            if np.linalg.norm(pt - target_center) <= target_radius:
                return True

    return False


def get_fixed_scenario() -> tuple[np.ndarray, list[dict], dict]:
    """Return the deterministic fixed world: (q_start, obstacles, objective)."""
    q_start = np.array([0.2, 0.2, 0.0], dtype=float)
    obstacles = [dict(w) for w in DEFAULT_WALLS] + [dict(o) for o in FIXED_OBSTACLES]
    objective = {
        "type": "sphere",
        "kind": "objective",
        "radius": 0.02,
        "center": (1.8, 1.8, 0.02),
    }
    return q_start, obstacles, objective


def generate_random_scenario(
    n_obstacles: int = 7,
    seed: int | None = None,
    robot_radius: float = ROBOT_RADIUS,
) -> tuple[np.ndarray, list[dict], dict]:
    """
    Generate a feasible random world.
    Guarantees corridor widths >= 2 * robot_radius + safety margin.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = 0.05, 1.95, 0.05, 1.95

    # Safe corridor: robot width + extra safety margin
    min_passage = 2.0 * robot_radius + 0.06
    safe_margin = robot_radius + 0.04

    # 1. Start pose
    sx = float(rng.uniform(x_min + safe_margin, 0.6))
    sy = float(rng.uniform(y_min + safe_margin, 0.6))
    stheta = float(rng.uniform(-np.pi, np.pi))
    q_start = np.array([sx, sy, stheta], dtype=float)
    start_xy = np.array([sx, sy], dtype=float)

    # 2. Obstacles (walls are always included)
    obstacles = [dict(w) for w in DEFAULT_WALLS]

    for _ in range(n_obstacles):
        for _ in range(500):
            if rng.random() < 0.5:
                cand = {
                    "type": "cylinder",
                    "kind": "obstacle",
                    "radius": float(rng.uniform(0.08, 0.16)),
                    "height": 0.2,
                }
            else:
                cand = {
                    "type": "box",
                    "kind": "obstacle",
                    "size": (
                        float(rng.uniform(0.12, 0.35)),
                        float(rng.uniform(0.10, 0.25)),
                        0.2,
                    ),
                    "yaw": float(rng.uniform(0.0, np.pi)),
                }

            r_cand = obstacle_bounding_radius(cand)
            cx = float(rng.uniform(x_min + r_cand, x_max - r_cand))
            cy = float(rng.uniform(y_min + r_cand, y_max - r_cand))
            cand["center"] = (cx, cy, 0.1)

            # Ensure clearance from robot start pose
            if np.hypot(cx - sx, cy - sy) < (r_cand + robot_radius + 0.05):
                continue

            # Ensure corridors between obstacles are traversable
            valid = True
            for existing in obstacles[len(DEFAULT_WALLS):]:
                r_ex = obstacle_bounding_radius(existing)
                ex_x, ex_y = existing["center"][0], existing["center"][1]
                if np.hypot(cx - ex_x, cy - ex_y) < (r_cand + r_ex + min_passage):
                    valid = False
                    break

            if valid:
                obstacles.append(cand)
                break

    # 3. Target (placed in the opposite half of the arena, away from obstacles)
    for _ in range(500):
        tx = float(rng.uniform(1.2, x_max - safe_margin))
        ty = float(rng.uniform(1.2, y_max - safe_margin))
        t_xy = np.array([tx, ty], dtype=float)

        if np.linalg.norm(t_xy - start_xy) < 1.0:
            continue
        if any(dist_point_to_obstacle(t_xy, obs) < safe_margin for obs in obstacles):
            continue

        objective = {
            "type": "sphere",
            "kind": "objective",
            "radius": 0.02,
            "center": (tx, ty, 0.02),
        }
        return q_start, obstacles, objective

    # Fallback target if placement fails
    objective = {"type": "sphere", "kind": "objective", "radius": 0.02, "center": (1.8, 1.8, 0.02)}
    return q_start, obstacles, objective
