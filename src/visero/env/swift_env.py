"""
Swift 3D visualization manager.
Handles scene loading, camera perspective, obstacle rendering and tab reuse across seeds.
"""

import numpy as np
import spatialgeometry as sg
import spatialmath as sm
import swift
import time
import logging

from visero.env.arena import DEFAULT_WALLS, ROBOT_SIZE


def init_swift(
    walls: list[dict] = DEFAULT_WALLS,
    browser: str = "browser",
) -> dict:
    """
    Launch Swift in a single browser tab, configure the camera at 60 deg,
    and add static perimeter walls and a persistent target sphere.
    """
    env = swift.Swift()
    env.launch(realtime=True, browser=browser)

    # 1. Add static perimeter walls (persistent across all trials)
    for w in walls:
        wall_shape = sg.Cuboid(scale=list(w["size"]), color=(0.4, 0.4, 0.45, 1.0))
        wall_shape.T = sm.SE3(*w["center"]) * sm.SE3.Rz(w.get("yaw", 0.0))
        env.add(wall_shape)

    # 2. Add persistent target sphere (repositioned in each scenario)
    target_shape = sg.Sphere(radius=0.02, color=(0.95, 0.1, 0.1, 1.0))
    target_shape.T = sm.SE3(0.0, 0.0, -1.0)  # Hidden initially
    env.add(target_shape)

    # 3. Position camera above the X-axis wall (y=0), elevated 2m, tilted 60 deg
    arena_center = np.array([1.0, 1.0, 0.0])
    cam_z = 2.0
    tilt_rad = np.deg2rad(60.0)
    dy = cam_z / np.tan(tilt_rad)  # ~1.155 m set-back
    cam_pos = [arena_center[0], arena_center[1] - dy, cam_z]
    env.set_camera_pose(cam_pos, arena_center.tolist())

    env.step(0.0)

    return {
        "env": env,
        "target_shape": target_shape,
        "dynamic_handles": [],  # Internal obstacles to clear between trials
    }


def load_scenario_in_swift(
    swift_state: dict,
    obstacles: list[dict],
    objective: dict,
) -> None:
    """
    Clear previous dynamic obstacles and load the new scenario into the existing Swift tab.
    """
    env = swift_state["env"]

    # Remove dynamic obstacles from the previous run
    for handle in swift_state["dynamic_handles"]:
        env.remove(handle)
    swift_state["dynamic_handles"].clear()

    # Add internal obstacles (skip perimeter walls since they are permanent)
    for ob in obstacles:
        if ob.get("kind") == "wall":
            continue

        color = (0.7, 0.35, 0.25, 1.0)
        yaw = float(ob.get("yaw", 0.0))

        if ob["type"] == "box":
            shape = sg.Cuboid(scale=list(ob["size"]), color=color)
        elif ob["type"] == "cylinder":
            shape = sg.Cylinder(radius=ob["radius"], length=ob["height"], color=color)
        else:
            continue

        shape.T = sm.SE3(*ob["center"]) * sm.SE3.Rz(yaw)
        handle = env.add(shape)
        swift_state["dynamic_handles"].append(handle)

    # Reposition target sphere
    swift_state["target_shape"].T = sm.SE3(*objective["center"])
    env.step(0.0)


def close_swift_env(swift_state: dict) -> None:
    """Close the Swift browser tab and shut down the WebSocket server cleanly."""
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    swift_state["env"].close()
    time.sleep(0.1)


def create_robot_marker(
    env: swift.Swift,
    color: tuple[float, float, float, float] = (0.2, 0.5, 0.9, 1.0),
    robot_size: tuple[float, float, float] = ROBOT_SIZE,
) -> sg.Cuboid:
    """Create a 3D visual cuboid for a robot and add it to Swift."""
    body = sg.Cuboid(scale=list(robot_size), color=color)
    env.add(body)
    return body


def update_robot_pose(
    robot_shape: sg.Cuboid,
    q: np.ndarray,
    robot_size: tuple[float, float, float] = ROBOT_SIZE,
) -> None:
    """Update the 3D pose of the robot body."""
    x, y, theta = q
    lz = robot_size[2]
    # Place robot bottom flush on the ground (z = lz / 2)
    robot_shape.T = sm.SE3(x, y, lz / 2.0) * sm.SE3.Rz(theta)
