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

from src.env.arena import DEFAULT_WALLS, ROBOT_SIZE
from src.control.camera import VirtualCamera


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


def create_fov_visual(
    env: swift.Swift,
    camera: VirtualCamera,
    color: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.75),  # Semi-transparent green
) -> tuple[list, list[tuple[float, float]]]:
    """
    Create visual boundary spheres on the floor for the camera FOV circular sector.
    Returns (markers_list, local_relative_points).
    """
    local_pts = camera.get_fov_local_points(n_arc_points=9)
    markers = []

    for _ in local_pts:
        marker = sg.Sphere(radius=0.007, color=color)
        env.add(marker)
        markers.append(marker)

    return markers, local_pts


def update_fov_visual(fov_markers: list, local_pts: list[tuple[float, float]], q_robot: np.ndarray) -> None:
    """Update 3D world pose of all FOV sector boundary markers."""
    rx, ry, r_theta = q_robot
    c, s = np.cos(r_theta), np.sin(r_theta)
    rot = np.array([[c, -s], [s, c]])

    for marker, (lx, ly) in zip(fov_markers, local_pts):
        world_pt = rot @ np.array([lx, ly]) + np.array([rx, ry])
        marker.T = sm.SE3(world_pt[0], world_pt[1], 0.005)  # Slightly above floor


def update_swift_camera(env: swift.Swift, q: np.ndarray, mode: str = "default") -> None:
    """
    Dynamically update Swift camera pose based on robot pose (x, y, theta):
    - 'default': Overview from above (static, no-op).
    - 'first_person': Camera mounted on robot's nose, looking in travel direction.
    - 'third_person': Camera placed behind and above, tracking robot smoothly.
    """
    if mode == "default":
        return

    x, y, theta = q
    c, s = np.cos(theta), np.sin(theta)

    if mode == "first_person":
        # Mounted on nose (lx/2 = 0.05m), looking forward into the FOV cone
        cam_pos = [float(x + 0.055 * c), float(y + 0.055 * s), 0.065]
        look_at = [float(x + 0.55 * c), float(y + 0.55 * s), 0.030]
        env.set_camera_pose(cam_pos, look_at)

    elif mode == "third_person":
        # Set back ~35 cm behind heading, elevated ~20 cm
        cam_pos = [float(x - 0.35 * c), float(y - 0.35 * s), 0.22]
        look_at = [float(x + 0.25 * c), float(y + 0.25 * s), 0.04]
        env.set_camera_pose(cam_pos, look_at)


def reset_default_camera(env: swift.Swift) -> None:
    """Reset camera to the standard 60-degree high-angle arena overview."""
    arena_center = np.array([1.0, 1.0, 0.0])
    cam_z = 2.0
    tilt_rad = np.deg2rad(60.0)
    dy = cam_z / np.tan(tilt_rad)
    cam_pos = [arena_center[0], arena_center[1] - dy, cam_z]
    env.set_camera_pose(cam_pos, arena_center.tolist())


def create_grid_visual(vis_res: float = 0.10) -> dict:
    """Create lightweight state tracker for coarse floor visualization of ExplorationGrid."""
    return {"res": float(vis_res), "free_tiles": set(), "occ_tiles": set()}


def update_grid_visual(env: swift.Swift, swift_state: dict, exp_grid, vis_state: dict) -> None:
    """
    Incrementally render mapped FREE and OCCUPIED cells onto the Swift floor using
    coarse dynamic tiles (independent of arena size) without removing existing tiles.
    """
    if exp_grid.width == 0:
        return

    vis_res = vis_state["res"]
    fine_res = exp_grid.res

    # 1. FREE tiles (value == 1)
    fr_rows, fr_cols = np.where(exp_grid.grid == 1)
    if len(fr_rows) > 0:
        f_vx = np.floor((exp_grid.x_min + (fr_cols + 0.5) * fine_res) / vis_res).astype(int)
        f_vy = np.floor((exp_grid.y_min + (fr_rows + 0.5) * fine_res) / vis_res).astype(int)
        for vx, vy in set(zip(f_vx.tolist(), f_vy.tolist())):
            if (vx, vy) not in vis_state["free_tiles"]:
                tile = sg.Cuboid(
                    scale=[vis_res * 0.92, vis_res * 0.92, 0.002],
                    color=(0.45, 0.85, 0.55, 0.30),
                )
                tile.T = sm.SE3((vx + 0.5) * vis_res, (vy + 0.5) * vis_res, 0.001)
                swift_state["dynamic_handles"].append(env.add(tile))
                vis_state["free_tiles"].add((vx, vy))

    # 2. OCCUPIED tiles (value == 2)
    oc_rows, oc_cols = np.where(exp_grid.grid == 2)
    if len(oc_rows) > 0:
        o_vx = np.floor((exp_grid.x_min + (oc_cols + 0.5) * fine_res) / vis_res).astype(int)
        o_vy = np.floor((exp_grid.y_min + (oc_rows + 0.5) * fine_res) / vis_res).astype(int)
        for vx, vy in set(zip(o_vx.tolist(), o_vy.tolist())):
            if (vx, vy) not in vis_state["occ_tiles"]:
                tile = sg.Cuboid(
                    scale=[vis_res * 0.92, vis_res * 0.92, 0.004],
                    color=(0.95, 0.30, 0.15, 0.75),
                )
                tile.T = sm.SE3((vx + 0.5) * vis_res, (vy + 0.5) * vis_res, 0.003)
                swift_state["dynamic_handles"].append(env.add(tile))
                vis_state["occ_tiles"].add((vx, vy))
