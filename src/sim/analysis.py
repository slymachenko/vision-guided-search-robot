"""
Plotting and performance analysis module.
Generates 2D arena reconstructions and comparative metric charts in results/plots.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch, Polygon
import numpy as np


def draw_arena_2d(ax, obstacles: list[dict], objective: dict, q_start: np.ndarray | None = None) -> None:
    """Draw perimeter walls, box obstacles (with yaw), cylinders and target on a 2D axis."""
    for obs in obstacles:
        if obs["type"] == "cylinder":
            cx, cy = obs["center"][0], obs["center"][1]
            ax.add_patch(Circle((cx, cy), obs["radius"], color="tab:red", alpha=0.5))
        elif obs["type"] == "box":
            cx, cy = obs["center"][0], obs["center"][1]
            hx, hy = obs["size"][0] / 2.0, obs["size"][1] / 2.0
            yaw = float(obs.get("yaw", 0.0))
            c, s = np.cos(yaw), np.sin(yaw)
            rot = np.array([[c, -s], [s, c]])
            local_corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
            world_corners = local_corners @ rot.T + np.array([cx, cy])

            is_wall = obs.get("kind") == "wall"
            color = "0.2" if is_wall else "tab:gray"
            alpha = 0.9 if is_wall else 0.6
            ax.add_patch(Polygon(world_corners, closed=True, color=color, alpha=alpha))

    # Target
    tx, ty = objective["center"][0], objective["center"][1]
    tr = objective.get("radius", 0.02)
    ax.add_patch(Circle((tx, ty), tr, color="red", zorder=5))
    ax.plot(tx, ty, "r*", markersize=10, zorder=6, label="Target")

    # Start pose
    if q_start is not None:
        ax.plot(q_start[0], q_start[1], "ko", markersize=6, zorder=6, label="Start")
        # Direction arrow
        dx = 0.08 * np.cos(q_start[2])
        dy = 0.08 * np.sin(q_start[2])
        ax.arrow(q_start[0], q_start[1], dx, dy, head_width=0.03, color="black", zorder=6)



def plot_trajectories_2d(batch_results: list[dict], save_path: str = "results/plots/trajectories.png") -> None:
    """
    Save 2D trajectory reconstructions comparing Open-Loop (orange) vs Closed-Loop (blue).
    Supports single runs or grid layouts for multiple random seeds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(batch_results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, item in enumerate(batch_results):
        ax = axes[idx]
        obstacles = item["obstacles"]
        objective = item["objective"]
        q_start = item["q_start"]
        res_ol = item["open_loop"]
        res_cl = item["closed_loop"]
        waypoints = item.get("waypoints")

        # 1. Draw arena layout
        draw_arena_2d(ax, obstacles, objective, q_start)

        # 2. Draw planned A* waypoints
        if waypoints is not None and len(waypoints) > 0:
            wp = np.array(waypoints)
            ax.plot(wp[:, 0], wp[:, 1], "g--o", markersize=3, lw=1.0, alpha=0.5, label="A* Plan")

        # 3. Draw executed trajectories
        q_ol = res_ol["q_hist"]
        q_cl = res_cl["q_hist"]
        ax.plot(q_ol[:, 0], q_ol[:, 1], color="tab:orange", lw=1.8, linestyle="--", label="Open-Loop")
        ax.plot(q_cl[:, 0], q_cl[:, 1], color="tab:blue", lw=1.8, linestyle="-", label="Closed-Loop")

        label = item.get("label", f"Trial {idx}")
        status_ol = "OK" if res_ol["reached"] else "FAIL"
        status_cl = "OK" if res_cl["reached"] else "FAIL"

        ax.set_title(
            f"[{label}] OL: {status_ol} (d={res_ol['final_distance']:.2f}m) | CL: {status_cl} (d={res_cl['final_distance']:.2f}m)",
            fontsize=8.5,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 2.05)
        ax.set_axisbelow(True)
        ax.grid(True, color="0.9", linestyle=":")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.5)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D trajectory plot: {save_path}")


def plot_metrics_comparison(batch_results: list[dict], dt: float = 0.05, save_path: str = "results/plots/metrics.png") -> None:
    """Save quantitative comparison charts between Open-Loop and Closed-Loop."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [r.get("label", str(i)) for i, r in enumerate(batch_results)]
    x = np.arange(len(labels))
    width = 0.35

    # Extract metrics
    dist_ol = [r["open_loop"]["final_distance"] for r in batch_results]
    dist_cl = [r["closed_loop"]["final_distance"] for r in batch_results]

    col_ol = [r["open_loop"]["collisions"] for r in batch_results]
    col_cl = [r["closed_loop"]["collisions"] for r in batch_results]

    len_ol = [r["open_loop"]["path_length"] for r in batch_results]
    len_cl = [r["closed_loop"]["path_length"] for r in batch_results]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # 1. Final distance from target center
    axs[0, 0].bar(x - width / 2, dist_ol, width, label="Open-Loop", color="tab:orange")
    axs[0, 0].bar(x + width / 2, dist_cl, width, label="Closed-Loop", color="tab:blue")
    axs[0, 0].axhline(0.08, color="red", linestyle=":", label="Stop Target (0.08m)")
    axs[0, 0].set_title("Final Distance to Target Center")
    axs[0, 0].set_ylabel("Distance [m]")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, axis="y", color="0.9")

    # 2. Number of collision events
    axs[0, 1].bar(x - width / 2, col_ol, width, label="Open-Loop", color="tab:orange")
    axs[0, 1].bar(x + width / 2, col_cl, width, label="Closed-Loop", color="tab:blue")
    axs[0, 1].set_title("Collision Events")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True, axis="y", color="0.9")

    # 3. Path length traveled
    axs[1, 0].bar(x - width / 2, len_ol, width, label="Open-Loop", color="tab:orange")
    axs[1, 0].bar(x + width / 2, len_cl, width, label="Closed-Loop", color="tab:blue")
    axs[1, 0].set_title("Trajectory Length")
    axs[1, 0].set_ylabel("Meters [m]")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, axis="y", color="0.9")

    # 4. Control input comparison v(t) and w(t) for trial 0
    if len(batch_results) > 0:
        res0_ol = batch_results[0]["open_loop"]
        res0_cl = batch_results[0]["closed_loop"]

        t_ol = np.arange(len(res0_ol["v_hist"])) * dt
        t_cl = np.arange(len(res0_cl["v_hist"])) * dt

        axs[1, 1].plot(t_ol, res0_ol["v_hist"], color="tab:orange", linestyle="--", label="v OL (turn-drive)")
        axs[1, 1].plot(t_cl, res0_cl["v_hist"], color="tab:blue", label="v CL (smooth APF)")
        axs[1, 1].set_title(f"Linear Velocity v(t) - {labels[0]}")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Velocity [m/s]")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, color="0.9")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics summary plot: {save_path}")


def plot_scenario3_maps(batch_results: list[dict], save_path: str = "results/plots/scenario_3_maps.png") -> None:
    """
    Plot 2D ExplorationGrid maps (UNKNOWN / FREE / traversed / OCCUPIED) overlaid with
    real obstacles, exploration/approach trajectory, and target detection pose.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(batch_results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    grid_cmap = ListedColormap(["#dcdcdc", "#eef9ee", "#c0392b"])  # UNKNOWN, FREE, OCCUPIED

    for idx, item in enumerate(batch_results):
        ax = axes[idx]
        res = item["result"]
        grid = res["grid"]
        q_hist = res["q_hist"]
        step_det = res["step_detected"]

        if grid.width > 0:
            extent = [grid.x_min, grid.x_max, grid.y_min, grid.y_max]
            ax.imshow(grid.grid, origin="lower", extent=extent, cmap=grid_cmap, vmin=0, vmax=2, interpolation="nearest")
            trav_overlay = np.ma.masked_where(~grid.traversed, np.ones_like(grid.grid))
            ax.imshow(trav_overlay, origin="lower", extent=extent, cmap=ListedColormap(["#b7e4c7"]), alpha=0.75, interpolation="nearest")

        # Draw ground-truth arena layout semi-transparently for map accuracy comparison
        draw_arena_2d(ax, item["obstacles"], item["objective"], item["q_start"])

        if step_det is not None and step_det < len(q_hist):
            ax.plot(q_hist[:step_det + 1, 0], q_hist[:step_det + 1, 1], color="tab:green", lw=1.8, label="Explore")
            ax.plot(q_hist[step_det:, 0], q_hist[step_det:, 1], color="tab:blue", lw=1.8, label="Approach")
            ax.plot(q_hist[step_det, 0], q_hist[step_det, 1], "gD", markersize=6, zorder=7, label="Detected")
        else:
            ax.plot(q_hist[:, 0], q_hist[:, 1], color="tab:green", lw=1.8, label="Explore")

        label = item.get("label", f"Trial {idx}")
        status = "OK" if res["reached"] else "FAIL"
        det_str = f"{res['time_detected']:.1f}s" if res["time_detected"] is not None else "Never"
        areas = res["mapped_areas"]
        ax.set_title(f"[{label}] {status} | Det: {det_str} | Free: {areas['free_m2']:.2f}m²", fontsize=8.5)
        ax.set_aspect("equal")
        xs = [obs["center"][0] for obs in item["obstacles"]] + [float(q_hist[:, 0].min()), float(q_hist[:, 0].max())]
        ys = [obs["center"][1] for obs in item["obstacles"]] + [float(q_hist[:, 1].min()), float(q_hist[:, 1].max())]
        ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
        ax.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
        ax.grid(True, color="0.85", linestyle=":")

        if idx == 0:
            legend_items = [
                Patch(facecolor="#dcdcdc", label="Unknown"),
                Patch(facecolor="#eef9ee", label="Free (FOV)"),
                Patch(facecolor="#b7e4c7", label="Free (Passed)"),
                Patch(facecolor="#c0392b", label="Occupied"),
            ]
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=legend_items + handles, loc="upper left", fontsize=6.5, ncol=2)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Scenario 3 map plot: {save_path}")


def plot_scenario3_metrics(batch_results: list[dict], dt: float = 0.05, save_path: str = "results/plots/scenario_3_metrics.png") -> None:
    """Save quantitative metrics charts for Scenario 3 trials."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = [r.get("label", str(i)) for i, r in enumerate(batch_results)]
    x = np.arange(len(labels))
    width = 0.35

    t_tot = [r["result"]["time"] for r in batch_results]
    t_det = [r["result"]["time_detected"] if r["result"]["time_detected"] is not None else r["result"]["time"] for r in batch_results]
    free_m2 = [r["result"]["mapped_areas"]["free_m2"] for r in batch_results]
    occ_m2 = [r["result"]["mapped_areas"]["occupied_m2"] for r in batch_results]
    lengths = [r["result"]["path_length"] for r in batch_results]
    cols = [r["result"]["collisions"] for r in batch_results]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    # 1. Detection vs Total Time
    axs[0, 0].bar(x - width / 2, t_det, width, label="Time to Detect", color="tab:green")
    axs[0, 0].bar(x + width / 2, t_tot, width, label="Total Time", color="tab:blue")
    axs[0, 0].set_title("Detection & Completion Time")
    axs[0, 0].set_ylabel("Time [s]")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels)
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, axis="y", color="0.9")

    # 2. Mapped Areas
    axs[0, 1].bar(x - width / 2, free_m2, width, label="Free [m²]", color="mediumseagreen")
    axs[0, 1].bar(x + width / 2, occ_m2, width, label="Occupied [m²]", color="indianred")
    axs[0, 1].set_title("Mapped Areas (ExplorationGrid)")
    axs[0, 1].set_ylabel("Area [m²]")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels)
    axs[0, 1].legend(fontsize=8)
    axs[0, 1].grid(True, axis="y", color="0.9")

    # 3. Path Length & Collisions
    bars = axs[1, 0].bar(x, lengths, width=0.45, color="tab:blue", label="Path Length")
    for bar, c_cnt in zip(bars, cols):
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"col={c_cnt}", ha="center", va="bottom", fontsize=7.5)
    axs[1, 0].set_title("Trajectory Length & Collisions")
    axs[1, 0].set_ylabel("Meters [m]")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels)
    axs[1, 0].grid(True, axis="y", color="0.9")

    # 4. Velocity profile for trial 0
    if len(batch_results) > 0:
        res0 = batch_results[0]["result"]
        t = np.arange(len(res0["v_hist"])) * dt
        axs[1, 1].plot(t, res0["v_hist"], color="tab:blue", label="v [m/s]")
        axs[1, 1].plot(t, res0["w_hist"], color="tab:orange", alpha=0.8, label="w [rad/s]")
        if res0["time_detected"] is not None:
            axs[1, 1].axvline(res0["time_detected"], color="tab:green", linestyle="--", label="Detected")
        axs[1, 1].set_title(f"Control Inputs - {labels[0]}")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, color="0.9")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Scenario 3 metrics plot: {save_path}")
