"""
Plotting and performance analysis module.
Generates 2D arena reconstructions and comparative metric charts in outputs/figures.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


def draw_arena_2d(
    ax, obstacles: list[dict], objective: dict, q_start: np.ndarray | None = None
) -> None:
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
        ax.arrow(
            q_start[0], q_start[1], dx, dy, head_width=0.03, color="black", zorder=6
        )


def plot_trajectories_2d(
    batch_results: list[dict], save_path: str = "outputs/figures/trajectories.png"
) -> None:
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
            ax.plot(
                wp[:, 0],
                wp[:, 1],
                "g--o",
                markersize=3,
                lw=1.0,
                alpha=0.5,
                label="A* Plan",
            )

        # 3. Draw executed trajectories
        q_ol = res_ol["q_hist"]
        q_cl = res_cl["q_hist"]
        ax.plot(
            q_ol[:, 0],
            q_ol[:, 1],
            color="tab:orange",
            lw=1.8,
            linestyle="--",
            label="Open-Loop",
        )
        ax.plot(
            q_cl[:, 0],
            q_cl[:, 1],
            color="tab:blue",
            lw=1.8,
            linestyle="-",
            label="Closed-Loop",
        )

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


def plot_metrics_comparison(
    batch_results: list[dict],
    dt: float = 0.05,
    save_path: str = "outputs/figures/metrics.png",
) -> None:
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
    axs[0, 0].axhline(0.09, color="red", linestyle=":", label="Stop Target (0.09m)")
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

        axs[1, 1].plot(
            t_ol,
            res0_ol["v_hist"],
            color="tab:orange",
            linestyle="--",
            label="v OL (turn-drive)",
        )
        axs[1, 1].plot(
            t_cl, res0_cl["v_hist"], color="tab:blue", label="v CL (smooth APF)"
        )
        axs[1, 1].set_title(f"Linear Velocity v(t) - {labels[0]}")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Velocity [m/s]")
        axs[1, 1].legend(fontsize=8)
        axs[1, 1].grid(True, color="0.9")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics summary plot: {save_path}")
