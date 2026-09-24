# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless previewer for the named layout terrains in :data:`~..config.layouts.LAYOUTS`.

Renders a top-down schematic and a 3D mesh per layout at its pinned difficulty, optionally with a GLB.
"""

import argparse

parser = argparse.ArgumentParser(description="Preview named layout terrains without launching a full environment.")
parser.add_argument("--headless", action="store_true", default=False, help="Don't open a viewer window.")
parser.add_argument("--layout", nargs="*", default=None, help="Layout names to render (default: all in LAYOUTS).")
parser.add_argument("--out", type=str, default=None, help="Output directory (default: ./output/layouts next to this script).")
parser.add_argument("--no-glb", action="store_true", default=False, help="Skip GLB export (PNG only).")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401 (registers the 3d projection)

from isaaclab.terrains.utils import color_meshes_by_height  # noqa: E402

from informed_exploration.terrains.bridge import resolve_layout  # noqa: E402
from informed_exploration.terrains.bridge.bridges import GapBridgeCfg  # noqa: E402
from informed_exploration.terrains.bridge.composer_cfg import LayoutResolution  # noqa: E402
from informed_exploration.terrains.bridge.patches import (  # noqa: E402
    derive_geometry_bounds,
    derive_keep_in_regions,
)
from informed_exploration.terrains.config.layouts import LAYOUTS  # noqa: E402

# rough AnymalD-scale reference box (length, width, height in m) for reading difficulty at a glance
ROBOT_REFERENCE_SIZE = (1.0, 0.5, 0.6)


def _reference_box(origin: np.ndarray) -> trimesh.Trimesh:
    box = trimesh.creation.box(ROBOT_REFERENCE_SIZE)
    box.apply_translation(origin + np.array([0.0, 0.0, ROBOT_REFERENCE_SIZE[2] / 2.0]))
    box.visual.vertex_colors = (0, 0, 0, 200)
    return box


START_MARKER_COLOR = (50, 205, 50, 255)  # limegreen, matching the top-down start platforms
GOAL_MARKER_COLOR = (255, 200, 0, 255)  # gold, matching the top-down goal platforms


def _pose_marker(pos: np.ndarray, color: tuple[int, int, int, int]) -> list[trimesh.Trimesh]:
    """Build a colored map-pin on a start or goal pose point, which need not be the platform center."""
    pole = trimesh.creation.cylinder(radius=0.05, height=0.8, sections=12)
    pole.apply_translation([pos[0], pos[1], pos[2] + 0.4])
    head = trimesh.creation.icosphere(radius=0.18, subdivisions=2)
    head.apply_translation([pos[0], pos[1], pos[2] + 0.8])
    for mesh in (pole, head):
        mesh.visual.vertex_colors = color
    return [pole, head]


def _pose_points(layout_res: LayoutResolution) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return the start and goal pose points in absolute tile coordinates."""
    starts = [node.pose_center for node in layout_res.nodes if node.is_start]
    goals = [goal.pose_center for goal in layout_res.goals]
    return starts, goals


def _feature_meshes(meshes: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Drop the full-tile ground plane the composer always returns as ``meshes[0]``."""
    return meshes[1:] if len(meshes) > 1 else meshes


def _plot_3d(
    ax,
    meshes: list[trimesh.Trimesh],
    origin: np.ndarray,
    title: str,
    margin: float = 1.0,
    start_poses: list[np.ndarray] | None = None,
    goal_poses: list[np.ndarray] | None = None,
) -> None:
    combined = trimesh.util.concatenate(meshes)
    verts, faces = combined.vertices, combined.faces
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="turbo", linewidth=0.05, antialiased=True)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="lime", s=40, depthshade=False)
    for poses, color in ((start_poses, "lime"), (goal_poses, "gold")):
        for pos in poses or []:
            ax.scatter([pos[0]], [pos[1]], [pos[2] + 0.2], color=color, marker="*", s=140, depthshade=False)

    x_min, x_max = verts[:, 0].min() - margin, verts[:, 0].max() + margin
    y_min, y_max = verts[:, 1].min() - margin, verts[:, 1].max() + margin
    z_min, z_max = verts[:, 2].min() - margin, verts[:, 2].max() + margin
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, max(z_max - z_min, 0.5)))

    ax.view_init(elev=32, azim=-60)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x", fontsize=7)
    ax.set_ylabel("y", fontsize=7)
    ax.tick_params(labelsize=6)


def _plot_topdown(ax, layout_res: LayoutResolution, title: str) -> None:
    """Draw an annotated xy schematic of platforms, bridge edges and pose points.

    Also overlays the derived task-space AABB (dashed) and keep-in regions (hatched), so open air a uniform
    sample could land in is visible.
    """
    # the derived regions are env-origin-relative, and the env origin is the root's POSE point
    origin = layout_res.root.pose_center

    bounds = derive_geometry_bounds(layout_res)
    ax.add_patch(
        Rectangle(
            (bounds["x_lo"] + origin[0], bounds["y_lo"] + origin[1]),
            bounds["x_hi"] - bounds["x_lo"],
            bounds["y_hi"] - bounds["y_lo"],
            facecolor="none",
            edgecolor="darkorange",
            linewidth=1.4,
            linestyle="--",
            zorder=4,
            label="task-space AABB",
        )
    )
    for i, (x_lo, x_hi, y_lo, y_hi) in enumerate(derive_keep_in_regions(layout_res)):
        ax.add_patch(
            Rectangle(
                (x_lo + origin[0], y_lo + origin[1]),
                x_hi - x_lo,
                y_hi - y_lo,
                facecolor="none",
                edgecolor="darkorange",
                hatch="///",
                alpha=0.35,
                linewidth=0.0,
                zorder=0,
                label="keep-in region" if i == 0 else None,
            )
        )

    goal_names = {g.name for g in layout_res.goals}
    goal_xy = [g.center[:2] for g in layout_res.goals]
    parents_with_outgoing_edge = {edge.parent.name for edge in layout_res.edges}

    for node in layout_res.nodes:
        is_start = node.is_start
        is_goal = node.name in goal_names
        is_dead_end = not is_start and not is_goal and node.name not in parents_with_outgoing_edge

        # skip cycle-closing terminal hubs coincident with a goal, which would paint over the goal pad
        if is_dead_end and any(np.allclose(node.center[:2], xy) for xy in goal_xy):
            continue

        if is_start:
            color, label = "limegreen", "start" if node.parent_name is None else node.name
        elif is_goal:
            color, label = "gold", node.name
        elif is_dead_end:
            color, label = "lightgray", "dead end"
        else:
            color, label = "lightsteelblue", node.name

        half_x, half_y = node.platform.size[0] / 2.0, node.platform.size[1] / 2.0
        ax.add_patch(
            Rectangle(
                (node.center[0] - half_x, node.center[1] - half_y),
                2 * half_x,
                2 * half_y,
                facecolor=color,
                edgecolor="black",
                linewidth=1.2,
                zorder=2,
            )
        )
        ax.annotate(label, (node.center[0], node.center[1]), ha="center", va="center", fontsize=8, zorder=3)

    for edge in layout_res.edges:
        p0, p1 = edge.parent.center[:2], edge.child.center[:2]
        is_gap = isinstance(edge.bridge, GapBridgeCfg)
        color = "crimson" if is_gap else "steelblue"
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-",
                color=color,
                linewidth=2.5 if is_gap else 2.0,
                linestyle="--" if is_gap else "-",
                zorder=1,
            )
        )
        mid = (p0 + p1) / 2.0
        bridge_name = type(edge.bridge).__name__.replace("BridgeCfg", "")
        text = f"{bridge_name}\nINFEASIBLE ({edge.bridge.gap_width_range[0]:.1f} m)" if is_gap else bridge_name
        ax.annotate(
            text,
            mid,
            ha="center",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold" if is_gap else "normal",
            zorder=3,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
        )

    start_poses, goal_poses = _pose_points(layout_res)
    for poses, color, label in ((start_poses, "green", "start pose"), (goal_poses, "darkgoldenrod", "goal pose")):
        for i, pos in enumerate(poses):
            ax.plot(
                pos[0],
                pos[1],
                marker="*",
                markersize=14,
                color=color,
                markeredgecolor="black",
                linestyle="none",
                zorder=5,
                label=label if i == 0 else None,
            )

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    all_x = [n.center[0] for n in layout_res.nodes]
    all_y = [n.center[1] for n in layout_res.nodes]
    margin = 2.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)


def main() -> None:
    out_dir = args_cli.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "layouts")
    os.makedirs(out_dir, exist_ok=True)

    names = args_cli.layout if args_cli.layout else list(LAYOUTS.keys())

    for name in names:
        if name not in LAYOUTS:
            print(f"[WARN] Unknown layout '{name}', skipping. Available: {list(LAYOUTS.keys())}")
            continue

        gen_cfg = LAYOUTS[name]()
        difficulty = gen_cfg.difficulty_range[0]
        sub_names = list(gen_cfg.sub_terrains.keys())

        fig, axes = plt.subplots(
            len(sub_names),
            2,
            figsize=(14.0, 6.5 * len(sub_names)),
            squeeze=False,
        )
        # right column needs a 3d projection, so swap in fresh 3d axes per row
        for row, sub_name in enumerate(sub_names):
            axes[row][1].remove()
            axes[row][1] = fig.add_subplot(len(sub_names), 2, row * 2 + 2, projection="3d")

        for row, sub_name in enumerate(sub_names):
            sub_cfg = gen_cfg.sub_terrains[sub_name]
            layout_res = resolve_layout(sub_cfg.layout, sub_cfg.bridge, sub_cfg.size, sub_cfg.cross_links)
            meshes, origin = sub_cfg.function(difficulty=difficulty, cfg=sub_cfg)
            feature_meshes = _feature_meshes(meshes)
            start_poses, goal_poses = _pose_points(layout_res)

            _plot_topdown(axes[row][0], layout_res, f"{sub_name} top-down (d={difficulty:.2f})")
            _plot_3d(
                axes[row][1],
                feature_meshes,
                origin,
                f"{sub_name} 3D (d={difficulty:.2f})",
                start_poses=start_poses,
                goal_poses=goal_poses,
            )

            if not args_cli.no_glb:
                scene_meshes = list(feature_meshes) + [_reference_box(origin)]
                colored = color_meshes_by_height(scene_meshes)
                scene = trimesh.Scene(colored)
                # pose pins go in AFTER height-coloring so they keep their start/goal colors
                for pos in start_poses:
                    for marker in _pose_marker(pos, START_MARKER_COLOR):
                        scene.add_geometry(marker)
                for pos in goal_poses:
                    for marker in _pose_marker(pos, GOAL_MARKER_COLOR):
                        scene.add_geometry(marker)
                glb_path = os.path.join(out_dir, f"{name}_{sub_name}_d{difficulty:.2f}.glb")
                scene.export(glb_path)

            print(f"[{name}] {sub_name}: goal_poses={sub_cfg.goal_poses}")

        fig.suptitle(f"Layout: {name}", fontsize=13)
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[{name}] wrote {png_path}" + ("" if args_cli.no_glb else f" + {len(sub_names)} GLB(s)"))

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
