# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Figures for a composed mesh-scan tile, free of Isaac Lab imports so they render without the runtime.

They catch a fence that misses the rock, an apron coplanar with the floor, or a pose the task space cannot reach.
"""

from __future__ import annotations

import numpy as np
import trimesh
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withStroke
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

STANDING_HEIGHT = 0.6
"""Nominal base height above the surface, matching the task space's ``height_offset``."""

ROBOT_FOOTPRINT = (1.0, 0.5)
"""Rough ANYmal footprint (length, width in m), drawn at the start and goal for scale."""


def _box_polys(mesh: trimesh.Trimesh, color: str, alpha: float) -> Poly3DCollection:
    return Poly3DCollection(mesh.vertices[mesh.faces], facecolor=color, edgecolor="none", alpha=alpha)


def _pose_marker(ax, xy, z: float, yaw: float, color: str) -> None:
    """Draw a footprint outline, standing-height pole and heading arrow.

    Uses ``ax.plot`` because Matplotlib does not depth-sort 3D collections against ``plot_trisurf``.
    """
    length, width = ROBOT_FOOTPRINT
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * [length / 2.0, width / 2.0]
    outline = corners @ np.array([[cos_y, sin_y], [-sin_y, cos_y]]) + xy
    ax.plot(outline[:, 0], outline[:, 1], np.full(len(outline), z + 0.01), color=color, lw=2.5)
    ax.plot([xy[0], xy[0]], [xy[1], xy[1]], [z, z + STANDING_HEIGHT], color=color, lw=3.5)
    ax.plot([xy[0]], [xy[1]], [z + STANDING_HEIGHT], marker="o", color=color, mec="black", ms=8)
    ax.plot(
        [xy[0], xy[0] + 0.6 * cos_y],
        [xy[1], xy[1] + 0.6 * sin_y],
        [z + STANDING_HEIGHT] * 2,
        color=color,
        lw=2.5,
        ls="--",
    )


def _extra_marker(ax, xy, z: float, color: str, label: str) -> None:
    """Draw a pairs-eval pose as a bare labelled pole, quieter than the measured pair."""
    ax.plot([xy[0], xy[0]], [xy[1], xy[1]], [z, z + STANDING_HEIGHT], color=color, lw=2.0)
    ax.plot([xy[0]], [xy[1]], [z + STANDING_HEIGHT], marker="o", color=color, mec="black", mew=0.6, ms=5.5)
    # stroked so the label stays readable over both the terrain colormap and the grey fence
    ax.text(
        xy[0], xy[1], z + STANDING_HEIGHT + 0.14, label,
        fontsize=6.5, color="white", ha="center",
        path_effects=[withStroke(linewidth=1.6, foreground="black")],
    )


def extra_poses(cfg) -> tuple[list, list]:
    """Return ``(starts, goals)``, each a list of ``(key, (x, y, z))`` in tile coordinates.

    Tolerates cfgs without extras, and sorts by the key's numeric index so a tenth pose does not relabel the set.
    """
    published = getattr(cfg, "extra_poses", None) or {}

    def by_index(prefix):
        rows = [(k, v) for k, v in published.items() if k.startswith(prefix)]
        return sorted(rows, key=lambda kv: int(kv[0].rpartition("_")[2]))

    return by_index("start"), by_index("goal")


def _extra_label(key: str) -> str:
    """Shorten ``start_3`` to ``s3``, matching the pairs-eval labels."""
    kind, _, index = key.partition("_")
    return f"{kind[0]}{index}"


def _draw_3d(ax, cfg, meshes, elev: float, azim: float, title: str, zoom: bool, show_floor: bool) -> None:
    floor, sheet, *walls = meshes
    vertices, faces = sheet.vertices, sheet.faces
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], cmap="terrain", linewidth=0, antialiased=False)
    for wall in walls:
        ax.add_collection3d(_box_polys(wall, "#6d6d78", 0.5))
    if show_floor:
        ax.add_collection3d(_box_polys(floor, "#3a3a42", 0.3))

    start, goal = cfg.goal_poses["start"], cfg.goal_poses["goal"]
    extra_starts, extra_goals = extra_poses(cfg)
    for key, pose in extra_starts:
        _extra_marker(ax, np.asarray(pose[:2]), pose[2], "limegreen", _extra_label(key))
    for key, pose in extra_goals:
        _extra_marker(ax, np.asarray(pose[:2]), pose[2], "gold", _extra_label(key))
    # the measured pair last, so its heavier marker draws over any extra standing near it
    _pose_marker(ax, np.asarray(start[:2]), start[2], cfg.heading, "limegreen")
    _pose_marker(ax, np.asarray(goal[:2]), goal[2], cfg.heading, "gold")

    rect = cfg.footprint_rect_tile
    if zoom:
        pad = 0.9
        limits = (rect[0] - pad, rect[1] + pad, rect[2] - pad, rect[3] + pad)
    else:
        limits = (0.0, cfg.size[0], 0.0, cfg.size[1])
    z_top = max(2.1, (cfg.perimeter_wall.height if cfg.perimeter_wall else 0.0) + 0.1)
    ax.set_xlim3d(limits[0], limits[1])
    ax.set_ylim3d(limits[2], limits[3])
    ax.set_zlim3d(-0.15, z_top)
    ax.set_box_aspect((limits[1] - limits[0], limits[3] - limits[2], z_top + 0.15))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [m] (tile)", fontsize=7)
    ax.set_ylabel("y [m] (tile)", fontsize=7)
    ax.tick_params(labelsize=6)


def _distance_to_tile_edge(start: np.ndarray, direction: np.ndarray, size) -> float:
    """Return how far a ray from ``start`` runs along ``direction`` before it leaves the tile.

    Zero components are skipped explicitly, since aligned poses are common and a divide warning would be noise.
    """
    limits = [
        (start[axis] / -component) if component < 0.0 else ((size[axis] - start[axis]) / component)
        for axis, component in enumerate(direction[:2])
        if abs(component) > 1e-12
    ]
    return min(limits)  # `direction` is a unit vector, so at least one component is non-zero


def _section(meshes, p0: np.ndarray, direction: np.ndarray, length: float):
    """Return every crossing of the tile mesh by the vertical plane through ``p0`` along ``direction``."""
    combined = trimesh.util.concatenate(meshes)
    normal = np.array([-direction[1], direction[0], 0.0])
    segments = trimesh.intersections.mesh_plane(combined, plane_normal=normal, plane_origin=np.append(p0, 0.0))
    if len(segments) == 0:
        return np.zeros(0), np.zeros(0)
    points = segments.reshape(-1, 3)
    along = (points[:, :2] - p0) @ direction
    keep = (along >= 0.0) & (along <= length)
    return along[keep], points[keep, 2]


def _draw_section(ax, cfg, meshes, p0, direction, length, window, title) -> None:
    along, heights = _section(meshes, p0, direction, length)
    ax.plot(along, heights, ".", ms=1.6, color="#2b6cb0")

    start, goal = cfg.goal_poses["start"], cfg.goal_poses["goal"]
    for pose, color, label in ((start, "limegreen", "start"), (goal, "gold", "goal")):
        at = float(np.dot(np.asarray(pose[:2]) - p0, direction))
        ax.plot([at, at], [pose[2], pose[2] + STANDING_HEIGHT], color=color, lw=3)
        ax.plot([at], [pose[2]], "o", color=color, mec="black", ms=9, label=label)
    ax.axhline(0.0, color="black", lw=0.9, ls="--", label="tile floor")
    ax.axhline(cfg.surface_lift, color="#b7791f", lw=0.9, ls=":", label="lowest rock")
    ax.set_xlim(*window)
    ax.set_xlabel("distance along the section [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def _draw_sampling(ax, cfg, meshes, task_bounds) -> None:
    """Draw a top-down view of the curriculum's sampling region against the fence and the scan rim.

    Args:
        task_bounds: The env cfg's env-relative ``(bounds, eval_bounds)`` x/y pairs, or ``None``.
    """
    sheet = meshes[1]
    vertices = sheet.vertices
    ax.tricontourf(vertices[:, 0], vertices[:, 1], sheet.faces, vertices[:, 2], levels=24, cmap="terrain")

    rect = cfg.footprint_rect_tile
    start = np.asarray(cfg.goal_poses["start"][:2])
    rects = [((rect[0], rect[1]), (rect[2], rect[3]), "black", "scan footprint", 1.6)]
    if cfg.perimeter_wall is not None:
        margin_x, margin_y = cfg.perimeter_wall.margin
        rects.append((
            (rect[0] - margin_x, rect[1] + margin_x),
            (rect[2] - margin_y, rect[3] + margin_y),
            "#c53030", "fence inner faces", 2.0,
        ))
    if task_bounds is not None:
        (bx, by), (ex, ey) = task_bounds
        rects.append(((bx[0] + start[0], bx[1] + start[0]), (by[0] + start[1], by[1] + start[1]),
                      "white", "task-space bounds", 2.2))
        rects.append(((ex[0] + start[0], ex[1] + start[0]), (ey[0] + start[1], ey[1] + start[1]),
                      "limegreen", "initial-state window", 2.2))
    for (x_lo, x_hi), (y_lo, y_hi), color, label, lw in rects:
        ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, fill=False, ec=color, lw=lw, label=label))

    goal = np.asarray(cfg.goal_poses["goal"][:2])

    # pairs-eval extras joined to their canonical partner, the fan eval_scan_pairs.py runs
    extra_starts, extra_goals = extra_poses(cfg)
    for key, pose in extra_starts:
        ax.annotate("", xy=goal, xytext=pose[:2], arrowprops=dict(arrowstyle="->", color="white", lw=0.8, alpha=0.5))
    for key, pose in extra_goals:
        ax.annotate("", xy=pose[:2], xytext=start, arrowprops=dict(arrowstyle="->", color="white", lw=0.8, alpha=0.5))
    for key, pose in extra_starts:
        ax.plot(pose[0], pose[1], "o", color="limegreen", mec="black", mew=0.6, ms=5, alpha=0.9)
    for key, pose in extra_goals:
        ax.plot(pose[0], pose[1], "*", color="gold", mec="black", mew=0.6, ms=9, alpha=0.9)
    for key, pose in extra_starts + extra_goals:
        ax.annotate(
            _extra_label(key), xy=pose[:2], xytext=(5, 4), textcoords="offset points",
            fontsize=6.5, color="white",
            path_effects=[withStroke(linewidth=1.6, foreground="black")],
        )
    if extra_starts or extra_goals:
        ax.plot([], [], "o", color="limegreen", mec="black", ms=5,
                label=f"pairs eval: {len(extra_starts)} extra start(s), {len(extra_goals)} goal(s)")

    ax.plot(*start, "o", color="limegreen", mec="black", ms=10)
    ax.plot(*goal, "*", color="gold", mec="black", ms=17)
    ax.annotate("", xy=goal, xytext=start, arrowprops=dict(arrowstyle="->", color="white", lw=2))
    # label both frames, since tile and scan coordinates differ by `rot90_k`, which changes when poses move
    for xy, obj_xy, color in ((start, cfg.start_xy_obj, "limegreen"), (goal, cfg.goal_xy_obj, "gold")):
        ax.annotate(
            f"tile ({xy[0]:.2f}, {xy[1]:.2f})\nscan ({obj_xy[0]:.2f}, {obj_xy[1]:.2f})",
            xy=xy, xytext=(8, 8), textcoords="offset points", fontsize=7, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="black", alpha=0.85),
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (tile)")
    ax.set_ylabel("y [m] (tile)")
    ax.set_title("where the curriculum may sample, vs fence and rim", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def tile_figure(cfg, meshes, task_bounds=None, name: str = "") -> Figure:
    """Six panels of the composed tile: three views, two sections and the sampling footprint."""
    fig = Figure(figsize=(19.5, 11.5))
    _draw_3d(fig.add_subplot(2, 3, 1, projection="3d"), cfg, meshes, 38, -55,
             "full tile: floor plane + scan sheet + arena fence", zoom=False, show_floor=True)
    _draw_3d(fig.add_subplot(2, 3, 2, projection="3d"), cfg, meshes, 18, -118,
             "fenced arena, low angle", zoom=True, show_floor=False)
    _draw_3d(fig.add_subplot(2, 3, 3, projection="3d"), cfg, meshes, 80, -90,
             "near top-down", zoom=True, show_floor=False)

    start = np.asarray(cfg.goal_poses["start"][:2])
    goal = np.asarray(cfg.goal_poses["goal"][:2])
    direction = goal - start
    direction = direction / np.linalg.norm(direction)
    # walk back from the start to the tile edge, so both fences are in frame
    p0 = start - direction * min(_distance_to_tile_edge(start, direction, cfg.size), 3.0)
    length = float(np.hypot(*(np.array(cfg.size))))

    rect = cfg.footprint_rect_tile
    # slab-clip the section against the scan footprint to find where it enters the rim
    spans = []
    for lo, hi, axis in ((rect[0], rect[1], 0), (rect[2], rect[3], 1)):
        if abs(direction[axis]) < 1e-9:
            spans.append((-np.inf, np.inf))
            continue
        t_lo = (lo - p0[axis]) / direction[axis]
        t_hi = (hi - p0[axis]) / direction[axis]
        spans.append((min(t_lo, t_hi), max(t_lo, t_hi)))
    rim = max(spans[0][0], spans[1][0])

    _draw_section(fig.add_subplot(2, 3, 4), cfg, meshes, p0, direction, length, (0.0, length),
                  "section along start->goal: fence, apron, rock")
    _draw_section(fig.add_subplot(2, 3, 5), cfg, meshes, p0, direction, length, (rim - 1.0, rim + 1.0),
                  "rim detail: fence, apron and the scan's edge")
    _draw_sampling(fig.add_subplot(2, 3, 6), cfg, meshes, task_bounds)

    triangles = sum(len(m.faces) for m in meshes)
    fig.suptitle(
        f"{name}: composed tile -- {cfg.size[0]} x {cfg.size[1]} m, rot90_k={cfg.resolved_rot90_k},"
        f" {triangles:,} triangles/tile",
        fontsize=13,
    )
    fig.tight_layout()
    return fig
