# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive ``init_pos``/``target`` flat-patch windows and task-space metadata from a resolved platform layout."""

from __future__ import annotations

import numpy as np

from isaaclab.terrains import FlatPatchSamplingCfg

from .composer_cfg import EdgeSpan, LayoutResolution, PlatformNode

# fraction of the platform half-extent kept as the patch window, so samples stay off the edge
_MARGIN_FRACTION = 0.6
_Z_MARGIN = 0.05
_MAX_HEIGHT_DIFF = 0.15


def _platforms_patch(
    nodes: list[PlatformNode], origin: np.ndarray, num_patches: int, patch_radius: float
) -> FlatPatchSamplingCfg:
    """A single patch window spanning the AABB of the given platforms' margin-shrunk footprints."""
    x_los, x_his, y_los, y_his, zs = [], [], [], [], []
    for node in nodes:
        rel = node.pose_center - origin
        half_x = node.platform.size[0] / 2.0 * _MARGIN_FRACTION
        half_y = node.platform.size[1] / 2.0 * _MARGIN_FRACTION
        x_los.append(rel[0] - half_x)
        x_his.append(rel[0] + half_x)
        y_los.append(rel[1] - half_y)
        y_his.append(rel[1] + half_y)
        zs.append(rel[2])
    return FlatPatchSamplingCfg(
        num_patches=num_patches,
        patch_radius=patch_radius,
        x_range=(float(min(x_los)), float(max(x_his))),
        y_range=(float(min(y_los)), float(max(y_his))),
        z_range=(float(min(zs) - _Z_MARGIN), float(max(zs) + _Z_MARGIN)),
        max_height_diff=_MAX_HEIGHT_DIFF,
    )


def _platform_patch(node: PlatformNode, origin: np.ndarray, num_patches: int, patch_radius: float) -> FlatPatchSamplingCfg:
    return _platforms_patch([node], origin, num_patches, patch_radius)


def _start_nodes(layout_res: LayoutResolution) -> list[PlatformNode]:
    return [node for node in layout_res.nodes if node.is_start]


def derive_flat_patches(
    layout_res: LayoutResolution, num_patches: int = 8, patch_radius: float = 0.3
) -> dict[str, FlatPatchSamplingCfg]:
    """Build the ``init_pos`` and ``target`` (or ``target_i`` for several goals) patch dict.

    ``init_pos`` spans every start platform, with ``num_patches`` scaled by their count.
    """
    assert layout_res.root is not None, "Layout must be resolved before deriving flat patches."
    origin = np.array([layout_res.root.pose_center[0], layout_res.root.pose_center[1], 0.0])

    starts = _start_nodes(layout_res)
    result: dict[str, FlatPatchSamplingCfg] = {
        "init_pos": _platforms_patch(starts, origin, num_patches * len(starts), patch_radius)
    }
    if len(layout_res.goals) == 1:
        result["target"] = _platform_patch(layout_res.goals[0], origin, num_patches, patch_radius)
    else:
        for i, goal in enumerate(layout_res.goals):
            result[f"target_{i}"] = _platform_patch(goal, origin, num_patches, patch_radius)
    return result


def derive_geometry_bounds(layout_res: LayoutResolution) -> dict[str, float]:
    """AABB of every platform footprint, the task-space source for any number of goals.

    XY is relative to the start hub and Z is absolute platform-top height. The ``start_*`` keys give
    the root's top height and the span of the start-platform centers.
    """
    assert layout_res.root is not None, "Layout must be resolved before deriving geometry bounds."
    origin = layout_res.root.pose_center

    xs: list[float] = []
    ys: list[float] = []
    tops: list[float] = []
    for node in layout_res.nodes:
        half_x, half_y = node.platform.size[0] / 2.0, node.platform.size[1] / 2.0
        xs += [node.center[0] - half_x, node.center[0] + half_x]
        ys += [node.center[1] - half_y, node.center[1] + half_y]
        tops.append(node.center[2])

    start_xs = [node.pose_center[0] - origin[0] for node in _start_nodes(layout_res)]
    start_ys = [node.pose_center[1] - origin[1] for node in _start_nodes(layout_res)]

    return {
        "x_lo": float(min(xs) - origin[0]),
        "x_hi": float(max(xs) - origin[0]),
        "y_lo": float(min(ys) - origin[1]),
        "y_hi": float(max(ys) - origin[1]),
        "z_lo": float(min(tops)),
        "z_hi": float(max(tops)),
        "start_z": float(origin[2]),
        "start_x_lo": float(min(start_xs)),
        "start_x_hi": float(max(start_xs)),
        "start_y_lo": float(min(start_ys)),
        "start_y_hi": float(max(start_ys)),
    }


def span_rect(edge: EdgeSpan, origin: np.ndarray) -> tuple[float, float, float, float]:
    """Env-relative ``(x_lo, x_hi, y_lo, y_hi)`` AABB of one bridge span's ``length x width`` envelope.

    Mirrors ``composer.platform_bridge_terrain``'s placement math exactly; exact for axis-aligned spokes.
    """
    direction = np.array([np.cos(edge.angle), np.sin(edge.angle)])
    normal = np.array([-direction[1], direction[0]])
    entry_xy = edge.parent.center[:2] + direction * (edge.parent.platform.size[0] / 2.0)
    half_w = edge.bridge.width / 2.0
    corners = np.stack([
        entry_xy + direction * along + normal * across for along in (0.0, edge.length) for across in (-half_w, half_w)
    ])
    return (
        float(corners[:, 0].min() - origin[0]),
        float(corners[:, 0].max() - origin[0]),
        float(corners[:, 1].min() - origin[1]),
        float(corners[:, 1].max() - origin[1]),
    )


def derive_keep_in_regions(layout_res: LayoutResolution) -> list[tuple[float, float, float, float]]:
    """Env-relative XY ``(x_lo, x_hi, y_lo, y_hi)`` rectangles covering every walkable footprint.

    One per platform and one per bridge span. Spans count as solid and rotated spans use their AABB,
    so the approximation only ever keeps too much; Z is left to the task space's own bounds.
    """
    assert layout_res.root is not None, "Layout must be resolved before deriving keep-in regions."
    origin = layout_res.root.pose_center

    regions: list[tuple[float, float, float, float]] = []

    for node in layout_res.nodes:
        half_x, half_y = node.platform.size[0] / 2.0, node.platform.size[1] / 2.0
        regions.append((
            float(node.center[0] - half_x - origin[0]),
            float(node.center[0] + half_x - origin[0]),
            float(node.center[1] - half_y - origin[1]),
            float(node.center[1] + half_y - origin[1]),
        ))

    for edge in layout_res.edges:
        regions.append(span_rect(edge, origin))

    return regions


def goal_pose_metadata(layout_res: LayoutResolution) -> dict[str, tuple[float, float, float]]:
    """Absolute tile-local ``{name: (x, y, z)}`` poses for the start and every goal."""
    assert layout_res.root is not None, "Layout must be resolved before deriving pose metadata."
    poses = {"start": tuple(float(c) for c in layout_res.root.pose_center)}
    for goal in layout_res.goals:
        poses[goal.name] = tuple(float(c) for c in goal.pose_center)
    return poses
