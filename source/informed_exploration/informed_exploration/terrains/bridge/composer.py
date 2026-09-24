# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The platform and pluggable-bridge composer, the sub-terrain generator every bridge preset uses."""

from __future__ import annotations

import warnings
from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import geometry
from . import patches as patch_utils
from .composer_cfg import LayoutResolution, PerimeterWallCfg, RadialLayoutCfg, WallCfg, resolve_layout
from .interface import BridgeCfg


def perimeter_wall_meshes_for_rect(
    cfg: PerimeterWallCfg, rect: tuple[float, float, float, float], size: tuple[float, float]
) -> list[trimesh.Trimesh]:
    """Four arena walls around an ``(x_lo, x_hi, y_lo, y_hi)`` rect, in tile coordinates.

    ``cfg.margin`` grows the rect per axis; a negative margin tucks the fence inside the footprint.
    """
    x_lo, x_hi, y_lo, y_hi = rect
    margin_x, margin_y = cfg.margin
    x_lo, x_hi = x_lo - margin_x, x_hi + margin_x
    y_lo, y_hi = y_lo - margin_y, y_hi + margin_y
    t = cfg.thickness

    if x_lo - t < 0.0 or x_hi + t > size[0] or y_lo - t < 0.0 or y_hi + t > size[1]:
        warnings.warn(
            f"perimeter wall (inner faces x [{x_lo:.2f}, {x_hi:.2f}], "
            f"y [{y_lo:.2f}, {y_hi:.2f}], thickness {t}) extends outside the tile bounds {size}. "
            "Shrink `margin`/`thickness` or grow the tile.",
            stacklevel=2,
        )

    # side walls span the full outer extent (covering the corners); top/bottom span between them
    x_span = x_hi - x_lo + 2.0 * t
    x_mid, y_mid = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    return [
        geometry.wall_box((x_lo - t / 2.0, y_mid), (t, y_hi - y_lo), cfg.height),
        geometry.wall_box((x_hi + t / 2.0, y_mid), (t, y_hi - y_lo), cfg.height),
        geometry.wall_box((x_mid, y_lo - t / 2.0), (x_span, t), cfg.height),
        geometry.wall_box((x_mid, y_hi + t / 2.0), (x_span, t), cfg.height),
    ]


def _perimeter_wall_meshes(
    cfg: PerimeterWallCfg, layout_res: LayoutResolution, size: tuple[float, float]
) -> list[trimesh.Trimesh]:
    """Four arena walls around the platform-footprint AABB (tile coordinates)."""
    xs: list[float] = []
    ys: list[float] = []
    for node in layout_res.nodes:
        half_x, half_y = node.platform.size[0] / 2.0, node.platform.size[1] / 2.0
        xs += [node.center[0] - half_x, node.center[0] + half_x]
        ys += [node.center[1] - half_y, node.center[1] + half_y]
    return perimeter_wall_meshes_for_rect(cfg, (min(xs), max(xs), min(ys), max(ys)), size)


def platform_bridge_terrain(difficulty: float, cfg: PlatformBridgeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a start-platform, bridge(s), goal-platform(s) terrain.

    Platform positions are difficulty-independent; only each bridge's interior geometry depends on ``difficulty``.
    """
    layout_res = resolve_layout(cfg.layout, cfg.bridge, cfg.size, cfg.cross_links)

    # One seeded stream for all bridges; inside a TerrainGenerator `cfg.seed` is the generator's seed.
    rng = np.random.default_rng(getattr(cfg, "seed", None))

    meshes: list[trimesh.Trimesh] = [
        geometry.plane(cfg.size, 0.0, center_xy=(cfg.size[0] / 2.0, cfg.size[1] / 2.0))
    ]

    for node in layout_res.nodes:
        # every platform is a solid pillar reaching true ground
        bottom_z = -geometry.GROUND_EMBED_DEPTH
        meshes.append(
            geometry.box(
                node.platform.size[0],
                node.platform.size[1],
                node.center[2] - bottom_z,
                center=(node.center[0], node.center[1], (node.center[2] + bottom_z) / 2.0),
            )
        )

    for edge in layout_res.edges:
        direction = np.array([np.cos(edge.angle), np.sin(edge.angle)])
        entry_xy = edge.parent.center[:2] + direction * (edge.parent.platform.size[0] / 2.0)

        local_meshes = edge.bridge.build(
            difficulty=difficulty,
            length=edge.length,
            width=edge.bridge.width,
            z_entry=edge.parent.center[2],
            z_exit=edge.child.center[2],
            rng=rng,
        )
        transform = trimesh.transformations.rotation_matrix(edge.angle, [0.0, 0.0, 1.0])
        transform[0:2, -1] = entry_xy
        for mesh in local_meshes:
            mesh.apply_transform(transform)
        meshes.extend(local_meshes)

    # env origin is the root's pose point, so env-relative x=y=0 is the start pose even when offset on its pad
    origin = np.array([layout_res.root.pose_center[0], layout_res.root.pose_center[1], 0.0])

    # interior separators and the arena fence are mesh-only, invisible to the derived task-space metadata
    for wall in cfg.walls:
        meshes.append(
            geometry.wall_box((origin[0] + wall.center[0], origin[1] + wall.center[1]), wall.size, wall.height)
        )
    if cfg.perimeter_wall is not None:
        meshes += _perimeter_wall_meshes(cfg.perimeter_wall, layout_res, cfg.size)

    return meshes, origin


@configclass
class PlatformBridgeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a :func:`platform_bridge_terrain` sub-terrain.

    .. important::
        :attr:`size` is read in :meth:`__post_init__`, so always pass it explicitly, matching
        ``TerrainGeneratorCfg.size``.
    """

    function = platform_bridge_terrain

    layout: RadialLayoutCfg = MISSING
    """The root (start) hub of the recursive platform graph."""

    bridge: BridgeCfg = MISSING
    """Default bridge used for any spoke that doesn't specify its own."""

    cross_links: bool = False
    """If True, add best-effort straight-edge bridges between sibling hubs (see :func:`resolve_layout`)."""

    seed: int | None = 0
    """Seed for the bridges' stochastic geometry; overwritten by ``TerrainGeneratorCfg.seed`` inside a generator.

    ``None`` gives non-reproducible geometry."""

    walls: list[WallCfg] = []
    """Interior wall boxes (root-relative), e.g. separators between parallel routes. Mesh-only."""

    perimeter_wall: PerimeterWallCfg | None = None
    """If set, four arena walls enclose the layout's platform-footprint AABB. Mesh-only."""

    patch_override: dict[str, FlatPatchSamplingCfg] | None = None
    """If set, used verbatim instead of the auto-derived ``init_pos``/``target`` patches."""

    patch_num_samples: int = 8
    """Number of samples per auto-derived flat patch."""

    patch_radius: float = 0.3
    """Radius (in m) of each auto-derived flat patch."""

    # Derived in __post_init__ from the resolved layout; never set by hand

    goal_poses: dict[str, tuple[float, float, float]] | None = None
    """Start + goal platform positions; see :func:`~.patches.goal_pose_metadata`."""

    geometry_bounds: dict[str, float] | None = None
    """AABB of every platform footprint; see :func:`~.patches.derive_geometry_bounds`."""

    keep_in_regions: list[tuple[float, float, float, float]] | None = None
    """Walkable XY footprints; see :func:`~.patches.derive_keep_in_regions`."""

    def __post_init__(self):
        if self.patch_override is not None:
            self.flat_patch_sampling = self.patch_override
        elif self.flat_patch_sampling is None:
            layout_res = resolve_layout(self.layout, self.bridge, self.size, self.cross_links)
            self.flat_patch_sampling = patch_utils.derive_flat_patches(
                layout_res, num_patches=self.patch_num_samples, patch_radius=self.patch_radius
            )
            self.goal_poses = patch_utils.goal_pose_metadata(layout_res)
            self.geometry_bounds = patch_utils.derive_geometry_bounds(layout_res)
            self.keep_in_regions = patch_utils.derive_keep_in_regions(layout_res)
