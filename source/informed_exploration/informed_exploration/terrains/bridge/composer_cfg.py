# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recursive platform-graph layout expressing linear, radial-star and grid layouts with one primitive.

Each :class:`RadialLayoutCfg` hub has :class:`SpokeCfg` children at polar offsets ``(angle, radius)``,
each ending in a leaf goal or recursing into another hub.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from isaaclab.utils import configclass

from .interface import BridgeCfg


@configclass
class PlatformCfg:
    """A flat platform: a start hub, an intermediate hub, or a goal."""

    size: tuple[float, float] = (2.5, 2.5)
    """Footprint of the platform (in m)."""

    top_z: float | None = None
    """Absolute height (in m) of the platform's top surface above the tile ground.

    ``None`` resolves to :data:`GROUND_LEVEL_PLATFORM_RISE`; an explicit value, even zero, is used verbatim."""

    pose_offset: tuple[float, float] = (0.0, 0.0)
    """XY offset (in m) of the start/goal reference point from the platform center, without moving the mesh.

    A root offset shifts the whole env-relative frame. Keep the point and its patch window inside the
    footprint; nothing checks."""


@configclass
class SpokeCfg:
    """A directed edge from a parent hub to a child platform (leaf goal or nested hub)."""

    angle: float = 0.0
    """World-frame angle (in rad) of the spoke direction, measured from +x."""

    radius: float = 3.0
    """Center-to-center distance (in m) from the parent hub to the child platform."""

    bridge: BridgeCfg | None = None
    """Bridge filling this span. If None, the terrain-level default bridge is used."""

    child: "RadialLayoutCfg | None" = None
    """If set, this spoke recurses into another hub. If None, it terminates in a leaf goal."""

    leaf_platform: PlatformCfg | None = None
    """Platform config for the leaf goal (only used when :attr:`child` is None); defaults if None."""

    goal_name: str | None = None
    """Name for the auto-derived flat patch / pose metadata of this leaf goal. Auto-numbered if None."""


@configclass
class RadialLayoutCfg:
    """Recursive node in the platform graph. The terrain's root node is the start hub."""

    platform: PlatformCfg = PlatformCfg()
    """The platform at this hub."""

    spokes: list[SpokeCfg] = []
    """Child spokes radiating from this hub."""

    is_start: bool = False
    """Mark this hub as part of the start region alongside the root.

    The start window is one box over all start platforms, so only mark hubs adjacent to the root
    with the same ``top_z``."""


@configclass
class WallCfg:
    """An axis-aligned interior wall box in root-relative coordinates, e.g. separating parallel routes.

    Mesh-only: invisible to the derived task-space metadata, so keep walls thin.
    """

    center: tuple[float, float] = (0.0, 0.0)
    """Wall footprint center (x, y), in m, relative to the root (start) hub."""

    size: tuple[float, float] = (1.0, 0.2)
    """Wall footprint extents (x, y), in m."""

    height: float = 1.0
    """Wall-top height (in m) above the tile floor; the box always reaches true ground."""


@configclass
class PerimeterWallCfg:
    """Arena fence of four walls around the platform-footprint AABB, sized from the layout, not the tile."""

    height: float = 1.0
    """Wall-top height (in m) above the tile floor (z=0)."""

    thickness: float = 0.2
    """Wall thickness (in m)."""

    margin: tuple[float, float] = (0.5, 0.5)
    """Clear floor (in m) between the footprint AABB and the walls' inner faces, per axis ``(x, y)``.

    Negative values pull the walls inside the AABB. The fence must stay inside the tile."""


"""
Layout resolution, difficulty-independent so it is shared by mesh generation and patch derivation.
"""


@dataclass
class PlatformNode:
    """A resolved platform: absolute position + the config that placed it."""

    name: str
    center: np.ndarray
    """Absolute (x, y, z) of the platform's top-surface center, in the tile-local frame."""
    platform: PlatformCfg
    parent_name: str | None
    is_goal: bool = False
    is_start: bool = False
    """True for the root hub and any hub whose :attr:`RadialLayoutCfg.is_start` is set."""

    @property
    def pose_center(self) -> np.ndarray:
        """Absolute (x, y, z) start/goal reference point: :attr:`center` shifted by the pose offset.

        Derived pose frames anchor here; meshes and bridge spans use :attr:`center`."""
        return self.center + np.array([self.platform.pose_offset[0], self.platform.pose_offset[1], 0.0])


@dataclass
class EdgeSpan:
    """A resolved bridge span between two platforms."""

    parent: PlatformNode
    child: PlatformNode
    angle: float
    bridge: BridgeCfg
    length: float
    """Usable span length (in m) between the two platform edges (radius minus half-extents)."""


@dataclass
class LayoutResolution:
    nodes: list[PlatformNode] = field(default_factory=list)
    edges: list[EdgeSpan] = field(default_factory=list)
    root: PlatformNode | None = None
    goals: list[PlatformNode] = field(default_factory=list)


def _default_leaf_platform() -> PlatformCfg:
    return PlatformCfg()


GROUND_LEVEL_PLATFORM_RISE = 1.0
"""Height (in m) of platforms whose ``top_z`` is unset, applied at layout resolution so all consumers agree."""


def _resolve_platform_top_z(top_z: float | None) -> float:
    return GROUND_LEVEL_PLATFORM_RISE if top_z is None else top_z


def _walk(
    layout: RadialLayoutCfg,
    origin_xy: np.ndarray,
    name: str,
    parent_name: str | None,
    default_bridge: BridgeCfg,
    nodes: list[PlatformNode],
    edges: list[EdgeSpan],
    goals: list[PlatformNode],
) -> PlatformNode:
    hub_center = np.array([origin_xy[0], origin_xy[1], _resolve_platform_top_z(layout.platform.top_z)])
    hub = PlatformNode(
        name=name,
        center=hub_center,
        platform=layout.platform,
        parent_name=parent_name,
        is_start=parent_name is None or layout.is_start,
    )
    nodes.append(hub)

    for i, spoke in enumerate(layout.spokes):
        bridge = spoke.bridge if spoke.bridge is not None else default_bridge
        direction = np.array([np.cos(spoke.angle), np.sin(spoke.angle)])
        child_xy = origin_xy + spoke.radius * direction

        if spoke.child is not None:
            child_name = f"{name}/{i}"
            child_node = _walk(spoke.child, child_xy, child_name, name, default_bridge, nodes, edges, goals)
            child_platform = spoke.child.platform
        else:
            leaf_platform = spoke.leaf_platform if spoke.leaf_platform is not None else _default_leaf_platform()
            goal_name = spoke.goal_name if spoke.goal_name is not None else f"goal_{len(goals)}"
            child_center = np.array([child_xy[0], child_xy[1], _resolve_platform_top_z(leaf_platform.top_z)])
            child_node = PlatformNode(
                name=goal_name, center=child_center, platform=leaf_platform, parent_name=name, is_goal=True
            )
            nodes.append(child_node)
            goals.append(child_node)
            child_platform = leaf_platform

        usable_length = spoke.radius - hub.platform.size[0] / 2.0 - child_platform.size[0] / 2.0
        usable_length = max(0.5, usable_length)
        edges.append(EdgeSpan(parent=hub, child=child_node, angle=spoke.angle, bridge=bridge, length=usable_length))

    return hub


def resolve_layout(
    layout: RadialLayoutCfg,
    default_bridge: BridgeCfg,
    size: tuple[float, float],
    cross_links: bool = False,
) -> LayoutResolution:
    """Walk the recursive layout tree, placing the root (start) hub at the tile center.

    Args:
        layout: The root :class:`RadialLayoutCfg` (the start hub).
        default_bridge: Bridge used for any spoke that doesn't specify its own.
        size: Tile size (in m); the root hub is placed at ``(size[0]/2, size[1]/2)``.
        cross_links: Whether to add best-effort bridges between sibling hubs, forming a lattice on small grids.

    Returns:
        The resolved node/edge graph.
    """
    nodes: list[PlatformNode] = []
    edges: list[EdgeSpan] = []
    goals: list[PlatformNode] = []
    origin_xy = np.array([size[0] / 2.0, size[1] / 2.0])
    root = _walk(layout, origin_xy, "start", None, default_bridge, nodes, edges, goals)

    if cross_links:
        by_parent: dict[str, list[PlatformNode]] = {}
        for node in nodes:
            if node.parent_name is not None and not node.is_goal:
                by_parent.setdefault(node.parent_name, []).append(node)
        for siblings in by_parent.values():
            for a in range(len(siblings)):
                for b in range(a + 1, len(siblings)):
                    n1, n2 = siblings[a], siblings[b]
                    delta = n2.center[:2] - n1.center[:2]
                    dist = float(np.linalg.norm(delta))
                    angle = float(np.arctan2(delta[1], delta[0]))
                    usable = max(0.5, dist - n1.platform.size[0] / 2.0 - n2.platform.size[0] / 2.0)
                    edges.append(EdgeSpan(parent=n1, child=n2, angle=angle, bridge=default_bridge, length=usable))

    _check_bounds(nodes, size)
    return LayoutResolution(nodes=nodes, edges=edges, root=root, goals=goals)


def _check_bounds(nodes: list[PlatformNode], size: tuple[float, float]) -> None:
    """Warn (do not raise) if any resolved platform footprint extends outside the tile."""
    for node in nodes:
        half_x, half_y = node.platform.size[0] / 2.0, node.platform.size[1] / 2.0
        x_min, x_max = node.center[0] - half_x, node.center[0] + half_x
        y_min, y_max = node.center[1] - half_y, node.center[1] + half_y
        if x_min < 0.0 or x_max > size[0] or y_min < 0.0 or y_max > size[1]:
            warnings.warn(
                f"PlatformBridgeTerrainCfg: platform '{node.name}' at "
                f"({node.center[0]:.2f}, {node.center[1]:.2f}) with size {node.platform.size} extends "
                f"outside the tile bounds {size}. Increase `size`, shrink `radius`/platform sizes, or "
                "accept clipping at the tile border.",
                stacklevel=3,
            )
