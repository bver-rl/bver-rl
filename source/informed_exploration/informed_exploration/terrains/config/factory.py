# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable layout builders (linear, star, grid) and `TerrainGeneratorCfg` factories."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg

from ..bridge import (
    BridgeCfg,
    PerimeterWallCfg,
    PlatformBridgeTerrainCfg,
    PlatformCfg,
    RadialLayoutCfg,
    SpokeCfg,
    WallCfg,
)

DEFAULT_TILE_SIZE = (14.0, 14.0)
"""Default sub-terrain tile size (m); the root sits at ``size/2``, so it needs ``size/2 >= radius + half-extent``."""


def linear_layout(
    bridge: BridgeCfg,
    radius: float = 4.5,
    start_platform: PlatformCfg | None = None,
    goal_platform: PlatformCfg | None = None,
) -> RadialLayoutCfg:
    """Start -> single bridge -> one goal, along a single 0-degree spoke."""
    start_platform = start_platform if start_platform is not None else PlatformCfg()
    goal_platform = goal_platform if goal_platform is not None else PlatformCfg()
    return RadialLayoutCfg(
        platform=start_platform,
        spokes=[SpokeCfg(angle=0.0, radius=radius, bridge=bridge, leaf_platform=goal_platform, goal_name="goal")],
    )


def star_layout(
    bridge: BridgeCfg,
    num_goals: int,
    radius: float = 4.0,
    start_platform: PlatformCfg | None = None,
    goal_platform: PlatformCfg | None = None,
) -> RadialLayoutCfg:
    """Start hub with `num_goals` evenly-spaced spokes, each ending in its own goal platform."""
    start_platform = start_platform if start_platform is not None else PlatformCfg()
    goal_platform = goal_platform if goal_platform is not None else PlatformCfg()
    spokes = [
        SpokeCfg(
            angle=2.0 * np.pi * i / num_goals,
            radius=radius,
            bridge=bridge,
            leaf_platform=goal_platform,
            goal_name=f"goal_{i}",
        )
        for i in range(num_goals)
    ]
    return RadialLayoutCfg(platform=start_platform, spokes=spokes)


def _chain(
    bridge: BridgeCfg,
    angle: float,
    radius: float,
    num_hops: int,
    platform_factory: Callable[[], PlatformCfg],
    extra_spoke_factory: Callable[[int], SpokeCfg | None] | None = None,
) -> RadialLayoutCfg | None:
    """A chain of `num_hops` spokes at a fixed angle and radius, each hop optionally growing an extra spoke.

    Returns None if ``num_hops <= 0``.
    """
    if num_hops <= 0:
        return None

    def build(hop: int) -> RadialLayoutCfg:
        spokes = []
        extra = extra_spoke_factory(hop) if extra_spoke_factory is not None else None
        if extra is not None:
            spokes.append(extra)
        if hop < num_hops - 1:
            spokes.append(SpokeCfg(angle=angle, radius=radius, bridge=bridge, child=build(hop + 1)))
        else:
            spokes.append(SpokeCfg(angle=angle, radius=radius, bridge=bridge, leaf_platform=platform_factory()))
        return RadialLayoutCfg(platform=platform_factory(), spokes=spokes)

    return build(0)


def grid_layout(
    bridge: BridgeCfg,
    rows: int,
    cols: int,
    row_spacing: float = 3.0,
    col_spacing: float = 3.0,
    start_platform: PlatformCfg | None = None,
    hub_platform: PlatformCfg | None = None,
) -> RadialLayoutCfg:
    """An XY grid tree: a spine of `cols` hubs along +x, each growing a `rows`-long column along +y.

    Pass ``cross_links=True`` on the owning :class:`~..bridge.PlatformBridgeTerrainCfg` for a lattice.
    """
    start_platform = start_platform if start_platform is not None else PlatformCfg()
    hub_platform = hub_platform if hub_platform is not None else PlatformCfg()

    def column_extra(_hop_index: int) -> SpokeCfg | None:
        if rows <= 1:
            return None
        chain = _chain(bridge, np.pi / 2.0, row_spacing, rows - 1, lambda: hub_platform)
        return SpokeCfg(angle=np.pi / 2.0, radius=row_spacing, bridge=bridge, child=chain)

    spine = _chain(bridge, 0.0, col_spacing, cols - 1, lambda: hub_platform, extra_spoke_factory=column_extra)
    if spine is None:
        # cols == 1: the root itself is the only spine cell.
        extra = column_extra(0)
        return RadialLayoutCfg(platform=start_platform, spokes=[extra] if extra is not None else [])

    spine.platform = start_platform
    return spine


def _flip_layout_bridges(layout: RadialLayoutCfg) -> None:
    """In-place flip of ``reverse`` on every spoke's bridge in a layout tree (recursively)."""
    for spoke in layout.spokes:
        if spoke.bridge is not None:
            spoke.bridge.reverse = not spoke.bridge.reverse
        if spoke.child is not None:
            _flip_layout_bridges(spoke.child)


def reversed_layout(layout: RadialLayoutCfg) -> RadialLayoutCfg:
    """A deep copy of ``layout`` with ``reverse`` flipped on every spoke's bridge.

    Spoke bridges shadow the terrain default, so flipping only the default would not reverse the terrain.
    """
    new = layout.copy()  # configclass .copy() deep-copies all mutable members, including bridges
    _flip_layout_bridges(new)
    return new


def single_type_cfg(
    layout: RadialLayoutCfg,
    bridge: BridgeCfg,
    num_rows: int = 10,
    num_cols: int = 4,
    size: tuple[float, float] = DEFAULT_TILE_SIZE,
    border_width: float = 4.0,
    reverse_columns: bool = False,
    cross_links: bool = False,
) -> TerrainGeneratorCfg:
    """A `TerrainGeneratorCfg` with one sub-terrain (rows = difficulty), for per-terrain scoring runs.

    ``reverse_columns`` adds a ``backward`` sub-terrain with every bridge reversed.
    """
    sub_terrains = {"forward": PlatformBridgeTerrainCfg(size=size, layout=layout, bridge=bridge, cross_links=cross_links)}
    if reverse_columns:
        reversed_bridge = bridge.copy()
        reversed_bridge.reverse = not bridge.reverse
        sub_terrains["backward"] = PlatformBridgeTerrainCfg(
            size=size, layout=reversed_layout(layout), bridge=reversed_bridge, cross_links=cross_links
        )
    return TerrainGeneratorCfg(
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains=sub_terrains,
        use_cache=False,
    )


def pinned_layout_cfg(
    layout: RadialLayoutCfg,
    default_bridge: BridgeCfg,
    difficulty: float,
    sub_terrain_name: str,
    size: tuple[float, float] = DEFAULT_TILE_SIZE,
    border_width: float = 4.0,
    num_rows: int = 1,
    num_cols: int = 1,
    cross_links: bool = False,
    walls: list[WallCfg] | None = None,
    perimeter_wall: PerimeterWallCfg | None = None,
) -> TerrainGeneratorCfg:
    """A single-sub-terrain `TerrainGeneratorCfg` pinned to one difficulty, for named mixed-bridge layouts.

    ``default_bridge`` is required but only fills spokes without their own bridge.
    """
    sub_terrain = PlatformBridgeTerrainCfg(
        size=size,
        layout=layout,
        bridge=default_bridge,
        cross_links=cross_links,
        walls=walls if walls is not None else [],
        perimeter_wall=perimeter_wall,
    )
    return TerrainGeneratorCfg(
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=False,
        difficulty_range=(difficulty, difficulty),
        sub_terrains={sub_terrain_name: sub_terrain},
        use_cache=False,
    )


def all_bridges_cfg(
    bridge_layouts: dict[str, tuple[RadialLayoutCfg, BridgeCfg]],
    num_rows: int = 10,
    size: tuple[float, float] = DEFAULT_TILE_SIZE,
    border_width: float = 4.0,
) -> TerrainGeneratorCfg:
    """A `TerrainGeneratorCfg` with every named bridge family as its own column, for a combined sweep."""
    sub_terrains = {
        name: PlatformBridgeTerrainCfg(size=size, layout=layout, bridge=bridge)
        for name, (layout, bridge) in bridge_layouts.items()
    }
    return TerrainGeneratorCfg(
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=len(sub_terrains),
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        sub_terrains=sub_terrains,
        use_cache=False,
    )
