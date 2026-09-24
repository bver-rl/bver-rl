# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DTSG layout: one goal reachable over three walled parallel routes (stairs, pillars, ramp) on floor-level platforms.

The layout tree cannot express the two cycles, so the pillars and ramp routes end in hubs coincident with the goal.
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg

from ...bridge import PerimeterWallCfg, PlatformCfg, RadialLayoutCfg, SpokeCfg, WallCfg, resolve_layout
from ...bridge.bridges import *
from ...bridge.interface import lerp
from ...bridge.patches import span_rect
from ..factory import pinned_layout_cfg

MULTIPLE_START_POSES = False
"""Default for :func:`build_dtsg_layout`: whether P_TL and P_BL also count as start platforms.

Env configs pass it explicitly and enable it for every task whose start column is open."""

DTSG_SPOKE_RADIUS = 6.0
"""Center-to-center distance (m) of every spoke in the grid."""

DTSG_SHORT_RADIUS = 1.8
"""Center-to-center distance (m) of the vertical connector spokes (the corridor spacing)."""

DTSG_ROUTE_WIDTH = 1.5
"""Width (m) of the three route bridges; the separator walls sit flush with their sides."""

DTSG_WALL_HEIGHT = 2.0
"""Wall-top height (m) of the separator walls and perimeter fence, tall enough to be impassable."""

_PLATFORM_HALF = 1.25
"""Half-extent (m) of the :class:`PlatformCfg` footprint every hub here uses."""
_WALL_MARGIN = 0.5

DTSG_TILE_SIZE = (16.0, 16.0)
"""Sub-terrain tile size (m): spoke radius plus platform half-width plus border clearance, on each side of the root."""

DTSG_DIFFICULTY = 0.5
"""Fixed difficulty this layout is calibrated at (see :func:`dtsg_cfg`'s pinned difficulty_range)."""

DTSG_PLATFORM_TOP_Z = 0.05
"""Platform top height (m), nearly flush with the floor but raised enough to avoid z-fighting."""

DTSG_POLE_EXTRA_HEIGHT_RANGE = (0.2, 0.4)
"""Pillar-top rise (m) above the floor-level platforms, from easy to hard."""

DTSG_POSE_OFFSET_X = 0.5
"""Distance (m) start poses are pulled back (-x) and the goal pose pushed forward (+x) on their pads.

It also shifts the env-relative frame, which :func:`dtsg_walls` compensates for."""

DTSG_SUBTERRAIN_NAME = "dtsg"


def _floor_platform(pose_offset: tuple[float, float] = (0.0, 0.0)) -> PlatformCfg:
    return PlatformCfg(top_z=DTSG_PLATFORM_TOP_Z, size=(2.5, 2.5), pose_offset=pose_offset)


def _start_platform() -> PlatformCfg:
    return _floor_platform(pose_offset=(-DTSG_POSE_OFFSET_X, 0.0))


def dtsg_walls(seal_goal_column: bool = False, seal_start_column: bool = False) -> list[WallCfg]:
    """Build the route-separator walls, walling both sides of every route span.

    ``seal_goal_column`` and ``seal_start_column`` extend the inner wall pair through the goal or start column
    so a route's exit or entry is reachable only through that route (corridor evals). They are mutually
    exclusive, since sealing both strands the goal pose, and change the terrain cache key.

    Args:
        seal_goal_column: Extend the inner walls through the goal column.
        seal_start_column: Extend the inner walls through the start column.

    Returns:
        The wall configs in env-origin-relative coordinates.
    """
    # env origin is the root's pose point, offset from the pad center, so shift to keep walls fixed in the world
    assert not (seal_goal_column and seal_start_column), (
        "sealing BOTH columns strands the goal pose in the central pocket, unreachable from the"
        " pillars and ramp routes; see this function's docstring."
    )
    x_lo_span = DTSG_POSE_OFFSET_X + _PLATFORM_HALF  # start column's inner edge
    x_lo_sealed = DTSG_POSE_OFFSET_X - _PLATFORM_HALF  # its outer edge
    x_hi_span = DTSG_POSE_OFFSET_X + DTSG_SPOKE_RADIUS - _PLATFORM_HALF  # goal column's inner edge
    x_hi_sealed = DTSG_POSE_OFFSET_X + DTSG_SPOKE_RADIUS + _PLATFORM_HALF  # its outer edge
    inner_pair = (DTSG_ROUTE_WIDTH / 2.0, DTSG_SHORT_RADIUS - DTSG_ROUTE_WIDTH / 2.0)
    outer_pair = (DTSG_SHORT_RADIUS + DTSG_ROUTE_WIDTH / 2.0, DTSG_SHORT_RADIUS + _PLATFORM_HALF)
    inner_x = (
        x_lo_sealed if seal_start_column else x_lo_span,
        x_hi_sealed if seal_goal_column else x_hi_span,
    )

    walls: list[WallCfg] = []
    for (y_lo, y_hi), (x_lo, x_hi) in ((inner_pair, inner_x), (outer_pair, (x_lo_span, x_hi_span))):
        x_center, x_size = (x_lo + x_hi) / 2.0, x_hi - x_lo
        y_center, y_size = (y_lo + y_hi) / 2.0, y_hi - y_lo
        walls += [
            WallCfg(center=(x_center, y_center), size=(x_size, y_size), height=DTSG_WALL_HEIGHT),
            WallCfg(center=(x_center, -y_center), size=(x_size, y_size), height=DTSG_WALL_HEIGHT),
        ]
    return walls


def dtsg_perimeter_wall() -> PerimeterWallCfg:
    """Build the arena fence, its inner faces flush with the outer platform edges."""
    return PerimeterWallCfg(height=DTSG_WALL_HEIGHT, thickness=_WALL_MARGIN, margin=(0.0, -_WALL_MARGIN))


def build_dtsg_layout(multiple_start_poses: bool = MULTIPLE_START_POSES) -> RadialLayoutCfg:
    """Build the difficulty-independent DTSG platform-graph tree.

    Args:
        multiple_start_poses: Mark P_TL and P_BL as additional start platforms.
    """
    up, right, down = np.pi / 2.0, 0.0, -np.pi / 2.0
    r = DTSG_SPOKE_RADIUS
    r_short = DTSG_SHORT_RADIUS

    # P_TR / P_BR descend to non-goal hubs coincident with the goal, closing the cycles the tree cannot express
    goal_dup_top = RadialLayoutCfg(
        platform=_floor_platform(),
        spokes=[],
    )
    goal_dup_bottom = RadialLayoutCfg(
        platform=_floor_platform(),
        spokes=[],
    )

    p_tr = RadialLayoutCfg(
        platform=_floor_platform(),
        spokes=[
            SpokeCfg(
                angle=down,
                radius=r_short,
                bridge=RoughTerrainBridgeCfg(
                    noise_amplitude_range=(0.0, 0.0),
                ),
                child=goal_dup_top,
            )
        ],
    )
    p_tl = RadialLayoutCfg(
        platform=_start_platform(),
        # optional start pad sharing the root's pose_offset so the start region stays one clean column
        is_start=multiple_start_poses,
        spokes=[
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=PillarsBridgeCfg(
                    width=DTSG_ROUTE_WIDTH,
                    pole_extra_height_range=(0.2, 0.2),
                    length_range=(2.0, 2.0),
                    pole_radius_range=(0.2, 0.2),
                    gap_range=(0.1, 0.1),
                ),
                child=p_tr,
            )
        ],
    )

    p_br = RadialLayoutCfg(
        platform=_floor_platform(),
        spokes=[
            SpokeCfg(
                angle=up,
                radius=r_short,
                bridge=RoughTerrainBridgeCfg(
                    noise_amplitude_range=(0.0, 0.0),
                ),
                child=goal_dup_bottom,
            )
        ],
    )
    p_bl = RadialLayoutCfg(
        platform=_start_platform(),
        is_start=multiple_start_poses,  # see p_tl
        spokes=[
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=RampBridgeCfg(
                    width=DTSG_ROUTE_WIDTH,
                    reverse=True,
                    peak_extra_height_range=(0.5, 0.5),
                    length_range=(2.0, 2.0),
                    surface_noise_range=(0.05, 0.05),
                ),
                child=p_br,
            )
        ],
    )

    # start hub: +x stairs to the leaf named "goal" (the sole goal), +y/-y rough paths to the other routes
    return RadialLayoutCfg(
        platform=_start_platform(),
        spokes=[
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=StairsBridgeCfg(
                    width=DTSG_ROUTE_WIDTH,
                    entry_num_steps=4,
                    exit_num_steps=1,
                    reverse=True,
                    peak_extra_height_range=(0.5, 0.5),
                    length_range=(2.0, 2.0),
                ),
                leaf_platform=_floor_platform(pose_offset=(DTSG_POSE_OFFSET_X, 0.0)),
                goal_name="goal",
            ),
            SpokeCfg(
                angle=up,
                radius=r_short,
                bridge=RoughTerrainBridgeCfg(
                    noise_amplitude_range=(0.0, 0.0),
                ),
                child=p_tl,
            ),
            SpokeCfg(
                angle=down,
                radius=r_short,
                bridge=RoughTerrainBridgeCfg(
                    noise_amplitude_range=(0.0, 0.0),
                ),
                child=p_bl,
            ),
        ],
    )


_ROUTE_BRIDGE_LABELS: tuple[tuple[type, str], ...] = (
    (StairsBridgeCfg, "stairs"),
    (PillarsBridgeCfg, "pillars"),
    (RampBridgeCfg, "ramp"),
)
"""Bridge family to route name; the rough vertical connectors are within-column links, not routes."""


def dtsg_route_rects(
    layout: RadialLayoutCfg | None = None,
    size: tuple[float, float] = DTSG_TILE_SIZE,
    include_start_column: bool = False,
) -> dict[str, tuple[float, float, float, float]]:
    """Compute env-relative ``{route_name: (x_lo, x_hi, y_lo, y_hi)}`` rects for the three routes.

    Derived from the resolved layout, so the rects are disjoint in y and track the mesh.

    Args:
        layout: Layout to resolve. Defaults to :func:`build_dtsg_layout`.
        size: Sub-terrain tile size (m).
        include_start_column: Extend each rect back over its route's entry platform.

    Returns:
        Route rects keyed ``"stairs"``, ``"pillars"`` and ``"ramp"``.
    """
    layout_res = resolve_layout(layout if layout is not None else build_dtsg_layout(), NarrowBeamBridgeCfg(), size)
    assert layout_res.root is not None, "layout resolution has no root hub"
    origin = layout_res.root.pose_center

    rects: dict[str, tuple[float, float, float, float]] = {}
    for edge in layout_res.edges:
        label = next((name for cls, name in _ROUTE_BRIDGE_LABELS if isinstance(edge.bridge, cls)), None)
        if label is None:
            continue
        assert label not in rects, f"two {label} spans in the DTSG layout; the route label is ambiguous"
        x_lo, x_hi, y_lo, y_hi = span_rect(edge, origin)
        if include_start_column:
            # far edge of the departure platform, read from the parent node so it tracks pad changes
            parent = edge.parent
            x_lo = float(parent.center[0] - parent.platform.size[0] / 2.0 - origin[0])
        rects[label] = (x_lo, x_hi, y_lo, y_hi)

    expected = {name for _, name in _ROUTE_BRIDGE_LABELS}
    assert set(rects) == expected, f"DTSG layout is missing route(s) {sorted(expected - set(rects))}"
    return rects


_ROUTE_RISE_FIELDS = ("peak_extra_height_range", "pole_extra_height_range")
"""Per-family fields giving a route's difficulty-interpolated rise above the platform tops.

Surface-roughness fields are excluded, since they are not a height the route reaches."""


def dtsg_route_top_z(
    layout: RadialLayoutCfg | None = None,
    size: tuple[float, float] = DTSG_TILE_SIZE,
    difficulty: float = DTSG_DIFFICULTY,
) -> dict[str, float]:
    """Compute the env-relative z of the highest walkable surface on each route.

    Read from the config rather than raycast, since rays can fall between the dart-thrown pillars.

    Args:
        layout: Layout to resolve. Defaults to :func:`build_dtsg_layout`.
        size: Sub-terrain tile size (m).
        difficulty: Difficulty at which the rises are interpolated.

    Returns:
        Top z per route, keyed like :func:`dtsg_route_rects`.
    """
    layout_res = resolve_layout(layout if layout is not None else build_dtsg_layout(), NarrowBeamBridgeCfg(), size)
    assert layout_res.root is not None, "layout resolution has no root hub"

    tops: dict[str, float] = {}
    for edge in layout_res.edges:
        label = next((name for cls, name in _ROUTE_BRIDGE_LABELS if isinstance(edge.bridge, cls)), None)
        if label is None:
            continue
        # mirror `_dtsg_peak_standing_z`: the harder side wins, so the bound tracks a per-side retune
        side_difficulty = max(edge.bridge.side_difficulties(difficulty))
        rises = [lerp(side_difficulty, *getattr(edge.bridge, f)) for f in _ROUTE_RISE_FIELDS if hasattr(edge.bridge, f)]
        assert rises, f"the {label} bridge declares no rise field; its top would collapse to the platform"
        # every DTSG platform is level (`_floor_platform`), so the span's own top_z is the base
        tops[label] = DTSG_PLATFORM_TOP_Z + max(rises)

    expected = {name for _, name in _ROUTE_BRIDGE_LABELS}
    assert set(tops) == expected, f"DTSG layout is missing route(s) {sorted(expected - set(tops))}"
    return tops


def dtsg_cfg(seal_goal_column: bool = False) -> TerrainGeneratorCfg:
    """Build a `TerrainGeneratorCfg` with a single DTSG sub-terrain pinned to :data:`DTSG_DIFFICULTY`.

    Args:
        seal_goal_column: Wall off the goal column into per-route pockets.
    """
    return pinned_layout_cfg(
        build_dtsg_layout(),
        NarrowBeamBridgeCfg(),  # every spoke sets its own bridge; this default is never used
        DTSG_DIFFICULTY,
        DTSG_SUBTERRAIN_NAME,
        size=DTSG_TILE_SIZE,
        walls=dtsg_walls(seal_goal_column),
        perimeter_wall=dtsg_perimeter_wall(),
    )


def dtsg_sealed_cfg() -> TerrainGeneratorCfg:
    """Build :func:`dtsg_cfg` with the goal column sealed, the corridor-eval terrain."""
    return dtsg_cfg(seal_goal_column=True)
