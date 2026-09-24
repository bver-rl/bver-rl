# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Difficulty-calibrated `TerrainGeneratorCfg` presets, one per bridge family.

Each preset is a start, bridge, goal :func:`~.factory.linear_layout`; directional bridges enable
``reverse_columns`` so both traversal directions appear side by side.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.terrains import TerrainGeneratorCfg

from ..bridge.bridges import (
    GapBridgeCfg,
    NarrowBeamBridgeCfg,
    ObstacleBridgeCfg,
    PillarsBridgeCfg,
    RampBridgeCfg,
    RoughTerrainBridgeCfg,
    StairsBridgeCfg,
    SteppingStonesBridgeCfg,
    WallBridgeCfg,
)
from .factory import DEFAULT_TILE_SIZE, linear_layout, single_type_cfg

_SPOKE_RADIUS = 8.5
_NUM_ROWS = 3
_NUM_COLS = 1


def stepping_stones_cfg() -> TerrainGeneratorCfg:
    bridge = SteppingStonesBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
    )


def narrow_beam_cfg() -> TerrainGeneratorCfg:
    bridge = NarrowBeamBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
    )


def obstacle_cfg() -> TerrainGeneratorCfg:
    """Crouch-under / climb-over. Directional by default: entry side stays open, exit side tightens."""
    bridge = ObstacleBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        reverse_columns=True,
    )


def stairs_cfg() -> TerrainGeneratorCfg:
    bridge = StairsBridgeCfg(
        reverse=False,
    )
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
    )


def rough_cfg() -> TerrainGeneratorCfg:
    bridge = RoughTerrainBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
    )


def gap_cfg() -> TerrainGeneratorCfg:
    """A leap across a gap. Directional by default: exit lip is raised, so jumping that way is harder."""
    bridge = GapBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        reverse_columns=True,
    )


def ramp_cfg() -> TerrainGeneratorCfg:
    """An A-frame ridge with the peak offset toward the entry side, steep one way and gentle the other."""
    bridge = RampBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        reverse_columns=True,
    )


def pillars_cfg() -> TerrainGeneratorCfg:
    bridge = PillarsBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
    )


def wall_cfg() -> TerrainGeneratorCfg:
    """A funnel: wide, fixed opening on entry, narrowing to a tight doorway on exit."""
    bridge = WallBridgeCfg()
    return single_type_cfg(
        linear_layout(bridge, radius=_SPOKE_RADIUS),
        bridge,
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        reverse_columns=True,
    )


PRESETS: dict[str, Callable[[], TerrainGeneratorCfg]] = {
    "stepping_stones": stepping_stones_cfg,
    "narrow_beam": narrow_beam_cfg,
    "obstacle": obstacle_cfg,
    "stairs": stairs_cfg,
    "rough": rough_cfg,
    "gap": gap_cfg,
    "ramp": ramp_cfg,
    "pillars": pillars_cfg,
    "wall": wall_cfg,
}
"""All bridge-family presets, keyed by name."""

__all__ = ["PRESETS", "DEFAULT_TILE_SIZE"] + [f"{name}_cfg" for name in PRESETS]
