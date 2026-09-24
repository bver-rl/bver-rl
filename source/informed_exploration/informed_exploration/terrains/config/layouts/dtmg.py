# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DTMG ("different terrain multi goal"): a radial star with three goals, each over a different bridge family.

Spokes go +y, +x and -y from the start hub, each ending in its own leaf goal.
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg

from ...bridge import PlatformCfg, RadialLayoutCfg, SpokeCfg
from ...bridge.bridges import *
from ..factory import pinned_layout_cfg

DTMG_SPOKE_RADIUS = 6.0
"""Center-to-center distance (m) of every spoke."""

DTMG_TILE_SIZE = (14.0, 14.0)
"""Sub-terrain tile size (m)."""

DTMG_DIFFICULTY = 0.75
"""Fixed difficulty this layout is calibrated at."""

DTMG_SUBTERRAIN_NAME = "dtmg"


def build_dtmg_layout() -> RadialLayoutCfg:
    """Build the difficulty-independent DTMG platform-graph tree."""
    up, right, down = np.pi / 2.0, 0.0, -np.pi / 2.0
    r = DTMG_SPOKE_RADIUS
    return RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=up,
                radius=r,
                bridge=GapBridgeCfg(
                    gap_width_range=(0.3, 0.3),
                ),
                leaf_platform=PlatformCfg(),
                goal_name="goal_nb",
            ),
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=ObstacleBridgeCfg(
                    reverse=True,
                    bar_thickness=1.0,
                ),
                leaf_platform=PlatformCfg(),
                goal_name="goal_ss",
            ),
            SpokeCfg(
                angle=down,
                radius=r,
                bridge=RampBridgeCfg(
                    peak_extra_height_range=(0.3, 0.9),
                ),
                leaf_platform=PlatformCfg(),
                goal_name="goal_ramp",
            ),
        ],
    )


def dtmg_cfg() -> TerrainGeneratorCfg:
    """A `TerrainGeneratorCfg` with a single DTMG sub-terrain, pinned to :data:`DTMG_DIFFICULTY`."""
    return pinned_layout_cfg(
        build_dtmg_layout(),
        NarrowBeamBridgeCfg(),  # every spoke sets its own bridge; this default is never used
        DTMG_DIFFICULTY,
        DTMG_SUBTERRAIN_NAME,
        size=DTMG_TILE_SIZE,
    )
