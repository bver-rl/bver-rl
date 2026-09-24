# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SGDE layout: a 2x2 platform grid with a feasible route to the goal and an infeasible gap dead end.

The gap branch ends in a non-goal terminal hub coincident with the goal, since the layout tree cannot express the cycle.
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg

from ...bridge import PlatformCfg, RadialLayoutCfg, SpokeCfg
from ...bridge.bridges import *
from ..factory import pinned_layout_cfg

SGDE_SPOKE_RADIUS = 6.0
"""Center-to-center distance (m) of every spoke in the grid."""

SGDE_TILE_SIZE = (16.0, 16.0)
"""Sub-terrain tile size (m), leaving clearance past the far platforms."""

SGDE_DIFFICULTY = 0.5
"""Fixed difficulty this layout is calibrated at (see :func:`sgde_cfg`'s pinned difficulty_range)."""

SGDE_GAP_WIDTH = 2.0
"""Fixed gap width (m), infeasible regardless of difficulty."""

SGDE_SUBTERRAIN_NAME = "sgde"


def build_sgde_layout() -> RadialLayoutCfg:
    """Build the difficulty-independent SGDE platform-graph tree."""
    right = 0.0
    down = -np.pi / 2.0

    # P_TR -gap-> a non-goal terminal hub (dead end), coincident with the goal's position.
    dead_end = RadialLayoutCfg(platform=PlatformCfg(), spokes=[])
    p_tr = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=down,
                radius=SGDE_SPOKE_RADIUS,
                bridge=GapBridgeCfg(gap_width_range=(SGDE_GAP_WIDTH, SGDE_GAP_WIDTH)),
                child=dead_end,
            )
        ],
    )

    # P_BL to the leaf named "goal", the sole goal
    p_bl = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=right,
                radius=SGDE_SPOKE_RADIUS,
                bridge=StairsBridgeCfg(
                    reverse=False,
                    entry_num_steps=2,
                    exit_num_steps=3,
                    peak_extra_height_range=(0.25, 0.5),
                ),
                leaf_platform=PlatformCfg(),
                goal_name="goal",
            )
        ],
    )

    # start hub: +x to P_TR (top edge), -y to P_BL (left edge)
    return RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=right,
                radius=SGDE_SPOKE_RADIUS,
                bridge=StairsBridgeCfg(
                    reverse=True,
                    entry_num_steps=2,
                    exit_num_steps=3,
                    peak_extra_height_range=(0.25, 0.5),
                ),
                child=p_tr,
            ),
            SpokeCfg(
                angle=down,
                radius=SGDE_SPOKE_RADIUS,
                bridge=StairsBridgeCfg(
                    reverse=True,
                    entry_num_steps=2,
                    exit_num_steps=3,
                ),
                child=p_bl,
            ),
        ],
    )


def sgde_cfg() -> TerrainGeneratorCfg:
    """Build a `TerrainGeneratorCfg` with a single SGDE sub-terrain, pinned to :data:`SGDE_DIFFICULTY`."""
    return pinned_layout_cfg(
        build_sgde_layout(),
        NarrowBeamBridgeCfg(),  # every spoke sets its own bridge; this default is never used
        SGDE_DIFFICULTY,
        SGDE_SUBTERRAIN_NAME,
        size=SGDE_TILE_SIZE,
    )
