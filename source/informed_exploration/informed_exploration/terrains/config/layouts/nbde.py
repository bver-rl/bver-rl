# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NBDE layout: a 2x3 platform grid of narrow beams with an infeasible dead-end gap directly between start and goal.

The two grid cycles are closed with non-goal terminal hubs coincident with the goal and with P_TR.
"""

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg

from ...bridge import PlatformCfg, RadialLayoutCfg, SpokeCfg
from ...bridge.bridges import *
from ..factory import pinned_layout_cfg

NBDE_SPOKE_RADIUS = 5.0
"""Center-to-center distance (m) of every spoke in the grid."""

NBDE_TILE_SIZE = (28.0, 16.0)
"""Sub-terrain tile size (m): grid extent plus platform half-width and border clearance around the root."""

NBDE_DIFFICULTY = 0.5
"""Fixed difficulty this layout is calibrated at (see :func:`nbde_cfg`'s pinned difficulty_range)."""

NBDE_GAP_WIDTH = 2.0
"""Fixed gap width (m) of the dead-end edge, infeasible regardless of difficulty."""

BEAM_WIDTH = 1.5

NBDE_SUBTERRAIN_NAME = "nbde"


def build_nbde_layout() -> RadialLayoutCfg:
    """Build the difficulty-independent NBDE platform-graph tree."""
    up, right, left = np.pi / 2.0, 0.0, np.pi
    r = NBDE_SPOKE_RADIUS

    # non-goal hub coincident with P_TR closes the top-row edge, since P_TR itself is reached through P_BR
    p_tr_dup = RadialLayoutCfg(platform=PlatformCfg(), spokes=[])

    # P_TM: -x to the leaf named "goal" (the sole goal), +x to the P_TR duplicate
    p_tm = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=left,
                radius=r,
                bridge=NarrowBeamBridgeCfg(beam_width_range=(BEAM_WIDTH, BEAM_WIDTH)),
                leaf_platform=PlatformCfg(),
                goal_name="goal",
            ),
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=NarrowBeamBridgeCfg(beam_width_range=(BEAM_WIDTH, BEAM_WIDTH)),
                child=p_tr_dup,
            ),
        ],
    )

    p_tr = RadialLayoutCfg(platform=PlatformCfg(), spokes=[])
    p_br = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=up,
                radius=r,
                bridge=NarrowBeamBridgeCfg(beam_width_range=(BEAM_WIDTH, BEAM_WIDTH)),
                child=p_tr,
            )
        ],
    )

    p_bm = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=up,
                radius=r,
                bridge=StairsBridgeCfg(
                    reverse=True, entry_num_steps=3, exit_num_steps=1, peak_extra_height_range=(0.5, 0.5)
                ),
                child=p_tm,
            ),
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=NarrowBeamBridgeCfg(beam_width_range=(BEAM_WIDTH, BEAM_WIDTH)),
                child=p_br,
            ),
        ],
    )

    # S -gap-> a non-goal terminal hub (dead end), coincident with the goal's position.
    dead_end = RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[],
    )

    # Start hub: +y (infeasible gap) -> dead end at G's position; +x (narrow beam) -> P_BM.
    return RadialLayoutCfg(
        platform=PlatformCfg(),
        spokes=[
            SpokeCfg(
                angle=up,
                radius=r,
                bridge=GapBridgeCfg(
                    gap_width_range=(NBDE_GAP_WIDTH, NBDE_GAP_WIDTH),
                    entry_lip_range=(0.0, 0.0),
                ),
                child=dead_end,
            ),
            SpokeCfg(
                angle=right,
                radius=r,
                bridge=NarrowBeamBridgeCfg(beam_width_range=(BEAM_WIDTH, BEAM_WIDTH)),
                child=p_bm,
            ),
        ],
    )


def nbde_cfg() -> TerrainGeneratorCfg:
    """Build a `TerrainGeneratorCfg` with a single NBDE sub-terrain, pinned to :data:`NBDE_DIFFICULTY`."""
    return pinned_layout_cfg(
        build_nbde_layout(),
        NarrowBeamBridgeCfg(),  # every spoke sets its own bridge; this default is never used
        NBDE_DIFFICULTY,
        NBDE_SUBTERRAIN_NAME,
        size=NBDE_TILE_SIZE,
    )
