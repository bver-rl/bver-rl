# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The bridge roster: every concrete :class:`~..interface.BridgeCfg` implementation."""

from .gap import GapBridgeCfg
from .narrow_beam import NarrowBeamBridgeCfg
from .obstacle import ObstacleBridgeCfg
from .pillars import PillarsBridgeCfg
from .ramp import RampBridgeCfg
from .rough import RoughTerrainBridgeCfg
from .stairs import StairsBridgeCfg
from .stepping_stones import SteppingStonesBridgeCfg
from .wall import WallBridgeCfg

__all__ = [
    "GapBridgeCfg",
    "NarrowBeamBridgeCfg",
    "ObstacleBridgeCfg",
    "PillarsBridgeCfg",
    "RampBridgeCfg",
    "RoughTerrainBridgeCfg",
    "StairsBridgeCfg",
    "SteppingStonesBridgeCfg",
    "WallBridgeCfg",
]
