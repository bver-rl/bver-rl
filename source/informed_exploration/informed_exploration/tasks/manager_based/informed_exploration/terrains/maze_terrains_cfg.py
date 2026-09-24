# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg
from .maze_terrains import maze_terrain


@configclass
class MeshMazeTerrainCfg(SubTerrainBaseCfg):

    function = maze_terrain

    cell_size: float = 1.0
    """Size of each cell in the maze."""

    wall_height: float = 5.0
    """Height of the walls in the maze."""

    maze_name: str | None = MISSING
    """Name of the maze to generate. Should be one of the mazes defined in `maze_terrains.py`."""
