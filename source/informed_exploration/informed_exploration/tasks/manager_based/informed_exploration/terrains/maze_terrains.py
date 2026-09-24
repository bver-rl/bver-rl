# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

from .maze_geometry import MAZE_LAYOUTS, MazeCell, maze_geometry  # noqa: F401

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def maze_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshMazeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    # compute the position of the terrain
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0)
    # compute the vertices of the terrain
    plane_mesh = make_plane(cfg.size, 0.0, center_zero=False)

    # construct the maze
    if cfg.maze_name not in MAZE_LAYOUTS:
        raise ValueError(f"Maze name {cfg.maze_name} not recognized.")
    maze_layout = MAZE_LAYOUTS[cfg.maze_name].layout

    rows, cols = len(maze_layout), len(maze_layout[0])
    cell_size = cfg.cell_size
    width, height = cols * cell_size, rows * cell_size
    assert rows * cell_size <= cfg.size[0] and cols * cell_size <= cfg.size[1], "Maze size exceeds terrain size."

    maze_meshes = list()
    for i in range(rows):
        for j in range(cols):
            cell = maze_layout[i][j]
            if cell.is_wall():
                wall_mesh = trimesh.creation.box(
                    (cell_size, cell_size, cfg.wall_height),
                    trimesh.transformations.translation_matrix(
                        (
                            (i + 0.5) * cell_size - height / 2.0 + origin[0],
                            (j + 0.5) * cell_size - width / 2.0 + origin[1],
                            cfg.wall_height / 2.0,
                        ),
                    ),
                )
                maze_meshes.append(wall_mesh)
    # return the tri-mesh and the origins
    return [plane_mesh] + maze_meshes, np.array(origin)
