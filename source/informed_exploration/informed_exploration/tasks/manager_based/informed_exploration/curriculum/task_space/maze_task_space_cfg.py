# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from ...terrains.maze_geometry import MazeGeometry
from .task_space_base_cfg import TaskSpaceBaseCfg

MAZE_BALL_RADIUS = 0.5
"""Radius of the point-mass ball, in metres."""

MAZE_BALL_Z = MAZE_BALL_RADIUS
"""Env-relative height of the ball's centre when it rests on the floor."""


@configclass
class MazeTaskSpaceCfg(TaskSpaceBaseCfg):
    """Root-type task space for the point-mass maze: ``[x, vx, y, vy, z, vz]``.

    Geometry is filled by :func:`maze_task_space_cfg`. z is pinned to the resting height but kept, since
    BVER builds goals from a fixed ``xyz_dimensions`` triple.
    """

    dimensions: tuple | None = (3, 2)
    total_dims: int | None = 6
    xyz_dimensions: tuple | None = (0, 2, 4)

    task_types: dict | None = {"reset_base": "root"}
    subterm_dims: dict | None = {"reset_base": (3, 2)}
    tasks: dict | None = {
        "event": {
            "reset": {
                "reset_base": {
                    "pose": ["x", "y", "z"],
                    "velocity": ["x", "y", "z"],
                },
            }
        }
    }

    # only x and y are filtered; z and the velocities are pinned
    filter_dimensions: tuple | None = (0, 2)

    optimal_trajectory_file: str | None = None


def maze_task_space_cfg(geom: MazeGeometry, z: float = MAZE_BALL_Z) -> MazeTaskSpaceCfg:
    """Build the task-space cfg for ``geom``: interior bounds, wall keep-out filter and INIT-cell eval bounds.

    ``eval_bounds`` spans only the INIT cell, so every arm's headline task is the same start-cell-to-goal-cell run.
    """
    x_lo, x_hi = geom.x_bounds
    y_lo, y_hi = geom.y_bounds
    (init_x_lo, init_x_hi), (init_y_lo, init_y_hi) = geom.init_region

    cfg = MazeTaskSpaceCfg()
    cfg.bounds = [[x_lo, x_hi], [0.0, 0.0], [y_lo, y_hi], [0.0, 0.0], [z, z], [0.0, 0.0]]
    cfg.eval_bounds = [
        [init_x_lo, init_x_hi],
        [0.0, 0.0],
        [init_y_lo, init_y_hi],
        [0.0, 0.0],
        [z, z],
        [0.0, 0.0],
    ]
    # A ball centre outside every wall box is clear of every wall, so start and goal filters coincide.
    cfg.sample_filter = geom.wall_keep_out
    cfg.goal_sample_filter = geom.wall_keep_out
    cfg.default_task = [0.0, 0.0, 0.0, 0.0, z, 0.0]
    return cfg
