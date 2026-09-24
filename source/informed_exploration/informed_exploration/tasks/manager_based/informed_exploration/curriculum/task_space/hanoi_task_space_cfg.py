# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.utils import configclass

from ...terrains.hanoi_geometry import HanoiGeometry
from .hanoi_trajectory_layout import HANOI_ARM_DOF
from .task_space_base_cfg import TaskSpaceBaseCfg

HANOI_RING_ROWS = 3

HANOI_XYZ_DIMENSIONS = tuple(2 * (HANOI_ARM_DOF + i) for i in range(HANOI_RING_ROWS))
"""Flat indices of the ring centre's x, y and z values in the interleaved ``[value, velocity]`` task vector."""

HANOI_INIT_SUBSPACE = [2 * row for row in range(HANOI_ARM_DOF + HANOI_RING_ROWS)]
"""Value dimensions of every row: the joint offsets that reach the sim and the derived ring centre."""

HANOI_JOINT_MARGIN = 1.0
"""How far the joint rows reach past the solved configurations, in radians."""


@configclass
class HanoiTaskSpaceCfg(TaskSpaceBaseCfg):
    """Joint-type task space for the Franka Hanoi task: arm-joint offsets from home, then the ring centre.

    Only the joint rows reach the sim; the ring rows are their forward kinematics, carried because BVER builds
    goals from ``xyz_dimensions``. Geometry is filled by :func:`hanoi_task_space_cfg`.
    """

    dimensions: tuple | None = (HANOI_ARM_DOF + HANOI_RING_ROWS, 2)
    total_dims: int | None = 2 * (HANOI_ARM_DOF + HANOI_RING_ROWS)
    xyz_dimensions: tuple | None = HANOI_XYZ_DIMENSIONS

    task_types: dict | None = {"reset_robot_joints": "joint"}
    subterm_dims: dict | None = {"reset_robot_joints": (HANOI_ARM_DOF, 2)}
    tasks: dict | None = {"event": {"reset": {"reset_robot_joints": ["position", "velocity"]}}}

    # keep-out boxes are stated on the ring centre
    filter_dimensions: tuple | None = HANOI_XYZ_DIMENSIONS

    optimal_trajectory_file: str | None = None

    marker_min_half_size: float | None = None
    """Floor on the half-extents of the drawn start-zone and bounds boxes; unset keeps the markers' own floor."""


def hanoi_task_space_cfg(
    geom: HanoiGeometry,
    joint_limits: Sequence[Sequence[float]],
    home_joint_pos: Sequence[float],
    start_joint_pos: Sequence[float],
    anchor_joint_pos: Sequence[float],
    ring_xyz_start: Sequence[float],
    ring_xyz_home: Sequence[float],
    start_box_half_width: float = 0.0,
    joint_margin: float = HANOI_JOINT_MARGIN,
) -> HanoiTaskSpaceCfg:
    """Build the task-space cfg for ``geom``, with joint rows as offsets from ``home_joint_pos``.

    ``eval_bounds`` is the start configuration widened by ``start_box_half_width``; its ring rows are pinned to
    ``ring_xyz_start``, which must be the start's true ring centre since BVER reads goal xyz off it.
    """
    bounds, eval_bounds, default_task = [], [], []
    for (lo, hi), home, start, anchor in zip(joint_limits, home_joint_pos, start_joint_pos, anchor_joint_pos):
        low = max(lo, min(home, start, anchor) - joint_margin)
        high = min(hi, max(home, start, anchor) + joint_margin)
        bounds += [[low - home, high - home], [0.0, 0.0]]
        eval_bounds += [[start - home - start_box_half_width, start - home + start_box_half_width], [0.0, 0.0]]
        default_task += [0.0, 0.0]
    for (lo, hi), start, home in zip(geom.workspace_bounds, ring_xyz_start, ring_xyz_home):
        bounds += [[lo, hi], [0.0, 0.0]]
        eval_bounds += [[start, start], [0.0, 0.0]]
        default_task += [home, 0.0]

    cfg = HanoiTaskSpaceCfg()
    cfg.bounds = bounds
    cfg.eval_bounds = eval_bounds
    cfg.sample_filter = geom.ring_keep_out
    cfg.goal_sample_filter = geom.ring_keep_out
    cfg.default_task = default_task
    # the start is one configuration, so its marker shrinks to the room the ring has around its peg
    cfg.marker_min_half_size = geom.side_clearance
    return cfg
