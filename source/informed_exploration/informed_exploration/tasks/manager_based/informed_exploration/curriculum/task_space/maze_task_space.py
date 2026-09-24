# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import Tensor

from .maze_task_space_cfg import MazeTaskSpaceCfg
from .parkour_task_space import ParkourTaskSpace


class MazeTaskSpace(ParkourTaskSpace):
    """Task space of the point-mass maze.

    Reuses :class:`ParkourTaskSpace`'s ``set_tasks`` and ``get_tasks``; only the observation-to-task
    conversion differs.
    """

    def __init__(self, cfg: MazeTaskSpaceCfg):
        super().__init__(cfg)

    def obs_to_task(self, obs: Tensor) -> Tensor:
        """Convert ``(N, 1, 6)`` trajectory frames ``[pos xyz | lin vel xyz]`` to ``(N, 1, 3, 2)`` tasks.

        Relies on the term order of the ``trajectory`` observation group.
        """
        return torch.stack([obs[..., :3], obs[..., 3:6]], dim=-1)
