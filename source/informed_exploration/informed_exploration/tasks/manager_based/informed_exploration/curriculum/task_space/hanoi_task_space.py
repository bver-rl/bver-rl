# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from torch import Tensor

from .hanoi_task_space_cfg import HanoiTaskSpaceCfg
from .hanoi_trajectory_layout import hanoi_obs_to_task
from .parkour_task_space import ParkourTaskSpace


class HanoiTaskSpace(ParkourTaskSpace):
    """Task space of the Franka Hanoi task.

    Reuses :class:`ParkourTaskSpace`'s ``set_tasks``; only the observation-to-task conversion differs.
    """

    def __init__(self, cfg: HanoiTaskSpaceCfg):
        super().__init__(cfg)

    def obs_to_task(self, obs: Tensor) -> Tensor:
        """Trajectory frames to task states; the ``trajectory`` group's term order is load-bearing here."""
        return hanoi_obs_to_task(obs)
