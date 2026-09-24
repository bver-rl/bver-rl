# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout of the Hanoi ``trajectory`` observation group and its conversion to task states.

Kept apart from the task-space class so the conversion can be checked without the package import chain.
"""

from __future__ import annotations

import torch
from torch import Tensor

HANOI_ARM_DOF = 7

TRAJ_JOINT_POS = slice(0, HANOI_ARM_DOF)
TRAJ_JOINT_VEL = slice(HANOI_ARM_DOF, 2 * HANOI_ARM_DOF)
TRAJ_RING_POS = slice(2 * HANOI_ARM_DOF, 2 * HANOI_ARM_DOF + 3)
TRAJ_DIM = 2 * HANOI_ARM_DOF + 3


def hanoi_obs_to_task(obs: Tensor) -> Tensor:
    """Convert ``(..., 17)`` trajectory frames to ``(..., 10, 2)`` task states.

    Arm-joint rows carry offsets and velocities; the ring-centre rows carry a zero velocity.
    """
    joints = torch.stack([obs[..., TRAJ_JOINT_POS], obs[..., TRAJ_JOINT_VEL]], dim=-1)
    ring = obs[..., TRAJ_RING_POS]
    ring = torch.stack([ring, torch.zeros_like(ring)], dim=-1)
    return torch.cat([joints, ring], dim=-2)
