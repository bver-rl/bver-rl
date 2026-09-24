# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action-scale halving, shared by the climb sparse arms and the scans."""

ACTION_SCALE_HALF = 0.5
"""Action scale of the halved arms, matching the parkour base's joint-position scale.

Replaying a checkpoint at a different scale silently changes the commanded amplitude, so eval bindings pin it here.
"""


def apply_action_scale(cfg, scale: float) -> None:
    """Set the joint-position action term's scale.

    Must run after ``super().__post_init__()``, which assembles ``actions``.
    """
    cfg.actions.joint_pos.scale = scale
