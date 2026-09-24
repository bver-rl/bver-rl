# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation curricula: fixed start/goal replay, single- and multi-goal."""

from isaaclab.utils import configclass

from ..common.env_cfg import *
from ...curriculum import EvalCfg


# Evaluation curriculum configs
@configclass
class ClimbBoxEvalCurriculumCfg:
    evaluation = EvalCfg(terrain_levels=True)
@configclass
class ClimbBoxMultiGoalEvalCurriculumCfg:
    """Like ClimbBoxEvalCurriculumCfg, but the eval queue also carries goals."""

    evaluation = EvalCfg(terrain_levels=True, goal_command_name="base_position")
