# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hanoi layout P12 arms: the ring moves from the bottom of the left peg to the bottom of the right peg.

Layouts are named by their peg height in centimetres, the dial that sets how long each funnel is.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .hanoi_arms import (
    HanoiEnvCfg_Baseline,
    HanoiEnvCfg_EVAL,
    HanoiEnvCfg_BVER,
    HanoiEnvCfg_BVER_BWD,
    HanoiEnvCfg_BVER_FWD,
    HanoiEnvCfg_PLAY,
    HanoiEnvCfg_Random,
)
from .hanoi_env_cfg import hanoi_poses_relpath

LAYOUT = "p12"
EPISODE_S = 8.0
# Forward and backward walk lengths in policy steps
BROWNIAN_HORIZONS = (10, 10)
POSES_PATH = hanoi_poses_relpath(LAYOUT)

# P12-only BVER knobs, overriding `_BVER_HANOI_KWARGS` on this layout's three BVER arms
BVER_OVERRIDES = {
    # "goal_connect_ratio": 0.5,
    # "start_connect_ratio": 0.5,
}


@configclass
class HanoiP12EnvCfg_BVER(HanoiEnvCfg_BVER):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH
    bver_overrides = {
        "p_goal_candidate": 0.8,
        "p_start_candidate": 0.8,
        "p_goal_from_solved": 0.2,
        "p_start_from_solved": 0.2,
        **BVER_OVERRIDES,
    }


@configclass
class HanoiP12EnvCfg_BVER_FWD(HanoiEnvCfg_BVER_FWD):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH
    bver_overrides = BVER_OVERRIDES


@configclass
class HanoiP12EnvCfg_BVER_BWD(HanoiEnvCfg_BVER_BWD):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH
    bver_overrides = BVER_OVERRIDES


@configclass
class HanoiP12EnvCfg_Baseline(HanoiEnvCfg_Baseline):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH


@configclass
class HanoiP12EnvCfg_Random(HanoiEnvCfg_Random):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH


@configclass
class HanoiP12EnvCfg_EVAL(HanoiEnvCfg_EVAL):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH


@configclass
class HanoiP12EnvCfg_PLAY(HanoiEnvCfg_PLAY):
    hanoi_layout = LAYOUT
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    poses_path = POSES_PATH
