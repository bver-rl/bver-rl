# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""U maze arms.

Layout u_maze; shortest INIT-to-GOAL path is 24 m.

The episode is half again as long as an optimal run, which caps the solved band BVER reads against.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .maze_arms import (
    MazeEnvCfg_Baseline,
    MazeEnvCfg_EVAL,
    MazeEnvCfg_BVER,
    MazeEnvCfg_BVER_BWD,
    MazeEnvCfg_BVER_FWD,
    MazeEnvCfg_PLAY,
    MazeEnvCfg_Random,
)

MAZE_NAME = "u_maze"
EPISODE_S = 9.0
# Forward and backward walk lengths in policy steps
BROWNIAN_HORIZONS = (80, 80)

# U-only BVER knobs, overriding `_BVER_MAZE_KWARGS` on this layout's three BVER arms
BVER_OVERRIDES = {
    "r_min": 0.1,
    "r_max": 0.85,
    "goal_connect_ratio": 0.5,
    "start_connect_ratio": 0.5,
    "goal_replay_ratio": 0.2,
    "start_replay_ratio": 0.2,
}


@configclass
class UMazeEnvCfg_BVER(MazeEnvCfg_BVER):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    bver_overrides = BVER_OVERRIDES


@configclass
class UMazeEnvCfg_BVER_FWD(MazeEnvCfg_BVER_FWD):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    bver_overrides = BVER_OVERRIDES


@configclass
class UMazeEnvCfg_BVER_BWD(MazeEnvCfg_BVER_BWD):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
    bver_overrides = BVER_OVERRIDES


@configclass
class UMazeEnvCfg_Baseline(MazeEnvCfg_Baseline):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS


@configclass
class UMazeEnvCfg_Random(MazeEnvCfg_Random):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS


@configclass
class UMazeEnvCfg_EVAL(MazeEnvCfg_EVAL):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS


@configclass
class UMazeEnvCfg_PLAY(MazeEnvCfg_PLAY):
    maze_name = MAZE_NAME
    episode_s = EPISODE_S
    brownian_horizons = BROWNIAN_HORIZONS
