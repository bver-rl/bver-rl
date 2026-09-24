# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from ..cfg.maze.maze_env_cfg import MazeEnvCfg
from ..curriculum.task_space.maze_task_space import MazeTaskSpace
from ..utils.logger import DummyLogger
from .base_env_cfg import BaseEnv


class MazeEnv(BaseEnv):
    """Point-mass maze environment; one class for every layout and arm."""

    cfg: MazeEnvCfg
    eval_data = {}
    num_steps_per_env: int | None = None

    def __init__(self, cfg: MazeEnvCfg, log_dir: str, **kwargs):
        # Instantiate before super().__init__(): the curricula read the task space during manager setup.
        self.task_space = MazeTaskSpace(cfg.task_space)
        self.logger = DummyLogger(log_dir=log_dir)

        super().__init__(cfg=cfg, **kwargs)
