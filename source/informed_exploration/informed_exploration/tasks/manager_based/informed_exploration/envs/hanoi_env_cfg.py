# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from ..cfg.hanoi.hanoi_env_cfg import HanoiEnvCfg
from ..curriculum.task_space.hanoi_task_space import HanoiTaskSpace
from ..utils.logger import DummyLogger
from .base_env_cfg import BaseEnv


class HanoiEnv(BaseEnv):
    """Franka Hanoi environment; one class for every layout and arm."""

    cfg: HanoiEnvCfg
    eval_data = {}
    num_steps_per_env: int | None = None

    def __init__(self, cfg: HanoiEnvCfg, log_dir: str, **kwargs):
        # Instantiate before super().__init__(): the curricula read the task space during manager setup.
        self.task_space = HanoiTaskSpace(cfg.task_space)
        self.logger = DummyLogger(log_dir=log_dir)

        super().__init__(cfg=cfg, **kwargs)
