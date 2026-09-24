import os
import pickle
import torch

from ..curriculum.task_space.parkour_task_space import ParkourTaskSpace
from ..cfg.common.env_cfg import IEParkourEnvCfg
from .base_env_cfg import BaseEnv

from ..utils.logger import DummyLogger, Logger


class IEParkourEnv(BaseEnv):
    cfg: IEParkourEnvCfg
    eval_data = {}
    num_steps_per_env: int | None = None

    def __init__(self, cfg: IEParkourEnvCfg, log_dir: str, **kwargs):
        # Instantiate before super().__init__() so the managers can read it during setup.
        self.task_space = ParkourTaskSpace(cfg.task_space)
        # self.logger = Logger(log_dir=log_dir)
        self.logger = DummyLogger(log_dir=log_dir)

        super().__init__(cfg=cfg, **kwargs)
