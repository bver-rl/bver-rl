from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.utils import configclass

from ..task_space.samplers import TaskSpaceSamplerCfg
from ...utils.curriculum_utils import get_task_space, make_sampler
from ..curriculum_handlers import parallel_eval

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


class RandomCurriculum(ManagerTermBase):

    cfg: "RandomCfg"

    def __init__(self, cfg: "RandomCfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)
        self._sampler = make_sampler(cfg.sampler_cfg, cfg.filtered_samples, cfg.subspace, self.task_space, env.device)

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        env_ids = parallel_eval(env, env_ids)

        if len(env_ids) == 0:
            return {}

        scaled_tasks = self._sampler.sample(len(env_ids))
        self.task_space.set_tasks(env, env_ids, scaled_tasks)

        return {}


@configclass
class RandomCfg(CurriculumTermCfg):

    func: callable = RandomCurriculum

    filtered_samples: bool = False
    subspace: Sequence[int] | None = None
    sampler_cfg: TaskSpaceSamplerCfg | None = None
