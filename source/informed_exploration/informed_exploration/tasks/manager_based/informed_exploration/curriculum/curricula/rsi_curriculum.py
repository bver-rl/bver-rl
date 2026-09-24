from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from ...utils.curriculum_utils import get_task_space
from ..curriculum_handlers import parallel_eval

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


class RSICurriculum(ManagerTermBase):

    cfg: "RSICfg"

    def __init__(self, cfg: "RSICfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        env_ids = parallel_eval(env, env_ids)

        if len(env_ids) == 0:
            return {}

        if self.cfg.terrain_level:
            terrain: TerrainImporter = env.scene.terrain
            levels = (
                terrain.terrain_levels[env_ids].cpu()
                if not self.cfg.one_level
                else torch.ones_like(env_ids).cpu() * self.cfg.terrain_level
            )
            types = terrain.terrain_types[env_ids].cpu()

            terrain_level_trajectories = env.train_data.get("terrain_level_trajectories", None)
            assert terrain_level_trajectories is not None, "No terrain level trajectories found in train_data"

            tasks = []
            for i in range(len(env_ids)):
                trajs = torch.stack(terrain_level_trajectories[levels[i]][types[i]], dim=0)
                traj_idx = torch.randint(0, trajs.shape[0], (1,)).cpu()
                random_trajectories = trajs[traj_idx].squeeze(0)

                random_idx = torch.randint(0, trajs.shape[1], (1,)).cpu()
                tasks.append(random_trajectories[random_idx])

            tasks = torch.stack(tasks, dim=0).to(env.device)
        else:
            trajectories = env.train_data.get("trajectories", None)
            assert trajectories is not None, "No trajectories found in train_data"

            trajectories = trajectories["trajectories"].to(env.device)

            traj_idx = torch.randint(0, trajectories.shape[0], (len(env_ids),), device=env.device)
            random_trajectories = trajectories[traj_idx]

            random_idx = torch.randint(0, trajectories.shape[1], (len(env_ids),), device=env.device)

            tasks = torch.stack([random_trajectories[i, random_idx[i]] for i in range(len(env_ids))], dim=0)
            tasks = tasks.unsqueeze(1)

        tasks = self.task_space.obs_to_task(tasks).reshape(
            len(env_ids),
            *self.task_space.task_dim(),
        )
        self.task_space.set_tasks(env, env_ids, tasks)
        return {}


@configclass
class RSICfg(CurriculumTermCfg):

    func: callable = RSICurriculum

    terrain_level: bool | int = False
    one_level: bool = False
