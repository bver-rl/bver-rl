from __future__ import annotations

import pickle
import torch
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ..envs.base_env_cfg import BaseEnv


class DummyLogger(object):
    def __init__(self, log_dir: str):
        pass

    def log(self, env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
        pass

    def log_trajectories(
        self, env: ManagerBasedRLEnv, trajectories: list[tuple[torch.Tensor, torch.float, int]]
    ) -> None:
        pass

    def write_logs(self) -> None:
        pass


class Logger(object):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.task_rewards = defaultdict(list)
        self.trajectories = list[tuple[torch.Tensor, float, int]]()
        self.log_interval = 99
        self.log_counter = 0

    def _step(
        self,
        counter: int,
        num_steps: int | None,
    ) -> None:

        multiplier = num_steps if num_steps is not None else 1

        if counter // (self.log_interval * multiplier) > self.log_counter:
            self.log_counter += 1
            self.write_logs()

    def log(
        self,
        env: BaseEnv,
        env_ids: torch.Tensor,
    ) -> None:

        # TODO: the issue with this approach is that each trajectory is controlled by multiple policies
        tasks = env.task_space.get_tasks(env, env_ids)
        episode_sums = (
            torch.concat([v.unsqueeze(1) for k, v in env.reward_manager._episode_sums.items()], axis=1)[env_ids]
            .sum(axis=1)
            .cpu()
        )

        # TODO: make this compatible with cartpole task space
        full_tasks = torch.concat([task for task in tasks.values()], dim=1).cpu()
        augmented_task = torch.concat(
            (full_tasks, torch.ones((full_tasks.shape[0], 1), device=full_tasks.device) * env.common_step_counter),
            axis=1,
        )

        for j in range(len(env_ids)):
            self.task_rewards[augmented_task[j]].append(episode_sums[j].item())

        self._step(env.common_step_counter, env.num_steps_per_env)

    def log_trajectories(
        self,
        env: BaseEnv,
        trajectories: list[tuple[torch.Tensor, float, int]],
    ) -> None:

        self.trajectories.extend(trajectories)
        self._step(env.common_step_counter, env.num_steps_per_env)

    def write_logs(self) -> None:
        with open(os.path.join(self.log_dir, "logs.pkl"), "wb") as f:
            pickle.dump(self.task_rewards, f)
        with open(os.path.join(self.log_dir, "trajectories.pkl"), "wb") as f:
            pickle.dump(self.trajectories, f)
