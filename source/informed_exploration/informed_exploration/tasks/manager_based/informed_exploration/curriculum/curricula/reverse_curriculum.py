from __future__ import annotations

from collections import deque, defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.utils import configclass

from ...utils.curriculum_utils import (
    get_task_space,
    compute_episode_sums,
    task_bbox,
)
from ..curriculum_handlers import parallel_eval

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


class ReverseCurriculum(ManagerTermBase):

    cfg: "RCCfg"

    def __init__(self, cfg: "RCCfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)

        self.iterations = 0
        self.random_starts = []
        goal_tensor = torch.tensor(self.cfg.goal, device=env.device)
        self.starts = [goal_tensor]
        self.old_starts = deque([goal_tensor], maxlen=self.cfg.max_old_starts)
        self.good_starts = deque([goal_tensor], maxlen=self.cfg.max_random_walk_seeds)
        self.starts_rewards = defaultdict(list)
        # Seed the goal with R_max so it is always kept as a good start
        self.starts_rewards[tuple(map(tuple, goal_tensor.tolist()))] = [self.cfg.R_max]
        self.new_frontier_points = 0.0
        self.num_frontier_candidates = 0
        self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        train_ids, random_ids = parallel_eval(env, env_ids, return_random_env_ids=True)

        if env.common_step_counter == 0:
            return {}

        # Collect brownian motion end states
        if env.common_step_counter > 1:
            all_random_ids = torch.arange(
                int(env.num_envs * (1 - env.random_env_ratio)), env.num_envs, device=env.device
            )
            valid_mask = env.episode_length_buf[all_random_ids] != 1
            valid_ids = all_random_ids[valid_mask & self.initialized[all_random_ids]]

            if len(valid_ids) > 0:
                raw_obs_random = env.observation_manager._obs_buffer["trajectory"][valid_ids, -1].reshape(
                    len(valid_ids), 1, -1
                )

                batched_random_starts = self.task_space.obs_to_task(raw_obs_random).reshape(
                    len(valid_ids), *self.task_space.task_dim()
                )
                self.random_starts.extend(list(batched_random_starts))

        # evaluate random steps and update starts
        if len(self.random_starts) >= self.cfg.num_random_steps:

            if self.iterations % 1 == 0:  # TODO: remove?

                # training starts
                new_starts = torch.stack(self.random_starts, dim=0)[
                    np.random.randint(0, len(self.random_starts), self.cfg.N_new)
                ]
                new_starts = [t for t in new_starts]
                old_starts_list = list(self.old_starts)
                old_starts = torch.stack(old_starts_list, dim=0)[
                    np.random.randint(0, len(old_starts_list), self.cfg.N_old)
                ]
                old_starts = [t for t in old_starts]

                self.starts = new_starts + old_starts

                # brownian starts
                new_good_starts = [
                    torch.tensor(start).to(env.device)
                    for start, rews in self.starts_rewards.items()
                    if self.cfg.R_min <= np.mean(rews) <= self.cfg.R_max
                ]
                self.new_frontier_points = len(new_good_starts)
                self.num_frontier_candidates = len(self.starts_rewards)

                if len(new_good_starts) > 0:
                    self.good_starts.extend(new_good_starts)
                    self.old_starts.extend(new_good_starts)

                self.random_starts = []
                self.starts_rewards.clear()
            self.iterations += 1

        # Record start-reward pairs from trajectories
        episode_sums = compute_episode_sums(env, train_ids, self.cfg.reward_terms)

        train_ids_ = train_ids[self.initialized[train_ids]]

        if len(train_ids_) > 0:
            # filter out uninitialized envs
            step_idx = -env.episode_length_buf[train_ids_]

            raw_obs_train = env.observation_manager._obs_buffer["trajectory"][train_ids_, step_idx].reshape(
                len(train_ids_), 1, -1
            )

            batched_trained_starts = self.task_space.obs_to_task(raw_obs_train).reshape(
                len(train_ids_), *self.task_space.task_dim()
            )
            trained_starts = list(batched_trained_starts)

            # make tensors hashable for unique dict keys
            [
                self.starts_rewards[tuple(map(tuple, start.tolist()))].append(reward.item())
                for start, reward in zip(trained_starts, episode_sums)
            ]

        # Sample new starts, first for random envs
        if len(random_ids) > 0:
            good_starts_seeds = torch.stack(list(self.good_starts), dim=0)[
                np.random.randint(0, len(self.good_starts), len(random_ids))
            ]

            # small noise works around a physics collision bug
            if len(self.good_starts) == 1:
                good_starts_seeds += torch.randn_like(good_starts_seeds, device=env.device) * 1e-3

            self.task_space.set_tasks(env, random_ids, good_starts_seeds)
            self.initialized[random_ids] = True

        # set tasks for training envs
        if len(self.starts) > 0:
            new_starts = torch.stack(self.starts, dim=0)[np.random.randint(0, len(self.starts), len(train_ids))]
            self.task_space.set_tasks(env, train_ids, new_starts)
            self.initialized[train_ids] = True

        acceptance_rate = (
            self.new_frontier_points / self.num_frontier_candidates if self.num_frontier_candidates > 0 else 0.0
        )
        xyz_dims = (
            getattr(self.task_space.cfg, "xyz_dimensions", (0, 1, 2)) if hasattr(self.task_space, "cfg") else (0, 1, 2)
        )
        return {
            "num_brownian_seeds": len(self.good_starts),
            "num_new_frontier_points": self.new_frontier_points,
            "num_frontier_candidates": self.num_frontier_candidates,
            "frontier_acceptance_rate": acceptance_rate,
            "RC iterations": self.iterations,
            **task_bbox(self.starts, xyz_dims, "starts"),
            **task_bbox(self.random_starts, xyz_dims, "random_starts"),
            **task_bbox(self.good_starts, xyz_dims, "good_starts"),
        }


@configclass
class RCCfg(CurriculumTermCfg):

    func: callable = ReverseCurriculum

    N_old: int = 1000
    N_new: int = 2000
    R_min: float = 0.0
    R_max: float = 4.9
    brownian_horizon: int = 300
    num_random_steps: int = 100_000
    max_random_walk_seeds: int = 400
    max_old_starts: int = 100_000

    reward_terms: str | list[str] = "all"

    goal: list[list[float, float]] = [
        [-8.0, 0.0],
        [8.0, 0.0],
    ]
