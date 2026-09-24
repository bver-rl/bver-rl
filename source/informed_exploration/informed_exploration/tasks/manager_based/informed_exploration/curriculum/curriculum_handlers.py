from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from ..utils.curriculum_utils import (
    get_task_space,
    make_sampler,
    resolve_anchor_goal_xyz,
    resolve_env_splits,
)
from .task_space.samplers import TaskSpaceSamplerCfg

if TYPE_CHECKING:
    from ..envs.base_env_cfg import BaseEnv


def random_eval_curriculum(
    env: BaseEnv,
    env_ids: torch.Tensor,
    filtered_samples: bool = False,
    subspace: Sequence[int] | None = None,
    sampler_cfg: TaskSpaceSamplerCfg | None = None,
) -> None:
    """Initialize tasks uniformly within the eval task bounds.

    Args:
        env: Env to set the tasks for.
        env_ids: Env ids to set the tasks for.
        filtered_samples: Whether to filter the samples.
        subspace: Flat task dimensions to sample; ``None`` samples all.
        sampler_cfg: Optional sampler config; must have ``eval_bounds=True`` (not enforced).
    """

    # generate random tasks
    task_space = get_task_space(env)

    # lazy-init sampler (cached on env to avoid re-construction every call)
    _cache_attr = "_curr_sampler_random_eval"
    if not hasattr(env, _cache_attr):
        setattr(
            env,
            _cache_attr,
            make_sampler(sampler_cfg, filtered_samples, subspace, task_space, env.device, eval_bounds=True),
        )
    sampler = getattr(env, _cache_attr)

    scaled_tasks = sampler.sample(len(env_ids))
    task_space.set_tasks(env, env_ids, scaled_tasks)


class EvalCurriculum(ManagerTermBase):

    cfg: "EvalCfg"

    def __init__(self, cfg: "EvalCfg", env: BaseEnv):
        self.cfg = cfg

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        eval_data = getattr(env, "eval_data", None)
        assert eval_data is not None, "env does not have eval_data attribute"

        if eval_data != {}:
            task_space = get_task_space(env)
            dims = task_space.task_dim()

            goal_command = None
            if self.cfg.goal_command_name is not None:
                goal_command = env.command_manager.get_term(self.cfg.goal_command_name)

            if eval_data["initialized"]:
                alive_ids = ~eval_data["finished"][env_ids.cpu()]
                env_ids_alive = env_ids[alive_ids]

                episode_rewards = torch.concat(
                    [v.unsqueeze(1) for _, v in env.reward_manager._episode_sums.items()], axis=1
                )[env_ids_alive].cpu()
                episode_sums = episode_rewards.sum(axis=1)
                solved_tasks = eval_data["old_tasks"][env_ids_alive].cpu()

                if goal_command is not None:
                    solved_goals = eval_data["old_goals"][env_ids_alive].cpu()

                if eval_data.get("trajectories", None) is not None:
                    trajectories = env.observation_manager._obs_buffer["trajectory"][env_ids_alive]
                    trajectory_lengths = env.episode_length_buf[env_ids_alive]

                for i in range(len(env_ids_alive)):
                    if eval_data["results"] is not None:
                        eval_data["results"][solved_tasks[i]].append(episode_sums[i].item())
                    eval_data["rewards"].append(episode_rewards[i])
                    # starts are recorded even without a goal command, since fixed-goal evals report by start
                    eval_data["starts"].append(solved_tasks[i])
                    if goal_command is not None:
                        eval_data["goals"].append(solved_goals[i])

                    if eval_data.get("trajectories", None) is not None:
                        if not self.cfg.terrain_levels:
                            eval_data["trajectories"].append(
                                (trajectories[i][-trajectory_lengths[i] :].cpu(), episode_sums[i].item())
                            )
                        else:
                            terrain: TerrainImporter = env.scene.terrain
                            eval_data["trajectories"].append(
                                (
                                    terrain.terrain_levels[env_ids_alive[i]].cpu(),
                                    terrain.terrain_types[env_ids_alive[i]].cpu(),
                                    trajectories[i][-trajectory_lengths[i] :].cpu(),
                                    episode_sums[i].item(),
                                )
                            )

            queue = eval_data["queue"]
            num_tasks = min(len(queue), len(env_ids))

            eval_data["finished"][env_ids[num_tasks:]] = True
            if num_tasks == 0:
                if eval_data["finished"].all():
                    eval_data["done"] = True
                return {}
            else:
                tasks = torch.zeros(len(env_ids), *dims, device=env.device)
                if goal_command is not None:
                    # queue entries are goal_xyz(3) followed by the flattened start state, as in BVER's pairs
                    goals = torch.zeros(len(env_ids), 3, device=env.device)
                    for i in range(num_tasks):
                        pair = queue.popleft()
                        goals[i] = pair[:3]
                        tasks[i] = pair[3:].reshape(*dims)
                else:
                    tasks[:num_tasks] = torch.stack(
                        [queue.popleft().reshape(*dims) for _ in range(num_tasks)], dim=0
                    )

                eval_data["old_tasks"][env_ids[:num_tasks]] = tasks[:num_tasks]
                task_space.set_tasks(env, env_ids, tasks)

                if goal_command is not None:
                    eval_data["old_goals"][env_ids[:num_tasks]] = goals[:num_tasks]
                    goal_command.set_goal(env_ids[:num_tasks], goals[:num_tasks])

                eval_data["initialized"] = True
        return {}


@configclass
class EvalCfg(CurriculumTermCfg):

    func: callable = EvalCurriculum

    terrain_levels: bool = False

    goal_command_name: str | None = None
    """When set, the queue holds (goal_xyz, start_state) pairs and each goal is written to this command term.
    ``None`` means the queue holds plain start states."""


def parallel_eval(
    env: BaseEnv,
    env_ids: torch.Tensor,
    return_random_env_ids: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Split env_ids into train, eval and random ids and apply the eval curriculum to the eval envs.

    Args:
        env: Env to reset.
        env_ids: Env ids being reset.
        return_random_env_ids: Whether to also return the random env ids.

    Returns:
        The train env ids, and the random env ids if requested.
    """

    num_train_envs, num_random_envs, num_eval_envs = resolve_env_splits(env)

    train_env_ids = env_ids[env_ids < num_train_envs]
    eval_env_ids = env_ids[(env_ids >= num_train_envs) & (env_ids < num_train_envs + num_eval_envs)]
    random_env_ids = env_ids[env_ids >= num_train_envs + num_eval_envs]

    if 1 - env.train_env_ratio - env.random_env_ratio > 1e-6:
        random_eval_curriculum(
            env,
            eval_env_ids,
            filtered_samples=True,
        )
        _command_anchor_goal(env, eval_env_ids)
    if return_random_env_ids:
        return train_env_ids, random_env_ids

    return train_env_ids


def _command_anchor_goal(env: BaseEnv, env_ids: torch.Tensor) -> None:
    """Command the curriculum's anchor goal on ``env_ids``, the held-out eval slice, which otherwise has no goal.

    A no-op for command terms without ``set_goal``, with a ``default_goal``, or when there is no anchor.
    """
    if len(env_ids) == 0:
        return

    init_cfg = getattr(getattr(env, "curriculum_manager", None), "cfg", None)
    name = "base_position"
    for term in vars(init_cfg).values() if init_cfg is not None else ():
        name = getattr(term, "goal_command_name", None) or name

    try:
        goal_term = env.command_manager.get_term(name)
    except (KeyError, AttributeError, ValueError):
        return
    if not hasattr(goal_term, "set_goal") or getattr(goal_term.cfg, "default_goal", None) is not None:
        return

    anchor = resolve_anchor_goal_xyz(env)
    if anchor is None:
        return
    goal_term.set_goal(env_ids, anchor.unsqueeze(0).expand(len(env_ids), 3))
