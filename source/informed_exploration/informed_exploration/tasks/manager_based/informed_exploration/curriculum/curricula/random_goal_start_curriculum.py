"""Uninformed (start, goal) control curriculum: the matched baseline for BVER.

Mirrors BVER's branch marginals with uniform draws: forward envs start from ``eval_bounds`` with pool goals,
backward envs start from the pool with the goal pinned to the anchor. ``fwd_ratio`` of one or zero gives the
single-direction ablations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.utils import configclass

from ..task_space.samplers import FeasiblePoolSamplerCfg, FilteredUniformSamplerCfg, UniformSamplerCfg
from ...utils.curriculum_utils import get_task_space, make_sampler, resolve_env_splits, resolve_xyz_dims
from ..curriculum_handlers import parallel_eval

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


class RandomGoalStartCurriculum(ManagerTermBase):

    cfg: "RandomGoalStartCfg"

    def __init__(self, cfg: "RandomGoalStartCfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)

        assert cfg.pool_path or cfg.sample_uniform, (
            "RandomGoalStartCfg needs either a pool_path (a feasible-start pool .pkl) or"
            " sample_uniform=True (draw from the task space's own filters). Terrains that fill"
            " pool_path in __post_init__ fail here if that step was missed, which is the point:"
            " silently drawing uniformly over a robot's task-space AABB spawns it mid-air."
        )
        assert not (cfg.pool_path and cfg.sample_uniform), (
            "RandomGoalStartCfg: set exactly one of pool_path and sample_uniform"
        )
        assert (
            cfg.anchor_goal is not None and len(cfg.anchor_goal) == 3
        ), "RandomGoalStartCfg.anchor_goal must be an env-relative (x, y, z), set per-terrain in __post_init__"
        assert 0.0 <= cfg.fwd_ratio <= 1.0, "RandomGoalStartCfg.fwd_ratio must lie in [0, 1]"

        self._xyz_dims = list(resolve_xyz_dims(self.task_space))

        # pool: starts and goals share one sampler; closed-form validity: separate filtered samplers
        if not cfg.sample_uniform:
            self._start_sampler = make_sampler(
                sampler_cfg=FeasiblePoolSamplerCfg(pool_path=cfg.pool_path),
                filtered_samples=False,
                subspace=None,
                task_space=self.task_space,
                device=env.device,
            )
            self._goal_sampler = self._start_sampler
        elif cfg.filter_uniform:
            self._start_sampler = make_sampler(
                sampler_cfg=FilteredUniformSamplerCfg(),
                filtered_samples=True,
                subspace=None,
                task_space=self.task_space,
                device=env.device,
            )
            self._goal_sampler = make_sampler(
                sampler_cfg=FilteredUniformSamplerCfg(use_goal_filter=True),
                filtered_samples=True,
                subspace=None,
                task_space=self.task_space,
                device=env.device,
            )
        else:
            # obstacle-agnostic uniform over the bounds, walls included (see RandomGoalStartCfg.filter_uniform)
            self._start_sampler = make_sampler(
                sampler_cfg=UniformSamplerCfg(),
                filtered_samples=False,
                subspace=None,
                task_space=self.task_space,
                device=env.device,
            )
            self._goal_sampler = self._start_sampler
        # matches BVERCurriculum's `_initial_state_sampler`: uniform over eval_bounds
        self._initial_state_sampler = make_sampler(
            sampler_cfg=None,
            filtered_samples=False,
            subspace=None,
            task_space=self.task_space,
            device=env.device,
            eval_bounds=True,
        )

        self._goal_command = env.command_manager.get_term(cfg.goal_command_name)
        self._anchor_goal = torch.as_tensor(cfg.anchor_goal, dtype=torch.float32, device=env.device)

        # resolved lazily: train_env_ratio is mirrored onto env after construction
        self._fwd_boundary: int | None = None

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        train_ids = parallel_eval(env, env_ids)
        if len(train_ids) == 0:
            return {}

        if self._fwd_boundary is None:
            num_train_envs, _, _ = resolve_env_splits(env)
            # backward takes whatever forward does not
            self._fwd_boundary = int(num_train_envs * self.cfg.fwd_ratio)

        # static per-env role by index, mirroring BVER's contiguous branch ranges
        fwd_mask = train_ids < self._fwd_boundary
        n_fwd = int(fwd_mask.sum())
        n_bwd = len(train_ids) - n_fwd

        states = torch.zeros(len(train_ids), *self.task_space.task_dim(), device=env.device)
        goals = torch.zeros(len(train_ids), 3, device=env.device)

        if n_fwd > 0:
            states[fwd_mask] = self._initial_state_sampler.sample(n_fwd)
            goals[fwd_mask] = self._sample_goal_xyz(n_fwd)
        if n_bwd > 0:
            states[~fwd_mask] = self._start_sampler.sample(n_bwd)
            goals[~fwd_mask] = self._anchor_goal.expand(n_bwd, 3)

        # non-init dims fall back to `default_task`, matching BVERCurriculum._set_tasks
        states = self.task_space.mask_to_subspace(states, self.cfg.init_subspace)

        self.task_space.set_tasks(env, train_ids, states)
        # goals are env-relative; the command term must have default_goal=None or resampling overwrites them
        self._goal_command.set_goal(train_ids, goals)

        return {"random_gs_fwd_envs": float(n_fwd), "random_gs_bwd_envs": float(n_bwd)}

    def _sample_goal_xyz(self, n: int) -> torch.Tensor:
        """``n`` goal positions: drawn states reduced to their xyz."""
        return self._goal_sampler.sample(n).reshape(n, -1)[:, self._xyz_dims]


@configclass
class RandomGoalStartCfg(CurriculumTermCfg):

    func: callable = RandomGoalStartCurriculum

    pool_path: str = ""
    """Feasible-start pool from ``generate_feasible_starts.py``; required unless ``sample_uniform`` is set."""

    sample_uniform: bool = False
    """Draw from the task space's own filters instead of a pool.

    Only valid where validity is a closed-form region (a maze ball clearing walls), not on legged robots."""

    filter_uniform: bool = True
    """Whether uniform draws respect ``sample_filter`` / ``goal_sample_filter``; ignored with a ``pool_path``.

    False allows starts and goals inside geometry, where simulator behavior is undefined."""

    fwd_ratio: float = 0.5
    """Share of training envs in the forward role (start-pad start, pool goal); the rest go backward."""

    goal_command_name: str = "base_position"
    """Command term written via ``set_goal``. Its ``default_goal`` must be ``None``."""

    anchor_goal: list[float] | None = None
    """Env-relative ``(x, y, z)`` commanded by backward envs; set per terrain in the env cfg's ``__post_init__``."""

    init_subspace: Sequence[int] | None = None
    """Task dimensions taken from the drawn state, the rest from ``default_task``; match BVER's ``init_subspace``."""
