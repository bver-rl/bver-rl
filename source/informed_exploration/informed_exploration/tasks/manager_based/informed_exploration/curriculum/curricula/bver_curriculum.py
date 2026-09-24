from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.utils import configclass

from ...utils.curriculum_utils import (
    get_task_space,
    make_sampler,
    compute_episode_sums,
    random_tail_boundaries,
    resolve_env_splits,
    resolve_xyz_dims,
)
from ..curriculum_handlers import TaskSpaceSamplerCfg, parallel_eval
from .similarity_buffer import SimilarityBuffer

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


class BVERCurriculum(ManagerTermBase):
    """Unified bidirectional, goal-conditioned CURE with ratio-controlled forward and backward branches.

    Train envs split into anchor, forward (initial start, expanded goal) and backward (expanded start, anchor
    goal) groups. Pair buffers store ``concat(goal_xyz, state.flatten())``.
    """

    cfg: "BVERCfg"

    def __init__(self, cfg: "BVERCfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)

        # achievable-reward scale for r_min/r_max banding, computed lazily (see _compute_reward_band_norm)
        self._reward_band_norm: float | None = None

        # sanity check: 5 branch ratios must sum to 1
        total_env_ratio = (
            self.cfg.anchor_envs_ratio
            + self.cfg.fwd_explore_envs_ratio
            + self.cfg.fwd_exploit_envs_ratio
            + self.cfg.bwd_explore_envs_ratio
            + self.cfg.bwd_exploit_envs_ratio
        )
        assert abs(total_env_ratio - 1.0) < 1e-6, (
            "anchor_envs_ratio, fwd_explore_envs_ratio, fwd_exploit_envs_ratio, bwd_explore_envs_ratio "
            f"and bwd_exploit_envs_ratio must sum to 1.0 (got {total_env_ratio})"
        )
        # backward walks also need the anchor when random_backward_ratio > 0, checked lazily in __call__
        if self.cfg.anchor_envs_ratio > 0 or self.cfg.bwd_explore_envs_ratio > 0 or self.cfg.bwd_exploit_envs_ratio > 0:
            assert (
                self.cfg.anchor_goal is not None
            ), "BVERCfg.anchor_goal must be set (full task-space state) when anchor/bwd ratios > 0"
        # connect partitions each direction's walk-seeding modes (remainder is Voronoi)
        assert 0.0 <= self.cfg.goal_connect_ratio <= 1.0, "goal_connect_ratio must be in [0, 1]"
        assert 0.0 <= self.cfg.start_connect_ratio <= 1.0, "start_connect_ratio must be in [0, 1]"
        # from_solved + candidate partition each explore branch; the remainder is the fallback chain
        assert (
            self.cfg.p_start_from_solved + self.cfg.p_goal_candidate <= 1.0 + 1e-6
        ), "p_start_from_solved + p_goal_candidate must be <= 1 (fwd_explore proposal-source partition)"
        assert (
            self.cfg.p_goal_from_solved + self.cfg.p_start_candidate <= 1.0 + 1e-6
        ), "p_goal_from_solved + p_start_candidate must be <= 1 (bwd_explore proposal-source partition)"

        # init_subspace: dims that reach the sim and the start novelty key; must cover the xyz search dims
        self._xyz_dims = resolve_xyz_dims(self.task_space)
        # derived so the projection widths cannot desync from the SimilarityBuffer dim
        self._search_dim = len(self._xyz_dims)
        if self.cfg.init_subspace is not None:
            assert self.task_space.cfg.default_task is not None, (
                "BVERCfg.init_subspace requires the task space's cfg.default_task to be set "
                "(it fills dims outside init_subspace; see TaskSpaceBase.mask_to_subspace)"
            )
            assert set(self._xyz_dims) <= set(self.cfg.init_subspace), (
                f"BVERCfg.init_subspace must be a superset of the task space's xyz dims "
                f"({self._xyz_dims}), got {self.cfg.init_subspace}"
            )
        self._init_subspace = self.cfg.init_subspace

        # subspace weights apply only to the start-side novelty metric
        self._subspace_weights = None
        if self.cfg.subspace_weights is not None:
            assert self.cfg.init_subspace is not None, "subspace_weights requires init_subspace to be set"
            assert len(self.cfg.subspace_weights) == len(self.cfg.init_subspace), (
                f"subspace_weights ({len(self.cfg.subspace_weights)}) must match init_subspace "
                f"length ({len(self.cfg.init_subspace)})"
            )
            weights = torch.tensor(self.cfg.subspace_weights, device=env.device, dtype=torch.float32)
            assert (weights >= 0).all(), "subspace_weights must be non-negative"
            self._subspace_weights = weights

        # pair layout: concat(goal_xyz, state.flatten()); _pair_state_xyz_dims index the state xyz inside a pair
        self._state_flat_dim = math.prod(self.task_space.task_dim())
        self._pair_dim = self._search_dim + self._state_flat_dim
        self._pair_state_xyz_dims = tuple(self._search_dim + d for d in self._xyz_dims)

        # novelty key dim, shared by all four start-side buffers
        self._novelty_dim = (
            len(self.cfg.init_subspace) if self.cfg.init_subspace is not None else self.task_space.total_task_dim()
        )

        # Datastructures: search is always xyz; start buffers dedup on init_subspace

        # goal-side pair buffers: one xyz-only key serving both search and novelty
        self.goal_frontier_buffer = SimilarityBuffer(
            dim=self._search_dim,
            max_size=self.cfg.goal_frontier_buffer_size,
            item_shape=(self._pair_dim,),
            projection=self._project_goal,
            scale=0.25,
        )
        self.goal_solved_buffer = SimilarityBuffer(
            dim=self._search_dim,
            max_size=self.cfg.goal_solved_buffer_size,
            item_shape=(self._pair_dim,),
            projection=self._project_goal,
        )
        self.mixed_goal_buffer = SimilarityBuffer(
            dim=self._search_dim,
            max_size=self.cfg.mixed_goal_buffer_size,
            item_shape=(self._pair_dim,),
            projection=self._project_goal,
        )
        self.goal_candidate_buffer = SimilarityBuffer(
            dim=self._search_dim,
            max_size=self.cfg.goal_candidate_buffer_size,
            item_shape=(self._pair_dim,),
            projection=self._project_goal,
        )

        # start-side pair buffers
        self.start_frontier_buffer = self._new_start_buffer(self.cfg.start_frontier_buffer_size, scale=0.25)
        self.start_solved_buffer = self._new_start_buffer(self.cfg.start_solved_buffer_size)
        self.mixed_start_buffer = self._new_start_buffer(self.cfg.mixed_start_buffer_size)
        self.start_candidate_buffer = self._new_start_buffer(self.cfg.start_candidate_buffer_size)

        # canonical anchor (goal_xyz, full_state), drawn directly by anchor/bwd_explore
        self._anchor_state: torch.Tensor | None = None
        self._anchor_xyz: torch.Tensor | None = None
        if self.cfg.anchor_goal is not None:
            self._anchor_state = torch.tensor(self.cfg.anchor_goal, device=env.device, dtype=torch.float32)
            self._anchor_xyz = self._extract_xyz(self._anchor_state)

            # reverse-curriculum seed on the start side only
            anchor_pair = self._make_pair(self._anchor_xyz, self._anchor_state).unsqueeze(0)
            self.start_frontier_buffer.add(anchor_pair)
            self.start_solved_buffer.add(anchor_pair)

        # curriculum-controlled goal command term (see mdp.commands.CurriculumGoalCommand)
        self._goal_command = env.command_manager.get_term(self.cfg.goal_command_name)

        self._commanded_pairs = torch.zeros(env.num_envs, self._pair_dim, device=env.device)
        self._walk_goals = torch.zeros(env.num_envs, 3, device=env.device)

        # general
        self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.iterations = 0
        # env splits are computed lazily on the first __call__ (ratios on env are not final here)
        self.anchor_envs: torch.Tensor | None = None
        self.fwd_explore_envs: torch.Tensor | None = None
        self.fwd_exploit_envs: torch.Tensor | None = None
        self.bwd_explore_envs: torch.Tensor | None = None
        self.bwd_exploit_envs: torch.Tensor | None = None
        # step of the last mixed rebuilds; compared with >= since __call__ runs on resets only
        self._last_goal_update = 0
        self._last_start_update = 0

        # per-window throughput counters, reset in _update_metrics
        self._new_goal_candidate_count = 0
        self._new_start_candidate_count = 0
        self._new_goal_frontier_count = 0
        self._new_goal_solved_count = 0
        self._new_start_frontier_count = 0
        self._new_start_solved_count = 0
        self._metrics: dict[str, float] = {}
        self._last_metrics_step = -1

        # self._sampler only feeds xyz attractor queries; filtering still matters since filters are positional
        self._sampler = make_sampler(
            self.cfg.sampler_cfg,
            self.cfg.filtered_voronoi_samples,
            self.cfg.init_subspace,
            self.task_space,
            env.device,
        )
        self._initial_state_sampler = make_sampler(
            sampler_cfg=None,
            filtered_samples=False,
            eval_bounds=True,
            device=env.device,
            subspace=None,
            task_space=self.task_space,
        )

    def _new_start_buffer(self, max_size: int, scale: float = 0.7) -> SimilarityBuffer:
        """A start-side pair buffer: xyz search on the state half, init_subspace novelty."""
        return SimilarityBuffer(
            dim=self._search_dim,
            max_size=max_size,
            item_shape=(self._pair_dim,),
            projection=self._project_start_search,
            scale=scale,
            novelty_dim=self._novelty_dim,
            novelty_projection=self._project_start_novelty,
        )

    # Main per-reset entry point

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        train_ids, random_ids = parallel_eval(env, env_ids, return_random_env_ids=True)

        # lazy computation of the 5-way env split as the ratios are not available until first call
        if not self.initialized.any():
            (
                self.anchor_envs,
                self.fwd_explore_envs,
                self.fwd_exploit_envs,
                self.bwd_explore_envs,
                self.bwd_exploit_envs,
            ) = self._split_train_envs(env)
            # tail ratios are mirrored onto env after construction, so these guards live here
            if getattr(env, "random_backward_ratio", 0.0) > 0:
                assert self.cfg.anchor_goal is not None, (
                    "BVERCfg.anchor_goal must be set (full task-space state) when the runner's "
                    "random_backward_ratio > 0"
                )
            assert getattr(env, "random_reachability_ratio", 0.0) == 0.0, (
                "BVER has no reachability walks; the runner's random_reachability_ratio must be 0.0 "
                "or those tail envs would never be seeded"
            )

        current_anchor_env_ids = self._ids_in_range(train_ids, self.anchor_envs, self.anchor_envs)
        current_fwd_explore_env_ids = self._ids_in_range(train_ids, self.fwd_explore_envs, self.fwd_explore_envs)
        current_fwd_exploit_env_ids = self._ids_in_range(train_ids, self.fwd_exploit_envs, self.fwd_exploit_envs)
        current_bwd_explore_env_ids = self._ids_in_range(train_ids, self.bwd_explore_envs, self.bwd_explore_envs)
        current_bwd_exploit_env_ids = self._ids_in_range(train_ids, self.bwd_exploit_envs, self.bwd_exploit_envs)

        # runner-owned random-action tail split [forward_walk | backward_walk]
        _, backward_start, _ = random_tail_boundaries(env)
        forward_walk_reset_ids = random_ids[random_ids < backward_start]
        backward_walk_reset_ids = random_ids[random_ids >= backward_start]

        # Data collection
        if env.common_step_counter > 0:
            forward_walk_states = self._collect_random_walks(env, forward_walk_reset_ids)
            backward_walk_states = self._collect_random_walks(env, backward_walk_reset_ids)

            if forward_walk_states is not None:
                # self-referential harvest: each visited state paired with its OWN xyz
                forward_walk_pairs = self._make_pair(self._extract_xyz(forward_walk_states), forward_walk_states)
                self._new_goal_candidate_count += self.goal_candidate_buffer.add_novel(
                    forward_walk_pairs, min_sq_dist=self.cfg.goal_candidate_novelty_min_dist**2
                )

            if backward_walk_states is not None:
                # inherited harvest: pair each state with its walk's seed goal
                inherited_goals = self._walk_goal_labels(env, backward_walk_reset_ids)
                if inherited_goals is not None:
                    backward_walk_pairs = self._make_pair(inherited_goals, backward_walk_states)
                    self._new_start_candidate_count += self.start_candidate_buffer.add_novel(
                        backward_walk_pairs, min_sq_dist=self.cfg.start_candidate_novelty_min_dist**2
                    )

            # policy rollouts: rewards are matched against the remembered commanded goal
            collected = self._collect_policy_trajectories(env, train_ids)
            if collected is not None:
                valid_ids, achieved_starts, rewards = collected
                commanded_pairs = self._commanded_pairs[valid_ids]
                commanded_goal, _ = self._split_pair(commanded_pairs)
                is_fwd, is_bwd = self._role_masks(valid_ids)
                skill_min_sq_dist = (self.cfg.novelty_min_dist / 100) ** 2

                if is_fwd.any():
                    fwd_pairs = commanded_pairs[is_fwd]
                    solved_g, frontier_g = self._identify_solved_and_frontier_pairs(fwd_pairs, rewards[is_fwd])
                    if len(solved_g) > 0:
                        self._new_goal_solved_count += self.goal_solved_buffer.add_novel(
                            solved_g, min_sq_dist=skill_min_sq_dist
                        )
                    if len(frontier_g) > 0:
                        self._new_goal_frontier_count += self.goal_frontier_buffer.add_novel(
                            frontier_g, min_sq_dist=skill_min_sq_dist
                        )
                if is_bwd.any():
                    bwd_pairs = self._make_pair(commanded_goal[is_bwd], achieved_starts[is_bwd])
                    solved_s, frontier_s = self._identify_solved_and_frontier_pairs(bwd_pairs, rewards[is_bwd])
                    if len(solved_s) > 0:
                        self._new_start_solved_count += self.start_solved_buffer.add_novel(
                            solved_s, min_sq_dist=skill_min_sq_dist
                        )
                    if len(frontier_s) > 0:
                        self._new_start_frontier_count += self.start_frontier_buffer.add_novel(
                            frontier_s, min_sq_dist=skill_min_sq_dist
                        )

        # Mixed-buffer refresh
        if env.common_step_counter - self._last_goal_update >= self.cfg.update_interval:
            self._update_mixed_goal_buffer(env)
        if env.common_step_counter - self._last_start_update >= self.cfg.update_interval:
            self._update_mixed_start_buffer(env)
            self.iterations += 1

        # Sample new starts + goals

        if current_anchor_env_ids is not None:
            n = len(current_anchor_env_ids)
            starts = self._initial_state_sampler.sample(n)
            self._set_tasks(env, current_anchor_env_ids, starts)
            goal_xyz = self._anchor_xyz.expand(n, -1)
            anchor_state = self._anchor_state.expand(n, *self._anchor_state.shape)
            self._commit(env, current_anchor_env_ids, goal_xyz, anchor_state)
            self.initialized[current_anchor_env_ids] = True

        if current_fwd_explore_env_ids is not None:
            n = len(current_fwd_explore_env_ids)
            starts = self._initial_state_sampler.sample(n)
            self._set_tasks(env, current_fwd_explore_env_ids, starts)
            goal_xyz, seed_state = self._split_pair(self._propose_goals(env, n))
            self._commit(env, current_fwd_explore_env_ids, goal_xyz, seed_state)
            self.initialized[current_fwd_explore_env_ids] = True

        if current_fwd_exploit_env_ids is not None:
            n = len(current_fwd_exploit_env_ids)
            starts = self._initial_state_sampler.sample(n)
            self._set_tasks(env, current_fwd_exploit_env_ids, starts)
            goal_xyz, seed_state = self._split_pair(self._sample_exploit_goal_pairs(env, n))
            self._commit(env, current_fwd_exploit_env_ids, goal_xyz, seed_state)
            self.initialized[current_fwd_exploit_env_ids] = True

        if current_bwd_explore_env_ids is not None:
            n = len(current_bwd_explore_env_ids)
            start_state = self._propose_starts(env, n)
            self._set_tasks(env, current_bwd_explore_env_ids, start_state)
            goal_xyz = self._anchor_xyz.expand(n, -1)
            self._commit(env, current_bwd_explore_env_ids, goal_xyz, start_state)
            self.initialized[current_bwd_explore_env_ids] = True

        if current_bwd_exploit_env_ids is not None:
            n = len(current_bwd_exploit_env_ids)
            goal_xyz, start_state = self._split_pair(self._sample_exploit_start_pairs(env, n))
            self._set_tasks(env, current_bwd_exploit_env_ids, start_state)
            self._commit(env, current_bwd_exploit_env_ids, goal_xyz, start_state)
            self.initialized[current_bwd_exploit_env_ids] = True

        # forward-frontier random walks: Voronoi/connect over mixed_goal_buffer
        if len(forward_walk_reset_ids) > 0:
            forward_walk_starts = self._sample_forward_walk_seeds(env, len(forward_walk_reset_ids))
            self._set_tasks(env, forward_walk_reset_ids, forward_walk_starts)
            self.initialized[forward_walk_reset_ids] = True

        # backward-frontier random walks: Voronoi/connect over mixed_start_buffer
        if len(backward_walk_reset_ids) > 0:
            backward_walk_starts = self._sample_backward_walk_seeds(env, backward_walk_reset_ids)
            self._set_tasks(env, backward_walk_reset_ids, backward_walk_starts)
            self.initialized[backward_walk_reset_ids] = True

        self._update_metrics(env)

        return {
            "iterations": self.iterations,
            "goal_frontier_size": len(self.goal_frontier_buffer),
            "goal_solved_size": len(self.goal_solved_buffer),
            "start_frontier_size": len(self.start_frontier_buffer),
            "start_solved_size": len(self.start_solved_buffer),
            **self._metrics,
        }

    # Data collection

    def _collect_random_walks(self, env: BaseEnv, random_ids: torch.Tensor) -> torch.Tensor | None:
        """Subsampled task-space trajectories of the random-walk envs resetting this call."""
        if len(random_ids) == 0:
            return None
        valid_mask = env.episode_length_buf[random_ids] != 1
        valid_ids = random_ids[valid_mask & self.initialized[random_ids]]
        if len(valid_ids) == 0:
            return None

        trajectory_lengths = env.episode_length_buf[valid_ids]
        trajectories = env.observation_manager._obs_buffer["trajectory"][valid_ids]

        traj_samples_per_walk = []
        for i in range(len(valid_ids)):
            full_traj = trajectories[i, -trajectory_lengths[i] :]
            traj_samples = full_traj[torch.linspace(0, len(full_traj) - 1, self.cfg.trajectory_subsamples).long()]
            traj_samples_per_walk.append(
                self.task_space.obs_to_task(traj_samples.unsqueeze(1)).reshape(
                    self.cfg.trajectory_subsamples, *self.task_space.task_dim()
                )
            )
        return torch.cat(traj_samples_per_walk, dim=0)

    def _walk_goal_labels(self, env: BaseEnv, random_ids: torch.Tensor) -> torch.Tensor | None:
        """Per-row inherited goal_xyz aligned with :meth:`_collect_random_walks`'s output."""
        if len(random_ids) == 0:
            return None
        valid_mask = env.episode_length_buf[random_ids] != 1
        valid_ids = random_ids[valid_mask & self.initialized[random_ids]]
        if len(valid_ids) == 0:
            return None
        return self._walk_goals[valid_ids].repeat_interleave(self.cfg.trajectory_subsamples, dim=0)

    def _collect_policy_trajectories(
        self, env: BaseEnv, train_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Env ids, achieved starts and episodic rewards of the resetting policy envs.

        Returns:
            ``(valid_ids, achieved_starts, rewards)``, or ``None`` if no env is valid.
        """
        if len(train_ids) == 0:
            return None
        valid_ids = train_ids[self.initialized[train_ids]]
        n = len(valid_ids)
        if n == 0:
            return None

        trajectory_lengths = env.episode_length_buf[valid_ids]
        trajectories = env.observation_manager._obs_buffer["trajectory"][valid_ids]
        rewards = compute_episode_sums(env, valid_ids, self.cfg.reward_terms)
        if self._reward_band_norm is None:
            self._reward_band_norm = self._compute_reward_band_norm(env)

        # achieved start = the episode's first obs, i.e. index (T - length) into the padded buffer
        start_obs = trajectories[torch.arange(n, device=env.device), -trajectory_lengths]
        achieved_starts = self.task_space.obs_to_task(start_obs.unsqueeze(1)).reshape(n, *self.task_space.task_dim())

        return valid_ids, achieved_starts, rewards

    def _compute_reward_band_norm(self, env: BaseEnv) -> float:
        """Achievable-reward scale that turns an episodic reward sum into the fraction ``r_min``/``r_max`` use.

        Sums ``weight * peak * active_seconds`` over positive-weight banded terms; windowed terms (with a
        ``duration`` param) contribute a flat ``weight``. Dense ``tracking_pos`` would be under-counted.
        """
        rm = env.reward_manager
        names = rm.active_terms if self.cfg.reward_terms == "all" else self.cfg.reward_terms
        episode_s = env.max_episode_length_s
        scale = 0.0
        for n in names:
            term = rm.get_term_cfg(n)
            if term.weight <= 0.0:
                continue
            duration = term.params.get("duration")
            if duration is None:
                scale += term.weight * episode_s
            else:
                scale += term.weight * min(duration, episode_s) / duration
        return max(scale, 1e-8)

    def _identify_solved_and_frontier_pairs(
        self, pairs: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Band-split pairs by normalized episodic reward into solved and frontier pairs."""
        frac = rewards / self._reward_band_norm
        solved_mask = frac > self.cfg.r_max
        frontier_mask = (frac >= self.cfg.r_min) & (frac <= self.cfg.r_max)
        return pairs[solved_mask], pairs[frontier_mask]

    def _role_masks(self, valid_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(is_fwd, is_bwd)`` masks for ``valid_ids``; ``anchor`` envs are in both."""
        is_anchor = self._id_in_group(valid_ids, self.anchor_envs)
        is_fwd = (
            is_anchor
            | self._id_in_group(valid_ids, self.fwd_explore_envs)
            | self._id_in_group(valid_ids, self.fwd_exploit_envs)
        )
        is_bwd = (
            is_anchor
            | self._id_in_group(valid_ids, self.bwd_explore_envs)
            | self._id_in_group(valid_ids, self.bwd_exploit_envs)
        )
        return is_fwd, is_bwd

    @staticmethod
    def _id_in_group(ids: torch.Tensor, group: torch.Tensor | None) -> torch.Tensor:
        if group is None:
            return torch.zeros_like(ids, dtype=torch.bool)
        return (ids >= group[0]) & (ids <= group[-1])

    # Mixed-buffer maintenance

    def _update_mixed_goal_buffer(self, env: BaseEnv) -> None:
        new_buffer = SimilarityBuffer(
            dim=self._search_dim,
            # +1 headroom
            max_size=self.cfg.mixed_goal_buffer_size + 1,
            item_shape=(self._pair_dim,),
            projection=self._project_goal,
        )
        n_solved = min(int(self.cfg.mixed_goal_buffer_size * self.cfg.goal_replay_ratio), len(self.goal_solved_buffer))
        n_frontier = min(self.cfg.mixed_goal_buffer_size - n_solved, len(self.goal_frontier_buffer))
        if n_solved > 0:
            solved_list = list(self.goal_solved_buffer)
            indices = np.random.choice(len(solved_list), n_solved, replace=False)
            new_buffer.add(torch.stack([solved_list[i] for i in indices], dim=0))
        if n_frontier > 0:
            frontier_list = list(self.goal_frontier_buffer)
            indices = np.random.choice(len(frontier_list), n_frontier, replace=False)
            new_buffer.add(torch.stack([frontier_list[i] for i in indices], dim=0))
        self.mixed_goal_buffer = new_buffer
        self._last_goal_update = env.common_step_counter

    def _update_mixed_start_buffer(self, env: BaseEnv) -> None:
        # +1 headroom
        new_buffer = self._new_start_buffer(self.cfg.mixed_start_buffer_size + 1)
        n_solved = min(
            int(self.cfg.mixed_start_buffer_size * self.cfg.start_replay_ratio), len(self.start_solved_buffer)
        )
        n_frontier = min(self.cfg.mixed_start_buffer_size - n_solved, len(self.start_frontier_buffer))
        if n_solved > 0:
            solved_list = list(self.start_solved_buffer)
            indices = np.random.choice(len(solved_list), n_solved, replace=False)
            new_buffer.add(torch.stack([solved_list[i] for i in indices], dim=0))
        if n_frontier > 0:
            frontier_list = list(self.start_frontier_buffer)
            indices = np.random.choice(len(frontier_list), n_frontier, replace=False)
            new_buffer.add(torch.stack([frontier_list[i] for i in indices], dim=0))
        self.mixed_start_buffer = new_buffer
        self._last_start_update = env.common_step_counter

    # Goal / start sampling

    def _set_tasks(self, env: BaseEnv, env_ids: torch.Tensor, state: torch.Tensor) -> None:
        """Write ``state`` into the sim, masked to ``cfg.init_subspace``; the single call site for every branch."""
        self.task_space.set_tasks(env, env_ids, self.task_space.mask_to_subspace(state, self._init_subspace))

    def _commit(self, env: BaseEnv, env_ids: torch.Tensor, goal_xyz: torch.Tensor, state: torch.Tensor) -> None:
        """Command ``goal_xyz`` and remember ``(goal_xyz, state)``."""
        self._goal_command.set_goal(env_ids, goal_xyz)
        self._commanded_pairs[env_ids] = self._make_pair(goal_xyz, state)

    def _source_band_counts(
        self,
        env: BaseEnv,
        num_envs: int,
        p_from_solved: float,
        p_candidate: float,
        from_solved_live: bool,
        candidate_live: bool,
    ) -> tuple[int, int, int]:
        """Per-env categorical over an explore branch's proposal sources ``[from_solved | candidate | remainder]``.

        A dormant from-solved source hands its mass to candidate, never to the remainder.

        Returns:
            ``(n_from_solved, n_candidate, n_remainder)``, summing to ``num_envs``.
        """
        eff_from_solved = p_from_solved if from_solved_live else 0.0
        candidate_hi = p_from_solved + p_candidate  # pinned; only the internal boundary moves
        eff_candidate = (candidate_hi - eff_from_solved) if candidate_live else 0.0

        u = torch.rand(num_envs, device=env.device)
        n_from_solved = int((u < eff_from_solved).sum())
        n_candidate = int(((u >= eff_from_solved) & (u < eff_from_solved + eff_candidate)).sum())
        return n_from_solved, n_candidate, num_envs - n_from_solved - n_candidate

    def _sample_goals_from_start_solved(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """fwd_explore goal pairs from backward-solved starts; only the state half is used, re-paired with its own xyz."""
        drawn = self.start_solved_buffer.get_random(num_envs, replace=True).to(env.device)
        _, states = self._split_pair(drawn)
        return self._make_pair(self._extract_xyz(states), states)

    def _propose_goals(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Goal proposals for fwd_explore envs from a categorical over from_solved, candidate and remainder.

        The remainder draws an initial-state sample paired with its own xyz.
        """
        n_from_solved, n_candidate, n_remainder = self._source_band_counts(
            env,
            num_envs,
            p_from_solved=self.cfg.p_start_from_solved,
            p_candidate=self.cfg.p_goal_candidate,
            from_solved_live=len(self.start_solved_buffer) > 0,
            candidate_live=len(self.goal_candidate_buffer) > 0,
        )

        pairs: list[torch.Tensor] = []
        if n_from_solved > 0:
            pairs.append(self._sample_goals_from_start_solved(env, n_from_solved))
        if n_candidate > 0:
            pairs.append(self.goal_candidate_buffer.get_random(n_candidate, replace=True).to(env.device))
        if n_remainder > 0:
            remainder_states = self._initial_state_sampler.sample(n_remainder)
            pairs.append(self._make_pair(self._extract_xyz(remainder_states), remainder_states))
        return torch.cat(pairs, dim=0)

    def _sample_exploit_goal_pairs(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Goal pairs for fwd_exploit envs from :attr:`mixed_goal_buffer`, falling back to exploration."""
        if len(self.mixed_goal_buffer) == 0 and (
            len(self.goal_frontier_buffer) > 0 or len(self.goal_solved_buffer) > 0
        ):
            self._update_mixed_goal_buffer(env)
        if len(self.mixed_goal_buffer) == 0:
            return self._propose_goals(env, num_envs)
        return self.mixed_goal_buffer.get_random(num_envs, replace=True).to(env.device)

    def _propose_starts(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Start proposals for bwd_explore envs, mirroring :meth:`_propose_goals`.

        from_solved spawns at forward-solved states; the remainder's mixed_start rung carries the anchor's
        reverse-curriculum seed, so bootstrap starts at the anchor.
        """
        n_from_solved, n_candidate, n_remainder = self._source_band_counts(
            env,
            num_envs,
            p_from_solved=self.cfg.p_goal_from_solved,
            p_candidate=self.cfg.p_start_candidate,
            from_solved_live=len(self.goal_solved_buffer) > 0,
            candidate_live=len(self.start_candidate_buffer) > 0,
        )

        starts: list[torch.Tensor] = []
        if n_from_solved > 0:
            solved_pairs = self.goal_solved_buffer.get_random(n_from_solved, replace=True).to(env.device)
            _, solved_states = self._split_pair(solved_pairs)
            starts.append(solved_states)
        if n_candidate > 0:
            candidate_pairs = self.start_candidate_buffer.get_random(n_candidate, replace=True).to(env.device)
            _, candidate_states = self._split_pair(candidate_pairs)
            starts.append(candidate_states)
        if n_remainder > 0:
            starts.append(self._sample_remainder_starts(env, n_remainder))
        return torch.cat(starts, dim=0)

    def _sample_remainder_starts(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Remainder rungs: the mixed_start buffer, then the initial-state sampler."""
        if len(self.mixed_start_buffer) == 0 and (
            len(self.start_frontier_buffer) > 0 or len(self.start_solved_buffer) > 0
        ):
            self._update_mixed_start_buffer(env)
        if len(self.mixed_start_buffer) > 0:
            _, seed_states = self._split_pair(self.mixed_start_buffer.get_random(num_envs, replace=True).to(env.device))
            return seed_states
        return self._initial_state_sampler.sample(num_envs)

    def _sample_exploit_start_pairs(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Start pairs for bwd_exploit envs from the mixed_start buffer, else fresh proposals paired with the anchor."""
        if len(self.mixed_start_buffer) == 0 and (
            len(self.start_frontier_buffer) > 0 or len(self.start_solved_buffer) > 0
        ):
            self._update_mixed_start_buffer(env)
        if len(self.mixed_start_buffer) == 0:
            goal_xyz = self._anchor_xyz.expand(num_envs, -1)
            return self._make_pair(goal_xyz, self._propose_starts(env, num_envs))
        return self.mixed_start_buffer.get_random(num_envs, replace=True).to(env.device)

    def _sample_forward_walk_seeds(self, env: BaseEnv, num_envs: int) -> torch.Tensor:
        """Seeds for forward walks from :attr:`mixed_goal_buffer`, selected by a per-env attractor mode.

        Connect pulls toward backward-solved starts, voronoi toward fresh task-space samples.
        """
        if len(self.mixed_goal_buffer) == 0:
            return self._initial_state_sampler.sample(num_envs)

        effective_connect = self.cfg.goal_connect_ratio if len(self.start_solved_buffer) > 0 else 0.0
        u = torch.rand(num_envs, device=env.device)
        n_connect = int((u < effective_connect).sum())
        n_voronoi = num_envs - n_connect

        seeds: list[torch.Tensor] = []
        if n_connect > 0:
            # attractor: a mastered backward start
            attractor_pairs = self.start_solved_buffer.get_random(n_connect, replace=True).to(env.device)
            _, attractor_states = self._split_pair(attractor_pairs)
            connect_xyz = self._extract_xyz(attractor_states)
            connect_pairs = self.mixed_goal_buffer.get_similar(connect_xyz, knn=self.cfg.select_knn).to(env.device)
            _, connect_seed_states = self._split_pair(connect_pairs)
            seeds.append(connect_seed_states)
        if n_voronoi > 0:
            voronoi_xyz = self._extract_xyz(self._sampler.sample(n_voronoi))
            voronoi_pairs = self.mixed_goal_buffer.get_similar(voronoi_xyz, knn=self.cfg.select_knn).to(env.device)
            _, voronoi_seed_states = self._split_pair(voronoi_pairs)
            seeds.append(voronoi_seed_states)
        return torch.cat(seeds, dim=0)

    def _sample_backward_walk_seeds(self, env: BaseEnv, walk_ids: torch.Tensor) -> torch.Tensor:
        """Seeds for backward walks from :attr:`mixed_start_buffer`, selected by a per-env attractor mode.

        Connect pulls toward forward-solved goals, voronoi toward fresh samples. Also writes :attr:`_walk_goals`
        so harvested candidates inherit the seed's goal.
        """
        num_envs = len(walk_ids)
        if len(self.mixed_start_buffer) == 0:
            self._walk_goals[walk_ids] = self._anchor_xyz.expand(num_envs, -1)
            return self._initial_state_sampler.sample(num_envs)

        effective_connect = self.cfg.start_connect_ratio if len(self.goal_solved_buffer) > 0 else 0.0
        u = torch.rand(num_envs, device=env.device)
        n_connect = int((u < effective_connect).sum())
        n_voronoi = num_envs - n_connect

        seed_states: list[torch.Tensor] = []
        seed_goals: list[torch.Tensor] = []
        if n_connect > 0:
            # attractor = a mastered forward goal's xyz
            attractor_pairs = self.goal_solved_buffer.get_random(n_connect, replace=True).to(env.device)
            _, attractor_states = self._split_pair(attractor_pairs)  # bare full state
            connect_query = self._extract_xyz(attractor_states)
            connect_pairs = self.mixed_start_buffer.get_similar(connect_query, knn=self.cfg.select_knn).to(env.device)
            connect_goal, connect_state = self._split_pair(connect_pairs)
            seed_states.append(connect_state)
            seed_goals.append(connect_goal)
        if n_voronoi > 0:
            voronoi_state = self._sampler.sample(n_voronoi)  # bare full state
            voronoi_query = self._extract_xyz(voronoi_state)
            voronoi_pairs = self.mixed_start_buffer.get_similar(voronoi_query, knn=self.cfg.select_knn).to(env.device)
            voronoi_goal, voronoi_state_out = self._split_pair(voronoi_pairs)
            seed_states.append(voronoi_state_out)
            seed_goals.append(voronoi_goal)

        seeds = torch.cat(seed_states, dim=0)
        self._walk_goals[walk_ids] = torch.cat(seed_goals, dim=0)
        return seeds

    # Env-id bookkeeping

    def _split_train_envs(
        self, env: BaseEnv
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Contiguous ``[anchor | fwd_explore | fwd_exploit | bwd_explore | bwd_exploit]`` id ranges.

        Any rounding residual goes to the first non-empty branch."""
        num_train_envs, _, _ = resolve_env_splits(env)
        ratios = [
            self.cfg.anchor_envs_ratio,
            self.cfg.fwd_explore_envs_ratio,
            self.cfg.fwd_exploit_envs_ratio,
            self.cfg.bwd_explore_envs_ratio,
            self.cfg.bwd_exploit_envs_ratio,
        ]
        counts = [int(num_train_envs * r) for r in ratios]
        residual = num_train_envs - sum(counts)
        if residual > 0:
            for i in range(len(counts)):
                if counts[i] > 0:
                    counts[i] += residual
                    break
            else:
                counts[0] += residual

        bounds = [0]
        for c in counts:
            bounds.append(bounds[-1] + c)

        def _mk(lo: int, hi: int) -> torch.Tensor | None:
            r = torch.arange(lo, hi, device=env.device)
            return r if len(r) > 0 else None

        return tuple(_mk(bounds[i], bounds[i + 1]) for i in range(5))

    @staticmethod
    def _ids_in_range(ids: torch.Tensor, lo_group: torch.Tensor | None, hi_group: torch.Tensor | None):
        """Select ``ids`` within ``[lo_group[0], hi_group[-1]]``, inclusive."""
        if lo_group is None and hi_group is None:
            return None
        lo = (lo_group if lo_group is not None else hi_group)[0]
        hi = (hi_group if hi_group is not None else lo_group)[-1]
        selected = ids[(ids >= lo) & (ids <= hi)]
        return selected if len(selected) > 0 else None

    # Pair helpers (goal_xyz <-> full state, flat pair <-> (goal_xyz, state))

    def _extract_xyz(self, state: torch.Tensor) -> torch.Tensor:
        """``(*, *task_dim)`` full state(s) -> ``(*, 3)`` xyz values of the interleaved ``[val, vel]`` layout."""
        flat = state.reshape(*state.shape[:-2], -1)
        return flat[..., self._xyz_dims]

    def _make_pair(self, goal_xyz: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``(*, 3)`` goal xyz + ``(*, *task_dim)`` state -> ``(*, pair_dim)`` flat pair, goal first."""
        flat_state = state.reshape(*state.shape[:-2], -1)
        return torch.cat([goal_xyz, flat_state], dim=-1)

    def _split_pair(self, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(*, pair_dim)`` flat pair -> ``((*, search_dim) goal_xyz, (*, *task_dim) state)``."""
        goal_xyz = pair[..., : self._search_dim]
        state = pair[..., self._search_dim :].reshape(*pair.shape[:-1], *self.task_space.task_dim())
        return goal_xyz, state

    def _project_goal(self, item: torch.Tensor) -> torch.Tensor:
        """Search and novelty metric for goal-side buffers; identity-safe on a bare xyz query."""
        return item[..., : self._search_dim]

    def _project_start_search(self, item: torch.Tensor) -> torch.Tensor:
        """Search-metric projection for start-side buffers: xyz of the state half; bare xyz queries pass through."""
        if item.shape[-1] == self._pair_dim:
            return item[..., self._pair_state_xyz_dims]
        return item

    def _project_start_novelty(self, item: torch.Tensor) -> torch.Tensor:
        """Novelty-metric projection for start-side buffers: ``cfg.init_subspace`` of the state half."""
        if item.shape[-1] == self._pair_dim:
            state = item[..., self._search_dim :].reshape(*item.shape[:-1], *self.task_space.task_dim())
        else:
            state = item
        return self._project_for_search(state)

    # Novelty-metric machinery

    def _project_to_subspace(self, tasks: torch.Tensor) -> torch.Tensor:
        if self.cfg.init_subspace is not None:
            flat = tasks.reshape(*tasks.shape[:-2], -1)
            return flat[..., self.cfg.init_subspace]
        else:
            return tasks

    def _project_for_search(self, tasks: torch.Tensor) -> torch.Tensor:
        """Novelty-metric projection for start-side dedup; not used for selection."""
        projected = self._project_to_subspace(tasks)
        if self.cfg.normalize_subspace:
            projected = self.task_space.normalize(projected, subspace=self.cfg.init_subspace)
        if self._subspace_weights is not None:
            projected = projected * self._subspace_weights.to(projected.device)
        return projected

    # Periodic scalar metrics (new-state throughput)

    def _update_metrics(self, env: BaseEnv) -> None:
        """Refresh the cached scalar metrics every ``cfg.metrics_log_interval`` env steps."""
        if env.common_step_counter % self.cfg.metrics_log_interval != 0:
            return
        if env.common_step_counter == self._last_metrics_step:
            return
        self._last_metrics_step = env.common_step_counter

        self._metrics = {
            "new_goal_candidate_states": self._new_goal_candidate_count,
            "new_start_candidate_states": self._new_start_candidate_count,
            "new_goal_frontier_pairs": self._new_goal_frontier_count,
            "new_goal_solved_pairs": self._new_goal_solved_count,
            "new_start_frontier_pairs": self._new_start_frontier_count,
            "new_start_solved_pairs": self._new_start_solved_count,
        }
        self._new_goal_candidate_count = 0
        self._new_start_candidate_count = 0
        self._new_goal_frontier_count = 0
        self._new_goal_solved_count = 0
        self._new_start_frontier_count = 0
        self._new_start_solved_count = 0


@configclass
class BVERCfg(CurriculumTermCfg):

    func: callable = BVERCurriculum

    # Curriculum parameters
    update_interval: int = 240

    # Environment split: fractions of the training envs, must sum to one
    anchor_envs_ratio: float = 0.05
    fwd_explore_envs_ratio: float = 0.325
    fwd_exploit_envs_ratio: float = 0.15
    bwd_explore_envs_ratio: float = 0.325
    bwd_exploit_envs_ratio: float = 0.15

    # Exploration parameters
    novelty_min_dist: float = 0.0
    select_knn: int = 4
    trajectory_subsamples: int = 4

    # Exploitation: bands are fractions of the achievable episodic reward (see _compute_reward_band_norm)
    r_min: float = 0.008
    r_max: float = 0.08  # must exceed the furthest goal's travel-limited fraction or the frontier stays empty
    # walk length in episode steps per tail slice; each must be set explicitly
    brownian_horizon_forward: int = MISSING
    brownian_horizon_backward: int = MISSING

    # Walk seeding: per-env categorical over connect, then voronoi remainder
    # connect targets the other tree's solved region; backward-connect waits for the first mastered goal
    goal_connect_ratio: float = 0.0
    start_connect_ratio: float = 0.0

    # Candidate-buffer parameters
    p_goal_candidate: float = 0.5
    goal_candidate_novelty_min_dist: float = 0.0
    p_start_candidate: float = 0.5
    start_candidate_novelty_min_dist: float = 0.0

    # From-solved proposals: explore draws from the other tree's solved region; dormant mass goes to candidate
    p_start_from_solved: float = 0.0
    p_goal_from_solved: float = 0.0

    # Datastructures
    goal_frontier_buffer_size: int = 2_000
    goal_solved_buffer_size: int = 10_000
    mixed_goal_buffer_size: int = 2_000
    goal_candidate_buffer_size: int = 5_000
    start_frontier_buffer_size: int = 2_000
    start_solved_buffer_size: int = 10_000
    mixed_start_buffer_size: int = 2_000
    start_candidate_buffer_size: int = 5_000
    # share of each mixed buffer rebuild drawn from *_solved, the rest from *_frontier
    goal_replay_ratio: float = 0.1
    start_replay_ratio: float = 0.1

    # General parameters
    reward_terms: str | list[str] = "all"
    # exogenous anchor goal as a full task-space state; required when anchor/bwd ratios > 0
    anchor_goal: list[list[float]] | None = None
    goal_command_name: str = "base_position"

    sampler_cfg: TaskSpaceSamplerCfg | None = None

    # filter Voronoi attractors to valid terrain so walks are not dragged toward holes
    filtered_voronoi_samples: bool = False

    init_subspace: list[int] | None = None
    normalize_subspace: bool = False
    subspace_weights: list[float] | None = None

    metrics_log_interval: int = 10 * 24
