# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum cfgs shared by every terrain-derived task family.

``anchor_goal`` is omitted and set per instance by :func:`~.goal_conditioning.apply_bver_goal_conditioning`.
"""

from typing import Any

from isaaclab.utils import configclass

from ...curriculum import BVERCfg, RandomCfg
from ...curriculum.task_space.parkour_task_space_cfg import STATE_SPACE_WEIGHTS

# Knobs shared by the BVER variants; only the branch ratios differ
_BVER_KWARGS = dict(
    reward_terms=["tracking_pos_sparse"],
    init_subspace=list(range(0, 36, 2)),
    subspace_weights=STATE_SPACE_WEIGHTS[::2],
    normalize_subspace=True,
    novelty_min_dist=0.2,
    goal_candidate_novelty_min_dist=0.01,
    start_candidate_novelty_min_dist=0.01,
    brownian_horizon_forward=128,
    brownian_horizon_backward=128,
    goal_replay_ratio=0.05,
    start_replay_ratio=0.05,
    p_goal_candidate=1.0,
    p_start_candidate=1.0,
    r_min=0.05,
    r_max=0.2,
)
# From-solved proposals, currently disabled; if enabled, from_solved + candidate must sum to one
_BVER_FROM_SOLVED_KWARGS: dict[str, Any] = dict(
    _BVER_KWARGS,
    # p_goal_candidate=0.5,  # fwd expansion
    # p_start_from_solved=0.5,
    # p_start_candidate=0.5,  # bwd expansion
    # p_goal_from_solved=0.5,
)


@configclass
class TerrainBVERCurriculumCfg:
    """Bidirectional: both forward and backward expansion active."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.35,
        fwd_exploit_envs_ratio=0.15,
        bwd_explore_envs_ratio=0.35,
        bwd_exploit_envs_ratio=0.15,
        goal_connect_ratio=0.5,
        start_connect_ratio=0.5,
        **_BVER_FROM_SOLVED_KWARGS,
    )


@configclass
class TerrainBVERFwdCurriculumCfg:
    """Forward-only ablation arm."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.70,
        fwd_exploit_envs_ratio=0.30,
        bwd_explore_envs_ratio=0.0,
        bwd_exploit_envs_ratio=0.0,
        **_BVER_KWARGS,
    )


@configclass
class TerrainBVERBwdCurriculumCfg:
    """Backward-only ablation arm."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.0,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.70,
        bwd_exploit_envs_ratio=0.30,
        **_BVER_KWARGS,
    )


@configclass
class GenFeasibleStartsCurriculumCfg:
    """Random curriculum driving ``generate_feasible_starts.py``, with the keep-in filter enabled.

    Samples base x, y, z and yaw only; the robot spawns in its default stance and settles onto the terrain.
    """

    initialization = RandomCfg(filtered_samples=True, subspace=[0, 2, 4, 10])
