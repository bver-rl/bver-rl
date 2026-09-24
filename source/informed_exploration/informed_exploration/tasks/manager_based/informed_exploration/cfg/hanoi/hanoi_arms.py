# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Franka Hanoi campaign's arms, layout-agnostic.

Arms differ only in how a reset's start and goal are chosen; a layout module subclasses them and sets the layout.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...curriculum import EvalCfg, BVERCfg, RandomGoalStartCfg
from ...curriculum.task_space.hanoi_task_space_cfg import HANOI_INIT_SUBSPACE
from .hanoi_env_cfg import HanoiEnvCfg, HanoiTrajectoryObservationsCfg, HanoiWalkTerminationsCfg

# Knobs shared by the BVER presets
_BVER_HANOI_KWARGS = dict(
    reward_terms=["hold_window"],
    p_goal_candidate=1.0,
    p_start_candidate=1.0,
    r_min=0.1,
    r_max=0.85,
    goal_replay_ratio=0.1,
    start_replay_ratio=0.1,
    # Joint offsets and the derived ring centre, velocities excluded; radians and metres, so normalized.
    init_subspace=HANOI_INIT_SUBSPACE,
    normalize_subspace=True,
    subspace_weights=None,
    metrics_log_interval=24 * 4,
    # buffer sizes
    goal_frontier_buffer_size=1000,
    goal_solved_buffer_size=1000,
    mixed_goal_buffer_size=1000,
    goal_candidate_buffer_size=500,
    start_frontier_buffer_size=1000,
    start_solved_buffer_size=1000,
    mixed_start_buffer_size=1000,
    start_candidate_buffer_size=500,
)


@configclass
class HanoiBVERCurriculumCfg:
    """Bidirectional: both forward and backward expansion active."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.5,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.5,
        bwd_exploit_envs_ratio=0.0,
        goal_connect_ratio=0.5,
        start_connect_ratio=0.5,
        # Horizons are per-layout, written by _apply_bver_layout
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_HANOI_KWARGS,
    )


@configclass
class HanoiBVERFwdCurriculumCfg:
    """Forward-only ablation: expansion moves the goal outward from the seated start."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.9,
        fwd_exploit_envs_ratio=0.1,
        bwd_explore_envs_ratio=0.0,
        bwd_exploit_envs_ratio=0.0,
        goal_connect_ratio=0.0,
        start_connect_ratio=0.0,
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_HANOI_KWARGS,
    )


@configclass
class HanoiBVERBwdCurriculumCfg:
    """Backward-only ablation: expansion moves the start outward from the seated anchor."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.0,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.9,
        bwd_exploit_envs_ratio=0.1,
        goal_connect_ratio=0.0,
        start_connect_ratio=0.0,
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_HANOI_KWARGS,
    )


@configclass
class HanoiRandomGSCurriculumCfg:
    """Uninformed control matched to bidirectional BVER's start and goal marginals, without a feasible pool.

    Goals and joint starts are drawn uniformly from the task space, unfiltered, so they need not be feasible.
    """

    initialization = RandomGoalStartCfg(
        sample_uniform=True,
        filter_uniform=False,
        fwd_ratio=0.5,
        goal_command_name="base_position",
        init_subspace=HANOI_INIT_SUBSPACE,
    )


@configclass
class HanoiEvalCurriculumCfg:
    evaluation = EvalCfg(terrain_levels=False)


_LAYOUT_OWNED_BVER_KEYS = frozenset(
    {
        "anchor_goal",
        "brownian_horizon_forward",
        "brownian_horizon_backward",
    }
)
"""BVER fields ``_apply_bver_layout`` derives from the layout, which ``bver_overrides`` refuses."""


def _apply_bver_overrides(init: BVERCfg, overrides: dict | None) -> None:
    """Write a layout's ``bver_overrides`` onto its BVER cfg; unknown keys raise."""
    for key, value in (overrides or {}).items():
        if key in _LAYOUT_OWNED_BVER_KEYS:
            raise ValueError(
                f"bver_overrides['{key}'] is derived from the layout: the anchor comes from the poses file and "
                "the walk horizons from brownian_horizons"
            )
        if not hasattr(init, key):
            raise AttributeError(f"bver_overrides['{key}'] is not a BVERCfg field")
        setattr(init, key, value)


def _apply_bver_layout(cfg: HanoiEnvCfg_BVER) -> None:
    """Point a BVER arm at its layout: anchor state, walk horizons and per-layout knob overrides."""
    init = cfg.curriculum.initialization
    # one anchor, given as a full task state: joint offsets from home, then the seated ring centre
    init.anchor_goal = cfg.anchor_task()
    fwd, bwd = cfg.brownian_horizons
    init.brownian_horizon_forward = fwd
    init.brownian_horizon_backward = bwd
    # the curriculum is the sole goal writer; a default_goal would overwrite it on every reset
    cfg.commands.base_position.default_goal = None
    _apply_bver_overrides(init, cfg.bver_overrides)


def _apply_fixed_goal(cfg: HanoiEnvCfg) -> None:
    """Command the anchor's ring centre on every reset, for the arms with no goal-writing curriculum."""
    cfg.commands.base_position.default_goal = cfg.goal_xyz()


def _apply_randomgs_layout(cfg: HanoiEnvCfg) -> None:
    """Point the random control at its layout: backward envs command the anchor's ring centre."""
    cfg.curriculum.initialization.anchor_goal = cfg.goal_xyz()
    # the curriculum is the sole goal writer; a default_goal would overwrite it on every reset
    cfg.commands.base_position.default_goal = None


@configclass
class HanoiEnvCfg_BVER(HanoiEnvCfg):
    """Bidirectional BVER."""

    observations: HanoiTrajectoryObservationsCfg = HanoiTrajectoryObservationsCfg()
    terminations: HanoiWalkTerminationsCfg = HanoiWalkTerminationsCfg()
    curriculum: HanoiBVERCurriculumCfg = HanoiBVERCurriculumCfg()

    bver_overrides: dict | None = None
    """Per-layout BVER knobs overriding ``_BVER_HANOI_KWARGS``; inherited by the forward and backward arms."""

    def __post_init__(self):
        super().__post_init__()
        _apply_bver_layout(self)


@configclass
class HanoiEnvCfg_BVER_FWD(HanoiEnvCfg_BVER):
    curriculum: HanoiBVERFwdCurriculumCfg = HanoiBVERFwdCurriculumCfg()


@configclass
class HanoiEnvCfg_BVER_BWD(HanoiEnvCfg_BVER):
    curriculum: HanoiBVERBwdCurriculumCfg = HanoiBVERBwdCurriculumCfg()


@configclass
class HanoiEnvCfg_Baseline(HanoiEnvCfg):
    """Vanilla PPO: every episode runs the headline task from the seated start, with no curriculum at all."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_goal(self)


@configclass
class HanoiEnvCfg_Random(HanoiEnvCfg):
    """Uninformed start and goal coverage: BVER's task distribution without its selection."""

    observations: HanoiTrajectoryObservationsCfg = HanoiTrajectoryObservationsCfg()
    curriculum: HanoiRandomGSCurriculumCfg = HanoiRandomGSCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_randomgs_layout(self)


@configclass
class HanoiEnvCfg_EVAL(HanoiEnvCfg):
    """Batch eval on the headline task: seated on the start peg to seated on the goal peg."""

    observations: HanoiTrajectoryObservationsCfg = HanoiTrajectoryObservationsCfg()
    curriculum: HanoiEvalCurriculumCfg = HanoiEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_goal(self)
        self.viewer.resolution = (1920, 1080)


@configclass
class HanoiEnvCfg_PLAY(HanoiEnvCfg_EVAL):
    """Single-viewport interactive playback of a checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
