# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The maze campaign's arms, layout-agnostic.

Arms differ only in how a reset's start and goal are chosen; a layout module subclasses them and sets the grid.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ...curriculum import EvalCfg, BVERCfg, RandomGoalStartCfg
from .maze_env_cfg import MazeEnvCfg, MazeTrajectoryObservationsCfg, MazeWalkTerminationsCfg

# Knobs shared by the BVER presets
_BVER_MAZE_KWARGS = dict(
    reward_terms=["hold_window"],
    p_goal_candidate=1.0,
    p_start_candidate=1.0,
    # Competence thresholds; keep r_max below one, since an off-centre ball earns less than one per step
    r_min=0.01,
    r_max=0.5,
    goal_replay_ratio=0.0,
    start_replay_ratio=0.0,
    # Position only; z is required by BVER but pinned by the task-space bounds
    init_subspace=[0, 2, 4],
    # Already metric and isotropic, so nothing to normalize
    normalize_subspace=False,
    subspace_weights=None,
    # Novelty radii in metres, scaled to the maze cell
    novelty_min_dist=0.5,
    goal_candidate_novelty_min_dist=0.05,
    start_candidate_novelty_min_dist=0.05,
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
class MazeBVERCurriculumCfg:
    """Bidirectional: both forward and backward expansion active."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.3,
        fwd_explore_envs_ratio=0.35,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.35,
        bwd_exploit_envs_ratio=0.0,
        goal_connect_ratio=1.0,
        start_connect_ratio=1.0,
        # per-layout, written by _apply_bver_layout
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_MAZE_KWARGS,
    )


@configclass
class MazeBVERFwdCurriculumCfg:
    """Forward-only ablation: expansion moves the goal outward from the initial state."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.3,
        fwd_explore_envs_ratio=0.7,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.0,
        bwd_exploit_envs_ratio=0.0,
        goal_connect_ratio=0.0,
        start_connect_ratio=0.0,
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_MAZE_KWARGS,
    )


@configclass
class MazeBVERBwdCurriculumCfg:
    """Backward-only ablation: expansion moves the start outward from the anchor goal."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.3,
        fwd_explore_envs_ratio=0.0,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.7,
        bwd_exploit_envs_ratio=0.0,
        goal_connect_ratio=0.0,
        start_connect_ratio=0.0,
        brownian_horizon_forward=0,
        brownian_horizon_backward=0,
        **_BVER_MAZE_KWARGS,
    )


@configclass
class MazeRandomGSCurriculumCfg:
    """Uninformed control matched to bidirectional BVER's start and goal marginals.

    ``sample_uniform`` replaces a feasible-start pool, since the task-space filter states feasibility in closed form.
    """

    initialization = RandomGoalStartCfg(
        sample_uniform=True,
        fwd_ratio=0.5,
        goal_command_name="base_position",
        init_subspace=[0, 2, 4],
    )


@configclass
class MazeEvalCurriculumCfg:
    evaluation = EvalCfg(terrain_levels=False)


_LAYOUT_OWNED_BVER_KEYS = frozenset(
    {
        "anchor_goal",
        "brownian_horizon_forward",
        "brownian_horizon_backward",
    }
)
"""BVER fields ``_apply_bver_layout`` derives from the grid, which ``bver_overrides`` refuses."""


def _apply_bver_overrides(init: BVERCfg, overrides: dict | None) -> None:
    """Write a layout's ``bver_overrides`` onto its BVER cfg; unknown keys raise."""
    for key, value in (overrides or {}).items():
        if key in _LAYOUT_OWNED_BVER_KEYS:
            raise ValueError(
                f"bver_overrides['{key}'] is derived from the layout: set the anchor via the "
                "grid's GOAL cell and the walk horizons via brownian_horizons"
            )
        if not hasattr(init, key):
            raise AttributeError(f"bver_overrides['{key}'] is not a BVERCfg field")
        setattr(init, key, value)


def _apply_bver_layout(cfg: MazeEnvCfg_BVER) -> None:
    """Point a BVER arm at its layout: anchor goal, walk horizons and per-layout knob overrides."""
    init = cfg.curriculum.initialization
    gx, gy, gz = cfg.goal_xyz()
    # one anchor, given as a full task state: (position, velocity) per xyz dimension
    init.anchor_goal = [[gx, 0.0], [gy, 0.0], [gz, 0.0]]
    fwd, bwd = cfg.brownian_horizons
    init.brownian_horizon_forward = fwd
    init.brownian_horizon_backward = bwd
    # the curriculum is the sole goal writer; a default_goal would overwrite it on every reset
    cfg.commands.base_position.default_goal = None
    _apply_bver_overrides(init, cfg.bver_overrides)


def _apply_fixed_goal(cfg: MazeEnvCfg) -> None:
    """Command the layout's GOAL cell on every reset, for the arms with no goal-writing curriculum."""
    cfg.commands.base_position.default_goal = cfg.goal_xyz()


def _apply_randomgs_layout(cfg: MazeEnvCfg) -> None:
    cfg.curriculum.initialization.anchor_goal = cfg.goal_xyz()
    cfg.commands.base_position.default_goal = None


@configclass
class MazeEnvCfg_BVER(MazeEnvCfg):
    """Bidirectional BVER."""

    observations: MazeTrajectoryObservationsCfg = MazeTrajectoryObservationsCfg()
    terminations: MazeWalkTerminationsCfg = MazeWalkTerminationsCfg()
    curriculum: MazeBVERCurriculumCfg = MazeBVERCurriculumCfg()

    bver_overrides: dict | None = None
    """Per-layout BVER knobs overriding ``_BVER_MAZE_KWARGS``; inherited by the forward and backward arms."""

    def __post_init__(self):
        super().__post_init__()
        _apply_bver_layout(self)


@configclass
class MazeEnvCfg_BVER_FWD(MazeEnvCfg_BVER):
    curriculum: MazeBVERFwdCurriculumCfg = MazeBVERFwdCurriculumCfg()


@configclass
class MazeEnvCfg_BVER_BWD(MazeEnvCfg_BVER):
    curriculum: MazeBVERBwdCurriculumCfg = MazeBVERBwdCurriculumCfg()


@configclass
class MazeEnvCfg_Baseline(MazeEnvCfg):
    """Vanilla PPO: every episode runs the INIT-to-GOAL task, with no curriculum at all."""

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_goal(self)


@configclass
class MazeEnvCfg_Random(MazeEnvCfg):
    """Uninformed start and goal coverage: BVER's task distribution without its selection."""

    observations: MazeTrajectoryObservationsCfg = MazeTrajectoryObservationsCfg()
    curriculum: MazeRandomGSCurriculumCfg = MazeRandomGSCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_randomgs_layout(self)


@configclass
class MazeEnvCfg_EVAL(MazeEnvCfg):
    """Batch eval on the headline task: INIT cell to GOAL cell."""

    observations: MazeTrajectoryObservationsCfg = MazeTrajectoryObservationsCfg()
    curriculum: MazeEvalCurriculumCfg = MazeEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_fixed_goal(self)
        self.viewer.resolution = (1920, 1080)


@configclass
class MazeEnvCfg_PLAY(MazeEnvCfg_EVAL):
    """Single-viewport interactive playback of a checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
