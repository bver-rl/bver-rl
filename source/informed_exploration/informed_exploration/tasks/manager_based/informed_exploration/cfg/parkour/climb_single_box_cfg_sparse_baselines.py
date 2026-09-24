"""Non-BVER controls for the sparse-reward single-box climb campaign.

Arms share the sparse MDP and localized observations and differ only in how resets are drawn, so their
checkpoints run in the sparse BVER play/eval bindings unchanged.
"""

import copy

from isaaclab.utils import configclass

from ..common.env_cfg import *
from ...curriculum.task_space.parkour_task_space_cfg import height_offset

from ...curriculum import BackplayCfg, RCCfg, RSICfg
from ..common.rewards import (
    PenSparseClimbBoxRewardsCfg,
    SparseClimbBoxRewardsCfg,
)
from ..common.terminations import ClimbBoxRCTerminationsCfg
from .climb_single_box_cfg import (
    ClimbBoxFeasibleStartsCurriculumCfg,
    IEAnymalDClimbSingleBoxEnvCfg,
    IEAnymalDClimbSingleBoxEnvCfg_INIT,
    _RC_GOAL_ON_BOX,
)
from ..common.box_height import CLIMB_BOX_HEIGHTS
from .climb_single_box_cfg_bver_sparse import _apply_fixed_on_box_goal, _feasible_pool_path


def _rsi_terrain_level(target_box_height: float) -> int:
    """Negative terrain level whose recorded expert trajectories match ``target_box_height``.

    Must be negative: ``RSICurriculum`` gates on truthiness, so level 0 would take the level-agnostic path.
    """
    level = round((target_box_height - 0.1) * 9.0 / 0.7)
    return level - 10


@configclass
class ClimbBoxSparseRSICurriculumCfg:
    """RSI for the sparse campaign, with ``terrain_level`` derived from the box height."""

    initialization = RSICfg(
        terrain_level=_rsi_terrain_level(CLIMB_BOX_HEIGHTS["0p41"]),
        one_level=True,
    )


def _apply_rsi_terrain_level(cfg) -> None:
    """Point the RSI and Backplay reference-trajectory level at the env's ``target_box_height``."""
    cfg.curriculum.initialization.terrain_level = _rsi_terrain_level(cfg.target_box_height)


_BACKPLAY_MAX_STEP = 300
"""Iteration at which the sparse campaign's Backplay curriculum finishes."""


@configclass
class ClimbBoxSparseBackplayCurriculumCfg:
    """Backplay for the sparse campaign, reading the same pool and terrain level as the RSI arm.

    Uses the default fractional window schedule; ``max_step`` restretches the whole curriculum.
    """

    initialization = BackplayCfg(
        terrain_level=_rsi_terrain_level(CLIMB_BOX_HEIGHTS["0p41"]),
        one_level=True,
        max_step=_BACKPLAY_MAX_STEP,
    )


@configclass
class ClimbBoxSparseReverseCurriculumCfg:
    """Reverse curriculum for the sparse campaign.

    ``R_min``/``R_max`` are rescaled to the sparse weight and ``brownian_horizon`` matches the BVER arms.
    """

    initialization = RCCfg(
        goal=_RC_GOAL_ON_BOX,
        reward_terms=["tracking_pos_sparse"],
        R_min=12.5,
        R_max=75.0,
        N_old=500,
        N_new=2500,
        brownian_horizon=64,
        num_random_steps=100_000,
        max_random_walk_seeds=4000,
        max_old_starts=100_000,
    )


# Env configs


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_Baseline(IEAnymalDClimbSingleBoxEnvCfg):
    """Vanilla-PPO baseline for the sparse campaign: the sparse arms without an initialization curriculum.

    Not derived from ``_INIT``, whose static fallback pose would spawn the robot on the box, already solved.
    """

    observations = XYZObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = None


    def __post_init__(self):
        super().__post_init__()
        # Match the arms' terrain, which the _INIT leaf enables
        self.scene.terrain.terrain_generator.curriculum = True
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_Baseline_PEN(IEAnymalDClimbSingleBoxEnvCfg_SPARSE_Baseline):
    """The vanilla-PPO baseline with the non-contact penalties active from step zero."""

    rewards = PenSparseClimbBoxRewardsCfg()


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_Random(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Random-initialization control: every reset draws a full robot state from the feasible-starts pool.

    The pool is height-specific, so ``__post_init__`` re-derives ``pool_path`` from ``target_box_height``.
    """

    observations = XYZObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxFeasibleStartsCurriculumCfg()


    def __post_init__(self):
        super().__post_init__()
        # The pool is height-specific
        self.curriculum.initialization.sampler_cfg.pool_path = _feasible_pool_path(self.target_box_height)
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_RSI(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Reference-state-initialization control: resets drawn from recorded expert trajectories.

    Trajectories come from ``ParkourTaskSpaceCfg.optimal_trajectory_file``, loaded into ``env.train_data``.
    """

    observations = XYZObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxSparseRSICurriculumCfg()


    def __post_init__(self):
        super().__post_init__()
        _apply_rsi_terrain_level(self)
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_Backplay(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Backplay control: expert-trajectory resets through a window sliding backward from the goal.

    Identical to :class:`IEAnymalDClimbSingleBoxEnvCfg_SPARSE_RSI` except for the curriculum term.
    """

    observations = XYZObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxSparseBackplayCurriculumCfg()


    def __post_init__(self):
        super().__post_init__()
        _apply_rsi_terrain_level(self)
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_SPARSE_RC(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Reverse-curriculum control: starts expanded backward from the on-box goal state by Brownian walkers.

    Requires the ``trajectory`` obs group and ``brownian_motion_time_out``.
    """

    observations = XYZTrajectoryObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxSparseReverseCurriculumCfg()
    terminations = ClimbBoxRCTerminationsCfg()


    def __post_init__(self):
        super().__post_init__()
        # Goal z tracks the box height; deep-copy so the shared goal is not mutated
        self.curriculum.initialization.goal = copy.deepcopy(self.curriculum.initialization.goal)
        self.curriculum.initialization.goal[2][0] = self.target_box_height + height_offset
        _apply_fixed_on_box_goal(self)
