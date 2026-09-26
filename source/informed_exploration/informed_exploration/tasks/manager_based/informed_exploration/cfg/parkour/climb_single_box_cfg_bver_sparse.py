"""Sparse-reward BVER arms for the single-box climb.

Presets with :class:`SparseClimbBoxRewardsCfg` and localized xyz observations everywhere, so
checkpoints are not obs-compatible with the dense tasks. Non-BVER controls live in the sparse baselines module.
"""

import copy

from isaaclab.utils import configclass

from ..common.env_cfg import *
from ...curriculum.task_space.parkour_task_space_cfg import STATE_SPACE_WEIGHTS, height_offset

from ...curriculum import BVERCfg
from ...mdp.commands import CurriculumGoalCfg
from ..common.eval_curricula import (
    ClimbBoxEvalCurriculumCfg,
    ClimbBoxMultiGoalEvalCurriculumCfg,
)
from ..common.rewards import (
    PenSparseClimbBoxRewardsCfg,
    SparseClimbBoxRewardsCfg,
)
from ..common.action_scale import (
    ACTION_SCALE_HALF,
    apply_action_scale,
)
from ..common.terminations import ClimbBoxRCTerminationsCfg
from .climb_single_box_cfg import (
    IEAnymalDClimbSingleBoxEnvCfg_EVAL,
    IEAnymalDClimbSingleBoxEnvCfg_INIT,
    _RC_GOAL_ON_BOX,
)

# Knobs shared by the sparse presets
_BVER_SPARSE_KWARGS = dict(
    anchor_goal=_RC_GOAL_ON_BOX,
    reward_terms=["tracking_pos_sparse"],
    init_subspace=list(range(0, 36, 2)),
    subspace_weights=STATE_SPACE_WEIGHTS[::2],
    normalize_subspace=True,
    novelty_min_dist=0.2,
    goal_candidate_novelty_min_dist=0.01,
    start_candidate_novelty_min_dist=0.01,
    brownian_horizon_forward=64,
    brownian_horizon_backward=64,
    goal_replay_ratio=0.05,
    start_replay_ratio=0.05,
    p_goal_candidate=1.0,
    p_start_candidate=1.0,
    r_min=0.05,
    r_max=0.2,
)


# Non-auto-fired event mode, applied only by the eval script; mirrored in scripts/rsl_rl/eval_utils.py
PERTURB_MODE = "perturb"


def _feasible_pool_path(target_box_height: float) -> str:
    """Path of the feasible-starts pool for ``target_box_height``, matching ``generate_feasible_starts.py``."""
    suffix = f"_{target_box_height:.2f}".replace(".", "p")
    return f"informed-exploration/data/parkour/feasible_starts_climb_box{suffix}.pkl"


@configclass
class ClimbBoxBVERSparseCurriculumCfg:
    """Bidirectional, sparse-reward arm."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.35,
        fwd_exploit_envs_ratio=0.15,
        bwd_explore_envs_ratio=0.35,
        bwd_exploit_envs_ratio=0.15,
        goal_connect_ratio=0.5,
        start_connect_ratio=0.5,
        **_BVER_SPARSE_KWARGS,
    )


@configclass
class ClimbBoxBVERSparseFwdCurriculumCfg:
    """Forward-only, sparse-reward arm."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.70,
        fwd_exploit_envs_ratio=0.30,
        bwd_explore_envs_ratio=0.0,
        bwd_exploit_envs_ratio=0.0,
        **_BVER_SPARSE_KWARGS,
    )


@configclass
class ClimbBoxBVERSparseBwdCurriculumCfg:
    """Backward-only, sparse-reward arm."""

    initialization = BVERCfg(
        anchor_envs_ratio=0.0,
        fwd_explore_envs_ratio=0.0,
        fwd_exploit_envs_ratio=0.0,
        bwd_explore_envs_ratio=0.70,
        bwd_exploit_envs_ratio=0.30,
        **_BVER_SPARSE_KWARGS,
    )


# Env configs


def _apply_bver_goal_conditioning(cfg) -> None:
    """Patch the anchor's z to the box height and set the curriculum-controlled 3D goal command and observation."""
    cfg.curriculum.initialization.anchor_goal = copy.deepcopy(cfg.curriculum.initialization.anchor_goal)
    cfg.curriculum.initialization.anchor_goal[2][0] = cfg.target_box_height + height_offset

    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
    )

    # 3D goal observation, since the goal can be on top of the box
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d


def _apply_fixed_on_box_goal(cfg) -> None:
    """Pin a fixed 3D goal on top of the box and switch the goal observation to 3D.

    For bindings where no curriculum writes goals; without it the policy sees an all-zeros goal.
    """
    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
        default_goal=[
            _RC_GOAL_ON_BOX[0][0],
            _RC_GOAL_ON_BOX[1][0],
            cfg.target_box_height + height_offset,
        ],
    )

    # 3D goal observation, since the goal can be on top of the box
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d


def _pin_startup_randomization(cfg) -> None:
    """Collapse the startup mass and material randomization onto its nominal point.

    The zero-perturbation level is therefore not numerically identical to the plain ``-EVAL-v0`` result.
    """
    cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    for key in ("static_friction_range", "dynamic_friction_range"):
        lo, hi = cfg.events.physics_material.params[key]
        cfg.events.physics_material.params[key] = (0.5 * (lo + hi), 0.5 * (lo + hi))
    cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    cfg.events.physics_material.params["num_buckets"] = 1


@configclass
class RobustEvalEventCfg(InitializationEventCfg):
    """Initialization events plus a zero-magnitude push term whose range the robustness eval rewrites."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode=PERTURB_MODE,
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}},
    )


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    # BVER records whole episodes through the `trajectory` obs group
    observations = XYZTrajectoryObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxBVERSparseCurriculumCfg()
    # Adds brownian_motion_time_out, without which the walk horizons never fire
    terminations = ClimbBoxRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_FWD(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    observations = XYZTrajectoryObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxBVERSparseFwdCurriculumCfg()
    terminations = ClimbBoxRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_BWD(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    observations = XYZTrajectoryObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxBVERSparseBwdCurriculumCfg()
    terminations = ClimbBoxRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PEN(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE):
    """The bidirectional sparse arm with the non-contact penalties active from step zero."""

    rewards = PenSparseClimbBoxRewardsCfg()


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Interactive single-viewport play for sparse BVER checkpoints, with the localized observations."""

    observations = XYZObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()
    curriculum = ClimbBoxEvalCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator.curriculum = True
        # reduce terrain size
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        # disable policy noise
        self.observations.policy.enable_corruption = False
        # remove external forces
        self.events.base_external_force_torque = None

        # render quality
        self.sim.render = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)

        # viewer settings
        self.viewer.eye = (-4.4, -4.1, 3.6)
        self.viewer.lookat = (-0.0, 2.1, -0.9)
        self.viewer.resolution = (1920, 1080)

        # The eval curriculum sets starts only, so the goal must be pinned
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGPLAY(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY):
    """Multi-goal play: goals are sampled from the (filtered) task space instead of fixed on-box."""

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # A set default_goal would overwrite every sampled goal on reset
        self.commands.base_position.default_goal = None


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL(IEAnymalDClimbSingleBoxEnvCfg_EVAL):
    """Batch and sweep single-goal eval for sparse BVER checkpoints, driven by scripts/rsl_rl/eval.py."""

    # Localized observations, matching what the sparse arms trained against
    observations = XYZTrajectoryObservationsCfg()
    rewards = SparseClimbBoxRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The eval curriculum has no .initialization, so the fixed-goal helper applies
        _apply_fixed_on_box_goal(self)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGEVAL(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL):
    """Batch and sweep multi-goal eval with goals sampled from the filtered task space."""

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # A set default_goal would overwrite every sampled goal on reset
        self.commands.base_position.default_goal = None


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTEVAL(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL):
    """Perturbation-robustness eval: the single-goal eval with startup randomization pinned.

    Goals travel through the eval queue; the fixed goal is kept as :attr:`robust_nominal_goal`.
    """

    events = RobustEvalEventCfg()
    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    robust_nominal_goal: list[float] | None = None
    """Env-relative xyz goal of the unperturbed task, filled in ``__post_init__``."""
    feasible_pool_path: str | None = None
    """Feasible-starts pool the start and goal levels draw from, filled in ``__post_init__``."""

    def __post_init__(self):
        super().__post_init__()
        _pin_startup_randomization(self)
        self.robust_nominal_goal = list(self.commands.base_position.default_goal)
        self.commands.base_position.default_goal = None
        self.feasible_pool_path = _feasible_pool_path(self.target_box_height)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTPLAY(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTEVAL):
    """Single-viewport counterpart of ``_ROBUSTEVAL``, for watching one perturbed rollout."""

    def __post_init__(self):
        super().__post_init__()

        # reduce terrain size
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1

        # render quality
        self.sim.render = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)

        # viewer settings
        self.viewer.eye = (-4.4, -4.1, 3.6)
        self.viewer.lookat = (-0.0, 2.1, -0.9)
        self.viewer.resolution = (1920, 1080)


# Action-scale variants of the play and eval bindings


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY_SCALE05(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY):
    """``_BVER_SPARSE_PLAY`` at :data:`ACTION_SCALE_HALF`. Also the binding record.py drives."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGPLAY_SCALE05(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGPLAY):
    """``_BVER_SPARSE_MGPLAY`` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL_SCALE05(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL):
    """``_BVER_SPARSE_EVAL`` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGEVAL_SCALE05(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGEVAL):
    """``_BVER_SPARSE_MGEVAL`` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTEVAL_SCALE05(
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTEVAL
):
    """``_BVER_SPARSE_ROBUSTEVAL`` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTPLAY_SCALE05(
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_ROBUSTPLAY
):
    """``_BVER_SPARSE_ROBUSTPLAY`` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)
