# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BVER on real-world rock scans: the arms, written once for every scan.

Each class reads its terrain from :attr:`ScanEnvCfg.scan`; ``scan_families.py`` binds one subclass per scan.
The bridge stack is reused unchanged except for the yaw window, which is re-centred on the scan's start
heading, and the initial-state window, a small patch around the measured start pose.
"""

from __future__ import annotations

import math
import warnings

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from ..common.curricula import (
    GenFeasibleStartsCurriculumCfg,
    TerrainBVERBwdCurriculumCfg,
    TerrainBVERCurriculumCfg,
    TerrainBVERFwdCurriculumCfg,
)
from ..common.goal_conditioning import (
    apply_baseline_spawn,
    apply_eval_goal_conditioning,
    apply_bver_goal_conditioning,
)
from ..common.goal_env_cfg import (
    TerrainDerivedTaskSpaceEnvCfg_EVAL,
    TerrainDerivedTaskSpaceEnvCfg_INIT,
    TerrainRCTerminationsCfg,
    only_sub_terrain,
)
from ...curriculum import RandomGoalStartCfg
from ...curriculum.task_space.bridge_task_space_cfg import bridge_anchors_from_goal_poses
from ...mdp.commands import CurriculumGoalCfg
from ..common.eval_curricula import ClimbBoxEvalCurriculumCfg, ClimbBoxMultiGoalEvalCurriculumCfg
from ..common.rewards import (
    PenSparseClimbBoxRewardsCfg,
    SparseClimbBoxRewardsCfg,
)
from ..common.action_scale import (
    ACTION_SCALE_HALF,
    apply_action_scale,
)
from ..common.env_cfg import *  # noqa: F403  (sim_utils, the observation groups, configclass)
from ...curriculum.task_space.parkour_task_space_cfg import height_offset

# absolute: `informed_exploration.terrains` is a top-level package out of relative-import reach
from informed_exploration.terrains.config.scans import ScanSpec

from .scan_naming import pool_path

# Family-wide knobs; anything that varies per rock lives on the ScanSpec

SCAN_EVAL_XY_HALF = 0.2
"""Half-width of the initial-state patch around the start pose, over which the success rate is measured."""

SCAN_BROWNIAN_HORIZON = 64
"""Forward and backward walk length in policy steps on every scan BVER arm, as on box climbing."""

PLAY_EPISODE_TIME_S = 7.0
"""Seconds per episode on the PLAY bindings.

EVAL bindings keep the scan's own episode time, since the sparse reward's ``duration`` is pinned to it."""

SCAN_BODY_FOOTPRINT = (1.0, 0.5)
"""Base footprint (length x width) used only to check that spawns clear the fence."""

SCAN_GPU_MAX_RIGID_PATCH_COUNT = 2**19
"""Contact-patch budget for triangle-mesh terrain; when short, PhysX silently drops contacts.

Scales linearly with ``num_envs`` and triangle density. Over-allocating exhausts device memory, so raise
it only to a modest multiple of what PhysX reports."""


# Shared __post_init__ tails


def _apply_scan_task_space(cfg) -> None:
    """Re-centre the task space on the scan's start heading and tighten the initial-state window."""
    sub = only_sub_terrain(cfg.scene.terrain.terrain_generator)
    heading = sub.heading
    space = cfg.task_space

    space.bounds[10] = [heading - math.pi / 2.0, heading + math.pi / 2.0]
    space.eval_bounds[10] = [heading, heading]
    space.default_task[10] = heading

    # bounds are already fence-clamped, so intersecting keeps the spawn patch off the wall
    for dim in (0, 2):
        lo = max(space.bounds[dim][0], -SCAN_EVAL_XY_HALF)
        hi = min(space.bounds[dim][1], SCAN_EVAL_XY_HALF)
        assert lo < hi, (
            f"the start pose leaves no room for an initial-state window in dimension {dim}:"
            f" bounds {space.bounds[dim]} do not overlap +-{SCAN_EVAL_XY_HALF} m around the start."
            " Move the start pose further from the scan's edge."
        )
        space.eval_bounds[dim] = [lo, hi]

    goal = sub.goal_poses["goal"]
    start = sub.goal_poses["start"]
    for dim, value in ((0, goal[0] - start[0]), (2, goal[1] - start[1])):
        assert space.bounds[dim][0] <= value <= space.bounds[dim][1], (
            f"the goal falls outside the task space in dimension {dim}: {value:.2f} not in"
            f" {space.bounds[dim]}. It is too close to the scan's edge for the fence standoff."
        )

    _warn_on_fence_clearance(sub)


def _warn_on_fence_clearance(sub) -> None:
    """Warn if the robot's body would reach into the fence from a corner of the initial-state window."""
    if sub.perimeter_wall is None:
        return
    rect = sub.footprint_rect_tile
    margin_x, margin_y = sub.perimeter_wall.margin
    faces_x = (rect[0] - margin_x, rect[1] + margin_x)
    faces_y = (rect[2] - margin_y, rect[3] + margin_y)

    length, width = SCAN_BODY_FOOTPRINT
    cos_y, sin_y = math.cos(sub.heading), math.sin(sub.heading)
    reach_x = abs(length * cos_y) / 2.0 + abs(width * sin_y) / 2.0
    reach_y = abs(length * sin_y) / 2.0 + abs(width * cos_y) / 2.0

    start = sub.goal_poses["start"]
    needed = (reach_x + SCAN_EVAL_XY_HALF, reach_y + SCAN_EVAL_XY_HALF)
    have = (
        min(start[0] - faces_x[0], faces_x[1] - start[0]),
        min(start[1] - faces_y[0], faces_y[1] - start[1]),
    )
    if have[0] < needed[0] or have[1] < needed[1]:
        warnings.warn(
            f"the start pose sits {have[0]:.2f} m (x) / {have[1]:.2f} m (y) from the fence, but a"
            f" spawn at the corner of the initial-state window needs {needed[0]:.2f} / {needed[1]:.2f} m"
            f" for the robot's body at heading {math.degrees(sub.heading):.0f} deg. Some resets will"
            " start inside the wall: move the start further in, or lower SCAN_EVAL_XY_HALF.",
            stacklevel=2,
        )


SCAN_VIEWER_RESOLUTION = (1920 * 2, 1080 * 2)
"""Viewport resolution unless a scan overrides it."""


def _apply_scan_viewer(cfg) -> None:
    """Point the viewport camera at this scan, if it sets ``viewer_eye``/``viewer_lookat``."""
    if cfg.scan.viewer_eye is not None:
        cfg.viewer.eye = cfg.scan.viewer_eye
    if cfg.scan.viewer_lookat is not None:
        cfg.viewer.lookat = cfg.scan.viewer_lookat
    cfg.viewer.resolution = cfg.scan.viewer_resolution or SCAN_VIEWER_RESOLUTION


def _apply_scan_physx(cfg) -> None:
    """Size the contact patch budget for a triangle-mesh terrain."""
    cfg.sim.physx.gpu_max_rigid_patch_count = SCAN_GPU_MAX_RIGID_PATCH_COUNT


def _apply_scan_anchor_yaw(cfg) -> None:
    """Point the anchor goal along the start heading so backward teleports are not sideways."""
    heading = only_sub_terrain(cfg.scene.terrain.terrain_generator).heading
    cfg.curriculum.initialization.anchor_goal[5][0] = heading  # yaw of the (18, 2) anchor layout


def _apply_scan_randomgs_goal_conditioning(cfg) -> None:
    """Pool, anchor and the curriculum-controlled 3D goal command for the RandomGS arms.

    ``default_goal`` stays None, since the curriculum is the sole goal writer.
    """
    goal_poses = only_sub_terrain(cfg.scene.terrain.terrain_generator).goal_poses
    anchor = bridge_anchors_from_goal_poses(goal_poses)[0]
    cfg.curriculum.initialization.anchor_goal = [anchor[0][0], anchor[1][0], anchor[2][0]]
    # the pool is per scan, and the curriculum cfgs below are shared by every scan
    cfg.curriculum.initialization.pool_path = pool_path(cfg.scan)

    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
    )
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d  # noqa: F405
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d  # noqa: F405


# Reward + curricula


@configclass
class ScanSparseRewardsCfg(SparseClimbBoxRewardsCfg):
    """The climb campaign's purely sparse MDP, with ``fell_off_platform`` folded into the failure penalty.

    With no per-step cost, standing still scores zero against the termination penalty, so waiting beats quitting.
    """

    termination = RewTerm(  # noqa: F405
        func=mdp.is_terminated_term,  # noqa: F405
        weight=-100.0,
        params={"term_keys": ["illegal_force", "bad_orientation", "fell_off_platform"]},
    )


@configclass
class ScanPenSparseRewardsCfg(PenSparseClimbBoxRewardsCfg):
    """The climb campaign's penalized sparse MDP on a scan, with ``fell_off_platform`` in the failure penalty."""

    termination = RewTerm(  # noqa: F405
        func=mdp.is_terminated_term,  # noqa: F405
        weight=-100.0,
        params={"term_keys": ["illegal_force", "bad_orientation", "fell_off_platform"]},
    )


@configclass
class ScanBVERCurriculumCfg(TerrainBVERCurriculumCfg):
    """Bidirectional bridge curriculum where some ``bwd_explore`` envs spawn at forward-solved states.

    The mass comes out of ``p_start_candidate``, since in a no-reachability arm the two must sum to one.
    """

    def __post_init__(self):
        self.initialization.p_start_from_solved = 0.1
        self.initialization.p_goal_candidate = 0.9


# Matched controls for the BVER arms (see ``bridge_cfg_random.py``); starts come from the settled pool

_SCAN_RANDOMGS_KWARGS = dict(
    # placeholder; rewritten per scan in `_apply_scan_randomgs_goal_conditioning`
    pool_path="",
    goal_command_name="base_position",
    # position dims, identical to the BVER arms' `init_subspace`
    init_subspace=list(range(0, 36, 2)),
)


@configclass
class ScanRandomGSCurriculumCfg:
    """Branch-mirrored control for the bidirectional arm."""

    initialization = RandomGoalStartCfg(fwd_ratio=0.5, **_SCAN_RANDOMGS_KWARGS)


@configclass
class ScanRandomGSFwdCurriculumCfg:
    """Goal-variety-only control for the forward arm: start pose fixed, pool goals."""

    initialization = RandomGoalStartCfg(fwd_ratio=1.0, **_SCAN_RANDOMGS_KWARGS)


@configclass
class ScanRandomGSBwdCurriculumCfg:
    """Uniform-reverse-curriculum control for the backward arm: pool starts, anchor goal."""

    initialization = RandomGoalStartCfg(fwd_ratio=0.0, **_SCAN_RANDOMGS_KWARGS)


# The arms


@configclass
class ScanEnvCfg(TerrainDerivedTaskSpaceEnvCfg_INIT):
    """Base binding for a rock scan; ``scan`` is bound per terrain by ``scan_families.py``.

    Observes env-relative base xyz plus yaw, so checkpoints are not compatible with plain-observation tasks.
    """

    @property
    def scan(self) -> ScanSpec:
        """Which rock this arm runs on, bound per subclass by ``scan_families.build_scan_family``.

        Stored under a dunder name so ``configclass`` and Hydra never treat it as an overridable field.
        """
        spec = getattr(type(self), "__scan_spec__", None)
        if spec is None:
            raise ValueError(
                f"{type(self).__name__} is not bound to a scan. The arms in this module are generic;"
                " use the per-scan classes in scan_families.py, which bind one."
            )
        return spec

    # grid size does not affect iteration time; a small grid saves memory and every tile is the same mesh
    terrain_num_rows: int = 2
    terrain_num_cols: int = 2

    observations = XYZObservationsCfg()  # noqa: F405
    rewards = ScanSparseRewardsCfg()
    z_margin: float = 0.3
    fall_margin: float = 0.0
    filtered_voronoi_samples: bool = False  # a solid surface: the footprint tiles its own AABB

    def _build_terrain_gen_cfg(self) -> TerrainGeneratorCfg:
        self.scan.require_measured()
        return self.scan.generator_cfg(
            num_rows=self.terrain_num_rows, num_cols=self.terrain_num_cols, difficulty=self.pin_difficulty
        )

    def _episode_time_s(self) -> float | None:
        """Seconds per episode, or ``None`` to keep the family default.

        A hook rather than a field because it must be known before ``super().__post_init__()``.
        """
        return self.scan.episode_time_s

    def __post_init__(self):
        # must precede super(), where `set_episode_length` derives timings from it
        episode_time_s = self._episode_time_s()
        if episode_time_s is not None:
            self.episode_time_s = episode_time_s
        super().__post_init__()
        _apply_scan_task_space(self)
        _apply_scan_physx(self)


@configclass
class ScanEnvCfg_EVAL(TerrainDerivedTaskSpaceEnvCfg_EVAL):
    """Batch/sweep eval base: the eval grid and quality render, on a scan."""

    scan = ScanEnvCfg.scan  # the same property; see ScanEnvCfg.scan

    observations = XYZTrajectoryObservationsCfg()  # noqa: F405
    rewards = ScanSparseRewardsCfg()
    z_margin: float = 0.3
    fall_margin: float = 0.0
    filtered_voronoi_samples: bool = False

    def _build_terrain_gen_cfg(self) -> TerrainGeneratorCfg:
        self.scan.require_measured()
        return self.scan.generator_cfg(
            num_rows=self.terrain_num_rows, num_cols=self.terrain_num_cols, difficulty=self.pin_difficulty
        )

    def __post_init__(self):
        # must precede super(), where `set_episode_length` derives timings from it
        if self.scan.episode_time_s is not None:
            self.episode_time_s = self.scan.episode_time_s
        super().__post_init__()
        _apply_scan_task_space(self)
        _apply_scan_physx(self)


@configclass
class ScanEnvCfg_BVER(ScanEnvCfg):
    observations = XYZTrajectoryObservationsCfg()  # noqa: F405
    curriculum = ScanBVERCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_bver_goal_conditioning(self)
        _apply_scan_anchor_yaw(self)
        self.curriculum.initialization.brownian_horizon_forward = SCAN_BROWNIAN_HORIZON
        self.curriculum.initialization.brownian_horizon_backward = SCAN_BROWNIAN_HORIZON


@configclass
class ScanEnvCfg_BVER_FWD(ScanEnvCfg_BVER):
    """Forward-only ablation arm."""

    curriculum = TerrainBVERFwdCurriculumCfg()


@configclass
class ScanEnvCfg_BVER_BWD(ScanEnvCfg_BVER):
    """Backward-only ablation arm."""

    curriculum = TerrainBVERBwdCurriculumCfg()


@configclass
class ScanEnvCfg_BVER_PEN(ScanEnvCfg_BVER):
    """Bidirectional BVER with the penalized sparse MDP (:class:`ScanPenSparseRewardsCfg`).

    The scale-0.5 recipe is applied per yaml, not here.
    """

    rewards = ScanPenSparseRewardsCfg()


@configclass
class ScanEnvCfg_BVER_PLAY(ScanEnvCfg):
    """Single tile, one goal, quality render: for looking at a checkpoint, with shorter episodes."""

    curriculum = ClimbBoxEvalCurriculumCfg()

    terrain_num_rows: int = 1
    terrain_num_cols: int = 1

    def _episode_time_s(self) -> float:
        return PLAY_EPISODE_TIME_S

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        self.sim.render = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)  # noqa: F405
        _apply_scan_viewer(self)

        apply_eval_goal_conditioning(self)


@configclass
class ScanEnvCfg_BVER_PLAY_SCALE05(ScanEnvCfg_BVER_PLAY):
    """:class:`ScanEnvCfg_BVER_PLAY` at :data:`ACTION_SCALE_HALF`, for checkpoints trained at that scale.

    Replaying them on the full-scale binding doubles joint amplitude and fails silently."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class ScanEnvCfg_BVER_EVAL(ScanEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        # after super(), which is where the bridge eval base sets its own camera
        _apply_scan_viewer(self)
        apply_eval_goal_conditioning(self)


def _apply_scan_multi_goal_sampling(cfg) -> None:
    """Free the goal for the pairs-eval bindings, the scan twin of ``_apply_dtsg_multi_goal_sampling``.

    Otherwise ``_resample_command`` overwrites every queued goal with ``default_goal`` on reset.
    """
    cfg.commands.base_position.default_goal = None


@configclass
class ScanEnvCfg_BVER_MGPLAY(ScanEnvCfg_BVER_PLAY):
    """Interactive pairs eval driven by ``scripts/rsl_rl/eval_scan_pairs.py --play``."""

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_scan_multi_goal_sampling(self)


@configclass
class ScanEnvCfg_BVER_MGEVAL(ScanEnvCfg_BVER_EVAL):
    """Batch/sweep pairs eval: EVAL's grid and trajectory observations with the goal freed."""

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_scan_multi_goal_sampling(self)


@configclass
class ScanEnvCfg_BVER_EVAL_SCALE05(ScanEnvCfg_BVER_EVAL):
    """:class:`ScanEnvCfg_BVER_EVAL` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class ScanEnvCfg_BVER_MGPLAY_SCALE05(ScanEnvCfg_BVER_MGPLAY):
    """:class:`ScanEnvCfg_BVER_MGPLAY` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class ScanEnvCfg_BVER_MGEVAL_SCALE05(ScanEnvCfg_BVER_MGEVAL):
    """:class:`ScanEnvCfg_BVER_MGEVAL` at :data:`ACTION_SCALE_HALF`."""

    def __post_init__(self):
        super().__post_init__()
        apply_action_scale(self, ACTION_SCALE_HALF)


@configclass
class ScanEnvCfg_Baseline(ScanEnvCfg):
    """Vanilla PPO: the BVER env with the initialization curriculum removed (see ``bridge_cfg_baseline.py``)."""

    # plain xyz group: same policy input, without the trajectory buffer only curricula read
    observations = XYZObservationsCfg()  # noqa: F405
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        apply_eval_goal_conditioning(self)
        apply_baseline_spawn(self)
        # without a curriculum the static spawn is the whole initial-state distribution, so use eval_bounds
        self.events.reset_base.params["pose"]["x"] = tuple(self.task_space.eval_bounds[0])
        self.events.reset_base.params["pose"]["y"] = tuple(self.task_space.eval_bounds[2])
        self.events.reset_base.params["pose"]["yaw"] = float(
            only_sub_terrain(self.scene.terrain.terrain_generator).heading
        )


@configclass
class ScanEnvCfg_Baseline_PEN(ScanEnvCfg_Baseline):
    """Vanilla PPO on the penalized sparse MDP; the uninformed control for :class:`ScanEnvCfg_BVER_PEN`."""

    rewards = ScanPenSparseRewardsCfg()


@configclass
class ScanEnvCfg_GenFeasibleStarts(ScanEnvCfg):
    """Generation-only env for the feasible-start pool: reset, settle, snapshot."""

    curriculum = GenFeasibleStartsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.use_cache = True
        # drop from above the high ground; the full standing band would put most starts inside rock
        surface_top_z = only_sub_terrain(self.scene.terrain.terrain_generator).surface_top_z
        self.task_space.bounds[4] = [
            surface_top_z + height_offset,
            surface_top_z + height_offset + 0.3,
        ]


@configclass
class ScanEnvCfg_RandomGS(ScanEnvCfg):
    """Branch-mirrored control, matching the BVER arm's observations, terminations and goal wiring."""

    # same as the baseline: nothing reads the trajectory buffer here
    observations = XYZObservationsCfg()  # noqa: F405
    curriculum = ScanRandomGSCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_scan_randomgs_goal_conditioning(self)


@configclass
class ScanEnvCfg_RandomGS_FWD(ScanEnvCfg_RandomGS):
    """Goal-variety-only arm (all envs forward)."""

    curriculum = ScanRandomGSFwdCurriculumCfg()


@configclass
class ScanEnvCfg_RandomGS_BWD(ScanEnvCfg_RandomGS):
    """Uniform-start-only arm (all envs backward)."""

    curriculum = ScanRandomGSBwdCurriculumCfg()
