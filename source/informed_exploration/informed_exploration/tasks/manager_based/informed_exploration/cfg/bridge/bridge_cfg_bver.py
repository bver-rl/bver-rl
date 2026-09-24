"""BVER curriculum wiring for bridge-terrain tasks, plus the play, eval and RSI bindings.

Three arms (bidirectional, forward-only, backward-only); anchors come from the terrain's ``goal_poses``.
"""

from typing import Any

from isaaclab.utils import configclass

from ..common.curricula import (
    TerrainBVERBwdCurriculumCfg,
    TerrainBVERCurriculumCfg,
    TerrainBVERFwdCurriculumCfg,
)
from ..common.goal_conditioning import (
    apply_eval_goal_conditioning,
    apply_bver_goal_conditioning,
)
from ..common.goal_env_cfg import (
    TerrainRCTerminationsCfg,
    only_sub_terrain,
)
from .bridge_cfg import (
    IEAnymalDBridgeDTSGEnvCfg,
    IEAnymalDBridgeDTSGEnvCfg_EVAL,
    center_dtsg_start_window_on_pads,
)
from ..common.env_cfg import *
from ..common.eval_curricula import ClimbBoxEvalCurriculumCfg, ClimbBoxMultiGoalEvalCurriculumCfg
from ...curriculum import RSICfg
from ...curriculum.task_space.parkour_task_space_cfg import height_offset

# `informed_exploration.terrains` is a sibling of `tasks/`, hence the absolute import.
from informed_exploration.terrains.bridge.interface import lerp


# DTSG (different terrain single goal); localized obs make checkpoints incompatible with other terrains.


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER(IEAnymalDBridgeDTSGEnvCfg):
    observations = XYZTrajectoryObservationsCfg()
    curriculum = TerrainBVERCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_FWD(IEAnymalDBridgeDTSGEnvCfg):
    observations = XYZTrajectoryObservationsCfg()
    curriculum = TerrainBVERFwdCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_BWD(IEAnymalDBridgeDTSGEnvCfg):
    observations = XYZTrajectoryObservationsCfg()
    curriculum = TerrainBVERBwdCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_bver_goal_conditioning(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_PLAY(IEAnymalDBridgeDTSGEnvCfg):
    """Interactive single-viewport play and the base of every other DTSG play binding.

    Batch eval tasks keep the training episode length so their success rates stay comparable.
    """

    curriculum = ClimbBoxEvalCurriculumCfg()

    episode_time_s: float = 6.0
    """Episode length (s), shorter than training so clips do not idle on the goal; slow traverses time out.

    A field, not a Hydra override, since ``set_episode_length`` consumes it in ``__post_init__``.
    """

    terrain_num_rows: int = 1
    terrain_num_cols: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        self.sim.render = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)
        
        self.viewer.resolution = (1920, 1080)

        apply_eval_goal_conditioning(self)
        _apply_dtsg_single_tile_view(self)
        # Same forward-shifted start window as the eval bindings.
        center_dtsg_start_window_on_pads(self)
        
        self.viewer.eye = (3.0, -6.0, 8.0)
        self.viewer.lookat = (3.0, 0.0, 0.0)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_EVAL(IEAnymalDBridgeDTSGEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        apply_eval_goal_conditioning(self)


# DTSG multi-goal eval: goals sampled from the task space by scripts/rsl_rl/eval_multi_goal.py.

_BRIDGE_RISE_FIELDS = ("peak_extra_height_range", "pole_extra_height_range", "noise_amplitude_range")
"""Per-family difficulty-interpolated fields giving how far a bridge climbs above the platform tops."""


def _dtsg_peak_standing_z(cfg) -> float:
    """Return the highest env-relative standing base z on DTSG: the tallest bridge rise plus `height_offset`.

    Derived from the layout at the pinned difficulty, using the harder side of each bridge.
    """

    def _spokes(layout):
        for spoke in layout.spokes:
            yield spoke
            if spoke.child is not None:
                yield from _spokes(spoke.child)

    sub = only_sub_terrain(cfg.scene.terrain.terrain_generator)
    geometry_bounds = sub.geometry_bounds
    assert (
        geometry_bounds is not None
    ), "sub-terrain has no derived `geometry_bounds` (a `patch_override` skips layout resolution)"

    rises: list[float] = []
    for spoke in _spokes(sub.layout):
        bridge = spoke.bridge
        assert bridge is not None, "every DTSG spoke sets its own bridge (the terrain-level default is unused)"
        difficulty = max(bridge.side_difficulties(cfg.pin_difficulty))
        rises += [lerp(difficulty, *getattr(bridge, f)) for f in _BRIDGE_RISE_FIELDS if hasattr(bridge, f)]
    assert rises, "no rise-bearing bridge on the DTSG layout; the cap below would silently be the platform top"

    return geometry_bounds["z_hi"] + max(rises) + height_offset


def _apply_dtsg_multi_goal_sampling(cfg) -> None:
    """Enable task-space goal sampling for the DTSG multi-goal cfgs.

    Clears `default_goal`, which `_resample_command` would otherwise restore on every reset, and caps
    goal z at the peak standing height, since subtracting `z_margin` would collapse the band to a plane.
    """
    cfg.commands.base_position.default_goal = None
    assert cfg.task_space.bounds is not None  # set by TerrainDerivedTaskSpaceEnvCfg.__post_init__
    cfg.task_space.bounds[4][1] = _dtsg_peak_standing_z(cfg)


def _apply_dtsg_single_tile_view(cfg) -> None:
    """Frame the whole DTSG tile instead of the start pad, so the goal end stays in view."""
    cfg.viewer.eye = (3.0, -3.0, 15.0)
    cfg.viewer.lookat = (3.0, 0.0, 0.0)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_MGPLAY(IEAnymalDBridgeDTSGEnvCfg_BVER_PLAY):
    """Interactive multi-goal eval: _BVER_PLAY with task-space-sampled goals."""

    # One shared tile so all sampled goals land on the same geometry; envs do not collide.
    terrain_num_rows: int = 1
    terrain_num_cols: int = 1

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_dtsg_multi_goal_sampling(self)
        _apply_dtsg_single_tile_view(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_MGEVAL(IEAnymalDBridgeDTSGEnvCfg_BVER_EVAL):
    """Batch/sweep multi-goal eval: _BVER_EVAL with task-space-sampled goals."""

    curriculum = ClimbBoxMultiGoalEvalCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_dtsg_multi_goal_sampling(self)


# DTSG corridor eval (eval_multi_goal.py --corridor_goals): per-corridor success reads route traversability.


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_CORRIDOR(IEAnymalDBridgeDTSGEnvCfg_BVER_MGEVAL):
    """Batch/sweep corridor eval: MGEVAL on the sealed terrain."""

    seal_goal_column: bool = True


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_CORRIDORPLAY(IEAnymalDBridgeDTSGEnvCfg_BVER_MGPLAY):
    """Interactive corridor eval: MGPLAY on the sealed terrain."""

    seal_goal_column: bool = True


# DTSG corridor-start eval (eval_corridor_starts.py): pooled mid-route starts, fixed goal, sealed start column.


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_CORRIDORSTART(IEAnymalDBridgeDTSGEnvCfg_BVER_EVAL):
    """Batch/sweep corridor-start eval."""

    seal_start_column: bool = True


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_CORRIDORSTARTPLAY(IEAnymalDBridgeDTSGEnvCfg_BVER_PLAY):
    """Interactive corridor-start eval."""

    seal_start_column: bool = True

    def __post_init__(self):
        super().__post_init__()
        _apply_dtsg_single_tile_view(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_BVER_HARVEST(IEAnymalDBridgeDTSGEnvCfg_BVER_CORRIDORSTART):
    """Corridor-start eval on a shortened episode, used to harvest RSI reference trajectories.

    The short episode trims goal dwell from the pool; it is a field since ``set_episode_length``
    consumes it in ``__post_init__``. Feeds ``scripts/helpers/build_dtsg_rsi_pool.py``.
    """

    episode_time_s: float = 6.0


# RSI (reference-state initialization) for DTSG, over a pool with an equal quota per route.
DTSG_RSI_TRAJECTORY_POOL = "informed-exploration/data/parkour/trajectories_bridge_dtsg.pkl"


@configclass
class BridgeDTSGRSICurriculumCfg:
    """RSI over the harvested DTSG pool.

    ``terrain_level=False`` samples uniformly over the whole pool, as the pinned layout needs no levels.
    """

    initialization = RSICfg(terrain_level=False, one_level=False)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_RSI(IEAnymalDBridgeDTSGEnvCfg):
    """DTSG RSI control arm, trained from harvested expert states on the unsealed terrain.

    ``optimal_trajectory_file`` must be set after ``super().__post_init__()``, and the fixed-goal helper is
    required since RSI writes no goal. A no-trajectories assert at reset means the harvest has not run.
    """

    curriculum = BridgeDTSGRSICurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # AFTER super(): the bridge base rebuilds task_space from the terrain geometry.
        self.task_space.optimal_trajectory_file = DTSG_RSI_TRAJECTORY_POOL
        apply_eval_goal_conditioning(self)
