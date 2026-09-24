"""Uninformed random (start, goal) control arms matched to the BVER arms in ``bridge_cfg_bver.py``.

They match BVER's branch marginals, draw starts and goals from a settled feasible pool, and mirror its
env split, so only the proposal source differs. The three arms differ only in ``fwd_ratio``.
"""

import math

from isaaclab.utils import configclass

from ... import mdp
from ...curriculum import RandomGoalStartCfg
from ...curriculum.task_space.bridge_task_space_cfg import bridge_anchors_from_goal_poses
from ...mdp.commands import CurriculumGoalCfg
from ..common.env_cfg import XYZTrajectoryObservationsCfg
from informed_exploration.terrains.config.layouts import dtsg_route_rects

from ..common.curricula import (
    GenFeasibleStartsCurriculumCfg,
    TerrainBVERBwdCurriculumCfg,
    TerrainBVERCurriculumCfg,
    TerrainBVERFwdCurriculumCfg,
)
from ..common.goal_env_cfg import TerrainRCTerminationsCfg, only_sub_terrain
from .bridge_cfg import IEAnymalDBridgeDTSGEnvCfg

DTSG_FEASIBLE_POOL = "informed-exploration/data/parkour/feasible_starts_dtsg.pkl"
"""Pool consumed by every DTSG RandomGS arm, repo-root-relative; produced by the generation task below."""

DTSG_SURFACE_TOP_Z = 0.55
"""Env-local z of DTSG's highest walkable surface; pass as ``--surface_top_z`` to the pool generator."""


# Pool generation


@configclass
class IEAnymalDBridgeDTSGEnvCfg_GenFeasibleStarts(IEAnymalDBridgeDTSGEnvCfg):
    """Generation-only env for the DTSG feasible-start pool; resets, settles and snapshots without a goal."""

    curriculum = GenFeasibleStartsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.use_cache = True


DTSG_CORRIDOR_POOL = "informed-exploration/data/parkour/feasible_starts_dtsg_corridor.pkl"
"""Pool consumed by the corridor-start eval, confined to the three routes; separate from :data:`DTSG_FEASIBLE_POOL`."""


CORRIDOR_START_YAW_RANGE = (-math.pi / 4.0, math.pi / 4.0)
"""Yaw window (rad) a corridor start may settle in; yaw zero faces down-route toward the goal."""


@configclass
class IEAnymalDBridgeDTSGEnvCfg_GenCorridorStarts(IEAnymalDBridgeDTSGEnvCfg_GenFeasibleStarts):
    """Generation-only env for the corridor-start pool, confined to the three routes via ``keep_in_filter``.

    Includes each route's entry pad, which the sealed start column keeps unambiguous per route.
    """

    seal_start_column: bool = True

    def __post_init__(self):
        super().__post_init__()
        sub = only_sub_terrain(self.scene.terrain.terrain_generator)
        rects = dtsg_route_rects(sub.layout, sub.size, include_start_column=True)
        self.task_space.keep_in_filter = [((x_lo, x_hi), (y_lo, y_hi)) for x_lo, x_hi, y_lo, y_hi in rects.values()]
        # The yaw position dim.
        self.task_space.bounds[10] = list(CORRIDOR_START_YAW_RANGE)


# Random (start, goal) control arms

# Shared by all three arms; `anchor_goal` is set per instance by the goal-conditioning helper below.
_RANDOMGS_KWARGS = dict(
    pool_path=DTSG_FEASIBLE_POOL,
    goal_command_name="base_position",
    # Position dims, identical to the BVER arms' `init_subspace`.
    init_subspace=list(range(0, 36, 2)),
)


@configclass
class BridgeRandomGSCurriculumCfg:
    """Branch-mirrored control for ``TerrainBVERCurriculumCfg``: pad start with pool goal, or pool start with anchor."""

    initialization = RandomGoalStartCfg(fwd_ratio=0.5, **_RANDOMGS_KWARGS)


@configclass
class BridgeRandomGSFwdCurriculumCfg:
    """Goal-variety-only control for ``TerrainBVERFwdCurriculumCfg``: pad starts with uniform pool goals."""

    initialization = RandomGoalStartCfg(fwd_ratio=1.0, **_RANDOMGS_KWARGS)


@configclass
class BridgeRandomGSBwdCurriculumCfg:
    """Start-variety-only control for ``TerrainBVERBwdCurriculumCfg``: pool starts with the anchor goal."""

    initialization = RandomGoalStartCfg(fwd_ratio=0.0, **_RANDOMGS_KWARGS)


def _apply_bridge_randomgs_goal_conditioning(cfg) -> None:
    """Wire the curriculum-controlled 3D goal command and the terrain's anchor position.

    ``default_goal`` stays None, since otherwise ``_resample_command`` would clobber the curriculum goal.
    """
    goal_poses = only_sub_terrain(cfg.scene.terrain.terrain_generator).goal_poses
    anchor = bridge_anchors_from_goal_poses(goal_poses)[0]
    cfg.curriculum.initialization.anchor_goal = [anchor[0][0], anchor[1][0], anchor[2][0]]

    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
    )

    # 3D goal observation, since goals are not coplanar with the start platform.
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d


@configclass
class IEAnymalDBridgeDTSGEnvCfg_RandomGS(IEAnymalDBridgeDTSGEnvCfg):
    """Branch-mirrored control matching ``IEAnymalDBridgeDTSGEnvCfg_BVER`` in observations, terminations and goals."""

    observations = XYZTrajectoryObservationsCfg()
    curriculum = BridgeRandomGSCurriculumCfg()
    terminations = TerrainRCTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_bridge_randomgs_goal_conditioning(self)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_RandomGS_FWD(IEAnymalDBridgeDTSGEnvCfg_RandomGS):
    """Goal-variety-only arm (all forward)."""

    curriculum = BridgeRandomGSFwdCurriculumCfg()


@configclass
class IEAnymalDBridgeDTSGEnvCfg_RandomGS_BWD(IEAnymalDBridgeDTSGEnvCfg_RandomGS):
    """Uniform-start-only arm (all backward)."""

    curriculum = BridgeRandomGSBwdCurriculumCfg()
