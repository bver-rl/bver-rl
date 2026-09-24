# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env config for tasks whose task space is derived from terrain metadata.

Bridge layouts and mesh scans publish ``geometry_bounds``, ``keep_in_regions`` and ``perimeter_wall``,
which this base turns into the task space, the spawn and the fall termination.
"""

from typing import cast

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from .env_cfg import *
from .eval_curricula import ClimbBoxEvalCurriculumCfg
from .rewards import ClimbBoxActionsCfg, ClimbBoxRewardsCfg
from .terminations import ClimbBoxTerminationsCfg

# `mdp`, `DoneTerm`, `RewTerm`, `configclass` and `MISSING` arrive via the star import above
from ...curriculum.task_space.bridge_task_space_cfg import BridgeTaskSpaceCfg

from informed_exploration.terrains.bridge import PlatformBridgeTerrainCfg


def only_sub_terrain(gen_cfg: TerrainGeneratorCfg) -> PlatformBridgeTerrainCfg:
    """Return the terrain generator's single sub-terrain, regardless of its key."""
    return cast(PlatformBridgeTerrainCfg, next(iter(gen_cfg.sub_terrains.values())))


_PERIMETER_WALL_STANDOFF = 0.3
"""Clearance [m] between the task-space x/y bounds and the fence's inner faces."""


def clamp_task_space_to_perimeter_wall(task_space, sub_terrain: PlatformBridgeTerrainCfg) -> None:
    """Pull the task space's x/y bounds inside the arena fence's inner faces; a no-op without a fence.

    Walls are mesh-only and invisible to the keep-in metadata, so without this, draws can land on the wall top.
    """
    fence = sub_terrain.perimeter_wall
    if fence is None:
        return
    geometry_bounds = sub_terrain.geometry_bounds
    assert geometry_bounds is not None  # caller asserted before deriving the task space
    margin_x, margin_y = fence.margin

    # eval_bounds too, since a multi-start window can reach under the fence
    for bounds in (task_space.bounds, task_space.eval_bounds):
        bounds[0] = [  # x
            max(bounds[0][0], geometry_bounds["x_lo"] - margin_x + _PERIMETER_WALL_STANDOFF),
            min(bounds[0][1], geometry_bounds["x_hi"] + margin_x - _PERIMETER_WALL_STANDOFF),
        ]
        bounds[2] = [  # y
            max(bounds[2][0], geometry_bounds["y_lo"] - margin_y + _PERIMETER_WALL_STANDOFF),
            min(bounds[2][1], geometry_bounds["y_hi"] + margin_y - _PERIMETER_WALL_STANDOFF),
        ]


class TerrainTerminationsCfg(ClimbBoxTerminationsCfg):
    """ClimbBox terminations plus a "fell off the platform" detector.

    ``minimum_height`` is written by :meth:`TerrainDerivedTaskSpaceEnvCfg.__post_init__` from ``geometry_bounds``.
    """

    fell_off_platform = DoneTerm(
        func=mdp.root_height_below_env_minimum,
        params={"minimum_height": MISSING},
        # A real failure: time_out=True would zero the termination penalty and bootstrap the value
        time_out=False,
    )


@configclass
class TerrainRCTerminationsCfg(TerrainTerminationsCfg):
    """Bridge terminations extended with the Brownian-motion timeout for random walkers."""

    brownian_motion_time_out = DoneTerm(func=mdp.brownian_motion_time_out, time_out=True)


@configclass
class TerrainRewardsCfg(ClimbBoxRewardsCfg):
    """ClimbBox rewards with a progress term and the fall folded into the termination penalty.

    A new DoneTerm is silently unpenalized unless listed in ``term_keys``.
    """

    tracking_pos = None

    progress = RewTerm(
        func=mdp.goal_closing_speed,
        weight=25.0,
        params={"command_name": "base_position", "max_speed": 1.5, "deadband": 0.25},
    )

    termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={"term_keys": ["illegal_force", "bad_orientation", "fell_off_platform"]},
    )


@configclass
class TerrainDerivedTaskSpaceEnvCfg(IEParkourEnvCfg):
    """Base config for a BVER bridge-terrain task; subclasses bind a preset via `_terrain_preset`."""

    pin_difficulty: float = 0.75
    """Fixed terrain difficulty in [0, 1] for every env; one is the hardest geometry."""

    z_margin: float = 0.3
    """Headroom [m] above the highest platform for the task space's z-bounds."""

    filtered_voronoi_samples: bool = False
    """Restrict BVER's Voronoi attractors to the terrain's walkable footprints.

    Only useful on layouts whose platforms do not tile their bounding box.
    """

    fall_margin: float = 0.0
    """Drop [m] below the lowest platform top before `fell_off_platform` fires (env-relative)."""

    terrain_num_rows: int = 5
    terrain_num_cols: int = 5

    episode_time_s: float = 10.0
    """Episode length [s], applied via :func:`set_episode_length`."""

    actions = ClimbBoxActionsCfg()
    rewards = TerrainRewardsCfg()
    terminations = TerrainTerminationsCfg()
    curriculum = None

    def _terrain_preset(self):
        """Return the unbound preset function this env binds to."""
        raise NotImplementedError

    def _build_terrain_gen_cfg(self) -> TerrainGeneratorCfg:
        """Build this env's pinned `TerrainGeneratorCfg`, by default from `_terrain_preset`."""
        return _pinned_bridge_terrain_gen_cfg(
            self._terrain_preset(), self.pin_difficulty, num_rows=self.terrain_num_rows, num_cols=self.terrain_num_cols
        )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.viewer.eye = (5.7, 8.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, -3.2)

        # Seed and cache must be set on this fresh object, which replaces the scene's generator
        gen_cfg = self._build_terrain_gen_cfg()
        gen_cfg.seed = 0
        gen_cfg.use_cache = True
        self.scene.terrain.terrain_generator = gen_cfg
        sub_terrain = only_sub_terrain(gen_cfg)
        geometry_bounds = sub_terrain.geometry_bounds
        # A `patch_override` skips layout resolution and leaves derived fields unset
        assert geometry_bounds is not None, (
            "sub-terrain has no derived `geometry_bounds`; a `patch_override` skips the layout"
            " resolution that produces it, so the task space cannot be derived from this terrain."
        )

        self.task_space = BridgeTaskSpaceCfg.from_geometry_bounds(
            geometry_bounds, z_margin=self.z_margin, keep_in_regions=sub_terrain.keep_in_regions
        )
        # Keep draws off the perimeter wall top, which the metadata does not see
        clamp_task_space_to_perimeter_wall(self.task_space, sub_terrain)
        # Fall threshold from the same geometry as the task-space z-bounds
        self.terminations.fell_off_platform.params["minimum_height"] = geometry_bounds["z_lo"] - self.fall_margin

        set_episode_length(self, self.episode_time_s)


@configclass
class TerrainDerivedTaskSpaceEnvCfg_INIT(TerrainDerivedTaskSpaceEnvCfg):
    """Override env-level resets needed for state-space initialization curricula (BVER)."""

    events = InitializationEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.base_external_force_torque = None


@configclass
class TerrainDerivedTaskSpaceEnvCfg_EVAL(TerrainDerivedTaskSpaceEnvCfg_INIT):
    """Batch/sweep eval base: bigger terrain grid, trajectory observations, quality render."""

    observations = TrajectoryObservationsCfg()
    curriculum = ClimbBoxEvalCurriculumCfg()
    terrain_num_rows: int = 2
    terrain_num_cols: int = 2

    def __post_init__(self):
        super().__post_init__()
        # use_cache is already set by TerrainDerivedTaskSpaceEnvCfg.__post_init__

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        self.sim.render = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)
        self.viewer.eye = (-12.0, -12.0, 11.5)
        self.viewer.lookat = (0.0, 0.0, -1.5)
        self.viewer.resolution = (1920, 1080)
