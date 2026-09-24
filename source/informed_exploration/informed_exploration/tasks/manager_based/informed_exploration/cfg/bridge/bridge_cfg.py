"""Base env configs for BVER bridge-terrain tasks; the curriculum wiring lives in ``bridge_cfg_bver.py``.

Terrain comes from ``informed_exploration.terrains`` and the task space from its ``geometry_bounds``.
Every env pins one difficulty and the forward direction.
"""

from typing import cast

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from ..common.env_cfg import *
from ..common.goal_env_cfg import (
    TerrainDerivedTaskSpaceEnvCfg_EVAL,
    TerrainDerivedTaskSpaceEnvCfg_INIT,
    only_sub_terrain,
)

# `mdp`, `DoneTerm`, `RewTerm`, `configclass` and `MISSING` all arrive via the star import above.
from ...curriculum.task_space.bridge_task_space_cfg import BridgeTaskSpaceCfg

# `informed_exploration.terrains` is a sibling of `tasks/`, hence the absolute import.
from informed_exploration.terrains.bridge.bridges import NarrowBeamBridgeCfg
from informed_exploration.terrains.config.factory import DEFAULT_TILE_SIZE, pinned_layout_cfg, single_type_cfg
from informed_exploration.terrains.config.layouts.dtsg import (
    DTSG_POSE_OFFSET_X,
    DTSG_SUBTERRAIN_NAME,
    DTSG_TILE_SIZE,
    build_dtsg_layout,
    dtsg_perimeter_wall,
    dtsg_walls,
)


def _pinned_bridge_terrain_gen_cfg(
    preset_fn,
    pin_difficulty: float,
    num_rows: int = 1,
    num_cols: int = 1,
    size: tuple[float, float] = DEFAULT_TILE_SIZE,
) -> TerrainGeneratorCfg:
    """Build a BVER-ready `TerrainGeneratorCfg` from a bridge preset, pinned to one difficulty, forward only."""
    base = preset_fn()
    fwd = base.sub_terrains["forward"]
    gen_cfg = single_type_cfg(
        fwd.layout, fwd.bridge, num_rows=num_rows, num_cols=num_cols, size=size, reverse_columns=False
    )
    gen_cfg.difficulty_range = (pin_difficulty, pin_difficulty)
    return gen_cfg


def center_dtsg_start_window_on_pads(cfg) -> None:
    """Shift DTSG's initial-state window forward by ``DTSG_POSE_OFFSET_X`` onto the start pads' centers.

    Used by the eval and play bases only, not training. Call exactly once per config, since it shifts
    relatively; the window is then re-clamped to the fence interior.
    """
    task_space = cfg.task_space
    assert task_space.eval_bounds is not None  # set by TerrainDerivedTaskSpaceEnvCfg.__post_init__
    task_space.eval_bounds[0] = [x + DTSG_POSE_OFFSET_X for x in task_space.eval_bounds[0]]
    gen_cfg = cast(TerrainGeneratorCfg, cfg.scene.terrain.terrain_generator)
    clamp_task_space_to_perimeter_wall(task_space, only_sub_terrain(gen_cfg))


# Terminations / rewards configs


@configclass
class IEAnymalDBridgeDTSGEnvCfg(TerrainDerivedTaskSpaceEnvCfg_INIT):
    """DTSG: one goal reached by three routes (stairs, pillars, ramp) on floor-level platforms.

    Floor-level, so ``fell_off_platform`` is inert by design. Every binding observes env-relative xyz and yaw
    to make the route choice learnable, so DTSG checkpoints are not obs-compatible with other terrains.
    """

    observations = XYZObservationsCfg()
    z_margin: float = 0.6
    multiple_start_poses: bool = True
    """Widen the initial-state window (``eval_bounds``) to span the whole left start column of adjacent pads.

    Inert on arms that place their own starts, and forced off when ``seal_start_column`` is set.
    """
    seal_goal_column: bool = False
    """Continue the separator walls through the goal column so each route exits into its own pocket.

    Used by the corridor evals; changes the mesh, so sealed and unsealed runs are not comparable.
    """
    seal_start_column: bool = False
    """Extend the separator walls through the start column, for the corridor-start eval.

    Mutually exclusive with ``seal_goal_column``.
    """

    def _build_terrain_gen_cfg(self) -> TerrainGeneratorCfg:
        return pinned_layout_cfg(
            # Sealed start columns are incompatible with the widened window, so derive it here.
            build_dtsg_layout(self.multiple_start_poses and not self.seal_start_column),
            NarrowBeamBridgeCfg(),  # every spoke sets its own bridge; this default is never used
            self.pin_difficulty,
            DTSG_SUBTERRAIN_NAME,
            size=DTSG_TILE_SIZE,
            num_rows=self.terrain_num_rows,
            num_cols=self.terrain_num_cols,
            walls=dtsg_walls(self.seal_goal_column, self.seal_start_column),
            perimeter_wall=dtsg_perimeter_wall(),
        )


@configclass
class IEAnymalDBridgeDTSGEnvCfg_EVAL(TerrainDerivedTaskSpaceEnvCfg_EVAL):
    observations = XYZTrajectoryObservationsCfg()  # xyz everywhere on DTSG
    z_margin: float = 1.2
    multiple_start_poses: bool = True
    """See ``IEAnymalDBridgeDTSGEnvCfg.multiple_start_poses``; redeclared since the eval bases are a sibling branch."""
    seal_goal_column: bool = False
    """See ``IEAnymalDBridgeDTSGEnvCfg.seal_goal_column``."""
    seal_start_column: bool = False
    """See ``IEAnymalDBridgeDTSGEnvCfg.seal_start_column``."""

    def __post_init__(self):
        super().__post_init__()
        center_dtsg_start_window_on_pads(self)

    def _build_terrain_gen_cfg(self) -> TerrainGeneratorCfg:
        return pinned_layout_cfg(
            # see the training base's `_build_terrain_gen_cfg` for the `seal_start_column` guard
            build_dtsg_layout(self.multiple_start_poses and not self.seal_start_column),
            NarrowBeamBridgeCfg(),
            self.pin_difficulty,
            DTSG_SUBTERRAIN_NAME,
            size=DTSG_TILE_SIZE,
            num_rows=self.terrain_num_rows,
            num_cols=self.terrain_num_cols,
            walls=dtsg_walls(self.seal_goal_column, self.seal_start_column),
            perimeter_wall=dtsg_perimeter_wall(),
        )


