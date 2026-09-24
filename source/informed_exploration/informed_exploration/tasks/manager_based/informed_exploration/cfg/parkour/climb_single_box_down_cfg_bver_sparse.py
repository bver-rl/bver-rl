"""Climb-DOWN sparse-reward BVER ablation arms (ANYmal-D only).

Reuses the climb-up sparse env cfgs and re-points only the direction-specific pieces, so up and down differ
only in task direction. Each arm is registered as a box-height twin.
"""

from isaaclab.utils import configclass

from ...terrains.climb_box import CLIMB_BOX_DOWN_SUBTERRAINS_SINGLE

from ...curriculum.task_space.parkour_task_space_cfg import height_offset

from .climb_single_box_cfg_bver_sparse import (
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_BWD,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_FWD,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGEVAL,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGPLAY,
    IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY,
)

# Far-side ground target x for climb-down, in the robot-state frame rather than the terrain patch frame
_CLIMB_DOWN_TARGET_X = -3.0


def _apply_climb_down_overrides(cfg) -> None:
    """Re-point an up climb env onto the climb-down subterrain and ground goal.

    Must be called after ``super().__post_init__()``.
    """
    tg = cfg.scene.terrain.terrain_generator
    # Terrain swap
    tg.sub_terrains = CLIMB_BOX_DOWN_SUBTERRAINS_SINGLE
    tg.sub_terrains["big_box_down"].height_range = (cfg.target_box_height, cfg.target_box_height)

    # Move the default spawn to the box top; only INIT-based flavors carry a "pose" dict
    reset_params = cfg.events.reset_base.params
    if "pose" in reset_params:
        reset_params["pose"]["x"] = -3.0
        reset_params["pose"]["z"] = height_offset

    # Eval spawn region on the box top instead of the ground before it
    cfg.task_space.eval_bounds[0] = [-0.2, 0.2]  # x: on box top, in the state frame
    cfg.task_space.eval_bounds[4] = [  # z: standing height on box top
        cfg.target_box_height + height_offset,
        cfg.target_box_height + height_offset,
    ]

    # increase bounds z
    cfg.task_space.bounds[4][1] = cfg.target_box_height + height_offset + 0.2

    # Reverse-curriculum goal on the ground beyond the box, for curricula that carry a ``goal``
    if hasattr(cfg.curriculum, "initialization") and hasattr(cfg.curriculum.initialization, "goal"):
        init = cfg.curriculum.initialization
        init.goal[0][0] = _CLIMB_DOWN_TARGET_X  # x, far-side ground in the state frame
        init.goal[1][0] = 0.0  # y
        init.goal[2][0] = height_offset  # z, nominal standing height on the ground

    # BVER anchor to the same target; already deep-copied by the up preset, so in-place is safe
    if hasattr(cfg.curriculum, "initialization") and getattr(cfg.curriculum.initialization, "anchor_goal", None):
        anchor = cfg.curriculum.initialization.anchor_goal
        anchor[0][0] = _CLIMB_DOWN_TARGET_X  # x, far-side ground in the state frame
        anchor[1][0] = 0.0  # y
        anchor[2][0] = height_offset  # z, nominal standing height on the ground


def _flip_default_goal_to_climb_down(cfg) -> None:
    """Re-point a fixed on-box ``default_goal`` to the climb-down ground target; a no-op when it is ``None``."""
    if cfg.commands.base_position.default_goal is not None:
        cfg.commands.base_position.default_goal = [_CLIMB_DOWN_TARGET_X, 0.0, height_offset]


# Training arms


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_FWD(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_FWD):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_BWD(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_BWD):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)


# Play and eval


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_PLAY(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_PLAY):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)
        _flip_default_goal_to_climb_down(self)


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_MGPLAY(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGPLAY):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_EVAL(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_EVAL):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)
        _flip_default_goal_to_climb_down(self)


@configclass
class IEAnymalDClimbSingleBoxDownEnvCfg_BVER_SPARSE_MGEVAL(IEAnymalDClimbSingleBoxEnvCfg_BVER_SPARSE_MGEVAL):

    def __post_init__(self):
        super().__post_init__()
        _apply_climb_down_overrides(self)
