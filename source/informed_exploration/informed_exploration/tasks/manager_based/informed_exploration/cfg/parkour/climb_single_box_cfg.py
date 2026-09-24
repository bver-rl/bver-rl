
from isaaclab.utils import configclass
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from ..common.env_cfg import *
from ..common.eval_curricula import ClimbBoxEvalCurriculumCfg
from ..common.rewards import (
    ClimbBoxActionsCfg,
    ClimbBoxRewardsCfg,
    SparseClimbBoxRewardsCfg,
)
from ..common.terminations import ClimbBoxTerminationsCfg

from ...terrains.climb_box import CLIMB_BOX_SUBTERRAINS_SINGLE
from ...curriculum.task_space.parkour_task_space_cfg import height_offset

from ...curriculum import (
    RandomCfg,
)
from ...curriculum.task_space.samplers import FeasiblePoolSamplerCfg


    # goal_reached = DoneTerm(
    #     func=mdp.StayedAtTarget,
    #     params={
    #         "distance_threshold": 0.25,
    #         "counter_threshold": 150,
    #         "command_name": "base_position",
    #     },
    #     time_out=False,
    # )


# Reverse-curriculum goal: AnymalD at rest on the box, one [pos, vel] pair per task dim, joints as default offsets
_RC_GOAL_ON_BOX: list[list[float]] = [
    # base pose (x, y, z, roll, pitch, yaw), as (position, velocity)
    [0.0, 0.0],  # x
    [0.0, 0.0],  # y
    [1.01, 0.0],  # z, overridden per env from target_box_height + height_offset
    [0.0, 0.0],  # roll
    [0.0, 0.0],  # pitch
    [0.0, 0.0],  # yaw
    # 12 joints (offset from default, velocity)
    [0.0, 0.0],  # LF_HAA
    [0.0, 0.0],  # LH_HAA
    [0.0, 0.0],  # RF_HAA
    [0.0, 0.0],  # RH_HAA
    [0.0, 0.0],  # LF_HFE
    [0.0, 0.0],  # LH_HFE
    [0.0, 0.0],  # RF_HFE
    [0.0, 0.0],  # RH_HFE
    [0.0, 0.0],  # LF_KFE
    [0.0, 0.0],  # LH_KFE
    [0.0, 0.0],  # RF_KFE
    [0.0, 0.0],  # RH_KFE
]


@configclass
class ClimbBoxFeasibleStartsCurriculumCfg:
    """Random curriculum that draws initial states from the feasible-starts pool."""

    initialization = RandomCfg(
        sampler_cfg=FeasiblePoolSamplerCfg(
            pool_path="informed-exploration/data/parkour/feasible_starts_climb_box_0p41.pkl",
            subspace=None,  # use full robot state from pool
        ),
    )


@configclass
class ClimbBoxGenFeasibleStartsCurriculumCfg:
    """Random curriculum used by the feasible-starts generator (filter enabled)."""

    initialization = RandomCfg(filtered_samples=True, subspace=list(range(6)))


@configclass
class IEAnymalDClimbSingleBoxEnvCfg(IEParkourEnvCfg):
    """Configuration for the Anymal D box climbing environment."""

    target_box_height: float = 0.41

    actions = ClimbBoxActionsCfg()
    rewards = ClimbBoxRewardsCfg()
    # rewards = SparseClimbBoxRewardsCfg()
    terminations = ClimbBoxTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch scene
        self.scene.terrain.terrain_generator.sub_terrains = CLIMB_BOX_SUBTERRAINS_SINGLE
        # fix seed for reproducibility
        self.scene.terrain.terrain_generator.seed = 0
        self.scene.terrain.terrain_generator.use_cache = True
        # replace robot
        self.scene.robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # viewer settings
        self.viewer.eye = (5.7, 8.5, 3.5)
        self.viewer.lookat = (0.0, 0.0, -3.2)

        # change box height
        self.scene.terrain.terrain_generator.sub_terrains["big_box_up"].height_range = (
            self.target_box_height,
            self.target_box_height,
        )
        self.task_space.update_box_height(self.target_box_height)

        # Keeps resample window, timeout, trajectory history and sparse `duration` locked together
        set_episode_length(self, 4.0)

        # Removed here so the Baseline, which binds this class directly, matches the curriculum arms
        self.events.base_external_force_torque = None

        # TODO: overwriting the terrain level from the box height has no effect; remove once fixed
        level = round((self.target_box_height - 0.1) * 9.0 / 0.7)
        negative_level = level - 10

        if self.curriculum is not None and hasattr(self.curriculum, "initialization"):
            if (
                hasattr(self.curriculum.initialization, "params")
                and "terrain_level" in self.curriculum.initialization.params
            ):
                self.curriculum.initialization.params["terrain_level"] = negative_level


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_INIT(IEAnymalDClimbSingleBoxEnvCfg):

    # override env level resets needed for initialization curriculum
    events = InitializationEventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.terrain.terrain_generator.curriculum = True

        # remove external forces
        self.events.base_external_force_torque = None


@configclass
class IEAnymalDClimbSingleBoxEnvCfg_GenFeasibleStarts(IEAnymalDClimbSingleBoxEnvCfg_INIT):
    """Generation-only env for ``generate_feasible_starts.py`` with tighter bounds and a sample filter."""

    curriculum = ClimbBoxGenFeasibleStartsCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.use_cache = True

        self.task_space.bounds[0][1] = 0.0
        bh = self.target_box_height

        # Lift the z bound above standing height and stop the keep-out there, else no start lands on the box
        on_box_z = bh + height_offset
        self.task_space.bounds[4][1] = on_box_z + 0.15
        self.task_space.sample_filter = [
            ((-1.0, 0.0), (0, on_box_z)),
            ((-3.5, -1.2), (0.7, self.task_space.bounds[4][1])),
        ]




@configclass
class IEAnymalDClimbSingleBoxEnvCfg_EVAL(IEAnymalDClimbSingleBoxEnvCfg_INIT):

    observations = TrajectoryObservationsCfg()

    curriculum = ClimbBoxEvalCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # enable curriculum to obtain the same terrains as in training
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.terrain_generator.use_cache = True

        # disable policy noise
        self.observations.policy.enable_corruption = False
        # remove external forces
        self.events.base_external_force_torque = None

        # render quality
        render_cfg = sim_utils.RenderCfg(rendering_mode="quality", dlss_mode=3)

        # viewer settings
        self.viewer.eye = (-12.0, -12.0, 11.5)
        self.viewer.lookat = (0.0, 0.0, -1.5)
        self.viewer.resolution = (1920, 1080)
