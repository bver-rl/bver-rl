from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import mdp
from ...terrains.parkour_terrain import PARKOUR_TERRAIN_CFG

from ...curriculum.task_space.parkour_task_space_cfg import ParkourTaskSpaceCfg


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # general
    num_envs: int = 2048
    env_spacing: float = 2.5

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.05, 0.05),
            roughness=0.6,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 1.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        drift_range=(0.0, 0.0),
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=300.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=8000.0),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_position = mdp.MetricsWrapperCfg(
        distance_threshold=0.25,
        asset_name="robot",
        resampling_time_range=(7.0, 9.0),
        simple_heading=False,
        debug_vis=False,
        viz_targets=False,
        viz_init_pos=False,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        pos_command = ObsTerm(func=mdp.pos_command, params={"command_name": "base_position"})
        time_to_target = ObsTerm(
            func=mdp.time_to_target, params={"command_name": "base_position"}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            # Small amplitude keeps obstacle geometry such as stair risers distinguishable from noise
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-3.0, 3.0),
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg(enable_corruption=True)
    critic: PolicyCfg = PolicyCfg(enable_corruption=False)


@configclass
class TrajectoryObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class TrajectoryObsCfg(ObsGroup):
        root_pos_w = ObsTerm(
            func=mdp.root_pos_w,
            history_length=200,
            flatten_history_dim=False,
        )
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            history_length=200,
            flatten_history_dim=False,
        )
        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w,
            history_length=200,
            flatten_history_dim=False,
        )
        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w,
            history_length=200,
            flatten_history_dim=False,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=200,
            flatten_history_dim=False,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            history_length=200,
            flatten_history_dim=False,
        )

    trajectory: TrajectoryObsCfg = TrajectoryObsCfg(enable_corruption=False)


@configclass
class XYZObservationsCfg(ObservationsCfg):
    """Policy and critic observations extended with env-relative base xyz and yaw as sin/cos.

    Resolves state aliasing of the otherwise egocentric observations on a fixed, memorizable terrain.
    """

    @configclass
    class XYZPolicyCfg(ObservationsCfg.PolicyCfg):
        # Env-relative despite the name: `mdp.root_pos_w` subtracts `env.scene.env_origins`
        root_pos_env = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        root_yaw_sincos = ObsTerm(func=mdp.root_yaw_sincos, noise=Unoise(n_min=-0.05, n_max=0.05))

    policy: XYZPolicyCfg = XYZPolicyCfg(enable_corruption=True)
    critic: XYZPolicyCfg = XYZPolicyCfg(enable_corruption=False)


@configclass
class XYZTrajectoryObservationsCfg(XYZObservationsCfg):
    """XYZ observations plus the ``trajectory`` group of :class:`TrajectoryObservationsCfg`."""

    trajectory: TrajectoryObservationsCfg.TrajectoryObsCfg = TrajectoryObservationsCfg.TrajectoryObsCfg(
        enable_corruption=False
    )


# Terms of `TrajectoryObservationsCfg.TrajectoryObsCfg`; keep in sync, `set_episode_length` resizes their history
_TRAJECTORY_OBS_TERMS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")


def set_episode_length(cfg, episode_time_s: float) -> None:
    """Set resample window, timeout, trajectory history and sparse ``tracking_pos`` duration together.

    Args:
        cfg: Env cfg to modify in place.
        episode_time_s: Episode length [s].
    """
    cfg.commands.base_position.resampling_time_range = (episode_time_s, episode_time_s)
    cfg.episode_length_s = episode_time_s

    group = getattr(cfg.observations, "trajectory", None)
    if group is not None:
        steps = round(episode_time_s / (cfg.decimation * cfg.sim.dt))
        for term in _TRAJECTORY_OBS_TERMS:
            getattr(group, term).history_length = steps

    # Pin sparse durations to the episode so r_min/r_max read as fraction of episode on goal
    rewards = getattr(cfg, "rewards", None)
    if rewards is not None:
        for term in vars(rewards).values():
            params = getattr(term, "params", None)
            if isinstance(params, dict) and params.get("sparse") and "duration" in params:
                params["duration"] = episode_time_s


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_resample = DoneTerm(
        func=mdp.command_resample,
        time_out=True,
        params={
            "command_name": "base_position",
            "num_resamples": 1,
        },
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    illegal_force = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 5000.0},
    )
    illegal_force_feet = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "threshold": 1500.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.5),
            "dynamic_friction_range": (1.0, 1.5),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-25.0, 25.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class InitializationEventCfg(EventCfg):
    """Configuration for initialization events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_per_env,
        mode="reset",
        params={
            "pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 1.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            "velocity": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_per_env,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*"],
            ),
            "position": 0.0,
            "velocity": 0.0,
        },
    )


@configclass
class IEParkourEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the parkour environment."""

    is_finite_horizon = True

    scene: BaseSceneCfg = BaseSceneCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()

    # task space
    task_space: ParkourTaskSpaceCfg = ParkourTaskSpaceCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = self.commands.base_position.resampling_time_range[1]
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # Arms that want terrain difficulty rows re-enable this in their own __post_init__
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
