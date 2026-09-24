# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP for the point-mass maze.

A velocity-controlled ball in a walled grid, goal-conditioned and sparsely rewarded. Geometry comes from the grid
via ``terrains.maze_geometry``; a layout module sets only a name, an episode length and the walk horizons.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from ... import mdp
from ...curriculum.task_space.maze_task_space_cfg import MAZE_BALL_RADIUS, MAZE_BALL_Z, maze_task_space_cfg
from ...terrains.maze_geometry import maze_geometry
from ...terrains.maze_terrains_cfg import MeshMazeTerrainCfg

MAZE_CELL_SIZE = 4.0
MAZE_WALL_HEIGHT = 3.0
MAZE_BORDER_WIDTH = 10.0
MAZE_SIM_DT = 1.0 / 100.0
MAZE_DECIMATION = 5
"""Five 10 ms physics steps per policy step: control at 20 Hz."""

MAZE_MAX_SPEED = 4.0
"""One cell per second."""

MAZE_GOAL_TOLERANCE = 1.0
"""Radius of the goal ball [m], a quarter of a cell.

Shared by the reward and the command's success threshold, so "rewarded" and "reached" mean one thing.
"""

MAZE_HOLD_WINDOW_S = 2.0
"""Length of the hold window the reward pays out over [s]; it opens at first arrival."""


MAZE_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.SphereCfg(
        radius=MAZE_BALL_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=4.0 * MAZE_MAX_SPEED,
            max_angular_velocity=10.0,
            max_depenetration_velocity=10.0,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        # Frictionless and inelastic, so the ball slides without spin and does not bounce off walls
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
            friction_combine_mode="min",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1), roughness=0.4),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, MAZE_BALL_Z)),
)


@configclass
class MazeSceneCfg(InteractiveSceneCfg):
    """One maze tile shared by every env: ``env_spacing`` is zero and the envs are stacked on it."""

    num_envs: int = 4096
    env_spacing: float = 0.0

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # size / maze_name are per-layout and are written in MazeEnvCfg.__post_init__
        terrain_generator=TerrainGeneratorCfg(
            size=(20.0, 20.0),
            border_width=MAZE_BORDER_WIDTH,
            num_rows=1,
            num_cols=1,
            sub_terrains={
                "maze": MeshMazeTerrainCfg(
                    cell_size=MAZE_CELL_SIZE,
                    wall_height=MAZE_WALL_HEIGHT,
                    maze_name="u_maze",
                )
            },
        ),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.0,
            friction_combine_mode="min",
            restitution_combine_mode="min",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.07), roughness=0.8),
        debug_vis=False,
    )

    robot: RigidObjectCfg = MAZE_BALL_CFG

    # No contact sensor: it would require `clone_in_fabric` off

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class MazeActionsCfg:
    velocity = mdp.MazeVelocityActionCfg(asset_name="robot", max_speed=MAZE_MAX_SPEED)


@configclass
class MazeCommandsCfg:
    """The goal, named ``base_position`` because the reward and curricula look it up by that name."""

    base_position = mdp.CurriculumGoalCfg(
        asset_name="robot",
        # rewritten to the episode length by set_maze_episode_length
        resampling_time_range=(8.0, 8.0),
        distance_threshold=MAZE_GOAL_TOLERANCE,
        debug_vis=True,
    )


@configclass
class MazeObservationsCfg:
    """Position, velocity and the goal vector, without noise; actor and critic see the same state."""

    @configclass
    class PolicyCfg(ObsGroup):
        root_pos_xy = ObsTerm(func=mdp.root_pos_xy_env)
        root_lin_vel_xy = ObsTerm(func=mdp.root_lin_vel_xy_w)
        goal_delta_xy = ObsTerm(func=mdp.goal_delta_xy, params={"command_name": "base_position"})

    policy: PolicyCfg = PolicyCfg(enable_corruption=False)
    critic: PolicyCfg = PolicyCfg(enable_corruption=False)


@configclass
class MazeTrajectoryObservationsCfg(MazeObservationsCfg):
    """Adds the whole-episode ``trajectory`` group the curricula record from.

    Its term order is the layout ``MazeTaskSpace.obs_to_task`` unpacks; keep the two in sync.
    """

    @configclass
    class TrajectoryObsCfg(ObsGroup):
        # history_length is rewritten to the episode length by set_maze_episode_length
        root_pos_w = ObsTerm(func=mdp.root_pos_w, history_length=1, flatten_history_dim=False)
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, history_length=1, flatten_history_dim=False)

    trajectory: TrajectoryObsCfg = TrajectoryObsCfg(enable_corruption=False)


@configclass
class MazeRewardsCfg:
    """The whole reward: a hold window at the goal, and nothing else (see :class:`mdp.HoldWindowReward`)."""

    hold_window = RewTerm(
        func=mdp.HoldWindowReward,
        weight=250.0,
        params={
            "command_name": "base_position",
            "duration": MAZE_HOLD_WINDOW_S,
            "tolerance": MAZE_GOAL_TOLERANCE,
        },
    )


@configclass
class MazeTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # a natural terminal, not a timeout: nothing can be earned after the window, so no bootstrap
    hold_window_closed = DoneTerm(
        func=mdp.HoldWindowClosed, time_out=False, params={"reward_term": "hold_window"}
    )


@configclass
class MazeWalkTerminationsCfg(MazeTerminationsCfg):
    """Adds the walk-horizon timeout for arms whose runner drives a random-action tail.

    Reads the horizons off ``curriculum.initialization``, so arms without one must not bind it.
    """

    brownian_motion_time_out = DoneTerm(func=mdp.brownian_motion_time_out, time_out=True)


@configclass
class MazeEventCfg:
    """Reset the ball's root state, uniformly within the INIT cell unless a curriculum writes per-env tensors."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_per_env,
        mode="reset",
        params={
            # Only keys MazeTaskSpaceCfg.tasks manages; `set_tasks` stacks every value, so extras break resets
            "pose": {"x": 0.0, "y": 0.0, "z": MAZE_BALL_Z},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
    )


def set_maze_episode_length(cfg, episode_time_s: float) -> None:
    """Set the episode timeout, the goal-resample window and the trajectory history together.

    The hold-window reward has no ``sparse`` flag, so its fixed window is left alone.
    """
    cfg.commands.base_position.resampling_time_range = (episode_time_s, episode_time_s)
    cfg.episode_length_s = episode_time_s

    group = getattr(cfg.observations, "trajectory", None)
    if group is not None:
        steps = round(episode_time_s / (cfg.decimation * cfg.sim.dt))
        for term in vars(group).values():
            if isinstance(term, ObsTerm):
                term.history_length = steps

    rewards = getattr(cfg, "rewards", None)
    if rewards is not None:
        for term in vars(rewards).values():
            params = getattr(term, "params", None)
            if isinstance(params, dict) and params.get("sparse") and "duration" in params:
                params["duration"] = episode_time_s


@configclass
class MazeEnvCfg(ManagerBasedRLEnvCfg):
    """Base maze environment; a layout module subclasses this and sets the three fields below."""

    maze_name: str = "u_maze"
    episode_s: float = 8.0
    brownian_horizons: tuple[int, int] = (200, 200)
    """Forward and backward walk lengths in policy steps, capped by the episode length."""

    scene: MazeSceneCfg = MazeSceneCfg(num_envs=4096, env_spacing=0.0, clone_in_fabric=True)
    observations: MazeObservationsCfg = MazeObservationsCfg()
    actions: MazeActionsCfg = MazeActionsCfg()
    commands: MazeCommandsCfg = MazeCommandsCfg()
    rewards: MazeRewardsCfg = MazeRewardsCfg()
    terminations: MazeTerminationsCfg = MazeTerminationsCfg()
    events: MazeEventCfg = MazeEventCfg()
    curriculum = None

    task_space = maze_task_space_cfg(maze_geometry("u_maze", MAZE_CELL_SIZE, MAZE_BALL_RADIUS))

    def goal_xyz(self) -> list[float]:
        """Env-relative xyz of the layout's GOAL cell: the anchor every arm is scored against."""
        gx, gy = maze_geometry(self.maze_name, MAZE_CELL_SIZE, MAZE_BALL_RADIUS).goal_xy
        return [gx, gy, MAZE_BALL_Z]

    def __post_init__(self):
        self.decimation = MAZE_DECIMATION
        self.sim.dt = MAZE_SIM_DT
        self.sim.render_interval = self.decimation
        self.is_finite_horizon = True

        geom = maze_geometry(self.maze_name, MAZE_CELL_SIZE, MAZE_BALL_RADIUS)

        generator = self.scene.terrain.terrain_generator
        generator.size = geom.terrain_size
        generator.sub_terrains["maze"].maze_name = self.maze_name
        generator.sub_terrains["maze"].cell_size = MAZE_CELL_SIZE
        generator.sub_terrains["maze"].wall_height = MAZE_WALL_HEIGHT

        self.task_space = maze_task_space_cfg(geom)

        x_range, y_range = geom.init_region
        self.events.reset_base.params["pose"]["x"] = x_range
        self.events.reset_base.params["pose"]["y"] = y_range

        set_maze_episode_length(self, self.episode_s)

        # top-down, framing the whole maze
        span = max(geom.terrain_size)
        self.viewer.eye = (0.0, 0.0, 1.4 * span)
        self.viewer.lookat = (0.0, 0.0, 0.0)
