# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP for the Franka Tower-of-Hanoi task.

A Franka holds a ring welded to its hand and must move it from seated on one peg to seated on another. Joint
configurations come from the poses file of ``scripts/helpers/hanoi_solve_poses.py``, loaded in ``__post_init__``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

from ... import mdp
from ...curriculum.task_space.hanoi_task_space_cfg import hanoi_task_space_cfg
from ...terrains.hanoi_geometry import (
    PANDA_ARM_JOINT_NAMES,
    PANDA_FINGER_JOINT_EXPR,
    PANDA_JOINT_LIMITS,
    HanoiGeometry,
    hanoi_geometry,
)

HANOI_SIM_DT = 1.0 / 120.0
HANOI_DECIMATION = 4
"""Four physics steps per policy step: control at 30 Hz."""

HANOI_ACTION_SCALE = 0.1
"""Joint displacement per unit action, in radians per control step."""
HANOI_ARM_JOINTS = ["panda_joint.*"]
HANOI_FINGER_JOINTS = [PANDA_FINGER_JOINT_EXPR]

HANOI_GOAL_TOLERANCE = 0.03
"""Radius of the goal ball around the seated ring centre, in metres.

Shared by the reward kernel and the command's success threshold, so "rewarded" and "reached" mean one thing.
"""

HANOI_HOLD_WINDOW_S = 1.0
"""Length of the hold window the reward pays out over, in seconds."""

HANOI_HAND_BODY = "panda_hand"
"""Link the ring is welded to. The ring centre is a fixed offset in its frame, read through the articulation's body
poses (see ``mdp.link_point_pos_w``)."""

HANOI_ROBOT_CFG_NAME = "FRANKA_PANDA_HIGH_PD_CFG"
"""Robot cfg the scene uses and the poses file must have been solved with.

Gravity-free with stiff joints, so a solved or teleported pose holds without sagging onto a peg.
"""

HANOI_POSE_ERROR_TOL = HANOI_GOAL_TOLERANCE / 10.0
"""Largest ring-centre error a solved pose may carry."""

HANOI_PHYSX = {
    "gpu_collision_stack_size": 2**28,
    "gpu_max_rigid_contact_count": 2**23,
    "gpu_max_rigid_patch_count": 2**23,
}
"""Contact buffers sized like Isaac Lab's gear-assembly task, the closest contact-rich upstream scene."""

HANOI_WORKSPACE_ROOT = Path(__file__).resolve().parents[9]
"""Root the repo-relative data paths resolve against when the working directory is elsewhere."""

_P12 = hanoi_geometry("p12")
_BARS = {name: (center, size) for name, center, size in _P12.ring_bar_specs()}
_PEG_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78), roughness=0.5)
_PLATE_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.18), roughness=0.7)
_RING_MATERIAL = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1), roughness=0.4)


def hanoi_poses_relpath(layout: str) -> str:
    """Repo-relative path of a layout's poses file."""
    return f"informed-exploration/data/hanoi/hanoi_{layout}_poses.yaml"


def resolve_workspace_path(path: str) -> str:
    """``path`` if absolute or present under the working directory, else resolved against the workspace root."""
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    return str(HANOI_WORKSPACE_ROOT / path)


HANOI_CAMERA_OFFSET = (1.2, 0.9, 0.7)
"""Viewer eye relative to its lookat: beyond the pegs on the start peg's side, looking back toward the arm."""


def hanoi_camera_pose(geom: HanoiGeometry) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(eye, lookat)`` of the viewer camera, aimed at the plate top on the peg row."""
    lookat = (geom.peg_x, 0.0, geom.plate_top_z)
    eye = tuple(target + offset for target, offset in zip(lookat, HANOI_CAMERA_OFFSET))
    return eye, lookat


def _peg(prim_name: str, index: int) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CylinderCfg(
            radius=_P12.peg_radius,
            height=_P12.peg_height,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_PEG_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_P12.peg_center(index)),
    )


def _ring_bar(name: str) -> AssetBaseCfg:
    center, size = _BARS[name]
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_hand/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size, collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_RING_MATERIAL
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center, rot=_P12.ring_rot_hand),
    )


@configclass
class HanoiSceneCfg(InteractiveSceneCfg):
    """Reach table, base plate, three pegs and a Franka whose fingers hold the ring.

    Field order is load-bearing: the ring bars are spawned as children of the hand link, so they must follow
    ``robot``. They carry collision only, no rigid body, so PhysX treats them as shapes of the hand.
    """

    # declared so code that reads ``scene.terrain`` sees no terrain instead of a missing attribute
    terrain = None

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    base_plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasePlate",
        spawn=sim_utils.CuboidCfg(
            size=_P12.plate_size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=_PLATE_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_P12.plate_center),
    )
    peg_left = _peg("PegLeft", 0)
    peg_middle = _peg("PegMiddle", 1)
    peg_right = _peg("PegRight", 2)

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ring_bar_px = _ring_bar("ring_bar_px")
    ring_bar_nx = _ring_bar("ring_bar_nx")
    ring_bar_py = _ring_bar("ring_bar_py")
    ring_bar_ny = _ring_bar("ring_bar_ny")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


def apply_hanoi_geometry(scene: HanoiSceneCfg, geom: HanoiGeometry) -> None:
    """Size and place the plate, pegs and ring bars for ``geom``."""
    scene.base_plate.spawn.size = geom.plate_size
    scene.base_plate.init_state.pos = geom.plate_center
    for index, peg in enumerate((scene.peg_left, scene.peg_middle, scene.peg_right)):
        peg.spawn.radius = geom.peg_radius
        peg.spawn.height = geom.peg_height
        peg.init_state.pos = geom.peg_center(index)
    for name, center, size in geom.ring_bar_specs():
        bar = getattr(scene, name)
        bar.spawn.size = size
        bar.init_state.pos = center
        bar.init_state.rot = geom.ring_rot_hand


def apply_hanoi_ring_offset(cfg: HanoiEnvCfg, geom: HanoiGeometry) -> None:
    """Point the command, the reward and the ring observation terms at ``geom``'s ring centre on the hand."""
    offset = geom.ring_center_hand
    cfg.commands.base_position.body_offset = offset
    cfg.rewards.hold_window.params["offset"] = offset
    for group in vars(cfg.observations).values():
        if isinstance(group, ObsGroup):
            for term in vars(group).values():
                if isinstance(term, ObsTerm) and "offset" in term.params:
                    term.params["offset"] = offset


@configclass
class HanoiActionsCfg:
    """Relative joint-position targets for the seven arm joints; a reset event holds the fingers on the ring.

    A zero action holds the current pose; absolute targets made home an attractor that lifted a seated ring.
    """

    arm = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=HANOI_ARM_JOINTS, scale=HANOI_ACTION_SCALE, use_zero_offset=True
    )


@configclass
class HanoiCommandsCfg:
    """The goal: the seated ring centre on the goal peg.

    Named ``base_position`` because that name is the interface the curricula and the reward look up. It measures
    the ring centre rather than the robot root, which never moves.
    """

    base_position = mdp.CurriculumGoalCfg(
        asset_name="robot",
        body_name=HANOI_HAND_BODY,
        body_offset=_P12.ring_center_hand,
        # rewritten to the episode length by set_hanoi_episode_length
        resampling_time_range=(8.0, 8.0),
        distance_threshold=HANOI_GOAL_TOLERANCE,
        marker_size=HANOI_GOAL_TOLERANCE,
        debug_vis=True,
    )


@configclass
class HanoiObservationsCfg:
    """Arm joint state, the ring centre, the vector to the goal and the last action, without noise.

    Actor and critic see the same state. The joint terms name the arm joints so the held fingers stay out.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_ARM_JOINTS)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_ARM_JOINTS)}
        )
        ring_pos = ObsTerm(
            func=mdp.LinkPointPosEnv,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[HANOI_HAND_BODY]),
                "offset": _P12.ring_center_hand,
            },
        )
        goal_delta = ObsTerm(
            func=mdp.GoalDeltaLinkPoint,
            params={
                "command_name": "base_position",
                "asset_cfg": SceneEntityCfg("robot", body_names=[HANOI_HAND_BODY]),
                "offset": _P12.ring_center_hand,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg(enable_corruption=False)
    critic: PolicyCfg = PolicyCfg(enable_corruption=False)


@configclass
class HanoiTrajectoryObservationsCfg(HanoiObservationsCfg):
    """Adds the whole-episode ``trajectory`` group the curricula record from.

    Its term order is the layout ``hanoi_obs_to_task`` unpacks; keep the two in sync.
    """

    @configclass
    class TrajectoryObsCfg(ObsGroup):
        # history_length is rewritten to the episode length by set_hanoi_episode_length
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_ARM_JOINTS)},
            history_length=1,
            flatten_history_dim=False,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_ARM_JOINTS)},
            history_length=1,
            flatten_history_dim=False,
        )
        ring_pos = ObsTerm(
            func=mdp.LinkPointPosEnv,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[HANOI_HAND_BODY]),
                "offset": _P12.ring_center_hand,
            },
            history_length=1,
            flatten_history_dim=False,
        )

    trajectory: TrajectoryObsCfg = TrajectoryObsCfg(enable_corruption=False)


@configclass
class HanoiRewardsCfg:
    """The whole reward: a hold window with the ring seated on the goal peg, and nothing else.

    See :class:`mdp.HoldWindowReward`. Measured on the ring centre in three dimensions, since a ring hovering above
    the goal peg is not seated.
    """

    hold_window = RewTerm(
        func=mdp.HoldWindowReward,
        weight=250.0,
        params={
            "command_name": "base_position",
            "duration": HANOI_HOLD_WINDOW_S,
            "tolerance": HANOI_GOAL_TOLERANCE,
            "distance_3d": True,
            "asset_cfg": SceneEntityCfg("robot", body_names=[HANOI_HAND_BODY]),
            "offset": _P12.ring_center_hand,
        },
    )


@configclass
class HanoiTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # a natural terminal, not a timeout: nothing can be earned after the window, so no bootstrap
    hold_window_closed = DoneTerm(func=mdp.HoldWindowClosed, time_out=False, params={"reward_term": "hold_window"})


@configclass
class HanoiWalkTerminationsCfg(HanoiTerminationsCfg):
    """Adds the walk-horizon timeout for arms whose runner drives a random-action tail."""

    brownian_motion_time_out = DoneTerm(func=mdp.brownian_motion_time_out, time_out=True)


@configclass
class HanoiEventCfg:
    """Reset the seven arm joints, as offsets from the home pose, and put the fingers back on the ring.

    ``position`` holds the start offsets written in ``__post_init__``; curricula replace it with per-env tensors.
    """

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_per_env,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_ARM_JOINTS),
            "position": 0.0,
            "velocity": 0.0,
        },
    )
    hold_fingers = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HANOI_FINGER_JOINTS)},
    )


def set_hanoi_episode_length(cfg, episode_time_s: float) -> None:
    """Set the episode timeout, the goal-resample window and the trajectory history together."""
    cfg.commands.base_position.resampling_time_range = (episode_time_s, episode_time_s)
    cfg.episode_length_s = episode_time_s

    group = getattr(cfg.observations, "trajectory", None)
    if group is not None:
        steps = round(episode_time_s / (cfg.decimation * cfg.sim.dt))
        for term in vars(group).values():
            if isinstance(term, ObsTerm):
                term.history_length = steps


@configclass
class HanoiEnvCfg(ManagerBasedRLEnvCfg):
    """Base Hanoi environment; a layout module subclasses this and sets the layout fields below."""

    hanoi_layout: str = "p12"
    episode_s: float = 8.0
    brownian_horizons: tuple[int, int] = (60, 60)
    """Forward and backward walk lengths in policy steps."""
    poses_path: str = hanoi_poses_relpath("p12")
    """Poses file written by ``scripts/helpers/hanoi_solve_poses.py``."""
    start_box_half_width: float = 0.0
    """Half-width of the start distribution on every joint offset, in radians."""

    # filled from the poses file in __post_init__
    home_joint_pos: list[float] | None = None
    start_joint_pos: list[float] | None = None
    anchor_joint_pos: list[float] | None = None
    joint_limits: list[list[float]] | None = None
    ring_xyz_home: list[float] | None = None
    ring_xyz_start: list[float] | None = None
    ring_xyz_anchor: list[float] | None = None

    scene: HanoiSceneCfg = HanoiSceneCfg(num_envs=4096, env_spacing=2.5, clone_in_fabric=False)
    observations: HanoiObservationsCfg = HanoiObservationsCfg()
    actions: HanoiActionsCfg = HanoiActionsCfg()
    commands: HanoiCommandsCfg = HanoiCommandsCfg()
    rewards: HanoiRewardsCfg = HanoiRewardsCfg()
    terminations: HanoiTerminationsCfg = HanoiTerminationsCfg()
    events: HanoiEventCfg = HanoiEventCfg()
    curriculum = None

    task_space = hanoi_task_space_cfg(
        _P12, PANDA_JOINT_LIMITS, [0.0] * 7, [0.0] * 7, [0.0] * 7, _P12.start_ring_xyz, _P12.home_ring_xyz
    )

    def geometry(self) -> HanoiGeometry:
        return hanoi_geometry(self.hanoi_layout)

    def goal_xyz(self) -> list[float]:
        """Env-relative ring centre of the anchor pose: the goal every arm is scored against."""
        if self.ring_xyz_anchor is not None:
            return list(self.ring_xyz_anchor)
        return list(self.geometry().goal_ring_xyz)

    def _task(self, joint_pos: list[float], ring_xyz: list[float]) -> list[list[float]]:
        rows = [[q - home, 0.0] for q, home in zip(joint_pos, self.home_joint_pos)]
        return rows + [[value, 0.0] for value in ring_xyz]

    def start_task(self) -> list[list[float]]:
        """The start configuration as a full task state."""
        return self._task(self.start_joint_pos, self.ring_xyz_start)

    def anchor_task(self) -> list[list[float]]:
        """The anchor configuration, ring seated on the goal peg, as a full task state."""
        return self._task(self.anchor_joint_pos, self.goal_xyz())

    def _load_poses(self) -> None:
        """Fill the joint configurations, joint limits and ring centres from the poses file."""
        path = resolve_workspace_path(self.poses_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Hanoi poses file not found: {path}. Solve the poses first with "
                f"'informed-exploration/scripts/helpers/hanoi_solve_poses.py --layout {self.hanoi_layout}'."
            )
        with open(path) as f:
            data = yaml.safe_load(f)
        if data.get("layout") != self.hanoi_layout:
            raise ValueError(f"{path} was solved for layout '{data.get('layout')}', not '{self.hanoi_layout}'.")
        if data.get("robot_cfg") != HANOI_ROBOT_CFG_NAME:
            raise ValueError(
                f"{path} was solved with '{data.get('robot_cfg')}', but the scene uses {HANOI_ROBOT_CFG_NAME}."
            )
        if list(data.get("joint_names", [])) != list(PANDA_ARM_JOINT_NAMES):
            raise ValueError(f"{path} lists joints {data.get('joint_names')}, expected {list(PANDA_ARM_JOINT_NAMES)}.")
        poses = data["poses"]
        geom = self.geometry()
        targets = {"home": geom.home_ring_xyz, "start": geom.start_ring_xyz, "anchor": geom.goal_ring_xyz}
        solved_quat = data.get("hand_quat_wxyz") or []
        stale = len(solved_quat) != 4 or any(abs(float(a) - b) > 1e-4 for a, b in zip(solved_quat, geom.hand_quat_wxyz))
        for name, target in targets.items():
            stale = stale or any(abs(float(a) - b) > 1e-4 for a, b in zip(poses[name]["ring_xyz_target"], target))
        if stale:
            raise ValueError(
                f"{path} was solved for another geometry of layout '{self.hanoi_layout}': its hand orientation or "
                "ring targets differ. Re-solve it with "
                f"'informed-exploration/scripts/helpers/hanoi_solve_poses.py --layout {self.hanoi_layout}'."
            )
        for name in ("home", "start", "anchor"):
            error = float(poses[name]["error_m"])
            if error > HANOI_POSE_ERROR_TOL:
                raise ValueError(
                    f"{path}: the {name} pose misses its ring target by {error:.4f} m, more than "
                    f"{HANOI_POSE_ERROR_TOL:.4f} m. Re-solve it."
                )
        self.joint_limits = [[float(lo), float(hi)] for lo, hi in data["joint_limits"]]
        self.home_joint_pos = [float(q) for q in poses["home"]["joint_pos"]]
        self.start_joint_pos = [float(q) for q in poses["start"]["joint_pos"]]
        self.anchor_joint_pos = [float(q) for q in poses["anchor"]["joint_pos"]]
        self.ring_xyz_home = [float(v) for v in poses["home"]["ring_xyz_measured"]]
        self.ring_xyz_start = [float(v) for v in poses["start"]["ring_xyz_measured"]]
        self.ring_xyz_anchor = [float(v) for v in poses["anchor"]["ring_xyz_measured"]]

    def __post_init__(self):
        self.decimation = HANOI_DECIMATION
        self.sim.dt = HANOI_SIM_DT
        self.sim.render_interval = self.decimation
        self.is_finite_horizon = True
        for key, value in HANOI_PHYSX.items():
            setattr(self.sim.physx, key, value)

        geom = self.geometry()
        apply_hanoi_geometry(self.scene, geom)
        apply_hanoi_ring_offset(self, geom)
        self._load_poses()

        joint_pos = dict(zip(PANDA_ARM_JOINT_NAMES, self.home_joint_pos))
        joint_pos[PANDA_FINGER_JOINT_EXPR] = geom.finger_joint_pos
        self.scene.robot.init_state.joint_pos = joint_pos

        self.task_space = hanoi_task_space_cfg(
            geom,
            self.joint_limits,
            self.home_joint_pos,
            self.start_joint_pos,
            self.anchor_joint_pos,
            self.ring_xyz_start,
            self.ring_xyz_home,
            self.start_box_half_width,
        )
        offsets = [q - home for q, home in zip(self.start_joint_pos, self.home_joint_pos)]
        width = self.start_box_half_width
        self.events.reset_robot_joints.params["position"] = (
            offsets if width == 0.0 else [(offset - width, offset + width) for offset in offsets]
        )

        set_hanoi_episode_length(self, self.episode_s)

        self.viewer.eye, self.viewer.lookat = hanoi_camera_pose(geom)
