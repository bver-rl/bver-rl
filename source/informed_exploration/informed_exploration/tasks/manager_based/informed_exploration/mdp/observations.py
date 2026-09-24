from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.utils.math import combine_frame_transforms, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pos_command_3d(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    """The generated 3D position command in the robot's base frame (x, y, z; no heading).

    Reads ``command_term.pos_command_b`` directly, so unlike ``mdp.pos_command`` it carries z.
    """
    command_term = env.command_manager.get_term(command_name)
    return command_term.pos_command_b


def root_yaw_sincos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The root yaw as ``(sin, cos)``, the heading partner of an env-relative position observation.

    Env origins are pure translations, so world yaw equals env-frame yaw.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def body_pos_w(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The world-frame position of the bodies configured in SceneEntityCfg."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0]][:, :2]  # only return xy position


def root_pos_xy_env(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Env-relative planar position of the asset's root, for the point-mass maze where z is constant."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w - env.scene.env_origins)[:, :2]


def root_lin_vel_xy_w(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """World-frame planar linear velocity of the asset's root."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :2]


def goal_delta_xy(
    env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Planar vector from the asset's root to the commanded goal, in the world frame.

    World frame rather than base frame, since the ball's actions are world-axis velocities and its orientation
    is meaningless.
    """
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    return (command_term.pos_command_w - asset.data.root_pos_w[:, :3])[:, :2]


def link_point_pos_w(asset: Articulation, body_idx: int, offset: torch.Tensor) -> torch.Tensor:
    """World position of a point with position ``offset`` in the frame of link ``body_idx`` of ``asset``.

    Reads the articulation's body poses rather than a FrameTransformer, which returns stale link poses on the
    reset step after a joint teleport.
    """
    pos_w = asset.data.body_link_pos_w[:, body_idx]
    quat_w = asset.data.body_link_quat_w[:, body_idx]
    return combine_frame_transforms(pos_w, quat_w, offset.expand_as(pos_w))[0]


class LinkPointPosEnv(ManagerTermBase):
    """Env-relative position of a point fixed to ``asset_cfg``'s first body, e.g. an object welded to a hand.

    ``offset`` is the point's position in that body's frame, read once at construction.
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._offset = torch.tensor(cfg.params["offset"], device=env.device, dtype=torch.float32)

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, offset: Sequence[float]) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        return link_point_pos_w(asset, asset_cfg.body_ids[0], self._offset) - env.scene.env_origins


class GoalDeltaLinkPoint(ManagerTermBase):
    """3D vector from a point fixed to ``asset_cfg``'s first body to the commanded goal, in the world frame.

    ``offset`` is the point's position in that body's frame, read once at construction.
    """

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._offset = torch.tensor(cfg.params["offset"], device=env.device, dtype=torch.float32)

    def __call__(
        self, env: "ManagerBasedRLEnv", command_name: str, asset_cfg: SceneEntityCfg, offset: Sequence[float]
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command_term = env.command_manager.get_term(command_name)
        return command_term.pos_command_w - link_point_pos_w(asset, asset_cfg.body_ids[0], self._offset)


# Vendored from the isaaclab-parkour extension (MIT).
# Copyright (c) 2025, BVER Team.
#
# Copied here so this package stands alone; behaviour is unchanged from upstream.

def pos_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated position command relative to the robot."""
    command = env.command_manager.get_command(command_name)
    # select only the x and y components
    pos_command = command[:, :2]
    # compute the relative yaw command
    yaw_command = torch.stack((torch.sin(command[:, 3]), torch.cos(command[:, 3])), dim=1)
    # concatenate the position and yaw commands
    return torch.cat((pos_command, yaw_command), dim=1)
def time_to_target(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The time until the position command is resampled."""
    command_term = env.command_manager.get_term(command_name)
    # return normalized time until the command is resampled
    return command_term.time_left.unsqueeze(1) / command_term.cfg.resampling_time_range[1]
