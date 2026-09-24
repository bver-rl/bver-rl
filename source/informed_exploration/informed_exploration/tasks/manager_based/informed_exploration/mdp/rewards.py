# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommand
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import wrap_to_pi
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def exp_tracking_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    # duration: float | None = None,
    distance_3d: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position-tracking reward averaging an exponential and a Gaussian kernel of width ``sigma``, in (0, 1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: TerrainBasedPose2dCommand = env.command_manager.get_term(command_name)
    target = command_term.pos_command_w[:, :3] if distance_3d else command_term.pos_command_w[:, :2]
    current = asset.data.root_pos_w[:, :3] if distance_3d else asset.data.root_pos_w[:, :2]
    pos_error = torch.norm(target - current, dim=1)

    reward = 0.5* (torch.exp(-pos_error / sigma) + torch.exp(-(pos_error / sigma)**2))
    # if duration is not None:
    #     mask = (command_term.time_left <= duration).float() / duration
    #     reward = reward * mask
    return reward


def goal_closing_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_speed: float = 1.5,
    deadband: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Closing speed toward the commanded goal, clipped to ``[-max_speed, max_speed]`` (m/s).

    Symmetric, so back-and-forth pacing nets zero. Returns zero inside ``deadband``, where the goal direction
    flips rapidly.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: TerrainBasedPose2dCommand = env.command_manager.get_term(command_name)
    to_goal = command_term.pos_command_w[:, :3] - asset.data.root_pos_w[:, :3]
    dist = torch.norm(to_goal, dim=1)
    closing = torch.sum(asset.data.root_lin_vel_w * (to_goal / dist.clamp_min(1e-6).unsqueeze(1)), dim=1)
    return torch.where(dist > deadband, closing.clamp(-max_speed, max_speed), torch.zeros_like(closing))


# Vendored from the isaaclab-parkour extension (MIT).
# Copyright (c) 2025, BVER Team.
#
# Copied here so this package stands alone; behaviour is unchanged from upstream.

def tracking_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    duration: float,
    distance_3d: bool = False,
    sparse: bool = False,
    tolerance: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking the position command during the last `duration` seconds before resampling, scaled by
    1/duration. Distance is planar unless `distance_3d` is set.

    `tolerance` is the sparse kernel's support in metres, where the reward falls linearly to zero.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term: TerrainBasedPose2dCommand = env.command_manager.get_term(command_name)
    # compute the position error
    if distance_3d:
        pos_error = torch.norm(command_term.pos_command_w[:, :3] - asset.data.root_pos_w[:, :3], dim=1)
    else:
        pos_error = torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    # compute which environments receive the reward and scale the reward by the duration
    mask = command_term.time_left <= duration
    mask = mask / duration
    # reward scales with the distance to the target
    if sparse:
        return torch.clip(1.0 - pos_error / tolerance, 0.0) * mask
    return torch.clip(2.0 - 0.5 * pos_error, min=0.0) * mask
def custom_contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces with a constant up to 200, then linearly."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    # compute the penalty
    violation = torch.where(violation > threshold, violation.clip(min=200.0), violation)
    return torch.sum(violation, dim=1) / 200.0
def squared_contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces squared if they surpass the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0].clip(
        max=2.0 * threshold
    )

    # compute the penalty
    return torch.sum(torch.square((violation - threshold).clip(min=0.0)), dim=1)
class body_lin_acc_l2_old_term(ManagerTermBase):
    """Penalize the linear acceleration of bodies using an L2 kernel, computed at the environment frequency."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_lin_vel_w_env = torch.zeros_like(asset.data.body_lin_vel_w)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # compute acceleration using the control frequency
        body_lin_acc_w_env = (asset.data.body_lin_vel_w - self.previous_lin_vel_w_env) / env.step_dt
        # store the current joint velocities for the next step
        self.previous_lin_vel_w_env = asset.data.body_lin_vel_w.clone()
        return torch.sum(torch.norm(body_lin_acc_w_env[:, asset_cfg.body_ids, :], dim=-1), dim=1)
class body_acc_weighted_l2_old_term(ManagerTermBase):
    """Penalize weighted linear and angular body accelerations using an L2 squared kernel, at the environment
    frequency."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_lin_vel_w_env = torch.zeros_like(asset.data.body_lin_vel_w)
        self.previous_ang_vel_w_env = torch.zeros_like(asset.data.body_ang_vel_w)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        linear_weight: float,
        angular_weight: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # compute acceleration using the control frequency
        body_lin_acc_w_env = (asset.data.body_lin_vel_w - self.previous_lin_vel_w_env) / env.step_dt
        body_ang_acc_w_env = (asset.data.body_ang_vel_w - self.previous_ang_vel_w_env) / env.step_dt
        # store the current joint velocities for the next step
        self.previous_lin_vel_w_env = asset.data.body_lin_vel_w.clone()
        self.previous_ang_vel_w_env = asset.data.body_ang_vel_w.clone()
        # add linear and angular accelerations with different weights
        return linear_weight * torch.sum(
            torch.square(body_lin_acc_w_env[:, asset_cfg.body_ids]), dim=(1, 2)
        ) + angular_weight * torch.sum(
            torch.square(body_ang_acc_w_env[:, asset_cfg.body_ids]), dim=(1, 2)
        )  # imp make this two rewards
def torque_limits_old(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum((torch.abs(asset.data.computed_torque[:, asset_cfg.joint_ids]) - limit).clip(min=0.0), dim=1)
