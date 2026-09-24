"""Event functions for :class:`isaaclab.managers.EventTermCfg`, mainly per-env reset terms."""

from __future__ import annotations

import logging
from isaaclab.assets.rigid_object.rigid_object import RigidObject
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with uniform random offsets around the default position and velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def _per_env_joint_values(env: ManagerBasedEnv, value, joint_dim: int) -> torch.Tensor:
    """Build ``(num_envs, joint_dim)`` values from a scalar, a per-joint sequence or a per-env tensor.

    A per-joint sequence holds a scalar or a ``(lo, hi)`` pair drawn uniformly per call; tensors pass through.
    """
    if isinstance(value, (int, float)):
        return torch.ones(env.num_envs, joint_dim, device=env.device) * value
    if isinstance(value, (list, tuple)):
        assert len(value) == joint_dim, f"expected {joint_dim} per-joint values, got {len(value)}"
        out = torch.empty(env.num_envs, joint_dim, device=env.device)
        for j, item in enumerate(value):
            if isinstance(item, (list, tuple)):
                out[:, j].uniform_(float(item[0]), float(item[1]))
            else:
                out[:, j] = float(item)
        return out
    return value if joint_dim > 1 else value.unsqueeze(-1)


def reset_joints_by_offset_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position: float | torch.Tensor,
    velocity: float | torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with per-env offsets around the default position and velocity.

    ``position`` and ``velocity`` accept the forms of :func:`_per_env_joint_values`.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # TODO: check why there was a unsqueeze
    joint_dim = len(asset_cfg.joint_ids)
    position = _per_env_joint_values(env, position, joint_dim)
    velocity = _per_env_joint_values(env, velocity, joint_dim)

    # get default joint state
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # bias these values
    joint_pos += position[env_ids]
    joint_vel += velocity[env_ids]

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

    pass


def reset_root_state_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: dict[str, float | tuple[float, float] | torch.Tensor],
    velocity: dict[str, float | tuple[float, float] | torch.Tensor],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the root state to env-origin-relative per-env values, not offset by the default root position.

    Each entry is a scalar, a per-env tensor indexed by ``env_ids``, or a ``(lo, hi)`` pair drawn per reset.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    full_pose_list = [pose.get(key, (0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_samples = torch.zeros(len(env_ids), 6, device=asset.device)

    for i, item in enumerate(full_pose_list):
        if isinstance(item, torch.Tensor):
            pose_samples[:, i] = item[env_ids]
        elif isinstance(item, (tuple, list)):
            pose_samples[:, i].uniform_(item[0], item[1])
        else:
            pose_samples[:, i] = item

    positions = env.scene.env_origins[env_ids] + pose_samples[:, 0:3]  # +root_states[:, 0:3] # absolute positions
    orientations_delta = math_utils.quat_from_euler_xyz(pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    full_velocity_list = [velocity.get(key, 0.0) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    velocity_samples = torch.zeros(len(env_ids), 6, device=asset.device)

    for i, item in enumerate(full_velocity_list):
        if isinstance(item, torch.Tensor):
            velocity_samples[:, i] = item[env_ids]
        elif isinstance(item, (tuple, list)):
            velocity_samples[:, i].uniform_(item[0], item[1])
        else:
            velocity_samples[:, i] = item

    velocities = root_states[:, 7:13] + velocity_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def hold_joints_at_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the selected joints to their default position at rest and make that position their target.

    Meant for joints no action term drives, whose targets otherwise stay where the articulation initialized them.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    asset.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target(joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
