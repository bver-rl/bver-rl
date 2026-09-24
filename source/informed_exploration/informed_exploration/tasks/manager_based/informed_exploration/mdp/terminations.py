# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.assets.rigid_object.rigid_object import RigidObject
import torch
from typing import TYPE_CHECKING

import omni.log
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from ..utils.curriculum_utils import random_tail_boundaries

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from ..envs.base_env_cfg import BaseEnv


class StayedAtTarget(ManagerTermBase):

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        distance_threshold: float,
        counter_threshold: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "goal_command",
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]

        if asset_cfg.body_names is not None:
            # use the body specifically configured in the SceneEntityCfg
            position = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
        else:
            position = asset.data.root_link_pos_w

        # TODO: add some type checking for different command types
        command_term = env.command_manager.get_term(command_name)
        distance_to_goal = torch.linalg.norm(position[:, :3] - command_term.pos_command_w[:, :3], dim=1)

        # don't terminate random environments (the tail); tail_start is resolve_env_splits-consistent
        tail_start, _, _ = random_tail_boundaries(env)
        random_env_mask = torch.arange(env.num_envs, device=env.device) < tail_start
        successful = (distance_to_goal <= distance_threshold) & random_env_mask

        self.counter[successful] += 1
        self.counter[~successful] = 0

        return self.counter >= counter_threshold


def root_height_violation(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height or above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < minimum_height) | (asset.data.root_pos_w[:, 2] > maximum_height)


def root_height_below_env_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root drops below ``minimum_height`` measured from its env origin.

    Intended for raised terrains with a walkable floor underneath. Only the root is compared, so leave clearance
    for the lowest legitimate pose on the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < minimum_height


def brownian_motion_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate a random-walk episode when it reaches its Brownian horizon.

    Each tail slice (forward, backward, reachability) has its own horizon, falling back to ``brownian_horizon``
    when the curriculum defines one. Only tail envs are terminated.
    """
    if not hasattr(env.curriculum_manager.cfg, "initialization"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    init_cfg = env.curriculum_manager.cfg.initialization
    base_horizon = getattr(init_cfg, "brownian_horizon", None)
    forward_horizon = getattr(init_cfg, "brownian_horizon_forward", None)
    backward_horizon = getattr(init_cfg, "brownian_horizon_backward", None)
    reachability_horizon = getattr(init_cfg, "brownian_horizon_reachability", None)
    if base_horizon is not None:
        forward_horizon = base_horizon if forward_horizon is None else forward_horizon
        backward_horizon = base_horizon if backward_horizon is None else backward_horizon
        reachability_horizon = base_horizon if reachability_horizon is None else reachability_horizon

    if forward_horizon is None and backward_horizon is None and reachability_horizon is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    tail_start, backward_start, reachability_start = random_tail_boundaries(env)
    env_ids = torch.arange(env.num_envs, device=env.device)
    forward_mask = (env_ids >= tail_start) & (env_ids < backward_start)
    backward_mask = (env_ids >= backward_start) & (env_ids < reachability_start)
    reachability_mask = env_ids >= reachability_start

    # per-env horizon: only slices with a resolved horizon time out
    horizon = torch.zeros(env.num_envs, device=env.device, dtype=env.episode_length_buf.dtype)
    tail_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if forward_horizon is not None:
        horizon[forward_mask] = forward_horizon
        tail_mask |= forward_mask
    if backward_horizon is not None:
        horizon[backward_mask] = backward_horizon
        tail_mask |= backward_mask
    if reachability_horizon is not None:
        horizon[reachability_mask] = reachability_horizon
        tail_mask |= reachability_mask
    return (env.episode_length_buf >= horizon) & tail_mask
