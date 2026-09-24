# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An arrival-triggered hold window reward and the termination that closes it.

The fixed-length window opens at first arrival, so every pair that arrives can earn the full weight
regardless of how far the goal was.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from collections.abc import Sequence
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg, TerminationTermCfg

from ..utils.curriculum_utils import random_tail_boundaries
from .observations import link_point_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HoldWindowReward(ManagerTermBase):
    """Pay ``clip(1 - d / tolerance, 0) / duration`` per step for ``duration`` seconds after first arrival.

    The tracked point is the asset root, or ``offset`` in ``asset_cfg``'s first body frame; distance is planar
    unless ``distance_3d`` is set. The window runs contiguously from arrival, so leaving early forfeits the rest.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.arrived = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.steps_paid = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.window_steps = max(1, round(cfg.params["duration"] / env.step_dt))
        offset = cfg.params.get("offset")
        self._offset = None if offset is None else torch.tensor(offset, device=env.device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.arrived[env_ids] = False
        self.steps_paid[env_ids] = 0

    @property
    def closed(self) -> torch.Tensor:
        """Envs whose window has opened and paid out in full."""
        return self.arrived & (self.steps_paid >= self.window_steps)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        duration: float,
        tolerance: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        distance_3d: bool = False,
        offset: Sequence[float] | None = None,
    ) -> torch.Tensor:
        command_term = env.command_manager.get_term(command_name)
        if self._offset is None:
            asset: RigidObject = env.scene[asset_cfg.name]
            pos_w = asset.data.root_pos_w
        else:
            arm: Articulation = env.scene[asset_cfg.name]
            pos_w = link_point_pos_w(arm, asset_cfg.body_ids[0], self._offset)
        dims = 3 if distance_3d else 2
        dist = torch.linalg.norm(pos_w[:, :dims] - command_term.pos_command_w[:, :dims], dim=1)
        kernel = torch.clip(1.0 - dist / tolerance, 0.0)

        # open the window on first arrival
        self.arrived |= dist < tolerance
        # pay only inside the window; the step that opens it is the window's first step
        active = self.arrived & (self.steps_paid < self.window_steps)
        self.steps_paid[active] += 1
        return torch.where(active, kernel / duration, torch.zeros_like(kernel))


class HoldWindowClosed(ManagerTermBase):
    """Terminate an episode once the hold window of reward term ``reward_term`` has paid out in full.

    A natural terminal, not a timeout, so the return is not bootstrapped. Random-action tail envs are never
    terminated here.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_term: HoldWindowReward | None = None

    def __call__(self, env: ManagerBasedRLEnv, reward_term: str) -> torch.Tensor:
        if self._reward_term is None:
            term = env.reward_manager.get_term_cfg(reward_term).func
            assert isinstance(term, HoldWindowReward), (
                f"HoldWindowClosed needs reward term {reward_term!r} to be a HoldWindowReward, got {type(term).__name__}"
            )
            self._reward_term = term
        tail_start, _, _ = random_tail_boundaries(env)
        not_tail = torch.arange(env.num_envs, device=env.device) < tail_start
        return self._reward_term.closed & not_tail
