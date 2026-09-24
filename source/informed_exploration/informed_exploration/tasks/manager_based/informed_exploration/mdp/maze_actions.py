# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term for the point-mass maze ball."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MazeVelocityAction(ActionTerm):
    """Planar velocity control of a free-floating ball.

    The commanded xy velocity is tracked by a proportional force applied every substep, so a zero action
    means stop. A spin damping torque keeps the base frame used by ``CurriculumGoalCommand`` from rotating.
    """

    cfg: MazeVelocityActionCfg
    _asset: RigidObject

    def __init__(self, cfg: MazeVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # (num_envs, num_bodies, 3); the ball is a single-body rigid object
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros_like(self._forces)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Commanded planar velocity in m/s. Shape is (num_envs, 2)."""
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-1.0, 1.0) * self.cfg.max_speed

    def apply_actions(self):
        vel = self._asset.data.root_lin_vel_w
        self._forces[:, 0, :2] = self.cfg.gain * (self._processed_actions - vel[:, :2])
        self._torques[:, 0, :] = -self.cfg.angular_damping * self._asset.data.root_ang_vel_w
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces, torques=self._torques, is_global=True
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class MazeVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`MazeVelocityAction`."""

    class_type: type[ActionTerm] = MazeVelocityAction

    asset_name: str = "robot"

    max_speed: float = 4.0
    """Speed commanded by an action of magnitude one, in m/s."""

    gain: float = 40.0
    """Proportional velocity-tracking gain, in N per m/s. Tracking is monotonic only while ``gain * dt < m``."""

    angular_damping: float = 1.0
    """Spin damping torque, in Nm per rad/s."""
