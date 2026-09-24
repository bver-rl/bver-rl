from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ..observations import link_point_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .curriculum_goal_cfg import CurriculumGoalCfg


marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/curriculum_goal",
    markers={
        "cube": sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.15),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.3)),
        )
    },
)


class CurriculumGoalCommand(CommandTerm):
    """3D position command whose value is written by the curriculum via :meth:`set_goal`, not resampled.

    ``_resample_command`` does not overwrite curriculum goals, which are set before the command reset.
    No heading is modeled.
    """

    cfg: CurriculumGoalCfg

    def __init__(self, cfg: CurriculumGoalCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        # optional tracked point: a fixed offset on one of the asset's links instead of the root
        self._body_idx: int | None = None
        self._body_offset: torch.Tensor | None = None
        if cfg.body_name is not None:
            self._body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
            self._body_offset = torch.tensor(cfg.body_offset, device=self.device, dtype=torch.float32)

        # commands: world-frame and base-frame (robot-relative) target position
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)

        # optional fixed default goal, written on reset when no curriculum drives set_goal (eval/play)
        self._default_goal_e: torch.Tensor | None = None
        if cfg.default_goal is not None:
            self._default_goal_e = torch.tensor(cfg.default_goal, device=self.device, dtype=torch.float32)

        # metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["achieved_goal"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "CurriculumGoalCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 3D position in the base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    """
    Operations.
    """

    def set_goal(self, env_ids: Sequence[int], goal_pos_e: torch.Tensor) -> None:
        """Set the commanded goal position for ``env_ids``.

        Args:
            env_ids: The environments to set the goal for.
            goal_pos_e: Env-origin-relative xyz positions, shape ``(len(env_ids), 3)``.
        """
        self.pos_command_w[env_ids] = goal_pos_e.to(self.device) + self._env.scene.env_origins[env_ids]

    """
    Implementation specific functions.
    """

    def _tracked_pos_w(self) -> torch.Tensor:
        """World position the goal is measured against: the link point when one is configured, else the root."""
        if self._body_idx is None:
            return self.robot.data.root_pos_w[:, :3]
        return link_point_pos_w(self.robot, self._body_idx, self._body_offset)

    def _update_metrics(self):
        tracked = self._tracked_pos_w()
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - tracked, dim=1)
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - tracked[:, :2], dim=1)
        self.metrics["achieved_goal"] = (self.metrics["error_pos"] < self.cfg.distance_threshold).float()

    def _resample_command(self, env_ids: Sequence[int]):
        # No resampling: the curriculum writes goals; only a configured default goal (eval/play) is written here.
        if self._default_goal_e is not None:
            self.pos_command_w[env_ids] = self._default_goal_e + self._env.scene.env_origins[env_ids]

    def _update_command(self):
        # re-target the world-frame goal to the robot's base frame (heading-only rotation)
        target_vec = self.pos_command_w - self._tracked_pos_w()
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_marker"):
                cfg = marker_cfg.copy()
                if self.cfg.marker_size is not None:
                    size = self.cfg.marker_size
                    cfg.markers["cube"].size = (size, size, size)
                self.goal_marker = VisualizationMarkers(cfg)
            self.goal_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_marker"):
                self.goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_marker.visualize(self.pos_command_w)
