# Vendored from the isaaclab-parkour extension (MIT).
# Copyright (c) 2025, BVER Team.
#
# Copied here so this package stands alone; behaviour is unchanged from upstream.

from __future__ import annotations
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommand
from isaaclab.envs.mdp.commands.commands_cfg import TerrainBasedPose2dCommandCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TerrainBasedPose2dCommandViz(TerrainBasedPose2dCommand):
    """Command generator that adds a vizualization to the terrain-based command generator."""

    cfg: TerrainBasedPose2dCommandVizCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the valid initial positions from the terrain
        if "init_pos" not in self.terrain.flat_patches and cfg.viz_init_pos:
            raise RuntimeError(
                "The visualization requires a valid flat patch under 'init_pos' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        self.valid_init_pos: torch.Tensor = self.terrain.flat_patches["init_pos"]

    def _set_debug_vis_impl(self, debug_vis: bool):
        if self.cfg.default_viz:
            super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            # create markers if necessary for the first time
            if self.cfg.viz_targets:
                if not hasattr(self, "target_visualizer"):
                    self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # set their visibility to true
                self.target_visualizer.set_visibility(True)
            else:
                if hasattr(self, "target_visualizer"):
                    self.target_visualizer.set_visibility(False)
            if self.cfg.viz_init_pos:
                if not hasattr(self, "init_pos_visualizer"):
                    self.init_pos_visualizer = VisualizationMarkers(self.cfg.init_pos_visualizer_cfg)
                # set their visibility to true
                self.init_pos_visualizer.set_visibility(True)
            else:
                if hasattr(self, "init_pos_visualizer"):
                    self.init_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.cfg.default_viz:
            super()._debug_vis_callback(event)
        if self.cfg.viz_targets:
            translations = self.valid_targets.reshape(-1, 3).clone()
            translations[:, 2] += 0.5
            self.target_visualizer.visualize(translations=translations)
        if self.cfg.viz_init_pos:
            translations = self.valid_init_pos.reshape(-1, 3).clone()
            translations[:, 2] += 0.5
            self.init_pos_visualizer.visualize(translations=translations)


class MetricsWrapper(TerrainBasedPose2dCommandViz):
    """Command generator wrapper that adds metrics to the terrain-based command generator."""

    cfg: MetricsWrapperCfg

    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandVizCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # add metrics for the error between the command and the robot state
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["achieved_goal"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_pos"] = torch.norm(self.pos_command_w - self.robot.data.root_pos_w, dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        self.metrics["achieved_goal"] = (self.metrics["error_pos"] < self.cfg.distance_threshold).float()


@configclass
class TerrainBasedPose2dCommandVizCfg(TerrainBasedPose2dCommandCfg):
    """Configuration for the terrain-based command generator with visualization."""

    class_type = TerrainBasedPose2dCommandViz

    default_viz: bool = True
    """Whether to show the default visualization of the parent class."""

    viz_targets: bool = False
    """Whether to visualize the target positions."""

    viz_init_pos: bool = False
    """Whether to visualize the initial positions."""

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/target",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    """The configuration for the target visualization marker."""

    init_pos_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Event/init_pos",
        markers={
            "cuboid": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    """The configuration for the initial position visualization marker."""


@configclass
class MetricsWrapperCfg(TerrainBasedPose2dCommandVizCfg):
    """Configuration for the terrain-based command generator with metrics."""

    class_type = MetricsWrapper

    distance_threshold: float = MISSING
