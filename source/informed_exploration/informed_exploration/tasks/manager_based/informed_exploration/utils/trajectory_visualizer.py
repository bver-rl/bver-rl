import torch
from typing import Sequence
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import SPHERE_MARKER_CFG


class TrajectoryVisualizer:
    """Visualizer rendering trajectory buffers in Isaac Sim as point markers.

    Re-renders only once enough new trajectories have arrived.
    """

    def __init__(
        self,
        buffer_size: int,
        num_markers_per_traj: int = 10,
        update_threshold_ratio: float = 0.5,
        prim_path: str = "/Visuals/TrajectoryBuffer",
        marker_radius: float = 0.01,
    ):
        """Initialize the visualizer.

        Args:
            buffer_size: Total capacity of the trajectory buffer.
            num_markers_per_traj: Maximum number of points to render per trajectory.
            update_threshold_ratio: Fraction of the buffer that must change before re-rendering.
            prim_path: USD prim path for the markers.
            marker_radius: Radius of the point markers.
        """
        self.num_markers_per_traj = num_markers_per_traj
        self.update_threshold = int(buffer_size * update_threshold_ratio)
        self.new_trajectories_count = 0

        # Setup the point marker configuration
        marker_cfg = SPHERE_MARKER_CFG.replace(prim_path=prim_path)
        marker_cfg.markers["sphere"].radius = marker_radius
        marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.455, 0.224, 1)  # purple
        self.cfg = marker_cfg
        self.visualizer = VisualizationMarkers(self.cfg)

    def count_and_update(
        self, trajectories: Sequence[torch.Tensor], num_new: int, env_origins: torch.Tensor | None = None
    ):
        """Increment the new-trajectory counter and re-render once the threshold is met.

        Args:
            trajectories: Sequence of trajectory tensors of shape (seq_len, obs_dim).
            num_new: Number of trajectories added in the current step.
            env_origins: Optional environment origins to offset the trajectories.
        """
        self.new_trajectories_count += num_new

        if self.new_trajectories_count >= self.update_threshold and len(trajectories) > 0:
            self.visualize(trajectories, env_origins)
            self.new_trajectories_count = 0

    def visualize(self, trajectories: Sequence[torch.Tensor], env_origins: torch.Tensor | None = None):
        """Subsample each trajectory and update the markers.

        Args:
            trajectories: Sequence of trajectory tensors of shape (seq_len, obs_dim).
            env_origins: Optional (num_envs, 3) environment origins to offset the trajectories.
        """
        all_sampled_points = []

        for traj in trajectories:
            traj_len = traj.shape[0]
            if traj_len == 0:
                continue

            # Subsample points evenly across the trajectory
            num_samples = min(self.num_markers_per_traj, traj_len)
            indices = torch.linspace(0, traj_len - 1, num_samples, dtype=torch.long)

            # Extract (X, Y, Z) assuming they are the first 3 coordinates
            sampled_positions = traj[indices, 0:3]
            all_sampled_points.append(sampled_positions)

        if all_sampled_points:
            # Concatenate all sampled points into a single flat tensor [N, 3]
            flat_translations = torch.cat(all_sampled_points, dim=0)

            if env_origins is not None:
                # Get unique origins to avoid drawing identical markers for envs on the same subterrain
                unique_origins = torch.unique(env_origins, dim=0).to(flat_translations.device)
                # Add origins: [M_unique, 1, 3] + [1, N, 3] -> [M_unique, N, 3] -> [M_unique * N, 3]
                flat_translations = (unique_origins.unsqueeze(1) + flat_translations.unsqueeze(0)).view(-1, 3)

            self.visualizer.visualize(translations=flat_translations)
