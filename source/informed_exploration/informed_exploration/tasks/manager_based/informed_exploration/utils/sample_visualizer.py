import torch
from typing import Optional
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG


class SampleVisualizer:
    """Visualizer rendering task-space samples in Isaac Sim as point markers."""

    def __init__(
        self,
        max_size: int = 10000,
        prim_path: str = "/Visuals/SamplesBuffer",
        marker_radius: float = 0.025,
        max_sizes: Optional[dict[str, int]] = None,
        global_viz: bool = True,
        global_origin: Optional[torch.Tensor] = None,
    ):
        self.global_viz = global_viz
        self.base_types = ["grey", "blue", "green", "dark_green", "purple", "magenta", "orange", "white"]
        self.types = list(self.base_types)
        if self.global_viz:
            self.types.extend([f"{t}_global" for t in self.base_types])

        self.colors = {
            "grey": (0.1, 0.1, 0.1),
            "blue": (0.0, 0.6, 1.0),
            "green": (0.0, 1.0, 0.4),
            "dark_green": (0.0, 0.431, 0.235),
            "purple": (0.455, 0.224, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "orange": (1.0, 0.5, 0.0),
            "white": (1.0, 1.0, 1.0),
        }
        if self.global_viz:
            self.colors.update({f"{t}_global": self.colors[t] for t in self.base_types})

        if max_sizes is None:
            self.max_sizes = {t: max_size for t in self.types}
        else:
            self.max_sizes = {t: max_sizes.get(t, max_size) for t in self.types}

        self.global_origin = global_origin  # (3,) offset applied to all _global marker types

        self.buffers = {t: torch.zeros((self.max_sizes[t], 3), dtype=torch.float32) for t in self.types}
        self.current_idxs = {t: 0 for t in self.types}
        self.num_samples = {t: 0 for t in self.types}
        self.visualizers = {}

        for t in self.types:
            marker_cfg = SPHERE_MARKER_CFG.replace(prim_path=f"{prim_path}_{t}")
            marker_cfg.markers["sphere"].radius = marker_radius
            marker_cfg.markers["sphere"].visual_material.diffuse_color = self.colors[t]
            self.visualizers[t] = VisualizationMarkers(marker_cfg)

    def refresh(self):
        """Re-assert all marker buffers to the renderer.

        Call once per rendered frame, since Fabric can drop instancer data on later simulation flushes.
        """
        for t in self.types:
            n = self.num_samples[t]
            if n > 0:
                self.visualizers[t].visualize(translations=self.buffers[t][:n].clone())

    def clear_buffer(self, marker_type: str):
        """Clears the buffer for the given marker type and hides its markers."""
        if marker_type not in self.base_types:
            raise ValueError(f"marker_type must be one of {self.base_types}, got {marker_type}")

        for t in [marker_type] + ([f"{marker_type}_global"] if self.global_viz else []):
            self.buffers[t].zero_()
            self.current_idxs[t] = 0
            self.num_samples[t] = 0
            self.visualizers[t].set_visibility(False)

    def add_samples(
        self,
        samples: torch.Tensor,
        xyz_dims: tuple[int, ...],
        env_origins: Optional[torch.Tensor] = None,
        marker_type: str = "blue",
    ):
        """Add new samples to the buffer and update the visualization."""
        if marker_type not in self.base_types:
            raise ValueError(f"marker_type must be one of {self.base_types}, got {marker_type}")

        device = samples.device
        self.buffers[marker_type] = self.buffers[marker_type].to(device)
        if self.global_viz:
            self.buffers[f"{marker_type}_global"] = self.buffers[f"{marker_type}_global"].to(device)

        # Flatten the spatial dimensions of samples before extracting xyz
        if len(samples.shape) > 2:
            # Shape (num_envs, num_components, dim), as for parkour
            samples = samples.view(samples.shape[0], -1)

        # Extract xyz coordinates
        xyz_samples = samples[:, xyz_dims].view(-1, len(xyz_dims))

        # Pad or truncate to 3D xyz
        if xyz_samples.shape[1] < 3:
            pad = torch.zeros((xyz_samples.shape[0], 3 - xyz_samples.shape[1]), device=device)
            xyz_samples = torch.cat([xyz_samples, pad], dim=-1)
        elif xyz_samples.shape[1] > 3:
            xyz_samples = xyz_samples[:, :3]

        global_xyz_samples = xyz_samples.clone()
        if self.global_origin is not None:
            global_xyz_samples = global_xyz_samples + self.global_origin.to(device)

        # Add origins if available
        if env_origins is not None:
            if xyz_samples.shape[0] == env_origins.shape[0]:
                xyz_samples += env_origins
            else:
                # Assume several samples per env
                num_envs = env_origins.shape[0]
                num_samples_per_env = xyz_samples.shape[0] // num_envs
                if xyz_samples.shape[0] % num_envs == 0:
                    xyz_samples += env_origins.repeat_interleave(num_samples_per_env, dim=0)

        update_types = [marker_type]
        update_samples = [xyz_samples]

        if self.global_viz:
            update_types.append(f"{marker_type}_global")
            update_samples.append(global_xyz_samples)

        for m_type, m_samples in zip(update_types, update_samples):
            n_new = m_samples.shape[0]
            max_size = self.max_sizes[m_type]

            if n_new >= max_size:
                self.buffers[m_type][:] = m_samples[-max_size:]
                self.current_idxs[m_type] = 0
                self.num_samples[m_type] = max_size
            else:
                end_idx = self.current_idxs[m_type] + n_new
                if end_idx <= max_size:
                    self.buffers[m_type][self.current_idxs[m_type] : end_idx] = m_samples
                else:
                    overflow = end_idx - max_size
                    self.buffers[m_type][self.current_idxs[m_type] :] = m_samples[: n_new - overflow]
                    self.buffers[m_type][:overflow] = m_samples[n_new - overflow :]

                self.current_idxs[m_type] = (self.current_idxs[m_type] + n_new) % max_size
                self.num_samples[m_type] = min(max_size, self.num_samples[m_type] + n_new)

            self.visualizers[m_type].visualize(translations=self.buffers[m_type][: self.num_samples[m_type]].clone())
