"""Visualizer drawing parent-to-child tree edges as thin cylinders, mirroring :class:`SampleVisualizer`."""

import torch
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def _quat_align_z_to(directions: torch.Tensor) -> torch.Tensor:
    """Return wxyz quaternions that rotate the local +Z axis onto each direction.

    Args:
        directions: (N, 3) unit vectors (zero-length vectors default to identity).

    Returns:
        (N, 4) quaternions in (w, x, y, z) convention.
    """
    N = directions.shape[0]
    device = directions.device

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)

    cross = torch.linalg.cross(z_axis, directions)          # (N, 3)
    dot   = (z_axis * directions).sum(dim=-1, keepdim=True)  # (N, 1)
    cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)

    # Numerically stable half-angle form: w = sqrt((1 + cos) / 2), xyz = cross / (2w).
    w = torch.sqrt((1.0 + dot.clamp(min=-1.0)).clamp(min=0.0) / 2.0)  # (N, 1)
    denom = (2.0 * w).clamp(min=1e-8)
    xyz   = cross / denom  # (N, 3)

    quats = torch.cat([w, xyz], dim=-1)  # (N, 4)  wxyz

    # near-antiparallel directions: rotate half a turn around X
    anti = (cross_norm.squeeze(-1) < 1e-6) & (dot.squeeze(-1) < 0.0)
    if anti.any():
        quats[anti] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    # normalise for safety
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return quats


class ConnectionVisualizer:
    """Render parent-to-child edges as thin cylinders in Isaac Sim.

    Args:
        max_size: maximum number of edges shown simultaneously, as a circular buffer.
        prim_path: USD prim root path prefix.
        cylinder_radius: radius of each cylinder.
        colors: override the default colour map.
    """

    BASE_TYPES = ("grey", "blue", "green")
    DEFAULT_COLORS = {
        "grey":  (0.60, 0.60, 0.62),  # light blue-grey, recedes against green
        "blue":  (0.30, 0.30, 0.85),  # soft blue
        "green": (0.0,  1.0,  0.4),   # vivid green, stands out as the solution path
    }

    def __init__(
        self,
        max_size: int = 5000,
        prim_path: str = "/Visuals/ConnectionsBuffer",
        cylinder_radius: float = 0.008,
        colors: Optional[dict] = None,
    ):
        self._max_size      = max_size
        self._cylinder_r    = cylinder_radius
        self._color_map     = {**self.DEFAULT_COLORS, **(colors or {})}

        # Buffers: midpoints, orientations (wxyz), scales (rx, ry, length)
        self._mid   = {t: torch.zeros(max_size, 3) for t in self.BASE_TYPES}
        self._ori   = {t: self._identity_quat(max_size) for t in self.BASE_TYPES}
        self._scl   = {t: self._default_scale(max_size, cylinder_radius) for t in self.BASE_TYPES}
        self._idx   = {t: 0 for t in self.BASE_TYPES}
        self._n     = {t: 0 for t in self.BASE_TYPES}

        # One marker set per colour type; unit-height cylinders are scaled along Z to the edge length.
        self._viz: dict[str, VisualizationMarkers] = {}
        for t in self.BASE_TYPES:
            cfg = VisualizationMarkersCfg(
                prim_path=f"{prim_path}_{t}",
                markers={
                    "cylinder": sim_utils.CylinderCfg(
                        radius=cylinder_radius,
                        height=1.0,
                        axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=self._color_map[t],
                        ),
                    )
                },
            )
            self._viz[t] = VisualizationMarkers(cfg)

    def add_edges(
        self,
        parent_xyz: torch.Tensor,
        child_xyz:  torch.Tensor,
        origin:     Optional[torch.Tensor] = None,
        marker_type: str = "blue",
    ) -> None:
        """Add edges defined by parent and child positions.

        Args:
            parent_xyz: world-frame or env-local parent positions.
            child_xyz: world-frame or env-local child positions.
            origin: offset applied to both endpoints, e.g. the env origin.
            marker_type: colour key.
        """
        if marker_type not in self.BASE_TYPES:
            raise ValueError(f"marker_type must be one of {self.BASE_TYPES}")

        p = parent_xyz.float()
        c = child_xyz.float()
        if origin is not None:
            o = origin.float().to(p.device)
            p = p + o
            c = c + o

        midpoints  = (p + c) * 0.5                      # (N, 3)
        diff       = c - p                               # (N, 3)
        lengths    = torch.linalg.norm(diff, dim=-1)     # (N,)
        mask_valid = lengths > 1e-4
        if not mask_valid.any():
            return

        p, c, midpoints, diff, lengths = (
            p[mask_valid], c[mask_valid],
            midpoints[mask_valid], diff[mask_valid], lengths[mask_valid],
        )

        directions = diff / lengths.unsqueeze(-1)        # (N, 3) unit
        orientations = _quat_align_z_to(directions)      # (N, 4) wxyz
        # Keep the radius (x, y at one) and scale z by the edge length.
        scales = torch.stack([
            torch.ones_like(lengths),
            torch.ones_like(lengths),
            lengths,
        ], dim=-1)                                       # (N, 3)

        self._write_to_buffer(marker_type, midpoints, orientations, scales)

    def refresh(self) -> None:
        """Re-assert all edge buffers to the renderer.

        Call once per rendered frame, since Fabric can drop instancer data on later simulation flushes.
        """
        for t in self.BASE_TYPES:
            n = self._n[t]
            if n > 0:
                self._viz[t].visualize(
                    translations=self._mid[t][:n].clone(),
                    orientations=self._ori[t][:n].clone(),
                    scales=self._scl[t][:n].clone(),
                )

    def clear(self, marker_type: str) -> None:
        """Clear all edges of the given type and hide the markers."""
        if marker_type not in self.BASE_TYPES:
            raise ValueError(f"marker_type must be one of {self.BASE_TYPES}")
        self._mid[marker_type].zero_()
        self._ori[marker_type] = self._identity_quat(self._max_size)
        self._scl[marker_type] = self._default_scale(self._max_size, self._cylinder_r)
        self._idx[marker_type] = 0
        self._n[marker_type]   = 0
        self._viz[marker_type].set_visibility(False)

    @staticmethod
    def _identity_quat(n: int) -> torch.Tensor:
        q = torch.zeros(n, 4)
        q[:, 0] = 1.0  # w=1
        return q

    @staticmethod
    def _default_scale(n: int, r: float) -> torch.Tensor:
        # identity scale: the base cylinder geometry already carries the desired radius
        return torch.ones(n, 3)

    def _write_to_buffer(
        self,
        t: str,
        midpoints:    torch.Tensor,
        orientations: torch.Tensor,
        scales:       torch.Tensor,
    ) -> None:
        n_new   = midpoints.shape[0]
        dev     = midpoints.device
        max_s   = self._max_size

        for buf, src in (
            (self._mid, midpoints.cpu()),
            (self._ori, orientations.cpu()),
            (self._scl, scales.cpu()),
        ):
            b   = buf[t]
            cur = self._idx[t]
            end = cur + n_new
            if n_new >= max_s:
                b[:] = src[-max_s:]
            elif end <= max_s:
                b[cur:end] = src
            else:
                over = end - max_s
                b[cur:]   = src[:n_new - over]
                b[:over]  = src[n_new - over:]

        self._idx[t] = (self._idx[t] + n_new) % max_s
        self._n[t]   = min(max_s, self._n[t] + n_new)

        n = self._n[t]
        self._viz[t].visualize(
            translations=self._mid[t][:n].clone(),
            orientations=self._ori[t][:n].clone(),
            scales=self._scl[t][:n].clone(),
        )
