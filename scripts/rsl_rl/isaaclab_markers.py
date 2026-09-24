"""In-sim Isaac Lab visualization of static environment markers.

Draws the ``rerun_markers`` start zone, goal and bounds in the viewport, with boxes as cylinder wireframes.
Geometry comes from :mod:`env_marker_geometry`; call :func:`draw_env_markers` once after ``gym.make``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import env_marker_geometry as geom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Radius (m) of the box wireframe edge cylinders, sized for a robot-scale scene.
EDGE_RADIUS = 0.005

# Quaternions (w, x, y, z) rotating the cylinder's local +Z axis onto each world axis.
_SQRT_HALF = float(np.sqrt(0.5))
_QUAT_Z_TO_X = (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0)
_QUAT_Z_TO_Y = (_SQRT_HALF, -_SQRT_HALF, 0.0, 0.0)
_QUAT_Z_TO_Z = (1.0, 0.0, 0.0, 0.0)


class EnvMarkers:
    """Holds the static environment markers and keeps them visible.

    Args:
        markers: The Isaac Lab ``VisualizationMarkers`` instance.
        translations: World-frame instance positions, shape ``(N, 3)``.
        orientations: Per-instance ``(w, x, y, z)`` quaternions, shape ``(N, 4)``.
        scales: Per-instance scales applied to the unit prototypes, shape ``(N, 3)``.
        marker_indices: Prototype index for each instance, shape ``(N,)``.
    """

    def __init__(self, markers, translations, orientations, scales, marker_indices):
        self._markers = markers
        self._translations = translations
        self._orientations = orientations
        self._scales = scales
        self._marker_indices = marker_indices
        self.refresh()

    def refresh(self) -> None:
        """Re-assert the marker instancer, which Fabric can drop on later flushes."""
        self._markers.visualize(
            translations=self._translations,
            orientations=self._orientations,
            scales=self._scales,
            marker_indices=self._marker_indices,
        )

    def set_visibility(self, visible: bool) -> None:
        """Show or hide every marker, re-asserting the instancer when shown."""
        self._markers.set_visibility(visible)
        if visible:
            self.refresh()


def _box_edges(center: np.ndarray, half_sizes: np.ndarray):
    """Return ``(midpoints, orientations, lengths)`` for a box's 12 wireframe edges.

    Orientations are ``(w, x, y, z)`` quaternions aligning a +Z cylinder onto each edge.
    """
    cx, cy, cz = (float(v) for v in center)
    hx, hy, hz = (float(v) for v in half_sizes)
    signs = (-1.0, 1.0)

    midpoints: list[tuple[float, float, float]] = []
    orientations: list[tuple[float, float, float, float]] = []
    lengths: list[float] = []

    # 4 edges parallel to X (at every y/z corner combination), and likewise for Y and Z.
    for sy in signs:
        for sz in signs:
            midpoints.append((cx, cy + sy * hy, cz + sz * hz))
            orientations.append(_QUAT_Z_TO_X)
            lengths.append(2.0 * hx)
    for sx in signs:
        for sz in signs:
            midpoints.append((cx + sx * hx, cy, cz + sz * hz))
            orientations.append(_QUAT_Z_TO_Y)
            lengths.append(2.0 * hy)
    for sx in signs:
        for sy in signs:
            midpoints.append((cx + sx * hx, cy + sy * hy, cz))
            orientations.append(_QUAT_Z_TO_Z)
            lengths.append(2.0 * hz)

    return (
        np.array(midpoints, dtype=np.float32),
        np.array(orientations, dtype=np.float32),
        np.array(lengths, dtype=np.float32),
    )


def draw_env_markers(
    env: "ManagerBasedRLEnv",
    prim_path: str = "/Visuals/EnvMarkers",
    edge_radius: float | None = None,
) -> EnvMarkers | None:
    """Spawn wireframe boxes / a solid goal sphere for the static environment markers.

    Args:
        env: Unwrapped ``ManagerBasedRLEnv`` (call ``env.unwrapped`` before passing).
        prim_path: USD prim path for the marker instancer.
        edge_radius: Radius (m) of the box wireframe edges; None uses :data:`EDGE_RADIUS`.

    Returns:
        An :class:`EnvMarkers` handle, or ``None`` if no markers were resolved.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    specs = geom.compute_env_markers(env)
    # The terrain obstacle ('box') is already visible as real geometry; skip it here.
    specs = [s for s in specs if s.name != "box"]
    if not specs:
        return None

    prototypes: dict = {}
    translations: list = []
    orientations: list = []
    scales: list = []
    marker_indices: list = []

    for proto_index, spec in enumerate(specs):
        r, g, b, _ = spec.color
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(r / 255.0, g / 255.0, b / 255.0))

        if isinstance(spec, geom.PointCloudMarker):
            # One sphere prototype instanced N times instead of one prim per point.
            prototypes[spec.name] = sim_utils.MeshSphereCfg(radius=1.0, visual_material=material)
            for center in spec.centers:
                translations.append(center)
                orientations.append(_QUAT_Z_TO_Z)
                scales.append((spec.radius, spec.radius, spec.radius))
                marker_indices.append(proto_index)
        elif isinstance(spec, geom.SphereMarker):
            prototypes[spec.name] = sim_utils.MeshSphereCfg(radius=1.0, visual_material=material)
            translations.append(spec.center)
            orientations.append(_QUAT_Z_TO_Z)
            scales.append((spec.radius, spec.radius, spec.radius))
            marker_indices.append(proto_index)
        else:  # BoxMarker -> wireframe: 12 thin cylinder edges sharing one prototype.
            prototypes[spec.name] = sim_utils.CylinderCfg(
                radius=edge_radius if edge_radius is not None else EDGE_RADIUS,
                height=1.0,
                visual_material=material,
            )
            mids, eoris, lengths = _box_edges(spec.center, spec.half_sizes)
            for k in range(mids.shape[0]):
                translations.append(mids[k])
                orientations.append(eoris[k])
                # Base cylinder keeps its radius (x/y scale 1); only the length (z) changes.
                scales.append((1.0, 1.0, float(lengths[k])))
                marker_indices.append(proto_index)

    markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers=prototypes))
    markers.set_visibility(True)

    return EnvMarkers(
        markers,
        np.asarray(translations, dtype=np.float32),
        np.asarray(orientations, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
        np.asarray(marker_indices, dtype=np.int32),
    )
