"""Process-global rerun.io logger for exploration visualization and training metrics.

Geometry is stamped on the ``env_step`` timeline; scalar metrics on ``global_step`` and ``iteration``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from typing import Optional, Sequence

# Module-level state

_ENABLED: bool = False
_RR = None  # rerun module reference, imported lazily

# One stream per sink (a rerun stream owns a single sink); holding the refs keeps the gRPC server alive.
_RECORDINGS: list = []


def _get_rr():
    global _RR
    if _RR is None:
        try:
            import rerun as rr

            _RR = rr
        except ImportError as e:
            raise ImportError(
                "rerun-sdk is required for rerun logging. " "Install it with: pip install rerun-sdk"
            ) from e
    return _RR


# Colour constants (uint8, mirroring the Isaac visualizers)

_COLORS = {
    "grey": np.array([140, 140, 140], dtype=np.uint8),
    "light_grey": np.array([180, 180, 180], dtype=np.uint8),
    "dark_grey": np.array([90, 90, 90], dtype=np.uint8),
    "blue": np.array([0, 180, 255], dtype=np.uint8),
    "green": np.array([0, 225, 129], dtype=np.uint8),
    "dark_green": np.array([0, 110, 60], dtype=np.uint8),
    "purple": np.array([116, 57, 255], dtype=np.uint8),
    "magenta": np.array([255, 0, 255], dtype=np.uint8),
    "orange": np.array([255, 128, 0], dtype=np.uint8),
    "yellow": np.array([255, 221, 0], dtype=np.uint8),
    "white": np.array([255, 255, 255], dtype=np.uint8),
    # Goal-side and start-side buffer families, plus one candidate-buffer colour per family.
    "cyan": np.array([0, 230, 230], dtype=np.uint8),
    "red": np.array([230, 40, 40], dtype=np.uint8),
    "teal": np.array([0, 150, 150], dtype=np.uint8),
    "pink": np.array([255, 105, 180], dtype=np.uint8),
    # Anchor goals in gold, outside both direction families so they stand out.
    "anchor": np.array([255, 200, 40], dtype=np.uint8),
}

# Marker colours rendered with the larger point radius; everything else uses the default.
_LARGE_RADIUS_COLORS = ("green", "magenta", "orange", "yellow", "anchor")

_ENTITY_PREFIX = "exploration"

# Rolling buffers of volatile task collections keyed by marker colour; see ``buffer_sample_tasks``.
_SAMPLE_BUFFERS: dict[str, deque] = {}
_DEFAULT_BUFFER_MAXLEN = 20_000


# Public API – lifecycle


def initialize(
    mode: str = "save",
    save_path: Optional[str] = None,
    app_id: str = "informed_exploration",
) -> None:
    """Initialize the rerun recording stream.

    Args:
        mode: ``"save"`` writes a .rrd file; ``"serve"`` additionally hosts a live gRPC stream.
        save_path: Path for the .rrd file (required).
        app_id: Rerun application ID shown in the viewer title bar.
    """
    import os

    global _ENABLED
    rr = _get_rr()

    if save_path is None:
        raise ValueError("save_path must be provided")

    if mode not in ("save", "serve"):
        raise ValueError(f"Unknown rerun mode '{mode}'. Choose save|serve.")

    _RECORDINGS.clear()

    # The file stream is present in both modes.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_rec = rr.RecordingStream(app_id, recording_id="file")
    file_rec.save(save_path)
    _RECORDINGS.append(file_rec)

    if mode == "serve":
        import subprocess
        import time

        # Free the gRPC port in case a previous training run left it occupied.
        subprocess.run(
            ["bash", "-c", 'pid=$(lsof -ti tcp:9876 2>/dev/null); [ -n "$pid" ] && kill -9 $pid 2>/dev/null; true']
        )
        time.sleep(0.5)

        # Separate serving stream; it buffers all history so late-connecting viewers see the full run.
        serve_rec = rr.RecordingStream(app_id, recording_id="serve")
        uri = serve_rec.serve_grpc(cors_allow_origin=["*"])
        _RECORDINGS.append(serve_rec)
        print(f"[INFO] Rerun gRPC server started at {uri}")
        print("[INFO] Launch 'General - Rerun Viewer (Live)' in VS Code, then open http://localhost:9090")

    _ENABLED = True

    # Send a default blueprint that sets the 3-D view background colour to every stream.
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            background=[10, 5, 18],
            line_grid=rrb.LineGrid3D(color=[197, 188, 255, 4]),
        ),
        auto_layout=True,
    )
    for rec in _RECORDINGS:
        rec.send_blueprint(blueprint)


def enabled() -> bool:
    """Return True if rerun logging has been initialized."""
    return _ENABLED


# Internal fan-out helpers, teeing every log and time call to all active streams


def _log(entity: str, archetype, static: bool = False) -> None:
    """Log a prebuilt archetype to every active recording stream."""
    for rec in _RECORDINGS:
        rec.log(entity, archetype, static=static)


def _set_time(timeline: str, sequence: int) -> None:
    """Stamp a timeline value on every active recording stream."""
    for rec in _RECORDINGS:
        rec.set_time(timeline, sequence=sequence)


# Public API – time


def set_time(env_step: Optional[int] = None, iteration: Optional[int] = None) -> None:
    """Stamp the current log position on one or both timelines.

    Args:
        env_step: Value for the ``env_step`` timeline (geometry).
        iteration: Value for the ``iteration`` timeline (metrics/geometry).
    """
    if not _ENABLED:
        return
    if env_step is not None:
        _set_time("env_step", env_step)
    if iteration is not None:
        _set_time("iteration", iteration)


# Public API – geometry


def log_samples(
    world_xyz: np.ndarray,
    marker_type: str = "blue",
    entity: Optional[str] = None,
) -> None:
    """Log a point cloud of task-space samples.

    Args:
        world_xyz: World-frame positions.
        marker_type: Colour key.
        entity: Rerun entity path, defaulting to one derived from the colour.
    """
    if not _ENABLED or world_xyz.shape[0] == 0:
        return
    entity = entity or f"{_ENTITY_PREFIX}/samples/{marker_type}"
    rr = _get_rr()
    color = _COLORS.get(marker_type, _COLORS["blue"])
    colors = np.tile(color, (world_xyz.shape[0], 1))
    radii = 0.004 if marker_type in _LARGE_RADIUS_COLORS else 0.002
    _log(entity, rr.Points3D(positions=world_xyz, colors=colors, radii=radii))


def clear_samples(entity: str) -> None:
    """Empty a sample cloud at the current timeline point by logging a zero-point cloud.

    Needed because :func:`log_samples` skips empty batches, leaving the last cloud visible.
    """
    if not _ENABLED:
        return
    rr = _get_rr()
    _log(entity, rr.Points3D(positions=np.zeros((0, 3), dtype=np.float32)))


def log_trajectories(
    strips: list[np.ndarray],
    marker_type: str = "purple",
    entity: Optional[str] = None,
) -> None:
    """Log a list of trajectory line strips.

    Args:
        strips: One array of positions per trajectory.
        marker_type: Colour key.
        entity: Rerun entity path, defaulting to one derived from the colour.
    """
    if not _ENABLED or len(strips) == 0:
        return
    entity = entity or f"{_ENTITY_PREFIX}/trajectories/{marker_type}"
    rr = _get_rr()
    color = _COLORS.get(marker_type, _COLORS["purple"])
    colors = [color] * len(strips)
    _log(entity, rr.LineStrips3D(strips=strips, colors=colors))


def log_connections(
    parent_xyz: np.ndarray,
    child_xyz: np.ndarray,
    marker_type: str = "blue",
    entity: Optional[str] = None,
    radii: Optional[float] = None,
) -> None:
    """Log parent-to-child edge connections as line segments.

    Args:
        parent_xyz: Parent positions.
        child_xyz: Child positions.
        marker_type: Colour key.
        entity: Rerun entity path, defaulting to one derived from the colour.
        radii: Line radius; ``None`` keeps rerun's default.
    """
    if not _ENABLED or parent_xyz.shape[0] == 0:
        return
    segments = edges_to_segments(parent_xyz, child_xyz)
    if segments.shape[0] == 0:
        return
    entity = entity or f"{_ENTITY_PREFIX}/connections/{marker_type}"
    rr = _get_rr()
    color = _COLORS.get(marker_type, _COLORS["blue"])
    # Each segment is a 2-point strip; broadcast one colour per strip.
    colors = np.tile(color, (segments.shape[0], 1))
    strips = [segments[i] for i in range(segments.shape[0])]
    _log(entity, rr.LineStrips3D(strips=strips, colors=colors, radii=radii))


# High-level callsite helpers, each a no-op when rerun is disabled


def log_sample_tasks(
    tasks: torch.Tensor,
    xyz_dims: tuple[int, ...],
    env_origins: Optional[torch.Tensor] = None,
    env_step: Optional[int] = None,
    marker_type: str = "blue",
    entity: Optional[str] = None,
) -> None:
    """Convert and log a batch of task-space samples, mirroring ``add_samples``.

    Args:
        tasks: Task-space tensor, flat or shaped.
        xyz_dims: Spatial coordinate indices in the flat task vector.
        env_origins: Per-env origin offsets, broadcast like ``add_samples``.
        env_step: Value for the ``env_step`` timeline; skipped if None.
        marker_type: Colour key.
        entity: Optional override for the rerun entity path.
    """
    if not _ENABLED:
        return
    if env_step is not None:
        set_time(env_step=env_step)
    log_samples(
        samples_to_world_xyz(tasks, xyz_dims, env_origins),
        marker_type=marker_type,
        entity=entity,
    )


def log_trajectory_set(
    trajectories: Sequence[torch.Tensor],
    env_origins: Optional[torch.Tensor] = None,
    env_step: Optional[int] = None,
    marker_type: str = "purple",
    entity: Optional[str] = None,
) -> None:
    """Convert and log a set of trajectories as line strips.

    Args:
        trajectories: One tensor per trajectory, whose leading columns are xyz.
        env_origins: Per-env origin offsets; unique origins are broadcast.
        env_step: Value for the ``env_step`` timeline; skipped if None.
        marker_type: Colour key.
        entity: Optional override for the rerun entity path.
    """
    if not _ENABLED:
        return
    if env_step is not None:
        set_time(env_step=env_step)
    log_trajectories(
        trajectories_to_strips(trajectories, env_origins),
        marker_type=marker_type,
        entity=entity,
    )


# Volatile-collection buffering: per-step collections accumulate here between periodic snapshots


def buffer_sample_tasks(
    tasks: torch.Tensor,
    marker_type: str = "magenta",
    maxlen: int = _DEFAULT_BUFFER_MAXLEN,
) -> None:
    """Append volatile task-space samples, as flat (D,) CPU rows, to the rolling buffer for ``marker_type``.

    Args:
        tasks: (N, D) or (N, C, K) task-space tensor.
        marker_type: Colour key the buffer is filed under.
        maxlen: Maximum rows retained, honoured only when the buffer is created.
    """
    if tasks.shape[0] == 0:
        return
    buf = _SAMPLE_BUFFERS.get(marker_type)
    if buf is None:
        buf = deque(maxlen=maxlen)
        _SAMPLE_BUFFERS[marker_type] = buf
    buf.extend(tasks.detach().reshape(tasks.shape[0], -1).cpu())


def buffered_sample_tasks(marker_type: str = "magenta") -> list[torch.Tensor]:
    """Return the buffered volatile samples for ``marker_type`` as flat (D,) rows, or an empty list."""
    buf = _SAMPLE_BUFFERS.get(marker_type)
    return list(buf) if buf is not None else []


# Public API – metrics (called from rsl_rl's wandb writer)


def log_boxes(
    centers: np.ndarray,
    half_sizes: Sequence[float] | np.ndarray,
    colors: Sequence[int],
    entity: str,
    static: bool = False,
) -> None:
    """Log a set of 3D axis-aligned boxes (one per environment).

    Args:
        centers: (N, 3) float32 world-frame box centres.
        half_sizes: [hx, hy, hz] broadcast to all boxes, or an (N, 3) array of per-box half-extents.
        colors: [r, g, b] or [r, g, b, a] uint8, broadcast to all boxes.
        entity: Rerun entity path.
        static: Whether to log as static data visible at every timeline point.
    """
    if not _ENABLED or centers.shape[0] == 0:
        return
    rr = _get_rr()
    n = centers.shape[0]
    hs = np.asarray(half_sizes, dtype=np.float32)
    hs = hs if hs.ndim == 2 else np.tile(hs, (n, 1))
    col = np.tile(np.array(colors, dtype=np.uint8), (n, 1))
    _log(entity, rr.Boxes3D(centers=centers, half_sizes=hs, colors=col), static=static)


def log_ellipsoids(
    centers: np.ndarray,
    half_sizes: Sequence[float],
    colors: Sequence[int],
    entity: str,
    static: bool = False,
) -> None:
    """Log a set of solid ellipsoids; ``FillMode.Solid`` is what honours the alpha channel.

    Args:
        centers: (N, 3) float32 world-frame ellipsoid centres.
        half_sizes: [hx, hy, hz] half-extents, broadcast to all ellipsoids.
        colors: [r, g, b, a] uint8, broadcast to all ellipsoids.
        entity: Rerun entity path.
        static: Whether to log as static data visible at every timeline point.
    """
    if not _ENABLED or centers.shape[0] == 0:
        return
    rr = _get_rr()
    import rerun.components as rrcomp

    n = centers.shape[0]
    hs = np.tile(np.array(half_sizes, dtype=np.float32), (n, 1))
    col = np.tile(np.array(colors, dtype=np.uint8), (n, 1))
    _log(
        entity,
        rr.Ellipsoids3D(centers=centers, half_sizes=hs, colors=col, fill_mode=rrcomp.FillMode.Solid),
        static=static,
    )


def log_meshes(
    meshes,
    entity: str,
    static: bool = True,
    color_by_height: bool = True,
    color: Optional[Sequence[int]] = None,
    flat_shading: bool = False,
) -> None:
    """Log trimesh geometry, such as the sim's terrain, concatenated into one solid ``rr.Mesh3D`` underlay.

    Args:
        meshes: A single ``trimesh.Trimesh`` or a list of them; empty input is a no-op.
        entity: Rerun entity path.
        static: Whether to log as static data visible at every timeline point.
        color_by_height: Whether to apply Turbo height colouring when ``color`` is None.
        color: Optional uniform ``[r, g, b]`` or ``[r, g, b, a]`` (0-255) that overrides ``color_by_height``.
        flat_shading: Whether to unmerge vertices for faceted shading on blocky terrain.
    """
    if not _ENABLED:
        return
    import trimesh

    mesh_list = list(meshes) if isinstance(meshes, (list, tuple)) else [meshes]
    mesh_list = [m for m in mesh_list if m is not None and len(m.vertices) > 0]
    if len(mesh_list) == 0:
        return

    mesh = trimesh.util.concatenate(mesh_list) if len(mesh_list) > 1 else mesh_list[0]

    if flat_shading:
        # Copy first so the caller's mesh is never mutated.
        mesh = mesh.copy()
        mesh.unmerge_vertices()

    vertex_colors = None
    if color is not None:
        col = np.asarray(color, dtype=np.uint8)
        if col.shape[0] == 3:
            col = np.append(col, np.uint8(255))
        vertex_colors = np.tile(col, (len(mesh.vertices), 1))
    else:
        if color_by_height:
            # Idempotent on an already-coloured mesh.
            from isaaclab.terrains.utils import color_meshes_by_height

            mesh = color_meshes_by_height(mesh)
        existing = getattr(mesh.visual, "vertex_colors", None)
        if existing is not None and len(existing) == len(mesh.vertices):
            vertex_colors = np.asarray(existing, dtype=np.uint8)

    rr = _get_rr()
    normals = mesh.vertex_normals if mesh.vertex_normals is not None else None
    _log(
        entity,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
            triangle_indices=np.asarray(mesh.faces, dtype=np.uint32),
            vertex_normals=normals,
            vertex_colors=vertex_colors,
        ),
        static=static,
    )


def log_scalar(name: str, value: float, global_step: Optional[int] = None) -> None:
    """Log a single scalar value (e.g. a training metric).

    Args:
        name: Metric name, appended to the ``metrics/`` entity path.
        value: Scalar float.
        global_step: If provided, stamps the ``global_step`` timeline first.
    """
    if not _ENABLED:
        return
    rr = _get_rr()
    if global_step is not None:
        _set_time("global_step", global_step)
    _log(f"metrics/{name}", rr.Scalars(value))


# Data conversion helpers – torch tensors → numpy for rerun


def samples_to_world_xyz(
    samples: torch.Tensor,
    xyz_dims: tuple[int, ...],
    env_origins: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Convert task-space samples to world-frame (N, 3) float32, matching ``SampleVisualizer.add_samples``.

    Args:
        samples: (N, D) or (N, C, K) task-space tensor.
        xyz_dims: Indices of the spatial coordinates in the flat sample vector.
        env_origins: (N, 3) or (num_envs, 3) environment origin offsets.

    Returns:
        (N, 3) float32 numpy array in world frame.
    """
    if len(samples.shape) > 2:
        samples = samples.view(samples.shape[0], -1)

    xyz = samples[:, list(xyz_dims)].float()

    # Pad / truncate to exactly 3 dimensions
    if xyz.shape[1] < 3:
        pad = torch.zeros((xyz.shape[0], 3 - xyz.shape[1]), device=xyz.device)
        xyz = torch.cat([xyz, pad], dim=-1)
    elif xyz.shape[1] > 3:
        xyz = xyz[:, :3]

    if env_origins is not None:
        env_origins = env_origins.to(xyz.device)
        if xyz.shape[0] == env_origins.shape[0]:
            xyz = xyz + env_origins
        else:
            num_envs = env_origins.shape[0]
            if xyz.shape[0] % num_envs == 0:
                reps = xyz.shape[0] // num_envs
                xyz = xyz + env_origins.repeat_interleave(reps, dim=0)

    return xyz.detach().cpu().numpy().astype(np.float32)


def trajectories_to_strips(
    trajectories: Sequence[torch.Tensor],
    env_origins: Optional[torch.Tensor] = None,
) -> list[np.ndarray]:
    """Convert trajectory tensors to full (L, 3) strips, one per unique env origin.

    Args:
        trajectories: Sequence of (seq_len, obs_dim) tensors; first 3 cols are xyz.
        env_origins: (num_envs, 3) environment origin offsets.

    Returns:
        List of (L, 3) float32 numpy arrays suitable for rr.LineStrips3D.
    """
    if len(trajectories) == 0:
        return []

    strips: list[np.ndarray] = []
    for traj in trajectories:
        if traj.shape[0] == 0:
            continue
        xyz = traj[:, :3].float().detach().cpu()
        if env_origins is not None:
            unique_origins = torch.unique(env_origins.cpu(), dim=0)
            for origin in unique_origins:
                shifted = (xyz + origin).numpy().astype(np.float32)
                strips.append(shifted)
        else:
            strips.append(xyz.numpy().astype(np.float32))

    return strips


def edges_to_segments(
    parent_xyz: np.ndarray,
    child_xyz: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert parent/child position pairs to a segment array, dropping near-zero-length edges.

    Args:
        parent_xyz: (N, 3) parent node world positions.
        child_xyz: (N, 3) child node world positions.
        origin: (3,) offset applied to both endpoints.

    Returns:
        (M, 2, 3) float32 array where M ≤ N (invalid edges removed).
    """
    if isinstance(parent_xyz, torch.Tensor):
        parent_xyz = parent_xyz.detach().cpu().numpy()
    if isinstance(child_xyz, torch.Tensor):
        child_xyz = child_xyz.detach().cpu().numpy()

    parent_xyz = parent_xyz.astype(np.float32)
    child_xyz = child_xyz.astype(np.float32)

    if origin is not None:
        if isinstance(origin, torch.Tensor):
            origin = origin.detach().cpu().numpy()
        origin = origin.astype(np.float32)
        parent_xyz = parent_xyz + origin
        child_xyz = child_xyz + origin

    lengths = np.linalg.norm(child_xyz - parent_xyz, axis=-1)
    mask = lengths > 1e-4
    parent_xyz = parent_xyz[mask]
    child_xyz = child_xyz[mask]

    if parent_xyz.shape[0] == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)

    return np.stack([parent_xyz, child_xyz], axis=1)
