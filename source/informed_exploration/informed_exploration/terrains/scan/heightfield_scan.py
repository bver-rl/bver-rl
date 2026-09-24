# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain built from a real-world surface scan, rasterized to a height field by ``prepare_scan.py``.

The tile is a floor plane, the scan sheet centred on it and an arena fence. Poses are measured in the OBJ frame,
the grid is rotated by ``rot90_k`` quarter turns so start-to-goal points along +x, and the terrain origin is the
start pose, so env-relative z equals tile-local z.
"""

from __future__ import annotations

import functools
import hashlib
import math
import warnings
from dataclasses import MISSING, dataclass
from pathlib import Path

import numpy as np
import trimesh

from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.utils import configclass

from . import fill
from ..bridge import geometry
from ..bridge.composer import perimeter_wall_meshes_for_rect
from ..bridge.composer_cfg import PerimeterWallCfg

__all__ = [
    "ScanHeightfieldTerrainCfg",
    "content_fingerprint",
    "load_scan_grid",
    "pinned_scan_cfg",
    "resolve_scan_path",
    "scan_heightfield_terrain",
]

SCAN_DIR = Path(__file__).resolve().parents[1] / "mesh_scans"
"""Directory bare scan file names resolve against."""

_PROBE_STATS = ("median", "p90", "max")


def scan_path(path: str) -> Path:
    """Return where a scan artifact would live, resolving bare names against :data:`SCAN_DIR`."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = SCAN_DIR / candidate
    return candidate


def resolve_scan_path(path: str) -> Path:
    """Like :func:`scan_path`, but raise if the artifact has not been prepared yet."""
    candidate = scan_path(path)
    if not candidate.exists():
        raise FileNotFoundError(
            f"scan height field '{path}' not found at '{candidate}'. Run"
            " scripts/terrains/prepare_scan.py on the raw scan first."
        )
    return candidate


def content_fingerprint(path: Path) -> str:
    """Return a short content hash of the height field, so the terrain cache key tracks the scan's contents."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]


@dataclass(frozen=True)
class ScanGrid:
    """The prepared height field as stored on disk. Heights are NaN outside ``valid``."""

    heights: np.ndarray
    valid: np.ndarray
    resolution: float
    x0: float
    y0: float


@functools.lru_cache(maxsize=8)
def load_scan_grid(path: str, fingerprint: str) -> ScanGrid:
    """Load a prepared height field; ``fingerprint`` only keys the cache so a re-prepared scan is re-read."""
    del fingerprint  # cache key only
    with np.load(path) as data:
        return ScanGrid(
            heights=np.asarray(data["heights"], dtype=np.float32),
            valid=np.asarray(data["valid"], dtype=bool),
            resolution=float(data["resolution"]),
            x0=float(data["x0"]),
            y0=float(data["y0"]),
        )


def _rotate_fractional_index(
    fi: float, fj: float, shape: tuple[int, int]
) -> tuple[float, float, tuple[int, int]]:
    """Return where one ``np.rot90`` step sends a fractional grid index, and the resulting shape."""
    n0, n1 = shape
    return (n1 - 1 - fj), fi, (n1, n0)


_FILL_METHODS = {"harmonic": fill.fill_harmonic, "nearest": fill.fill_nearest}
"""How the un-scanned parts of the bounding box are closed (see :mod:`.fill`)."""


@dataclass(frozen=True)
class PreparedGrid:
    """The height field as it sits in the tile: cropped, hole-free, rotated and lifted."""

    heights: np.ndarray
    resolution: float
    """Cell size in the tile, i.e. the scan's cell size times :attr:`xy_scale`."""
    obj_resolution: float
    """Cell size in the scan's own frame, used only to turn a measured pose into a grid index."""
    origin_xy: tuple[float, float]
    """Tile xy of the rotated grid's cell ``(0, 0)``."""
    obj_x0: float
    obj_y0: float
    """OBJ xy of the CROPPED grid's cell ``(0, 0)``, i.e. before rotation."""
    crop_shape: tuple[int, int]
    rot90_k: int

    @property
    def footprint_rect(self) -> tuple[float, float, float, float]:
        """Tile-local ``(x_lo, x_hi, y_lo, y_hi)`` spanned by the surface's cell centres."""
        nx, ny = self.heights.shape
        return (
            self.origin_xy[0],
            self.origin_xy[0] + (nx - 1) * self.resolution,
            self.origin_xy[1],
            self.origin_xy[1] + (ny - 1) * self.resolution,
        )

    def obj_to_tile(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Map a point from the OBJ frame into tile-local xy."""
        fi = (xy[0] - self.obj_x0) / self.obj_resolution
        fj = (xy[1] - self.obj_y0) / self.obj_resolution
        shape = self.crop_shape
        for _ in range(self.rot90_k):
            fi, fj, shape = _rotate_fractional_index(fi, fj, shape)
        return self.origin_xy[0] + fi * self.resolution, self.origin_xy[1] + fj * self.resolution

    def probe(self, xy_tile: tuple[float, float], radius: float, stat: str) -> float:
        """Return the surface height over a footprint disc around a tile-local point.

        ``median`` suits a goal; ``p90`` and ``max`` bias upward so a spawn clears the local rock.
        """
        assert stat in _PROBE_STATS, f"unknown probe stat '{stat}', expected one of {_PROBE_STATS}"
        nx, ny = self.heights.shape
        ci = (xy_tile[0] - self.origin_xy[0]) / self.resolution
        cj = (xy_tile[1] - self.origin_xy[1]) / self.resolution
        r_cells = max(radius / self.resolution, 1.0)
        i_lo, i_hi = max(0, int(np.floor(ci - r_cells))), min(nx - 1, int(np.ceil(ci + r_cells)))
        j_lo, j_hi = max(0, int(np.floor(cj - r_cells))), min(ny - 1, int(np.ceil(cj + r_cells)))
        ii, jj = np.meshgrid(np.arange(i_lo, i_hi + 1), np.arange(j_lo, j_hi + 1), indexing="ij")
        inside = (ii - ci) ** 2 + (jj - cj) ** 2 <= r_cells**2
        values = self.heights[i_lo : i_hi + 1, j_lo : j_hi + 1][inside]
        if values.size == 0:
            values = self.heights[i_lo : i_hi + 1, j_lo : j_hi + 1].ravel()
        if stat == "median":
            return float(np.median(values))
        if stat == "p90":
            return float(np.percentile(values, 90.0))
        return float(np.max(values))


    def height_at(self, xy: np.ndarray) -> np.ndarray:
        """Return the nearest-cell surface height at tile-frame points, over any leading shape."""
        nx, ny = self.heights.shape
        i = np.clip(np.round((xy[..., 0] - self.origin_xy[0]) / self.resolution).astype(int), 0, nx - 1)
        j = np.clip(np.round((xy[..., 1] - self.origin_xy[1]) / self.resolution).astype(int), 0, ny - 1)
        return self.heights[i, j]

    def ring_spreads(self, centre, window: float, radius: float, step: float | None = None):
        """Return the height spread over the flat-patch ring for every candidate centre in a square window.

        Mirrors :func:`isaaclab.terrains.utils.find_flat_patches` so a patch window can be judged up front.
        """
        step = step or self.resolution
        angle = np.linspace(0.0, 2.0 * np.pi, 10)
        ring = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * radius
        axis_x = np.arange(centre[0] - window, centre[0] + window + 1e-9, step)
        axis_y = np.arange(centre[1] - window, centre[1] + window + 1e-9, step)
        centres = np.stack(np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1).reshape(-1, 2)
        heights = self.height_at(centres[:, None, :] + ring[None])
        return heights.max(axis=1) - heights.min(axis=1)

    def window_z_range(self, centre, window: float) -> tuple[float, float]:
        """Return the surface height extent over a square window, for the patch sampler's z filter."""
        axis_x = np.arange(centre[0] - window, centre[0] + window + 1e-9, self.resolution)
        axis_y = np.arange(centre[1] - window, centre[1] + window + 1e-9, self.resolution)
        grid = np.stack(np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1).reshape(-1, 2)
        heights = self.height_at(grid)
        return float(heights.min()), float(heights.max())


def _prepare_grid(cfg: ScanHeightfieldTerrainCfg, rot90_k: int) -> PreparedGrid:
    """Crop, fill, rotate and lift the stored grid, and place it centred in the tile."""
    grid = load_scan_grid(str(resolve_scan_path(cfg.npz_path)), cfg.fingerprint or "")

    heights, valid = grid.heights, grid.valid
    i_lo = j_lo = 0
    if cfg.footprint_rect_obj is not None:
        x_lo, x_hi, y_lo, y_hi = cfg.footprint_rect_obj
        i_lo = max(0, int(math.ceil((x_lo - grid.x0) / grid.resolution)))
        i_hi = min(heights.shape[0] - 1, int(math.floor((x_hi - grid.x0) / grid.resolution)))
        j_lo = max(0, int(math.ceil((y_lo - grid.y0) / grid.resolution)))
        j_hi = min(heights.shape[1] - 1, int(math.floor((y_hi - grid.y0) / grid.resolution)))
        assert i_hi > i_lo and j_hi > j_lo, f"footprint_rect_obj {cfg.footprint_rect_obj} selects no cells"
        heights = heights[i_lo : i_hi + 1, j_lo : j_hi + 1]
        valid = valid[i_lo : i_hi + 1, j_lo : j_hi + 1]

    assert cfg.fill_method in _FILL_METHODS, (
        f"unknown fill_method {cfg.fill_method!r}, expected one of {sorted(_FILL_METHODS)}"
    )
    # scale before the lift: the lift is clearance in the tile, not part of the measured surface
    heights = _FILL_METHODS[cfg.fill_method](heights, valid) * cfg.z_scale + cfg.surface_lift
    rotated = np.rot90(heights, rot90_k)
    spacing = grid.resolution * cfg.xy_scale

    nx, ny = rotated.shape
    ring = cfg.apron_ring_cells
    span_x = (nx - 1 + 2 * ring) * spacing
    span_y = (ny - 1 + 2 * ring) * spacing
    assert span_x < cfg.size[0] and span_y < cfg.size[1], (
        f"scan sheet ({span_x:.2f} x {span_y:.2f} m incl. apron) does not fit the tile {cfg.size}"
    )
    origin_xy = (
        (cfg.size[0] - span_x) / 2.0 + ring * spacing,
        (cfg.size[1] - span_y) / 2.0 + ring * spacing,
    )

    return PreparedGrid(
        heights=np.ascontiguousarray(rotated, dtype=np.float32),
        resolution=spacing,
        obj_resolution=grid.resolution,
        origin_xy=origin_xy,
        obj_x0=grid.x0 + i_lo * grid.resolution,
        obj_y0=grid.y0 + j_lo * grid.resolution,
        crop_shape=(heights.shape[0], heights.shape[1]),
        rot90_k=rot90_k,
    )


def _auto_rot90_k(bearing: float) -> int:
    """Return the quarter turns that bring a start-to-goal bearing closest to +x."""
    return int(round(-bearing / (math.pi / 2.0))) % 4


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def scan_heightfield_terrain(
    difficulty: float, cfg: ScanHeightfieldTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build the tile: floor plane, scan sheet and arena fence. ``difficulty`` is ignored."""
    del difficulty

    prepared = _prepare_grid(cfg, cfg.resolved_rot90_k)
    meshes: list[trimesh.Trimesh] = [
        geometry.plane(cfg.size, 0.0, center_xy=(cfg.size[0] / 2.0, cfg.size[1] / 2.0))
    ]

    # sink the apron ring just below the floor so the sheet crosses it instead of z-fighting
    padded = np.pad(
        prepared.heights, cfg.apron_ring_cells, mode="constant", constant_values=-cfg.apron_drop
    )
    vertices, triangles = convert_height_field_to_mesh(
        padded, horizontal_scale=prepared.resolution, vertical_scale=1.0, slope_threshold=cfg.slope_threshold
    )
    sheet = trimesh.Trimesh(vertices=vertices, faces=triangles)
    sheet.apply_translation((
        prepared.origin_xy[0] - cfg.apron_ring_cells * prepared.resolution,
        prepared.origin_xy[1] - cfg.apron_ring_cells * prepared.resolution,
        0.0,
    ))
    meshes.append(sheet)

    if cfg.perimeter_wall is not None:
        meshes += perimeter_wall_meshes_for_rect(cfg.perimeter_wall, prepared.footprint_rect, cfg.size)

    start_tile = prepared.obj_to_tile(cfg.start_xy_obj)
    origin = np.array([start_tile[0], start_tile[1], 0.0])
    return meshes, origin


@configclass
class ScanHeightfieldTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a :func:`scan_heightfield_terrain` sub-terrain.

    Publishes the same derived metadata as a bridge sub-terrain, so the bridge env base works unchanged.

    .. important::
        :attr:`size` is read at construction time, so pass it explicitly to match ``TerrainGeneratorCfg.size``.
    """

    function = scan_heightfield_terrain

    npz_path: str = MISSING
    """Prepared height field, absolute or a bare name under ``terrains/mesh_scans/``."""

    fingerprint: str | None = None
    """Derived content hash of the height field, so editing the scan in place invalidates the terrain cache."""

    rot90_k: int | None = None
    """Quarter turns applied to the scan. ``None`` picks the turn that puts the goal roughly along +x."""

    start_xy_obj: tuple[float, float] = MISSING
    """Start pose in the OBJ frame; also the terrain origin, so it is env-relative zero."""

    start_yaw_obj: float = 0.0
    """Start heading in the OBJ frame; rotated along with the scan into :attr:`heading`."""

    goal_xy_obj: tuple[float, float] = MISSING
    """Goal pose in the OBJ frame."""

    extra_starts_obj: tuple[tuple[float, float], ...] = ()
    extra_goals_obj: tuple[tuple[float, float], ...] = ()
    """Further pairs-eval poses in the OBJ frame. Only published, so they cannot affect training."""

    footprint_rect_obj: tuple[float, float, float, float] | None = None
    """OBJ-frame ``(x_lo, x_hi, y_lo, y_hi)`` crop. ``None`` keeps the whole grid, filled corners included."""

    fill_method: str = "harmonic"
    """Fill for un-scanned cells, ``"harmonic"`` or ``"nearest"`` (see :mod:`.fill`); the latter only
    reproduces older meshes."""

    xy_scale: float = 1.0
    """Horizontal scale applied to the scan; alone it flattens every slope."""

    z_scale: float = 1.0
    """Vertical scale applied to the scan; equal to :attr:`xy_scale` it preserves every slope angle."""

    surface_lift: float = 0.1
    """Height (m) the surface is raised above the tile floor, keeping the lowest hollow above the
    start-pool feasibility bound on base height."""

    apron_ring_cells: int = 2
    """Width, in cells, of the flat ring joining the scan rim to the tile floor, outside the fence."""

    apron_drop: float = 0.02
    """Depth (m) the apron ring sinks below the tile floor, so sheet and floor are not coplanar."""

    slope_threshold: float | None = None
    """Passed to the height-field converter; ``None`` keeps the raster's own slopes."""

    start_probe_radius: float = 0.25
    start_probe_stat: str = "p90"
    """Footprint disc for the start surface height (see :meth:`PreparedGrid.probe`), biased upward for spawning."""

    goal_probe_radius: float = 0.3
    goal_probe_stat: str = "median"

    goal_z_offset: float = 0.0
    """Shift (m) of the goal's commanded height, applied to the published goal.

    Negative suits rough ground, where the base settles below full standing height. Flat patches stay on the
    true surface."""
    """Footprint disc for the goal surface height, where the base comes to rest."""

    patch_num_samples: int = 8
    """Flat patches sampled per pose."""

    patch_radius: float = 0.3
    patch_max_height_diff: float = 0.35
    """Maximum height spread (m) of a flat patch, independent of :attr:`z_scale`.

    Deliberately loose: no arm samples these patches, so it only lets terrain generation construct.
    """

    patch_window: float = 0.3
    """Half-width (m) of the square patches are drawn from around each pose; wider lowers the pass rate."""

    patch_z_margin: float = 0.05
    """Slack added to the surface's own height range when filtering patches by height."""

    z_percentiles: tuple[float, float] = (0.5, 99.5)
    """Height percentiles reported as ``z_lo`` / ``z_hi``, so a single spike does not stretch the z band."""

    fence_clearance: float = 0.8
    """Minimum height (m) of the fence above the highest rock; the wall is raised to meet it."""

    perimeter_wall: PerimeterWallCfg | None = PerimeterWallCfg(height=2.0, thickness=0.5, margin=(-0.05, -0.05))
    """Mesh-only arena fence, tucked just inside the rim so the edge cliff is unreachable."""

    # derived in __post_init__, never set by hand

    goal_poses: dict[str, tuple[float, float, float]] | None = None
    """Start and goal positions, tile-local, at surface height."""

    extra_poses: dict[str, tuple[float, float, float]] | None = None
    """Pairs-eval poses, tile-local at surface height, keyed ``start_1..`` / ``goal_1..``.

    Kept out of :attr:`goal_poses`, where every non-start entry would become a training anchor."""

    geometry_bounds: dict[str, float] | None = None
    """Walkable AABB in the bridge terrains' convention: xy env-relative, z tile-local."""

    keep_in_regions: list[tuple[float, float, float, float]] | None = None
    """The scan footprint, env-relative. A solid surface, so this is one rect."""

    footprint_rect_tile: tuple[float, float, float, float] | None = None
    surface_top_z: float | None = None
    """Env-local height of the highest walkable surface, the ``--surface_top_z`` for the feasible-start pool."""

    heading: float | None = None
    """Start heading after rotation."""

    bearing: float | None = None
    """Start-to-goal bearing after rotation."""

    resolved_rot90_k: int | None = None

    def _pose_patch(self, prepared: PreparedGrid, xy_tile, name: str) -> FlatPatchSamplingCfg:
        """Build a flat-patch window around one pose, with its z filter taken from the local surface."""
        start_tile = prepared.obj_to_tile(self.start_xy_obj)
        z_lo, z_hi = prepared.window_z_range(xy_tile, self.patch_window + self.patch_radius)

        max_height_diff = self.patch_max_height_diff
        passing = _patch_pass_fraction(prepared, xy_tile, self.patch_window, self.patch_radius, max_height_diff)
        if passing < 0.05:
            warnings.warn(
                f"only {passing:.1%} of the '{name}' patch window is flat enough"
                f" (radius {self.patch_radius}, max height diff {max_height_diff:.2f});"
                " the terrain generator's rejection sampling may fail. Widen the tolerance, shrink"
                " the radius, or move the pose onto flatter rock.",
                stacklevel=3,
            )

        return FlatPatchSamplingCfg(
            num_patches=self.patch_num_samples,
            patch_radius=self.patch_radius,
            x_range=(float(xy_tile[0] - start_tile[0] - self.patch_window),
                     float(xy_tile[0] - start_tile[0] + self.patch_window)),
            y_range=(float(xy_tile[1] - start_tile[1] - self.patch_window),
                     float(xy_tile[1] - start_tile[1] + self.patch_window)),
            z_range=(float(z_lo - self.patch_z_margin), float(z_hi + self.patch_z_margin)),
            max_height_diff=max_height_diff,
        )

    def __post_init__(self):
        path = resolve_scan_path(self.npz_path)
        self.fingerprint = content_fingerprint(path)

        # bearing decides the rotation, so it is measured before one is applied
        grid = load_scan_grid(str(path), self.fingerprint)
        del grid  # loaded here only to fail early on a corrupt archive
        bearing_obj = math.atan2(
            self.goal_xy_obj[1] - self.start_xy_obj[1], self.goal_xy_obj[0] - self.start_xy_obj[0]
        )
        k = _auto_rot90_k(bearing_obj) if self.rot90_k is None else int(self.rot90_k) % 4
        self.resolved_rot90_k = k
        self.heading = _wrap_to_pi(self.start_yaw_obj + k * math.pi / 2.0)
        self.bearing = _wrap_to_pi(bearing_obj + k * math.pi / 2.0)

        prepared = _prepare_grid(self, k)
        rect = prepared.footprint_rect
        self.footprint_rect_tile = tuple(float(v) for v in rect)

        start_tile = prepared.obj_to_tile(self.start_xy_obj)
        goal_tile = prepared.obj_to_tile(self.goal_xy_obj)
        for name, xy in (("start", start_tile), ("goal", goal_tile)):
            assert rect[0] <= xy[0] <= rect[1] and rect[2] <= xy[1] <= rect[3], (
                f"{name} pose {xy} (tile-local) lies outside the scan footprint {rect}; check the"
                " OBJ-frame coordinates against the prep script's preview."
            )

        start_z = prepared.probe(start_tile, self.start_probe_radius, self.start_probe_stat)
        goal_z = prepared.probe(goal_tile, self.goal_probe_radius, self.goal_probe_stat) + self.goal_z_offset
        z_lo, z_hi = (float(v) for v in np.percentile(prepared.heights, self.z_percentiles))

        self.goal_poses = {
            "start": (float(start_tile[0]), float(start_tile[1]), start_z),
            "goal": (float(goal_tile[0]), float(goal_tile[1]), goal_z),
        }

        # extras use the same probe and offset as the pose they stand in for
        extra: dict[str, tuple[float, float, float]] = {}
        for prefix, poses, radius, stat, offset in (
            ("start", self.extra_starts_obj, self.start_probe_radius, self.start_probe_stat, 0.0),
            ("goal", self.extra_goals_obj, self.goal_probe_radius, self.goal_probe_stat, self.goal_z_offset),
        ):
            for idx, xy_obj in enumerate(poses):
                xy = prepared.obj_to_tile(xy_obj)
                assert rect[0] <= xy[0] <= rect[1] and rect[2] <= xy[1] <= rect[3], (
                    f"extra {prefix} pose {idx + 1} at {xy_obj} maps to {xy} (tile-local), outside the"
                    f" scan footprint {rect}. Check it against the prep script's preview."
                )
                extra[f"{prefix}_{idx + 1}"] = (
                    float(xy[0]),
                    float(xy[1]),
                    prepared.probe(xy, radius, stat) + offset,
                )
        self.extra_poses = extra
        # env-relative xy: the start pose is the terrain origin
        self.geometry_bounds = {
            "x_lo": float(rect[0] - start_tile[0]),
            "x_hi": float(rect[1] - start_tile[0]),
            "y_lo": float(rect[2] - start_tile[1]),
            "y_hi": float(rect[3] - start_tile[1]),
            "z_lo": z_lo,
            "z_hi": z_hi,
            "start_z": start_z,
            "start_x_lo": 0.0,
            "start_x_hi": 0.0,
            "start_y_lo": 0.0,
            "start_y_hi": 0.0,
        }
        self.keep_in_regions = [(
            self.geometry_bounds["x_lo"],
            self.geometry_bounds["x_hi"],
            self.geometry_bounds["y_lo"],
            self.geometry_bounds["y_hi"],
        )]
        self.surface_top_z = z_hi

        # flat patches are still required by the terrain pose command and the stock reset event
        if self.flat_patch_sampling is None:
            self.flat_patch_sampling = {
                "init_pos": self._pose_patch(prepared, start_tile, "start"),
                "target": self._pose_patch(prepared, goal_tile, "goal"),
            }

        if self.perimeter_wall is not None:
            needed = z_hi + self.fence_clearance
            if self.perimeter_wall.height < needed:
                self.perimeter_wall.height = float(needed)
        if abs(_wrap_to_pi(self.bearing)) > math.pi / 4.0:
            warnings.warn(
                f"goal bearing {math.degrees(self.bearing):.0f} deg is far off +x even after"
                f" {k} quarter turn(s); the task space's yaw window is centred on the start heading,"
                " so check `rot90_k`.",
                stacklevel=2,
            )


def _patch_pass_fraction(prepared: PreparedGrid, centre, window: float, radius: float, max_diff: float) -> float:
    return float((prepared.ring_spreads(centre, window, radius) <= max_diff).mean())


def pinned_scan_cfg(
    sub_terrain: ScanHeightfieldTerrainCfg,
    sub_terrain_name: str,
    difficulty: float = 0.5,
    num_rows: int = 1,
    num_cols: int = 1,
    border_width: float = 4.0,
) -> TerrainGeneratorCfg:
    """Build a single-sub-terrain generator cfg for a scan, pinned to one difficulty."""
    return TerrainGeneratorCfg(
        size=sub_terrain.size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=False,
        difficulty_range=(difficulty, difficulty),
        sub_terrains={sub_terrain_name: sub_terrain},
        use_cache=False,
    )
