# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn a raw surface scan into the height field the mesh-scan terrain consumes.

Sim-free. Rasterizes the top surface (max per cell), fills holes, despikes, and writes ``_hf.npz``,
``_meta.json`` and ``_preview.png`` next to the source; ``--debug`` adds a comparison against the mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as patheffects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy import ndimage  # noqa: E402

APRON = 0
MEASURED = 1
INPAINTED = 2
DESPIKED = 3
PROVENANCE_LABELS = {APRON: "outside", MEASURED: "measured", INPAINTED: "filled", DESPIKED: "despiked"}


# Raster pipeline


def load_scan(path: Path, up_axis: str, scale: float) -> trimesh.Trimesh:
    """Load the scan and bring it into a metres, z-up frame."""
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if scale != 1.0:
        mesh.apply_scale(scale)
    if up_axis == "y":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0]))
    return mesh


def synthetic_scan(extent: float = 3.5, cell: float = 0.02) -> trimesh.Trimesh:
    """A deterministic bumpy surface with one tall spike, for tests and dry runs."""
    n = int(extent / cell) + 1
    xs = np.linspace(0.0, extent, n)
    ys = np.linspace(0.0, extent, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = 0.15 * np.sin(2.2 * xx) * np.cos(1.7 * yy) + 0.05 * np.sin(9.0 * xx + 3.0 * yy)
    zz += 0.6 * np.exp(-(((xx - 2.4) ** 2 + (yy - 1.2) ** 2) / 0.02))  # the spike
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, d, b], [a, c, d]]
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def densify(mesh: trimesh.Trimesh, resolution: float, samples_per_cell: int, seed: int) -> np.ndarray:
    """Vertices plus surface samples, enough that every cell the scan covers receives a point.

    Seeded, since the sampling decides every cell height and thus the terrain cache hash.
    """
    area_cells = max(1.0, mesh.area / (resolution * resolution))
    n_samples = int(min(4e6, area_cells * samples_per_cell))
    points, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    return np.vstack([np.asarray(mesh.vertices), points])


def rasterize_max_z(points: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Highest sample per cell, which keeps ridges intact and flattens overhang undersides."""
    lo = points[:, :2].min(axis=0)
    hi = points[:, :2].max(axis=0)
    nx, ny = (np.floor((hi - lo) / resolution).astype(int) + 1)
    idx = np.clip(np.floor((points[:, :2] - lo) / resolution).astype(int), 0, [nx - 1, ny - 1])

    heights = np.full((nx, ny), -np.inf, dtype=np.float64)
    np.maximum.at(heights, (idx[:, 0], idx[:, 1]), points[:, 2])
    covered = np.isfinite(heights)
    heights[~covered] = np.nan
    # Origin is the cell centre, matching consumers that index `x0 + i * resolution`.
    return heights, covered, float(lo[0] + resolution / 2.0), float(lo[1] + resolution / 2.0)


def footprint_mask(covered: np.ndarray, close_iters: int) -> np.ndarray:
    """Covered cells closed over sampling gaps, with interior holes filled."""
    if close_iters > 0:
        # Padded so closing does not erode against the array border.
        pad = close_iters + 1
        padded = np.pad(covered, pad, mode="constant", constant_values=False)
        closed = ndimage.binary_closing(padded, iterations=close_iters)[pad:-pad, pad:-pad]
    else:
        closed = covered
    return ndimage.binary_fill_holes(closed)


def despike(heights: np.ndarray, covered: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Pull cells that stand above their neighbourhood median by more than ``threshold`` back down to it.

    Lower-only, since max binning errs upward; uncovered cells take their nearest measured neighbour as context.
    """
    if threshold <= 0.0:
        return heights, np.zeros_like(covered)
    _, indices = ndimage.distance_transform_edt(~covered, return_indices=True)
    filled = heights[tuple(indices)]
    median = ndimage.median_filter(filled, size=3, mode="nearest")
    spikes = covered & (filled - median > threshold)
    out = heights.copy()
    out[spikes] = median[spikes]
    return out, spikes


def inpaint(heights: np.ndarray, covered: np.ndarray, inside: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Give every cell inside the footprint a height, taken from the nearest measured cell."""
    missing = inside & ~covered
    if not missing.any():
        return heights, missing
    out = heights.copy()
    if method == "linear":
        from scipy.interpolate import griddata

        src = np.argwhere(covered)
        dst = np.argwhere(missing)
        values = griddata(src, heights[covered], dst, method="linear")
        nan = np.isnan(values)
        if nan.any():
            _, indices = ndimage.distance_transform_edt(~covered, return_indices=True)
            values[nan] = heights[tuple(indices)][missing][nan]
        out[missing] = values
    else:
        _, indices = ndimage.distance_transform_edt(~covered, return_indices=True)
        out[missing] = heights[tuple(indices)][missing]
    return out, missing


def normalise_z(heights: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift so the lowest cell of the footprint sits at zero."""
    offset = float(np.nanmin(heights[inside]))
    return heights - offset, offset


def slope_degrees(heights: np.ndarray, resolution: float) -> np.ndarray:
    gx, gy = np.gradient(np.nan_to_num(heights, nan=float(np.nanmin(heights))), resolution)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def probe_disc(heights: np.ndarray, x0: float, y0: float, resolution: float, xy, radius: float) -> dict:
    """Height statistics over a footprint-sized disc, mirroring the terrain cfg's probe."""
    nx, ny = heights.shape
    ci, cj = (xy[0] - x0) / resolution, (xy[1] - y0) / resolution
    r = max(radius / resolution, 1.0)
    i_lo, i_hi = max(0, int(ci - r)), min(nx - 1, int(math.ceil(ci + r)))
    j_lo, j_hi = max(0, int(cj - r)), min(ny - 1, int(math.ceil(cj + r)))
    ii, jj = np.meshgrid(np.arange(i_lo, i_hi + 1), np.arange(j_lo, j_hi + 1), indexing="ij")
    inside = (ii - ci) ** 2 + (jj - cj) ** 2 <= r * r
    values = heights[i_lo : i_hi + 1, j_lo : j_hi + 1][inside]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(values.max()),
        "relief": float(values.max() - values.min()),
    }


# Debug comparisons against the original mesh


def sample_heights(heights: np.ndarray, x0: float, y0: float, resolution: float, xy: np.ndarray) -> np.ndarray:
    """Nearest-cell lookup of the height field at arbitrary xy."""
    nx, ny = heights.shape
    i = np.clip(np.round((xy[:, 0] - x0) / resolution).astype(int), 0, nx - 1)
    j = np.clip(np.round((xy[:, 1] - y0) / resolution).astype(int), 0, ny - 1)
    return heights[i, j]


def residual_map(
    mesh: trimesh.Trimesh,
    heights: np.ndarray,
    inside: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    z_offset: float,
    seed: int,
    n: int = 1_000_000,
) -> tuple[np.ndarray, dict]:
    """Per-cell worst disagreement between the height field and the original surface.

    Positive residuals are expected under overhangs; negative ones should come only from despiking.
    """
    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nx, ny = heights.shape
    i = np.clip(np.round((points[:, 0] - x0) / resolution).astype(int), 0, nx - 1)
    j = np.clip(np.round((points[:, 1] - y0) / resolution).astype(int), 0, ny - 1)

    # Only cells the field describes.
    on_footprint = inside[i, j]
    outside_fraction = float((~on_footprint).mean())
    points, i, j = points[on_footprint], i[on_footprint], j[on_footprint]
    residual = heights[i, j] - (points[:, 2] - z_offset)

    # -inf, not NaN, since np.maximum propagates NaN.
    worst = np.full((nx, ny), -np.inf)
    np.maximum.at(worst, (i, j), np.abs(residual))
    worst[~np.isfinite(worst)] = np.nan

    stats = {
        "samples": int(points.shape[0]),
        "samples_off_footprint_fraction": outside_fraction,
        "abs_percentiles": {
            str(p): float(v) for p, v in zip([50, 90, 99, 99.9, 100], np.percentile(np.abs(residual), [50, 90, 99, 99.9, 100]))
        },
        "signed_percentiles": {
            str(p): float(v) for p, v in zip([0.1, 1, 50, 99, 99.9], np.percentile(residual, [0.1, 1, 50, 99, 99.9]))
        },
        "frac_above_5cm_below_field": float((residual < -0.05).mean()),
    }
    return worst, stats


def cross_section(mesh: trimesh.Trimesh, p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original-mesh profile along the segment ``p0 -> p1`` (arc length, z of every crossing)."""
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    direction = direction / length
    normal = np.array([-direction[1], direction[0], 0.0])
    segments = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=np.append(p0, 0.0))
    if len(segments) == 0:
        return np.zeros(0), np.zeros(0)
    pts = segments.reshape(-1, 3)
    t = (pts[:, :2] - p0) @ direction
    keep = (t >= 0.0) & (t <= length)
    order = np.argsort(t[keep])
    return t[keep][order], pts[keep][order, 2]


# Outputs


def draw_poses(ax, args) -> None:
    """Mark the measured pair and the ``eval_scan_pairs.py`` extras on one map panel.

    Each extra is drawn one size down and joined to the measured pose it is paired with.
    """
    start, goal = args.start_xy, args.goal_xy
    extra_starts = getattr(args, "extra_starts", ()) or ()
    extra_goals = getattr(args, "extra_goals", ()) or ()
    # Stroked labels stay readable on both light and dark panels.
    label_style = dict(
        fontsize=6.5,
        color="white",
        path_effects=[patheffects.withStroke(linewidth=1.6, foreground="black")],
    )

    # Pairing fan first, so the markers sit on top of it.
    for xy in extra_starts:
        if goal is not None:
            ax.annotate("", xy=goal, xytext=xy, arrowprops=dict(arrowstyle="->", color="white", lw=0.7, alpha=0.45))
    for xy in extra_goals:
        if start is not None:
            ax.annotate("", xy=xy, xytext=start, arrowprops=dict(arrowstyle="->", color="white", lw=0.7, alpha=0.45))

    for i, xy in enumerate(extra_starts, 1):
        ax.plot(*xy, "o", color="limegreen", ms=5, mec="black", mew=0.6, alpha=0.85)
        ax.annotate(f"s{i}", xy=xy, xytext=(4, 3), textcoords="offset points", **label_style)
    for j, xy in enumerate(extra_goals, 1):
        ax.plot(*xy, "*", color="gold", ms=9, mec="black", mew=0.6, alpha=0.85)
        ax.annotate(f"g{j}", xy=xy, xytext=(4, -9), textcoords="offset points", **label_style)

    # Measured pair last and largest.
    if start is not None:
        ax.plot(*start, "o", color="limegreen", ms=9, mec="black")
    if goal is not None:
        ax.plot(*goal, "*", color="gold", ms=16, mec="black")
    if start is not None and goal is not None:
        ax.annotate("", xy=goal, xytext=start, arrowprops=dict(arrowstyle="->", color="white", lw=1.5))


def write_preview(
    png: Path,
    heights: np.ndarray,
    inside: np.ndarray,
    provenance: np.ndarray,
    x0: float,
    y0: float,
    resolution: float,
    args,
    residual: np.ndarray | None,
    sections: list | None,
) -> None:
    nx, ny = heights.shape
    extent = [x0, x0 + nx * resolution, y0, y0 + ny * resolution]
    shown = np.where(inside, heights, np.nan)
    slope = np.where(inside, slope_degrees(heights, resolution), np.nan)

    # Hillshade.
    gx, gy = np.gradient(np.nan_to_num(heights), resolution)
    az, alt = math.radians(315.0), math.radians(45.0)
    shade = (np.sin(alt) + np.cos(alt) * (np.cos(az) * -gx + np.sin(az) * -gy)) / np.sqrt(1 + gx**2 + gy**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    panels = [
        (shown.T, "height [m]", "terrain"),
        (np.where(inside, shade, np.nan).T, "hillshade", "gray"),
        (slope.T, "slope [deg]", "magma"),
        (provenance.T, "provenance", "viridis"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)

    ax = axes.ravel()[4]
    if residual is not None:
        im = ax.imshow(np.where(inside, residual, np.nan).T, origin="lower", extent=extent, cmap="inferno")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("|residual| vs original mesh [m]")
    else:
        ax.hist(heights[inside].ravel(), bins=80)
        ax.set_title("height histogram [m]")
        ax.set_xlabel("z [m]")

    ax = axes.ravel()[5]
    if sections:
        for label, t_mesh, z_mesh, t_line, z_hf in sections:
            ax.plot(t_mesh, z_mesh, ".", ms=1.5, alpha=0.35, label=f"{label}: mesh")
            ax.plot(t_line, z_hf, "-", lw=1.5, label=f"{label}: field")
        ax.legend(fontsize=8)
        ax.set_xlabel("distance along section [m]")
        ax.set_ylabel("z [m]")
        ax.set_title("cross sections")
    else:
        ax.axis("off")

    for ax in axes.ravel()[:4]:
        ax.set_xlabel("x [m]  (OBJ frame)")
        ax.set_ylabel("y [m]  (OBJ frame)")
        draw_poses(ax, args)

    fig.suptitle(png.stem, fontsize=13)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def _load_by_path(name: str, path: Path):
    """Load a module from file, keeping this script free of the Isaac runtime."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_tile_glb(
    glb: Path, heights: np.ndarray, inside: np.ndarray, resolution: float, ring: int, lift: float, drop: float
) -> None:
    """Export the tile sheet exactly as the terrain builds it, using the same fill and converter."""
    fill = _load_by_path(
        "_scan_fill",
        Path(__file__).resolve().parents[2]
        / "source/informed_exploration/informed_exploration/terrains/scan/fill.py",
    )
    module = _load_by_path(
        "_hf_utils", Path(__file__).resolve().parents[3] / "source/isaaclab/isaaclab/terrains/height_field/utils.py"
    )

    filled = fill.fill_harmonic(np.nan_to_num(heights), inside)
    padded = np.pad(filled + lift, ring, mode="constant", constant_values=-drop)
    vertices, faces = module.convert_height_field_to_mesh(padded, resolution, 1.0, None)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(str(glb))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize a surface scan into a terrain height field.")
    parser.add_argument("--obj", type=str, default=None, help="Raw scan (.obj/.ply/.stl).")
    parser.add_argument("--synthetic", action="store_true", help="Use a generated test surface instead of --obj.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: next to the source).")
    parser.add_argument("--name", type=str, default=None, help="Artifact base name (default: the source stem).")
    parser.add_argument("--resolution", type=float, default=0.025, help="Grid cell size in m.")
    parser.add_argument("--up_axis", choices=["z", "y"], default="z", help="Up axis of the source.")
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Real-world metres per unit of the source file. REQUIRED, pass 1.0 for a scan already"
        " exported in metres. The height field on disk is always metric, so --resolution is always a"
        " real cell size; a scan that is merely small should be enlarged with a ScanSpec's"
        " xy_scale/z_scale instead, never here.",
    )
    parser.add_argument("--fill", choices=["nearest", "linear"], default="nearest", help="Hole filling.")
    parser.add_argument("--despike", type=float, default=0.05, help="Spike threshold in m (0 disables).")
    parser.add_argument("--close_iters", type=int, default=2, help="Binary closing iterations for the footprint.")
    parser.add_argument("--samples_per_cell", type=int, default=20, help="Surface samples per grid cell.")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the surface sampling; keep it fixed.")
    parser.add_argument("--start_xy", type=float, nargs=2, default=None, help="Start pose in the OBJ frame.")
    parser.add_argument("--goal_xy", type=float, nargs=2, default=None, help="Goal pose in the OBJ frame.")
    parser.add_argument(
        "--scan",
        type=str,
        default=None,
        help=(
            "snake_case scan name, e.g. small_rocks. Fills the poses from that ScanSpec instead of"
            " the command line: the measured pair AND the pairs eval's extra starts and goals, all"
            " drawn on the preview. 'auto' picks the spec whose stem matches --obj. An explicit"
            " --start_xy / --goal_xy still wins, which is what lets a new pose be tried out before"
            " it is written into scans.py."
        ),
    )
    parser.add_argument("--probe_radius", type=float, default=0.35, help="Footprint disc radius for the probes.")
    parser.add_argument("--debug", action="store_true", help="Compare the field against the original mesh.")
    parser.add_argument("--export_glb", action="store_true", help="Also write the tile sheet as a GLB.")
    parser.add_argument("--surface_lift", type=float, default=0.1, help="Lift used for the debug GLB only.")
    parser.add_argument("--apron_ring_cells", type=int, default=2, help="Apron ring used for the debug GLB only.")
    parser.add_argument("--apron_drop", type=float, default=0.02, help="Apron drop used for the debug GLB only.")
    args = parser.parse_args()

    assert args.obj or args.synthetic, "pass --obj or --synthetic"
    if args.synthetic:
        mesh = synthetic_scan()
        source = Path("synthetic_scan.obj")
        source_md5 = "synthetic"
    else:
        source = Path(args.obj).resolve()
        mesh = load_scan(source, args.up_axis, args.scale)
        source_md5 = hashlib.md5(source.read_bytes()).hexdigest()[:12]

    name = args.name or source.stem
    out_dir = Path(args.out_dir) if args.out_dir else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    args.extra_starts, args.extra_goals = (), ()
    if args.scan is not None:
        import scan_specs

        spec = scan_specs.scan_for_stem(source.stem) if args.scan == "auto" else scan_specs.resolve_scan(args.scan)
        if spec is None:
            print(f"[poses] --scan auto: no ScanSpec has stem {source.stem!r}; drawing no poses")
        else:
            # An explicit pose still wins.
            args.start_xy = args.start_xy or (list(spec.start_xy_obj) if spec.start_xy_obj else None)
            args.goal_xy = args.goal_xy or (list(spec.goal_xy_obj) if spec.goal_xy_obj else None)
            args.extra_starts, args.extra_goals = spec.extra_starts_obj, spec.extra_goals_obj
            print(
                f"[poses] {spec.snake}: start {args.start_xy}, goal {args.goal_xy},"
                f" plus {len(args.extra_starts)} extra start(s) and {len(args.extra_goals)} extra goal(s)"
                " from the pairs eval"
            )

    print(f"[scan] {source.name}: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    print(f"[scan] bounds {np.round(mesh.bounds, 3).tolist()}  extents {np.round(mesh.extents, 3).tolist()}")
    down = float((mesh.face_normals[:, 2] < -0.5).mean())
    print(f"[scan] downward-facing faces (overhangs, flattened by the raster): {down:.1%}")

    points = densify(mesh, args.resolution, args.samples_per_cell, args.seed)
    heights, covered, x0, y0 = rasterize_max_z(points, args.resolution)
    inside = footprint_mask(covered, args.close_iters)
    heights, spikes = despike(heights, covered, args.despike)
    heights, filled = inpaint(heights, covered, inside, args.fill)
    heights, z_offset = normalise_z(heights, inside)
    heights = np.where(inside, heights, np.nan).astype(np.float32)

    provenance = np.full(heights.shape, APRON, dtype=np.uint8)
    provenance[inside & covered] = MEASURED
    provenance[filled] = INPAINTED
    provenance[spikes] = DESPIKED

    nx, ny = heights.shape
    print(f"[grid] {nx} x {ny} cells at {args.resolution} m, footprint {inside.mean():.1%} of the AABB")
    print(f"[grid] filled {int(filled.sum())} cells, despiked {int(spikes.sum())} cells, z shifted by {z_offset:+.3f} m")
    print(f"[grid] z range 0.000 .. {float(np.nanmax(heights)):.3f} m")
    # The fence follows the bounding box, so unscanned cells inside it are extrapolated walkable ground.
    extrapolated = float((~covered).mean())
    print(
        f"[grid] {extrapolated:.1%} of the tile's bounding box was never scanned and is blended in"
        f" from the rim{' -- consider a footprint_rect_obj crop' if extrapolated > 0.1 else ''}"
    )

    residual, residual_stats, sections = None, None, []
    if args.debug:
        residual, residual_stats = residual_map(mesh, np.nan_to_num(heights), inside, x0, y0, args.resolution, z_offset, args.seed)
        print(f"[debug] |residual| percentiles {residual_stats['abs_percentiles']}")
        print(f"[debug] fraction of surface above the field by >5 cm: {residual_stats['frac_above_5cm_below_field']:.3%}")

        if args.start_xy and args.goal_xy:
            lines = [("start->goal", np.array(args.start_xy), np.array(args.goal_xy))]
        else:
            mid_x, mid_y = x0 + nx * args.resolution / 2.0, y0 + ny * args.resolution / 2.0
            lines = [
                ("mid-x", np.array([x0, mid_y]), np.array([x0 + (nx - 1) * args.resolution, mid_y])),
                ("mid-y", np.array([mid_x, y0]), np.array([mid_x, y0 + (ny - 1) * args.resolution])),
            ]
        for label, p0, p1 in lines:
            t_mesh, z_mesh = cross_section(mesh, p0, p1)
            length = float(np.linalg.norm(p1 - p0))
            t_line = np.linspace(0.0, length, 400)
            xy = p0 + np.outer(t_line, (p1 - p0) / length)
            z_hf = sample_heights(np.nan_to_num(heights), x0, y0, args.resolution, xy)
            sections.append((label, t_mesh, z_mesh - z_offset, t_line, z_hf))

    npz = out_dir / f"{name}_hf.npz"
    np.savez_compressed(
        npz,
        heights=heights,
        valid=inside,
        resolution=np.float64(args.resolution),
        x0=np.float64(x0),
        y0=np.float64(y0),
        z_offset=np.float64(z_offset),
        source_name=str(source.name),
        source_md5=source_md5,
        version=np.int64(1),
    )

    meta = {
        "source": {"name": source.name, "md5": source_md5, "faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))},
        "source_bounds": np.round(mesh.bounds, 4).tolist(),
        "downward_face_fraction": round(down, 5),
        "grid": {"nx": int(nx), "ny": int(ny), "resolution": args.resolution, "x0": x0, "y0": y0, "z_offset": z_offset},
        "footprint": {
            "fraction_of_aabb": float(inside.mean()),
            "cells_measured": int((inside & covered).sum()),
            "cells_filled": int(filled.sum()),
            "cells_despiked": int(spikes.sum()),
            # Unscanned share of the bounding box; crop with a ScanSpec's `footprint_rect_obj`.
            "fraction_extrapolated": extrapolated,
        },
        "z_percentiles": {
            str(p): float(v) for p, v in zip([0, 0.5, 2, 50, 98, 99.5, 100], np.nanpercentile(heights[inside], [0, 0.5, 2, 50, 98, 99.5, 100]))
        },
        "slope_percentiles_deg": {
            str(p): float(v) for p, v in zip([50, 90, 99], np.percentile(slope_degrees(np.nan_to_num(heights), args.resolution)[inside], [50, 90, 99]))
        },
        "args": vars(args),
    }
    if args.start_xy:
        meta["start_probe"] = probe_disc(heights, x0, y0, args.resolution, args.start_xy, args.probe_radius)
    if args.goal_xy:
        meta["goal_probe"] = probe_disc(heights, x0, y0, args.resolution, args.goal_xy, args.probe_radius)
    if residual_stats:
        meta["residual"] = residual_stats
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))

    write_preview(
        out_dir / f"{name}_preview.png", heights, inside, provenance, x0, y0, args.resolution, args, residual, sections
    )

    if args.export_glb:
        export_tile_glb(
            out_dir / f"{name}_tile.glb", heights, inside, args.resolution, args.apron_ring_cells,
            args.surface_lift, args.apron_drop,
        )

    print(f"[out] {npz}")
    print(f"[out] {out_dir / f'{name}_meta.json'}")
    print(f"[out] {out_dir / f'{name}_preview.png'}")
    if args.export_glb:
        print(f"[out] {out_dir / f'{name}_tile.glb'}")


if __name__ == "__main__":
    main()
