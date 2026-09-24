# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Propose extra start and goal poses on a prepared scan for the pairs eval, ready to paste into scans.py.

Poses need a measured, flat probe disc clear of the footprint edge, judged in the tile frame. Starts are
spread by bearing around the goal and goals by height. Verify with "Check Start/Goal Poses" afterwards.
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np

import scan_specs

REPO = pathlib.Path(__file__).resolve().parents[3]
PKG = REPO / "informed-exploration/source/informed_exploration/informed_exploration"
MESH = PKG / "terrains/mesh_scans"

PROBE_R_TILE = 0.35
"""Footprint disc the surface is judged over. Mirrors ``ScanHeightfieldTerrainCfg.start_probe_radius``."""

FLAT_FLOOR = 0.20
"""Spread (m, tile frame) allowed across that disc, unless the scan's own canonical pose is rougher."""

MEASURED_FLOOR_GOAL = 0.8
"""Fraction of a goal's disc that must be measured rather than filled; below one since canonical goals may be too."""

RIM_START_TILE = 0.7
RIM_GOAL_TILE = 0.5
"""Standoff (m, tile frame) from the footprint edge; the start's also covers the body and eval xy jitter."""

MIN_SEP_TILE = 1.0
MIN_TASK_TILE = 1.5
"""Minimum pose separation and pair length; the separation must exceed the eval jitter for label matching."""


def load_grid(stem: str):
    with np.load(MESH / f"{stem}_hf.npz") as data:
        return (
            np.asarray(data["heights"], np.float64),
            np.asarray(data["valid"], bool),
            float(data["resolution"]),
            float(data["x0"]),
            float(data["y0"]),
        )


def disc_stats(heights, valid, res, xy_scale, z_scale):
    """Per-cell ``(fully_measured, measured_fraction, tile-frame spread)`` over the probe disc."""
    r = max(int(round((PROBE_R_TILE / xy_scale) / res)), 1)
    ii, jj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    keep = (ii**2 + jj**2) <= r * r
    offs = np.stack([ii[keep], jj[keep]], -1)
    nx, ny = heights.shape
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ai, aj = gi[..., None] + offs[:, 0], gj[..., None] + offs[:, 1]
    inside = (ai >= 0) & (ai < nx) & (aj >= 0) & (aj < ny)
    hh = heights[np.clip(ai, 0, nx - 1), np.clip(aj, 0, ny - 1)]
    vv = valid[np.clip(ai, 0, nx - 1), np.clip(aj, 0, ny - 1)] & inside
    full = (vv | ~inside).all(-1) & inside.all(-1)
    with np.errstate(invalid="ignore"):
        frac = vv.sum(-1) / np.maximum(inside.sum(-1), 1)
        masked = np.where(vv, hh, np.nan)
        spread = (np.nanmax(masked, -1) - np.nanmin(masked, -1)) * z_scale
    return full, frac, spread


def propose(spec, n_extra: int):
    heights, valid, res, x0, y0 = load_grid(spec.stem)
    nx, ny = heights.shape
    xy_s, z_s = spec.xy_scale, spec.z_scale
    full, frac, spread = disc_stats(heights, valid, res, xy_s, z_s)

    xs, ys = x0 + np.arange(nx) * res, y0 + np.arange(ny) * res
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    rim = np.minimum.reduce([grid_x - xs[0], xs[-1] - grid_x, grid_y - ys[0], ys[-1] - grid_y]) * xy_s

    def canonical_spread(p):
        ci, cj = (p[0] - x0) / res, (p[1] - y0) / res
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        r = max((PROBE_R_TILE / xy_s) / res, 1.0)
        mask = ((gi - ci) ** 2 + (gj - cj) ** 2) <= r * r
        return float((np.nanmax(heights[mask]) - np.nanmin(heights[mask])) * z_s)

    def height_at(p):
        return float(heights[int(round((p[0] - x0) / res)), int(round((p[1] - y0) / res))])

    start, goal = spec.start_xy_obj, spec.goal_xy_obj
    flat_start = max(FLAT_FLOOR, canonical_spread(start))
    flat_goal = max(FLAT_FLOOR, canonical_spread(goal))
    ok_start = full & (spread <= flat_start) & (rim >= RIM_START_TILE)
    ok_goal = (frac >= MEASURED_FLOOR_GOAL) & (spread <= flat_goal) & (rim >= RIM_GOAL_TILE)

    def candidates(mask, anchor):
        out = []
        for i, j in zip(*np.nonzero(mask)):
            p = (float(grid_x[i, j]), float(grid_y[i, j]))
            if math.hypot(p[0] - anchor[0], p[1] - anchor[1]) < MIN_TASK_TILE / xy_s:
                continue
            out.append((p, float(spread[i, j]), float(heights[i, j]), float(rim[i, j])))
        return out

    sep = MIN_SEP_TILE / xy_s

    def separated(p, refs):
        return all(math.hypot(p[0] - q[0], p[1] - q[1]) >= sep for q in refs)

    # Starts: one per bearing sector around the canonical goal.
    cand = candidates(ok_start, goal)
    starts, taken = [], [start, goal]
    base = math.atan2(start[1] - goal[1], start[0] - goal[0])
    for k in range(1, n_extra + 1):
        centre, best = base + 2.0 * math.pi * k / (n_extra + 1), None
        for p, sp, _z, rm in cand:
            if not separated(p, taken):
                continue
            ang = math.atan2(p[1] - goal[1], p[0] - goal[0])
            if abs((ang - centre + math.pi) % (2.0 * math.pi) - math.pi) > math.pi / (n_extra + 1):
                continue
            score = -sp + 0.05 * min(rm, 2.0)  # flattest footing, well inside the fence
            if best is None or score > best[0]:
                best = (score, p, sp)
        if best is not None:
            starts.append((best[1], best[2]))
            taken.append(best[1])

    # A narrow scan cannot fill every sector, so top up by distance.
    while len(starts) < n_extra:
        best = None
        for p, sp, _z, _rm in cand:
            if not separated(p, taken):
                continue
            d = min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in taken)
            if best is None or d > best[0]:
                best = (d, p, sp)
        if best is None:
            break
        starts.append((best[1], best[2]))
        taken.append(best[1])

    # Goals: the summit first, then height quantiles below it.
    cand_g = candidates(ok_goal, start)
    goals, taken_g = [], [goal]
    if cand_g:
        zs = np.array([c[2] for c in cand_g])
        for target in ([zs.max()] + list(np.percentile(zs, [70.0, 40.0, 10.0])))[:n_extra]:
            best = None
            for p, sp, z, rm in cand_g:
                if not separated(p, taken_g + [start]):
                    continue
                score = -abs(z - target) - 0.3 * sp + 0.02 * min(rm, 2.0)
                if best is None or score > best[0]:
                    best = (score, p, sp)
            if best is not None:
                goals.append((best[1], best[2]))
                taken_g.append(best[1])

    print(f"\n=== {spec.snake} ===")
    print(
        f"  tolerances: start spread <= {flat_start:.2f} m, goal <= {flat_goal:.2f} m"
        f"  (this scan's canonical start {canonical_spread(start):.2f}, goal {canonical_spread(goal):.2f})"
    )
    print(f"  feasible cells: {ok_start.mean():.1%} for a start, {ok_goal.mean():.1%} for a goal")
    if len(starts) < n_extra or len(goals) < n_extra:
        print(
            f"  NOTE: only {len(starts)} start / {len(goals)} goal candidates fit; this scan has"
            " little ground that clears the thresholds. Widen FLAT_FLOOR or lower the rim standoff"
            " if you want more, and say so in the spec's comment."
        )
    print("    extra_starts_obj=(")
    for p, sp in starts:
        d = math.hypot(p[0] - goal[0], p[1] - goal[1]) * xy_s
        bearing = math.degrees(math.atan2(p[1] - goal[1], p[0] - goal[0]))
        print(f"        ({p[0]:.2f}, {p[1]:.2f}),  # {d:.1f} m out, bearing {bearing:+4.0f} deg, spread {sp:.2f} m")
    print("    ),")
    print("    extra_goals_obj=(")
    for p, sp in goals:
        d = math.hypot(p[0] - start[0], p[1] - start[1]) * xy_s
        rise = (height_at(p) - height_at(start)) * z_s
        print(f"        ({p[0]:.2f}, {p[1]:.2f}),  # {d:.1f} m out, {rise:+.2f} m of rise, spread {sp:.2f} m")
    print("    ),")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", default=None, help="snake_case scan name; every prepared scan if omitted.")
    parser.add_argument("--n_extra", type=int, default=4, help="Candidates proposed per end.")
    args = parser.parse_args()

    specs = scan_specs.load_scan_specs()
    if args.scan is not None:
        if args.scan not in specs:
            raise SystemExit(f"unknown scan {args.scan!r}; known: {', '.join(sorted(specs))}")
        chosen = {args.scan: specs[args.scan]}
    else:
        chosen = specs

    for snake, spec in chosen.items():
        if not (MESH / f"{spec.stem}_hf.npz").exists():
            print(f"\n=== {snake} ===\n  not prepared yet; run prepare_scan.py on {spec.obj} first")
            continue
        if not spec.measured:
            print(f"\n=== {snake} ===\n  no measured start/goal yet; the extras are placed relative to them")
            continue
        propose(spec, args.n_extra)

    print(
        "\nPaste into the scan's entry in terrains/config/scans.py, marked PROVISIONAL, then verify"
        '\nwith the VS Code "Check Start/Goal Poses" config before trusting any pairs result.'
    )


if __name__ == "__main__":
    main()
