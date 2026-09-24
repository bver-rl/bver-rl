# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline previewer for the mesh-scan terrains in :data:`~..config.scans.SCANS`.

Renders the composed tile with the measured poses and, when the task cfg imports, the sampling region.
Kit is launched only because building the meshes imports USD.
"""

import argparse

parser = argparse.ArgumentParser(description="Preview mesh-scan terrains without launching a full environment.")
parser.add_argument("--scan", nargs="*", default=None, help="Scan names to render (default: all in SCANS).")
parser.add_argument("--out", type=str, default=None, help="Output directory (default: ./output/scans next to this script).")
parser.add_argument("--no-glb", action="store_true", default=False, help="Skip GLB export (PNG only).")
args_cli = parser.parse_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import trimesh  # noqa: E402

from informed_exploration.terrains.config.scans import SCANS  # noqa: E402
from informed_exploration.terrains.debug.scan_figures import tile_figure  # noqa: E402


def _spec_for(stem: str):
    """Return the :class:`ScanSpec` whose artifacts use this file stem, or ``None``."""
    from informed_exploration.terrains.config.scans import SCAN_SPECS

    return next((spec for spec in SCAN_SPECS.values() if spec.stem == stem), None)


def _task_bounds(stem: str):
    """Return the env cfg's x/y task-space and initial-state windows, or ``None`` for an unmeasured scan."""
    spec = _spec_for(stem)
    if spec is None or not spec.measured:
        print(f"[{stem}] no measured poses yet; skipping the sampling overlay.")
        return None
    try:
        from informed_exploration.tasks.manager_based.informed_exploration.cfg.scan import scan_families

        cfg = getattr(scan_families, scan_families.env_cfg_name(spec, "BVER"))()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not build the task cfg for '{stem}' ({exc}); skipping the overlay.")
        return None
    task_space = cfg.task_space
    return (
        (task_space.bounds[0], task_space.bounds[2]),
        (task_space.eval_bounds[0], task_space.eval_bounds[2]),
    )


def main() -> None:
    out_dir = args_cli.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "scans")
    os.makedirs(out_dir, exist_ok=True)

    names = args_cli.scan if args_cli.scan else list(SCANS.keys())
    for name in names:
        if name not in SCANS:
            print(f"[WARN] Unknown scan '{name}', skipping. Available: {list(SCANS.keys())}")
            continue

        gen_cfg = SCANS[name]()
        sub_cfg = next(iter(gen_cfg.sub_terrains.values()))
        meshes, origin = sub_cfg.function(gen_cfg.difficulty_range[0], sub_cfg)

        print(f"[{name}] {len(meshes)} meshes, {sum(len(m.faces) for m in meshes):,} triangles/tile")
        print(f"[{name}] origin (start pose, tile) {origin.round(3).tolist()}")
        print(f"[{name}] heading {sub_cfg.heading:.3f} rad, rot90_k {sub_cfg.resolved_rot90_k}")
        print(f"[{name}] goal_poses {sub_cfg.goal_poses}")

        spec = _spec_for(name)
        label = name if spec is None or spec.measured else f"{name}  [UNMEASURED: placeholder poses]"
        if spec is not None and not spec.measured:
            print(f"[{name}] start/goal are PLACEHOLDERS at the footprint centre, not measured poses")
        figure = tile_figure(sub_cfg, meshes, task_bounds=_task_bounds(name), name=label)
        png = os.path.join(out_dir, f"{name}_tile.png")
        figure.savefig(png, dpi=105)
        print(f"[{name}] wrote {png}")

        if not args_cli.no_glb:
            glb = os.path.join(out_dir, f"{name}_tile.glb")
            trimesh.util.concatenate(meshes).export(glb)
            print(f"[{name}] wrote {glb}")


if __name__ == "__main__":
    main()
    simulation_app.close()
