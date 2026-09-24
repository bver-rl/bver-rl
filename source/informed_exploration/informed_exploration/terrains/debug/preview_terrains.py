# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless previewer rendering each bridge preset across a difficulty ladder as a contact-sheet PNG and GLBs.

The sim app is launched only because ``isaaclab.terrains`` imports ``pxr``; nothing is stepped.
"""

import argparse
import re


def _parse_difficulty_tokens(tokens: list[str]) -> list[float]:
    """Parse difficulties from separate tokens or one bracketed/comma-separated token, as launch.json passes it."""
    if len(tokens) == 1 and ("," in tokens[0] or "[" in tokens[0]):
        cleaned = tokens[0].strip().strip("[]")
        parts = [p for p in re.split(r"[,\s]+", cleaned) if p]
    else:
        parts = tokens
    return [float(p) for p in parts]


parser = argparse.ArgumentParser(description="Preview bridge-terrain presets without launching a full environment.")
parser.add_argument("--headless", action="store_true", default=False, help="Don't open a viewer window.")
parser.add_argument(
    "--bridge", nargs="*", default=None, help="Preset names to render (default: all presets in PRESETS)."
)
parser.add_argument(
    "--difficulties",
    nargs="*",
    type=str,
    default=None,
    help="Difficulty ladder: space-separated ('0.0 0.5 1.0') or a single bracketed/comma token"
    " ('[0.0,0.5,1.0]'). Defaults to [0.0, 0.25, 0.5, 0.75, 1.0].",
)
parser.add_argument("--out", type=str, default=None, help="Output directory (default: ./output/bridge_terrains next to this script).")
parser.add_argument("--no-glb", action="store_true", default=False, help="Skip per-cell GLB export (PNG contact-sheets only).")
args_cli = parser.parse_args()
args_cli.difficulties = (
    [0.0, 0.25, 0.5, 0.75, 1.0] if args_cli.difficulties is None else _parse_difficulty_tokens(args_cli.difficulties)
)

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401 (registers the 3d projection)

from isaaclab.terrains.utils import color_meshes_by_height  # noqa: E402

from informed_exploration.terrains.config.presets import PRESETS  # noqa: E402

# rough AnymalD-scale reference box (length, width, height in m) for reading difficulty at a glance
ROBOT_REFERENCE_SIZE = (1.0, 0.5, 0.6)


def _reference_box(origin: np.ndarray) -> trimesh.Trimesh:
    box = trimesh.creation.box(ROBOT_REFERENCE_SIZE)
    box.apply_translation(origin + np.array([0.0, 0.0, ROBOT_REFERENCE_SIZE[2] / 2.0]))
    box.visual.vertex_colors = (0, 0, 0, 200)
    return box


def _plot_terrain(ax, meshes: list[trimesh.Trimesh], origin: np.ndarray, title: str, margin: float = 1.0) -> None:
    """Plot ``meshes``, without the ground plane, cropped to their own bounding box."""
    combined = trimesh.util.concatenate(meshes)
    verts, faces = combined.vertices, combined.faces
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="turbo", linewidth=0.05, antialiased=True)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="lime", s=40, depthshade=False)

    x_min, x_max = verts[:, 0].min() - margin, verts[:, 0].max() + margin
    y_min, y_max = verts[:, 1].min() - margin, verts[:, 1].max() + margin
    z_min, z_max = verts[:, 2].min() - margin, verts[:, 2].max() + margin
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, max(z_max - z_min, 0.5)))

    ax.view_init(elev=32, azim=-60)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x", fontsize=6)
    ax.set_ylabel("y", fontsize=6)
    ax.tick_params(labelsize=5)


def _feature_meshes(meshes: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Drop the full-tile ground plane the composer returns as ``meshes[0]``, which clutters the preview."""
    return meshes[1:] if len(meshes) > 1 else meshes


def main() -> None:
    out_dir = args_cli.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "bridge_terrains")
    os.makedirs(out_dir, exist_ok=True)

    names = args_cli.bridge if args_cli.bridge else list(PRESETS.keys())
    difficulties = args_cli.difficulties

    for name in names:
        if name not in PRESETS:
            print(f"[WARN] Unknown preset '{name}', skipping. Available: {list(PRESETS.keys())}")
            continue
        gen_cfg = PRESETS[name]()
        sub_names = list(gen_cfg.sub_terrains.keys())

        fig, axes = plt.subplots(
            len(difficulties),
            len(sub_names),
            figsize=(4.5 * len(sub_names), 4.0 * len(difficulties)),
            subplot_kw={"projection": "3d"},
            squeeze=False,
        )

        for row, difficulty in enumerate(difficulties):
            for col, sub_name in enumerate(sub_names):
                sub_cfg = gen_cfg.sub_terrains[sub_name]
                meshes, origin = sub_cfg.function(difficulty=difficulty, cfg=sub_cfg)
                feature_meshes = _feature_meshes(meshes)
                _plot_terrain(axes[row][col], feature_meshes, origin, f"{sub_name} d={difficulty:.2f}")

                if not args_cli.no_glb:
                    scene_meshes = list(feature_meshes) + [_reference_box(origin)]
                    colored = color_meshes_by_height(scene_meshes)
                    glb_path = os.path.join(out_dir, f"{name}_{sub_name}_d{difficulty:.2f}.glb")
                    trimesh.Scene(colored).export(glb_path)

        fig.suptitle(f"Bridge terrain: {name}", fontsize=12)
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{name}_contact_sheet.png")
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[{name}] wrote {png_path}" + ("" if args_cli.no_glb else f" + {len(difficulties) * len(sub_names)} GLBs"))

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
