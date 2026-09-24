# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""In-sim viewer for a bridge, layout or scan terrain, showing origins and derived patches with physics running."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactively view a bridge-terrain preset.")
parser.add_argument(
    "--bridge", type=str, default="stepping_stones", help="Terrain name from PRESETS or LAYOUTS to load."
)
parser.add_argument(
    "--use_curriculum", action="store_true", default=False, help="Arrange sub-terrains by difficulty (rows)."
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Visualize the auto-derived init_pos/target patches.",
)
parser.add_argument(
    "--show_keep_in_regions",
    action="store_true",
    default=False,
    help="Visualize the auto-derived task-space keep-in regions (walkable platform/span footprints).",
)
parser.add_argument("--color_scheme", type=str, default="height", choices=["height", "random", "none"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from informed_exploration.terrains.config.layouts import LAYOUTS
from informed_exploration.terrains.config.presets import PRESETS
from informed_exploration.terrains.config.scans import SCANS

# merge the preset, layout and scan registries so --bridge selects any of them by name
_TERRAINS = {**PRESETS, **LAYOUTS, **SCANS}


def _visualize_keep_in_regions(terrain_gen_cfg, terrain_importer: TerrainImporter) -> None:
    """Overlay the derived task-space keep-in regions as translucent slabs on the decks.

    Only drawn for a single-sub-terrain generator, since tiles of a mixed generator carry different layouts.
    """
    if len(terrain_gen_cfg.sub_terrains) != 1:
        print(
            f"[WARN] --show_keep_in_regions needs a single sub-terrain, got"
            f" {list(terrain_gen_cfg.sub_terrains.keys())}; skipping the overlay."
        )
        return
    sub_cfg = next(iter(terrain_gen_cfg.sub_terrains.values()))
    regions = getattr(sub_cfg, "keep_in_regions", None)
    if not regions:
        print("[WARN] sub-terrain has no derived `keep_in_regions`; skipping the overlay.")
        return

    vis_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TerrainKeepInRegions",
        markers={
            "keep_in": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),  # unit cube; per-instance `scales` below give each its extent
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0), opacity=0.35),
            )
        },
    )
    regions_visualizer = VisualizationMarkers(vis_cfg)

    # regions are XY-only and env-relative to each tile origin; park the slabs just above the lowest deck
    slab_thickness = 0.04
    deck_z = sub_cfg.geometry_bounds["z_lo"] + slab_thickness
    tile_origins = terrain_importer.terrain_origins
    tile_origins = terrain_importer.env_origins if tile_origins is None else tile_origins.reshape(-1, 3)

    translations, scales = [], []
    for origin in tile_origins:
        for x_lo, x_hi, y_lo, y_hi in regions:
            translations.append([
                float(origin[0]) + (x_lo + x_hi) / 2.0,
                float(origin[1]) + (y_lo + y_hi) / 2.0,
                float(origin[2]) + deck_z,
            ])
            scales.append([x_hi - x_lo, y_hi - y_lo, slab_thickness])

    regions_visualizer.visualize(
        translations=torch.tensor(translations, dtype=torch.float32),
        scales=torch.tensor(scales, dtype=torch.float32),
        marker_indices=[0] * len(translations),
    )
    print(f"[INFO] drew {len(regions)} keep-in regions on {len(tile_origins)} tile(s).")


def design_scene() -> tuple[dict, torch.Tensor]:
    """Design the scene."""
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    if args_cli.bridge not in _TERRAINS:
        raise ValueError(f"Unknown --bridge '{args_cli.bridge}'. Available: {list(_TERRAINS.keys())}")
    terrain_gen_cfg = _TERRAINS[args_cli.bridge]()
    terrain_gen_cfg.curriculum = args_cli.use_curriculum
    terrain_gen_cfg.color_scheme = args_cli.color_scheme

    terrain_importer_cfg = TerrainImporterCfg(
        num_envs=terrain_gen_cfg.num_rows * terrain_gen_cfg.num_cols,
        env_spacing=0.0,
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=terrain_gen_cfg,
        debug_vis=True,
    )
    if args_cli.color_scheme in ["height", "random"]:
        terrain_importer_cfg.visual_material = None
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    if args_cli.show_flat_patches:
        vis_cfg = VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers={})
        for name in terrain_importer.flat_patches:
            vis_cfg.markers[name] = sim_utils.CylinderCfg(
                radius=0.3,
                height=0.1,
                visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
            )
        flat_patches_visualizer = VisualizationMarkers(vis_cfg)

        all_patch_locations = []
        all_patch_indices = []
        for i, patch_locations in enumerate(terrain_importer.flat_patches.values()):
            num_patch_locations = patch_locations.view(-1, 3).shape[0]
            all_patch_locations.append(patch_locations.view(-1, 3))
            all_patch_indices += [i] * num_patch_locations
        flat_patches_visualizer.visualize(torch.cat(all_patch_locations), marker_indices=all_patch_indices)

    if args_cli.show_keep_in_regions:
        _visualize_keep_in_regions(terrain_gen_cfg, terrain_importer)

    scene_entities = {"terrain": terrain_importer}
    return scene_entities, terrain_importer.env_origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, AssetBase], origins: torch.Tensor):
    """Run the simulation loop."""
    del entities, origins
    while simulation_app.is_running():
        sim.step()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[10.0, 30.0, 30.0], target=[3.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    sim.reset()
    print(f"[INFO]: Setup complete... previewing '{args_cli.bridge}'.")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
