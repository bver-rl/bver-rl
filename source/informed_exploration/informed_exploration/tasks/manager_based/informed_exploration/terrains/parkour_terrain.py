# Vendored from the isaaclab-parkour extension (MIT).
# Copyright (c) 2025, BVER Team.
#
# Copied here so this package stands alone; behaviour is unchanged from upstream.

"""The parkour scene's default terrain generator, vendored with its sub-terrain dicts."""

from __future__ import annotations

from isaaclab.terrains import TerrainGeneratorCfg


PARKOUR_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=15.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,  # does not influence rough surfaces in mesh terrains
    vertical_scale=0.005,  # does not influence rough surfaces in mesh terrains
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
)
