# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout builders and difficulty-calibrated `TerrainGeneratorCfg` presets and factories."""

from .factory import (
    DEFAULT_TILE_SIZE,
    all_bridges_cfg,
    grid_layout,
    linear_layout,
    pinned_layout_cfg,
    single_type_cfg,
    star_layout,
)
from .layouts import LAYOUTS, build_dtmg_layout, build_sgde_layout, dtmg_cfg, sgde_cfg
from .presets import PRESETS
from .scans import SCAN_SPECS, SCANS, ScanSpec
