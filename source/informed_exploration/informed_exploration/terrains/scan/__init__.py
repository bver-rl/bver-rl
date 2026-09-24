# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrains built from real-world surface scans (see :mod:`.heightfield_scan`)."""

from .heightfield_scan import (
    ScanHeightfieldTerrainCfg,
    content_fingerprint,
    load_scan_grid,
    pinned_scan_cfg,
    resolve_scan_path,
    scan_path,
    scan_heightfield_terrain,
)
