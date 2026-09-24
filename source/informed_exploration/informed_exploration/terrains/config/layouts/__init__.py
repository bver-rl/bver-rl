# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named multi-spoke :class:`~..bridge.RadialLayoutCfg` terrains beyond the linear :data:`~..presets.PRESETS`."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.terrains import TerrainGeneratorCfg

from .dtmg import build_dtmg_layout, dtmg_cfg
from .dtsg import build_dtsg_layout, dtsg_cfg, dtsg_route_rects, dtsg_route_top_z, dtsg_sealed_cfg
from .nbde import build_nbde_layout, nbde_cfg
from .sgde import build_sgde_layout, sgde_cfg

LAYOUTS: dict[str, Callable[[], TerrainGeneratorCfg]] = {
    "sgde": sgde_cfg,
    "dtmg": dtmg_cfg,
    "nbde": nbde_cfg,
    "dtsg": dtsg_cfg,
    "dtsg_sealed": dtsg_sealed_cfg,
}
"""All named layout-terrain factories, keyed by name."""

__all__ = [
    "LAYOUTS",
    "build_dtmg_layout",
    "build_dtsg_layout",
    "build_nbde_layout",
    "build_sgde_layout",
    "dtmg_cfg",
    "dtsg_cfg",
    "dtsg_route_rects",
    "dtsg_route_top_z",
    "dtsg_sealed_cfg",
    "nbde_cfg",
    "sgde_cfg",
]
