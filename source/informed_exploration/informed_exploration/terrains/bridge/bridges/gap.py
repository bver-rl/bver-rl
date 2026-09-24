# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A single gap the robot must leap across; harder means wider.

An optional lip at each apron's gap edge makes one jump direction harder independently of platform heights.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


def _apron(x0: float, x1: float, top_z: float, width: float) -> list[trimesh.Trimesh]:
    """A solid apron slab reaching down to true ground, regardless of how high ``top_z`` sits."""
    bottom_z = -geometry.GROUND_EMBED_DEPTH
    center = ((x0 + x1) / 2.0, 0.0, (top_z + bottom_z) / 2.0)
    return [geometry.box(x1 - x0, width, top_z - bottom_z, center=center)]


@configclass
class GapBridgeCfg(BridgeCfg):
    """Gap/jump bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    gap_width_range: tuple[float, float] = (0.3, 1.0)
    """Width of the gap (in m): narrow/easy -> wide/hard."""

    entry_lip_range: tuple[float, float] = (0.0, 0.3)
    """Extra height (in m) of the entry apron right at the gap edge, on top of ``z_entry``."""

    exit_lip_range: tuple[float, float] = (0.0, 0.0)
    """Extra height (in m) of the exit apron right at the gap edge, on top of ``z_exit``."""

    def build(
        self,
        difficulty: float,
        length: float,
        width: float,
        z_entry: float,
        z_exit: float,
        rng: np.random.Generator | None = None,
    ) -> list[trimesh.Trimesh]:
        z0, z1 = self.resolve_sides(z_entry, z_exit)
        d0, d1 = self.side_difficulties(difficulty)
        lip0, lip1 = self.resolve_side_value(difficulty, self.entry_lip_range, self.exit_lip_range)

        gap_w = lerp((d0 + d1) / 2.0, *self.gap_width_range)
        gap_w = min(gap_w, max(0.1, length - 0.2))

        mid = length / 2.0
        entry_edge = mid - gap_w / 2.0
        exit_edge = mid + gap_w / 2.0

        meshes = _apron(0.0, entry_edge, z0 + lip0, width)
        meshes += _apron(exit_edge, length, z1 + lip1, width)
        return meshes
