# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A field of discrete stepping stones. Harder = smaller stones, wider gaps, more height noise."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class SteppingStonesBridgeCfg(BridgeCfg):
    """Stepping-stone bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    stone_size_range: tuple[float, float] = (0.55, 0.30)
    """Stone footprint side length (in m): large/easy -> small/hard."""

    gap_range: tuple[float, float] = (0.10, 0.15)
    """Gap between adjacent stones (in m): small/easy -> wide/hard."""

    height_noise_range: tuple[float, float] = (0.0, 0.10)
    """Per-stone vertical offset randomness (in m): none/easy -> large/hard."""

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

        meshes: list[trimesh.Trimesh] = []
        x = 0.0
        rng = rng if rng is not None else np.random.default_rng()
        while x < length:
            x_frac = x / length if length > 0 else 0.0
            d_local = lerp(x_frac, d0, d1)
            stone_size = lerp(d_local, *self.stone_size_range)
            gap = lerp(d_local, *self.gap_range)
            noise = lerp(d_local, *self.height_noise_range)

            # same pitch in x and y so the gap between adjacent stones is equal in both directions
            pitch = stone_size + gap
            n_rows = max(1, int(width / pitch))
            row_span = n_rows * pitch - gap  # n stones + (n-1) gaps, centered in the width
            y_start = -row_span / 2.0
            for row in range(n_rows):
                y = y_start + row * pitch + stone_size / 2.0
                z_base = lerp(x_frac, z0, z1)
                dz = float(rng.uniform(-noise, noise)) if noise > 0.0 else 0.0
                # pillar always reaches true ground; only the top surface (walking height) varies
                top = z_base + dz
                bottom = -geometry.GROUND_EMBED_DEPTH
                center = (x + stone_size / 2.0, y, (top + bottom) / 2.0)
                meshes.append(geometry.box(stone_size, stone_size, top - bottom, center=center))

            x += stone_size + gap

        return meshes
