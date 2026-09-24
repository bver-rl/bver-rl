# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A field of cylindrical stepping poles filling the span; harder means smaller poles and wider gaps.

Poles are placed by dart throwing until saturation, so the typical hop distance is set by ``gap_range``.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class PillarsBridgeCfg(BridgeCfg):
    """Stepping-pole bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    pole_radius_range: tuple[float, float] = (0.35, 0.2)
    """Pole radius (in m): large/easy -> small/hard."""

    gap_range: tuple[float, float] = (0.1, 0.25)
    """Minimum edge-to-edge clearance between poles (in m): small/easy -> wide/hard."""

    pole_extra_height_range: tuple[float, float] = (0.0, 0.0)
    """Extra pole-top height (in m) above the ``z_entry -> z_exit`` line: low/easy -> high/hard.

    Raise it on floor-level layouts, where flush tops would degenerate into plain floor.
    """

    max_placement_failures: int = 250
    """Consecutive rejected placement candidates before the field counts as saturated."""

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
        rng = rng if rng is not None else np.random.default_rng()

        # dart throwing until `max_placement_failures` candidates in a row fail the clearance test
        placed: list[tuple[float, float, float, float]] = []  # (x, y, radius, gap)
        failures = 0
        while failures < self.max_placement_failures:
            x = float(rng.uniform(0.0, length))
            x_frac = x / length if length > 0 else 0.0
            d_local = lerp(x_frac, d0, d1)
            radius = lerp(d_local, *self.pole_radius_range)
            gap = lerp(d_local, *self.gap_range)

            # clamp the candidate fully inside the span footprint (skip spans too small for a pole)
            if length < 2.0 * radius or width < 2.0 * radius:
                break
            x = float(np.clip(x, radius, length - radius))
            y = float(rng.uniform(-width / 2.0 + radius, width / 2.0 - radius))

            ok = all(
                (x - px) ** 2 + (y - py) ** 2 >= (radius + pr + max(gap, pg)) ** 2
                for px, py, pr, pg in placed
            )
            if not ok:
                failures += 1
                continue
            failures = 0
            placed.append((x, y, radius, gap))

        meshes: list[trimesh.Trimesh] = []
        bottom = -geometry.GROUND_EMBED_DEPTH
        for x, y, radius, _ in placed:
            x_frac = x / length if length > 0 else 0.0
            d_local = lerp(x_frac, d0, d1)
            top = lerp(x_frac, z0, z1) + lerp(d_local, *self.pole_extra_height_range)
            center = (x, y, (top + bottom) / 2.0)
            meshes.append(geometry.cylinder(radius, top - bottom, center=center))

        return meshes
