# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two side walls forming a grounded channel between the platforms.

Different entry and exit openings give a funnel that is wide on one approach and narrow on the other.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class WallBridgeCfg(BridgeCfg):
    """Wall/funnel bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    wall_height: float = 1.0
    """Height of the side walls (in m)."""

    wall_thickness: float = 0.08
    """Thickness of each wall segment (in m)."""

    entry_opening_range: tuple[float, float] = (2.0, 2.0)
    """Channel opening width (in m) at the entry side."""

    exit_opening_range: tuple[float, float] = (2.0, 0.5)
    """Channel opening width (in m) at the exit side: wide/easy -> narrow/hard."""

    num_segments: int = 10
    """Number of segments each side wall is built from."""

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
        entry_open, exit_open = self.resolve_side_value(difficulty, self.entry_opening_range, self.exit_opening_range)
        max_open = max(0.3, width - 2.0 * self.wall_thickness)

        # grounded walking floor over the whole span, a single tread flat-topped at z1
        meshes: list[trimesh.Trimesh] = geometry.terraced_ramp(0.0, length, z0, z1, width=width, num_steps=1)

        ground = -geometry.GROUND_EMBED_DEPTH
        seg_len = length / self.num_segments
        for i in range(self.num_segments):
            x0 = i * seg_len
            x_frac = (x0 + seg_len / 2.0) / length if length > 0 else 0.0
            opening = min(lerp(x_frac, entry_open, exit_open), max_open)
            half_open = opening / 2.0
            z = lerp(x_frac, z0, z1)
            wall_top = z + self.wall_height
            wall_center_z = (wall_top + ground) / 2.0

            for sign in (1.0, -1.0):
                y = sign * (half_open + self.wall_thickness / 2.0)
                meshes.append(
                    geometry.box(
                        seg_len * 1.05, self.wall_thickness, wall_top - ground, center=(x0 + seg_len / 2.0, y, wall_center_z)
                    )
                )

        return meshes
