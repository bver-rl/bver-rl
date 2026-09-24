# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A horizontal bar the robot must crouch under or climb over.

Clearance ramps between entry and exit values, so one approach can be easier than the other. Builds
its own grounded walking floor so raised platforms need no detour down to the ground plane.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class ObstacleBridgeCfg(BridgeCfg):
    """Crouch-under / climb-over bridge. ``*_range`` = ``(value at difficulty 0, value at difficulty 1)``."""

    bar_thickness: float = 0.1
    """Thickness of the overhead bar (in m)."""

    entry_clearance_range: tuple[float, float] = (1.0, 0.75)
    """Gap (in m) between local ground and the bar's underside, entry side: open/easy -> tight/hard."""

    exit_clearance_range: tuple[float, float] = (0.5, 0.5)
    """Gap (in m) between local ground and the bar's underside, exit side: open/easy -> tight/hard."""

    num_segments: int = 10
    """Number of segments the bar underside ramp is built from."""

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
        entry_clearance, exit_clearance = self.resolve_side_value(
            difficulty, self.entry_clearance_range, self.exit_clearance_range
        )

        # grounded walking floor over the whole span, a single tread flat-topped at z1
        meshes: list[trimesh.Trimesh] = geometry.terraced_ramp(0.0, length, z0, z1, width=width, num_steps=1)

        seg_len = length / self.num_segments
        for i in range(self.num_segments):
            x0 = i * seg_len
            x_frac = (x0 + seg_len / 2.0) / length if length > 0 else 0.0
            clearance = lerp(x_frac, entry_clearance, exit_clearance)
            ground_z = lerp(x_frac, z0, z1)
            bar_bottom = ground_z + clearance
            center = (x0 + seg_len / 2.0, 0.0, bar_bottom + self.bar_thickness / 2.0)
            meshes.append(geometry.box(seg_len * 1.05, width, self.bar_thickness, center=center))

        return meshes
