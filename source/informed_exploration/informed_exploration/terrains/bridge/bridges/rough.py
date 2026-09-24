# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A patch of randomly bumpy ground; harder means larger noise amplitude.

The noise sheet has no volume, so a solid backing block grounds it.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class RoughTerrainBridgeCfg(BridgeCfg):
    """Rough-ground bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    noise_amplitude_range: tuple[float, float] = (0.01, 0.09)
    """Peak-to-peak height noise (in m): smooth/easy -> bumpy/hard. Averaged over both sides' difficulties."""

    noise_step: float = 0.005
    """Discretization step of the noise height-field (in m)."""

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
        d_avg = (d0 + d1) / 2.0
        amplitude = lerp(d_avg, *self.noise_amplitude_range)

        z_mid = (z0 + z1) / 2.0
        meshes = geometry.rough_patch(
            size=(length, width),
            difficulty=1.0,
            center=(length / 2.0, 0.0, z_mid),
            noise_range=(-amplitude, amplitude),
            noise_step=self.noise_step,
            rng=rng,
        )

        # backing tops out at the lowest possible dip so it never pokes through the walking surface
        backing_top = z_mid - amplitude
        backing_bottom = -geometry.GROUND_EMBED_DEPTH
        backing_center = (length / 2.0, 0.0, (backing_top + backing_bottom) / 2.0)
        meshes.append(geometry.box(length, width, backing_top - backing_bottom, center=backing_center))
        return meshes
