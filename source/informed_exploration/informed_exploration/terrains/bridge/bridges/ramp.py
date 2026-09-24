# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A smooth A-frame ramp climbing to a ridge peak, then descending.

An off-center :attr:`peak_fraction` gives a steep side and a gentle side even between equal-height platforms.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class RampBridgeCfg(BridgeCfg):
    """Ramp/A-frame bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    peak_extra_height_range: tuple[float, float] = (0.5, 1.2)
    """Ridge height (in m) above ``max(z_entry, z_exit)``: low/easy -> high/hard."""

    peak_fraction: float = 0.3
    """Where the ridge sits along the span, in ``(0, 1)``; off-center makes one side steeper. Mirrored by `reverse`."""

    surface_noise_range: tuple[float, float] = (0.0, 0.0)
    """Height noise (in m, ``+-`` about the slope) roughening the walking surface: smooth/easy -> bumpy/hard.

    When non-zero the profile becomes one rough sheet (:func:`~..geometry.noisy_ramp_surface`) tapered
    to zero at the platform junctions."""

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

        fraction = (1.0 - self.peak_fraction) if self.reverse else self.peak_fraction
        fraction = float(np.clip(fraction, 0.05, 0.95))
        extra = lerp((d0 + d1) / 2.0, *self.peak_extra_height_range)
        z_peak = max(z0, z1) + extra
        split = length * fraction

        noise = lerp((d0 + d1) / 2.0, *self.surface_noise_range)
        if noise > 0.0:
            return geometry.noisy_ramp_surface([(0.0, z0), (split, z_peak), (length, z1)], width, noise, rng=rng)

        meshes = [geometry.ramp_wedge(0.0, split, z0, z_peak, width)]
        meshes.append(geometry.ramp_wedge(split, length, z_peak, z1, width))
        return [m for m in meshes if m is not None]
