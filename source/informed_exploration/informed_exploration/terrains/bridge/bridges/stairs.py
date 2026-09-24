# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A staircase climbing from the entry platform to a peak landing, then descending to the exit platform.

Peak height is the only difficulty knob; per-side step counts set riser height, so fewer steps make a side steeper.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class StairsBridgeCfg(BridgeCfg):
    """Staircase bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``.

    Per-side difficulty scales only affect the mean peak height; use the step counts for asymmetry.
    """

    entry_num_steps: int = 2
    """Number of risers between the entry platform and the peak (>= 1); fewer means taller and harder."""

    exit_num_steps: int = 3
    """Number of risers between the peak and the exit platform (>= 1); fewer means taller and harder."""

    peak_extra_height_range: tuple[float, float] = (0.5, 1.0)
    """Extra height (in m) of the peak landing above ``max(z_entry, z_exit)``: low/easy -> high/hard."""

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

        # structural values: no difficulty interpolation, just the `reverse` swap onto (x0, xlength)
        n0, n1 = self.entry_num_steps, self.exit_num_steps
        if self.reverse:
            n0, n1 = n1, n0
        n0, n1 = max(1, n0), max(1, n1)
        extra = lerp((d0 + d1) / 2.0, *self.peak_extra_height_range)
        z_peak = max(z0, z1) + extra

        # Tread tops: n0 risers up, n1 down; the last riser lands on the exit platform, so it gets no tread.
        rise, drop = (z_peak - z0) / n0, (z_peak - z1) / n1
        tops = [z0 + i * rise for i in range(1, n0 + 1)] + [z_peak - j * drop for j in range(1, n1)]
        if len(tops) == 1:
            tops = tops * 2  # a lone landing (n0 == n1 == 1): split it so both halves are covered

        # Under `reverse` the peak landing goes to the descending half; clamped so neither half is empty.
        split = min(max(n0 - 1 if self.reverse else n0, 1), len(tops) - 1)

        half = length / 2.0
        meshes = geometry.tread_run(0.0, half, tops[:split], width)
        meshes += geometry.tread_run(half, length, tops[split:], width)
        return meshes
