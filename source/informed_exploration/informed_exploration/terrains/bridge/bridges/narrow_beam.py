# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A single straight balance beam along the traverse axis; harder means narrower and wobblier."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .. import geometry
from ..interface import BridgeCfg, lerp


@configclass
class NarrowBeamBridgeCfg(BridgeCfg):
    """Balance-beam bridge. ``*_range`` tuples are ``(value at difficulty 0, value at difficulty 1)``."""

    beam_width_range: tuple[float, float] = (1.5, 0.5)
    """Beam width (in m): wide/easy -> narrow/hard."""

    beam_thickness: float = 0.06
    """Beam slab thickness (in m)."""

    wobble_range: tuple[float, float] = (0.0, 0.1)
    """Per-segment lateral offset randomness (in m): none/easy -> large/hard."""

    num_segments: int = 14
    """Number of segments the beam is built from (affects wobble granularity, not visible seams)."""

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

        seg_len = length / self.num_segments
        meshes: list[trimesh.Trimesh] = []
        for i in range(self.num_segments):
            x0 = i * seg_len
            x_frac = (x0 + seg_len / 2.0) / length if length > 0 else 0.0
            d_local = lerp(x_frac, d0, d1)
            beam_w = lerp(d_local, *self.beam_width_range)
            wobble = lerp(d_local, *self.wobble_range)
            y_off = float(rng.uniform(-wobble, wobble)) if wobble > 0.0 else 0.0
            z = lerp(x_frac, z0, z1)
            beam_bottom = z - self.beam_thickness
            center = (x0 + seg_len / 2.0, y_off, z - self.beam_thickness / 2.0)
            # slight x-overlap so adjacent segments (with different y offsets) don't leave a gap
            meshes.append(geometry.box(seg_len * 1.05, beam_w, self.beam_thickness, center=center))

            # same-footprint support under each segment so the thin plank never floats
            ground = -geometry.GROUND_EMBED_DEPTH
            support_center = (x0 + seg_len / 2.0, y_off, (beam_bottom + ground) / 2.0)
            meshes.append(geometry.box(seg_len * 1.05, beam_w, beam_bottom - ground, center=support_center))

        return meshes
