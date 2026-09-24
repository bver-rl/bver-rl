# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common contract every 'bridge' terrain (the span between two platforms) must implement."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass


def lerp(x_frac: float, v0: float, v1: float) -> float:
    """Linearly interpolate between ``v0`` (at ``x_frac=0``) and ``v1`` (at ``x_frac=1``), clipped."""
    return float(v0 + (v1 - v0) * np.clip(x_frac, 0.0, 1.0))


@configclass
class BridgeCfg:
    """Base configuration for a pluggable bridge spanning two platforms.

    Meshes fill the local span frame: ``x`` in ``[0, length]`` from the entry platform, ``y`` in
    ``[-width/2, width/2]``, and ``z`` in absolute tile-local coordinates.
    """

    width: float = 2.5
    """Width of the bridge span (in m), perpendicular to the traverse axis."""

    length_range: tuple[float, float] = (1.5, 3.5)
    """Difficulty-interpolated length (in m) of the active hard feature region within the span.

    Single-feature bridges may ignore this and use the full span length.
    """

    reverse: bool = False
    """If True, swap which physical end (x=0 vs x=length) is treated as entry vs exit."""

    entry_difficulty_scale: float = 1.0
    """Multiplier applied to the global difficulty for the entry-side geometry."""

    exit_difficulty_scale: float = 1.0
    """Multiplier applied to the global difficulty for the exit-side geometry."""

    def side_difficulties(self, difficulty: float) -> tuple[float, float]:
        """Return ``(difficulty_at_x0, difficulty_at_xlength)``, honoring :attr:`reverse`.

        For distinct per-side ranges use :meth:`resolve_side_value` instead.
        """
        d_entry = float(np.clip(difficulty * self.entry_difficulty_scale, 0.0, 1.0))
        d_exit = float(np.clip(difficulty * self.exit_difficulty_scale, 0.0, 1.0))
        return (d_exit, d_entry) if self.reverse else (d_entry, d_exit)

    def resolve_sides(self, z_entry: float, z_exit: float) -> tuple[float, float]:
        """Return ``(z_at_x0, z_at_xlength)``, honoring :attr:`reverse`."""
        return (z_exit, z_entry) if self.reverse else (z_entry, z_exit)

    def resolve_side_value(
        self, difficulty: float, entry_range: tuple[float, float], exit_range: tuple[float, float]
    ) -> tuple[float, float]:
        """Interpolate distinct entry and exit ranges by their own side's difficulty, placed at ``(x0, xlength)``.

        The swap for :attr:`reverse` happens after interpolation, so each value stays paired with its range.
        """
        d_entry = float(np.clip(difficulty * self.entry_difficulty_scale, 0.0, 1.0))
        d_exit = float(np.clip(difficulty * self.exit_difficulty_scale, 0.0, 1.0))
        v_entry = lerp(d_entry, *entry_range)
        v_exit = lerp(d_exit, *exit_range)
        return (v_exit, v_entry) if self.reverse else (v_entry, v_exit)

    def resolve_active_length(self, difficulty: float, span_length: float) -> float:
        """Interpolate :attr:`length_range` by difficulty, clipped to the available span."""
        lo, hi = self.length_range
        active = lo + difficulty * (hi - lo)
        return float(np.clip(active, 0.0, span_length))

    def build(
        self,
        difficulty: float,
        length: float,
        width: float,
        z_entry: float,
        z_exit: float,
        rng: np.random.Generator | None = None,
    ) -> list[trimesh.Trimesh]:
        """Build meshes filling the local span frame.

        Args:
            difficulty: Global difficulty in ``[0, 1]``.
            length: Usable span length (in m) between the two platform edges.
            width: Width of the span (in m).
            z_entry: Absolute tile-local height (m) of the entry-side platform's top surface.
            z_exit: Absolute tile-local height (m) of the exit-side platform's top surface.
            rng: Generator for stochastic geometry; ``None`` falls back to an entropy-seeded one.

        Returns:
            Meshes positioned in the local span frame.
        """
        raise NotImplementedError
