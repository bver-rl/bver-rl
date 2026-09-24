# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named terrains built from real-world scans, one :class:`ScanSpec` per scan with its measured poses.

Prepared height fields are always metric; :attr:`ScanSpec.xy_scale` and :attr:`ScanSpec.z_scale` are for
deliberate enlargement, never unit correction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from isaaclab.terrains import TerrainGeneratorCfg

from ..scan import (
    ScanHeightfieldTerrainCfg,
    content_fingerprint,
    load_scan_grid,
    pinned_scan_cfg,
    scan_path,
)

SUBTERRAIN_NAME = "rock_scan"
"""Sub-terrain key shared by every scan's single-sub-terrain generator cfg."""

TILE_APRON = 2.0
"""Clearance (m) between the scan footprint and the tile edge when :attr:`ScanSpec.tile_size` is derived.

It must hold the apron ring and the fence."""


@dataclass(frozen=True)
class ScanSpec:
    """One real-world scan and the task laid out on it.

    Frozen and free of derived state, so arms of the same scan can share it; the height field is only read
    in :meth:`sub_terrain`.
    """

    name: str
    """CamelCase identity, used in task ids and class names (``Parkour-Scan-RockyAscent-BVER-v0``)."""

    snake: str
    """snake_case identity, used in log dirs, wandb projects, pool and yaml file names."""

    stem: str
    """File stem of the raw scan and its prepared artifacts under ``terrains/mesh_scans/``."""

    start_xy_obj: tuple[float, float] | None = None
    """Measured start position in the scan's own frame. ``None`` makes every task on this scan refuse to build."""

    start_yaw_obj: float | None = None
    """Measured start heading in the scan's own frame. ``None`` faces the goal."""

    goal_xy_obj: tuple[float, float] | None = None
    """Measured goal position in the scan's own frame."""

    extra_starts_obj: tuple[tuple[float, float], ...] = ()
    """Further start positions in the scan's own frame, each paired with the canonical goal in the pairs eval.

    Eval only; training sees the measured pair alone."""

    extra_goals_obj: tuple[tuple[float, float], ...] = ()
    """Further goal positions in the scan's own frame, each paired with the canonical start in the pairs eval."""

    xy_scale: float = 1.0
    z_scale: float = 1.0
    """Deliberate enlargement, not unit correction. Keep them equal to preserve every slope angle."""

    tile_size: tuple[float, float] | None = None
    """Tile the scan is centred in. ``None`` derives it from the scaled footprint plus :data:`TILE_APRON`."""

    footprint_rect_obj: tuple[float, float, float, float] | None = None
    """Optional crop ``(x_lo, x_hi, y_lo, y_hi)`` in the scan's frame, to drop extrapolated corners of an
    irregular outline."""

    patch_max_height_diff: float | None = None
    """Maximum height spread (m) of a flat patch. ``None`` keeps the terrain default.

    Raise it only if a scan cannot find patches; the config warns with the measured pass rate first."""

    goal_z_offset: float = 0.0
    """Shift (m) of the goal's commanded height. Negative suits rough ground, where the robot settles lower."""

    episode_time_s: float | None = None
    """Seconds per episode. ``None`` keeps the family default.

    Raise it when the traverse leaves no time parked on the goal; all time-related settings follow it, and
    the recorded trajectory buffer grows in proportion."""

    viewer_eye: tuple[float, float, float] | None = None
    viewer_lookat: tuple[float, float, float] | None = None
    """Viewport camera position and target in world coordinates, for the PLAY and EVAL bindings.

    ``None`` leaves the family default. They do not follow pose or scale changes."""

    viewer_resolution: tuple[int, int] | None = None
    """Viewport resolution. ``None`` leaves the family default."""

    notes: str = ""
    """What the scan is, in one line, for the previewer and the generated docstrings."""

    @property
    def npz(self) -> str:
        """Prepared height field, written by ``scripts/terrains/prepare_scan.py``."""
        return f"{self.stem}_hf.npz"

    @property
    def obj(self) -> str:
        """Raw scan, as exported."""
        return f"{self.stem}.obj"

    @property
    def measured(self) -> bool:
        """Whether the start and goal positions are known; the heading has a default."""
        return None not in (self.start_xy_obj, self.goal_xy_obj)

    def resolved_start_yaw(self) -> float:
        """The start heading, defaulting to facing the goal."""
        if self.start_yaw_obj is not None:
            return self.start_yaw_obj
        return math.atan2(self.goal_xy_obj[1] - self.start_xy_obj[1], self.goal_xy_obj[0] - self.start_xy_obj[0])

    def pairs(self) -> list[tuple[str, str, tuple[float, float], tuple[float, float]]]:
        """Return the pairs-eval ``(label, kind, start_obj, goal_obj)`` rows, canonical pair first.

        Extra starts go against the canonical goal and extra goals against the canonical start, not the
        cross product, so each failure is attributable to one end.
        """
        assert self.measured, f"scan {self.snake!r} has no measured pair; pairs() has nothing to build on"
        start, goal = self.start_xy_obj, self.goal_xy_obj
        assert start is not None and goal is not None  # narrowed by `measured`
        rows = [("canonical", "canonical", start, goal)]
        rows += [(f"s{i + 1}->g", "start_star", xy, goal) for i, xy in enumerate(self.extra_starts_obj)]
        rows += [(f"s->g{j + 1}", "goal_star", start, xy) for j, xy in enumerate(self.extra_goals_obj)]
        return rows

    @property
    def prepared(self) -> bool:
        return scan_path(self.npz).exists()

    def require_measured(self) -> None:
        """Raise unless this scan is prepared and measured, naming what is missing."""
        if not self.prepared:
            raise ValueError(
                f"scan {self.snake!r} has no prepared height field at {scan_path(self.npz)}."
                f" Run scripts/terrains/prepare_scan.py on {self.obj} first"
                f' (VS Code: "Parkour IE Scan {self.name}: Prepare Scan").'
            )
        missing = [field for field in ("start_xy_obj", "goal_xy_obj") if getattr(self, field) is None]
        if missing:
            raise ValueError(
                f"scan {self.snake!r} has no measured {', '.join(missing)}. Read the pose off"
                f" {self.stem}_preview.png and fill it into SCAN_SPECS in terrains/config/scans.py;"
                " the rotation, the surface heights and the whole task space follow from it."
            )

    def resolved_tile_size(self) -> tuple[float, float]:
        """Return :attr:`tile_size`, or derive it from the prepared height field when ``None``."""
        if self.tile_size is not None:
            return self.tile_size
        path = scan_path(self.npz)
        grid = load_scan_grid(str(path), content_fingerprint(path))
        span_x = (grid.heights.shape[0] - 1) * grid.resolution * self.xy_scale
        span_y = (grid.heights.shape[1] - 1) * grid.resolution * self.xy_scale
        return (
            float(math.ceil(span_x + 2.0 * TILE_APRON)),
            float(math.ceil(span_y + 2.0 * TILE_APRON)),
        )

    def sub_terrain(self, **overrides) -> ScanHeightfieldTerrainCfg:
        """Build a fresh sub-terrain cfg.

        Args:
            overrides: Field overrides, used by the previewer for placeholder poses.
        """
        fields: dict[str, Any] = dict(
            npz_path=self.npz,
            size=self.resolved_tile_size(),
            start_xy_obj=self.start_xy_obj,
            start_yaw_obj=self.resolved_start_yaw(),
            goal_xy_obj=self.goal_xy_obj,
            extra_starts_obj=self.extra_starts_obj,
            extra_goals_obj=self.extra_goals_obj,
            footprint_rect_obj=self.footprint_rect_obj,
            xy_scale=self.xy_scale,
            z_scale=self.z_scale,
            goal_z_offset=self.goal_z_offset,
        )
        if self.patch_max_height_diff is not None:
            fields["patch_max_height_diff"] = self.patch_max_height_diff
        fields.update(overrides)
        return ScanHeightfieldTerrainCfg(**fields)

    def generator_cfg(self, num_rows: int = 1, num_cols: int = 1, difficulty: float = 0.5) -> TerrainGeneratorCfg:
        """Single-sub-terrain generator cfg, for the env bindings and the previewers."""
        return pinned_scan_cfg(
            self.sub_terrain(), SUBTERRAIN_NAME, difficulty=difficulty, num_rows=num_rows, num_cols=num_cols
        )

    def with_placeholder_poses(self) -> "ScanSpec":
        """Return a copy with placeholder poses so an unmeasured scan renders. Previewer only."""
        if self.measured:
            return self
        path = scan_path(self.npz)
        grid = load_scan_grid(str(path), content_fingerprint(path))
        span_x = (grid.heights.shape[0] - 1) * grid.resolution
        span_y = (grid.heights.shape[1] - 1) * grid.resolution
        centre = (grid.x0 + span_x / 2.0, grid.y0 + span_y / 2.0)
        start = (grid.x0 + span_x / 6.0, centre[1])
        return replace(self, start_xy_obj=start, goal_xy_obj=centre)  # heading follows the goal


# The scans. Poses are in each scan's metric frame; a spec without poses previews but refuses to train.

BOULDER_SMALL_ROCKS = ScanSpec(
    name="BoulderSmallRocks",
    snake="boulder_small_rocks",
    stem="boulder_with_small_rocks",
    # provisional poses read off the preview: start on flat ground east of the boulder, goal on its plateau
    start_xy_obj=(1.6353, 1.0986),
    goal_xy_obj=(0.0, 0.1),
    # start_yaw_obj omitted: facing the goal, which is what these provisional poses assume anyway
    # doubled so there is ground to cross; the climb then exceeds the trained box height, lower z_scale first
    xy_scale=2.0,
    z_scale=2.0,
    tile_size=(12.0, 12.0),
    # PROVISIONAL extra poses from propose_scan_poses.py; verify with "Check Start/Goal Poses"
    extra_starts_obj=((0.77, -1.0),),
    # canonical pair measured on the published poses: 3.8 m out, +0.79 m of rise
    extra_goals_obj=(
        # (0.3, 1.2),
        # (1.42, 0.09),
        # (0.77, -1.31),
    ),
    # framed by hand, the only tuned camera of the four
    viewer_eye=(-9.5, 6.3, 4.3),
    viewer_lookat=(-0.0, -0.5, 0.1),
    notes="a boulder rising out of small rocky ground; climb from the low ground to the top plateau",
)

FREESTANDING_BOULDER = ScanSpec(
    name="FreestandingBoulder",
    snake="freestanding_boulder",
    stem="freestanding_boulder",
    start_xy_obj=(1.3, -1.3),
    goal_xy_obj=(0.0, 0.0),
    xy_scale=2.0,
    z_scale=1.5,
    # PROVISIONAL; starts at evenly spread bearings since the level ground allows approach from any side
    extra_starts_obj=(
        # (1.14, 1.26),
        # (1.2, 0.01),
        # (0.01, -1.44),
        # (-1.4, -1.44),
    ),
    # published-pose figures; the canonical pair is 3.7 m out, +1.56 m of rise
    extra_goals_obj=(
        # (-0.6, 1.36),
        # (1.24, 1.31),
        # (0.3, -0.3),
    ),
    viewer_eye=(6.0, 9.0, 2.5),
    viewer_lookat=(-0.0, -0.0, 0.4),
    notes="a single boulder on level ground, approachable from any side",
)

ROCKY_ASCENT = ScanSpec(
    name="RockyAscent",
    snake="rocky_ascent",
    stem="rocky_ascent",
    episode_time_s=12.0,
    start_xy_obj=(-2.8, -0.0),
    goal_xy_obj=(3.2, 0.6),
    xy_scale=0.75,
    z_scale=0.8,
    goal_z_offset=0.0,
    # PROVISIONAL and weakest set: little flat ground, so starts are filled in by distance, not bearing
    extra_starts_obj=(
        # (2.83, -0.46),  # 1.6 m out, bearing -100 deg
        # (-0.07, 1.69),  # 2.5 m out, bearing +178 deg
        # (-1.07, -0.56),  # 3.6 m out, bearing -153 deg
        # (1.13, -0.41),  # 2.2 m out, bearing -136 deg
    ),
    # published-pose figures; the canonical pair is 4.7 m out, +1.33 m of rise
    extra_goals_obj=(
        # (3.63, -0.31),  # 4.8 m out, +1.38 m: the top of the ascent, just above the canonical goal
        # (1.63, 1.94),  # 3.6 m out, +1.13 m
        # (2.28, -0.16),  # 3.8 m out, +1.10 m
        # (0.43, 0.94),  # 2.5 m out, +0.71 m: a way-point short of the canonical goal
    ),
    viewer_eye=(-7.5, 4.5, 2.1),
    viewer_lookat=(-1.0, 0.0, 1.3),
    # Cover video
    # viewer_eye=(-11.0, 7.4, 1.7),
    # viewer_lookat=(-2.2, -0.5, 2.2),
    notes="a continuous rocky slope with no summit; the goal is a point up the ascent",
)

SMALL_ROCKS = ScanSpec(
    name="SmallRocks",
    snake="small_rocks",
    stem="small_freestanding_rocks",
    start_xy_obj=(2.2, 0.0),
    goal_xy_obj=(-0.85, 0.12),
    xy_scale=1.5,
    z_scale=1.5,
    goal_z_offset=-0.1,
    # PROVISIONAL; a footing problem, so the goals barely rise except the tallest rock
    extra_starts_obj=(
        (0.48, 1.44),  # 2.5 m out, bearing  +38 deg
        (-2.07, 1.39),  # 2.1 m out, bearing +131 deg
        (-1.67, -1.11),  # 2.2 m out, bearing -124 deg
        (0.92, -1.56),  # 2.1 m out, bearing  -81 deg
    ),
    # published-pose figures; the canonical pair is 4.6 m out, +0.76 m of rise
    extra_goals_obj=(
        (0.43, -0.01),  # 2.7 m out, +0.78 m: on top of the tallest rock, the one real climb here
        # (1.83, -1.26),  # 2.0 m out, +0.03 m
        # (0.58, 1.34),  # 3.2 m out, -0.04 m
        # (-1.32, 1.74),
    ),
    # starting point, not yet framed by hand
    viewer_eye=(-5.3, -8.8, 6.5),
    viewer_lookat=(-0.7, -0.7, 0.5),
    notes="a field of small freestanding rocks on level ground; a footing problem, not a climb",
)

SCAN_SPECS: dict[str, ScanSpec] = {
    spec.snake: spec for spec in (BOULDER_SMALL_ROCKS, FREESTANDING_BOULDER, ROCKY_ASCENT, SMALL_ROCKS)
}
"""Every scan, keyed by its snake_case name. The task families are generated from this."""


def _preview_cfg(spec: ScanSpec) -> Callable[[], TerrainGeneratorCfg]:
    """Generator-cfg factory for the previewers, tolerant of a scan that has not been measured."""

    def build() -> TerrainGeneratorCfg:
        return spec.with_placeholder_poses().generator_cfg()

    return build


SCANS: dict[str, Callable[[], TerrainGeneratorCfg]] = {spec.stem: _preview_cfg(spec) for spec in SCAN_SPECS.values()}
"""All scan terrains keyed by file stem, for the previewers. Unmeasured scans get placeholder poses."""

__all__ = [
    "BOULDER_SMALL_ROCKS",
    "FREESTANDING_BOULDER",
    "ROCKY_ASCENT",
    "SCANS",
    "SCAN_SPECS",
    "SMALL_ROCKS",
    "SUBTERRAIN_NAME",
    "ScanSpec",
]
