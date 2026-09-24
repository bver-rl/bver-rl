# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze grid layouts and the geometry derived from them, free of Isaac Sim imports.

The grid is the single source of truth: mesh, task-space bounds, wall keep-out and start/goal poses
all derive from it. Rows run along +x and columns along +y, matching ``maze_terrains.maze_terrain``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class MazeCell(enum.Enum):
    WALL = 0
    EMPTY = 1
    INIT = 2
    GOAL = 3

    def is_wall(self):
        return self == MazeCell.WALL

    def is_empty(self):
        return self == MazeCell.EMPTY or self == MazeCell.INIT or self == MazeCell.GOAL

    def is_init(self):
        return self == MazeCell.INIT

    def is_goal(self):
        return self == MazeCell.GOAL


W, o, I, G = MazeCell.WALL, MazeCell.EMPTY, MazeCell.INIT, MazeCell.GOAL


class EmptyMaze:
    name = "empty_maze"
    layout = [
        [W, W, W, W, W, W],
        [W, o, o, o, G, W],
        [W, o, o, o, o, W],
        [W, I, o, o, o, W],
        [W, W, W, W, W, W],
    ]


class UMaze:
    """5x5 U maze: rllab-curriculum's ``maze_id=0`` and mujoco-maze's ``UMaze``."""

    name = "u_maze"
    layout = [
        [W, W, W, W, W],
        [W, G, o, o, W],
        [W, W, W, o, W],
        [W, I, o, o, W],
        [W, W, W, W, W],
    ]


class GMaze:
    """7x7 spiral: rllab-curriculum's ``maze_id=11`` and mujoco-maze's ``SpiralMaze``."""

    name = "g_maze"
    layout = [
        [W, W, W, W, W, W, W],
        [W, o, o, o, o, G, W],
        [W, o, W, W, W, W, W],
        [W, o, W, I, o, o, W],
        [W, o, W, W, W, o, W],
        [W, o, o, o, o, o, W],
        [W, W, W, W, W, W, W],
    ]


class MediumMaze:
    """9x12 multi-route maze with dead ends; start and goal sit in opposite corners."""

    name = "medium_maze"
    layout = [
        [W, W, W, W, W, W, W, W, W, W, W, W],
        [W, o, o, W, o, o, o, W, o, o, G, W],
        [W, W, o, W, o, W, o, W, o, W, W, W],
        [W, o, o, W, o, W, o, o, o, o, o, W],
        [W, o, W, W, W, W, o, W, W, W, o, W],
        [W, o, o, o, o, o, o, W, o, o, o, W],
        [W, o, W, W, o, W, o, W, o, W, o, W],
        [W, I, o, o, o, W, o, o, o, o, o, W],
        [W, W, W, W, W, W, W, W, W, W, W, W],
    ]


class SerpentineMaze:
    """Two full-height barriers with gaps at opposite ends, forcing a single long detour route.

    The route first leads away from the goal direction, which is where an undirected explorer stalls.
    """

    name = "serpentine_maze"
    layout = [
        [W, W, W, W, W, W, W, W, W, W, W],
        [W, o, o, o, W, o, o, o, o, o, W],
        [W, o, I, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, o, o, W],
        [W, o, o, o, W, o, W, o, G, o, W],
        [W, o, o, o, o, o, W, o, o, o, W],
        [W, W, W, W, W, W, W, W, W, W, W],
    ]


class ScatterMaze:
    """13x13 of scattered blocks with many routes of similar length and no forced detour.

    The complement of the serpentine: tests whether a curriculum still helps when the task never funnels.
    """

    name = "scatter_maze"
    layout = [
        [W, W, W, W, W, W, W, W, W, W, W, W, W],
        [W, W, o, o, o, o, o, W, o, o, o, o, W],
        [W, o, o, o, W, W, o, o, o, o, W, o, W],
        [W, o, W, o, o, o, o, W, W, o, W, o, W],
        [W, o, o, o, I, o, W, o, o, o, W, o, W],
        [W, W, W, o, o, o, o, o, W, o, W, o, W],
        [W, o, W, o, W, W, o, W, o, o, o, o, W],
        [W, o, o, o, o, o, W, W, o, W, o, o, W],
        [W, o, o, W, W, o, o, o, G, o, o, W, W],
        [W, W, o, o, o, W, o, o, o, o, W, o, W],
        [W, W, W, o, o, o, o, o, W, o, W, o, W],
        [W, o, o, o, o, W, o, W, W, o, o, o, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W],
    ]


MAZE_LAYOUTS = {
    layout.name: layout
    for layout in (EmptyMaze, UMaze, GMaze, MediumMaze, SerpentineMaze, ScatterMaze)
}
"""Registry the terrain generator and the geometry helper both resolve ``maze_name`` through."""


@dataclass(frozen=True)
class MazeGeometry:
    """Everything a maze env cfg needs about a layout, in env-relative metres."""

    name: str
    layout: list[list[MazeCell]]
    cell_size: float
    ball_radius: float

    @property
    def rows(self) -> int:
        return len(self.layout)

    @property
    def cols(self) -> int:
        return len(self.layout[0])

    @property
    def terrain_size(self) -> tuple[float, float]:
        """Sub-terrain size the generator must use for the grid to fill its tile exactly."""
        return (self.rows * self.cell_size, self.cols * self.cell_size)

    def cell_center(self, i: int, j: int) -> tuple[float, float]:
        height, width = self.terrain_size
        return ((i + 0.5) * self.cell_size - height / 2.0, (j + 0.5) * self.cell_size - width / 2.0)

    @property
    def x_bounds(self) -> tuple[float, float]:
        """Interior x range, border walls excluded and inset by the ball radius."""
        height = self.terrain_size[0]
        return (self.cell_size - height / 2.0 + self.ball_radius, (self.rows - 1) * self.cell_size - height / 2.0 - self.ball_radius)

    @property
    def y_bounds(self) -> tuple[float, float]:
        width = self.terrain_size[1]
        return (self.cell_size - width / 2.0 + self.ball_radius, (self.cols - 1) * self.cell_size - width / 2.0 - self.ball_radius)

    @property
    def wall_keep_out(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """One ``((x_lo, x_hi), (y_lo, y_hi))`` keep-out box per interior wall cell, grown by the ball radius.

        Border cells are omitted since the bounds already exclude them.
        """
        boxes = []
        for i in range(1, self.rows - 1):
            for j in range(1, self.cols - 1):
                if not self.layout[i][j].is_wall():
                    continue
                cx, cy = self.cell_center(i, j)
                half = self.cell_size / 2.0 + self.ball_radius
                boxes.append(((cx - half, cx + half), (cy - half, cy + half)))
        return boxes

    def _unique_cell(self, predicate, what: str) -> tuple[float, float]:
        cells = [
            (i, j) for i in range(self.rows) for j in range(self.cols) if predicate(self.layout[i][j])
        ]
        if len(cells) != 1:
            raise ValueError(f"maze '{self.name}' must mark exactly one {what} cell, found {len(cells)}")
        return self.cell_center(*cells[0])

    @property
    def init_xy(self) -> tuple[float, float]:
        return self._unique_cell(MazeCell.is_init, "INIT")

    @property
    def cell_half_extent(self) -> float:
        """Half-width of the region a ball centre can occupy inside one cell."""
        return self.cell_size / 2.0 - self.ball_radius

    @property
    def init_region(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((x_lo, x_hi), (y_lo, y_hi))`` spanning the INIT cell, inset by the ball radius.

        The initial-state distribution every arm's episodes start from.
        """
        cx, cy = self.init_xy
        e = self.cell_half_extent
        return ((cx - e, cx + e), (cy - e, cy + e))

    @property
    def goal_xy(self) -> tuple[float, float]:
        return self._unique_cell(MazeCell.is_goal, "GOAL")


def maze_geometry(name: str, cell_size: float = 4.0, ball_radius: float = 0.5) -> MazeGeometry:
    if name not in MAZE_LAYOUTS:
        raise ValueError(f"Maze name {name} not recognized. Expected one of {sorted(MAZE_LAYOUTS)}.")
    return MazeGeometry(
        name=name, layout=MAZE_LAYOUTS[name].layout, cell_size=cell_size, ball_radius=ball_radius
    )
