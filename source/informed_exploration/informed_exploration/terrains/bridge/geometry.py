# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic trimesh primitive helpers used by every bridge.

Unlike ``make_box``/``make_cylinder`` these apply no random orientation; stochastic helpers take an explicit ``rng``.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.trimesh.utils import make_border, make_plane

__all__ = [
    "GROUND_EMBED_DEPTH",
    "box",
    "border",
    "cylinder",
    "noisy_ramp_surface",
    "plane",
    "ramp_wedge",
    "rough_patch",
    "terraced_ramp",
    "tread_run",
    "wall_box",
]

GROUND_EMBED_DEPTH = 0.1
"""Depth (in m) below the tile ground that every support structure extends to, avoiding seams and z-fighting."""


def box(length: float, width: float, height: float, center: tuple[float, float, float], yaw: float = 0.0) -> trimesh.Trimesh:
    """A box of the given dimensions, centered at ``center`` with an optional yaw (rotation about z)."""
    transform = trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0])
    transform[0:3, -1] = np.asarray(center)
    return trimesh.creation.box((length, width, height), transform=transform)


def wall_box(center_xy: tuple[float, float], size_xy: tuple[float, float], height: float) -> trimesh.Trimesh:
    """A solid wall box from true ground up to ``height`` above the tile floor, shared by all walls."""
    bottom_z = -GROUND_EMBED_DEPTH
    return box(size_xy[0], size_xy[1], height - bottom_z, center=(center_xy[0], center_xy[1], (height + bottom_z) / 2.0))


def border(
    outer_size: tuple[float, float], inner_size: tuple[float, float], height: float, center: tuple[float, float, float]
) -> list[trimesh.Trimesh]:
    """Rectangular border frame (4 wall segments) with a rectangular hole. See :func:`make_border`."""
    return make_border(outer_size, inner_size, height, center)


def cylinder(radius: float, height: float, center: tuple[float, float, float], sections: int = 16) -> trimesh.Trimesh:
    """A cylinder of the given radius/height, centered at ``center``, no random orientation."""
    transform = np.eye(4)
    transform[0:3, -1] = np.asarray(center)
    return trimesh.creation.cylinder(radius, height, sections=sections, transform=transform)


def plane(size: tuple[float, float], height: float, center_xy: tuple[float, float] = (0.0, 0.0)) -> trimesh.Trimesh:
    """A flat quad of the given size at ``height``, spanning ``center_xy +- size/2``."""
    mesh = make_plane(size, height, center_zero=True)
    mesh.apply_translation((center_xy[0], center_xy[1], 0.0))
    return mesh


def rough_patch(
    size: tuple[float, float],
    difficulty: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_range: tuple[float, float] = (-0.03, 0.05),
    noise_step: float = 0.005,
    slope_threshold: float = 0.75,
    rng: np.random.Generator | None = None,
) -> list[trimesh.Trimesh]:
    """A height-field patch of uniform random noise, centered at ``center`` (``center[2]`` offsets it vertically).

    With ``rng`` given, the global numpy state is seeded from it for the call and restored afterwards.
    """
    cfg = HfRandomUniformTerrainCfg(
        size=size, noise_range=noise_range, noise_step=noise_step, slope_threshold=slope_threshold
    )
    if rng is None:
        meshes, _ = random_uniform_terrain(difficulty, cfg)
    else:
        state = np.random.get_state()
        np.random.seed(int(rng.integers(2**31)))
        try:
            meshes, _ = random_uniform_terrain(difficulty, cfg)
        finally:
            np.random.set_state(state)
    offset = (center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, center[2])
    for mesh in meshes:
        mesh.apply_translation(offset)
    return meshes


def noisy_ramp_surface(
    profile: list[tuple[float, float]],
    width: float,
    noise_amplitude: float,
    noise_step: float = 0.005,
    resolution: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[trimesh.Trimesh]:
    """A rough sheet following the piecewise-linear ``(x, z)`` ``profile`` with ``+-noise_amplitude`` noise.

    Noise is zero on the end rows so the sheet meets the platforms without a lip; solid backing sits
    at the deepest possible dip.
    """
    px = np.array([p[0] for p in profile])
    pz = np.array([p[1] for p in profile])
    rng = rng if rng is not None else np.random.default_rng()

    n_x = max(2, int(np.ceil((px[-1] - px[0]) / resolution)) + 1)
    n_y = max(2, int(np.ceil(width / resolution)) + 1)
    xs = np.linspace(px[0], px[-1], n_x)
    ys = np.linspace(-width / 2.0, width / 2.0, n_y)

    noise = rng.uniform(-noise_amplitude, noise_amplitude, (n_x, n_y))
    noise = np.round(noise / noise_step) * noise_step
    noise[0, :] = 0.0
    noise[-1, :] = 0.0
    zs = np.interp(xs, px, pz)[:, None] + noise

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack([grid_x.ravel(), grid_y.ravel(), zs.ravel()], axis=1)
    quads = np.array([
        [i * n_y + j, (i + 1) * n_y + j, (i + 1) * n_y + j + 1, i * n_y + j + 1]
        for i in range(n_x - 1)
        for j in range(n_y - 1)
    ])
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    sheet = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    sheet.fix_normals()

    meshes: list[trimesh.Trimesh] = [sheet]
    for (xa, za), (xb, zb) in zip(profile[:-1], profile[1:]):
        backing = ramp_wedge(xa, xb, za - noise_amplitude, zb - noise_amplitude, width)
        if backing is not None:
            meshes.append(backing)
    return meshes


def ramp_wedge(x0: float, x1: float, z0: float, z1: float, width: float) -> trimesh.Trimesh | None:
    """A smooth sloped solid from ``(x0, z0)`` to ``(x1, z1)``, ``width`` wide, reaching true ground.

    Returns None if ``x1 <= x0``.
    """
    if x1 <= x0:
        return None
    zg = -GROUND_EMBED_DEPTH
    hw = width / 2.0
    vertices = np.array(
        [
            [x0, -hw, z0],
            [x0, hw, z0],
            [x1, -hw, z1],
            [x1, hw, z1],
            [x0, -hw, zg],
            [x0, hw, zg],
            [x1, -hw, zg],
            [x1, hw, zg],
        ]
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # sloped top
            [4, 6, 7], [4, 7, 5],  # bottom
            [0, 4, 5], [0, 5, 1],  # x0 (entry) side
            [2, 3, 7], [2, 7, 6],  # x1 (exit) side
            [0, 2, 6], [0, 6, 4],  # -y side
            [1, 5, 7], [1, 7, 3],  # +y side
        ]
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.fix_normals()
    return mesh


def terraced_ramp(
    x0: float, x1: float, z0: float, z1: float, width: float, num_steps: int
) -> list[trimesh.Trimesh]:
    """A run of ``num_steps`` grounded, full-width tread boxes climbing or descending from ``z0`` to ``z1``.

    For a continuous slope use :func:`ramp_wedge` instead.
    """
    num_steps = max(1, int(num_steps))
    step_rise = (z1 - z0) / num_steps
    return tread_run(x0, x1, [z0 + step_rise * (i + 1) for i in range(num_steps)], width)


def tread_run(x0: float, x1: float, tops: list[float], width: float) -> list[trimesh.Trimesh]:
    """Equal-depth, full-width grounded tread boxes tiling ``[x0, x1]``, the ``i``-th topped at ``tops[i]``."""
    if x1 <= x0 or not tops:
        return []
    step_len = (x1 - x0) / len(tops)
    bottom_z = -GROUND_EMBED_DEPTH

    meshes: list[trimesh.Trimesh] = []
    for i, top_z in enumerate(tops):
        xs = x0 + i * step_len
        xe = xs + step_len
        center = ((xs + xe) / 2.0, 0.0, (top_z + bottom_z) / 2.0)
        meshes.append(box(xe - xs, width, top_z - bottom_z, center=center))
    return meshes
