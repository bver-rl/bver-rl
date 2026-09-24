# Vendored from the isaaclab-parkour extension (MIT).
# Copyright (c) 2025, BVER Team.
#
# Copied here so this package stands alone; behaviour is unchanged from upstream.

"""Climb-box sub-terrains: the box generator and the sub-terrain dicts the climb family uses."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Optional

import numpy as np
import scipy.spatial.transform as tf
import trimesh

from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.trimesh.utils import make_plane

from isaaclab.utils import configclass


def big_box_terrain(difficulty: float, cfg: BigBoxTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with one big box."""

    # set the difficulty
    box_height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    box_width = cfg.width_range[1] - difficulty * (cfg.width_range[1] - cfg.width_range[0])
    box_length = cfg.length_range[1] - difficulty * (cfg.length_range[1] - cfg.length_range[0])
    box_yaw = np.random.uniform(difficulty * cfg.yaw_range[0], difficulty * cfg.yaw_range[1])

    # roughness from the bool flags or optional probabilities
    do_rough_ground = bool(getattr(cfg, "rough_ground", False))
    if getattr(cfg, "rough_ground_prob", None) is not None:
        p = float(cfg.rough_ground_prob) * (0.5 + 0.5 * float(difficulty))
        do_rough_ground = do_rough_ground or (np.random.rand() < np.clip(p, 0.0, 1.0))

    do_rough_box = bool(getattr(cfg, "rough_box", False))
    if getattr(cfg, "rough_box_prob", None) is not None:
        p = float(cfg.rough_box_prob) * (0.5 + 0.5 * float(difficulty))
        do_rough_box = do_rough_box or (np.random.rand() < np.clip(p, 0.0, 1.0))

    # initialize the list of meshes
    meshes_list = list()

    # add ground plane, rough height field or flat
    if do_rough_ground:
        rough_ground_cfg = HfRandomUniformTerrainCfg(
            size=cfg.size,
            noise_range=(-0.05, 0.075),
            noise_step=0.005,
            slope_threshold=0.75,
        )
        rough_ground = random_uniform_terrain(difficulty, rough_ground_cfg)[0]
        meshes_list.extend(rough_ground)
    else:
        meshes_list.append(make_plane(cfg.size, 0.0, center_zero=False))

    # add the box with optional pitch/roll tilt
    pitch = np.random.uniform(cfg.pitch_range[0], cfg.pitch_range[1]) if hasattr(cfg, "pitch_range") else 0.0
    roll  = np.random.uniform(cfg.roll_range[0],  cfg.roll_range[1])  if hasattr(cfg, "roll_range")  else 0.0
    Rz = trimesh.transformations.rotation_matrix(box_yaw, [0, 0, 1])
    Ry = trimesh.transformations.rotation_matrix(pitch,   [0, 1, 0])
    Rx = trimesh.transformations.rotation_matrix(roll,    [1, 0, 0])
    pose = Rz @ Ry @ Rx @ np.eye(4)
    pose[0, -1] = cfg.size[0] / 2.0
    pose[1, -1] = cfg.size[1] / 2.0
    pose[2, -1] = box_height / 2.0

    box = trimesh.creation.box([box_length, box_width, box_height])
    box.apply_transform(pose)
    meshes_list.append(box)

    # make the box rough
    if do_rough_box:
        rough_box_cfg = HfRandomUniformTerrainCfg(
            size=(box_length, box_width),
            noise_range=(-0.05, 0.075),
            noise_step=0.005,
            slope_threshold=0.75,
        )
        rough_box = random_uniform_terrain(difficulty, rough_box_cfg)[0][0]
        pose[0, -1] = (cfg.size[0] - box_length * np.cos(box_yaw) + box_width * np.sin(box_yaw)) / 2.0
        pose[1, -1] = (cfg.size[0] - box_width * np.cos(box_yaw) - box_length * np.sin(box_yaw)) / 2.0
        pose[2, -1] = box_height
        rough_box.apply_transform(pose)
        meshes_list.append(rough_box)

    # define the origin of the terrain
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])

    return meshes_list, origin


@configclass
class BigBoxTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a mesh terrain with one box in the center."""

    function = big_box_terrain

    height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the box (in m)."""
    width_range: tuple[float, float] = MISSING
    """The minimum and maximum width of the box (in m)."""
    length_range: tuple[float, float] = MISSING
    """The minimum and maximum length of the box (in m)."""

    yaw_range: tuple[float, float] = MISSING
    """The minimum and maximum yaw of the box (in rad)."""
    pitch_range: tuple[float, float] = (0.0, 0.0)
    """The min and max pitch tilt (radians)."""
    roll_range: tuple[float, float] = (0.0, 0.0)
    """The min and max roll tilt (radians)."""

    rough_ground: bool = False
    """If True, always add uniform random-height noise to the surrounding ground."""
    rough_ground_prob: Optional[float] = None
    """If set, probability of adding rough ground (scaled by difficulty). Ignored if None."""

    rough_box: bool = False
    """If True, always add uniform random-height noise to the top of the box."""
    rough_box_prob: Optional[float] = None
    """If set, probability of adding rough noise to the box top (scaled by difficulty). Ignored if None."""


CLIMB_BOX_SUBTERRAINS_SINGLE = {
    "big_box_up": BigBoxTerrainCfg(
        proportion=1.0,
        height_range=(0.56, 0.56),
        width_range=(3.2, 3.2),
        length_range=(1.2, 1.2),
        yaw_range=(0.0, 0.0),
        rough_ground=False,
        rough_box=False,
        flat_patch_sampling={
            "init_pos": FlatPatchSamplingCfg(
                num_patches=1,
                patch_radius=0.35,
                x_range=(-3.2, -2.8),
                y_range=(-0.2, 0.2),
                z_range=(0.0, 0.05),
                max_height_diff=0.1,
            ),
            "target": FlatPatchSamplingCfg(
                num_patches=1,
                patch_radius=0.35,
                x_range=(-1e-4, 1e-4),
                y_range=(-1e-4, 1e-4),
                z_range=(0.05, 10.0),
                max_height_diff=10.0,
            ),
        },
    )
}

CLIMB_BOX_DOWN_SUBTERRAINS_SINGLE = {
    "big_box_down": BigBoxTerrainCfg(
        proportion=1.0,
        height_range=(0.56, 0.56),
        width_range=(3.2, 3.2),
        length_range=(1.2, 1.2),
        yaw_range=(0.0, 0.0),
        rough_ground=False,
        rough_box=False,
        flat_patch_sampling={
            "init_pos": FlatPatchSamplingCfg(
                num_patches=1,
                patch_radius=0.35,
                x_range=(-0.2, 0.2),
                y_range=(-0.2, 0.2),
                z_range=(0.05, 10.0),
                max_height_diff=10.0,
            ),
            "target": FlatPatchSamplingCfg(
                num_patches=1,
                patch_radius=0.35,
                x_range=(-3.0 - 1e-4, -3.0 + 1e-4),
                y_range=(-1e-4, 1e-4),
                z_range=(0.0, 0.05),
                max_height_diff=0.1,
            ),
        },
    )
}
