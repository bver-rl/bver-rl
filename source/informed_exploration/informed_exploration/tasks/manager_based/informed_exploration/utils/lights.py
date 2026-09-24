# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A rectangular area light config, which Isaac Lab's spawners do not ship.

``spawn_light`` is generic over its cfg, so a config class is all that is needed.
"""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.lights.lights_cfg import LightCfg
from isaaclab.utils import configclass


@configclass
class RectLightCfg(LightCfg):
    """A light emitting from one face of a rectangle, along its local -Z.

    See `USDLux RectLight <https://openusd.org/dev/api/class_usd_lux_rect_light.html>`_.
    """

    prim_type = "RectLight"

    width: float = 1.0
    """Width of the rectangle, along its local x (in m)."""

    height: float = 1.0
    """Height of the rectangle, along its local y (in m)."""


def floor_light_cfg(
    size: tuple[float, float],
    height: float,
    intensity: float,
    prim_path: str = "/World/light",
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> AssetBaseCfg:
    """A rect light spanning ``size``, lying ``height`` metres above the floor and facing it.

    Deliberately not normalized, so ``intensity`` is a radiance and floor brightness does not depend
    on the layout size.
    """
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=RectLightCfg(
            width=size[0], height=size[1], intensity=intensity, color=color, normalize=False
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
