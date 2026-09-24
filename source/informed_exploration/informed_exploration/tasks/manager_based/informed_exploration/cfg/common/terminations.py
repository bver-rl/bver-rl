# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination cfgs shared across task families.

``fell_off_platform`` keeps its name because reward ``term_keys`` and logged metrics refer to it.
"""

from isaaclab.utils import configclass

from ..common.env_cfg import *


@configclass
class ClimbBoxTerminationsCfg(TerminationsCfg):
    illegal_contact = None
    illegal_force_feet = None
    illegal_force = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*FOOT).*"), "threshold": 5000.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi * 0.5},
    )


@configclass
class ClimbBoxRCTerminationsCfg(ClimbBoxTerminationsCfg):
    """ClimbBox terminations extended with the Brownian-motion timeout for random walkers."""

    brownian_motion_time_out = DoneTerm(func=mdp.brownian_motion_time_out, time_out=True)
