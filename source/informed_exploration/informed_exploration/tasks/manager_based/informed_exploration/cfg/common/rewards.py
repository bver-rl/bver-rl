# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action and reward cfgs shared by climb, the terrain-derived families and their sparse arms."""

from isaaclab.utils import configclass

from ..common.env_cfg import *


@configclass
class ClimbBoxActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ClimbBoxRewardsCfg:
    # task rewards
    tracking_pos_sparse = RewTerm(
        func=mdp.tracking_pos,
        weight=250.0,
        params={"command_name": "base_position", "duration": 4.0, "distance_3d": True, "sparse": True},
    )
    tracking_pos = RewTerm(
        func=mdp.exp_tracking_pos,
        weight=10.0,
        params={"command_name": "base_position", "sigma": 0.6, "distance_3d": True},
    )

    # contact and termination penalties
    termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": ["illegal_force", "bad_orientation"]},
    )
    collision = RewTerm(
        func=mdp.custom_contact_forces,  # imp use standard collision penalty
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(THIGH|SHANK)"), "threshold": 1.0},
    )
    collision_base = RewTerm(
        func=mdp.custom_contact_forces,  # imp use standard collision penalty
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=mdp.squared_contact_forces,
        weight=-2.5e-6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 700.0},
    )

    # shaping penalties
    torque_limits = RewTerm(
        func=mdp.torque_limits_old,
        weight=-0.02,
        params={"limit": 85.0},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.25,
        params={"soft_ratio": 0.9},
    )
    base_acc = RewTerm(
        func=mdp.body_acc_weighted_l2_old_term,
        weight=-0.5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "linear_weight": 1.0, "angular_weight": 0.02},
    )
    feet_acc = RewTerm(
        func=mdp.body_lin_acc_l2_old_term,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT")},
    )
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)


@configclass
class SparseClimbBoxRewardsCfg:
    """Purely sparse climb MDP: goal occupancy plus the failure penalty, nothing else."""

    tracking_pos_sparse = RewTerm(
        func=mdp.tracking_pos,
        weight=250.0,
        params={
            "command_name": "base_position",
            "duration": 4.0,
            "distance_3d": True,
            "sparse": True,
        },
    )
    # contact and termination penalties
    termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": ["illegal_force", "bad_orientation"]},
    )


@configclass
class PenSparseClimbBoxRewardsCfg(SparseClimbBoxRewardsCfg):
    """The sparse climb MDP plus :class:`ClimbBoxRewardsCfg`'s non-contact shaping penalties."""

    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.25,
        params={"soft_ratio": 0.9},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
    )
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    feet_acc = RewTerm(
        func=mdp.body_lin_acc_l2_old_term,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*FOOT")},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1.0e-2,
    )
