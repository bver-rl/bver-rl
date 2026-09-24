"""Utilities for generating, snapshotting and validating feasible start states for the quadruped climb-box task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .task_space.parkour_task_space import ParkourTaskSpace


def resolve_foot_indices(env: "ManagerBasedRLEnv") -> tuple[list[int], list[int]]:
    """Resolve foot body indices in the articulation and contact-sensor views, which index bodies independently."""
    robot_cfg = SceneEntityCfg("robot", body_names=".*FOOT")
    robot_cfg.resolve(env.scene)
    contact_cfg = SceneEntityCfg("contact_forces", body_names=".*FOOT")
    contact_cfg.resolve(env.scene)
    return list(robot_cfg.body_ids), list(contact_cfg.body_ids)


def compute_body_up_z(quat: torch.Tensor) -> torch.Tensor:
    """Return the world-frame z-component of the body's local +z axis from a wxyz quaternion."""
    x = quat[..., 1]
    y = quat[..., 2]
    return 1.0 - 2.0 * (x * x + y * y)


def compute_foot_contact_force_mag(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    contact_sensor_foot_ids: list[int],
) -> torch.Tensor:
    """Net contact force magnitude per foot. Shape: (len(env_ids), 4)."""
    contact_sensor = env.scene.sensors["contact_forces"]
    foot_forces = contact_sensor.data.net_forces_w[env_ids][:, contact_sensor_foot_ids]
    return torch.linalg.norm(foot_forces, dim=-1)


def feasibility_predicate(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    bad_term_mask: torch.Tensor,
    contact_sensor_foot_ids: list[int],
    box_top_z: float,
    min_base_z: float = 0.55,
    max_base_z_above_box: float = 0.65,
    upright_threshold: float = 0.7,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Per-env mask of states that survived without a bad termination, have a foot in contact, an env-local
    base z inside the allowed window and an upright body."""
    robot = env.scene["robot"]

    survived = ~bad_term_mask[env_ids]

    foot_force_mag = compute_foot_contact_force_mag(env, env_ids, contact_sensor_foot_ids)
    has_contact = (foot_force_mag > contact_threshold).any(dim=-1)

    base_z_w = robot.data.root_pos_w[env_ids, 2]
    env_origins_z = env.scene.env_origins[env_ids, 2]
    base_z_local = base_z_w - env_origins_z
    z_in_range = (base_z_local > min_base_z) & (base_z_local < box_top_z + max_base_z_above_box)

    upright = compute_body_up_z(robot.data.root_quat_w[env_ids]) > upright_threshold

    return survived & has_contact & z_in_range & upright


def snapshot_state(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    task_space: "ParkourTaskSpace",
) -> torch.Tensor:
    """Snapshot the robot's full state for `env_ids` as a task tensor replayable via `set_tasks`.

    Base position is env-local and joint states are offsets from defaults. Replay assumes the default root
    quaternion is identity.
    """
    robot = env.scene["robot"]
    env_origins = env.scene.env_origins

    root_pos_local = robot.data.root_pos_w[env_ids] - env_origins[env_ids]
    root_quat = robot.data.root_quat_w[env_ids]
    root_lin_vel = robot.data.root_lin_vel_w[env_ids]
    root_ang_vel = robot.data.root_ang_vel_w[env_ids]
    joint_pos_rel = robot.data.joint_pos[env_ids] - robot.data.default_joint_pos[env_ids]
    joint_vel_rel = robot.data.joint_vel[env_ids] - robot.data.default_joint_vel[env_ids]

    obs = torch.cat(
        [root_pos_local, root_quat, root_lin_vel, root_ang_vel, joint_pos_rel, joint_vel_rel],
        dim=-1,
    ).unsqueeze(
        1
    )  # (N, 1, 37)

    return task_space.obs_to_task(obs).squeeze(1)  # (N, 18, 2)


def get_contact_pattern(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    articulation_foot_ids: list[int],
    contact_sensor_foot_ids: list[int],
    contact_threshold: float = 1.0,
    surface_tolerance: float = 0.10,
) -> torch.Tensor:
    """Classify each foot as air, ground or box, in the order the foot-name pattern resolves to.

    Any contact above ``surface_tolerance`` counts as box, including the box's vertical faces.
    """
    robot = env.scene["robot"]
    env_origins = env.scene.env_origins

    foot_pos_local = robot.data.body_pos_w[env_ids][:, articulation_foot_ids] - env_origins[env_ids].unsqueeze(1)
    foot_z = foot_pos_local[..., 2]  # (N, 4)

    foot_force_mag = compute_foot_contact_force_mag(env, env_ids, contact_sensor_foot_ids)
    has_contact = foot_force_mag > contact_threshold

    on_ground = has_contact & (foot_z < surface_tolerance)
    on_box = has_contact & ~on_ground

    pattern = torch.zeros_like(foot_z, dtype=torch.long)
    pattern[on_ground] = 1
    pattern[on_box] = 2
    return pattern
