# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solve the joint configurations of a Franka Hanoi layout and write its poses file.

Start, goal and home poses are solved with kinematic differential IK, then held under PD control and re-measured.
The Hanoi environments refuse to load until this file exists.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Solve a Hanoi layout's joint poses with differential IK.")
parser.add_argument("--layout", type=str, default="p12", help="Hanoi layout name.")
parser.add_argument(
    "--output", type=str, default=None, help="Poses file to write. Defaults to the layout's file under data/hanoi."
)
parser.add_argument("--iterations", type=int, default=500, help="Maximum IK iterations.")
parser.add_argument("--pos_tol", type=float, default=5e-4, help="Hand position tolerance, in metres.")
parser.add_argument("--rot_tol", type=float, default=1e-2, help="Hand orientation tolerance, in radians.")
parser.add_argument(
    "--hold_steps", type=int, default=120, help="Physics steps each solution is held for before re-measuring."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms  # noqa: E402

from informed_exploration.tasks.manager_based.informed_exploration.cfg.hanoi.hanoi_env_cfg import (  # noqa: E402
    HANOI_HAND_BODY,
    HANOI_POSE_ERROR_TOL,
    HANOI_ROBOT_CFG_NAME,
    HANOI_SIM_DT,
    HanoiSceneCfg,
    apply_hanoi_geometry,
    hanoi_camera_pose,
    hanoi_poses_relpath,
    resolve_workspace_path,
)
from informed_exploration.tasks.manager_based.informed_exploration.mdp import link_point_pos_w  # noqa: E402
from informed_exploration.tasks.manager_based.informed_exploration.terrains.hanoi_geometry import (  # noqa: E402
    PANDA_ARM_JOINT_NAMES,
    PANDA_FINGER_JOINT_EXPR,
    hanoi_geometry,
)

POSE_NAMES = ("start", "anchor", "home")

IK_SEEDS = {
    "p12": {
        "start": (1.7501, -1.0869, -2.0093, -2.4149, 1.7994, 2.4923, -0.9401),
        "anchor": (1.3938, -1.1236, -2.1094, -2.101, 0.892, 2.2658, -0.1491),
        "home": (1.1921, -0.8868, -1.577, -2.4704, 1.342, 2.5001, -0.1973),
    },
}
"""IK seed per layout and pose, all on the grip branch that connects the three solved poses."""


def _write_arm_joints(robot, arm: SceneEntityCfg, joint_pos: torch.Tensor) -> torch.Tensor:
    """Teleport the arm joints to ``joint_pos`` at rest and make that the PD target; returns the full joint vector."""
    full = robot.data.default_joint_pos.clone()
    full[:, arm.joint_ids] = joint_pos
    robot.write_joint_state_to_sim(full, torch.zeros_like(full))
    robot.set_joint_position_target(full)
    return full


def _rounded(values) -> list[float]:
    return [round(float(v), 6) for v in values]


def main():
    geom = hanoi_geometry(args_cli.layout)
    if args_cli.layout not in IK_SEEDS:
        raise ValueError(f"No IK seeds for Hanoi layout '{args_cli.layout}'. Add them to IK_SEEDS.")
    seeds = IK_SEEDS[args_cli.layout]

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=HANOI_SIM_DT, device=args_cli.device))
    eye, lookat = hanoi_camera_pose(geom)
    sim.set_camera_view(eye=eye, target=lookat)

    scene_cfg = HanoiSceneCfg(num_envs=len(POSE_NAMES), env_spacing=2.5)
    apply_hanoi_geometry(scene_cfg, geom)
    seed_pose = dict(zip(PANDA_ARM_JOINT_NAMES, seeds["home"]))
    seed_pose[PANDA_FINGER_JOINT_EXPR] = geom.finger_joint_pos
    scene_cfg.robot.init_state.joint_pos = seed_pose
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    arm = SceneEntityCfg(
        "robot", joint_names=list(PANDA_ARM_JOINT_NAMES), body_names=[HANOI_HAND_BODY], preserve_order=True
    )
    arm.resolve(scene)
    # a fixed-base robot's root body is not in the Jacobian, so body indices shift down by one
    jacobian_index = arm.body_ids[0] - 1 if robot.is_fixed_base else arm.body_ids[0]
    device = sim.device
    ring_offset = torch.tensor(geom.ring_center_hand, device=device)
    dt = sim.get_physics_dt()
    num = len(POSE_NAMES)

    ring_targets = torch.tensor([geom.start_ring_xyz, geom.goal_ring_xyz, geom.home_ring_xyz], device=device)
    hand_pos_targets = torch.tensor([geom.hand_pos_for_ring(p) for p in ring_targets.tolist()], device=device)
    hand_quat_targets = torch.tensor(geom.hand_quat_wxyz, device=device).repeat(num, 1)

    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=num,
        device=device,
    )
    controller.reset()
    controller.set_command(torch.cat([hand_pos_targets, hand_quat_targets], dim=-1))

    limits = robot.data.soft_joint_pos_limits[0, arm.joint_ids]
    joint_pos = torch.tensor([seeds[name] for name in POSE_NAMES], device=device)
    _write_arm_joints(robot, arm, joint_pos)
    robot.reset()
    scene.write_data_to_sim()
    sim.forward()
    scene.update(dt)

    iterations = torch.full((num,), args_cli.iterations, dtype=torch.long)
    for it in range(args_cli.iterations + 1):
        root_pose_w = robot.data.root_pose_w
        hand_pose_w = robot.data.body_pose_w[:, arm.body_ids[0]]
        hand_pos_b, hand_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], hand_pose_w[:, :3], hand_pose_w[:, 3:7]
        )
        pos_err = torch.linalg.norm(hand_pos_b - hand_pos_targets, dim=-1)
        rot_err = quat_error_magnitude(hand_quat_b, hand_quat_targets)
        converged = ((pos_err < args_cli.pos_tol) & (rot_err < args_cli.rot_tol)).cpu()
        iterations[converged & (iterations == args_cli.iterations)] = it
        if bool(converged.all()) or it == args_cli.iterations:
            break
        jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, arm.joint_ids]
        joint_pos = controller.compute(hand_pos_b, hand_quat_b, jacobian, joint_pos)
        joint_pos = torch.clamp(joint_pos, limits[:, 0], limits[:, 1])
        _write_arm_joints(robot, arm, joint_pos)
        scene.write_data_to_sim()
        # kinematics only: the solve teleports through the pegs, which must not push back
        sim.forward()
        scene.update(dt)
        if not args_cli.headless:
            sim.render()

    kinematic_ring = link_point_pos_w(robot, arm.body_ids[0], ring_offset) - scene.env_origins

    solved = joint_pos.clone()
    full = _write_arm_joints(robot, arm, solved)
    for _ in range(args_cli.hold_steps):
        robot.set_joint_position_target(full)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    held_ring = link_point_pos_w(robot, arm.body_ids[0], ring_offset) - scene.env_origins
    held_joints = robot.data.joint_pos[:, arm.joint_ids]
    ring_error = torch.linalg.norm(held_ring - ring_targets, dim=-1)
    hold_drift = (held_joints - solved).abs().max(dim=-1).values
    limit_margin = torch.minimum(solved - limits[:, 0], limits[:, 1] - solved).min(dim=-1).values

    data = {
        "layout": args_cli.layout,
        "robot_cfg": HANOI_ROBOT_CFG_NAME,
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "joint_limits": [_rounded(row) for row in limits.tolist()],
        "hand_quat_wxyz": _rounded(geom.hand_quat_wxyz),
        "poses": {},
    }
    header = f"{'pose':<8}{'converged':>10}{'iters':>7}{'hand_mm':>9}{'rot_rad':>9}{'ring_mm':>9}{'drift_rad':>11}{'margin_rad':>12}"
    print("\n" + header)
    print("-" * len(header))
    failed = []
    for i, name in enumerate(POSE_NAMES):
        is_converged = bool(pos_err[i] < args_cli.pos_tol) and bool(rot_err[i] < args_cli.rot_tol)
        if not is_converged or float(ring_error[i]) > HANOI_POSE_ERROR_TOL:
            failed.append(name)
        print(
            f"{name:<8}{str(is_converged):>10}{int(iterations[i]):>7}{1000 * float(pos_err[i]):>9.2f}"
            f"{float(rot_err[i]):>9.4f}{1000 * float(ring_error[i]):>9.2f}{float(hold_drift[i]):>11.4f}"
            f"{float(limit_margin[i]):>12.3f}"
        )
        data["poses"][name] = {
            "joint_pos": _rounded(solved[i].tolist()),
            "ring_xyz_target": _rounded(ring_targets[i].tolist()),
            "ring_xyz_kinematic": _rounded(kinematic_ring[i].tolist()),
            "ring_xyz_measured": _rounded(held_ring[i].tolist()),
            "error_m": round(float(ring_error[i]), 6),
            "hold_drift_rad": round(float(hold_drift[i]), 6),
            "min_limit_margin_rad": round(float(limit_margin[i]), 6),
            "iterations": int(iterations[i]),
            "converged": is_converged,
        }

    output = resolve_workspace_path(args_cli.output or hanoi_poses_relpath(args_cli.layout))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        f.write(f"# Written by informed-exploration/scripts/helpers/hanoi_solve_poses.py --layout {args_cli.layout}\n")
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"\n[INFO] Wrote {output}")
    if failed:
        print(
            f"[WARN] {', '.join(failed)}: not converged or ring error above {1000 * HANOI_POSE_ERROR_TOL:.1f} mm. "
            "The Hanoi environments will refuse this file; adjust the geometry or the solver settings."
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
