# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drop robots on a scan's published start, goal and extra poses and report whether they stand there.

Half the envs spawn at each pose and settle under zero action. Pass ``--canonical_only`` to skip the extra poses.
"""

from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser(description="Drop and settle robots on a scan's start and goal poses.")
parser.add_argument("--task", type=str, required=True, help="Any arm of the scan to check.")
parser.add_argument("--envs_per_pose", type=int, default=4, help="Robots dropped on each pose.")
parser.add_argument("--settle_steps", type=int, default=150, help="Zero-action steps to settle for.")
parser.add_argument(
    "--drop_height",
    type=float,
    default=0.05,
    help="Extra height above the standing pose to drop from. Small on purpose: this checks footing,"
    " not a fall.",
)
parser.add_argument(
    "--jitter",
    type=float,
    default=None,
    help="Half-width of the uniform xy spread around each pose. Defaults to the initial-state"
    " window's, so the start is sampled the way training samples it. 0 pins every robot to the pose.",
)
parser.add_argument("--seed", type=int, default=0, help="Seeds the jitter.")
parser.add_argument("--hold_steps", type=int, default=0, help="Extra steps to keep the viewer open after settling.")
parser.add_argument(
    "--canonical_only",
    action="store_true",
    default=False,
    help="Check the measured start and goal only, skipping the pairs eval's extra poses.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    # `hydra_task_config` resolves the agent cfg too, even though nothing here trains.
    help="Name of the RL agent configuration entry point.",
)

from isaaclab.app import AppLauncher  # noqa: E402

import os
import sys

# cli_args.py lives with the entry points in scripts/rsl_rl/; put that directory on the path (one-way dependency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "rsl_rl"))

import cli_args  # isort: skip  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import informed_exploration.tasks  # noqa: F401, E402
from informed_exploration.tasks.manager_based.informed_exploration.curriculum.feasibility import (  # noqa: E402
    compute_body_up_z,
    compute_foot_contact_force_mag,
    resolve_foot_indices,
)

CANONICAL_POSES = ("start", "goal")
"""The two poses every scan publishes; order fixes which envs get which."""


def _collect_poses(sub_terrain, canonical_only: bool) -> dict:
    """Return the poses to check, canonical first so a truncated run still covers the measured pair."""
    poses = {name: sub_terrain.goal_poses[name] for name in CANONICAL_POSES}
    if not canonical_only:
        poses.update(sub_terrain.extra_poses or {})
    return poses


def _pose_tasks(env, poses, names, per_pose: int, drop_height: float, jitter: float, seed: int) -> torch.Tensor:
    """Build one task row per env, ``per_pose`` of them at each pose.

    Rows use the flat (18, 2) task layout; everything but the base position is zero, i.e. default stance at rest.
    """
    from informed_exploration.tasks.manager_based.informed_exploration.curriculum.task_space.parkour_task_space_cfg import (
        height_offset,
    )

    device = env.device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = poses["start"]
    rows = []
    for name in names:
        pose = poses[name]
        # env-relative: the start pose is the env origin
        centre = torch.tensor(
            [pose[0] - start[0], pose[1] - start[1], pose[2] + height_offset + drop_height],
            device=device,
        )
        offsets = torch.zeros((per_pose, 3), device=device)
        if jitter > 0.0:
            offsets[:, :2] = (torch.rand((per_pose, 2), generator=generator) * 2.0 - 1.0).to(device) * jitter
        rows.append(centre.unsqueeze(0) + offsets)

    task = torch.zeros((per_pose * len(names), 18, 2), device=device)
    task[:, :3, 0] = torch.cat(rows, dim=0)
    return task


def _report(env, env_ids, labels, names, bad_term_mask, foot_ids, commanded) -> None:
    """One line per robot, then a verdict per pose."""
    robot = env.scene["robot"]
    base = robot.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    # body +z projected on world +z, the same helper the feasibility predicate uses
    upright = compute_body_up_z(robot.data.root_quat_w[env_ids])
    forces = compute_foot_contact_force_mag(env, env_ids, foot_ids)
    contacts = (forces > 1.0).sum(dim=1)
    drift = torch.linalg.norm(base[:, :2] - commanded[:, :2], dim=1)
    sank = commanded[:, 2] - base[:, 2]

    width = max(len(name) for name in names)
    print(
        f"\n{'env':>4s} {'pose':>{width}s} {'survived':>9s} {'upright':>8s} {'feet':>5s}"
        f" {'drift xy':>9s} {'settled':>8s}"
    )
    for i, label in enumerate(labels):
        print(
            f"{int(env_ids[i]):4d} {label:>{width}s} {str(not bool(bad_term_mask[i])):>9s} {float(upright[i]):8.2f}"
            f" {int(contacts[i]):5d} {float(drift[i]):9.2f} {float(sank[i]):+8.2f}"
        )

    print()
    for name in names:
        rows = [i for i, label in enumerate(labels) if label == name]
        held = [
            i
            for i in rows
            if not bool(bad_term_mask[i]) and float(upright[i]) > 0.7 and int(contacts[i]) > 0
        ]
        worst_drift = max(float(drift[i]) for i in rows)
        verdict = "STANDS" if len(held) == len(rows) else ("PARTIAL" if held else "FAILS")
        print(
            f"{name:>{width}s}: {len(held)}/{len(rows)} standing upright with a foot down,"
            f" worst drift {worst_drift:.2f} m   -> {verdict}"
        )
    print(
        "\ndrift is how far the robot slid from where it was placed; a large one means the pose is on"
        "\na slope it cannot hold. 'settled' is how far it sank, roughly the leg compression."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    per_pose = args_cli.envs_per_pose
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand = None
    # a curriculum runs inside `_reset_idx` before the events and would overwrite the poses written below
    env_cfg.curriculum = None
    env_cfg.observations.policy.enable_corruption = False
    # one tile: every robot should stand on the same rock, where the poses were measured
    env_cfg.terrain_num_rows = 1
    env_cfg.terrain_num_cols = 1

    jitter = args_cli.jitter
    if jitter is None:
        from informed_exploration.tasks.manager_based.informed_exploration.cfg.scan.scan_env_cfg import (
            SCAN_EVAL_XY_HALF,
        )

        jitter = SCAN_EVAL_XY_HALF

    # num_envs is needed before the env exists, so resolve the sub-terrain cfg from the generator
    probe_sub = next(iter(env_cfg.scene.terrain.terrain_generator.sub_terrains.values()))
    n_poses = len(_collect_poses(probe_sub, args_cli.canonical_only))
    env_cfg.scene.num_envs = per_pose * n_poses

    print(f"[INFO] Creating environment: {args_cli.task}")
    # `log_dir` is a required positional on IEParkourEnv; nothing is written there
    env = gym.make(args_cli.task, cfg=env_cfg, log_dir="logs/scan_pose_check", render_mode="rgb_array")
    core_env = env.unwrapped

    try:
        sub_terrain = next(iter(core_env.cfg.scene.terrain.terrain_generator.sub_terrains.values()))
        poses = _collect_poses(sub_terrain, args_cli.canonical_only)
        names = list(poses)
        print(f"[INFO] scan poses (tile frame): {poses}")
        print(
            f"[INFO] {per_pose} robots per pose, jitter +-{jitter:.2f} m, dropped {args_cli.drop_height:.2f} m"
            f" above standing height, settling {args_cli.settle_steps} steps"
        )

        tasks = _pose_tasks(core_env, poses, names, per_pose, args_cli.drop_height, jitter, args_cli.seed)
        labels = [name for name in names for _ in range(per_pose)]
        env_ids = torch.arange(core_env.num_envs, device=core_env.device)

        core_env.task_space.set_tasks(core_env, env_ids, tasks)
        env.reset()

        # the articulation and the contact sensor index bodies independently; the force lookup wants the sensor's
        _, contact_foot_ids = resolve_foot_indices(core_env)
        zero_actions = torch.zeros((core_env.num_envs, core_env.action_space.shape[1]), device=core_env.device)
        bad_term_mask = torch.zeros(core_env.num_envs, dtype=torch.bool, device=core_env.device)
        for _ in range(args_cli.settle_steps):
            _, _, terminated, _, _ = env.step(zero_actions)
            bad_term_mask |= terminated

        _report(core_env, env_ids, labels, names, bad_term_mask, contact_foot_ids, tasks[:, :3, 0])

        for _ in range(args_cli.hold_steps):
            env.step(zero_actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
