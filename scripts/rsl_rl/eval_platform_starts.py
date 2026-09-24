# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start-PLATFORM evaluation of a goal-conditioned RSL-RL policy checkpoint.

Spawns a third of the episodes on each DTSG start platform with the goal fixed on the goal pad, so each
episode is a full route run; needs a sealed start column (``*-CORRIDORSTART-v0``) and no start pool.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Start-platform evaluation of an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Episodes to run, rounded down to a multiple of three so each start platform gets the same share.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_folder", type=str, default=None, help="Name of the checkpoint folder.")
parser.add_argument(
    "--eval_config",
    type=str,
    default=None,
    help="Path to eval config yaml with a list of runs (used with --sweep_index).",
)
parser.add_argument("--models", type=json.loads, default=None, help="Name of the models to evaluate.")
parser.add_argument("--checkpoint_base_path", type=str, default=None, help="Base path to the checkpoint folder.")
parser.add_argument("--results_dir", type=str, default="logs/results", help="Directory to save the results.")
parser.add_argument(
    "--results_suffix",
    type=str,
    default=None,
    help=(
        "Suffix appended to the results folder name, so this sits beside the corridor-goal and"
        " corridor-start output for the same checkpoint_folder. Falls back to the eval config yaml."
    ),
)
parser.add_argument(
    "--platform_margin",
    type=float,
    default=None,
    help=(
        "Inset (m) from every edge of a start platform's pocket, so the robot's whole footprint is"
        " clear of the separator walls it is spawned between. Default 0.4; yaml-overridable."
    ),
)
parser.add_argument(
    "--start_height",
    type=float,
    default=None,
    help=(
        "Base height (m) every start is spawned at ABOVE THE PAD SURFACE it stands on, not above the"
        " tile ground plane. The pad top is read from the terrain (geometry_bounds['start_z']), so"
        " this tracks a pad retune instead of baking in today's 0.05. Default 0.6, i.e. the robot's"
        " nominal standing height. yaml-overridable."
    ),
)
parser.add_argument(
    "--start_yaw_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help=(
        "Heading window (rad) about +x, which points down-route at the goal for all three routes."
        " Default '0 0', i.e. every robot faces straight down its route, so no part of the per-route"
        " number is heading noise. Pass a real window (e.g. -0.785 0.785, matching the pooled"
        " corridor starts) to reintroduce it. yaml-overridable."
    ),
)
parser.add_argument(
    "--store_trajectories",
    action="store_true",
    default=False,
    help=(
        "Also dump every episode's full per-step state to"
        " <results>/dist/trajectories_<model>.pkl.gz, the RSI reference-trajectory harvest."
        " Entries are (terrain_level, terrain_type, (T, 37) tensor, episode_reward_sum) tuples; the"
        " 37 columns are the 'trajectory' observation group (root_pos_w env-relative, root_quat_w,"
        " root_lin_vel_w, root_ang_vel_w, joint_pos_rel, joint_vel_rel). Pair with the 6 s"
        " Parkour-BridgeDTSG-BVER-HARVEST-v0 task and feed the dump to build_dtsg_rsi_pool.py."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Enable wandb logging.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import gc
import wandb

from collections import deque

import eval_utils
import rsl_tasks  # noqa: F401
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import informed_exploration.tasks  # noqa: F401
from informed_exploration.tasks.manager_based.informed_exploration.cfg.bridge.bridge_cfg import _only_sub_terrain
from informed_exploration.terrains.config.layouts import dtsg_route_rects

_X, _Y, _Z, _YAW = 0, 2, 4, 10
"""Flat indices of x, y, z, yaw in the interleaved (pos, vel) 18x2 task.

Other dims stay 0; joint dims are offsets from the default pose, so 0 is the default stance."""

DEFAULTS = {"platform_margin": 0.4, "start_yaw_range": [0.0, 0.0], "start_height": 0.6}
"""Applied when neither the CLI nor the eval yaml set them.

Placement is pinned (heading straight down-route, footprint clear of the walls) so per-route numbers carry
no placement noise; a degenerate yaw range is valid."""


def _label_by_corridor(xy, corridor_rects: dict) -> list[str]:
    """Route name per position by point-in-rect test; ``"other"`` outside every route."""
    labels = []
    for x, y in xy:
        inside = (
            name
            for name, (x_lo, x_hi, y_lo, y_hi) in corridor_rects.items()
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi
        )
        labels.append(next(inside, "other"))
    return labels


def _platform_regions(corridor_rects: dict, span_rects: dict, margin: float) -> dict:
    """``{route: (x_lo, x_hi, y_lo, y_hi)}`` for the start platform each route departs from.

    The pad is the route's full region minus its span, with every edge inset by ``margin``.
    """
    regions = {}
    for name, (x_lo, _, y_lo, y_hi) in corridor_rects.items():
        x_hi = span_rects[name][0]  # the span's near edge == the pad's far edge
        region = (x_lo + margin, x_hi - margin, y_lo + margin, y_hi - margin)
        assert region[0] < region[1] and region[2] < region[3], (
            f"start platform for '{name}' collapses under --platform_margin {margin}"
        )
        regions[name] = region
    return regions


def _platform_starts(n: int, regions: dict, start_z: float, yaw_range, device, generator) -> tuple:
    """``n`` starts spread evenly over the three start platforms, with their route labels.

    Default stance at absolute height ``start_z``, heading down-route within ``yaw_range``.
    """
    names = sorted(regions)  # sorted so a fixed seed reproduces the draw
    k = n // len(names)
    # interleave routes so a run cut short still covers all three evenly
    labels = [name for _ in range(k) for name in names]

    starts = torch.zeros(len(labels), 36, device=device)
    starts[:, _Z] = start_z
    for i, name in enumerate(names):
        rows = torch.arange(i, len(labels), len(names), device=device)
        x_lo, x_hi, y_lo, y_hi = regions[name]
        u = torch.rand(k, 3, generator=generator).to(device)
        starts[rows, _X] = x_lo + (x_hi - x_lo) * u[:, 0]
        starts[rows, _Y] = y_lo + (y_hi - y_lo) * u[:, 1]
        starts[rows, _YAW] = yaw_range[0] + (yaw_range[1] - yaw_range[0]) * u[:, 2]
    return starts, labels, k


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Start-platform evaluation with an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # resolve the eval config first: it fills args_cli fields (seed, num_envs, ...) used below
    eval_cfg = eval_utils.resolve_eval_config(args_cli, env_cfg)
    eval_utils.init_wandb_from_eval_config(args_cli, eval_cfg)
    for key, value in DEFAULTS.items():
        if getattr(args_cli, key) is None:
            setattr(args_cli, key, value)

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the seed here since some randomizations happen at environment initialization
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # this eval needs a fixed goal (the opposite of eval_multi_goal.py's guard)
    base_position_cfg = env_cfg.commands.base_position
    if getattr(base_position_cfg, "default_goal", None) is None:
        raise ValueError(
            f"--task {args_cli.task} has no commands.base_position.default_goal. This eval holds the"
            " goal fixed on the goal pad, so use a *-CORRIDORSTART-v0 task; the *_MGEVAL tasks"
            " deliberately clear default_goal."
        )
    if not getattr(env_cfg, "seal_start_column", False):
        raise ValueError(
            f"--task {args_cli.task} does not seal the start column. Spawning on the start platforms"
            " only names a route BECAUSE those walls split the left column into one pocket per"
            " route; without them a robot spawned on any pad can take any route and the per-route"
            " numbers are meaningless. Use a *-CORRIDORSTART-v0 task."
        )

    # retrieve all models of one run
    pt_files = eval_utils.discover_checkpoints(args_cli)

    # results path
    results_path = eval_utils.make_results_path(args_cli)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, log_dir=results_path, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.abspath(results_path), "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    task_space = getattr(env.unwrapped, "task_space", None)
    if task_space is None:
        raise ValueError("The environment does not have a task space defined for evaluation.")

    # draw the start set once so every checkpoint (and same-seed arm) sees identical starts
    sub_terrain = _only_sub_terrain(env.unwrapped.cfg.scene.terrain.terrain_generator)
    # full-passage regions label episodes; span-only regions mark where each pad ends
    corridor_rects = dtsg_route_rects(sub_terrain.layout, sub_terrain.size, include_start_column=True)
    span_rects = dtsg_route_rects(sub_terrain.layout, sub_terrain.size)
    regions = _platform_regions(corridor_rects, span_rects, args_cli.platform_margin)

    device = task_space.task_bounds(eval_bounds=True).device
    # --start_height is above the pad, whose top comes from the terrain's derived geometry
    pad_top = float(sub_terrain.geometry_bounds["start_z"])
    start_z = pad_top + float(args_cli.start_height)
    generator = torch.Generator().manual_seed(int(agent_cfg.seed))
    n = args_cli.num_samples if args_cli.num_samples is not None else 3
    queue, corridor_labels, per_route = _platform_starts(
        n, regions, start_z, args_cli.start_yaw_range, device, generator
    )

    # scatter the starts this run evaluates, so the viewport shows the real distribution
    env.unwrapped._corridor_start_points = queue[:, [_X, _Y, _Z]].cpu().numpy()

    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    if args_cli.env_markers:
        import isaaclab_markers

        # store the handle on the env so it is kept alive (and re-assertable via refresh())
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    if per_route == 0:
        raise SystemExit(f"--num_samples {n} is fewer than the three start platforms")
    print(
        f"[INFO]: Starting START-PLATFORM evaluation of {len(pt_files)} models:"
        f" {[os.path.basename(f) for f in pt_files]} on {task_name} for {queue.shape[0]} episodes"
        f" ({per_route} per platform, {args_cli.start_height:.2f} m above the pad -> z={start_z:.2f},"
        f" margin={args_cli.platform_margin},"
        f" yaw={[round(v, 3) for v in args_cli.start_yaw_range]};"
        f" seal_start_column={getattr(env.unwrapped.cfg, 'seal_start_column', None)})."
    )
    for resume_path in reversed(pt_files):

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        try:
            runner.load(resume_path)
        except FileNotFoundError as e:
            print(f"[WARNING]: Could not load checkpoint from {resume_path}: {e}")
            continue

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        dims = env.unwrapped.task_space.task_dim()

        env.unwrapped.eval_data["queue"] = deque(torch.clone(queue))
        env.unwrapped.eval_data["done"] = False
        env.unwrapped.eval_data["initialized"] = False
        env.unwrapped.eval_data["results"] = None
        env.unwrapped.eval_data["rewards"] = []
        env.unwrapped.eval_data["starts"] = []
        env.unwrapped.eval_data["finished"] = torch.tensor([0] * env.num_envs, dtype=torch.bool)
        env.unwrapped.eval_data["old_tasks"] = torch.zeros(env.unwrapped.num_envs, *dims, device=env.device)

        # the key's presence switches on EvalCurriculum's recorder; eval_data is cleared per model
        if args_cli.store_trajectories:
            env.unwrapped.eval_data["trajectories"] = []

        # run everything in inference mode
        with torch.inference_mode():
            env.reset()
            obs = env.get_observations()

            while simulation_app.is_running():
                start_time = time.time()

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

                if env.unwrapped.eval_data["done"]:
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        # log results; rows are row-aligned with "starts", which is what this eval labels by
        rewards_per_episode = torch.stack(env.unwrapped.eval_data["rewards"], dim=0)
        logs_dict = {"rewards": rewards_per_episode}
        logs_dict["starts"] = torch.stack(env.unwrapped.eval_data["starts"], dim=0)
        logs_dict["reward_names"] = list(env.unwrapped.reward_manager._episode_sums.keys())
        logs_dict["reward_band_norms"] = eval_utils.reward_band_norms(env.unwrapped)
        # same keys as eval_corridor_starts.py, so analyze_corridor_eval.py needs no flag to read it
        logs_dict["corridor_starts"] = True
        logs_dict["corridor_rects"] = corridor_rects
        logs_dict["corridor_per_route"] = per_route
        logs_dict["start_source"] = "platforms"  # provenance: synthesized on the pads, not pooled
        logs_dict["platform_regions"] = regions
        logs_dict["start_height"] = args_cli.start_height  # above the pad; absolute z was pad_top + this
        logs_dict["start_z"] = start_z
        logs_dict["seal_start_column"] = getattr(env.unwrapped.cfg, "seal_start_column", None)
        logs_dict["geometry_bounds"] = getattr(sub_terrain, "geometry_bounds", None)

        model_name = os.path.splitext(os.path.basename(resume_path))[0]

        rewards = rewards_per_episode.mean(dim=0)
        mean_rewards = rewards.sum()
        print(f"[INFO]: Mean reward: {mean_rewards:.4f}")

        reward_names = logs_dict["reward_names"]
        success_term = "tracking_pos_sparse"
        per_corridor: dict[str, float] = {}
        if success_term in reward_names:
            reached = rewards_per_episode[:, reward_names.index(success_term)] > 0
            success_rate = reached.float().mean().item() * 100
            print(
                f"[INFO]: Success rate (reached_goal > 0): {success_rate:.1f}%"
                f" ({int(reached.sum())}/{len(rewards_per_episode)} episodes)"
            )
            # label from each episode's recorded start, not the draw order, so reordering cannot shift it
            episode_labels = _label_by_corridor(logs_dict["starts"][:, :2, 0].tolist(), corridor_rects)
            for name in sorted(set(episode_labels)):
                mask = torch.tensor([label == name for label in episode_labels], device=reached.device)
                per_corridor[name] = reached[mask].float().mean().item() * 100
                print(
                    f"[INFO]:   started on the {name} platform: {per_corridor[name]:.1f}%"
                    f" ({int(reached[mask].sum())}/{int(mask.sum())} episodes)"
                )

        if wandb.run is not None:
            log_dict = {"mean_reward": mean_rewards.item()}
            try:
                log_dict["model_nr"] = int(model_name.split("_")[-1])
            except ValueError:
                pass
            if success_term in reward_names:
                log_dict["success_rate"] = success_rate
                for name, rate in per_corridor.items():
                    log_dict[f"success_rate/{name}"] = rate
            if len(reward_names) == len(rewards):
                for name, val in zip(reward_names, rewards):
                    log_dict[f"rewards/{name}"] = val.item()
            wandb.log(log_dict)

        eval_utils.save_dist_pickle(results_path, model_name, logs_dict)

        if args_cli.store_trajectories:
            traj_dict = {
                "trajectories": env.unwrapped.eval_data.get("trajectories", None),
                # the term-name list documents the 37-column layout for downstream consumers
                "observations": (
                    env.unwrapped.observation_manager.active_terms["trajectory"]
                    if "trajectory" in env.unwrapped.observation_manager.active_terms
                    else None
                ),
            }
            eval_utils.save_trajectories_pickle(results_path, traj_dict, model_name=model_name)
            del traj_dict

        # Free memory by clearing and deleting large objects
        del logs_dict
        env.unwrapped.eval_data.clear()

        del runner
        del policy
        if "policy_nn" in locals():
            del policy_nn

        gc.collect()
        torch.cuda.empty_cache()

    # close the simulator
    env.close()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
