# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Corridor-START evaluation of a goal-conditioned RSL-RL policy, the dual of ``eval_multi_goal.py --corridor_goals``.

Spawns the robot mid-route from a pool of settled feasible states with the goal fixed on the goal pad and
the start column sealed, so per-route success measures traversal isolated from approach.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Corridor-start evaluation of an RSL-RL agent.")
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
    help=(
        "Upper bound on episodes. The actual count is 3 * k where k is the per-route draw, capped by"
        " the smallest route bucket, since repeats would re-run identical episodes, so the pool's size is"
        " the real limit and it is reported rather than padded around."
    ),
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
        "Suffix appended to the results folder name (results_dir/<checkpoint_folder>_<suffix>/...), so"
        " the corridor-start output sits beside the corridor-goal and single-goal runs for the same"
        " checkpoint_folder. Falls back to the eval config yaml's results_suffix key."
    ),
)
parser.add_argument(
    "--start_pool",
    type=str,
    default=None,
    help=(
        "Path to the corridor feasible-start pool (.pkl from generate_feasible_starts.py on"
        " *-GenCorridorStarts-v0), repo-root-relative. Defaults to the DTSG pool; yaml-overridable."
    ),
)
parser.add_argument(
    "--min_per_route",
    type=int,
    default=None,
    help=(
        "Refuse to run if any route's bucket holds fewer than this many pooled starts, rather than"
        " reporting a route whose rate rests on a handful of episodes. Default 32; yaml-overridable."
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
import pickle
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
from informed_exploration.tasks.manager_based.informed_exploration.cfg.bridge.bridge_cfg_random import (
    DTSG_CORRIDOR_POOL,
)
from informed_exploration.terrains.config.layouts import dtsg_route_rects

_INIT_SUBSPACE = list(range(0, 36, 2))
"""The position dims of the 18x2 task; velocities stay zero, matching the BVER and RandomGS arms."""

DEFAULTS = {"start_pool": DTSG_CORRIDOR_POOL, "min_per_route": 32}
"""Applied when neither the CLI nor the eval yaml set them."""


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


def _load_corridor_pool(pool_path: str, corridor_rects: dict) -> dict:
    """Bucket a feasible-start pool by which route each state's base sits on.

    Returns ``{route: Tensor(n_route, 18, 2)}``. States outside every route are dropped and counted;
    many of them means the pool came from the layout-wide ``-GenFeasibleStarts-v0`` task.
    """
    resolved = pool_path if os.path.isabs(pool_path) else os.path.join(os.getcwd(), pool_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"corridor start pool not found at {resolved}. Generate it first:\n"
            "  generate_feasible_starts.py --task Parkour-BridgeDTSG-GenCorridorStarts-v0"
            " --surface_top_z 0.55 --num_states <N> --out <path>"
        )
    with open(resolved, "rb") as f:
        pool = torch.as_tensor(pickle.load(f)["tasks"], dtype=torch.float32)

    labels = _label_by_corridor(pool[:, :2, 0].tolist(), corridor_rects)  # base (x, y) position
    buckets = {
        name: pool[torch.tensor([i for i, label in enumerate(labels) if label == name], dtype=torch.long)]
        for name in sorted(corridor_rects)
    }
    dropped = sum(1 for label in labels if label == "other")
    print(
        f"[INFO]: corridor start pool {pool_path}: {len(pool)} states ->"
        f" { {k: len(v) for k, v in buckets.items()} }, {dropped} outside every route"
    )
    return buckets


def _draw_balanced(buckets: dict, num_samples: int | None, min_per_route: int, generator) -> tuple:
    """Equal draw per route, without replacement, capped by the smallest bucket.

    Returns ``(queue Tensor(3k, 36), labels list[str], k)``. No repeats, since a deterministic eval
    would re-run the identical episode.
    """
    smallest = min(len(v) for v in buckets.values())
    for name, states in buckets.items():
        if len(states) < min_per_route:
            raise SystemExit(
                f"route '{name}' has only {len(states)} pooled starts (< --min_per_route"
                f" {min_per_route}); its success rate would rest on too few episodes. Generate a"
                " larger pool with --num_states."
            )

    k = smallest if num_samples is None else min(smallest, num_samples // len(buckets))
    if num_samples is not None and k * len(buckets) < num_samples:
        print(
            f"[INFO]: capping at {k * len(buckets)} episodes ({k} per route), not the requested"
            f" {num_samples}: the smallest route bucket holds {smallest} distinct starts and repeats"
            " would re-run identical episodes."
        )

    names = sorted(buckets)  # sorted so a fixed seed reproduces the draw
    rows = []
    for name in names:
        states = buckets[name]
        pick = torch.randperm(len(states), generator=generator)[:k]
        rows.append(states[pick].reshape(k, -1))

    # interleave the routes so a run cut short still covers all of them evenly
    queue = torch.stack(rows, dim=1).reshape(-1, rows[0].shape[-1])
    labels = [name for _ in range(k) for name in names]
    return queue, labels, k


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Corridor-start evaluation with an RSL-RL agent."""
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
            f"--task {args_cli.task} has no commands.base_position.default_goal. The corridor-start"
            " eval holds the goal fixed on the goal pad, so use an *-CORRIDORSTART-v0 task (or"
            " another *_EVAL binding); the *_MGEVAL tasks deliberately clear default_goal."
        )
    if not getattr(env_cfg, "seal_start_column", False):
        print(
            "[WARN]: this task does not seal the start column, so a robot spawned mid-route can"
            " retreat across the open start column and finish via a different route. Per-route"
            " numbers will be optimistic. Use a *-CORRIDORSTART-v0 task."
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
    # use the same regions the pool was generated in so no pooled state is dropped
    corridor_rects = dtsg_route_rects(sub_terrain.layout, sub_terrain.size, include_start_column=True)
    buckets = _load_corridor_pool(args_cli.start_pool, corridor_rects)
    generator = torch.Generator().manual_seed(int(agent_cfg.seed))
    queue, corridor_labels, per_route = _draw_balanced(
        buckets, args_cli.num_samples, args_cli.min_per_route, generator
    )
    queue = queue.to(task_space.task_bounds(eval_bounds=True).device)

    # only the position dims are teleported; velocities stay at the default (see _INIT_SUBSPACE)
    keep = torch.zeros_like(queue)
    keep[:, _INIT_SUBSPACE] = queue[:, _INIT_SUBSPACE]
    queue = keep

    # scatter the starts this run actually evaluates, so the viewport shows the real distribution
    env.unwrapped._corridor_start_points = queue[:, [0, 2, 4]].cpu().numpy()

    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    if args_cli.env_markers:
        import isaaclab_markers

        # store the handle on the env so it is kept alive (and re-assertable via refresh())
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    n = queue.shape[0]
    print(
        f"[INFO]: Starting CORRIDOR-START evaluation of {len(pt_files)} models:"
        f" {[os.path.basename(f) for f in pt_files]} on {task_name} for {n} episodes"
        f" ({per_route} per route, drawn without replacement from {args_cli.start_pool};"
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
        # the analyzer keys off `corridor_starts` to label by start instead of goal
        logs_dict["corridor_starts"] = True
        logs_dict["corridor_rects"] = corridor_rects
        logs_dict["corridor_per_route"] = per_route
        logs_dict["corridor_start_pool"] = args_cli.start_pool
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
                    f"[INFO]:   started on {name}: {per_corridor[name]:.1f}%"
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
