# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pairs evaluation of a goal-conditioned RSL-RL checkpoint on a real-world rock scan.

Runs the measured pair plus extra starts to the canonical goal and the canonical start to extra goals
(no cross product), all read from the scan's :meth:`ScanSpec.pairs`. Needs a ``*-BVER-MGEVAL-v0`` task.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Pairs evaluation of an RSL-RL agent on a rock scan.")
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
        "Episodes PER PAIR, not in total: a scan with more extras costs more episodes rather than"
        " thinning each pair's sample. Default 64, which is what the per-pair interval is sized for."
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
        "Suffix appended to the results folder name, so this sits beside the single-pair eval's"
        " output for the same checkpoint_folder. Falls back to the eval config yaml."
    ),
)
parser.add_argument(
    "--jitter_xy",
    type=float,
    default=None,
    help=(
        "Half-width (m) of the uniform xy spread around every start, extra and canonical alike."
        " Defaults to the task's own SCAN_EVAL_XY_HALF, i.e. the window eval.py draws from, so the"
        " canonical row reproduces the single-pair number. yaml-overridable."
    ),
)
parser.add_argument(
    "--play",
    action="store_true",
    default=False,
    help=(
        "Viewer mode: put every pair on one tile at once and run without writing results. Use with a"
        " *-BVER-MGPLAY-v0 task, and give --num_envs at least one env per pair."
    ),
)
parser.add_argument(
    "--store_trajectories",
    action="store_true",
    default=False,
    help=(
        "Also dump every episode's full per-step state to <results>/dist/trajectories_<model>.pkl.gz."
        " Entries are (terrain_level, terrain_type, (T, 37) tensor, episode_reward_sum) tuples."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Enable wandb logging.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gc
import gymnasium as gym
import os
import time
import torch
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
from informed_exploration.tasks.manager_based.informed_exploration.curriculum.task_space.parkour_task_space_cfg import (
    height_offset,
)
from informed_exploration.tasks.manager_based.informed_exploration.mdp.commands import CurriculumGoalCfg

DEFAULTS = {"num_samples": 64}
"""Applied when neither the CLI nor the eval yaml set them.

``--jitter_xy`` falls back to the task's ``SCAN_EVAL_XY_HALF`` so the canonical row matches eval.py."""


def _pair_poses(env, spec, jitter, num_samples, device, generator):
    """The eval queue: ``num_samples`` episodes for every pair the scan declares, interleaved by pair.

    Returns ``(pairs_39d, labels, kinds, table)``. Poses come from the terrain's tile-frame poses, and the
    start z is re-raycast at the jittered xy so a spawn on a slope is neither buried nor dropped.
    """
    sub = _only_sub_terrain(env.cfg.scene.terrain.terrain_generator)
    origin = sub.goal_poses["start"]  # the terrain origin: every env-relative pose is measured from it
    published = dict(sub.goal_poses)
    published.update(sub.extra_poses or {})

    # map to the published tile pose by key; `pairs()` and `extra_poses` share the same order
    def key_of(kind, index, end):
        if kind == "canonical" or (kind == "start_star" and end == "goal") or (kind == "goal_star" and end == "start"):
            return end
        return f"{end}_{index}"

    rows = spec.pairs()
    starts_e, goals_e, labels, kinds, table = [], [], [], [], {}
    counters = {"start_star": 0, "goal_star": 0}
    for label, kind, start_obj, goal_obj in rows:
        if kind in counters:
            counters[kind] += 1
        index = counters.get(kind, 0)
        start_pose = published[key_of(kind, index, "start")]
        goal_pose = published[key_of(kind, index, "goal")]
        starts_e.append((start_pose[0] - origin[0], start_pose[1] - origin[1]))
        goals_e.append((goal_pose[0] - origin[0], goal_pose[1] - origin[1], goal_pose[2] + height_offset))
        labels.append(label)
        kinds.append(kind)
        table[label] = {
            "kind": kind,
            "start_obj": tuple(float(v) for v in start_obj),
            "goal_obj": tuple(float(v) for v in goal_obj),
            "start_tile": tuple(float(v) for v in start_pose),
            "goal_tile": tuple(float(v) for v in goal_pose),
        }

    n_pairs = len(rows)
    total = n_pairs * num_samples
    # interleaved: episode i belongs to pair i % n_pairs
    order = torch.arange(total) % n_pairs
    start_xy = torch.tensor(starts_e, dtype=torch.float32)[order]
    goal_xyz = torch.tensor(goals_e, dtype=torch.float32)[order].to(device)

    if jitter > 0.0:
        offsets = (torch.rand((total, 2), generator=generator) * 2.0 - 1.0) * jitter
        start_xy = start_xy + offsets
    start_xy = start_xy.to(device)

    # z at the jittered xy, plus the standing height the task space works in
    start_z = eval_utils.surface_z_at(env, start_xy) + height_offset
    heading = float(sub.heading)
    starts = eval_utils.synthesize_start_states(start_xy, start_z, heading, device)

    pairs = torch.cat([goal_xyz, starts], dim=-1)
    return pairs, [labels[i] for i in order.tolist()], [kinds[i] for i in order.tolist()], table


def _assert_reachable(task_space, table, published_start, sub) -> None:
    """Raise unless every pair lies inside the task-space bounds and keep-in regions.

    Extras arrive via ``set_goal`` at runtime and skip the config-time check done for the measured pair.
    """
    # use the accessors, which also tolerate a cfg without a keep-in filter
    bounds = task_space.task_bounds(eval_bounds=False).tolist()
    keep_in = task_space.get_keep_in_filter()
    keep_in_dims = task_space.get_keep_in_dimensions()
    assert keep_in is None or tuple(keep_in_dims) == (0, 2), (
        f"the keep-in rects index dimensions {keep_in_dims}, not the flat (x, y) = (0, 2) this check"
        " assumes; update it alongside BridgeTaskSpaceCfg.keep_in_dimensions."
    )
    problems = []
    for label, entry in table.items():
        gx = entry["goal_tile"][0] - published_start[0]
        gy = entry["goal_tile"][1] - published_start[1]
        gz = entry["goal_tile"][2] + height_offset
        sx = entry["start_tile"][0] - published_start[0]
        sy = entry["start_tile"][1] - published_start[1]
        for name, value, row in (("goal x", gx, 0), ("goal y", gy, 2), ("goal z", gz, 4),
                                 ("start x", sx, 0), ("start y", sy, 2)):
            if not bounds[row][0] <= value <= bounds[row][1]:
                problems.append(f"  {label}: {name} {value:.2f} outside task-space bounds {bounds[row]}")
        if keep_in is not None:
            for name, x, y in (("goal", gx, gy), ("start", sx, sy)):
                if not any(lo_x <= x <= hi_x and lo_y <= y <= hi_y for (lo_x, hi_x), (lo_y, hi_y) in keep_in):
                    problems.append(f"  {label}: {name} ({x:.2f}, {y:.2f}) outside every walkable region")
    if problems:
        raise ValueError(
            "some pairs on this scan do not lie inside the task space:\n"
            + "\n".join(problems)
            + f"\n(scan footprint, tile frame: {sub.footprint_rect_tile}). Move the offending"
            " extra_starts_obj / extra_goals_obj in terrains/config/scans.py further inside the rock."
        )


def _match_labels(starts, goals, table, published_start, tolerance):
    """Label each finished episode by the nearest pair to its recorded start and goal xy.

    Episodes farther than ``tolerance`` from every pair are labelled ``"unmatched"``.
    """
    labels = []
    for i in range(starts.shape[0]):
        sx, sy = float(starts[i, 0, 0]), float(starts[i, 1, 0])
        gx, gy = float(goals[i, 0]), float(goals[i, 1])
        best, best_d = "unmatched", float("inf")
        for label, entry in table.items():
            esx = entry["start_tile"][0] - published_start[0]
            esy = entry["start_tile"][1] - published_start[1]
            egx = entry["goal_tile"][0] - published_start[0]
            egy = entry["goal_tile"][1] - published_start[1]
            d = max(((sx - esx) ** 2 + (sy - esy) ** 2) ** 0.5, ((gx - egx) ** 2 + (gy - egy) ** 2) ** 0.5)
            if d < best_d:
                best, best_d = label, d
        labels.append(best if best_d <= tolerance else "unmatched")
    return labels


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Pairs evaluation with an RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]

    # resolve the eval config first: it fills args_cli fields (seed, num_envs, ...) used below
    eval_cfg = eval_utils.resolve_eval_config(args_cli, env_cfg)
    if not args_cli.play:
        eval_utils.init_wandb_from_eval_config(args_cli, eval_cfg)
    for key, value in DEFAULTS.items():
        if getattr(args_cli, key) is None:
            setattr(args_cli, key, value)

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # The goal must be FREE for the queue to write, the mirror of eval_platform_starts.py's guard.
    base_position_cfg = env_cfg.commands.base_position
    if not isinstance(base_position_cfg, CurriculumGoalCfg):
        raise ValueError(
            f"--task {args_cli.task} does not use a CurriculumGoalCfg for commands.base_position (got"
            f" {type(base_position_cfg).__name__}); this eval pushes a goal per episode, so it needs"
            " a *-BVER-MGEVAL-v0 / *-BVER-MGPLAY-v0 scan task."
        )
    if base_position_cfg.default_goal is not None:
        raise ValueError(
            f"--task {args_cli.task} sets commands.base_position.default_goal="
            f"{base_position_cfg.default_goal}. CurriculumGoalCommand._resample_command rewrites the"
            " goal to this value on EVERY reset, so every pair would silently be run against the"
            " canonical goal. Use a *-BVER-MGEVAL-v0 task, which clears default_goal."
        )
    goal_command_name = getattr(getattr(env_cfg.curriculum, "evaluation", None), "goal_command_name", None)
    if goal_command_name is None:
        raise ValueError(
            f"--task {args_cli.task}'s curriculum.evaluation.goal_command_name is not set, so the eval"
            " queue would push starts only. Use a *-BVER-MGEVAL-v0 task"
            " (ClimbBoxMultiGoalEvalCurriculumCfg)."
        )
    spec = getattr(env_cfg, "scan", None)
    if spec is None:
        raise ValueError(f"--task {args_cli.task} is not a scan task; this eval reads its pairs from a ScanSpec.")

    pt_files = eval_utils.discover_checkpoints(args_cli)
    results_path = eval_utils.make_results_path(args_cli)

    env = gym.make(
        args_cli.task, cfg=env_cfg, log_dir=results_path, render_mode="rgb_array" if args_cli.video else None
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    task_space = getattr(env.unwrapped, "task_space", None)
    if task_space is None:
        raise ValueError("The environment does not have a task space defined for evaluation.")

    sub_terrain = _only_sub_terrain(env.unwrapped.cfg.scene.terrain.terrain_generator)
    device = task_space.task_bounds(eval_bounds=True).device
    dims = task_space.task_dim()

    jitter = args_cli.jitter_xy
    if jitter is None:
        from informed_exploration.tasks.manager_based.informed_exploration.cfg.scan.scan_env_cfg import (
            SCAN_EVAL_XY_HALF,
        )

        jitter = SCAN_EVAL_XY_HALF

    # draw the pair set once so every checkpoint (and same-seed arm) sees identical pairs
    generator = torch.Generator().manual_seed(int(agent_cfg.seed))
    queue, labels, kinds, table = _pair_poses(
        env.unwrapped, spec, float(jitter), int(args_cli.num_samples), device, generator
    )
    _assert_reachable(task_space, table, sub_terrain.goal_poses["start"], sub_terrain)

    if args_cli.play:
        # every robot on the rock the poses were measured on, so the fan-out reads as one picture
        ref_origin = env.unwrapped.scene.env_origins[0].clone()
        env.unwrapped.scene.env_origins[:] = ref_origin
        if env.num_envs < len(table):
            print(
                f"[WARNING]: {env.num_envs} envs for {len(table)} pairs; the queue is drained across"
                " envs, so they will be shown in waves rather than all at once. Pass --num_envs"
                f" {len(table)} or more to see every pair together."
            )

    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    if args_cli.env_markers:
        import isaaclab_markers

        # store the handle on the env so it is kept alive (and re-assertable via refresh())
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    counts = {label: labels.count(label) for label in table}
    print(
        f"[INFO]: Starting PAIRS evaluation of {len(pt_files)} models:"
        f" {[os.path.basename(f) for f in pt_files]} on {task_name} for {queue.shape[0]} episodes"
        f" over {len(table)} pairs ({args_cli.num_samples} each, jitter +-{jitter:.2f} m)."
    )
    for label, entry in table.items():
        print(f"[INFO]:   {label:>10s} [{entry['kind']:>10s}]  start {entry['start_obj']} -> goal {entry['goal_obj']}"
              f"  ({counts[label]} episodes)")

    for resume_path in reversed(pt_files):

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
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

        policy = runner.get_inference_policy(device=env.unwrapped.device)

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

        env.unwrapped.eval_data["queue"] = deque(torch.clone(queue))
        env.unwrapped.eval_data["done"] = False
        env.unwrapped.eval_data["initialized"] = False
        env.unwrapped.eval_data["results"] = None
        env.unwrapped.eval_data["rewards"] = []
        env.unwrapped.eval_data["goals"] = []
        env.unwrapped.eval_data["starts"] = []
        env.unwrapped.eval_data["finished"] = torch.tensor([0] * env.num_envs, dtype=torch.bool)
        env.unwrapped.eval_data["old_tasks"] = torch.zeros(env.unwrapped.num_envs, *dims, device=env.device)
        env.unwrapped.eval_data["old_goals"] = torch.zeros(env.unwrapped.num_envs, 3, device=env.device)

        # the key's presence switches on EvalCurriculum's recorder; eval_data is cleared per model
        if args_cli.store_trajectories:
            env.unwrapped.eval_data["trajectories"] = []

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
                    if args_cli.play:
                        # loop the queue so the fan-out keeps playing for as long as it is watched
                        env.unwrapped.eval_data["queue"] = deque(torch.clone(queue))
                        env.unwrapped.eval_data["done"] = False
                        env.unwrapped.eval_data["finished"][:] = False
                        # drop each lap's records so the lists do not grow for the whole session
                        for key in ("rewards", "goals", "starts"):
                            env.unwrapped.eval_data[key].clear()
                    else:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        if args_cli.play:
            # play mode writes nothing and loads no further checkpoint
            env.unwrapped.eval_data.clear()
            break

        rewards_per_episode = torch.stack(env.unwrapped.eval_data["rewards"], dim=0)
        logs_dict = {"rewards": rewards_per_episode}
        logs_dict["goals"] = torch.stack(env.unwrapped.eval_data["goals"], dim=0)
        logs_dict["starts"] = torch.stack(env.unwrapped.eval_data["starts"], dim=0)
        logs_dict["reward_names"] = list(env.unwrapped.reward_manager._episode_sums.keys())
        # per-term norms that turn a raw episodic sum into BVER's [0, 1] r_min/r_max fraction
        logs_dict["reward_band_norms"] = eval_utils.reward_band_norms(env.unwrapped)
        # everything the offline analyzer needs, so it never imports Isaac
        logs_dict["scan_pairs"] = True
        logs_dict["scan_snake"] = spec.snake
        logs_dict["scan_name"] = spec.name
        logs_dict["pair_table"] = table
        logs_dict["pair_jitter_xy"] = float(jitter)
        logs_dict["pair_num_samples"] = int(args_cli.num_samples)
        logs_dict["geometry_bounds"] = getattr(sub_terrain, "geometry_bounds", None)
        logs_dict["terrain_origin"] = tuple(float(v) for v in sub_terrain.goal_poses["start"])

        episode_labels = _match_labels(
            logs_dict["starts"], logs_dict["goals"], table, sub_terrain.goal_poses["start"], max(2.0 * jitter, 0.5)
        )
        logs_dict["pair_labels"] = episode_labels
        logs_dict["pair_kinds"] = [table[label]["kind"] if label in table else "unmatched" for label in episode_labels]

        model_name = os.path.splitext(os.path.basename(resume_path))[0]

        rewards = rewards_per_episode.mean(dim=0)
        mean_rewards = rewards.sum()
        print(f"[INFO]: Mean reward: {mean_rewards:.4f}")

        reward_names = logs_dict["reward_names"]
        success_term = "tracking_pos_sparse"
        per_pair: dict[str, float] = {}
        if success_term in reward_names:
            reached = rewards_per_episode[:, reward_names.index(success_term)] > 0
            success_rate = reached.float().mean().item() * 100
            print(
                f"[INFO]: Success rate (reached_goal > 0): {success_rate:.1f}%"
                f" ({int(reached.sum())}/{len(rewards_per_episode)} episodes)"
            )
            print(f"[INFO]: {'pair':>10s} {'kind':>10s} {'success':>8s}  episodes")
            for label in list(table) + (["unmatched"] if "unmatched" in episode_labels else []):
                mask = torch.tensor([lab == label for lab in episode_labels], device=reached.device)
                if not bool(mask.any()):
                    continue
                per_pair[label] = reached[mask].float().mean().item() * 100
                kind = table[label]["kind"] if label in table else "unmatched"
                print(
                    f"[INFO]: {label:>10s} {kind:>10s} {per_pair[label]:7.1f}%"
                    f"  ({int(reached[mask].sum())}/{int(mask.sum())})"
                )

        if wandb.run is not None:
            log_dict = {"mean_reward": mean_rewards.item()}
            try:
                log_dict["model_nr"] = int(model_name.split("_")[-1])
            except ValueError:
                pass
            if success_term in reward_names:
                log_dict["success_rate"] = success_rate
                for label, rate in per_pair.items():
                    log_dict[f"success_rate/{label}"] = rate
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
                "pair_labels": episode_labels,
            }
            eval_utils.save_trajectories_pickle(results_path, traj_dict, model_name=model_name)
            del traj_dict

        del logs_dict
        env.unwrapped.eval_data.clear()

        del runner
        del policy
        if "policy_nn" in locals():
            del policy_nn

        gc.collect()
        torch.cuda.empty_cache()

    env.close()

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
