# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--store_trajectories", action="store_true", default=False, help="Store trajectories during evaluation."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_samples", type=int, default=None, help="Number of samples per task dimension for the evaluation grid."
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
parser.add_argument(
    "--checkpoint_base_path",
    type=str,
    default=None,
    help="Base path to the checkpoint folder.",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="logs/results",
    help="Directory to save the results.",
)
parser.add_argument(
    "--results_suffix",
    type=str,
    default=None,
    help=(
        "Suffix appended to the results folder name (results_dir/<checkpoint_folder>_<suffix>/...), so"
        " e.g. a single-goal and a multi-goal eval of the same checkpoint_folder don't overwrite each"
        " other's output. Falls back to the eval config yaml's results_suffix key if set there."
    ),
)
parser.add_argument(
    "--perturbations",
    type=json.loads,
    default=None,
    help=(
        "JSON list of perturbation levels, each evaluated as its own full pass over the eval queue and"
        " written to <results>/<level name>/dist/. Needs a task whose cfg declares the push term (the"
        " -ROBUSTEVAL / -ROBUSTPLAY bindings). Normally set from the eval yaml's `perturbations:` key"
        " instead; see eval_utils.normalize_perturbations for the schema. Without it the eval runs a"
        " single unperturbed pass, writing to <results>/dist/ as before."
    ),
)
parser.add_argument(
    "--detailed", action="store_true", default=False, help="Wether to store task-reward pairs or just rewards."
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

import math
from collections import deque, defaultdict

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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import informed_exploration.tasks  # noqa: F401


def _yaw_task_index(task_space) -> int:
    """Flat index of the start yaw inside one task vector.

    A task is ``(n_params, 2)`` flattened, so this is the yaw's pose row times the pair width.
    Raises when the task space has no yaw start dimension.
    """
    pose_params = task_space.get_task_cfg()["event"]["reset"]["reset_base"]["pose"]
    row = task_space.cfg.subterm_ids["reset_base"][pose_params.index("yaw")]
    return row * task_space.task_dim()[1]


def _goal_command_name(env_cfg) -> str | None:
    """Name of the goal command the eval curriculum drives, or None for a start-only eval."""
    return getattr(getattr(env_cfg.curriculum, "evaluation", None), "goal_command_name", None)


def _box_footprint(env_cfg) -> tuple[float, float, float, float] | None:
    """Env-relative ``(x_lo, x_hi, y_lo, y_hi)`` of the climb box, for the offline maps.

    The box is centred on the env origin; size ranges are reported at their widest.
    """
    sub_terrains = getattr(env_cfg.scene.terrain.terrain_generator, "sub_terrains", None) or {}
    box = sub_terrains.get("big_box_up")
    if box is None:
        return None
    length, width = max(box.length_range), max(box.width_range)
    return (-length / 2, length / 2, -width / 2, width / 2)


def _load_feasible_pool(path: str, task_space, device) -> torch.Tensor:
    """The feasible-starts pool as flat task vectors, ``(N, total_dims)``.

    Raises when the pool was generated for a different task space.
    """
    with open(os.path.join(os.getcwd(), path), "rb") as f:
        pool = pickle.load(f)["tasks"]
    pool = torch.as_tensor(pool, dtype=torch.float32, device=device)
    dims = tuple(task_space.task_dim())
    if tuple(pool.shape[1:]) != dims:
        raise ValueError(
            f"feasible pool {path} holds states of shape {tuple(pool.shape[1:])}; task space expects {dims}"
        )
    return pool.reshape(len(pool), -1)


def _draw_from_pool(pool: torch.Tensor, n: int, what: str) -> torch.Tensor:
    """``n`` distinct pool rows, or with replacement (and a warning) when the pool is smaller."""
    if n <= len(pool):
        return pool[torch.randperm(len(pool), device=pool.device)[:n]]
    print(f"[WARN]: {n} {what} requested from a pool of {len(pool)}; drawing with replacement")
    return pool[torch.randint(0, len(pool), (n,), device=pool.device)]


def _level_queue(
    level: dict | None,
    nominal_starts: torch.Tensor,
    pool: torch.Tensor | None,
    nominal_goal: torch.Tensor | None,
    goal_mode: bool,
    yaw_index: int,
    xyz_index: list[int],
) -> torch.Tensor:
    """The eval queue for one level, built once so every checkpoint sees the same tasks.

    Fixed levels reuse the nominal starts so levels differ only by the perturbation. On a goal-carrying
    task each entry is ``goal_xyz ++ start``, with the nominal goal wherever the level does not vary it.
    """
    if level is None:
        starts, goals = nominal_starts, None
    else:
        n = level["num_samples"] or len(nominal_starts)
        if level["starts"] == "pool":
            starts = _draw_from_pool(pool, n, "starts")
        else:
            repeats = -(-n // len(nominal_starts))
            starts = nominal_starts.repeat(repeats, 1)[:n].clone()
            if level["yaw_init"] is not None:
                if level["alternate_sign"]:
                    starts[0::2, yaw_index] = level["yaw_init"]
                    starts[1::2, yaw_index] = -level["yaw_init"]
                else:
                    starts[:, yaw_index] = level["yaw_init"]
        goals = _draw_from_pool(pool, n, "goals")[:, xyz_index] if level["goals"] == "pool" else None
    if not goal_mode:
        return starts
    if goals is None:
        goals = nominal_goal.expand(len(starts), 3)
    return torch.cat([goals, starts], dim=-1)


def _fire_push(env, level: dict) -> None:
    """Kick every env whose per-env episode counter just reached this level's push step, exactly once."""
    due = (env.episode_length_buf == level["push"]["step"]).nonzero(as_tuple=False).flatten()
    if len(due) == 0:
        return
    term_cfg = env.event_manager.get_term_cfg(eval_utils.PUSH_TERM)
    velocity_range = level["push"]["velocity_range"]
    if not level["alternate_sign"]:
        term_cfg.params["velocity_range"] = velocity_range
        env.event_manager.apply(mode=eval_utils.PERTURB_MODE, env_ids=due)
        return
    # mirror the lateral component on odd env slots, so one level covers a push from either side
    for sign, parity in ((1.0, 0), (-1.0, 1)):
        subset = due[due % 2 == parity]
        if len(subset) == 0:
            continue
        lo, hi = velocity_range["y"]
        term_cfg.params["velocity_range"] = dict(velocity_range, y=(sign * lo, sign * hi))
        env.event_manager.apply(mode=eval_utils.PERTURB_MODE, env_ids=subset)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # resolve the eval config first: it fills args_cli fields (seed, num_envs, ...) used below
    eval_cfg = eval_utils.resolve_eval_config(args_cli, env_cfg)
    eval_utils.init_wandb_from_eval_config(args_cli, eval_cfg)

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the seed here since some randomizations happen at environment initialization
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # retrieve all models of one run
    pt_files = eval_utils.discover_checkpoints(args_cli)

    # results path
    results_path = eval_utils.make_results_path(args_cli)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, log_dir=results_path, render_mode="rgb_array" if args_cli.video else None
    )

    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    # plain-number terrain description for offline analyses that must not import Isaac
    terrain_meta = eval_utils.terrain_metadata(env.unwrapped)

    # draw static start / goal / terrain-box / task-space-bounds markers in the viewport
    if args_cli.env_markers:
        import isaaclab_markers

        # store the handle on the env so it is kept alive (and re-assertable via refresh())
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_log_dir = os.path.abspath(results_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(video_log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # randomly sample task combinations for evaluation
    task_space = getattr(env.unwrapped, "task_space", None)
    if task_space is None:
        raise ValueError("The environment does not have a task space defined for evaluation.")

    n = args_cli.num_samples
    d = task_space.total_task_dim()
    bounds = task_space.task_bounds(eval_bounds=True)
    queue = torch.rand(n, d, device=bounds.device)
    queue = queue * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # filter out invalid tasks
    # queue = task_space.filter_tasks(queue.unsqueeze(1)).squeeze(1)

    # each perturbation level is a full pass written to its own sub-folder; none means an ordinary eval
    levels = eval_utils.normalize_perturbations(
        args_cli.perturbations, env.unwrapped.step_dt, max_steps=int(env.unwrapped.max_episode_length)
    )
    if levels and eval_utils.PERTURB_MODE not in env.unwrapped.event_manager.available_modes:
        raise ValueError(
            f"--perturbations needs a task whose env cfg declares the '{eval_utils.PUSH_TERM}' event term"
            f" under mode '{eval_utils.PERTURB_MODE}' (the -ROBUSTEVAL / -ROBUSTPLAY bindings);"
            f" '{task_name}' does not."
        )
    sweep_levels = levels or [None]

    # goal-carrying tasks consume (goal ++ start) pairs and need a nominal goal; start-only tasks cannot vary goals
    goal_mode = _goal_command_name(env.unwrapped.cfg) is not None
    nominal_goal = None
    if goal_mode:
        nominal = getattr(env.unwrapped.cfg, "robust_nominal_goal", None)
        if nominal is None:
            raise ValueError(
                f"'{task_name}' carries goals through the eval queue but declares no robust_nominal_goal;"
                " eval.py drives the -ROBUSTEVAL / -ROBUSTPLAY bindings, eval_multi_goal.py the -MGEVAL ones."
            )
        nominal_goal = torch.tensor(nominal, dtype=torch.float32, device=queue.device)
    elif any(level["goals"] != eval_utils.NOMINAL for level in levels):
        raise ValueError(f"a 'goals' level needs a task whose eval curriculum carries goals; '{task_name}' does not.")

    pool = None
    if any(level["starts"] != eval_utils.NOMINAL or level["goals"] != eval_utils.NOMINAL for level in levels):
        pool_path = getattr(env.unwrapped.cfg, "feasible_pool_path", None)
        if pool_path is None:
            raise ValueError(
                f"a pool-drawn level needs a task whose cfg names its feasible_pool_path; '{task_name}' does not."
            )
        pool = _load_feasible_pool(pool_path, task_space, queue.device)
        print(f"[INFO]: Feasible pool {pool_path}: {len(pool)} states")

    yaw_index = _yaw_task_index(task_space) if any(level["yaw_init"] is not None for level in levels) else 0
    xyz_index = list(task_space.get_xyz_dimensions())
    # built once, before the checkpoint loop, so every checkpoint sees the same tasks per level
    level_queues = [
        _level_queue(level, queue, pool, nominal_goal, goal_mode, yaw_index, xyz_index) for level in sweep_levels
    ]

    print(
        f"[INFO]: Starting evaluation of {len(pt_files)} models: {[os.path.basename(f) for f in pt_files]} on {task_name} for {n} task combinations."
    )
    if levels:
        print(f"[INFO]: Sweeping {len(levels)} perturbation levels: {[level['name'] for level in levels]}")
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
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt
        model_name = os.path.splitext(os.path.basename(resume_path))[0]

        # create data structures for task queue and result logging
        dims = env.unwrapped.task_space.task_dim()

        # one wandb row per checkpoint so every level shares the model_nr step metric
        log_dict = {}

        for level, level_queue in zip(sweep_levels, level_queues):
            # each level gets its own results folder, laid out like an ordinary eval folder
            level_results_path = results_path if level is None else os.path.join(results_path, level["name"])
            metric_prefix = "" if level is None else f"robust/{level['name']}/"
            if level is not None:
                print(f"[INFO]: Perturbation level '{level['name']}': {eval_utils.describe_perturbation(level)}")

            env.unwrapped.eval_data["queue"] = deque(torch.clone(level_queue))
            env.unwrapped.eval_data["done"] = False
            env.unwrapped.eval_data["initialized"] = False
            env.unwrapped.eval_data["results"] = defaultdict(list) if args_cli.detailed else None
            env.unwrapped.eval_data["rewards"] = []
            # EvalCurriculum records the start of every finished episode, goal command or not
            env.unwrapped.eval_data["starts"] = []
            env.unwrapped.eval_data["finished"] = torch.tensor([0] * env.num_envs, dtype=torch.bool)
            env.unwrapped.eval_data["old_tasks"] = torch.zeros(env.unwrapped.num_envs, *dims, device=env.device)
            if goal_mode:
                env.unwrapped.eval_data["goals"] = []
                env.unwrapped.eval_data["old_goals"] = torch.zeros(env.unwrapped.num_envs, 3, device=env.device)

            if args_cli.store_trajectories:
                env.unwrapped.eval_data["trajectories"] = []

            # run everything in inference mode
            with torch.inference_mode():
                # reset environment
                env.reset()
                obs = env.get_observations()

                # simulate environment
                while simulation_app.is_running():
                    start_time = time.time()

                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                    # push after the policy acted, so recovery starts on the next step like a real push
                    if level is not None and level["push"] is not None:
                        _fire_push(env.unwrapped, level)

                    if env.unwrapped.eval_data["done"]:
                        break

                    # time delay for real-time evaluation
                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)

            # log results etc.
            logs_dict = (
                {"results": env.unwrapped.eval_data["results"]}
                if args_cli.detailed
                else {"rewards": torch.stack(env.unwrapped.eval_data["rewards"], dim=0)}
            )
            # reward column order and per-term band norms for offline consumers
            logs_dict["reward_names"] = list(env.unwrapped.reward_manager._episode_sums.keys())
            logs_dict["reward_band_norms"] = eval_utils.reward_band_norms(env.unwrapped)
            logs_dict["starts"] = torch.stack(env.unwrapped.eval_data["starts"], dim=0)
            logs_dict.update(terrain_meta)
            if goal_mode:
                logs_dict["goals"] = torch.stack(env.unwrapped.eval_data["goals"], dim=0)
                logs_dict["nominal_goal"] = nominal_goal.tolist()
            if level is not None:
                # make the folder self-describing for the offline analysis
                logs_dict["perturbation"] = level
                logs_dict["step_dt"] = dt
                logs_dict["box_footprint"] = _box_footprint(env.unwrapped.cfg)
                logs_dict["feasible_pool_path"] = getattr(env.unwrapped.cfg, "feasible_pool_path", None)

            if args_cli.store_trajectories:
                traj_dict = {
                    "trajectories": env.unwrapped.eval_data.get("trajectories", None),
                    "observations": (
                        env.unwrapped.observation_manager.active_terms["trajectory"]
                        if "trajectory" in env.unwrapped.observation_manager.active_terms
                        else None
                    ),
                }

            # print quick summary of results
            rewards_per_episode = torch.stack(env.unwrapped.eval_data["rewards"], dim=0)
            rewards = rewards_per_episode.mean(dim=0)
            if args_cli.detailed:
                mean_rewards = (torch.tensor(list(env.unwrapped.eval_data["results"].values()))).mean()
            else:
                mean_rewards = rewards.sum()
            print(f"[INFO]: Mean reward: {mean_rewards:.4f}")

            reward_names = list(env.unwrapped.reward_manager._episode_sums.keys())
            # the sparse goal term whose positive episode sum marks a reached goal
            success_term = next((t for t in ("tracking_pos_sparse", "hold_window") if t in reward_names), None)
            if success_term is not None:
                goal_idx = reward_names.index(success_term)
                success_rate = (rewards_per_episode[:, goal_idx] > 0).float().mean().item() * 100
                print(
                    f"[INFO]: Success rate (reached_goal > 0): {success_rate:.1f}% ({int((rewards_per_episode[:, goal_idx] > 0).sum())}/{len(rewards_per_episode)} episodes)"
                )

            if wandb.run is not None:
                log_dict[f"{metric_prefix}mean_reward"] = mean_rewards.item()
                try:
                    model_num = int(model_name.split("_")[-1])
                    log_dict["model_nr"] = model_num
                except ValueError:
                    pass

                if success_term is not None:
                    log_dict[f"{metric_prefix}success_rate"] = success_rate

                # split and log individual reward terms
                if len(reward_names) == len(rewards):
                    for name, val in zip(reward_names, rewards):
                        log_dict[f"{metric_prefix}rewards/{name}"] = val.item()

            eval_utils.save_dist_pickle(level_results_path, model_name, logs_dict)

            if args_cli.store_trajectories:
                # per model, so a multi-checkpoint --models list keeps every dump instead of the last
                eval_utils.save_trajectories_pickle(level_results_path, traj_dict, model_name=model_name)

            # Free memory by clearing and deleting large objects
            del logs_dict
            if args_cli.store_trajectories:
                del traj_dict
            env.unwrapped.eval_data.clear()

        if wandb.run is not None:
            wandb.log(log_dict)

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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
