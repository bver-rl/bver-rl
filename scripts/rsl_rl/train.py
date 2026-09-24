# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import experiment_config  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length", type=int, default=None, help="Length of the recorded video (in steps). Default: 200."
)
parser.add_argument(
    "--video_interval", type=int, default=None, help="Interval between video recordings (in steps). Default: 2000."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Must run before AppLauncher so --video / --enable_cameras from the YAML take effect.
_experiment_data, _experiment_meta_for_run, _sweep_overrides, _sweep_grid_index = (
    experiment_config.parse_and_apply_experiment_config(args_cli, hydra_args)
)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import time
import torch
import pickle
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rsl_tasks  # noqa: F401 # isort:skip

# import logger
logger = logging.getLogger(__name__)

import informed_exploration.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Set here since some randomizations happen at environment initialization.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Run directory: {time-stamp}_{run_name}
    # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    # The Ray Tune workflow parses the experiment name from the line below; do not change it.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # initialize rerun logging if requested
    if args_cli.rerun:
        from informed_exploration.tasks.manager_based.informed_exploration.utils import rerun_logger
        rerun_logger.initialize(
            mode=args_cli.rerun_mode,
            save_path=os.path.join(log_dir, "rerun.rrd"),
        )
        print(f"[INFO] Rerun logging enabled (mode={args_cli.rerun_mode}).")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, log_dir=log_dir, render_mode="rgb_array" if args_cli.video else None)

    # log static start / goal / terrain-box markers once after env creation
    if args_cli.rerun:
        import rerun_markers
        rerun_markers.log_env_markers(env.unwrapped)
        # Terrain underlay so markers and samples sit on the terrain.
        if args_cli.rerun_terrain:
            rerun_markers.log_env_terrain(env.unwrapped)

    # Same static markers in the Isaac Sim viewport.
    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    if args_cli.env_markers:
        import isaaclab_markers
        # Stored on the env to keep the handle alive.
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        # Argparse defaults are None to allow YAML override.
        video_length = args_cli.video_length if args_cli.video_length is not None else 200
        video_interval = args_cli.video_interval if args_cli.video_interval is not None else 2000
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Must precede the wrapper, whose reset makes curricula lock their env split on first compute.
    env.unwrapped.train_env_ratio = getattr(agent_cfg, "train_env_ratio", 1.0)
    env.unwrapped.random_env_ratio = getattr(agent_cfg, "random_env_ratio", 0.0)
    env.unwrapped.random_reachability_ratio = getattr(agent_cfg, "random_reachability_ratio", 0.0)
    env.unwrapped.random_backward_ratio = getattr(agent_cfg, "random_backward_ratio", 0.0)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # patch wandb writer with experiment metadata (group/tags/notes) before runner construction
    if _experiment_data is not None and agent_cfg.logger == "wandb":
        experiment_config.patch_wandb_writer(_experiment_meta_for_run or _experiment_data["experiment"])

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # store full experiment YAML in wandb config and log directory
    if _experiment_data is not None:
        dump_yaml(os.path.join(log_dir, "params", "experiment.yaml"), _experiment_data["raw"])
        if agent_cfg.logger == "wandb":
            experiment_config.store_experiment_config_in_wandb(_experiment_data["raw"])

    # store sweep metadata (if any) so each run is fully self-describing
    if _sweep_overrides:
        assert _experiment_data is not None
        sweep_info = {
            "sweep_index": args_cli.sweep_index,
            "grid_index": _sweep_grid_index,
            "grid_size": experiment_config.get_sweep_grid_size(_experiment_data),
            "overrides": _sweep_overrides,
        }
        dump_yaml(os.path.join(log_dir, "params", "sweep.yaml"), sweep_info)
        if agent_cfg.logger == "wandb":
            try:
                import wandb

                if wandb.run is not None:
                    wandb.config.update({"sweep": sweep_info}, allow_val_change=True)
            except Exception:
                pass

    # Env-split ratios are mirrored before the wrapper; this one is not needed at the first reset.
    env.unwrapped.num_steps_per_env = runner.cfg["num_steps_per_env"]

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    # print(f"trajectory indices: {env.unwrapped.eval_data['traj_index'] if hasattr(env.unwrapped, 'eval_data') else 'N/A'}") # print trajectory indices for debugging
    # print(f"last rewards: {env.unwrapped.eval_data['last_rewards'] if hasattr(env.unwrapped, 'eval_data') else 'N/A'}") # print last rewards for debugging
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
