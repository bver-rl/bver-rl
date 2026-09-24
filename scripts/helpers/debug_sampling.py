# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to debug sampling regions of tasks for RL environments."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import os
import sys

# cli_args.py lives with the entry points in scripts/rsl_rl/; put that directory on the path (one-way dependency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "rsl_rl"))

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Debug Sampling Tasks.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Stub to match play arguments.")
parser.add_argument(
    "--filtered_samples", action="store_true", default=False, help="Filter the generated random task samples."
)
parser.add_argument(
    "--pool_path",
    type=str,
    default=None,
    help="Path to a feasible-starts .pkl pool file. When set, samples are drawn from the pool instead of uniformly.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import informed_exploration.tasks  # noqa: F401
from informed_exploration.tasks.manager_based.informed_exploration.curriculum.curriculum_handlers import (
    random_curriculum,
)
from informed_exploration.tasks.manager_based.informed_exploration.curriculum.task_space.samplers import (
    FeasiblePoolSamplerCfg,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
):
    """Debug task sampling interactively."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disable randomizations if possible to focus on just task sampling
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand = None

    print(f"[INFO] Creating environment: {args_cli.task}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, log_dir="logs/debug", render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    core_env = env.unwrapped

    print("[INFO] Starting sampling loop. The task targets are visualized as red spheres.")
    obs = env.reset()

    # resample all envs every step
    env_ids = torch.arange(core_env.num_envs, device=core_env.device)

    step_counter = 0

    sampler_cfg = FeasiblePoolSamplerCfg(pool_path=args_cli.pool_path) if args_cli.pool_path else None

    env.reset()

    while simulation_app.is_running():
        random_curriculum(core_env, env_ids, debug_viz=True, filtered_samples=args_cli.filtered_samples, sampler_cfg=sampler_cfg)

        actions = torch.zeros((core_env.num_envs, core_env.action_space.shape[1]), device=core_env.device)
        obs, _, dones, _, log = env.step(actions)

        step_counter += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
