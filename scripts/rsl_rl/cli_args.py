# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # experiment config
    arg_group.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to a YAML experiment config file. When provided, launch-time parameters "
        "(num_envs, max_iterations, seed, logger, video, etc.) and wandb metadata (group, tags, notes) "
        "are loaded from this file instead of being specified via CLI flags.",
    )
    arg_group.add_argument(
        "--sweep_index",
        type=int,
        default=None,
        help="Zero-based index into the sweep grid defined in the experiment YAML. "
        "Each index maps deterministically to one combination of swept hyperparameters. "
        "Designed for SLURM array jobs: --sweep_index=$SLURM_ARRAY_TASK_ID.",
    )
    # rerun.io logging
    arg_group.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Enable rerun.io logging of exploration geometry and training metrics.",
    )
    arg_group.add_argument(
        "--rerun_mode",
        type=str,
        default="save",
        choices={"save", "serve"},
        help="Rerun output mode: 'save' writes a .rrd file for later replay (default); "
        "'serve' also streams live via gRPC (open VS Code 'General - Rerun Viewer (Live)').",
    )
    arg_group.add_argument(
        "--rerun_metrics",
        action="store_true",
        default=False,
        help="Mirror RSL-RL training scalars (losses, rewards, etc.) into the rerun recording.",
    )
    arg_group.add_argument(
        "--rerun_terrain",
        action="store_true",
        default=True,
        help="Log the regenerated terrain mesh as a static underlay in the rerun recording so the "
        "markers / curriculum samples sit on the terrain instead of floating (requires --rerun).",
    )
    arg_group.add_argument(
        "--env_markers",
        action="store_true",
        default=False,
        help="Draw semi-transparent start-zone / goal / terrain-box / task-space-bounds markers "
        "in the Isaac Sim viewport (mirrors the rerun static markers).",
    )
    arg_group.add_argument(
        "--hide_walls",
        action="store_true",
        default=False,
        help="Stop drawing the terrain's arena fence, so it does not stand between the camera and "
        "the terrain. Rendering only: the fence keeps its collider and stays in the height scan.",
    )
    arg_group.add_argument(
        "--path_tracing",
        action="store_true",
        default=False,
        help="Render with path tracing instead of the default real-time renderer (higher quality, "
        "slower; mainly useful for recorded videos). Requires --enable_cameras / --video.",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    # propagate rerun flags so the runner cfg dict carries them to the wandb writer
    if hasattr(args_cli, "rerun") and args_cli.rerun:
        agent_cfg.rerun = True
    if hasattr(args_cli, "rerun_metrics") and args_cli.rerun_metrics:
        agent_cfg.rerun_metrics = True

    return agent_cfg
