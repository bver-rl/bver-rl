# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-goal evaluation of a goal-conditioned RSL-RL policy checkpoint.

Like eval.py, but goals are sampled per episode; needs an *_MGPLAY task (no ``default_goal``, eval
curriculum with ``goal_command_name``). ``--corridor_goals`` places goals inside DTSG's walled routes
instead, so per-corridor success reads out which routes a policy can cross (pair with *-CORRIDOR-v0).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import json

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Multi-goal evaluation of an RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--store_trajectories",
    action="store_true",
    default=False,
    help="Not supported by *_MGPLAY tasks (they omit TrajectoryObservationsCfg); passing this errors out.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_samples", type=int, default=None, help="Number of (start, goal) pairs to evaluate."
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
    "--detailed", action="store_true", default=False, help="Wether to store task-reward pairs or just rewards."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--enable_wandb", action="store_true", default=False, help="Enable wandb logging.")
parser.add_argument(
    "--filter_goals",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Reject sampled goals inside the box / floating in front of it (task_space.goal_sample_filter)."
        " Use --no-filter_goals to sample raw over the goal bounds instead."
    ),
)
parser.add_argument(
    "--goal_z_band",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help=(
        "Replace each sampled goal's z with terrain_height(x, y) + U[LO, HI], via a downward raycast"
        " against the terrain mesh (e.g. 0.55 0.65 keeps goals in a tight band around standing height,"
        " so every goal is attainable). Default: z stays uniform over the task-space z-bounds. Falls"
        " back to the eval config yaml's goal_z_band key if set there."
    ),
)
parser.add_argument(
    "--corridor_goals",
    action="store_true",
    default=False,
    help=(
        "DTSG only. Place goals uniformly at random INSIDE the three routes (stairs / pillars / ramp),"
        " one third each, instead of sampling the whole task space. Since every route is walled over"
        " its full length, reaching such a goal requires traversing that route, so the per-corridor"
        " success rate reads out which routes a policy can actually cross, the BVER-vs-PPO"
        " comparison this mode exists for. Pair with a *-CORRIDOR-v0 task, whose sealed goal column"
        " closes the 'enter the corridor backwards from its exit' loophole. Falls back to the eval"
        " config yaml's corridor_goals key if set there."
    ),
)
parser.add_argument(
    "--corridor_wall_clearance",
    type=float,
    default=None,
    help=(
        "--corridor_goals: inset (m) from the separator walls on both sides of a route, so a goal is"
        " never so close to a wall that no base pose can occupy it. Default 0.3; falls back to the"
        " eval config yaml's key of the same name."
    ),
)
parser.add_argument(
    "--corridor_end_margin",
    type=float,
    default=None,
    help=(
        "--corridor_goals: inset (m) from the route's two open ends. Unlike the sides these are"
        " mouths, not walls: a goal right at the mouth could be satisfied within the 0.25 m success"
        " tolerance by a robot still standing on the adjoining platform column, which would score a"
        " traversal that never happened. Default 0.5 (2x the tolerance); yaml-overridable."
    ),
)
parser.add_argument(
    "--corridor_z_offsets",
    type=json.loads,
    default=None,
    help=(
        "--corridor_goals: JSON map of corridor name -> how far (m) above the local reference height its"
        " goals sit, replacing --goal_z_band for those corridors. A [lo, hi] pair gives that corridor"
        ' its own band, a scalar pins a fixed rise, e.g. \'{"pillars": [0.2, 0.3], "ramp":'
        ' [0.35, 0.45]}\' (the default). Corridors not listed keep the band. yaml-overridable.'
    ),
)
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
from informed_exploration.tasks.manager_based.informed_exploration.mdp.commands import CurriculumGoalCfg
from informed_exploration.tasks.manager_based.informed_exploration.utils.curriculum_utils import make_sampler
from informed_exploration.terrains.config.layouts import dtsg_route_rects, dtsg_route_top_z


def _goal_z_above_terrain(env, goal_xy_e: torch.Tensor, z_band: tuple[float, float]) -> torch.Tensor:
    """Per-goal z as the terrain surface height at (x, y) plus U[z_band], via :func:`eval_utils.surface_z_at`."""
    surface_z_e = eval_utils.surface_z_at(env, goal_xy_e)
    lo, hi = z_band
    return surface_z_e + lo + (hi - lo) * torch.rand(surface_z_e.shape[0], device=surface_z_e.device)



CORRIDOR_Z_OFFSETS = {"pillars": [0.2, 0.3], "ramp": [0.35, 0.45]}
"""Per-corridor goal rise (m) above the route's reference height, as a ``[lo, hi]`` band or a scalar.

Unlisted corridors use ``--goal_z_band``; these sit lower because their reference geometry is raised.
"""

CORRIDOR_DEFAULTS = {
    "corridor_wall_clearance": 0.3,
    "corridor_end_margin": 0.5,
    "corridor_z_offsets": CORRIDOR_Z_OFFSETS,
}
"""Corridor knob defaults, applied when neither the CLI nor the eval yaml set them."""




def _corridor_z_rise_range(name: str, args) -> tuple[float, float]:
    """``(lo, hi)`` metres above the slice reference height that this corridor's goal z is drawn from.

    Uses the corridor's ``--corridor_z_offsets`` entry when set, else the global ``--goal_z_band``.
    """
    override = args.corridor_z_offsets.get(name)
    if override is None:
        return float(args.goal_z_band[0]), float(args.goal_z_band[1])
    if isinstance(override, (int, float)):
        return float(override), float(override)
    assert len(override) == 2, (
        f"--corridor_z_offsets['{name}'] = {override}: expected a [lo, hi] pair (that corridor's own"
        " band) or a scalar (a fixed rise)."
    )
    return float(override[0]), float(override[1])


def _corridor_sample_regions(corridor_rects: dict, args, xy_bounds) -> dict:
    """The xy rectangles corridor goals are drawn from.

    Each route footprint is inset by the wall clearance and end margin, then intersected with the task-space bounds.
    """
    assert args.goal_z_band is not None, (
        "--corridor_goals needs --goal_z_band: the routes rise up to ~0.6 m above the floor-level"
        " platforms, so a task-space-uniform z would put most corridor goals in unreachable air."
    )
    assert args.corridor_wall_clearance > 0.0, (
        "--corridor_wall_clearance must be positive, or goals land flush against the separator walls"
        " where no base pose can occupy them."
    )

    regions = {}
    for name in sorted(corridor_rects):  # sorted: a fixed seed reproduces the round-robin assignment
        x_lo, x_hi, y_lo, y_hi = corridor_rects[name]
        x_lo, x_hi = x_lo + args.corridor_end_margin, x_hi - args.corridor_end_margin
        y_lo, y_hi = y_lo + args.corridor_wall_clearance, y_hi - args.corridor_wall_clearance
        # intersect rather than assert: outer routes coincide with the fence standoff up to float noise
        x_lo, x_hi = max(x_lo, float(xy_bounds[0][0])), min(x_hi, float(xy_bounds[0][1]))
        y_lo, y_hi = max(y_lo, float(xy_bounds[1][0])), min(y_hi, float(xy_bounds[1][1]))
        assert x_lo < x_hi and y_lo < y_hi, (
            f"corridor '{name}' collapses under the requested insets (end_margin"
            f" {args.corridor_end_margin}, wall_clearance {args.corridor_wall_clearance}) intersected"
            f" with the task-space xy bounds {xy_bounds.tolist()}."
        )
        regions[name] = (x_lo, x_hi, y_lo, y_hi)
    return regions


def _corridor_goal_samples(n: int, regions: dict, route_top_z: dict, args, device) -> tuple[torch.Tensor, list[str]]:
    """``n`` goals drawn uniformly inside the DTSG routes, balanced across routes, with their labels.

    z is the route's config-derived top surface raised by :func:`_corridor_z_rise_range`.
    """
    names = sorted(regions)
    # round-robin, so every corridor gets ~n/3 goals rather than a lucky multinomial draw
    labels = [names[i % len(names)] for i in range(n)]
    goals = torch.empty(n, 3, device=device)

    for name in names:
        idx = torch.tensor([i for i, label in enumerate(labels) if label == name], device=device)
        x_lo, x_hi, y_lo, y_hi = regions[name]
        rise_lo, rise_hi = _corridor_z_rise_range(name, args)

        xy = torch.rand(idx.numel(), 2, device=device)
        xy[:, 0] = x_lo + (x_hi - x_lo) * xy[:, 0]
        xy[:, 1] = y_lo + (y_hi - y_lo) * xy[:, 1]

        goals[idx, :2] = xy
        goals[idx, 2] = route_top_z[name] + rise_lo + (rise_hi - rise_lo) * torch.rand(idx.numel(), device=device)

    return goals, labels


def _corridor_marker_boxes(regions: dict, route_top_z: dict, args) -> dict:
    """``{name: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)}``: the exact sampling volume of each route."""
    boxes = {}
    for name, (x_lo, x_hi, y_lo, y_hi) in regions.items():
        rise_lo, rise_hi = _corridor_z_rise_range(name, args)
        boxes[name] = (x_lo, x_hi, y_lo, y_hi, route_top_z[name] + rise_lo, route_top_z[name] + rise_hi)
    return boxes


def _label_by_corridor(goals_xy, corridor_rects: dict) -> list[str]:
    """Corridor name per goal by point-in-rect test; ``"other"`` outside every route."""
    labels = []
    for x, y in goals_xy:
        inside = (
            name
            for name, (x_lo, x_hi, y_lo, y_hi) in corridor_rects.items()
            if x_lo <= x <= x_hi and y_lo <= y <= y_hi
        )
        labels.append(next(inside, "other"))
    return labels


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Multi-goal evaluation with an RSL-RL agent."""
    if args_cli.store_trajectories:
        raise ValueError(
            "--store_trajectories is not supported here: *_MGPLAY tasks omit TrajectoryObservationsCfg"
            " (same as their single-goal *_PLAY counterparts). Use eval.py for trajectory capture."
        )

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # resolve the eval config first: it fills args_cli fields (seed, num_envs, ...) used below
    eval_cfg = eval_utils.resolve_eval_config(args_cli, env_cfg)
    eval_utils.init_wandb_from_eval_config(args_cli, eval_cfg)
    for key, value in CORRIDOR_DEFAULTS.items():  # neither the CLI nor the yaml set these
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

    # fail loudly on anything but goal-conditioned *_MGPLAY tasks
    base_position_cfg = env_cfg.commands.base_position
    if not isinstance(base_position_cfg, CurriculumGoalCfg):
        raise ValueError(
            f"--task {args_cli.task} does not use a CurriculumGoalCfg for commands.base_position (got"
            f" {type(base_position_cfg).__name__}); eval_multi_goal.py only supports goal-conditioned *_MGPLAY"
            " tasks (e.g. Parkour-ClimbSingleBox-FwdCURE-MGPLAY-v0)."
        )
    if base_position_cfg.default_goal is not None:
        raise ValueError(
            f"--task {args_cli.task} sets commands.base_position.default_goal="
            f"{base_position_cfg.default_goal}. CurriculumGoalCommand._resample_command rewrites the goal to"
            " this value on EVERY reset, which would silently overwrite every sampled goal. Use an *_MGPLAY"
            " task, which clears default_goal."
        )
    goal_command_name = getattr(getattr(env_cfg.curriculum, "evaluation", None), "goal_command_name", None)
    if goal_command_name is None:
        raise ValueError(
            f"--task {args_cli.task}'s curriculum.evaluation.goal_command_name is not set; eval_multi_goal.py"
            " only supports *_MGPLAY tasks (ClimbBoxMultiGoalEvalCurriculumCfg)."
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

    # sample (start, goal) pairs for evaluation, once, so every checkpoint sees the same pairs
    task_space = getattr(env.unwrapped, "task_space", None)
    if task_space is None:
        raise ValueError("The environment does not have a task space defined for evaluation.")

    n = args_cli.num_samples
    device = task_space.task_bounds(eval_bounds=True).device
    dims = task_space.task_dim()

    # starts: same eval_bounds distribution as eval.py's single-goal sweep (unfiltered)
    start_sampler = make_sampler(None, False, None, task_space, device, eval_bounds=True)
    starts = start_sampler.sample(n).reshape(n, -1)

    if args_cli.corridor_goals:
        # goals inside the DTSG routes; no task-space filter since each route rect is already walkable
        sub_terrain = _only_sub_terrain(env.unwrapped.cfg.scene.terrain.terrain_generator)
        corridor_rects = dtsg_route_rects(sub_terrain.layout, sub_terrain.size)
        route_top_z = dtsg_route_top_z(sub_terrain.layout, sub_terrain.size, env.unwrapped.cfg.pin_difficulty)
        # fence-clamped training xy bounds, the same ones the plain multi-goal draw uses
        xy_bounds = task_space.task_bounds(eval_bounds=False)[list(task_space.get_xyz_dimensions())[:2]]
        corridor_regions = _corridor_sample_regions(corridor_rects, args_cli, xy_bounds)
        goals, corridor_labels = _corridor_goal_samples(n, corridor_regions, route_top_z, args_cli, device)
        # publish the sampling volumes for the viewport markers
        env.unwrapped._corridor_regions = _corridor_marker_boxes(corridor_regions, route_top_z, args_cli)
    else:
        # goals uniform over the training xyz bounds, optionally filtered by goal_sample_filter (not sample_filter)
        corridor_rects = corridor_regions = route_top_z = None
        xyz_dims = list(task_space.get_xyz_dimensions())
        total = task_space.total_task_dim()
        goal_raw = torch.rand(n, 1, total, device=device)
        goals_full = task_space.scale_to_bounds(goal_raw, eval_bounds=False, subspace=xyz_dims)
        if args_cli.filter_goals:
            goals_full = task_space.filter_tasks(
                goals_full, eval_bounds=False, subspace=xyz_dims, filter_override=task_space.get_goal_sample_filter()
            )
        goals = goals_full.reshape(n, -1)[:, xyz_dims]

        # optional: snap z to a band above the local terrain surface, replacing the uniform z draw
        if args_cli.goal_z_band is not None:
            goals[:, 2] = _goal_z_above_terrain(env.unwrapped, goals[:, :2], args_cli.goal_z_band)

    # 39-dim pairs goal_xyz(3) ++ start_state.flatten()(36), the EvalCurriculum goal-branch layout
    pairs = torch.cat([goals, starts], dim=-1)

    # draw markers only now, after the corridor regions are published
    import render_utils

    render_utils.hide_walls_if_requested(env, args_cli)

    if args_cli.env_markers:
        import isaaclab_markers

        # store the handle on the env so it is kept alive (and re-assertable via refresh())
        env.unwrapped._env_markers = isaaclab_markers.draw_env_markers(env.unwrapped)

    if args_cli.corridor_goals:
        counts = {name: corridor_labels.count(name) for name in sorted(corridor_rects)}
        print(
            f"[INFO]: Starting CORRIDOR evaluation of {len(pt_files)} models:"
            f" {[os.path.basename(f) for f in pt_files]} on {task_name} for {n} (start, goal) pairs"
            f" (goals per corridor: {counts}, wall_clearance={args_cli.corridor_wall_clearance},"
            f" end_margin={args_cli.corridor_end_margin}, goal_z_band={args_cli.goal_z_band},"
            f" z_offsets={args_cli.corridor_z_offsets},"
            f" seal_goal_column={getattr(env.unwrapped.cfg, 'seal_goal_column', None)})."
        )
    else:
        print(
            f"[INFO]: Starting multi-goal evaluation of {len(pt_files)} models:"
            f" {[os.path.basename(f) for f in pt_files]} on {task_name} for {n} (start, goal) pairs"
            f" (filter_goals={args_cli.filter_goals}, goal_z_band={args_cli.goal_z_band})."
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

        # create data structures for the (start, goal) queue and result logging
        env.unwrapped.eval_data["queue"] = deque(torch.clone(pairs))
        env.unwrapped.eval_data["done"] = False
        env.unwrapped.eval_data["initialized"] = False
        env.unwrapped.eval_data["results"] = defaultdict(list) if args_cli.detailed else None
        env.unwrapped.eval_data["rewards"] = []
        env.unwrapped.eval_data["goals"] = []
        env.unwrapped.eval_data["starts"] = []
        env.unwrapped.eval_data["finished"] = torch.tensor([0] * env.num_envs, dtype=torch.bool)
        env.unwrapped.eval_data["old_tasks"] = torch.zeros(env.unwrapped.num_envs, *dims, device=env.device)
        env.unwrapped.eval_data["old_goals"] = torch.zeros(env.unwrapped.num_envs, 3, device=env.device)

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

                if env.unwrapped.eval_data["done"]:
                    break

                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        # log results etc., row-aligned with "rewards"/"results", so no dict keyed by float tensors
        logs_dict = (
            {"results": env.unwrapped.eval_data["results"]}
            if args_cli.detailed
            else {"rewards": torch.stack(env.unwrapped.eval_data["rewards"], dim=0)}
        )
        logs_dict["goals"] = torch.stack(env.unwrapped.eval_data["goals"], dim=0)
        logs_dict["starts"] = torch.stack(env.unwrapped.eval_data["starts"], dim=0)
        # geometry and column order so the offline analyzer needs no Isaac imports
        logs_dict["reward_names"] = list(env.unwrapped.reward_manager._episode_sums.keys())
        # per-term norms that turn a raw episodic sum into BVER's [0, 1] r_min/r_max fraction
        logs_dict["reward_band_norms"] = eval_utils.reward_band_norms(env.unwrapped)
        logs_dict["corridor_goals"] = args_cli.corridor_goals
        logs_dict["corridor_rects"] = corridor_rects
        logs_dict["corridor_regions"] = corridor_regions
        logs_dict["corridor_route_top_z"] = route_top_z
        logs_dict["corridor_z_offsets"] = args_cli.corridor_z_offsets if args_cli.corridor_goals else None
        logs_dict["seal_goal_column"] = getattr(env.unwrapped.cfg, "seal_goal_column", None)
        logs_dict["geometry_bounds"] = getattr(
            _only_sub_terrain(env.unwrapped.cfg.scene.terrain.terrain_generator), "geometry_bounds", None
        )

        model_name = os.path.splitext(os.path.basename(resume_path))[0]

        # print quick summary of results, identical to eval.py's, so runs stay comparable
        rewards_per_episode = torch.stack(env.unwrapped.eval_data["rewards"], dim=0)
        rewards = rewards_per_episode.mean(dim=0)
        if args_cli.detailed:
            mean_rewards = (torch.tensor(list(env.unwrapped.eval_data["results"].values()))).mean()
        else:
            mean_rewards = rewards.sum()
        print(f"[INFO]: Mean reward: {mean_rewards:.4f}")

        reward_names = list(env.unwrapped.reward_manager._episode_sums.keys())
        success_term = "tracking_pos_sparse"
        per_corridor: dict[str, float] = {}
        if success_term in reward_names:
            goal_idx = reward_names.index(success_term)
            reached = rewards_per_episode[:, goal_idx] > 0
            success_rate = reached.float().mean().item() * 100
            print(
                f"[INFO]: Success rate (reached_goal > 0): {success_rate:.1f}%"
                f" ({int(reached.sum())}/{len(rewards_per_episode)} episodes)"
            )

            # per-route breakdown labelled from each episode's recorded goal, not the sampling-time label
            if corridor_rects is not None:
                episode_labels = _label_by_corridor(logs_dict["goals"][:, :2].tolist(), corridor_rects)
                for name in sorted(set(episode_labels)):
                    mask = torch.tensor([label == name for label in episode_labels], device=reached.device)
                    per_corridor[name] = reached[mask].float().mean().item() * 100
                    print(
                        f"[INFO]:   corridor {name}: {per_corridor[name]:.1f}%"
                        f" ({int(reached[mask].sum())}/{int(mask.sum())} episodes)"
                    )

        if wandb.run is not None:
            log_dict = {"mean_reward": mean_rewards.item()}
            try:
                model_num = int(model_name.split("_")[-1])
                log_dict["model_nr"] = model_num
            except ValueError:
                pass

            if success_term in reward_names:
                log_dict["success_rate"] = success_rate
                for name, rate in per_corridor.items():
                    log_dict[f"success_rate/{name}"] = rate

            # split and log individual reward terms
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
