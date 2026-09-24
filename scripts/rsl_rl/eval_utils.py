# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the scripts/rsl_rl eval scripts; does no argparse or AppLauncher work at import."""

from __future__ import annotations

import gzip
import os
import pickle

import math

import wandb
import yaml

# must match RobustEvalEventCfg: the push term sits under a mode Isaac Lab never auto-fires
PERTURB_MODE = "perturb"
PUSH_TERM = "push_robot"

CONTROL_LEVEL = "control"
"""Name of the implicit unperturbed level every sweep is measured against."""

_LEVEL_KEYS = {"name", "push", "yaw_init", "alternate_sign", "starts", "goals", "num_samples"}
_PUSH_KEYS = {"vx", "vy", "vz", "t"}
_DRAW_SOURCES = {"pool"}
"""Where a level may draw its starts or goals from, besides the nominal eval distribution."""
NOMINAL = "nominal"


def reward_band_norms(env) -> dict[str, float]:
    """Per-term achievable episodic reward ``{term_name: norm}`` over the positive-weight terms.

    Dividing a term's episodic sum by its norm gives BVER's ``[0, 1]`` ``r_min``/``r_max`` fraction,
    mirroring ``BVERCurriculum._compute_reward_band_norm`` per term; windowed terms use their ``duration``.
    """
    rm = env.reward_manager
    episode_s = env.max_episode_length_s
    norms: dict[str, float] = {}
    for name in rm.active_terms:
        term = rm.get_term_cfg(name)
        if term.weight <= 0.0:  # penalties add no achievable reward
            continue
        duration = (term.params or {}).get("duration")
        norms[name] = float(term.weight * (episode_s if duration is None else min(duration, episode_s) / duration))
    return norms


def resolve_eval_config(args_cli, env_cfg) -> dict | None:
    """Resolve ``--eval_config`` into ``args_cli`` fields and nested ``env.*`` overrides on ``env_cfg``.

    Must run before anything is derived from ``args_cli``. The yaml only fills settings the CLI left unset,
    and the run comes from ``--sweep_index`` into ``runs:`` or an explicit ``--checkpoint_folder``.

    Returns:
        The loaded yaml dict, or ``None`` if ``--eval_config`` wasn't given.
    """
    eval_cfg = None
    if args_cli.eval_config is not None:
        with open(args_cli.eval_config, "r") as f:
            eval_cfg = yaml.safe_load(f)
        if args_cli.sweep_index is not None:
            args_cli.checkpoint_folder = eval_cfg["runs"][args_cli.sweep_index]
        elif args_cli.checkpoint_folder is None:
            raise ValueError(
                "--eval_config needs either --sweep_index (take the run from the yaml's `runs:` list,"
                " what run_sweep.py does) or an explicit --checkpoint_folder (evaluate one folder"
                " with the yaml supplying the eval knobs). Neither was given."
            )
        # seed must be set before `cli_args.update_rsl_rl_cfg` copies it onto agent_cfg
        if args_cli.seed is None:
            args_cli.seed = eval_cfg.get("seed")
        if args_cli.num_envs is None:
            args_cli.num_envs = eval_cfg.get("num_envs")
        if args_cli.num_samples is None:
            args_cli.num_samples = eval_cfg.get("num_samples")
        if args_cli.models is None:
            args_cli.models = eval_cfg.get("models")
        if args_cli.results_suffix is None:
            args_cli.results_suffix = eval_cfg.get("results_suffix")
        # eval_multi_goal.py only; eval.py's parser has no --goal_z_band, hence the hasattr guard
        if hasattr(args_cli, "goal_z_band") and args_cli.goal_z_band is None:
            args_cli.goal_z_band = eval_cfg.get("goal_z_band")
        # corridor knobs; the store_true flag can only be turned on by the yaml
        if hasattr(args_cli, "corridor_goals"):
            args_cli.corridor_goals = args_cli.corridor_goals or bool(eval_cfg.get("corridor_goals", False))
            for key in ("corridor_wall_clearance", "corridor_end_margin", "corridor_z_offsets"):
                if getattr(args_cli, key) is None:
                    setattr(args_cli, key, eval_cfg.get(key))
        # store_true flag, so the yaml can only turn it on
        if hasattr(args_cli, "store_trajectories"):
            args_cli.store_trajectories = args_cli.store_trajectories or bool(eval_cfg.get("store_trajectories", False))
        # the start-side evals only; same "yaml fills what the CLI left unset" rule
        for key in ("start_pool", "min_per_route", "platform_margin", "start_yaw_range", "start_height"):
            if hasattr(args_cli, key) and getattr(args_cli, key) is None:
                setattr(args_cli, key, eval_cfg.get(key))
        # the robustness eval only (eval.py); see normalize_perturbations for the entry schema
        if hasattr(args_cli, "perturbations") and args_cli.perturbations is None:
            args_cli.perturbations = eval_cfg.get("perturbations")

        # dynamically apply any nested environment parameters from the evaluation yaml
        for key, value in eval_cfg.items():
            if key.startswith("env."):
                parts = key.split(".")[1:]
                _obj = env_cfg
                for part in parts[:-1]:
                    _obj = getattr(_obj, part)
                setattr(_obj, parts[-1], value)

    # fall back to default for num_samples if still unset
    if args_cli.num_samples is None:
        args_cli.num_samples = 3

    return eval_cfg


def normalize_perturbations(levels, step_dt: float, max_steps: int | None = None) -> list[dict]:
    """Validate the eval yaml's ``perturbations:`` list into one resolved dict per sweep level.

    Each level is a full pass over the eval queue. Keys: ``name``, ``push`` (exact root velocity delta
    ``vx``/``vy``/``vz`` fired at ``t`` s), ``yaw_init`` (rad), ``alternate_sign`` (mirror lateral push and
    yaw on odd envs), ``starts``/``goals`` (``pool`` draws from the feasible pool), ``num_samples`` (pool
    levels only). A ``control`` level is prepended if missing.

    Args:
        levels: The raw ``perturbations:`` list, or ``None``.
        step_dt: Environment step duration (s).
        max_steps: Episode length in steps, used to reject pushes that would never fire.

    Returns:
        The resolved levels, or ``[]`` for ``None`` (an ordinary eval).
    """
    if levels is None:
        return []
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"'perturbations' must be a non-empty list of level dicts; got {levels!r}")

    resolved: list[dict] = []
    for entry in levels:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"every perturbation level needs a 'name'; got {entry!r}")
        name = str(entry["name"])
        unknown = set(entry) - _LEVEL_KEYS
        if unknown:
            raise ValueError(f"perturbation '{name}': unknown key(s) {sorted(unknown)}; allowed: {sorted(_LEVEL_KEYS)}")
        if os.sep in name or "/" in name:
            raise ValueError(f"perturbation '{name}': the name becomes a results sub-folder, so it cannot be a path")

        alternate_sign = bool(entry.get("alternate_sign", False))

        push = entry.get("push")
        if push is not None:
            if not isinstance(push, dict):
                raise ValueError(f"perturbation '{name}': 'push' must be a mapping of {sorted(_PUSH_KEYS)}")
            unknown = set(push) - _PUSH_KEYS
            if unknown:
                raise ValueError(f"perturbation '{name}': unknown push key(s) {sorted(unknown)}")
            if "t" not in push:
                raise ValueError(f"perturbation '{name}': 'push' needs 't', the time into the episode it fires")
            t = float(push["t"])
            if t < 0.0:
                raise ValueError(f"perturbation '{name}': push t={t} is negative")
            velocity = {axis: float(push.get(f"v{axis}", 0.0)) for axis in ("x", "y", "z")}
            if not any(velocity.values()):
                raise ValueError(
                    f"perturbation '{name}': 'push' sets no velocity, so it is the control level with extra"
                    " steps. Give it a vx / vy / vz, or drop the 'push' key."
                )
            step = int(round(t / step_dt))
            if max_steps is not None and step >= max_steps:
                raise ValueError(
                    f"perturbation '{name}': push t={t}s is step {step}, at or past the {max_steps}-step"
                    f" episode ({max_steps * step_dt:.2f}s), so it would never fire."
                )
            push = {
                "t": t,
                "step": step,
                # lo == hi expresses an exact delta-v through the uniform sampler
                "velocity_range": {axis: (value, value) for axis, value in velocity.items()},
            }

        yaw_init = entry.get("yaw_init")
        if yaw_init is not None:
            yaw_init = float(yaw_init)
            if abs(yaw_init) > math.pi:
                raise ValueError(f"perturbation '{name}': yaw_init {yaw_init} rad is outside +-pi")

        starts, goals = entry.get("starts", NOMINAL), entry.get("goals", NOMINAL)
        for key, value in (("starts", starts), ("goals", goals)):
            if value != NOMINAL and value not in _DRAW_SOURCES:
                raise ValueError(f"perturbation '{name}': {key}={value!r} is not one of {sorted(_DRAW_SOURCES)}")
        if starts != NOMINAL and yaw_init is not None:
            raise ValueError(
                f"perturbation '{name}': 'yaw_init' rewrites the heading of a nominal start, but a pool"
                " start is a settled state whose heading is part of it. Use one or the other."
            )
        num_samples = entry.get("num_samples")
        if num_samples is not None:
            num_samples = int(num_samples)
            if num_samples <= 0:
                raise ValueError(f"perturbation '{name}': num_samples must be positive; got {num_samples}")
            if starts == NOMINAL and goals == NOMINAL:
                raise ValueError(
                    f"perturbation '{name}': 'num_samples' only applies to a pool-drawn level; a fixed level"
                    " reuses the shared nominal draw so it stays comparable to the control."
                )

        if alternate_sign and not (yaw_init or (push and push["velocity_range"]["y"][0])):
            raise ValueError(
                f"perturbation '{name}': 'alternate_sign' has nothing to flip. It mirrors the lateral"
                " push component and the yaw only, since a forward push and a backward one are"
                " different conditions rather than the same one mirrored."
            )

        resolved.append(
            {
                "name": name,
                "push": push,
                "yaw_init": yaw_init,
                "alternate_sign": alternate_sign,
                "starts": starts,
                "goals": goals,
                "num_samples": num_samples,
            }
        )

    names = [level["name"] for level in resolved]
    if len(names) != len(set(names)):
        raise ValueError(f"perturbation names must be unique (each is a results sub-folder); got {names}")
    if CONTROL_LEVEL not in names:
        resolved.insert(0, control_level())
    return resolved


def control_level() -> dict:
    """The implicit unperturbed level: nominal starts, nominal goal, nothing fired."""
    return {
        "name": CONTROL_LEVEL,
        "push": None,
        "yaw_init": None,
        "alternate_sign": False,
        "starts": NOMINAL,
        "goals": NOMINAL,
        "num_samples": None,
    }


def is_unperturbed(level: dict) -> bool:
    return (
        level["push"] is None
        and level["yaw_init"] is None
        and level.get("starts", NOMINAL) == NOMINAL
        and level.get("goals", NOMINAL) == NOMINAL
    )


def describe_perturbation(level: dict) -> str:
    """One-line human form of a resolved level, for the eval's progress prints."""
    if is_unperturbed(level):
        return "unperturbed"
    parts = []
    if level.get("starts", NOMINAL) == "pool":
        parts.append("starts drawn from the feasible pool")
    if level.get("goals", NOMINAL) == "pool":
        parts.append("goals at the base positions of feasible-pool states")
    if level.get("num_samples"):
        parts.append(f"{level['num_samples']} samples")
    if level["push"] is not None:
        velocity = ", ".join(
            f"v{axis}={lo:+.2f}" for axis, (lo, _) in level["push"]["velocity_range"].items() if lo
        )
        parts.append(f"push {velocity} m/s at t={level['push']['t']:.2f}s (step {level['push']['step']})")
    if level["yaw_init"] is not None:
        parts.append(f"start yaw {math.degrees(level['yaw_init']):+.0f} deg")
    if level["alternate_sign"]:
        parts.append("sign alternating across envs")
    return "; ".join(parts)


def init_wandb_from_eval_config(args_cli, eval_cfg: dict | None) -> None:
    """Init wandb from ``eval_cfg["wandb"]`` when ``--enable_wandb`` is set.

    The group is the run name without its timestamp prefix; ``--results_suffix`` is appended to the group
    and tags.
    """
    if not (args_cli.enable_wandb and eval_cfg is not None and "wandb" in eval_cfg):
        return
    wandb_cfg = eval_cfg["wandb"]
    run_group = "_".join(args_cli.checkpoint_folder.split("_")[2:]) or wandb_cfg.get("group", None)
    tags = list(wandb_cfg.get("tags", []))

    suffix = getattr(args_cli, "results_suffix", None)
    if suffix:
        if run_group:
            run_group = f"{run_group}_{suffix}"
        if suffix not in tags:
            tags.append(suffix)

    wandb.init(
        project=wandb_cfg.get("project", "isaaclab_eval"),
        group=run_group,
        name=args_cli.checkpoint_folder,
        tags=tags,
        config={"task": args_cli.task, "checkpoint_folder": args_cli.checkpoint_folder},
    )
    # Define a custom x-axis metric to allow logging in reverse order
    wandb.define_metric("model_nr")
    wandb.define_metric("mean_reward", step_metric="model_nr")


def discover_checkpoints(args_cli) -> list[str]:
    """Resolve the sorted list of checkpoint ``.pt`` paths to evaluate, or the ``--models`` subset."""
    if args_cli.checkpoint_folder is None:
        raise ValueError(
            "Please provide a valid checkpoint folder name with --checkpoint_folder or use --eval_config with"
            " --sweep_index"
        )
    if args_cli.checkpoint_base_path is None:
        raise ValueError(
            "Please provide a valid checkpoint base path with --checkpoint_base_path or use --eval_config with"
            " --sweep_index"
        )

    checkpoint_path = os.path.join(args_cli.checkpoint_base_path, args_cli.checkpoint_folder)
    if args_cli.models is None:
        pt_files = sorted(
            [
                os.path.abspath(os.path.join(checkpoint_path, f))
                for f in os.listdir(checkpoint_path)
                if f.endswith(".pt")
            ]
        )
    else:
        pt_files = [
            os.path.abspath(os.path.join(checkpoint_path, f"model_{model_id}.pt")) for model_id in args_cli.models
        ]
    return pt_files


def make_results_path(args_cli) -> str:
    """Resolve and create ``<results_dir>/<checkpoint_folder>[_<results_suffix>]``.

    The suffix keeps different evals of the same checkpoint folder from overwriting each other.
    """
    folder_name = args_cli.checkpoint_folder
    suffix = getattr(args_cli, "results_suffix", None)
    if suffix:
        folder_name = f"{folder_name}_{suffix}"
    results_path = os.path.join(args_cli.results_dir, folder_name)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def save_dist_pickle(results_path: str, model_name: str, logs_dict: dict) -> str:
    """Gzip-pickle ``logs_dict`` to ``<results_path>/dist/<model_name>.pkl.gz``. Returns the filename."""
    os.makedirs(os.path.join(results_path, "dist"), exist_ok=True)
    filename = os.path.join(results_path, "dist", model_name + ".pkl.gz")
    print(f"[INFO]: Evaluation finished. Saving results to {filename}")
    with gzip.open(filename, "wb", compresslevel=1) as f:
        pickle.dump(logs_dict, f)
    return filename


def save_trajectories_pickle(results_path: str, traj_dict: dict, model_name: str | None = None) -> str:
    """Gzip-pickle ``traj_dict`` to ``<results_path>/dist/trajectories.pkl.gz``. Returns the filename.

    Pass ``model_name`` to write ``trajectories_<model_name>.pkl.gz`` so multi-checkpoint runs keep every dump.
    """
    stem = "trajectories" if model_name is None else f"trajectories_{model_name}"
    filename = os.path.join(results_path, "dist", stem + ".pkl.gz")
    with gzip.open(filename, "wb", compresslevel=1) as f:
        pickle.dump(traj_dict, f)
    print(f"[INFO]: Saved trajectories to {filename}")
    return filename


def terrain_metadata(env) -> dict:
    """Plain-number description of a single-sub-terrain bridge terrain for the results pickle.

    Lets offline analyses draw the terrain without importing Isaac. Rects are env-relative
    ``(x_lo, x_hi, y_lo, y_hi)``; empty when the terrain has no ``geometry_bounds``.
    """
    gen_cfg = getattr(env.cfg.scene.terrain, "terrain_generator", None)
    subs = getattr(gen_cfg, "sub_terrains", None)
    if not subs or len(subs) != 1:
        return {}
    name, sub = next(iter(subs.items()))
    bounds = getattr(sub, "geometry_bounds", None)
    if bounds is None:
        return {}

    meta = {
        "geometry_bounds": dict(bounds),
        "keep_in_regions": [tuple(float(v) for v in rect) for rect in (sub.keep_in_regions or [])],
        "walls": [
            (
                float(wall.center[0] - wall.size[0] / 2.0),
                float(wall.center[0] + wall.size[0] / 2.0),
                float(wall.center[1] - wall.size[1] / 2.0),
                float(wall.center[1] + wall.size[1] / 2.0),
            )
            for wall in getattr(sub, "walls", [])
        ],
        "perimeter_wall": None,
        "route_rects": None,
        "default_goal": None,
    }
    fence = getattr(sub, "perimeter_wall", None)
    if fence is not None:
        # inner faces: platform AABB grown by the per-axis margin, as in `perimeter_wall_meshes_for_rect`
        margin_x, margin_y = fence.margin
        meta["perimeter_wall"] = {
            "inner": (
                bounds["x_lo"] - margin_x,
                bounds["x_hi"] + margin_x,
                bounds["y_lo"] - margin_y,
                bounds["y_hi"] + margin_y,
            ),
            "thickness": float(fence.thickness),
        }
    # deferred import: the layouts package needs the Omniverse runtime
    from informed_exploration.terrains.config.layouts.dtsg import DTSG_SUBTERRAIN_NAME, dtsg_route_rects

    if name == DTSG_SUBTERRAIN_NAME:
        meta["route_rects"] = dtsg_route_rects(sub.layout, sub.size)
    goal_command = getattr(getattr(env.cfg, "commands", None), "base_position", None)
    default_goal = getattr(goal_command, "default_goal", None)
    if default_goal is not None:
        meta["default_goal"] = [float(v) for v in default_goal]
    return meta


# Terrain queries and start-state synthesis

# flat base-pose indices in the interleaved (pos, vel) 18x2 task; zero joint dims mean the default stance
TASK_X, TASK_Y, TASK_Z, TASK_YAW = 0, 2, 4, 10


def surface_z_at(env, xy_e):
    """Env-relative terrain surface height under each ``(x, y)``, by downward raycast on the terrain mesh.

    Probes env 0's tile, valid for every env since pinned tiles are identical. Imports Isaac Lab lazily.
    """
    import torch

    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh

    mesh = RayCaster.meshes[env.cfg.scene.height_scanner.mesh_prim_paths[0]]
    origin = env.scene.env_origins[0].to(xy_e.device)

    ray_starts = torch.empty(xy_e.shape[0], 3, device=xy_e.device)
    ray_starts[:, :2] = xy_e + origin[:2]
    ray_starts[:, 2] = origin[2] + 50.0  # comfortably above any terrain geometry
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0

    hits = raycast_mesh(ray_starts, ray_directions, mesh=mesh)[0]
    surface_z_e = hits[:, 2] - origin[2]
    assert torch.isfinite(surface_z_e).all(), (
        "downward raycast missed the terrain mesh for some (x, y); those points lie outside the"
        " terrain geometry, so no surface height can be read for them."
    )
    return surface_z_e


def synthesize_start_states(xy_e, z_e, yaw, device):
    """``(n, 36)`` flattened task rows in the default stance at ``xy_e`` / ``z_e`` facing ``yaw``.

    Pass a ``z_e`` read from the surface, not a constant.
    """
    import torch

    starts = torch.zeros(xy_e.shape[0], 36, device=device)
    starts[:, TASK_X] = xy_e[:, 0].to(device)
    starts[:, TASK_Y] = xy_e[:, 1].to(device)
    starts[:, TASK_Z] = z_e.to(device)
    starts[:, TASK_YAW] = yaw.to(device) if hasattr(yaw, "to") else float(yaw)
    return starts
