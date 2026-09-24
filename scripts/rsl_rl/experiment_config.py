# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for loading experiment YAML configs and patching the wandb writer.

The wandb writer is monkey-patched to inject group, tags and notes without modifying rsl_rl.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Experiment YAML loading


def load_experiment_config(yaml_path: str, task_name: str | None = None) -> dict[str, Any]:
    """Load an experiment YAML and resolve the effective config for a specific run.

    The YAML holds ``experiment``, ``defaults`` and ``runs`` blocks; the first run whose ``task`` matches wins.

    Args:
        yaml_path: Path to the experiment YAML file.
        task_name: The ``--task`` value selecting the run entry; if None only ``defaults`` is used.

    Returns:
        A dict with ``"experiment"`` (metadata), ``"effective"`` (defaults merged with run overrides)
        and ``"raw"`` (the parsed YAML).

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        ValueError: If *task_name* is given but no matching run entry is found.
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Experiment config not found: {yaml_path}")

    with open(yaml_path) as f:
        raw: dict = yaml.safe_load(f)

    experiment: dict = raw.get("experiment", {})
    defaults: dict = raw.get("defaults", {})
    runs: list[dict] = raw.get("runs", [])

    effective = copy.deepcopy(defaults)

    if task_name is not None:
        matched_run: dict | None = None
        for run_entry in runs:
            if run_entry.get("task") == task_name:
                matched_run = run_entry
                break
        if matched_run is None:
            available = [r.get("task", "<no task>") for r in runs]
            raise ValueError(f"No run entry for task '{task_name}' in {yaml_path}. " f"Available tasks: {available}")
        for key, value in matched_run.items():
            if key != "task":
                effective[key] = value

        run_tags = matched_run.get("tags", [])
        if run_tags:
            exp_wandb = experiment.get("wandb", {})
            base_tags = list(exp_wandb.get("tags", []))
            # Deduplicated, order preserved.
            merged = base_tags + [t for t in run_tags if t not in base_tags]
            experiment = copy.deepcopy(experiment)
            experiment.setdefault("wandb", {})["tags"] = merged

    return {
        "experiment": experiment,
        "effective": effective,
        "raw": raw,
    }


def get_experiment_meta_for_run(
    experiment_data: dict[str, Any], task_name: str | None = None, sweep_index: int | None = None
) -> dict[str, Any]:
    """Return experiment metadata with a deterministic wandb group for this run.

    The group joins ``wandb.group``, ``run_name`` (or the task name) and ``sw{index}`` so seeds group together.

    Args:
        experiment_data: The dict returned by :func:`load_experiment_config`.
        task_name: Fallback group component when no group or run name exists.
        sweep_index: Sweep index appended for sweep jobs.

    Returns:
        A deep copy of the ``experiment`` metadata with ``["wandb"]["group"]`` resolved.
    """
    meta = copy.deepcopy(experiment_data.get("experiment", {}))
    effective = experiment_data.get("effective", {})

    wandb_meta = dict(meta.get("wandb", {}))
    base_group = wandb_meta.get("group")
    run_name = effective.get("run_name")

    group_parts: list[str] = []
    if base_group:
        group_parts.append(str(base_group))
    if run_name:
        group_parts.append(str(run_name))
    if not group_parts and task_name:
        group_parts.append(str(task_name))
    if sweep_index is not None:
        group_parts.append(f"sw{sweep_index}")

    resolved_group = "_".join(group_parts) if group_parts else None
    if resolved_group is not None:
        wandb_meta["group"] = resolved_group

    if wandb_meta:
        meta["wandb"] = wandb_meta

    return meta


def apply_experiment_config_to_args(experiment: dict[str, Any], args_cli) -> None:
    """Apply effective experiment config values onto the argparse namespace; CLI flags always win.

    Args:
        experiment: The ``"effective"`` dict returned by :func:`load_experiment_config`.
        args_cli: The ``argparse.Namespace`` from ``parser.parse_known_args()``.
    """
    # Experiment YAML keys to argparse attribute names.
    YAML_TO_ARG = {
        "num_envs": "num_envs",
        "max_iterations": "max_iterations",
        "seed": "seed",
        "logger": "logger",
        "video": "video",
        "video_interval": "video_interval",
        "video_length": "video_length",
        "run_name": "run_name",
    }

    for yaml_key, arg_attr in YAML_TO_ARG.items():
        if yaml_key not in experiment:
            continue
        yaml_value = experiment[yaml_key]

        current = getattr(args_cli, arg_attr, None)
        # None or a False store_true flag counts as not set on the CLI.
        if current is None or (isinstance(current, bool) and current is False):
            setattr(args_cli, arg_attr, yaml_value)
            logger.debug(f"Experiment config: {arg_attr} = {yaml_value}")

    # Video needs cameras.
    if getattr(args_cli, "video", False):
        args_cli.enable_cameras = True


# WandB monkey-patch

# Module-level storage so the subclass can read it at construction time.
_wandb_experiment_meta: dict[str, Any] = {}


def patch_wandb_writer(experiment_meta: dict[str, Any]) -> None:
    """Monkey-patch ``WandbSummaryWriter`` to inject group, tags and notes.

    Must run before the ``OnPolicyRunner`` is built, since the rsl_rl Logger late-imports the writer class.

    Args:
        experiment_meta: The ``"experiment"`` dict from the YAML (``wandb.group``, ``wandb.tags``, ``description``).
    """
    global _wandb_experiment_meta
    _wandb_experiment_meta = experiment_meta

    import rsl_rl.utils.wandb_utils as wandb_module

    OriginalWriter = wandb_module.WandbSummaryWriter

    class EnhancedWandbSummaryWriter(OriginalWriter):
        """Drop-in replacement that passes group/tags/notes to ``wandb.init``."""

        def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
            # Replicates OriginalWriter.__init__, whose wandb.init lacks group/tags/notes.
            import wandb
            from torch.utils.tensorboard import SummaryWriter

            SummaryWriter.__init__(self, log_dir, flush_secs)

            run_name = os.path.split(log_dir)[-1]

            try:
                project = cfg["wandb_project"]
            except KeyError:
                raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
            try:
                entity = os.environ["WANDB_USERNAME"]
            except KeyError:
                entity = None

            meta = _wandb_experiment_meta
            wandb_meta = meta.get("wandb", {})

            wandb.init(
                project=wandb_meta.get("project", project),
                entity=entity,
                name=run_name,
                group=wandb_meta.get("group"),
                tags=wandb_meta.get("tags"),
                notes=_build_notes(meta),
                config={"log_dir": log_dir},
            )

            self.logged_videos: set[str] = set()
            self._rerun_enabled: bool = bool(cfg.get("rerun_metrics", False))

    wandb_module.WandbSummaryWriter = EnhancedWandbSummaryWriter
    logger.info("Patched WandbSummaryWriter with experiment group/tags/notes.")


def store_experiment_config_in_wandb(raw_config: dict[str, Any]) -> None:
    """Store the full experiment YAML in the active wandb run's config; call after ``wandb.init``.

    Args:
        raw_config: The ``"raw"`` dict returned by :func:`load_experiment_config`.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update({"experiment_config": raw_config}, allow_val_change=True)
    except Exception as exc:
        logger.warning(f"Could not store experiment config in wandb: {exc}")


# Sweep grid resolution


def get_sweep_grid_size(experiment_data: dict[str, Any]) -> int:
    """Return the number of combinations in the effective config's ``sweep`` grid.

    The size is the product of the lengths of all ``values`` lists.

    Args:
        experiment_data: The dict returned by :func:`load_experiment_config`.

    Returns:
        The number of grid points, or zero if no sweep is defined.
    """
    sweep = experiment_data["effective"].get("sweep")
    if not sweep:
        return 0
    size = 1
    for axis in sweep.values():
        size *= len(axis["values"])
    return size


def resolve_sweep(experiment_data: dict[str, Any], sweep_index: int) -> list[str]:
    """Map a flat *sweep_index* to a list of Hydra-style CLI override strings.

    Uses row-major ordering (last axis varies fastest) so SLURM array indices map deterministically.

    Args:
        experiment_data: The dict returned by :func:`load_experiment_config`.
        sweep_index: Zero-based index into the cartesian product grid.

    Returns:
        A list of ``"dotpath=value"`` override strings suitable for Hydra.

    Raises:
        ValueError: If *sweep_index* is out of range or no sweep is defined.
    """
    sweep = experiment_data["effective"].get("sweep")
    if not sweep:
        raise ValueError("No 'sweep' block defined for the current run entry.")

    axes: list[tuple[str, list]] = [(k, v["values"]) for k, v in sweep.items()]

    grid_size = 1
    for _, vals in axes:
        grid_size *= len(vals)

    if not (0 <= sweep_index < grid_size):
        raise ValueError(f"sweep_index={sweep_index} is out of range [0, {grid_size}).")

    # Row-major decomposition.
    overrides: list[str] = []
    remainder = sweep_index
    for key, vals in axes:
        stride = 1
        for _, later_vals in axes[axes.index((key, vals)) + 1 :]:
            stride *= len(later_vals)
        idx = remainder // stride
        remainder %= stride
        overrides.append(f"{key}={vals[idx]}")

    return overrides


# Helpers


def _build_notes(meta: dict[str, Any]) -> str | None:
    """Combine description + hypothesis into a wandb notes string."""
    parts: list[str] = []
    if meta.get("description"):
        parts.append(meta["description"])
    if meta.get("hypothesis"):
        parts.append(f"Hypothesis: {meta['hypothesis']}")
    return "\n".join(parts) if parts else None


def parse_and_apply_experiment_config(
    args_cli, hydra_args: list[str]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str], int | None]:
    """Load experiment config from YAML, apply to args_cli, and resolve sweeps.

    Args:
        args_cli: The ``argparse.Namespace`` from ``parser.parse_known_args()``.
        hydra_args: Leftover Hydra arguments; sweep overrides are appended in place.

    Returns:
        A tuple of the experiment data, the resolved run metadata, the sweep overrides and the sweep
        grid index. The data, metadata and index are None when not applicable.
    """
    experiment_data = None
    experiment_meta_for_run = None
    sweep_overrides: list[str] = []
    sweep_grid_index: int | None = None

    if args_cli.experiment_config is not None:
        experiment_data = load_experiment_config(args_cli.experiment_config, args_cli.task)
        apply_experiment_config_to_args(experiment_data["effective"], args_cli)

        if getattr(args_cli, "sweep_index", None) is not None:
            # With a seeds list and no --seed, the index is grid_index * num_seeds + seed_index.
            seeds_list = experiment_data["effective"].get("seeds")
            if isinstance(seeds_list, list) and len(seeds_list) > 0 and getattr(args_cli, "seed", None) is None:
                num_seeds = len(seeds_list)
                sweep_grid_index = args_cli.sweep_index // num_seeds
                seed_index = args_cli.sweep_index % num_seeds
                args_cli.seed = int(seeds_list[seed_index])
            else:
                sweep_grid_index = args_cli.sweep_index

            sweep_overrides = resolve_sweep(experiment_data, sweep_grid_index)
            hydra_args.extend(sweep_overrides)
            # A Hydra num_envs override must not lose to the CLI arg.
            if any(o.startswith("env.scene.num_envs=") for o in sweep_overrides):
                args_cli.num_envs = None
            # Keeps log dirs of different grid points apart.
            suffix = f"_sw{sweep_grid_index}"
            if getattr(args_cli, "run_name", None):
                args_cli.run_name += suffix
            elif experiment_data["effective"].get("run_name"):
                args_cli.run_name = experiment_data["effective"]["run_name"] + suffix

        experiment_meta_for_run = get_experiment_meta_for_run(
            experiment_data,
            task_name=args_cli.task,
            sweep_index=sweep_grid_index,
        )

    return experiment_data, experiment_meta_for_run, sweep_overrides, sweep_grid_index
