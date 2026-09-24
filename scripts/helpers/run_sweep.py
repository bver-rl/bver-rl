#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run all training jobs defined in an experiment YAML sequentially, including sweep grids.

Each ``runs:`` entry yields one job per seed, or per (grid point, seed) pair when it has a ``sweep:`` block.
Jobs run as subprocesses of ``train.py`` with the same interpreter; ``--dry_run`` only prints the commands.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import yaml

# Resolve paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The entry points live in scripts/rsl_rl/, so resolve them there rather than next to this helper.
RSL_RL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "rsl_rl"))
TRAIN_SCRIPT = os.path.join(RSL_RL_DIR, "train.py")


# Sweep helpers (duplicated here to keep the runner self-contained)


def _get_grid_size(sweep: dict) -> int:
    size = 1
    for axis in sweep.values():
        size *= len(axis["values"])
    return size


def _resolve_sweep_overrides(sweep: dict, sweep_index: int) -> list[str]:
    """Resolve one sweep index to concrete key=value overrides in row-major order (last axis fastest)."""
    axes: list[tuple[str, list]] = [(k, v["values"]) for k, v in sweep.items()]

    grid_size = 1
    for _, values in axes:
        grid_size *= len(values)
    if not (0 <= sweep_index < grid_size):
        raise ValueError(f"sweep_index={sweep_index} is out of range [0, {grid_size}).")

    overrides: list[str] = []
    remainder = sweep_index
    for i, (key, values) in enumerate(axes):
        stride = 1
        for _, later_values in axes[i + 1 :]:
            stride *= len(later_values)
        idx = remainder // stride
        remainder %= stride
        overrides.append(f"{key}={values[idx]}")

    return overrides


def _resolve_run_effective(defaults: dict, run_entry: dict) -> dict:
    """Return defaults merged with run-level overrides."""
    effective = dict(defaults)
    for key, value in run_entry.items():
        if key != "task":
            effective[key] = value
    return effective


def _resolve_seed_list(effective: dict, extra_args: list[str] | None) -> list[int | None]:
    """Return seed values to schedule.

    An explicit ``--seed`` in ``extra_args`` wins, then ``seeds``, then ``seed``, else ``[None]``.
    """
    if extra_args and "--seed" in extra_args:
        return [None]

    seeds = effective.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list):
            raise ValueError("Expected 'seeds' to be a list in experiment config.")
        return [int(s) for s in seeds]

    seed = effective.get("seed")
    if seed is not None:
        return [int(seed)]

    return [None]


# Build commands


def build_commands(
    config_file: str,
    target_script: str,
    config_arg_name: str,
    task_filter: str | None = None,
    extra_args: list[str] | None = None,
) -> list[tuple[str, list[str], dict]]:
    """Return a list of ``(description, argv, metadata)`` tuples for every job."""

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    defaults: dict = raw.get("defaults", {})
    raw_runs = raw.get("runs", [])
    commands: list[tuple[str, list[str], dict]] = []

    if config_arg_name == "--eval_config":
        task = task_filter if task_filter else "UnknownTask"
        for i, run_entry in enumerate(raw_runs):
            desc = f"Eval [{run_entry}] sweep_index={i}"
            cmd = [
                sys.executable,
                target_script,
                "--headless",
                config_arg_name,
                os.path.abspath(config_file),
                "--sweep_index",
                str(i),
            ]
            if task_filter:
                cmd.extend(["--task", task_filter])
            if extra_args:
                cmd.extend(extra_args)
            metadata = {"task": task, "sweep_index": i, "checkpoint": run_entry}
            commands.append((desc, cmd, metadata))
        return commands

    runs = raw_runs
    for run_entry in runs:
        task = run_entry["task"]
        if task_filter and task != task_filter:
            continue

        sweep = run_entry.get("sweep")
        effective = _resolve_run_effective(defaults, run_entry)
        seeds = _resolve_seed_list(effective, extra_args)
        base_args = [
            sys.executable,
            target_script,
            "--task",
            task,
            "--headless",
            config_arg_name,
            os.path.abspath(config_file),
        ]
        if extra_args:
            base_args.extend(extra_args)

        if not sweep:
            for seed in seeds:
                cmd = list(base_args)
                desc = f"{task}"
                metadata = {"task": task, "seed": seed, "sweep_index": None, "sweep_overrides": []}
                if seed is not None:
                    cmd.extend(["--seed", str(seed)])
                    desc += f"  [seed {seed}]"
                commands.append((desc, cmd, metadata))
        else:
            # global_idx = grid_index * num_seeds + seed_index (SLURM array convention); train.py decodes it
            grid_size = _get_grid_size(sweep)
            num_seeds = len(seeds)
            total = grid_size * num_seeds
            for global_idx in range(total):
                grid_idx = global_idx // num_seeds
                seed_idx = global_idx % num_seeds
                seed = seeds[seed_idx]
                sweep_overrides = _resolve_sweep_overrides(sweep, grid_idx)
                desc = f"{task}  [sweep {global_idx}/{total}]"
                cmd = list(base_args) + ["--sweep_index", str(global_idx)]
                # seed is encoded in the global index; an explicit --seed already sits in base_args
                metadata = {
                    "task": task,
                    "seed": seed,
                    "sweep_index": global_idx,
                    "sweep_grid_index": grid_idx,
                    "sweep_grid_size": grid_size,
                    "sweep_overrides": sweep_overrides,
                }
                if seed is not None:
                    desc += f"  [seed {seed}]"
                commands.append((desc, cmd, metadata))

    return commands


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequentially run all training/eval jobs from a config YAML.")
    parser.add_argument(
        "--experiment_config",
        default=None,
        help="Path to the experiment YAML file (for training).",
    )
    parser.add_argument(
        "--eval_config",
        default=None,
        help="Path to the evaluation YAML file (for evaluation).",
    )
    parser.add_argument(
        "--eval_script",
        default="eval.py",
        help=(
            "Script (in scripts/rsl_rl/) to run for --eval_config jobs, e.g. eval_multi_goal.py."
            " Ignored for --experiment_config jobs, which always use train.py."
        ),
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional: only run jobs matching this task name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them.",
    )
    args, extra = parser.parse_known_args()

    if args.experiment_config:
        config_file = args.experiment_config
        target_script = os.path.join(RSL_RL_DIR, "train.py")
        config_arg_name = "--experiment_config"
    elif args.eval_config:
        config_file = args.eval_config
        target_script = os.path.join(RSL_RL_DIR, args.eval_script)
        config_arg_name = "--eval_config"
    else:
        print("[ERROR] Must provide either --experiment_config or --eval_config", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(config_file):
        print(f"[ERROR] Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    commands = build_commands(config_file, target_script, config_arg_name, args.task, extra if extra else None)

    if not commands:
        print("[WARN] No matching jobs found.")
        sys.exit(0)

    total = len(commands)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Scheduled {total} job(s):\n")
    for i, (desc, _, metadata) in enumerate(commands):
        print(f"  [{i}] {desc}")
        if args.dry_run:
            if metadata.get("sweep_index") is not None:
                grid_idx = metadata.get("sweep_grid_index", metadata["sweep_index"])
                print(
                    f"      sweep_index: {metadata['sweep_index']}  (grid {grid_idx}/{metadata.get('sweep_grid_size', '?')})"
                )
            if metadata.get("checkpoint"):
                print(f"      checkpoint: {metadata['checkpoint']}")
            if metadata.get("sweep_overrides"):
                print(f"      params: {', '.join(metadata['sweep_overrides'])}")
            if metadata.get("seed") is not None:
                print(f"      seed: {metadata['seed']}")
    print()

    failed: list[tuple[int, str, int]] = []

    for i, (desc, cmd, metadata) in enumerate(commands):
        header = f"[{i + 1}/{total}] {desc}"
        print("=" * 80)
        print(header)
        print("=" * 80)

        if args.dry_run:
            if metadata.get("sweep_index") is not None:
                grid_idx = metadata.get("sweep_grid_index", metadata["sweep_index"])
                print(
                    f"sweep_index: {metadata['sweep_index']}  (grid {grid_idx}/{metadata.get('sweep_grid_size', '?')})"
                )
            if metadata.get("checkpoint"):
                print(f"checkpoint: {metadata['checkpoint']}")
            if metadata.get("sweep_overrides"):
                print(f"params: {', '.join(metadata['sweep_overrides'])}")
            if metadata.get("seed") is not None:
                print(f"seed: {metadata['seed']}")
            print(" ".join(cmd))
            print()
            continue

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            failed.append((i, desc, result.returncode))
            print(f"\n[FAIL] {desc}  (exit code {result.returncode}, {elapsed:.0f}s)\n")
        else:
            print(f"\n[OK]   {desc}  ({elapsed:.0f}s)\n")

    # Summary
    print("=" * 80)
    print(f"Finished: {total - len(failed)}/{total} succeeded")
    if failed:
        print("Failed jobs:")
        for idx, desc, rc in failed:
            print(f"  [{idx}] {desc}  (exit code {rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
