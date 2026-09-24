# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Names of each scan's arms, cfg classes and task ids for the gym registry.

Deliberately import-free: the registry runs while the task package initialises and would otherwise
hit a circular import, so it registers by ``module:Class`` strings formed here.
"""

from __future__ import annotations

ARM_NAMES: tuple[str, ...] = (
    "BVER",
    "BVER_FWD",
    "BVER_BWD",
    "BVER_PEN",
    "BVER_PLAY",
    "BVER_EVAL",
    "BVER_MGPLAY",
    "BVER_MGEVAL",
    "BVER_PLAY_SCALE05",
    "BVER_EVAL_SCALE05",
    "BVER_MGPLAY_SCALE05",
    "BVER_MGEVAL_SCALE05",
    "Baseline",
    "Baseline_PEN",
    "GenFeasibleStarts",
    "RandomGS",
    "RandomGS_FWD",
    "RandomGS_BWD",
)
"""The arms every scan gets, in the order they appear in the registry and the launch group."""

ARM_TASK_SUFFIX: dict[str, str] = {
    "BVER": "BVER-v0",
    "BVER_FWD": "BVER-fwd-v0",
    "BVER_BWD": "BVER-bwd-v0",
    "BVER_PEN": "BVER-pen-v0",
    "BVER_PLAY": "BVER-PLAY-v0",
    "BVER_EVAL": "BVER-EVAL-v0",
    "BVER_MGPLAY": "BVER-MGPLAY-v0",
    "BVER_MGEVAL": "BVER-MGEVAL-v0",
    "BVER_PLAY_SCALE05": "BVER-PLAY-scale05-v0",
    "BVER_EVAL_SCALE05": "BVER-EVAL-scale05-v0",
    "BVER_MGPLAY_SCALE05": "BVER-MGPLAY-scale05-v0",
    "BVER_MGEVAL_SCALE05": "BVER-MGEVAL-scale05-v0",
    "Baseline": "Baseline-v0",
    "Baseline_PEN": "Baseline-pen-v0",
    "GenFeasibleStarts": "GenFeasibleStarts-v0",
    "RandomGS": "RandomGS-v0",
    "RandomGS_FWD": "RandomGS-fwd-v0",
    "RandomGS_BWD": "RandomGS-bwd-v0",
}
"""Task-id suffix per arm: lowercase for training ablations, uppercase ``-PLAY-``/``-EVAL-``/``-MG*-``
for inspection bindings, and ``-scale05-`` on bindings that replay scale-0.5 checkpoints."""

ARM_RUNNER: dict[str, str] = {
    "BVER": "BVER",
    "BVER_FWD": "BVERFwd",
    "BVER_BWD": "BVERBwd",
    # penalized reward, same runner as the bidirectional arm
    "BVER_PEN": "BVER",
    # play/eval bindings replay a BVER checkpoint, so they carry its runner for the network shape
    "BVER_PLAY": "BVER",
    "BVER_EVAL": "BVER",
    "BVER_MGPLAY": "BVER",
    "BVER_MGEVAL": "BVER",
    "BVER_PLAY_SCALE05": "BVER",
    "BVER_EVAL_SCALE05": "BVER",
    "BVER_MGPLAY_SCALE05": "BVER",
    "BVER_MGEVAL_SCALE05": "BVER",
    # no random-action tail: every env trains
    "Baseline": "",
    "Baseline_PEN": "",
    "GenFeasibleStarts": "",
    # one runner for all three: the tail is discarded whatever it contains
    "RandomGS": "RandomGS",
    "RandomGS_FWD": "RandomGS",
    "RandomGS_BWD": "RandomGS",
}
"""Which runner cfg each arm pairs with, as the infix in :func:`runner_name`."""


def env_cfg_name(spec, arm: str) -> str:
    """Class name of one arm on one scan."""
    return f"IEAnymalDScan{spec.name}EnvCfg_{arm}"


def runner_name(spec, arm_infix: str = "") -> str:
    """Class name of a scan's runner cfg; ``arm_infix`` is one of :data:`ARM_RUNNER`'s values."""
    return f"AnymalDScan{spec.name}{arm_infix}PPORunnerCfg"


def task_id(spec, arm: str) -> str:
    """Gym id of one arm on one scan."""
    return f"Parkour-Scan-{spec.name}-{ARM_TASK_SUFFIX[arm]}"


def experiment_name(spec) -> str:
    """Log directory and rsl_rl experiment name for a scan."""
    return f"anymal_d_scan_{spec.snake}"


def pool_path(spec) -> str:
    """Feasible-start pool for one scan, repo-root-relative like the other pools."""
    return f"informed-exploration/data/parkour/feasible_starts_scan_{spec.snake}.pkl"
