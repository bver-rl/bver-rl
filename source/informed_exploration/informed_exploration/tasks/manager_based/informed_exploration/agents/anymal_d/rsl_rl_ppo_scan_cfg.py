# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configs for the real-world scan terrains, one set per scan.

Generated from :data:`~informed_exploration.terrains.config.scans.SCAN_SPECS` and mirroring the bridge runners.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from informed_exploration.terrains.config.scans import SCAN_SPECS, ScanSpec

from ...cfg.scan.scan_naming import experiment_name, runner_name  # noqa: F401
from .rsl_rl_ppo_cfg import ParkourPPORunnerCfg, RandomActionNoiseCfg


def build_scan_runners(spec: ScanSpec) -> dict[str, type]:
    """Return the runner cfgs for one scan, keyed by class name.

    Plain PPO, the BVER trio, and a RandomGS control that copies the BVER split so its PPO batch matches.
    """
    experiment = experiment_name(spec)

    plain = configclass(
        type(
            runner_name(spec),
            (ParkourPPORunnerCfg,),
            {
                "experiment_name": experiment,
                "max_iterations": 4000,
                "__module__": __name__,
                "__doc__": f"Vanilla PPO on the {spec.snake} scan (baseline, pool generation).",
            },
        )
    )
    bver = configclass(
        type(
            runner_name(spec, "BVER"),
            (plain,),
            {
                "train_env_ratio": 0.85,
                "random_env_ratio": 0.15,
                "random_reachability_ratio": 0.0,
                "random_backward_ratio": 0.5,
                "random_action_noise_forward": RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0),
                "random_action_noise_backward": RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=10.0),
                "__module__": __name__,
                "__doc__": f"Bidirectional BVER, no reachability, on the {spec.snake} scan.",
            },
        )
    )
    fwd = configclass(
        type(
            runner_name(spec, "BVERFwd"),
            (bver,),
            {
                "random_backward_ratio": 0.0,
                "__module__": __name__,
                "__doc__": "Forward-only ablation arm.",
            },
        )
    )
    bwd = configclass(
        type(
            runner_name(spec, "BVERBwd"),
            (bver,),
            {
                "random_backward_ratio": 1.0,
                "__module__": __name__,
                "__doc__": "Backward-only ablation arm.",
            },
        )
    )
    random_gs = configclass(
        type(
            runner_name(spec, "RandomGS"),
            (plain,),
            {
                "train_env_ratio": 0.85,
                "random_env_ratio": 0.15,
                "random_reachability_ratio": 0.0,
                "random_backward_ratio": 0.5,
                "random_action_noise_forward": RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0),
                "random_action_noise_backward": RandomActionNoiseCfg(noise_type="ou", beta=0.95, alpha=5.0),
                "__module__": __name__,
                "__doc__": (
                    "Uninformed random (start, goal) control; one runner serves all three RandomGS"
                    " arms, since the tail is discarded whatever it contains."
                ),
            },
        )
    )
    return {cls.__name__: cls for cls in (plain, bver, fwd, bwd, random_gs)}


SCAN_RUNNERS: dict[str, dict[str, type]] = {
    snake: build_scan_runners(spec) for snake, spec in SCAN_SPECS.items()
}
"""``{scan snake name: {class name: runner cfg class}}`` for every registered scan."""

for _runners in SCAN_RUNNERS.values():
    globals().update(_runners)

__all__ = [
    "SCAN_RUNNERS",
    "build_scan_runners",
    "runner_name",
    *sorted(name for runners in SCAN_RUNNERS.values() for name in runners),
]
