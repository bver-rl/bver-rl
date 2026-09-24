# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for every task in this extension.

Families are tables of ``(task suffix, env cfg class suffix, runner infix)`` rows expanded by
:func:`_register_family`; cfgs are bound by ``module:Class`` string since they cannot be imported yet.
"""

import gymnasium as gym

from informed_exploration.terrains.config import scans as scan_specs

from .cfg.scan import scan_naming

from . import agents
from .cfg.common.box_height import CLIMB_BOX_HEIGHTS

_ENVS = "informed_exploration.tasks.manager_based.informed_exploration.envs"
PARKOUR_ENV = f"{_ENVS}:IEParkourEnv"
MAZE_ENV = f"{_ENVS}:MazeEnv"
HANOI_ENV = f"{_ENVS}:HanoiEnv"


def _register(task_id: str, env_entry_point: str, env_cfg: str, runner: str) -> None:
    """Register one task; ``env_cfg`` and ``runner`` are ``module:Class`` relative to this package and :mod:`agents`."""
    gym.register(
        id=task_id,
        entry_point=env_entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.{runner}",
        },
    )


def _register_family(
    *,
    id_prefix: str,
    env_entry_point: str,
    cfg_module: str,
    cfg_prefix: str,
    runner_module: str,
    runner_prefix: str,
    arms: tuple[tuple[str, str, str], ...],
    heights: tuple[str, ...] = (),
) -> None:
    """Expand one table of ``(task suffix, env cfg class suffix, runner infix)`` rows.

    An empty runner infix binds the plain PPO runner. ``heights`` registers each arm once per box-height
    token, placed before ``-v0`` and appended to the class name, since the height cannot be a hydra override.
    """
    for task_suffix, cfg_suffix, runner_infix in arms:
        runner = f"{runner_module}:{runner_prefix}{runner_infix}PPORunnerCfg"
        if not heights:
            _register(f"{id_prefix}-{task_suffix}", env_entry_point,
                      f"{cfg_module}:{cfg_prefix}_{cfg_suffix}", runner)
            continue
        stem = task_suffix.removesuffix("-v0")
        for token in heights:
            _register(f"{id_prefix}-{stem}-{token}-v0", env_entry_point,
                      f"{cfg_module}:{cfg_prefix}_{cfg_suffix}_{token}", runner)


# Templates

_register(
    "Template-v0",
    "isaaclab.envs:ManagerBasedRLEnv",
    "cfg.template_env_cfg:TemplateEnvCfg",
    "template.rsl_rl_ppo_cfg:PPORunnerCfg",
)


# Goal-conditioned arm set shared by maze and Hanoi; eval and playback bind the plain runner.

GOAL_ARMS: tuple[tuple[str, str, str], ...] = (
    # task suffix        env cfg suffix   runner infix
    ("BVER-v0", "BVER", "BVER"),
    ("BVER-fwd-v0", "BVER_FWD", "BVERFwd"),
    ("BVER-bwd-v0", "BVER_BWD", "BVERBwd"),
    ("Baseline-v0", "Baseline", ""),
    ("Random-v0", "Random", ""),
    ("EVAL-v0", "EVAL", ""),
    ("PLAY-v0", "PLAY", ""),
)


# Maze (point mass): U is the literature UMaze, Serpentine and Scatter are hand-drawn 11x11 stages.

MAZE_LAYOUTS: tuple[tuple[str, str], ...] = (
    # id token, cfg module stem (the env cfg class is `{token}MazeEnvCfg`)
    ("U", "maze_u_cfg"),
    ("Serpentine", "maze_serpentine_cfg"),
    ("Scatter", "maze_scatter_cfg"),
)

for _token, _module in MAZE_LAYOUTS:
    _register_family(
        id_prefix=f"Maze-{_token}",
        env_entry_point=MAZE_ENV,
        cfg_module=f"cfg.maze.{_module}",
        cfg_prefix=f"{_token}MazeEnvCfg",
        runner_module="maze.rsl_rl_ppo_cfg",
        runner_prefix="Maze",
        arms=GOAL_ARMS,
    )


# Franka Hanoi (ring between pegs, token = peg height in cm); needs scripts/helpers/hanoi_solve_poses.py output.

_register_family(
    id_prefix="Franka-Hanoi-P12",
    env_entry_point=HANOI_ENV,
    cfg_module="cfg.hanoi.hanoi_p12_cfg",
    cfg_prefix="HanoiP12EnvCfg",
    runner_module="hanoi.rsl_rl_ppo_cfg",
    runner_prefix="Hanoi",
    arms=GOAL_ARMS,
)


# Parkour Single Box (ANYmal-D climbs onto a box): sparse-reward no-reachability BVER with xyz obs.

_register_family(
    id_prefix="Parkour-ClimbSingleBox",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.parkour.climb_heights",
    cfg_prefix="IEAnymalDClimbSingleBoxEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_cfg",
    runner_prefix="AnymalDClimbBox",
    arms=(
        # training
        ("BVER-sparse-v0", "BVER_SPARSE", "BVERSparse"),
        ("BVER-fwd-sparse-v0", "BVER_SPARSE_FWD", "BVERSparseFwd"),
        ("BVER-bwd-sparse-v0", "BVER_SPARSE_BWD", "BVERSparseBwd"),
        ("BVER-pen-sparse-v0", "BVER_SPARSE_PEN", "BVERSparse"),  # subset of non-contact penalties plus dof_acc
        # Play / eval take the bidirectional runner; fwd/bwd differ only in an inert ratio.
        ("BVER-sparse-PLAY-v0", "BVER_SPARSE_PLAY", "BVERSparse"),
        ("BVER-sparse-MGPLAY-v0", "BVER_SPARSE_MGPLAY", "BVERSparse"),
        ("BVER-sparse-EVAL-v0", "BVER_SPARSE_EVAL", "BVERSparse"),
        ("BVER-sparse-MGEVAL-v0", "BVER_SPARSE_MGEVAL", "BVERSparse"),
        ("BVER-sparse-ROBUSTEVAL-v0", "BVER_SPARSE_ROBUSTEVAL", "BVERSparse"),  # push term per --perturbations level
        ("BVER-sparse-ROBUSTPLAY-v0", "BVER_SPARSE_ROBUSTPLAY", "BVERSparse"),
        # Action-scale-0.5 counterparts; replaying those checkpoints on scale 1.0 fails silently.
        ("BVER-sparse-PLAY-scale05-v0", "BVER_SPARSE_PLAY_SCALE05", "BVERSparse"),
        ("BVER-sparse-MGPLAY-scale05-v0", "BVER_SPARSE_MGPLAY_SCALE05", "BVERSparse"),
        ("BVER-sparse-EVAL-scale05-v0", "BVER_SPARSE_EVAL_SCALE05", "BVERSparse"),
        ("BVER-sparse-MGEVAL-scale05-v0", "BVER_SPARSE_MGEVAL_SCALE05", "BVERSparse"),
        ("BVER-sparse-ROBUSTEVAL-scale05-v0", "BVER_SPARSE_ROBUSTEVAL_SCALE05", "BVERSparse"),
        ("BVER-sparse-ROBUSTPLAY-scale05-v0", "BVER_SPARSE_ROBUSTPLAY_SCALE05", "BVERSparse"),
    ),
    heights=CLIMB_BOX_HEIGHTS,
)

# Non-BVER controls on the same sparse MDP and xyz obs (climb_single_box_cfg_sparse_baselines.py).
_register_family(
    id_prefix="Parkour-ClimbSingleBox",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.parkour.climb_heights",
    cfg_prefix="IEAnymalDClimbSingleBoxEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_cfg",
    runner_prefix="AnymalDClimbBox",
    arms=(
        ("Baseline-sparse-v0", "SPARSE_Baseline", "Sparse"),  # vanilla PPO
        ("Baseline-pen-sparse-v0", "SPARSE_Baseline_PEN", "Sparse"),  # vanilla PPO on the -pen- rewards
        ("RSI-sparse-v0", "SPARSE_RSI", "Sparse"),  # resets replay the expert trajectory
        ("Random-sparse-v0", "SPARSE_Random", "Sparse"),  # uniform draws from the feasible-starts pool
        ("Backplay-sparse-v0", "SPARSE_Backplay", "Sparse"),  # RSI pool through a backward-sliding window
        ("RC-sparse-v0", "SPARSE_RC", "RCSparse"),  # reverse curriculum with its own runner
    ),
    heights=CLIMB_BOX_HEIGHTS,
)

# Feasible-start pool generators, run once per height to write the pool the Random arm expects.
_register_family(
    id_prefix="Parkour-ClimbSingleBox",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.parkour.climb_heights",
    cfg_prefix="IEAnymalDClimbSingleBoxEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_cfg",
    runner_prefix="AnymalDClimbBox",
    arms=(
        ("GenFeasibleStarts-v0", "GenFeasibleStarts", ""),
    ),
    heights=CLIMB_BOX_HEIGHTS,
)


# Parkour Single Box DOWN (ANYmal-D climbs down from the box), mirroring the climb-up sparse arms.

_register_family(
    id_prefix="Parkour-ClimbSingleBoxDown",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.parkour.climb_heights",
    cfg_prefix="IEAnymalDClimbSingleBoxDownEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_cfg",
    runner_prefix="AnymalDClimbDownBox",
    arms=(
        ("BVER-sparse-v0", "BVER_SPARSE", "BVERSparse"),
        ("BVER-fwd-sparse-v0", "BVER_SPARSE_FWD", "BVERSparseFwd"),
        ("BVER-bwd-sparse-v0", "BVER_SPARSE_BWD", "BVERSparseBwd"),
        ("BVER-sparse-PLAY-v0", "BVER_SPARSE_PLAY", "BVERSparse"),
        ("BVER-sparse-MGPLAY-v0", "BVER_SPARSE_MGPLAY", "BVERSparse"),
        ("BVER-sparse-EVAL-v0", "BVER_SPARSE_EVAL", "BVERSparse"),
        ("BVER-sparse-MGEVAL-v0", "BVER_SPARSE_MGEVAL", "BVERSparse"),
    ),
    heights=CLIMB_BOX_HEIGHTS,
)


# Bridge DTSG (stairs, pillars and ramp routes to one goal); MG, corridor and harvest bindings are unwired.

_register_family(
    id_prefix="Parkour-BridgeDTSG",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.bridge.bridge_cfg_bver",
    cfg_prefix="IEAnymalDBridgeDTSGEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_bridge_cfg",
    runner_prefix="AnymalDBridgeDTSG",
    arms=(
        ("BVER-v0", "BVER", "BVER"),
        ("BVER-fwd-v0", "BVER_FWD", "BVERFwd"),
        ("BVER-bwd-v0", "BVER_BWD", "BVERBwd"),
        ("BVER-PLAY-v0", "BVER_PLAY", "BVER"),
        ("BVER-MGPLAY-v0", "BVER_MGPLAY", "BVER"),
        ("BVER-EVAL-v0", "BVER_EVAL", "BVER"),
        ("BVER-MGEVAL-v0", "BVER_MGEVAL", "BVER"),  # goals sampled from the task space
        ("BVER-CORRIDOR-v0", "BVER_CORRIDOR", "BVER"),  # goals inside one route
        ("BVER-CORRIDORPLAY-v0", "BVER_CORRIDORPLAY", "BVER"),
        ("BVER-CORRIDORSTART-v0", "BVER_CORRIDORSTART", "BVER"),  # pooled mid-route starts
        ("BVER-CORRIDORSTARTPLAY-v0", "BVER_CORRIDORSTARTPLAY", "BVER"),
        ("BVER-HARVEST-v0", "BVER_HARVEST", "BVER"),  # regenerates the RSI trajectory pool
        ("RSI-v0", "RSI", ""),  # resets replay the harvested pool
    ),
)

# Uninformed controls and pool generators; the RandomGS arms share one runner since the tail is discarded.
_register_family(
    id_prefix="Parkour-BridgeDTSG",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.bridge.bridge_cfg_random",
    cfg_prefix="IEAnymalDBridgeDTSGEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_bridge_cfg",
    runner_prefix="AnymalDBridgeDTSG",
    arms=(
        ("GenFeasibleStarts-v0", "GenFeasibleStarts", ""),
        ("GenCorridorStarts-v0", "GenCorridorStarts", ""),
        ("RandomGS-v0", "RandomGS", "RandomGS"),
        ("RandomGS-fwd-v0", "RandomGS_FWD", "RandomGS"),
        ("RandomGS-bwd-v0", "RandomGS_BWD", "RandomGS"),
    ),
)

# Vanilla-PPO baseline without the curriculum; `-hient-` differs only by a higher-entropy runner.
_register_family(
    id_prefix="Parkour-BridgeDTSG",
    env_entry_point=PARKOUR_ENV,
    cfg_module="cfg.bridge.bridge_cfg_baseline",
    cfg_prefix="IEAnymalDBridgeDTSGEnvCfg",
    runner_module="anymal_d.rsl_rl_ppo_bridge_cfg",
    runner_prefix="AnymalDBridgeDTSG",
    arms=(
        ("Baseline-v0", "Baseline", ""),
        ("Baseline-hient-v0", "Baseline", "HighEntropy"),
    ),
)


# Real-world scan terrains, one arm family per SCAN_SPECS entry; missing scan data only fails at instantiation.

for _scan_spec in scan_specs.SCAN_SPECS.values():
    for _arm in scan_naming.ARM_NAMES:
        _register(
            scan_naming.task_id(_scan_spec, _arm),
            PARKOUR_ENV,
            f"cfg.scan.scan_families:{scan_naming.env_cfg_name(_scan_spec, _arm)}",
            (
                f"anymal_d.rsl_rl_ppo_scan_cfg:"
                f"{scan_naming.runner_name(_scan_spec, scan_naming.ARM_RUNNER[_arm])}"
            ),
        )
