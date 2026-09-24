# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate, visualize, and replay a pool of feasible start states for any `random_curriculum` task.

Modes: `generate` (random init, settle, keep feasible states), `visualize` (matplotlib only, no Isaac Sim) and
`replay` (reset from the pool and report survival). Non-box terrains must pass ``--surface_top_z``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

# Argument parsing (split so `visualize` mode can skip the Isaac Sim launch)

parser = argparse.ArgumentParser(description="Generate / visualize / replay feasible start states.")
parser.add_argument(
    "--mode",
    type=str,
    default="generate",
    choices=["generate", "visualize", "replay"],
    help="What to do. `visualize` does not launch IsaacSim.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Parkour-ClimbSingleBox-GenFeasibleStarts-0p41-v0",
    help="Task id. Use a variant whose curriculum is random_curriculum (e.g. *-GenFeasibleStarts-v0 or *-Random-v0).",
)
parser.add_argument("--num_envs", type=int, default=64, help="Parallel envs for generation/replay.")
parser.add_argument(
    "--num_states",
    type=int,
    default=1000,
    help="Target number of states to collect (generate) / replay (replay). In replay mode 0 means the whole pool.",
)
parser.add_argument("--settle_steps", type=int, default=20, help="Settle steps under zero action.")
parser.add_argument(
    "--source",
    type=str,
    default="random",
    choices=["random"],
    help="Generation source. Only Method 1 (random) is implemented in this iteration.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output .pkl path. Defaults to data/parkour/feasible_starts_climb_box_{height}.pkl.",
)
parser.add_argument("--pool", type=str, default=None, help="Existing pool .pkl to load (for visualize/replay).")
parser.add_argument("--replay_steps", type=int, default=100, help="Steps to step after each replay reset.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Stub to match play arguments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--live_viz",
    action="store_true",
    help="Show 3D markers in IsaacSim for accepted states during generation.",
)
parser.add_argument(
    "--max_iters",
    type=int,
    default=1000,
    help="Safety cap on outer generation iterations.",
)
parser.add_argument(
    "--surface_top_z",
    type=float,
    default=None,
    help=(
        "Env-local z of the highest walkable surface, used as the top of the feasibility "
        "predicate's base-z window. Defaults to the task space's `box_height`; required for "
        "terrains that have none (e.g. --surface_top_z 0.55 for DTSG)."
    ),
)
parser.add_argument(
    "--subspace",
    type=str,
    default=None,
    choices=["pos", "pos_yaw", "pos_vel", "pos_vel_yaw", "base", "all"],
    help=(
        "Override which task-space dimensions the curriculum samples from during generation. "
        "Omit to keep whatever the env cfg's curriculum term declares (e.g. the DTSG "
        "GenFeasibleStarts binding's [0, 2, 4, 10] = base x/y/z + yaw). "
        "'pos': base x/y/z position only (flat indices 0/2/4). "
        "'pos_yaw': pos + yaw (flat indices 0/2/4/10). "
        "'pos_vel': base x/y/z position + velocity (flat indices 0-5). "
        "'pos_vel_yaw': pos_vel + yaw (flat indices 0-5 and 10). "
        "'base': full base pose + velocity (flat indices 0-11, adds roll/pitch/yaw). "
        "'all': full robot state including joints (all 36 flat dims). "
        "Dimensions outside the subspace are zeroed, which means default stance / level / at rest "
        "for the relative dims but TERRAIN LEVEL for base z, so dropping index 4 spawns the base "
        "in the ground."
    ),
)

# Parse the mode first; Hydra/Isaac Lab CLI args are only added when Isaac Sim launches.

_pre_args, _ = parser.parse_known_args()


# Visualize mode: matplotlib only

if _pre_args.mode == "visualize":
    args_cli = _pre_args
    if args_cli.pool is None:
        print("[ERROR] --pool is required for visualize mode.", file=sys.stderr)
        sys.exit(1)

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    with open(args_cli.pool, "rb") as f:
        pool = pickle.load(f)

    tasks = pool["tasks"]  # numpy (N, 18, 2)
    contact = pool["contact_pattern"]  # numpy (N, 4) values in {0, 1, 2}

    # Task layout: 6 base params (x, y, z, roll, pitch, yaw) then 12 joint params.
    base_x = tasks[:, 0, 0]
    base_z = tasks[:, 2, 0]

    # Collapse the 4-foot contact pattern to categories; box contact is the informative axis for the climb.
    n_ground = (contact == 1).sum(axis=1)
    n_box = (contact == 2).sum(axis=1)

    def categorize(g, b):
        if b == 0 and g > 0:
            return "all-ground"
        if b > 0 and g == 0:
            return "all-box"
        if b > 0 and g > 0:
            return "mixed (mid-climb)"
        return "no-contact"

    categories = np.array([categorize(g, b) for g, b in zip(n_ground, n_box)])
    palette = {
        "all-ground": "#4477AA",
        "all-box": "#EE6677",
        "mixed (mid-climb)": "#228833",
        "no-contact": "#CCBB44",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    bh = pool.get("metadata", {}).get("box_height", None)
    if bh is not None:
        ax.axhline(bh, color="grey", linewidth=1.0, linestyle=":", alpha=0.5, label=f"box top z={bh}")
        box_rect = mpatches.Rectangle(
            (-0.6, 0.0), 1.2, bh, linewidth=0.8, edgecolor="#aaaaaa", facecolor="#dddddd", zorder=0, label="box"
        )
        ax.add_patch(box_rect)

    for cat, color in palette.items():
        mask = categories == cat
        if mask.any():
            ax.scatter(base_x[mask], base_z[mask], s=8, c=color, alpha=0.7, label=f"{cat} (n={mask.sum()})", zorder=1)
    ax.set_xlabel("base x (env-local)")
    ax.set_ylabel("base z (env-local)")
    ax.set_title(
        f"Feasible-starts pool ({tasks.shape[0]} states):  box_height = {pool.get('metadata', {}).get('box_height', 'unknown')}"
    )
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out = Path(args_cli.pool).with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"[INFO] Wrote scatter plot to {out}")
    sys.exit(0)


# Generate / replay modes: launch Isaac Sim

from isaaclab.app import AppLauncher  # noqa: E402

# local imports
import os
import sys

# cli_args.py lives with the entry points in scripts/rsl_rl/; put that directory on the path (one-way dependency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "rsl_rl"))

import cli_args  # isort: skip  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import informed_exploration.tasks  # noqa: F401, E402
from informed_exploration.tasks.manager_based.informed_exploration.curriculum.feasibility import (  # noqa: E402
    feasibility_predicate,
    get_contact_pattern,
    resolve_foot_indices,
    snapshot_state,
)


def _infer_surface_top_z(env, args) -> float:
    """Return the env-local z of the highest walkable surface, the top of the feasibility base-z window.

    Prefers ``--surface_top_z``, then a terrain-published ``surface_top_z``, then the climb-box ``box_height``.
    """
    if args.surface_top_z is not None:
        return float(args.surface_top_z)
    # prefer a terrain-published surface over a hand-copied number that goes stale on rebuild
    generator = getattr(getattr(getattr(env.cfg.scene, "terrain", None), "terrain_generator", None), "sub_terrains", None)
    if generator:
        published = getattr(next(iter(generator.values())), "surface_top_z", None)
        if published is not None:
            return float(published)
    task_space = getattr(env, "task_space", None)
    assert task_space is not None and hasattr(task_space, "cfg"), "env has no task_space.cfg"
    box_height = getattr(task_space.cfg, "box_height", None)
    assert box_height is not None, (
        "task space has no `box_height`, so pass --surface_top_z explicitly (the env-local z of the "
        "highest walkable surface) for terrains that are not the climb box"
    )
    return float(box_height)


def _output_path(env_cfg, args) -> Path:
    if args.output is not None:
        return Path(args.output)
    box_h = getattr(env_cfg, "target_box_height", None)
    suffix = f"_{box_h:.2f}".replace(".", "p") if box_h is not None else ""
    return Path("/workspace/isaaclab/informed-exploration/data/parkour") / f"feasible_starts_climb_box{suffix}.pkl"


# Generate


def generate(env, args, env_cfg):
    """Sample random states, settle them and keep feasible ones until the pool reaches `args.num_states`."""
    core_env = env.unwrapped
    device = core_env.device
    num_envs = core_env.num_envs

    # Ensure parallel_eval treats all envs as train envs (so random_curriculum runs on all).
    core_env.train_env_ratio = 1.0
    core_env.random_env_ratio = 0.0

    task_space = core_env.task_space
    surface_top_z = _infer_surface_top_z(core_env, args)
    articulation_foot_ids, contact_foot_ids = resolve_foot_indices(core_env)

    pool_tasks: list[torch.Tensor] = []
    pool_contact: list[torch.Tensor] = []

    zero_actions = torch.zeros((num_envs, core_env.action_space.shape[1]), device=device)
    all_env_ids = torch.arange(num_envs, device=device)

    # Initial reset to apply the random curriculum to every env.
    env.reset()

    out_path = _output_path(env_cfg, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    iter_idx = 0
    total_collected = 0
    while total_collected < args.num_states and iter_idx < args.max_iters:
        iter_idx += 1

        # Track which envs hit a bad termination at any point during settle.
        bad_term_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for _ in range(args.settle_steps):
            _, _, terminated, _, _ = env.step(zero_actions)
            bad_term_mask |= terminated

        passed = feasibility_predicate(
            core_env,
            all_env_ids,
            bad_term_mask,
            contact_foot_ids,
            box_top_z=surface_top_z,
        )

        if passed.any():
            survived_ids = all_env_ids[passed]
            tasks = snapshot_state(core_env, survived_ids, task_space)
            contact = get_contact_pattern(
                core_env,
                survived_ids,
                articulation_foot_ids,
                contact_foot_ids,
            )
            pool_tasks.append(tasks.cpu())
            pool_contact.append(contact.cpu())
            total_collected += tasks.shape[0]

            if args.live_viz:
                _live_viz_add(core_env, tasks, survived_ids, marker_type="green")

        # Force another batch of random samples by resetting all envs.
        env.reset()

        if iter_idx % 10 == 0 or total_collected >= args.num_states:
            elapsed = time.time() - t0
            print(
                f"[generate] iter={iter_idx:4d}  collected={total_collected:6d}/{args.num_states}  "
                f"yield_this_iter={passed.sum().item():3d}/{num_envs}  elapsed={elapsed:6.1f}s",
                flush=True,
            )

    if not pool_tasks:
        print("[ERROR] Generation produced 0 feasible states. Loosen the predicate or check the task.")
        return

    tasks_cat = torch.cat(pool_tasks, dim=0)[: args.num_states]
    contact_cat = torch.cat(pool_contact, dim=0)[: args.num_states]

    payload = {
        "tasks": tasks_cat.numpy(),
        "contact_pattern": contact_cat.numpy(),
        "metadata": {
            "source": args.source,
            "settle_steps": args.settle_steps,
            "num_envs": num_envs,
            "task": args.task,
            "box_height": getattr(env_cfg, "target_box_height", None),
            "surface_top_z": surface_top_z,
            "iterations": iter_idx,
            "elapsed_s": time.time() - t0,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[generate] wrote {tasks_cat.shape[0]} states to {out_path}")


def _live_viz_add(core_env, tasks: torch.Tensor, env_ids: torch.Tensor, marker_type: str = "green"):
    """Push accepted states to the SampleVisualizer, offset by each env's origin."""
    if not hasattr(core_env, "sample_visualizer"):
        from informed_exploration.tasks.manager_based.informed_exploration.utils.sample_visualizer import (
            SampleVisualizer,
        )

        core_env.sample_visualizer = SampleVisualizer()
    xyz_dims = getattr(core_env.task_space.cfg, "xyz_dimensions", (0, 2, 4))
    origins = core_env.scene.env_origins[env_ids]
    core_env.sample_visualizer.add_samples(
        tasks.reshape(tasks.shape[0], -1),
        xyz_dims,
        origins,
        marker_type=marker_type,
    )


# Replay


def replay(env, args, env_cfg):
    """Apply each pool state as a reset, step `replay_steps` and count terminations."""
    if args.pool is None:
        print("[ERROR] --pool is required for replay mode.", file=sys.stderr)
        return

    core_env = env.unwrapped
    device = core_env.device
    num_envs = core_env.num_envs

    core_env.train_env_ratio = 1.0
    core_env.random_env_ratio = 0.0

    task_space = core_env.task_space

    with open(args.pool, "rb") as f:
        pool = pickle.load(f)

    pool_tasks = torch.from_numpy(pool["tasks"]).to(device)  # (N_pool, 18, 2)
    n_pool = pool_tasks.shape[0]
    print(f"[replay] loaded pool of {n_pool} states from {args.pool}")

    zero_actions = torch.zeros((num_envs, core_env.action_space.shape[1]), device=device)
    all_env_ids = torch.arange(num_envs, device=device)

    # initial reset; each batch then overrides the tasks via set_tasks before env.reset()
    env.reset()

    total_terminated_during = 0
    total_replayed = 0
    n_batches = (
        (args.num_states + num_envs - 1) // num_envs if args.num_states > 0 else (n_pool + num_envs - 1) // num_envs
    )
    sample_target = args.num_states if args.num_states > 0 else n_pool

    perm = torch.randperm(n_pool, device=device)
    cursor = 0

    for batch_idx in range(n_batches):
        if cursor + num_envs > n_pool:
            perm = torch.randperm(n_pool, device=device)
            cursor = 0
        sample_ids = perm[cursor : cursor + num_envs]
        cursor += num_envs

        batch_tasks = pool_tasks[sample_ids]  # (num_envs, 18, 2)

        # curriculum is None here, so reset applies the set_tasks params; env.reset() commits the physics state
        task_space.set_tasks(core_env, all_env_ids, batch_tasks)
        env.reset()

        bad_term_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        for _ in range(args.replay_steps):
            _, _, terminated, _, _ = env.step(zero_actions)
            bad_term_mask |= terminated

        total_terminated_during += int(bad_term_mask.sum().item())
        total_replayed += num_envs

        if total_replayed >= sample_target:
            break

    survived = total_replayed - total_terminated_during
    survival_rate = survived / max(1, total_replayed)
    print(
        f"[replay] replayed={total_replayed}  survived_full_{args.replay_steps}_steps={survived}  "
        f"survival_rate={survival_rate:6.2%}"
    )


# Hydra-wrapped main


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable domain randomization to focus on the curriculum samples themselves.
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand = None

    if args_cli.mode == "replay":
        # `_reset_idx` runs the curriculum before events, so disable it to let `set_tasks` win
        env_cfg.curriculum = None
    elif (
        args_cli.subspace is not None
        and hasattr(env_cfg, "curriculum")
        and env_cfg.curriculum is not None
        and hasattr(env_cfg.curriculum, "initialization")
    ):
        # only applied when --subspace is given; otherwise the env cfg's own subspace stands
        _subspace_map = {
            "pos": [0, 2, 4],
            "pos_yaw": [0, 2, 4, 10],
            "pos_vel": list(range(6)),
            "pos_vel_yaw": list(range(6)) + [10],
            "base": list(range(12)),
            "all": None,  # None is the task space's "sample every dim" sentinel, not "unset"
        }
        subspace = _subspace_map[args_cli.subspace]
        init_term = env_cfg.curriculum.initialization
        # `subspace` is a configclass field on RandomCfg-style terms but a `params` entry on function terms
        if hasattr(init_term, "subspace"):
            init_term.subspace = subspace
        elif hasattr(init_term, "params") and "subspace" in init_term.params:
            init_term.params["subspace"] = subspace
        else:
            print(
                f"[WARN] --subspace {args_cli.subspace} ignored: curriculum term "
                f"{type(init_term).__name__} exposes no `subspace` field or param.",
                file=sys.stderr,
            )
        # `filtered_samples` is left as the env cfg sets it

    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, log_dir="logs/feasible_starts", render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        if args_cli.mode == "generate":
            generate(env, args_cli, env_cfg)
        elif args_cli.mode == "replay":
            replay(env, args_cli, env_cfg)
        else:
            print(f"[ERROR] Unhandled mode after sim launch: {args_cli.mode}", file=sys.stderr)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
