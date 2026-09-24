"""Turn a platform-start trajectory harvest into a sim-free DTSG RSI reference-trajectory pool.

Success is recomputed from ``root_pos_w`` (the dumped reward sum includes dense terms) and routes are labelled by
the start frame; writes ``trajectories`` of shape (N, T, 37) plus per-episode ``levels`` and ``types``.
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle

import torch

# Env-relative route geometry duplicated from terrains/config/layouts/dtsg.py, which needs pxr to import.
ROUTE_CENTERS_Y = {"stairs": 0.0, "pillars": 1.8, "ramp": -1.8}
ROUTE_IDS = {"stairs": 0, "pillars": 1, "ramp": 2}
ROUTE_HALF_WIDTH = 0.9  # route half-width plus margin, unambiguous given the center spacing

# Env-relative goal: goal pad center plus pose offset, one nominal standing height above the pad top.
DEFAULT_GOAL = (7.0, 0.0, 0.65)

# The sparse tracking_pos kernel pays out below this distance.
DEFAULT_SUCCESS_RADIUS = 0.25

TRAJ_DIM = 37


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DTSG RSI reference-trajectory pool from a platform-start harvest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        metavar="PKL_GZ",
        help=(
            "One or more trajectories_<model>.pkl.gz dumps from"
            " 'eval_platform_starts.py --store_trajectories'. Passing several (e.g. two checkpoints,"
            " or two harvest runs) pools them before the quota is applied."
        ),
    )
    parser.add_argument(
        "--output",
        default="informed-exploration/data/parkour/trajectories_bridge_dtsg.pkl",
        help="Where to write the RSI pool. Must match the RSI env cfg's optimal_trajectory_file.",
    )
    parser.add_argument(
        "--per_route",
        type=int,
        default=100,
        help="Successful trajectories to keep per route. Equal by construction; short routes error.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seeds the per-route subsample, so a rerun picks the same set."
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        default=list(DEFAULT_GOAL),
        metavar=("X", "Y", "Z"),
        help="Goal position (env-relative) the success test measures against.",
    )
    parser.add_argument(
        "--success_radius",
        type=float,
        default=DEFAULT_SUCCESS_RADIUS,
        help="An episode counts as successful if any frame is within this 3-D distance of the goal.",
    )
    return parser.parse_args()


def load_dump(path: str) -> list:
    """Return the raw entry list from one ``trajectories_<model>.pkl.gz`` harvest dump."""
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    entries = payload.get("trajectories") if isinstance(payload, dict) else None
    if entries is None:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        basename = os.path.basename(path)
        if not basename.startswith("trajectories_"):
            sibling = os.path.join(os.path.dirname(path), f"trajectories_{basename}")
            hint = (
                f"\n  This looks like the results pickle, not the trajectory dump. Try:\n    {sibling}"
                if os.path.exists(sibling)
                else (
                    "\n  This looks like the results pickle, not the trajectory dump; the dump is the"
                    f" sibling 'trajectories_{basename}', which is missing, so re-run the harvest with"
                    " --store_trajectories."
                )
            )
        else:
            hint = "\n  Re-run the harvest with --store_trajectories."
        raise ValueError(f"{path}: no 'trajectories' key (found: {keys}).{hint}")
    return list(entries)


def entry_trajectory(entry) -> torch.Tensor:
    """Pull the (T, 37) state tensor out of one recorded 4-tuple or 2-tuple entry."""
    if isinstance(entry, (tuple, list)):
        for item in entry:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
        raise ValueError(f"no (T, D) tensor in entry of length {len(entry)}")
    if torch.is_tensor(entry) and entry.ndim == 2:
        return entry
    raise ValueError(f"unrecognized trajectory entry type: {type(entry)}")


def label_route(traj: torch.Tensor) -> str | None:
    """Return the route name from the y of the first frame, or ``None`` if it lies between the bands."""
    y = float(traj[0, 1])
    name, center = min(ROUTE_CENTERS_Y.items(), key=lambda kv: abs(y - kv[1]))
    return name if abs(y - center) <= ROUTE_HALF_WIDTH else None


def reached_goal(traj: torch.Tensor, goal: torch.Tensor, radius: float) -> bool:
    """Reproduce the sparse-reward success test: any frame within ``radius`` of the goal in 3-D."""
    return bool((torch.linalg.norm(traj[:, :3] - goal, dim=-1) < radius).any())


def pad_front(traj: torch.Tensor, length: int) -> torch.Tensor:
    """Right-align ``traj`` to ``length`` frames, repeating its first frame at the front.

    The pool is indexed from the end, so the goal side of a trajectory must stay put.
    """
    if traj.shape[0] == length:
        return traj
    if traj.shape[0] > length:
        return traj[-length:]
    pad = traj[:1].expand(length - traj.shape[0], -1)
    return torch.cat([pad, traj], dim=0)


def main() -> None:
    args = parse_args()
    goal = torch.tensor(args.goal, dtype=torch.float32)

    entries: list = []
    for path in args.trajectories:
        loaded = load_dump(path)
        print(f"[INFO]: {path}: {len(loaded)} episodes")
        entries.extend(loaded)
    if not entries:
        raise SystemExit("no episodes loaded")

    by_route: dict[str, list[torch.Tensor]] = {name: [] for name in ROUTE_CENTERS_Y}
    n_failed = 0
    n_unlabeled = 0
    for entry in entries:
        traj = entry_trajectory(entry).float()
        if traj.shape[1] != TRAJ_DIM:
            raise ValueError(f"expected {TRAJ_DIM} state columns, got {traj.shape[1]}")
        route = label_route(traj)
        if route is None:
            n_unlabeled += 1
            continue
        if not reached_goal(traj, goal, args.success_radius):
            n_failed += 1
            continue
        by_route[route].append(traj)

    print(f"\n[INFO]: {len(entries)} episodes -> {sum(len(v) for v in by_route.values())} successes")
    print(f"         {n_failed} failed (never within {args.success_radius} m of the goal)")
    if n_unlabeled:
        print(f"         {n_unlabeled} unlabeled (start y outside every route band), unexpected here")
    print(f"\n  {'route':<10}{'successes':>11}{'kept':>7}")
    for name in ROUTE_CENTERS_Y:
        kept = min(len(by_route[name]), args.per_route)
        print(f"  {name:<10}{len(by_route[name]):>11}{kept:>7}")

    short = {name: len(v) for name, v in by_route.items() if len(v) < args.per_route}
    if short:
        raise SystemExit(
            f"\n[ERROR]: --per_route {args.per_route} not met for {sorted(short)} (have"
            f" {short}).\n         Re-run the harvest with a larger --num_samples, or with a"
            " checkpoint that solves those routes; the pool must be balanced across routes to be a"
            " valid RSI reference."
        )

    generator = torch.Generator().manual_seed(args.seed)
    kept: list[torch.Tensor] = []
    types: list[torch.Tensor] = []
    for name in sorted(ROUTE_CENTERS_Y):  # sorted so the pool order is seed-reproducible
        pool = by_route[name]
        pick = torch.randperm(len(pool), generator=generator)[: args.per_route]
        for idx in pick.tolist():
            kept.append(pool[idx])
            types.append(torch.tensor(ROUTE_IDS[name], dtype=torch.int64))

    length = max(t.shape[0] for t in kept)
    lengths = {t.shape[0] for t in kept}
    if len(lengths) > 1:
        print(f"\n[INFO]: episode lengths {sorted(lengths)} -> front-padding all to {length}")
    trajectories = torch.stack([pad_front(t, length) for t in kept], dim=0)

    payload = {
        "trajectories": trajectories,
        "levels": [torch.tensor(0, dtype=torch.int64) for _ in kept],
        "types": types,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n[INFO]: wrote {args.output}  ({tuple(trajectories.shape)}, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
