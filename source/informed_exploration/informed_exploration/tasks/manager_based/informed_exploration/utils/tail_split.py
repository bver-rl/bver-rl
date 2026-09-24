"""Dependency-free arithmetic for the random-action tail's ``[forward | backward | reachability]`` split.

Single source of truth shared by the RSL-RL runner and the curriculum.
"""

from __future__ import annotations


def split_random_tail(
    num_envs: int,
    train_ratio: float,
    random_ratio: float,
    reachability_ratio: float,
    backward_ratio: float,
) -> tuple[int, int, int]:
    """Boundaries of the random-action tail's three-way split.

    Args:
        num_envs: Total number of envs.
        train_ratio: Fraction of train envs.
        random_ratio: Fraction of random-action envs.
        reachability_ratio: Fraction of the whole random tail used for reachability walks.
        backward_ratio: Fraction of the remaining (post-reachability) tail used for backward walks.

    Returns:
        ``(tail_start, backward_start, reachability_start)`` delimiting the forward, backward and
        reachability blocks.
    """
    num_train_envs = int(num_envs * train_ratio)
    num_random_envs = int(num_envs * random_ratio)
    num_eval_envs = int(num_envs * (1.0 - train_ratio - random_ratio))
    residual_envs = num_envs - (num_train_envs + num_eval_envs + num_random_envs)
    num_random_envs += residual_envs  # add any residual envs to random, matching resolve_env_splits

    tail_start = num_train_envs + num_eval_envs
    num_reachability_random = int(num_random_envs * reachability_ratio)
    reachability_start = num_envs - num_reachability_random

    # Keep this truncation order; subtracting int(frontier_size * backward_ratio) rounds differently.
    frontier_size = reachability_start - tail_start
    backward_start = tail_start + int(frontier_size * (1.0 - backward_ratio))
    return tail_start, backward_start, reachability_start
