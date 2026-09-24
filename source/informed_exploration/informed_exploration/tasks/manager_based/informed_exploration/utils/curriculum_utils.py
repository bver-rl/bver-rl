from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from ..curriculum.task_space.samplers import (
    FilteredUniformSamplerCfg,
    UniformSamplerCfg,
    TaskSpaceSamplerCfg,
)
from ..curriculum.task_space.task_space_base import TaskSpaceBase
from .tail_split import split_random_tail

if TYPE_CHECKING:
    from ..envs.base_env_cfg import BaseEnv


def task_bbox(
    collection: list | deque,
    xyz_dims: tuple[int, int, int],
    prefix: str,
) -> dict:
    """Axis-aligned bounding box over the xyz dimensions of a task collection.

    Args:
        collection: List or deque of same-shape tensors, flattened before indexing.
        xyz_dims: Indices of x, y and z into the flattened task vector.
        prefix: Key prefix for the returned dict.

    Returns:
        Six-entry dict ``{prefix}_bbox_{x,y,z}_{min,max}``, or ``{}`` when the collection is empty.
    """
    if len(collection) == 0:
        return {}
    flat = torch.stack(list(collection)).reshape(len(collection), -1)  # [N, D]
    t_min = flat.min(dim=0).values
    t_max = flat.max(dim=0).values
    x, y, z = xyz_dims
    return {
        f"{prefix}_bbox_x_min": t_min[x].item(),
        f"{prefix}_bbox_x_max": t_max[x].item(),
        f"{prefix}_bbox_y_min": t_min[y].item(),
        f"{prefix}_bbox_y_max": t_max[y].item(),
        f"{prefix}_bbox_z_min": t_min[z].item(),
        f"{prefix}_bbox_z_max": t_max[z].item(),
    }


def get_task_space(env):
    """Return the task space from env, asserting it exists."""
    task_space = getattr(env, "task_space", None)
    assert task_space is not None, "env does not have a task space"
    return task_space


def compute_episode_sums(env, env_ids, reward_terms="all") -> torch.Tensor:
    """Aggregate episodic reward sums over ``env_ids``.

    Args:
        env: The RL environment.
        env_ids: Indices of the environments to aggregate.
        reward_terms: ``"all"`` to sum all reward terms, or a list of term names to sum.

    Returns:
        Summed episodic reward per environment, shape ``(len(env_ids),)``.
    """
    if reward_terms == "all":
        return torch.concat([v.unsqueeze(1) for _, v in env.reward_manager._episode_sums.items()], axis=1)[env_ids].sum(
            axis=1
        )
    return torch.stack([v for k, v in env.reward_manager._episode_sums.items() if k in reward_terms], axis=1)[
        env_ids
    ].sum(axis=1)


def resolve_xyz_dims(task_space) -> tuple[int, int, int]:
    """Return the (x, y, z) dimension indices into the flattened task vector."""
    return getattr(task_space.cfg, "xyz_dimensions", (0, 1, 2)) if hasattr(task_space, "cfg") else (0, 1, 2)


def resolve_anchor_goal_xyz(env) -> "torch.Tensor | None":
    """The env-relative xyz of the curriculum's anchor goal, or ``None`` if it has none.

    ``anchor_goal`` is either a full task state (BVER) or a plain xyz (RandomGoalStart).
    """
    import torch

    # May run right after `gym.make` on a partly built env, so assume only the cfg and task space exist.
    device = getattr(env, "device", "cpu")

    cfg = getattr(getattr(env, "cfg", None), "curriculum", None)
    if cfg is None:
        cfg = getattr(getattr(env, "curriculum_manager", None), "cfg", None)
    if cfg is None:
        return None

    for term in vars(cfg).values():
        anchor = getattr(term, "anchor_goal", None)
        if anchor:
            flat = torch.as_tensor(anchor, dtype=torch.float32, device=device).reshape(-1)
            if flat.numel() == 3:
                return flat
            return flat[list(resolve_xyz_dims(get_task_space(env)))]
    return None


def resolve_viz_origin(env, global_viz_subterrain=None, fallback_to_env_origin: bool = True):
    """Resolve a single world-space origin tensor for anchoring markers.

    Args:
        env: The RL environment.
        global_viz_subterrain: Optional ``(terrain_level, terrain_type)`` selecting the first env in that tile.
        fallback_to_env_origin: Whether to fall back to ``env_origins[0]`` instead of returning ``None``.

    Returns:
        A ``(3,)`` tensor for the chosen origin, or ``None``.
    """
    if global_viz_subterrain is not None:
        terrain = getattr(getattr(env, "scene", None), "terrain", None)
        if terrain is not None:
            target_level, target_type = global_viz_subterrain
            mask = (terrain.terrain_levels == target_level) & (terrain.terrain_types == target_type)
            if mask.any():
                return terrain.env_origins[mask][0]
    if fallback_to_env_origin:
        env_origins = getattr(getattr(env, "scene", None), "env_origins", None)
        if env_origins is not None and len(env_origins) > 0:
            return env_origins[0]
    return None


def make_sampler(
    sampler_cfg: TaskSpaceSamplerCfg | None,
    filtered_samples: bool,
    subspace: list[int] | None,
    task_space: TaskSpaceBase,
    device,
    eval_bounds: bool = False,
):
    """Construct a task-space sampler.

    Args:
        sampler_cfg: An explicit sampler config. When provided, the other arguments are ignored.
        filtered_samples: Whether to use ``FilteredUniformSamplerCfg`` when ``sampler_cfg`` is ``None``.
        subspace: Optional list of task dimension indices.
        task_space: The environment's task space.
        device: The torch device for the sampler.
        eval_bounds: When ``True``, sample within eval bounds instead of train bounds.

    Returns:
        An instantiated sampler.
    """
    if sampler_cfg is not None:
        return sampler_cfg.class_type(sampler_cfg, task_space, device)

    sub = list(subspace) if subspace is not None else None
    if filtered_samples:
        cfg = FilteredUniformSamplerCfg(subspace=sub, eval_bounds=eval_bounds)
    else:
        cfg = UniformSamplerCfg(subspace=sub, eval_bounds=eval_bounds)
    return cfg.class_type(cfg, task_space, device)


def resolve_env_splits(env: BaseEnv) -> tuple[int, int, int]:
    """Split envs into ``[train | eval | random]`` counts, with the rounding residual going to random.

    Must match ``rsl_rl.utils.resolve_env_splits`` exactly. Returns ``(train, random, eval)``, whereas
    rsl_rl returns ``(train, eval, random)``.
    """
    num_train_envs = int(env.num_envs * env.train_env_ratio)
    num_random_envs = int(env.num_envs * env.random_env_ratio)
    num_eval_envs = int(env.num_envs * (1.0 - env.train_env_ratio - env.random_env_ratio))
    residual = env.num_envs - (num_train_envs + num_eval_envs + num_random_envs)
    num_random_envs += residual  # leftover envs go to random

    return num_train_envs, num_random_envs, num_eval_envs


def random_tail_boundaries(env: BaseEnv) -> tuple[int, int, int]:
    """Boundaries of the random-action tail's ``[forward | backward | reachability]`` split.

    Returns ``(tail_start, backward_start, reachability_start)``, computed exactly as the RSL-RL runner
    does so both sides agree on which envs receive which noise.
    """
    return split_random_tail(
        env.num_envs,
        env.train_env_ratio,
        env.random_env_ratio,
        getattr(env, "random_reachability_ratio", 0.0),
        getattr(env, "random_backward_ratio", 0.0),
    )
