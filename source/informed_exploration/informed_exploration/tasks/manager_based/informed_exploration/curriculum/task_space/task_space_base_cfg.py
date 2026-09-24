import torch

from isaaclab.utils import configclass


@configclass
class TaskSpaceBaseCfg:

    # (joint, terms per joint)
    dimensions: tuple[int, int] | None = None

    total_dims: int | None = None

    bounds: list | None = None

    eval_bounds: list | None = None

    xyz_dimensions: tuple | None = None

    task_types: dict | None = None

    subterm_dims: dict | None = None

    tasks: dict | None = None

    sample_filter: list[tuple[tuple[float, float], ...]] | None = None

    goal_sample_filter: list[tuple[tuple[float, float], ...]] | None = None
    """Keep-out regions for goal sampling, distinct from ``sample_filter`` (start-state validity).

    ``None`` falls back to ``sample_filter`` in :meth:`TaskSpaceBase.get_sample_filter`.
    """

    filter_dimensions: tuple | None = None

    keep_in_filter: list[tuple[tuple[float, float], ...]] | None = None
    """Regions a task must lie inside (at least one of) to be valid; applies to both starts and goals.

    :meth:`TaskSpaceBase.get_filter_mask` rejects a task inside any keep-out region or outside every
    keep-in region. ``None`` disables it.
    """

    keep_in_dimensions: tuple | None = None
    """Dimensions indexed by ``keep_in_filter``; ``None`` falls back to ``filter_dimensions``."""

    optimal_trajectory_file: str | None = None

    default_task: list | None = None
    """Flat ``(total_dims,)`` nominal task that fills dimensions outside a subspace in
    :meth:`TaskSpaceBase.mask_to_subspace`. Required by curricula that mask tasks to a subspace."""
