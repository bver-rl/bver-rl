"""Task-space samplers: uniform, filtered-uniform, and feasible-pool.

Each :meth:`sample` returns ``(n, *task_space.task_dim())`` tasks ready for ``task_space.set_tasks``.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .task_space_base import TaskSpaceBase


# Configuration base


@configclass
class TaskSpaceSamplerCfg:
    """Base configuration for task-space samplers; subclasses set ``class_type``."""

    class_type: type = MISSING
    """Concrete sampler class to instantiate."""

    subspace: list[int] | None = None
    """Flat indices into the ``(total_dims,)`` task vector that this sampler fills; the rest are zeroed.
    ``None`` means all dimensions."""

    eval_bounds: bool = False
    """If True use ``task_space.cfg.eval_bounds`` instead of the training bounds."""


class TaskSpaceSampler(ABC):
    """Abstract base class for task-space samplers.

    Args:
        cfg: Sampler configuration.
        task_space: Task space providing bounds, scale, and filter helpers.
        device: Torch device for output tensors.
    """

    def __init__(
        self,
        cfg: TaskSpaceSamplerCfg,
        task_space: "TaskSpaceBase",
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.task_space = task_space
        self.device = device

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """Draw ``n`` tasks.

        Returns:
            Tensor of shape ``(n, *task_space.task_dim())``.
        """
        ...


# Concrete samplers


class UniformSampler(TaskSpaceSampler):
    """Uniform draw from task-space bounds (no filtering)."""

    def sample(self, n: int) -> torch.Tensor:
        total = self.task_space.total_task_dim()
        tasks = torch.rand(n, 1, total, device=self.device)
        scaled = self.task_space.scale_to_bounds(
            tasks,
            eval_bounds=self.cfg.eval_bounds,
            subspace=self.cfg.subspace,
        )
        return scaled.reshape(n, *self.task_space.task_dim())


class FilteredUniformSampler(TaskSpaceSampler):
    """Uniform draw from task-space bounds with sample-filter applied."""

    def sample(self, n: int) -> torch.Tensor:
        total = self.task_space.total_task_dim()
        tasks = torch.rand(n, 1, total, device=self.device)
        scaled = self.task_space.scale_to_bounds(
            tasks,
            eval_bounds=self.cfg.eval_bounds,
            subspace=self.cfg.subspace,
        )
        filtered = self.task_space.filter_tasks(
            scaled,
            eval_bounds=self.cfg.eval_bounds,
            subspace=self.cfg.subspace,
            filter_override=self.task_space.get_goal_sample_filter() if self.cfg.use_goal_filter else None,
        )
        return filtered.reshape(n, *self.task_space.task_dim())


class FeasiblePoolSampler(TaskSpaceSampler):
    """Uniform draw (with replacement) from a pre-generated feasible-state pool.

    The pool is a ``.pkl`` holding ``{"tasks": Tensor(N, ..., 2), ...}``.
    """

    def __init__(
        self,
        cfg: "FeasiblePoolSamplerCfg",
        task_space: "TaskSpaceBase",
        device: torch.device,
    ) -> None:
        super().__init__(cfg, task_space, device)
        if not cfg.pool_path:
            raise ValueError("FeasiblePoolSamplerCfg.pool_path must be set")
        with open(os.path.join(os.getcwd(), cfg.pool_path), "rb") as f:
            data = pickle.load(f)
        self._pool: torch.Tensor = torch.as_tensor(data["tasks"], dtype=torch.float32)
        # pool stays on CPU; slices move to device in sample()

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, len(self._pool), (n,))
        tasks = self._pool[idx].to(self.device)  # (n, 18, 2)
        if self.cfg.subspace is not None:
            flat = tasks.reshape(n, -1)  # (n, total_dims)
            result = torch.zeros_like(flat)
            result[:, self.cfg.subspace] = flat[:, self.cfg.subspace]
            tasks = result.reshape(n, *self.task_space.task_dim())
        return tasks


# Concrete cfg classes, defined after the implementations so defaults can reference them


@configclass
class UniformSamplerCfg(TaskSpaceSamplerCfg):
    """Configuration for :class:`UniformSampler`."""

    class_type: type = UniformSampler


@configclass
class FilteredUniformSamplerCfg(TaskSpaceSamplerCfg):
    """Configuration for :class:`FilteredUniformSampler`."""

    class_type: type = FilteredUniformSampler

    use_goal_filter: bool = False
    """Reject with ``task_space.get_goal_sample_filter()`` instead of the start-side ``sample_filter``,
    for sampling goals."""


@configclass
class FeasiblePoolSamplerCfg(TaskSpaceSamplerCfg):
    """Configuration for :class:`FeasiblePoolSampler`."""

    class_type: type = FeasiblePoolSampler

    pool_path: str = ""
    """Absolute or relative path to the ``.pkl`` pool file."""
