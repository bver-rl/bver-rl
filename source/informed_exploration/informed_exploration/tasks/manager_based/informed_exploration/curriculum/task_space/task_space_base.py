import warnings
from abc import ABC, abstractmethod
from torch import Tensor
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .task_space_base_cfg import TaskSpaceBaseCfg

_MAX_RESAMPLE_ROUNDS = 256
"""Cap on :meth:`TaskSpaceBase.filter_tasks`' rejection-sampling loop; only a filter that keeps almost nothing
of the bounds (wrong frame or dimension index) should hit it."""


class TaskSpaceBase(ABC):

    def __init__(self, cfg: TaskSpaceBaseCfg):
        super().__init__()
        self.cfg = cfg

        self.cfg.subterm_ids = {}
        if self.cfg.subterm_dims is not None:
            offset = 0
            for k, v in self.cfg.subterm_dims.items():
                self.cfg.subterm_ids[k] = list(range(offset, offset + v[0]))
                offset += v[0]

    def task_dim(self) -> tuple[int, int]:
        return self.cfg.dimensions

    def total_task_dim(self) -> int:
        return self.cfg.total_dims

    def task_bounds(self, eval_bounds: bool = False) -> torch.Tensor:
        bounds = self.cfg.eval_bounds if eval_bounds else self.cfg.bounds
        return torch.tensor(bounds)

    def get_xyz_dimensions(self) -> tuple | None:
        return self.cfg.xyz_dimensions

    def get_task_cfg(self) -> dict:
        return self.cfg.tasks

    def get_sample_filter(self) -> list[tuple[tuple[float, float], ...]] | None:
        return self.cfg.sample_filter

    def get_goal_sample_filter(self) -> list[tuple[tuple[float, float], ...]] | None:
        """Filter region for goal sampling; falls back to :meth:`get_sample_filter` when unset."""
        return self.cfg.goal_sample_filter if self.cfg.goal_sample_filter is not None else self.cfg.sample_filter

    def get_filter_dimensions(self) -> tuple | None:
        return self.cfg.filter_dimensions

    def get_keep_in_filter(self) -> list[tuple[tuple[float, float], ...]] | None:
        """Regions a task must be inside at least one of; see ``TaskSpaceBaseCfg.keep_in_filter``.

        Read via ``getattr`` since some task-space cfgs do not subclass ``TaskSpaceBaseCfg``."""
        return getattr(self.cfg, "keep_in_filter", None)

    def get_keep_in_dimensions(self) -> tuple | None:
        """Dimensions :meth:`get_keep_in_filter`'s tuples index; falls back to the keep-out filter's."""
        dimensions = getattr(self.cfg, "keep_in_dimensions", None)
        return dimensions if dimensions is not None else self.cfg.filter_dimensions

    def get_optimal_trajectory_file(self) -> str | None:
        return self.cfg.optimal_trajectory_file

    @abstractmethod
    def set_tasks(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        task: Tensor,
    ) -> None:
        pass

    @abstractmethod
    def get_tasks(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ) -> Tensor:
        pass

    @abstractmethod
    def obs_to_task(
        self,
        obs: Tensor,
    ) -> Tensor:
        pass

    def scale_to_bounds(
        self,
        tasks: torch.Tensor,
        eval_bounds: bool = False,
        subspace: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Scale ``tasks``, given in ``[0, 1]``, to ``bounds``.

        Args:
            tasks: Shape ``(num envs, num params, terms per param)``.
            eval_bounds: Whether to use the evaluation bounds.
            subspace: Flat indices to keep; the rest are zeroed. ``None`` keeps everything.
        """
        num_envs, nr_joints, terms_per_joint = tasks.shape
        # reshape bounds from (nr. joints * terms per joint, 2) to (nr. joints, terms per joint, 2)
        bounds_reshaped = self.task_bounds(eval_bounds).view(nr_joints, terms_per_joint, 2).to(tasks.device)
        low = bounds_reshaped[..., 0]  # (nr. joints, terms per joint)
        high = bounds_reshaped[..., 1]  # (nr. joints, terms per joint)
        # Scale tasks from [0, 1] to [low, high], broadcasting over the env dimension

        scaled = low + tasks * (high - low)
        if subspace is not None:
            tasks_ = torch.zeros_like(tasks)
            tasks_[..., subspace] = scaled[..., subspace]
            scaled = tasks_

        return scaled

    def mask_to_subspace(
        self,
        tasks: torch.Tensor,
        subspace: Sequence[int] | None,
    ) -> torch.Tensor:
        """Keep the ``subspace`` dimensions of ``tasks`` and overwrite the rest with ``cfg.default_task``.

        Args:
            tasks: Full tasks, shape ``(*, *task_dim())``.
            subspace: Flat indices into the flattened task to keep; ``None`` keeps everything.
        """
        if subspace is None:
            return tasks
        assert self.cfg.default_task is not None, "mask_to_subspace requires cfg.default_task to be set"
        shape = tasks.shape
        flat = tasks.reshape(*shape[:-2], -1)
        default = torch.as_tensor(self.cfg.default_task, device=tasks.device, dtype=tasks.dtype)
        out = default.broadcast_to(flat.shape).clone()
        idx = torch.as_tensor(subspace, device=tasks.device, dtype=torch.long)
        out[..., idx] = flat[..., idx]
        return out.reshape(*shape)

    def _subspace_bounds(
        self,
        subspace: Sequence[int] | None,
        eval_bounds: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the flat ``(low, high)`` task bounds, optionally restricted to the flat indices ``subspace``."""
        bounds = self.task_bounds(eval_bounds).to(device).reshape(-1, 2)  # (total_dims, 2)
        if subspace is not None:
            bounds = bounds[torch.as_tensor(subspace, device=device, dtype=torch.long)]
        return bounds[..., 0], bounds[..., 1]

    def normalize(
        self,
        values: torch.Tensor,
        subspace: Sequence[int] | None = None,
        eval_bounds: bool = False,
    ) -> torch.Tensor:
        """Map flattened task values from their physical bounds to ``[0, 1]``; inverse of :meth:`denormalize`.

        Args:
            values: Task values in physical units, shape ``(..., D)``.
            subspace: Flat indices the values correspond to, or ``None`` for the full flat task.
            eval_bounds: Whether to use the evaluation bounds instead of the training bounds.
        """
        low, high = self._subspace_bounds(subspace, eval_bounds, values.device)
        return (values - low) / (high - low + 1e-8)

    def denormalize(
        self,
        values: torch.Tensor,
        subspace: Sequence[int] | None = None,
        eval_bounds: bool = False,
    ) -> torch.Tensor:
        """Map normalized ``[0, 1]`` task values back to their physical bounds; inverse of :meth:`normalize`."""
        low, high = self._subspace_bounds(subspace, eval_bounds, values.device)
        return low + values * (high - low)

    def filter_tasks(
        self,
        tasks: torch.Tensor,
        eval_bounds: bool = False,
        subspace: Sequence[int] | None = None,
        filter_override: list[tuple[tuple[float, float], ...]] | None = None,
    ) -> torch.Tensor:
        """Resample any task that falls inside the filter region.

        Args:
            tasks: Shape ``(num_envs, *dims)``.
            eval_bounds: Whether to resample within the evaluation bounds.
            subspace: Flat indices to resample; ``None`` resamples all.
            filter_override: Keep-out list used instead of :meth:`get_sample_filter`.
        """
        tasks_ = tasks.clone()

        for _ in range(_MAX_RESAMPLE_ROUNDS):
            mask = self.get_filter_mask(tasks_, filter_override=filter_override)

            if mask.sum() == 0:
                return tasks_

            tasks_[mask] = self.scale_to_bounds(
                torch.rand(mask.sum(), *tasks_.shape[1:], device=tasks_.device),
                eval_bounds=eval_bounds,
                subspace=subspace,
            ).squeeze()

        # Bounded so a filter with an empty valid set warns and returns the last draw instead of hanging.
        warnings.warn(
            f"filter_tasks did not converge in {_MAX_RESAMPLE_ROUNDS} rounds:"
            f" {int(self.get_filter_mask(tasks_, filter_override=filter_override).sum())} of"
            f" {tasks_.shape[0]} tasks still rejected. Either the filter regions are disjoint from the"
            " task-space bounds (check their frame and their dimension indices) or they keep so little"
            " of the bounds that the bounds themselves should be tightened instead.",
            stacklevel=2,
        )
        return tasks_

    def get_filter_mask(
        self,
        tasks: torch.Tensor,
        filter_override: list[tuple[tuple[float, float], ...]] | None = None,
    ) -> torch.Tensor:
        """Mask of rejected tasks: inside any keep-out region or outside every keep-in region.

        Args:
            tasks: Shape ``(num_envs, *dims)``.
            filter_override: Keep-out list used instead of :meth:`get_sample_filter`; keep-in is unaffected.
        """
        mask = torch.zeros(
            (tasks.shape[0], 1),
            dtype=torch.bool,
            device=tasks.device,
        )

        filter_list = filter_override if filter_override is not None else self.get_sample_filter()

        if filter_list is not None:
            for filter_ in filter_list:
                mask |= self.in_region(tasks, filter_)

        keep_in_list = self.get_keep_in_filter()
        if keep_in_list is not None:
            keep_in_dim = self.get_keep_in_dimensions()
            inside_any = torch.zeros_like(mask)
            for region in keep_in_list:
                inside_any |= self.in_region(tasks, region, dimensions=keep_in_dim)
            mask |= ~inside_any

        return mask

    def in_region(
        self,
        tasks: torch.Tensor,
        filter_list: tuple[tuple[float, float], ...],
        dimensions: tuple | None = None,
    ) -> torch.Tensor:
        """Mask of tasks inside the box ``filter_list``, one ``(low, high)`` per entry of ``dimensions``
        (defaults to :meth:`get_filter_dimensions`)."""

        filter_dim = dimensions if dimensions is not None else self.get_filter_dimensions()

        low = torch.tensor([f[0] for f in filter_list], device=tasks.device)
        high = torch.tensor([f[1] for f in filter_list], device=tasks.device)

        mask = ((tasks[:, :, filter_dim] >= low) & (tasks[:, :, filter_dim] <= high)).all(dim=-1)
        return mask
