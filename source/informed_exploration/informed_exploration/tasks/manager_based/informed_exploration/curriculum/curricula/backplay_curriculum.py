from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass

from ...utils.curriculum_utils import get_task_space
from ..curriculum_handlers import parallel_eval

if TYPE_CHECKING:
    from ...envs.base_env_cfg import BaseEnv


# Doubling-width windows over fractions of max_step (threshold) and of the trajectory counted back from the goal.
DEFAULT_WINDOW_SCHEDULE = [
    (0.0, (0.0, 1 / 15)),
    (0.2, (1 / 15, 2 / 15)),
    (0.4, (2 / 15, 4 / 15)),
    (0.6, (4 / 15, 8 / 15)),
    (0.8, (8 / 15, 1.0)),
    (1.0, (1.0, 1.0)),
]


class BackplayCurriculum(ManagerTermBase):
    """Backplay (Resnick et al. 2018): reset near the end of a demonstration and move the reset window backward.

    Each reset draws a demo state from the window the schedule pins to the current iteration. Both
    window edges slide, so the terminal stage is plain RL from ``s_0``; advancement is never success-gated.
    """

    cfg: "BackplayCfg"

    def __init__(self, cfg: "BackplayCfg", env: BaseEnv):
        self.cfg = cfg
        self.task_space = get_task_space(env)

        # thresholds resolve to iterations once; windows resolve per reset against the loaded pool
        assert cfg.window_schedule, "BackplayCfg.window_schedule must contain at least one (fraction, (j, k)) entry"
        assert cfg.max_step is not None, (
            "BackplayCfg.max_step has no default; the right value is the iteration this run's"
            " curriculum should finish at, which only the env cfg knows. Set it there, e.g."
            " BackplayCfg(max_step=1000)."
        )
        assert cfg.max_step >= 0, f"BackplayCfg.max_step must be non-negative; got {cfg.max_step}"
        self._max_step: int = int(cfg.max_step)

        fractions = sorted(
            [(float(threshold), (float(low), float(high))) for threshold, (low, high) in cfg.window_schedule],
            key=lambda entry: entry[0],
        )
        assert fractions[0][0] <= 0.0, (
            "BackplayCfg.window_schedule must start at fraction 0.0; got a first threshold of" f" {fractions[0][0]}"
        )
        assert fractions[-1][0] <= 1.0, (
            "BackplayCfg.window_schedule thresholds are fractions of max_step and must lie in [0, 1];"
            f" got a last threshold of {fractions[-1][0]}"
        )
        assert all(0.0 <= low <= 1.0 and 0.0 <= high <= 1.0 for _, (low, high) in fractions), (
            "BackplayCfg.window_schedule windows are fractions of the trajectory length and must lie"
            f" in [0, 1]; got {[w for _, w in fractions]}"
        )
        self._schedule = self.to_absolute_schedule(fractions, self._max_step)

    @staticmethod
    def to_absolute_schedule(
        fractions: list[tuple[float, tuple[float, float]]], max_step: int
    ) -> list[tuple[int, tuple[float, float]]]:
        """Scale fractional thresholds into absolute iteration thresholds, leaving the windows fractional.

        A ``max_step`` of zero collapses every threshold, so only the final window applies.
        """
        return [(int(round(fraction * max_step)), window) for fraction, window in fractions]

    @staticmethod
    def resolve_window(
        schedule: list[tuple[int, tuple[float, float]]], iteration: int, length: int
    ) -> tuple[int, int, int]:
        """Resolve the schedule at ``iteration`` into inclusive index bounds on a pool of ``length`` steps.

        Windows are half-open fractions ``[j, k)`` counted backward from the trajectory end and clamped;
        an empty window collapses to its lowest offset.

        Returns:
            ``(stage_idx, low_index, high_index)`` with both index bounds inclusive.
        """
        stage_idx = 0
        for i, (threshold, _) in enumerate(schedule):
            if iteration >= threshold:
                stage_idx = i
            else:
                break
        low_fraction, high_fraction = schedule[stage_idx][1]

        low_offset = max(0, min(int(round(low_fraction * length)), length - 1))
        high_offset = max(0, min(int(round(high_fraction * length)), length))
        if high_offset <= low_offset:
            high_offset = low_offset + 1

        return stage_idx, length - high_offset, length - 1 - low_offset

    def __call__(self, env: BaseEnv, env_ids: torch.Tensor) -> dict:
        env_ids = parallel_eval(env, env_ids)

        if len(env_ids) == 0:
            return {}

        # One epoch is one PPO iteration, i.e. num_steps_per_env environment steps.
        steps = getattr(env.unwrapped, "num_steps_per_env", None) or 1
        iteration = env.common_step_counter // steps

        if self.cfg.terrain_level:
            terrain: TerrainImporter = env.scene.terrain
            levels = (
                terrain.terrain_levels[env_ids].cpu()
                if not self.cfg.one_level
                else torch.ones_like(env_ids).cpu() * self.cfg.terrain_level
            )
            types = terrain.terrain_types[env_ids].cpu()

            terrain_level_trajectories = env.train_data.get("terrain_level_trajectories", None)
            assert terrain_level_trajectories is not None, "No terrain level trajectories found in train_data"

            tasks = []
            for i in range(len(env_ids)):
                trajs = torch.stack(terrain_level_trajectories[levels[i]][types[i]], dim=0)
                traj_idx = torch.randint(0, trajs.shape[0], (1,)).item()
                random_trajectory = trajs[traj_idx]

                stage_idx, low_idx, high_idx = self.resolve_window(self._schedule, iteration, trajs.shape[1])
                step_idx = torch.randint(low_idx, high_idx + 1, (1,)).item()
                tasks.append(random_trajectory[step_idx])

            tasks = torch.stack(tasks, dim=0).to(env.device)
        else:
            trajectories = env.train_data.get("trajectories", None)
            assert trajectories is not None, "No trajectories found in train_data"

            trajectories = trajectories["trajectories"].to(env.device)

            traj_idx = torch.randint(0, trajectories.shape[0], (len(env_ids),), device=env.device)
            random_trajectories = trajectories[traj_idx]

            stage_idx, low_idx, high_idx = self.resolve_window(self._schedule, iteration, trajectories.shape[1])
            step_idx = torch.randint(low_idx, high_idx + 1, (len(env_ids),), device=env.device)
            tasks = random_trajectories[torch.arange(len(env_ids), device=env.device), step_idx]

        tasks = self.task_space.obs_to_task(tasks.unsqueeze(1)).reshape(
            len(env_ids),
            *self.task_space.task_dim(),
        )
        self.task_space.set_tasks(env, env_ids, tasks)

        return {
            "backplay_stage": stage_idx,
            "schedule_progress": min(iteration / self._max_step, 1.0) if self._max_step > 0 else 1.0,
            "window_low_fraction": self._schedule[stage_idx][1][0],
            "window_high_fraction": self._schedule[stage_idx][1][1],
            "range_low_idx": low_idx,
            "range_size": high_idx - low_idx + 1,
        }


@configclass
class BackplayCfg(CurriculumTermCfg):

    func: callable = BackplayCurriculum

    window_schedule: list = DEFAULT_WINDOW_SCHEDULE
    """Step-function ``(threshold, (j, k))`` entries; threshold is a fraction of :attr:`max_step`, the
    window a half-open fraction of the trajectory counted back from the goal. First threshold must be zero."""

    max_step: int | None = None
    """Iteration (``common_step_counter // num_steps_per_env``) at which the curriculum finishes; required.

    Training past it is plain RL from ``s_0``. Zero switches the curriculum off.
    """

    terrain_level: bool | int = False
    """Index into ``train_data["terrain_level_trajectories"]``, or ``False`` for the flat pool.

    Truthiness-gated, so level zero reads as ``False``."""

    one_level: bool = False
    """Pin every env to :attr:`terrain_level` instead of reading each env's own terrain level."""
