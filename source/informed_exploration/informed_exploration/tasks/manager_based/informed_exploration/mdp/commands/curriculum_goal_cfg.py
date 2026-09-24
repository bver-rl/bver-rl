from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .curriculum_goal import CurriculumGoalCommand


@configclass
class CurriculumGoalCfg(CommandTermCfg):

    class_type: type = CurriculumGoalCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    distance_threshold: float = 0.25
    """Distance (m) under which ``achieved_goal`` reports success in the command metrics."""

    default_goal: list[float] | None = None
    """Optional env-relative xyz goal written on every reset by ``_resample_command``.

    Leave ``None`` when a curriculum writes goals via ``set_goal``. Set it for eval/play configs, otherwise the
    policy receives an all-zeros goal."""

    body_name: str | None = None
    """Optional link the goal is measured against, for a fixed-base robot; ``None`` measures the asset root."""

    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the tracked point in that link's frame."""

    marker_size: float | None = None
    """Edge length of the goal marker cube; ``None`` keeps the default cube."""

