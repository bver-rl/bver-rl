"""Vanilla-PPO baseline env configs for bridge-terrain tasks: the BVER env without its curriculum.

Spawns on the start platform with a fixed 3D goal; checkpoints are evaluated through the BVER eval task.
"""

from isaaclab.utils import configclass

from ..common.env_cfg import XYZTrajectoryObservationsCfg
from ..common.goal_conditioning import (
    apply_baseline_spawn,
    apply_eval_goal_conditioning,
)
from .bridge_cfg import (
    IEAnymalDBridgeDTSGEnvCfg,
)


@configclass
class IEAnymalDBridgeDTSGEnvCfg_Baseline(IEAnymalDBridgeDTSGEnvCfg):
    observations = XYZTrajectoryObservationsCfg()  # xyz everywhere on DTSG, see the base cfg
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        apply_eval_goal_conditioning(self)
        apply_baseline_spawn(self)
        # Without a curriculum, draw spawn x and y uniformly over the eval window each reset.
        self.events.reset_base.params["pose"]["x"] = tuple(self.task_space.eval_bounds[0])
        self.events.reset_base.params["pose"]["y"] = tuple(self.task_space.eval_bounds[2])


