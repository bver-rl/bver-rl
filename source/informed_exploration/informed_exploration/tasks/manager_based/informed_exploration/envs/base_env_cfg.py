import os
import pickle
import torch

from isaaclab.envs import ManagerBasedRLEnv
from abc import ABC
from collections import deque
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from ..curriculum.task_space.task_space_base import TaskSpaceBase
from ..utils.logger import Logger
from .train_eval_reset import TrainEvalResetMixin


class BaseEnv(TrainEvalResetMixin, ManagerBasedRLEnv, ABC):
    cfg: ManagerBasedRLEnvCfg
    train_data = {}
    eval_data = {}
    num_steps_per_env = None
    train_env_ratio = 1.0
    random_env_ratio = 0.0
    random_reachability_ratio = 0.0
    random_backward_ratio = 0.0
    logger: Logger
    task_space: TaskSpaceBase
    # trajectory_buffer: deque #deprecated, moved to curriculum handler

    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)

        filename = self.task_space.get_optimal_trajectory_file()
        if filename is not None:
            if os.path.exists(filename):
                print(f"[INFO]: Loading optimal trajectories from: {filename}")
                with open(filename, "rb") as f:
                    optimal_trajectories = pickle.load(f)

                num_types = max(optimal_trajectories["types"]) + 1
                num_levels = max(optimal_trajectories["levels"]) + 1

                terrain_level_trajectories = [
                    [
                        [
                            traj
                            for traj, l, t in zip(
                                optimal_trajectories["trajectories"],
                                optimal_trajectories["levels"],
                                optimal_trajectories["types"],
                            )
                            if l == level and t == terrain_type
                        ]
                        for terrain_type in range(num_types)
                    ]
                    for level in range(num_levels)
                ]

                self.train_data = {
                    "trajectories": optimal_trajectories,
                    "terrain_level_trajectories": terrain_level_trajectories,
                    "traj_index": torch.zeros(len(optimal_trajectories["trajectories"][0]), dtype=torch.int).cpu(),
                    "last_rewards": torch.zeros(self.num_envs).cpu(),
                }
        else:
            print(
                "[INFO]: No optimal trajectory file specified in the task space config. Skipping loading optimal trajectories."
            )

    def get_action_reset_state(self) -> torch.Tensor:
        """Return raw actions that reproduce each env's current pose, shape ``(num_envs, total_action_dim)``.

        Used to warm-start the action-noise filter on reset. Only rows of just-reset envs are valid; other
        rows hold their last commanded action.
        """
        action_manager = self.action_manager
        return torch.cat(
            [action_manager.get_term(name).raw_actions for name in action_manager.active_terms], dim=-1
        )
