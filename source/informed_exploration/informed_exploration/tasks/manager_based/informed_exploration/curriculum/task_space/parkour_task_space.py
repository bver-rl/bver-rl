from abc import ABC, abstractmethod
from torch import Tensor
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal
from isaaclab.utils.math import euler_xyz_from_quat

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .task_space_base import TaskSpaceBase
from .parkour_task_space_cfg import ParkourTaskSpaceCfg


class ParkourTaskSpace(TaskSpaceBase):

    def __init__(self, cfg: ParkourTaskSpaceCfg):
        super().__init__(cfg)

    def set_tasks(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], task: Tensor) -> None:

        assert task.shape == (
            len(env_ids),
            *self.cfg.dimensions,
        ), "task shape does not match dimensions of the task space"

        for manager, manager_term in self.cfg.tasks.items():
            if manager == "event":
                for term in manager_term.values():
                    for i, (subterm, params) in enumerate(term.items()):
                        event_cfg = env.event_manager.get_term_cfg(subterm)

                        # TODO: remove duplicate code between the root and joint branches
                        if self.cfg.task_types[subterm] == "root":
                            for j, param in enumerate(params.keys()):
                                assert (
                                    param in event_cfg.params
                                ), f"Event {subterm} does not have parameter {param} needed for the task space"

                                # the stack below spans every param key, so a stray key breaks later calls
                                assert list(event_cfg.params[param].keys()) == list(params[param]), (
                                    f"Event {subterm}'s '{param}' declares {list(event_cfg.params[param].keys())}"
                                    f" but the task space drives {list(params[param])}. They must match exactly;"
                                    " drop the extra keys from the event (omitted ones default to zero)."
                                )

                                if not isinstance(list(event_cfg.params[param].values())[0], Tensor):
                                    full_tasks = torch.zeros(
                                        env.num_envs, self.cfg.subterm_dims[subterm][0], device=task.device
                                    )
                                else:
                                    full_tasks = torch.stack(list(event_cfg.params[param].values()), dim=1)

                                # here we assign row-wise
                                full_tasks[env_ids, :] = task[:, self.cfg.subterm_ids[subterm], j]
                                event_cfg.params[param].update({k: v for k, v in zip(params[param], full_tasks.T)})

                                pass
                        elif self.cfg.task_types[subterm] == "joint":
                            # joint level
                            for j, param in enumerate(params):
                                assert (
                                    param in event_cfg.params
                                ), f"Event {subterm} does not have parameter {param} needed for the task space"

                                if not isinstance(event_cfg.params[param], Tensor):
                                    full_tasks = torch.zeros(
                                        env.num_envs, self.cfg.subterm_dims[subterm][0], device=task.device
                                    )
                                else:
                                    full_tasks = event_cfg.params[param]

                                # here we assign column-wise (for all joints together)
                                full_tasks[env_ids, :] = task[:, self.cfg.subterm_ids[subterm]][:, :, j]
                                event_cfg.params[param] = full_tasks
                                pass
                        else:
                            raise NotImplementedError(f"task type {self.cfg.task_types[subterm]} not implemented")

            else:
                raise NotImplementedError(f"manager {manager} not implemented")

    def get_tasks(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
    ) -> Tensor:
        raise NotImplementedError("get_tasks is not implemented for ParkourTaskSpace yet")
        pass

    def obs_to_task(
        self,
        obs: Tensor,
    ) -> Tensor:

        # base
        base_pos_xyz = obs[:, :, :3]
        base_orient_quat = obs[:, :, 3:7].squeeze(-2)
        base_vel = obs[:, :, 7:13]
        # convert quaternion to euler angles
        base_orient_euler_xyz = torch.stack(euler_xyz_from_quat(base_orient_quat), dim=-1).unsqueeze(1)
        base_pose = torch.cat([base_pos_xyz, base_orient_euler_xyz], dim=-1)
        base_pose_vel = torch.stack([base_pose, base_vel], dim=-1)

        # joints
        joint_pos = obs[:, :, 13:25]
        joint_vel = obs[:, :, 25:37]
        joint_pos_vel = torch.stack([joint_pos, joint_vel], dim=-1)

        task = torch.cat([base_pose_vel, joint_pos_vel], dim=-2)

        return task
