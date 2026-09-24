# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-instance goal wiring, applied from an arm's ``__post_init__``.

Each helper re-points the goal command, its observation terms and the curriculum anchor at the terrain in the cfg.
"""

from ... import mdp
from ...curriculum.task_space.bridge_task_space_cfg import bridge_anchors_from_goal_poses
from ...curriculum.task_space.parkour_task_space_cfg import height_offset
from ...mdp.commands import CurriculumGoalCfg
from .goal_env_cfg import only_sub_terrain


def apply_bver_goal_conditioning(cfg) -> None:
    """Set the curriculum anchor at the goal platform plus the curriculum-controlled 3D goal command and observation."""
    goal_poses = only_sub_terrain(cfg.scene.terrain.terrain_generator).goal_poses
    anchors = bridge_anchors_from_goal_poses(goal_poses)
    assert len(anchors) == 1, f"BVER supports a single anchor goal, but the terrain has {len(anchors)} goal platforms"
    cfg.curriculum.initialization.anchor_goal = anchors[0]
    # Per-terrain: whether the task-space AABB contains non-terrain depends on the layout
    cfg.curriculum.initialization.filtered_voronoi_samples = cfg.filtered_voronoi_samples

    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
    )

    # 3D goal observation, since goals need not be coplanar with the start platform
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d
def apply_eval_goal_conditioning(cfg) -> None:
    """Set a fixed default goal at the first goal platform plus the 3D goal observation for play and eval.

    No curriculum writes goals in eval, so multi-goal terrains exercise only the first goal.
    """
    goal_poses = only_sub_terrain(cfg.scene.terrain.terrain_generator).goal_poses
    anchors = bridge_anchors_from_goal_poses(goal_poses)
    anchor = anchors[0]

    cfg.commands.base_position = CurriculumGoalCfg(
        asset_name="robot",
        resampling_time_range=cfg.commands.base_position.resampling_time_range,
        debug_vis=True,
        default_goal=[anchor[0][0], anchor[1][0], anchor[2][0]],
    )
    cfg.observations.policy.pos_command.func = mdp.pos_command_3d
    cfg.observations.critic.pos_command.func = mdp.pos_command_3d


def apply_baseline_spawn(cfg) -> None:
    """Put the static spawn on top of the start platform at nominal standing height.

    The inherited ``pose.z`` would spawn the base inside the raised pedestal; the curriculum-free baseline reads it.
    """
    geometry_bounds = only_sub_terrain(cfg.scene.terrain.terrain_generator).geometry_bounds
    cfg.events.reset_base.params["pose"]["z"] = geometry_bounds["start_z"] + height_offset
