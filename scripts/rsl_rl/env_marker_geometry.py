"""Backend-agnostic geometry for static environment markers.

Shared by ``rerun_markers`` and ``isaaclab_markers`` so both stay in sync. Centres are world-frame
``(3,)`` float32 arrays and colours are RGBA in the ``0-255`` range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Draw the start zone from the task space's ``eval_bounds`` instead of the terrain's ``init_pos`` patch config.
USE_EVAL_BOUNDS_FOR_START_ZONE = True


@dataclass
class BoxMarker:
    """Axis-aligned box, centred at ``center`` with the given half extents."""

    name: str
    center: np.ndarray  # (3,) world frame
    half_sizes: np.ndarray  # (3,)
    color: tuple[int, int, int, int]  # RGBA, 0-255


@dataclass
class SphereMarker:
    """Sphere centred at ``center`` with the given radius."""

    name: str
    center: np.ndarray  # (3,) world frame
    radius: float
    color: tuple[int, int, int, int]  # RGBA, 0-255


@dataclass
class PointCloudMarker:
    """Many identical small spheres: one prototype, N instances.

    Backends key a prototype off the spec name, so this avoids one prototype per sphere in large scatters.
    """

    name: str
    centers: np.ndarray  # (N, 3) world frame
    radius: float
    color: tuple[int, int, int, int]  # RGBA, 0-255


def compute_env_markers(env) -> list[BoxMarker | SphereMarker]:
    """Return all static markers for the visualization reference subterrain.

    Markers whose source config is unavailable are omitted.

    Args:
        env: Unwrapped ``ManagerBasedRLEnv`` (call ``env.unwrapped`` first).
    """
    ref_origin, ref_env_idx = resolve_ref_env(env)

    candidates = [
        start_zone(env, ref_origin),
        goal_zone(env, ref_env_idx),
        terrain_box(env, ref_origin),
    ]
    # Corridor evals replace the task-space box with their goal volumes or their drawn start points.
    corridors = corridor_boxes(env, ref_origin)
    starts = corridor_start_points(env, ref_origin)
    candidates += corridors or ([starts] if starts is not None else [bounds_box(env, ref_origin)])
    return [m for m in candidates if m is not None]


# Reference-env resolution


def resolve_ref_env(env) -> tuple[np.ndarray, int]:
    """Return ``(ref_origin, ref_env_idx)`` for the visualization reference subterrain.

    The env index is the one whose origin is nearest the curriculum's ``global_viz_subterrain`` origin.
    """
    from informed_exploration.tasks.manager_based.informed_exploration.utils.curriculum_utils import (
        resolve_viz_origin,
    )

    global_viz_subterrain = None
    try:
        for attr in vars(env.cfg.curriculum).values():
            gvs = getattr(attr, "global_viz_subterrain", None)
            if gvs is not None:
                global_viz_subterrain = gvs
                break
    except Exception:
        pass

    ref_tensor = resolve_viz_origin(env, global_viz_subterrain)

    all_origins = env.scene.env_origins.detach().cpu().numpy().astype(np.float32)

    if ref_tensor is not None:
        ref_origin = ref_tensor.detach().cpu().numpy().astype(np.float32)
        diffs = np.linalg.norm(all_origins - ref_origin[np.newaxis], axis=1)
        ref_env_idx = int(np.argmin(diffs))
    else:
        ref_env_idx = 0
        ref_origin = all_origins[0]

    return ref_origin, ref_env_idx


# Individual markers


def start_zone(env, ref_origin: np.ndarray) -> BoxMarker | None:
    """Blue flat slab at the start-zone centre (reference env).

    Source is chosen by ``USE_EVAL_BOUNDS_FOR_START_ZONE``.
    """
    if USE_EVAL_BOUNDS_FOR_START_ZONE:
        marker = _start_zone_from_eval_bounds(env, ref_origin)
        if marker is not None:
            return marker
    return _start_zone_from_init_pos(env, ref_origin)


def _start_zone_from_eval_bounds(env, ref_origin: np.ndarray) -> BoxMarker | None:
    """Start zone from the x/y/z entries of the task space's ``eval_bounds``.

    The z bound is already a base height, so no base-height offset is added.
    """
    try:
        from informed_exploration.tasks.manager_based.informed_exploration.utils.curriculum_utils import (
            resolve_xyz_dims,
        )

        task_space = getattr(env, "task_space", None)
        eval_bounds = getattr(getattr(task_space, "cfg", None), "eval_bounds", None)
        if eval_bounds is None:
            return None

        xi, yi, zi = resolve_xyz_dims(task_space)
        x_range, y_range, z_range = eval_bounds[xi], eval_bounds[yi], eval_bounds[zi]
        local = np.array(
            [
                (x_range[0] + x_range[1]) / 2,
                (y_range[0] + y_range[1]) / 2,
                (z_range[0] + z_range[1]) / 2,
            ],
            dtype=np.float32,
        )
        own_floor = _task_space_marker_floor(task_space)
        floor = MIN_MARKER_HALF_SIZE if own_floor is None else own_floor
        half_sizes = np.array(
            [
                max((-x_range[0] + x_range[1]) / 2, floor),
                max((-y_range[0] + y_range[1]) / 2, floor),
                # a flat slab unless the task space sizes its own markers
                0.05 if own_floor is None else max((z_range[1] - z_range[0]) / 2, own_floor),
            ],
            dtype=np.float32,
        )
        return BoxMarker("start", ref_origin + local, half_sizes, (0, 142, 255, 155))
    except Exception:
        return None


def _start_zone_from_init_pos(env, ref_origin: np.ndarray) -> BoxMarker | None:
    """Start zone from the terrain subterrain's ``init_pos`` flat-patch config."""
    try:
        sub_terrains = env.cfg.scene.terrain.terrain_generator.sub_terrains
        for tcfg in sub_terrains.values():
            fp = getattr(tcfg, "flat_patch_sampling", None) or {}
            ip = fp.get("init_pos")
            if ip is None:
                continue
            local = np.array(
                [
                    (ip.x_range[0] + ip.x_range[1]) / 2,
                    (ip.y_range[0] + ip.y_range[1]) / 2,
                    (ip.z_range[0] + ip.z_range[1]) / 2 + 0.6,  # offset anymal base height
                ],
                dtype=np.float32,
            )
            half_sizes = np.array(
                [
                    (-ip.x_range[0] + ip.x_range[1]) / 2,
                    (-ip.y_range[0] + ip.y_range[1]) / 2,
                    0.05,
                ],
                dtype=np.float32,
            )
            return BoxMarker("start", ref_origin + local, half_sizes, (0, 142, 255, 155))
    except Exception:
        pass
    return None


def goal_zone(env, ref_env_idx: int) -> SphereMarker | None:
    """Yellow-green sphere at the goal position of the reference env.

    Uses ``valid_targets`` for terrain-sampled commands (``pos_command_w`` is still zero at ``gym.make``),
    else the ``default_goal`` or curriculum anchor for a ``CurriculumGoalCommand``.
    """
    goal = _goal_from_valid_targets(env, ref_env_idx)
    if goal is None:
        goal = _goal_from_curriculum_anchor(env, ref_env_idx)
    if goal is None:
        return None
    radius = _goal_radius(env)
    # yellow-green, well clear of the start zone's blue, which the viewport renders as cyan
    return SphereMarker("goal", goal, radius, (0, 153, 76, 30))


def _goal_radius(env) -> float:
    """The goal sphere's radius: the command's success threshold when it has one."""
    try:
        threshold = env.command_manager.get_term("base_position").cfg.distance_threshold
        if threshold is not None and threshold > 0:
            return float(threshold)
    except Exception:
        pass
    return 0.25


def _goal_from_valid_targets(env, ref_env_idx: int) -> np.ndarray | None:
    """Goal position from a terrain-sampled command term's flat patches."""
    try:
        goal_term = env.command_manager.get_term("base_position")
        terrain = env.scene["terrain"]
        level = terrain.terrain_levels[ref_env_idx]
        ttype = terrain.terrain_types[ref_env_idx]
        # valid_targets: (terrain_level, terrain_type, num_patches, 3)
        goal = goal_term.valid_targets[level, ttype, 0].clone()
        # match the command's root-height offset so the sphere sits at base height
        goal[2] += env.scene["robot"].data.default_root_state[ref_env_idx, 2]
        return goal.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def _goal_from_curriculum_anchor(env, ref_env_idx: int) -> np.ndarray | None:
    """Goal position for a curriculum-written command term, in world coordinates.

    Prefers the command's ``default_goal`` (eval and play bindings), else the xyz of the curriculum's
    ``anchor_goal``. Both are env-relative, so the reference origin is added.
    """
    try:
        origin = env.scene.env_origins[ref_env_idx].detach().cpu().numpy().astype(np.float32)

        default_goal = getattr(env.command_manager.get_term("base_position").cfg, "default_goal", None)
        if default_goal is not None:
            return origin + np.asarray(default_goal, dtype=np.float32)

        from informed_exploration.tasks.manager_based.informed_exploration.utils.curriculum_utils import (
            resolve_anchor_goal_xyz,
        )

        anchor = resolve_anchor_goal_xyz(env)
        if anchor is None:
            return None
        return origin + anchor.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def terrain_box(env, ref_origin: np.ndarray) -> BoxMarker | None:
    """Dark semi-transparent cuboid matching the BigBox terrain obstacle (reference env)."""
    try:
        from informed_exploration.tasks.manager_based.informed_exploration.terrains.climb_box import (
                BigBoxTerrainCfg,
            )

        sub_terrains = env.cfg.scene.terrain.terrain_generator.sub_terrains
        for tcfg in sub_terrains.values():
            if not isinstance(tcfg, BigBoxTerrainCfg):
                continue
            bh = (tcfg.height_range[0] + tcfg.height_range[1]) / 2
            bw = (tcfg.width_range[0] + tcfg.width_range[1]) / 2
            bl = (tcfg.length_range[0] + tcfg.length_range[1]) / 2
            center = ref_origin + np.array([0.0, 0.0, bh / 2], dtype=np.float32)
            half_sizes = np.array([bl / 2, bw / 2, bh / 2], dtype=np.float32)
            return BoxMarker("box", center, half_sizes, (50, 40, 26, 255))
    except Exception:
        pass
    return None


CORRIDOR_COLOR = (235, 104, 52, 40)
"""Colour shared by all corridor markers; routes are told apart by position."""

MIN_CORRIDOR_HALF_HEIGHT = 0.01
"""Floor on a corridor box's half-height, so a zero-thickness region still renders."""

MIN_MARKER_HALF_SIZE = 0.05
"""Floor on any marker box half-extent, so a flat or point-like region still renders.

A task space with smaller regions sets its own floor through ``marker_min_half_size``."""


def _task_space_marker_floor(task_space) -> float | None:
    """The floor a task space sets on its markers' half-extents, or ``None`` when it keeps the default."""
    return getattr(getattr(task_space, "cfg", None), "marker_min_half_size", None)


START_POINT_RADIUS = 0.04
"""Radius (m) of a scattered corridor-start marker."""


def corridor_start_points(env, ref_origin: np.ndarray) -> PointCloudMarker | None:
    """The base positions of the pooled starts a corridor-START run will evaluate, or ``None``.

    Reads the env-relative ``env._corridor_start_points`` published by ``eval_corridor_starts.py``.
    """
    points = getattr(env, "_corridor_start_points", None)
    if points is None or len(points) == 0:
        return None
    centers = np.asarray(points, dtype=np.float32) + ref_origin
    return PointCloudMarker("corridor_starts", centers, START_POINT_RADIUS, CORRIDOR_COLOR)


def corridor_boxes(env, ref_origin: np.ndarray) -> list[BoxMarker]:
    """Boxes covering the corridor goal-sampling regions, or ``[]`` when this isn't a corridor eval.

    Reads the env-relative ``env._corridor_regions`` published by ``eval_multi_goal.py --corridor_goals``.
    """
    regions = getattr(env, "_corridor_regions", None)
    if not regions:
        return []

    boxes = []
    for name, (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) in regions.items():
        local = np.array([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2, (z_lo + z_hi) / 2], dtype=np.float32)
        half_sizes = np.array(
            [
                (x_hi - x_lo) / 2,
                (y_hi - y_lo) / 2,
                # a zero-thickness corridor (level pole tops) still needs a visible slab
                max((z_hi - z_lo) / 2, MIN_CORRIDOR_HALF_HEIGHT),
            ],
            dtype=np.float32,
        )
        boxes.append(BoxMarker(f"corridor_{name}", ref_origin + local, half_sizes, CORRIDOR_COLOR))
    return boxes


def bounds_box(env, ref_origin: np.ndarray) -> BoxMarker | None:
    """Very transparent grey cuboid spanning the x/y/z components of the task space ``bounds``.

    The z bound is already a base height, so no offset is added.
    """
    try:
        from informed_exploration.tasks.manager_based.informed_exploration.utils.curriculum_utils import (
            resolve_xyz_dims,
        )

        task_space = getattr(env, "task_space", None)
        bounds = getattr(getattr(task_space, "cfg", None), "bounds", None)
        if bounds is None:
            return None

        xi, yi, zi = resolve_xyz_dims(task_space)
        x_range, y_range, z_range = bounds[xi], bounds[yi], bounds[zi]
        local = np.array(
            [
                (x_range[0] + x_range[1]) / 2,
                (y_range[0] + y_range[1]) / 2,
                (z_range[0] + z_range[1]) / 2,
            ],
            dtype=np.float32,
        )
        own_floor = _task_space_marker_floor(task_space)
        floor = MIN_MARKER_HALF_SIZE if own_floor is None else own_floor
        half_sizes = np.array(
            [
                max((-x_range[0] + x_range[1]) / 2, floor),
                max((-y_range[0] + y_range[1]) / 2, floor),
                max((-z_range[0] + z_range[1]) / 2, floor),
            ],
            dtype=np.float32,
        )
        return BoxMarker("bounds", ref_origin + local, half_sizes, (128, 128, 128, 30))
    except Exception:
        return None
