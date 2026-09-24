"""Task space for BVER bridge-terrain envs, with bounds and anchor goals derived from the terrain's own metadata."""

from __future__ import annotations

import copy
import math

from isaaclab.utils import configclass

from .parkour_task_space_cfg import ParkourTaskSpaceCfg, height_offset, jpl, jvl

# Attitude and joint bounds shared by every terrain; deep-copied so per-terrain bounds never alias it.
_ORIENTATION_AND_JOINT_BOUNDS: list[list[float]] = [
    [-math.pi / 6, math.pi / 6],  # roll
    [-math.pi / 18, math.pi / 18],  # v roll
    [-math.pi / 4, math.pi / 4],  # pitch
    [-math.pi / 18, math.pi / 18],  # v pitch
    [-math.pi / 2, math.pi / 2],  # yaw
    [-math.pi / 18, math.pi / 18],  # v yaw
    # 12 joints (position, velocity), same limits for every joint
    *([[-jpl, jpl], [-jvl, jvl]] * 12),
]

_DEFAULT_TASK_TEMPLATE: list[float] = [
    0.0,
    0.0,  # x, vx
    0.0,
    0.0,  # y, vy
    height_offset,
    0.0,  # z, vz; see ParkourTaskSpaceCfg.default_task for why z is non-zero
    0.0,
    0.0,  # roll, v roll
    0.0,
    0.0,  # pitch, v pitch
    0.0,
    0.0,  # yaw, v yaw
    *([0.0, 0.0] * 12),  # 12 joints (offset, velocity)
]


@configclass
class BridgeTaskSpaceCfg(ParkourTaskSpaceCfg):
    """Bridge task space whose geometry-dependent fields are set by its ``from_*`` constructors.

    Uses walkable keep-in footprints instead of keep-out sample filters.
    """

    sample_filter: list | None = None
    goal_sample_filter: list | None = None
    optimal_trajectory_file: str | None = None

    # Declared here since `ParkourTaskSpaceCfg` does not subclass `TaskSpaceBaseCfg`.
    keep_in_filter: list | None = None
    """Walkable xy footprints written by :meth:`from_geometry_bounds`; see ``TaskSpaceBaseCfg.keep_in_filter``."""

    keep_in_dimensions: tuple | None = (0, 2)
    """Flat x and y indices, since bridge keep-in regions are xy footprints."""

    @classmethod
    def from_goal_poses(
        cls,
        goal_poses: dict[str, tuple[float, float, float]],
        platform_half_extent: float = 0.75,
        bridge_width: float = 2.0,
        margin: float = 0.3,
        z_margin: float = 0.3,
    ) -> "BridgeTaskSpaceCfg":
        """Derive bounds and eval_bounds from a terrain's tile-local ``goal_poses``.

        Args:
            goal_poses: Env-origin-relative start and goal platform positions.
            platform_half_extent: Half the platform footprint; pads the x/y bounds.
            bridge_width: Bridge span width; sets the y bounds if wider than the platform.
            margin: Extra padding added to every spatial bound.
            z_margin: Headroom above the higher platform's standing height; widen for tall mid-traverse excursions.
        """
        # the start platform is the terrain origin, so only the goal's offset matters
        start = goal_poses["start"]
        goal = goal_poses["goal"]
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]

        half_y = max(platform_half_extent, bridge_width / 2.0)
        # span both platforms' y-extent, for layouts with an off-axis goal
        y_lo = min(0.0, dy) - half_y
        y_hi = max(0.0, dy) + half_y
        z_lo = min(start[2], goal[2]) + height_offset - margin
        z_hi = max(start[2], goal[2]) + height_offset + z_margin

        bounds: list[list[float]] = [
            [-platform_half_extent - margin, dx + platform_half_extent + margin],  # x
            [-1.0, 1.0],  # vx
            [y_lo, y_hi],  # y
            [-0.5, 0.5],  # vy
            [z_lo, z_hi],  # z
            [0.0, 1.0],  # vz
        ]
        bounds += copy.deepcopy(_ORIENTATION_AND_JOINT_BOUNDS)

        # eval_bounds: a small region on the start platform at standing height, everything else zero
        eval_bounds = [[0.0, 0.0] for _ in range(len(bounds))]
        eval_bounds[0] = [-0.2, 0.2]
        eval_bounds[2] = [-0.2, 0.2]
        eval_bounds[4] = [start[2] + height_offset, start[2] + height_offset]

        return cls(bounds=bounds, eval_bounds=eval_bounds, default_task=copy.deepcopy(_DEFAULT_TASK_TEMPLATE))

    @classmethod
    def from_geometry_bounds(
        cls,
        geometry_bounds: dict[str, float],
        margin: float = -0.3,
        z_margin: float = 0.3,
        keep_in_regions: list[tuple[float, float, float, float]] | None = None,
    ) -> "BridgeTaskSpaceCfg":
        """Derive bounds and eval_bounds from a terrain's ``geometry_bounds``, the AABB of every platform footprint.

        Works for any number of goals, unlike :meth:`from_goal_poses`.

        Args:
            geometry_bounds: Platform-top bounding box, in the tile-local frame with the start platform at the origin.
            margin: Extra padding added to every spatial bound.
            z_margin: Headroom above the highest platform's top; widen for tall mid-traverse excursions.
            keep_in_regions: Walkable xy footprints stored as ``keep_in_filter``; ``None`` disables the filter.
        """
        x_lo = geometry_bounds["x_lo"] - margin
        x_hi = geometry_bounds["x_hi"] + margin
        y_lo = geometry_bounds["y_lo"] - margin
        y_hi = geometry_bounds["y_hi"] + margin
        z_lo = geometry_bounds["z_lo"] + height_offset
        z_hi = geometry_bounds["z_hi"] + height_offset + z_margin

        bounds: list[list[float]] = [
            [x_lo, x_hi],  # x
            [-1.0, 1.0],  # vx
            [y_lo, y_hi],  # y
            [-0.5, 0.5],  # vy
            [z_lo, z_hi],  # z
            [0.0, 1.0],  # vz
        ]
        bounds += copy.deepcopy(_ORIENTATION_AND_JOINT_BOUNDS)

        # eval_bounds: a window spanning every start platform at standing height, everything else zero
        eval_bounds = [[0.0, 0.0] for _ in range(len(bounds))]
        eval_bounds[0] = [geometry_bounds.get("start_x_lo", 0.0) - 0.5, geometry_bounds.get("start_x_hi", 0.0) + 0.5]
        eval_bounds[2] = [geometry_bounds.get("start_y_lo", 0.0) - 0.5, geometry_bounds.get("start_y_hi", 0.0) + 0.5]
        start_z = geometry_bounds["start_z"]
        eval_bounds[4] = [start_z + height_offset, start_z + height_offset]

        keep_in_filter = None
        if keep_in_regions is not None:
            # flat rect tuples to per-axis pairs, matching keep_in_dimensions
            keep_in_filter = [((x_lo, x_hi), (y_lo, y_hi)) for x_lo, x_hi, y_lo, y_hi in keep_in_regions]

        return cls(
            bounds=bounds,
            eval_bounds=eval_bounds,
            default_task=copy.deepcopy(_DEFAULT_TASK_TEMPLATE),
            keep_in_filter=keep_in_filter,
        )


def _bridge_anchor_state(
    start: tuple[float, float, float], goal: tuple[float, float, float], extra_z: float = 0.0
) -> list[list[float]]:
    """Build an anchor-goal state for standing on ``goal`` in the default posture, in the layout of
    ``_RC_GOAL_ON_BOX``."""
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    z = goal[2] + height_offset + extra_z

    anchor: list[list[float]] = [
        [dx, 0.0],  # x
        [dy, 0.0],  # y
        [z, 0.0],  # z
        [0.0, 0.0],  # roll
        [0.0, 0.0],  # pitch
        [0.0, 0.0],  # yaw
    ]
    anchor += [[0.0, 0.0] for _ in range(12)]  # 12 joints (offset, velocity)
    return anchor


def bridge_anchor_from_goal_poses(
    goal_poses: dict[str, tuple[float, float, float]], extra_z: float = 0.0
) -> list[list[float]]:
    """Return the anchor state for a single-goal terrain; see :func:`bridge_anchors_from_goal_poses`."""
    return _bridge_anchor_state(goal_poses["start"], goal_poses["goal"], extra_z)


def bridge_anchors_from_goal_poses(
    goal_poses: dict[str, tuple[float, float, float]], extra_z: float = 0.0
) -> list[list[list[float]]]:
    """Return one anchor-goal state per goal platform, in ``goal_poses`` iteration order (the flat-patch numbering)."""
    start = goal_poses["start"]
    return [_bridge_anchor_state(start, pose, extra_z) for name, pose in goal_poses.items() if name != "start"]
