# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry of the Franka Tower-of-Hanoi task: base plate, pegs and the disk (called ring) the fingers hold.

Positions are env-relative, with the robot base at the origin and the table top at zero height. The hand frame is
``panda_hand``, whose +z is the approach axis and whose fingers slide along y. Free of Isaac Sim imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))

PANDA_FINGER_JOINT_EXPR = "panda_finger_joint.*"

PANDA_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
"""Published Franka Panda position limits in radians, used until a solved-poses file supplies the asset's own."""

PANDA_TCP_Z = 0.1034
"""Tool centre point along the hand's approach axis, as in Isaac Lab's Franka manipulation tasks."""

SIDEWAYS_QUAT_WXYZ = (0.5, -0.5, 0.5, -0.5)
"""Hand orientation with the approach axis along world +x and the fingers sliding vertically; maps the hand's x, y, z
axes to world -y, -z, +x."""

NUM_PEGS = 3


def quat_mul(a, b) -> tuple[float, float, float, float]:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conjugate(q) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q, v) -> tuple[float, float, float]:
    """``v`` rotated by the unit ``(w, x, y, z)`` quaternion ``q``."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


@dataclass(frozen=True)
class HanoiGeometry:
    """Everything a Hanoi env cfg needs about a layout, in env-relative metres.

    Pegs are indexed left to right as seen by the robot facing +x: peg 0 at +y, peg 1 on the centre line and peg 2
    at -y.
    """

    name: str
    start_peg: int
    goal_peg: int
    peg_height: float
    peg_x: float = 0.6
    peg_spacing: float = 0.15
    peg_radius: float = 0.01
    plate_size: tuple[float, float, float] = (0.10, 0.40, 0.12)
    """Plate footprint and thickness; the thickness is also the plate's top height."""
    ring_size: float = 0.20
    """Side length of the square disk, wider than the plate so the fingers can pinch it where it overhangs."""
    ring_hole: float = 0.025
    """Side length of the disk's square hole."""
    ring_thickness: float = 0.02
    grip_yaw: float = math.pi / 4
    """Yaw of the approach axis about world z, toward +y.

    A straight-ahead grip puts start and goal on mirrored IK branches; the yawed grip keeps them on one branch.
    """
    grip_distance: float = 0.11
    """Distance from the TCP to the disk centre along the approach axis."""
    tcp_z: float = PANDA_TCP_Z
    bottom_clearance: float = 0.005
    """Gap between the plate top and the disk's lower face when the disk is seated."""
    home_height: float = 0.05
    """Gap between the peg tops and the disk's lower face in the home pose."""
    workspace_margin: tuple[float, float, float] = (0.15, 0.15, 0.2)
    """How far the task space reaches past the pegs in x and y, and above the peg tops in z."""

    @property
    def plate_top_z(self) -> float:
        return self.plate_size[2]

    @property
    def plate_center(self) -> tuple[float, float, float]:
        return (self.peg_x, 0.0, self.plate_size[2] / 2.0)

    def peg_y(self, index: int) -> float:
        return (1 - index) * self.peg_spacing

    def peg_center(self, index: int) -> tuple[float, float, float]:
        """Centre of peg ``index``, which is where a cylinder spawner places it."""
        return (self.peg_x, self.peg_y(index), self.plate_top_z + self.peg_height / 2.0)

    @property
    def peg_top_z(self) -> float:
        return self.plate_top_z + self.peg_height

    @property
    def side_clearance(self) -> float:
        """Gap between a centred peg and the sides of the disk's hole."""
        return self.ring_hole / 2.0 - self.peg_radius

    @property
    def ring_bottom_z(self) -> float:
        """Height of the disk centre when the disk is seated at the bottom of a peg."""
        return self.plate_top_z + self.bottom_clearance + self.ring_thickness / 2.0

    @property
    def ring_lift_z(self) -> float:
        """Height above which the disk centre has cleared the peg tops."""
        return self.peg_top_z + self.ring_thickness / 2.0

    def ring_xyz_at_peg_bottom(self, index: int) -> tuple[float, float, float]:
        return (self.peg_x, self.peg_y(index), self.ring_bottom_z)

    @property
    def start_ring_xyz(self) -> tuple[float, float, float]:
        return self.ring_xyz_at_peg_bottom(self.start_peg)

    @property
    def goal_ring_xyz(self) -> tuple[float, float, float]:
        return self.ring_xyz_at_peg_bottom(self.goal_peg)

    @property
    def home_ring_xyz(self) -> tuple[float, float, float]:
        return (self.peg_x, self.peg_y(1), self.peg_top_z + self.home_height + self.ring_thickness / 2.0)

    @property
    def finger_joint_pos(self) -> float:
        """Opening of each finger that puts its pad on a face of the disk."""
        return self.ring_thickness / 2.0

    @property
    def hand_quat_wxyz(self) -> tuple[float, float, float, float]:
        """Hand orientation in the env frame at which the held disk is flat and square to the env axes."""
        half = self.grip_yaw / 2.0
        return quat_mul((math.cos(half), 0.0, 0.0, math.sin(half)), SIDEWAYS_QUAT_WXYZ)

    @property
    def ring_center_hand(self) -> tuple[float, float, float]:
        return (0.0, 0.0, self.tcp_z + self.grip_distance)

    @property
    def ring_rot_hand(self) -> tuple[float, float, float, float]:
        """Orientation of the disk in the hand frame: the env frame as seen from a hand at :attr:`hand_quat_wxyz`."""
        return quat_conjugate(self.hand_quat_wxyz)

    def ring_bar_specs(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        """Return ``(name, centre in the hand frame, size along the disk's axes)`` of the four bars forming the disk.

        The bars do not overlap, and each is oriented by :attr:`ring_rot_hand`.
        """
        width = (self.ring_size - self.ring_hole) / 2.0
        offset = (self.ring_hole + width) / 2.0
        thickness = self.ring_thickness
        local = [
            ("ring_bar_px", (offset, 0.0, 0.0), (width, self.ring_size, thickness)),
            ("ring_bar_nx", (-offset, 0.0, 0.0), (width, self.ring_size, thickness)),
            ("ring_bar_py", (0.0, offset, 0.0), (self.ring_hole, width, thickness)),
            ("ring_bar_ny", (0.0, -offset, 0.0), (self.ring_hole, width, thickness)),
        ]
        rot = self.ring_rot_hand
        cx, cy, cz = self.ring_center_hand
        specs = []
        for name, bar_offset, size in local:
            dx, dy, dz = quat_rotate(rot, bar_offset)
            specs.append((name, (cx + dx, cy + dy, cz + dz), size))
        return specs

    def hand_pos_for_ring(self, ring_xyz) -> tuple[float, float, float]:
        """Hand position that puts the disk centre at ``ring_xyz`` with the hand at :attr:`hand_quat_wxyz`."""
        ox, oy, oz = quat_rotate(self.hand_quat_wxyz, self.ring_center_hand)
        x, y, z = ring_xyz
        return (x - ox, y - oy, z - oz)

    @property
    def workspace_bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Box the disk centre is confined to in the task space, as ``(lo, hi)`` per x, y and z."""
        margin_x, margin_y, margin_z = self.workspace_margin
        reach_y = self.peg_spacing * (NUM_PEGS - 1) / 2.0 + margin_y
        return (
            (self.peg_x - margin_x, self.peg_x + margin_x),
            (-reach_y, reach_y),
            (self.plate_top_z, self.peg_top_z + margin_z),
        )

    @property
    def ring_keep_out(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """Disk-centre regions no state may occupy, as ``((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))`` boxes.

        One box keeps the disk above the plate and one keeps it above the table.
        """
        half = self.ring_size / 2.0
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, _) = self.workspace_bounds
        floor = z_lo - 1.0
        plate = (
            (self.peg_x - self.plate_size[0] / 2.0 - half, self.peg_x + self.plate_size[0] / 2.0 + half),
            (-self.plate_size[1] / 2.0 - half, self.plate_size[1] / 2.0 + half),
            (floor, self.plate_top_z + self.ring_thickness / 2.0),
        )
        table = ((x_lo - 1.0, x_hi + 1.0), (y_lo - 1.0, y_hi + 1.0), (floor, self.ring_thickness / 2.0))
        return [plate, table]


HANOI_LAYOUTS = {
    "p12": {"start_peg": 0, "goal_peg": 2, "peg_height": 0.12},
}
"""Layouts by name: which peg the ring starts on, which it must reach, and how tall the pegs are."""


def hanoi_geometry(name: str) -> HanoiGeometry:
    if name not in HANOI_LAYOUTS:
        raise ValueError(f"Hanoi layout '{name}' not recognized. Expected one of {sorted(HANOI_LAYOUTS)}.")
    return HanoiGeometry(name=name, **HANOI_LAYOUTS[name])
