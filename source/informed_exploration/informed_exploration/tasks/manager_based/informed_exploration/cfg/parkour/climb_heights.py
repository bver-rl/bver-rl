# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every climb arm at every box height.

Pairs each arm with each height in :data:`~...cfg.common.box_height.CLIMB_BOX_HEIGHTS`, so the gym registry can
point at ``climb_heights:<Arm>_<token>``.
"""

from ..common.box_height import CLIMB_BOX_HEIGHTS, build_height_twins  # noqa: F401
from . import climb_single_box_cfg as _base
from . import climb_single_box_cfg_bver_sparse as _bver
from . import climb_single_box_cfg_sparse_baselines as _controls
from . import climb_single_box_down_cfg_bver_sparse as _down

_BVER_ARMS = (
    "BVER_SPARSE", "BVER_SPARSE_FWD", "BVER_SPARSE_BWD", "BVER_SPARSE_PEN",
    "BVER_SPARSE_PLAY", "BVER_SPARSE_MGPLAY", "BVER_SPARSE_EVAL", "BVER_SPARSE_MGEVAL",
    "BVER_SPARSE_ROBUSTEVAL", "BVER_SPARSE_ROBUSTPLAY",
    "BVER_SPARSE_PLAY_SCALE05", "BVER_SPARSE_MGPLAY_SCALE05", "BVER_SPARSE_EVAL_SCALE05",
    "BVER_SPARSE_MGEVAL_SCALE05", "BVER_SPARSE_ROBUSTEVAL_SCALE05", "BVER_SPARSE_ROBUSTPLAY_SCALE05",
)
_CONTROL_ARMS = (
    "SPARSE_Baseline", "SPARSE_Baseline_PEN", "SPARSE_RSI", "SPARSE_Random", "SPARSE_Backplay",
    "SPARSE_RC",
)
_GENERATOR_ARMS = ("GenFeasibleStarts",)
_DOWN_ARMS = (
    "BVER_SPARSE", "BVER_SPARSE_FWD", "BVER_SPARSE_BWD",
    "BVER_SPARSE_PLAY", "BVER_SPARSE_MGPLAY", "BVER_SPARSE_EVAL", "BVER_SPARSE_MGEVAL",
)

_UP = "IEAnymalDClimbSingleBoxEnvCfg_"
_DN = "IEAnymalDClimbSingleBoxDownEnvCfg_"

_ARMS: dict[str, type] = {}
for _mod, _prefix, _names in (
    (_bver, _UP, _BVER_ARMS),
    (_controls, _UP, _CONTROL_ARMS),
    (_base, _UP, _GENERATOR_ARMS),
    (_down, _DN, _DOWN_ARMS),
):
    for _n in _names:
        _ARMS[f"{_prefix}{_n}"] = getattr(_mod, f"{_prefix}{_n}")

build_height_twins(_ARMS, globals())

__all__ = [f"{name}_{token}" for name in _ARMS for token in CLIMB_BOX_HEIGHTS]
