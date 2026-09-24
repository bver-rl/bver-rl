# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular, parametric bridge-terrain library.

Platforms are connected by swappable "bridges" whose geometry sets the difficulty. See :mod:`.bridge`
for the composer and :mod:`.config` for presets. Depends only on Isaac Lab core.
"""

from .bridge import *  # noqa: F401, F403
from .config import *  # noqa: F401, F403
