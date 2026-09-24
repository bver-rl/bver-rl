# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach ``SCAN_SPECS`` from a bare Python process, for the sim-free scripts in this directory.

Loads the real ``terrains/config/scans.py`` with its Omniverse imports stubbed, since the package needs ``pxr``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

PKG = pathlib.Path(__file__).resolve().parents[3] / "informed-exploration/source/informed_exploration/informed_exploration"
SCANS_PY = PKG / "terrains/config/scans.py"

_STUBBED_FROM_SCAN = ("ScanHeightfieldTerrainCfg", "content_fingerprint", "load_scan_grid", "pinned_scan_cfg", "scan_path")
"""What ``scans.py`` imports from ``..scan``; a new name there fails loudly on import."""


def load_scan_specs() -> dict:
    """Return ``{snake: ScanSpec}`` from the real ``scans.py`` with its Omniverse imports stubbed.

    Terrain-building methods (``sub_terrain``, ``generator_cfg``) do not work here, since they return stubs.
    """
    if "_scanspecs.scans" in sys.modules:
        return sys.modules["_scanspecs.scans"].SCAN_SPECS

    stub_scan = types.ModuleType("_scanspecs.scan")
    for name in _STUBBED_FROM_SCAN:
        setattr(stub_scan, name, object)
    stub_terrains = types.ModuleType("isaaclab.terrains")
    stub_terrains.TerrainGeneratorCfg = object
    stub_isaaclab = types.ModuleType("isaaclab")
    stub_isaaclab.__path__ = []
    sys.modules.setdefault("isaaclab", stub_isaaclab)
    sys.modules.setdefault("isaaclab.terrains", stub_terrains)

    package = types.ModuleType("_scanspecs")
    package.__path__ = []
    sys.modules["_scanspecs"] = package
    sys.modules["_scanspecs.scan"] = stub_scan
    # `from ..scan import ...` resolves against the parent of __package__, hence the nesting.
    config = types.ModuleType("_scanspecs.config")
    config.__path__ = []
    sys.modules["_scanspecs.config"] = config

    spec = importlib.util.spec_from_file_location("_scanspecs.scans", SCANS_PY)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "_scanspecs.config"
    sys.modules["_scanspecs.scans"] = module
    spec.loader.exec_module(module)
    return module.SCAN_SPECS


def resolve_scan(snake: str):
    """One spec by its snake_case name, with the known names listed if it is not one of them."""
    specs = load_scan_specs()
    if snake not in specs:
        raise SystemExit(f"unknown scan {snake!r}; known: {', '.join(sorted(specs))}")
    return specs[snake]


def scan_for_stem(stem: str):
    """Return the spec whose raw scan has this file stem, or ``None``."""
    for spec in load_scan_specs().values():
        if spec.stem == stem:
            return spec
    return None
