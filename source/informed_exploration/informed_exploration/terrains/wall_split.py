# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hide a terrain's walls in the viewport without changing what the robot stands on or senses.

The single terrain mesh keeps its collider and is made invisible; collider-free render copies split into surface
and wall carry the visuals. A face is wall when all three vertices sit on a wall box's corners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporter

__all__ = ["RENDER_ROOT", "split_perimeter_wall"]

RENDER_ROOT = "render"
"""Child of the terrain prim path holding the render-only copies, authored after the collider mesh."""

_CORNER_EPS = 1e-4
"""Tolerance (m) for a vertex to count as on a box corner, loose enough for the cached ``.obj`` round trip."""


def _tile_wall_meshes(sub_cfg) -> list:
    """Rebuild one tile's wall boxes in tile coordinates, for mesh-scan and platform-bridge sub-terrains.

    Uses the same calls the terrain was built with; the layout's repeated geometry warnings are silenced.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _wall_meshes(sub_cfg)


def _wall_meshes(sub_cfg) -> list:
    """Body of :func:`_tile_wall_meshes`, split out so its warnings stay scoped."""
    from .bridge import geometry
    from .bridge.composer import perimeter_wall_meshes_for_rect
    from .bridge.composer_cfg import resolve_layout

    wall = getattr(sub_cfg, "perimeter_wall", None)
    rect = getattr(sub_cfg, "footprint_rect_tile", None)
    if rect is not None:
        return perimeter_wall_meshes_for_rect(wall, rect, sub_cfg.size) if wall is not None else []

    if getattr(sub_cfg, "layout", None) is None:
        return []
    layout_res = resolve_layout(sub_cfg.layout, sub_cfg.bridge, sub_cfg.size, sub_cfg.cross_links)
    origin = layout_res.root.pose_center
    meshes = [
        geometry.wall_box((origin[0] + w.center[0], origin[1] + w.center[1]), w.size, w.height)
        for w in sub_cfg.walls
    ]
    if wall is not None:
        xs = [c for node in layout_res.nodes for c in (node.center[0] - node.platform.size[0] / 2.0,
                                                       node.center[0] + node.platform.size[0] / 2.0)]
        ys = [c for node in layout_res.nodes for c in (node.center[1] - node.platform.size[1] / 2.0,
                                                       node.center[1] + node.platform.size[1] / 2.0)]
        meshes += perimeter_wall_meshes_for_rect(wall, (min(xs), max(xs), min(ys), max(ys)), sub_cfg.size)
    return meshes


def _tile_wall_bounds(sub_cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return AABBs of one tile's wall boxes, in tile coordinates."""
    return [(mesh.bounds[0], mesh.bounds[1]) for mesh in _tile_wall_meshes(sub_cfg)]


def _tile_offsets(gen_cfg) -> list[np.ndarray]:
    """Return the tile-to-world translation for every tile, composing the generator's three moves.

    Checked against the importer's origins by :func:`_assert_offsets` before use.
    """
    size_x, size_y = gen_cfg.size
    return [
        np.array([
            row * size_x - gen_cfg.num_rows * size_x / 2.0,
            col * size_y - gen_cfg.num_cols * size_y / 2.0,
            0.0,
        ])
        for row in range(gen_cfg.num_rows)
        for col in range(gen_cfg.num_cols)
    ]


def _wall_face_mask(points: np.ndarray, faces: np.ndarray, boxes) -> tuple[np.ndarray, list[int]]:
    """Return the union mask of faces whose vertices are all corners of a wall box, and per-box match counts."""
    corners = points[faces]  # (num_faces, 3, 3)
    mask = np.zeros(len(faces), dtype=bool)
    counts = []
    for lo, hi in boxes:
        on_corner = (np.abs(corners - lo) <= _CORNER_EPS) | (np.abs(corners - hi) <= _CORNER_EPS)
        found = on_corner.all(axis=2).all(axis=1)
        counts.append(int(found.sum()))
        mask |= found
    return mask, counts


def _copy_render_mesh(prim_path: str, source, faces: np.ndarray, material_path: str | None):
    """Create a collider-free Mesh prim over ``source``'s points and a subset of its faces."""
    import isaaclab.sim as sim_utils

    prim = sim_utils.create_prim(
        prim_path,
        "Mesh",
        attributes={
            "points": source.GetAttribute("points").Get(),
            "faceVertexIndices": faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(faces)),
            "subdivisionScheme": source.GetAttribute("subdivisionScheme").Get() or "bilinear",
        },
    )
    # per-vertex colour primvars carry over whole since the points are shared
    for name in ("primvars:displayColor", "primvars:displayOpacity"):
        attr = source.GetAttribute(name)
        if attr and attr.HasAuthoredValue():
            prim.GetAttribute(name).Set(attr.Get())
    if material_path is not None:
        sim_utils.bind_visual_material(prim_path, material_path)
    return prim


def split_perimeter_wall(terrain: TerrainImporter | None, wall_visible: bool = False) -> int:
    """Draw the terrain from render-only copies so its walls can be shown or hidden on their own.

    Covers a mesh-scan's fence and a bridge layout's fence and route separators. Idempotent.

    Args:
        terrain: The scene's terrain importer.
        wall_visible: Whether the wall copy is rendered.

    Returns:
        Number of wall triangles separated out; zero leaves the stage untouched.
    """
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdShade

    if terrain is None:
        print("[WARN] --hide_walls: the scene has no terrain importer.")
        return 0

    prim_path = terrain.cfg.prim_path
    render_root = f"{prim_path}/{RENDER_ROOT}"
    wall_path = f"{render_root}/wall"
    stage = sim_utils.get_current_stage()

    # already split: only the visibility is left to set
    wall_prim = stage.GetPrimAtPath(wall_path)
    if wall_prim.IsValid():
        _set_visible(wall_prim, wall_visible)
        return len(wall_prim.GetAttribute("faceVertexCounts").Get())

    gen_cfg = terrain.cfg.terrain_generator
    if terrain.cfg.terrain_type != "generator" or gen_cfg is None:
        print(f"[WARN] --hide_walls: terrain type '{terrain.cfg.terrain_type}' has no fence to split off.")
        return 0
    if len(gen_cfg.sub_terrains) != 1:
        print(
            "[WARN] --hide_walls: the terrain grid mixes"
            f" {len(gen_cfg.sub_terrains)} sub-terrains, so which fence sits in which tile is not"
            " knowable from the config. Skipped."
        )
        return 0

    sub_cfg = next(iter(gen_cfg.sub_terrains.values()))
    tile_boxes = _tile_wall_bounds(sub_cfg)
    if not tile_boxes:
        print(
            f"[WARN] --hide_walls: sub-terrain '{type(sub_cfg).__name__}' declares no walls, or is"
            " not one of the families whose walls can be located (see `_tile_wall_meshes`)."
        )
        return 0

    offsets = _tile_offsets(gen_cfg)
    _assert_offsets(terrain, sub_cfg, offsets)
    boxes = [(lo + offset, hi + offset) for offset in offsets for lo, hi in tile_boxes]

    terrain_path = terrain.terrain_prim_paths[0]
    source = sim_utils.get_first_matching_child_prim(terrain_path, lambda prim: prim.GetTypeName() == "Mesh")
    if source is None:
        raise RuntimeError(f"no mesh prim under {terrain_path!r} to split.")
    counts = np.asarray(source.GetAttribute("faceVertexCounts").Get())
    assert (counts == 3).all(), "expected a triangulated terrain mesh"
    points = np.asarray(source.GetAttribute("points").Get(), dtype=np.float64)
    faces = np.asarray(source.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)

    is_wall, counts = _wall_face_mask(points, faces, boxes)
    missed = [n for n in counts if n != 12]  # a box is 12 triangles
    if missed:
        raise RuntimeError(
            f"{len(missed)} of {len(boxes)} wall boxes did not match 12 triangles each under"
            f" {terrain_path!r} (matched {sorted(set(missed))}). The terrain on the stage and the"
            " config it is read from have diverged; leaving the mesh alone rather than hiding the"
            " wrong geometry."
        )

    # inherit the material so the copies shade exactly like the mesh they replace
    material_path = str(UsdShade.MaterialBindingAPI(source).GetDirectBinding().GetMaterialPath()) or None

    sim_utils.create_prim(render_root, "Xform")
    _copy_render_mesh(f"{render_root}/surface", source, faces[~is_wall], material_path)
    wall_prim = _copy_render_mesh(wall_path, source, faces[is_wall], material_path)

    # the original keeps its collider and its physics material and simply stops being drawn
    UsdGeom.Imageable(stage.GetPrimAtPath(terrain_path)).MakeInvisible()
    _set_visible(wall_prim, wall_visible)
    return int(is_wall.sum())


def _set_visible(prim, visible: bool) -> None:
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _assert_offsets(terrain: TerrainImporter, sub_cfg, offsets: list[np.ndarray]) -> None:
    """Check the derived tile placement against the importer's terrain origins minus the start pose."""
    origins = getattr(terrain, "terrain_origins", None)
    poses = getattr(sub_cfg, "goal_poses", None)
    if origins is None or not poses:
        return
    start = np.asarray(poses["start"][:2])
    measured = origins.reshape(-1, 3).cpu().numpy()[:, :2] - start
    assert np.allclose(measured, np.stack(offsets)[:, :2], atol=1e-4), (
        f"tile placement disagrees with the terrain origins ({measured} vs {np.stack(offsets)[:, :2]});"
        " isaaclab's terrain generator no longer places tiles the way this module assumes."
    )
