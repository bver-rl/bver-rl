"""One-shot rerun logging of static environment markers (start zone, goal zone, terrain box).

Geometry comes from :mod:`env_marker_geometry`, shared with ``isaaclab_markers``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import env_marker_geometry as geom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Rerun entity paths keyed by marker name.
_ENTITY_PATHS = {
    "start": "exploration/markers/start",
    "goal": "exploration/markers/goal",
    "box": "exploration/markers/box",
    "bounds": "exploration/markers/bounds",
}


def log_env_markers(env: "ManagerBasedRLEnv") -> None:
    """Log static start-zone, goal-zone and terrain-box markers into rerun.

    Args:
        env: Unwrapped ``ManagerBasedRLEnv`` (call ``env.unwrapped`` before passing).
    """
    from informed_exploration.tasks.manager_based.informed_exploration.utils import rerun_logger

    if not rerun_logger.enabled():
        return

    for marker in geom.compute_env_markers(env):
        entity = _ENTITY_PATHS.get(marker.name, f"exploration/markers/{marker.name}")
        if isinstance(marker, geom.SphereMarker):
            rerun_logger.log_ellipsoids(
                marker.center[None],
                half_sizes=[marker.radius, marker.radius, marker.radius],
                colors=list(marker.color),
                entity=entity,
                static=True,
            )
        else:  # BoxMarker
            rerun_logger.log_boxes(
                marker.center[None],
                half_sizes=list(marker.half_sizes),
                colors=list(marker.color),
                entity=entity,
                static=True,
            )


def log_env_terrain(env: "ManagerBasedRLEnv") -> None:
    """Regenerate the sim's terrain mesh from its cfg and log it as a static rerun underlay.

    The importer does not keep its generator, so the mesh is rebuilt; it shares the ``env_origins`` frame.
    Best-effort no-op unless rerun is on and the terrain is generator-based.

    Args:
        env: Unwrapped ``ManagerBasedRLEnv`` (call ``env.unwrapped`` before passing).
    """
    from informed_exploration.tasks.manager_based.informed_exploration.utils import rerun_logger

    if not rerun_logger.enabled():
        return

    terrain_cfg = getattr(env.cfg.scene, "terrain", None)
    if terrain_cfg is None or getattr(terrain_cfg, "terrain_type", None) != "generator":
        return
    gen_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if gen_cfg is None:
        return

    try:
        generator = gen_cfg.class_type(cfg=gen_cfg, device="cpu")
        rerun_logger.log_meshes(
            generator.terrain_mesh,
            entity="exploration/terrain/mesh",
            static=True,
            color=(40, 40, 40),  # flat dark grey underlay
            flat_shading=True,  # faceted normals; avoids smeared shading on hard edges
        )
    except Exception as exc:  # regeneration is best-effort; never break training
        print(f"[WARN] rerun terrain underlay skipped: {exc}")
