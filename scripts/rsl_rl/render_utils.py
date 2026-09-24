"""Helpers for configuring the RTX renderer (e.g. path tracing) from the play/eval/train scripts."""

from __future__ import annotations


def enable_path_tracing(total_spp: int = 64) -> None:
    """Switch the RTX renderer to path tracing; call after ``AppLauncher``.

    Args:
        total_spp: Samples per pixel accumulated for each rendered frame.
    """
    try:
        import carb

        settings = carb.settings.get_settings()
        settings.set_string("/rtx/rendermode", "PathTracing")
        settings.set_int("/rtx/pathtracing/totalSpp", total_spp)
        settings.set_int("/rtx/pathtracing/spp", 1)
        settings.set_int("/rtx/pathtracing/clampSpp", total_spp)
        # Denoise the low-sample result.
        settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)
        print(f"[INFO] Path tracing enabled (totalSpp={total_spp}).")
    except Exception as exc:  # pragma: no cover - depends on the live Omniverse app
        print(f"[WARN] Could not enable path tracing: {exc}")


def hide_walls_if_requested(env, args_cli) -> None:
    """Honour ``--hide_walls`` by hiding the arena fence; its collider and height scan are kept."""
    if not getattr(args_cli, "hide_walls", False):
        return
    from informed_exploration.terrains.wall_split import split_perimeter_wall

    num_faces = split_perimeter_wall(env.unwrapped.scene.terrain, wall_visible=False)
    print(f"[INFO] Hid the arena fence ({num_faces} triangles).")
