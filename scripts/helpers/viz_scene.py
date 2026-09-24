# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open an environment with no policy under zero action, for figures.

The viewport mode holds the scene for manual camera placement; ``--screenshot PATH`` (run via ``record.py`` with
``--enable_cameras``) settles, writes a PNG and exits. Use a ``-PLAY-`` binding, whose fixed goal the markers read.
The robot is pinned to the centre of its start distribution and timeouts are disabled so the scene holds still.
"""

from __future__ import annotations

import argparse
import sys

SCREENSHOT_RESOLUTION = (1600, 1600)
"""Default screenshot size. Square, because a figure of a maze is."""

FLOOR_LIGHT_HEIGHT = 0.1
"""How far above the floor the light rectangle sits, in metres."""

FLOOR_LIGHT_PRIM = "/World/light"
"""Prim path the floor light replaces: the one the scene's own light already occupies."""

VIEW_AXIS_TOL = 1e-6
"""How far the eye may sit off the lookat's vertical before the camera aims itself, in metres."""

MARKER_EDGE_FRACTION = 0.004
"""Marker wireframe radius as a fraction of the framed width, so lines keep a constant pixel thickness."""

DEFAULT_APERTURE = 20.955
"""Fallback horizontal aperture, for a camera that has not authored one."""

DEFAULT_FOCAL = 18.14756202697754
"""Fallback focal length, likewise."""

HORIZON_DEPTH_PERCENTILE = 99.0
"""Depth percentile past which a pixel counts as the far terrain the background is matched to."""

BACKGROUND_TOLERANCE = 1.0
"""How close, per channel on the 0-255 scale, the rendered background has to come to the terrain."""

BACKGROUND_MAX_ROUNDS = 8
"""Most background adjustments tried before the closest one is kept."""

BACKGROUND_SETTLE_RENDERS = 8
"""Renders after each background adjustment before the frame is measured."""

BACKGROUND_OVERRIDE_COLOR = 2
"""Value of ``/rtx/background/source/type`` that forces the background to a flat colour, over the dome light."""

parser = argparse.ArgumentParser(description="View an environment with no policy, under zero action.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=0, help="Seeds whatever the midpoint pinning leaves random.")
parser.add_argument(
    "--screenshot",
    type=str,
    default=None,
    help="Write one PNG here and exit, instead of holding the viewport open. Requires"
    " --enable_cameras; pair with --headless to render without a window.",
)
parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    default=None,
    metavar=("WIDTH", "HEIGHT"),
    help=f"Render resolution. A screenshot defaults to {SCREENSHOT_RESOLUTION[0]}x"
    f"{SCREENSHOT_RESOLUTION[1]}; the viewport defaults to the env cfg's.",
)
parser.add_argument(
    "--frame_margin",
    type=float,
    default=0.05,
    help="Slack left on each side of the task space when the camera zooms to it, as a fraction of"
    " the image. Zero puts the task space exactly at the image border.",
)
parser.add_argument(
    "--floor_light_intensity",
    type=float,
    nargs="+",
    default=[7500.0],
    help="Brightness of the floor light, as a radiance rather than a total power, so it does not"
    " fall off as the layout grows. Several values write one screenshot each, suffixed with the"
    " value, so a brightness can be chosen by eye from a single run.",
)
parser.add_argument(
    "--scene_light",
    action="store_true",
    help="Keep the scene's own lights, untouched, instead of the floor light. For terrain with relief,"
    " whose raised surfaces a light lying on the floor cannot reach. Ignores --floor_light_intensity.",
)
parser.add_argument(
    "--terrain_background",
    action="store_true",
    help="Render the sky in the colour of the farthest terrain in view, so the background matches the"
    " terrain and no horizon shows. Screenshot only.",
)
parser.add_argument(
    "--settle_steps", type=int, default=30, help="Zero-action steps taken before the screenshot."
)
parser.add_argument(
    "--warmup_renders",
    type=int,
    default=30,
    help="Renders discarded before the screenshot is kept. The annotator returns empty frames while"
    " the renderer warms up, and path tracing accumulates its samples over these.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    # `hydra_task_config` resolves the agent cfg too, even though nothing here trains.
    help="Name of the RL agent configuration entry point.",
)

from isaaclab.app import AppLauncher  # noqa: E402

import os
import sys

# cli_args.py lives with the entry points in scripts/rsl_rl/; put that directory on the path (one-way dependency).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "rsl_rl"))

import cli_args  # isort: skip  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.screenshot:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.managers import TerminationTermCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import informed_exploration.tasks  # noqa: F401, E402


def _disable_timeouts(terminations) -> list[str]:
    """Switch off every timeout term so the scene resets once and then holds; returns the disabled names."""
    disabled = []
    for name, term in list(vars(terminations).items()):
        if isinstance(term, TerminationTermCfg) and term.time_out:
            setattr(terminations, name, None)
            disabled.append(name)
    return disabled


def _pin_initial_state(events) -> int:
    """Collapse every ``(lo, hi)`` range in an event's nested params to its midpoint.

    Written back as a degenerate range, since the reset events build tensors from these pairs.
    """
    pinned = 0
    for term in vars(events).values():
        params = getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        for value in params.values():
            if not isinstance(value, dict):
                continue
            for key, rng in value.items():
                if isinstance(rng, (tuple, list)) and len(rng) == 2:
                    if not all(isinstance(bound, (int, float)) for bound in rng):
                        continue
                    mid = 0.5 * (rng[0] + rng[1])
                    value[key] = (mid, mid)
                    pinned += 1
    return pinned


def _queue_initial_state(env) -> bool:
    """Queue the centre of ``eval_bounds`` as every env's first start, as ``eval.py`` fills ``env.eval_data``.

    Returns whether anything was queued; not without an eval curriculum or ``eval_bounds``, or with goal-carrying terms.
    """
    from collections import deque

    from informed_exploration.tasks.manager_based.informed_exploration.curriculum import EvalCfg

    curriculum = getattr(env.cfg, "curriculum", None)
    terms = [t for t in vars(curriculum).values() if isinstance(t, EvalCfg)] if curriculum is not None else []
    task_space = getattr(env, "task_space", None)
    if not terms or getattr(getattr(task_space, "cfg", None), "eval_bounds", None) is None:
        return False
    if terms[0].goal_command_name is not None:
        print("[WARN] The eval curriculum carries goals, so no start was queued; use a single-goal -PLAY- binding.")
        return False

    centre = task_space.task_bounds(eval_bounds=True).float().mean(dim=1)
    env.eval_data["queue"] = deque(centre.clone() for _ in range(env.num_envs))
    env.eval_data["done"] = False
    env.eval_data["initialized"] = False
    env.eval_data["results"] = None
    env.eval_data["rewards"] = []
    env.eval_data["starts"] = []
    env.eval_data["finished"] = torch.zeros(env.num_envs, dtype=torch.bool)
    env.eval_data["old_tasks"] = torch.zeros(env.num_envs, *task_space.task_dim(), device=env.device)
    return True


def _terrain_tile_size(scene_cfg) -> tuple[float, float] | None:
    """The generated terrain's tile size, which the floor light and the marker frames scale to."""
    generator = getattr(scene_cfg.terrain, "terrain_generator", None)
    return getattr(generator, "size", None)


def _report_light(prim_path: str = FLOOR_LIGHT_PRIM) -> None:
    """Print what the light prim actually became on the stage."""
    import isaacsim.core.utils.stage as stage_utils

    prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] No prim at {prim_path}: the scene spawned no light there.")
        return
    inputs = {
        attr.GetName().removeprefix("inputs:"): attr.Get()
        for attr in prim.GetAttributes()
        if attr.GetName().startswith("inputs:")
    }
    translate = prim.GetAttribute("xformOp:translate")
    print(f"[INFO] {prim_path} is a {prim.GetTypeName()} at {translate.Get() if translate else None}")
    print(f"[INFO]   {inputs}")


def _set_light_intensity(intensity: float, prim_path: str = FLOOR_LIGHT_PRIM) -> None:
    """Retune the spawned light in place, so one run can sweep brightnesses."""
    import isaacsim.core.utils.stage as stage_utils

    prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
    attr = prim.GetAttribute("inputs:intensity") if prim.IsValid() else None
    if attr:
        attr.Set(intensity)


def _use_floor_light(scene_cfg, intensity: float) -> None:
    """Swap the scene's light for a floor rect light sized to the terrain tile; without a tile the light is kept."""
    from informed_exploration.tasks.manager_based.informed_exploration.utils.lights import floor_light_cfg

    size = _terrain_tile_size(scene_cfg)
    if not size:
        print("[INFO] No terrain tile size to span; keeping the scene's own light.")
        return
    scene_cfg.light = floor_light_cfg(
        size=size, height=FLOOR_LIGHT_HEIGHT, intensity=intensity, prim_path=FLOOR_LIGHT_PRIM
    )
    print(f"[INFO] Floor light: {size[0]:.1f}x{size[1]:.1f} m at z={FLOOR_LIGHT_HEIGHT}, facing down.")


def _task_space_frame(env, margin: float):
    """Return ``(width, z)`` in metres framing the task-space marker box, or ``None`` without a task space."""
    import env_marker_geometry as geom

    ref_origin, _ = geom.resolve_ref_env(env)
    box = geom.bounds_box(env, ref_origin)
    if box is None:
        return None
    extent = 2.0 * max(float(box.half_sizes[0]), float(box.half_sizes[1]))
    return extent / max(1.0 - 2.0 * margin, 1e-3), float(box.center[2])


def _zoom_to_frame(env, width: float, subject_z: float) -> None:
    """Frame exactly ``width`` metres at the subject's height, by lens if possible.

    The lens is written on the session layer and trusted only if it composes; otherwise the camera is moved.
    """
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Usd

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(env.cfg.viewer.cam_prim_path)
    if not prim.IsValid():
        return

    aperture_attr = prim.GetAttribute("horizontalAperture")
    aperture = aperture_attr.Get() if aperture_attr else None
    aperture = float(aperture) if aperture else DEFAULT_APERTURE

    eye = [float(v) for v in env.cfg.viewer.eye]
    distance = abs(eye[2] - subject_z)
    focal_attr = prim.GetAttribute("focalLength")
    wanted = distance * aperture / width

    res_x, res_y = env.cfg.viewer.resolution
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        if focal_attr:
            focal_attr.Set(wanted)
        vertical = prim.GetAttribute("verticalAperture")
        if vertical:
            vertical.Set(aperture * res_y / res_x)

    focal = float(focal_attr.Get()) if focal_attr and focal_attr.Get() else DEFAULT_FOCAL
    if abs(focal - wanted) < 1e-3:
        print(f"[INFO] Framed {width:.2f} m from {distance:.2f} m up: focal length {focal:.2f}.")
        return

    # the lens would not take, so move the camera to where this lens frames the same width
    from isaacsim.core.utils.viewports import set_camera_view

    height = subject_z + (width / 2.0) / math.tan(math.atan(aperture / (2.0 * focal)))
    set_camera_view([eye[0], eye[1], height], list(env.cfg.viewer.lookat), env.cfg.viewer.cam_prim_path)
    print(
        f"[INFO] Lens stayed at {focal:.2f}; framed {width:.2f} m by moving the camera to"
        f" {height:.2f} m instead of {eye[2]:.2f} m."
    )


def _write_xform_op(op, values) -> None:
    """Set an existing xform op, keeping whatever Gf type it already holds."""
    from pxr import Gf, UsdGeom

    current = op.Get()
    if current is not None:
        op.Set(type(current)(*values))
        return
    precision = op.GetPrecision()
    vec = Gf.Vec3f if precision == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d
    op.Set(vec(*values))


def _aim_camera_if_vertical(env) -> bool:
    """Aim the viewer camera straight down (or up) at its lookat, which ``set_camera_view`` fails to do.

    Existing xform ops are written in place, since re-adding them raises. Returns whether the view is vertical.
    """
    import isaacsim.core.utils.stage as stage_utils
    from pxr import Gf, Usd, UsdGeom

    eye, lookat = env.cfg.viewer.eye, env.cfg.viewer.lookat
    if abs(eye[0] - lookat[0]) > VIEW_AXIS_TOL or abs(eye[1] - lookat[1]) > VIEW_AXIS_TOL:
        return False

    prim = stage_utils.get_current_stage().GetPrimAtPath(env.cfg.viewer.cam_prim_path)
    xform = UsdGeom.Xformable(prim)
    # a USD camera looks along its own -z; looking up is a half turn about x
    looking_down = lookat[2] <= eye[2]
    half_turn = 0.0 if looking_down else 180.0

    eulers = {
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }
    rotated = False
    for op in xform.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            _write_xform_op(op, [float(v) for v in eye])
        elif op_type in eulers:
            # only one axis turns, so every euler order gives the same rotation
            _write_xform_op(op, [half_turn, 0.0, 0.0])
            rotated = True
        elif op_type == UsdGeom.XformOp.TypeOrient:
            current = op.Get()
            quat = type(current) if current is not None else Gf.Quatd
            op.Set(quat(1.0, 0.0, 0.0, 0.0) if looking_down else quat(0.0, 1.0, 0.0, 0.0))
            rotated = True
    if not rotated:
        # nothing on the camera carries a rotation, so give it one
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(1.0, 0.0, 0.0, 0.0) if looking_down else Gf.Quatd(0.0, 1.0, 0.0, 0.0)
        )

    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    view = tuple(round(-matrix[2][k], 6) for k in range(3))
    up = tuple(round(matrix[1][k], 6) for k in range(3))
    print(f"[INFO] Aimed the camera down its vertical: view {view}, up {up}.")
    wanted = (0.0, 0.0, -1.0) if looking_down else (0.0, 0.0, 1.0)
    if any(abs(a - b) > 1e-3 for a, b in zip(view, wanted)):
        print(
            f"[WARN] The camera resolves to {view}, not {wanted}: a stronger layer is overriding"
            " the aim, so the view is not the straight-down one it is configured to be."
        )
    return True


def _report_camera(env) -> None:
    """Print what the viewer camera actually holds, the counterpart to :func:`_report_light`."""
    import isaacsim.core.utils.stage as stage_utils

    prim_path = env.cfg.viewer.cam_prim_path
    prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] No prim at {prim_path}.")
        return
    names = [
        "projection",
        "horizontalAperture",
        "verticalAperture",
        "focalLength",
        "clippingRange",
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:orient",
    ]
    print(f"[INFO] {prim_path} holds:")
    for name in names:
        attr = prim.GetAttribute(name)
        if attr:
            print(f"[INFO]   {name} = {attr.Get()}")


def _intensity_path(path: str, intensity: float, sweeping: bool) -> str:
    """Where one intensity's frame goes: the given path, or a suffixed sibling when sweeping."""
    if not sweeping:
        return path
    stem, ext = os.path.splitext(path)
    value = int(intensity) if float(intensity).is_integer() else intensity
    return f"{stem}_i{value}{ext}"


def _attach_depth(env):
    """Attach a depth annotator to the render product of ``env.render()``; call after the first render."""
    import omni.replicator.core as rep

    annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane", device="cpu")
    annotator.attach([env.unwrapped._render_product])
    return annotator


def _read_depth(annotator, shape: tuple[int, int]):
    """The annotator's latest depth frame as an ``(H, W)`` array, or ``None`` while it is still empty."""
    import numpy as np

    data = annotator.get_data()
    if isinstance(data, dict):
        data = data["data"]
    depth = np.asarray(data, dtype=np.float32)
    if depth.size != shape[0] * shape[1]:
        return None
    return depth.reshape(shape)


def _srgb_to_linear(srgb):
    """0-255 sRGB to linear 0-1."""
    import numpy as np

    c = np.asarray(srgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _sky_and_far_terrain(rgb, depth):
    """Return the median colour of the sky (infinite depth) and of the farthest terrain, ``None`` if not in view."""
    import numpy as np

    sky = ~np.isfinite(depth)
    sky_colour = np.median(rgb[sky], axis=0) if sky.any() else None
    if sky.all():
        return sky_colour, None
    hit_depth = depth[~sky]
    far = hit_depth >= np.percentile(hit_depth, HORIZON_DEPTH_PERCENTILE)
    return sky_colour, np.median(rgb[~sky][far], axis=0)


def _render_with_depth(env, depth_annotator, renders: int):
    """Render ``renders`` frames and return the last one's colour and depth."""
    rgb = None
    for _ in range(renders):
        rgb = env.render()
    return rgb, _read_depth(depth_annotator, rgb.shape[:2])


def _match_background_to_terrain(env, depth_annotator) -> None:
    """Set the renderer's flat background colour to whatever the farthest terrain renders as.

    Tone mapping makes the result unpredictable, so it is matched iteratively with the markers hidden.
    """
    import carb
    import numpy as np

    markers = getattr(env.unwrapped, "_env_markers", None)
    if markers is not None:
        markers.set_visibility(False)
    try:
        rgb, depth = _render_with_depth(env, depth_annotator, BACKGROUND_SETTLE_RENDERS)
        if depth is None:
            print("[WARN] The depth annotator is still empty; the background is left as it is.")
            return
        sky, terrain = _sky_and_far_terrain(rgb, depth)
        if sky is None or terrain is None:
            print("[INFO] No sky in view; the background is left as it is.")
            return

        settings = carb.settings.get_settings()
        settings.set_int("/rtx/background/source/type", BACKGROUND_OVERRIDE_COLOR)
        colour = _srgb_to_linear(terrain)
        for round_index in range(BACKGROUND_MAX_ROUNDS):
            settings.set_float_array("/rtx/background/source/color", [float(c) for c in colour])
            rgb, depth = _render_with_depth(env, depth_annotator, BACKGROUND_SETTLE_RENDERS)
            sky, terrain = _sky_and_far_terrain(rgb, depth)
            error = np.abs(sky - terrain)
            if np.all(error <= BACKGROUND_TOLERANCE):
                break
            colour = colour * _srgb_to_linear(terrain) / np.maximum(_srgb_to_linear(sky), 1e-4)
        print(
            f"[INFO] Background matched to the far terrain after {round_index + 1} adjustments:"
            f" sky {np.round(sky).astype(int).tolist()}, terrain {np.round(terrain).astype(int).tolist()}."
        )
        if np.any(error > BACKGROUND_TOLERANCE):
            print("[WARN] The background did not come within tolerance of the terrain; the closest match is kept.")
    finally:
        if markers is not None:
            markers.set_visibility(True)


def _save_screenshot(env, zero_actions, path: str, settle_steps: int, warmup_renders: int,
                     intensities: list[float] | None, terrain_background: bool = False) -> None:
    """Settle the scene, then write one frame per light intensity, or one as spawned when ``intensities`` is None."""
    from PIL import Image

    for _ in range(settle_steps):
        env.step(zero_actions)

    _report_light()
    _report_camera(env.unwrapped)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    sweeping = intensities is not None and len(intensities) > 1
    depth_annotator = None

    for intensity in intensities or [None]:
        if intensity is not None:
            _set_light_intensity(intensity)
        rgb = None
        for _ in range(warmup_renders):
            rgb = env.render()
            if terrain_background and depth_annotator is None:
                depth_annotator = _attach_depth(env)
        if depth_annotator is not None:
            _match_background_to_terrain(env, depth_annotator)
            # the markers just came back, so let the renderer settle on them again
            for _ in range(warmup_renders):
                rgb = env.render()
        if rgb is None or rgb.size == 0 or not rgb.any():
            raise RuntimeError(
                "The renderer returned an empty frame. Check that --enable_cameras is set, and"
                " raise --warmup_renders if it is."
            )
        out = _intensity_path(path, intensity, sweeping)
        Image.fromarray(rgb).save(out)
        lit = "scene lights" if intensity is None else f"intensity {intensity:g}"
        print(f"[INFO] Wrote {os.path.abspath(out)} ({rgb.shape[1]}x{rgb.shape[0]}), {lit}.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand = None
    env_cfg.observations.policy.enable_corruption = False

    resolution = args_cli.resolution
    if resolution is None and args_cli.screenshot:
        resolution = SCREENSHOT_RESOLUTION
    if resolution is not None:
        env_cfg.viewer.resolution = tuple(resolution)

    if args_cli.scene_light:
        print("[INFO] Keeping the scene's own lights.")
    else:
        _use_floor_light(env_cfg.scene, args_cli.floor_light_intensity[0])

    disabled = _disable_timeouts(env_cfg.terminations)
    if disabled:
        print(f"[INFO] Timeouts disabled, the scene will not reset: {', '.join(disabled)}")
    print(f"[INFO] Pinned {_pin_initial_state(env_cfg.events)} reset ranges to their midpoint.")

    print(f"[INFO] Creating environment: {args_cli.task}")
    # `log_dir` is a required positional on the env classes; nothing is written there
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        log_dir="logs/viz_scene",
        render_mode="rgb_array" if args_cli.screenshot else None,
    )
    core_env = env.unwrapped

    try:
        import render_utils

        if args_cli.path_tracing:
            render_utils.enable_path_tracing()
        render_utils.hide_walls_if_requested(env, args_cli)

        frame = _task_space_frame(core_env, args_cli.frame_margin)

        if args_cli.env_markers:
            import isaaclab_markers

            # store the handle on the env so it is kept alive (and re-assertable via refresh())
            core_env._env_markers = isaaclab_markers.draw_env_markers(
                core_env,
                edge_radius=None if frame is None else frame[0] * MARKER_EDGE_FRACTION,
            )

        if _queue_initial_state(core_env):
            print("[INFO] Queued the centre of eval_bounds as the start.")
        else:
            print("[INFO] No eval start queued; spawning from the reset events' midpoints.")
        env.reset()
        # only a straight-down view has a single subject distance to frame against
        if _aim_camera_if_vertical(core_env) and frame is not None:
            _zoom_to_frame(core_env, *frame)

        # hold the pose the scene was reset to; a zero raw action is the default pose on joint-position robots
        get_reset_state = getattr(core_env, "get_action_reset_state", None)
        hold_actions = (
            get_reset_state().clone()
            if get_reset_state is not None
            else torch.zeros((core_env.num_envs, core_env.action_space.shape[1]), device=core_env.device)
        )

        if args_cli.screenshot:
            with torch.inference_mode():
                _save_screenshot(
                    env,
                    hold_actions,
                    args_cli.screenshot,
                    args_cli.settle_steps,
                    args_cli.warmup_renders,
                    None if args_cli.scene_light else args_cli.floor_light_intensity,
                    terrain_background=args_cli.terrain_background,
                )
            return

        _report_light()
        _report_camera(core_env)
        print("[INFO] Holding the reset pose. Move the camera in the viewport; close the window to quit.")
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(hold_actions)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
