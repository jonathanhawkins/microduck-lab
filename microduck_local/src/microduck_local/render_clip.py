"""Render a saved 🎬 clip as forward-kinematics playback — mp4 + contact sheet.

    uv run render-clip g1-front-kick --out /tmp/rc
    uv run render-clip backflip --camera side --fps 25

This is the AUTHORED motion, not a policy: every frame is the clip resampled
at 50 Hz (motion.load_clip, exactly what the imitation reward tracks), posed
on the robot's scratch model and grounded on its lowest sole (pose.PoseScratch,
exactly what the editor shows). It exists so a clip can be looked at and
shared before a policy is trained on it, and so a trained policy's rollout
(render-rollout) can be put beside the reference it was paid to follow.

The sheet's captions carry the balance read per tile — where the centre of
mass sits against the grounded soles — because a clip that cannot be stood
in statically is a clip the policy will have to cheat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import motion
from .pose import pose_scratch
from .render_rollout import CAMERAS, build_sheet, sheet_indices


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip", help="clip name (clips/<name>.json)")
    ap.add_argument("--out", default="/tmp/render-clip")
    ap.add_argument("--camera", default="three-quarter", choices=sorted(CAMERAS))
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--loops", type=int, default=1,
                    help="how many times to play a looping clip (default 1)")
    ap.add_argument("--sheet-frames", type=int, default=12)
    return ap.parse_args(argv)


def clip_frames(clip: motion.Clip, loops: int = 1) -> list[int]:
    n = clip.steps * (max(loops, 1) if clip.loop else 1)
    return list(range(n))


def main(argv=None) -> None:
    import imageio.v2 as imageio
    import mujoco

    args = parse_args(argv)
    clip = motion.load_clip(args.clip)
    scratch = pose_scratch(clip.robot)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    width, height = args.width - args.width % 2, args.height - args.height % 2

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = CAMERAS[args.camera]
    cam.distance = max(0.7, 2.6 * scratch.stand_height)
    cam.lookat[:] = (0.0, 0.0, scratch.stand_height * 0.55)
    renderer = mujoco.Renderer(scratch.model, height=height, width=width)

    stride = max(1, round(motion.CONTROL_HZ / max(args.fps, 1.0)))
    real_fps = motion.CONTROL_HZ / stride
    frames: list[np.ndarray] = []
    captions: list[list[str]] = []
    for i in clip_frames(clip, args.loops):
        if i % stride:
            continue
        q, pitch = clip.at(i)
        scratch.solve(q, pitch, ground=True)
        bal = scratch.balance()
        renderer.update_scene(scratch.data, camera=cam)
        frames.append(renderer.render().copy())
        feet = scratch.effector_positions()
        lifted = [f"{s} foot {feet[f'{s}_foot'][2]:.2f} m up"
                  for s in ("left", "right")
                  if not bal["feet"][s]["grounded"]]
        captions.append([
            f"t={i / motion.CONTROL_HZ:5.2f}s  frame {i}/{clip.steps}",
            f"CoM {('over the ' + bal['over'] + ' sole') if bal['over'] else 'off both soles'}"
            f"  stance {bal['support']['marginMm']:+.0f} mm",
            "  ".join(lifted) or "both feet down",
            f"pelvis {scratch.data.xpos[scratch.base_body][2]:.3f} m",
        ])
    mp4 = out / f"{clip.name}.mp4"
    with imageio.get_writer(mp4, fps=real_fps, macro_block_size=None) as w:
        for f in frames:
            w.append_data(f)
    idx = sheet_indices(len(frames), args.sheet_frames)
    header = [
        f"clip {clip.name}  robot={clip.robot}  {clip.steps} frames "
        f"({clip.duration:.2f} s, {'loop' if clip.loop else 'one-shot'})  camera={args.camera}",
        "forward kinematics of the AUTHORED clip, grounded on the lowest sole — "
        "not a policy",
    ]
    sheet = out / f"{clip.name}_sheet.png"
    build_sheet([frames[i] for i in idx], [captions[i] for i in idx], header,
                ["balance: CoM vs the soles' footprints, static — how hard the "
                 "pose is to hold, not whether it stands"],
                [False] * len(idx), sheet)
    print(f"wrote {mp4} ({len(frames)} frames @ {real_fps:.1f} fps) and {sheet}")


if __name__ == "__main__":
    main()
