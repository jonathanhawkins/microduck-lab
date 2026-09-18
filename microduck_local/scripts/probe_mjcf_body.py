"""LOOK at a level-0 body: one offscreen frame at its spawn keyframe, to PNG.

    uv run python scripts/probe_mjcf_body.py menagerie:unitree_go2 --out /tmp/p

`AGENTS.md`'s first verification rule is that reward curves and eval sums have
repeatedly lied here, and the answer is always to render it and look. The same
rule applies to a body that arrived from a stranger's MJCF, and more sharply:
every number `MjcfBody` reads is self-consistent by construction — a keyframe
that spawns the robot folded into the floor, a mesh that loaded as a hull, a
scene whose light is inside the chassis all produce a perfectly valid stage
pitch and a perfectly valid joint table. A picture is the only thing that
says the body is the robot its README shows.

Deliberately NOT a `render-rollout`: there is no policy at level 0 and nothing
to roll out. One frame, at the keyframe the body says it spawns from, plus the
numbers burned into the caption line so the PNG and the report cannot drift
apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


def frame(body, width: int = 900, height: int = 640,
          azimuth: float = 135.0, elevation: float = -20.0) -> np.ndarray:
    """One RGB frame of `body` at its spawn keyframe.

    The camera is framed off the body's OWN measured width — 2.1x it, which
    is `lab_spacing_m * 0.6` since the pitch is 3.52x the width — so a 0.33 m
    arm and a 1.3 m humanoid both fill the frame without a per-robot number.
    MEASURED on the Go2: the pitch itself (3.05 m out on a 0.75 m robot) put
    the body in about a seventh of the frame, which is a picture of a
    checkerboard.

    The scene goes through `MjSpec` rather than `MjModel.from_xml_path` for
    one reason, MEASURED: MuJoCo's offscreen framebuffer is 640x480 unless a
    model asks for more, and a stranger's MJCF never does — every Menagerie
    `scene.xml` leaves `<global offwidth>` alone, so a 900 px render dies
    with "Image width 900 > framebuffer width 640". Raising it on the spec
    before compiling is the fix that needs nothing from the model.
    """
    spec = mujoco.MjSpec.from_file(str(body.scene_fn()))
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), width)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), height)
    model = spec.compile()
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, body.stand_keyframe)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = data.xpos[1:].mean(axis=0) if model.nbody > 1 else [0, 0, 0]
    cam.distance = float(body.lab_spacing_m) * 0.6
    cam.azimuth, cam.elevation = azimuth, elevation
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=cam)
        return r.render()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("body", help="a registry id, e.g. menagerie:unitree_go2")
    ap.add_argument("--out", default="/tmp/probe-mjcf",
                    help="directory for <id>.png")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    args = ap.parse_args(argv)

    from microduck_local.robots import registry as R

    try:
        body = R.get(args.body)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = out / (args.body.replace(":", "-").replace("/", "-") + ".png")
    rgb = frame(body, width=args.width, height=args.height,
                azimuth=args.azimuth, elevation=args.elevation)
    try:
        import imageio.v3 as iio
        iio.imwrite(png, rgb)
    except ImportError:                      # pillow is the render dep here
        from PIL import Image
        Image.fromarray(rgb).save(png)

    extra = dict(getattr(body, "extra", {}) or {})
    print(f"{body.id}  ({body.title})")
    print(f"  {png}  {args.width}x{args.height}")
    print(f"  kind={body.kind} look={body.look()} joints={body.num_joints} "
          f"actions={body.num_actions} obs={body.obs_dim}")
    print(f"  keyframe={body.stand_keyframe!r} pose_source="
          f"{extra.get('pose_source')} scene={extra.get('scene_source')}")
    print(f"  measured {extra.get('measured_width_m')} m across -> stage "
          f"pitch {body.lab_spacing_m:.3f} m")
    if extra.get("license"):
        print(f"  licence: {extra['license']}")
    try:
        print(f"  {body.contract().describe()}")
    except NotImplementedError:
        print("  no contract declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
