"""LOOK at what a MARS reach policy does, and score it deterministically.

    uv run python scripts/probe_mars_reach.py runs/mars-reach-v1/policy.onnx \
        --seeds 8 --out /tmp/mars-reach

Phase 4a's eye. `AGENTS.md`'s verification discipline says two things about a
trained policy and both of them need a tool that exists: claims come from the
DETERMINISTIC exported ONNX and never from `ep_rew` curves, and you look at
the frames before you conclude anything. `render-rollout` is that tool for a
walker; it is built around `MicroduckWalkEnv`'s trunk height, foot contacts
and fall rule, none of which a wheeled body with an arm has. So this is the
same discipline with an arm's instruments:

  * every seed's FINAL distance and whether it held inside 2 cm for the last
    second (the task's own success rule, read off the env's `info`),
  * a 2x4 contact sheet at 1 Hz with the distance, the near-streak and any
    self-collision burned into each tile, and the target drawn as a sphere,
  * the per-term reward columns, because a number that looks right for the
    wrong reason is this repo's most expensive recurring failure.

`--null` scores a random-init policy through the identical path, so the
trained number has the baseline `AGENTS.md` rule 3 asks for.

Offscreen rendering uses `mujoco.Renderer`, which on macOS picks the bundled
CGL backend with no `MUJOCO_GL` setting and no display. Set `MUJOCO_GL=egl`
(or `osmesa`) on Linux.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

TARGET_RGBA = (0.2, 1.0, 0.4, 0.75)
EE_RGBA = (1.0, 0.45, 0.1, 0.9)
MONO_FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def make_camera(distance: float, azimuth: float, elevation: float,
                lookat: np.ndarray):
    """A fixed camera framing the arm, not the whole robot.

    The lookat sits at chassis-lid height rather than at the base origin: the
    arm and the target shell live between 0.05 and 0.35 m, and a camera aimed
    at z = 0 spent half the tile on floor (the same mistake the first G1
    contact sheet made, twelve tiles of shin).
    """
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = lookat
    return cam


def add_sphere(scene, pos, radius: float, rgba) -> None:
    """A visual-only marker. The target is the TASK's, not the physics' — no
    body in the model is at that point — so the renderer has to be told."""
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], np.float64),
        np.asarray(pos, np.float64), np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _font(size: int):
    from PIL import ImageFont

    for p in MONO_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_sheet(tiles, captions, header, footer, out_path: Path) -> None:
    """Grid of frames with the captions burned in below each one.

    Deliberately plain — white monospace on near-black. The audience is a
    model reading a downscaled PNG, so contrast and glyph size beat
    prettiness. (`render_rollout.build_sheet` is the same idea for a walker;
    it is not imported because that module is under concurrent edit and a
    contact sheet is forty lines.)
    """
    import textwrap

    from PIL import Image, ImageDraw

    n = len(tiles)
    th, tw = tiles[0].shape[:2]
    cols = 4 if n >= 4 else n
    rows = math.ceil(n / cols)
    fs = max(14, round(tw / 24))
    font, hfont = _font(fs), _font(fs + 4)
    line_h, pad = fs + 5, 10
    W = cols * tw + (cols + 1) * pad
    wrap_cols = max(20, int((W - 2 * pad) // (font.getlength("M") or 8.0)))
    header = [w for ln in header for w in (textwrap.wrap(ln, wrap_cols) or [""])]
    footer = [w for ln in footer for w in (textwrap.wrap(ln, wrap_cols) or [""])]
    cap_h = max(len(c) for c in captions) * line_h + 10
    head_h = len(header) * (fs + 9) + 2 * pad
    foot_h = len(footer) * line_h + 2 * pad
    H = head_h + rows * (th + cap_h + pad) + pad + foot_h

    img = Image.new("RGB", (W, H), (10, 10, 12))
    dr = ImageDraw.Draw(img)
    y = pad
    for line in header:
        dr.text((pad, y), line, font=hfont, fill=(255, 255, 255))
        y += fs + 9
    for i in range(n):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        ty = head_h + r * (th + cap_h + pad)
        img.paste(Image.fromarray(tiles[i]), (x, ty))
        dr.rectangle([x, ty, x + tw - 1, ty + th - 1], outline=(90, 90, 100))
        dr.rectangle([x, ty, x + 5 * fs // 2, ty + line_h + 4], fill=(10, 10, 12))
        dr.text((x + 5, ty + 2), f"#{i:02d}", font=font, fill=(230, 230, 240))
        cy = ty + th + 4
        dr.rectangle([x, ty + th, x + tw - 1, ty + th + cap_h - 1],
                     fill=(22, 22, 26))
        for line in captions[i]:
            dr.text((x + 6, cy), line[:int(tw // (font.getlength("M") or 8.0))],
                    font=font, fill=(230, 230, 240))
            cy += line_h
    y = H - foot_h + pad
    for line in footer:
        dr.text((pad, y), line, font=font, fill=(190, 200, 215))
        y += line_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def run_action_kwargs(policy: Path | None, args) -> dict:
    """The action map to build the scoring env with.

    A MARS policy's floats mean something different under each map — `delta`'s
    six arm actions are increments where `absolute`'s are positions — so an
    env built with the WRONG map scores a different controller than the one
    that was trained, and the number it prints is meaningless rather than bad.
    This is the same failure as Phase 4a's unclipped-action eval (the probe
    that handed the raw ONNX output to `step` scored a 2.6 cm policy at
    20.9 cm), so the map comes from the run's own record rather than from a
    default: `run.json`'s `env_kwargs` is where `train.py` writes it.

    An explicit flag still wins — scoring one policy under two maps is a
    legitimate question — and a loose .onnx with no run directory falls back
    to `mars_env`'s default, which is what it would have got anyway.
    """
    import json

    from microduck_local.robots import mars_env as ME

    kw: dict = {}
    if policy is not None:
        meta = Path(policy).resolve().parent / "run.json"
        if meta.exists():
            try:
                recorded = (json.loads(meta.read_text()).get("env_kwargs")
                            or {})
            except (OSError, ValueError):
                recorded = {}
            for key in ("action_mode", "action_scale_rad"):
                if key in recorded:
                    kw[key] = recorded[key]
    if args.action_mode is not None:
        kw["action_mode"] = args.action_mode
    if args.action_scale_rad is not None:
        kw["action_scale_rad"] = args.action_scale_rad
    # `delta` refuses a width it has no use for, so do not hand it one that
    # only came along for the ride.
    if kw.get("action_mode") == "delta":
        kw.pop("action_scale_rad", None)
    kw.setdefault("action_mode", ME.DEFAULT_ACTION_MODE)
    return kw


def rollout(env, act_fn, seed: int, renderer=None, cam=None,
            frame_every: int | None = None):
    """One deterministic episode. Returns (record, tiles, captions)."""
    obs, _ = env.reset(seed=seed)
    tiles, captions = [], []
    dists, terms_sum = [env.distance()], {}
    # Saturation and tail spread: a policy whose action sits on the box edge
    # has only bang-bang authority, and a bang-bang controller with a
    # rate-limited target CHATTERS across a 2 cm ball instead of settling in
    # it. Both numbers are here so that "it gets close but will not hold"
    # comes with its mechanism attached rather than as an adjective.
    sats, ee_tail = [], []
    info: dict = {"dist": dists[0], "success": False, "near_steps": 0,
                  "self_collision": None}
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if renderer is not None and frame_every and step % frame_every == 0:
            renderer.update_scene(env.data, camera=cam)
            add_sphere(renderer.scene, env.target, 0.02, TARGET_RGBA)
            add_sphere(renderer.scene, env.effector(), 0.012, EE_RGBA)
            tiles.append(renderer.render().copy())
            captions.append([
                f"t={step * env.dt:4.1f}s  d={info['dist'] * 100:5.1f}cm",
                f"near {info['near_steps']:3d}/{env.hold_steps}",
            ])
        action = act_fn(obs)
        edge = float(np.abs(env.action_space.high[0]))
        sats.append(float(np.mean(np.abs(action[:6]) >= edge - 1e-6)))
        obs, _rew, terminated, truncated, info = env.step(action)
        step += 1
        dists.append(info["dist"])
        ee_tail.append(env.effector().copy())
    for k, v in info["episode_rewards"].items():
        terms_sum[k] = v
    tail = np.array(ee_tail[-50:])            # the last 2 s
    record = {
        "seed": seed, "steps": step, "start": dists[0], "final": info["dist"],
        "best": min(dists), "success": bool(info["success"]),
        "near_steps": int(info["near_steps"]),
        "self_collision": info["self_collision"],
        "terminated": bool(terminated), "terms": terms_sum,
        "target_base": np.round(env.target_base_sample, 3).tolist(),
        "saturated": float(np.mean(sats)),
        "tail_spread": float(np.linalg.norm(tail.std(axis=0))),
        "tail_dist": float(np.mean(dists[-50:])),
    }
    return record, tiles, captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=None,
                    help="an exported policy.onnx (omit with --null)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mars-reach"))
    ap.add_argument("--sheet-seed", type=int, default=None,
                    help="which seed to render (default: the first)")
    ap.add_argument("--null", action="store_true",
                    help="score the ZERO action instead of a policy — the "
                         "arm parked at ARM_HOME, which is what the reward "
                         "has to beat before anything is credited to it")
    ap.add_argument("--action-mode", default=None,
                    help="the action map to score under (mars_env.ACTION_MODES). "
                         "Default: the one the policy's own run.json trained "
                         "with — see `run_action_kwargs`")
    ap.add_argument("--action-scale-rad", type=float, default=None,
                    help="override the absolute/cubic box width in rad "
                         "(default: the run's, else mars_env.ACTION_SCALE_RAD)")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--cam-distance", type=float, default=1.05)
    ap.add_argument("--cam-azimuth", type=float, default=130.0)
    ap.add_argument("--cam-elevation", type=float, default=-18.0)
    args = ap.parse_args()

    import mujoco

    from microduck_local.robots.mars_env import MarsArmEnv

    action_kwargs = run_action_kwargs(
        Path(args.policy) if args.policy else None, args)
    env = MarsArmEnv(seed=args.seed0, **action_kwargs)
    if args.null:
        label = "NULL (zero action: the arm parked at ARM_HOME)"

        def act_fn(_obs):
            return np.zeros(env.action_space.shape, np.float32)
    else:
        if not args.policy:
            ap.error("a policy path, or --null")
        import onnxruntime as ort

        sess = ort.InferenceSession(args.policy)
        in_name = sess.get_inputs()[0].name
        label = str(args.policy)

        def act_fn(obs):
            # CLIP to the action box, exactly as SB3 does in training and as
            # a deployed code skill must: the box is the contract, and this
            # policy's mean reaches |a| = 90 inside a +-1 space. Handing the
            # raw output over instead scored the same policy at 0.209 m
            # instead of 0.026 m (see `mars_env._get_obs`).
            raw = sess.run(None, {in_name: obs[None]})[0][0]
            return np.clip(raw, env.action_space.low,
                           env.action_space.high).astype(np.float32)

    # The camera looks at the chassis lid, where the arm and the shell are.
    lookat = np.array([0.10, 0.0, 0.20])
    cam = make_camera(args.cam_distance, args.cam_azimuth,
                      args.cam_elevation, lookat)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    # 8 tiles over an 8 s episode = 1 Hz, a 2x4 sheet.
    frame_every = max(1, env.max_steps // 8)

    sheet_seed = (args.sheet_seed if args.sheet_seed is not None
                  else args.seed0)
    records, sheet = [], None
    for i in range(args.seeds):
        seed = args.seed0 + i
        want_sheet = seed == sheet_seed
        rec, tiles, caps = rollout(
            env, act_fn, seed,
            renderer if want_sheet else None, cam,
            frame_every if want_sheet else None)
        records.append(rec)
        if want_sheet:
            sheet = (tiles, caps, rec)
        hit = rec["self_collision"]
        print("seed %3d  start %.3f  best %.3f  FINAL %.3f m  held %3d/%d  "
              "%-7s  steps %3d  %s"
              % (seed, rec["start"], rec["best"], rec["final"],
                 rec["near_steps"], env.hold_steps,
                 "SUCCESS" if rec["success"] else "-", rec["steps"],
                 f"self-collision {hit[0]}<->{hit[1]}" if hit else ""))

    finals = np.array([r["final"] for r in records])
    bests = np.array([r["best"] for r in records])
    n_ok = int(sum(r["final"] <= 0.02 for r in records))
    n_success = int(sum(r["success"] for r in records))
    n_hit = int(sum(r["self_collision"] is not None for r in records))
    terms = {}
    for r in records:
        for k, v in r["terms"].items():
            terms[k] = terms.get(k, 0.0) + v / len(records)

    box = ("" if env.action_mode == "delta"
           else f" box +-{env.action_scale_rad:g} rad")
    summary = [
        f"policy: {label}",
        f"action map: {env.action_mode}{box}  "
        f"(from {'--action-mode' if args.action_mode else 'the run'})",
        f"seeds {args.seed0}..{args.seed0 + args.seeds - 1}  "
        f"final dist: median {np.median(finals):.4f}  mean {finals.mean():.4f}  "
        f"min {finals.min():.4f}  max {finals.max():.4f} m",
        f"best-in-episode: median {np.median(bests):.4f} m",
        f"FINAL <= 2 cm: {n_ok}/{args.seeds}      "
        f"held 1 s inside 2 cm (the task's rule): {n_success}/{args.seeds}",
        f"self-collision terminations: {n_hit}/{args.seeds}",
        "mechanism: arm-action dims on the box edge "
        f"{np.mean([r['saturated'] for r in records]) * 100:.0f}% of steps; "
        "effector spread over the last 2 s "
        f"{np.mean([r['tail_spread'] for r in records]) * 1000:.1f} mm; "
        "mean distance over the last 2 s "
        f"{np.mean([r['tail_dist'] for r in records]) * 100:.1f} cm",
        "per-episode terms: " + "  ".join(f"{k} {v:+.2f}"
                                          for k, v in terms.items()),
    ]
    print()
    for line in summary:
        print(line)

    if sheet is not None:
        tiles, caps, rec = sheet
        header = [f"MARS reach — {Path(label).name if not args.null else label}",
                  f"seed {rec['seed']}  target(base) {rec['target_base']}  "
                  f"final {rec['final'] * 100:.1f} cm  "
                  f"{'SUCCESS' if rec['success'] else 'not held'}"]
        out = args.out / "sheet.png"
        build_sheet(tiles, caps, header, summary, out)
        print(f"\nsheet: {out}  ({len(tiles)} tiles at "
              f"{frame_every * env.dt:.1f} s)")


if __name__ == "__main__":
    main()
