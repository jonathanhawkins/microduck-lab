"""Render a policy rollout to mp4 AND to a captioned contact sheet.

    uv run render-rollout --policy runs/<run>/policy.onnx --behavior backflip \
        --episodes 2 --seconds 8 --out /tmp/out

The mp4 is for a human. The **contact sheet** is for a model: evenly sampled
frames with the numbers burned in (trunk height vs the standing reference, head
height, pitch, foot contacts, which policy is driving), because the failure
modes this project keeps hitting are invisible to reward batteries but obvious
in a picture with a ruler next to it:

  - a "stand" that scores perfectly can be a folded crouch — orientation and
    ELEVATION are independent, so every caption carries trunk/head z against
    the STAND-keyframe reference;
  - a maneuver can be the demo spotter's assist torque rather than the policy
    (render `--policy limp` as a null control and compare);
  - a "hold" can be rapid cycling rather than one sustained pose — hence the
    reversal counts in the printed summary.

Offscreen rendering uses mujoco.Renderer, which on macOS picks the bundled CGL
backend (`mujoco.cgl`) with no MUJOCO_GL setting and no display — verified
working on Apple Silicon. Set MUJOCO_GL=egl/osmesa only on a Linux box.

The env is built exactly the way eval and the lab build it (BehaviorEnv with
obs_noise/domain_rand/action_delay/random_yaw off, fixed seed) so what you see
is the policy, not the randomizers. `--env KEY=VALUE` sets the behaviors'
per-stage knobs (MICRODUCK_BF_SPAWN_LO/HI, MICRODUCK_SPAWN_FAMILY_PROBS,
MICRODUCK_EPISODE_S) both in os.environ and as per-instance spawn_overrides,
so one phase of a staged trick can be rendered on its own.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import contract as C

# Free-camera presets: (azimuth, elevation) in degrees. Azimuth 90 puts the
# camera on -y looking toward +y, which lands the robot's forward axis (+x) on
# the right of the frame — the view that reads a pitch maneuver (backflip) as
# rotation in the image plane.
CAMERAS: dict[str, tuple[float, float]] = {
    "side": (90.0, -8.0),
    "front": (180.0, -8.0),
    "three-quarter": (125.0, -14.0),
}
LOOKAT_Z = 0.14   # m — mid-body of a 25 cm robot; fixed (only x/y track the
                  # trunk) so the floor stays in frame while the trunk bobs.
# 0.70 m at MuJoCo's default 45 deg fovy spans ~0.58 m vertically: the robot
# fills ~45% of the tile — big enough to read a posture, loose enough that a
# flip's whole arc stays inside the frame.
CAM_DISTANCE = 0.70

# The lab's handoff rule, mirrored from viz_server.Duck._handoff_due: the
# trick is finished and the duck is on both feet.
HANDOFF_ROT_RAD = 5.2

MONO_FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

HANDOFF_RGB = (255, 176, 46)   # amber: frames after the policy switch
PLAIN_RGB = (235, 235, 235)

# Caption budget: tiles are `--width` px and the caption font is width/22, so a
# 480 px tile holds ~34 monospace columns. Captions are generated here, so they
# are BUILT to fit rather than trusted to (tests/test_render_rollout.py locks
# it) — a clipped caption is a caption the reader silently mis-reads.
CAPTION_COLUMNS = 34


# --------------------------------------------------------------- pure helpers

def parse_env_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """`--env KEY=VALUE` repeated -> dict. Values may contain '='."""
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--env expects KEY=VALUE, got {p!r}")
        out[k] = v
    return out


def sheet_indices(n_frames: int, k: int) -> list[int]:
    """`k` evenly spaced frame indices spanning [0, n_frames-1], endpoints
    included — the last frame is where a trick either landed or did not, so it
    must never be sampled away."""
    if n_frames <= 0 or k <= 0:
        return []
    k = min(k, n_frames)
    if k == 1:
        return [0]
    return [round(i * (n_frames - 1) / (k - 1)) for i in range(k)]


def grid_cols(n: int, cap: int = 4) -> int:
    """Near-square grid, capped so tiles stay big enough to read after the
    viewer downscales the sheet."""
    return max(1, min(cap, math.ceil(math.sqrt(max(n, 1)))))


def count_reversals(values: Sequence[float], deadband: float) -> int:
    """Direction changes in a 1-D signal, ignoring wiggles below `deadband`.

    The 'is it holding or is it cycling?' test: one sustained pose gives ~0,
    a policy flapping in and out of the pose gives many. Reward batteries
    average the two into the same number.
    """
    if len(values) < 3:
        return 0
    reversals = 0
    direction = 0
    anchor = float(values[0])
    for v in values[1:]:
        delta = float(v) - anchor
        if abs(delta) < deadband:
            continue
        sign = 1 if delta > 0 else -1
        if direction and sign != direction:
            reversals += 1
        direction = sign
        anchor = float(v)
    return reversals


@dataclass(frozen=True)
class FrameDiag:
    """Everything burned into one contact-sheet caption."""
    index: int
    t: float
    trunk_z: float
    head_z: float
    pitch_deg: float          # signed backward pitch: 0 upright, +180 inverted
    tilt_deg: float           # total lean from vertical, sign-free
    rot_deg: float | None     # behavior's accumulated trick rotation, if tracked
    contact_l: bool
    contact_r: bool
    ground_bodies: tuple[str, ...]   # non-foot bodies resting on the floor
    driver: str
    handed_off: bool


def short_label(label: str, width: int = 13) -> str:
    """Truncate a policy label from the LEFT: run names disambiguate in their
    tail (`...-402439-s5`), so keeping the head loses exactly the useful part."""
    return label if len(label) <= width else "…" + label[-(width - 1):]


def pack_names(names: Sequence[str], budget: int) -> str:
    """As many names as fit in `budget` columns, then '+N' for the rest.

    A caption that runs off its tile is a caption the reader silently
    mis-reads, so the ground-contact list is budgeted rather than trusted to
    be short; the episode summary prints the full list.
    """
    if not names:
        return "none"
    kept: list[str] = []
    used = 0
    for i, n in enumerate(names):
        extra = len(n) + (1 if kept else 0)
        rest = len(names) - i - 1
        tail = len(f",+{rest}") if rest else 0
        if kept and used + extra + tail > budget:
            break
        kept.append(n)
        used += extra
    left = len(names) - len(kept)
    out = ",".join(kept) + (f",+{left}" if left else "")
    return out[:budget]


def format_caption(d: FrameDiag, stand_z: float, head_ref_z: float) -> tuple[str, ...]:
    """Caption lines for one frame — kept under ~34 monospace columns so they
    fit a tile at the default render width.

    Every height carries its standing reference inline: a bare '0.09 m' means
    nothing to a reader who does not already know the model, and that ignorance
    is exactly how a folded crouch got scored as a stand.
    """
    head = "  n/a" if not math.isfinite(d.head_z) else f"{d.head_z:.3f}"
    head_ref = "n/a" if not math.isfinite(head_ref_z) else f"{head_ref_z:.3f}"
    rot = "" if d.rot_deg is None else f" rot={d.rot_deg:+.0f}"
    ground = pack_names(d.ground_bodies, CAPTION_COLUMNS - len("feet L=1 R=1  floor:"))
    return (
        f"#{d.index:02d} t={d.t:5.2f}s drv={short_label(d.driver)}",
        f"trunk_z={d.trunk_z:.3f} (stand {stand_z:.3f})",
        f"head_z ={head} (stand {head_ref})",
        f"deg: pitch={d.pitch_deg:+.0f} tilt={d.tilt_deg:.0f}{rot}",
        f"feet L={int(d.contact_l)} R={int(d.contact_r)}  floor:{ground}",
    )


# ------------------------------------------------------------------- drivers

@dataclass
class Driver:
    label: str
    fn: Callable[[np.ndarray, object], np.ndarray]


def _policy_label(path: Path) -> str:
    # runs/<run>/policy.onnx is named by its run, not by "policy".
    return path.parent.name if path.stem in ("policy", "live") else path.stem


def load_driver(spec: str) -> Driver:
    """An .onnx path, or one of the two NULL CONTROLS.

    `zero`  — action 0 everywhere, i.e. hold DEFAULT_POSE stiffly.
    `limp`  — action = (current joint pos - DEFAULT_POSE), so every servo's
              target is where it already is: no restoring torque, the body
              slumps. This is the comparison that proves a maneuver came from
              the policy; if the limp duck does the same thing, something else
              (gravity, a spawn pose, an assist torque) is driving it.
    """
    key = spec.strip().lower()
    if key in ("zero", "none"):
        return Driver("zero", lambda obs, env: np.zeros(C.NUM_JOINTS, np.float32))
    if key == "limp":
        def limp(obs, env):
            q = env.data.qpos[env.joint_qpos_adr]
            return (q - C.DEFAULT_POSE).astype(np.float32)
        return Driver("limp", limp)

    path = Path(spec).expanduser()
    if not path.exists():
        raise SystemExit(f"policy not found: {path} (or use 'limp'/'zero')")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path))
    in_name = sess.get_inputs()[0].name

    def infer(obs, env):
        return sess.run(None, {in_name: obs[None]})[0][0].astype(np.float32)

    return Driver(_policy_label(path), infer)


def handoff_due(env) -> bool:
    """Mirrors viz_server.Duck._handoff_due."""
    rot = getattr(env, "_bf_rot", None)
    if rot is None or rot < HANDOFF_ROT_RAD:
        return False
    c = getattr(env, "foot_contact_state", {})
    if not (c.get("left") and c.get("right")):
        return False
    # Rate gate, mirrored: hand off only once the spin is braked (see
    # viz_server._handoff_due — an early handoff pivots on landing).
    w = env.data.sensordata[env.gyro_adr]
    return float(w[0] ** 2 + w[1] ** 2 + w[2] ** 2) < 2.0


# ----------------------------------------------------------------- rendering

class Probe:
    """Per-frame diagnostics pulled off a live env."""

    def __init__(self, env):
        import mujoco

        self.env = env
        m = env.model
        self.head_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
        # Bodies owning the foot collision geoms — everything else touching the
        # floor is the robot dragging, which is what separates a stand from a
        # slump (behaviors.py prices the same distinction).
        self.foot_bids = {int(m.geom_bodyid[g]) for g in env.foot_geoms.values()}
        self.body_name = lambda b: (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
        # Standing references, measured off the model (never hand-carried) on a
        # scratch MjData so the live episode is untouched.
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, env.key_stand)
        mujoco.mj_forward(m, d)
        self.stand_z = float(d.xpos[env.trunk_body_id][2])
        self.head_ref_z = (float(d.xpos[self.head_bid][2])
                           if self.head_bid >= 0 else float("nan"))

    def _ground_bodies(self) -> tuple[str, ...]:
        env = self.env
        d, m = env.data, env.model
        names: list[str] = []
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
            if env.floor_geom not in (g1, g2):
                continue
            other = g2 if g1 == env.floor_geom else g1
            b = int(m.geom_bodyid[other])
            if b in self.foot_bids:
                continue
            n = self.body_name(b)
            if n not in names:
                names.append(n)
        return tuple(names)

    def sample(self, index: int, step: int, driver: str, handed: bool) -> FrameDiag:
        env = self.env
        g = env._projected_gravity()
        # For a pure backward pitch by `rot`, gravity in the trunk frame is
        # (-sin rot, 0, -cos rot) — so this angle equals the trick's own
        # rotation convention (behaviors._bf_update integrates the same sign).
        pitch = math.degrees(math.atan2(-float(g[0]), -float(g[2])))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))
        contacts = env._foot_contacts()
        rot = getattr(env, "_bf_rot", None)
        head_z = (float(env.data.xpos[self.head_bid][2])
                  if self.head_bid >= 0 else float("nan"))
        return FrameDiag(
            index=index,
            t=step * C.CTRL_DT,
            trunk_z=float(env.data.xpos[env.trunk_body_id][2]),
            head_z=head_z,
            pitch_deg=pitch,
            tilt_deg=tilt,
            rot_deg=None if rot is None else math.degrees(float(rot)),
            contact_l=bool(contacts["left"]),
            contact_r=bool(contacts["right"]),
            ground_bodies=self._ground_bodies(),
            driver=driver,
            handed_off=handed,
        )


def make_camera(name: str, distance: float):
    import mujoco

    if name not in CAMERAS:
        raise SystemExit(f"--camera must be one of {sorted(CAMERAS)}")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = CAMERAS[name]
    cam.distance = distance
    return cam


def run_episode(env, renderer, cam, probe: Probe, seed: int, driver: Driver,
                handoff: Driver | None, stride: int):
    """One rollout; returns (frames, diags, meta)."""
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    diags: list[FrameDiag] = []
    handed = False
    handoff_t: float | None = None

    def capture(step: int) -> None:
        cam.lookat[:] = (float(env.data.xpos[env.trunk_body_id][0]),
                         float(env.data.xpos[env.trunk_body_id][1]), LOOKAT_Z)
        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render().copy())
        label = (handoff.label if handed and handoff else driver.label)
        diags.append(probe.sample(len(diags), step, label, handed))

    capture(0)
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        if handoff is not None and not handed and handoff_due(env):
            handed = True
            handoff_t = step * C.CTRL_DT
        action = (handoff.fn if handed else driver.fn)(obs, env)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        if step % stride == 0 or terminated or truncated:
            capture(step)

    meta = {
        "steps": step,
        "seconds": step * C.CTRL_DT,
        "outcome": "FELL (terminated)" if terminated else "completed (truncated)",
        "terminated": bool(terminated),
        "handoff_t": handoff_t,
        "spawn": getattr(env, "last_spawn", None),
    }
    return frames, diags, meta


def summarize(diags: Sequence[FrameDiag], meta: dict, probe: Probe) -> list[str]:
    """The numbers a reader needs next to the pictures."""
    tz = [d.trunk_z for d in diags]
    hz = [d.head_z for d in diags if math.isfinite(d.head_z)]
    both = sum(d.contact_l and d.contact_r for d in diags) / len(diags)
    air = sum(not (d.contact_l or d.contact_r) for d in diags) / len(diags)
    dragging = sum(bool(d.ground_bodies) for d in diags) / len(diags)
    lines = [
        f"outcome: {meta['outcome']} after {meta['steps']} steps "
        f"({meta['seconds']:.2f} s); spawn={meta['spawn']}",
        f"trunk_z  min {min(tz):.3f}  max {max(tz):.3f}  final {tz[-1]:.3f}   "
        f"(STAND reference {probe.stand_z:.3f} m)",
    ]
    if hz:
        lines.append(
            f"head_z   min {min(hz):.3f}  max {max(hz):.3f}  final {hz[-1]:.3f}   "
            f"(STAND reference {probe.head_ref_z:.3f} m)")
    rots = [d.rot_deg for d in diags if d.rot_deg is not None]
    if rots:
        lines.append(f"trick rotation: max {max(rots):.0f} deg, final {rots[-1]:.0f} deg "
                     f"(a full flip is 360; the lab hands off at 298)")
    # Captions budget this list to fit a tile; the summary is where the full
    # set lives, because "which part of the robot is on the ground" is the
    # difference between standing and slumping.
    ground: list[str] = []
    for d in diags:
        for n in d.ground_bodies:
            if n not in ground:
                ground.append(n)
    lines.append(
        f"contacts: both feet {both:.0%} of frames, airborne {air:.0%}, "
        f"non-foot body on floor {dragging:.0%} "
        f"({', '.join(ground) if ground else 'nothing but feet'})")
    # Hold vs cycling: a sustained pose reverses ~0 times; a policy flapping
    # in and out of the pose reverses many, and the mean hides it.
    lines.append(
        f"reversals (hold-vs-cycling): trunk_z {count_reversals(tz, 0.015)}, "
        f"pitch {count_reversals([d.pitch_deg for d in diags], 15.0)} "
        f"over {meta['seconds']:.1f} s")
    if meta["handoff_t"] is not None:
        lines.append(f"handoff fired at t={meta['handoff_t']:.2f} s")
    elif meta.get("had_handoff"):
        lines.append("handoff NEVER fired (trick did not complete on both feet)")
    return lines


# --------------------------------------------------------------- contact sheet

def _font(size: int):
    from PIL import ImageFont

    for p in MONO_FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _cols_for(font, width_px: int) -> int:
    """How many monospace columns fit in `width_px` — the sheet's own text is
    generated here, so it must never silently run off the edge of a tile."""
    try:
        adv = font.getlength("M") or 1.0
    except AttributeError:  # very old Pillow bitmap fonts
        adv = 8.0
    return max(8, int(width_px // adv))


def _wrap(lines: Sequence[str], cols: int) -> list[str]:
    import textwrap

    out: list[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, cols) or [""])
    return out


def build_sheet(tiles: Sequence[np.ndarray], captions: Sequence[Sequence[str]],
                header: Sequence[str], footer: Sequence[str],
                highlight: Sequence[bool], out_path: Path) -> None:
    """Grid of frames with the captions burned in below each one.

    Deliberately plain: white monospace on near-black, no decoration. The
    audience is a model reading a downscaled PNG, so contrast and glyph size
    beat prettiness.
    """
    from PIL import Image, ImageDraw

    n = len(tiles)
    th, tw = tiles[0].shape[:2]
    cols = grid_cols(n)
    rows = math.ceil(n / cols)
    fs = max(16, round(tw / 22))
    font, hfont = _font(fs), _font(fs + 4)
    line_h = fs + 5
    pad = 10

    W = cols * tw + (cols + 1) * pad
    tile_cols = _cols_for(font, tw - 12)
    captions = [[ln[:tile_cols] for ln in c] for c in captions]
    header = _wrap(header, _cols_for(hfont, W - 2 * pad))
    footer = _wrap(footer, _cols_for(font, W - 2 * pad))

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

    y = head_h
    for i in range(n):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        ty = y + r * (th + cap_h + pad)
        img.paste(Image.fromarray(tiles[i]), (x, ty))
        colour = HANDOFF_RGB if highlight[i] else PLAIN_RGB
        # Frame border doubles as the handoff annotation: amber = a different
        # policy produced this frame's action.
        dr.rectangle([x, ty, x + tw - 1, ty + th - 1], outline=colour,
                     width=3 if highlight[i] else 1)
        # Index badge on the image itself, so a reader can cite a frame.
        dr.rectangle([x, ty, x + 5 * fs // 2, ty + line_h + 4], fill=(10, 10, 12))
        dr.text((x + 5, ty + 2), f"#{i:02d}", font=font, fill=colour)
        cy = ty + th + 4
        dr.rectangle([x, ty + th, x + tw - 1, ty + th + cap_h - 1], fill=(22, 22, 26))
        for line in captions[i]:
            dr.text((x + 6, cy), line, font=font, fill=colour)
            cy += line_h

    y = H - foot_h + pad
    for line in footer:
        dr.text((pad, y), line, font=font, fill=(190, 200, 215))
        y += line_h

    img.save(out_path)


def sheet_footer(probe: Probe) -> list[str]:
    """The legend travels WITH the image: a sheet read weeks later, or by a
    model with no other context, still has to be interpretable on its own."""
    head_ref = ("n/a" if not math.isfinite(probe.head_ref_z)
                else f"{probe.head_ref_z:.3f} m")
    return [
        f"REFERENCE (STAND keyframe): trunk_z {probe.stand_z:.3f} m, "
        f"head_z (jaw_soft) {head_ref}.  Heights far below these are a CROUCH "
        f"or COLLAPSE, however upright the pose looks.",
        "deg line: pitch = signed backward pitch of the trunk, WRAPPED to "
        "+/-180 (0 upright); tilt = total lean from vertical; rot = the "
        "behavior's own accumulated trick rotation, which keeps counting past "
        "180 (backflip: 360 = a full turn).",
        "feet L/R = 1 when that foot geom touches the floor.  floor: = non-foot "
        "bodies resting on the ground (trunk/hips = dragging, not standing).",
        "drv = the policy that produced this frame's action; amber frames ran "
        "the --handoff policy instead of the trick policy.",
    ]


# ------------------------------------------------------------------------ cli

def build_env(behavior_id: str, env_overrides: dict[str, str], seed: int):
    """Exactly how eval and the lab build a behavior env: randomizers off, so
    what the sheet shows is the policy and not the noise."""
    from .behaviors import BEHAVIORS, BehaviorEnv

    if behavior_id not in BEHAVIORS:
        raise SystemExit(f"unknown behavior {behavior_id!r}; "
                         f"choose from {sorted(BEHAVIORS)}")
    # Both channels: the process environment (what a trainer subprocess would
    # see, and what _spawn_knob falls back to) and per-instance overrides
    # (what the lab's preview envs use). Set before the env is constructed —
    # BehaviorEnv reads MICRODUCK_EPISODE_S in __init__.
    os.environ.update(env_overrides)
    return BehaviorEnv(behavior_id, obs_noise=False, domain_rand=False,
                       action_delay=False, random_yaw=False, seed=seed,
                       spawn_overrides=dict(env_overrides))


def behavior_from_policy(policy: str) -> str | None:
    """runs/<run>/behavior.json records what a run was trained on."""
    try:
        bj = Path(policy).parent / "behavior.json"
        return json.loads(bj.read_text()).get("behavior")
    except (OSError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a policy rollout to mp4 + a captioned contact sheet.")
    ap.add_argument("--policy", required=True,
                    help="path to an .onnx policy, or 'limp'/'zero' for a null control")
    ap.add_argument("--behavior", default=None,
                    help="behavior id (default: read runs/<run>/behavior.json)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=None,
                    help="episode length (default: the behavior's own; an explicit "
                         "--env MICRODUCK_EPISODE_S wins)")
    ap.add_argument("--seed", type=int, default=0, help="episode N uses seed+N")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="behavior env knob, repeatable (MICRODUCK_BF_SPAWN_LO/HI, "
                         "MICRODUCK_SPAWN_FAMILY_PROBS, MICRODUCK_EPISODE_S)")
    ap.add_argument("--handoff", default=None,
                    help="second .onnx that takes over once the trick completes "
                         "and both feet are down (the lab's rule)")
    ap.add_argument("--camera", default="side", choices=sorted(CAMERAS))
    ap.add_argument("--distance", type=float, default=CAM_DISTANCE)
    ap.add_argument("--fps", type=int, default=30,
                    help="target video fps; control runs at 50 Hz so the frame "
                         "stride is rounded and the real fps is reported")
    ap.add_argument("--sheet-frames", type=int, default=12)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    try:
        overrides = parse_env_overrides(args.env)
    except ValueError as e:
        raise SystemExit(str(e))
    # --seconds is spelled as the behaviors' own episode knob so it survives
    # BehaviorEnv's hard override; an explicit --env of the same key wins.
    if args.seconds is not None:
        overrides.setdefault("MICRODUCK_EPISODE_S", str(args.seconds))

    behavior = args.behavior or behavior_from_policy(args.policy)
    if not behavior:
        raise SystemExit("--behavior is required (no behavior.json next to the policy)")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    # h264 (yuv420p) rejects odd dimensions, and a silent ffmpeg rescale would
    # blur the captions' companion video.
    width, height = args.width - args.width % 2, args.height - args.height % 2

    import imageio.v2 as imageio
    import mujoco

    driver = load_driver(args.policy)
    handoff = load_driver(args.handoff) if args.handoff else None
    env = build_env(behavior, overrides, seed=args.seed)
    probe = Probe(env)
    cam = make_camera(args.camera, args.distance)
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    ctrl_hz = 1.0 / C.CTRL_DT
    stride = max(1, round(ctrl_hz / max(args.fps, 1)))
    real_fps = ctrl_hz / stride

    knobs = " ".join(f"{k}={v}" for k, v in sorted(overrides.items())) or "(none)"
    print(f"policy: {args.policy}  behavior: {behavior}  camera: {args.camera}")
    print(f"env knobs: {knobs}")
    print(f"render {width}x{height} @ {real_fps:.1f} fps "
          f"(stride {stride} of the 50 Hz control loop), backend "
          f"{mujoco.GLContext.__module__}")
    if handoff:
        print(f"handoff: {args.handoff} (label {handoff.label}) at rot>="
              f"{HANDOFF_ROT_RAD} rad with both feet down")

    for ep in range(args.episodes):
        frames, diags, meta = run_episode(
            env, renderer, cam, probe, seed=args.seed + ep,
            driver=driver, handoff=handoff, stride=stride)
        meta["had_handoff"] = handoff is not None

        mp4 = out / f"ep{ep}.mp4"
        # macro_block_size=None keeps the rendered resolution exactly (no silent
        # upscale); h264's yuv420p — imageio's default — needs even dimensions,
        # which main() has already enforced.
        with imageio.get_writer(mp4, fps=real_fps, macro_block_size=None) as w:
            for f in frames:
                w.append_data(f)

        idx = sheet_indices(len(frames), args.sheet_frames)
        picked = [diags[i] for i in idx]
        captions = [format_caption(d, probe.stand_z, probe.head_ref_z)
                    for d in picked]
        header = [
            f"{args.policy}   behavior={behavior}   episode {ep} (seed "
            f"{args.seed + ep})   camera={args.camera}",
            f"env knobs: {knobs}",
            "  |  ".join(summarize(diags, meta, probe)[:1]
                         + [f"handoff={args.handoff or 'none'}"]),
        ]
        sheet = out / f"ep{ep}_sheet.png"
        build_sheet([frames[i] for i in idx], captions, header,
                    sheet_footer(probe), [d.handed_off for d in picked], sheet)

        print(f"\n--- episode {ep} ---")
        for line in summarize(diags, meta, probe):
            print("  " + line)
        # Same captions as the sheet, in text: an agent that cannot see the
        # PNG still gets every number, and the two can never disagree.
        print(f"  sampled frames ({len(idx)} of {len(frames)}):")
        for c in captions:
            print("    " + " | ".join(c))
        print(f"  wrote {mp4} ({mp4.stat().st_size // 1024} KB) "
              f"and {sheet} ({sheet.stat().st_size // 1024} KB)")

    renderer.close()
    print(f"\nREAD the *_sheet.png files — the captions carry the numbers. "
          f"Output dir: {out}")


if __name__ == "__main__":
    main()
