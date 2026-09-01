---
name: render-rollout
description: >-
  Render a microduck policy rollout to video AND to a frame contact sheet with
  per-frame diagnostics burned in, then READ the sheet to see what the policy
  actually does. Use before concluding anything about a trained policy or
  behavior: when an eval battery / reward breakdown is surprising or looks too
  good, when the user says the robot "isn't doing X" but the metrics disagree,
  when comparing curriculum stages or checkpoints, and when hunting for new
  reward-term ideas. Trigger on: "what is this policy doing", "why did the
  backflip not work", "the numbers say it stands", "render the rollout",
  "look at the policy", "show me the trick", "check the behavior visually".
---

# render-rollout — look at the policy before you believe the numbers

Reward batteries have repeatedly lied in this project, and a human glancing at
the 3D viewer caught it instantly:

- what the metrics scored as **"standing"** was a **folded crouch** — orientation
  was perfect, elevation was on the floor;
- what looked like a **"backflip"** was mostly the **demo spotter's assist
  torque** shoving the robot over, not the policy;
- what the reward called a **"hold"** was **rapid cycling** in and out of the pose,
  averaged into the same number as a real hold.

You can look at still images. So render the rollout and **read the contact
sheet**. The mp4 is for the human; `ep<N>_sheet.png` is for you.

## When to use it

Use it **before** stating what a policy does — not after. Specifically:

- Before concluding a behavior works, half-works, or failed.
- When a battery result is surprising, suspiciously good, or contradicts a
  previous run.
- When the user says the robot "isn't doing X" and the metrics say it is.
  (The user is watching pixels; you are reading sums. Go look.)
- When comparing curriculum stages (`teach-<id>-<hash>-s1..sN`) or checkpoints:
  render each and diff the sheets.
- When you need **new reward-term ideas**: the sheet shows what the policy
  actually settled into, and the gap between that and the intent is the term
  you are missing.

Do **not** start or restart training, and do **not** restart `duck-lab`, to
look at a policy — rendering is a separate read-only process.

## How to run it

From `microduck_local/` (uv for everything):

```bash
uv run render-rollout --policy runs/<run>/policy.onnx --behavior backflip \
    --episodes 2 --seconds 8 --out /tmp/rr-<run>
```

`--behavior` is optional when the run has a `behavior.json` (every `teach-*`
run does) — it is read from `runs/<run>/behavior.json`. Then:

**READ the generated `/tmp/rr-<run>/ep0_sheet.png` and `ep1_sheet.png` with the
Read tool.** They are images; the Read tool renders them. The same numbers are
also printed to stdout, so you get them either way — but look at the pictures,
that is the point of the tool.

Key options:

| flag | what it does |
|---|---|
| `--policy` | `.onnx` path, **or `limp` / `zero`** for a null control (see below) |
| `--behavior` | behavior id; defaults to the run's `behavior.json` |
| `--env KEY=VALUE` | repeatable behavior env knob, set before the env is built |
| `--handoff <onnx>` | second policy that takes over when the trick completes |
| `--camera side\|front\|three-quarter` | `side` (default) reads pitch maneuvers best |
| `--seconds`, `--episodes`, `--seed` | episode length / count / seeds (`seed+N` per episode) |
| `--sheet-frames` | frames on the sheet (default 12; use 20+ to check for cycling) |
| `--fps`, `--width`, `--height` | video/tile size (defaults 30 / 480 / 360) |

Defaults are deliberately modest (2 episodes, 480x360) — an 8 s rollout renders
in ~10 s and must not starve live trainers. Prefix with `nice -n 10` if training
is running.

### Rendering one phase of a staged trick

The behaviors read per-stage knobs from the environment, so `--env` picks the
phase you want to see. From the backflip curriculum in `behaviors/backflip.py`:

```bash
# just the landing rehearsal: always spawn already-landed
uv run render-rollout --policy runs/<your-backflip-run>/policy.onnx \
    --env MICRODUCK_SPAWN_FAMILY_PROBS=1.0,0.0 --out /tmp/rr-landing

# just the mid-roll carry, in a narrow rotation window
uv run render-rollout --policy runs/<your-backflip-run>/policy.onnx \
    --env MICRODUCK_SPAWN_FAMILY_PROBS=0.0,1.0 \
    --env MICRODUCK_BF_SPAWN_LO=2.6 --env MICRODUCK_BF_SPAWN_HI=5.0 \
    --out /tmp/rr-carry

# the honest whole-trick attempt: plain standing starts only
uv run render-rollout --policy runs/<your-backflip-run>/policy.onnx \
    --env MICRODUCK_SPAWN_FAMILY_PROBS=0.0,0.0 --out /tmp/rr-entry
```

Read the knob names off the behavior's `curriculum` stages in
`src/microduck_local/behaviors/` (one module per trick) — never guess them. A
human may be editing those files at the same time as you: re-read before
editing rather than
working from memory.

### Handoff

`--handoff <onnx>` mirrors the lab's rule (`viz_server.Duck._handoff_due`):
once `env._bf_rot >= 5.2` **and** both feet are in contact, the second policy
drives. Frames after the switch are annotated — amber border and amber caption,
with `drv=<handoff label>`.

```bash
uv run render-rollout --policy runs/<your-backflip-run>/policy.onnx \
    --handoff ../microduck/policies/alpha_stand.onnx --out /tmp/rr-handoff
```

The summary reports `handoff fired at t=…` or **`handoff NEVER fired`** — which
alone tells you the trick did not complete on both feet.

## What to look for in the sheet

**Read the burned-in numbers. Do not trust the impression the picture gives.**
A duck can look plausibly upright in a 480 px tile and be 4 cm off the floor.

Each caption carries:

```
#04 t= 1.44s drv=…ip-402439-s5     <- frame index, time, WHICH POLICY drove it
trunk_z=0.103 (stand 0.120)        <- height vs the STAND-keyframe reference
head_z =0.044 (stand 0.233)        <- head (jaw_soft) height vs its reference
deg: pitch=-49 tilt=49 rot=+308    <- pitch wraps +/-180; rot accumulates
feet L=1 R=1  floor:jaw_soft       <- foot contacts; non-foot bodies on the ground
```

The three failure patterns learned the hard way here, and how the sheet
exposes each:

### 1. "Upright" can be a collapsed crouch — check HEIGHT, not orientation

Orientation and elevation are independent. `tilt=0` proves nothing.

- Compare `trunk_z` and `head_z` against the `(stand …)` reference printed in
  every caption and in the sheet footer.
- `head_z=0.044` against a 0.233 reference is a duck lying on its face, not a
  stand — however tidy `pitch`/`tilt` look.
- `floor:` lists **non-foot** bodies touching the ground. `floor:jaw_soft` or
  `floor:trunk_base` means dragging/slumping. A real stand shows `floor:none`
  with `feet L=1 R=1`.
- The summary's `non-foot body on floor NN% of frames` is the one-number
  version of this test.

### 2. A maneuver may not be the policy — always render a null control

`--policy limp` re-runs the exact same rollout with every servo target pinned
to where the joint already is: no restoring torque, the body just slumps.
`--policy zero` holds `DEFAULT_POSE` stiffly.

```bash
uv run render-rollout --policy limp --behavior backflip \
    --env MICRODUCK_SPAWN_FAMILY_PROBS=0.0,1.0 --out /tmp/rr-null
```

If the limp duck produces the same rotation, landing, or "pose", the policy is
not what caused it — the spawn pose, gravity, or an assist is. Note that the
`spotter_fn` assist torque in `behaviors/backflip.py` is a **showcase-only** feature and
`render-rollout` never enables it, so anything you see here is the policy plus
the spawn. Check `spawn=` in the header: a `spawn=landed` or `spawn=mid-roll
246°` episode was *handed* most of the trick by the reverse curriculum. To see
whether the policy can do it from scratch, force plain standing starts
(`--env MICRODUCK_SPAWN_FAMILY_PROBS=0.0,0.0`).

### 3. A "hold" may be rapid cycling — check consecutive frames

One sustained pose and a policy flapping in and out of it average to the same
reward.

- Compare **consecutive** captions: a genuine hold shows `trunk_z`, `pitch` and
  the foot contacts nearly constant across frames (`0.114, 0.114, 0.114`).
  Cycling shows them swinging frame to frame.
- The summary prints `reversals (hold-vs-cycling): trunk_z N, pitch M` computed
  over **every** rendered frame, not just the sampled ones — a sustained hold is
  ~0, cycling is many. This catches oscillation faster than the sheet, which can
  alias it.
- If reversals are high but the sheet looks static, re-render with
  `--sheet-frames 24` or a shorter `--seconds` to zoom in on the cycle.
- Diagnostics are sampled at the render stride (~25 Hz at the default
  `--fps 30`), so an oscillation faster than ~12 Hz can alias. Add `--fps 50`
  to sample every control step when you suspect fast chatter.

### Also worth reading

- **Header**: `outcome: FELL (terminated)` vs `completed (truncated)`, plus
  `spawn=…` (which reverse-curriculum family this episode got).
- **Summary**: `trunk_z min/max/final`, `trick rotation max/final`
  (360 = a full flip; the lab hands off at 298), `both feet NN%`,
  `airborne NN%`.
- **Frame #00** is the spawn. If the interesting thing already happened by
  #01, the spawn family did it, not the policy.
- Run **two episodes** (the default) with different seeds before generalizing —
  one lucky rollout is not a result.

## Worked example

> "The backflip battery says rotation 320°, both feet down 91% — it landed,
> right?"

```bash
cd microduck_local
nice -n 10 uv run render-rollout \
    --policy runs/<your-backflip-run>/policy.onnx \
    --episodes 2 --seconds 8 --out /tmp/rr-bf402439
# then: Read /tmp/rr-bf402439/ep0_sheet.png  and  ep1_sheet.png
```

What the sheet actually showed for that run: after the roll, every frame from
t=0.7 s to t=8.0 s sat at `trunk_z=0.088 (stand 0.120)`,
`head_z=0.040 (stand 0.233)`, `floor:jaw_soft`, with the summary reporting
`non-foot body on floor 97%`. Both feet *were* down — while the duck lay on its
beak. "Landed" was wrong; the missing sub-skill was rising from the arrival
crouch, which is what `--handoff ../microduck/policies/alpha_stand.onnx` is for.
Rendering the handoff version showed a genuine 6 s stand at
`trunk_z=0.114 / head_z=0.231`, `floor:none`.

That difference is invisible in the reward sums and obvious in the sheet.

## Notes

- **Match the actuator to the training run.** The env default is the strong
  `xml` phantom actuator; the farm trains under `MICRODUCK_ACTUATOR=bam`
  (restart.sh exports it). A BAM-trained policy rendered without
  `--env MICRODUCK_ACTUATOR=bam` runs on stronger servos than it ever
  trained with — a whole afternoon of renders carried this flattery
  (2026-08-31) before it was caught. Always pass it for lab/teach runs — EXCEPT when the run's own
  curriculum stage declares `MICRODUCK_ACTUATOR` (the headstand ladder's
  stage 1 trains on `xml` training wheels); mirror the stage's env dict
  instead of forcing bam, or the drill stage reads as a failure.

- Offscreen rendering uses `mujoco.Renderer`; on this Mac it picks the bundled
  **CGL** backend (`mujoco.cgl`) with no `MUJOCO_GL` set and no display. The tool
  prints the backend it used. On a Linux box set `MUJOCO_GL=egl` (or `osmesa`).
- The env is built the way training and the lab build it —
  `BehaviorEnv(behavior_id, obs_noise=False, domain_rand=False,
  action_delay=False, random_yaw=False, seed=…)` — so what you see is the
  policy, not the randomizers. That also means it is *not* a robustness test;
  use `uv run eval-walk` for noise/DR survival.
- It renders **behavior** envs (`BehaviorEnv`), so `--behavior` must name a
  behavior from the `behaviors/` package. To look at a plain walking policy, give it a
  behavior whose env is the walking scene (e.g. `--behavior stand`) and read
  the sheet knowing the twist command is pinned to zero.
- Implementation: `microduck_local/src/microduck_local/render_rollout.py`;
  helpers locked by `tests/test_render_rollout.py`.
