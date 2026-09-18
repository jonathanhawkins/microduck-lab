# Mars roadmap — a third body: the Innate MARS (wheeled base + 6-DoF arm)

*Written 2026-09-17. A plan, not a record: nothing below is built except the
Phase 0 measurement. Follow `docs/roadmap.md`'s convention — `[ ]` / `[~]` /
`[x]` with the answer written back into the item.*

The ask: an easy download that puts Innate's
[MARS](https://docs.innate.bot/robots/mars) on the stage beside the ducks and
the G1, with the UX already there (the robot switch, the 🎓 teach panel, the
🧠 palette, `/sim`), and a seam clean enough that the *next* robot is a
directory and a registry entry, not a fourth pass over every `if robot ==
"g1"`. Possible: yes, and cheaper per step than the duck. Different: yes,
and the difference is the whole design — MARS does not walk, so the thing
the harness calls "the policy" (a velocity-command walker) does not exist
for it. What exists instead is an **arm**, and the room-and-brain layer fits
it better than the walking layer ever will.

---

## 0. What MARS is, measured, and what Innate already publishes — DONE (2026-09-17)

**The robot.** A differential-drive base carrying a 5-joint arm with a
parallel-jaw gripper and a pitching head: 7 Dynamixels — XL430-W250-T (joints
1, 3), XC430-T240BB-T (joint 2), **XL330-M288-T (joints 4–6 and the head)** —
the same XL330 family the duck's BAM actuator fit was identified on. 40 cm
reach, 250 g payload, 2 mm repeatability. Sensors: a stereo RGB head camera
(640×480 on the wire, calibrated fx 200.3 / fy 267.3 / cx 319.1 / cy 248.7,
depth 0.4–6 m), a wrist RGB camera (80° vertical FOV in their sim, uncalibrated),
a 360° 2-D LiDAR (0.15–6 m, 6 Hz) on the chassis lid. Compute: a Jetson Orin
Nano Super. Everything is Apache-2.0 in
[innate-inc/innate-os](https://github.com/innate-inc/innate-os).

**What is downloadable.** `ros2_ws/src/mars_bot/mars_description/` holds
`urdf/mars.urdf`, `urdf/arm.srdf` and nine STL meshes — **7.2 MB total**
(base.STL is 5.4 MB of it), pinned here at innate-os
`0ca73670ae20940e381de8b4356ac9855f66e25a` (2026-09-17). The URDF carries
hand-measured collision boxes for the chassis, rear tray, wheels, arm mount,
LiDAR turret and neck, and placeholder inertias (every link `ixx=iyy=izz=0.001`;
the masses sum to 1.365 kg, which is the printed parts and not the battery or
the Jetson — see *Risks*). There is no shipped policy file to fetch: Innate's
learned skills are ACT checkpoints trained from teleop demonstrations on their
cloud, not ONNX, and the repo ships none.

**Innate's own MuJoCo recipe** (`mars_sim_driver/world.py`, `core.py` —
their simulator is MuJoCo 3 headless, the same major this repo pins) is the
one to mirror, not reinvent:

- load the URDF with `MjSpec.from_string` after two rewrites — `package://`
  → a directory, and `<mujoco><compiler discardvisual="false"/></mujoco>`
  injected so the visual STLs survive the import;
- a **planar base**: `base_x`, `base_y` slides and a `base_yaw` hinge on
  `base_link` ("a wheeled chassis can't pitch, and a free joint lets the arm's
  reaction torque tip the 0.89 kg base over") — the wheels are collision
  cylinders made **frictionless** (`condim 1, priority 1`) because the planar
  constraint pins z and a tangent wheel otherwise answers every step with ~50 N
  of spurious normal force;
- the base is driven by a velocity PD through `xfrc_applied` (KP forward 200,
  lateral 40, yaw 3) with a station-keeping hold after 0.4 s of quiet, and a
  0.5 s `cmd_vel` watchdog; commands clamp at 0.8 m/s and 2.0 rad/s (their
  real-robot defaults ×2, the "Mad" mode);
- arm/head **position servos** KP 50 / KD 1, torque clamp 50 N·m (the
  gripper 2 N·m — 50 on a 45 mm finger was 1.1 kN of pinch), a structural-sag
  model (25 N·m/rad per joint + 0.055 rad backlash) so the link settles below
  the encoder, `joint6M` mirroring `joint6` at −1, finger contact tuned for
  grasping (condim 6, friction 2.0/0.05/0.02, armature 1e-4, damping 1.0), and
  a joint2 floor re-clamped from joint1 so the arm ducks under the head;
- `ARM_HOME` = joint1 1.445, joint2 −1.388, joint3 1.517, joint4 0.446,
  joint5 −0.089, joint6 0.0015, head 0.

**Phase 0 measurement — it loads and it is cheap.** `scripts/probe_mars.py`
(in `microduck_local/`) applies that recipe and then the lab's own pattern —
`MjSpec.attach(robot, prefix="m0/", frame=...)` into a floor world, which is
how `world/compose.py` puts N ducks in one model:

```
nq=11 nv=11 nbody=18 ngeom=59 nmesh=9      # 3 planar + 5 arm + 2 finger + head
total mass 1.365 kg
collision AABB extent (x,y,z) ~ [0.668 0.311 0.415]   # rbound-based, conservative
after 2 s at ARM_HOME: max |q - home| = 0.0034 rad      # innate's PD, python-side
mars: 92,700 physics steps/s   (single env, 2 ms step)
duck: 72,700 physics steps/s   (nv=20, same loop)
```

So one MARS costs **0.78× a duck** per physics step (the G1 is ~2×). The
attach-under-prefix works on 3.10 with no edits to the URDF beyond the two
rewrites. That is the whole feasibility question answered; everything below
is design and labour.

---

## 1. The design decision: MARS is not a walker, so do not make it one

Every robot in this harness so far entered through `robots/spec.py`'s
`RobotSpec`, which is honestly named in its own docstring: "what the **walking
env** needs to know about a robot". Its fields are foot geoms, fall thresholds,
air-time windows, twist-command ranges, a gyro sensor. MARS has none of these
and a wheeled base cannot fall. Threading it through `MicroduckWalkEnv` would
be the architecture-scale version of the mistake `microduck_local/AGENTS.md`
warns about most — zeroing inherited terms because their default target is
wrong for the new body. The clean cut is:

**Split the lab's contract from the walker's.** Today a single dataclass
carries both. Introduce `robots/body.py`:

```python
class Body(Protocol):                 # what the LAB, the VIEWER and /sim need
    id: str; title: str; noun: str    # "mars", "Innate MARS", "MARS"
    kind: str                         # "legged" | "wheeled"
    joint_names: tuple[str, ...]; joint_groups; default_pose
    obs_dim: int; num_actions: int    # the policy contract this body speaks
    lab_spacing_m: float
    def ready(self) -> bool           # assets on disk?
    def fetch(self) -> Path           # the one-click download (POST /robots/{id}/fetch)
    def setup_hint(self) -> str       # "uv run fetch-robot mars"
    def visual_scene(self) -> dict    # GET /scene?robot=<id>: meshes for the viewer
    def look(self) -> str             # the viewer's material set: "duck" | "g1" | "mars"
    def env_class(self, task) -> type # the trainer's env for a task ("walk", "reach", ...)
    def tasks(self) -> tuple[Behavior, ...]   # what the 🎓 panel offers
    def shipped_policies(self) -> tuple[dict, ...]
    def attach(self, spec, prefix, frame)     # put one in a /sim world
    def driver(self, model, prefix) -> Driver # step it in that world (G1Walker today)
```

`RobotSpec` becomes a `Body` **plus** the walker fields — the duck and the
G1 keep every byte of behaviour (the golden-bit tests are the proof, as they
were for the G1). `MarsSpec` is a `Body` with an `ArmSpec` instead. A
`robots/registry.py` maps id → Body and replaces the literal lists. Count of
what that literal list costs today, so the refactor has a number:

| where | `"g1"` literals | what they decide |
|---|---|---|
| `viz_server.py` | 15 | available robots, noun, `/scene?robot`, `/robots/{id}/fetch`, shipped-policy group, trick lookup, imitation recipe, teach suggestions |
| `train.py` | 3 | `env_class` if-chain, `--robot` choices |
| `world/*.py`, `world_server.py`, `render_rollout.py` | 6 | the person kind, compose attach, arena driver, `--robot` choices |
| `duck-viewer` (`lib/lab.ts`, `lib/sim.ts`, `Duck.tsx`, `Viewer.tsx`, `PolicyPanel.tsx`, `SimStage.tsx`, `SimViewer.tsx`) | 21 | the `RobotId` union, the palette group, the `look`, the /sim person branch |

**Settles Phase 1:** those 45 become ~6 (the registry module, `RobotId`
widened to `string` with a per-robot `look` map, the G1 person kind kept for
old scenarios), every existing test green, goldens bit-identical.

**Vocabulary, so the two projects do not talk past each other.** In Innate's
stack an *agent* is a prompt plus skills (an LLM picks), a *skill* is Python
or a trained ACT policy. In this lab a *brain* is a controller over senses
that emits intents, and a *policy* is the reflex tier. The mapping that
holds: a lab **brain** would deploy as one Innate **code skill** (Python,
runs onnxruntime, streams `/cmd_vel` and `/mars/arm/commands`); an Innate
**agent** sits above both. A lab-trained arm **policy** is not an Innate
*learned skill* (those are ACT from demonstrations, 25 Hz, both cameras in)
— it is a code skill that happens to run a network. Say so on the run.

---

## 2. Phases, each with the command and the number that settles it

### Phase 1 — the seam: `Body` + registry, no behaviour change  `[ ]`

- Add `robots/body.py`, `robots/registry.py`; make `RobotSpec` a `Body`;
  move the G1's fetch/ready/scene/look/tasks/attach/driver behind it (they all
  exist, scattered across `g1.py`, `fetch_g1.py`, `viz_server.py`,
  `compose.py`, `arena.py`).
- `viz_server`: `available_robots()`, `robot_noun()`, `GET /scene`,
  `POST /robots/{id}/fetch`, `GET /robots`, `discover_policies()`'s shipped
  group and `run_trick()` read the registry. `train.env_class` →
  `registry.get(robot).env_class(task)`. `--robot` choices → `registry.ids()`.
- Viewer: `RobotId` → `string`; a `ROBOT_LOOKS: Record<string, Look>` beside
  `G1Look.tsx`; `PolicyPanel` builds the shipped group per robot from
  `GET /policies.robots`; `fetchRobot(id)` already takes an id.
- Generalise `fetch-g1` to `uv run fetch-robot <id>` (keep `fetch-g1` as an
  alias — the README and memory both name it).
- **Settles:** `grep -c '"g1"'` over the files in the table above ≤ 6;
  `uv run --with pytest pytest tests/` green including the goldens;
  `npm test` green; `tests/test_lab_robots.py` extended with a fake third
  registry entry so the seam is proven by something that is not the G1.

### Phase 2 — MARS on the stage: download, spec, look  `[ ]`

- `robots/mars.py`: `MarsSpec` — `CACHE_DIR = .cache/innate_mars/`,
  `INNATE_OS_SHA` pinned, joint names in the order Innate's `/mars/arm/state`
  reports them (`joint1..joint6`, `joint_head`; `joint6M` is the mimic and is
  not a policy joint), `ARM_HOME` as `default_pose`, `lab_spacing_m`
  measured off the AABB the way `g1.BODY_WIDTH_M` was (0.311 m wide at home →
  the duck's 3.52× gives **1.09 m**; the 0.668 m *length* with the arm folded
  is what actually neighbours a slot, so measure both and pick the larger);
  `noun = "MARS"`; `kind = "wheeled"`.
- `fetch-robot mars`: nine raw-file downloads + the URDF + the SRDF at the
  pinned sha with sha256 checks, not a clone (the repo is 185 MB; the
  directory is 7.2 MB). Writes `.cache/innate_mars/mars_description/`. The
  palette's existing "not set up yet — ⤓" affordance does the rest via the
  generalised endpoint.
- `mars.model_spec()`: Innate's recipe verbatim (the two rewrites, planar
  base, wheel condim, finger contact, mimic range, exclude) as a function
  that returns an `MjSpec` ready to attach — one implementation for the lab
  slot, the `/sim` world and the training env, like `g1.g1_spec()`.
- `mars.visual_scene()`: the same mm-int mesh dump `g1.extract_visual_scene`
  does; the viewer's `MarsLook` paints links 1/3/5 Innate orange and the base
  charcoal (their `style_robot_geoms`), hides the frame-marker links.
- A lab slot with a MARS and no policy sits at `ARM_HOME` on the stage;
  the ▶ chips of duck/G1 policies are refused on it by obs width (10-ish ≠
  61 ≠ 99 — `test_lab_robots` already pins the mechanism).
- **Settles:** from a fresh checkout, `uv run fetch-robot mars` lands in
  **< 30 s** on home broadband; `GET /scene?robot=mars` serves 9 meshes; a
  roster of `duck, g1, mars` renders with no overlap (`tests/test_lab_robots.py`
  re-measures the width like it does for the G1); a screenshot in
  `docs/media/lab-mars.png` (`.claude/skills/capture-viewer-screenshots`).

### Phase 3 — MARS in a room: drive it, sense with it, give it the existing brains  `[ ]`

This is where MARS earns its place fastest, because the room-and-brain
layer was built around a twist-emitting brain over a reflex tier, and for a
wheeled base the reflex tier *is* the base controller — no policy to train
before anything moves.

- `world/scenario.py`: a first-class `robot` field on `Duck` (default
  `"microduck"`) rather than a second `Person.kind` hack; `compose()` calls
  `registry.get(robot).attach(...)`; `arena.WorldDuck` takes a `Driver` from
  the registry — the duck's is the ONNX walker + command block, the MARS's is
  `robots/mars_drive.py`: Innate's base PD + station keeping + watchdog +
  arm/head servos + sag model, ported line for line with their constants
  named, so a drive that is wrong here is wrong there too.
- Senses: `Senses.lidar` — a `sensors/ray.planar_fan(360, 360°)` at the
  `base_laser` link, 6 Hz, 0.15–6 m, with its own noise preset; the detector
  gets a MARS lens (`DetectorSpec` from the calibrated fx/fy/cx/cy — the lab
  already learned what an uncalibrated lens does to bearings). `Senses.odom`
  exists; `holding` reads gripper load.
- Brains: `wander` and `follow` emit twists and work on day one; their
  ToF-based avoidance reads `lidar` through a small adapter (64 zones ←
  nearest bins in the ToF's FOV, so nothing in the controllers changes). The
  `/sim` WASD drive, the possess-a-person loop, `record-world`, the ring
  replay all come free.
- **Settles:** `uv run record-world playroom --robot d0=mars --brain d0=wander
  --seconds 60`: **0 wall contacts** on 3 seeds (events.txt), sheet shows the
  arm parked at home; `follow-me` with a MARS follower holds the follow band
  at the duck's measured level (**≥ 0.95 in-band**, `eval-brain`); a
  `docs/media/sim-mars-wander.gif`.

### Phase 4 — arm policies: `MarsArmEnv`, `reach` → `pick`, the teach panel, ONNX  `[ ]`

The "train it, make new policies" half. The policy is the arm.

- **The MARS obs/action contract** (`robots/mars.py`, documented like
  `contract.py`; one fixed layout, zero-pad what a task does not use):
  obs = `[arm qpos(6), arm qvel(6), head(1), gripper load(1), last action(8),
  target in base frame(3), target seen(1), reserved]`; actions = `[6 joint
  targets, vx, wz]` at **25 Hz** (decimation 20 on a 2 ms step) — the rate
  Innate's policy-defined skills run at, so a lab policy and a real skill
  tick the same clock. The base pair is zero for arm-only tasks.
- **Actuators.** Joints 4–6 and the head are XL330s: `bam_actuator` applies
  as-is (its identification is that servo). Joints 1–3 are XL430/XC430 with
  no BAM fit here → Innate's PD + sag model as the `"xml"` path, and a note
  on the run that the shoulder is a proxy. Do not pretend the BAM fit covers
  the shoulder.
- `MarsArmEnv` (gymnasium, its own base class — not `MicroduckWalkEnv`),
  tasks as `Behavior`s in `behaviors/mars_tasks.py` so the 🎓 panel lists them
  under a MARS roster the way `g1_tasks.py` does: `reach` (end-effector to a
  sampled point in the 40 cm shell, ±30° of the front arc), then `pick` (a
  4 cm block on the floor in front, gripper closes on it, lift 5 cm), then
  `place` (into the playroom basket). Rewards follow the playbook: distance
  progress paid, not distance held; termination on self-collision; the
  physics ladder (block spawn box widens per rung) before any reward
  surgery.
- `export-walk runs/<mars-run>` already reads the run's robot; the exporter
  bakes the normaliser and writes `obs[1,N] → actions[1,8]`. `render-rollout
  --robot mars` renders it. `describe-run --pick` names the rung.
- **Settles:** `reach` ≤ **2 cm** final error on **8/8** seeds within 10 s;
  `pick` ≥ **80 %** lifts over 20 episodes on the widest rung, deterministic
  ONNX, rendered and looked at (`render-rollout`), before any number is
  quoted.

### Phase 5 — a MARS brain: tidy with an arm, and `train-brain --robot mars`  `[ ]`

- `Tidy` today is beak-shaped (13 beak/mouth/ground_pick references). A
  `TidyArm` brain keeps its state machine (search → approach → pick → carry →
  release) and swaps the `skill="ground_pick"` cycle for the Phase 4 `pick`
  policy plus `Intent.arm` (a joint target block, alongside `twist` and
  `head`); the `eval-tidy` harness measures toys/5 min unchanged.
- `brain_env`: the 80-float obs has 64 ToF zones at `[0:64]`; a MARS body
  needs a lidar layout → `obs_version 3` with a per-body `ObsBuilder`, the
  version written into `brain.json` as it already is, so `learned:<name>`
  brains stay self-describing. `train-brain --robot mars` follows.
- **Settles:** `eval-tidy --robot mars --seeds 3 --seconds 300` reports
  toys/5 min; the duck's 0.83 is the yardstick. A learned follow brain on
  MARS reaches the scripted `follow`'s in-band level, as it did on the duck.

### Phase 6 — the way out: a skill template, and the docs  `[ ]`

- `microduck_local/deploy/mars_skill_template.py`: an Innate code skill
  (`Skill` subclass) that loads `policy.onnx`, reads `/mars/arm/state` and the
  head detector, and streams `/mars/arm/commands` + `/cmd_vel` at 25 Hz.
  **Untested on hardware — nobody here has a MARS.** Ship it as a template
  with that sentence at the top, and verify it against Innate's `VirtualMars`
  (their `sim/` venv, no ROS) as the closest thing to a golden test.
- README: "A third robot: the Innate MARS" beside the G1 section; the
  command crib in `AGENTS.md`; `scripts/setup.sh` gains an optional
  `--with-mars`; this file gets its answers written back.

---

## 3. Order and cost

Phase 1 is the one that pays for every later robot and it is pure refactor
against green tests — do it first and alone, in its own commit series (the
shared-checkout rule: commit shared files whole, check the index). Phases 2
and 3 are each a day; Phase 3 delivers the first thing worth a GIF. Phase 4
is the RL work and the only phase with real uncertainty (grasping in MuJoCo
with a mimic gripper — Innate's finger contact tuning is a head start, and
the `pick` ladder is a physics ladder, per the playbook). Phase 5 rides on 4.
Phase 6 is documentation plus one file.

The per-step cost story is the opposite of the G1's: at 0.78× a duck, a
32-env `MarsArmEnv` run should sit near `train-behavior`'s throughput, so
`bench-envs --robot mars` will say the right `--envs` and there is no
distil-first constraint.

## 4. Risks and honesty

- **No hardware here.** Everything above is a lab contract. The deploy
  template is the one artefact aimed at the robot and it is untested by
  construction; say so on it and on every MARS run's `record.json`.
- **The URDF's dynamics are placeholders.** 0.001 inertias everywhere, 1.37
  kg total, no battery/Jetson mass. Fine for a velocity-driven planar base
  (Innate drives it the same way), weak for anything that depends on the
  base's reaction to the arm. If a task needs it, measure a real MARS's mass
  first; do not tune the numbers to look right.
- **Shoulder servos are a proxy** (XL430/XC430 have no BAM fit); the XL330
  wrist and gripper do. Write which joints are which on the run.
- **Their apartment assets are not ours to bundle.** `innate-sim-assets`
  tarballs are 130–156 MB, content-addressed, with third-party meshes under
  `sim/ATTRIBUTION.md` and no licence field on the repo. Use the lab's own
  rooms; an optional "load Innate's apartment" is a later item, licence
  read first.
- **Vocabulary drift.** Their agent/skill vs our brain/policy — §1 fixes the
  mapping; use it in the UI copy ("teach MARS a task", not "a trick").

## 5. Open decisions (the user's, not the code's)

1. **Emoji and noun.** "🛸 MARS" / "teach MARS a task"? The robot switch and
   every sentence in the panel take the noun from the spec.
2. **Which task first in the 🎓 panel** — `reach` is the honest starting rung;
   `pick` is the demo people want. Both are listed; the palette's ⭐ goes to
   whichever a measurement earns.
3. **Whether the `/sim` MARS carries the duck's detector classes** (toy,
   basket, ball, person) from day one. Yes by default — it is what makes the
   existing brains work — with the inspector labelling them simulated, as it
   does for the duck's non-shipped classes.

---

## 6. Architecture cleanup for a global pipeline — what a body touches today, and what to change (2026-09-17)

Asked: what should be cleaned up so bringing in *any* robot is easy, not just
the third one. Measured against the tree as it stands (a `Body` is the term
from §1: the lab's contract with a robot, walker or not). Ranked by leverage;
the first four are the ones to do before MARS, because MARS would otherwise
be built on the same forks the G1 was.

### 6.1 Two task systems — the duck's and everyone else's  (do first)

The duck learns tricks from a **reward recipe**: `behaviors/core.py`'s
`Behavior` (30 fields — terms, curriculum, spawn families, handoff, clips)
evaluated inside `BehaviorEnv(MicroduckWalkEnv)` by `train-behavior`, which
has no `--robot` flag. The G1 learns tasks from **env subclasses**
(`robots/g1_env.py`, `g1_karate.py`, `g1_imitate.py`) run by
`train-walk --robot g1 --task …`, and its `Behavior` entries carry
display-only terms (`_env_owned`) plus a `trainer=(...)` tuple naming the
CLI to launch. Two authoring paths, two trainers, two ways a curriculum
stage is applied. A MARS task would be a third unless the fork is closed.

**Fix:** `BehaviorEnv` wraps *the body's* base env (`Body.env_class("base")`)
instead of subclassing the duck's; `train-behavior --robot <id>` is the one
teach trainer; a `Behavior` declares `robot` (it already does) and its terms
are callables on the env (they already are). `RewardTerm` grows a `bodies`
tag so the ＋ term picker only offers what applies (arm terms to an arm,
foot terms to feet). The G1's env-owned rewards become terms — the G1 idle's
ten rows are already listed as `RewardTerm`s, they just do not compute.
**Settles:** one trainer name in every `Behavior.trainer` (or the field
gone), `train_behavior.py` has `--robot`, and the G1 idle retrains to its
recorded level through the recipe path.

### 6.2 A conformance suite: define "supported" as a test  (do first)

112 test files; the robot tests are per-robot (`test_g1_*`, `needs_g1`
skips) and only `test_robot_spec.py` and `test_pose.py` iterate the
registry. Nothing says what a body must satisfy to be listed.

**Fix:** `tests/test_body_conformance.py`, parameterised over
`registry()`, skipping bodies whose assets are not fetched: compiles; attaches
under a prefix beside a duck in one model; spawns at `default_pose` and
settles 2 s without a fall, a NaN or a contact explosion; obs width and
action width match the spec; a random-weight policy exports to ONNX and
round-trips within 1e-5; `visual_scene()` serves `nmesh` meshes with a
material kind per geom; `lab_spacing_m` is re-measured from the AABB (the
G1 test does this by hand); `mirror_joint_perm()` is a permutation; every
declared sensor site and effector body resolves by name. **Settles:** a new
body is "supported" when this file is green for it, and the README says so.

**DONE (2026-09-17)** — `tests/test_body_conformance.py`, 38 cases (19 a
body) in 2.7 s, every positive shown to fail on a planted break (27/27
plants caught). Both specs pass clean: every name resolves, both mirrors
are involutions, both pitches are within 0.1 % of 3.52× the re-measured
width, ONNX round-trips at 1.2e-8. Two measurements changed the test as
written above: **neither body settles open-loop** (the duck falls at 0.98 s
on xml servos, 1.62 s under BAM; the G1 at 1.22 s), so the 2 s hold runs the
body's shipped idle (`alpha_stand`, `walker.onnx`) and the open-loop topple
is the planted negative — a wheeled body will need a `kind` guard there;
and the duck env's `_get_obs` writes a literal 61 while `observation_space`
is spec-sized, so a wrong `obs_dim` on the duck disagrees silently where the
G1's raises — the width case checks all three numbers. Four seams are faked
by id tables until 6.4 lands: `ready()`, `setup_hint()`, which shipped
policy is the idle, and `visual_scene()`.

### 6.3 Self-describing policies: a contract id, not a width  (do first)

`run.json` records `robot`; `export_onnx.run_robot()` reads it; the lab
refuses a policy on the wrong body by **observation width** (61 vs 99 —
`viz_server` guards at four sites). Width is a proxy: two bodies with the
same width would cross silently, and an ONNX handed around without its run
directory says nothing about itself. The brain layer already got this right
(`brain.json` carries `obs_version`).

**Fix:** a `Contract` record on the body — `{id: "microduck-61-v1",
obs_dim, act_dim, slots: [...], rate_hz}` — written into `run.json`,
`record.json` and the ONNX file's `metadata_props` at export, and read back
by the palette, `render-rollout` and the world (`policy_robot()` becomes
`policy_contract()`). The duck's is the deployment contract by name; the
G1's says "lab"; MARS's says "code skill". **Settles:** the four width
guards become one contract comparison, and `eval-walk some.onnx` with no
run directory still knows its body.

### 6.4 The registry, and the lab file it lives in  (do first — §1 Phase 1)

`viz_server.py` is 4,540 lines: the 50 Hz loop, the socket, HTTP, teach
jobs, captures, policy discovery, run management *and* 15 robot branches.
The registry from §1 is the fix for the branches; the extraction that makes
it stick is `lab/robots.py` (available robots, fetch, scene, shipped
policies, nouns, teach suggestions) so the next body never opens
`viz_server.py`. The viewer's half: one `lib/robots.ts` map (`id → emoji,
look, chip label`) and a per-geom material `kind` sent by the server, so
`Duck.tsx` maps kind → material and no robot needs its own component
(`G1Look.tsx` becomes the material table, not a special case).

### 6.5 World mode has one robot slot and one hack  (Phase 3)

`WorldDuck` (`world/arena.py`, 1,611 lines) is sized by `C.NUM_JOINTS`,
`head_cmd(4)`, `body_cmd(6)` — the 61-obs walker — and a G1 in a room is a
`Person` with `kind == "g1"` stepped by `G1Walker`. Sensors mount by the
duck's site names (`"tof"`, `"head_camera"`) hard-coded in `arena.py`;
`Intent.head` is four duck neck joints, `Intent.beak` is the mouth servo,
`Senses.tof` is 64 zones, and `brain_env`'s observation bakes those 64 at
`[0:64]`.

**Fix:** `WorldRobot` takes a `Driver` from the body (the duck's = ONNX
walker + command block; the G1's = `G1Walker`; MARS's = the base PD + arm
servos); the body declares its **sense channels** (`tof`, `camera`, `lidar`
with mount sites) and **intent channels** (`twist`, `head`, `beak`, `arm`),
and the brain observation builder is chosen by `obs_version` per body as
`learned.py` already selects by version. `Person.kind == "g1"` stays for old
scenarios and stops being the way a second body enters a room.

### 6.6 Per-joint servo models, not one string  (Phase 4)

`actuator="xml"|"bam"` is one switch for the whole body; BAM is the XL330
identification and the G1 env raises on it. MARS mixes XL430, XC430 and
XL330. **Fix:** `Body.servo_models` in joint order (`"xl330-bam"`,
`"xml"`, later `"xl430-…"`), the actuator layer applying each per DoF.
The duck's table is all XL330; the G1's all `"xml"`; nothing changes for
either.

### 6.7 One asset story  (Phase 2)

Three conventions today: the duck's MJCF from the upstream checkout via
`MICRODUCK_RL_DIR`; the G1 from `.cache/unitree_g1` via `fetch-g1` and
`MICRODUCK_G1_DIR`; the shipped policies from `../microduck/policies` via
`setup.sh`'s Hub download. **Fix:** `Body.assets` — a manifest (source,
pinned revision, files, sha256) — behind `fetch-robot <id>`, `.cache/<id>/`
and `ready()`; `setup.sh` iterates the registry; `POST /robots/{id}/fetch`
is the same call. The duck keeps its upstream-checkout source (the goldens
depend on the pinned sha), expressed in the same manifest form.

### 6.8 Smaller, and worth doing as the seams are touched

- **CLI:** `--robot` has `choices=("microduck", "g1")` written by hand in
  six commands; take the choices from the registry, and make every command
  that takes a run directory read the robot from `run.json` (the exporter
  already does), so `--robot` is only for commands that start from nothing.
- **Env-var knobs:** 112 `MICRODUCK_*` variables, ten of them `MICRODUCK_G1_*`.
  New per-body knobs go in the `Behavior`'s curriculum `env` dict and the
  run's `env_kwargs`, both of which already exist and are recorded, not in
  new variables.
- **Naming:** the generic slot is called a duck everywhere — `Duck` in
  `viz_server`, `WorldDuck`, `ducks: DuckFrame[]`, `Duck.tsx`, `spawn_duck`,
  `duck_prefix`. Do not mass-rename (the shared-checkout rule, and a rename
  commit scrambles history); rename at the seam being touched
  (`WorldDuck → WorldRobot` with 6.5, `Duck.tsx → RobotBody.tsx` with 6.4)
  and leave the rest.
- **Deployment honesty as data:** `Body.deploy` — one sentence per body on
  whether an export is drop-in (duck), a lab contract (G1) or a code skill
  (MARS) — printed by `export-walk` and written onto the run, so the caveat
  travels with the file instead of living in a README section.

### 6.9 Order

6.4 (registry) → 6.2 (conformance, written against duck + G1 first, so it is
proven by two bodies before a third arrives) → 6.3 (contract ids) → 6.1
(one task system). Then MARS Phase 2 onward picks up 6.7, 6.5, 6.6 as it
goes. Each step is a no-behaviour-change refactor against the goldens and
`tests/`, in its own commit series, shared files committed whole.

---

## 7. Generalising for wider use — where it pays, and where it is a trap (2026-09-17)

Asked: can any of this be made generic enough that other people bring their
own robots? Yes, and the cheapest test of "generic" is already available:
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) holds
**71 robot models** — humanoids, quadrupeds, arms, hands, mobile manipulators
— each with a `home` keyframe, under per-model licences (Apache/BSD; check
each model's `LICENSE`). If a Menagerie body reaches the stage with zero
per-robot code, the seam is generic. If it needs a file of its own, it is not.

### 7.1 A ladder: what you get for what you declare

Generic "where it makes sense" means a body earns features by declaring
things, and nothing is required that the feature does not need.

| level | you declare | you get |
|---|---|---|
| **0 — on the stage** | an MJCF or URDF and an id (`MjcfBody` reads joint names, the `home` keyframe as the default pose, actuators, geoms) | the palette chip, the meshes, a slot in the lab, the 🎬 animate panel's joint sliders, a kinematic body in a `/sim` room, `record-world` |
| **1 — it trains** | a `kind` (`legged` / `wheeled` / `arm` / `fixed`) and the few names that kind's base env needs (feet + base + gyro; an effector; base joints) | the conformance suite, the kind's env, kind-tagged recipes in the 🎓 panel, `export-walk`, `render-rollout` |
| **2 — it lives in a room** | a driver (an ONNX with a contract id, or a scripted controller) and its sense/intent channels | brains (`wander`, `follow`, learned), scenarios, `eval-*` batteries, the `/sim` inspector |
| **3 — honest about hardware** | servo models per joint, measured sensor specs, a deploy sentence | the sim2real caveats travel with the run and the ONNX |

The duck is level 3 and stays the reference; the G1 is 2; MARS enters at 2;
a Menagerie body arrives at 0 and climbs only when someone needs it to.
**Settles:** `uv run fetch-robot menagerie:unitree_go2` puts a Go2 on the
stage, posable, with **0 lines** of Go2-specific code in this repo.

### 7.2 Where generic pays

1. **A body plugin is a directory, and a Python entry point.**
   `robots/<id>/{body.py, assets.json, look.json, tasks.py}`, and an
   entry-point group so `pip install someone-elses-robot` registers a body
   without forking this repo. The standard plugin pattern; the registry from
   §6.4 is the only thing it needs. **Settles:** a test package in `tests/`
   registers a fake body through the entry point and the palette lists it.
2. **Standards over invention.** MJCF/URDF through `MjSpec` (already),
   gymnasium and SB3 (already), ONNX carrying its contract id in
   `metadata_props` (§6.3), the 🤗 Hub upload that exists today. With the
   contract id in the file, a policy someone downloads from the Hub lands on
   the right body in someone else's lab.
3. **Recipes as portable data.** A `Behavior` is nearly data now
   (plain-English terms, curriculum, spawns). With body-tagged terms (§6.1),
   "stand still" runs on any legged body and "reach" on any arm — with the
   rule the G1 taught: the *structure* ports, the *windows* do not (the
   duck's air-time window paid on 0 % of G1 swings until re-measured). A
   recipe carries its measured constants per body, or a test fails.
4. **Brains are generic by design and not yet in code.** Senses in, intents
   out; the channel declaration from §6.5 makes `follow` run on anything
   with a twist channel and a camera.
5. **The verification tooling is the most reusable thing here** and has
   nothing duck-shaped in it: contact sheets with burned-in diagnostics,
   `record-world` with an events log, eval batteries that print the
   minimum detectable effect, the null-control rule. Anyone training in
   MuJoCo lacks this. It could stand alone as a package before anything
   else does.
6. **A package boundary, enforced, instead of a rename.** `core` (env base,
   trainer, exporter, lab, world, brains, tooling) must never import `duck`
   (the 61-obs contract, BAM, colorways, the beak, tricks, soccer, tidy).
   **Settles:** an import-direction test (`grep`-level: no `from ..contract`
   in core modules except through the registry). Renaming the package or
   the product is a separate, optional decision once the boundary holds.

### 7.3 Where generic is a trap

- **One physics engine.** Everything rests on `MjSpec.attach`; Isaac,
  Genesis or PyBullet backends would be a second harness, not a feature.
- **"Any robot walks out of the box"** is not a promise this stack can make.
  Level 1 is "the kind's env runs"; every reward window is measured per body.
- **The duck stays the duck.** The deployment contract, BAM, the beak,
  colorways, soccer and tidy are the reference plugin, not abstractions to
  generalise. Abstracting them would hollow out the one sim2real-honest path.
- **No agent layer.** Innate's LLM-picks-a-skill loop is theirs; this lab's
  brains stop at controllers over senses.
- **No mass rename now.** The shared checkout and the history rules make a
  rename commit the most expensive kind; rename at the seams as they change.

### 7.4 What to build for it, in order

1. `MjcfBody` + `fetch-robot menagerie:<name>` (level 0, after §6.4).
2. Entry-point registration and the fake-plugin test.
3. Body-tagged terms and one recipe proven on two legged bodies.
4. The verification tooling's own README, so it can be pointed at without
   the rest.
