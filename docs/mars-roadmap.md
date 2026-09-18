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

### Phase 1 — the seam: `Body` + registry, no behaviour change  `[~]`

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

**1a DONE (2026-09-17)** — `robots/body.py` (`Body` protocol + `BodyBase`,
`conforms()` because `isinstance` cannot see through the G1's lazy proxy),
`robots/registry.py` (built-ins declared without importing, a
`microduck_local.bodies` entry-point group for pip-installed bodies,
`register()` for tests, `ids()` lists a known-but-unfetched body so
`--robot g1` still answers with the fetch command), `robots/microduck.py`
(`MicroduckBody(RobotSpec)`: every duck-asset answer lives here, not on the
generic walker — a first draft put them on `RobotSpec` behind an id guard,
which is the shape of the mistake the split exists to prevent), `G1Body` at
module level in `robots/g1.py`, `fetch-robot <id>` with `fetch-g1` kept as an
alias, and `train`/`export`/`bench`/`distill`/`eval-onnx` reading choices
from the registry: `"g1"` literals in those five files 7 → 0. Behaviour
preserved to the bit: 200-step rollout fingerprints identical to HEAD for
duck xml, duck bam and G1 xml; every `MICRODUCK` and `G1_SPEC` field
unchanged (only `noun`, `kind` added). 33 registry tests, each shown to fail
on a planted break; the conformance suite unchanged and green. Still open
for 1b: `viz_server.py` (15 literals), `render_rollout.py`, the viewer — and
`MicroduckBody.visual_scene()/shipped_policies()` import from `viz_server`
in the wrong direction until `lab/robots.py` exists (marked `PHASE 1B:`).

### Phase 2 — MARS on the stage: download, spec, look  `[~]`

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

**2a DONE (2026-09-17)** — the BACKEND: `robots/mars.py` (`MarsBody(BodyBase)`
— NOT a `RobotSpec`, so no feet, no gyro, no fall height), registered as
`mars`, `fetch-robot mars`, Innate's recipe as `robot_spec()`, the scene and
`HOME` keyframe as `scene_xml()`, their arm/head PD as `arm_servo()`, the
viewer's dump as `visual_scene()`. The lab server and the viewer are NOT in
it (Phase 2b). `tests/test_body_conformance.py` is split by kind — generic /
`RobotSpec` / `kind == "wheeled"` — plus `tests/test_mars.py`: 36 + 59 cases,
3.7 s, and all 24 planted breaks caught. Measured:

- **`fetch-robot mars` takes 8.0 s** cold on home broadband (11 files, 7.2 MB,
  sha256 each) and **0.28 s** warm, which answers the < 30 s bar.
- **`lab_spacing_m` is 1.46 m, not the planned 1.09 m.** The AABB at HOME is
  0.4135 x 0.3665 x 0.4483 m: the arm folds over the chassis, so the body is
  LONGER than it is wide and the plan's estimate (the y extent at qpos0)
  would have parked two MARSes inside each other's rear trays. 3.52x the
  larger extent, as for the G1.
- **One more URDF rewrite than Innate's two: `fusestatic="false"`.** The
  importer defaults it on, which merges 8 jointless frames away — and then
  `robot_spec().compile()` alone gives 10 bodies / 1.3650 kg while attaching
  that already-compiled spec gives 10 bodies / **1.3340 kg**, silently
  dropping the marker links' mass. Two compiles that disagree on the body
  list would draw a robot with its parts on the wrong joints (the lab indexes
  poses positionally), and `base_laser` / `head_camera_left` / `ee_link` —
  Phase 3's senses and Phase 4's reward — have no name once fused.
- **The hold: max |q − home| 0.00344 rad** over 2 s at HOME under
  `arm_servo`, base z drift 0.0 mm, planar drift 1e-5 m / 1.4e-4 rad, ncon ≤ 6,
  identical at 5 ms (the lab's timestep, what the scene uses) and at Innate's
  2 ms — so their finger tuning survives the coarser step.
- **134,700 physics steps/s** bare, 80,000 with the servo in python, against
  the duck's own scene at 70,400: **0.52x a duck per step**, cheaper than the
  0.78x the probe suggested (that number had the python servo in the loop).
- `nu == 0` and it is correct: mars.urdf has no `<actuator>` block, because
  Innate's driver commands positions through `qfrc_applied`.
- The obs contract is declared but unfilled: **32 floats** (28 used + 4
  reserved) and **8 actions** (6 joint targets + `vx, wz`) at 25 Hz.
- Two constants that a first draft of the tests could not see: comparing the
  model to the module's own constant is a tautology, so the ported numbers
  are now pinned against literals (`test_mars.INNATE_CONSTANTS`) — a planted
  `FINGER_ARMATURE = 0.0` and `GRIPPER_CLOSED_ON_AIR_RAD = 0.0` both passed
  the first version.

Still open for 2b: `viz_server`'s 15 robot literals and `lab/robots.py`, the
viewer's material table for `look() == "mars"` (the colours already travel in
the dump), the roster screenshot, and `scripts/setup.sh --with-mars`. Two of
those literals are now WRONG rather than merely duplicated, measured with a
third body in the registry:

- `GET /robots` lists every registry entry for the 🎬 editor, and answers
  `"ready": True if rid == "microduck" else bool(g1_ready())` — so MARS
  inherits the G1's download state. `Body.ready()` is the fix and it exists.
- worse, that list is what the editor offers, and `pose.pose_scratch("mars")`
  raises `AttributeError: 'MarsBody' object has no attribute 'base_body'`.
  `PoseScratch` is a walker's tool (effectors, soles, a base body), so the
  editor's list has to be filtered by capability — `isinstance(body,
  RobotSpec)`, or an `effectors`-shaped question — before a MARS appears in
  it. Nothing else in the tree breaks on a non-walker entry: `train
  --robot mars` raises NotImplementedError naming Phase 4, the palette's
  shipped groups come back empty, and the stage pitch is per body already.

### Phase 3 — MARS in a room: drive it, sense with it, give it the existing brains  `[x]`

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

**3a DONE (2026-09-17)** — the STANDALONE half: `robots/mars_drive.py`
(`MarsDriver` — Innate's base velocity PD, station keeping, the `cmd_vel`
watchdog, the safety governor and `arm_servo`, `MarsBody.driver()` now returns
one) and `sensors/lidar.py` (`LidarSensor`/`LidarFrame`/`LidarNoise` on
`ray.RayFan`), each unit-tested in MARS's own scene: `tests/test_mars_drive.py`
28 cases + `tests/test_lidar.py` 18 cases, 0.7 s, and all 20 planted breaks
caught. `world/`, the lab server and the viewer are untouched (Phase 3b).
Measured:

- **Innate's `KP_YAW = 3` is UNSTABLE at this repo's 5 ms step** and it is the
  one constant that could not be ported verbatim. `xfrc_applied` is an
  explicit force, so the velocity loop `v += (KP*dt/M)(v* - v)` needs
  `KP*dt/M < 2`; the apparent inertia at the base DoFs (`mj_fullM`, the base
  3x3's Schur complement, arm folded at HOME) is 1.315 kg / 1.297 kg /
  **0.005761 kg*m^2**, so their yaw gain is 1.04 at their 2 ms and **2.60 at
  ours**. Unguarded it diverges: 1.0 rad/s commanded for 3 s ends at yaw
  7.13 rad with |wz| peaking at 13.3 rad/s, and a straight 0.3 m/s line
  wanders 229 mm sideways. `_stable_gain` clamps each gain to
  `GAIN_LIMIT * M / dt` with GAIN_LIMIT 1.0 (deadbeat, factor 2 in hand):
  forward 200 and lateral 40 are untouched at either timestep, yaw becomes
  1.152 at 5 ms and 2.88 (96 % of theirs) at 2 ms. The empirical boundary —
  KP_YAW 2.0 tracks, 2.5 diverges — is the 2.0 bound to two digits. The yaw
  inertia is small because mars.urdf carries placeholder inertias, so a
  revision with real ones will raise the clamp on its own.
- **The drive cannot ride the 50 Hz control tick.** At a 20 ms tick the
  forward loop's gain is 3.04 and a 0.3 m/s run ends going backwards at
  2.6 m/s; at 10 ms it rings. Phase 3b must call `step()` every physics step.
  It costs 6.3-6.6 us of a ~14 us step — **66-71 k steps/s driven against
  119-127 k bare over four runs, 0.56x either way**, two driven MARSes in one
  model 36,400-37,000 — and decimating the drive would only buy ~1.5 us.
- **The lidar must exclude `base_link`, not its mount.** `base_laser` is a
  jointless frame INSIDE `base_turret`, the box modelling the scanner's own
  housing, and that box is a geom of `base_link` — so with `RayFan`'s own
  default (exclude the mount) **all 360 rays return 0.039-0.101 m** and the
  frame is full, plausible and useless. Innate's `lidar_scan` excludes
  `self._base_id` for the same reason. `RayFan` grew one optional
  `exclude_body=` (default unchanged) and the sensor defaults it to the
  mount's PARENT, resolved from the model so a prefix needs no id table.
- **`planar_fan` could not author a full turn.** Left-first across 360 deg
  puts ray 0 on -x, repeats +-180 as two rays and leaves NO ray on +x. One
  optional `ccw=` flag (default unchanged, the old sector test still green)
  gives Innate's convention: `arange(n) * 2*pi/n` CCW from +x.
- **The laser sits 76.4 mm BEHIND the base origin**, so a 2 m wall reads
  2.0264 m and a consumer that assumed the base frame would put every
  obstacle 76 mm nearer than it is — a third of the robot's length.
  `LidarFrame.mount_pos` carries the offset in the base's heading frame.
- **The scan sees the robot's own folded arm**: 4 rays of 360, at 7-10 deg,
  from 0.1562 m off `link5`, and nothing else of the chassis. Documented, not
  masked — the shadow moves with the arm, so ignoring returns inside the
  footprint belongs to the consumer. One 360-ray scan is 30.4 us; polled at
  6 Hz it is 1.7 us a step.
- Drive numbers: 0.3 m/s x 5 s -> x 1.4994 m, y -6.7 mm, yaw -0.0045 rad;
  1.0 rad/s x 3 s -> 2.9995 rad; cmd 2.0 m/s -> 0.8000 m/s; the watchdog has
  |v| < 0.02 m/s 0.52 s after the last command; a 0.5 m/s shove peaks 0.9 mm
  and settles 0.2 mm out; the arm holds 0.0034 rad while driving.
- **The one-shot shove does not test station keeping.** The drive's own
  forward damper sheds a velocity injection in ~7 ms, so that case passes at
  0.91 mm with the hold stubbed out. A 200 ms *sustained* push is the
  discriminating one: 51 mm out, back to 5.6 mm with the hold and still
  51 mm without it.

**3b DONE (2026-09-17)** — MARS IN A ROOM. `Duck.robot` is a first-class
field validated against `robots.registry.ids()`; `compose()` routes a
non-duck entry through `Body.attach`; `world/arena.WorldRobot` is a SEPARATE
class from `WorldDuck` (1,600 lines of walker fields answer nothing a wheeled
base asks) that holds the driver, steps it every physics step, and shares the
odometry, bump and brain plumbing by being written once in `World`; a body
mounts its own sensors through the new `Body.make_sensors` and names its own
root link and gaze joint through the new `Body.frames()` (a `RobotFrames`, in
`robots/body.py` so a plugin can import it — the first cut of this phase held
those two names in a `ROBOT_FRAMES` dict in the arena, which is the §1 pattern
being deleted, and the review caught it); `Senses.lidar` /
`Intent.arm` exist; and `sensors.lidar.tof_from_lidar` hands the duck's
`wander` and `follow` a 64-zone frame so `brain/controllers.py` is UNEDITED.
`tests/test_world_robot.py` + the ToF-adapter block in `tests/test_lidar.py`,
25 planted breaks, 25 caught; 1802 passed suite-wide, duck fingerprints
unchanged. **Four things measured, two of them bugs the phase found:**

- **A planar base's attach frame must be the IDENTITY.** A frame's transform
  is baked into the attached body's `pos`/`quat`, which a FREE joint's qpos
  replaces (why the duck keeps its spawn frame) but which a SLIDE or HINGE
  joint treats as its reference — so the two compose. A MARS spawned at
  (0.4, -0.3, 0.9 rad) and then written to the same numbers landed at
  (0.884, -0.173): the spawn applied twice, silently, because both halves
  were individually right. The driver poses it; the frame is the origin.
- **A body-mounted camera's housing is its PARENT.** `head_camera_left` is a
  5 mm marker inside `head`'s 113x121x36 mm collision box, and the occlusion
  ray excluded only the mount — so every ray hit `head_body` at 0.0206 m,
  `_unoccluded` returned 0.0 for a person 1.22 m dead ahead, and `follow` sat
  in `search` for a whole 60 s run. Exactly the failure the lidar's turret
  had; `Detector` got the same `exclude_body` default and override.
- **The world's option block costs MARS nothing.** `attach` keeps the
  parent's `<option>`, and MARS's own three (`implicitfast`, elliptic cone,
  impratio 10) agree with `compose`'s block to five decimals on the hold
  (0.00344 rad), the line (1.4994 m / -6.7 mm / -0.0045 rad) and the contact
  count — because the arm is a `qfrc_applied` servo with no contacts and the
  wheels are frictionless, so there is no friction cone to model. NOT tested
  for a GRASP, which is what those options were set for; re-measure in
  Phase 4.
- **`wander` is the wrong brain for a wheeled body, and the scan already
  says why.** The bar (`0 wall contacts` x 3 seeds, 60 s) is MET on
  `mars-playroom` — 0, 0, 0 — but only because the run never reaches a wall:
  the MARS cruises 1.2 m to the basket and is then pinned on its 6 cm rim for
  53 of 60 s (5-6 contact episodes, ~2680 of 3000 ticks, path 3.7-3.8 m,
  states steer/unstick). The rim is 11 cm BELOW the 17 cm scan plane, so the
  lidar cannot see it — and the head camera CAN (the basket marker is a
  detector target). In a bare walled room the same brain covers 17.55 m and
  bumps 0 / 0 / 2 on seeds 0 / 1 / 2, and the two grazes are the ARM'S ELBOW
  (`link2_elbow`, 0.199 m to the SIDE, measured at the contact) against a
  wall it is driving parallel to at 0.28 m/s with 2.3-3.2 m clear AHEAD. The
  360-degree scan has that wall; the 45-degree ToF adapter throws it away.
  **Phase 5's lever is a brain that reads `Senses.lidar` and the camera, not
  a tuned `stop_at`** — for reference `wander`'s 0.30 m stop leaves a duck
  0.21 m of clearance and leaves MARS 0.014 m, against a 6 Hz scanner whose
  staleness is 0.05 m of travel at cruise.
- **`follow` works unchanged: 0.988 in band** (0.45-0.95 m, `FollowParams.
  distance` 0.7 +- `FollowTask.band` 0.25), median 0.849 m, p10-p90
  0.769-0.903, 10.82 m of path, 0 wall contacts, 0 falls, `approach` for 2808
  of 3000 ticks. Against the bar of >= 0.95 and the duck's own measured
  0.955-1.00. The MARS drops the brain's `vy` (a differential-drive base
  cannot strafe, so `idle_vy`'s gait-warming sidestep is inert) and does not
  need it: it has no gait to keep warm.

### Phase 4 — arm policies: `MarsArmEnv`, `reach` → `pick`, the teach panel, ONNX  `[~]`

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

**4a DONE, and the bar is MISSED — the reach is solved, the HOLD is not
(2026-09-17).** `robots/mars_env.py` (`MarsArmEnv`, its own `gym.Env`),
`behaviors/mars_tasks.py` (`mars_reach` in the 🎓 panel),
`MarsBody.env_class("reach")`, `train.MARS_TASKS`, the stamped export, and
`scripts/probe_mars_reach.py` as the eye — `render-rollout` is a walker's tool
(trunk height, foot contacts, a fall rule) so the arm got its own. 43 cases in
`tests/test_mars_env.py`, each shown to fail on a planted break. The
acceptance measurement was a CLI run under `MICRODUCK_RUNS_DIR` (the batteries
exception), 1.5 M steps at 8 envs in 2.6 min, **8,200-9,900 steps/s** — MARS
trains at about the duck's throughput, as Phase 0's cost story predicted.

`reach` on the deterministic export, 8 seeds x 8 s:

| | final median | best median | ≤ 2 cm | held 1 s | sat. | tail spread |
|---|---|---|---|---|---|---|
| null (untrained net) | 30.1 cm | 30.1 cm | 0/8 | 0/8 | 0% | 0.0 mm |
| null (zero action) | 30.2 cm | 30.1 cm | 0/8 | 0/8 | 0% | 0.0 mm |
| v1 | 2.62 cm | 1.10 cm | 1/8 | 0/8 | 64% | 9.5 mm |
| **v2 (the pick)** | **2.02 cm** | **0.75 cm** | **3/8** | **0/8** | 58% | 11.8 mm |

The contact sheet is unambiguous: the arm unfolds from HOME and puts the
gripper ON the target inside 1 s, then **limit-cycles between 1.1 and 2.3 cm
with a ~2 s period**, dipping inside the 2 cm ball for 1-5 control steps at a
time where the success rule wants 25 consecutive. So this is not a reaching
failure at all — it is a settling failure, and the bar as written (8/8 within
2 cm) turns entirely on the last centimetre.

Four things measured on the way, each of which changed the build:

- **The action box's reachable set had to be measured before the reward was
  written.** A shell solution costs up to **3.09 rad** on its worst joint
  (median 2.01, p95 2.6-3.0; 60 targets a window, solved by coordinate
  descent over the full ranges with self-colliding poses rejected), because
  HOME parks the arm folded at joint1 = +1.445 rad — pointing 83° to the
  robot's LEFT, while the task samples the front arc. Hence
  `ACTION_SCALE_RAD = 3.0`. The shell itself is the arm's: residual 0.9 mm
  median, 95% of draws inside 2 cm.
- **An absolute joint target at 25 Hz is a 3 rad teleport, and the chassis is
  in the way.** 60 of 60 random episodes ended in a self-collision inside
  **1-5 control steps**. No reward fixes a rollout distribution that never
  contains the skill, so the fix was the world: the commanded target is
  rate-limited to **6.0 rad/s** `[datasheet — the XL430 shoulder pair's
  61 rpm; not measured here]`, which is what the real servos could follow
  anyway. Random episodes go 1.6 → 67.7 steps, and the ZERO action survives
  all 200.
- **"Terminate on any self-contact" terminates on the URDF's own error.**
  Within ±0.1 rad of HOME, 15.3% of poses report a contact and **0.0% are
  deeper than 5 mm** — all `link1↔link3` / `link2↔link4`, links two apart in
  a chain folded tight, in boxes Innate already documents as overlapping
  ~9 mm (`JOINT2_GUARD_MIN`). Driving the arm properly into the chassis is
  20-60 mm. The predicate is therefore a **10 mm depth**, and HOME is clean.
- **The bug that cost the first eval, and the transferable one.** SB3 clips a
  Box action before the env sees it, so during training the raw and the
  clipped action coincide and `last_action` is unambiguous. At inference
  nothing clips: v1's exported mean reaches **|a| = 90 in a ±1 box**, and the
  probe that handed the raw output straight to `step` wrote 86.6 into the
  observation — scoring the same policy at 0.209 m instead of 0.026 m. **A
  limit applied at both training and inference must come from one place**;
  the env now clips what it OBSERVES (the penalties still price the raw).
  The saturation itself is what `action_mag_penalty` was added for after v1:
  it improved every accuracy number and did not fix the hold.

**What 4b needs before `pick`:** fine control near the target, and none of it
is reward work. The mean still sits on the box edge 58% of the time, so a
rate-limited joint target can only alternate — it has no "stay". Three
candidates, in order of cheapness: a **non-linear action map** (cubic, so the
same box gives millimetre resolution near zero and full range at the edge); a
**narrower action box as a second curriculum rung** once the arm is near (the
physics ladder this file already plans for the block spawn); and
`LOG_STD_MAX`, which is **std 0.6065 — 61% of a ±1 half-box against 15% of
the duck's ±4**, so MARS's exploration pressure is 4× the duck's in the units
that matter. That coupling between a body's action-box width and a global
trainer constant is undocumented and should be written down wherever the
fourth body's box is chosen.

*(All three of those candidates were measured in 4a-2, directly below. The
short version before you act on this paragraph: the non-linear map LOST, the
narrower box was structurally excluded and never trained, `LOG_STD_MAX`
needed no change — and a fourth option nobody had listed, an INCREMENTAL
action, is what worked.)*

**4a-2 DONE — the action map was the mechanism, and `delta` is the default
now. The 8/8 bar is still open (2026-09-18).** Three variants as env options
(`MarsArmEnv(action_mode=..., action_scale_rad=...)`, reached from the CLI by
`--action-mode` and recorded in `run.json`'s `env_kwargs`), matched at 1.5 M
steps / 8 envs / seed 0, scored on the deterministic export over 8 seeds × 8 s
with `scripts/probe_mars_reach.py` — which now builds its env from the run's
own map, because the same six floats are an increment under one map and a
position under another.

| variant | final median | best | ≤ 2 cm | **held 1 s** | sat. | tail spread | self-hits | sheet |
|---|---|---|---|---|---|---|---|---|
| null (zero action) | 30.2 cm | 30.1 cm | 0/8 | 0/8 | 0% | 0.0 mm | 0/8 | `scratchpad/mars-null-eval/` |
| **absolute** — 4a's v2, the baseline | 2.02 cm | 0.75 cm | 3/8 | **0/8** | 58% | 11.8 mm | 0/8 | `scratchpad/mars-reach-v2-eval/` |
| **B. cubic** `HOME + sign(a)|a|³·3.0` | 7.62 cm | 5.33 cm | 0/8 | **0/8** | 58% | 22.9 mm | 2/8 | `scratchpad/mars-reach-cubic-eval/` |
| **A. delta** `target += a·0.24`, seed 0 | **1.13 cm** | 0.42 cm | **5/8** | **5/8** | 58% | 5.3 mm | 0/8 | `scratchpad/mars-reach-delta-eval/` |
| A. delta, training seed 1 | 2.38 cm | 1.23 cm | 3/8 | **2/8** | 55% | 7.0 mm | 0/8 | `scratchpad/mars-reach-delta-s1-eval/` |

**The mechanism was a missing fixed point, not saturation.** A rate-limited
ABSOLUTE target cannot be told to stay: the action names a destination, the
target walks there at 6 rad/s, and holding still needs the action to keep
pointing at wherever the target already is — a moving quantity, re-hit every
40 ms. `delta` integrates instead (`a = 0` is exact), and it is the first map
the task's own success rule fires on at all. The sheets are qualitatively
different, not just better: v2 crossed the 2 cm ball for 1–5 control steps at
a time on a ~2 s cycle, while the delta sheet reads 0.5–0.7 cm from t = 2 s to
the end with the near-streak climbing past 150 of the 25 it needs.
`at_target` doubles, +110 → +212. **All three maps sit at 55–58% of dims on
the box edge and `delta` holds anyway** — so 4a's 58% was a symptom, and under
`delta` a saturated action means "slew at the fastest legal rate", which is
the right command while travelling and simply is not what the policy emits
once it arrives. (`AGENTS.md`'s "KL was a symptom, not the cause".)

Corroboration from the optimizer's side: `train.LOG_STD_MAX` caps std at
0.6065 and the cap BINDS on 6 of 8 dims under `absolute` (v2: 0.584–0.612) and
4 of 8 under `cubic`, but `delta` pulls its three shoulder dims down by itself
to 0.451 / 0.530 / 0.489. Only under `delta` does noise cost anything — it
integrates into target drift, so there is gradient pressure to be quiet near
the target. So `LOG_STD_MAX`, the third candidate 4a listed, needed no change;
the map made the cap stop binding where it mattered. The box-width/`LOG_STD_MAX`
coupling 4a asked to be written down still should be, for the fourth body.

**Cubic is a real negative result.** Worst of the three, and the only one that
self-collides. `|a|³` means an action must run to the box edge to travel at
all (`|0.5|³·3.0` = 0.375 rad, an eighth of the linear map at the same output),
so the policy lives at `|a| ≈ 1` — where the cubic slope is 9.0 rad per unit
action against the linear 3.0. It sold the interior resolution it was bought
for (0.99 mrad per 0.01 of action at `|a| = 0.1`, against 30.0 linear) in
exchange for 3× the coarseness where the policy actually operates. Its sheet
shows it PARKING stably at 2.9–3.2 cm, so the map does settle — it just cannot
settle close. **Stretching an action map's interior only helps if the policy's
operating point is in the interior, and 58% on the edge was the measurement
saying it is not.**

**The narrower box (4a's second candidate) was never trained, and that is the
result.** `ACTION_SCALE_RAD` 3.0 → 1.0 is structurally excluded, measured
before spending a run on it (`AGENTS.md`, "check a knob's reachable set
first"; `scratchpad/probe_box_reach2.py`, the coordinate-descent solver the
3.0 was chosen with, 40 draws from the env's own shell):

| box | residual median | p90 | ≤ 2 cm | the 8 eval targets |
|---|---|---|---|---|
| full joint ranges | 0.14 cm | 3.67 cm | 72% | — |
| HOME ± 3.0 (today) | 0.15 cm | 3.15 cm | 78% | **8/8 inside 0.3 mm** |
| HOME ± 2.0 | 2.30 cm | 6.91 cm | 48% | — |
| HOME ± 1.0 (the rung) | 12.96 cm | 24.96 cm | **0%** | **0/8** (5.0–27.8 cm) |

HOME parks the arm folded at joint1 = +1.445 rad, 83° to the robot's LEFT,
while the task samples the front arc — so a 1.0 rad box cannot reach a single
one of the eight targets the A/B scores. Its run would have printed ~13 cm and
said nothing about the hold: an expected null, not an informative one. The
third run went to a second training SEED instead. `RUNG_SCALE_RAD` stays as a
refuted option with that table on it, and the useful reading is that narrowing
a box only helps when its centre follows the arm — which is what `delta` is.
The same table also confirms the bar is **geometrically attainable** on these
eight seeds (8/8 inside 0.3 mm), so nothing about the shell excuses a miss.

**What 4b still owes, from 4a-2's own numbers:**

- **Two training seeds, always.** Seed 1 of the identical delta recipe holds
  2/8 against seed 0's 5/8 — wider than the 8-seed eval spread, which is
  "eval seeds don't measure training runs" landing exactly as written. The
  claim is "delta holds about half the seeds". `absolute` has only one
  training seed here, so the comparison rests on 0/8 → {5,2}/8 being
  structural rather than on a paired statistic: an absolute map's zero is the
  absence of a fixed point, not variance.
- **What still misses is not the hold — it is the shell's edges.** On seed 0's
  run: seed 4's target is at 0.386 m of the 0.40 m shell AND 0.343 m of its
  0.35 m ceiling (the arm arrives stretched out and stalls 2.7–3.4 cm short,
  though the solver gets within 3 mm — so the policy, not the geometry);
  seed 1 is at +54° of the ±60° arc; seed 6 reaches 0.2 cm then drifts to
  2.1 cm. Two edges and a drift is a **physics-ladder** shape — ladder the
  target shell, which `behaviors/mars_tasks.py` already names as the rung to
  add — and not a reward one.
- The `reach` reward was not touched in this phase, by design: the mechanism
  was in the action space, and a reward cannot pay for a behaviour the action
  space has no way to express.

**3b's open question is still open, and 4a could not close it.** Phase 3b
left "the world's option block costs MARS nothing… NOT tested for a GRASP,
which is what those options were set for; re-measure in Phase 4". `reach`
never closes the gripper on anything — there is no object in its scene — so
nothing here exercises the elliptic cone or `impratio 10` either. It is
`pick`, in a world model, that settles it.

**Also for 4b, from the gripper slot's own measurement:** `gripper_load`
carries the torque `arm_servo` wrote at joint6, and closing on AIR drives the
blade into its own hard stop where the position error is zero, so the slot
reads **0.0 N·m — identical to an open claw**. What distinguishes "holding" is
that an object stops the blades SHORT of that stop, which shows up in
`arm_qpos[5]` with `qfrc_constraint` as the force behind it. So `pick` wants
the constraint torque as well as this slot, a pickable body in the scene (its
own model, hence `MarsArmEnv(own_model=True)`), and the `target_seen` gate
finally doing something — it is hard-wired to 1.0 today because `reach`'s
target is privileged, and the slot exists so that adding the head detector
does not change the layout.

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

**PRODUCER SIDE DONE (2026-09-17)** — `robots/policy_contract.py`:
`PolicyContract(id, robot, obs_dim, act_dim, rate_hz, slots, deploy)` with
`Slot(name, start, stop)`, and `resolve(run_dir_or_onnx)` as the ONE
precedence in the tree — the ONNX's own `metadata_props`, then `run.json`'s
`"contract"`, then `run.json`'s `"robot"` through the registry, then the duck
(a file with nothing to say has always been a duck). A DIRECTORY is asked
`run.json` first, deliberately: naming the run asks what the run drives, and
the lab resolves every run in the palette on a timer where 300 bytes of JSON
beats a protobuf parse per run per poll. `declare()` takes the dims FROM the
body so they cannot drift, and `__post_init__` refuses a table that does not
tile `[0, obs_dim)` exactly — so a bad layout is a construction error at the
first `fetch-robot` listing. The three contracts:

```
microduck-61-v1  obs[1,61] -> actions[1,14] at 50 Hz   drop-in: the robot's own contract
g1-lucky-99-v1   obs[1,99] -> actions[1,29] at 50 Hz   lab contract: base lin vel is not observed
mars-arm-32-v1   obs[1,32] -> actions[1,8]  at 25 Hz   code skill, untested on hardware
```

Written by `export()` (into `metadata_props`, before its own onnxruntime
cross-check, so every export proves the stamp is free), by `train.py` and
`distill.py` into `run.json` (`"robot"` kept beside it — an addition, not a
migration), and as `contract_id` alone into `record.json`, which is what a
person reads. Read by `export_onnx.run_robot()` (now `resolve(...).robot`)
and by `eval-walk`, which refuses a `--robot` that contradicts a contract the
file RECORDED and still honours it for a file that declares nothing — the
flag's only real use is an old `.onnx` moved away from its run. `export-walk`
prints `describe()` so the deploy caveat leaves the building with the file;
`fetch-robot` lists each body's id beside its state.

MEASURED: the stamp is free — graph protos byte-identical once the props are
stripped, onnxruntime outputs bit-identical over 200 random observations,
+456 bytes. 50 new cases in `tests/test_policy_contract.py`, every positive
shown to fail on a planted break (25/25 caught); the duck's table is pinned
by reading all EIGHT slots back off a live `MicroduckWalkEnv` and the G1's
four off a live `G1WalkEnv`, because a swap of two same-width slots still
tiles perfectly and only an env can tell them apart. Two of those plants
first went MISSED and fixed the tests: at the STAND keyframe `base_lin_vel`
and `base_ang_vel` are both three zeros, so the G1 case now steps 40 times
before it reads (the "perturb what the dynamics can feel" rule), and the
rung-1 case had been competing the metadata against a robot NAME instead of
a recorded contract, which the reversed precedence also satisfied. The
conformance suite's exporter round-trip now asserts the stamp, so "supported"
includes "its exports are self-describing". The duck/BAM/G1 rollout
fingerprints are unchanged.

**Left for 1b** (`viz_server.py` is contended): `policy_robot()` at
viz_server.py:843 becomes `resolve(path)` — it already takes either a run dir
or a file, which is the shape `resolve` accepts — and the four width guards
(`do_assign` :3918, `do_spawn_helper` :3971, `do_spawn_duck` :4040,
`apply_snapshot` :4078, each comparing `infer.obs_dim` against
`duck.env.observation_space.shape[0]`) become one `matches()` against the
body's contract, with the graph width kept as the last-ditch check that the
FILE is what its metadata claims. `render_rollout.main()` :723 calls
`run_robot(Path(args.policy).parent)` and its `--robot` choices are still the
literal `("microduck", "g1")`.

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

**DONE in 3b (2026-09-17)**, with one deliberate difference from the sketch
above: `WorldDuck` was NOT generalised. `WorldRobot` is a second class, and
the sense/intent channels are declared by `Body.make_sensors` plus the
`Senses`/`Intent` field per channel rather than by a channel table; the
frames world mode needs by name come from `Body.frames()`. **No `if robot ==
...` was added anywhere** — `world/arena.py` branches on the CLASS it holds
(`isinstance(d, WorldRobot)`), never on an id, and `world/compose.py` and
`world/scenario.py` branch only on `!= DUCK_ROBOT` to keep the duck's own
attach byte-identical. The
`obs_version`-per-body observation builder is still Phase 4's, because
nothing trains on MARS yet. What is left of §6.5 is the `/sim` VIEWER
(Phase 2b) and `brain_env`'s hard-coded `[0:64]`.

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
