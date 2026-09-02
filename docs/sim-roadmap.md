# `/sim` roadmap: real-time sensing, autonomy, and multi-duck play

Where the lab goes after tricks: simulate the parts of the Microduck that are
not servos (the head ToF depth matrix, the camera + NPU detector, the
microphones and the speaker, BLE proximity, odometry), give the ducks a
**brain layer** that turns those senses into the same `robot.move` /
`robot.head` intents the real robot takes, and open a new `/sim` page in the
viewer where people build worlds, watch what each duck senses, and train ducks
to follow, map, tidy a playroom, and play soccer against each other.

This document is a brainstorm and a plan. **Status:** Phases 1 and 2 are in
the tree, and Track 12 has its first working loop:

- Phase 1: `world/` (0.1–0.3 incl. the page's editor), `sensors/` ToF (1.1,
  1.2, 1.8, 1.9), `world_server.py` + `/sim` (0.4, 0.6 record/replay, 7.1–7.4,
  7.8), the CI matrix (9.6, Windows runs the spawn-safe modules).
- Phase 2: the geometric detector (1.3) and a tracker with ids over it,
  persons + possess, the brain runtime with freshness (2.1, 2.4), `Wander`
  and `Follow` (2.3), `BrainEnv` + `train-brain` + `eval-brain` (3.1, 3.2),
  odometry drift presets (1.7) and an occupancy map per duck in its own
  odometry frame, painted on the page, with a wall-line loop closure
  against drift (5.1, 5.5: pose error under a 1.5°/s bias halved);
  the `pitch` scenario with two `chase` brains, the robot's shipped kick
  policies as skills, goal counting and `eval-pitch` (soccer, first form:
  1.38 goals, 6.5 kicks, 0.50 falls a run over 300 s and 8 seeds, with a
  body-aware avoid of the other duck — falls were 2.1 before it — and
  kick windows at robotd's standing gain), then the second form — the
  head tracks the ball on the way in, a 3D-placed ToF bumper, wall and
  retreat rules, goal-centre aiming — at 1.12 / 11.8 / 0.50, and teams
  (`brain/team.py`, `pitch-2v2` / `pitch-3v3`): 2v2 2.25 goals, 10.8
  kicks, 3.0 falls a run; 3v3 1.00 / 5.8 / 5.25 — falls per duck climb
  with the roster. Dribbling, a walk-round and close-range
  re-planning were built, measured worse, and ship off with the numbers.
  `brains/follow-v1` ships in the repo. Upstream is pinned (microduck_rl
  badc4e7, the 2026-09 CAD re-export; microduck 2c61dcc) and every
  model-dependent number above was re-measured against it — the tidy loop
  needed its release distance re-measured (the new model's stop drifts
  1–2 cm further), nothing else moved. Binary framing (0.5) stays open on a measurement:
  the page's perf readout puts the JSON encode at 0.8 ms per 40 ms frame
  for two ducks with maps streaming (physics 1.6, sensors 0.6 — the whole
  loop is ~7% of a core), so JSON holds until rosters of four or more
  ducks make the encode the largest term. **Measured:** on identical
  follow-me episodes (12, the pinned model) the learned brains hold the
  distance band 0.80 / 0.63 (`follow-v2` under the reflex tier — the env
  yaws the head toward the tracked target and refuses to walk into
  something 0.25 m ahead; 0.69 / 0.68 without it), 0.71 / 0.63
  (`follow-v3`, trained with the reflex tier and variety: boxes and a
  wandering duck; best on the variety benchmark at 0.68) and 0.73 / 0.60
  (`follow-v1`, version-1 observation, scored in the world it was trained
  in) under the datasheet / hostile presets against the scripted
  controller's 0.46 / 0.42; in sight 0.75 / 0.61, 0.68 / 0.57, 0.85 / 0.75
  vs 0.53 / 0.40. The scripted one loses because it stands still and goes
  cold; an idle sidestep took it from 0.36 to 0.51 in sight, the head gaze
  to 0.53, and the rest of the gap is the learned brain's continuous motion.
- Track 12: toys, a basket, grasp-as-attachment, the shipped ground-pick as
  a skill, and the `tidy` brain with `eval-tidy` (12.1–12.4, 12.6, 12.7,
  12.13), the tether toggle (12.10: `--tether-ms`, `POST /world/tether`).
  **Measured:** 0.94 of six scattered toys are in the basket after five
  minutes on the pinned 2026-09 model (8 seeds, 0.50 falls a run; 0.88 /
  0.50 under datasheet odometry drift, 0.62 / 0.75 under hostile drift,
  0.79 / 1.50 with a 250 ms brain tether — every traced tethered fall was
  the stopping stride at the rim on a stop decided 250 ms late) — up from
  0.88 / 0.38 before rim toys were approached from the outside, 0.67 /
  1.7 falls at the first close of the loop and 0.11 before that. What it took, each a measurement on
  the walker, not a tune: releases only after a standing re-measure of
  the basket at 0.42 m (walking rocks the head 0.02 rad and holds it
  0.08 rad higher than standing — detection frames now carry the camera
  pose); a release geometry with 3 cm to spare (beak 8 cm past the trunk,
  feet 4 cm, rim contact from 0.185 m); a turn-around back-off because the
  walker does not reverse at all; a forward kick on turns from a standstill
  because a cold right turn never starts; tracked ids instead of a
  confidence bar (a 2 cm toy at 1.5 m is 1.5° wide and confidence 0.2); a
  keep-out disc around the basket and "a toy that projects into the basket
  is delivered". `walker-facts` and `trace-tidy` (skill `tidy-trace`) are
  the tools that found each of these. The carry-walk reflex (12.5) was not
  needed: the shipped walker carries a 20 g block. The basket is designated
  in the scenario/editor (12.9 in first form); the tether toggle (12.10),
  graspability learning (12.8) and VLM designation (12.12) are still plans.

Numbers that shaped the design, all measured in this world: the walker only
turns in place at a yaw command of 1.0 and stands still below ~0.2 m/s
asked; it honours a head-pitch intent of +0.6 (camera 37° down, no falls,
even walking) but cannot turn in place with the head down; it coasts ~1 cm
after a stop and walks straight under the heading hold; the ground-pick
tip bottoms 2 cm up and 7.8 cm ahead of the trunk; a 4 cm block is found by
the simulated detector about half the time at 0.6 m and rarely beyond 1 m.
Everything else below is still a plan. It was written after reading this repo (both checkouts), upstream
`pollen-robotics/microduck` (the onboard Rust daemons and their design docs),
`pollen-robotics/microduck_rl` (the official training stack), and the public
Hugging Face and press material. Facts about the robot below cite where they
came from; when something is an assumption it says so.

Contents:

1. [What the robot actually has](#1-what-the-robot-actually-has)
2. [The one architectural decision](#2-the-one-architectural-decision-reflex-policy--brain)
3. [The `/sim` page](#3-the-sim-page)
4. [Task list, by track](#4-task-list-by-track)
5. [What to build first, and why](#5-what-to-build-first-and-why)
6. [Teaching with it](#6-teaching-with-it)
7. [UX, debugging, and visualization principles](#7-ux-debugging-and-visualization-principles)
8. [Performance plan](#8-performance-plan)
9. [Windows support](#9-windows-support)
10. [Sim2real for the brain layer](#10-sim2real-for-the-brain-layer)
11. [Open questions and assumptions](#11-open-questions-and-assumptions)

---

## 1. What the robot actually has

The relevant facts, from upstream code and docs rather than marketing pages.

| Component | What the sources say | Where |
|---|---|---|
| Compute | Rockchip RK3566 with NPU, 1 GB RAM, 32 GB storage. Radxa Zero 3W is the bring-up board. | press kit via search; `docs/project/slice-2-bringup.md` |
| Depth | **One** VL53L5/8CX 8×8 time-of-flight matrix on the head HAT's I²C bus, served by `tofd` on `/run/tofd/tof.sock` as a `tof.stream` subscription. The daemon publishes the sensor view only; consumers reproject it through `robotd`'s kinematics. | `docs/design/architecture.md` |
| "LiDAR" | Press calls the ToF matrix an "8×8 time-of-flight LiDAR matrix". No scanning LiDAR appears anywhere in the onboard repo. **Assumption:** the "Compact LiDAR" and the "8×8 ToF matrix" are the same device. Section 11 covers what changes if they are not. | press via search; repo tree |
| Camera | Front camera through `mediad` (libcamera, V4L2 M2M H.264, 30 fps, WebRTC out). Raw frames never cross a socket; only derived features do (tens of bytes, 10–30 Hz). | `docs/design/architecture.md` |
| NPU detector | `duck-detect`: a quantized YOLO11n, 320×320 input, one class (duck), 3.9 MB INT8, mAP50 0.976, p50 25.7 ms / p95 58.4 ms per frame on the board. Detections are not yet exposed over IPC; the plan is to run it inside `mediad` and publish detections as state, then build "approaching, following, facing" on top. | `docs/project/npu-bringup.md` |
| Microphones | `pet-detect`: 16 kHz mono from an AIC3104 codec, 40-band log-mel windows, a ~20 KB CNN in ONNX that classifies head-petting from the sound, with hysteresis. `robotd` polls it and plays a "coo" on onset. | `pet-detect/` |
| Speaker / voice | `sounds/` crate, a procedural synth with pitch, level, and vowel set at runtime. `theremin.rs` maps depth to sound at 15 Hz. Press: each unit generates its own voice on first wake and keeps it. | `sounds/`, `docs/ideas/autonomous_behavior.md`, press |
| IMUs | Two: body (v2 `imu_to_dxl` board, id 200 on the Dynamixel bus, gives the policy its gyro and projected gravity) and head. | `docs/design/robotd-design.md`, press |
| Proximity | BLE beacons: nearby duck ids and RSSI as a distance proxy, plus a shared beat "±20 ms across ducks" for synchronized moves. Two NFC antennas. | `docs/ideas/autonomous_behavior.md`, press |
| Odometry | Contact-anchored: one sole corner is pinned to the ground and the trunk pose follows by forward kinematics; heading is integrated IMU yaw with no magnetometer. Drift is inherent. Position and yaw are in the telemetry frame. | `docs/design/robotd-design.md` |
| Control | 50 Hz loop in `robotd`, exactly `[f32; 61]` obs, 14 actions, ONNX Runtime, priority chain `roulade > kick > ground pick > sit/rise > stand > walk`, velocity deadman (intents stop, velocity zeroes, torque stays). | `docs/design/robotd-design.md` |
| API | JSON-RPC 2.0 NDJSON over Unix sockets, one socket per daemon. Continuous intents: `robot.move {vx, vy, vyaw}`, `robot.head {neck_pitch, head_pitch, head_yaw, head_roll}`. Requests: `robot.enable/stop/init/relax/subscribe/health`. The same API is fronted by BLE, WebSocket, and the WebRTC "control" datachannel. | `docs/design/architecture.md` |
| Brain | Milestone M9 "Autonomous brain" is future work with no code. The idea doc sketches a 16-state FSM (Chill, LookAround, Wander, TurnInPlace, Zoomies, Startle, Stretch, Ruffle, Preen, Sneeze, Dance, GroundPick, Nap, BallPlay, Petted, Held) over an energy/mood model, fed by ToF, the NPU detector, audio events, BLE, and the gamepad. Follow-the-leader is sketched as "RSSI holds spacing, ToF handles the duck directly ahead". | `docs/project/roadmap.md`, `docs/ideas/autonomous_behavior.md` |
| Upstream RL and objects | The ball-kick task adds a 70 mm, 15 g ball but keeps the actor **blind**: only the critic sees ball position and velocity. The stated reason is that the real robot has no ball sensing and the operator aims it. The kick policy is hot-swapped in and auto-swaps back after ~2 s. | `microduck_rl` `microduck_ball_kick_env_cfg.py` |
| Hub | The nine shipped ONNX policies are on the Hub as `pollen-robotics/microduck-policies`; M8 is the on-robot "policies from Hub" channel with a `manifest.json` contract. The `microduck-simulator` Space runs MuJoCo in WebAssembly with onnxruntime-web. | awesome-microduck, roadmap |

Two things follow from this table and shape everything below:

- **The RL policy is deliberately blind.** Everything exteroceptive on the
  real robot reaches the legs only as a *command*: a twist and a head pose in
  obs slots 48–60. That is the hot-swap contract this repo already refuses to
  bend. So "train it to follow people" cannot mean "add ToF to the obs vector".
- **The brain layer does not exist yet upstream.** Pollen has deferred it to
  M9 with a design doc still to be written. A laptop lab that can simulate the
  sensors, run a brain against them, and hand that brain to the real daemons
  over the same JSON-RPC intents is a contribution, not a duplicate.

## 2. The one architectural decision: reflex policy + brain

Everything in `/sim` sits on a two-tier control stack that mirrors the robot:

```
                  sensors (simulated here, real on the robot)
   ToF 8×8 @15 Hz · detector bearing/size @10 Hz · sound events · BLE RSSI · odometry
                                   │
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  BRAIN  (10–30 Hz)  scripted FSM · hand-written controller ·   │
   │         RL "brain policy" · LLM tool-caller · a human on WASD  │
   │  emits INTENTS only:  robot.move {vx,vy,vyaw}                  │
   │                       robot.head {neck_pitch,head_pitch,yaw,roll}
   │                       policy select (walk / stand / kick / …)  │
   │                       speaker {pitch, level, vowel}            │
   └────────────────────────────────────────────────────────────────┘
                                   │  obs[48:61] + policy slot
                                   ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  REFLEX  (50 Hz)  the unchanged 61-obs / 14-action ONNX policy │
   │  shipped alpha_walking, or any run trained in this lab          │
   └────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                          MuJoCo world (one mjData: N ducks + objects)
```

Why this is the right split for a laptop lab:

- **Contract-safe.** The 61-obs invariant in `microduck_local/AGENTS.md` is
  untouched. Brains only write the command block, which the reflex policies
  were trained on (with the tiny keep-alive ranges in `contract.py`).
- **Cheap to train.** A brain policy is a small MLP over ~80 features acting at
  10 Hz through a frozen ONNX walker. Physics cost per step is what it is
  today; the brain adds a few microseconds. Brain RL runs are minutes long on
  the same `ForkVecEnv`.
- **Portable.** A brain's outputs are the robot's public intents. The same
  Python brain can run against the sim lab or against a real duck over its
  WebSocket/WebRTC control channel (section 10). For the brain layer the
  sim2real gap is *perception* (noise, latency, dropouts), which this lab can
  randomize, not actuator physics.
- **Teachable.** The split *is* the lesson: reflexes are learned, deliberation
  is composed; this is how the robot is built and how most real systems are.

Hierarchical RL (a learned brain that also selects which reflex policy to run)
is a natural later step, and section 4 lists it.

## 3. The `/sim` page

A second route in `duck-viewer` (`app/sim/page.tsx`) that reuses the stage,
protocol client, and panel conventions of the current page, but is organised
around **a world** rather than a roster of isolated ducks.

### Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 🌍 world: Living room ▾   ▶ ⏸ ⏮  RTF 1.00×   3 ducks · 1 person · 1 ball   ⚙ │
├───────────────┬──────────────────────────────────────────┬───────────────────┤
│ SCENARIOS     │                                          │ INSPECTOR (duck)  │
│ · empty floor │          3D stage (shared Viewer)        │ ▸ Sensors         │
│ · living room │   overlays toggled per layer:            │   ToF 8×8 heatmap │
│ · obstacle    │   ▫ ToF frustum + 64 hit dots            │   detector cone   │
│   course      │   ▫ detector cone + bearing needle       │   sound events    │
│ · follow me   │   ▫ sound rings, BLE links               │   BLE RSSI        │
│ · soccer 2v2  │   ▫ odometry trail vs truth trail        │   odom vs truth   │
│ · custom…     │   ▫ occupancy grid on the floor          │ ▸ Brain           │
│               │   ▫ goals / lines / score                │   FSM live graph  │
│ WORLD EDITOR  │                                          │   inputs+freshness│
│ walls · boxes │                                          │   intents out     │
│ ball · goals  │                                          │ ▸ Policy (reflex) │
│ person agent  │                                          │ ▸ Perf            │
│ duck ↓ spawn  │                                          │                   │
├───────────────┴──────────────────────────────────────────┴───────────────────┤
│ TIMELINE  ●rec  ◀ ▶  scrub ───────────────────●─────────  events: goal! · … │
│ MATCH   🔵 2 – 1 🔴   03:12   possession 61%   │ 📚 LESSON: "why is it blind?"│
└──────────────────────────────────────────────────────────────────────────────┘
```

- **Scenarios** are JSON files in `microduck_local/scenarios/` (like clips):
  a layout (walls, boxes, ball, goals, lights), spawn slots, the agents in
  it, a scoring rule, and a lesson card. The picker loads one; the editor
  edits it in place; save writes it back (`PUT /scenarios/{name}`).
- **The stage** is the existing `Viewer` canvas with a world group: static
  geometry instanced from the scenario, dynamic bodies (ball, boxes, person
  capsules) streamed like duck bodies. Overlays are separate toggleable
  layers so a screenshot can show exactly one idea.
- **The inspector** is per selected duck, tabbed. Sensors shows raw
  sensor views next to what they mean (the 8×8 heatmap with the hit dots
  lit in the scene; the detector's bearing needle over the detected duck).
  Brain shows the FSM as a live graph with the active state pulsing, each
  input with its age (the "freshness gating" idea from upstream, made
  visible), and the intents going out. Perf shows per-duck sensor cost,
  brain tick time, and the world's real-time factor.
- **Timeline** records everything the lab streamed (frames, sensors, brain
  state, intents) to a ring buffer and to disk on demand, scrubs it, and
  replays it with the overlays live. This is the debug tool everything else
  leans on.
- **Match / lesson strip** is scenario-dependent: a scoreboard for soccer, a
  coverage percentage for mapping, a "distance held" gauge for follow-me, and
  the lesson card for the scenario.
- **Possess.** `P` takes over the selected agent with WASD (the protocol
  already carries `{"cmd"}`): drive the person the duck should follow, or
  drive a duck yourself in a match against brains. Turning yourself into a
  sensor input is the fastest way to understand what a brain sees.

### Backend additions (in `microduck_local`)

```
src/microduck_local/
├── world/            # one MjModel via MjSpec composition: N ducks + objects
│   ├── scenario.py   # scenario JSON contract, validation, procedural rooms
│   ├── arena.py      # WorldEnv: shared mjData, per-duck obs/reward views
│   └── agents.py     # person capsule agent (scripted paths / possessed)
├── sensors/          # simulated exteroception, sampled at REAL rates
│   ├── ray.py        # generic ray rig on mj_multiRay (ToF today, LiDAR if ever)
│   ├── tof.py        # 8×8 zones, FOV, range, 15 Hz, noise + dropouts
│   ├── detector.py   # geometric "camera": frustum, occlusion ray, bbox, latency
│   ├── audio.py      # sound sources, attenuation, event classifier stub
│   ├── ble.py        # RSSI from distance + log-normal noise, shared beat
│   └── odometry.py   # contact-anchored FK + integrated yaw (drifts on purpose)
├── brain/            # the M9 layer, prototyped here
│   ├── runtime.py    # 10–30 Hz tick, freshness gating, intent bus
│   ├── fsm.py        # scripted states (upstream's 16), energy/mood model
│   ├── controllers.py# follow, avoid, go-to, sweep-scan (hand-written)
│   ├── brain_env.py  # gymnasium env: features -> intents, frozen reflex ONNX
│   └── bridge.py     # same brain against a REAL duck over JSON-RPC
└── viz_server.py     # +/world +/scenarios +/sensors framing +/brain +/replay
```

### Protocol additions

- Frames gain `world: {objects: [{id, kind, bodies}], score, clock}` and
  per-duck `sensors` and `brain` blocks. Depth cells and feature arrays go as
  a **binary WebSocket frame** (a `Uint16Array` per duck for the 64 depth
  cells) interleaved with the JSON frame; the viewer README already names
  binary framing as the next lever.
- New HTTP: `GET/PUT /scenarios`, `POST /world/load`, `POST /world/object`,
  `POST /brain/{duck}` (set brain kind + params), `GET /replay/{id}`,
  `POST /sensors/{duck}/noise` (live sliders on noise, latency, dropout).
- New WS intents: `{"possess": {"agent": "person1"}}`, `{"brain": {"duck":
  "d2", "kind": "fsm"|"follow"|"run:<brain-run>"|"none"}}`.

## 4. Task list, by track

Sizes: **S** ≈ a day, **M** ≈ a week, **L** ≈ several weeks. "Value" is the
author's judgement on platform leverage + wow + teaching, argued in section 5.

### Track 0: Foundations (everything depends on these)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 0.1 | **World model via `MjSpec` composition** | Build one `MjModel` from a scenario: attach N duck MJCFs (from `microduck_rl`) plus floor, walls, ball, goals, boxes, person capsules. One `mjData` for the whole world; per-duck views for obs. Keep `shared_model_scope` semantics (one compiled model per scenario). | L | ★★★★★ |
| 0.2 | **`WorldEnv` + per-duck reflex loop** | Step all ducks' 50 Hz reflex policies in one physics step. Reuse `Duck` for brain provenance, falls, speed. Contract test: a single-duck world is bitwise the walk env. | M | ★★★★★ |
| 0.3 | **Scenario contract + editor endpoints** | JSON schema, validation, `scenarios/` directory, procedural room generator (seeded). | M | ★★★★ |
| 0.4 | **`/sim` route skeleton** | Route, shared stage, scenario picker, world objects rendering (instanced), inspector shell, timeline shell. | M | ★★★★★ |
| 0.5 | **Binary frame channel** | Interleaved binary WS messages for sensor arrays; JSON stays for everything else. | S | ★★★ |
| 0.6 | **Record / replay ring buffer** | Lab-side recorder of frames + sensors + brain + intents, save as `.duckrec`, scrub and replay with overlays. | M | ★★★★★ |
| 0.7 | **Spawn-safe process model** | Write all new multi-process code with `spawn` in mind from day one (section 9) so Windows is not a retrofit. | S | ★★★★ |

### Track 1: Sensor simulation (real rates, real limits, tunable lies)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 1.1 | **Ray rig** on `mj_multiRay` | Generic N-ray sensor attached to a body: origins, directions, groups, cutoff. ToF is one config; a planar scan is another. | S | ★★★★★ |
| 1.2 | **ToF 8×8** | 8×8 zones over ~45°×45° (≈63–65° diagonal, VL53L5CX/L8CX), ~4 m max, 15 Hz at 8×8. Multiple rays per zone averaged; per-zone noise, range-dependent dropout, min-range saturation, and a "firmware upload" warm-up delay so freshness gating means something. Output shaped like `tof.stream`. | M | ★★★★★ |
| 1.3 | **Geometric detector** ("camera" without rendering) | Project other ducks / people / the ball into the head camera's frustum, occlusion-check with one ray to the target centre, emit `{class, bearing, elevation, bbox_w, conf}` at 10 Hz with 26–60 ms latency (the measured p50/p95), false negatives by apparent size, occasional false positives. Classes: `duck` (exists on the NPU today), `person` and `ball` (would need a retrained detector, flagged as such). | M | ★★★★★ |
| 1.4 | **Rendered camera (debug only)** | Optional real offscreen render from the head camera for the inspector, off by default; never an input the brain depends on. | M | ★★ |
| 1.5 | **Audio events** | Sound sources in the world (person voice, another duck's quack, a clap); attenuation and bearing by inter-mic delay; a stub classifier emitting `{event, bearing, level}`; the pet-detect "petted" event from a click on the head. | M | ★★★ |
| 1.6 | **BLE proximity** | RSSI from distance with log-normal noise and body shadowing; shared beat clock with ±20 ms jitter; NFC as a "touched" event. | S | ★★★ |
| 1.7 | **Odometry that drifts** | Port the contact-anchored FK + yaw-integration scheme; expose estimate and truth so the trail overlay shows the gap. | M | ★★★★ |
| 1.8 | **Noise console** | Live sliders for every sensor's noise, latency, dropout, and rate (`POST /sensors/{duck}/noise`), plus "presets": *ideal*, *datasheet*, *hostile*. | S | ★★★★ |
| 1.9 | **Sensor contract tests** | Lock frame shapes, rates, and the ToF zone geometry against the datasheet so a future lab and a future bridge agree. | S | ★★★★ |

### Track 2: The brain layer (upstream's M9, prototyped here)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 2.1 | **Brain runtime** | 10–30 Hz tick decoupled from physics, last-value-wins sensor cache with ages, intent bus to the reflex command block + policy slot + speaker. Pluggable `Brain` interface. | M | ★★★★★ |
| 2.2 | **Scripted FSM** | The 16 states from the idea doc over an energy/mood model; each state a small function. Start with Chill / LookAround / Wander / TurnInPlace / Startle / Petted / BallPlay. | M | ★★★★ |
| 2.3 | **Hand-written controllers** | `follow(bearing, distance)`, `avoid(tof)`, `goto(odom target)`, `sweep_scan(head yaw)`. Simple, readable, the baselines RL is measured against. | S | ★★★★★ |
| 2.4 | **Brain inspector** | Live FSM graph, input freshness bars, intent traces, "why did it transition" log. | M | ★★★★★ |
| 2.5 | **Brain packaging** | A brain is a directory: `brain.json` (kind, params, feature spec) + optional `brain.onnx`. Assignable from the palette like a policy; shareable on the Hub. | S | ★★★ |
| 2.6 | **LLM tool-calling brain (optional)** | A brain that exposes intents as tools to a language model at ~1 Hz with the FSM underneath for reflexes. Off by default; interesting for the community, not for the core. | M | ★★ |

### Track 3: Learned skills with exteroception (brain RL)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 3.1 | **`BrainEnv`** | Gymnasium env: obs = sensor features (64 ToF cells, detector tuple, odom delta, last intents, ~80 dims), action = `[vx, vy, vyaw, head_yaw, head_pitch]` at 10 Hz, frozen reflex ONNX underneath, `ForkVecEnv` parallelism as today. Domain randomization on sensor noise/latency, not physics. | M | ★★★★★ |
| 3.2 | **Follow-me** | Scenario: a person capsule walks a random path; reward = keep 0.5–0.8 m and bearing near zero, penalize losing sight, collisions, jerky intents. Compare RL vs the scripted controller in the same scenario with the same noise preset. | M | ★★★★★ |
| 3.3 | **Obstacle avoidance from ToF only** | Wander without collisions in procedural rooms; the classic "learn a policy over a depth image" lesson at 64 pixels. | M | ★★★★ |
| 3.4 | **Go-to under odometry drift** | Reach a target given only drifting odometry; lesson on why closed-loop sensing beats dead reckoning. | S | ★★★ |
| 3.5 | **Hierarchical brain** | Add a discrete head to the brain action: which reflex policy to run (walk / stand / kick / ground-pick). Needed for soccer kicking and for "sit when petted". | M | ★★★★ |
| 3.6 | **Brain teach panel** | Same UX as tricks: plain-English recipe cards, sliders, live snapshots hot-loaded on the trainee duck in `/sim`. Reuse `TeachPanel` with a brain behavior family. | M | ★★★★ |
| 3.7 | **Head-aware locomotion (reflex side)** | A walk policy fine-tuned so head-pose commands are honoured while walking (the shipped walker only sees keep-alive ranges). Trained on the existing `train-walk` path with wider `HEAD_CMD_RANGES`, still inside the contract. Ports to `microduck_rl` as an mjlab cfg change. | M | ★★★★ |

### Track 4: Multi-duck and soccer

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 4.1 | **Multi-duck world** | N ducks in one `mjData` with duck–duck collisions; per-duck reflex + brain; palette drag onto any duck. | (0.1/0.2) | ★★★★★ |
| 4.2 | **Soccer pitch scenario** | Pitch, two goals, the 70 mm / 15 g ball from upstream, spawn slots, out-of-bounds reset, scoreboard, match clock. | M | ★★★★★ |
| 4.3 | **Ball perception** | Detector class `ball` (bearing, size ⇒ distance) plus ToF blob; honest label in the UI that the real robot cannot see a ball until someone trains that detector class. | S | ★★★★ |
| 4.4 | **Striker brain (RL)** | 1v0: dribble toward the goal; reward = ball progress toward goal, being behind the ball, no collisions; discrete kick selection via 3.5. | M | ★★★★★ |
| 4.5 | **Self-play ladder** | 1v1, then 2v2 with parameter sharing; league of past checkpoints as opponents; ELO in the scoreboard; the lesson is non-stationarity. | L | ★★★★★ |
| 4.6 | **Team play tooling** | Team assignment UI, role tags, possession and heatmap stats, replay of goals, "possess a duck and play against the brains". **Done (first form):** `Duck.team`, `make_pitch(per_side)`, a team blackboard with attacker/support roles (`brain/team.py`), `pitch-2v2` / `pitch-3v3` built-ins, `eval-pitch --per-side`; measured 2v2 2.25 goals, 10.8 kicks, 3.0 falls a run, 3v3 1.00 / 5.8 / 5.25 (falls per duck climb with the roster: 0.25 → 0.75 → 0.88). | M | ★★★★ |
| 4.7 | **Goal / pitch sensing honesty** | Options: known pitch + drifting odometry (real-ish), or detector classes for goal markers. Expose the choice in the scenario; teach why it matters. | S | ★★★ |
| 4.8 | **Flocking / follow-the-leader** | Upstream's sketch verbatim: RSSI holds spacing, ToF handles the duck ahead; then a learned version. Cheap once 1.2 + 1.6 exist. | S | ★★★ |
| 4.9 | **Synchronized dance** | Shared BLE beat drives head-bobs across ducks; the speaker plays each duck's voice on the beat. Pure delight, ten lines once 1.6 and 6.x exist. | S | ★★ |

### Track 5: Scan and map a room

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 5.1 | **Occupancy grid from ToF** | Reproject 64 zones through head + trunk kinematics (as `tofd` consumers must) into a 2–5 cm grid with log-odds updates; render as a floor overlay. | M | ★★★★★ |
| 5.2 | **Head sweep scanning** | `sweep_scan` controller: stand and yaw the head through its range, tilting to cover floor and obstacles; "lighthouse" mode; the ToF's FOV limits become visible. | S | ★★★★ |
| 5.3 | **Truth vs estimate** | Ground-truth map from the scenario next to the built map; coverage % and error metrics on the lesson strip. | S | ★★★★ |
| 5.4 | **Exploration** | Frontier-based explorer (scripted) then RL with a novelty grid (the idea doc's Wander memory); coverage-per-minute leaderboard. | M | ★★★★ |
| 5.5 | **Drift correction (light SLAM)** | Scan-to-map matching to correct yaw drift; optional, the lesson is "here is why real robots need loop closure". **Done (first form):** wall-line matching per frame (`brain/mapping.py`), pose error under a 1.5°/s bias 0.21 → 0.12 m, wall cells within 10 cm 0.64 → 0.85; a correlative search was measured to trade yaw for sideways error with a 45° FOV. | L | ★★★ |
| 5.6 | **Map export** | Save the grid + trajectory; replay a real `tof.stream` log through the same mapper (section 10). | S | ★★★ |

### Track 6: Audio and voice

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 6.1 | **Per-duck voice in the browser** | WebAudio synth with pitch / level / vowel (the same three knobs as `sounds/`), seeded from the duck id so each duck keeps its voice; spatialized by scene position. | S | ★★★ |
| 6.2 | **Brain → speaker intents** | The FSM's coos, startles, greetings play in the viewer; the ToF theremin mode as a toy. | S | ★★ |
| 6.3 | **Sound events as inputs** | Clap / voice / other-duck sources feed 1.5; "turn toward the sound" state. | S | ★★★ |
| 6.4 | **Pet-detect in sim** | Click-and-hold on a head fires the petted event with the real classifier's hysteresis timing; ships the "Petted" state. | S | ★★ |
| 6.5 | **Voice greeting memory** | Persisted list of duck ids met; a warmer sound for a friend (from the idea doc). | S | ★ |

### Track 7: `/sim` UX, debugging, visualization

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 7.1 | **Overlay layers** | One toggle per idea (ToF, detector, sound, BLE, odom, grid, score); persisted; screenshot-friendly. | S | ★★★★★ |
| 7.2 | **Sensor inspector** | 8×8 heatmap with the same cells lit in 3D on hover; detector bearing needle; event timeline; RSSI sparkline. | M | ★★★★★ |
| 7.3 | **Freshness everywhere** | Every sensor value shows its age; stale turns amber then red; brain transitions caused by staleness are logged. | S | ★★★★ |
| 7.4 | **Timeline scrub + step** | Frame-step forward/back, jump to events (goal, fall, transition), export a clip of the range with the capture pipeline. | M | ★★★★★ |
| 7.5 | **Intent tracer** | Plot of commanded vs achieved twist per duck (the lab already knows both), with the deadman visible. | S | ★★★ |
| 7.6 | **Compare mode** | Two brains in the same scenario with the same seed side by side (the current page's checkpoints trick, applied to brains). | M | ★★★★ |
| 7.7 | **Reward and feature probes** | Click a term or a feature and see it drawn in the scene (the bearing that a reward scores; the ToF cells a feature averages). Enforces "never reward what the policy cannot observe". | M | ★★★★ |
| 7.8 | **Perf HUD** | RTF, per-duck sensor µs, brain tick µs, physics µs, WS bytes/s; the stats strip already exists, extend it. | S | ★★★ |
| 7.9 | **Keyboard and possess** | `P` possess, `1–9` select, `G` toggle grid, `T` toggle ToF, `L` lesson card. | S | ★★★ |
| 7.10 | **Render-rollout for `/sim`** | Extend `render-rollout` and its contact sheet to world scenarios with sensor overlays burned in, so agents (and the `watch-training` skill) can *read* what a brain did. | M | ★★★★ |

### Track 8: Performance

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 8.1 | **One world, one step** | N ducks in one `mjData` costs one `mj_step` per substep instead of N; measure against today's per-duck envs. | (0.1) | ★★★★ |
| 8.2 | **Sensor cost budget** | ToF at 15 Hz × 64 rays via `mj_multiRay`; detector at 10 Hz with one occlusion ray per candidate; target < 5% of a core for 8 ducks. Bench + test. | S | ★★★★ |
| 8.3 | **Binary framing + delta compression** | Sensor arrays binary; static world geometry sent once; ball/boxes as deltas. | S | ★★★ |
| 8.4 | **Vectorised brain training** | `BrainEnv` under `ForkVecEnv` with `envs_per_worker` packing (worlds are heavier than single ducks); `bench-envs` support for world envs. | M | ★★★★ |
| 8.5 | **Viewer instancing** | Instanced meshes for walls/boxes; keep the no-shadow rule; grid overlay as one textured quad updated from a `Uint8Array`, not per-cell meshes. | S | ★★★★ |
| 8.6 | **Replay off the main loop** | Recorder writes in a thread; replay serves from disk without a live world. | S | ★★★ |

### Track 9: Windows and cross-platform (detail in section 9)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 9.1 | **`spawn` vec-env backend** | Works on Windows; per-worker model compile or pickled `MjModel` handoff; parity test vs `fork`. | M | ★★★★ |
| 9.2 | **Process-tree control without signals** | Replace SIGTERM plumbing with `psutil` tree terminate; already a dependency. | S | ★★★★ |
| 9.3 | **Rendering backend** | `MUJOCO_GL=wgl`/glfw path for `render-rollout` on Windows; document. | S | ★★★ |
| 9.4 | **Scripts → entry points** | The three skills' bash scripts become `uv run lab-restart` / `lab-teach` / `lab-watch` Python entry points with PowerShell one-liners in the skill docs. | S | ★★★ |
| 9.5 | **Viewer script portability** | `next dev -p ${PORT:-63317}` is bash-only; use a node launcher. | S | ★★★ |
| 9.6 | **Cross-OS CI** | GitHub Actions matrix macOS / Windows / Ubuntu running the contract tests (clone `microduck_rl` in CI). | S | ★★★★ |
| 9.7 | **Bench on Windows** | Re-run `bench-envs`; thread settings were tuned on Apple P/E cores. | S | ★★ |

### Track 10: Education

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 10.1 | **Lesson cards per scenario** | Short, in-page explanations tied to what is on screen: "why the policy is blind", "what 64 pixels can and cannot tell you", "why odometry drifts", "why self-play is unstable". | S | ★★★★ |
| 10.2 | **Guided tours** | Click-through walkthroughs that toggle overlays and possess agents at each step. | M | ★★★ |
| 10.3 | **Challenges** | Obstacle course, follow test, mapping coverage, soccer ladder; a scenario declares its metric; results export to the Hub. | M | ★★★★ |
| 10.4 | **Docs** | `/sim` README, `AGENTS.md` rules for brains ("never feed the brain a feature the robot cannot produce"), and a skill for reading `/sim` contact sheets. | S | ★★★★ |

### Track 11: Sim2real bridge for brains (section 10)

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 11.1 | **JSON-RPC client** | Python client for the robot's API (`robot.move`, `robot.head`, `robot.subscribe`, `tof.stream`) over WebSocket / the WebRTC control channel, mirroring the sim's intent bus. | M | ★★★★ |
| 11.2 | **Real-log replay** | Play a recorded `tof.stream` + odometry log into the sensor inspector and the mapper; the first thing to do when hardware arrives. | S | ★★★★ |
| 11.3 | **Hub round-trip** | Import `pollen-robotics/microduck-policies` into the palette; publish brains and scenario results to the user's Hub account with the existing BYOK token. | M | ★★★ |
| 11.4 | **Detector dataset export** | Sim frames + labels (ducks, ball, person) as a Hub dataset for training the NPU detector classes the robot does not have yet. | M | ★★★ |

### Track 12: Tidy the playroom (pick up, carry, deliver to a bin)

The request: bring the duck into a room full of scattered toys (bricks,
socks, blocks), point at a low basket, say "tidy", and have it find each
pickable object, pick it up with its beak, carry it to the basket, drop it,
and repeat until the floor is clear. Upstream already ships a ground-pick
policy; this track is everything around it.

**What upstream gives us, precisely.** The ground-pick policy is a *blind,
phase-driven* 4-second cycle: the command slots carry `[cos 2πφ, sin 2πφ, 0]`
and the reward walks the mouth tip to the floor (descent 0–1.5 s), holds
briefly, then rises with a simulated 10–40 g payload pulling on `mouth_tip`.
The MJCF has no actuated jaw ("the passive jaw joints are no longer part of
the articulation"); on the robot the beak is motor slot 9 of 15, driven outside
the policy and undocumented. So the real primitive is "crouch until the beak
touches the floor under the head, then rise", and the operator aims the robot
so the object is under the beak. Everything else in the loop is a *brain*
problem, which is exactly what the rest of this roadmap builds.

**Where the computation should live.** Three tiers, chosen by rate:

| Rate | What | Where | Why |
|---|---|---|---|
| 50 Hz | Walk, carry-walk, ground-pick, stand (reflex ONNX) | Onboard `robotd` | Already true by architecture; `robotd` never blocks on another service. |
| 5–15 Hz | Detector (objects, basket, duck), ToF, odometry, the tidy FSM, visual-servo approach | Onboard (NPU + CPU) **or** tethered Mac during development | A 320 px INT8 YOLO at ~26 ms is what the NPU already does for ducks; adding classes is a retrain, not a new capability. The FSM is trivial. A tether adds 100–300 ms, fine at these rates. |
| ≤ 1 Hz | Open-vocabulary designation ("the blue bin", "everything that is a toy"), map memory, planning, data logging | Tethered Mac or cloud | Where a VLM earns its keep: one call per scan, results cached in the map. |

A full **end-to-end VLA** (pixels and language in, 14 joint targets out at
50 Hz) is the wrong tool here: the deployment contract is *intents*, not
joints; `robotd` is authoritative on safety; a 1 GB RK3566 cannot host one;
and the duck has no joint-level teleop to collect the demonstrations a VLA
needs. What fits is a **VLA over skills**: a language/vision model at the
brain tier whose actions are `goto(object)`, `pick`, `carry_to(bin)`,
`release`, with the skills below doing the physics. In the lab this is 2.6
plus a skill vocabulary. For the first version, a detector plus the scripted
FSM does the whole loop without any language model, and the lab should prove
that first; the VLM is an upgrade for designation and open-vocabulary
"pickable", not a prerequisite.

**The honest hard parts.**

- *Grasp physics.* A beak grasp of a brick or a sock is a contact problem the
  sim cannot cheaply model (deformables, jaw servo, friction). Plan: model
  grasp as an **attachment event**: when the beak is closed with the mouth tip
  within a tolerance of an object's grasp point, a weld constraint attaches
  it, with success probability as a function of alignment error and object
  class. Those probabilities are a *dataset* to learn from real attempts
  (12.8), not a physics claim. The UI labels it as such.
- *Alignment.* Ground-pick is blind; the approach must put the object under
  the beak to ±2 cm. That is a visual-servo controller over the detector
  bearing + ToF floor blob, then a final blind dead-reckoned step, exactly the
  kick task's "±2 cm placement noise" in reverse.
- *Carrying while walking.* The walker never trained with a payload in the
  mouth or the head pitched to hold one. Needs a carry-walk reflex (a
  `train-walk` variant with a mouth payload DR and a held head pose; ports to
  an mjlab cfg), plus a check that walking with the head down does not tip the
  gait. Bricks (2–10 g) are easy; a sock is a pendulum.
- *Finding the basket again.* Odometry drifts, so "return home" is not enough
  across a room. The basket needs a re-acquirable signature: a printed marker
  the NPU can detect, a distinct colour/shape class in the detector, or the
  spot the duck was NFC-tapped at, refined visually on approach. The rim must
  sit below the beak (roughly ≤ 10 cm for a 25 cm duck standing), so "short
  basket" means a tray or a cut-down bin, and the release is "head over rim,
  open beak".
- *Pickable or not.* Size class from bbox + ToF distance, on the floor plane,
  not attached to anything, not the duck; plus a learned graspability score
  from attempts. Bricks yes, a plush yes, a book no.
- *Coverage.* The mapping track (5.x) turns "scan the room" into a frontier
  sweep with remembered object positions in the grid, which becomes the work
  queue; objects that keep failing get demoted to the end of it.

| # | Task | What | Size | Value |
|---|---|---|---|---|
| 12.1 | **Playroom scenario** | Clutter generator (bricks, blocks, socks, balls, a plush), a low-rim basket with a marker, spawn slots, tidy score = objects in basket / total plus time and drops. | M | ★★★★★ |
| 12.2 | **Beak grasp as attachment** | Closed-beak intent + mouth-tip tolerance ⇒ weld to the object; release intent breaks it; per-class success curve vs alignment error; payload mass rides along for the carry-walk. Labelled as a model, not physics. | M | ★★★★★ |
| 12.3 | **Ground-pick in the lab** | Run upstream's ground-pick ONNX (or a lab-trained one via `train-behavior`) as a reflex skill with its 4 s phase in the command slots; verify the return with the payload. | S | ★★★★ |
| 12.4 | **Visual-servo approach** | Detector bearing + ToF blob ⇒ align the object under the beak; final blind step; success measured against the ±2 cm window. Scripted first, then RL over approach offsets (3.1-style). | M | ★★★★★ |
| 12.5 | **Carry-walk reflex** | `train-walk` variant with mouth payload DR (10–40 g, matching upstream) and a held head pose; eval: falls per metre while carrying. Ports to an mjlab cfg. | M | ★★★★ |
| 12.6 | **Deliver and release** | Go-to basket with visual re-acquisition (marker / class), align, head over rim, open beak, verify the object left (payload gone, detector sees it in the bin). | M | ★★★★ |
| 12.7 | **Tidy FSM** | Scan → Select → Approach → Pick → Verify → Carry → Deliver → Release → repeat; retry budgets, give-up rules, a work queue in the map; the brain inspector shows it. | M | ★★★★★ |
| 12.8 | **Graspability learning** | Log every attempt (features → success) in sim and, later, on hardware; train a small classifier; feed it back into Select and into 12.2's curves. | M | ★★★ |
| 12.9 | **Basket designation UX** | Three ways, all in sim: click it in `/sim`; NFC-tap "home" (the robot has two antennas); a printed marker. On the tether: tap it on the live video. | S | ★★★★ |
| 12.10 | **Compute placement toggle** | Run the brain onboard (in the lab process) or "tethered" with simulated 100–300 ms link latency and dropouts from the noise console; the inspector shows the latency budget per tier. The lesson is *where should this run?* | S | ★★★★ |
| 12.11 | **Detector classes + export** | Add brick / sock / toy / basket / marker to the geometric detector; render a sim dataset plus real photos; the same INT8 320 px recipe upstream used for ducks, so it runs on the NPU. | M | ★★★★ |
| 12.12 | **VLM designation (optional)** | One call per scan labels pickables and the bin from a frame ("everything that is a toy, into the blue bin"); tethered or cloud; cached in the map; 2.6's skill vocabulary underneath. | M | ★★★ |
| 12.13 | **Tidy benchmark** | N objects, time to clear, success rate, drops, collisions; a 10.3 challenge with a leaderboard. | S | ★★★★ |

**Order inside the track:** 12.1 → 12.2 → 12.3 → 12.4 (first pick in sim)
→ 12.6 + 12.7 (first full loop, one object) → 12.5 (carry properly) →
12.9 + 12.10 → 12.11 (hardware path) → 12.8, 12.12, 12.13. The first
screenshot is a duck walking one brick to a tray. Everything through 12.7 is
a brain over existing reflexes plus one attachment trick, which is why this
track slots in right after follow-me in the phase plan: it reuses 1.2, 1.3,
2.1, 2.3, 3.1 and 5.1 and adds only the grasp model and the carry gait.

## 5. What to build first, and why

Ranked by leverage. Each phase ends in a demo somebody can screenshot.

**Phase 1: the world and the eyes (Tracks 0 + 1.1–1.3 + 1.8 + 7.1–7.3).**
Foundation for everything, and the first payoff is already educational:
a duck standing in a living-room scenario with its ToF frustum drawn, the
8×8 heatmap in the inspector, the cells lit on the wall, and a noise console
that makes the sensor lie on demand. This is the most valuable phase because
every later feature is a consumer of it, and because it settles the two hard
engineering decisions (one-world `MjSpec` composition; binary sensor framing)
while they are cheap to change. Demo: *"what the duck sees"*.

**Phase 2: a brain and follow-me (Tracks 2.1–2.4 + 2.3 + 3.1–3.2 + 0.6).**
The first end-to-end behavior that uses sensing, and the one with the most
direct path to the real robot (a duck detector exists on the NPU today; a
person class is one retrain away). Scripted follow first, then RL follow in
the same scenario under the same noise preset, side by side. The record /
replay timeline lands here because the first "why did it lose me?" question
needs it. Demo: *possess the person, walk around, the duck follows; flip the
noise to "hostile" and watch the scripted controller fail where the trained
brain copes*.

**Phase 2b: tidy the playroom (Track 12).**
The first *useful* behavior, and the one most people asked for once they saw
the beak. It is follow-me's brain pointed at a brick plus one honest trick
(grasp as an attachment event) and one reflex variant (carry-walk). Demo:
*point at a tray, scatter five bricks, watch the duck clear them*.

**Phase 3: many ducks and soccer (Track 4).**
Highest wow and the strongest community pull, but only worth doing once
sensing and brains exist, because a soccer duck is a follow-me brain with a
ball class, a goal, and an opponent. Start with 1v0 dribbling, then a
self-play ladder. It teaches non-stationarity and credit assignment, the two
multi-agent lessons no single-duck task can. Demo: *2v2, humans possess one
duck per team against the brains, replay the goals*.

**Phase 4: map the room (Track 5).**
Lower wow than soccer but the cleanest robotics lesson in the set: what a 64
pixel depth sensor on a nodding head can reconstruct, and how far odometry
drift gets you before you need loop closure. Demo: *sweep-scan, watch the
grid fill, compare to truth, then explore*.

**Phase 5: sound, voice, social (Track 6 + 4.8–4.9).**
Cheap once the brain exists and very on-brand for the product (every duck has
its own voice), but it advances the platform least. Do it as the polish pass.

**Throughout: Windows (Track 9) and education (Track 10).**
Windows support is mostly a matter of not adding POSIX-only code to the new
modules (section 9); the `spawn` backend and CI matrix should land during
Phase 1 so nothing accretes. Lesson cards are written with each scenario, not
afterwards.

The features that expand the *platform* most are 0.1 (one world), 1.1 (ray
rig), 2.1 (brain runtime), 3.1 (`BrainEnv`) and 0.6 (record/replay): each is
a primitive that many later tasks are a thin layer over. The features that
*educate* most are 1.8 (noise console), 2.4 (brain inspector), 5.3 (truth vs
estimate) and 4.5 (self-play ladder). The features with the most *wow* are
soccer, follow-me, and the per-duck voices.

## 6. Teaching with it

Each scenario carries one lesson, and the page makes the lesson visible
rather than explained:

| Scenario | The lesson | What the screen shows |
|---|---|---|
| Empty floor + ToF | Partial observability. The reflex policy is blind by design; the brain sees 64 numbers. | ToF frustum, heatmap, the wall cells lighting up; the obs vector with slots 48–60 highlighted as "the only door in". |
| Noise console | Sim2real is a perception gap. | The same brain under *ideal* / *datasheet* / *hostile* presets; freshness bars going amber. |
| Follow me | Classical control vs learned control; POMDPs with dropouts. | Scripted vs RL side by side, same seed; the moment the detector drops out and each one reacts. |
| Obstacle course | Learning from a depth image; reward and feature probes. | The feature the reward reads drawn in the scene ("never reward what the policy cannot observe"). |
| Map the room | Sensing geometry, dead reckoning, and drift. | Truth trail vs odometry trail; grid coverage %; loop-closure toggle. |
| Soccer 1v0 → 2v2 | Hierarchical control, then self-play and non-stationarity. | Discrete kick selection lighting up; ELO over league generations; a checkpoint that beats its parent and loses to its grandparent. |
| Dance / voices | Distributed timing, identity from a seed. | Shared beat jitter; each duck's voice parameters. |

Every lesson has a "look, don't trust the curve" step, in keeping with the
verification discipline in `microduck_local/AGENTS.md`: render the rollout
with overlays, read the contact sheet, then believe the metric.

## 7. UX, debugging, and visualization principles

- **Every number has a picture and every picture has a toggle.** A sensor
  value is drawn where it happens in the scene, in the inspector, and on the
  timeline; each overlay is one switch so a screenshot can carry one idea.
- **Show freshness, not just value.** Age is the most common cause of a
  brain doing something odd. Every input is stamped and coloured by age.
- **Truth next to estimate.** The lab knows the truth; the brain does not.
  Draw both (trails, maps, ball position) so the gap is the thing you look at.
- **Scrub before you speculate.** The recorder is always on (ring buffer);
  jumping to "the fall at 01:12" is one click. Replays keep the overlays live.
- **Same seed, side by side.** Comparing two brains or two noise presets
  under one seed is the standard experiment; the UI makes it the default.
- **Possess anything.** Driving the person or a duck yourself is the fastest
  intuition for what a sensor delivers and what an intent does.
- **Honest labels.** When a simulated sense does not exist on the robot yet
  (ball class, person class, goal markers), the UI says so on the sensor
  card, the way the current page marks assisted stretches with a spotter tag.
- **Keep the stage light.** Instancing, no shadow maps, DOM labels, one quad
  for the grid; the viewer README's context-loss story still applies.
- **Agents can read it too.** `render-rollout` grows scenario support and
  burned-in overlays so the existing skills keep working for `/sim` runs.

## 8. Performance plan

Budgets, to be measured with `bench-envs`-style tooling before any default
changes (the repo's rule: throughput is not learning speed).

| Item | Budget | Basis |
|---|---|---|
| World physics, 8 ducks + ball | ≤ 1 core at 50 Hz real time | One `mj_step` per substep for the whole world; today 8 separate ducks cost ~5% of a core, so headroom is large. |
| ToF, per duck | 64 rays × 15 Hz via `mj_multiRay` | Sub-millisecond per call; ~1 ms/s per duck. |
| Detector, per duck | ≤ 8 candidates × 1 occlusion ray × 10 Hz | Geometric, no rendering. |
| Brain tick | ≤ 0.2 ms scripted, ≤ 1 ms ONNX | Small MLP; runs inside the lab loop. |
| WS bandwidth | ≤ 200 kB/s at 8 ducks | Bodies as today; 64 × uint16 depth per duck binary; world deltas. |
| Brain RL throughput | ≥ 10k brain-steps/s at 32 workers | One brain step = 5 physics-control steps of a frozen ONNX walker; same worker fleet as `train-behavior`. |
| Viewer | 60 fps at 8 ducks + room | Instanced statics, single grid quad, no new per-frame React renders. |

Two measurements to take early: (1) world-with-N-ducks vs N single-duck envs,
to confirm the one-world model is a win and not a contention problem in
`ForkVecEnv`; (2) `MjSpec` compile time for a room scenario, since the lab
recompiles on scenario load and it should feel instant.

## 9. Windows support

macOS stays first, but nothing in the plan is Mac-specific. What is
POSIX-specific today, and the fix:

| Today | Problem on Windows | Plan |
|---|---|---|
| `vec_env.py` `fork` backend (one compiled `MjModel` inherited copy-on-write) | No `fork` on Windows. | Add a `spawn` backend: workers receive the MJCF path and compile privately (the old `subproc` cost: ~100 MB and a few seconds per worker), or receive a pickled `MjModel` (the Python bindings pickle models) to skip the compile. Semaphores and `multiprocessing.shared_memory` both work under `spawn`. Parity test against `fork` on macOS. |
| `viz_server.py` SIGTERMs the trainer and its workers | No POSIX signals. | `psutil` process-tree terminate (already a dependency), one helper used on all platforms. |
| `render-rollout` relies on CGL on macOS and documents EGL/OSMESA for Linux | Neither exists on Windows. | Support `MUJOCO_GL=wgl` (GLFW window context, the default on Windows desktops) and document it; the contact sheet path is pure PIL. |
| `.claude/skills/*/*.sh` bash scripts | No bash by default. | Python entry points (`lab-restart`, `lab-teach`, `lab-watch`) plus PowerShell one-liners in the skill docs. |
| `duck-viewer` `npm run dev` uses `${PORT:-63317}` | bash-only expansion. | A tiny node launcher or `cross-env`. |
| `hf-token.json` written mode `0600` | `chmod` is a no-op on NTFS. | Document, and set an ACL where practical. |
| MPS update path | Mac only. | Already auto-disables elsewhere; CUDA on Windows could be added later as a measured opt-in. |
| Thread pinning tuned on P/E cores | Different scheduler. | Re-run `bench-envs` on a Windows box and record it. |

Rules for new code so this does not regress: no `os.fork`, no signals, no
`/tmp`, `Path` everywhere, `multiprocessing.get_context("spawn")`-safe
picklable arguments, and a CI matrix (macOS, Windows, Ubuntu) running the
contract tests from day one of `/sim`.

## 10. Sim2real for the brain layer

For the reflex policies, the honest path stays what `AGENTS.md` says: port
the env design to `microduck_rl`, retrain on GPU, ship from there. For the
brain layer the story is better, because the brain speaks the robot's
intents:

1. **Same interface both ways.** `brain/bridge.py` swaps the sim's sensor
   cache and intent bus for the robot's `tof.stream` subscription,
   `robot.subscribe` telemetry (odometry, active policy, safety verdict), and
   `robot.move` / `robot.head` notifications. Detections arrive once upstream
   publishes them from `mediad`; until then the bridge exposes the same
   field as "unavailable" so the brain's freshness gating does its job.
2. **Log first, run second.** Record real `tof.stream` and odometry, replay
   into the sensor inspector and the mapper, and compare against the sim's
   noise presets. Tune the presets to the data before trusting a trained
   brain on hardware.
3. **Respect the robot's authority.** `robotd` owns fall detection, limits,
   and the deadman; the bridge never sends joint targets and never assumes an
   intent was applied (telemetry carries `move.applied`).
4. **Hub as the exchange.** Policies come from `microduck-policies`; brains
   and scenario results go back under the user's own token, as the README
   already promises for GPU jobs.

## 11. Open questions and assumptions

- **Is there a scanning LiDAR?** The onboard repo has only the 8×8 ToF. If
  final hardware adds a planar LiDAR, the ray rig (1.1) makes it a config,
  and the mapper (5.1) gets a much easier job. The plan is written for the
  ToF and gets better, not different, with a LiDAR.
- **Detector classes.** Only `duck` exists on the NPU today. Person and ball
  classes are needed for follow-me and soccer on hardware; the lab flags
  them as simulated-only and can export a training dataset (11.4).
- **Detection publication.** Upstream has not decided the IPC shape for
  detections (a `media.frame` call vs publishing from `mediad`). The bridge
  mirrors the sim's `{class, bearing, elevation, bbox_w, conf}` and adapts
  when upstream lands one.
- **Head-pose commands while walking.** The shipped walker was trained with
  keep-alive head ranges only; a brain that steers the gaze while walking
  will need 3.7 (or the upstream head-pose curricula) to be honoured.
- **Multi-duck on hardware** is BLE + ToF only; soccer on real ducks is a
  long way off and the page should say so.
- **`MjSpec` composition of the upstream MJCF.** Attaching several copies of
  `robot_walk.xml` with prefixed names and separate keyframes needs a spike
  early in Phase 1; it is the riskiest single technical item. The three
  MuJoCo APIs the plan leans on (`mj_multiRay` for the ray rig,
  `MjSpec.attach` for world composition, and pickling a compiled `MjModel`
  for the `spawn` backend) were checked to exist in the pinned MuJoCo 3.10.0
  while writing this; what has not been checked is how well `attach` handles
  this particular robot file.
