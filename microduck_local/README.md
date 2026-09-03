# microduck_local

Local CPU-MuJoCo PPO training for Microduck on Apple Silicon — the
[jenga-stacker](https://github.com/jonathanhawkins/jenga-stacker) recipe
(Stable Baselines 3 + CPU MuJoCo), scaled for locomotion and made
contract-faithful to the official stack.

**Why this exists:** [microduck_rl](../microduck_rl) trains through MuJoCo Warp,
which requires a CUDA GPU — on a Mac, warp reports `devices: ['cpu']` and
training cannot start. This project keeps the same MJCF models, the same
61-obs / 14-action contract and the same 50 Hz timing as
`microduck_rl/scripts/infer_policy.py`, and lets you iterate locally.

**Actuator model** — pick with `actuator="xml"|"bam"` or `MICRODUCK_ACTUATOR`:

| | `xml` (default) | `bam` |
|---|---|---|
| what | MJCF `position` servos, kp=0.55 | the xl330/m6 voltage model microduck_rl trains with |
| matches | `infer_policy.py` deployment rehearsal | `microduck_constants.py` `FrictionDRBamActuatorCfg` |
| speed | ~6.4k env-steps/s | ~3.3k env-steps/s |

The XML class is *literally* the BAM model run through `to_mujoco()` at
vin=7.5 V (every one of its five numbers reproduces — see
`tests/test_bam_actuator.py`), so it is the small-signal linearization. It
drops the 1.75 A firmware current limit (peak torque 0.96 vs BAM's 0.64 Nm),
turns back-EMF into a passive damper that *adds* to the drive when a joint is
back-driven (1.60 vs 0.70 Nm at −12 rad/s), and drops m6's load-dependent
gearbox friction (~27% of motor torque) along with battery sag and the 3–6
step bus lag. Inside the linear region (|error| ≲ 0.5 rad) the two agree to
~2%; they diverge on hard transients and impacts. `bam` is the honest choice
when a maneuver saturates the servos; `xml` is faster and unchanged.

**What it is for:** prototyping — new behaviors, reward designs, obs experiments,
curriculum ideas — with minutes-long feedback loops. Validated: the shipped
`alpha_walking.onnx` (trained on the official stack) runs upright in this env
with 0 falls (that's a permanent regression test), and exported ONNX is drop-in
compatible with `infer_policy.py`.

**What it is NOT for:** the final policy you put on the robot. Even under
`actuator="bam"` this is single-env CPU with a subset of microduck_rl's DR
stack, and microduck_rl *is* the sim2real recipe (read its AGENTS.md). Once a
behavior works here, port the env design to an mjlab cfg and retrain on GPU:
`uv run train <TASK> --hf-jobs`.

## Commands

```bash
uv sync                                     # one-time (needs ../microduck_rl checked out at the pinned sha —
                                            # ../scripts/setup.sh does the clones, the pins, uv and npm in one go)
uv run --with pytest pytest tests/          # contract tests — run before training
uv run bench-walk                           # raw env-stepping throughput
uv run bench-envs                           # PPO throughput vs --envs → the right worker count
uv run train-walk --envs 12 --steps 3_000_000 --run-name my-run
MICRODUCK_ACTUATOR=bam uv run train-walk --envs 12 --run-name my-run-bam  # BAM physics
uv run export-walk runs/my-run              # → runs/my-run/policy.onnx (normalizer baked)
uv run eval-walk runs/my-run/policy.onnx    # headless: falls, tracking error
uv run eval-walk ../microduck/policies/alpha_walking.onnx   # baseline comparison

# LOOK at what a policy does — mp4 (for you) + a captioned frame contact sheet
# (for an AI assistant to read). See .claude/skills/render-rollout/SKILL.md.
uv run render-rollout --policy runs/my-run/policy.onnx --out /tmp/rr --episodes 2
uv run render-rollout --policy limp --behavior backflip --out /tmp/rr-null  # null control

# watch it in the viewer (the official rehearsal tool):
cd ../microduck_rl && uv run scripts/infer_policy.py \
    --walking ../microduck_local/runs/my-run/policy.onnx --new-cmd-obs

# training curves:
uv run tensorboard --logdir runs/
```

Measured on an M5 Max: ~26k control-steps/s raw env throughput at 12 workers
(run `uv run bench-envs` to get the numbers for your machine).
Continue a run with `--init-from runs/my-run`.

## How many envs? (`uv run bench-envs`)

`bench-walk` measures raw env *stepping*, which scales nearly linearly and so
always answers "more workers". Training is not that shape: PPO alternates a
parallel rollout with a **serial** gradient update, and the update's share grows
with the env count because the rollout buffer does (`n_steps=256` per env).
`bench-envs` measures the number that actually sets how long a run takes —
env-steps/second sustained through real `model.learn()`, rollouts *and* updates
— on the real training path (`BehaviorEnv` + `train_behavior`'s PPO).

Measured 2026-08-30 on an 18-CPU M5 Max (6P + 12E), `behavior=run`, `fork`
backend, best of 3 interleaved repeats on an otherwise idle machine (each point
reproduced within 2%):

| envs | steps/s | steps/s per env | scaling efficiency | 1M steps |
|---:|---:|---:|---:|---:|
| 4 | 6,836 | 1,709 | 100% | 2.4 min |
| 8 | 10,074 | 1,259 | 74% | 1.7 min |
| 10 | 11,072 | 1,107 | 65% | 1.5 min |
| 12 | 12,327 | 1,027 | 60% | 1.4 min |
| **16** | **14,288** | **893** | **52%** | **1.2 min** |
| 18 | 14,529 | 807 | 47% | 1.1 min |
| 20 | 15,068 | 753 | 44% | 1.1 min |
| **24** | **15,626** | **651** | **38%** | **1.1 min** |
| 32 | 16,484 | 515 | 30% | 1.0 min |
| 48 | 17,031 | 355 | 21% | 1.0 min |
| 64 | 17,053 | 266 | 16% | 1.0 min |

**More workers than cores is correct here.** The curve is still climbing at 24
workers on an 18-core machine, because the rollout (worker-bound) and the update
(trainer-bound, multi-threaded torch) take turns — extra workers fill the cores
the update phase leaves idle. Capping workers at `cpu_count` (18) is 7% slower
than the 24-worker knee; the old default of 10 was 29% slower than the new 16.

**It saturates, though.** 24 envs is 92% of the ~17.1k asymptote on an idle
machine; 32 is 97%, and that is the shipped default (`viz_server.BASE_ENVS`
and `--envs` alike). An earlier live-lab test seemed to invert the curve —
16 envs held 10.0k steps/s (`teach-run-79675b`) while "26 envs" held 6.8k
(`teach-run-be11cc`) — but the 26-env run got its extra envs from 5 helper
ducks, i.e. five extra 50 Hz viewer sims in the lab process: a confound, not
an inversion. Re-measured with helpers as viewers only (2026-08-30, lab +
browser live), the idle-machine ordering holds and widens: 16 → 4.7k, 24 →
5.4k, 32 → 6.5k. At 16 envs the workers are ~11% busy — the parent's serial
per-vec-step work is what extra envs amortize.

Things that did **not** help, both measured:

- **`torch.set_num_threads(1)` in the trainer is a 21-24% regression**, not the
  classic worker-oversubscription win. The update is a real matmul workload on a
  512-256-128 MLP and wants every core it can get while the workers sit blocked
  on their pipes. The training path now sets intra-op to 8 (the measured peak
  on a 6P+12E M5 Max; PyTorch's default is 6 P-cores) and inter-op to 1
  (the train loop is a single stream — the default of 18 idle inter-op threads
  just contends). `bench-envs --pin-threads` still reproduces the 1-thread
  regression.
- **Helper ducks used to make the UPDATE longer, not the rollout.** PPO
  minibatches stayed at 1024 as `--envs` grew, so 28 helpers meant 7
  minibatches × 5 epochs instead of 4 × 5 — the serial phase grew 75% for ~8%
  more samples, which is why Activity Monitor sat at 40% with 28 workers at
  16% each. `ppo_batch_size` now holds 4 minibatches (rsl_rl's
  `num_mini_batches`) once the buffer is large enough, so 28 envs is 4 × 1792.
  16 envs is still 4 × 1024.
- **In-process env stepping never wins, even at 4 envs.** At 8 envs `dummy`
  (serial) reaches 42% of `fork` and `thread` 38%; at 4 envs they are still
  1.75x and 1.87x behind. `fork` and `subproc` tie on throughput (1.00x) — model
  sharing is a memory and startup win, not a speed one.

Caveat: these are `train-behavior`'s envs, which run with `domain_rand=False`.
`train-walk` adds domain randomization and observation noise per step and is
correspondingly slower; the *shape* of the curve is the transferable part.

```bash
uv run bench-envs                              # full sweep + recommendation (~2 min)
uv run bench-envs --envs 8,16,24 --repeats 3   # narrower, more repeats
uv run bench-envs --compare-vec --compare-threads
```

### The 2026-08-31 speed pass (measured, in order applied)

- **Semaphore IPC in `ForkVecEnv`** (`vec_env.py`): step traffic had cost two
  pipe syscalls + a pickle round-trip per worker per step; steps now travel
  entirely through shared memory with semaphore signalling. No-policy
  vec-step floor at 32 envs: **1.61 → 1.08 ms** (same load, same day).
  Exact-semantics change, covered by the existing parity tests.
- **numba-fused BAM substep kernels** (`bam_actuator.py`): ~48 numpy
  dispatches per substep (more than `mj_step` itself) became three compiled
  kernels. BITWISE ports — `exp`/`power` stay in numpy because their SIMD
  code is not reproducible by another libm; everything arithmetic fused, op
  order preserved — held to the golden float64 rollouts in
  `test_bam_perf_parity.py`. numba is optional; without it the numpy path
  remains.
- **`FastActorCriticPolicy`** (`symmetry.py`): hand-rolled diag-Gaussian
  rollout forward, 234 → 215 us at batch 32. Small and exact (log-probs match
  `evaluate_actions` to float tolerance — `test_fast_policy.py`).
- **Overlapped update** (`SymmetryPPO(overlap_update=True)`, opt-in via
  `--overlap` / `MICRODUCK_OVERLAP=1`): the PPO update on rollout k runs in a
  background thread while rollout k+1 is collected with a frozen pre-update
  policy snapshot. Hides the shorter of the two phases — measured +25%
  steps/s at 32 envs — **but the A/B says don't use it for real runs**: on
  the `run` recipe the one-update-stale data cost ~2x the reward-per-step in
  early training (two seeds, one RNG-matched with frozen code; e.g. seed 4 at
  1.45M steps: ep_rew 530 sync vs 270 overlap, ep_len 113 vs 61). At our
  lr=1e-3 the per-update KL is large enough that stale data lands outside
  the clip window and the surrogate discards it. Net wall-clock-to-quality
  is a LOSS. Left in as opt-in for throughput benchmarking, and possibly
  worth revisiting for low-lr fine-tunes where the lag is smaller.
- **Env batching** (`ForkVecEnv(envs_per_worker=k)` /
  `MICRODUCK_ENVS_PER_WORKER`): several envs per worker process, stepped
  serially inside it — semaphore ops and pipes scale with workers, not envs.
  Default 1 (identical behavior, pinned by a step-for-step parity test);
  meant for 48+ env counts where 1:1 packing costs 2 sem ops per env per
  vec-step.
- **MPS update** (`SymmetryPPO(update_device="mps")` / `--update-device`,
  default `auto`): the minibatch loop hops policy + Adam state to the GPU;
  rollouts stay on CPU where batch-32 inference beats GPU dispatch latency.
  Quality is device-independent (same seed, same curve to 3 decimal places
  at 0.5M steps); throughput flips sign with minibatch size — a LOSS at
  batch 1024 (13.5k vs 14.8k steps/s), a big win at 4096 (18k vs 12.5k on
  the real recipe) — so `auto` engages it only at batch >= 2048.

**Where that leaves the numbers** (quiet machine, 2026-08-31): stock-PPO
bench throughput 14.3k (old code, 16 envs) → 26.8k steps/s (128 envs, MPS
update, 4 envs/worker), saturating ~27k — the parent's serial per-vec-step
work is the remaining wall. A single worker's `env.step` is down to 0.135 ms
idle, so physics itself would allow ~130k; closing the gap means replacing
the SB3 rollout loop wholesale, not tuning it.

**And the number that actually matters** — reward per wall-second on the
real recipe — does NOT follow raw steps/s. Both throughput-maximizing
configs sacrificed per-step learning in A/Bs (seed-matched, frozen code):
the overlapped update (+25% steps/s, ~2x less reward per step), and 64-env
batch-4096 (+40% steps/s, ~2x less per step at lr 1e-3, mostly — not fully —
recovered by doubling lr, which this repo's history says to fear). The
quality-safe sweet spot stays **32 envs**: ~14.8k steps/s on the real
recipe, ~2.2x the 6.8k this workspace started at. Raw-throughput configs
remain available behind flags for benchmarking and for experiments that can
tolerate the trade.

## One model, many workers (`MICRODUCK_VEC_ENV`)

The compiled `mjModel` costs ~470 MB the first time a process compiles it; the
`mjData` that holds the *actual* simulation state costs ~0.9 MB. MuJoCo is built
so one read-only model backs many data, but a worker per env each compiling its
own threw that away — ~99% of per-env memory was a private copy of an
identical, never-written model.

`vec_env.py` compiles the model ONCE, in the trainer, and forks the workers so
they inherit it copy-on-write. Same N processes, same parallelism; the fleet now
costs one model. Measured (footprint = summed private bytes + the one shared
model):

| envs | before (`subproc`) | after (`fork`) |
|---:|---:|---:|
| 10 | 6,954 MB | 678 MB |
| 32 | 20,862 MB | 1,031 MB |
| 64 | 41,128 MB | 1,526 MB |

Throughput is neutral-to-slightly-positive, which is the honest result: four
paired `subproc`/`fork` PPO runs (`one_leg`, train_behavior's hyperparameters,
40k timed steps each) gave a median of +3.9% at 10 envs and +8.3% at 16, but one
pair came out at −4.5%, so treat it as "not slower" rather than as free speed.
The workers were never bottlenecked on the model. Startup is not a wash:
building the vec env drops from 7.8 s to 0.2 s at 64 envs, because no child ever
opens the MJCF.

Backends, via `MICRODUCK_VEC_ENV`:

- `fork` — **the default.** N worker processes, one shared model.
- `subproc` — the old `forkserver` path, a private compile per worker. The
  escape hatch, and the baseline the table above measures against.
- `dummy` — every env in this process, stepped serially, one model. The memory
  floor; no parallelism.
- `thread` — every env in this process, stepped by a thread pool, one model.
  MuJoCo *does* release the GIL in `mj_step` (measured 3.5-4.2x on raw physics
  threads), but a full `env.step()` is only ~40% physics and the numpy rest
  holds the GIL, so this measured ~1.0x end to end. Kept because the question
  keeps getting asked, not because it is fast. Refuses `domain_rand=True`,
  whose per-step re-assert races across threads.

Domain randomization stays per-worker under sharing: `fork` gives each child its
own copy-on-write pages, and the in-process backends re-point the model at the
stepping env's own draw (`walk_env._sync_model`). `tests/test_vec_env.py` locks
both. `actuator="bam"` cannot share a model *in-process* — it rewrites
`dof_frictionloss` every physics substep — so it raises there; under `fork` it
is fine.

The lab server shares the same way: `Duck._make_env` builds every duck inside
`shared_model_scope(exclusive=False)`, so a roster costs one compiled model per
scene rather than one per duck. Measured on a live 6-duck lab, 1520 MiB →
1037 MiB, and the marginal duck drops from ~88 MB to ~1 MB. Sharing is free of
caveats there because the lab pins `domain_rand=False` and steps its ducks
serially in one frame loop; a lab launched with `MICRODUCK_ACTUATOR=bam` falls
back to private models.

## World mode: rooms, sensors, brains (the viewer's `/sim` page)

The roster above gives every duck a private env. **World mode** composes a
whole room into ONE MuJoCo model — walls, boxes, a ball, N ducks — so the
ducks can bump into each other and a sensor on one can see another
(`world/compose.py`, MjSpec attaching the upstream robot MJCF once per duck
under an id prefix; `world/arena.py` steps every duck's reflex policy in one
`mj_step`). It is the base for everything in `docs/sim-roadmap.md`.

```bash
uv run duck-lab --world living-room          # no roster needed; built-ins:
                                             # empty-floor, wall-test, living-room
# then open the viewer's /sim page
```

What a duck senses lives in `sensors/`: `TofSensor` is the head's 8×8
time-of-flight matrix on the MJCF's `tof` site (45° FOV, 4 m, 15 Hz on a
fixed grid, uint16 mm frames shaped like the robot's `tof.stream`, with
`ideal` / `datasheet` / `hostile` noise presets you can flip live from the
inspector). It never reaches the policy: `brain/` is where senses become
intents. `Wander` is the first brain — it reads the middle columns of the
depth matrix and emits only a twist, the same `robot.move` the real robot
takes — and it drives every ToF-equipped duck in auto mode; press **P** on
the page to take the wheel yourself.

Scenarios are JSON in `scenarios/` (`world/scenario.py` is the contract and
validator; `make_room(seed)` generates one). `GET/PUT/DELETE /scenarios/{n}`,
`POST /world/load`, `POST /world/noise` and the `/ws/sim` socket are
documented in `world_server.py`. Invariants worth knowing: a one-duck world
reproduces `MicroduckWalkEnv` step for step (`tests/test_arena.py`), and the
world never runs rewards or domain randomization — reflex training keeps its
own env.

### Senses → intents: the brain layer (roadmap Phase 2)

The policy on the robot is blind by contract; everything that *sees* is a
separate tier that only emits intents — a twist for `robot.move`, a head
pose, a beak open/close, a skill name. `brain/runtime.py` is that contract
(`Senses` in, `Intent` out, every input stamped with its age so a brain can
refuse stale data); `brain/controllers.py` has the scripted brains (`Wander`,
`Follow`, `Script`), and the page's inspector shows each brain's inputs and
its current intent live.

- `sensors/detector.py` is a **geometric detector** in place of a neural
  one: what a camera-side model would output (class, bearing, elevation,
  apparent width, confidence) computed from the sim, gated by a frustum, an
  occlusion ray and apparent size, with latency, misses and ghosts per noise
  preset. Every frame carries the camera pose it was captured from (height
  and depression) — on the robot that is IMU pitch plus the neck servos
  through the kinematics — because the walking gait holds the head 0.08 rad
  higher than the standing pose and rocks it ±0.02 rad, which is tens of
  centimetres of range for anything ranged by elevation.
- **Persons** are mocap capsules that walk waypoint paths; press the page's
  possess button (or send `{"possess": "p0"}`) to drive one yourself and
  see how a brain reacts.
- `brain/tracker.py` is the **tracker** over the detector's frames: ids
  from association (class, bearing gate, range gate), smoothing, hit
  counts, and coasting through misses with the remembered bearing turning
  with the body (from odometry). `Follow` acts on a track, not on whatever
  the last frame said, and a one-frame ghost never confirms. In the sim the
  detector also hands out true names; the tracker keeps them for the tools,
  its ids are its own.
- `brain/gait.py` holds the **walker facts every brain shares** (measured
  with `walker-facts`): it does not walk backwards, a right turn from a
  standstill never starts and a left one sometimes does not, so a cold
  turn gets a 0.2 m/s forward kick that is dropped as soon as the body is
  turning. `GaitWatch` decides "cold" from odometry, never from the intent.
- **Odometry drift** (roadmap 1.7): the `(x, y, yaw)` a brain gets is dead
  reckoning — a per-run distance scale, a gyro bias, per-step noise — under
  the same `ideal` / `datasheet` / `hostile` presets (`Duck.odom`, the
  inspector's odom select, `eval-tidy --odom`). The presets are
  assumptions until someone measures the robot; the point is that a brain
  has to live with them.
- **Room mapping** (`brain/mapping.py`): an occupancy grid per duck in its
  own odometry frame, built from the ToF frames' mount pose (each frame
  carries where the sensor was on the body) and its own pose, floor hits
  traced as free space. The page paints it on the floor (`M`). Under
  drift it **closes the loop on its own walls**: each frame's wall hits
  are fitted with a line, the map's wall near it with another, and the
  angle and gap between them correct the heading and position (a
  deadband keeps line-fit noise from moving good odometry). Measured on a
  duck walking at a corner under a 1.5°/s gyro bias (1.5× the hostile
  preset's σ): pose error 0.21 → 0.12 m, occupied cells within 10 cm of a
  true wall 0.64 → 0.85; under a 3°/s bias 0.42 → 0.20 m; ideal odometry
  stays within 2 cm and 0.5°. A correlative search over cells was tried
  first and measured to trade yaw error for sideways error — a 45° depth
  matrix cannot tell the two apart by overlap, which is the lesson.
- `brain/brain_env.py` turns the brain tier into a gymnasium env
  (`BrainEnv`: 80-float obs from senses, a 3-float twist action, decisions
  at 10 Hz, the shipped walker frozen underneath, domain randomization on
  the *senses* only) and `train-brain` / `eval-brain` train and score one:

  ```bash
  uv run train-brain --run-name follow-v4 --envs 12 --steps 2_000_000 --variety   # ~15 min
  uv run eval-brain --brain learned:follow-v4 --preset hostile --episodes 24   # vs `--brain follow`
  uv run duck-lab --world follow-me         # inspector: pick brain "learned:follow-v4"
  # …and watch the run live at http://localhost:63317/train (duck-viewer)
  ```

  `brains/follow-v1` … `follow-v4` (112 kB each, `brain.onnx` + the
  contract in `brain.json`) ship in the repo, so `learned:follow-v1` /
  `-v2` / `-v3` / `-v4` work from a fresh clone; retrain to replace them
  (the SB3 checkpoints and `progress.jsonl` stay local — which is why a
  cloned brain shows no curve on the `/train` page). The observation comes in two
  versions (`brain_env.py`): v1 fed the nearest raw detection; v2 feeds
  the TRACKER's target (its bearing keeps turning with the body through
  a miss), a coasting flag, confirmation and the odometry yaw rate, in
  the three slots v1 reserved. A brain's `brain.json` says which it was
  trained on and `LearnedBrain` builds that one, so v1 still runs
  bit-for-bit. follow-v2 is v2, trained on the pinned 2026-09 model
  (400k decisions, then 400k more warm-started).

  **A person is seen by its legs.** The detector reports the part of a
  tall target that is inside its 48° vertical frustum: from a 24 cm-high
  camera a person's middle leaves the frustum at 1.2 m, and until this
  the follow band (0.7 m) sat inside the range where the person was
  invisible — every brain's "in sight" fell as it closed in, and the
  scripted one coasted and searched with the person a metre ahead. A
  point-like target (a ball) is unchanged. Every row below moved with
  it (in sight 0.75 → 0.96 for `follow-v2`), so the table is the new
  world. `follow-v4` is the first brain TRAINED in it; v1–v3 are scored in
  it but were trained before it.

  Measured on identical follow-me episodes, in band / in sight under the
  datasheet and hostile presets. **240 episodes a cell** (24 episodes x 10
  eval seeds) — the whole table is one sample, six seeds deeper than the
  12-episode figures it replaces, which moved some cells by up to 0.04 and
  narrowed the seed spread to +-0.02..0.04:

  | brain | datasheet | hostile | +variety, datasheet | +variety, hostile |
  |---|---|---|---|---|
  | `follow-v4` (retrained on the legs detector) + reflex tier | **0.81 / 0.95** | **0.76** / 0.87 | **0.82 / 0.96** | **0.74** / 0.87 |
  | `follow-v2` + reflex tier | 0.77 / 0.96 | 0.67 / 0.87 | 0.76 / 0.94 | 0.65 / 0.85 |
  | `follow-v3` (trained with the reflex tier and variety) + reflex tier | 0.74 / 0.95 | 0.63 / 0.88 | 0.75 / 0.94 | 0.62 / 0.86 |
  | `follow-v1` (version-1 observation, no reflex tier) | 0.72 / 0.94 | 0.57 / 0.88 | 0.69 / 0.94 | 0.58 / 0.88 |
  | scripted `follow` + reflex tier | 0.48 / 0.83 | 0.41 / 0.74 | 0.48 / 0.82 | 0.41 / 0.74 |

  (Before the legs, at 12 episodes: 0.80 / 0.75, 0.63 / 0.61; 0.71 / 0.68,
  0.63 / 0.57; 0.73 / 0.85, 0.60 / 0.75; 0.46 / 0.53, 0.42 / 0.40.)

  **`follow-v4` is the follower to pick.** Every shipped brain before it was
  trained while a person vanished inside 1.2 m; v4 is the first trained in
  the world the legs detector made, and it is ahead in all four cells — by
  0.05 on the datasheet preset and 0.09 under hostile noise, on 9/10 and
  10/10 eval seeds respectively. The gap is widest exactly where the old
  brains were worst, which is the shape you would expect if the thing they
  were missing was the close-in half of the band. It also *bumps less*:
  15.5 an episode against v2's 21.8 on the datasheet preset (21.2 vs 26.9
  hostile) — the first learned brain to move that number, which had sat at
  14-27 for all of them.

  It cost 2M decisions (15 min, 12 envs, `--variety`, ~2.2k steps/s on an
  M-series Mac) against v2's 800k and v3's 600k. Its reward curve is flat
  from ~900k, so the budget, not the recipe, is what v3 was short of:

  ```bash
  uv run train-brain --run-name follow-v4 --envs 12 --steps 2_000_000 --variety
  ```

  Watch a run go at `/train` in the viewer (below) rather than tailing the
  log.

  The **reflex tier** is the thing that moved: under a version-2 brain
  the env yaws the head toward the tracked target (0.8 × the body
  bearing, ±0.6 rad — the 62° camera keeps the person while the body
  catches up) and refuses a forward command with something inside 0.25 m
  ahead. It took `follow-v2` from 0.69 to 0.80 in band on the datasheet
  preset and the scripted brain from 0.47 to 0.53 in sight; hostile
  noise moved less. Detection frames now carry the camera's yaw and the
  tracker keeps tracks in the BODY frame, so a brain steers the body
  whatever the head is doing. A version-1 brain is scored in the world it
  was trained in — no tracker, no reflex (`eval-brain` runs the env at
  the brain's observation version; measured under a gaze it never saw,
  v1 fell from 0.73 to 0.60). `+variety` (`eval-brain --variety`,
  `train-brain --variety`) is two free 30 cm boxes re-scattered every
  episode and a second duck walking a slow circle: every brain loses
  0.03–0.14 to it; v3 and v4, trained on it, lose least (v4 is
  fractionally BETTER with variety than without, 0.82 vs 0.81). v3 was
  never otherwise better than v2 (600k decisions vs v2's 800k; its reward
  was still climbing) — `follow-v4` settles that by running the same recipe
  to 2M, and it is the follower to pick. Bumps sat at 14–27 an episode for
  every learned brain — the person walks through the duck as often as the
  duck walks into the person, and a reflex stop cannot help with the first
  — until v4 took the datasheet figure to 15.5.

  **Getting out of the way** (`brain/controllers.ClosingWatch`, `eval-brain
  --charge S --avoid`): the case where the person — or another duck —
  walks straight at the follower. `--charge 6` makes the person's next
  waypoint a point past the duck every 6 s, so it walks through the
  duck's spot; the benchmark then reports contact seconds (truth: the
  capsule against the body) and dodges. The watch reads the ToF's
  clearance ahead (body-height returns): it shrinks at the duck's own
  speed when it walks at a wall and faster when something comes at it,
  and the difference, fitted over 0.4 s, is the closing rate. Past
  0.12 m/s inside 1.2 m the manoeuvre is a turn toward the freer side,
  then a walk. It ships OFF, with the numbers: the walker cannot clear a
  person (measured from a standstill it sidesteps 1 cm in its first
  second and 6 cm in two at the largest lateral command; a turn-and-walk
  moves it 0.1 m off the line in 1.8 s; walking diagonally 0.13 m in
  2 s — and the person arrives in 2–5 s). On the charge case (datasheet,
  12 episodes) the scripted follow's contact is 5.3 s an episode without
  the dodge and 4.9 with it, falls 0.17 → 0.08; `follow-v3` under the
  reflex dodge 3.4 s either way, falls 0.50 → 0.25 — within the noise of
  12 episodes. In ordinary following the dodge fires on a person who is
  merely walking toward the duck and loses it: in band 0.49 → 0.39, in
  sight 0.86 → 0.57 (lazier triggers, 0.8 m / 0.2 m/s and 0.6 m / 0.3
  m/s, cost less and buy the same nothing). The mechanism and the flag
  stay for the robot, where a person does not walk through it.

  **A polite person** (`Person.yield_m`, `eval-brain --polite M`) settles
  it. The mocap capsule walked through the duck, so contact seconds could
  not fall whatever the duck did; a real person stops. With `yield_m`
  the walker stands when a duck is inside that range on its way (facing
  the way it wants to go) and after 2.5 s gives the waypoint up and
  steps around — turning in place first, not arcing through what is
  beside it. On the charge case with the person stopping 0.55 m centre
  to centre (its surface 0.35 m from the duck), datasheet, 12 episodes:
  the scripted follow holds the band **0.92** of the time with **no
  contact and no falls** (`follow-v2` 0.93, `follow-v3` 0.93), and the
  dodge only costs (0.83 in band, 25 bumps an episode, the dodge walking
  into the person). Facing the person and standing — what the follow
  already does inside the band — is the right behaviour; the dodge stays
  off. (With the person stopping 0.35 m centre to centre, nearly touching
  a 0.2 m capsule, every brain fell out of band, 0.28–0.32, with the
  same zero contact: the band is the metric that moved, not safety.)

  Why the scripted one loses: a probe of both on the same episodes showed
  the learned brain sidestepping ±0.23 the whole time and holding the
  bearing at 0.13 rad, while the scripted one stood still between
  corrections, went cold (a standing walker cannot start a right turn and
  starts a left one slowly) and averaged 0.82 rad off — the person walked
  out of the frame before the body followed. An idle sidestep toward the
  target's side plus a turn gain of 8 (swept over 8 episodes: 0.46 → 0.49
  in band, 0.36 → 0.51 in sight) is what ships; speed, lead-on-bearing-
  rate and crab-strafe variants all measured worse. The rest of the gap is
  the learned brain's continuous motion, which a hand-written
  hold-and-correct loop does not have. The obs layout is a contract shared
  by training and the in-world `LearnedBrain` (`brain/learned.py`), and
  the exported ONNX bakes the normalizer in, like `export-walk`.

### Two ducks, one ball (the soccer track's first form)

`pitch` is a walled 3 × 2.5 m room with a ball in the middle and two ducks
running the `chase` brain: track the ball, line up behind it, stand, and
kick it with the robot's own shipped kick policy (`ball_kick_left` /
`ball_kick_right`, run as a 0.5 s window with an all-zero command exactly
as robotd does). The World counts a goal whenever the ball crosses either
short wall's line inside the goal width and puts it back in the centre;
the page shows the score; `eval-pitch` is the benchmark.

What was measured on the way: a ball 8 cm ahead and 6 cm to the foot's
side flies 1.6 m, 10 cm dead ahead barely moves, the other side is not
touched; a floor ball leaves both the camera and the ToF in the last
0.3 m, so the line-up is dead reckoning in odometry; a kick started
mid-stride fell 4 times in 7, so the duck stands 0.4 s first; a full
walk-around to kick toward the goal crossed walls and the other duck
(4 kicks and 2 falls a run for 1.0 goal), plain line-of-sight kicks were
11 kicks, 1.25 falls and 1.0 goal, and aiming at the goal only when it
costs under a 60° detour is **1.75 goals, 6.8 kicks and 1.5 falls a run**
(`eval-pitch --seeds 4 --seconds 300`; over 8 seeds 1.5 goals, 8.5
kicks, 2.1 falls). The falls were then duck-on-duck: 5 of 7 falls in 4
traced runs had the other duck 3–9 cm away and this one turning in place
(searching, blocked, or lining up) — the walker tips over when it turns
against a body its ToF rows do not see. A yield rule (stand when the
other duck is close and clearly nearer the ball) was measured over the
same 8 seeds and shipped OFF: 1.1 goals for the same 2.1 falls. The
**body-aware avoid** that replaced it — a tracked duck inside 0.4 m and
ahead: turn away from it, never toward it, and stand still while it is
touching — measures **1.38 goals, 6.5 kicks and 0.50 falls a run** over
the same 8 seeds. The kick window also runs at robotd's standing tuning
now (`STANDING_GAIN_RATIO`: the walking Kp × 0.8 for the 0.5 s, the
whole action as always), which the measured kick distances above survive.

**Second form** (`pitch`, `pitch-2v2`, `pitch-3v3`; `eval-pitch
--per-side N`). The chase brain now tracks the ball with its head: a
gaze law pitches the camera so a floor ball sits on its axis while the
duck walks in (measured: 0.6 of head command = 0.647 rad of camera,
0.20 m up), which keeps the ball in view to 0.2 m instead of 0.3 and
refreshes the line-up spot to 0.35 m before the blind leg. The ToF
bumper places every zone's hit in 3D from the mount pose the frame
carries and counts only body-height returns, so the head looking down
does not read the floor as a wall and the ball never counts as one; a
wall beside the duck turns it toward the freer side instead of into the
boards; a duck stood against something for 1.5 s retreats (turn to the
free side, walk clear — two attackers otherwise waited nose to nose for
8 s); after a kick it stands and looks down before searching, and the
search dips the head every 1.5 s (a 9 s spin once passed a ball 0.17 m
ahead). Kicks aim at the goal's centre (`goal` in the odometry frame)
within a 60° detour of the line of sight, else along it.

Three things were built, measured and shipped OFF, with the numbers next
to the switch: **re-planning the spot from sightings inside 0.35 m**
(at 0.2 m the bearing noise is centimetres and the foot choice flipped;
the spot dithered for 8 s), a **walk-round via-point** to get behind a
ball on the wrong side (it oscillated with the re-plan and crossed the
other duck), and **dribbling** (`push_beyond`: stand behind the ball
and walk through it — a pushed ball rolls on at about the walking speed
on this floor and the duck follows it without ever lining up; the kick
and a look win). Measured 1v1 over 8 seeds × 300 s: **1.12 goals, 11.8
kicks, 0.50 falls a run** (1.38 / 6.5 / 0.50 before: kicks nearly
doubled, goals within the noise of eight seeds, falls held).

**A goal restarts play from a kickoff** (`World.kickoff`): the ball on
the centre spot with 5 cm of random nudge (two mirror-image ducks would
otherwise meet nose to nose), every duck back on its spawn with its
odometry frame, and a 1 s hold during which every walker gets a zero
command whatever its brain asks — play resumes from standing ducks, not
from the heap at the goal mouth. `World.goal_seq` says a goal happened;
`brain/team.kickoff_brains` makes every brain forget its plan (the chase
brain keeps its kick tally) and wipes the team boards; the lab loop and
`eval-pitch` both do this, and the page's score panel counts the kickoff
down. Re-measured over 4 seeds × 300 s: **1v1 2.00 goals, 8.2 kicks,
0.50 falls a run** (1.12 / 11.8 / 0.50 before), since every goal is
followed by a clean approach from the spawns instead of a scramble at
the wall.

**Which goals are kicks.** `eval-pitch` now attributes a goal to a kick
within 4 s of it, else to a bump: one run scored four goals from one
kick. Over 8 seeds × 300 s of 1v1 the shipped brain scores **0.75
kicked and 1.25 bumped goals a run from 8.4 kicks** — a chase at
0.45 m/s sends a walked-into ball rolling about as far as a kick does
on this floor, and most goals are that. "One kick in four" above was
this: read it as one goal per four kicks, most of them not the kick's.

**The kick map** (a standing duck, the ball swept over ahead × side of
the trunk, `kick_left`; the right kick checked mirrored): the ball
leaves at an angle to the BODY heading that depends on the side offset —
15°/cm near 2 cm, 4.5°/cm around 4–8 cm — and at the shipped spot
(8 cm ahead, 6 cm to the side) it is +21.6° for the left foot (2.1 m)
and −11° for the right (1.9 m), the same whichever way the body is
yawed; 12 cm ahead the kick dies, 2 cm to the wrong side it barely
moves. The sweet spot is 6–10 cm ahead and 4–8 cm to the side. Standing
the body rotated by the deflection so a sweet-spot kick flies along the
line to the goal (`kick_deflect_left` / `_right`) was built and measured
OFF: in play the ball is 2–3 cm off the sweet spot when the kick fires
— the line-up, not the map, scatters the shots (a 12-spot lone-shot
probe: 35° mean absolute direction error without it, 28° with it, both
noise-dominated) — and the rotated stance scored **1.38 goals a run
against 2.00 without it** over 8 seeds × 300 s (10.4 vs 8.4 kicks). So
the map's lesson that ships is where the next goals are: line-up
precision at the spot (the stop lands 2–5 cm long, the aim tolerance is
0.25 rad), not the aim.

**Line-up precision, measured and shipped off** (`ChaseParams.two_stage`).
A landing probe (12 lone shots: where the ball is in the body frame when
the kick fires) said the ball was 9 ± 11 cm further ahead and 5 ± 11 cm
further to the side than planned, with the heading 11° off; a second
probe split that into 7 cm of sighting error and **15 cm of ball motion
between the last sighting and the kick** — the walk-in's last steering
steps and the square-up's turn in place, 8 cm from the ball, pushed it.
Tightening the aim tolerance to 0.12 or 0.06 rad changed nothing. A
two-stage line-up — square up at a pre-spot 22 cm behind the kick spot
on the kick line, then walk in steering onto the line (on a pure forward
command the walker holds its yaw but crabs 9 cm sideways), and back off
rather than turn in place when the ball is at the feet or the heading is
missed on the spot — puts the ball on the sweet spot: side error 3 cm,
heading 4°, the ball moving 2 cm, **4 goals from 11 lone shots against 1
from 12**. On the pitch it kicks **3.5 times a run against 8.4** and
those kicks scored 0.00 kicked goals a run against 0.75 (bumped 1.25
either way; falls 0.25 against 0.50), so it ships off: with this walker
the kick that happens beats the kick that is placed. The kick-map
deflection on top of it scored 1.88 goals a run, all but 0.4 of them
bumps. A search that walks after 3 s and a **hunt** (walk the line a
lost ball rolled off along — the kick's, or the duck's own heading —
before the standing search; state histograms said half of every run is
search) were measured with it: the hunt found the ball (2.00 bumped
goals a run) and walked into the other duck (1.50 falls a run); aiming
every kick at the goal cut kicks to 0.6 a run. The two-stage line-up and
the search walk ship off behind their parameters with the numbers next
to them; the hunt ships on, below.

**The hunt, with its stops** (`hunt_s`). Traced with the hunt on, it
alternated with "blocked" against the boards at 3 cm, where the ToF
returns nothing, and walked turning at full rate into the other duck
beside it at 90–110° — outside the camera's avoid cone and the ToF's
middle columns. It is now slower (0.3 m/s), turns at most 0.5 rad/s,
and ends — does not alternate — the moment the ToF has something inside
0.45 m, any duck track is beside, or the boards are 0.35 m ahead in
odometry. Over 8 seeds × 300 s: **8.6 kicks a run against 8.4, 0.12
falls against 0.50**, goals within the noise of eight seeds (1.50
against 2.00).

**The search is a walking circle.** Instrumented over 300 s, during
search the ball was inside the camera's frustum 1% of the time and
detected 0%: it sat 90–120° off the nose, and a standing turn barely
turns the walker (the cold-turn kick fires once, the next dip stands it
still again). Probed with a lone duck, the search now sweeps at ~24°/s
walking a slow circle (0.2 m/s with the turn): a ball straight behind is
found in 7 s, one to the right — by the always-left turn — in 10 s.
With the hunt and the circle, over 8 seeds × 300 s of 1v1: **2.25 goals
(0.25 kicked, 2.00 bumped), 9.4 kicks, 0.38 falls a run** against 2.00
(0.75 / 1.25), 8.4 and 0.50 before. Turning the search toward the side
the ball was last on probed 4 s instead of 10 for a ball to the right
and measured off (1.62 goals, 8.9 kicks, 0.62 falls a run: within the
noise, the falls on the walker's weak right turn).

**Teams** (`brain/team.py`): teammates share a blackboard — one message
a second over Wi-Fi on the robot: my id, my distance to the ball, where
I put it. The nearest attacks, the others support (0.7 m behind the
ball toward their own goal, spread sideways by rank, facing it, never
inside 0.45 m of it); the attacker keeps the role until a teammate is
clearly nearer, so two ducks a centimetre apart in range do not swap
every frame. Measured over 4 seeds × 300 s, with the kickoff: **2v2 2.00
goals, 7.8 kicks, 2.75 falls a run (0.69 per duck); 3v3 0.75 goals, 5.2
kicks, 4.75 falls a run (0.79 per duck)** on a pitch that grows 0.4 ×
0.35 m a side (before the kickoff 2.25 / 10.8 / 3.0 and 1.00 / 5.8 /
5.25). Six ducks crowd one ball: falls per duck climb with the roster
(0.25 → 0.69 → 0.79) as the avoid and retreat rules fire against three
bodies at once, and the supporters' spots overlap the opponents'
attackers. The next form is positional play — supporters that mark
rather than shadow — and a scoreboard that counts possession.

**3v3 falls, traced** (3 seeds × 300 s, 14 falls): 10 were supporters
turning in place with a teammate 5–28 cm away or against the boards —
a body beside the duck is outside the camera's 62° and the ToF's 45°,
so neither the avoid rule nor the wall rule saw it. Two answers: the
support spot stays 0.35 m inside the pitch (the brain gets the pitch's
bounds from `make_pitch`), and a supporter with any duck track inside
0.3 m within the last 1.5 s (a track's bearing turns with the body, so
a duck seen a second ago still says where it is) stands instead of
turning in place. 3v3 over 4 seeds: **1.00 goals, 7.8 kicks, 3.50 falls
a run (0.58 per duck)**, from 0.75 / 5.2 / 4.75 (0.79).

### Tidy the playroom (roadmap Track 12)

`playroom` is the built-in that scatters toys on the floor of a walled room
with a low basket in a corner, and `tidy` is the brain that clears it:

```bash
uv run duck-lab --world playroom          # watch it on /sim; the tidy score is top-left
uv run eval-tidy --seeds 3 --toys 6 --seconds 300   # the benchmark: toys in the basket
```

What is real and what is a model here, so nobody mistakes one for the other:

- **Grasp is an attachment event**, not contact physics (`World.grasp`): when
  the beak closes, the nearest toy within 4 cm of the mouth tip is welded to
  the jaw with probability falling from 1 at zero error to 0 at the edge.
  Release drops the weld. Contact-based grasping is roadmap 12.2's later step.
- **The ground pick is the shipped skill**: `alpha_ground_pick.onnx` runs one
  cycle as a hard swap of the reflex tier, exactly as the robot does, and the
  beak closes at the phase where the tip bottoms out (measured: 2 cm up,
  7.8 cm ahead of the trunk, 1.4 cm left). The carry needs no new reflex —
  the shipped walker carries a 20 g block.
- **The brain is a state machine over senses** (`brain/tidy.py`): scan,
  approach (head down, camera 37° down), a blind last half-metre in
  odometry, settle, pick, verify, carry, deliver (head level, the basket
  re-measured standing still at 0.42 m, then a straight blind leg), drop,
  back off. Every constant in it is a measurement on the walker, written
  next to the number — the 8 cm beak overhang past the feet is why a
  release is tight, and why the brain never walks at a toy that projects
  into the basket.
- Odometry is the sim's truth for now (roadmap 1.7 adds drift); the real
  robot's drifts, which is why the basket is re-acquired by sight every trip.

**Measured** (`eval-tidy --seeds 8 --toys 6 --seconds 300 --jobs 2`,
datasheet sensor noise, upstream models at the pinned shas):

| odometry | tether | tidied (mean of 16 seeds) | falls / run |
|---|---|---|---|
| ideal | onboard | **0.89** | 0.31 |
| datasheet drift | onboard | 0.84 | 0.56 |
| hostile drift | onboard | 0.79 | 0.75 |
| ideal | 250 ms round trip | 0.76 | 2.19 |

All four rows are on the 2026-09 CAD re-export (microduck_rl badc4e7),
with the staged approach for rim toys, the sidestep-then-turn back-off,
the slower last leg, and the brain steering by its own loop-closed pose
(`loop_closure`, the wall-line matcher of `brain/mapping.py` folded
into the tidy brain; `eval-tidy --no-loop-closure` steers by raw
odometry). The loop closure is **measured neutral** here: the hostile
row reads 0.79 / 0.75 with it on and off over the same 16 seeds (the
matcher fires — 96 corrections in the first minute of a hostile run —
but the brain re-acquires the basket and every toy by sight each trip,
so only the odometry between two sightings matters, and that is short).
It stays on for the map it builds; the win it was built for is the room
map itself (5.5). Sixteen seeds now (`--seeds` default), each its own toy
layout: eight seeds of six toys moved the hostile row 0.81 → 0.79 →
0.62 across three runs of the same brain with different back-offs, so a
difference under 0.05 is noise even here. The 8-seed rows before the
loop-closed pose read 0.94 / 0.50 ideal, 0.88 / 0.50 datasheet, 0.62 /
0.75 hostile, 0.79 / 1.50 tethered; before the rim staging 0.88 / 0.38
ideal; on the previous export 0.88 / 0.12, 0.79 / 0.50 and 0.79 / 1.88
for the drift and tether rows. A model bump is a re-measure, not a
merge.

The tether row is roadmap 12.10's answer in one line: a laptop brain over
Wi-Fi keeps most of the tidying but trips at the rim three times as
often — every traced tethered fall was the stopping stride at the rim,
because the stop was decided on senses a quarter of a second old (4.7 cm
of overshoot at a 0.3 command, 3.0 at 0.25, which is why the last leg is
walked at 0.25). The 50 Hz reflex stays onboard either way.

Up from 0.67 and 1.7 falls a run at the first close of the loop, and 0.11
before that. What moved it: the basket is re-measured standing still and
never released on a long-range guess; far sightings are directions, not
ranges (at 2.3 m the elevation-to-range map is 34 m per radian); the held
toy is masked out of the ToF guard (it sits 2.5 cm from the sensor and
read as a wall); a stop out of a steering step lunges 2–3 cm, so the last
centimetres are walked straight; the obstacle detour walks past whatever
the servo wanted. Every step of the way there was a measurement, not a
guess — `uv run walker-facts` re-measures them and `uv run trace-tidy`
shows one run state by state, with every release, landing and fall in
context. `.claude/skills/tidy-trace/SKILL.md` is the debugging guide.
`--tether-ms` (roadmap 12.10) delays every intent by a brain round trip;
`POST /world/tether` does the same live on the page.

## Teachable behaviors (the viewer's 🎓 teach panel)

The `behaviors/` package is a library of trick recipes — plain-English reward terms over
the walking env ("stand on one leg", "crouch", "spin"). The lab server matches
chat text to a behavior (`POST /teach`), launches `train-behavior` as a
subprocess, tails its `progress.jsonl` into the frame stream, and hot-loads the
`live.onnx` snapshot (exported every 150k steps) onto a 🎓 trainee duck — so
you watch the policy improve every ~15 s while it trains.

```bash
uv run train-behavior one_leg          # standalone, same artifacts as train-walk
```

Add a trick = add a `Behavior` to its module under `behaviors/` (reward fns + friendly
strings + keywords). The sign conventions and no-jackpot rules from
microduck_rl/AGENTS.md apply; `tests/test_behaviors.py` locks them.

Complex tricks can declare a **staged curriculum** (`Behavior.curriculum`,
a tuple of `CurriculumStage(label, steps, env)`): `/teach` then trains a
CHAIN of runs (`teach-<id>-<hash>-s1`, `-s2`, …), each fine-tuning from the
previous via `--init-from` with the stage's `env` vars on its subprocess
(how the backflip marches its spawn window back toward the entry). Frames
carry `training.stage {idx, count, label}` plus cumulative
`progress.overallSteps/overallTotal`, so the viewer narrates the chain
("stage 2 of 3 · carrying the roll over the top"); `/teach/stop` stops the
whole chain, and an explicit `initFrom` skips it (single run, final stage's
env).

`behaviors/core.py` also has a **term CATALOG** — composable optional terms
mirroring the official stack's reward vocabulary (`head_up`, `flat_feet`,
`calm_body`, `no_limit_parking`, `smooth_torque`, `soft_landings`). A /teach
`weights` key outside the recipe adopts that catalog term at the given weight
(the UI's "＋ add a term"); behavior cards list the rest under
`availableTerms`. Posture basics (head up, flat feet) are baseline defaults
on the static-pose behaviors — a trick that looks wrong usually means a
missing term, and the catalog is where it should come from.

`POST /teach` also takes `"weights": {termKey: value}` (reward-slider
overrides, clamped ≥ 0) and `"initFrom": "<run name>"` (fine-tune an existing
run's policy under the edited recipe). `POST /teach/load {"policy":
"run:<name>"}` seats a FINISHED run in the teach panel without training
anything — its recipe (behavior + the weights recorded in the run's
`behavior.json`) streams in "done" state, so the viewer's fine-tune button
targets that run; the viewer calls it when a duck running a teach-run policy
is selected or a policy chip is dropped on the teach panel. Refused while a
job is actively training. Helper ducks (`{"spawn_helper": true}`
over WS) are extra viewers of the same `live.onnx` snapshot — they do **not**
add trainer workers (the trainer runs `viz_server.BASE_ENVS` workers; see
the bench section for why helpers-as-workers measured slower and why that
measurement was a confound). Frames carry `stats` (machine/lab/
trainer cpu+mem, training steps/s). The roster persists to `lab-state.json`
(`LAB_STATE_PATH` to relocate, `--fresh` to reseed from the CLI); training
jobs do NOT survive a server restart — a restored trainee keeps its last
`live.onnx` brain.

Runs pile up fast, so `DELETE /runs/<name>` erases one run directory —
policy, checkpoints, progress log — and `?chain=true` treats the name as a
curriculum-chain prefix and takes every stage of it in one all-or-nothing
call (the palette's ✕, always behind a confirmation). Any run of the job
that is training right now is refused with 409, the whole chain included:
a stage warm-starts from the previous stage's dir. Deleting does not disturb
a duck already running that brain — the ONNX session is loaded in memory —
but the duck drops out of the roster at the next lab restart.

## Layout

```
src/microduck_local/
├── contract.py     # THE contract: obs layout, DEFAULT_POSE, timing, ranges — mirrors infer_policy.py
├── walk_env.py     # Gymnasium env: velocity-command walking, mjlab-distilled rewards
├── train.py        # SB3 PPO + vec envs + VecNormalize (+ penalty-sign guard)
├── vec_env.py      # ONE compiled mjModel behind every worker (see above)
├── export_onnx.py  # bake normalizer → ONNX obs[1,61]->actions[1,14]
├── eval_onnx.py    # headless eval battery (fall rate, tracking error)
├── render_rollout.py  # offscreen rollout → mp4 + captioned contact sheet
├── bench.py        # steps/sec benchmark
├── world/          # scenario contract, MjSpec composition, World (N ducks, one mjData,
│                   # persons, toys, basket, grasp-as-attachment, skills)
├── sensors/        # ray rig on mj_multiRay, the 8×8 ToF, the geometric detector
├── brain/          # senses → intents: runtime contract, tracker, gait facts, Wander/Follow/
│                   # Script, the tidy state machine, occupancy mapping, BrainEnv + LearnedBrain
├── eval_tidy.py    # the playroom benchmark (eval-tidy)
└── world_server.py # the /sim page's backend: /scenarios, /world, /ws/sim
tests/              # contract locks: obs order, action semantics, DR non-accumulation,
                    # penalty signs, shipped-alpha-survives regression test
```

## Conventions inherited from microduck_rl/AGENTS.md

- Obs layout is the shared 61D hot-swap contract; unused command slots are
  zero-padded (tiny keep-alive sampling ranges), never removed.
- Penalty reward terms are ≤ 0 by construction; a callback aborts training if
  any episode penalty sum goes positive (the double-negation lab bug).
- Domain randomization restores compile-time defaults before applying
  (never accumulates across resets).
- ONNX always ships with the obs normalizer baked in — never deploy a raw
  checkpoint.
- `joint_vel` observation lags one control step (Dynamixel moving-average),
  matching training and hardware.
