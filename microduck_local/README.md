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

## Cloud and Linux training (`machine.py`)

Every number above was measured on an 18-core M5 Max, and **that stays the
default** — this harness is written for a Mac and advertised as one. But two
of those tunings are wrong on a small Linux or cloud box, badly enough to
lose a third of the machine, so `machine.py` detects which machine it is on
and picks a profile. `uv run machine-facts` prints what yours resolved to.

**What goes wrong on a 4-core cloud box.** The trainer, not the physics
workers, eats the machine. torch's OpenMP pool spin-waits between the tiny
batch-N policy forwards that make up a rollout, and those spinning threads
take the cores the workers need: measured at 4 envs, the trainer process sat
at **311% CPU while each worker got 18%**. The cores looked 90% busy and were
mostly spinning. An M5 Max has enough cores to absorb that, which is exactly
why `--pin-threads` measured a 21-24% *regression* there.

The fix is not to pin threads — that gives the update one core, which is what
cost 21% on the Mac. It is to give each PPO phase the threads it actually
wants. Measured per phase at 32 envs on this box: **rollout 22.8 s, update
2.9 s**, so the rollout is ~89% of wall time and is the phase the threads
must not disturb.

Medians of **four interleaved repetitions** on a 4-vCPU Xeon container — the
arms measured back to back with their order rotated each rep, 40k timed steps
per point — because this box drifts far too much to compare across windows
(see the warning below):

| envs | mac profile | phase-aware threads | gain |
|---:|---:|---:|---:|
| 8 | 892 | **1,420** | +59% |
| 16 | 1,228 | **1,727** | +41% |
| 32 (the default) | 1,764 | **2,155** | +22% |

Per-arm spread across the four reps was 4-8%, and the mac arm reproduced
within 4% at 32 envs, so the machine held still for this run.

> **Do not compare these to a number you measure later.** The same script and
> configuration on this same container ran 13.1 s in one window and 19.5 s a
> few hours later — a 49% drift from noisy neighbours and CPU-credit
> throttling, larger than every optimization in this file. Only the ordering
> within a window transfers; re-measure both arms together
> (`--compare-profiles`) rather than trusting any absolute figure here.

The `linux` profile is that thread split, and it is the default off Darwin:

- **Phase-aware torch threads.** One intra-op thread while collecting
  rollouts, every usable core for the update. Implemented as an SB3
  *callback* on `on_rollout_start` / `on_rollout_end` — the two hooks that
  bracket the phase — so `train-walk`, `train-behavior` and `train-brain` all
  get it without a vendored train loop. On the mac profile the callback list
  is **empty**, not a no-op.
- **Cores from the container, not the host.** `os.cpu_count()` reports the
  *host's* cores inside Docker or Kubernetes, so a 4-CPU pod on a 64-core node
  would start 64 spinning threads. `usable_cores()` takes the smallest of
  `cpu_count`, CPU affinity and the cgroup quota.
- **numba warmed once, in the parent.** The BAM kernels JIT on first call:
  **528 ms** against a 0.95 ms steady step, and `cache=True` does not avoid it
  across processes. `vec_env._warm_jit` pays it once before forking so the
  workers inherit compiled code — the same trade as the shared `mjModel`,
  for code. It restores every model array a warm-up step touches, so children
  still inherit exactly the model the probe's construction left.

`train-brain` also never called `configure_torch_cpu` at all, so it ran with
torch's defaults (intra-op *and* inter-op at the core count). It does now.

**Worker packing was measured and rejected.** Packing the fleet into one
process per core (32 envs as 4 × 8) is the obvious companion optimization —
extra processes cannot run in parallel anyway, and each costs the parent two
semaphore ops per vec-step — and a single unreplicated point made it look
like a +7% win. Four interleaved repetitions put it slightly *behind* one
process per env at every count: **−1.7%** at 8 envs, **−3.9%** at 16,
**−2.6%** at 32, never once ahead. Its other claimed advantage, faster
startup, turned out to be the per-worker numba JIT that `_warm_jit` now
removes for every layout (32-env setup 7.5 s unpacked vs 7.2 s packed). So no
profile packs; `MICRODUCK_ENVS_PER_WORKER` remains the manual knob it always
was, worth re-measuring only at env counts far above these. This is the
second time here that one measurement disagreed with four — see the drift
warning above.

**What is deliberately NOT in a profile: the env count.** `--envs` sets the
PPO batch size and therefore the learning dynamics, and this repo's rule is
that throughput is not learning speed. It stays 32 everywhere. The one knob a
profile does carry is quality-neutral by construction: thread counts do not
enter the PPO math.

```bash
uv run machine-facts                           # cores + the profile they imply
uv run bench-envs --compare-profiles           # both arms, interleaved repeats
MICRODUCK_PROFILE=mac uv run train-behavior run   # force the historical settings
MICRODUCK_ROLLOUT_THREADS=2 MICRODUCK_UPDATE_THREADS=6 uv run bench-envs  # try your own
```

**Sizing a cloud box.** A single trainer's ceiling is the serial parent loop,
not the cores — the Mac numbers saturate near 27k steps/s. Past ~16 cores the
way to use the machine is *several independent trainers* (`taskset` each to
its own core group, each with its own `MICRODUCK_RUNS_DIR`), which is also
what the seed-matched A/Bs this playbook demands need anyway. Per-core clock
matters more than core count: one env steps in 0.34 ms here against 0.135 ms
on the M5 Max, so a 4-vCPU box will not match the laptop whatever the
threading does.

Two things measured and **not** adopted: `OMP_WAIT_POLICY=PASSIVE` on top of
phase-aware threads is a small loss (the update's threads then sleep between
minibatches — update 3.3 s → 5.1 s at 16 envs), and pinning OpenBLAS to one
thread is neutral, because the workers never spawn BLAS pools on 61-dim ops.
Two known gaps: the lock file resolves CUDA torch wheels on Linux (a 7.2 GB
venv for a CPU trainer — CI installs from the CPU index first, and a
`[tool.uv.sources]` entry would fix `uv sync` too), and `--update-device`'s
`auto` only knows about MPS, so a CUDA cloud box needs an explicit
`--update-device cuda` (the code path is device-generic but unmeasured).

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
  with `walker-facts`): from a standstill NO turn starts below the full
  ±1.0 command — not a weak turn, an exactly zero one, in both directions —
  so a cold turn gets a 0.2 m/s forward kick that starts the gait and is
  dropped as soon as the body is turning. `GaitWatch` decides "cold" from
  odometry, never from the intent. **And it walks backwards, faster than it
  walks forwards** — see below.
- **A locomotion "fact" that was a dead band, and cost three brains a
  workaround each.** `gait.py` said "it does not walk backwards (a -0.3 m/s
  command moves 4 mm in 2 s)" and every brain believed it: `tidy.py` leaves
  a rim with a ~7 s sidestep-turn-around-walk, the chase brain retreats by
  turning away, and the line-up gives up on any pre-spot behind the duck.
  The measurement was right and the fact was wrong — **-0.3 is the inside
  of a dead band**. One notch past it:

  | vx command | −0.40 | −0.35 | −0.30 | −0.25 | +0.20 | +0.25 | +0.30 | +0.40 |
  |---|---|---|---|---|---|---|---|---|
  | steady m/s | **−0.23** | **−0.20** | −0.00 | +0.00 | −0.00 | +0.11 | +0.13 | +0.19 |

  The walker reverses at 0.23 m/s — **faster than it moves in any other
  direction**, on an empty floor, over six start headings, with no falls.
  The turn is the same shape: cold, every |wz| below 1.0 is exactly zero
  and so is −1.0, while warm the rate is roughly linear (0.15 / 0.28 /
  0.47 / 0.61 rad/s at +0.25…+1). And `TURN_KICK = 0.2` is inside the
  forward dead band as well: it starts the gait for a cold turn, but a
  brain that "walks" at 0.2 moves 9 mm in 6 s.

  Three things came out of this beyond the numbers. `walker-facts` now
  **sweeps the command ranges** instead of sampling one value each
  (`command_deadbands()`), on an empty floor — `make_room`'s 3.0 × 2.5 m
  room with four boxes cannot hold the 1.3 m a 6 s reverse covers, so
  measuring the gait in it measures the boxes, which is how the first
  re-measurement of this came out wrong too. `gait.back_up()` hands out a
  reverse clamped past the dead band's edge, because a brain that politely
  asks for −0.3 gets 4 mm and no error. And `tests/test_walker_facts.py`
  locks the edge as well as the inside, so the next walker that changes
  this breaks a test instead of a benchmark six months later. **The general
  rule: a locomotion limit read off a single command value is a reading of
  the dead band, not of the robot.** The brains still use their old
  workarounds — swapping `tidy`'s backoff re-opens the fall and tether rows
  it was measured on, so that is its own A/B, not a drive-by.

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
  uv run eval-brain --brain learned:follow-v4 --preset hostile --episodes 24 --jobs 0   # …on every core
  uv run duck-lab --world follow-me         # inspector: pick brain "learned:follow-v4"
  # …and watch the run live at http://localhost:63317/train (duck-viewer)
  ```

  **An episode is a pure function of `(seed, ep)`.** `BrainEnv.reset()`
  re-seeds every generator that outlives `world.reset()` — the ToF's, the
  detector's and the world's own, all three seeded once at construction
  and untouched by their `reset()` — and zeroes the commanded twist that
  `_respawn` leaves standing, so the warm-up steps that let the first
  sensor frames land are not driven by whatever the last episode was
  asking for. Nothing carries from episode k-1 into episode k. That is
  what `--jobs N` rests on: it splits a battery over N processes, one env
  each, and returns exactly the rows `--jobs 1` returns (2.6x on 4 cores
  for a 24-episode battery; `tests/test_eval_brain_jobs.py` pins the
  exactness, and 7 of its 8 cases fail if a carrier ever comes back).
  Episodes used to be chained — episode 0 reproduced, everything after it
  continued the previous episode's noise stream. Re-measuring both follow
  tables independently moved no cell by as much as one seed-level sigma
  (largest in band: 0.019), so no published ranking changed; what it did
  change is that v4's lead over v5 now clears the noise in three cells of
  four, where under chained sampling it cleared in none.

  `brains/follow-v1` … `follow-v5` (112 kB each, `brain.onnx` + the
  contract in `brain.json`) ship in the repo, so `learned:follow-v1` /
  `-v2` / `-v3` / `-v4` / `-v5` work from a fresh clone; retrain to replace them
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

  Measured on identical follow-me episodes (the pinned model) with the
  **polite person that is now the benchmark's default** — it stops
  0.55 m centre to centre short of a duck in its way and steps around
  after 2.5 s — **240 episodes a cell** (24 episodes x 10 eval seeds;
  every row has zero contact, falls at most 0.06), in band / in sight
  under the datasheet and hostile presets. Re-measured on **independent
  episodes** (below) on a 4-core Linux container; the same pass taken the
  old chained way agreed with the M-series figures it replaces to within
  0.02 a cell, and no cell moved by as much as one seed-level sigma:

  | brain | datasheet | hostile | +variety, datasheet | +variety, hostile |
  |---|---|---|---|---|
  | `follow-v4` (retrained on the legs detector) + reflex tier | **0.94** / 0.99 | **0.88** / 0.90 | **0.93** / 0.98 | **0.89** / 0.92 |
  | `follow-v5` (v4's recipe against the polite person) + reflex tier | 0.92 / 0.97 | 0.86 / 0.86 | 0.92 / 0.97 | 0.89 / 0.89 |
  | `follow-v2` + reflex tier | 0.91 / 0.97 | 0.83 / 0.90 | 0.90 / 0.96 | 0.84 / 0.91 |
  | `follow-v3` (trained with the reflex tier and variety) + reflex tier | 0.90 / 0.98 | 0.82 / 0.92 | 0.90 / 0.97 | 0.81 / 0.90 |
  | `follow-v1` (version-1 observation, no reflex tier) | 0.86 / 0.97 | 0.74 / 0.94 | 0.86 / 0.97 | 0.75 / 0.94 |
  | scripted `follow` + reflex tier | 0.76 / 0.96 | 0.66 / 0.91 | 0.76 / 0.95 | 0.67 / 0.90 |

  v4 leads under hostile noise by 0.05 (10/10 eval seeds over v2, 10/10
  on the variety cell) and by 0.04 on the clean preset (9/10); the
  polite person lifted every band by 0.1–0.3 and took the bump counts
  from 15–27 an episode to 0.3–4.6. The capsule that walks
  through the duck (`--polite 0`) had capped every band for a reason
  that has nothing to do with following; measured with it, and much
  deeper — **240 episodes a cell** (24 episodes x 10 eval seeds), six
  seeds deeper than 12-episode figures, which moved some cells by up to
  0.04 and narrowed the seed spread to +-0.02..0.04 — the table is:

  | brain | datasheet | hostile | +variety, datasheet | +variety, hostile |
  |---|---|---|---|---|
  | `follow-v4` (retrained on the legs detector) + reflex tier | **0.84 / 0.97** | **0.73** / 0.85 | **0.83 / 0.96** | **0.74** / 0.87 |
  | `follow-v5` (v4's recipe against the polite person) + reflex tier | 0.74 / 0.94 | 0.65 / 0.79 | 0.74 / 0.94 | 0.66 / 0.80 |
  | `follow-v2` + reflex tier | 0.76 / 0.94 | 0.67 / 0.85 | 0.76 / 0.95 | 0.66 / 0.86 |
  | `follow-v3` (trained with the reflex tier and variety) + reflex tier | 0.74 / 0.94 | 0.64 / 0.86 | 0.75 / 0.95 | 0.63 / 0.85 |
  | `follow-v1` (version-1 observation, no reflex tier) | 0.70 / 0.95 | 0.57 / 0.89 | 0.70 / 0.93 | 0.57 / 0.87 |
  | scripted `follow` + reflex tier | 0.48 / 0.82 | 0.41 / 0.75 | 0.47 / 0.81 | 0.41 / 0.73 |

  (Before the legs, at 12 episodes: 0.80 / 0.75, 0.63 / 0.61; 0.71 / 0.68,
  0.63 / 0.57; 0.73 / 0.85, 0.60 / 0.75; 0.46 / 0.53, 0.42 / 0.40.)

  **`follow-v4` is the follower to pick.** Every shipped brain before it was
  trained while a person vanished inside 1.2 m; v4 is the first trained in
  the world the legs detector made, and it is ahead in all four cells — by
  0.07 on the datasheet preset and 0.07 under hostile noise, on 9/10 and
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

  **Retraining against the polite person did not widen the lead.**
  `follow-v5` is that command run with the person the benchmark now
  uses (`train-brain` defaults to `--polite 0.55` and `brain.json`
  records it; v1–v4 were trained against the capsule that walks through
  the duck). It trains faster and settles higher (reward 171 from ~410k
  decisions against v4's 161 from ~610k — the polite world is the easier
  one; 9 min at 3.7k steps/s), and **the interesting part is where it
  loses.** v5's TYPICAL episode is the better one: its median band is
  level or ahead in all four cells (0.962 against v4's 0.955 on the clean
  preset, 0.940 against 0.927 hostile) and per episode it is ahead more
  often than not (104 of the 159 non-tied episodes on the clean preset).
  Its BAD episodes are much worse, and that is what the means show: 15
  episodes under 0.7 band against v4's 2 on the clean preset, 39 against
  23 hostile, and a 10th percentile of 0.775 against 0.865 (0.579 against
  0.705 hostile). Paired over the same 240 episodes a cell, v4's lead is
  +0.018 / +0.020 / +0.014 / +0.007 in band and clears the seed noise in
  three cells of four (t = 2.9 / 2.0 / 2.1, and 0.8 on +variety hostile,
  which does not). Under the old chained sampling it cleared in NONE of
  them (t <= 1.6) — the carried noise was hiding the tail this comparison
  turns on.

  What v5 learned instead is that the person stops for it: it trips the
  bump signal 3.4 times an episode against v4's 0.4 on the datasheet
  preset and 12.5 against 4.1 under hostile noise, is in sight less
  (0.86 vs 0.90 hostile) and falls more with furniture about (0.05 vs
  0.02). Scored back in the world v4 was trained in (`--polite 0`, 240
  episodes a cell) the habit shows plainly: 0.74 / 0.65 in band against
  v4's 0.84 / 0.73, ahead on at most 1/10 seeds in any cell. A brain
  trained against a person who walks through the duck had to keep out of
  the way; one trained against a person who yields is paid for standing
  in it. **`follow-v4` stays the follower to pick** — trained in the
  harder world, scored in the polite one — and v5 ships as the
  measurement.

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

### Where the soccer track actually stands (read this before the rest)

The sections below are chronological and they read like a run of wins.
They are not. **Of everything tried at the brain tier, nothing has
survived a confirmation on layouts it was not found on.** The list, with
what killed each:

| tried | how it died |
|---|---|
| poacher supporter | reversed on 12 fresh seeds (10 v 3, then 21 v 12, then 13 v 19) |
| ball memory (`seek_s`) | dissolved over 48 seeds |
| seven shelved knobs, re-screened | nothing came back; the one p < 0.05 was a false positive |
| a searching head sweep | that false positive — and it makes the body turn MORE, 5/5 seeds |
| kick cone, ToF floor ball, aimed look, head gaze | never cleared the noise |
| two-stage line-up | loses to the shipped brain on goals, 45 v 71 over 24 seeds |
| `lineup_lat` (that line-up made 21% faster) | first block promised +26% kicks; fresh block gave +5% |
| **the bump-stand rule** | **the last one standing, and it failed too: −1.88 on its own 12 layouts, +0.25 on 12 fresh ones** |

What *has* survived is of three kinds, and none of it is a brain idea:
**bugs fixed** (the ToF placed hits without the head's rotation; clearance
read by sensor column instead of by bearing), **hardware questions
answered** (kicks scale with lens width, p < 0.001, but only if the pixels
scale too — the shipped camera is adequate), and **measurement
discipline** (event counts, paired reads, power, `--out`/`--tag` resume,
and the rule that catches all of the above: confirm on seeds the effect
was not found on).

The measured reason is in "Where the run actually goes" below: **the duck
spends 47% of a 1v1 run rotating on the spot**, at the walker's ceiling of
0.655 rad/s, with the command range already asking for everything it has.
A brain that decides better still has to wait for the body to point. That
is why roadmap 3.7 was rewritten from head-pose to turn rate, and why the
next thing worth spending a GPU on is the walker, not the brain.

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

**The two-stage line-up, made fast, and still not worth it — and the
premise it took down with it.** The state histogram said where its time
goes: over 12 duck-runs it spends 68.9 s a run lining up, of which 48.4 s
is stage one and 27.1 s of *that* is walking to the pre-spot, with 55
attempts a run dying in the back-off whose cause is "the pre-spot is
behind me". `lineup_lat` cuts that — a duck already on the kick line,
squared up and short of the spot, starts stage two where it stands — and
it works as designed: a kicking attempt 5.63 → 4.43 s (−21%), the
back-offs 55 → 35, the ball moving 28.0 → 19.5 cm during an attempt. On
**24 paired seeds over two independent blocks** (100–111, then 200–211)
it buys nothing the benchmark can see: against `two_stage` alone, kicks
72 → 83 (p = 0.44), goals 45 → 34 (p = 0.10), falls 16 → 11 (p = 0.54),
advance and progress flat. The first block promised +26% kicks and falls
12 → 4; the fresh block gave 37 → 39 kicks and falls 4 → 7. Neither
replicated. It ships at 0.

The premise is what actually died. Both careful arms lose to the shipped
brain on goals — **71 against 45 and 34** over the same 24 seeds
(p = 0.038 and p = 0.0003) — and the placed kick is not worth more when
measured directly: the ball's travel in the two seconds after a swing is
17.7 ± 3.9 cm for the shipped brain against 14.4 ± 3.4 and 13.6 ± 3.0,
and toward the goal +0.8 cm against −5.1 and −4.9. Nor is the kick
actually placed — it fires with the ball a median 21–25 cm ahead of the
trunk where the sweet spot is 6–10 cm, because the plan is laid at
`refresh_min` and goes stale while the ball moves 20–28 cm. **The
accuracy limit is the stale plan, not the approach geometry**, which is
why a faster approach barely moved it (25.5 → 22.6 cm). With this walker
the kick that happens still beats the kick that is placed, and that now
rests on 24 seeds and a direct measurement rather than on 8 seeds and a
ratio.

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

Measured after it, on the same 8 seeds, and shipped off with the
numbers: **deliberate bumping** (`push_beyond` 1.4, a push spot behind
the ball and a walk through it toward the goal) scored the same 2.25
goals a run (0.25 kicked, 2.00 bumped) from 6.4 kicks and 1.8 pushes
with **0.75 falls against 0.38** — the deliberate bump scores no more
than the accidental one and falls twice as often; and a **ball memory**
(`seek_s`: the centre spot at a kickoff, every fresh sighting, the end
of a hunted line, walked to before the circle) at 2.38 goals, 7.8 kicks
and 0.75 falls — the goals did not move and the blind walks fell. (Both
were re-screened later on fresh seeds with the continuous metrics, and
neither came back: "The shelf, re-screened", below.) For
the rosters (4 seeds): with the hunt and the circle off, 2v2 1.00 goals,
8.8 kicks, 3.50 falls and 3v3 0.75 / 7.0 / 3.75 against 1.50 / 9.8 /
3.50 and 1.50 / 5.0 / 4.50 with them on — more goals with them in both,
the same falls in 2v2, 0.75 more in 3v3 — and a wider support standoff
(1.0 m back, 0.6 m to the side) 3v3 1.75 / 5.2 / 4.50: the team
defaults stay.

**Where the ball is going** (`brain/tracker.py`): every track now carries
an odometry-frame position and, from consecutive hits, a velocity, and
`predict(t)` rolls it forward under the floor's deceleration. Probed with
a kicked ball: it leaves at 1.4 m/s, slows at 0.04 m/s² (it rolls 3 m, to
the boards) and leaves the level camera at once, 30–55° off the nose, so
the old track coasted with a stale range for two seconds and a new one
was born when the ball was found again. The chase brain can act on the
prediction three ways — yaw the head toward the predicted bearing
(`predict_s`, `head_yaw_gain`; always, or only while searching), open
the search toward the predicted side and walk the hunt to the predicted
point (`predict_steer`) — and **all of it measured off** over the same 8
seeds × 300 s of 1v1, against 2.25 goals / 9.4 kicks / 0.38 falls a run
with it off: yaw always with the steering 1.12 / 10.5 / 1.62; yaw off
with the steering 2.12 / 9.1 / 1.12; yaw while searching with the
steering 2.12 / 10.0 / 0.75; yaw while searching without it 2.12 / 6.5 /
0.75 (with prediction off the run is bit-identical to before the
tracker learned positions, so the tracker itself is neutral). Why: the
head yaws 34° at most and the searching duck's ball sits 90–120° off its
nose (instrumented: in the frustum 3% of search time with the gaze on,
5% without), so the gaze cannot reach it, and the steering walked blind
lines into things. The prediction stays tracked and drawn on the /sim
page (an orange line from the ball to where it will stop); `predict_s`
turns it back on with the best-measured settings.

**The ToF placed by the head pose — a bug, fixed.** The body-height
clearance (`tof_clearance_3d`, what every stop, hunt and wall rule reads)
placed each zone's hit without the head's rotation: with the head dipped
0.6 rad it reported a wall 0.33–0.37 m ahead in every column — the floor.
Hits go through the mount rotation now, trunk-relative, and body height
starts 8.7 cm above the floor (a ball is 7 cm tall). The 1v1 baseline on
the fixed geometry, same 8 seeds × 300 s: **2.38 goals (0.25 kicked),
7.4 kicks, 0.38 falls a run** (2.25 / 9.4 / 0.38 before — two fewer
kicks a run, the ones the false wall had been interrupting one way or
another). Every soccer number below this line is on the fixed geometry;
the tidy brain never read that helper, so its rows stand.

**How much can this benchmark actually resolve?** Read the event counts,
not the per-run averages. Over 8 seeds × 300 s of 1v1 a battery contains
roughly 50–130 kicks, ~20 goals, ~1–6 of them kicked, and **3–8 falls**.
So kicks separate cleanly and falls barely separate at all: a baseline
measured twice, once per clearance rule, gave 3 falls and 6 falls — the
same brain to within 1% of its stopping decisions, and a 3/6 split of 9
events comes up by chance half the time (p ≈ 0.5). Telling 0.38 falls a
run from 0.75 would take dozens of seeds. Goals are nearly as bad: the
standard error is about 0.5 a run, so anything under a goal and a half
of difference at 8 seeds is noise. **Every claim below is stated with
its event counts**, and a difference that does not clear them is written
as "no effect measured", not as a result. Several rows in earlier
revisions of this file did not clear them and have been demoted.

**So the benchmark grew two continuous metrics** (`world/metrics.py`,
re-exported by `eval_pitch`; both accumulated per team at the 50 Hz
control tick). Goals are ~2.5 a run and no amount of care in the
reporting fixes that; what the ducks do thousands of times a run is
*reach the ball* and *move it*:

- **`possession`** — seconds a minute the duck nearest the ball is ours
  and within `POSSESSION_R` = 0.25 m (about twice the kick spot's 0.12 m,
  and inside the 0.40 m at which the chase brain starts avoiding the
  other duck). No carry: one of ours was on the ball or it was not.
  `possessionWide` runs the same clock at 0.40 m as a robustness check.
- **`ballAdvance`** — metres a minute of the ball's displacement toward
  the goal the team attacks, summed over the ticks that team is in
  control, control persisting `CARRY_S` = 2 s past the last touch so a
  kick's roll counts for the kicker. **`ballProgress`** is the same sum
  *signed*, so a team that shoves the ball back toward its own goal is
  charged for it. Attributing per possession is what keeps the signed sum
  from telescoping into "goals in disguise": summed over a whole run and
  both teams it collapses to the ball's net start-to-end position, which
  after every kickoff recentring is goals again, with goals' variance.

Measured 16 seeds an arm (two independent 8-seed batteries that agree),
with each metric's coefficient of variation and the seeds ONE ARM needs
to resolve a 25% shift at p < 0.05 and 80% power — the metric's own
noise, with the size of that particular contrast divided out:

| metric | CV | seeds for a 25% shift | r with goals (32 runs) |
|---|---|---|---|
| goals | 0.76 | **146** | — |
| kicks | 0.49 | 62 | 0.30 [−0.06, +0.59] |
| falls | 1.22 | **376** | 0.07 [−0.29, +0.40] |
| `ballAdvance` | 0.40 | **43** | **0.50 [+0.19, +0.72]** |
| `possession` | 0.16 | **9** | 0.33 [−0.02, +0.61] |
| `possessionWide` | 0.11 | 6 | 0.23 [−0.13, +0.53] |
| `ballProgress` (run total) | 4.20 | — | 0.11 [−0.25, +0.44] |

So `ballAdvance` is the discriminator — the only metric here, `kicks`
included, whose association with goals is resolved away from zero, at
about a third of goals' seeds — `possession` is the cheap screen that
says two variants differ *at all*, and goals stay reported and stop being
the judge. Nine seeds against a hundred and forty-six is the difference
between a screen that runs in an afternoon and one that never runs.

**Never quote `ballAdvance` alone.** It keeps only the forward part of
each step, so anything that makes the ball move MORE scores higher on it
without moving the ball anywhere, and that caught a wrong conclusion the
day the metric shipped: the attacker-claim fix below raised advance
**+0.18 ± 0.06 (2.9 σ)** with kicks +64%, while signed `ballProgress`
stayed **flat (−0.003 ± 0.136, 0.0 σ)** and the advance PER KICK
**halved, 0.202 → 0.106**. The ball moved more and got no nearer the
goal. Read advance, signed progress and advance-per-kick together or
read none of them — and for any new metric, ask first which cheap
behaviour maximises it.

**The `left` / `right` keys of a row are goal MOUTHS, not team scores**
(documented, not renamed): `World._check_goal` files a ball crossing at
+x under `right`, and `World.goal_for` sends the LEFT team at +x, so a
row's `right` count is what the left team scored. Run totals are
unaffected — the failed replications below compared totals between arms
of identical brains — but every per-side reading is inverted if it is
missed, and reading it the natural way flipped the sign of a whole
correlation table here before it was caught.

The lab ticks the same `PitchMetrics` on its own step, so the /sim pitch
panel shows the three per-team rates live (the signed one in red when a
team is losing ground) and a number on the page is the number the battery
reports.

**The camera, as a hardware question — three batteries, one answer.** The
detector's field of view is one constant (`DetectorSpec.fov_h_deg`,
62° × 48° as shipped — an assumption about a Pi-camera-class module), so
the sim can price a wide-angle camera. It has to price it *honestly*:
the two apparent-width thresholds that decide whether a target is found
were written as angles but justified in **pixels** of a 320 px frame
over 62°, so a wider lens on the same sensor must find small distant
things *less* often. They are derived from pixels per radian now
(exactly unchanged at the shipped lens, verified bit-identical), which
costs 120° a duck at 3.9 m — found 73 times in 400 against 260 at 62°.
The first battery is soccer, where the wide lens wins: paying that gate,
over 8 seeds × 300 s of 1v1 against a 51-kick, 19-goal, 1-kicked-goal
baseline:

| lens | kicks | goals | kicked goals | falls |
|---|---|---|---|---|
| 62° × 48° (shipped) | 51 | 19 | 1 | 6 |
| 90° × 70° | 94 | 17 | 4 | 4 |
| 120° × 93° | **105** | 22 | 6 | 3 |
| 150° × 116° | **130** | 17 | 4 | 7 |

Kicks scale with the lens and the effect is overwhelming (105 against
51, p < 0.001) — a duck with a wide camera loses the ball far less often
and spends its run playing instead of searching. Goals do not follow
(22 against 19 is noise), and kicked goals only hint at it (6 against 1,
p = 0.13).

**The second battery pointed the same lens at the tidy task, and it
lost.** Soccer wants a wide view of a big orange ball; tidying wants to
resolve a 3 cm brick across a room, and the honest size gate takes that
away. 24 seeds a lens, paired on the same layouts, `--toys 6`:

| lens, same 320 px | tidied | vs shipped, per seed |
|---|---|---|
| 62° × 48° (shipped) | **0.889** (128/144) | — |
| 90° × 70° | 0.743 (107/144) | −0.88, worse on 14 of 24 |
| 120° × 93° | 0.632 (91/144) | **−1.54, worse on 24 of 24** (sign p = 1e-7) |
| 120° × 93° on **640 px** | 0.819 (59/72) | −0.33, inside the noise (p ≈ 0.13) |

It replicated across both halves of the battery, and it breaks in exactly
one place: **the scan**, not the deliver leg and not the grasp. Traced,
a toy at 1–1.5 m is found in 36% of the frames it appears in at 62° and
**2%** at 120°; one seed had a toy in frustum for 1182 frames at 1.5–2 m
and found it three times. The geometric floor for a 3.2 cm brick falls
from 1.83 m to 0.95 m, which in a 3 × 2.5 m room turns "scan the room"
into "bump into things": scanning goes from 8% of the run to 40%, and at
120° no seed ever reached 6/6. The basket is unaffected (hit rate
0.95–0.98 to 2 m at both lenses — it is 12 cm across and re-measured at
0.42 m before any release), grasp success is unchanged (0.87 vs 0.85),
falls are identical (8 vs 8), and release accuracy is if anything better.

**Then the third battery: more pixels at the lens the robot already
has.** 62° on a 640 px sensor is 591 px/rad against the shipped 296, the
size gate halving with it, and it buys **nothing**. Soccer, 12 seeds
paired against the same baseline: possession 17.5 → 17.0 s/min
(−0.49 ± 0.70, p = 0.50), `ballAdvance` +0.74 → +0.64 m/min
(−0.10 ± 0.10, p = 0.36), signed progress −0.01 → −0.06 m/min, 28 goal
events against 26 and 71 kicks against 97 (kicks −2.17 ± 1.15, p = 0.09 —
if anything fewer). Tidying, the same 12 layouts as the shipped lens:
0.917 against 0.875 (66 of 72 toys against 63), +0.25 toys a seed, better
on 5 seeds and worse on 2 — inside the noise on the cheap screen and on
the discriminator both.

So the whole recommendation, as one finding: **the shipped camera is
adequate, and what breaks it is widening the lens without adding pixels.**
Doubling the resolution at 62° changes neither task; spreading the same
320 px over 120° takes the tidy rate from 0.889 to 0.632 (worse on 24 of
24 seeds, sign p = 1e-7); and 120° on 640 px — 305 px/rad, which is what
the shipped 62°/320 px frame already had — tidies back inside the shipped
lens's noise. The variable is pixels per radian, and `DetectorSpec.px_h` is the
knob that says so. A wide lens is worth buying only WITH the sensor to
pay for it, and then what it buys is the soccer contact above, not a
robot that sees better. A 320 px sensor spread over 120° is a downgrade
wearing a wide-angle badge.

**Shooting only from inside the goal's cone — no effect measured.** One
shot in four scores, and a lone shot's direction error is 28–35°, so the
obvious move is to refuse the long ones: with `kick_cone`, a ball whose
goal mouth subtends less than that half-angle is dribbled closer (the
push spot) until the cone opens. It does exactly what it says and buys
nothing. At 0.35 rad (about a metre out on a 0.7 m goal): 36 kicks, 23
pushes, 16 goals, **1 kicked goal** against the baseline's 51 / 0 / 19 /
1. At 0.5 rad: 47 kicks, 30 pushes, 17 goals, **0 kicked**. Kicks turn
into pushes and the kicked goals do not move, so the shot that a
28–35° error scatters is not scattered any less from a metre out. Ships
at 0 (off).

**The head, unlocked, still cannot help.** The chase brain had capped
head yaw at 0.6 rad; the walker is trained to ±1.40 (upstream's head-pose
curriculum). With the cap at 1.4: the look after a kick yawed to the
foot's kick-map exit angle near the horizon (`look_aim`) 2.25 goals,
6.4 kicks, 0.50 falls; that plus a gaze on the ball track while
searching (`predict_s`, `head_yaw_when="search"`) 1.88 / 4.6 / 0.25;
the aimed look plus a searching head that sweeps ±1.4 rad
(`search_sweep`; old geometry) 1.88 / 6.8 / 0.38 — all against 2.38 /
7.4 / 0.38. Fewer kicks every time: a head turned away from the walking
line leaves the ToF bumper looking sideways, and the brain stops for
what it then sees. That last part was a real bug and is fixed
(`tof_clearance_bearings`, below); re-measured on the fixed geometry the
aimed look gives 61 kicks and 21 goals against a 51-kick, 19-goal
baseline, which does not clear the noise either. All three ship off
behind their flags — on the evidence that nothing has yet shown them
helping, not on evidence that they hurt. All three were re-screened on
12 fresh seeds with `possession` and `ballAdvance` ("The shelf,
re-screened", below); the sweeping head is the false positive in that
table.

Worth saying plainly, because the cap being raised to 1.4 reads like a
capability that shipped: **with all three off, the shipped brain never
yaws its head at all.** Every path that sets `look_at` is behind one of
them, so `head_yaw_max` is inert, and an instrumented 1v1 run confirms
it — `max |head yaw| commanded 0.000` over 1200 duck-seconds. The only
head motion shipping is the pitch that looks down at the ball.

And the reason none of them helped is now measured rather than guessed.
The walker's head is not the bottleneck: it tracks a yaw command to
1.42 rad *while walking*, at 7.5 rad/s — 11× the body's yaw rate — for a
12% forward-speed cost and no falls, which turns a 62° camera into a
210° gaze cone for free. Handing the brain that cone does not buy back
one second of body rotation: with `search_sweep` on, the body turned
**more**, on 5 of 5 paired seeds (+8.6% of the in-place yaw, +9.2% of
the total, n ≈ 150,000 ticks an arm). It finds the ball sooner and then
still has to turn the body to it. **Finding the ball was never the
bottleneck; pointing the body at it is** — see the next paragraph.

**Where the run actually goes: 47% of it is the robot spinning on the
spot.** Instrumenting a 1v1 run (1200 duck-seconds, commanded twist and
achieved body yaw every tick) says the chase brain spends 47.1% of it
turning in place and another 26.6% steering while walking — sweeping
371 rad, twenty-odd full revolutions a duck a run, at 0.655 rad/s. That
0.655 is the walker's ceiling and `ANG_VEL_Z_RANGE`'s ±1.0 is already
asking for all of it. Holding the yaw demand fixed, a walker that turned
at 1.0 rad/s would free 16% of every run, and one at 1.5 rad/s would free
**27%** — bigger than any brain-level change measured in this repo, and
the reason roadmap 3.7 was rewritten from "head-aware locomotion" (done,
by upstream, already) to "a faster body yaw" (not done, and the ceiling).

This is a claim about time, not about falls. Turning beside a *static*
body fell 0 times in 98 trials down to 8 cm of separation; the falls need
a duck that is *moving* into you. Do not sell a faster turn on falls.

**Clearance by bearing, not by sensor column.** The ToF is *in the
head*, so calling the middle columns "ahead" only holds while the head
looks along the walking line. Yawed 1.2 rad at boards 0.40 m away, those
columns reported a wall at 0.52–1.19 m that was really 69° off the nose,
and the brain stopped for it. `tof_clearance_bearings` places every hit
in the body's heading frame and selects by bearing, so a turned head
reads +inf ahead: honestly blind rather than confidently wrong. Ranges
stay aperture-relative so the tuned thresholds mean what they meant, and
a synthetic frame with no mount pose still falls back to the level-head
columns. It ships on its mechanism, not on a score: traced against the
column version over two full matches, the shipped brain never yaws its
head at all (the gaze is off), the two disagree about stopping in 0.7%
of samples, and the bearing version reads 1.2 cm nearer on average. The
batteries agree to within their noise (19 goals / 51 kicks against 19 /
59). What it buys is that the head experiments above are now *testable*.

**The ToF sees the ball at the feet — and ships off.** The level camera
loses a floor ball inside 0.3 m, exactly where the line-up and the kick
live, and the 8×8 ToF at 45° shows a 7 cm ball at 0.3 m as a 3–6 zone
blob 3–10 cm above the floor plane (`tof_floor_ball`: at least two
adjacent such zones, in columns with nothing taller near — a wall or a
duck has hits above the band in the same columns, a ball has the floor
behind it; tested against a ball, an empty floor, a level head and the
boards). Fed to the chase brain's tracker as a ball sighting whenever the
camera has none (`tof_ball_m`), it measured **1.62 goals, 8.1 kicks,
0.75 falls a run against 2.38 / 7.4 / 0.38**: a blob at the feet is as
often the other duck's foot as the ball, and a line-up on a foot is a
fall. The detector stays for the page and for a pitch with one duck on
it, and the knob was re-screened on 12 fresh seeds below without coming
back.

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
rather than shadow. The scoreboard that counts possession exists now
(`world/metrics.py`, above); positional play does not.

**3v3 falls, traced** (3 seeds × 300 s, 14 falls): 10 were supporters
turning in place with a teammate 5–28 cm away or against the boards —
a body beside the duck is outside the camera's 62° and the ToF's 45°,
so neither the avoid rule nor the wall rule saw it. Two answers: the
support spot stays 0.35 m inside the pitch (the brain gets the pitch's
bounds from `make_pitch`), and a supporter with any duck track inside
0.3 m within the last 1.5 s (a track's bearing turns with the body, so
a duck seen a second ago still says where it is) stands instead of
turning in place. 3v3 over 4 seeds: **1.00 goals, 7.8 kicks, 3.50 falls
a run (0.58 per duck)**, from 0.75 / 5.2 / 4.75 (0.79). With the hunt
and the walking search (below, measured on 1v1) the same 4 seeds give
2v2 1.50 goals, 9.8 kicks, 3.50 falls a run (0.88 per duck; 2.00 / 7.8
/ 2.75 without) and 3v3 1.50 / 5.0 / 4.50 (0.75; 1.00 / 7.8 / 3.50
without): more goals in 3v3, more falls in both — four seeds of six
ducks walking blind lines through a crowd. `hunt_s` and `search_vx`
switch them off per brain if a roster prefers it.

**Teammates on the board** (`Team.mates`): every claim now carries the
claimant's own pose too — the same one-message-a-second — because a
teammate beside or behind a duck is invisible to its camera and its ToF.
The chase brain can treat a teammate inside `mate_keepout` as a duck
beside it (no turn in place, no hunt) and, ahead, as a duck to avoid.
**Measured off** (3v3, 4 seeds × 300 s): 1.50 goals, 4.5 kicks, 5.25
falls a run with it at 0.4 m against 1.50 / 5.0 / 4.50 without. A fresh
trace of two seeds (13 falls) said why: 12 were beside an *opponent* —
which no board carries — and all but one were standing turns, most of
them a supporter at its spot turning to face the ball. The poses stay on
the board (the inspector shows them); the keep-out ships at 0. Making
that turn a walking one (`support_turn_vx` 0.2, like the search circle)
measured off too: 2v2 1.50 goals, 8.8 kicks, 4.00 falls a run against
1.50 / 9.8 / 3.50, 3v3 1.00 / 4.2 / 5.25 against 1.50 / 5.0 / 4.50 — a
walking turn bumps what it cannot see. What a crowded pitch needs is a
sense of the bodies beside the duck — a wider ToF field, or the bump the
IMU could read — before any rule can act on them.

**The attacker claim is a predicted TIME to the ball, not a distance**
(`brain/team.py`). Traced, the role churned: it changed hands **14.0
times a duck a run** with a **median tenure of 4.30 s**, a fifth of the
spells lasted under a second, and the duck the board called the attacker
was the team's actually-nearest one only **56.1%** of the time. A
straight line was wrong in three ways the trace named. It **ignores the
turn** — this walker turns in place at ~0.7 rad/s once the gait is going
and walks at 0.45 m/s (`walker-facts`, `ChaseParams.speed`), so a duck
facing away at 0.4 m is 4.7 s from the ball and one facing it at 0.6 m is
1.0 s, and the line sent the second one back to its support spot. It read
**losing sight as resignation** — the chase brain claims `inf` when its
track goes cold, and the level camera loses a floor ball inside 0.3 m,
which is exactly where an attacker lines up, so the duck ON the ball
handed the role to one a metre away and walked off. And **a stale claim
competed on equal terms**. The cost is now a predicted time built from
those measured walker facts; a duck that cannot see the ball is costed
off the board's freshest fix plus `blind_s`; age is priced at `age_rate`
a second and stops counting past `stale_s`; and the role moves only for a
challenger `switch_s` = 0.6 s quicker held `hold_s` = 1.2 s continuously,
unless it is `give_up_s` = 2.0 s quicker (the incumbent is out of the
play — fallen, or the ball kicked past it).

Measured over 6 seeds × 300 s of 3v3, **516 attacker spells against
365**, and replicated on three seeds it was not tuned on: **handovers
14.0 → 9.8 a duck a run, median tenure 4.30 → 6.96 s**, spells under a
second 21.3% → 10.4%, **the attacker really the nearest duck 56.1% →
68.2%** and really the quickest 68.4% → 79.3%.

**What it does not do, stated plainly.** It does not improve the play:
over 16 paired seeds kicks rose 64% and `ballAdvance` +0.18 ± 0.06 (2.9
σ), but that metric keeps only forward motion and churn inflates it —
signed `ballProgress` is **flat** (−0.003 ± 0.136, 0.0 σ), advance per
kick **halved** (0.202 → 0.106) and possession is down 13%. The ball
moves more and gets no nearer the goal. And it does not fix the crowding
the defect report opened with: nobody within 0.3 m of the ball 41.2% →
40.9%, two teammates inside 0.5 m of it 24.5% → 23.7%. Role churn was not
what kept six robots off the ball. Aiming the claim at an intercept
(`lead_max_s`: ball velocity plus lead) was measured **worse**, not
merely unhelpful — 18.2 handovers a duck a run against 12.3, a median
spell of 3.0 s against 5.8, 31% of spells under a second against 13%,
over the 3 seeds the hysteresis was swept on — because the board's
velocity is differenced from fixes a walking duck drags between frames;
it ships at 0. So this change stands on role stability and correctness,
measured on hundreds of events and replicated, and claims nothing about
the score.

**A bump sense** (`Senses.bumped`), and a lesson about four seeds. The
World reads its contact list — in the walk scene only the feet carry
collision geometry, so a bump is feet touching feet, which is what a
duck-duck fall is; on the robot it is the IMU and the servo loads — and a
chase brain that has been bumped stands instead of turning in place for
`bump_stand_s`. Over 4 seeds × 300 s it looked decisive — 3v3 falls 5.00 →
1.75 a run — and **that number did not replicate**: the same rule over
twelve seeds gives 3.17 and over twelve others 4.17. So it was measured
properly, the rebuilt rule against no rule at all — **and it did not
survive the confirmation either.** Three batteries, 3v3, 300 s a seed:

| block | no rule | rule at 0.5 s | falls a run | |
|---|---|---|---|---|
| seeds 24–35 | 4.83 | 3.25 | −1.58 ± 0.92 | p = 0.14, better on 8/12 |
| seeds 24–35, again | 6.17 | 4.00 | −2.17 ± 1.01 | p = 0.060, better on 8/12 |
| **seeds 200–211, fresh** | **4.08** | **4.33** | **+0.25 ± 1.04** | **p = 0.88, better on 5/12** |
| **all 24 distinct layouts** | | | **−0.81 ± 0.69** | **p = 0.264, better on 15/24** |

The first two batteries are the same twelve layouts measured twice, not
24 seeds — averaged per layout they give −1.88 ± 0.84, p = 0.055 (pooling
them as 24 says p = 0.012, which is repeated measures, not replication,
and was written here first). On twelve layouts nobody had run, the effect
is **absent and slightly reversed**. Pooled over all 24 distinct layouts
it is −0.81 ± 0.69, p = 0.264. **The claim "a third fewer falls" is
withdrawn.** This is the poacher's shape exactly — found, reproduced on
its own seeds, gone on fresh ones — and the rule that catches it is this
repo's own third: confirm on seeds the effect was NOT found on.

**And backing out instead of standing changes nothing either — which is
the most closed null here.** Standing was measured never to end a
contact (from 0.10 m of separation, 16 trials, a standing duck is still
at 0.099 m four seconds later and clears 0.30 m in 0 of them, where a
straight reverse clears it in a median 1.6 s), and the walker reverses at
0.23 m/s once you leave the dead band. So `bump_back` backs out on
exactly the stand's gate. A third arm on the same twelve fresh layouts:

| 3v3, seeds 200–211 | falls | kicks | goals | possession | advance |
|---|---|---|---|---|---|
| no rule | 4.08 | 6.50 | 2.17 | 11.83 | 0.40 |
| stand | 4.33 | 7.33 | 1.58 | 13.00 | 0.42 |
| back out | 4.33 | 6.67 | 2.00 | 11.97 | 0.44 |

`stand → back` on falls is **exactly 0.00 ± 1.13, p = 1.000** (52 events
against 52), and nothing else resolves. The knob is not inert: over one
instrumented 3v3 run it issues 690 reverse commands (13.8 s a run) and
drops the ticks spent touching another body from 4473 to 2529 of 90 000 —
**a 43% cut**, which is precisely the self-feeding the 838-bump trace
found ("standing on a body keeps touching it"). The mechanism is real, it
operates, and it does not matter. **Time in contact is not what makes a
3v3 duck fall** — which agrees with the probe the idea came from: a turn
beside a *static* body fell 0 times in 98 trials down to 8 cm, and every
fall in those probes needed a duck that was *moving* into you. The
remaining lever is the closing duck, not the contact.

The stand still ships on for rosters (`team_bump_stand_s`), and that is a
default nobody has earned in either direction. Over all 24 layouts the
point estimate still favours it (−0.81, better on 15/24) and nothing it
was suspected of costing moved — kicks +0.83 (p = 0.51), goals −0.58
(p = 0.50), advance and signed progress flat, on the confirmation block —
so flipping it off now would be reading noise in the other direction.
Falls need ~376 seeds to resolve a shift this size and this is 24: the
question is open, not settled, and the honest label is "not shown to
help", the same as every other knob on the shelf. Possession did move on
the fresh block (11.83 → 13.00, +1.17 ± 0.48, p = 0.034) — one arm in six
at p < 0.05 is what chance produces about a quarter of the time, and
possession is the screen, never the verdict.

Its first form did have a real defect, found by tracing 838 bumps, and
the trace is worth more than the numbers above. The obvious premise — two
attackers' feet meet at the ball and the one that stands loses it — is
**false**: the feet meet a median 0.66 m from the ball, both ducks are
inside 0.35 m of it in 18% of bumps, and two seconds later the ball is
further from *both* by the same 7.4 cm. What the rule actually did was
cancel the *escape*: 70% of its firing was in `blocked`, a state that is
12.6% of the run, where the walk is already zeroed and the turn is the
only command left — 6 of 8 falls were a stand leaning on the other duck —
and 5 more were in `search`, whose circle walks. And it fed itself:
standing on a body keeps touching it, so the timer never expired, bumps
went 44 → 105 a run and one freeze ran 74 s. It is edge-triggered from
the onset of a contact and scoped to the states where a standing turn
beside a body is the danger.

**A poacher supporter — found, confirmed, then killed by fresh seeds.**
Supporters shadow the ball (traced: 0.89 m from it on median, 29% of
their time inside 0.5 m, while 41% of the run has no duck at all within
0.3 m of it), so standing them between the ball and the goal they attack
looked obvious. It scored 10 goals against 3 over four seeds, then 21
against 12 over twelve. On twelve seeds nobody had looked at, it
**reversed**: 13 against 19. Pooled over all 24, 34 against 31, p = 0.80.
The middle battery was never independent — it contained the four seeds
the effect was found on. `support_mode="ahead"` stays as the record of
it.

**The shelf, re-screened — and nothing came back.** Most of the knobs in
the index at the top of `ChaseParams` ship off on differences that never
cleared the noise: "measured off" meant "not shown to help", and they were
judged on goals at 8 seeds, which resolves nothing. `possession` resolves
a 25% shift in 9 seeds, so re-screening them is an afternoon. Seven
knobs and a baseline, eight batteries of **12 seeds × 300 s of 1v1 each,
on seeds 40–51 — fresh ground, and the same seeds for every arm so every
reading is paired** — the baseline run interleaved in the same window on
the same code (2.17 goals, 8.08 kicks, 0.67 falls a run; possession
17.50 s/min, signed progress −0.01 m/min, advance +0.74 m/min, 0.091 m of
advance per kick):

| knob (what it is) | possession Δ (s/min) | signed progress Δ (m/min) | advance Δ (m/min) | advance / kick | kicks Δ (a run) |
|---|---|---|---|---|---|
| `predict_s` 3.0 (head gaze at the predicted ball) | +2.04 ± 1.69 (p = 0.25) | +0.01 | +0.00 | 0.091 → 0.083 | +0.83 |
| `look_aim` (the look after a kick aims by the kick map) | +0.21 ± 1.41 (p = 0.88) | +0.07 | +0.01 | → 0.105 | −1.00 |
| `search_sweep` 1.4 (a searching head that sweeps ±1.4 rad) | **+3.75 ± 1.21 (p = 0.010)**, up on 10/12 | +0.11 | −0.01 | → 0.073 | +1.83 |
| `tof_ball_m` 0.5 (the ToF's ball at the feet) | +1.83 ± 1.81 (p = 0.33) | +0.21 | +0.17 | → 0.118 | −0.33 |
| `seek_s` 8.0 (a ball memory the search walks to) | +1.68 ± 1.85 (p = 0.38) | +0.34 ± 0.29 (p = 0.26) | +0.26 | → 0.148 | −1.33 |
| `two_stage` (the two-stage line-up) | −3.47 ± 1.98 (p = 0.11) | +0.04 | −0.24 ± 0.10 (p = 0.034) | → 0.163 | **−5.00 (p < 0.001)** |
| `push_beyond` 1.4 (deliberate dribbling) | +2.30 ± 1.86 (p = 0.24) | +0.12 | −0.04 | → 0.136 | −2.92 (p = 0.047) |

**Read the `advance / kick` column as 1/kicks, not as kick quality.** It
is arithmetically advance ÷ kicks, and `ballAdvance` is flat in every row
here, so the column only ever reports its denominator: an arm that kicks
less "improves" it. Measured on three line-up arms whose kick counts
differ 2.6× over the same 24 seeds (185, 72, 83), total advance is
statistically identical (0.400, 0.360, 0.342 m/min, every pairwise
p > 0.17), while the ball's *actual* travel in the two seconds after each
swing goes the other way — 17.7 ± 3.9 cm for the arm with the worst
ratio, against 14.4 ± 3.4 and 13.6 ± 3.0 for the two with the best. The
column stays in the table because it was quoted in the decisions below
and deleting it would hide that; it is evidence of nothing. The rule it
cost is in `AGENTS.md`: a ratio whose numerator is flat is its
denominator, upside down.

**The one arm that cleared p < 0.05 is the instructive row, and it is a
false positive.** The head sweep took possession from 17.50 to 21.25
s/min — +3.75 ± 1.21, p = 0.010, up on 10 of 12 seeds — and every reading
churn cannot inflate says *worse*: `ballAdvance` −0.01 (flat), goals
2.17 → 1.17 a run (−1.00 ± 0.52) and falls 0.67 → 1.67 (+1.00 ± 0.49, 8
events against 20). A duck whose head sweeps
stays near the ball and does less with it, which is precisely what
possession pays for. That is the metric working as designed — it detected
a real behavioural difference — and it is why possession is the screen and
never the verdict. Seven arms tested at p < 0.05 also carry a
30% chance of throwing at least one "significant" result from nothing at
all; this is what one looks like from the inside.

**Ball memory, the last candidate: dead at 48 seeds.** `seek_s` was the
only arm whose signed progress looked like anything (+0.34 on the twelve
discovery seeds), so it was carried onto fresh ground twice: **+0.20 over
the next twelve seeds, then −0.11 over the twenty-four after that.**
Pooled over all 48 paired seeds: **+0.08 ± 0.10, p = 0.45, up on 25 of
48** — a coin flip. It never reversed the way the poacher did; it
dissolved, which is what a null effect looks like when seeds keep being
added to it. (Run together with `tof_ball_m` on the second twelve, the
pair measured worse than either alone: −0.13 m/min of signed progress
against the baseline's −0.05.) **Nothing on the shelf was rehabilitated**,
and every knob above still ships off — now on a paired 12-seed reading
with a metric that could have found an effect, which is a different and
better kind of "off" than the one they had before.

### Tidy the playroom (roadmap Track 12)

`playroom` is the built-in that scatters toys on the floor of a walled room
with a low basket in a corner, and `tidy` is the brain that clears it:

```bash
uv run duck-lab --world playroom          # watch it on /sim; the tidy score is top-left
uv run eval-tidy --seeds 3 --toys 6 --seconds 300   # the benchmark: toys in the basket
```

**Long batteries are resumable.** A 16-seed tidy battery or a 12-seed 3v3
one is well over an hour, so `eval-tidy` and `eval-pitch` print each seed
the moment it lands and, given `--out FILE`, append it there as a JSON
line and skip on a later run whatever the file already holds. Re-running
the *same command* after an interruption continues it; `--seed0` extends a
finished battery onto fresh seeds instead of redoing the same ones. The
brain's own parameters never appear in a row, so `--tag` says which
variant a file belongs to and a resume **refuses** a file written under a
different tag, roster or run length rather than fabricating a comparison
out of two halves. This was written after a cloud container was reclaimed
twice mid-battery and, because the results were buffered to the end, took
about ninety minutes of measurement with it both times.

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
| ideal | 250 ms round trip | 0.81 | 0.19 |

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
Wi-Fi keeps most of the tidying (0.81 against 0.89) and falls **no more
than the onboard brain** (0.19 against 0.31). It read 0.76 / 2.19 before:
every traced tethered fall was the stopping stride at the rim, because
the stop was decided on senses a quarter of a second old (4.7 cm of
overshoot at a 0.3 command, 3.0 at 0.25, which is why the last leg is
walked at 0.25), and the fix is the one thing a tethered brain *can* do
about its link — know it. The tether is modelled honestly now
(`brain/tether.py`: senses reach the brain half a round trip late,
re-aged; its intent lands half a round trip later — the sim used to
delay only the intent, which let the brain see senses it would never
get), so the brain reads the link off its own sensor ages: the floor of
the ToF age over the last second is the one-way lag (near zero onboard,
the sensor runs at 15 Hz), twice it is the round trip, and every stop
moves out by its speed times that (`latency_gain`; ~4 cm at 0.16 m/s and
250 ms). At the rim that was the margin every traced fall had spent
(nine of ten falls in four traced runs were within 0.6 s of a release
with the trunk 0.22–0.24 m from the basket): 0.71 / 0.25. At the toy it
was the time: traced, the tethered seed 0 picked 3 toys in 8 attempts
after its first three (onboard 5 in 5) and spent 80 s scanning, because
the pick's stop landed late too and the beak came down past the toy —
with the margin on the pick as well, 0.81 / 0.19. The 50 Hz reflex stays
onboard either way.

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
`--tether-ms` (roadmap 12.10) is a brain round trip — senses out half of
it late, intents back the other half (`brain/tether.py`); `POST
/world/tether` does the same live on the page, and the inspector's
`latency` is what the tidy brain reads off its sensor ages.

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
