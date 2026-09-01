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
uv sync                                     # one-time (needs ../microduck_rl checked out)
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
└── bench.py        # steps/sec benchmark
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
