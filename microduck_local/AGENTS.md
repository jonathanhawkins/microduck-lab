# Training playbook for agents (and humans)

Read this before touching rewards, observations, behaviors, or training code.
It encodes the invariants of the deployment contract and the lessons this
project paid for in wasted training runs. `README.md` covers what each command
does; this file covers how not to fool yourself.

## The invariants (do not negotiate with these)

- **61-dim obs / 14-action contract.** Policies are hot-swapped on the robot
  behind one shared ONNX interface (`contract.py`, mirrored from upstream
  `microduck_rl/scripts/infer_policy.py`). Never change the obs layout
  per-task. Unused command slots are zero-padded and keep tiny sampling
  ranges so the normalizer stays alive — never removed.
- **Penalty terms are ≤ 0 by construction.** A training callback aborts the
  run if any episode's penalty sum goes positive (a double-negation bug
  shipped once; the guard stays).
- **Domain randomization restores compile-time defaults before applying.**
  It must never accumulate across resets. `tests/test_env_contract.py` locks
  this.
- **ONNX ships with the obs normalizer baked in** (`export-walk`). A raw
  checkpoint is not a deliverable.
- **`joint_vel` lags one control step** (Dynamixel moving-average), matching
  training and hardware. Don't "fix" it.
- Tests in `tests/` are contract locks, including a regression test that the
  shipped `alpha_walking.onnx` survives upright in this env. Run
  `uv run --with pytest pytest tests/` before and after your change.

## Reward design rules

- **Never reward what the policy cannot observe.** The obs contract has no
  yaw and no world position. A heading- or world-anchored reward term is
  unlearnable noise — one such term produced 30M steps of circling that
  looked like an optimizer problem. Straightness/heading belongs to the
  velocity commander, not the reward. Before adding a term, point at the obs
  indices that let the policy see what you're scoring.
- **Mind MuJoCo's velocity frames.** `mj_objectVelocity(..., flg_local=0)`
  is the world frame; naive indexing once rewarded a sideways shuffle as
  "forward". When you write a term that reads a velocity, print it in a pose
  you can reason about first.
- **No jackpots.** Dense, bounded shaping beats sparse windfalls; a term a
  policy can spike once and farm will get farmed. (Inherited from upstream
  `microduck_rl/AGENTS.md`, which is worth reading in full.)
- Composable extras belong in the **term catalog** in `behaviors/core.py`
  (`head_up`, `flat_feet`, `calm_body`, `smooth_torque`, …) so the viewer's
  teach panel can offer them as sliders. A trick that "looks wrong" usually
  needs a catalog term, not a new bespoke one.
- **Reward design cannot fix an exploration gap.** If rollouts never contain
  the target skill — not even transiently, not even stochastically — no term,
  weight, gate, or ramp will teach it: the value of a state that is never
  sampled is never learned. This cost a full night on the headstand: five
  recipe variants (extended pay ramps, persistence bonuses, 3× gradient on
  the missing motion, penalty/terminal removal, still spawns) all converged
  to the same easy local pose, because a held *extended* headstand is not
  samplable under honest BAM servos from random behavior. The fix was never
  in the reward: ladder the **physics** — a strong-servo (`xml`) drill stage
  with ~80% dropped-in spawns made the skill appear in rollouts within
  minutes, then the servos step back down to honest BAM. Diagnose with one
  question before touching terms: "does ANY rollout ever do the thing?" If
  no, change the world, not the pay.
- **Curriculum stages may ladder only physics, spawns, and strictness —
  never the reward.** The first staged era failed because rungs carried
  their own term edits and every rung grew its own exploit to nap in. The
  headstand ladder keeps the identical sealed term set in every stage
  (a test enforces the stage-env knob allowlist), so a stage can make the
  world easier or the judging stricter, but never change what is paid.
  The viewer's per-stage weight sliders (`/teach` `stageWeights`) are the
  deliberate exception, and the asymmetry is the point: a human watching a
  rung stall can re-price it live, but what they produce is an experiment,
  not a recipe. `CurriculumStage` has no weights field to persist one into,
  so a stage tuning that works has to earn its way back as a term or a
  physics knob before it can ship.

## How much can the benchmark actually resolve? (read before any A/B)

Every rule below exists because it was broken on 2026-09-03 and cost a day
of wrong conclusions. Two results were published from four-seed batteries
and later reversed; several "measured off" verdicts turned out to be noise.

1. **Count the EVENTS, not the runs.** An 8-seed x 300 s 1v1 battery holds
   ~50-130 kicks but only ~20 goals and 3-8 FALLS. The same brain measured
   twice gave 3 falls and 6 - a chance split (p = 0.5). Quote a difference
   with its event totals or do not quote it.
2. **Know what your metric costs.** Measured, 16 seeds an arm: to resolve a
   25% shift at p<0.05 / 80% power you need **goals 146 seeds, falls 376,
   kicks 62, ballAdvance 43, possession 9**. If you are about to decide
   something on goals at 8 seeds, you are about to decide it on nothing.
   `eval-pitch` prints `ballAdvance` (the discriminator) and `possession`
   (the cheap screen); goals stay reported and are not the judge.
3. **Confirm on seeds the effect was NOT found on.** A "confirmation" that
   re-uses the discovery seeds is not one. A poacher supporter scored 10
   goals against 3 over four seeds and 21 against 12 over twelve - the
   twelve CONTAINED the four - then reversed on twelve fresh ones, 13
   against 19, for 34 against 31 over all 24 (p = 0.80). `--seed0` exists
   so a battery extends onto fresh seeds instead of re-running the old ones.
4. **Prefer a paired reading.** Both arms run the same seed layouts, so
   report per-seed wins/losses alongside the totals; it is strictly more
   powerful than comparing two means.
5. **Ask what would inflate your metric.** `ballAdvance` keeps only the
   forward part of the ball's motion, so anything that makes the ball move
   MORE scores higher without moving it anywhere: the handover fix raised
   it 2.9 sigma while signed `ballProgress` stayed flat (0.0 sigma). Read
   advance and signed progress together, and for any metric ask first which
   cheap behaviour maximises it.
6. **A ratio whose numerator is flat is its denominator, upside down.**
   "Advance per kick" was read here as kick QUALITY and it is not one.
   Three line-up arms whose kick counts differ 2.6x (185, 72, 83 over the
   same 24 seeds) have statistically identical total advance (0.400, 0.360,
   0.342 m/min, every pairwise p > 0.17) — so the ratio moved 0.052 ->
   0.120 -> 0.099 purely because the denominator fell. Measured DIRECTLY
   (ball travel in the 2 s after each swing) the arms with the flattering
   ratio kicked the ball LESS far: 17.7 +/- 3.9 cm against 14.4 +/- 3.4 and
   13.6 +/- 3.0. Before quoting a per-X figure, test the numerator on its
   own; if it does not move, you are reporting 1/X with extra steps, and it
   will point whichever way costs you the most to believe.
7. **"Measured off" usually means "not shown to help".** Say which one you
   mean. Several knobs in `ChaseParams` ship off on differences that never
   cleared the noise; re-screening them with `possession` is cheap and at
   least one of those verdicts is probably wrong.
8. **A battery must survive the machine.** Use `--out FILE --tag TAG`:
   every seed is appended as it lands and a re-run of the same command
   skips what is already there. A cloud container reclaimed mid-run cost
   about ninety minutes of 3v3 twice before the benchmarks streamed. The
   tag is refused if it disagrees, so two variants can never be stitched
   into one comparison.

## Verification discipline — the rules that exist because of false reports

1. **Training charts measure the noise-crutched stochastic policy.** Claims
   about a run are made from the deterministic exported ONNX, never from
   `ep_rew` curves. Export, then eval, then look.
2. **Look before you conclude.** `uv run render-rollout` writes an mp4 for
   humans and a captioned frame contact sheet for agents — read the sheet
   (`.claude/skills/render-rollout/SKILL.md` documents every caption field
   and the three classic failure patterns: the collapsed "stand", the
   spawn-assisted "trick", the cycling "hold"). Reward batteries here have
   scored a face-down crouch as standing.
3. **Render a null control** (`--policy limp` / `--policy zero`) before
   crediting the policy with anything a spawn pose or gravity could have done.
4. **Throughput is not learning speed.** Two "optimizations" (overlapped
   updates, big-batch) each raised steps/s ~25–40% and *halved*
   reward-per-step. Any change meant to make training faster gets a
   seed-matched A/B at matched *step counts* before it becomes a default.
   `uv run bench-ab <a> <b>` is that comparison.
5. **One training run per arm resolves nothing — pair the seeds.** Eval
   seeds control the *eval*, not the run. On the follow benchmark,
   run-to-run variance is **±0.02 in band**, larger than the ±0.013–0.023
   eval-seed spread and larger than every hyperparameter effect measured
   against it. Three changes each "lost" 0.015–0.018 and were "ahead on only
   1–2 of 10 eval seeds" — then the same recipe at a different *training*
   seed moved 0.021, and the whole result was one run's luck. Train both
   arms on the **same** seeds and compare the per-seed DIFFERENCE
   (`bench_ab.paired_delta`); that turned a spurious −0.015 into a real
   −0.002. Use Student's t, not 1.96: at n=2 the normal value understates
   the interval 6.5-fold and manufactures significance out of two runs.
6. **A weird optimizer metric is usually a broken reward.** A KL blow-up here
   was chased as a PPO tuning problem for a day; the cause was an unlearnable
   reward term. Check what you're asking for before re-tuning how hard to
   ask.
7. **Warm-start chains silently ratchet the action std into bang-bang.**
   Every `--init-from` reloads the previous run's `log_std`, and the entropy
   bonus pushes it up each generation; one long chain reached std 21–26 in a
   ±4 action space. At that point the *clipped noise distribution* carries
   the behavior (stochastic episodes survived 6.9 s) while the exported
   deterministic mean is saturated garbage (fell in 0.5 s) — and telemetry
   cannot tell the difference. `LOG_STD_MAX` in `train_behavior.py` caps it
   on load and every rollout; a mean-poisoned lineage cannot be consolidated
   and must be restarted from scratch. Probe `live.onnx` deterministically
   at every checkpoint — that is the policy that ships.
7. **An eval env that carries state between episodes hides the tail.**
   `BrainEnv` used to reseed nothing on reset: the ToF's, the detector's and
   the world's generators were seeded once at construction, and `_respawn`
   left the commanded twist standing, so episode 0 reproduced and every
   episode after it continued the one before. The cell MEANS barely noticed
   (re-measuring both follow tables independently moved no cell by as much
   as one seed-level sigma) — what it cost was resolution: v4's lead over
   v5 clears the seed noise in three cells of four on independent episodes
   and in NONE of them chained, because the comparison turns on v5's bad
   episodes and carried noise smears exactly those. If a battery is the
   evidence, make an episode a pure function of `(seed, ep)` and pin it
   with an exactness test (`tests/test_eval_brain_jobs.py`); a battery you
   can shard is also a battery you can trust.

## Adding a behavior (the main community extension point)

1. Add a `Behavior` to the trick module it belongs in under
   `src/microduck_local/behaviors/` (one file per trick family; shared
   helpers and the catalog live in `core.py`): reward terms
   (signed per the rules above), friendly strings, chat keywords, optional
   `curriculum` stages for hard tricks (see `backflip` for the pattern —
   reverse curriculum via spawn-family env knobs, staged `--init-from`
   chaining).
2. Lock it: extend `tests/test_behaviors.py` (term signs, keyword matching).
3. Train it: `uv run train-behavior <id>` — or better, through the lab
   (`POST /teach` or the viewer's 🎓 panel) so you and anyone watching the
   browser see live snapshots every ~15 s. Prefer the lab path when a human
   is in the loop; CLI runs are invisible to the viewer.
4. Verify per the discipline above, including from plain standing starts
   (`--env MICRODUCK_SPAWN_FAMILY_PROBS=0.0,0.0` for spawn-curriculum tricks).
5. Iterate weights from the viewer's sliders (`/teach` `weights` +
   `initFrom` fine-tuning) rather than editing numbers blind.

## Performance work

`README.md` documents the measured optimization history (shared-model fork
vec env, semaphore IPC, numba BAM kernels and the fused substep, MPS updates,
the per-machine thread profile) including the ones that were **rejected** —
for hurting learning, for not reproducing, or for costing more than they
bought. Follow that precedent: measure
with `bench-envs` (real PPO, not raw stepping), and A/B learning quality
before shipping any throughput win as a default.

**Mac is the default; other machines get a profile, not a rewrite.**
`machine.py` picks a per-machine thread policy, and the `mac` profile
reproduces the historical settings term for term —
`tests/test_machine.py` pins that, including that a Mac run gets *no* extra
callback in its training loop. When you find that a tuning constant here was
measured on an M5 Max and is wrong elsewhere (they mostly were), add it to a
profile rather than changing the shared default. Three rules bound what a
profile may contain:

- **Only quality-neutral knobs.** Thread counts do not enter the PPO math.
  The env count is the line: it sets the PPO batch size and therefore the
  learning dynamics, so `--envs` stays 32 on every machine and is never
  profiled. Verification discipline #4 applies to a profile like anything
  else.
- **A profile earns each knob separately, against the same window.** Worker
  packing shipped in the first draft of the linux profile on one point that
  showed +7%; four interleaved reps then put it behind 1:1 at every env
  count and it was removed. Measure each knob with the others held fixed
  (`MICRODUCK_ENVS_PER_WORKER=1` isolates packing from the thread split),
  and never let a bundle of changes ride on one arm's total.
- **Measure the profile you ship, on the machine you ship it for.**
  `uv run bench-envs --compare-profiles` runs both arms interleaved with the
  same repeats, which is the only form of that comparison worth reading
  (`--profile mac` on a Linux box reproduces the old behavior exactly).

**Prototype the win before you plumb it.** A profile decomposition tells you
where time *is*, not what removing it *buys* — the two differ once the pieces
interact. The double-buffered rollout was estimated at ~18% from the vec-step
split (hide the parent's 1.95 ms of forward + dispatch behind worker
compute); prototyped in 60 lines with two independent vec envs and no repo
changes, it measured +6.2/+7.1/+6.2% at 8/16/32 envs, because splitting pays
a second wait's sync cost and two half-batch forwards cost more than one full
one. That prototype cost an hour; the real version would have rewritten the
rollout buffer path. **When a change would touch a correctness-critical
seam, build the throwaway that measures it first** — and let the measured
number, not the estimate, decide whether to build the real one.

**Price a change by what it redefines, not by its diff.** The same rollout
split was rejected at ~5% end-to-end because it changes *what a step is*:
`test_overlap.py::test_collect_matches_stock_sb3_bitwise` pins the vendored
collect loop against stock SB3 and a split fleet cannot satisfy it,
`VecNormalize` updates `obs_rms` once per step so halves change the running
normalizer that gets baked into every exported ONNX, and the rollout buffer's
`add()` takes a whole row. A change that forces you to delete an invariant
test is not a throughput change, it is an architecture change; hold it to
that bar. The three that did ship (thread policy, numba warm-up, the fused
BAM substep) are all provably invisible to the math and each *added* a test
rather than removing one.

**A cloud VM is not a stable ruler.** On the box these profiles were measured,
the *identical* script and configuration ran 13.1 s in one window and 19.5 s
an hour later — a 49% drift with nothing changed, from noisy neighbours and
CPU-credit throttling. That is larger than every optimization in this file, so
a number compared against one taken earlier is worthless, and this bit almost
shipped a false result here: a configuration re-measured in a later window
looked like a regression in code that was provably not running (the callback
was verified firing, and disabling the change reproduced the same "slow"
number). Rules that follow, on any shared or virtualized machine:

- **Interleave the arms.** Every comparison runs its arms back to back inside
  one window and repeats the whole cycle, which is what `bench-envs
  --repeats N` and `--compare-profiles` already do. Never quote arm A from
  this hour against arm B from the last one.
- **Re-measure the baseline whenever you re-measure anything.** A surprising
  result is a drifted machine until the baseline says otherwise.
- **Prefer ratios within a window to absolute steps/s across windows.** The
  absolute numbers in `README.md` date a specific machine on a specific day;
  the ORDERING of the arms is the transferable part.

## Sim2real honesty

This harness is for prototyping with minutes-long feedback loops. Even under
`actuator="bam"` it is a subset of the official domain-randomization stack.
Once a behavior works here, port the env design to an mjlab cfg in upstream
`microduck_rl` and retrain on GPU — that stack, not this one, is the recipe
for policies that survive real hardware.
