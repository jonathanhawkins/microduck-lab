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
5. **A weird optimizer metric is usually a broken reward.** A KL blow-up here
   was chased as a PPO tuning problem for a day; the cause was an unlearnable
   reward term. Check what you're asking for before re-tuning how hard to
   ask.
6. **Warm-start chains silently ratchet the action std into bang-bang.**
   Every `--init-from` reloads the previous run's `log_std`, and the entropy
   bonus pushes it up each generation; one long chain reached std 21–26 in a
   ±4 action space. At that point the *clipped noise distribution* carries
   the behavior (stochastic episodes survived 6.9 s) while the exported
   deterministic mean is saturated garbage (fell in 0.5 s) — and telemetry
   cannot tell the difference. `LOG_STD_MAX` in `train_behavior.py` caps it
   on load and every rollout; a mean-poisoned lineage cannot be consolidated
   and must be restarted from scratch. Probe `live.onnx` deterministically
   at every checkpoint — that is the policy that ships.

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
vec env, semaphore IPC, numba BAM kernels, MPS updates) including the ones
that were **rejected for hurting learning**. Follow that precedent: measure
with `bench-envs` (real PPO, not raw stepping), and A/B learning quality
before shipping any throughput win as a default.

## Sim2real honesty

This harness is for prototyping with minutes-long feedback loops. Even under
`actuator="bam"` it is a subset of the official domain-randomization stack.
Once a behavior works here, port the env design to an mjlab cfg in upstream
`microduck_rl` and retrain on GPU — that stack, not this one, is the recipe
for policies that survive real hardware.
