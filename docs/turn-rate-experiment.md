# Brief: does a faster body yaw actually buy anything? (roadmap 3.7)

**For an agent running locally, with cores to spare.** Self-contained: everything
you need to know is here or named by path. Read `microduck_local/AGENTS.md`
("How much can the benchmark actually resolve?") before you measure anything.

---

## 1. Why this experiment exists

Every brain-tier idea tried in the soccer track has failed to survive a
confirmation on fresh seeds — poacher, ball memory, seven shelved knobs, the
head sweep, the kick cone, the ToF floor ball, the two-stage line-up,
`lineup_lat`, the bump-stand rule, `bump_back`, and a learned striker. The
README's "Where the soccer track actually stands" section lists them with what
killed each.

The measured reason is not a brain problem:

| | 1v1 | 3v3 |
|---|---|---|
| run spent turning **in place** | **47.1%** | ~58% |
| run spent steering while walking | 26.6% | ~31% |
| yaw swept in place | 371 rad / 1200 duck-s | — |
| rate while doing it | **0.655 rad/s** | 0.665 |

(1v1 measured over 1200 duck-seconds, seeds 60–61, commanded twist and achieved
yaw tallied every tick.)

A duck spends about half its life rotating on the spot at 0.655 rad/s. Holding
the yaw *demand* fixed, a walker that turned at 1.0 rad/s frees ~16% of every
run and one at 1.5 rad/s frees ~27% — larger than any brain-level change ever
measured in this repo.

**And there is nothing left to ask for.** `contract.py:65` sets
`ANG_VEL_Z_RANGE = (-1.0, 1.0)`, and at full command the shipped walker already
delivers 0.61–0.78 rad/s warm. The command ceiling and the policy ceiling are
the same wall. So this is a *training* question, and it is the only lever the
measurements point at.

### What this experiment is NOT

Do not sell a faster turn on **falls**. Turning beside a *static* body fell 0
times in 98 trials down to 8 cm of separation; every fall in those probes needed
a duck that was *moving* into you. Falls also need ~376 seeds to resolve a 25%
shift. Falls are not the endpoint here.

Do not try to fix this with **gaze**. Already measured: handing the brain a 210°
gaze cone (`search_sweep`) made the body turn *more*, on 5 of 5 paired seeds
(+8.6% in-place yaw, +9.2% total, n ≈ 150,000 ticks an arm). Finding the ball
was never the bottleneck; pointing the body at it is.

---

## 2. The experiment

Train **two** walkers from the identical recipe, differing only in the turn
command range and its curriculum weight, then measure both.

```
A  baseline    ANG_VEL_Z_RANGE = (-1.0, 1.0)     — the shipped range
B  fast-turn   ANG_VEL_Z_RANGE = (-2.0, 2.0)     — and see §4 for the ladder
```

**The A/B must be local-vs-local.** A locally trained walker is not
`alpha_walking` and will differ from it in speed, stability and gait quality —
`microduck_local/README.md` is explicit that this harness is for minutes-long
prototyping, not for the policy you put on hardware. Comparing B against the
*shipped* walker measures "local training vs Pollen's GPU run", which is not the
question. Comparing B against A, trained the same way on the same budget with
the same seeds, measures the turn range. **Train A even though a baseline feels
redundant — it is the whole experiment.**

---

## 3. Where the code is

| what | where |
|---|---|
| the command range | `microduck_local/src/microduck_local/contract.py:65` |
| command sampling | `walk_env.py:401` `_sample_commands()`, lines 410 and 419 draw `wz` |
| curriculum knobs | `walk_env.py:183-185` `zero_command_prob` 0.02, `turn_in_place_prob` 0.15, `forward_command_prob` 0.2 |
| the turn-tracking reward | `walk_env.py:747` `track_ang = W_TRACK_ANG * exp(-ang_err2 / ANG_TRACK_STD2)`; `W_TRACK_ANG = 2.0` (line 141), `ANG_TRACK_STD2 = 0.5` (line 149) |
| the command in the observation | `walk_env.py:685` `obs[48:51] = self.twist_cmd` |
| trainer | `uv run train-walk` → `microduck_local/train.py` |
| export | `uv run export-walk runs/<name>` (bakes the obs normalizer in — never ship a raw checkpoint) |

**The 61-obs contract is safe.** `obs[48:51]` carries the twist command in raw
rad/s, not normalised by the range, so widening `ANG_VEL_Z_RANGE` changes which
values the policy has *seen*, not the layout or the units. A brain that sends
`wz = 1.0` to a walker trained on ±2.0 still gets 1.0 rad/s asked of it. Do not
change the obs layout; `AGENTS.md`'s first convention forbids it.

---

## 4. Sequencing — do the cheap check before the expensive one

**Step 0 (minutes, do this first).** Widening the range is only worth training
if the walker's *ceiling* is a policy limit rather than a physical one. Command
the shipped walker beyond its trained range and see what it does:

```bash
cd microduck_local
uv run walker-facts        # command_deadbands() sweeps vx and wz on an empty floor
```

then extend that sweep past ±1.0 (the function is in `walker_facts.py`; it takes
a list of commands). If wz = 1.5 or 2.0 already produces more than 0.78 rad/s on
the shipped policy, the ceiling is not the training range and this whole
experiment changes shape — say so and stop. If it saturates at ~0.65–0.78, the
range is the wall and the training run is justified.

**Step 1.** Train A and B. Same `--envs`, `--steps`, `--seed`, same machine.
Use `uv run bench-envs` to pick `--envs` for that box; `uv run machine-facts`
prints its thread profile. Budget for at least 3M steps each — the README's
own examples use `--steps 3_000_000`.

**Step 2. Look at both before believing any number.**
```bash
uv run export-walk runs/<name>
uv run render-rollout --policy runs/<name>/policy.onnx --behavior stand --out /tmp/rr
```
and READ the contact sheet (`.claude/skills/render-rollout/SKILL.md`). This
repo's reward curves have lied repeatedly — most recently the striker's
`ep_rew` climbed the whole way while the exported brain carried the ball
backwards. A walker that turns fast by falling over, hopping, or pirouetting on
one foot will look excellent in `track_ang` and be useless. Check gait quality,
trunk height and forward speed, not just yaw rate.

**Step 3. Measure the turn rate you actually got**, with `walker-facts` pointed
at each policy — cold and warm, both directions, the whole sweep. If B's warm
rate is not meaningfully above A's, stop here and report that: the rest of the
pipeline is measuring nothing.

**Step 4. Only then, the game.**

---

## 5. What to measure, and what you can actually resolve

This is the part that decides whether the result is worth anything.

**Primary endpoint: TIME, not goals.** The in-place-turning fraction is a
per-tick tally with n ≈ 100,000+ per arm and it resolves easily. Goals need ~146
seeds for a 25% shift, falls ~376, kicks ~62. A 12-seed battery **cannot** call
goals or falls, and reporting them as a verdict is the mistake this repo has
made most often.

So report, per arm:

1. **fraction of the run spent turning in place** (`vx <= TURN_KICK and wz != 0`) — the headline
2. **yaw swept in place, and the rate achieved** — confirms the walker is delivering
3. **possession** (resolves a 25% shift in ~9 seeds) — the cheapest outcome metric
4. goals / kicks / falls — *reported, not judged*, with their event counts

Do not quote **advance per kick**. It is `ballAdvance / kicks` and `ballAdvance`
is flat across arms differing 2.6× in kicks, so the ratio reports its
denominator upside down. `AGENTS.md` rule 6.

**Confirm on fresh seeds.** Whatever you find on your first block, re-run on a
block of seeds you have never used (`--seed0`). Every single effect this repo
has lost, it lost at exactly that step.

**Use the resumable batteries.** `--out FILE --tag TAG` appends per seed and
skips what is already there; a reclaimed container then costs one seed, not
ninety minutes. Two batteries were lost before this existed.

---

## 5b. The trap that would make this measure nothing

**Both sides of the loop cap the turn at ±1.0, and if you widen only one
the experiment silently measures nothing.**

*Training side.* `ANG_VEL_Z_RANGE` lives in `contract.py:65`, which is the
shared 61-obs contract. **Do not edit it.** Grep says it is read in exactly
two places that matter — `walk_env.py:410` and `:419`, both command
sampling — so add a `WalkEnv` parameter for the training range (in the shape
of `turn_in_place_prob` at `walk_env.py:184`) defaulting to
`C.ANG_VEL_Z_RANGE`, plus a `train-walk --wz-max` flag. Nothing on the
deploy side reads the constant, so a wider training range needs no contract
change at all.

*Brain side — the easy one to miss.* Ten call sites hardcode ±1.0 as the
turn magnitude or the clip:

```
brain/gait.py:69                 wz = 1.0 if sign > 0 else -1.0     ← turn(), used by every scripted brain
brain/controllers.py:426, 1118, 1277, 1394, 1400   np.clip(..., -1.0, 1.0)
brain/controllers.py:457, 1448, 1456, 1458         wz = 1.0
ChaseParams.search_wz = 1.0 · TidyParams.scan_wz = 1.0
```

Train a walker that does 1.5 rad/s at `wz = 1.5`, leave these alone, and
every brain still asks for 1.0 — you would benchmark two walkers that are
never commanded differently and conclude a faster turn does not help. Put
the cap behind one constant (`gait.MAX_WZ`, defaulting to 1.0) that the
clips and the `turn()` helper read, and set it per arm alongside the
walker. **Verify before the battery**: log the commanded `|wz|` in a short
run and confirm arm B actually issues more than 1.0.

## 6. Code you will have to add

`eval-pitch`, `eval-tidy`, `walker-facts` and `trace-tidy` all hardcode
`POLICIES_DIR / "alpha_walking.onnx"` (see `eval_pitch.py:82`,
`eval_tidy.py:40`, `walker_facts.py:47` and `:106`, `trace_tidy.py:55`). Add a
`--walker PATH` argument that defaults to the current behaviour, so a battery
can be pointed at `runs/<name>/policy.onnx`. `brain_env.py:216` and
`striker.py:352` already take a `walker` override — follow that shape.

The in-place-turning tally is not in the repo. It is a small addition to
`eval_pitch.run_one`'s loop: count ticks where the commanded twist has
`|wz| > 0` and `vx <= TURN_KICK`, and integrate `|Δyaw|` from odometry over
those ticks. Add it as a metric rather than a one-off script, so it lands in the
`--out` rows alongside possession.

---

## 7. What a good report looks like

State, in this order: the step-0 result (is the range really the wall); the two
walkers' measured turn rates cold and warm; what the rollouts looked like; the
in-place-turning fraction for each arm with its tick count; possession with its
seed count; and goals/falls with their event counts and an explicit note that
they resolve nothing at this size. Then the recommendation: is a turn-rate
retrain worth GPU time on the official `microduck_rl` stack.

If B trains badly, or turns fast by walking badly, **that is the result** —
report it. Do not tune until it wins; that is how a false positive is
manufactured, and this repo has a section about the one it caught.
