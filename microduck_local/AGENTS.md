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
  It must never accumulate across resets. `tests/test_env_contract.py` and
  `tests/test_walk_env_physics.py` lock this — for every field in
  `walk_env.DR_MODEL_FIELDS` (mass, inertia, CoM, armature, friction) and
  the `mj_setConst` outputs that depend on them. A new randomized quantity
  joins those tuples, gets a knob with the upstream range as its default,
  and gets a test that shows it LANDS in the model.
- **The IMU blocks of the obs are refreshed after the substep loop**
  (`_refresh_derived`). Without it gyro / gravity / trunk height describe
  the state one substep before the joint blocks. Don't remove it, and don't
  "upgrade" it to `mj_step1`: that rebuilds the constraint rows the BAM
  friction scan reads without solving them.
- **Velocity pushes are part of the walk env's DR and OFF in every
  behavior recipe** (`BehaviorEnv` defaults `push_robot=False`). Turning
  them on for a trick is an experiment to name and measure, not a fix.
- **ONNX ships with the obs normalizer baked in** (`export-walk`). A raw
  checkpoint is not a deliverable.
- **`joint_vel` lags one control step** (Dynamixel moving-average), matching
  training and hardware. Don't "fix" it.
- Tests in `tests/` are contract locks, including a regression test that the
  shipped `alpha_walking.onnx` survives upright in this env. Run
  `uv run --with pytest pytest tests/` before and after your change.
- **A second robot is a `RobotSpec`, never a fork of `walk_env`**
  (`robots/spec.py`). The env resolves its body by NAME from the spec —
  base body, gyro sensor, floor, foot pads, stand keyframe, joints, default
  pose, action scale. Two rules hold when you add one:
  1. the duck path stays bit-identical (hash a 200-step rollout under `xml`
     and `bam` against the previous code — `tests/test_robot_spec.py` pins
     the ids and names the spec resolves to);
  2. a name the model does not have RAISES at construction. It used to come
     back as id -1, and a mistyped foot pad then paid no air-time reward
     with nothing anywhere saying so.
  The duck's 61/14 contract is not negotiable for the DUCK; another body has
  its own layout, and the one thing that must never happen is a policy
  stepped in a body whose observation width differs from its input.
- **The G1 is a lab contract, not a sim2real one.** Its 99-d layout carries
  base LINEAR velocity, which no real humanoid observes without state
  estimation. Prototype and demo with it; never present a G1 policy trained
  here as deployable. `bam` is refused for it (an XL330 identification does
  not describe a 34 kg humanoid's actuators).

## Reward design rules

- **Never reward what the policy cannot observe.** The obs contract has no
  yaw and no world position. A heading- or world-anchored reward term is
  unlearnable noise — one such term produced 30M steps of circling that
  looked like an optimizer problem. Straightness/heading belongs to the
  velocity commander, not the reward. Before adding a term, point at the obs
  indices that let the policy see what you're scoring.
- **A task the robot must SENSE puts its sensing in the command slots, in
  the robot's own terms.** `find_ball` has no ball in the physics: the env
  projects a point through the robot's `head_camera` and writes what a
  detector would report (bearing across/up the frame, seen, a gyro-
  dead-reckoned memory of which side it went) into the four head slots,
  with the detector's cadence and jitter. The reward may use the true
  bearing (privileged, like every reward here); the obs may not. The
  memory slot exists because the policy is memoryless: without it the
  ball rolling out of frame leaves nothing observable to say which way to
  look, and the daemon can produce it with one gyro integral.
- **A symmetric observation leaves the mean nothing to learn.** With the
  ball equally likely on either side and no cue in the obs, "turn left"
  and "turn right" carry the same advantage, the mean head-yaw action sits
  at zero, and the exploration noise does the finding: the first find_ball
  export stood and stared at a ball 42° off while the stochastic trainer
  saw it half the time. Diagnose it by probing the exported ONNX directly
  (feed one obs, vary one slot, read the action) — the network had learned
  every slot's sign correctly; the failure was in what the obs could not
  say. Break the tie in the obs (a belief slot with a fixed convention
  when nothing is known), never by hoping PPO finds a side.
- **A memoryless policy cannot sweep.** A search is a limit cycle in head
  yaw and 2M PPO steps produced a static gaze-vs-belief instead. Give it a
  clock (sin/cos of a phase in two command slots, the imitation recipe's
  trick) and the sweep becomes a static mapping. Same rule as the phase
  signal: anything the policy must do *over time* needs time in the obs.
- **Mind MuJoCo's velocity frames.** `mj_objectVelocity(..., flg_local=0)`
  is the world frame; naive indexing once rewarded a sideways shuffle as
  "forward". When you write a term that reads a velocity, print it in a pose
  you can reason about first.
- **No jackpots.** Dense, bounded shaping beats sparse windfalls; a term a
  policy can spike once and farm will get farmed. (Inherited from upstream
  `microduck_rl/AGENTS.md`, which is worth reading in full.)
- **Subclassing a task: RETARGET a term, never switch it off — and check it
  is not FLAT where the policy starts.** This is the most repeated mistake in
  this repo: five retrains in one day (2026-09-14, the G1 squat and karate
  strikes). When a new task inherits a reward, the reasoning "this term now
  pays for the wrong thing, so weight 0" feels obviously right and is almost
  always wrong. An objection to a term's TARGET is not an objection to the
  term; zeroing it deletes the only description of that aspect of the task and
  leaves a proxy that FAILURE also satisfies. Measured:
  - `W_POSE = 0` on the squat ("a squat is precisely not the default pose"):
    nothing then priced the configuration, only "be 20 cm lower" — and the
    cheapest way to drop a pelvis 20 cm is to pitch forward over the ankles.
    2 of 3 seeds ended face-down while the height term paid 5.42 of 6.0.
  - `W_HEIGHT = 0` on the punch ("a strike is not defined by pelvis height"):
    nothing paid to stay up, so it sagged from 0.735 m to 0.48 and tripped its
    own fall floor on 5/5 seeds.
  The second form is a term left ON at a width that is flat where the policy
  actually is, which is the same bug wearing a number: `STRIKE_STD2 = 0.35`
  paid a STANDING robot 0.00 of 6, so the punch abandoned the pose entirely
  (27° mean joint error); the idle's `HEIGHT_STD2 = 0.004` is e^-14 at the
  0.24 m sag that really happened, so there was no gradient back up.
  Two questions before you commit a weight or a width. **What else satisfies
  the terms that remain?** If collapsing, folding or sagging does, the term
  you are zeroing is load-bearing and needs a new target. **What does this
  term pay at the policy's STARTING state, and at the failure state I am
  trying to price out?** Print both numbers; a term that pays ~0 at both is
  decoration. A retarget is usually a one-line hook (`pose_target_rel()`,
  `target_height()`) whose base returns the old behaviour, so the original
  task is provably unchanged.
- **A pose that meets every number can still be wrong to look at, and that
  needs its own term.** The first squat that held its height, uprightness and
  foot contacts for 60 s on 5/5 seeds did it with its back twisted 30° and one
  arm folded onto its thigh, because `pose` spread ONE Gaussian over all 29
  joints at std2 4.0 — ~1.05 rad² of upper-body error still paid 0.77 of it.
  A separate tight `carriage` term (waist + arms, std2 0.25) fixed it in one
  retrain and made the HEIGHT more accurate too. Score the joints that decide
  how it *looks* separately from the ones it *balances* with, and never let
  the two sets overlap: on the punch, `carriage` covered exactly the punching
  joints and paid 0.00 at the target, i.e. it was charging the robot for
  striking.
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

- **When no reward can find the motion, DRAW it.** Six reward formulations
  and four start points all converged the G1's kick to ~0.1 m (roadmap
  13.3); the same kick posed in the 🎬 editor with the IK solver was 0.62 m
  high and balanced in every frame in twenty milliseconds of solves, and
  `imitate` / `g1_imitate` turns it into a tracking problem. The rule from
  the duck's backflip holds for every body: choreography is authored, RL
  solves the physics. The editor works for any `RobotSpec` (`pose.py`:
  `/joints?robot=`, `/pose`, `/ik`, `render-clip`), so "the G1 has no
  animation tools" is no longer a reason to grind on weights.
- **The IK solver has to respect the root trap too.** The pelvis is the free
  joint. Two feet pinned with the root held is a CLOSED chain — a "weight
  over the left foot" request stalled 6 mm short, served by the arms — so
  `solve_ik` frees the root's translation whenever the goals span more than
  one limb, and the editor draws in a frame anchored on the grounded soles
  (the feet stay, the hips move). And a soft constraint at equal weight with
  a hard one is a trade: a pinned foot's `level` row against its ankle-roll
  stop SLID the foot 5 cm to buy back tilt. Position is the promise; level
  is a 0.2-weight preference, and a joint on its stop has its column
  dropped.

## Launch training through the LAB, never straight from the CLI

**A run started with a bare `uv run train-*` is invisible.** The human who
asked for it cannot watch it, cannot see the reward breakdown move, cannot
stop it, and finds out how it went only when you report — which means they are
trusting your summary of a thing they were never shown. Start it through the
lab instead:

```bash
curl -s -X POST http://127.0.0.1:8788/teach \
     -H 'Content-Type: application/json' -d '{"text": "do a squat"}'
```

The lab then runs the recipe's whole curriculum, streams `progress.jsonl` into
the teach card, loads each `live.onnx` snapshot onto the 🎓 trainee on stage,
and the viewer at `http://localhost:63317` shows the robot practising, live,
while the numbers move. That is the point: **the human watching the stage
catches things the reward curves do not.** Every diagnosis in Track 13 started
that way — an idle "getting worse and going into weird positions", a squat
that met every number with its back twisted, a kick that held one leg when it
had been asked to recover.

If the backend is holding stale code, restart it (it does not hot-reload the
env modules) and then POST:

```bash
bash .claude/skills/restart-servers/restart.sh
```

### "Has the trainer finished?" — ask the lab, never the process table

```bash
curl -s localhost:8788/teach/status      # {"running": bool, "status": ..., "job": {...}}
```

`GET /teach/status` is the authoritative answer, because the lab owns the
subprocess (`TrainingJob._poll` reads `proc.poll()`). A finished `train-*` run
also closes `progress.jsonl` with a terminal `{"done": true}` line, which
answers the same question from the artifact alone with no server running.

**Do not `pgrep` for it.** Measured on this machine, both obvious patterns are
wrong in the same silent direction — they report a trainer forever, so a wait
loop never ends:

| pattern | what it ALSO matches |
|---|---|
| `pgrep -f "microduck_local.train "` | the shell running the pgrep — its own command line contains the pattern |
| `pgrep -f "Python.*microduck_local"` | **duck-lab itself**, whose venv path is `…/microduck_local/.venv/bin/duck-lab` |

This is the general rule from "Verify filters against the complement" applied
to processes: check what a filter MATCHES THAT IT SHOULD NOT, not just that it
finds the thing you had in mind. A wait loop that cannot terminate looks
exactly like a slow training run, which is why this cost a session's worth of
confusion before anyone noticed.

```bash
# correct: poll the owner
until [ "$(curl -s localhost:8788/teach/status \
           | python3 -c 'import json,sys;print(json.load(sys.stdin)["running"])')" = "False" ]
do sleep 30; done
```

The CLI trainers are for BATTERIES and A/Bs — paired-seed comparisons, sweeps,
anything whose output is a number rather than a behaviour. Use
`MICRODUCK_RUNS_DIR` for those so scratch runs stay out of the palette. If a
task has no recipe yet, add one (`behaviors/`, and `Behavior.trainer` for a
non-duck body) rather than reaching for the CLI: the recipe is what makes it
watchable, teachable and re-runnable by the person who asked.

This has been got wrong repeatedly — four CLI runs in one sitting before the
user asked, a second time, why they could not see any of it.

## Every run is a record: name it, describe it, file it

A board of 49 runs called `p-batch-s14`, `p-de-s11`, `z1` and `mix` could
not be read a day after it was made. A name that encodes the knob still
says nothing about the question or the answer, and `z1`/`z2` had no note
anywhere — identical recipes on the same seeds was all their `brain.json`
could say. So three fields ride in `brain.json` beside the recipe:

* **`title`** — the human name the `/train` page shows in place of the run
  name: `"Capacity sweep: 256-256"`, `"Follower v4 — the pick"`. The run
  NAME is an identifier (`--init-from`, `learned:<name>`, `select-brain`,
  the directory) and never changes; the title is what people read.
* **`description`** — one or two sentences: what it tests, against what,
  and — once known — **what it found, with the interval**: `"+0.000 paired,
  95% −0.012…+0.012 — neutral, shipped on"`. Write the finding back in when
  the experiment resolves; this is where the next person learns whether a
  run was worth making.
* **`group`** — the use case it files under, one of `describe-brain --help`'s
  keys (`shipped-followers`, `trainer-ab`, `paired-sweeps`, `capacity`, …).
  The `/train` cards and the `/sim` brain menu list by it. A group is the
  QUESTION a set of runs answers, not the knob they turned.

Set them at launch — `train-brain --title ... --description ... --group ...`
warns when the title is missing — or after, with
`describe-brain <run> --title ... --description ... --group ...`. A family's
seeds share a title (`p-n256-s31..36` are one experiment); the seed is
already in the name. Never launch a sweep with bare codes: ten seconds per
launch against an hour of forensic annotation after the fact.

### A WALK or TASK run says it in `record.json` — and the chain names its pick

`brain.json` is the navigation brains'. Everything trained by `train-walk` /
`train.py` (every duck walker, every G1 task) carries `runs/<name>/record.json`
instead — same idea, smaller file, and the lab's POLICY PALETTE is what reads
it: `title` on the chip, `note` in its tooltip, `pick` on the one stage worth
using. `train.py` writes the facts automatically (task, body, clip, warm-start
parent, curriculum rung); `describe-run` writes the rest.

**`pick` is the field that matters, and only a measurement may set it.** A
curriculum chain's `▶` button and its `⤓` download used to hand over the FINAL
stage, on the assumption that the last stage carries the whole trick. That is
false whenever a ladder steps one rung too far: in the G1 kick chain the final
stage over-shot the clip by half and survived 3 seeds in 8, while the stage
before it held 8 in 8 — and the palette pointed at the wrong one, with nothing
on screen to tell them apart. The pick is what a measured stage looks like from
the outside; setting it clears the flag on its siblings, because two picks is
the same "which do I use?" question it exists to end.

```bash
uv run python scripts/eval_imitate.py runs/<run>/policy.onnx <clip> --seeds 8 --record --pick
uv run describe-run <run> --title "Front kick (G1)" --note "8/8 hold 20 s, apex 0.62 m" --pick
uv run describe-run <old-run> --backfill      # title/description from its own run.json
```

**Where a number is quoted decides whether it is read.** The kick chain's
measurements existed in a chat log and in `docs/roadmap.md` the whole time; the
person looking at the palette saw five hashes. A finding that never reaches the
artifact has not been reported — write it into the record as well as the doc.

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
3. **A null needs its MDE, or it is not a null — it is "no result".** This
   is the rule that cost the most: a whole run of soccer knobs was written up
   as "measured off" on p-values alone, from batteries that could never have
   seen the effect being denied. At the 24 seeds these run, the minimum
   detectable effect is **28% of baseline on kicks, 33% on falls, 19% on ball
   advance, 48% on goals** — so a real 10% improvement is invisible BY
   CONSTRUCTION and comes out as a null. The MDE was on screen the whole
   time: it is the 95% half-width, which the table always printed and nobody
   read as one (a difference is significant exactly when it exceeds it).
   `compare_pitch.py` now prints it as a percentage of baseline and calls a
   non-significant row `null` only when the MDE is tight enough to mean it;
   otherwise it prints `NO RESULT` and the seeds that would settle it. Never
   write "measured off" against a `NO RESULT`.

   **Turning a null into a positive claim, and the trap in it.** A rule that
   only changes outcomes when it fires moves a whole-match metric by about
   (firing rate) x (per-firing effect), so `required per-firing effect =
   MDE / firing rate` — which reads as "this arm does not produce more than
   X% on the ticks it fires". **That holds for the ARM measured, not for the
   knob.** Measured on `board_margin` at 15 cm: feasible fractions 95.2% and
   52.8%, total effects +2.50 and +3.04 points, so per-acting went +2.63 →
   +5.76 — rising 2.19x as the fraction FELL, because a bigger clearance puts
   the kick spot further off the wall and a further spot is a better one. Where
   a knob's value changes both how often it acts and how well, both terms move
   and a null at one value bounds nothing at another. A registered prediction
   of 1.80:1 came back 0.82:1 on exactly this.

   **`compare_pitch` prints NINE metrics — pick the primary before you look.**
   If nothing is real, the chance at least one row comes back p < 0.05 is
   **37%**; three arms is 27 reads. So name the primary metric, its threshold
   and its direction in advance, Bonferroni over the contrasts, and quote
   everything else as exploratory with the read count attached. Pick it from
   what can actually resolve (possession 10%, spread 9%, depth 5% at 24 seeds —
   not kicks at 28%), and check the metric is structurally capable of moving:
   in `eval-pitch` BOTH teams get the knob, so a share-shaped metric cancels by
   construction and its null was never going to be anything else.

   **And the correction is a FORECAST, not a tax — this is the measured part.**
   A camera study read nine metrics across three arms with nothing named in
   advance. Applying the correction afterwards sorted its rows into two it said
   to trust and two not to lean on. A confirmatory block on fresh seeds, with
   the primary registered before launch, reproduced *exactly that split*:

       ballAdvance   -0.377 -> -0.356      the correction said trust it
       kicks         -41%   -> -42%        the correction said trust it
       crowd     p 0.039 -> p 0.341        it said do not lean on this
       falls     8->1     -> 4->3          it said do not lean on this

   The two flagged rows evaporated and the two strong ones came back within a
   few percent on independent seeds. The correction did not merely counsel
   caution — **it predicted which findings would survive replication**, which is
   the only evidence that it is doing work rather than making you timid.

   **A falsifier that scores a refutation and a discovery identically is badly
   built.** A registered rule read "~99% means the mechanism holds, ~60% means
   it does not". But ~60% was two opposite worlds: the surviving cases sitting
   *beyond* the gate would mean it was failing to act (a refutation), and the
   same 60% with them sitting *inside* it would mean the duck had waited until
   its plan was no longer stale — the mechanism wrong and the knob better than
   anyone had described. The original rule would have filed the second as
   "unsupported" and stopped. **Before registering a threshold, ask what else
   could produce the number you are about to treat as a refutation** — and if
   an interesting outcome and a boring one land on the same side of it, the
   rule needs a second dimension, not a tighter cut. (Here: the fraction of
   surviving cases inside the gate, which separated them at 94.4% against 0.7%.)

   **The constructive form: register a quantity your hypotheses DISAGREE on,
   not one your hypothesis predicts.** The rule above failed because the
   `None` fraction was something the favoured mechanism predicted — and so did
   its rival, which is why one number could not tell them apart. And the
   deeper caution: that registered prediction came back at 97.9% against ~99%
   and the mechanism behind it was still wrong (the gate *replaces* swings
   rather than declining them; both models give the same number).
   **Holding to a percentage point is not evidence that the model behind the
   number was correct.** What exposed the real mechanism was a *distribution*
   asked for as an extra — the true `|ahead|` of the surviving swings, which
   sat on the close mode at 0.111 against 0.108 with the far mode gone. So:
   register the number, and ask for a shape that would look different under a
   rival.

   **And a registration that names only the outcome its author expects cannot
   report a world where more than one mechanism is live.** Two registrations
   were made on the same number — "the crop is blinder everywhere" (expect a
   high blind fraction) and "the crop's surviving swings are the ones that kept
   sight" (expect a low one). They are not rivals: **both can be true at once**,
   and any middling value would have let each author claim the number landed
   nearer theirs. The fix is to register a **pair** rather than a verdict —
   here, degradation `(0.636 − (1−f)·0.92)/f` and selection `(0.979 − f)`,
   reported together and neither picked. Same family as the falsifier point
   above, one level along: there the rule could not tell a refutation from a
   discovery; here it could not describe a composite. Both registrations looked
   complete on their own, which is what makes this the subtler of the two.

   **MDE is per-METRIC, not per-battery — so never say a battery "found
   nothing".** One 24-seed `eval-pitch` run settled `board_margin` on
   possession at an MDE of **5% of baseline** (a real null, tight enough to
   mean it) while the *same seeds, same runs* gave an MDE of **18% on kicks** —
   wide enough that a boards-confined effect of exactly the size the gym
   measured would have been invisible in it. Same battery, decisive about one
   question and blind to another. Quote the metric's MDE, never the battery's.

   What does extrapolate is the **precondition** population — the ceiling no
   value can raise. `contest_margin` fires on 0.07% of ticks at 0.15, but its
   precondition (an opponent inside `duck_touch` with the ball visible) is
   0.41% at *any* margin, and 19% ÷ 0.0041 needs a 4634% per-firing effect. So
   ask: **does the knob's value set its own population?** `board_margin` does
   (0.25 acts on 39% of a boards draw, 0.10 on 95%) and was rescuable by
   re-valuing — that is what turned it from a null into 1.46-1.71x.
   `contest_margin` cannot, at any value, and stays dead.

4. **`ballProgress` is not quotable.** Its MDE has never once been under
   100% of baseline on a real battery; the median seed budget for a 10%
   change is ~21,000. It is printed for shape only. Differences in it have
   been quoted in this repo and none of them meant anything.

5. **Match the instrument to the event you are studying.** Measured cost per
   kick event: the 3v3 pitch is **~15 CPU-seconds**, `scripts/kick_gym.py`
   is **~0.8** — about **19x** cheaper, because a gym episode is one
   placement and one swing while a pitch run is six bodies and ~85% dead
   ball. Anything about the kick itself belongs in the gym; spend the pitch
   only on what genuinely needs a team. And before spending either, measure
   how often your knob's condition even fires: a rule that triggers on 1.3%
   of the run cannot move a whole-match average, whatever it does when it
   fires (that measurement is what closed item 12h's ball-memory arm).

6. **Before believing a null, check the thing you measured COULD have moved.**
   Three failures in this repo share one shape — a measurement that was never
   able to produce the answer it then reported, in the language of a result:
   - **A gated knob.** `contest_margin=0.15` ran a 16-seed contested gym and
     reproduced the baseline episode for episode; the rule is gated on
     `use_color`, which ships `False`. `brain/knob_gates.py` warns before the
     battery, `kick_gym.is_identical` catches it after.
   - **A shadowed knob.** `gaze_bearing_max` was a live knob with a hard-coded
     `0.6` in the gate it names. No static analysis sees that; only
     `is_identical` does.
   - **The wrong subject.** A post-clamp measured as a no-op — against the
     STRIKER, whose post was already clear. The DEFENDER's post lands four
     centimetres inside the board. The knob was right and the subject was not.
   - **A probe that forgets its own gate** reports 0% and reads as a profound
     finding. `scripts/probe_contest.py` refuses to present zero firings as a
     result for exactly this reason, and two tests hold its gate above its
     brain imports.

   The cheap general defence is a **positive control**: show the measurement
   producing a NON-zero on something you already believe, before you quote a
   zero. Rule 0's "a knob that changes nothing is broken, not null" is this
   rule for knobs; this is it for probes, subjects and populations too.

3. **Confirm on seeds the effect was NOT found on.** A "confirmation" that
   re-uses the discovery seeds is not one. A poacher supporter scored 10
   goals against 3 over four seeds and 21 against 12 over twelve - the
   twelve CONTAINED the four - then reversed on twelve fresh ones, 13
   against 19, for 34 against 31 over all 24 (p = 0.80). `--seed0` exists
   so a battery extends onto fresh seeds instead of re-running the old ones.
4. **A paired reading is free, but here it buys nothing — do not budget for
   it.** Both arms run the same seed layouts, so report per-seed wins/losses
   alongside the totals; pairing can never be worse. But it is not the power
   this section long implied. Measured across all thirteen A/B batteries on
   disk (`scripts/audit_power.py`), the median between-arm correlation is
   **r = 0.05** and the variance reduction is **1.03x**. The sim diverges
   within seconds of any knob that actually fires, so by 300 s the arms are
   effectively independent runs and the shared seed cancels nothing. The one
   battery where pairing paid (`t9 hunt`, r = 0.6) is the one whose knob
   barely fired — a high pairing gain measures how little your arm perturbed
   the run, not how good your design is. `compare_pitch.py` prints the
   observed gain per metric; when it reads ~1.0, the seeds were decorative.
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
7. **A downstream constant may have absorbed the bias you are about to
   fix. Fix them together or the fix measures as a regression.**
   `brain/tidy.py`'s `stale_fix` places a detection from the pose the duck
   HAD when the frame was taken instead of the pose it has now. It is
   unambiguously correct and it works: the toy's placement error drops from
   5.4 cm to 3.7 cm (medians, 4 seeds). It also lost **0.38 toys
   (p = 0.031)**, with grasp success falling 88% -> 76%.

   The diagnostic that explained it measured the TRUE duck-to-toy distance
   at `settle`, where the beak has to reach: 8.7 cm with the bias, 10.5 cm
   without it. The bias put the toy slightly AHEAD along the direction of
   travel, so the duck walked past its own estimate and arrived at 8.7 cm —
   which is `reach_ahead + reach_pad` = 8.8 cm, exactly where the grasp
   wants it. The stop pad had been fitted, by hand, against the biased
   estimate. Remove the bias and the duck stops honestly, 1.7 cm short, and
   fumbles.

   Correcting the pad by the 1.8 cm the bias was worth recovers it —
   **+0.44 toys (p = 0.026)** with grasp at 93% and attempts per pick
   1.15 -> 1.08. So the fix was right, the measurement was right, and the
   conclusion "the fix hurts" would have been wrong.

   The general shape: **any hand-fitted constant downstream of a biased
   estimate has absorbed some of that bias**, because it was tuned to make
   the system work WITH the bias present. Fixing the estimate alone moves
   the system off the operating point the constant encodes. So when you
   correct a systematic error, find the constants fitted against it and
   sweep them in the same battery — a 2x2, not two separate arms. And run
   the control cell (the corrected constant WITHOUT the fix): if that also
   improves, the constant was simply mistuned and the fix is not what
   earned it.

   Corollary for reading results: a correct change that measures as a
   regression is evidence about the SYSTEM, not only about the change. Ask
   what was tuned around the thing you just fixed before you file it as a
   null.
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

9. **Count the events, not the runs — the SAME measurement can be cheap or
   hopeless depending on which you pick.** Measured on the soccer benchmark
   (24 seeds × 300 s of 2v2, `docs/roadmap.md` Track 4.1.5): `kicksBack` as
   a per-run mean has a CV of 0.78 and needs ~151 seeds to resolve a 25%
   shift. The identical quantity as a PROPORTION of the arm's kick events —
   90 of 183 — is a binomial with 183 events behind it, and the aim-rule
   change resolved at p = 0.011 on 24 seeds. Two orders of magnitude of
   cost, from nothing but how the number was pooled. Before concluding a
   metric is too noisy to judge with, ask whether it is really a rate over
   runs or a fraction over events; if events, test the proportion.

   The same table says which instrument to reach for at all: `depth` (CV
   0.07, one seed), `possession` (0.18, 8) and `spread` (0.21, 11) resolve
   a POSITIONAL change for the price of a coffee, while goals need 136 and
   `ownGoals` — 19 events over 24 seeds — needs 347 and can never be the
   judge of anything here. Report it; decide on something else.
10. **A knob a battery sets must be shown to REACH the thing being
    measured, on the roster being measured.** `brain_kwargs` handed any
    roster with two ducks a side the bare `ChaseParams()` defaults, so every
    knob set through `MICRODUCK_CHASE` was silently discarded in 2v2 and
    3v3: both arms of such an A/B would have run the same brain and the seed
    noise between them would have been reported as the effect. The comment
    beside the line claimed the opposite. This is rule 0 of the next section
    wearing team colours, and the check is the same one: reach into the
    constructed object and read the value back off the thing that is
    running — `assert brain.p.aim_mode == "clamp"` — not off a fresh
    `ChaseParams()`, which will happily agree with you while the live brain
    does something else.

## Editing a file a running battery imports (write ATOMICALLY)

Another session usually shares this checkout, and batteries run for tens of
minutes. On 2026-09-09 an edit to `brain/controllers.py` killed both arms of
somebody else's `kick_gym` run — 40 minutes of compute — with
`TypeError: must be called with a dataclass type or instance` out of
`ChaseParams.from_env`.

**The mechanism was a torn read, not stale code.** `Path.write_text` truncates
the file and then writes it. A worker that opened it inside that window got a
prefix: `class ChaseParams` existed, `@dataclass` had not been applied yet. It
reproduces once and then passes a minute later, which is the signature.

**Two assumptions that made it feel safe, both false on this machine:**

- macOS defaults to **spawn**, not fork, so every worker re-imports the module
  from disk when it starts.
- `ProcessPoolExecutor` starts workers **lazily** as tasks are submitted, not
  all at pool creation. A battery running for minutes can start a fresh worker
  — and re-read your file — at any moment.

So there is no "the pool already imported it, therefore I am safe" window.

**The fix is structural, and this repo already does it one module over.**
`contract.py:161-163` writes the generated ball scene through a temp file and
`os.replace`, with the comment *"atomic: a worker never reads a half-written
scene"* — it got that in a code review the same morning, for this exact
failure with vec-env workers. So this is not a new commandment, it is a rule
the codebase applies in one place and needs everywhere. Temp file in the same
directory, then `os.replace`, which is atomic — a spawning worker opens either
the whole old file or the whole new one, never a prefix:

```python
fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
with os.fdopen(fd, "w") as fh:
    fh.write(text)
os.replace(tmp, path)
```

This beats coordination because it does not depend on anyone remembering.

**But atomicity fixes the TORN read, not the SPLIT one.** Workers that start
either side of an `os.replace` load two different *whole* versions, and the
battery is then measuring two programs and averaging them. Nothing in the
output says so. That is what the process check is for, and it is why it stays
after the writes are atomic — **do not delete it as redundant.** Before
touching `brain/` or `world/`, check what is running —

```bash
pgrep -fl python | grep -E '^[0-9]+ .*(kick_gym|eval[-_](pitch|striker|brain|tidy)|train[-_](walk|brain|behavior)|probe_|duck-lab)'
```

**The hyphens are load-bearing too, and the first version of this line was
wrong.** The console scripts are `eval-pitch`, `eval-striker`, `train-walk` —
hyphens — so a pattern written `eval_pitch` never matches `.venv/bin/eval-pitch`
and reports "nothing running" over a live battery. It was verified against two
live `kick_gym` arms (which DO have an underscore) and passed, so the check was
confirmed on the one case that could not fail it. Measured later with both kinds
running: the underscore-only form found 2 of 5 processes. **A check verified
only where it works is not verified** — the same defect as agreeing on a
quantity that could not have differed. Include the long-running writers too
(`train-*`, `duck-lab`): they import `brain/` and `world/` exactly as a battery
does, and the user starts teach runs mid-session.

(The rest is load-bearing as well, and two simpler forms do NOT work — verified
against live `kick_gym` arms. `pgrep -f 'kick_gym|...'` matches the shell
running the check, because `-f` matches whole command lines and the pattern is
in your own argv; so does anchoring on `python`. The `[k]ick_gym` bracket trick
does not save you either, because other shells in the session legitimately have
`kick_gym` in their command lines. Restricting `pgrep` to python processes and
then anchoring the grep on `^<pid> ` is what works: `-fl` prints multi-line
command lines, and only the first line of each carries the PID, so the anchor
drops the continuation lines that produce the phantom matches. The output is
the drivers AND their spawned workers.

**And do not count it with `grep -c`.** `pgrep -fl` prints a whole command line,
and the watcher shells on this box embed newlines, so one process can print as
many lines as it contains — `grep -c` counts lines, not processes. Measured with
both kinds live: `grep -c` said 3 where the true count was 13. To count, drop
`-l` and count PIDs:

```bash
pgrep -f 'kick_gym|eval-pitch|eval-striker|train-behavior|duck-lab' | wc -l
```

Use `-fl` to READ what is running and `-f | wc -l` to COUNT it. A zero from
either is still a reliable "nothing matched", so the counting bug inflates or
deflates a magnitude rather than manufacturing a false all-clear — that failure
came from the *pattern*, above.)

— and treat those modules as owned by whichever battery is live, not by a
session. **The unit of ownership is the import graph, not the file.**

### Before believing an EDIT, check the thing you changed is the thing you meant

The companion to "before believing a null, check the thing you measured could
have moved". Same defect, one on the measuring side and one on the writing
side. Four instances in one day, all of which parsed, imported and passed
`precommit.sh`:

- An anchored edit landed on **`Follow.step` instead of `Chase.step`** —
  identical signatures. That brain then raised `AttributeError` every tick.
- Inserting a class above `class ChaseParams:` put it **between
  `@dataclass(frozen=True)` and the class it decorates**. The new class became
  the dataclass; `ChaseParams` silently lost all 164 fields.
- A probe **re-entered** the method it measured and reported a 7.56% firing
  rate that a direct sweep showed was zero.
- A knob was measured against **the wrong subject** (the striker, whose post
  was already clear) and reported as a no-op.

The check that catches all four is embarrassingly cheap and needs no tooling:
**construct the object and call the method.** `is_dataclass(X)` and
`len(fields(X))` catch the second in one line; `Follow().step(...)` catches the
first; a direct knob sweep catches the third and fourth. A parse-and-import
gate proves the file is a program. It does not prove the program is the one
you wrote.

### Mark a number that a reconstruction produced, not a call

A reconstruction and a measurement look identical once written down — that is
the failure, not the reconstructing. Three instances in one evening, and in
each the reader could not have told without being told:

- A probe re-derived a clamped target by calling the method again, and reported
  a 7.56% firing rate that a direct sweep showed was **zero**.
- A truth table extended by rebuilding a spot calculation from the source got
  one cell wrong (the foot selection ignored `rel` and the hysteresis). Real
  calls to the same function were correct in the same message.
- A dilution figure computed with correct arithmetic on a **guessed** constant
  (a draw's lower bound taken as 0.055; the ball radius makes it 0.045) came
  out 40.8% against a measured 39.2%.

The third is the one to watch: real maths on an assumed input reads exactly
like a measurement, and is the easiest to produce by accident.

**The convention:** a bare number is one the real code path produced. Anything
recomputed, re-derived, or rebuilt from reading the source is marked inline —
`[reconstructed]`, `[assumed input]` — so a reader can weight it without having
to ask. It costs a word and it removes the whole class, because the reader no
longer has to trust that the writer noticed.

**A third shape, at the labelling end: a real number whose PROSE says
something other than what the expression computed.** A corner share was
reported as "within 0.30 m of two boards"; the expression was
`min(dx, dy) < 0.30 and max(dx, dy) - min(dx, dy) < 0.30` — "the nearer board
is under 0.30 away and the two distances are within 0.30 **of each other**",
which counts a ball 0.10 off one board and 0.35 off the other. 31.4% for the
predicate written, 19.2% for the one described. The measurement was real, the
code was real, and the sentence was about a different quantity. Neither rule
above catches it: nothing was reconstructed and no input was guessed. **So
state a predicate, do not paraphrase one** — quote the condition or write it
out whenever a number counts things satisfying something. A paraphrase is a
reconstruction of your own code, performed in prose, and it fails the same way.

**A fourth shape, and the nastiest: an unstated SCENARIO — a systematically
wrong input that agrees with the truth wherever the quantity does not depend
on it.** One session drew from `make_pitch(per_side=2)` out of habit while the
gym runs `gym_scenario()` (the 1v1 floor). Three feasibility fractions came out
right anyway, because feasibility is pitch-invariant — it depends on the ball's
gap from the NEAR board, not on where the others are. Only the corner share,
a ratio of areas, is pitch-dependent, and only it disagreed: 16.3% against
19.2%. Three correct numbers from a wrong setup, and exactly one detectable.

**And near-invariance is worse than invariance.** The same wrong pitch shifted
three other fractions too — 36.5 against 35.8, 96.0 against 95.2, 53.5 against
52.8 — all under a point. The large error in the corner share was challenged
and traced within one exchange; the identical error in the fractions sat inside
what both parties read as agreement, and the wrong numbers were adopted. A
quantity that is *nearly* invariant to a bad input produces agreement with a
small residue, and a residue reads as noise.

**The behavioural failure that let it through: scrutiny proportional to the
SIZE of a disagreement rather than to whether the quantity could have
differed.** One session challenged a five-sigma gap and deferred on three
sub-point ones in the same message, from the same unexamined setup. Small
disagreements between two setups are not noise — they are the same signal,
attenuated. Ask what could have differed before deciding a gap is too small to
chase.

**Recency is not provenance.** Twice in one night each session corrected the
other toward its own number and was wrong to. Having just run something makes
it feel verified, so whoever ran it most recently argues hardest — the
corrector has a fresh terminal, the correctee has a memory, and freshness reads
as evidence. **The operational form: when correcting someone toward your own
number, state your setup in the same message.** Both cases die instantly under
it — `[on make_pitch(per_side=2)]` beside a fraction, or the predicate beside a
count, and the other party spots it at once. Neither failure needed more care;
both needed the setup in the same breath as the number.

**And a peer conceding is not a measurement.** Deference from a competent peer
is nearly indistinguishable from confirmation and far cheaper for them to give
than to check. One session deferred on three sub-point gaps; the other read the
deference as agreement rather than asking what the setup was. Scrutinising
small disagreements is only half of it — the other half is not banking an
agreement that was a courtesy.

**The consequence is worse than the error.** Both sessions had independently
reproduced those three fractions to under a point and read the agreement as
strong mutual confirmation. It was not: **where a quantity is invariant to an
input, agreement about it carries no information about that input.** Two
parties agreeing is only independent confirmation if their setups are actually
independent AND the measurement is sensitive to what differs. Reproducing an
invariant is nearly free and proves almost nothing about provenance.

So: **state the scenario a measurement was drawn from, not just the call.**
"Real calls over the real draw" was true in every word and never said which
pitch, which was the whole error.

**But do not read that as "reproduction is useless" — the opposite.** Checked
against the record: all four shapes above were caught by two parties computing
the same quantity and DISAGREEING. Reproduction is how you buy the chance to
disagree, and it is the only thing that has ever caught anything here.
Agreement is the half that carries no weight when the quantity could not have
differed. So keep reproducing, and apply this test in the moment: **before
treating a reproduction as confirmation, name the input it was sensitive to.**
If nothing you measured would have differed under a different setup, the
reproduction confirms the arithmetic and nothing about the setup.

**A fifth shape, of a different kind: a correct observation generalised one
step too far, in the direction that flatters it.** "Agreement about a quantity
invariant to an input carries no information about that input" is true. It
became "independent reproduction has caught none of our errors and sensitivity
has caught all of them", which the record contradicts — all four were caught by
two parties disagreeing. The first four above are errors of EXECUTION; this one
is an error of INFERENCE, and it is the hardest to catch alone because it feels
like insight rather than arithmetic. **It runs in both directions**: a
pre-registration that condemned five measurements which never depended on the
law that might fail is the same error pointed inward, and the self-critical
form is harder to challenge because objecting to it looks like defensiveness.
Both parties contributed: one over-reached,
and the other had sold the observation as "the most important thing found
tonight", which is the framing that invites the over-reach. **When an
observation feels like the best thing you have found, that is when to check
what it does NOT license.**

**The same trap catches a prediction and its test.** When both are derived from
one model, agreement between them confirms the arithmetic, not the model — a
shared error produces a match without either being right. What is worth having
is a prediction the model could fail: here, the feasibility model says a margin
of 0.25 can NEVER fire in a corner, so any effect at all in a corner arm
falsifies it outright.

**Marking is not enough for `[assumed input]`, and saying so is part of the
rule.** A reconstruction announces itself to anyone who looks; a probe can be
contradicted by a sweep. Correct arithmetic on a guessed constant has no tell
at all — every digit is right and the derivation is sound. The 40.8% above was
caught only because someone else happened to open `_place_at_boards` for an
unrelated reason. That is luck, not method.

**What makes it method: cite the constant's source in the same breath.** Write
`r_ball = 0.035 (gym_scenario().balls[0].radius)`, not `r_ball = 0.035`. A
number carrying its provenance can be checked by a reader; a bare one cannot,
and the writer is the last person able to notice the gap.

If the reconstruction and the call disagree, **the call wins**, and the
reconstruction is the thing to throw away. See also: when a probe and a direct
sweep disagree, the sweep wins.

### A rate's denominator must come from the same population as its numerator

The sixth shape, and the only one that produces a *correct* number answering a
question nobody asked. A gym re-bin recorded the planned kick spot on every
SWING row and found **0.0% of 862 swings had the spot inside a board** — a real
measurement, from real calls, over 862 real episodes. It is worthless: a swing
happens only when a legal spot was found, so conditioning on the swing
guarantees the spot is legal. The selection produced the statistic, not the
physics.

The others above were wrong numbers and a reader could in principle have caught
them by re-measuring. This one survives re-measurement — it reproduces exactly,
forever — because the defect is in the population, not the arithmetic. **Ask
what a row had to do to be IN your sample before reading a rate off it.** If
the answer involves the outcome, the rate is about your filter.

**The instrumentation form:** record the episode's inputs at the episode's
start, on every row, not inside the branch where the interesting thing
happened. Here `ball_board` was written only on swing rows, so the rate had a
numerator and no denominator — the whole measurement had to be re-run.

**It catches people who are looking for it.** The other session hit this shape
two hours after we wrote it up, with the write-up open, on their own throw-in
measurement: a fix that expired stale positions "made things worse" (49% → 63%)
because the metric skips ticks whose track has no position, so the fix shrank
its own denominator from 1005 duck-ticks to 451 and the survivors were a biased
subset. It is not a knowledge failure. **The trap is that the biased metric is
the natural one to reuse — it is already running — and reusing a probe across a
change that alters what the probe can SEE is exactly when this bites.** Ask,
before reusing any probe to measure a fix: does the fix change which rows the
probe can observe?

**And attack a mechanism from the denominator side where you can.** The same
question — "does the collapse happen because no legal spot exists?" — was
answerable by drawing placements and calling the planner directly, with no
outcome in the loop and so no way for selection to bite. Two routes that cannot
share a bias are worth more than two runs of the route that can.

### A characterisation that silently inherits a default stops characterising anything the day the default moves

The most expensive error of 2026-09-09, and it hid for months behind tests that
all passed. The sim's camera was `62.3 x 48.8` — the **stock** Pi Camera v2's
full array. The fitted module is a wide M12 board at **116 x 60** (D 142.2,
H 116, V 60), which the owner supplied on request and which `camera-hardware.md`
§1 had listed as an open question all along. So a day of camera work compared
two geometries, **neither of which is on the robot**, and landed a headline —
*"every number here is measured on a camera the robot does not have, and is
optimistic"* — whose premise was right and whose **direction was backwards**:
the real camera is nearly twice as wide, so the sim was PESSIMISTIC.

**Why nothing caught it.** Eight tests asserted things *about* that geometry —
"outside a 62 degree FOV", "below a 48 degree vertical field",
`px_per_rad == 295.72`, a bearing bound of `0.6 rad` that was silently the old
half-angle — and **every one of them inherited the default rather than naming
it**. The geometry was pinned in eight places and stated in none. Moving it
produced nine unrelated-looking failures instead of one clear "the camera
changed", so the tests locked the value in without ever documenting it, and
locked in the wrong one.

**The rule.** A test that fixes a number it does not name is not a
characterisation, it is a hostage. **Write the constant into the assertion, or
read it from the spec** — `assert ... > spec.fov_h_deg / 2`, not `> 0.6`. Then
one changed default gives you one honest failure that says what changed,
instead of a scatter that looks like nine regressions.

**What survived, and why it is the part worth having.** Every result whose
subject was a *comparison between two geometries* is untouched: the gate
disabling itself without a track, `kick_ahead_max` replacing far seen swings
with close blind ones, the corner saturation, the 6.8 cm reachability floor,
the whole placement-versus-acceptance argument. **Levels fall; shapes hold.**
When a shared assumption turns out to be wrong, sort results by whether their
subject was an absolute or a difference before withdrawing anything — the
absolutes go, and the mechanisms usually do not.

### Every mechanical check here was wrong on first contact

Worth knowing before you trust a new one — over a single day: a `pgrep`
recipe that matched its own shell, two drafts of the atomic-write check in
this very section, a knob-gate analysis that read gating symmetrically and
reported backwards, a re-entry probe that corrupted its subject, a post-clamp
measured on the wrong duck. **Every one looked right.** The ones that survived
did so because something *independent* contradicted them — a direct sweep
against a probe, a live battery against a process check, a worked example
against an analysis — never because they read well. Budget for the second,
independent measurement; it is the one that does the work.

**A check whose failure mode is SILENCE returns the answer you expect for a
reason you did not check.** Three instances in one day, and all three were in
checks used to decide *when it was safe to act*, not in measurements — which is
the more dangerous place for them to live:

| check | what it returned | what it meant |
|---|---|---|
| `pgrep -fl python \| grep 'eval_pitch'` | no match | the console script is `eval-pitch`, hyphen — it never could match |
| `pgrep -fl … \| wc -l` | 286 | a count of *lines*; a multi-line cmdline is one process printed many times |
| `git log origin/development..HEAD 2>/dev/null` | empty | **"no such ref"**, swallowed by the redirect — not "nothing pending" |

**The general fix is to prefer a check whose answer is non-empty.** `git
ls-remote --heads origin` returns `refs/heads/main` and nothing else: a
non-empty answer that does not contain the branch you asked about is positive
evidence it was never pushed, where an empty result from a revision range is
ambiguous between "nothing there" and "the range was invalid". **An empty
result is ambiguous; a non-empty result missing the thing you looked for is
not.** When a check gates an action — is anything running, is this safe to
edit, has this been pushed — build it so that a broken check looks different
from a clean one.

### The one failure none of this catches: a favourable framing accepted in silence

Every mechanism above works on a number or on a claim someone has already made.
Registration constrains a metric before you read it; the MDE tells you what a
battery could have seen; the pair reports two mechanisms instead of picking one;
the falsifier's second dimension separates a refutation from a discovery. **Not
one of them fires when someone simply does not raise an objection they could
have raised.**

Two sessions spent a day on this and the only thing that ever caught it was the
*other session reading the same evidence and saying so*: multiplicity raised
against a table already written; a `pgrep` recipe verified only where it worked;
a bounded finding stated unbounded, twice, in both directions; a conditional
offered as a bound; a falsifier that could not describe a composite world. Each
was caught by a second reader, none by a rule.

**So do not mistake the discipline for a substitute for one.** Registering a
primary makes you honest about a number you have decided to look at. It does
nothing about the framing you accepted without noticing there was a choice —
and a framing that favours you is exactly the one you will not notice. When a
result matters, get it read by something that is not you, and give it the
evidence rather than your summary of the evidence.

**And it is not reproducible by one session working carefully.** That is the
part to say out loud before recommending any of this to someone working alone:
registration, the MDE, the pair and the falsifier's second dimension all
transfer to a solo worker unchanged — the second reader does not. A single
session can be rigorous about every number it decides to look at and still never
learn which framings it never questioned. If you are working alone, the nearest
substitutes are weak but real: write the registration where someone else will
read it, state the rival mechanism in the same breath as your own, and prefer
the measurement that would look different under the rival even when it costs
more than the one that would merely confirm you.

**A test that skips is not a test.** From the same day: a first version of the
`pred_ahead` assertion skipped, because with the gate ON 98% of swings are blind
and no swing carried a live prediction — the skip *was* the finding, and a
skipping test still protects nothing. It was rewritten to run with the gate
deliberately off, where ~40% of swings keep a prediction and the assertion has a
population to bite on. If a test's precondition is rare, construct the
population; do not let the rarity silence the check.

**And beware the easy version of this virtue.** Declining a claim right after
deriving the arithmetic that undercuts it is not the same act as noticing a
favourable framing with nothing in front of you contradicting it. The first is
just reading your own output. Do not let a record of the first stand in for
evidence of the second.

## CI runs on every pushed branch — precommit is the gate you run first

Until 2026-09-10, `development` had never been pushed (`git ls-remote --heads
origin` returned only `main`), so every guarantee in this repo came from
`precommit.sh` and the tests you remembered to run. That is how a test sat red
on `development` for five commits
(`test_the_chase_brain_tracks_a_ball_the_tof_sees_at_its_feet`, broken by
e6dd8d3, found by finally running the full suite): the subsets were green,
nobody ran `pytest tests/`, and nothing else was watching.

**2026-09-10: pushed, and the gate is real.** `development` is on the remote and
`main` fast-forwards to it; the workflow triggers on `push: branches: ["**"]`,
so every push runs the matrix (ubuntu, macOS, Windows, viewer). The first run
found two things the local suite could not: the `record-world` tests render
video and the hosted runners have no GL context (macOS raised `CGLError`,
ubuntu ABORTED the interpreter from inside GLFW and took the whole run with
it) — they now skip where there is no offscreen context; and the Linux
goldens, stale since the 2026-09-06 physics change, are re-recorded on a
hosted runner by `.github/workflows/record-goldens.yml` (`gh workflow run
record-goldens.yml`, then commit the artifact) — the store's policy makes a
hosted runner a valid recorder, bit-exact on its own CPU and by tolerance on
every other. A permanently-red CI is the cry-wolf failure at the level of the
whole project, so when the goldens go stale again, re-record them the same
morning, not "before someone pushes".

**The golden bits are a property of the HOST, not of the CPU model string
(2026-09-10, commit 7c7ddc4).** GitHub's hosted ubuntu runners report one
model string ("AMD EPYC 9V74 80-Core Processor") from more than one host, and
those hosts do not agree to the last bit: a commit whose whole diff was 38
lines of `docs/roadmap.md` failed all 11 golden assertions minutes after the
identical tree passed them, by 1-2 ulp on `qpos` and up to 5.6e-13 relative
on the reward sums an episode accumulates. So the exact comparison
(`float.hex` and the rollout digest) runs only where the lab can pin the
hardware — `BIT_EXACT_PLATFORMS` in `tests/golden_store.py`, today Apple
Silicon — and everywhere else the same rollout is compared by tolerance:
1e-9 when the machine reports the recorder's CPU model, 1e-7 when it does
not. That is still a physics regression test: perturbing one BAM constant by
1e-9 relative fails it, against the 5th-digit move a model re-export makes.
`MICRODUCK_GOLDEN_BITS=1` forces the exact comparison on (the record-goldens
workflow uses it on its own re-run, the one place on Linux where the bits are
verifiable), `=0` forces it off. Each Linux CI job prints `lscpu` so the next
disagreement says which host it landed on.

Still true: run `pytest tests/` — the whole suite, ~6 minutes — before anything
you would be embarrassed to have broken. Subsets are not a substitute; they
were green throughout the five commits above, and CI takes 25 minutes to tell
you what the suite tells you in six.

## Before you commit: `./scripts/precommit.sh` (1 second)

It runs `ruff` and imports every battery entry point. It exists because
this repo has twice been broken by a commit nobody ran anything against:
an unterminated docstring in `brain/tracker.py` broke every import and
killed three batteries that had been running for an hour, and the log that
would have said so was not read until much later.

The full suite is ~12 minutes, which is exactly why it gets skipped on
"just a doc tweak" and why those got through. Run the suite before pushing
anything that changes behaviour; run this before every commit without
exception. `ruff` parses but does not execute, so the import step is not
redundant — a module that parses can still fail on a bad relative import.

## Verification discipline — the rules that exist because of false reports

0. **A knob that changes NOTHING is broken, not null.** `stale_fix` came
   back bit-for-bit identical in all four cells of a 32-seed 2x2 — which is
   not a null result, it is a dead code path. The guard read `det.t`, and a
   `Detection` has no timestamp (only the FRAME does), so the branch never
   ran. A real null still moves the seeds it does not help; identical rows
   mean the knob is not wired. **And the test passed**, because it built a
   `SimpleNamespace` stub carrying the `det.t` the real type lacks: a stub
   that does not match the type it stands in for can make a dead path look
   alive. Prefer the real type in a test; if you must stub, assert the shape
   you are relying on.

   The same shape has now bitten three times, and the third has its own
   mechanism worth knowing. **A default argument object is bound once, at
   `def` time.** `Detector.__init__(self, ..., spec=DetectorSpec())` holds
   ONE `DetectorSpec` made at import; `World` constructs detectors without
   a `spec=`, so patching `DetectorSpec.__init__.__defaults__` — the usual
   trick for sweeping a frozen dataclass through a battery — changes what
   `DetectorSpec()` returns from then on and changes NOTHING about the
   detector that actually runs. A frustum sweep built that way returns
   identical arms and looks like a null. (`TidyParams` is safe from this
   only because `eval_tidy` builds a fresh `TidyParams(...)` per run; the
   difference is invisible from the call site, so do not reason about it —
   check.)

   **So: assert on the object that is running, not on a fresh one.**
   `assert DetectorSpec().fov_h_deg == 116` passes while the live detector
   is at 62. `assert duck.detector.spec.fov_h_deg == 116` is the check that
   would have caught it. Reach into the constructed world and read the
   value back off the thing being measured — every sweep harness in
   `scratchpad/` does this now, and the one that did not produced two
   bit-identical arms.

   **And it is not only a harness problem.** The turn-rate work found the
   same shape in SHIPPING inference code: `LearnedBrain` clipped its
   actions with the *module's* `ACT_HIGH`, so a brain trained at
   wz ±2.0 was silently re-clipped to the current default when it ran —
   halving its turn rate, and making any A/B across that bound measure
   nothing at all. A trained policy's action bounds are part of its
   contract, so they now come from its own `brain.json`. The general form:
   **a limit applied at BOTH training and inference must come from one
   place, and that place must travel with the artifact.** When a global
   default and a per-artifact value can disagree, the global one wins
   silently and the artifact is quietly worse than it was trained to be.
8. **To call a sensor wrong, reproduce its exact frame and definition
   first.** A probe splitting the ball's placement error into bearing and
   range made the same mistake twice, and both times the artefact looked
   like a finding:
   - `range_est` is the 3-D SLANT range from the camera SITE to the target's
     centre. The camera sits ~21 cm above a ball on the floor, so comparing
     it against a 2-D ground distance shows a "+5.7 cm systematic range
     bias" that is entirely the measurer's — at 0.35 m,
     `hypot(0.35, 0.21) - 0.35 = 5.8 cm`, which is the whole "bias".
   - `bearing` is in the CAMERA's frame, and brains yaw the head. Compared
     against a body-frame bearing it charges the detector for the head's
     rotation: it inflated the measured bearing error from 3.35° to 5.65°
     and invented a -1.5° bias. `frame.cam_yaw` is what
     `Tracker._associate` adds for exactly this reason.

   Both errors are the arc lesson wearing a different hat (net displacement
   in a start frame is not body-axis speed): **a difference between two
   quantities is only an error when they are the same quantity.** Before
   attributing a residual to the thing you are measuring, write down the
   sensor's frame, its origin, and whether its number is 2-D or 3-D — and
   check a case where you can predict the answer by hand. A bias that
   happens to equal a geometric term you left out is the tell.
1. **Training charts measure the noise-crutched stochastic policy.** Claims
   about a run are made from the deterministic exported ONNX, never from
   `ep_rew` curves. Export, then eval, then look.
2. **Look before you conclude.** `uv run render-rollout` writes an mp4 for
   humans and a captioned frame contact sheet for agents — read the sheet
   (`.claude/skills/render-rollout/SKILL.md` documents every caption field
   and the three classic failure patterns: the collapsed "stand", the
   spawn-assisted "trick", the cycling "hold"). Reward batteries here have
   scored a face-down crouch as standing.
   For WORLD mode (a room, the playroom, a pitch) the same eye is
   `uv run record-world <scenario>`: the lab's own `WorldState` run headless
   under a seed, to `world.mp4` + `sheet.png` + `events.txt` (every brain
   transition, fall, pick, release and goal, with sim time). Read the log
   and the sheet before saying what a duck did on the `/sim` page
   (`.claude/skills/record-world/SKILL.md`).
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

A SECOND ROBOT raises the same question one level up. The Unitree G1 here is
the Lucky Robots MJCF plus its shipped `walker.onnx`, trained by someone else
against a 99-d observation that includes base linear velocity — a privileged
quantity a real humanoid only gets from a state estimator. Nothing in this
repo identifies that robot's actuators, and its hands are frozen. So: it is a
body to prototype behaviours and to populate the `/sim` world with, and a
policy trained on it here is a lab artifact. If it ever needs to leave the
lab, it leaves through Unitree's own stack, the way a duck policy leaves
through `microduck_rl`.

This harness is for prototyping with minutes-long feedback loops. Even under
`actuator="bam"` — `train-walk`'s default since the 2026-09-06 audit — it is
a subset of the official domain-randomization stack (no IMU misalignment,
encoder bias or IMU delay; see README "Physics parity").
Once a behavior works here, port the env design to an mjlab cfg in upstream
`microduck_rl` and retrain on GPU — that stack, not this one, is the recipe
for policies that survive real hardware.
