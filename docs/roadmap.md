# Roadmap — what to run next, and what would settle it

A working list, not a plan of record. Each item carries the command to run and
the **number that decides it**, because this project's history is full of
things that looked right in a reward curve and were wrong on screen
(`microduck_local/AGENTS.md`, "Verification discipline").

Convention: `[ ]` not started · `[~]` running · `[x]` done, with the answer
written back into the item. Keep the answers — a negative result that took an
hour is worth as much as the positive one, and this file is where the next
person finds out it was already tried.

---

## Now: the 🔎 `find_ball` brain

Context: `find_ball` is a scan-and-track behavior that aims the duck at a
ball — the eyes the ball-blind kick and ground-pick policies never had. It was
prototyped and trained end to end in a 4-core cloud container at ~1.2k
steps/s (~8M steps, ~2.5 h across six warm-started stages). Everything below
wants a real machine: an M-series Mac runs the same recipe at ~12× that, so a
stage that took 25 minutes there takes ~2 minutes here.

The recipe, the slot layout, and the measured results are documented in
`microduck_local/README.md` ("🔎 `find_ball`"); the shipped export and its
lineage are in `microduck_local/policies/find_ball/`.

### 0. Verify the branch on real hardware — DONE (2026-09-03, M-series Mac)

- [x] **The test suite is green — confirmed, no bisect needed.**
      `cd microduck_local && uv run --with pytest pytest tests/`
      → **411 passed, 1 skipped, 0 failed in 37 s** on the M-series Mac
      (2026-09-03). All 12 golden bit-parity tests (`test_step_perf_parity`,
      `test_bam_perf_parity`, the `test_symmetry` drift case) pass here, which
      settles it: the cloud failures were environment drift (a Mesa/BLAS-level
      float difference), not this branch. The suite is 412 passed with the
      `body_aimed` term and its test added below.
- [x] **The battery reproduces.** `uv run eval-find-ball policies/find_ball/policy.onnx --episodes 40`
      → **it does**, and closely (Mac, 2026-09-03; cloud numbers in brackets):

      | bucket | n | found | t_first med | in frame | centred | fell |
      |---|---:|---:|---:|---:|---:|---:|
      | front | 10 | 100% [100%] | 0.03 s [0.03] | 100% | 98% | 0 [0] |
      | side | 20 | 90% [85%] | 0.61 s [0.60] | 77% | 62% | 0 [0] |
      | back | 10 | 60% [60%] | 0.94 s [0.94] | 32% | 0% | 2 [2] |
      | all | 40 | 85% | 0.43 s | 72% | 56% | 2 |

      Only the side bucket moved (85% → 90%, i.e. one episode). Note the
      battery takes **2.2 s** here — it is free, run it on everything.
      Worth recording for every item below: this Mac trains at **16-27k
      steps/s** against the container's ~1.2k, so the full 8M-step chain is
      ~7 minutes, not 2.5 hours.
- [x] **Looked at it — and item 1 is visible in the frames.** `uv run
      render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb --episodes 4`
      → it stands cleanly the whole time (`trunk_z` 0.113-0.116 against the
      0.120 stand reference, `floor:none`, both feet 98-99%, 0 reversals —
      no collapsed-crouch or cycling failure here). What the sheet shows is
      the gaze-policy problem, plainly: in ep3 the **feet never move for
      10 s** while the head is visibly cranked round, and the true body
      bearing goes p+98° → p+75° in the first second and then sits at
      **p+60° for the last 8 s** with the detector holding `x+0.21`. Aim
      streak 0 steps in every episode — the handoff never comes close.
      Confirms the item-1 trace independently on this machine.
- [x] **It works in the lab — first browser run of the ball path, and it
      is correct.** `cp -r microduck_local/policies/find_ball
      microduck_local/runs/find_ball`, `uv run duck-lab --port 8789
      runs/find_ball`, viewer on `?lab=127.0.0.1:8789`, then drop
      `run:find_ball` on a duck from the 🧠 palette.
      → the orange ball draws next to the duck, and the label carries the
      spawn note exactly as specified (`↻ ball -61° 0.7m blind`,
      `↻ ball +87° 0.9m prior` — both the prior and blind variants). The
      marker follows ball events live. Assigned duck: 0.00 m/s, 1 fall,
      r̄ 6.6-8.9 over 16 s.

      **Rough edge found on the way (pre-existing, not this branch):** a run
      dir passed as a `duck-lab` CLI positional (`duck-lab runs/find_ball`)
      is NOT recognised as a trick duck, because `build_ducks` never sets
      `policy_id` for it (`viz_server.py:897`) and `is_trick_duck` bails on
      anything without a `run:` prefix (`viz_server.py:2141`). The lab then
      sends this ball-brain a 0.9 m/s **walk command**, which is pure
      out-of-distribution noise to it: **1058 falls and r̄ -3.2** in ~4
      minutes. Dropping the identical run dir from the palette sets
      `policy_id="run:find_ball"`, commands go to zero, and the same policy
      is immediately healthy. So the documented path works and the CLI path
      silently does not — the fix is one argument
      (`add(p.name, ..., policy_id=f"run:{p.name}")` for dirs under
      `RUNS_DIR`), and it is exactly the failure mode the `is_trick_duck`
      docstring says it exists to prevent.

### 1. The open problem: the body never turns — `find_ball` is a gaze policy

**Sharpened by the handoff work (see item 4, which is done): the duck aims its
HEAD at the ball and leaves its body where it was.** Traced over a full
8 s episode with the ball 15° off at 1.2 m: it centres the camera perfectly
(bearing +0.11, elevation 0.00, in frame 100% of steps) using **21° of head
yaw**, while the body bearing to the ball stays at 18–20° and drifts slightly
*further* away. It never squares up, so it never satisfies the kick handoff.

This is one problem with the back-bucket weakness, not two. The head does the
eyes-on job alone and for free, while turning the body costs steps, smoothness
penalties, and fall risk — so the policy takes the cheap option. `face_the_ball`
is paid while the ball is seen, but its tight layer (std 0.4 rad) still pays
~2/3 at 19° off, so the last 20° has almost no gradient behind it; and
`turn_to_belief` is gated to fire only while the ball is *out* of frame, so
once the head finds the ball nothing pays for the body to catch up.


**FIRST, AND IT RE-BASELINES EVERYTHING BELOW: most of this was
under-training, not mis-pricing.** Before A/B-ing any fix, the *unchanged*
recipe was retrained on the Mac — the declared 3-stage curriculum via the
lab's `/teach` (which is the only path that chains stages; `train-behavior`
does not), 8M steps total, ~7 minutes, `body_aimed` pinned to 0 so it is
literally the shipped recipe. Run `teach-find_ball-5f89d9`. Against the
shipped stage-5 export, on 40 static-ball episodes:

| | shipped s5 | s4 baseline | **control (same recipe, Mac, 8M)** |
|---|---:|---:|---:|
| head yaw while centred, front | 21.1° | 18.7° | **13.9°** |
| head yaw while centred, all | 40.8° | 41.1° | **21.1°** |
| body bearing turned out, all | 23.7° | 17.9° | **57.6°** |
| handoff fired (front/side/back) | 40/10/0% | 50/0/0% | **60/15/10%** |
| falls / 40 | 2 | 0 | **0** |
| back-bucket in frame / centred | 32% / 0% | — | **54% / 52%** |
| median time to first sight, all | 0.43 s | — | **0.15 s** |

So the shipped export was **not** a converged instance of its own recipe: the
same terms, trained straight through the declared curriculum, turn 2.4× more
body, halve the head yaw, drop the falls to zero and go from *never* centring
a back-bucket ball to centring it half the time. Any A/B run against the
shipped ONNX would have credited a reward change with all of that.

**But item 1's symptom survives retraining**, which is why the fixes below are
still worth running: rendered (`/tmp/rr-ctrl`), the control visibly steps its
body round — ball at p+70° is p+20° by 1.4 s, and after a ball event throws it
to p−119° it turns all the way back — and then **parks 18–20° off and leaves
the last stretch to the neck**, aim streak 0, handoff never fired in that
episode. 13.9° of head yaw on a front start is under the 0.25 rad gate; 24.4°
on a side start is not. The remaining problem is precisely "the last 20°".

Two process notes for whoever runs the fixes:

- **The shipped ONNX cannot be warm-started from.** `policies/find_ball/` has
  no `model.zip` / `vecnormalize.pkl` (the container was ephemeral), so
  `--init-from runs/find_ball` in the item below cannot work as written. Every
  arm has to retrain the chain, which on this machine is fine (~7 min).
- **The lab always trains at seed 0** (`TrainingJob` never passes `--seed`), so
  two `/teach` arms are seed-matched for free — but a *second* training seed,
  which `AGENTS.md` wants before crediting a small effect, needs the CLI and a
  hand-chained curriculum.

Three candidate fixes, cheapest first — **A/B them, do not stack them**:

- [x] **Price the state you actually want — SHIPPED AS `body_aimed`, and it
      is the fix.** Added to `behaviors/ball.py` as
      `exp(-(bx²+by²)/0.25²) × exp(-head_yaw²/0.3²)` while seen, weight 2.0,
      locked by `test_find_ball_body_aimed_pays_the_body_not_the_neck`. Both
      factors are detector output + one joint encoder, so it prices exactly
      what `_ball_handoff_due` gates on and needs no privileged state.
      A/B: run `teach-find_ball-3fc099` vs the control `teach-find_ball-5f89d9`
      — identical recipe, identical 3-stage curriculum, 8M steps, same seed
      (the lab always trains seed 0), the **only** difference the weight.

      | 40 static-ball episodes | control (0.0) | **body_aimed 2.0** |
      |---|---:|---:|
      | head yaw while centred, front / side / back | 13.9 / 24.4 / 23.0° | **4.3 / 6.2 / 9.1°** |
      | head yaw while centred, all | 21.1° | **6.4°** |
      | final body bearing, all | 32.4° | **9.3°** |
      | body bearing turned out, all | 57.6° | **80.7°** |
      | **handoff fired, front / side / back** | 60 / 15 / 10% | **100 / 95 / 70%** |
      | handoff fired, all | 25% | **90%** |
      | battery: found, all | 85% | **98%** |
      | battery: back-bucket found | 60% | **100%** |
      | battery: in frame / centred, all | 78% / 76% | **85% / 81%** |
      | **battery: falls / 40** | **0** | **4** |

      → **decide-on met, and not marginally**: head yaw 6.2° on a side start
      against the 14° (0.25 rad) gate, and the handoff fires on **95% of side
      starts** against 15%. Rendered and read (`/tmp/rr-treat-solo`), so this
      is not a reward-sum claim: from a ball at **p+98°** the body is at
      **p+11° by 0.72 s** and holds p+9…p+12 square-on, head straight, for the
      remaining 8 s — aim streaks of **7.2 s and 7.8 s** on the −80° and +98°
      side starts, where the control never held one at all. `trunk_z` 0.115-0.117,
      `floor:none`, 0 reversals.

      **The cost is falls: 0 → 4 per 40** (1 side, 3 back) — it commits to big
      turns now and sometimes topples doing one. Nothing else regressed.
      The weight-down arm (`teach-find_ball-0ad2b0`, `body_aimed` 1.0, same
      chain) says it is a genuine trade, not a mispricing artifact — every
      column moves monotonically with the weight:

      | body_aimed | head yaw, all | handoff, all | battery found | back in frame | falls / 40 |
      |---:|---:|---:|---:|---:|---:|
      | 0.0 (control) | 21.1° | 25% | 85% | 54% | **0** |
      | 1.0 | 10.8° | 78% | 82% | 44% | 2 |
      | 2.0 | **6.4°** | **90%** | **98%** | **80%** | 4 |

      Halving the weight halves the falls and gives up most of the win, so
      there is no free setting on this axis — the falls are the duck
      committing to turns it did not attempt before, and 2.0 is the
      recommended ship. If they have to go, the honest levers are more steps
      or a stability term, not a smaller `body_aimed`. Caveat worth keeping:
      one training seed per arm (the lab pins seed 0), and `AGENTS.md` wants a
      second seed before crediting a small effect — the head-yaw and handoff
      columns are far too big to be seed noise, the ±2 falls plausibly are.

      **Do not read the `--handoff` render as find_ball falling.** With
      `--handoff ball_kick_right`, all 4 episodes fired the handoff (0.72-2.32 s,
      side starts included — that answers item 4's re-render) and all 4 then
      fell within ~0.5 s of the switch. The same 4 seeds with no handoff fell
      **zero** times. That is the kick toppling a duck it was handed at
      0.8-1.9 m, exactly as item 4 predicted; it is an argument for the
      approach behavior, not evidence against this term.
- [x] **Tighten `face_the_ball`'s tight layer** (std 0.4 → 0.2 rad) —
      **SHIPPED.** `_BALL_FACE_TIGHT_STD` is 0.2 in the recipe, and
      `policies/find_ball/policy.onnx` is this arm's export. It beats
      `body_aimed` on the trade that matters. The std is now the named constant `_BALL_FACE_TIGHT_STD`
      (`behaviors/ball.py`) precisely so this A/B is one edit. Arm
      `teach-find_ball-c60e89`, `body_aimed` pinned to 0 so nothing is
      stacked. The feared failure — a steeper Gaussian producing a policy
      that fights to hold an exact pose — did not appear: 0 reversals, and
      rendered it holds a clean square-up.
- [x] **Ungate `turn_to_belief`** — **negative result, and it is the wiggle
      the item predicted.** Arm `teach-find_ball-961dfd` (`body_aimed` 0,
      `_BALL_TURN_GATED_TO_LOST = False`). It is worse than the *untouched
      control* on every axis: handoff 53% vs the control's 38% but with
      **7 falls / 60** against 0, and `psi_turned` collapses to **19.1°**
      (control 42.4°, fix 2 52.1°) while the final bearing gets *worse*
      (70.9° vs 47.6°). Rendered at 50 fps, the mechanism is visible in one
      episode: from a −80° start it turns in to **p−15° by 2.9 s** and then
      **drifts back out to p−28° and parks there** for the last 5 s. A
      yaw-rate pay that never switches off makes *arriving* worth nothing, so
      the policy keeps some bearing in hand to turn back toward. The gate is
      correct; the constant stays `True` and this is why.

### The three fixes, side by side

Every arm is the same 3-stage curriculum, 8M steps, seed 0, via `/teach`;
they differ in exactly one thing each. **Read the events-on block, not the
battery block, when judging falls** — `eval-find-ball` pins
`MICRODUCK_BALL_EVENT_RATE=0`, and the two regimes disagree about which arm
is safest (fix 2 shows 0 falls in the battery and does fall under events).
The recipe trains, and `render-rollout` runs, with events at 0.33.

Battery FINDING table, 40 static-ball episodes (`uv run eval-find-ball`):

| arm | found | in frame | centred | falls |
|---|---:|---:|---:|---:|
| shipped s5 | 85% | 72% | 56% | 2 |
| control (unchanged recipe) | 85% | 78% | 76% | 0 |
| fix 1 `body_aimed` 1.0 | 82% | 71% | 69% | 2 |
| fix 1 `body_aimed` 2.0 | **98%** | **85%** | **81%** | 4 |
| **fix 2 face std 0.2** | 95% | 79% | 76% | 0 |
| fix 3 turn ungated | 85% | 79% | 77% | 3 |

AIMING table, 60 episodes with **ball events on** — the honest regime
(`uv run eval-find-ball <onnx> --episodes 60 --events 0.33`):

| arm | head yaw \| centred | handoff fired | **falls / 60** |
|---|---:|---:|---:|
| shipped s5 | 44.8° | 15% | 2 |
| control (unchanged recipe) | 25.0° | 38% | **0** |
| fix 1 `body_aimed` 1.0 | 13.3° | 68% | 4 |
| fix 1 `body_aimed` 2.0 | **8.1°** | **83%** | **10** |
| **fix 2 face std 0.2** | 18.6° | 68% | **1** |
| fix 3 turn ungated | 20.4° | 53% | 7 |

**Fix 2 is shipped** (`_BALL_FACE_TIGHT_STD` 0.2, and the arm's export
promoted to `policies/find_ball/`). It nearly doubles the handoff rate over
the control (38% → 68%) for one extra fall in sixty, and it is a one-constant
change to a term that already exists. `body_aimed` 2.0 is the better *aimer*
by a distance — it is the only arm that puts head yaw under the gate and the
only one that holds multi-second aim streaks — but 10 falls / 60 is a 17%
fall rate, and this file's own rule is that falls are the veto. `body_aimed`
is therefore left in the recipe **at weight 0**: measured, tested, documented,
one edit away. The thing that would settle it is whether those falls survive
more steps or a stability term, which is the item below, not a fourth term.

**Left in the tree:** fix 2 shipped (`_BALL_FACE_TIGHT_STD = 0.2`); fix 3
rejected and its gate kept (`_BALL_TURN_GATED_TO_LOST = True`); fix 1 present
but unpriced (`body_aimed` weight 0). `policies/find_ball/policy.onnx` is the
fix-2 export — and unlike its cloud-trained predecessor it has a real SB3
checkpoint behind it, at `runs/teach-find_ball-c60e89-s3/` on the machine that
trained it (4.6 MB, not in git: `runs/` is gitignored and this repo does not
ship raw checkpoints). If warm-startability should survive a machine, that is
the decision to make. All five trained chains are under `runs/`
(`5f89d9` control, `0ad2b0` / `3fc099` fix 1 at 1.0 / 2.0, `c60e89` fix 2,
`961dfd` fix 3). The aim probe that produced these numbers is now
**part of `eval-find-ball`** (an AIMING table beside the existing FINDING one,
plus `--events`), not a second command: the two share one rollout loop, so the
battery and the aim columns cannot drift apart, and every future measurement
gets these columns for free. Locked by `tests/test_eval_find_ball.py`.

**Caveat on all of it:** one training seed per arm. The lab pins `--seed 0`
(`TrainingJob` never passes one), so the arms are seed-matched to each other
for free, but `AGENTS.md` wants a second seed before crediting a small
effect. The head-yaw and handoff columns move far too much to be seed noise;
the fall counts (0 / 1 / 4 / 7 / 10) are small enough that only the extremes
are safe to lean on.

Also still open, and probably the same root cause:

- [~] **Train the full circle longer.** The command as written **cannot run**:
      `--init-from runs/find_ball` needs `model.zip` + `vecnormalize.pkl` and
      `policies/find_ball/` has only the ONNX. Answered instead by the control
      arm above — the same recipe trained straight through its own curriculum
      for 8M steps.
      → **half met.** Back-bucket found **60%**, so the ≥ 80% bar is missed;
      but **0 falls / 40** and **0 falls / 60 with events on**, and the
      back-bucket goes from 32%/0% in frame/centred to 54%/52%. The bar is met
      by two other arms — fix 1 at 2.0 (back 100% found, 4 and 10 falls) and
      fix 2 (back 100% found, 0 and 1 falls) — so **fix 2 is the first arm to
      satisfy both halves of this item at once**, on the battery at least.
      Still open: whether fix 2's single events-on fall goes away with more
      steps.
- [ ] **A/B the turn term** at weight 0 vs 1.0, same warm start, same seed, 2M
      steps each (`--weights-json` or the viewer's sliders).
      → still open, but **re-scope it before running it**: the premise was
      that stage 5's `turn_to_belief` bought back-bucket found rate at the
      cost of the chain's only falls. The control arm shows the term is not
      what was doing either — with the identical recipe trained through, back
      found stays 60% and the falls go to **zero**. So most of what stage 5
      looked like it bought (and cost) was under-training. Worth running as a
      clean 0-vs-1.0 A/B on the Mac chain, where it is 14 minutes.
- [ ] **If falls persist:** drop the weight to 0.5 before adding anything new.
      A recipe that needs a fourth term to survive its third is usually
      mispriced, not underspecified.

### 2. Sim2real realism (cheap, run each as a 1M-step fine-tune)

The whole chain trained under `actuator="xml"` with no domain randomization —
a prototype, not a robot policy. These are the knobs that decide whether the
behavior survives contact with reality.

- [ ] **BAM servos.** `MICRODUCK_ACTUATOR=bam` fine-tune.
      → **decide on:** side-bucket median time-to-first-sight stays under 1 s.
      BAM's real current limit slows head yaw, and head yaw *is* the search.
- [ ] **Detector dropout.** `MICRODUCK_BALL_DROPOUT=0.1` fine-tune, then run
      the battery with dropout still on.
      → **decide on:** in-frame share drops by no more than a few points. The
      real NPU detector will miss frames; the brain should not lose the ball
      when it does.
- [x] **FOV sensitivity — NOT a blocker, and the reason is worth knowing.**
      `eval-find-ball` grew the `--env KEY=VALUE` this item always assumed it
      had. Swept far wider than the item asked (40 static-ball episodes each,
      shipped brain):

      | HFOV | 24° | 30° | 40° | **48°** | 56° | 70° | 90° |
      |---|---|---|---|---|---|---|---|
      | found | 95% | 95% | 95% | 95% | 95% | 95% | 95% |
      | in frame | 71% | 77% | 78% | 79% | 79% | 81% | 82% |
      | handoff | 82% | 80% | 80% | 80% | 80% | 78% | 85% |
      | head yaw \| centred | 12.7° | 13.9° | 14.0° | 14.4° | 14.6° | 14.1° | 14.0° |
      | falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

      → **decide-on says ship**: found rate does not move AT ALL across a
      3.75× range, and in-frame share degrades gracefully rather than
      swinging. The placeholder IMX219 intrinsics do not have to be measured
      before this goes near hardware.

      The knob was verified to actually reach the detector before believing
      the flatness (`half_h` moves, and a ball 22° off is LOST at 40° and SEEN
      at 56°) — a flat sweep is only good news if the thing swept.

      **Why it is flat, because this is a design property worth keeping:** the
      detector reports `bx = angle / half_h` — a bearing NORMALIZED by the
      field of view — so the policy's control loop works in frame-relative
      units and the geometry cancels. FOV only sets how wide the "seen" window
      is in angle, and a head sweep covers the circle regardless.

      **What this does NOT clear, and it is the sharper version of the same
      worry:** the sweep moves the camera and the normalizer together. The
      dangerous case on hardware is a MISMATCH — a real 56° camera whose
      daemon divides by an assumed 48° — which is a constant gain error on
      bx/by, not a FOV change, and is untested. The contract to write down is
      "bearing normalized by the camera's ACTUAL horizontal/vertical FOV";
      get that right and, per the table, the precise value stops mattering.

- [x] **The camera is not an IMX219, and VFOV — not HFOV — is the axis that
      matters.** The datasheet (2026-09-03) is 1920x1080 on 2.75 µm pixels,
      1/2.9 in: **16:9**, 5.28 x 2.97 mm active, 6.06 mm diagonal, up to
      90 fps. `ball.py` derived its `48° x 62°` pair from a 4:3 IMX219, and a
      4:3 assumption is the one thing a 16:9 sensor definitely breaks — in
      portrait the tall/wide angular ratio is **1.778**, so VFOV should be
      ~77° against that 48°, not 62°. The absolute pair still cannot be
      derived without the **lens focal length**, which is the one number
      nobody has written down (portrait reference: f=2.8 mm → 56 x 87°,
      f=3.6 mm → 45 x 73°, f=4.0 mm → 41 x 67°).

      VFOV swept on the shipped brain, 60 episodes at `--events 0.33`:

      | VFOV | 40° | 50° | **62°** | 77° | 90° |
      |---|---|---|---|---|---|
      | found | 95% | 97% | 97% | 97% | 97% |
      | in frame | 58% | 62% | 66% | 69% | 71% |
      | t_first med | 0.58 s | 0.27 s | 0.24 s | 0.21 s | 0.20 s |
      | falls | 1 | 1 | 1 | 3 | 3 |

      Unlike HFOV (flat), VFOV has a real gradient — 13 points of in-frame
      share — which is what you would expect of a behavior built around
      nodding down for near-floor balls.

      **Orientation matters more than the sensor swap**, and it is the thing
      to confirm on the hardware:

      | mount | found | in frame | centred | handoff | t_handoff |
      |---|---|---|---|---|---|
      | current placeholder 48 x 62 | 97% | 66% | 62% | 68% | 2.08 s |
      | **16:9 PORTRAIT 48 x 77** | 97% | **69%** | 63% | **72%** | 2.10 s |
      | 16:9 LANDSCAPE 77 x 48 | 93% | 60% | 56% | 65% | 3.30 s |

      **The focal length arrived (2026-09-04) and it is a fisheye: EFL 2.9 mm,
      H 116° / V 60° / D 142.2°, max DFOV 165°.** Those are the sensor's own
      axes; mounted rotated 90° they swap into the robot's frame as **60°
      across, 116° up** — far wider than the 48 x 62 placeholder, and tall,
      which is the right shape for a duck hunting a floor ball. The knobs are
      now these values.

      | config | found | in frame | centred | handoff | falls |
      |---|---|---|---|---|---|
      | placeholder 48 x 62 | 97% | 66% | 62% | 68% | 1 |
      | **real, PORTRAIT 60 x 116** | 98% | **75%** | **66%** | **72%** | 5 |
      | real, LANDSCAPE 116 x 60 | 97% | 65% | 61% | 67% | 2 |

      Portrait is better at finding across three eval seeds (+6 to +9 points
      of in-frame share); landscape is worse than the placeholder. **Mount it
      portrait.**

      The projection needed no rework, which is worth knowing before someone
      "fixes" it: `_ball_sense` divides an ANGLE by the half-FOV angle, an
      equidistant f-theta mapping — exactly what a fisheye does. A rectilinear
      (tan/tan) model would have been badly wrong at 58°.

      **Two consequences that are not about numbers:**

      - **The near-floor nod is obsolete.** A ball at the closest spawn
        distance (0.3 m) sits 35.6° below a level gaze: outside the old 31°
        half-VFOV, comfortably inside the real 58°. The whole spawn window is
        now visible with the head level, so the recipe's nose-down pitch band
        (`_BALL_PITCH_DOWN`) covers a case the hardware does not have. Locked
        by `test_find_ball_near_balls_no_longer_need_a_nod` so a narrower lens
        or a landscape remount fails loudly.
      - **The aim tolerances are FOV-RELATIVE, not angular — and that is a
        design bug the wider camera exposed.** `eyes_on_ball`'s stds and the
        handoff gate are all in normalized-bearing units, so widening the
        camera silently loosened them:

        | | placeholder 48 x 62 | real 60 x 116 |
        |---|---|---|
        | "centred" (tight layer) | ±6.0° across, **±7.8° up** | ±7.5° across, **±14.5° up** |
        | kick handoff gate | ±6.0°, ±7.8° | ±7.5°, **±14.5°** |

        The vertical tolerance nearly DOUBLED without anyone editing a reward.
        This matters on hardware too: the daemon runs the same gate, and it
        will now hand the kick a ball 14.5° off vertically.

- [ ] **Retraining at the real FOV made it WORSE — negative result, and the
      tolerance rescale above is the leading suspect.** Run
      `teach-find_ball-c06bbb` (same recipe, 8M steps, only the FOV knobs
      changed). Against the shipped brain, both evaluated in the real-FOV env
      at `--events 0.33`, three eval seeds:

      | seed | old brain (trained 48 x 62) | new brain (trained 60 x 116) |
      |---|---|---|
      | 123 | 98% found, 75% in frame, **5** falls | 88%, 58%, **12** |
      | 200 | 100%, 66%, **1** | 93%, 63%, **9** |
      | 300 | 97%, 64%, **2** | 85%, 58%, **14** |

      Rendered, it is bimodal rather than uniformly bad: two of four episodes
      hold 7.24 s aim streaks at 98-99% in frame, one falls at 1.26 s, one
      holds the ball 24% of the time. So the skill is there and something is
      destabilising it.
      → **done, and the diagnosis was right about the AIM and incomplete about
      the FALLS.** The tolerances are now in degrees (`_BALL_AIM_DEG` 7,
      `_BALL_EYES_TIGHT_DEG` 7, `_BALL_EYES_WIDE_DEG` 16, chosen to reproduce
      what the 48x62 placeholder happened to mean), the battery's `centred`
      column is measured the same way, and the recipe retrained as
      `teach-find_ball-72af49`. Three eval seeds, real-FOV env,
      `--events 0.33`, all three brains scored with the new angular metric:

      | | old brain (48x62) | FOV retrain (normalized) | **angular tolerances** |
      |---|---|---|---|
      | in frame | 75 / 66 / 64% | 58 / 63 / 58% | **86 / 79 / 79%** |
      | centred | 56 / 54 / 54% | 45 / 49 / 45% | **74 / 69 / 70%** |
      | worst t_first | 6.9 / 7.6 / 7.9 s | 7.1 / 5.7 / 4.7 s | **1.1 / 4.6 / 1.7 s** |
      | found | 98 / 100 / 97% | 88 / 93 / 85% | 98 / 92 / 92% |
      | **falls** | **5 / 1 / 2** | 12 / 9 / 14 | 5 / 8 / 11 |

      The aim recovered and then some — best of all three on in-frame, centred
      and worst-case first sight, by a wide margin. **The falls did not.** They
      improved on the FOV retrain (12/9/14 -> 5/8/11) but remain well above
      the old brain's 5/1/2, so the tolerance rescale was *a* cause of the
      regression and not the only one.

      **Keep the angular tolerances regardless of the falls.** They fix a real
      defect that has nothing to do with any policy: with tolerances in
      normalized bearing, changing the camera silently changes what "centred"
      means, what the reward pays, and what the kick handoff accepts — on
      hardware as well as in sim. A recipe whose meaning moves when a lens
      changes is wrong even when it happens to score well.

- [ ] **The remaining falls are an over-aggressive belief-driven turn, and
      that is a stability problem, not an aiming one.** Rendered
      (`/tmp/rr-ang`), 5 of 6 episodes are excellent — in frame 88-99%,
      centred 74-98%, aim streaks to 7.26 s. The one fall is legible and is
      the same shape as the other arms' back-start falls: from a ball at
      p+173° the duck turns ~57° in 0.7 s while `trunk_z` sinks 0.127 ->
      0.066, tilt climbs 0° -> 41°, and it goes single-footed at 0.76 s. It
      never sees the ball at all, so this is purely the belief-driven turn,
      executed as **a one-footed leaning pivot instead of steps**.
      → **decide on:** back-bucket falls at or below the old brain's ~2/60 at
      `--events 0.33`, without losing the in-frame/centred gain above.
      Candidates, cheapest first, and **not by adding a fourth term** (this
      file's own rule — a recipe that needs a new term to survive its last one
      is mispriced).

      **`step_dont_skid` was the obvious candidate and it is NOT the lever.**
      Swept as a weight override through `/teach` (no code change), three eval
      seeds each at `--events 0.33`:

      | `step_dont_skid` | falls / 60 | in frame | centred |
      |---|---|---|---|
      | 0.5 (`teach-find_ball-2a07a0`) | 18 / 12 / 18 | 66 / 68 / 70% | 53 / 55 / 56% |
      | **1.0 — the recipe's own value** | **5 / 8 / 11** | **86 / 79 / 79%** | **74 / 69 / 70%** |
      | 2.0 (`teach-find_ball-e92868`) | 12 / 18 / 17 | 80 / 73 / 74% | 61 / 53 / 57% |

      Both directions are worse on **every** axis, so 1.0 is at or near a local
      optimum and the falls come from somewhere else. Raising it was reasoned
      backwards, and the mistake is worth keeping: the term is a **positive pay
      for a foot being AIRBORNE while rotating** (0.5 per foot inside a
      0.06-0.35 s window), so a bigger weight buys *more* one-footed turning —
      which is precisely the fall mode it was supposed to suppress. It declines
      to pay a sustained lean only because the lean outlasts the window; it
      never charges for one.

      **Revised hypothesis, and it survives everything measured so far: the
      wide camera removed an implicit posture constraint.** Every arm trained
      at the real 60x116 FOV falls more (5-18 per 60) than the brain trained
      at the narrow 48x62 placeholder (5 / 1 / 2), regardless of tolerance
      units or skid weight. With a 31° half-VFOV, tilting far enough to topple
      LOST the ball, so the centring pay was quietly buying uprightness; at
      58° the duck can lean hard and keep the ball in frame, and nothing in the
      recipe replaces what the narrow lens used to enforce.
      **`stay_upright` swept too, and it does not control the falls either.**
      Weight overrides through `/teach`, three eval seeds each at
      `--events 0.33`:

      | `stay_upright` | falls / 60 | in frame | centred | t_first med |
      |---|---|---|---|---|
      | **1.5 — the recipe's own value** | 5 / 8 / 11 | **86 / 79 / 79%** | **74 / 69 / 70%** | **0.18 / 0.22 / 0.34 s** |
      | 2.0 (`teach-find_ball-17d994`) | **21 / 21 / 22** | 71 / 76 / 72% | 62 / 67 / 62% | 0.24 / 0.26 / 0.30 s |
      | 3.0 (`teach-find_ball-60be16`) | **0 / 0 / 0** | 52 / 52 / 49% | 46 / 47 / 45% | 0.88 / 1.84 / 2.31 s |
      | 5.0 (`teach-find_ball-02e577`) | 8 / 4 / 6 | 74 / 75 / 69% | 57 / 58 / 53% | 0.36 / 0.66 / 0.74 s |

      Falls go 5-11 -> 21-22 -> 0 -> 4-8 as the weight rises monotonically.
      That is not a function of the weight in any usable sense, and the one
      setting that does eliminate falls (3.0) buys it by **abandoning the
      task**: in-frame share collapses to ~50% and median time to first sight
      goes from 0.2 s to 0.9-2.3 s. It stops falling by stopping turning.

      **`1.5` stays.** Nothing in the sweep beats it on a single aim column.

      **Training here is bit-for-bit deterministic, which is what makes that
      table interpretable — and it also kills the obvious excuse.** Re-running
      the 1.5 configuration from scratch (`teach-find_ball-38d788`) reproduces
      `teach-find_ball-72af49` EXACTLY: same found rate, in-frame, centred and
      fall count on all three eval seeds. So none of the swings above are
      run-to-run noise; they are exact single-sample draws, and the map from
      this weight to the fall count is simply not smooth. Two consequences:

      - A weight sweep at one training seed cannot find a "good weight" for
        falls, because neighbouring weights land in qualitatively different
        basins. More sweeping is not the answer.
      - Determinism is a *reproducibility* guarantee, not a *generalization*
        one. Every arm in this file is one draw from seed 0 (the lab pins it);
        a second seed would need the CLI and a hand-chained curriculum.

- [ ] **The falls want a structural fix, and there is a specific candidate
      this codebase has already used once.** `_upright` pays
      `exp(-sin²(tilt)/0.05)` — a std of ~12.9° of tilt, so it is worth 0.55
      at 10°, 0.10 at 20° and 0.007 at 30°. Past ~25° there is essentially no
      pay left and therefore **no gradient pulling the duck back**: the term
      prices being upright but not *recovering*, so once the lean is committed
      the fall is free. That is exactly the failure `face_the_ball`'s docstring
      records for its own std-1.5 Gaussian ("paid 0.04 with the ball straight
      behind and sloped nowhere the policy was"), and the fix there was a
      raised cosine that slopes all the way round.
      → **tried, and it is Pareto-DOMINATED.** `_upright_wide` (in the catalog,
      `_upright_term(w, wide=True)`) is the two-layer shape `eyes_on_ball` and
      `head_up` use: `0.5*exp(-s/0.5) + 0.5*exp(-s/0.05)`, worth 0.44 at 20°
      of tilt, 0.31 at 30 and 0.21 at 41 where the narrow term pays 0.10,
      0.007 and 0.0002. It is **opt-in**, because `_upright` gates `one_leg`'s
      hold and SCALES backflip's brake and two of imitate's terms — widening
      it globally would silently re-price four other behaviors.

      A/B on find_ball (`teach-find_ball-fe8c25`) did cut the falls and did
      cost the aim: 5/4/5 against 5/8/11, in-frame 66/61/54% against
      86/79/79%. Rendered, 4/4 upright with no falls — but the aim streak is
      **zero in three of four episodes**: it keeps the ball and stops squaring
      up. Nothing adopts it; `test_no_recipe_has_adopted_the_wide_upright`
      locks that and points here.

### The falls/aim frontier — why three sweeps in a row "failed"

Every arm trained at the real FOV, mean over three eval seeds at
`--events 0.33`:

| arm | falls / 60 | in frame | centred | on the frontier |
|---|---:|---:|---:|---|
| `stay_upright` 3.0 | **0.0** | 51.0% | 46.0% | yes |
| **old brain (trained at the NARROW 48x62)** | **2.7** | 68.3% | 54.7% | **yes** |
| wide `_upright` | 4.7 | 60.3% | 50.0% | no — dominated |
| `stay_upright` 5.0 | 6.0 | 72.7% | 56.0% | yes |
| **angular tolerances, recipe as it stands** | 8.0 | **81.3%** | **71.0%** | **yes** |
| `step_dont_skid` 2.0 | 15.7 | 75.7% | 57.0% | no |
| `step_dont_skid` 0.5 | 16.0 | 68.0% | 54.7% | no |
| `stay_upright` 2.0 | 21.3 | 73.0% | 63.7% | no |

The three sweeps did not fail to find a good weight. They found that **falls
and aim are in direct tension in this recipe, and every reward lever tried
moves ALONG that frontier rather than pushing it out**: 0 falls at 51%
in-frame, 2.7 at 68%, 6.0 at 73%, 8.0 at 81%. Pick a point; there is no
setting that gets both.

The most informative row is the second. The brain trained at the WRONG,
narrow camera sits on the frontier and beats every wide-FOV-trained arm at
its fall count — because at a 31° half-VFOV, leaning far enough to topple
LOST THE BALL, so the centring pay was buying uprightness for free. The real
lens removed that coupling, and no reward term has replaced it.

- [ ] **Push the frontier out by changing the WORLD, not the pay.** Three
      reward sweeps is enough evidence that this is not a pricing problem, and
      it is this file's own lesson: "if rollouts never contain the skill you
      are paying for, ladder the physics, not the reward" (`AGENTS.md`). The
      skill missing here is **recovering from a committed lean**, and no
      episode ever starts in one — the duck only ever reaches 30-40° of tilt
      on its way to the floor, where the value function has nothing to learn
      from. Give it a spawn family that starts already leaning 20-40°, the
      reverse-curriculum pattern `backflip` and `headstand` use, so recovery
      is practised rather than hoped for.
      → **built and A/B'd. It teaches real recovery, it does NOT move the
      frontier, and it turned up the physical reason why.**
      `_ball_spawn_leaning` starts 20-40° tipped about a random horizontal
      axis and still tipping (`MICRODUCK_BALL_LEAN_LO_DEG` / `_HI_DEG` /
      `_RATE`), using the backflip spawn's attitude-aware clearance so nothing
      wedges. Validated before training: 22.5% of resets, tilt 20.1-39.9°,
      **zero** terminating on step 1 — and the shipped brain survives only
      **1/30** of them, which is the gap stated as a number.

      Trained at a 0.25 mix (`teach-find_ball-3c1b2e`) and evaluated on the
      STANDARD spawn distribution (`--env MICRODUCK_SPAWN_FAMILY_PROBS=0.0`,
      or the battery silently tests a different thing):
      **falls 8.0 → 2.7 per 60, in-frame 81.3% → 68.0%.** That is
      (2.7, 68.0) — the old brain's point, reached for the first time by a
      policy trained on the RIGHT camera, and still on the frontier rather
      than outside it. Decide-on not met.

      **The recovery skill is real, and only visible when split by angle.** A
      first pass over the whole 20-40° band read 18% vs 15% and looked like
      nothing; the band is wide enough to average a real effect away:

      | spawned tilt | 5-10° | 10-15° | 15-20° | 20-25° | 30-35° |
      |---|---|---|---|---|---|
      | baseline survival | **87%** | 80% | 50% | 20% | **0%** |
      | lean-trained | 77% | 77% | **67%** | **47%** | **17%** |

      It roughly doubles survival at 20-25° and takes 30-35° from zero, paying
      a little near-upright robustness for it.

- [x] **Why the frontier is real: the fall line is ~20-25° of tilt, not 70°.**
      `FALL_GRAVITY_Z` terminates at 70°, which is nowhere near where this
      robot is actually lost — even the lean-trained brain only saves 17% of
      30-35° leans, and neither arm saves anything past ~40°. So a big body
      turn that produces a 30° lean is already a fall; the policy cannot
      "recover better", it can only not get there. **That is the frontier's
      physical root**, and it explains why three reward sweeps and a spawn
      curriculum all slid along the same line: every one of them was trading
      turn commitment for tilt, because tilt past the fall line is
      unrecoverable by construction.

      Consequence for what to try next: stop looking for a reward or a spawn
      that buys both. The remaining honest options are (a) **pick a point** —
      or (b) **change how the duck turns**, so a big turn does not produce a
      30° lean in the first place. (b) is a gait/technique problem — stepping
      round versus pivoting — and belongs to locomotion, not to this recipe's
      reward. It is also the one thing here that could move the frontier
      rather than slide along it.

- [x] **Point picked and SHIPPED: the lean-trained arm** (`teach-find_ball-3c1b2e`
      → `policies/find_ball/`), with the spawn family on at 0.25.

      Calling it "the safe point" undersold it. It loses in-frame share and
      **wins the deliverable**, on all three eval seeds:

      | | aim-heavy (`72af49`) | **shipped** (`3c1b2e`) |
      |---|---:|---:|
      | in frame | **86 / 79 / 79%** | 63 / 70 / 71% |
      | **handoff fired** | 85 / 77 / 82% | **92 / 87 / 93%** |
      | head yaw \| centred | 15.9 / 14.2 / 11.9° | **9.5 / 11.5 / 9.5°** |
      | falls / 60 | 5 / 8 / 11 | **2 / 3 / 3** |

      In-frame share is a means; the kick handoff is the end, and a brain that
      holds a ball in frame while never squaring up is precisely the failure
      item 1 opened with. 92% handoff at 9.5° of head yaw, under the gate,
      with a third of the falls. Rendered before shipping: 3 of 4 episodes hold
      7.5 s / 2.6 s / 7.4 s aim streaks, the fourth falls on a +173° back start.

      **`eval-find-ball` now pins `MICRODUCK_SPAWN_FAMILY_PROBS=0.0`.** The
      recipe trains with leaning starts; the battery must not fire them, or it
      silently stops being the test every number in this file was taken with.
      Pass `--env MICRODUCK_SPAWN_FAMILY_PROBS=0.25` to measure the training
      distribution on purpose. Locked by a test — this is the same class of
      trap as measuring `centred` in normalized bearing across two cameras.

      **Left in the tree, deliberately inconsistent:** the knobs are the real
      camera's, and `policies/find_ball/policy.onnx` is still the brain
      trained at 48 x 62 — because it is *better in the real-FOV env* than the
      one trained there. Fidelity beats consistency here: modelling the wrong
      camera would have hidden all of the above. One training seed per arm, as
      always.

- [x] **Detector rate is the real compute requirement: ≥ 10 Hz.** The sensor's
      90 fps is ~9x more than this pipeline can consume, so frame rate is not
      where compute should go; the NPU's detection rate is. Swept
      `MICRODUCK_BALL_DETECT_EVERY` against the 50 Hz control loop, 60
      episodes at `--events 0.33`:

      | detector | 50 Hz | 25 Hz (modelled) | 17 Hz | **10 Hz** | 6 Hz | 4 Hz |
      |---|---|---|---|---|---|---|
      | in frame | 65% | 66% | 66% | 64% | 51% | 19% |
      | centred | 62% | 62% | 60% | 60% | **23%** | **5%** |
      | handoff | 80% | 68% | 72% | 75% | **13%** | **0%** |
      | falls | 1 | 1 | 1 | 1 | 4 | 16 |

      → **50 / 25 / 17 / 10 Hz are indistinguishable, and it falls off a cliff
      between 10 and 6** — *as measured then, before the stale-pose fix below.
      The cliff turned out to be an artifact and the requirement is now ~4 Hz;
      this table is what an uncompensated detector costs.*

      **CORRECTED — it IS the tidy pipeline's stale-pose bug, and the same
      cure works.** An earlier revision of this item concluded the opposite.
      That conclusion came from a bug in the compensation, not from the data.

      find_ball holds `det[0]/det[1]` — a bearing measured off the camera at
      capture — unchanged while the head keeps sweeping, so the policy acts on
      a pose the duck has already left. The error tracks the cliff exactly:

      | detector | 25 Hz | 17 Hz | **10 Hz** | **6 Hz** | 4 Hz |
      |---|---|---|---|---|---|
      | stale bearing, mean / p95 | 0.4° / 2.0° | 0.6° / 3.1° | 1.1° / **5.0°** | 5.5° / **20.9°** | 31° / 79° |

      against an aim tolerance of 7°. Rotating the held report by the head's
      own rotation since capture — head encoders and the gyro, no new sensor,
      the same ingredients `brain/tidy.py`'s `stale_fix` used — **removes the
      cliff outright.** Shipped brain, unretrained, 60 episodes at
      `--events 0.33`:

      | rate | handoff / falls, fix OFF | fix ON |
      |---|---|---|
      | 25 Hz | 92% / 2 | 90% / 2 |
      | 10 Hz | 88% / 2 | 88% / 3 |
      | **6 Hz** | **35%** / 3 | **83%** / 5 |
      | **4 Hz** | **2%** / **16** | **80%** / **4** |

      → **the requirement drops from ≥ 10 Hz to ~4 Hz**, for about two points
      of handoff at the nominal rate. Worth taking: the NPU's real throughput
      is the least certain number in the pipeline. `MICRODUCK_BALL_STALE_FIX`
      is ON by default.

      **Two mistakes made getting here, both worth keeping.** The signs were
      guessed first, which made the error ten times worse; they are measured
      now (d(bx)/d(cam yaw) = +1/half_h, d(by)/d(cam pitch) = −1/half_v). Then
      the correction was written onto `det` in place while measuring rotation
      from capture, so every held step re-added the whole accumulated rotation
      — ~2.5× over-correction at 10 Hz and worse below. That version degraded
      every rate and read convincingly as "the compensation does not work",
      which is exactly what produced the wrong conclusion. It corrects from
      the captured bearing now, and a test locks the linear-growth property.

      **Training with it on is NOT needed, and is worse.** Two arms
      (`teach-find_ball-ce44a0` against the buggy version, `38ffd2` against the
      corrected one): the correctly-trained arm scores 77% handoff and **25
      falls / 60** at 25 Hz, against the shipped brain's 90% and 2. It buys
      tracking and spends it on falls — the same frontier every other lever in
      this file lands on. The win is the DEPLOYMENT-side fix applied to a brain
      that never saw it, which is precisely tidy's story: a hand-written daemon
      can do this, and a learned policy need not be retrained for it.

      **The trap in this table is worth its own line:** `found` stays 97% at
      EVERY rate, including the 4 Hz column where the handoff never fires and
      the duck falls in 16 of 60 episodes. "Ever saw the ball" is satisfied
      eventually by any detector. Only the AIMING columns see the collapse —
      one more reason the battery grew them.

### 3. Search speed itself — the actual ask

### Time spent LOST — the strongest signal found so far, and it says ship the blind recipe

Chasing the one column that tracked performance in the sweep audit above.
"Steps spent lost" conflates two failures, so split them: **search** (steps
before the ball is EVER seen) and **tracking** (steps lost after the first
sight — dropping a ball you already had). 40 episodes, no prior:

With a STATIC ball, tracking is essentially solved for every arm (0.0-3.7%
lost after first sight) — so with events off, "time lost" is **search, almost
entirely**. With the recipe's own events (0.33) the two separate:

| arm | search steps | tracking lost | drops/ep | regain med / p90 | never regained |
|---|---:|---:|---:|---:|---:|
| shipped s5 (cloud) | 3689 | 26.0% | 0.82 | 0.76s / 4.80s | 12 |
| control | 3013 | 18.7% | 0.82 | 0.52s / 2.32s | 9 |
| fix1 `body_aimed` 2.0 | 1943 | 27.0% | 0.82 | 1.41s / 3.38s | 15 |
| fix2 (shipped) | 2313 | 17.2% | 0.95 | **0.27s** / 0.71s | 14 |
| fix3 turn ungated | 2010 | 32.7% | 0.72 | 0.49s / 3.66s | 17 |
| **blind-trained** | **735** | **8.5%** | 1.07 | 0.45s / 1.11s | **3** |

The blind-trained arm searches 3x faster than the shipped brain and loses the
ball half as often, and it is the cleanest A/B on this page: it is the fix2
recipe with **one knob changed**, `MICRODUCK_BALL_PRIOR_PROB` 0.7 -> 0.0,
same curriculum, same 8M steps, same seed.

**Head to head in the deployment regime** (60 episodes, events 0.33):

| brain | prior at eval | found | in frame | centred | head yaw | handoff | falls | worst t_first |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fix2 (shipped) | none | 97% | 68% | 63% | 19.4° | 73% | 1 | 6.50 s |
| fix2 (shipped) | on | 95% | 66% | 62% | 19.1° | 68% | 1 | 7.72 s |
| blind-trained | none | **100%** | **83%** | **74%** | **15.3°** | **77%** | 1 | 5.78 s |
| **blind-trained** | **on** | **100%** | **85%** | **78%** | **14.2°** | **78%** | 1 | **1.16 s** |

It dominates every column in both prior conditions, back-bucket found goes
94% -> 100%, and the side bucket's worst-case time to first sight collapses
from **6.30 s to 0.58 s** — that is the wrong-side tail this section is about,
essentially gone.

**This overturns the previous section's recommendation, and the error is worth
naming: those conclusions were measured with `--events 0`.** At events 0 the
shipped brain looked like the better aimer (93% handoff vs 68%); with the
events the recipe actually trains on, the ordering flips (73% vs 77%). That is
the second time in this file the static-ball battery has pointed the wrong way.
**Judge find_ball at `--events 0.33`.** Full stop.

Rendered before believing it (`/tmp/rr-blind`): 4/4 episodes upright, in frame
86-98%. From a ball at **p+173°** — directly behind — it turns ~180° by
**1.44 s**, holds the bearing within a few degrees for the rest of the episode,
and re-acquires cleanly after a mid-episode ball event. The shipped fix2 brain
FELL on that same seed.

**Why, and it is the project's own lesson:** with `PRIOR_PROB=0.7`, 70% of
training episodes started by handing the duck the answer, so a real
search-from-nothing was under-practised — the same shape as the headstand
lesson in `AGENTS.md` ("if rollouts never contain the skill you are paying
for, change the world, not the pay"), except here the world was too *easy*
rather than too hard. Removing the prior makes every episode a real search.
Note what it does NOT do: the blind-trained brain still uses a prior perfectly
well when handed one — better than without — so this is about what it
PRACTISED, not what it can read.

- [ ] **Ship it?** The case is strong and one-knob, but it rests on a **single
      training seed** (the lab pins `--seed 0`), and it would mean replacing
      `policies/find_ball/` a second time and flipping a default this file
      currently justifies at 0.7. Effect sizes are far larger than the
      run-to-run variance seen elsewhere here (in frame 68 -> 83, worst
      t_first 7.72 s -> 1.16 s), which is the argument for; one seed is the
      argument for a confirmation run first.
      → **decide on:** a second blind-trained chain reproducing in-frame ≥ 80%
      and back-bucket found ≥ 95% at `--events 0.33`.


**Why this section now has a mechanism, not just a hunch (measured
2026-09-03).** "Which way should it look first?" has a fixed answer: with
nothing known the belief slot is seeded `_BALL_NO_PRIOR_MEM = +pi/2`, "sweep
left first". Splitting the battery by which side the ball was really on shows
what that convention costs, on the shipped brain, blind episodes, balls
outside the initial view (n=66):

| ball side | found | t_first mean | t_first median |
|---|---:|---:|---:|
| LEFT (with the convention) | 100% | **0.35 s** | 0.18 s |
| RIGHT (against it) | 97% | **2.48 s** | 2.75 s |

A **7x mean penalty** for being on the wrong side; 17 of 40 right-side balls
took over 2 s against 2 of 40 on the left. Note what this is NOT: it is not
fixable by choosing the side better. Spawns are uniform in bearing, so no
observable information distinguishes left from right, and the convention
exists precisely because a symmetric obs leaves the mean action with nothing
to learn (AGENTS.md). The wrong guess is unavoidable half the time; what is
tunable is what it COSTS, which is ~2/3 of a sweep period. That makes
SCAN_PERIOD the dominant term in mean time-to-first-sight for half of all
blind episodes, and it is why the item below is now the most valuable one
here rather than a nice-to-have.

**Unexplained, and probably the same bug as the prior item below:** force the
prior ON and the asymmetry REVERSES — left 2.29 s / 91% found, right 1.61 s /
97%. If the convention were the whole story, a seeded belief should make the
two sides symmetric, not swap them. Worth understanding before tuning either.

- [ ] **Faster sweep.** `MICRODUCK_BALL_SCAN_PERIOD=2.5` (from 4.0) fine-tune.
      → **decide on:** median time-to-first-sight on side and back, weighed
      against falls and centred share. A faster sweep that loses the ball on
      the way past is not faster. **Split the result by ball side** (above):
      the number this is really moving is the 2.48 s wrong-side mean, and a
      whole-battery median will hide it behind the 0.35 s right-side half.
- [x] **Is the prior worth producing? NO — and an ORACLE prior is worse than
      none, which rules out "the prior is just too noisy".** 60 episodes each
      on the shipped brain:

      | belief seeding | found | t_first med | in frame | centred | falls |
      |---|---:|---:|---:|---:|---:|
      | **blind (convention only)** | **98%** | 0.36 s | **85%** | **84%** | **1** |
      | prior, ORACLE (noise 0) | 93% | 0.36 s | 74% | 71% | 2 |
      | prior, noise 0.3 rad | 92% | 0.24 s | 73% | 70% | 2 |
      | prior, noise 0.6 rad (the recipe) | 93% | 0.22 s | 75% | 72% | 2 |
      | prior, noise 1.2 rad | 97% | 0.34 s | 77% | 74% | 2 |

      → **decide-on met: the daemon does not need to synthesize a prior**, and
      the slot can carry the sweep convention alone. Note the shape: prior
      ACCURACY is irrelevant (93 / 92 / 93 / 97% found across a 4x noise
      range) while prior PRESENCE costs ~11 points of in-frame share. A belief
      pointing exactly at the ball still loses. So this is not a bad estimate,
      it is the mechanism: a seeded belief buys a marginally faster first
      sight (0.22 s vs 0.36 s) and then keeps the ball worse.

      **Why, from probing the exported ONNX directly** (feed one obs, vary one
      slot, read the action — the method that found the original
      symmetric-obs bug). The belief slot is NOT ignored and NOT sign-flipped:
      with the ball lost, head yaw responds monotonically and correctly across
      the whole range. But the response is badly **asymmetric in magnitude**,
      and that is what produces every oddity here:

      | belief slot | −1.0 | −0.45 | −0.25 | **0.0** | +0.15 | +0.25 | +0.45 | +1.0 |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|
      | head-yaw action | −3.75 | −2.10 | −1.45 | **−0.39** | +0.23 | +0.57 | +1.09 | +2.13 |

      A rightward belief commands a **2.6× stronger** turn than the mirrored
      leftward one, and a NEUTRAL belief already commands −0.39 (rightward).
      Two consequences, both of which had been observed and unexplained:

      - The no-prior convention (+0.15) sits in the flattest part of that
        curve — it produces a limp +0.23 — which is why blind episodes find
        left-side balls in 0.35 s and right-side ones in 2.48 s (section 3's
        table above).
      - Force a prior on and the asymmetry **reverses** (left 2.29 s, right
        1.61 s), because now the slot carries a real bearing and rightward
        ones pull much harder. That was the unexplained reversal noted above;
        this is the explanation.

      **RETRACTED — that 2.6× is an artifact of the probe, not a property of
      the policy.** `vecnormalize` says slot 54 has **std 0.11** against a
      nominal range of ±1, so sweeping the slot to ±1 probes the network at
      **±8 sigma**: inputs it has effectively never seen. Three checks, all
      negative:

      - The baked normalizer is symmetric — `mean[54]` is -0.004..+0.011
        across all five arms, and ±1 maps to ±8.2..8.8 sigma with a
        |left|/|right| ratio of 0.95-1.02. The normalizer is not doing it.
      - Within ±2 sigma the head-yaw response is monotonic with a **correctly
        signed positive slope in every arm** (+0.31 to +0.71 per sigma). What
        looked like asymmetry at the extremes is a large constant offset,
        and that offset is just *where the scan clock happens to be pointing*
        — the probe had frozen obs[59]/obs[60] at one phase.
      - Swept across a full clock cycle instead, the belief slot shifts the
        sweep's **centre** while leaving its **span** roughly unchanged, which
        is exactly what a belief is supposed to do.

      Measured behaviourally instead — head yaw over 40 blind episodes on
      every step the ball is not seen — the arms DO sweep lopsidedly, and it
      turns out not to matter:

      | arm | steps lost | span | L/R balance | found |
      |---|---:|---:|---:|---:|
      | shipped s5 (cloud) | 4737 | 192° | 0.87 | 82% |
      | control | 2749 | 177° | **0.99** | 95% |
      | fix1 `body_aimed` 2.0 | 2255 | 90° | 0.44 | 98% |
      | **fix2 (shipped)** | 2173 | 181° | 0.66 | **100%** |
      | fix3 turn ungated | 2974 | 153° | 2.43 | 75% |
      | blind-trained | **769** | 159° | 0.62 | **100%** |

      Balance ranges 0.44-2.43 and predicts nothing: the control has the most
      symmetric sweep of any arm (0.99) and is not the best finder, while fix2
      is lopsided (0.66) and finds everything. **The column that tracks
      performance is steps spent LOST** — 4.8% of steps for the blind-trained
      arm against 30% for the old cloud export — which is re-acquisition and
      tracking, not search symmetry. Sweep span does not explain it either
      (blind-trained sweeps NARROWER than fix2 and finds better).

      So: the belief slot is read correctly, the sweep asymmetry is real but
      inert, and the only left/right effect with a measured cost is the plain
      one in section 3 — the convention decides which way it looks FIRST, so
      left balls come in at 0.18 s and right balls wait for the sweep.

      Method note worth keeping, since this cost two wrong hypotheses: before
      reading anything off a slot sweep, check that slot's std in
      `vecnormalize.pkl` and stay inside ±2 sigma, and vary the scan clock
      rather than freezing it — a frozen clock turns "what phase am I at" into
      a fake constant bias.

      Caveat on method: sweeping slot 54 while holding bx and head yaw fixed
      leaves the manifold for the SEEN case (while the ball is visible the
      slot carries the body-frame bearing, which is tied to bx and head yaw by
      geometry), so only the ball-LOST rows above are trustworthy. They are
      the rows that matter — the belief exists for when the ball is lost.

      **CORRECTION, from the blind-TRAINED arm this queued**
      (`MICRODUCK_BALL_PRIOR_PROB=0` through the whole curriculum, run
      `teach-find_ball-31f14b`). Everything above is measured on a brain that
      saw a prior in 70% of its training episodes, and **it does not
      generalize**: fed a prior, the blind-trained brain gets BETTER, not
      worse. The effect is an interaction between how a brain trained and what
      it is fed at run time — not a fact about priors. 60 episodes per cell:

      | | evaluated BLIND | evaluated WITH prior |
      |---|---|---|
      | **trained with prior** (the shipped brain) | 98% found · 85% in frame · 84% centred · **93% handoff** · 1 fall | 93% · 75% · 72% · 75% handoff · 2 falls |
      | **trained blind** | 100% · 94% · 86% · 68% handoff · 0 falls | **100% · 95% · 92%** · 73% handoff · **0 falls** (worst-case first sight **0.98 s**, against 6.30 s fed nothing) |

      Neither arm dominates, and the two halves of the task come apart:

      - **Best FINDER: blind-trained, fed a prior** — 95% in frame, 92%
        centred, no falls, and a worst case of 0.98 s where every other cell
        has a 6-8 s tail. Training without the prior makes a brain that
        searches better and then *uses* a prior well when handed one.
      - **Best AIMER: the shipped prior-trained brain, fed nothing** — 93% of
        episodes reach the kick handoff, against 68-75% in all three other
        cells. Aiming is the deliverable, so this is the cell that matters
        most today.

      So the section's decide-on ("does the daemon need to synthesize a
      prior?") splits by which brain ships. **For the brain in
      `policies/find_ball/`, no — feed it the sweep convention and it aims
      better** (93% vs 75%). The stronger claim, that a prior is dead weight
      in general, is WRONG and the row above is the counterexample.

      `MICRODUCK_BALL_PRIOR_PROB` is left at its 0.7 default: nothing here
      justifies changing what the recipe trains, and the two candidate
      recipes trade finding against aiming rather than one beating the other.
      The experiment worth running next is the obvious missing cell —
      blind-trained, then judged on the handoff after more steps — since the
      only thing it is clearly worse at is the one thing this behavior is for.

### 4. The soccer handoff — the demo that proves the point

`find_ball` aims; `ball_kick_right.onnx` kicks but is blind. One clip of find →
square up → kick is the whole argument for this behavior.

- [x] **Handoff condition — done.** It lives on the behavior
      (`Behavior.handoff_fn`), so `render-rollout --handoff` and the lab's
      showcase duck ask one implementation instead of two copies kept in step
      by a comment. Fires on: detector reports the ball centred (|bx|, |by| <
      0.25) **and** the head straight ahead (|head_yaw| < 0.25 rad), held
      0.5 s. Head centred plus head aligned means the *body* is pointing at
      the ball, and both halves are detector output plus joint encoders — the
      daemon can run the same test, no privileged state. `handoff_policy`
      names the kick; `handoff_recenter=False` keeps the lab from spinning
      away the turn the duck just made.
- [x] **Rendered it, and it found the real problem.** The gate is correct and
      the policy does not meet it: an episode with the ball centred 98% of the
      time never handed off, because the duck aims with 21° of head yaw and
      leaves its body put (item 1). The two episodes that *did* fire were
      near-frontal starts, and the kick toppled the duck within 0.4 s in
      both — expected, since it was handed a ball 0.9–1.3 m away.
- [x] **Re-rendered after item 1 — the handoff fires from a side start, and
      the kick does not connect.** Both halves of the prediction confirmed, on
      the `body_aimed` 2.0 arm with
      `--handoff ../microduck/policies/ball_kick_right.onnx`:
      **all 4 of 4 episodes fired** (t = 0.72, 0.72, 1.30, 2.32 s), including
      the **+70° and +98° side starts** that never fired before — against 2 of
      4 near-frontal-only on the shipped brain. And all 4 then **fell within
      ~0.5 s of the switch**. The same 4 seeds rendered with no handoff fell
      **zero** times, so this is the kick toppling a duck handed a ball at
      0.8-1.9 m, not the aimer failing — exactly the "asserts *aimed*, never
      *in range*" gap. The aiming half of the soccer demo is done; the clip
      needs the approach behavior below.
- [ ] **(Stretch) Approach.** Walking to the ball is a `forward_cmd` locomotion
      task steered by the bearing slot — a different recipe, not a stage of
      this one. Range needs the detector to report box size (distance ≈
      focal × real diameter / box height), which the sim's fake detector does
      not yet produce; adding it is a prerequisite, not an afterthought.

### 5. Lab ergonomics, while the above trains

- [ ] **Teach a `find_ball` chain from the browser** (`/teach`, or the 🎓 panel)
      and use the `watch-training` skill mid-stage.
      → **decide on:** do the three stage descriptions read correctly in the
      viewer's stage inspector, and does the ball marker follow a teleport
      without visible lag at 25 Hz? The stage narration and the marker stream
      have only ever been exercised headlessly.

---

## Later / parked

- **Port `find_ball` to an mjlab cfg** and retrain on GPU in upstream
  `microduck_rl`. That stack, not this one, is the sim2real recipe. Blocked on
  the items above: there is no point porting a recipe whose back-bucket
  behavior is still moving.
- **What the fake detector cannot produce.** The env fakes the detector by
  projecting a point through the MJCF camera: FOV bounds plus a range check,
  and nothing else. Three things a real one does are therefore untested, and
  they are worth doing in this order rather than as one "real detector" job:

  1. **Box size, hence RANGE.** Cheapest, and it unblocks queued work rather
     than opening new questions: the handoff gate asserts *aimed* and never
     *in range*, which is exactly why the kick whiffs in every soccer render,
     and the approach behavior (item 4's stretch) cannot start without it.
     `distance ~= focal x real diameter / box height`.
  2. **Occlusion — i.e. WALLS.** Today "not seen" ALWAYS means "not inside my
     camera cone", so the belief's dead-reckoning is never wrong about
     anything except direction. Put an occluder in the scene and that
     assumption breaks: not-seen can mean *hidden*, and the right response is
     to look around the obstacle rather than sweep past it — the memory slot
     would have to represent "hidden there", not just "it went that way".
     That is a qualitatively harder search than the one this recipe solves,
     and it is the one a robot in a real room faces. Needs geometry in the
     scene AND a ray test in `_ball_sense`; note the lab arena on the
     `robot-lab-sim-roadmap` branch may bring the geometry along anyway, in
     which case the deployed duck meets walls its training never had.
     (Worth being clear about what walls are NOT: they are not a better
     search-direction cue. Spawns are uniform in bearing, so no wall makes one
     side likelier — see section 3, where the wrong-side cost is real but the
     fix is a faster sweep, not a smarter choice.)
  3. **False positives.** A real detector reports something orange that is not
     the ball. The policy currently trusts `seen` completely, and nothing in
     training has ever lied to it.

  The fully honest version — render the head camera and run the actual
  `duck_detect` ONNX — subsumes all three and is far slower per step. Worth it
  only once the behavior is otherwise settled.
- **Other things to find.** The slot layout is not ball-specific: the same
  four head slots and scan clock would serve "find the other duck" (upstream
  wants precise bearing for gaze and following) or "find the charging dock".
  A second target is a cheap test of whether the recipe generalizes or whether
  it memorized a ball-sized blob.
