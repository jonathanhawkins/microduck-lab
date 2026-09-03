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

### 0. Verify the branch on real hardware (~15 min, do this first)

- [ ] **The test suite is green.** `cd microduck_local && uv run --with pytest pytest tests/`
      → 12 golden bit-parity tests (`test_step_perf_parity`,
      `test_bam_perf_parity`, one `test_symmetry` drift case) failed in the
      cloud container **and failed identically at the base commit there**, so
      they are environment drift (a Mesa/BLAS-level float difference), not
      this branch. On the Mac they should pass. **If they fail here too, stop
      and bisect — that is a real regression and nothing below matters.**
- [ ] **The battery reproduces.** `uv run eval-find-ball policies/find_ball/policy.onnx`
      → expect roughly front 100% found (median 0.03 s), side 85% (0.60 s),
      back 60% (0.94 s), 2 falls / 40. Cross-machine agreement is the check;
      the exact hex-level numbers will differ.
- [ ] **Look at it.** `uv run render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb --episodes 4`
      → watch the mp4. Nobody has seen this brain at full frame rate yet.
      The ball renders orange, the gaze dot cyan when the ball is in frame.
- [ ] **It works in the lab.** `cp -r microduck_local/policies/find_ball microduck_local/runs/find_ball`,
      start `duck-lab` + the viewer, drop `run:find_ball` on a duck.
      → the ball should draw next to the duck and the duck's label should show
      the spawn note (`↻ ball +143° 1.3m prior`). This is the first time the
      viewer's ball-drawing path runs against a browser.

### 1. The open problem: balls that start directly behind

The head reaches ±170°, so the *camera* can see behind; what is missing is the
body committing to the turn. Stage 5 added `turn_to_belief` (signed yaw-rate
pay aimed by the belief slot) and back-bucket found rate went 30% → 60% — but
it also introduced the only falls in the chain (2/40 in the battery, 2 of 4
back starts in a render, at 0.98 s and 2.48 s). That term has had exactly 1M
steps. **This is the highest-value item on the list.**

- [ ] **Just train it longer.** `uv run train-behavior find_ball --init-from runs/find_ball --steps 3_000_000 --run-name fb-long`
      (full-circle knobs: `MICRODUCK_BALL_BEARING_MAX=3.1416 MICRODUCK_BALL_EVENT_RATE=0.33`)
      → **decide on:** back-bucket found ≥ 80% **and** 0 falls / 40. Falls are
      the veto: a duck that finds the ball by falling over has not found it.
- [ ] **A/B the turn term** at weight 0 vs 1.0, same warm start, same seed, 2M
      steps each (`--weights-json` or the viewer's sliders).
      → **decide on:** does the term buy the turn, or only the falls? If back
      found rate is the same at weight 0, the term is not what fixed it and the
      credit belongs to the coverage-band change that shipped alongside it.
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
- [ ] **FOV sensitivity** (no training — run the battery on the existing brain
      with `--env MICRODUCK_BALL_HFOV_DEG=40` and `=56`).
      → **decide on:** if found rates swing hard, the daemon's real IMX219
      intrinsics have to be measured before this goes near hardware. Upstream
      flags them as placeholders (`microduck/docs/ideas/autonomous_behavior.md`).

### 3. Search speed itself — the actual ask

- [ ] **Faster sweep.** `MICRODUCK_BALL_SCAN_PERIOD=2.5` (from 4.0) fine-tune.
      → **decide on:** median time-to-first-sight on side and back, weighed
      against falls and centred share. A faster sweep that loses the ball on
      the way past is not faster.
- [ ] **Is the prior worth producing?** `uv run eval-find-ball <policy> --prior 0`
      vs `--prior 1` on the same brain.
      → **decide on:** in the cloud runs the prior barely helped (blind was
      *better* on the side bucket: 100% vs 85%). If that holds on a
      longer-trained brain, the daemon does not need to synthesize one and the
      slot can carry the sweep convention alone.

### 4. The soccer handoff — the demo that proves the point

`find_ball` aims; `ball_kick_right.onnx` kicks but is blind. One clip of find →
square up → kick is the whole argument for this behavior.

- [ ] **Add a handoff condition** to `render_rollout.handoff_due` and
      `viz_server.Duck._handoff_due`: ball centred (|bx|, |by| < 0.25) and body
      squared (|psi| < ~0.2 rad) held for ~0.5 s. Mirror the two
      implementations, as the backflip's rotation handoff already does.
- [ ] **Render it.** `uv run render-rollout --policy policies/find_ball/policy.onnx --handoff ../microduck/policies/ball_kick_right.onnx --out /tmp/rr-soccer`
      → **decide on:** does the handoff fire from a side or back start, and
      does the kick connect? The kick policy expects the ball ~9 cm in front
      of the kicking foot, so "squared up" may need to become "squared up and
      the ball is close", which is an approach behavior we do not have yet.
- [ ] **(Stretch) Approach.** Walking to the ball is a `forward_cmd` locomotion
      task steered by the bearing slot — a different recipe, not a stage of
      this one. Scope it only after the handoff render says what is missing.

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
- **A real detector in the loop.** The env fakes the detector by projecting a
  point through the MJCF camera. The honest version renders the head camera and
  runs the actual `duck_detect` ONNX — far slower per step, but it is the only
  way to find out what the brain does with a *wrong* box rather than a noisy
  one. Worth it only once the behavior is otherwise settled.
- **Other things to find.** The slot layout is not ball-specific: the same
  four head slots and scan clock would serve "find the other duck" (upstream
  wants precise bearing for gaze and following) or "find the charging dock".
  A second target is a cheap test of whether the recipe generalizes or whether
  it memorized a ball-sized blob.
