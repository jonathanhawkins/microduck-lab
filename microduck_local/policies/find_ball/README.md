# find_ball — prototype brain (CPU, this harness)

`policy.onnx` is the `find_ball` curriculum trained end to end in this harness
(xml actuator, no DR — a prototype, not a robot policy; see "sim2real honesty"
in `../../AGENTS.md`). 61-obs / 14-action, normalizer baked in, deterministic
mean. It expects the head slots and the scan clock filled as documented in
`../../README.md` ("find_ball").

Lineage: the declared 3-stage curriculum straight through, 8M steps, seed 0
(run `teach-find_ball-3c1b2e`), then a 3M-step **fine-tune with the detector's
stale-pose correction on** (`teach-find_ball-f31a4f`). Launched through the
lab's `/teach`; ~10 minutes total on an M-series Mac. Trained under the real
camera (60° × 116° portrait), with aim tolerances in **degrees**, 25% leaning
spawns, and `MICRODUCK_BALL_STALE_FIX`. The SB3 checkpoint is at
`runs/teach-find_ball-f31a4f/` on the machine that trained it and is not in git.

The fine-tune is worth a note, because it is the only change all session that
improved aim **and** falls together instead of trading them. Applying the stale
fix at deployment to the un-retrained brain already removed the detector-rate
cliff; warm-starting from that brain and letting it adapt to the corrected
signal added the rest — 100% found in every bucket, +11 points of in-frame
share, and the falls unchanged. Starting a FRESH chain with the fix on does not
do this (77% handoff, 25 falls / 60): the low-fall behavior is what the warm
start preserves.

`uv run eval-find-ball policies/find_ball/policy.onnx --episodes 60 --events 0.33`
— **judge this behavior with ball events on**, which is what the recipe trains
and `render-rollout` runs at; the static-ball battery has twice pointed the
wrong way. Measured 2026-09-04:

| ball starts | found | t_first med | in frame | head yaw \| centred | **handoff** | falls |
|---|---:|---:|---:|---:|---:|---:|
| front (< 45°) | 100% | 0.02 s | 77% | 7.3° | **100%** | 0 |
| side (45–135°) | 100% | 0.22 s | 75% | 10.3° | 89% | 1 |
| back (> 135°) | 100% | 0.81 s | 71% | 10.6° | 94% | 0 |
| **all** | **100%** | **0.23 s** | **74%** | **9.6°** | **93%** | **1** |

It holds up as the detector slows, which is the point of the stale fix —
handoff 97% at 10 Hz, 92% at 6 Hz, 90% at 4 Hz, against 88 / 83 / 80 for the
un-retrained brain and 88 / 35 / 2 with no correction at all.

`handoff` is the deliverable: the share of episodes that reach the state a
ball-blind kick wants handed to it (ball centred **and** head straight, held
0.5 s). 92%, with head yaw 9.5° against the gate's 14°.

## Why this export

`docs/roadmap.md` establishes a **falls/aim frontier** for this recipe: for a
long time every reward lever traded one against the other, because the duck's
effective fall line is ~20–25° of tilt and a big turn that produces a 30° lean
is already a fall. Two things got off that line, and neither was a reward:

| | aim-heavy arm | low-falls arm | + stale fix | **shipped** |
|---|---:|---:|---:|---:|
| in frame | 86 / 79 / 79% | 63 / 70 / 71% | 63% | **74%** |
| **handoff** | 85 / 77 / 82% | 92 / 87 / 93% | 90% | **93%** |
| falls / 60 | 5 / 8 / 11 | 2 / 3 / 3 | 2 | **1** |

The **leaning-spawn curriculum** bought the low-fall point (a physics change,
not a pay), and the **stale-pose correction plus a warm-started fine-tune**
bought back the tracking the low-fall point had cost. In-frame share is a
means; the kick handoff is the end, and a brain that holds a ball in frame
while never squaring up is exactly the failure this behavior was built to fix.

Known gaps: the back bucket is still the weak one (88% found, 81% handoff, and
both falls), and the leaning-spawn curriculum that bought the stability teaches
recovery only up to ~25° of tilt — past that the robot is lost whatever it has
practised. Moving the frontier rather than sliding along it needs a change to
*how the duck turns* (stepping round rather than pivoting into a lean), which
is a locomotion problem, not this recipe's reward.

## `policy_s4_pre_turn_term.onnx` — still the turn-term A/B baseline

The stage-4 cloud export: no `turn_to_belief`, and the old three-band gaze
coverage. **Keep it.** Its A/B ("does the turn term buy the turn, or only the
falls?") is still open in `docs/roadmap.md`, and the control arm re-scoped that
question rather than answering it. Its numbers were taken under the placeholder
camera and normalized tolerances, so they are not comparable to the table above.

To seat this brain in the lab as a run: `cp -r policies/find_ball runs/find_ball`
and drop `run:find_ball` from the 🧠 palette. To look at it:
`uv run render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb`.
