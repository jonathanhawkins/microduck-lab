# find_ball — prototype brain (CPU, this harness)

`policy.onnx` is the `find_ball` curriculum trained end to end in this harness
(xml actuator, no DR — a prototype, not a robot policy; see "sim2real honesty"
in `../../AGENTS.md`). 61-obs / 14-action, normalizer baked in, deterministic
mean. It expects the head slots and the scan clock filled as documented in
`../../README.md` ("find_ball").

Lineage: the declared 3-stage curriculum straight through, 8M steps, 32 envs,
seed 0, launched through the lab's `/teach` (the only path that chains stages).
Run `teach-find_ball-3c1b2e`; ~8 minutes on an M-series Mac. Trained under the
**real camera** (60° × 116° portrait, from the 2.9 mm lens datasheet), with the
aim tolerances in **degrees**, and with **25% leaning spawns** — the three
things `docs/roadmap.md` changed after the placeholder-camera era. The SB3
checkpoint is at `runs/teach-find_ball-3c1b2e-s3/` on the machine that trained
it and is not in git (`runs/` is gitignored).

`uv run eval-find-ball policies/find_ball/policy.onnx --episodes 60 --events 0.33`
— **judge this behavior with ball events on**, which is what the recipe trains
and `render-rollout` runs at; the static-ball battery has twice pointed the
wrong way. Measured 2026-09-04:

| ball starts | found | t_first med | in frame | head yaw \| centred | **handoff** | falls |
|---|---:|---:|---:|---:|---:|---:|
| front (< 45°) | 94% | 0.02 s | 63% | 7.6° | 88% | 0 |
| side (45–135°) | 100% | 0.68 s | 68% | 9.2° | **100%** | 0 |
| back (> 135°) | 88% | 0.56 s | 56% | 12.3° | 81% | 2 |
| **all** | **95%** | **0.44 s** | **63%** | **9.5°** | **92%** | **2** |

`handoff` is the deliverable: the share of episodes that reach the state a
ball-blind kick wants handed to it (ball centred **and** head straight, held
0.5 s). 92%, with head yaw 9.5° against the gate's 14°.

## Why this export and not the aim-heavy one

`docs/roadmap.md` establishes a **falls/aim frontier** for this recipe: every
reward lever tried trades one against the other, because the duck's effective
fall line is ~20–25° of tilt and a big body turn that produces a 30° lean is
already a fall. Two points were worth shipping, and this is the one chosen:

| | aim-heavy (`teach-find_ball-72af49`) | **shipped** (`3c1b2e`) |
|---|---:|---:|
| in frame | **86 / 79 / 79%** | 63 / 70 / 71% |
| **handoff fired** | 85 / 77 / 82% | **92 / 87 / 93%** |
| head yaw \| centred | 15.9 / 14.2 / 11.9° | **9.5 / 11.5 / 9.5°** |
| **falls / 60** | 5 / 8 / 11 | **2 / 3 / 3** |

(three eval seeds each, `--events 0.33`)

The aim-heavy arm holds the ball in frame more of the time. This one **aims
better when it matters and falls a third as often** — and holding a ball in
frame while never squaring up is exactly the failure this behavior was built
to fix. In-frame share is a means; the handoff is the end.

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
