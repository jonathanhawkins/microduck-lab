# find_ball — prototype brain (CPU, this harness)

`policy.onnx` is the stage-5 export of the `find_ball` curriculum trained in
this harness (xml actuator, no DR — a prototype, not a robot policy; see
"sim2real honesty" in `../../AGENTS.md`). 61-obs / 14-action, normalizer
baked in, deterministic mean. It expects the head slots and the scan clock
filled as documented in `../../README.md` ("find_ball").

Lineage: s1 (front 140°, 1M) → s1c (+ belief slot & scan clock, 1M) → s2
(anywhere, moving, 2M) → s3 (rolling, 1M) → s4 (+ raised-cosine facing, 2M)
→ s5 (+ turn_to_belief, no up-band coverage, 1M). 8 envs, seeds 1-7.

`uv run eval-find-ball policies/find_ball/policy.onnx` (40 static-ball
episodes × 8 s, deterministic, randomizers off), measured 2026-09-03:

| ball starts | found | time to first sight (median / max) | in frame | falls |
|---|---:|---:|---:|---:|
| front (< 45°) | 100% | 0.03 s / 0.16 s | 100% | 0 |
| side (45–135°) | 85% | 0.60 s / 2.46 s | 77% | 0 |
| back (> 135°) | 60% | 0.94 s / 6.98 s | 34% | 2 |
| blind (no prior), all | 88% | 0.64 s / 7.06 s | 74% | 1 |

Known gap: **it aims with its head, not its body** — it will hold the ball
dead centre in frame with ~21° of head yaw and never square up, so the kick
handoff (which requires the head straight ahead, i.e. the body pointing at the
ball) does not fire on a side start. It also falls on some back starts while
turning (2/40 in the battery,
2/4 in a full-circle render) — `turn_to_belief` has had only 1M steps. The
fall-free choice is the s4 recipe (no turn term), which found 30% of back
balls; retrain longer before putting either near a real duck.

To seat it in the lab as a run: `cp -r policies/find_ball runs/find_ball`
(the palette lists `run:find_ball`); to look at it:
`uv run render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb`.
