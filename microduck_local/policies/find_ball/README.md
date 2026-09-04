# find_ball — prototype brain (CPU, this harness)

`policy.onnx` is the `find_ball` curriculum trained end to end in this harness
(xml actuator, no DR — a prototype, not a robot policy; see "sim2real honesty"
in `../../AGENTS.md`). 61-obs / 14-action, normalizer baked in, deterministic
mean. It expects the head slots and the scan clock filled as documented in
`../../README.md` ("find_ball").

Lineage: the declared 3-stage curriculum straight through, 8M steps, 32 envs,
seed 0, launched through the lab's `/teach` (the only path that chains stages).
Run `teach-find_ball-c60e89`; ~7 minutes on an M-series Mac. The SB3 checkpoint
it was exported from is at `runs/teach-find_ball-c60e89-s3/` on the machine
that trained it and is **not** in git (`runs/` is gitignored) — so a fine-tune
can warm-start from it there, which the previous cloud-trained export could not
offer anywhere.

This export is the **fix 2** arm of `docs/roadmap.md` item 1: identical recipe
except `_BALL_FACE_TIGHT_STD` 0.4 → 0.2, which is now the shipped default.

`uv run eval-find-ball policies/find_ball/policy.onnx --episodes 40`
(40 static-ball episodes × 8 s, deterministic, randomizers off), 2026-09-03:

| ball starts | found | time to first sight (median / max) | in frame | head yaw \| centred | handoff | falls |
|---|---:|---:|---:|---:|---:|---:|
| front (< 45°) | 100% | 0.03 s / 0.06 s | 100% | 7.5° | 100% | 0 |
| side (45–135°) | 90% | 0.49 s / 7.16 s | 75% | 16.0° | 75% | 0 |
| back (> 135°) | 100% | 1.92 s / 7.16 s | 66% | 18.4° | 70% | 0 |
| **all** | **95%** | **0.23 s** | **79%** | **14.4°** | **80%** | **0** |

**Judge falls with ball events on**, which is what the recipe trains and
`render-rollout` runs at — the static-ball battery and the events-on battery
disagree about which policy is safest.
`uv run eval-find-ball policies/find_ball/policy.onnx --episodes 60 --events 0.33`:
found 97% all / 94% back, in frame 66%, head yaw 18.6°, handoff 68%,
**1 fall / 60**.

Against the previous shipped export (the 6-stage cloud chain, `--events 0.33`,
60 episodes): found 97% vs 85%, handoff **68% vs 15%**, head yaw **18.6° vs
44.8°**, falls 1 vs 2. Two separate things produced that gap and it is worth
keeping them apart: most of it was **under-training** — the old export was not
a converged instance of its own recipe, and retraining the *unchanged* terms
took head yaw to 25° and handoff to 38% on its own — and the rest is fix 2.

Known gap: it still aims partly with its neck. Head yaw is 14.4° averaged over
centred steps on a static ball, right at the handoff gate's 14°, and 18.6° with
events on; the handoff fires on 68–80% of episodes rather than all of them.
`body_aimed` (in the recipe at weight 0) is the stronger lever — it takes head
yaw to 8° and the handoff to 83% — and it triples the falls, which is why it is
not priced. See `docs/roadmap.md` item 1 for the full A/B.

## `policy_s4_pre_turn_term.onnx` — still the turn-term A/B baseline

The stage-4 cloud export: no `turn_to_belief`, and the old three-band gaze
coverage. **Keep it.** Its A/B ("does the turn term buy the turn, or only the
falls?") is still open in `docs/roadmap.md` — and the control arm re-scoped the
question rather than answering it, since with the recipe trained through, the
falls stage 5 was blamed for go to zero on their own.

| ball starts | found | time to first sight (median) | falls |
|---|---:|---:|---:|
| front | 100% | 0.03 s | 0 |
| side | 85% | 0.62 s | 0 |
| back | 30% | 3.16 s | 0 |

To seat this brain in the lab as a run: `cp -r policies/find_ball runs/find_ball`
and drop `run:find_ball` from the 🧠 palette (**not** as a `duck-lab` CLI
positional — a run dir passed that way is not recognised as a trick duck and
gets driven with a walk command). To look at it:
`uv run render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb`.
