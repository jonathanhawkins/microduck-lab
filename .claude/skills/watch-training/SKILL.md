---
name: watch-training
description: >-
  Look at what the ACTIVE teach run is practicing right now: renders the live
  checkpoint under the trainer's own env knobs (actuator, spawn mix, reward
  gates — read from the live trainer process) and writes contact sheets to
  read. Use during any training run to visually verify the behavior instead
  of steering by reward curves — charts have hidden one-sided flops,
  neck-leans, and parking that a single glance at frames caught. Trigger on:
  "watch the training", "what is it practicing", "screenshot the training",
  "is training doing the right thing".
---

# watch-training — look at the live student, under its own rules

Reward curves between stages have repeatedly hidden what a policy actually
practices (a flat hold that was a one-sided leg-flop; a rising hold that was
spawn handouts). This skill renders the ACTIVE run's `live.onnx` under the
exact env the trainer subprocess is using — same actuator, same spawn family
mix, same reward-gate knobs — so the frames show the true current drill.

```bash
bash .claude/skills/watch-training/watch.sh
```

Then **Read the printed `ep*_sheet.png` files** (they are images; captions
carry trunk/head heights, pitch, contacts — see the render-rollout skill for
how to read them). Optional args pass through to render-rollout, e.g.
`--seconds 12 --sheet-frames 20`.

Notes:
- Read-only: never restarts training or the lab; safe at any time. It does
  compete for CPU, so it runs under `nice` and short episodes by default.
- It needs a live trainer (it reads the knobs from the process env); for a
  FINISHED run use render-rollout directly and pass the stage knobs by hand
  (including the actuator the run's own stage declares (watch.sh harvests it
from the live trainer, so it is already correct)).
- Monitoring loop pattern: call it at every stage boundary and mid-stage
  telemetry checkpoint, and READ the sheet before trusting any curve. A
  subagent can do the same: run the script, read the sheets, report what
  the duck is physically doing and whether it matches the stage's `detail`.
