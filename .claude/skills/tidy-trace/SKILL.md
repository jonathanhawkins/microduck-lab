---
name: tidy-trace
description: Debug the playroom tidy loop (Track 12) — trace one run state by state, see every release, landing and fall with context, and re-measure the walker facts the brain's constants rest on. Use when eval-tidy's numbers drop, a duck stalls or falls near the basket, or after changing the walker, the MJCF, tidy.py or the detector.
---

# Tracing the tidy loop

`eval-tidy` tells you *how many* toys ended up in the basket. When the
number is wrong, these two tools tell you *why*. Both run headless on CPU
from `microduck_local/` and need the `microduck_rl` + `microduck` checkouts
next door (the shipped `alpha_walking` / `alpha_ground_pick` policies).

## 1. `trace-tidy` — one run under a microscope

```bash
uv run trace-tidy --seed 2 --seconds 300            # transitions, releases, landings, falls
uv run trace-tidy --seed 0 --every 5                 # + a position/intent line every 5 s
uv run trace-tidy --seed 0 --odom hostile            # under odometry drift (roadmap 1.7)
uv run trace-tidy --seed 3 --tether-ms 250           # under a brain tether (12.10), the same queue eval-tidy uses
uv run eval-tidy --seeds 8 --jobs 4 --tether-ms 250  # the benchmark, parallel, over a brain tether (12.10)
```

Read it like this:

- `-> state` lines are the brain's transitions (scan, explore, approach,
  blind, settle, pick, verify, carry, carry_explore, deliver, aim, drop,
  backoff, done). A run that is all `carry`/`deliver` turning is not seeing
  the basket; a run stuck in `scan`/`explore` is not seeing toys.
- `RELEASE` prints the trunk→basket and beak→basket distances and the
  estimate's error against the truth; `landed … IN/OUT` follows 1.5 s later.
  Geometry that matters: the beak reaches 0.08 m past the trunk, the feet
  0.04 m, a held toy sits 0.005–0.023 m beyond the tip, and the feet touch
  a 0.3 m tray's rim from 0.185 m to its centre. A release at trunk→basket
  0.21–0.24 with an estimate error under 0.03 lands in; outside that, look at
  the estimate (the `aim` state should have fixed it at 0.42 m) or the leg.
- `=== FALL` prints the two seconds before a fall (state, position,
  projected gravity, intent, head pitch, skill, ToF minimum, held toy) and
  what was within 0.35 m; the last column is the trunk's distance to the
  basket centre. Almost every fall so far was the rim: an explore or
  approach leg ending 0.2 m from the basket centre, or the turn right after
  a release (under a tether or drift the stop lands 1–3 cm closer, and a
  turn in place with the feet 2 cm from the rim trips — measured standing
  at 0.17–0.23 m: a left sidestep first never fell, the plain left turn
  fell at 0.17, a kicked or right turn at every distance). The keep-out
  disc, the staged approach for rim toys and the sidestep-then-turn
  back-off exist for those; if they reappear, that is where to look first.
- `--every` lines carry the intent's `note` — `blocked` (ToF guard),
  `detour`, `basket keep-out`, `scan k/6` — which is how the stalls were
  found (a duck that prints `blocked` for a minute is turning at the
  servo's steering rate, which does not turn the walker).

The `/sim` page shows the same brain live (`playroom` scenario, inspector →
brain `tidy`); `.claude/skills/sim-smoke` brings it up and screenshots it.

## 2. `walker-facts` — measure, don't assume

```bash
uv run walker-facts
```

Prints the beak/feet reach vs head pitch, the camera depression standing vs
walking (the gait holds the head 0.08 rad higher and rocks it ±0.02 —
that alone put basket estimates 0.8 m off until detection frames carried
their camera pose), the stopping coast, the fact that the walker does not
reverse at all, and the turn-in-place asymmetry (a right turn from a
standstill barely happens unless the gait is already going or a 0.2 m/s
forward kick is added). Every constant in `brain/tidy.py` quotes one of
these; if a number here moves, the constant next to its quote is the one to
revisit.

## Rules of thumb that came out of this

- Walk toward a target with the head LEVEL when it is 8 cm or higher off
  the floor (the basket marker), head DOWN only for floor toys, and never
  turn in place with the head down.
- Never release on a long-range estimate: re-measure standing still at
  0.42 m, square up, then walk the last 0.2 m straight with no steering.
- A toy that projects into the basket is one already delivered; walking at
  it is walking into the rim.
- Trust tracked ids, not confidence: a 2 cm toy at 1.5 m is found one frame
  in six at confidence 0.2 and is still a toy to walk toward.
