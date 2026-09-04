"""`eval-pitch`: the soccer benchmark (first form) — two `chase` brains,
one ball, goals in a fixed time, over seeds.

    uv run eval-pitch --seeds 4 --seconds 300 --jobs 2
    uv run eval-pitch --seeds 4 --per-side 2      # 2v2 (teams share a blackboard, brain/team.py)

Per seed: goals scored on each side, kicks and pushes attempted, falls, and
the CONTINUOUS metrics below (ball progress, possession); then the means per
run. Headless, the same World `pitch` / `pitch-2v2` / `pitch-3v3` streams
on /sim.

READ THE `left`/`right` GOAL KEYS THE WAY THE WORLD WRITES THEM. They are
goal MOUTHS, not team scores: `World._check_goal` puts a ball crossing at +x
into `goals["right"]`, and `World.goal_for` puts the LEFT team's attack at
+x — so a row's `right` count is the number of goals the LEFT TEAM SCORED.
The run TOTAL (`left + right`) is unaffected, but every per-side reading of
a row is inverted if this is missed, and reading it the natural way flipped
the sign of a whole correlation table here before it was caught.

Measured, 16 seeds x 300 s of 1v1 per arm, shipped lens vs a 120x93 deg one
(two independent 8-seed batteries, both replicated), coefficient of variation
and the seeds an arm needs to resolve a 25% shift in the metric at p<0.05 /
80% power — i.e. the metric's own noise, with the size of this particular
contrast divided out:

READ `ballAdvance` WITH `ballProgress` BESIDE IT. Advance keeps only the
forward part (max(0, dx)), so it is INFLATED BY CHURN: a change that merely
makes the ball move more scores higher on it without sending the ball
anywhere. Measured, the attacker-handover fix: kicks +64%, advance +0.18
+/- 0.06 (2.9 sigma) — and signed `ballProgress` FLAT at -0.003 +/- 0.136
(0.0 sigma) with advance per kick HALVED, 0.202 -> 0.106. The ball moved
more and no further toward the goal. So: advance is the sensitive
instrument, signed progress is the one that says the motion had a
direction, and advance-per-kick says whether each touch was worth more.
Quote all three or none.

    goals            CV 0.76   146 seeds     r with goals   —
    kicks            CV 0.49    62 seeds     r  0.30 [-0.06, +0.59]
    falls            CV 1.22   376 seeds     r  0.07 [-0.29, +0.40]
    ballAdvance      CV 0.40    43 seeds     r  0.50 [+0.19, +0.72]
    possession       CV 0.16     9 seeds     r  0.33 [-0.02, +0.61]
    possessionWide   CV 0.11     6 seeds     r  0.23 [-0.13, +0.53]
    ballProgress     CV 4.20      —          r  0.11 [-0.25, +0.44]

`ballAdvance` is the one to judge a variant on: it is the only metric here
whose association with goals is resolved away from zero (and the only one at
all, `kicks` included), at ~3x fewer seeds than goals. `possession` is the
cheapest DETECTOR of any difference at all — 6-9 seeds against goals' 146 —
but it is not evidence a variant is better, because what it tracks is how
much of the run a duck spends standing over the ball. Ball progress summed
over both teams is near zero by construction in self-play (the two sides
attack opposite goals) and carries nothing; it is a per-TEAM metric, for an
asymmetric matchup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_pitch
from .world.arena import (
    KICK_GOAL_S,  # noqa: F401  (the attribution window; the World keeps the counts)
)
from .world.metrics import (  # noqa: F401  (re-exported: tooling imports these from here)
    CARRY_S,
    METRIC_FIELDS,
    POSSESSION_R,
    POSSESSION_WIDE_R,
    SPIN_FIELDS,
    PitchMetrics,
    SpinMetrics,
)


def run_one(seed: int, seconds: float, per_side: int = 1, walker: str | None = None) -> dict:
    from .brain.team import brain_kwargs, kickoff_brains
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(Path(walker) if walker else POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: REGISTRY.make("chase", **brain_kwargs(d, w, teams)) for d in sc.ducks}
    # A little seed-dependent asymmetry: nudge the ball off centre.
    rng = np.random.default_rng(seed)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    w.data.qpos[q:q + 2] = rng.uniform(-0.2, 0.2, 2)
    metrics = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    goal_seq = 0
    spin = SpinMetrics(w)
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            intent = brains[d.id].step(s)
            spin.tick(d, intent.twist)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        metrics.tick()
        if w.goal_seq != goal_seq:              # a goal: play restarts from the spawns
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams)
    score = w.soccer_score()
    return {"seed": seed, "perSide": per_side, "left": score["left"], "right": score["right"],
            "kickGoals": score["kicked"], "bumpGoals": score["bumped"],   # attributed by the World (KICK_GOAL_S)
            "kicks": {k: b.kicks for k, b in brains.items()}, "pushes": {k: b.pushes for k, b in brains.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()}, "simSeconds": round(w.t, 1),
            "seconds": seconds,
            **spin.row(), **metrics.row()}


def load_done(path: str | None, tag: str, per_side: int, seconds: float) -> dict[int, dict]:
    """Seeds already measured into `path` (JSON lines, one row a seed), for a
    resume. A battery is the best part of an hour and this machine reclaims
    its container mid-run, so a killed run should cost the seed it was on and
    nothing else. Rows written under different settings are REFUSED rather
    than silently mixed: the brain's own parameters do not appear in a row,
    so `--tag` is how a caller says which variant a file belongs to.

    A row written BEFORE the ball-progress/possession metrics existed is not
    refused — the goals, kicks and falls in it are as good as they ever were,
    and re-running an hour of seeds to recover metrics nobody measured then
    would be the wrong trade. Its missing metrics come back as None, which
    every consumer here treats as "not measured": `_seed_line` prints a dash
    and the summary averages the seeds that have the field and SAYS how many
    that was. Filling them with 0.0 instead would quietly drag a mean toward
    zero, which is the one outcome worth engineering against."""
    if not path or not os.path.exists(path):
        return {}
    done: dict[int, dict] = {}
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("tag", ""), r.get("perSide"), r.get("seconds")) != (tag, per_side, seconds):
                raise SystemExit(
                    f"{path}:{n} was measured with tag={r.get('tag', '')!r} perSide={r.get('perSide')} "
                    f"seconds={r.get('seconds')}, not tag={tag!r} perSide={per_side} seconds={seconds}. "
                    "Write a different variant to a different file.")
            for f in METRIC_FIELDS:
                r.setdefault(f, None)
            done[int(r["seed"])] = r
    return done


def _total(r: dict, field: str) -> float | None:
    """A row's metric summed over both teams, or None if this row predates it."""
    v = r.get(field)
    return None if v is None else float(sum(v.values()))


def _mean_field(rows: list[dict], field: str) -> tuple[float | None, int]:
    """(mean over both teams' total, how many rows carried the field)."""
    vals = [v for v in (_total(r, field) for r in rows) if v is not None]
    return (float(np.mean(vals)) if vals else None), len(vals)


def _fmt(v: dict | None, unit: str) -> str:
    if v is None:
        return "—"
    return " ".join(f"{k} {x:+.2f}" if unit == "m/min" else f"{k} {x:.1f}" for k, x in sorted(v.items())) + f" {unit}"


def _seed_line(r: dict) -> str:
    return (f"seed {r['seed']}: goals left {r['left']} · right {r['right']} ({r['kickGoals']} kicked, {r['bumpGoals']} bumped)"
            f" · kicks {sum(r['kicks'].values())} · pushes {sum(r['pushes'].values())} · falls {r['falls']}"
            f" · progress {_fmt(r.get('ballProgress'), 'm/min')}"
            f" · possession {_fmt(r.get('possession'), 's/min')}")


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed: --seeds 12 --seed0 12 EXTENDS a 12-seed battery instead of redoing it "
                         "(a promising result found on one set of seeds has to be confirmed on fresh ones)")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--per-side", type=int, default=1, help="ducks a side: 1 (1v1), 2, 3")
    ap.add_argument("--walker", default=None, metavar="PATH",
                    help="run this walk policy instead of the shipped alpha_walking.onnx "
                         "(roadmap 3.7: comparing two LOCALLY trained walkers, never one against the shipped one)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="append each seed's result here as a JSON line AND resume from it: "
                                  "seeds already in the file are not re-run")
    ap.add_argument("--tag", default="", help="recorded in --out rows; a resume refuses to mix tags, so a file "
                                              "written under different brain settings is never reused by mistake")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    # Each seed PRINTS as it finishes, rather than the battery printing at the
    # end: a 12-seed 3v3 battery is the best part of an hour, and a machine
    # that reclaims its container mid-run should cost one seed, not all of
    # them (it cost all of them, twice). Resume the rest with --seed0.
    done = load_done(args.out, args.tag, args.per_side, args.seconds)
    rows = [done[sd] for sd in seeds if sd in done]
    if not args.json:
        for r in rows:
            print(_seed_line(r) + "  (already measured)", flush=True)
    todo = [sd for sd in seeds if sd not in done]
    out = open(args.out, "a") if args.out else None

    def keep(r: dict) -> None:
        rows.append(r)
        if out is not None:
            out.write(json.dumps({**r, "tag": args.tag}) + "\n")
            out.flush()
        if not args.json:
            print(_seed_line(r), flush=True)

    args_list = [(sd, args.seconds, args.per_side, args.walker) for sd in todo]
    try:
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                for r in pool.imap(_run_one_args, args_list):
                    keep(r)
        else:
            for a in args_list:
                keep(run_one(*a))
    finally:
        if out is not None:
            out.close()
    rows.sort(key=lambda r: r["seed"])
    if args.json:
        print(json.dumps(rows))
        return
    goals = [r["left"] + r["right"] for r in rows]
    print(f"{args.per_side}v{args.per_side}: mean goals {np.mean(goals):.2f}/run ({np.mean([r['kickGoals'] for r in rows]):.2f} kicked, "
          f"{np.mean([r['bumpGoals'] for r in rows]):.2f} bumped) · kicks "
          f"{np.mean([sum(r['kicks'].values()) for r in rows]):.1f}/run · pushes "
          f"{np.mean([sum(r['pushes'].values()) for r in rows]):.1f}/run"
          f" · falls {np.mean([sum(r['falls'].values()) for r in rows]):.2f}/run"
          f" (per duck {np.mean([sum(r['falls'].values()) for r in rows]) / (2 * args.per_side):.2f})")
    # The continuous metrics, both teams together (in self-play the two sides
    # run the same brain, so the pair's total is the run's number; the per-team
    # split in each seed line is what an asymmetric matchup is read from).
    parts, missing = [], 0
    for f, unit in (("ballProgress", "m/min"), ("ballAdvance", "m/min"),
                    ("possession", "s/min"), ("possessionWide", "s/min")):
        m, n = _mean_field(rows, f)
        missing = max(missing, len(rows) - n)
        parts.append(f"{f} {'—' if m is None else f'{m:+.2f}' if unit == 'm/min' else f'{m:.1f}'} {unit}")
    print("both teams: " + " · ".join(parts)
          + (f"   ({len(rows) - missing}/{len(rows)} seeds; the rest predate these metrics)" if missing else ""))


if __name__ == "__main__":
    main()
