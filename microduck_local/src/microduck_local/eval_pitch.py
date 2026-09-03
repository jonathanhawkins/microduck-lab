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
import math
import os

import numpy as np

from . import contract as C
from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_pitch
from .world.arena import (
    KICK_GOAL_S,  # noqa: F401  (the attribution window; the World keeps the counts)
)

# --- the continuous metrics ---------------------------------------------------
# Goals are a terrible ruler for this benchmark: ~2.5 a run against ~13 kicks,
# so the same brain measured twice gave 3 falls and 6, and four brain variants
# landed inside each other's noise on goals in one day. What the ducks actually
# do thousands of times a run is *reach the ball* and *move it*; these two
# accumulate that at the control tick.
#
# POSSESSION_R — a duck is "on the ball" inside this radius (trunk centre to
# ball centre, in the plane). The kick sweet spot is 6-10 cm ahead and 4-8 cm
# to the side of the trunk (`ChaseParams.kick_ahead/kick_side`), i.e. ~0.12 m
# out; 0.25 m is about twice that, so a duck lined up on the ball or one step
# short of it counts, while it stays well inside `duck_keepout` (0.40 m) — the
# range at which the chase brain starts avoiding the other duck — so the two
# contesting ducks are rarely both on the ball. Only the NEAREST duck can hold
# possession anyway, so the radius sets how much loitering counts, not who wins
# a contest.
POSSESSION_R = 0.25
# The same clock at the keepout radius, recorded only as a robustness check:
# it says whether a conclusion turns on the exact choice of POSSESSION_R.
#
# It did, and not in the direction that was expected: over 16 seeds an arm the
# WIDER radius came out both quieter (CV 0.11 vs 0.16) and far more sensitive
# to the lens contrast (p=0.0001 vs p=0.06). The primary stays 0.25 m anyway.
# 0.25 m was fixed before the battery and 0.40 m was not, and a radius chosen
# because it won on the one battery that could confirm it is exactly the kind
# of result this benchmark has already had to withdraw twice; 0.40 m also sits
# at the chase brain's own `duck_keepout` and is occupied 45 s in every 60 in
# 1v1, which is close enough to saturation to be measuring the pitch rather
# than the brain. Promote it when a battery run FOR that question says so.
POSSESSION_WIDE_R = 0.40
# After the last tick a team was on the ball it keeps CREDIT for the ball's
# motion for this long. A kick is exactly the case that matters: the ball
# leaves at speed and travels most of its distance with no duck within
# POSSESSION_R, so progress credited only while inside the radius would score
# dribbling and ignore the shot — the one thing the benchmark is about. 2.0 s
# is deliberately shorter than the World's own kick→goal attribution window
# (KICK_GOAL_S = 4.0 s): long enough for a kicked ball to run out, short
# enough that a ball rattling round the boards minutes later is nobody's.
CARRY_S = 2.0

# The per-team metric keys a row carries. A row written before these existed
# has none of them; `load_done` fills them with None rather than 0.0, and the
# summary reports how many seeds it could use (see `_mean_field`).
METRIC_FIELDS = ("ballProgress", "ballAdvance", "possession", "possessionWide")


class PitchMetrics:
    """Ball progress and possession, accumulated at the 50 Hz control tick.

    `tick()` is called once per control step, AFTER `World.step()`, and costs
    one distance per duck plus a few floats — nothing measurable against an
    `mj_step` of a 2-duck pitch.

    **Ball progress** is the ball's displacement toward the goal a team
    attacks (the pitch's long axis is x; `World.goal_for` gives the sign),
    summed over the ticks that team is in control. Why not the other two
    candidates:

    - *Signed per-step displacement over the whole run* telescopes. Sum every
      Δx and all that survives is the ball's net start-to-end position, and
      since every goal recentres the ball, that quantity is ~1.25 m × (goals
      for − goals against) plus a remainder — goals in disguise, with goals'
      variance. It cannot carry more signal than the thing it collapses to.
    - *Positive-only displacement, uncredited* measures how much the ball
      moved at all, which both teams get paid for equally; it is a ball-motion
      counter, not a per-side metric.

    Attributing each tick's displacement to the team in control keeps the
    telescoping LOCAL — inside one possession, where net start-to-end is
    exactly what "this team took the ball 40 cm toward the goal" means — and
    a run contains hundreds of possessions instead of two or three goals.
    `progress` is the signed sum (a team that shoves the ball back toward its
    own goal is charged for it); `advance` sums only the forward part, and is
    recorded alongside so the two can be compared on the same battery. They
    were, over 32 runs, and `advance` won on both counts: the signed sum has a
    CV of 4.2 as a run total, because in self-play the two sides' signs oppose
    and the run total is a difference of two similar numbers, while `advance`
    has a CV of 0.40 and is the only metric here whose correlation with goals
    is resolved away from zero (r 0.50, 95% CI [+0.19, +0.72]). Keep both:
    the signed one is the meaningful per-TEAM reading in an asymmetric
    matchup, where the cancellation that kills its run total is the point.

    **Possession** is the strict clock: seconds where the nearest duck to the
    ball is within POSSESSION_R and belongs to that team. No carry — the task
    it measures is "was one of mine on the ball", which is either true or not.
    It is by far the quietest number the benchmark produces (CV 0.16, against
    goals' 0.76) and therefore the cheapest way to tell that two variants
    differ AT ALL; it is not a measure of playing well, and the battery could
    not resolve its correlation with goals away from zero. Screen with it,
    judge with `advance`.
    """

    def __init__(self, world: World, team_of: dict[str, str]):
        self.w = world
        self.team_of = dict(team_of)
        self.teams = sorted(set(team_of.values()))
        # Which way is "forward" for each team, from the goal it attacks.
        self.sign: dict[str, float] = {}
        for did, tm in team_of.items():
            g = world.goal_for(world.ducks[did])
            s = 1.0 if g is None or g[0] >= 0 else -1.0
            if self.sign.setdefault(tm, s) != s:
                raise ValueError(f"team {tm!r} has ducks attacking opposite goals")
        self.progress = {t: 0.0 for t in self.teams}
        self.advance = {t: 0.0 for t in self.teams}
        self.possession = {t: 0.0 for t in self.teams}
        self.possession_wide = {t: 0.0 for t in self.teams}
        self._prev = world.ball_xy()
        self._holder: str | None = None       # team credited with the ball right now
        self._holder_t = -1e9                 # when it was last strictly on the ball
        self._goal_seq = world.goal_seq

    def nearest(self) -> tuple[str | None, float]:
        """(duck id, planar distance) of the duck closest to the ball."""
        ball = self.w.ball_xy()
        if ball is None:
            return None, math.inf
        best, best_d = None, math.inf
        for did, d in self.w.ducks.items():
            p = d.trunk_pos(self.w.data)
            r = math.hypot(float(p[0]) - ball[0], float(p[1]) - ball[1])
            if r < best_d:
                best, best_d = did, r
        return best, best_d

    def tick(self) -> None:
        w = self.w
        ball = w.ball_xy()
        if ball is None:
            return
        # The displacement first, and it is credited to whoever held the ball
        # THROUGH the step that just ended, not to whoever is standing over it
        # now: the tick a duck arrives to win the ball back is a tick of motion
        # its opponent caused, and charging a couple of centimetres to the
        # wrong side at every changeover — hundreds a run — is a real bias, not
        # a rounding one.
        if w.goal_seq != self._goal_seq:
            # A goal: the World teleported the ball back to the centre spot.
            # That jump is not anybody's progress (counted, it would cancel
            # almost exactly the goal it followed), and the next possession
            # starts clean.
            self._goal_seq = w.goal_seq
            self._prev, self._holder = ball, None
        elif self._prev is not None and self._holder is not None and w.t - self._holder_t <= CARRY_S:
            dx = self.sign[self._holder] * (ball[0] - self._prev[0])
            self.progress[self._holder] += dx
            self.advance[self._holder] += max(0.0, dx)
        self._prev = ball
        # Then who is on the ball at the end of the step, which is who the NEXT
        # step's motion belongs to.
        who, r = self.nearest()
        if who is not None and r <= POSSESSION_R:
            tm = self.team_of[who]
            self.possession[tm] += C.CTRL_DT
            self._holder, self._holder_t = tm, w.t          # control (and credit) passes
        if who is not None and r <= POSSESSION_WIDE_R:
            self.possession_wide[self.team_of[who]] += C.CTRL_DT

    def row(self) -> dict:
        """The four per-team metrics, as RATES per minute of play, so a row is
        comparable across `--seconds` (the totals divide out; `simSeconds` is
        in the row if anyone wants them back)."""
        per_min = 60.0 / max(self.w.t, 1e-9)
        return {"ballProgress": {t: round(v * per_min, 3) for t, v in self.progress.items()},
                "ballAdvance": {t: round(v * per_min, 3) for t, v in self.advance.items()},
                "possession": {t: round(v * per_min, 3) for t, v in self.possession.items()},
                "possessionWide": {t: round(v * per_min, 3) for t, v in self.possession_wide.items()}}


def run_one(seed: int, seconds: float, per_side: int = 1) -> dict:
    from .brain.team import brain_kwargs, kickoff_brains
    sc = make_pitch(per_side=per_side)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
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
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill, bumped=w.bumped(d))
            intent = brains[d.id].step(s)
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
            "seconds": seconds, **metrics.row()}


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

    args_list = [(sd, args.seconds, args.per_side) for sd in todo]
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
