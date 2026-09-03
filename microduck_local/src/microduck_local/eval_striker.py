"""`eval-striker`: the learned striker against the scripted `Chase`, on the
pitch, on identical seeds (roadmap 4.4).

    uv run python -m microduck_local.eval_striker --left chase  --solo --seeds 8 --out runs/s.jsonl --tag solo-chase
    uv run python -m microduck_local.eval_striker --left striker:striker-v1 --solo --seeds 8 --out runs/k.jsonl --tag solo-v1
    uv run python -m microduck_local.eval_striker --left striker:striker-v1 --seeds 8   # 1v1 against a scripted chase

Why not `eval-pitch`: that runs `chase` on both sides by construction. This
is the same world, the same walker, the same metrics and the same resume
machinery, with the LEFT duck's brain swappable — so the arms differ in one
thing. `--solo` drops the right duck (a 1v0 pitch), which is what the
striker is trained on and the cleanest reading of "can it take the ball to
the goal"; without it the right duck is the scripted `Chase` and the row is
a head-to-head.

READ THE THREE BALL NUMBERS TOGETHER (`eval_pitch`'s docstring is the long
version): `ballAdvance` keeps only the forward part and is inflated by
churn, so it is quoted with signed `ballProgress` beside it and with
advance PER KICK, which says whether each touch was worth anything. And
read the event counts: at 8 seeds a battery holds tens of kicks but a
handful of goals and falls, so goals here are reported, not judged.

The `left`/`right` keys of `goals` are goal MOUTHS, not teams: a ball
crossing at +x lands in `goals["right"]` and the LEFT team attacks +x, so
a row's `right` count is what the left team scored. `metrics` keys are
teams and need no such translation.
"""

from __future__ import annotations

import argparse
import json


from dataclasses import replace

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .brain.striker import LearnedStriker
from .eval_pitch import load_done
from .world import World, make_pitch
from .world.metrics import PitchMetrics


def pitch_scenario(per_side: int, solo: bool):
    """`make_pitch`'s pitch, with the right side removed for `--solo`. Both
    arms build it the same way, so a solo battery compares brains and not
    worlds."""
    sc = make_pitch(per_side=per_side)
    if solo:
        sc = replace(sc, ducks=[d for d in sc.ducks if d.team == "left"])
    return sc


def make_brain(kind: str, spec, world, teams: dict):
    """A brain for one duck: the scripted `chase` with the pitch kwargs the
    world gives it, or `striker:<name>` — a trained brain from brains/, which
    gets the same goal, bounds and team."""
    from .brain.team import Team, brain_kwargs
    if kind.startswith("striker:"):
        d = world.ducks[spec.id]
        hx, hy = world.scenario.floor[0] / 2 - 0.25, world.scenario.floor[1] / 2 - 0.25
        team = teams.setdefault(spec.team, Team(spec.team)) if spec.team else None
        return LearnedStriker(kind.split(":", 1)[1], goal=world.goal_for(d), duck_id=spec.id,
                              bounds=(hx, hy), goal_w=world.goal_width, team=team)
    return REGISTRY.make(kind, **brain_kwargs(spec, world, teams))


def run_one(seed: int, seconds: float, left: str = "chase", right: str = "chase",
            per_side: int = 1, solo: bool = False) -> dict:
    """One run. Byte-for-byte `eval_pitch.run_one` when left == right ==
    "chase" and solo is False — the test in tests/test_striker.py pins that,
    which is what makes the scripted arm here the published baseline and not
    a re-implementation of it."""
    from .brain.team import kickoff_brains
    sc = pitch_scenario(per_side, solo)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={d.id: infer for d in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {d.id: make_brain(left if d.team == "left" else right, d, w, teams) for d in sc.ducks}
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
    return {"seed": seed, "perSide": per_side, "solo": solo, "left": score["left"], "right": score["right"],
            "leftBrain": left, "rightBrain": right if not solo else None,
            "kickGoals": score["kicked"], "bumpGoals": score["bumped"],
            "kicks": {k: b.kicks for k, b in brains.items()}, "pushes": {k: b.pushes for k, b in brains.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()},
            "team": {d.id: (d.team or d.id) for d in sc.ducks},
            "simSeconds": round(w.t, 1), "seconds": seconds, **metrics.row()}


# --- reading a battery --------------------------------------------------------
def team_of(row: dict, side: str) -> dict:
    return {k: v for k, v in row["team"].items() if v == side}


def side_reading(rows: list[dict], side: str = "left") -> dict:
    """One side's three ball numbers plus its event counts, over a battery."""
    adv = [r["ballAdvance"][side] for r in rows if r.get("ballAdvance")]
    prog = [r["ballProgress"][side] for r in rows if r.get("ballProgress")]
    poss = [r["possession"][side] for r in rows if r.get("possession")]
    kicks = [sum(v for k, v in r["kicks"].items() if k in team_of(r, side)) for r in rows]
    falls = [sum(v for k, v in r["falls"].items() if k in team_of(r, side)) for r in rows]
    # The left team attacks +x, whose mouth the World records as "right".
    scored = [r["right" if side == "left" else "left"] for r in rows]
    tot_adv = float(np.sum([a * r["simSeconds"] / 60.0 for a, r in zip(adv, rows)])) if adv else 0.0
    return {"seeds": len(rows), "advance": float(np.mean(adv)) if adv else None,
            "advanceSd": float(np.std(adv, ddof=1)) if len(adv) > 1 else None,
            "progress": float(np.mean(prog)) if prog else None,
            "progressSd": float(np.std(prog, ddof=1)) if len(prog) > 1 else None,
            "possession": float(np.mean(poss)) if poss else None,
            "kicks": int(np.sum(kicks)), "falls": int(np.sum(falls)), "goals": int(np.sum(scored)),
            "advPerKick": (tot_adv / np.sum(kicks)) if np.sum(kicks) else None,
            "perSeedAdvance": adv, "perSeedGoals": scored}


def _fmt(v, nd=2, sign=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def summarize(rows: list[dict], side: str = "left") -> str:
    s = side_reading(rows, side)
    return (f"{side}: ballAdvance {_fmt(s['advance'])} ± {_fmt(s['advanceSd'])} m/min · "
            f"ballProgress {_fmt(s['progress'], sign=True)} ± {_fmt(s['progressSd'])} m/min · "
            f"advance/kick {_fmt(s['advPerKick'], 3)} m · possession {_fmt(s['possession'], 1)} s/min\n"
            f"    events over {s['seeds']} seeds: {s['kicks']} kicks · {s['goals']} goals · {s['falls']} falls")


def _run_args(a: tuple) -> dict:
    return run_one(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="chase", help="chase | striker:<brains/ name>")
    ap.add_argument("--right", default="chase", help="the opponent (ignored with --solo)")
    ap.add_argument("--solo", action="store_true", help="1v0: no opponent duck (what the striker trains on)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0, help="first seed — extends a battery onto FRESH seeds")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--per-side", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="append each seed as a JSON line AND resume from it")
    ap.add_argument("--tag", default="", help="recorded in --out rows; a resume refuses to mix tags")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    done = load_done(args.out, args.tag, args.per_side, args.seconds)
    rows = [done[sd] for sd in seeds if sd in done]
    todo = [sd for sd in seeds if sd not in done]
    out = open(args.out, "a") if args.out else None

    def keep(r: dict) -> None:
        rows.append(r)
        if out is not None:
            out.write(json.dumps({**r, "tag": args.tag}) + "\n")
            out.flush()
        if not args.json:
            print(f"seed {r['seed']}: goals L{r['left']}/R{r['right']} · kicks {sum(r['kicks'].values())}"
                  f" · falls {sum(r['falls'].values())}"
                  f" · advance {r['ballAdvance']} · progress {r['ballProgress']}", flush=True)

    todo_args = [(sd, args.seconds, args.left, args.right, args.per_side, args.solo) for sd in todo]
    try:
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                for r in pool.imap(_run_args, todo_args):
                    keep(r)
        else:
            for a in todo_args:
                keep(run_one(*a))
    finally:
        if out is not None:
            out.close()
    rows.sort(key=lambda r: r["seed"])
    if args.json:
        print(json.dumps(rows))
        return
    label = f"{args.left} vs {'nobody' if args.solo else args.right}"
    print(f"\n{label}, {len(rows)} seeds x {args.seconds:g} s:")
    print("  " + summarize(rows, "left"))
    if not args.solo:
        print("  " + summarize(rows, "right"))


if __name__ == "__main__":
    main()
