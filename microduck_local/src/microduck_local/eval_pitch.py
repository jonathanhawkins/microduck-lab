"""`eval-pitch`: the soccer benchmark (first form) — two `chase` brains,
one ball, goals in a fixed time, over seeds.

    uv run eval-pitch --seeds 4 --seconds 300 --jobs 2
    uv run eval-pitch --seeds 4 --per-side 2      # 2v2 (teams share a blackboard, brain/team.py)

Per seed: goals scored on each side, kicks and pushes attempted, falls;
then the means per run. Headless, the same World `pitch` / `pitch-2v2` /
`pitch-3v3` streams on /sim.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_pitch


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
    goal_seq = 0
    while w.t < seconds:
        for d in w.ducks.values():
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill)
            intent = brains[d.id].step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if w.goal_seq != goal_seq:              # a goal: play restarts from the spawns
            goal_seq = w.goal_seq
            kickoff_brains(brains, teams)
    score = w.soccer_score()
    return {"seed": seed, "perSide": per_side, "left": score["left"], "right": score["right"],
            "kicks": {k: b.kicks for k, b in brains.items()}, "pushes": {k: b.pushes for k, b in brains.items()},
            "falls": {k: d.falls for k, d in w.ducks.items()}, "simSeconds": round(w.t, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--per-side", type=int, default=1, help="ducks a side: 1 (1v1), 2, 3")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    if args.jobs > 1 and len(seeds) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(seeds))) as pool:
            rows = pool.starmap(run_one, [(sd, args.seconds, args.per_side) for sd in seeds])
    else:
        rows = [run_one(sd, args.seconds, args.per_side) for sd in seeds]
    if args.json:
        print(json.dumps(rows))
        return
    for r in rows:
        print(f"seed {r['seed']}: goals left {r['left']} · right {r['right']} · kicks {sum(r['kicks'].values())}"
              f" · pushes {sum(r['pushes'].values())} · falls {r['falls']}")
    goals = [r["left"] + r["right"] for r in rows]
    print(f"{args.per_side}v{args.per_side}: mean goals {np.mean(goals):.2f}/run · kicks "
          f"{np.mean([sum(r['kicks'].values()) for r in rows]):.1f}/run · pushes "
          f"{np.mean([sum(r['pushes'].values()) for r in rows]):.1f}/run"
          f" · falls {np.mean([sum(r['falls'].values()) for r in rows]):.2f}/run"
          f" (per duck {np.mean([sum(r['falls'].values()) for r in rows]) / (2 * args.per_side):.2f})")


if __name__ == "__main__":
    main()
