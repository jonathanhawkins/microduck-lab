"""The tidy benchmark (roadmap 12.13): scatter toys, start the tidy brain,
report what ended up in the basket.

    uv run eval-tidy --seeds 16 --toys 6 --seconds 300 --jobs 4   # seeds in parallel processes (one core each)

Each seed scatters its own toy layout; sixteen of them is the smallest
battery whose mean moves less than one toy (0.01) from chaos alone
(measured: eight seeds moved 0.1 between runs of the same brain).

Per seed: toys in the basket / total, picks, deliveries, falls, grasp
attempts and misses, and how long it took to finish or give up. Headless,
the same World the /sim page streams, so a number here is a number you
can watch there (`playroom` scenario, brain `tidy`).
"""

from __future__ import annotations

import argparse
import json
import os
import time

from pathlib import Path

import numpy as np

from .brain import REGISTRY, Senses
from .brain import tidy as _tidy  # noqa: F401
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_playroom


def run_one(seed: int, toys: int, seconds: float, quiet: bool = True, odom: str = "ideal",
            tether_ms: float = 0.0, loop_closure: bool = True,
            walker: str | None = None) -> dict:
    """`tether_ms` (roadmap 12.10): the brain runs somewhere else — a laptop
    over Wi-Fi, a cloud model — its senses reach it half this late and its
    intent lands on the robot half this later (brain/tether.py). 0 = onboard."""
    from .brain.tether import Tether
    sc = make_playroom(seed=seed, n=toys)
    sc.ducks[0].odom = odom
    tether = Tether(tether_ms / 1000.0)
    w = World(sc, infer_for={"d0": onnx_infer(Path(walker) if walker else POLICIES_DIR / "alpha_walking.onnx")}, seed=seed)
    d = w.ducks["d0"]
    from .brain.tidy import TidyParams
    brain = REGISTRY.make("tidy", p=TidyParams(loop_closure=loop_closure))
    t0 = time.time()
    states = []
    while w.t < seconds:
        tof, det = d.tof.last, d.detector.last
        s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                   det=det, det_age=None if det is None else w.t - det.t,
                   speed=d.heading_speed(w.data), odom=w.odom(d),
                   holding=d.holding is not None, skill=d.skill, bumped=w.bumped(d))
        intent = tether.intent_out(brain.step(tether.senses_in(s)), w.t)
        w.apply_intent(d, intent)
        if d.skill is None:
            d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        if not states or states[-1][1] != brain.state:
            states.append((round(w.t, 1), brain.state))
            if not quiet:
                print(f"  t={w.t:6.1f} {brain.state}", flush=True)
        if brain.state == "done":
            break
    score = w.tidy_score()
    return {"seed": seed, "toys": toys, "odom": odom, "tetherMs": tether_ms, "loopClosure": loop_closure,
            "seconds": seconds,
            "inBasket": score["inBasket"], "picked": brain.picked,
            "delivered": brain.delivered, "falls": d.falls, "attempts": d.grasp_attempts,
            "grasps": d.grasp_successes, "givenUp": sorted(brain.given_up),
            "simSeconds": round(w.t, 1), "wallSeconds": round(time.time() - t0, 1),
            "done": brain.state == "done", "transitions": len(states)}


def _seed_line(r: dict) -> str:
    return (f"seed {r['seed']}: {r['inBasket']}/{r['toys']} in the basket · picked {r['picked']} · delivered {r['delivered']}"
            f" · grasps {r['grasps']}/{r['attempts']} · falls {r['falls']} · {r['simSeconds']} s sim"
            f"{' · done' if r['done'] else ''}{' · gave up ' + ','.join(r['givenUp']) if r['givenUp'] else ''}")


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=16, help="each seed scatters its own toy layout (make_playroom)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--toys", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="append each seed's result here as a JSON line AND resume from it: "
                                  "seeds already in the file are not re-run (see eval_pitch.load_done)")
    ap.add_argument("--tag", default="", help="recorded in --out rows; a resume refuses to mix tags")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="seeds run in this many processes (each is single-threaded)")
    ap.add_argument("--odom", default="ideal", choices=["ideal", "datasheet", "hostile"],
                    help="odometry drift preset the brain has to live with (roadmap 1.7)")
    ap.add_argument("--walker", default=None, metavar="PATH",
                    help="run this walk policy instead of the shipped alpha_walking.onnx")
    ap.add_argument("--tether-ms", type=float, default=0.0,
                    help="brain round-trip latency: senses out, intent back (12.10; 0 = onboard)")
    ap.add_argument("--no-loop-closure", action="store_true",
                    help="steer by raw odometry instead of the brain's own loop-closed pose (5.5)")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    # Resumable, and each seed prints as it lands: a 16-seed battery is well
    # over an hour, and a machine that reclaims its container mid-run should
    # cost the seed it was on, not the battery (see eval_pitch.load_done).
    from .eval_pitch import load_done
    key = f"{args.odom}/{args.tether_ms}/{not args.no_loop_closure}/{args.toys}|{args.tag}"
    done = load_done(args.out, key, args.toys, args.seconds)
    rows = [done[sd] for sd in seeds if sd in done]
    if not args.json:
        for r in rows:
            print(_seed_line(r) + "  (already measured)", flush=True)
    todo = [sd for sd in seeds if sd not in done]
    out = open(args.out, "a") if args.out else None

    def keep(r: dict) -> None:
        rows.append(r)
        if out is not None:
            out.write(json.dumps({**r, "tag": key, "perSide": args.toys}) + "\n")
            out.flush()
        if not args.json:
            print(_seed_line(r), flush=True)

    args_list = [(sd, args.toys, args.seconds, True, args.odom, args.tether_ms,
                  not args.no_loop_closure, args.walker) for sd in todo]
    try:
        if args.jobs > 1 and len(todo) > 1:
            import multiprocessing as mp
            # spawn/forkserver: no forked ONNX runtime or MuJoCo state (the Windows-safe rule).
            ctx = mp.get_context("forkserver" if hasattr(mp, "get_context") and "forkserver" in mp.get_all_start_methods() else "spawn")
            with ctx.Pool(min(args.jobs, len(todo))) as pool:
                for r in pool.imap(_run_one_args, args_list):
                    keep(r)
        else:
            for a in args_list:
                keep(run_one(*a[:3], quiet=not args.verbose, odom=a[4], tether_ms=a[5],
                             loop_closure=a[6], walker=a[7] if len(a) > 7 else None))
    finally:
        if out is not None:
            out.close()
    rows.sort(key=lambda r: r["seed"])
    if args.json:
        print(json.dumps(rows))
        return
    frac = float(np.mean([r["inBasket"] / r["toys"] for r in rows]))
    print(f"mean tidied {frac:.2f} · falls {np.mean([r['falls'] for r in rows]):.2f}/run")


if __name__ == "__main__":
    main()
