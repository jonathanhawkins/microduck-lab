"""The tidy benchmark (roadmap 12.13): scatter toys, start the tidy brain,
report what ended up in the basket.

    uv run eval-tidy --seeds 3 --toys 6 --seconds 300
    uv run eval-tidy --seeds 8 --jobs 4          # seeds in parallel processes (one core each)

Per seed: toys in the basket / total, picks, deliveries, falls, grasp
attempts and misses, and how long it took to finish or give up. Headless,
the same World the /sim page streams, so a number here is a number you
can watch there (`playroom` scenario, brain `tidy`).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .brain import REGISTRY, Senses
from .brain import tidy as _tidy  # noqa: F401
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_playroom


def run_one(seed: int, toys: int, seconds: float, quiet: bool = True, odom: str = "ideal",
            tether_ms: float = 0.0) -> dict:
    """`tether_ms` (roadmap 12.10): the brain runs somewhere else — a laptop
    over Wi-Fi, a cloud model — and its intent lands on the robot this much
    after the senses it was decided from. 0 = onboard."""
    from collections import deque
    sc = make_playroom(seed=seed, n=toys)
    sc.ducks[0].odom = odom
    queue: deque = deque()
    delay = tether_ms / 1000.0
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=seed)
    d = w.ducks["d0"]
    brain = REGISTRY.make("tidy")
    t0 = time.time()
    states = []
    while w.t < seconds:
        tof, det = d.tof.last, d.detector.last
        s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                   det=det, det_age=None if det is None else w.t - det.t,
                   speed=d.heading_speed(w.data), odom=w.odom(d),
                   holding=d.holding is not None, skill=d.skill)
        queue.append((w.t + delay, brain.step(s)))
        while queue and queue[0][0] <= w.t + 1e-9:
            _, intent = queue.popleft()
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
    return {"seed": seed, "toys": toys, "odom": odom, "tetherMs": tether_ms, "inBasket": score["inBasket"], "picked": brain.picked,
            "delivered": brain.delivered, "falls": d.falls, "attempts": d.grasp_attempts,
            "grasps": d.grasp_successes, "givenUp": sorted(brain.given_up),
            "simSeconds": round(w.t, 1), "wallSeconds": round(time.time() - t0, 1),
            "done": brain.state == "done", "transitions": len(states)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--toys", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="seeds run in this many processes (each is single-threaded)")
    ap.add_argument("--odom", default="ideal", choices=["ideal", "datasheet", "hostile"],
                    help="odometry drift preset the brain has to live with (roadmap 1.7)")
    ap.add_argument("--tether-ms", type=float, default=0.0,
                    help="brain round-trip latency: senses out, intent back (12.10; 0 = onboard)")
    args = ap.parse_args()
    seeds = [args.seed0 + k for k in range(args.seeds)]
    if args.jobs > 1 and len(seeds) > 1:
        import multiprocessing as mp
        # spawn/forkserver: no forked ONNX runtime or MuJoCo state (the Windows-safe rule).
        ctx = mp.get_context("forkserver" if hasattr(mp, "get_context") and "forkserver" in mp.get_all_start_methods() else "spawn")
        with ctx.Pool(min(args.jobs, len(seeds))) as pool:
            rows = pool.starmap(run_one, [(sd, args.toys, args.seconds, True, args.odom, args.tether_ms) for sd in seeds])
    else:
        rows = [run_one(sd, args.toys, args.seconds, quiet=not args.verbose, odom=args.odom, tether_ms=args.tether_ms)
                for sd in seeds]
    if args.json:
        print(json.dumps(rows))
        return
    for r in rows:
        print(f"seed {r['seed']}: {r['inBasket']}/{r['toys']} in the basket · picked {r['picked']} · delivered {r['delivered']}"
              f" · grasps {r['grasps']}/{r['attempts']} · falls {r['falls']} · {r['simSeconds']} s sim"
              f"{' · done' if r['done'] else ''}{' · gave up ' + ','.join(r['givenUp']) if r['givenUp'] else ''}")
    frac = float(np.mean([r["inBasket"] / r["toys"] for r in rows]))
    print(f"mean tidied {frac:.2f} · falls {np.mean([r['falls'] for r in rows]):.2f}/run")


if __name__ == "__main__":
    main()
