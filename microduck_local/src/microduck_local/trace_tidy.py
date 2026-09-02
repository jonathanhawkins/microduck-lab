"""`trace-tidy`: the tidy loop under a microscope (roadmap 12.13's debugger).

    uv run trace-tidy --seed 2 --seconds 300
    uv run trace-tidy --seed 0 --every 5        # plus a position line every 5 s

Runs the `playroom` scenario headless with the `tidy` brain — the same
World the /sim page streams — and prints what the benchmark only counts:
every state transition, every release (trunk and beak distance to the
basket centre, the estimate's error against the truth) and where the toy
landed 1.5 s later, and every fall WITH the two seconds before it (state,
position, projected gravity, the intent, the ToF minimum, what was held)
and what was within 35 cm. The benchmark said "1/6"; this said "the feet
touch the rim at 0.185 m and the estimate was 6 cm short".

The truth (toy poses, the basket, falls) is read from the sim; the brain
only ever sees its senses. Nothing here is on the wire to the page.
"""

from __future__ import annotations

import argparse
import collections
import math

import numpy as np

from .brain import REGISTRY, Senses
from .brain import tidy as _tidy  # noqa: F401
from .brain.brain_env import POLICIES_DIR, onnx_infer
from .world import World, make_playroom


def senses_of(w: World, d) -> Senses:
    tof, det = d.tof.last, d.detector.last
    pos = d.trunk_pos(w.data)
    return Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                  det=det, det_age=None if det is None else w.t - det.t,
                  speed=d.heading_speed(w.data), odom=(float(pos[0]), float(pos[1]), d.yaw(w.data)),
                  holding=d.holding is not None, skill=d.skill)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--toys", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--every", type=float, default=0.0, help="print a position line every N s (0 = off)")
    ap.add_argument("--history", type=float, default=2.0, help="seconds of context printed before a fall")
    args = ap.parse_args()

    sc = make_playroom(seed=args.seed, n=args.toys)
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=args.seed)
    d = w.ducks["d0"]
    brain = REGISTRY.make("tidy")
    bx, by = sc.basket.pos
    hx, hy = sc.floor[0] / 2 - 0.25, sc.floor[1] / 2 - 0.25       # make_playroom pads the floor by 0.5
    print(f"playroom seed {args.seed}: {len(sc.pickables)} toys, basket at ({bx:.2f}, {by:.2f}) "
          f"{sc.basket.size[0]:.2f}×{sc.basket.size[1]:.2f} m rim {sc.basket.rim:.2f} m")
    hist: collections.deque = collections.deque(maxlen=int(args.history * 50))
    falls, last, landings, next_print = 0, None, [], 0.0
    while w.t < args.seconds:
        intent = brain.step(senses_of(w, d))
        w.apply_intent(d, intent)
        if d.skill is None:
            d.set_cmd(w.data, intent.twist, intent.head)
        pos = d.trunk_pos(w.data).copy()
        g = d.projected_gravity(w.data)
        tof = d.tof.last
        tmin = None if tof is None or not (tof.depth_mm > 0).any() else int(tof.depth_mm[tof.depth_mm > 0].min())
        hist.append((round(w.t, 2), brain.state, (round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)),
                     tuple(round(float(v), 2) for v in g), tuple(round(float(v), 2) for v in intent.twist),
                     round(float(intent.head[1]), 2), d.skill, tmin, d.holding))
        held = d.holding
        if args.every and w.t >= next_print:
            next_print += args.every
            e = None if brain.est is None else round(float(math.hypot(brain.est[0] - pos[0], brain.est[1] - pos[1])), 2)
            print(f"t={w.t:6.1f}    @({pos[0]:+.2f},{pos[1]:+.2f}) yaw={d.yaw(w.data):+.2f} {brain.state} "
                  f"dist-to-est={e} twist={tuple(round(float(v), 2) for v in intent.twist)} note={intent.note}")
        w.step()
        if d.falls > falls:
            falls = d.falls
            print(f"\n=== FALL #{falls} t={w.t:.1f} state={brain.state}  (t, state, pos, gravity, twist, head, skill, tof_min_mm, holding)")
            rows = list(hist)
            for h in rows[::10] + ([rows[-1]] if rows else []):
                print("   ", h)
            p = rows[-1][2] if rows else pos
            near = [(t, round(float(np.hypot(w.data.xpos[b][0] - p[0], w.data.xpos[b][1] - p[1])), 2)) for t, b in w.pickables.items()]
            print(f"   walls: x {hx - abs(p[0]):.2f} m, y {hy - abs(p[1]):.2f} m · basket {np.hypot(p[0] - bx, p[1] - by):.2f} m"
                  f" · toys within 0.35 m: {[n for n in near if n[1] < 0.35]}\n")
        if brain.state != last:
            print(f"t={w.t:6.1f} -> {brain.state:14s} holding={d.holding} falls={d.falls} in-basket={w.tidy_score()['inBasket']}")
            if brain.state == "drop":
                tip = w.mouth_tip(d)
                err = None if brain.est is None else math.hypot(brain.est[0] - bx, brain.est[1] - by)
                print(f"t={w.t:6.1f} RELEASE trunk→basket {np.hypot(pos[0] - bx, pos[1] - by):.3f} m · tip→basket "
                      f"{np.hypot(tip[0] - bx, tip[1] - by):.3f} m · holding={held} · estimate error "
                      f"{'?' if err is None else f'{err:.3f}'} m")
                if held:
                    landings.append((held, w.t + 1.5))
            last = brain.state
        for toy, tt in list(landings):
            if w.t >= tt:
                q = w.data.xpos[w.pickables[toy]]
                print(f"t={w.t:6.1f}   {toy} landed ({q[0] - bx:+.3f}, {q[1] - by:+.3f}) from the basket centre → "
                      f"{'IN' if w.in_basket(toy) else 'OUT'}")
                landings.remove((toy, tt))
        if brain.state == "done":
            break
    s = w.tidy_score()
    print(f"\nFINAL {s['inBasket']}/{s['total']} in the basket · held {s['held']} · picked {brain.picked} · "
          f"delivered {brain.delivered} · falls {d.falls} · {w.t:.1f} s sim · gave up {sorted(brain.given_up)}")


if __name__ == "__main__":
    main()
