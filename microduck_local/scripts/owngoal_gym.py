"""THE OWN-GOAL GYM: one duck, one ball, near ITS OWN MOUTH, repeated.

The thing the user actually complained about is a duck putting the ball in
its own net, and that is the one outcome no instrument in this repo can
resolve: `ownGoals` needs ~347 seeds (item 1.5), and a 240 s match yields
about one goal of which a fifth are own. So own goals were never measured —
backward TOUCHES were measured instead, which is a different thing, and
every arm that cut them left the own-goal share alone or worse.

This puts the duck in the stance that makes them: the ball near our own
mouth, the duck coming at it from up-pitch, so the line of sight points into
our own net. One episode per placement. The row is whether the ball ended in
OUR goal, THEIRS, or neither.

    uv run python owngoal_gym.py --episodes 40 --seeds 8 --jobs 8 \
        --arm 'off=' --arm 'strict=kick_select_t_own=0.02'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "scripts")


def run(seed: int, episodes: int, knobs: str, seconds: float, arm: str) -> list[dict]:
    if knobs:
        os.environ["MICRODUCK_CHASE"] = knobs
    else:
        os.environ.pop("MICRODUCK_CHASE", None)
    from kick_gym import gym_scenario                      # noqa: PLC0415
    from microduck_local.brain import REGISTRY, Senses     # noqa: PLC0415
    from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer  # noqa: PLC0415
    from microduck_local.brain.team import brain_kwargs    # noqa: PLC0415
    from microduck_local.world import World                # noqa: PLC0415

    sc = gym_scenario(opponents=0)
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={x.id: infer for x in sc.ducks}, seed=seed)
    teams: dict = {}
    brains = {x.id: REGISTRY.make("chase", **brain_kwargs(x, w, teams)) for x in sc.ducks}
    brain, d = brains["d0"], w.ducks["d0"]
    live = {k: getattr(brain.p, k) for k in
            ("kick_select_t_own", "kick_select_safest", "kick_select_n", "chase_behind")}
    gx = brain.goal[0] if brain.goal is not None else 1.0
    sgn = 1.0 if gx >= 0 else -1.0                  # +1 when we attack +x
    ours = "left" if sgn > 0 else "right"           # the mouth WE defend
    hx = abs(brain.bounds[0]) if brain.bounds else 1.5
    q = int(w.model.jnt_qposadr[w._ball_joint])
    qv = int(w.model.jnt_dofadr[w._ball_joint])
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        # The ball just off OUR goal line; the duck up-pitch of it, so the
        # line of sight from duck to ball points into our own net.
        bx = -sgn * float(rng.uniform(hx - 0.55, hx - 0.18))
        by = float(rng.uniform(-0.45, 0.45))
        rad = float(rng.uniform(0.40, 0.95))
        off = float(rng.uniform(-0.8, 0.8))
        a = math.atan2(-by, sgn * hx - bx) + off     # from the ball toward THEIR goal, jittered
        dx, dy = bx + rad * math.cos(a), by + rad * math.sin(a)
        d.spawn = (dx, dy, math.atan2(by - dy, bx - dx))
        w._respawn(d)
        w.data.qpos[q:q + 2] = (bx, by)
        w.data.qpos[q + 2] = 0.05
        w.data.qvel[qv:qv + 6] = 0.0
        for b in brains.values():
            b.kickoff()
        seq0, t0, x0 = w.goal_seq, w.t, sgn * bx
        k0, b0 = w.goals_kicked, w.goals_bumped
        own = theirs = False
        while w.t - t0 < seconds and not (own or theirs):
            tof, det = d.tof.last, d.detector.last
            s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                       det=det, det_age=None if det is None else w.t - det.t,
                       speed=d.heading_speed(w.data), odom=w.odom(d), skill=d.skill,
                       bumped=w.bumped(d))
            intent = brain.step(s)
            w.apply_intent(d, intent)
            if d.skill is None:
                d.set_cmd(w.data, intent.twist, intent.head)
            w.step()
            if w.goal_seq != seq0:
                if w.last_goal == ours:
                    own = True
                else:
                    theirs = True
                seq0 = w.goal_seq
        rows.append({"arm": arm, "seed": seed, "ep": ep,
                     "own_goal": own, "their_goal": theirs,
                     "own_kicked": bool(own and w.goals_kicked > k0),
                     "own_bumped": bool(own and w.goals_bumped > b0),
                     "advance": round(sgn * float(w.data.qpos[q]) - x0, 4),
                     "kicks": brain.kicks, "declines": brain.declines,
                     **{f"k_{k}": v for k, v in live.items()}})
        brain.kicks = brain.declines = 0
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=KNOBS")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.arm = a.arm or ["off="]
    tasks = [(s, a.episodes, spec.partition("=")[2], a.seconds, spec.partition("=")[0])
             for spec in a.arm for s in range(a.seed0, a.seed0 + a.seeds)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(run, *zip(*tasks)):
            rows += r
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} episodes over {len(a.arm)} arms x {a.seeds} seeds -> {a.out}")


if __name__ == "__main__":
    main()
