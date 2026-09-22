"""THE WRONG-SIDE GYM: one duck, one still ball, and the duck starts on the
GOAL SIDE of it — the exact stance every arm in this session was built for,
repeated cheaply.

`kick_gym._place` always spawns the duck BEHIND the ball, which is the case
that already works. This places it up-pitch instead: the ball between the
duck and the duck's OWN goal, so anything the duck does badly sends the ball
the wrong way. One episode = one placement + `--seconds` of play; the row is
how far the ball ended up toward the goal the duck attacks.

The point is events per CPU-second. A 240 s 2v2 pitch run yields ~35 touches
of which ~9 are from the wrong side; this yields one clean wrong-side trial
every few seconds, which is what it takes to resolve an effect this small.

    uv run python wrongside_gym.py --episodes 40 --seeds 6 --jobs 6 \
        --arm 'off=' --arm 'side=chase_behind=0.40' --out /tmp/ws.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mujoco
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
    live = {"chase_behind": brain.p.chase_behind, "behind_ball": brain.p.behind_ball,
            "approach_keepout": brain.p.approach_keepout, "team_side_s": brain.p.team_side_s}
    gx = brain.goal[0] if brain.goal is not None else 1.0
    sgn = 1.0 if gx >= 0 else -1.0
    q = int(w.model.jnt_qposadr[w._ball_joint])
    qv = int(w.model.jnt_dofadr[w._ball_joint])
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        # The ball somewhere central; the duck UP-PITCH of it, within a
        # drawn angle of the goal direction, at a drawn walk-in range.
        bx = float(rng.uniform(-0.6, 0.2)) * sgn
        by = float(rng.uniform(-0.5, 0.5))
        rad = float(rng.uniform(0.45, 1.2))
        off = float(rng.uniform(-1.0, 1.0))              # rad off the goal direction: the wrong-side cone
        a = math.atan2(0.0 - by, gx - bx) + off
        dx, dy = bx + rad * math.cos(a), by + rad * math.sin(a)
        d.spawn = (dx, dy, math.atan2(by - dy, bx - dx))  # facing the ball
        w._respawn(d)
        w.data.qpos[q:q + 2] = (bx, by)
        w.data.qpos[q + 2] = 0.05
        w.data.qvel[qv:qv + 6] = 0.0
        # Placing writes qpos/qvel; without this the DERIVED state (xpos,
        # xquat, sensordata) still describes the end of the previous episode,
        # so the first step builds obs from a stale gyro and gravity vector
        # and `set_cmd` seeds `_hold_yaw` from the heading the duck had in the
        # last episode. Every _place* in kick_gym ends the same way.
        mujoco.mj_forward(w.model, w.data)
        for b in brains.values():
            b.kickoff()
        t0, x0 = w.t, sgn * bx
        kicks0, falls0, touched = brain.kicks, d.falls, False
        while w.t - t0 < seconds:
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
            if not touched and math.hypot(float(w.data.qvel[qv]), float(w.data.qvel[qv + 1])) > 0.12:
                touched = True
        rows.append({"arm": arm, "seed": seed, "ep": ep, "off": round(off, 3), "rad": round(rad, 3),
                     "advance": round(sgn * float(w.data.qpos[q]) - x0, 4),
                     "touched": touched, "kicks": brain.kicks - kicks0,
                     "fell": int(d.falls) - falls0,
                     **{f"k_{k}": v for k, v in live.items()}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=KNOBS")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None,
                    help="stamp every row and RESUME: re-running the same command "
                         "skips (arm, seed) pairs already in --out, and a file "
                         "written under a different tag is refused")
    a = ap.parse_args()
    a.arm = a.arm or ["off="]
    tasks = []
    for spec in a.arm:
        label, _, knobs = spec.partition("=")
        for s in range(a.seed0, a.seed0 + a.seeds):
            tasks.append((s, a.episodes, knobs, a.seconds, label))
    # A battery must survive the machine (microduck_local/AGENTS.md): append
    # each seed as it lands, skip what is already banked, and refuse a tag
    # that disagrees so two variants can never be stitched into one file.
    done: set[tuple] = set()
    if a.out and a.tag and os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("tag") != a.tag:
                    raise SystemExit(
                        f"{a.out} holds tag {r.get('tag')!r}, not {a.tag!r} — "
                        f"write a different --out or use the original --tag")
                done.add((r["arm"], r["seed"]))
    tasks = [t for t in tasks if (t[4], t[0]) not in done]
    if done:
        print(f"resuming: {len(done)} (arm, seed) pairs already in {a.out}")

    n = 0
    out = open(a.out, "a") if a.out else None
    try:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(run, *zip(*tasks)) if tasks else ():
                for row in r:
                    n += 1
                    if out is not None:
                        out.write(json.dumps({**row, "tag": a.tag} if a.tag else row) + "\n")
                        out.flush()          # …so an interrupt keeps what landed
    finally:
        if out is not None:
            out.close()
    print(f"{n} episodes over {len(a.arm)} arms x {a.seeds} seeds -> {a.out}")


if __name__ == "__main__":
    main()
