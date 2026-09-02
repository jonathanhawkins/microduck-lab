"""Score a brain on the follow task under a sensor preset (roadmap 3.2):

    uv run eval-brain --brain follow --preset hostile --episodes 8
    uv run eval-brain --brain learned:follow-v2 --preset hostile --episodes 8

Metrics per episode: fraction of decisions inside the distance band,
mean |distance error|, fraction of time the target was in sight, bumps
(ToF says something within 22 cm ahead), and falls. The same BrainEnv,
the same seeds, the same person paths — so scripted and learned brains are
compared on identical episodes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import BrainEnv, FollowTask


def run(brain_kind: str, preset: str | None, episodes: int, seed: int, task: FollowTask = FollowTask()) -> dict:
    env = BrainEnv(task, seed=seed, fixed_preset=preset, sense_dr=preset is None)
    brain = REGISTRY.make(brain_kind)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        brain.reset()
        in_band = err = seen = bumps = 0.0
        n = 0
        ret = 0.0
        falls = 0
        while True:
            s: Senses = env.senses()
            if brain.kind.startswith("learned"):
                # The learned brain decides on the same cadence as the env.
                brain._tick = 0
            intent = brain.step(s)
            obs, r, term, trunc, info = env.step(np.array(intent.twist, np.float32))
            ret += r
            n += 1
            in_band += abs(info["dist"] - task.distance) <= task.band
            err += abs(info["dist"] - task.distance)
            seen += info["seen"]
            bumps += info["bumped"]
            if term:
                falls += 1
            if term or trunc:
                break
        rows.append({"return": ret, "in_band": in_band / n, "dist_err": err / n,
                     "seen": seen / n, "bumps": bumps, "falls": falls, "decisions": n})
    keys = ("return", "in_band", "dist_err", "seen", "bumps", "falls")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return {"brain": brain_kind, "preset": preset or "random", "episodes": episodes,
            "variety": bool(task.furniture or task.distractor), "reflex": bool(task.gaze_gain or task.bump_stop),
            **summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", default="follow")
    ap.add_argument("--preset", default=None, help="ideal | datasheet | hostile (default: drawn per episode)")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--variety", action="store_true", help="two free boxes re-scattered each episode + a duck walking a circle")
    ap.add_argument("--no-reflex", action="store_true", help="no gaze and no bump stop under the brain")
    args = ap.parse_args()
    task = FollowTask(furniture=2 if args.variety else 0, distractor=args.variety,
                      gaze_gain=0.0 if args.no_reflex else 0.8, bump_stop=0.0 if args.no_reflex else 0.25)
    res = run(args.brain, args.preset, args.episodes, args.seed, task)
    if args.json:
        print(json.dumps(res))
    else:
        print(f"{res['brain']} @ {res['preset']}{' +variety' if res['variety'] else ''}{'' if res['reflex'] else ' no-reflex'}: in-band {res['in_band']:.2f} · |err| {res['dist_err']:.2f} m · "
              f"seen {res['seen']:.2f} · bumps {res['bumps']:.1f}/ep · falls {res['falls']:.2f}/ep · return {res['return']:.1f}")


if __name__ == "__main__":
    main()
