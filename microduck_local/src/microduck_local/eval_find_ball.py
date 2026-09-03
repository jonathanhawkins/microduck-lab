"""Headless battery for the find_ball brain: how fast does it find the ball,
by where the ball started, and does it keep it?

    uv run eval-find-ball runs/<run>/policy.onnx [--episodes 60] [--seconds 8]

Ball events are OFF (a static ball, one search per episode) so every episode
answers one question: time to first sight from that bearing. Bearings are
swept uniformly round the circle and reported in three buckets — front
(|psi| < 45 deg), side (45-135) and back (135-180) — with the share of steps
the ball was in frame and centred after the first sight, and the fall count.
Same env as the lab and render-rollout (randomizers off), deterministic ONNX
mean: the policy that ships, not the noise-crutched trainer.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

from . import contract as C

BUCKETS = (("front", 0.0, 45.0), ("side", 45.0, 135.0), ("back", 135.0, 180.001))


def run_battery(onnx_path: str, episodes: int, seconds: float, seed: int,
                prior: float | None = None) -> dict:
    import onnxruntime as ort

    from .behaviors import BehaviorEnv, _ball_place, _ball_sense

    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    overrides = {"MICRODUCK_BALL_EVENT_RATE": "0"}
    if prior is not None:
        overrides["MICRODUCK_BALL_PRIOR_PROB"] = str(prior)
    env = BehaviorEnv("find_ball", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=seed,
                      spawn_overrides=overrides, max_episode_s=seconds)
    rows = []
    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        # Uniform sweep of bearings (both signs), distances across the window.
        bearing = (-1.0 if ep % 2 else 1.0) * math.pi * ((ep // 2) + 0.5) / max(1, episodes // 2)
        dist = float(rng.uniform(0.35, 1.4))
        _ball_place(env, dist, bearing)
        _ball_sense(env, force=True)
        obs = env._get_obs()
        first, seen_steps, centred, steps, fell = None, 0, 0, 0, False
        for _ in range(env.max_steps):
            obs, _, term, trunc, _ = env.step(sess.run(None, {name: obs[None]})[0][0])
            steps += 1
            if env._ball_seen:
                seen_steps += 1
                if first is None:
                    first = steps * C.CTRL_DT
                if abs(env._ball_bx) < 0.25 and abs(env._ball_by) < 0.25:
                    centred += 1
            if term:
                fell = True
                break
            if trunc:
                break
        rows.append({"bearing": math.degrees(bearing), "dist": dist, "first": first,
                     "seen": seen_steps / steps, "centred": centred / steps,
                     "fell": fell, "steps": steps})
    return {"rows": rows, "seconds": seconds}


def summarize(result: dict) -> list[str]:
    rows = result["rows"]
    lines = [f"{'bucket':6} {'n':>3} {'found':>6} {'t_first med':>12} {'t_first max':>12} "
             f"{'in frame':>9} {'centred':>8} {'fell':>5}"]
    for label, lo, hi in BUCKETS + (("all", 0.0, 180.001),):
        b = [r for r in rows if lo <= abs(r["bearing"]) < hi]
        if not b:
            continue
        found = [r["first"] for r in b if r["first"] is not None]
        med = f"{float(np.median(found)):.2f} s" if found else "  never"
        mx = f"{max(found):.2f} s" if found else "  never"
        lines.append(
            f"{label:6} {len(b):3d} {len(found) / len(b):6.0%} {med:>12} {mx:>12} "
            f"{np.mean([r['seen'] for r in b]):9.0%} {np.mean([r['centred'] for r in b]):8.0%} "
            f"{sum(r['fell'] for r in b):5d}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("onnx_path")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prior", type=float, default=None,
                    help="MICRODUCK_BALL_PRIOR_PROB for the battery (0 = blind, 1 = always a prior)")
    args = ap.parse_args()
    if not os.path.exists(args.onnx_path):
        sys.exit(f"{args.onnx_path} not found")
    res = run_battery(args.onnx_path, args.episodes, args.seconds, args.seed, args.prior)
    print(f"find_ball battery: {args.onnx_path}  ({args.episodes} static-ball episodes x "
          f"{args.seconds:.0f} s, prior={'recipe default' if args.prior is None else args.prior})")
    print("\n".join(summarize(res)))


if __name__ == "__main__":
    main()
