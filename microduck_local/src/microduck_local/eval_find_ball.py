"""Headless battery for the find_ball brain: how fast does it find the ball,
by where the ball started, does it keep it — and does its BODY do the aiming?

    uv run eval-find-ball runs/<run>/policy.onnx [--episodes 60] [--seconds 8]
    uv run eval-find-ball runs/<run>/policy.onnx --events 0.33   # see below

Bearings are swept uniformly round the circle and reported in three buckets —
front (|psi| < 45 deg), side (45-135) and back (135-180). Same env as the lab
and render-rollout (randomizers off), deterministic ONNX mean: the policy that
ships, not the noise-crutched trainer.

Two tables come out, because the first one cannot see the failure this
behavior actually has:

FINDING — time to first sight, share of steps in frame and centred, falls.

AIMING — the numbers that separate a policy that aims its BODY from one that
just cranks its neck. `find_ball` exists to hand a ball-blind kick a
squared-up duck (`Behavior.handoff_fn`), and a gaze policy scores 100% in
frame and 98% centred while never once satisfying that gate: the shipped
stage-5 export held the ball dead centre using 21 deg of head yaw with its
body 18-20 deg off, and every FINDING column called that a success. So:

  head_yaw|centred  mean |head_yaw| over the steps the ball was centred. The
                    handoff gate wants < 14 deg (_BALL_AIM_HEAD_YAW, 0.25 rad).
  psi_final         |true body bearing to the ball| at the end of the episode.
  psi_turned        how much of the start bearing the body actually turned out.
  handoff           share of episodes where handoff_fn ever fired, and when.
                    This is the deliverable.

`--events` is the other thing worth knowing. It defaults to 0 — a static ball,
one search per episode, so every episode answers one question — but the recipe
TRAINS at 0.33 and `render-rollout` runs at 0.33, and the two regimes disagree
about which policy is safest: an A/B arm measured 0 falls here at `--events 0`
and 1 fall at 0.33, while another went 4 -> 10. Judge falls at `--events 0.33`.
See docs/roadmap.md item 1 for the table this paragraph is summarizing.
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
                prior: float | None = None, events: float = 0.0) -> dict:
    import onnxruntime as ort

    from .behaviors import _BALL_HEAD_YAW_ID, BEHAVIORS, BehaviorEnv, _ball_place, _ball_sense

    sess = ort.InferenceSession(onnx_path)
    name = sess.get_inputs()[0].name
    # The behavior owns the handoff test (one implementation, so this battery,
    # render-rollout --handoff and the lab's showcase duck cannot drift apart).
    handoff_fn = BEHAVIORS["find_ball"].handoff_fn
    overrides = {"MICRODUCK_BALL_EVENT_RATE": str(events)}
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
        hy_centred, handoff_t = [], None
        for _ in range(env.max_steps):
            obs, _, term, trunc, _ = env.step(sess.run(None, {name: obs[None]})[0][0])
            steps += 1
            if env._ball_seen:
                seen_steps += 1
                if first is None:
                    first = steps * C.CTRL_DT
                if abs(env._ball_bx) < 0.25 and abs(env._ball_by) < 0.25:
                    centred += 1
                    # Head yaw is only meaningful as an AIM measure while the
                    # ball is centred: mid-sweep the head is supposed to be
                    # cranked round, and averaging that in would hide the
                    # thing this column exists to expose.
                    hy_centred.append(abs(float(env._joint_qpos()[_BALL_HEAD_YAW_ID])))
            if handoff_t is None and handoff_fn(env):
                handoff_t = steps * C.CTRL_DT
            if term:
                fell = True
                break
            if trunc:
                break
        rows.append({"bearing": math.degrees(bearing), "dist": dist, "first": first,
                     "seen": seen_steps / steps, "centred": centred / steps,
                     "fell": fell, "steps": steps,
                     "start_psi": abs(math.degrees(bearing)),
                     "psi_final": math.degrees(abs(env._ball_psi)),
                     "hy_centred": (math.degrees(float(np.mean(hy_centred)))
                                    if hy_centred else None),
                     "handoff_t": handoff_t})
    return {"rows": rows, "seconds": seconds, "events": events}


def _buckets(rows: list[dict]):
    for label, lo, hi in BUCKETS + (("all", 0.0, 180.001),):
        b = [r for r in rows if lo <= abs(r["bearing"]) < hi]
        if b:
            yield label, b


def summarize(result: dict) -> list[str]:
    """FINDING: does it get eyes on the ball, and keep them there?"""
    lines = [f"{'bucket':6} {'n':>3} {'found':>6} {'t_first med':>12} {'t_first max':>12} "
             f"{'in frame':>9} {'centred':>8} {'fell':>5}"]
    for label, b in _buckets(result["rows"]):
        found = [r["first"] for r in b if r["first"] is not None]
        med = f"{float(np.median(found)):.2f} s" if found else "  never"
        mx = f"{max(found):.2f} s" if found else "  never"
        lines.append(
            f"{label:6} {len(b):3d} {len(found) / len(b):6.0%} {med:>12} {mx:>12} "
            f"{np.mean([r['seen'] for r in b]):9.0%} {np.mean([r['centred'] for r in b]):8.0%} "
            f"{sum(r['fell'] for r in b):5d}")
    return lines


def summarize_aim(result: dict) -> list[str]:
    """AIMING: is it the BODY pointing at the ball, or just the neck?"""
    # Read from the behavior, not retyped: this footer tells the reader the
    # number to beat, and a stale copy makes every reading of the table wrong.
    from .behaviors import _BALL_AIM_HEAD_YAW

    gate = math.degrees(_BALL_AIM_HEAD_YAW)
    lines = [f"{'bucket':6} {'n':>3} {'head_yaw|centred':>17} {'psi_final':>10} "
             f"{'psi_turned':>11} {'handoff':>8} {'t_handoff':>10}"]
    for label, b in _buckets(result["rows"]):
        hys = [r["hy_centred"] for r in b if r["hy_centred"] is not None]
        # "never centred" is not 0 degrees of head yaw — it is no measurement.
        hy = f"{np.mean(hys):.1f} deg" if hys else "  (never centred)"
        hos = [r["handoff_t"] for r in b if r["handoff_t"] is not None]
        ht = f"{np.median(hos):.2f} s" if hos else "  never"
        lines.append(
            f"{label:6} {len(b):3d} {hy:>17} "
            f"{np.mean([r['psi_final'] for r in b]):9.1f}° "
            f"{np.mean([r['start_psi'] - r['psi_final'] for r in b]):10.1f}° "
            f"{len(hos) / len(b):8.0%} {ht:>10}")
    lines.append(f"(head_yaw|centred is the aim gate: the handoff needs < {gate:.0f} deg "
                 f"with the ball centred. psi_turned = start bearing - final.)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("onnx_path")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prior", type=float, default=None,
                    help="MICRODUCK_BALL_PRIOR_PROB for the battery (0 = blind, 1 = always a prior)")
    ap.add_argument("--events", type=float, default=0.0,
                    help="MICRODUCK_BALL_EVENT_RATE. 0 (default) = a static ball, "
                         "one search per episode. 0.33 = what the recipe trains "
                         "and render-rollout runs at — judge FALLS there, the two "
                         "regimes disagree (docs/roadmap.md item 1)")
    args = ap.parse_args()
    if not os.path.exists(args.onnx_path):
        sys.exit(f"{args.onnx_path} not found")
    res = run_battery(args.onnx_path, args.episodes, args.seconds, args.seed,
                      args.prior, args.events)
    ball = "static-ball" if not args.events else f"events={args.events}"
    print(f"find_ball battery: {args.onnx_path}  ({args.episodes} {ball} episodes x "
          f"{args.seconds:.0f} s, prior={'recipe default' if args.prior is None else args.prior})")
    print("\nFINDING — eyes on the ball")
    print("\n".join(summarize(res)))
    print("\nAIMING — is it the body doing it, or the neck?")
    print("\n".join(summarize_aim(res)))


if __name__ == "__main__":
    main()
