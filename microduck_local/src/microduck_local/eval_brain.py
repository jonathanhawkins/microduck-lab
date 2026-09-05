"""Score a brain on the follow task under a sensor preset (roadmap 3.2):

    uv run eval-brain --brain follow --preset hostile --episodes 8
    uv run eval-brain --brain learned:follow-v2 --preset hostile --episodes 8
    uv run eval-brain --brain follow --preset hostile --episodes 24 --jobs 0

Metrics per episode: fraction of decisions inside the distance band,
mean |distance error|, fraction of time the target was in sight, bumps
(ToF says something within 22 cm ahead), and falls. The same BrainEnv,
the same seeds, the same person paths — so scripted and learned brains are
compared on identical episodes.

An episode is a pure function of (seed, ep) — BrainEnv.reset() re-seeds
everything it touches — so `--jobs N` splits the battery over N processes
and returns exactly the rows `--jobs 1` returns, in the same order.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import BRAIN_OBS_VERSION, BrainEnv, FollowTask


def obs_version_of(brain) -> int:
    """The observation version the env should run for this brain: a learned
    brain's own (its brain.json), so a version-1 brain — trained with no
    tracker, no gaze, no bump stop — is scored in the world it was trained
    in (measured: scored under the reflex tier it never saw, follow-v1 fell
    from 0.73 to 0.60 in band); everything else gets the current version."""
    return int(getattr(brain, "obs_version", BRAIN_OBS_VERSION))


def _build(brain_kind: str, preset: str | None, seed: int, task: FollowTask, avoid: bool):
    """The (brain, env) pair episodes are scored on. Built once and reused:
    BrainEnv.reset() re-seeds everything an episode touches, so the env
    carries nothing from the episode before it and rebuilding one per
    episode would buy nothing — it only adds the build, which against a
    20 s episode measured +7% on an M-series Mac and +24% on a 4-core
    Linux box (the ONNX session is most of it)."""
    brain = REGISTRY.make(brain_kind)
    if hasattr(brain, "p") and hasattr(brain.p, "avoid"):
        from dataclasses import replace
        brain.p = replace(brain.p, avoid=avoid)          # the scripted follow's own dodge
    env = BrainEnv(task, seed=seed, fixed_preset=preset, sense_dr=preset is None, obs_version=obs_version_of(brain))
    return brain, env


def _episode(brain, env, task: FollowTask, seed: int, ep: int) -> dict:
    """One episode's row — a pure function of (seed, ep). The env is built
    from `seed` and reset to `seed + ep`, and neither reads anything the
    episode before it left behind, so the row does not depend on which
    worker ran it or on what ran there first. `--jobs N` rests on that; the
    exactness test in tests/test_eval_brain_jobs.py pins it."""
    obs, _ = env.reset(seed=seed + ep)
    brain.reset()
    in_band = err = seen = bumps = contact = 0.0
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
        contact += info["contact"]
        if term:
            falls += 1
        if term or trunc:
            break
    return {"return": ret, "in_band": in_band / n, "dist_err": err / n,
            "seen": seen / n, "bumps": bumps, "contact": contact / 10.0, "falls": falls, "decisions": n,
            "dodges": int(brain.closing.count if hasattr(brain, "closing") else info.get("dodges", 0))}


def _run_chunk(payload) -> tuple[list[tuple[int, dict]], int]:
    """A worker's whole share: ONE env, every episode it was given. The
    episodes are numbered, so the caller can put the table back in order,
    and the env's charge tally rides back with them."""
    brain_kind, preset, seed, task, avoid, eps = payload
    brain, env = _build(brain_kind, preset, seed, task, avoid)
    return [(ep, _episode(brain, env, task, seed, ep)) for ep in eps], int(env.charges)


def _chunks(episodes: int, jobs: int) -> list[list[int]]:
    """Episodes dealt round-robin, so a worker that draws a run of short
    episodes does not finish early while another still has all the long
    ones."""
    out = [list(range(j, episodes, jobs)) for j in range(jobs)]
    return [c for c in out if c]


def _run_parallel(brain_kind: str, preset: str | None, episodes: int, seed: int,
                  task: FollowTask, avoid: bool, jobs: int) -> tuple[list[dict], int]:
    payloads = [(brain_kind, preset, seed, task, avoid, c) for c in _chunks(episodes, jobs)]
    numbered: list[tuple[int, dict]] = []
    charges = 0
    # "spawn", not the fork Linux would pick: a worker forked from a parent
    # that has already built an env inherits ONNX Runtime's thread state
    # without its threads, and the child's first InferenceSession deadlocks
    # (a `run(jobs=1)` before a `run(jobs=N)` in one process hung forever).
    # macOS spawns anyway, so this is the path that has to work regardless.
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=mp.get_context("spawn")) as ex:
        for part, ch in ex.map(_run_chunk, payloads):
            numbered += part
            charges += ch
    return [row for _, row in sorted(numbered, key=lambda kv: kv[0])], charges


def run(brain_kind: str, preset: str | None, episodes: int, seed: int, task: FollowTask = FollowTask(),
        avoid: bool = False, jobs: int = 1) -> dict:
    jobs = max(1, min(int(jobs), episodes))
    if jobs > 1:
        rows, charges = _run_parallel(brain_kind, preset, episodes, seed, task, avoid, jobs)
        obs_version = obs_version_of(REGISTRY.make(brain_kind))
    else:
        brain, env = _build(brain_kind, preset, seed, task, avoid)
        rows = [_episode(brain, env, task, seed, ep) for ep in range(episodes)]
        charges, obs_version = int(env.charges), env.obs_version
    keys = ("return", "in_band", "dist_err", "seen", "bumps", "contact", "falls", "dodges")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return {"brain": brain_kind, "preset": preset or "random", "episodes": episodes,
            "variety": bool(task.furniture or task.distractor), "charge": task.charge, "polite": task.polite,
            "avoid": bool(task.avoid or avoid),
            "charges": charges,
            "reflex": bool((task.gaze_gain or task.bump_stop) and obs_version >= 2),
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
    ap.add_argument("--charge", type=float, default=0.0, metavar="S", help="the person walks straight at the duck every S seconds")
    ap.add_argument("--avoid", action="store_true", help="the dodge (ClosingWatch): the scripted follow's own, or the reflex tier's under a learned brain")
    ap.add_argument("--polite", type=float, default=0.55, metavar="M", help="the person stops M m (centre to centre) short of the duck in its way and steps around after 2.5 s; 0: walks through it")
    ap.add_argument("--jobs", type=int, default=1, metavar="N", help=f"score the episodes on N processes (0: one per core, {os.cpu_count()} here). Every row is identical to --jobs 1's.")
    args = ap.parse_args()
    jobs = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    task = FollowTask(furniture=2 if args.variety else 0, distractor=args.variety,
                      gaze_gain=0.0 if args.no_reflex else 0.8, bump_stop=0.0 if args.no_reflex else 0.25,
                      charge=args.charge, avoid=args.avoid and not args.brain == "follow", polite=args.polite)
    res = run(args.brain, args.preset, args.episodes, args.seed, task, avoid=args.avoid, jobs=jobs)
    if args.json:
        print(json.dumps(res))
    else:
        tags = (" +variety" if res["variety"] else "") + ("" if res["reflex"] else " no-reflex") \
            + (f" +charge {res['charge']:g}s" if res["charge"] else "") + (f" +polite {res['polite']:g}m" if res.get("polite") else "") \
            + (" +avoid" if res["avoid"] else "")
        print(f"{res['brain']} @ {res['preset']}{tags}: in-band {res['in_band']:.2f} · |err| {res['dist_err']:.2f} m · "
              f"seen {res['seen']:.2f} · bumps {res['bumps']:.1f}/ep · contact {res['contact']:.1f} s/ep · dodges {res['dodges']:.1f}/ep · "
              f"falls {res['falls']:.2f}/ep · return {res['return']:.1f}")


if __name__ == "__main__":
    main()
