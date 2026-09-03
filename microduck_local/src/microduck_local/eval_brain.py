"""Score a brain on the follow task under a sensor preset (roadmap 3.2):

    uv run eval-brain --brain follow --preset hostile --episodes 8
    uv run eval-brain --brain learned:follow-v2 --preset hostile --episodes 8
    uv run eval-brain --brain follow --preset hostile --episodes 24 --jobs 6

Metrics per episode: fraction of decisions inside the distance band,
mean |distance error|, fraction of time the target was in sight, bumps
(ToF says something within 22 cm ahead), and falls. The same BrainEnv,
the same seeds, the same person paths — so scripted and learned brains are
compared on identical episodes.

`--jobs N` and what it costs (READ THIS BEFORE PUBLISHING A NUMBER)
-------------------------------------------------------------------
The episodes of one `run()` are NOT independent, so they cannot be split
across processes and still come out bit-identical. `env.reset(seed=…)`
reseeds `env.rng` — the person's path, speed and spawn, the furniture
scatter, the per-episode noise preset — but two things ride from one
episode into the next and neither is seeded:

* the SENSORS' own generators. `TofSensor.rng` and `Detector.rng` are
  seeded once when the world is built, and neither sensor's `reset()` nor
  `World.reset()` touches them again, so episode k's noise starts wherever
  episode k-1 left off.
* the duck's last COMMAND. `World._respawn` clears `last_action`, the
  gains and the sensors but not `twist_cmd` / `head_cmd`, so the five
  warm-up steps `BrainEnv.reset()` runs "to let the first sensor frames
  land" are driven by whatever the previous episode was asking for when it
  ended — a different pose under the first decision.

Measured: a fresh env reproduces episode 0 bit for bit and diverges from
episode 1 on. And the chain cannot be parallelised at all — episode k-1
must be *simulated* before episode k's starting state exists, so every
exact schedule has a makespan of the whole serial chain however many cores
it is given.

So `--jobs N` (N > 1) buys its speed by changing what is measured: each
episode gets its OWN `BrainEnv` and its own brain (26 ms against a 356 ms
20 s episode — 7%, against an N-fold win), which makes an episode a pure
function of `(seed, ep)` and the battery a pure function of
`(seed, episodes)` — identical for every N ≥ 2, whatever the core count.
It is a different noise REALISATION, not a different distribution and not
a bias, but it is a different number: over 24 episodes it moved `in_band`
by ≤0.02 and `seen` by up to 0.04 under `hostile`. The published follow
table was measured on the chained path, so keep `--jobs 1` (the default,
and byte-for-byte the original serial loop) for anything that goes in it.
Every result carries `sampling` ("chained" or "independent") and `jobs`,
and the printed line says so too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .brain import REGISTRY, Senses
from .brain.brain_env import BRAIN_OBS_VERSION, BrainEnv, FollowTask

SUMMARY_KEYS = ("return", "in_band", "dist_err", "seen", "bumps", "contact", "falls", "dodges")


def obs_version_of(brain) -> int:
    """The observation version the env should run for this brain: a learned
    brain's own (its brain.json), so a version-1 brain — trained with no
    tracker, no gaze, no bump stop — is scored in the world it was trained
    in (measured: scored under the reflex tier it never saw, follow-v1 fell
    from 0.73 to 0.60 in band); everything else gets the current version."""
    return int(getattr(brain, "obs_version", BRAIN_OBS_VERSION))


def _make(brain_kind: str, preset: str | None, seed: int, task: FollowTask, avoid: bool):
    brain = REGISTRY.make(brain_kind)
    if hasattr(brain, "p") and hasattr(brain.p, "avoid"):
        from dataclasses import replace
        brain.p = replace(brain.p, avoid=avoid)          # the scripted follow's own dodge
    env = BrainEnv(task, seed=seed, fixed_preset=preset, sense_dr=preset is None, obs_version=obs_version_of(brain))
    return brain, env


def _episode(env: BrainEnv, brain, task: FollowTask) -> dict:
    """One episode's row, from an env and a brain that were both just reset.
    Extracted so the serial and the parallel paths cannot drift apart."""
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


def _single_threaded() -> None:
    """One core per worker. ONNX Runtime is already pinned to one thread in
    `brain_env.onnx_infer`; this is the BLAS/OpenMP half (same guard as
    select_brain's scorer)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _run_episodes(brain_kind: str, preset: str | None, eps: list[int], seed: int,
                  task: FollowTask, avoid: bool) -> dict:
    """A worker's share of the battery. Each episode gets a brand new env
    and brain, which is the only way a row depends on `(seed, ep)` alone and
    not on which worker drew it or how many there were: the env's carried
    state is not one thing to clear but a class of them (the sensor
    generators and the leftover walk command today — see the module
    docstring), and a fresh build is immune to the next one too."""
    _single_threaded()
    rows, charges, obs_version = [], 0, BRAIN_OBS_VERSION
    for ep in eps:
        brain, env = _make(brain_kind, preset, seed, task, avoid)
        env.reset(seed=seed + ep)
        brain.reset()
        rows.append((ep, _episode(env, brain, task)))
        charges += int(getattr(env, "charges", 0))
        obs_version = env.obs_version
    return {"rows": rows, "charges": charges, "obs_version": obs_version}


def _result(brain_kind: str, preset: str | None, episodes: int, task: FollowTask, avoid: bool,
            charges: int, obs_version: int, rows: list[dict], jobs: int) -> dict:
    summary = {k: float(np.mean([r[k] for r in rows])) for k in SUMMARY_KEYS}
    return {"brain": brain_kind, "preset": preset or "random", "episodes": episodes,
            "variety": bool(task.furniture or task.distractor), "charge": task.charge, "polite": task.polite,
            "avoid": bool(task.avoid or avoid),
            "charges": int(charges),
            "reflex": bool((task.gaze_gain or task.bump_stop) and obs_version >= 2),
            "jobs": int(jobs), "sampling": "chained" if jobs <= 1 else "independent",
            **summary, "rows": rows}


def run(brain_kind: str, preset: str | None, episodes: int, seed: int, task: FollowTask = FollowTask(),
        avoid: bool = False, jobs: int = 1) -> dict:
    """`jobs=1` (default): the chained serial battery the follow table was
    measured on. `jobs>1`: the same episodes over that many processes, each
    seeded independently — the same for every jobs>1, NOT the same as
    jobs=1. The module docstring says why, and the result says which."""
    if jobs > 1 and episodes > 1:
        return _run_parallel(brain_kind, preset, episodes, seed, task, avoid, jobs)
    brain, env = _make(brain_kind, preset, seed, task, avoid)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        brain.reset()
        rows.append(_episode(env, brain, task))
    return _result(brain_kind, preset, episodes, task, avoid,
                   getattr(env, "charges", 0), env.obs_version, rows, jobs=1)


def _run_parallel(brain_kind: str, preset: str | None, episodes: int, seed: int,
                  task: FollowTask, avoid: bool, jobs: int) -> dict:
    import multiprocessing as mp
    # spawn/forkserver: no forked ONNX runtime or MuJoCo state (the Windows-safe rule).
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
    n = min(jobs, episodes)
    # Round robin, not contiguous blocks: an episode that ends in a fall is
    # short, and the shares stay even when the falls cluster.
    shares = [list(range(episodes))[k::n] for k in range(n)]
    args = [(brain_kind, preset, eps, seed, task, avoid) for eps in shares if eps]
    with ctx.Pool(len(args), initializer=_single_threaded) as pool:
        out = pool.starmap(_run_episodes, args)
    rows = [row for _, row in sorted((pair for o in out for pair in o["rows"]), key=lambda p: p[0])]
    return _result(brain_kind, preset, episodes, task, avoid,
                   sum(o["charges"] for o in out), out[0]["obs_version"], rows, jobs=jobs)


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
    ap.add_argument("--jobs", type=int, default=1,
                    help="episodes run in this many processes (each single-threaded). NOT the same numbers as "
                         "--jobs 1: an episode inherits the sensors' RNG streams and the duck's last walk command "
                         "from the one before it, and that chain cannot be split, so jobs>1 gives every episode its "
                         "own env instead (same numbers for every jobs>1, whatever the core count). Keep --jobs 1 "
                         "for the published follow table")
    args = ap.parse_args()
    task = FollowTask(furniture=2 if args.variety else 0, distractor=args.variety,
                      gaze_gain=0.0 if args.no_reflex else 0.8, bump_stop=0.0 if args.no_reflex else 0.25,
                      charge=args.charge, avoid=args.avoid and not args.brain == "follow", polite=args.polite)
    if args.jobs > 1:
        print(f"[eval-brain] --jobs {args.jobs}: episodes seeded independently, not the chained sampling "
              f"--jobs 1 measures (see eval_brain.py). Same numbers for every --jobs > 1.", file=sys.stderr, flush=True)
    res = run(args.brain, args.preset, args.episodes, args.seed, task, avoid=args.avoid, jobs=args.jobs)
    if args.json:
        print(json.dumps(res))
    else:
        tags = (" +variety" if res["variety"] else "") + ("" if res["reflex"] else " no-reflex") \
            + (f" +charge {res['charge']:g}s" if res["charge"] else "") + (f" +polite {res['polite']:g}m" if res.get("polite") else "") \
            + (" +avoid" if res["avoid"] else "") + ("" if res["sampling"] == "chained" else f" [{res['sampling']} sampling]")
        print(f"{res['brain']} @ {res['preset']}{tags}: in-band {res['in_band']:.2f} · |err| {res['dist_err']:.2f} m · "
              f"seen {res['seen']:.2f} · bumps {res['bumps']:.1f}/ep · contact {res['contact']:.1f} s/ep · dodges {res['dodges']:.1f}/ep · "
              f"falls {res['falls']:.2f}/ep · return {res['return']:.1f}")


if __name__ == "__main__":
    main()
