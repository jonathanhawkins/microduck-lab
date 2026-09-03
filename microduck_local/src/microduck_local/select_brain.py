"""Pick a run's best brain by DETERMINISTIC benchmark score, not by reward.

    uv run select-brain follow-v6 --seeds 100,200,300 --jobs 6
    uv run select-brain follow-v6 --dry-run            # score, ship nothing

Why this exists
---------------
`train-brain` writes numbered checkpoints; this scores every one of them —
plus the final model — on the follow benchmark and ships the winner as
`brain.onnx`. Two measured facts make that worth doing:

* `follow-v4`'s reward curve was flat from ~900k of its 2M decisions, and
  every trick run in this workspace "peaked and then came apart". The last
  checkpoint is not reliably the best one.
* The reward curve cannot make the call anyway. It measures the
  noise-crutched STOCHASTIC policy (AGENTS.md, verification discipline #1),
  while `brain.onnx` ships the deterministic mean.

This is deliberately NOT the best-checkpoint selection that `train_behavior`
tried and reverted. That one scored a REWARD TERM and collapsed into
"longest-surviving"; its own comment says to re-introduce selection only on
an achieved-performance criterion. `in_band` is exactly that: the fraction of
decisions the duck actually held the follow distance, measured on the
deterministic export over the benchmark's own episodes.

Scoring runs one process per (checkpoint, seed) pair, so a sweep costs about
as long as its slowest single evaluation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from .brain.learned import brains_dir

# The benchmark's headline metric, then the tie-breaks. `in_band` is the
# fraction of decisions inside the follow band; `seen` the fraction the target
# was in sight. `falls` and `contact` are safety floors, not objectives — a
# checkpoint that falls more is rejected outright rather than traded off.
METRIC = "in_band"
MAX_FALLS = 0.25


def checkpoints(run_dir: Path) -> list[tuple[str, Path, Path]]:
    """(tag, model.zip, vecnormalize.pkl) for every checkpoint, then final.

    The final model is included under the tag "final" so the selection always
    contains the artifact that would otherwise have shipped — the comparison
    is only honest if the incumbent is in it.
    """
    out: list[tuple[str, Path, Path]] = []
    d = run_dir / "checkpoints"
    if d.is_dir():
        for m in sorted(d.glob("model_*.zip")):
            vn = d / f"vecnormalize_{m.stem.split('_', 1)[1]}.pkl"
            if vn.exists():
                out.append((m.stem.split("_", 1)[1], m, vn))
    final = run_dir / "model.zip"
    if final.exists() and (run_dir / "vecnormalize.pkl").exists():
        out.append(("final", final, run_dir / "vecnormalize.pkl"))
    return out


def _cached_export(export_fn, run_dir: Path, model_path: Path, vn_path: Path,
                   out: Path) -> bool:
    """Export only if the cached ONNX is older than what it was built from.

    Every invocation used to re-export every candidate through
    `torch.onnx.export`, which is the slowest part of a scoring pass and is a
    pure function of (checkpoint, normalizer, exporter). Re-scoring a run
    under different seeds or presets — which is the normal way to use these
    tools — paid for all of it again. Returns True when it exported.
    """
    if out.exists():
        stamp = out.stat().st_mtime_ns
        fresh = all(p.stat().st_mtime_ns <= stamp for p in (model_path, vn_path))
        if fresh:
            return False
    export_fn(run_dir, model_path=model_path, vn_path=vn_path, out=out)
    return True


def _score_one(job: tuple[str, str, int, int, bool, str | None, float]) -> dict:
    """One (checkpoint, seed) evaluation, in its own single-threaded process."""
    probe_dir, tag, seed, episodes, variety, preset, polite = job
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    from .brain.brain_env import FollowTask
    from .eval_brain import run
    task = FollowTask(furniture=2 if variety else 0, distractor=variety, polite=polite)
    res = run(f"learned:{probe_dir}", preset, episodes, seed, task)
    return {"tag": tag, "seed": seed,
            **{k: float(res[k]) for k in ("in_band", "seen", "falls", "contact", "bumps", "return")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_name", help="brains/<run_name>")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per seed")
    ap.add_argument("--seeds", default="100,200,300",
                    help="comma-separated eval seeds (the follow table uses 100..1000 step 100)")
    ap.add_argument("--preset", default=None,
                    help="ideal | datasheet | hostile, or a comma list to AVERAGE over "
                         "(default: drawn per episode). The follow table reports both "
                         "datasheet and hostile and brains rank differently across them, "
                         "so '--preset datasheet,hostile' picks a checkpoint that is good "
                         "under both rather than one tuned to a single noise model")
    ap.add_argument("--variety", action="store_true")
    ap.add_argument("--polite", type=float, default=0.55)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--dry-run", action="store_true", help="score and print; ship nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run_dir = brains_dir() / args.run_name
    cks = checkpoints(run_dir)
    if not cks:
        raise SystemExit(f"no checkpoints or final model under {run_dir}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    # One preset, a comma list to average over, or None to draw per episode.
    presets = ([x.strip() for x in args.preset.split(",") if x.strip()]
               if args.preset else [None])

    # Export each candidate into its own probe dir, carrying the run's
    # brain.json so the env is built at the observation version this brain
    # was TRAINED at (a version-1 brain scored under a reflex tier it never
    # saw measured 0.13 worse — the world has to match the training).
    from .train_brain import export_brain
    meta = json.loads((run_dir / "brain.json").read_text()) if (run_dir / "brain.json").exists() else {}
    probes = run_dir / ".probe"
    jobs, exported = [], 0
    for tag, model_path, vn_path in cks:
        d = probes / tag
        d.mkdir(parents=True, exist_ok=True)
        exported += _cached_export(export_brain, run_dir, model_path, vn_path,
                                   d / "brain.onnx")
        (d / "brain.json").write_text(json.dumps({**meta, "name": f"{args.run_name}@{tag}"}, indent=2))
        for preset in presets:
            for seed in seeds:
                jobs.append((str(d), tag, seed, args.episodes, args.variety, preset,
                             args.polite))

    print(f"[select-brain] {len(cks)} candidates x {len(presets)} preset(s) x "
          f"{len(seeds)} seeds x {args.episodes} episodes "
          f"= {len(jobs)} evaluations on {args.jobs} processes "
          f"({exported} exported, {len(cks) - exported} cached)", flush=True)
    if args.jobs > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            rows = pool.map(_score_one, jobs)
    else:
        rows = [_score_one(j) for j in jobs]

    table = []
    for tag, _, _ in cks:
        mine = [r for r in rows if r["tag"] == tag]
        table.append({"tag": tag, "seeds": len(mine),
                      **{k: float(np.mean([r[k] for r in mine]))
                         for k in ("in_band", "seen", "falls", "contact", "bumps", "return")}})

    # Falls are a REJECTION FLOOR, not a term traded against the band: a
    # brain that falls is not a better follower for holding the distance
    # while it does. If nothing clears the floor, nothing is shipped — the
    # fastest faller is not a winner.
    safe = [r for r in table if r["falls"] <= MAX_FALLS]
    viable = bool(safe)
    best = max(safe or table, key=lambda r: (r[METRIC], r["seen"]))
    final = next(r for r in table if r["tag"] == "final")

    # Ship BEFORE reporting, so `--json` emits exactly one object on stdout
    # (a trailing human line after it makes the output unparseable, which is
    # how a caller reading the last line finds out).
    shipped = None
    if not args.dry_run and not viable:
        print(f"\n  !! every candidate falls more than {MAX_FALLS} times an episode — "
              "nothing shipped.")
    if not args.dry_run and viable:
        shutil.copy(probes / best["tag"] / "brain.onnx", run_dir / "brain.onnx")
        shipped = {"tag": best["tag"], "metric": METRIC, "score": round(best[METRIC], 4),
                   "final_score": round(final[METRIC], 4), "seeds": seeds,
                   "episodes": args.episodes, "presets": presets,
                   "variety": args.variety, "polite": args.polite}
        meta["selected"] = shipped
        (run_dir / "brain.json").write_text(json.dumps(meta, indent=2))

    if args.json:
        print(json.dumps({"run": args.run_name, "metric": METRIC, "best": best,
                          "final": final, "table": table, "rows": rows,
                          "viable": viable, "shipped": shipped}))
    else:
        print(f"\n  {'checkpoint':>12}  {'in_band':>8} {'seen':>6} {'falls':>6} {'bumps':>6} {'return':>8}")
        for r in table:
            mark = " <- best" if r["tag"] == best["tag"] else ""
            print(f"  {r['tag']:>12}  {r['in_band']:8.3f} {r['seen']:6.3f} {r['falls']:6.2f} "
                  f"{r['bumps']:6.1f} {r['return']:8.1f}{mark}")
        gain = best[METRIC] - final[METRIC]
        print(f"\n  best is '{best['tag']}': {METRIC} {best[METRIC]:.3f} vs final {final[METRIC]:.3f} "
              f"({gain:+.3f})")
        if shipped:
            print(f"  shipped brains/{args.run_name}/brain.onnx from checkpoint {best['tag']}")


if __name__ == "__main__":
    main()
