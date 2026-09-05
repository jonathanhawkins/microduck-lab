"""Pick a locomotion run's shipping policy by ACHIEVED GROUND SPEED.

    uv run select-run runs/my-run --episodes 8 --jobs 8 --dry-run

Why this criterion, specifically
--------------------------------
`train_behavior.py` carries a comment about the last time best-checkpoint
selection was tried here, and it is the specification for this file:

    NO best-checkpoint selection. It was tried and it made things WORSE:
    scoring on `keep_pace * ep_len` is ~90% ep_len ... so it collapsed into
    "longest-surviving", and `_run_speed` pays 0.29 at ZERO velocity — a
    motionless duck scores well. ... Re-introduce only with a criterion
    scored on ACHIEVED GROUND SPEED, not on a reward term.

So nothing here reads a reward term. Each candidate is exported to ONNX,
rolled out DETERMINISTICALLY, and scored on the mean body-frame forward
velocity it actually reached — the same quantity `eval-walk` prints as
"achieved body-x speed". Survival is a REJECTION FLOOR rather than a term in
the score: a policy that falls more than `--max-falls` of the time is dropped
outright instead of trading its falls against its speed, which is exactly the
trade that produced "longest-surviving" last time.

A standing policy therefore scores ~0, not 0.29.

For brains use `select-brain` (the follow benchmark's in-band fraction);
this is its counterpart for anything that is supposed to travel.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import numpy as np

from .train import RUNS_DIR

METRIC = "speed"


def checkpoints(run_dir: Path) -> list[tuple[str, Path, Path]]:
    """(tag, model.zip, vecnormalize.pkl) for every checkpoint, then final.

    The final model is always a candidate: a selection that cannot choose the
    artifact which would otherwise have shipped is not a comparison.
    """
    out: list[tuple[str, Path, Path]] = []
    d = run_dir / "checkpoints"
    if d.is_dir():
        for m in sorted(d.glob("model_*.zip")):
            tag = m.stem.split("_", 1)[1]
            vn = d / f"vecnormalize_{tag}.pkl"
            if vn.exists():
                out.append((tag, m, vn))
    if (run_dir / "model.zip").exists() and (run_dir / "vecnormalize.pkl").exists():
        out.append(("final", run_dir / "model.zip", run_dir / "vecnormalize.pkl"))
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


def _score_one(job) -> dict:
    """Roll ONE exported candidate out deterministically. Own process."""
    onnx_path, tag, seed, episodes, behavior, cmd, actuator = job
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    import onnxruntime as ort

    from .behaviors import BEHAVIORS, BehaviorEnv
    b = BEHAVIORS[behavior]
    kw = dict(obs_noise=True, domain_rand=True, action_delay=True, random_yaw=True,
              seed=seed, max_episode_s=b.episode_s)
    if b.forward_cmd:
        kw["height_termination"] = False
        kw["actuator"] = actuator or "bam"
    elif actuator:
        kw["actuator"] = actuator
    # $MICRODUCK_RUN_CMD is read by the command sampler on EVERY episode
    # reset, not at construction, so it has to stay set for the whole scoring
    # call — and be restored at the end. This normally runs in its own pool
    # process where a leak would be harmless, but it is also called in-process
    # (tests, one-off scoring), and a stray pinned command silently changes
    # what every later env in that process samples. Both halves were learned
    # the hard way: leaking it broke a command-mix test in another file, and
    # the first fix restored it too early, which un-pinned the command and
    # dropped the shipped walker from 0.196 to 0.130 m/s.
    prev = os.environ.get("MICRODUCK_RUN_CMD")
    if cmd is not None:
        os.environ["MICRODUCK_RUN_CMD"] = str(cmd)
    try:
        env = BehaviorEnv(behavior, **kw)
        sess = ort.InferenceSession(str(onnx_path))
        name = sess.get_inputs()[0].name

        fwds, lens, falls = [], [], 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            term = trunc = False
            while not (term or trunc):
                # The DETERMINISTIC mean action — the policy that ships. A
                # reward curve measures the noise-crutched stochastic one.
                a = sess.run(None, {name: obs[None]})[0][0].astype(np.float32)
                obs, _, term, trunc, _ = env.step(a)
                fwds.append(float(env.body_lin_vel()[0]))
            lens.append(env.step_count)
            falls += int(term)
    finally:
        if cmd is not None:
            if prev is None:
                os.environ.pop("MICRODUCK_RUN_CMD", None)
            else:
                os.environ["MICRODUCK_RUN_CMD"] = prev
    return {"tag": tag, "seed": seed, "speed": float(np.mean(fwds)),
            "ep_len": float(np.mean(lens)), "falls": falls / max(episodes, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="runs/<name> (or a bare name under runs/)")
    ap.add_argument("--behavior", default=None,
                    help="behavior id (default: read from the run's behavior.json)")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per seed")
    ap.add_argument("--seeds", default="123,456,789")
    ap.add_argument("--cmd", type=float, default=None, help="pinned forward command")
    ap.add_argument("--actuator", default=None, choices=("xml", "bam"))
    ap.add_argument("--max-falls", type=float, default=0.25, metavar="F",
                    help="reject a candidate that falls more than this fraction of "
                         "episodes. A FLOOR, not a term: trading falls against speed "
                         "is what collapsed the last attempt into longest-surviving")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run_dir = args.run if args.run.exists() else RUNS_DIR / str(args.run)
    cks = checkpoints(run_dir)
    if not cks:
        raise SystemExit(f"no checkpoints or final model under {run_dir} "
                         "(train with --checkpoint-every to keep them)")
    behavior = args.behavior
    if behavior is None:
        meta_path = run_dir / "behavior.json"
        if not meta_path.exists():
            raise SystemExit(f"no behavior.json in {run_dir}; pass --behavior")
        behavior = json.loads(meta_path.read_text())["behavior"]
    from .behaviors import BEHAVIORS
    if not BEHAVIORS[behavior].forward_cmd:
        raise SystemExit(
            f"'{behavior}' is not a locomotion recipe, so ground speed is not its "
            "criterion. This tool deliberately has no fallback: a trick needs a "
            "measure of the trick, and picking one by reward term is what failed "
            "before (see the module docstring).")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    from .export_onnx import export
    probes = run_dir / ".probe"
    jobs, exported = [], 0
    for tag, model_path, vn_path in cks:
        onnx = probes / tag / "policy.onnx"
        onnx.parent.mkdir(parents=True, exist_ok=True)
        # export()'s signature puts `out` second; adapt it to the shared helper.
        exported += _cached_export(
            lambda rd, model_path, vn_path, out: export(rd, out, model_path=model_path,
                                                        vn_path=vn_path),
            run_dir, model_path, vn_path, onnx)
        for sd in seeds:
            jobs.append((str(onnx), tag, sd, args.episodes, behavior, args.cmd,
                         args.actuator))

    print(f"[select-run] {len(cks)} candidates x {len(seeds)} seeds x {args.episodes} "
          f"episodes = {len(jobs)} rollouts on {args.jobs} processes "
          f"({exported} exported, {len(cks) - exported} cached)", flush=True)
    if args.jobs > 1 and len(jobs) > 1:
        with mp.get_context("spawn").Pool(min(args.jobs, len(jobs))) as pool:
            rows = pool.map(_score_one, jobs)
    else:
        rows = [_score_one(j) for j in jobs]

    table = []
    for tag, _, _ in cks:
        mine = [r for r in rows if r["tag"] == tag]
        table.append({"tag": tag, **{k: float(np.mean([r[k] for r in mine]))
                                     for k in ("speed", "ep_len", "falls")}})
    safe = [r for r in table if r["falls"] <= args.max_falls]
    best = max(safe or table, key=lambda r: r[METRIC])
    final = next(r for r in table if r["tag"] == "final")
    # NOTHING cleared the floor. Do not quietly ship the fastest faller —
    # among policies that all fall, "fastest" selects the one diving forward
    # hardest, which is the failure mode this criterion exists to avoid.
    viable = bool(safe)

    shipped = None
    if not args.dry_run and not viable:
        print(f"\n  !! no candidate falls at most {args.max_falls:.0%} of episodes — "
              "nothing shipped.\n     This run has no policy worth keeping; "
              "train longer, or look at it with render-rollout.")
    if not args.dry_run and viable:
        shutil.copy(probes / best["tag"] / "policy.onnx", run_dir / "policy.onnx")
        shipped = {"tag": best["tag"], "metric": METRIC, "score": round(best[METRIC], 4),
                   "final_score": round(final[METRIC], 4), "seeds": seeds,
                   "episodes": args.episodes, "max_falls": args.max_falls}
        (run_dir / "selected.json").write_text(json.dumps(shipped, indent=2))

    if args.json:
        print(json.dumps({"run": str(run_dir), "behavior": behavior, "metric": METRIC,
                          "best": best, "final": final, "table": table,
                          "rejected": [r["tag"] for r in table if r["falls"] > args.max_falls],
                          "viable": viable, "shipped": shipped}))
        return
    print(f"\n  {'checkpoint':>12}  {'m/s':>7} {'ep_len':>8} {'falls':>7}")
    for r in table:
        rejected = r["falls"] > args.max_falls
        mark = ("  (falls)" if rejected else "")
        if r["tag"] == best["tag"]:
            mark = " <- best" if viable else "  (falls; fastest, not shipped)"
        print(f"  {r['tag']:>12}  {r['speed']:7.3f} {r['ep_len']:8.0f} {r['falls']:7.2f}{mark}")
    if viable:
        print(f"\n  best is '{best['tag']}': {best[METRIC]:.3f} m/s vs final "
              f"{final[METRIC]:.3f} ({best[METRIC] - final[METRIC]:+.3f})")
    else:
        print(f"\n  NO viable candidate: every one falls more than "
              f"{args.max_falls:.0%} of episodes. The fastest of them "
              f"('{best['tag']}', {best[METRIC]:.3f} m/s) is not a winner — among "
              "policies that all fall, fastest is the one diving forward hardest.")
    if shipped:
        print(f"  shipped {run_dir}/policy.onnx from checkpoint {best['tag']}")


if __name__ == "__main__":
    main()
