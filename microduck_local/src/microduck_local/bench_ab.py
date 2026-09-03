"""Compare two training runs at MATCHED STEP COUNTS, not at matched wall time.

    uv run bench-ab brains/ab-legacy brains/ab-fixed
    uv run bench-ab runs/a runs/b --metric ep_len --json

Why this exists
---------------
AGENTS.md's verification discipline #4: "Throughput is not learning speed."
Two optimizations in this workspace each raised steps/s 25-40% and HALVED
reward per step — the overlapped update and the 64-env big-batch config —
and both were caught only because someone compared the arms at the same step
count instead of the same wall clock. That comparison was done by hand every
time. This is it as a command, so the next change is cheap to hold to the
same standard.

What it reports, for each arm and for the pair:

* the metric interpolated onto a COMMON step grid, so arms that logged at
  different rollout sizes are comparable at all;
* the mean over the last quarter of the run (where a policy has settled) and
  the best value anywhere, with the paired difference;
* a paired sign test over the grid — how much of the run one arm led — which
  is the honest summary when two noisy curves cross;
* throughput and wall clock, reported SEPARATELY and never mixed into the
  quality verdict.

A verdict here is about learning per step. It says nothing about what the
DETERMINISTIC export does — that still needs `select-brain`, `eval-brain` or
`render-rollout`, because a reward curve measures the noise-crutched
stochastic policy (verification discipline #1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Student's t, two-sided 95%, by degrees of freedom. Small-n A/Bs are the
# whole point of this file and the normal 1.96 is simply wrong there: at n=2
# it understates the interval 6.5-fold, which is enough to manufacture a
# significant result out of two training runs. (Measured: two paired seeds of
# a real arm gave "excludes 0" under 1.96 and "includes 0" under t.)
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t_critical(df: int) -> float:
    """Two-sided 95% t value for `df` degrees of freedom (nearest tabulated
    df at or below, so the interval is never optimistic)."""
    if df <= 0:
        return float("inf")
    for k in sorted(T_95, reverse=True):
        if df >= k:
            return T_95[k]
    return T_95[1]


def paired_delta(baseline: list[float], candidate: list[float]) -> dict:
    """A PAIRED comparison of two arms scored on the same training seeds.

    Why paired: run-to-run variance on the follow benchmark is about +-0.02
    in band, which is larger than any hyperparameter effect measured here.
    Comparing two arms' MEANS at one training seed each therefore measures
    luck. Training both arms on the SAME seed and taking the per-seed
    DIFFERENCE cancels that, and turned a spurious -0.015 into a real
    -0.002.

    `baseline[i]` and `candidate[i]` must be the same training seed.
    """
    a, b = np.asarray(baseline, float), np.asarray(candidate, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired arms must be the same non-empty length")
    d = b - a
    n = d.size
    out = {"n": int(n), "mean": float(d.mean()),
           "candidate_ahead": int((d > 0).sum()), "diffs": [float(x) for x in d]}
    if n < 2:
        out.update(sd=None, lo=None, hi=None,
                   verdict="one pair resolves nothing — run more training seeds")
        return out
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = t_critical(n - 1)
    lo, hi = d.mean() - t * se, d.mean() + t * se
    out.update(sd=sd, lo=float(lo), hi=float(hi))
    out["verdict"] = ("includes 0: unresolved" if lo <= 0 <= hi else
                      "candidate is better" if lo > 0 else "baseline is better")
    return out


def read_progress(run_dir: Path, metric: str = "ep_rew") -> tuple[np.ndarray, np.ndarray, float]:
    """(steps, metric, elapsed_s) from a run's progress.jsonl.

    Both trainers write one JSON object per PPO rollout with `steps` and
    `ep_rew`; the trick trainer adds `terms` and a terminating {"done": true}
    line. Rows without the metric (the done marker) are skipped, and rows are
    sorted by step so a warm-restarted run — which appends to the same file —
    still reads as one curve.
    """
    path = run_dir / "progress.jsonl"
    if not path.exists():
        raise SystemExit(f"no progress.jsonl under {run_dir}")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if metric in r and r.get(metric) is not None and "steps" in r:
            v = float(r[metric])
            if np.isfinite(v):
                rows.append((int(r["steps"]), v, float(r.get("elapsed_s", 0.0))))
    if not rows:
        raise SystemExit(f"{path} has no rows carrying {metric!r}")
    rows.sort(key=lambda x: x[0])
    steps = np.array([r[0] for r in rows], float)
    vals = np.array([r[1] for r in rows], float)
    return steps, vals, rows[-1][2]


def compare(a_dir: Path, b_dir: Path, metric: str = "ep_rew", points: int = 200,
            tail: float = 0.25) -> dict:
    a_s, a_v, a_t = read_progress(a_dir, metric)
    b_s, b_v, b_t = read_progress(b_dir, metric)

    # The common grid stops at the SHORTER run: comparing an arm's late
    # steps against nothing is how a shorter arm wins by default.
    lo = max(a_s.min(), b_s.min())
    hi = min(a_s.max(), b_s.max())
    if hi <= lo:
        raise SystemExit(f"the two runs do not overlap in steps ({lo:.0f}..{hi:.0f})")
    grid = np.linspace(lo, hi, points)
    a_g, b_g = np.interp(grid, a_s, a_v), np.interp(grid, b_s, b_v)

    cut = grid >= hi - tail * (hi - lo)
    diff = b_g - a_g
    lead = float((diff > 0).mean())          # fraction of the run B is ahead
    out = {
        "metric": metric, "overlap": [float(lo), float(hi)], "points": points,
        "a": {"run": str(a_dir), "rows": int(a_s.size), "final_step": float(a_s.max()),
              "tail_mean": float(a_g[cut].mean()), "best": float(a_v.max()),
              "elapsed_s": a_t, "steps_per_s": float(a_s.max() / a_t) if a_t else None},
        "b": {"run": str(b_dir), "rows": int(b_s.size), "final_step": float(b_s.max()),
              "tail_mean": float(b_g[cut].mean()), "best": float(b_v.max()),
              "elapsed_s": b_t, "steps_per_s": float(b_s.max() / b_t) if b_t else None},
        "tail_frac": tail,
        "tail_delta": float(b_g[cut].mean() - a_g[cut].mean()),
        "best_delta": float(b_v.max() - a_v.max()),
        "b_leads_frac": lead,
        # Where B first reaches A's final tail level: "how many steps did the
        # change save", which is the number a throughput claim has to beat.
        "b_reaches_a_tail_at": None,
    }
    target = out["a"]["tail_mean"]
    reached = np.flatnonzero(b_g >= target)
    if reached.size:
        out["b_reaches_a_tail_at"] = float(grid[reached[0]])
    rel = out["tail_delta"] / abs(target) if target else float("nan")
    out["tail_delta_rel"] = float(rel)
    # A verdict, stated conservatively. Curves this noisy cross constantly, so
    # a lead has to show up in BOTH the settled tail and most of the run.
    if abs(rel) < 0.01 or 0.35 < lead < 0.65:
        out["verdict"] = "no difference this A/B can resolve"
    elif out["tail_delta"] > 0 and lead >= 0.65:
        out["verdict"] = "B is better"
    elif out["tail_delta"] < 0 and lead <= 0.35:
        out["verdict"] = "A is better"
    else:
        out["verdict"] = "mixed — tail and lead disagree, run more seeds"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="baseline run dir (runs/<name> or brains/<name>)")
    ap.add_argument("b", type=Path, help="candidate run dir")
    ap.add_argument("--metric", default="ep_rew")
    ap.add_argument("--points", type=int, default=200, help="common-grid resolution")
    ap.add_argument("--tail", type=float, default=0.25,
                    help="fraction of the overlap treated as 'settled' (default the last quarter)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = compare(args.a, args.b, args.metric, args.points, args.tail)
    if args.json:
        print(json.dumps(r))
        return
    a, b = r["a"], r["b"]
    print(f"\n  {r['metric']} over the common range "
          f"{r['overlap'][0]:,.0f} .. {r['overlap'][1]:,.0f} steps\n")
    print(f"  {'':>10}  {'tail mean':>10} {'best':>9} {'rows':>6} {'steps/s':>9} {'wall':>8}")
    for tag, x in (("A", a), ("B", b)):
        sps = f"{x['steps_per_s']:,.0f}" if x["steps_per_s"] else "-"
        print(f"  {tag} {Path(x['run']).name:>8.8}  {x['tail_mean']:10.2f} {x['best']:9.2f} "
              f"{x['rows']:6d} {sps:>9} {x['elapsed_s']:7.0f}s")
    print(f"\n  tail delta (B-A): {r['tail_delta']:+.2f} ({r['tail_delta_rel']:+.1%})")
    print(f"  best delta (B-A): {r['best_delta']:+.2f}")
    print(f"  B leads over {r['b_leads_frac']:.0%} of the run")
    if r["b_reaches_a_tail_at"] is not None:
        frac = r["b_reaches_a_tail_at"] / r["overlap"][1]
        print(f"  B reaches A's settled level at {r['b_reaches_a_tail_at']:,.0f} steps "
              f"({frac:.0%} of the way)")
    print(f"\n  verdict: {r['verdict']}")
    print("  (learning per STEP only — the deterministic export still has to be "
          "scored with select-brain / eval-brain / render-rollout)\n")


if __name__ == "__main__":
    main()
