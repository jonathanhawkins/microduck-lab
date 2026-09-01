"""How many parallel envs should we actually train with on this machine?

    uv run bench-envs                       # the full sweep (~2 min)
    uv run bench-envs --envs 8,12,16        # a narrower sweep
    uv run bench-envs --compare-vec --compare-threads

`bench-walk` measures RAW env stepping, which scales almost linearly and
therefore says "more workers is always better". That is a lie about training:
PPO alternates a parallel rollout with a SERIAL gradient update (Amdahl), and
the update's share grows with the env count because the rollout buffer does
(n_steps=256 per env). This script measures the number that actually decides
run length — env-steps/second sustained through real `model.learn()`, rollouts
AND updates — on the real training path (`BehaviorEnv` + train_behavior's PPO).

Each data point runs in a FRESH SUBPROCESS. That is not tidiness: thread pools
(torch/OpenMP/Accelerate) are process-global and initialize on first use, so a
`--compare-threads` pair measured in one process would compare a pinned run
against an already-warm 18-thread pool. Fresh processes also guarantee every
point pays the same worker spawn cost and none inherits the previous point's
leaked workers.

What it found here (18-CPU M5 Max, 2026-08-30, behavior=run, fork backend):

* Throughput keeps climbing PAST cpu_count — 11.1k steps/s at 10 envs, 14.3k
  at 16, 15.6k at 24, asymptote ~17.1k — because PPO alternates a worker-bound
  rollout with a trainer-bound update and extra workers fill the cores the
  update leaves idle. Capping workers at cpu_count (18) is 7% behind the
  24-worker knee and 15% behind the asymptote.
* `torch.set_num_threads(1)` is a 21-24% REGRESSION, not the classic
  oversubscription win: the update is a real matmul workload on a
  512-256-128 MLP and wants every core it can get while the workers block.
  It is available as `--pin-threads`. The training path (and this bench,
  unless `--pin-threads`) now sets intra-op to 8 and inter-op to 1.
* Worker PROCESSES earn their keep at every count measured, including the
  low ones where the IPC might have eaten the win: at 8 envs 'dummy' (serial)
  reaches 42% of 'fork' and 'thread' 38%, and even at 4 envs they are 1.75x
  and 1.87x behind. 'fork' and 'subproc' tie on throughput (1.00x) — model
  sharing is a memory and startup win (0.6 s vs 1.3 s), not a speed one.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

# Fine resolution where the curve is still moving, then two coarse points PAST
# saturation. The tail is not decoration: `recommend` measures every point
# against the best one it was given, so a sweep that stops while the curve is
# still climbing can only ever nominate its own widest entry.
DEFAULT_SWEEP = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32, 48)
# ~90k steps is 5-15 s per point at the throughputs this machine reaches, and
# spans >= 7 PPO iterations even at the widest env count (256*48 = 12288/iter),
# so no point is decided by a single unlucky rollout. Twelve points then land
# the whole sweep in under two minutes.
DEFAULT_BUDGET = 90_000
# One warmup rollout per point is DISCARDED before the clock starts: the first
# iteration pays MJCF compilation in every worker plus torch's lazy allocator,
# which is startup cost, not throughput.
WARMUP_ROLLOUTS = 1
# A point counts as "as fast as the best" within this margin. It has to clear
# two bars. (1) Measurement noise: a point repeats within ~2% on a quiet
# machine, but spread to 5-15% while another agent's training job shared the
# cores — and a 3% band (the first thing tried) then just crowned whichever
# point got the quietest stretch. (2) The trade itself: the curve saturates
# without ever turning back down, so ANY tolerance below the plateau's spread
# nominates the widest count swept, which is an artifact of the sweep range and
# not an answer. 10% buys more worker processes only for a speedup you could
# actually see.
KNEE_TOLERANCE = 0.10

# Thread caps for the point subprocess AND the SubprocVecEnv workers it spawns
# (children inherit the environment). These MUST be set before numpy/torch are
# imported — BLAS reads them at library init — which is why the driver puts
# them in the child's env rather than the child setting them on itself.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Point:
    """One measured (envs, vec-env kind, thread-pinning) configuration."""

    envs: int
    vec: str            # a vec_env.BACKENDS name ("fork", "subproc", ...)
    pinned: bool        # threads capped to 1 in trainer + workers
    steps: int          # env-steps actually run through learn(), warmup excluded
    seconds: float      # wall time for those steps
    setup_s: float      # venv spawn + PPO construction + warmup rollout
    overlap: bool = False  # SymmetryPPO overlap_update (update || rollout)

    @property
    def steps_per_s(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0

    @property
    def per_env(self) -> float:
        """Throughput per worker — the scaling efficiency, in one number."""
        return self.steps_per_s / self.envs if self.envs else 0.0


# --------------------------------------------------------------------------
# Pure logic (unit-tested in tests/test_bench_envs.py — no MuJoCo required)
# --------------------------------------------------------------------------


def recommend(points, tolerance: float = KNEE_TOLERANCE) -> int:
    """The KNEE of the curve, not its peak.

    Throughput vs env count saturates without ever turning back down, so
    `max()` returns the widest count swept — the most expensive answer, and one
    that says more about the sweep's range than about the machine. Instead:
    find the best measured throughput, then return the SMALLEST env count that
    lands within `tolerance` of it.

    This is only meaningful if the sweep actually REACHED saturation; see
    DEFAULT_SWEEP's tail.
    """
    ordered = sorted(points, key=lambda p: p.envs)
    if not ordered:
        raise ValueError("no measured points")
    best = max(p.steps_per_s for p in ordered)
    threshold = best * (1.0 - tolerance)
    for p in ordered:
        if p.steps_per_s >= threshold:
            return p.envs
    return ordered[-1].envs  # unreachable: the best point always clears it


def speedup_table(points) -> list[dict]:
    """Rows for the printed table, normalized against the smallest env count."""
    ordered = sorted(points, key=lambda p: p.envs)
    base = ordered[0].steps_per_s if ordered else 0.0
    return [
        {
            "envs": p.envs,
            "steps_per_s": p.steps_per_s,
            "seconds": p.seconds,
            "per_env": p.per_env,
            "speedup": (p.steps_per_s / base) if base else 0.0,
            "efficiency": ((p.steps_per_s / base) / (p.envs / ordered[0].envs)
                           if base else 0.0),
        }
        for p in ordered
    ]


def format_table(points, title: str = "PPO training throughput") -> str:
    rows = speedup_table(points)
    out = [
        f"{title}",
        f"{'envs':>5}  {'steps/s':>9}  {'wall s':>7}  {'steps/s/env':>11}  "
        f"{'speedup':>7}  {'efficiency':>10}",
        "-" * 62,
    ]
    for r in rows:
        out.append(
            f"{r['envs']:>5}  {r['steps_per_s']:>9,.0f}  {r['seconds']:>7.1f}  "
            f"{r['per_env']:>11,.0f}  {r['speedup']:>6.2f}x  {r['efficiency']:>9.0%}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _run_point(envs: int, vec: str, pinned: bool, budget: int,
               behavior: str, seed: int, overlap: bool = False,
               update_device: str | None = None) -> Point:
    """Measure ONE configuration. Runs inside the point subprocess."""
    from .ppo_hparams import N_STEPS, configure_torch_cpu, ppo_batch_size
    from .train_behavior import make_env  # the real training env factory
    from .vec_env import as_sb3_vec_env, make_vec_env

    t_setup = time.perf_counter()
    factories = [make_env(behavior, i, seed) for i in range(envs)]
    # Fork BEFORE importing torch, matching the training path.
    venv = make_vec_env(factories, backend=vec)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    if overlap or update_device:
        # SymmetryPPO at coefficient 0 is bit-identical stock PPO (see
        # test_symmetry.py); overlap_update / update_device are the things
        # being measured.
        from .symmetry import SymmetryPPO as PPO  # noqa: N814

    if pinned:
        # The env vars above cap BLAS; this caps torch's own intra-op pool,
        # which torch sets from cpu_count() when OMP_NUM_THREADS is unset.
        torch.set_num_threads(1)
    else:
        configure_torch_cpu(torch)

    venv = VecMonitor(as_sb3_vec_env(venv))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
    # Mirrors train_behavior.py's PPO construction. Duplicated rather than
    # imported because that block is a plain call inside main(); if it changes,
    # change it here too — but the SHAPE of the curve is what matters and that
    # is robust to the exact coefficients.
    model = PPO(
        "MlpPolicy", venv,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                           activation_fn=torch.nn.ELU, log_std_init=0.0),
        n_steps=N_STEPS, batch_size=ppo_batch_size(N_STEPS, envs), n_epochs=5,
        learning_rate=1e-3,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        max_grad_norm=1.0, device="cpu", seed=seed, verbose=0,
        **(dict(symmetry_coef=0.0, desired_kl=None,
                overlap_update=overlap, update_device=update_device)
           if (overlap or update_device) else {}),
    )
    warmup = WARMUP_ROLLOUTS * model.n_steps * envs
    model.learn(total_timesteps=warmup, progress_bar=False)
    setup_s = time.perf_counter() - t_setup

    before = model.num_timesteps
    t0 = time.perf_counter()
    # reset_num_timesteps=False keeps _last_obs, so this continues the warmed
    # rollout instead of paying another env reset storm.
    model.learn(total_timesteps=budget, progress_bar=False,
                reset_num_timesteps=False)
    seconds = time.perf_counter() - t0
    steps = int(model.num_timesteps) - int(before)
    venv.close()
    return Point(envs=envs, vec=vec, pinned=pinned, steps=steps,
                 seconds=seconds, setup_s=setup_s)


def _spawn_point(envs: int, vec: str, pinned: bool, budget: int,
                 behavior: str, seed: int, timeout: float,
                 overlap: bool = False,
                 update_device: str | None = None) -> Point | None:
    """Run one point in a fresh interpreter; parse its JSON result line."""
    env = dict(os.environ)
    for var in THREAD_ENV_VARS:
        if pinned:
            env[var] = "1"
        else:
            env.pop(var, None)
    spec = json.dumps({"envs": envs, "vec": vec, "pinned": pinned,
                       "budget": budget, "behavior": behavior, "seed": seed,
                       "overlap": overlap,
                       "update_device": update_device})
    proc = subprocess.run(
        [sys.executable, "-m", "microduck_local.bench_envs", "--point", spec],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("POINT "):
            return Point(**json.loads(line[len("POINT "):]))
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    print(f"  !! point failed (envs={envs}, {vec}, pinned={pinned}): "
          + " / ".join(tail))
    return None


def _sweep(counts, vec: str, pinned: bool, budget: int, behavior: str,
           seed: int, timeout: float, label: str,
           order: list[int] | None = None, overlap: bool = False,
           update_device: str | None = None) -> list[Point]:
    points: list[Point] = []
    for n in (order if order is not None else list(counts)):
        print(f"  {label}: {n:>2} envs ...", end="", flush=True)
        p = _spawn_point(n, vec, pinned, budget, behavior, seed, timeout,
                         overlap=overlap, update_device=update_device)
        if p:
            points.append(p)
            print(f" {p.steps_per_s:,.0f} steps/s  ({p.seconds:.1f} s run, "
                  f"{p.setup_s:.1f} s setup)")
    return points


def _best_of(counts, vec: str, pinned: bool, budget: int, behavior: str,
             seed: int, timeout: float, label: str, repeats: int,
             rng: random.Random, overlap: bool = False,
             update_device: str | None = None) -> list[Point]:
    """Repeat a comparison arm the same way the main sweep is repeated.

    An arm measured ONCE against a main sweep that kept its best of five is not
    a comparison, it is a handicap — and it flipped this benchmark's answer on
    thread pinning twice before the arms were evened up.
    """
    best: dict[int, Point] = {}
    for r in range(repeats):
        order = list(counts)
        rng.shuffle(order)
        tag = label if repeats == 1 else f"{label} {r + 1}/{repeats}"
        for p in _sweep(counts, vec, pinned, budget, behavior, seed, timeout,
                        tag, order=order, overlap=overlap,
                        update_device=update_device):
            if p.envs not in best or p.steps_per_s > best[p.envs].steps_per_s:
                best[p.envs] = p
    return sorted(best.values(), key=lambda p: p.envs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", default=",".join(str(n) for n in DEFAULT_SWEEP),
                    help="comma-separated env counts to sweep")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="env-steps timed per data point (warmup excluded)")
    ap.add_argument("--behavior", default="run",
                    help="behavior recipe to train (physics cost dominates, so "
                         "this mostly picks the episode length)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the whole sweep and keep the best per env "
                         "count (noise is one-sided: interference only slows)")
    ap.add_argument("--backend", default=None,
                    help="vec_env backend for the main sweep (default: whatever "
                         "training itself would pick)")
    ap.add_argument("--compare-vec", action="store_true",
                    help="also measure the OTHER vec_env backends at low env "
                         "counts — is the process parallelism worth its cost?")
    ap.add_argument("--compare-threads", action="store_true",
                    help="also measure the OTHER thread setting, to check for "
                         "the N-workers x N-BLAS-threads oversubscription "
                         "pathology (measured: it is not the pathology here)")
    ap.add_argument("--pin-threads", action="store_true",
                    help="cap the trainer and its workers to one BLAS/torch "
                         "thread. OFF by default because it measured 21-24%% "
                         "SLOWER — see the module docstring")
    ap.add_argument("--overlap", action="store_true",
                    help="measure with SymmetryPPO's overlapped update "
                         "(symmetry_coef=0, so stock PPO math + the overlap)")
    ap.add_argument("--update-device", default=None,
                    help="measure with the PPO minibatch loop on this device "
                         "(e.g. mps); rollouts stay on cpu")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json", default=None, help="also write raw points here")
    ap.add_argument("--point", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.point:  # child mode: one data point, one JSON line on stdout
        spec = json.loads(args.point)
        p = _run_point(spec["envs"], spec["vec"], spec["pinned"],
                       spec["budget"], spec["behavior"], spec["seed"],
                       overlap=spec.get("overlap", False),
                       update_device=spec.get("update_device"))
        print("POINT " + json.dumps(asdict(p)))
        return

    from .vec_env import BACKENDS, resolve_backend

    counts = [int(x) for x in args.envs.split(",") if x.strip()]
    pinned = args.pin_threads
    backend = resolve_backend(args.backend)  # what training would pick
    cpus = os.cpu_count() or 0
    print(f"bench-envs: behavior={args.behavior} backend={backend} "
          f"budget={args.budget:,} steps/point cpus={cpus} "
          f"threads={'pinned to 1' if pinned else 'default'}")
    print("(close other heavy apps — this measures the machine, not the code)\n")

    # Repeats visit the env counts in a SHUFFLED order, and each count keeps
    # its best result. Both halves matter. Background load on a laptop drifts
    # over minutes (another agent's training job, a browser waking up); walking
    # 4->32 in order makes that drift indistinguishable from a trend in env
    # count — an earlier version of this sweep "measured" two opposite curves
    # back to back that way. Shuffling decorrelates the two, and because
    # interference can only ever SLOW a point down, the max over repeats is the
    # estimate closest to the idle machine.
    rng = random.Random(args.seed)
    points = _best_of(counts, backend, pinned, args.budget, args.behavior,
                      args.seed, args.timeout, "sweep", args.repeats, rng,
                      overlap=args.overlap, update_device=args.update_device)
    if not points:
        raise SystemExit("every data point failed — see the errors above")

    print("\n" + format_table(points))
    pick = recommend(points)
    peak = max(points, key=lambda p: p.steps_per_s)
    print(f"\nrecommended --envs: {pick}   "
          f"(cheapest count within {KNEE_TOLERANCE:.0%} of the best measured "
          f"{peak.steps_per_s:,.0f} steps/s at {peak.envs} envs)")
    if pick > cpus:
        # Worth saying out loud, because it looks like a bug: the rollout and
        # the multi-threaded update take turns, so workers beyond cpu_count
        # fill cores the update phase would otherwise leave idle.
        print(f"  ({pick} workers on {cpus} cores — oversubscribing is correct "
              "for PPO; rollout and update alternate)")
    at_pick = next(p for p in points if p.envs == pick)
    print(f"at {pick} envs: 1M steps ~ {1e6 / at_pick.steps_per_s / 60:.1f} min, "
          f"10M ~ {1e7 / at_pick.steps_per_s / 3600:.1f} h")

    extra: list[Point] = []
    if args.compare_vec:
        # Low counts only: that is where the "are worker processes even worth
        # their IPC and spawn cost?" question is live. At 16 envs nobody
        # seriously proposes stepping them serially.
        low = [n for n in counts if n <= 8] or counts[:2]
        arm_repeats = max(1, min(args.repeats, 3))
        for other in [b for b in BACKENDS if b != backend]:
            print(f"\nbackend {other!r} vs the training default {backend!r}:")
            got = _best_of(low, other, pinned, args.budget, args.behavior,
                           args.seed, args.timeout, other, arm_repeats, rng)
            extra += got
            for g in got:
                ref = next((p for p in points if p.envs == g.envs), None)
                if ref:
                    print(f"  {g.envs:>2} envs: {backend} {ref.steps_per_s:,.0f} vs "
                          f"{other} {g.steps_per_s:,.0f} steps/s "
                          f"({ref.steps_per_s / g.steps_per_s:.2f}x) — "
                          f"setup {ref.setup_s:.1f} s vs {g.setup_s:.1f} s")

    if args.compare_threads:
        print(f"\nthread pinning at the knee ({pick} envs) and above:")
        probe = sorted({pick, max(counts)})
        other = _best_of(probe, backend, not pinned, args.budget, args.behavior,
                         args.seed, args.timeout,
                         "unpinned" if pinned else "pinned",
                         max(1, min(args.repeats, 3)), rng)
        extra += other
        for o in other:
            ref = next((p for p in points if p.envs == o.envs), None)
            if ref:
                delta = (ref.steps_per_s / o.steps_per_s - 1.0) * 100
                print(f"  {o.envs:>2} envs: pinned {'yes' if pinned else 'no'} "
                      f"{ref.steps_per_s:,.0f} vs {o.steps_per_s:,.0f} steps/s "
                      f"({delta:+.1f}% for the main sweep's setting)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"recommended_envs": pick, "cpus": cpus,
                       "behavior": args.behavior, "budget": args.budget,
                       "points": [asdict(p) for p in points + extra]}, f, indent=2)
        print(f"\nwrote {args.json}")

    spread = statistics.pstdev([p.steps_per_s for p in points]) if len(points) > 1 else 0
    if spread / max(peak.steps_per_s, 1) < 0.02:
        print("\nNOTE: the curve is flat across the whole sweep — env count is "
              "not the bottleneck here. Suspect a serial stage (the PPO update) "
              "or an oversubscribed thread pool.")


if __name__ == "__main__":
    main()
