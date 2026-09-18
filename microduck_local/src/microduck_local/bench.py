"""Measure env throughput on this machine.

    uv run bench-walk [--envs 16] [--robot g1]

Prints control-steps/sec for a single env and for the parallel vec env —
the number that decides how long a training run takes here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .robots import registry
from .train import env_class
from .vec_env import make_vec_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--robot", default="microduck", choices=registry.ids(),
                    help="which body to bench (measured here: the G1 holds "
                         "32.5k ctrl steps/s at 32 envs against the duck's "
                         "61.3k — 1.9x the cost. Measure, do not extrapolate)")
    args = ap.parse_args()
    cls = env_class(args.robot)

    env = cls(seed=0)
    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        _, _, term, trunc, _ = env.step(env.action_space.sample() * 0.1)
        if term or trunc:
            env.reset()
    single = args.steps / (time.perf_counter() - t0)
    print(f"single env: {single:,.0f} ctrl steps/s ({single * 4:,.0f} physics steps/s)")

    venv = make_vec_env([
        (lambda r: (lambda: cls(seed=r)))(i) for i in range(args.envs)
    ])
    venv.reset()
    acts = np.stack([venv.action_space.sample() * 0.1 for _ in range(args.envs)])
    n = max(args.steps // 4, 250)
    t0 = time.perf_counter()
    for _ in range(n):
        venv.step(acts)  # auto-resets
    parallel = n * args.envs / (time.perf_counter() - t0)
    venv.close()
    print(f"{args.envs} envs:  {parallel:,.0f} ctrl steps/s "
          f"→ 1M steps ≈ {1e6 / parallel / 60:.1f} min, 10M ≈ {1e7 / parallel / 3600:.1f} h")


if __name__ == "__main__":
    main()
