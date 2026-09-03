"""What THIS machine can do — usable cores, and the thread/packing policy that
suits them.

**Mac is the default this repo is tuned for and advertised on**, and the
`mac` profile below reproduces the historical numbers exactly: intra-op 8,
inter-op 1, one env per worker process, no thread switching between PPO
phases. Nothing about a Mac run changes because this module exists — that is
what `tests/test_machine.py` locks.

Linux and cloud boxes are a different machine and were measured as one.
`train-behavior`'s recipe on a 4-vCPU Xeon (this session's container), 40k
timed steps per point, `behavior=run`, fork backend:

| 32 envs                          | steps/s | CPU busy |
|----------------------------------|--------:|---------:|
| the mac profile's settings       |  2,231  |     90%  |
| + phase-aware threads (1/4)      |  2,915  |     77%  |
| + 4 workers x 8 envs             |  3,129  |     74%  |

The cause of the gap is the TRAINER, not the physics workers. torch's
OpenMP pool spin-waits between the tiny batch-N policy forwards of a
rollout, and on a 4-core box those spinning threads take the cores the
workers need: at 4 envs the trainer process sat at 311% CPU while each
worker got 18%. "90% busy" was mostly spin. The M5 Max has enough cores to
absorb that, which is why `bench-envs --pin-threads` measured a 21-24%
REGRESSION there and the opposite here.

The fix is not to pin threads (that gives the update one core and cost 21%
on the Mac); it is to give each PPO phase the thread count it actually
wants — 1 during rollout collection, all of them during the update, where a
512-256-128 MLP really is a matmul workload. Measured per phase at 32 envs:
rollout 22.8 s vs update 2.9 s, so the rollout is ~89% of wall time on this
box and is what the threads must not disturb.

Both knobs are QUALITY-NEUTRAL by construction, which is what lets them be
defaults at all under AGENTS.md's "throughput is not learning speed" rule:
thread counts do not enter the math, and worker packing is pinned
step-for-step by `test_envs_per_worker_batching_matches_one_per_worker`.
The env count — which DOES set the PPO batch size and therefore the
learning dynamics — is deliberately NOT part of any profile; it stays 32
everywhere.

Overrides, for benchmarking and for a machine that disagrees:

    MICRODUCK_PROFILE=mac|linux|auto   force a profile (default: auto)
    MICRODUCK_ROLLOUT_THREADS=N        torch intra-op during rollouts
    MICRODUCK_UPDATE_THREADS=N         torch intra-op during the PPO update
    MICRODUCK_ENVS_PER_WORKER=K        envs per worker process (see vec_env)

Torch-free on purpose, like `ppo_hparams`: the trainers fork their vec-env
workers BEFORE importing torch, and a module that pulled torch in here would
put OpenMP pools in the parent and deadlock the fork on macOS.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass

# The Mac constant this repo was tuned on: a 1024x512 Linear peaks at 8
# intra-op threads on an M5 Max (0.38 ms) vs PyTorch's default of 6 P-cores
# (0.42 ms) vs 18 (0.42 ms — E-cores hurt). Kept here as the mac profile's
# ceiling; `ppo_hparams.TORCH_INTRA_THREADS` re-exports it for compatibility.
MAC_INTRA_THREADS = 8

PROFILES = ("mac", "linux")


# --------------------------------------------------------------- core counting


def parse_cpu_max(text: str) -> int | None:
    """cgroup v2 `cpu.max` -> whole cores, or None when unlimited.

    The file is "<quota> <period>" in microseconds, or "max <period>". A
    fractional quota floors to at least one core: half a core still has to
    run the trainer.
    """
    try:
        quota_s, period_s = text.split()[:2]
    except ValueError:
        return None
    if quota_s == "max":
        return None
    try:
        quota, period = int(quota_s), int(period_s)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def _cgroup_quota() -> int | None:
    """Cores this container is actually allowed, or None if unlimited.

    `os.cpu_count()` reports the HOST's cores inside a container, so a
    4-CPU-quota pod on a 64-core node would otherwise start 64 spinning
    OpenMP threads — the pathology above, magnified. cgroup v2 first
    (`cpu.max`), then v1's quota/period pair.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            found = parse_cpu_max(f.read())
        if found is not None:
            return found
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def usable_cores() -> int:
    """Cores this PROCESS may actually run on — affinity and cgroup aware.

    Three answers can disagree on a cloud box, and the smallest is the true
    one: `os.cpu_count()` (the host's cores, wrong in a container),
    `sched_getaffinity` (what a taskset/cpuset allows, Linux-only) and the
    cgroup CPU quota (what a Docker/Kubernetes limit allows).
    """
    counts = [os.cpu_count() or 1]
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            counts.append(len(getaffinity(0)))
        except OSError:  # pragma: no cover - platform dependent
            pass
    quota = _cgroup_quota()
    if quota is not None:
        counts.append(quota)
    return max(1, min(counts))


# ------------------------------------------------------------------- profiles


@dataclass(frozen=True)
class Profile:
    """The thread and packing policy for one kind of machine.

    ``rollout_threads is None`` means "do not touch torch's thread count
    between PPO phases" — the mac behavior, and the reason a Mac run is
    bit-for-bit what it was before this module existed.
    """

    name: str
    cores: int
    update_threads: int
    rollout_threads: int | None
    interop_threads: int = 1
    pack_workers: bool = False

    @property
    def phase_threads(self) -> bool:
        """Does this profile switch thread counts between rollout and update?"""
        return (self.rollout_threads is not None
                and self.rollout_threads != self.update_threads)

    def envs_per_worker(self, n_envs: int) -> int:
        """How many envs to pack into each worker PROCESS.

        1 (one process per env) until the fleet outnumbers the cores. Beyond
        that point the extra processes cannot run in parallel anyway, and each
        one costs the parent two semaphore operations per vec-step.

        Measured on both profiles, and the `n_envs > cores` threshold is what
        both sets of numbers say — packing helps only past it:

        * linux, 32 envs on 4 cores: packing to 4 workers is +7% throughput
          and halves vec-env startup (8.2 s -> 4.3 s).
        * mac, 18 cores (bench-envs, real PPO, best of 3-4 repeats, quiet
          machine, 2026-09-03):

              envs |  k=1    |  k=2    |
                 8 | 11,516  | 10,829  |  -6.0%   too few workers, cores idle
                16 | 15,651  | 14,713  |  -6.0%
                32 | 19,456  | 20,536  |  +5.5%   semaphore traffic dominates
                32 |         | 17,824  |  -8.2% at k=4 — past the knee again

          which this rule reproduces exactly: 8 and 16 are under 18 cores so
          they get k=1, and 32 gets ceil(32/18) = 2, never the 4 that lost.

        Packing is invisible to the caller — same obs/rew/done stream, pinned
        by tests/test_vec_env.py's
        test_envs_per_worker_batching_matches_one_per_worker, so unlike most
        throughput changes here there is no learning-quality question to A/B.
        """
        if not self.pack_workers or n_envs <= self.cores:
            return 1
        return -(-int(n_envs) // max(1, self.cores))  # ceil

    def describe(self) -> str:
        rollout = ("unchanged" if self.rollout_threads is None
                   else str(self.rollout_threads))
        return (f"{self.name} profile: {self.cores} usable cores, "
                f"torch intra-op {rollout} during rollouts / "
                f"{self.update_threads} during the update, "
                f"inter-op {self.interop_threads}, "
                f"worker packing {'on' if self.pack_workers else 'off'}")


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def detect_platform() -> str:
    """Which profile this machine gets when nobody says: mac on Darwin,
    linux everywhere else (Linux, and Windows once the spawn backend lands —
    both lack the P/E-core scheduler the mac numbers were measured on)."""
    forced = (os.environ.get("MICRODUCK_PROFILE") or "auto").strip().lower()
    if forced in PROFILES:
        return forced
    if forced not in ("", "auto"):
        raise ValueError(
            f"unknown MICRODUCK_PROFILE {forced!r}; expected one of "
            f"{PROFILES + ('auto',)}"
        )
    return "mac" if sys.platform == "darwin" else "linux"


def build_profile(name: str | None = None, cores: int | None = None) -> Profile:
    """The profile for `name` (default: whatever this machine is)."""
    name = name or detect_platform()
    cores = cores or usable_cores()
    if name == "mac":
        # EXACTLY the historical settings. No phase switching, no packing.
        profile = Profile(
            name="mac", cores=cores,
            update_threads=min(MAC_INTRA_THREADS, cores),
            rollout_threads=None, interop_threads=1, pack_workers=True,
        )
    elif name == "linux":
        profile = Profile(
            name="linux", cores=cores,
            # The update is the one phase that wants every core: the workers
            # are blocked on their semaphores while it runs.
            update_threads=cores,
            # One thread for the rollout's batch-N forwards. More only spin,
            # and the spinning is what starves the physics workers.
            rollout_threads=1,
            interop_threads=1,
            pack_workers=True,
        )
    else:  # pragma: no cover - guarded by detect_platform
        raise ValueError(f"unknown profile {name!r}")

    rollout = _int_env("MICRODUCK_ROLLOUT_THREADS")
    update = _int_env("MICRODUCK_UPDATE_THREADS")
    if rollout is not None or update is not None:
        profile = Profile(
            name=profile.name, cores=profile.cores,
            update_threads=update if update is not None else profile.update_threads,
            rollout_threads=rollout if rollout is not None else profile.rollout_threads,
            interop_threads=profile.interop_threads,
            pack_workers=profile.pack_workers,
        )
    return profile


_CACHED: Profile | None = None


def profile() -> Profile:
    """This machine's profile, detected once per process."""
    global _CACHED
    if _CACHED is None:
        _CACHED = build_profile()
    return _CACHED


def reset_cache() -> None:
    """Forget the detected profile (tests that manipulate the environment)."""
    global _CACHED
    _CACHED = None


# ------------------------------------------------------- the trainer-side hook


def phase_thread_callbacks(BaseCallback, prof: Profile | None = None) -> list:
    """`[callback]` that re-threads torch per PPO phase, or `[]` when the
    profile does not want it (every Mac run).

    A CALLBACK rather than a vendored train loop: SB3 fires
    ``on_rollout_start`` before collection and ``on_rollout_end`` after it,
    which is exactly the seam between the two phases — so this works for
    stock `PPO` (train-walk, train-brain) and `SymmetryPPO` (train-behavior)
    alike without touching either. Returning an empty list on the mac
    profile keeps even the per-step no-op out of the callback chain.

    Built from the caller's `BaseCallback` (imported after the vec-env
    workers fork) so this module stays torch-free — the same pattern
    `train.py`'s penalty-sign guard uses.
    """
    prof = prof or profile()
    if not prof.phase_threads:
        return []

    rollout_threads = prof.rollout_threads
    update_threads = prof.update_threads

    class PhaseThreadCallback(BaseCallback):
        """One intra-op thread while collecting, all of them while training.

        Thread counts do not enter the PPO math, so this changes throughput
        and nothing else — no seed, no sample, no gradient.
        """

        def _set(self, n: int) -> None:
            import torch
            if torch.get_num_threads() != n:
                torch.set_num_threads(n)

        def _on_training_start(self) -> None:
            self._set(rollout_threads)

        def _on_rollout_start(self) -> None:
            self._set(rollout_threads)

        def _on_rollout_end(self) -> None:
            self._set(update_threads)

        def _on_step(self) -> bool:
            return True

    return [PhaseThreadCallback()]


def with_phase_callbacks(callback, BaseCallback, prof: Profile | None = None):
    """`callback` plus the profile's thread callback — or `callback` ITSELF,
    unchanged, when the profile does not want one.

    The identity return is the point. On the mac profile a trainer must hand
    SB3 exactly what it handed before, so it does not even pay the
    `CallbackList` wrapper's extra call per step on the hottest loop in the
    repo.
    """
    extra = phase_thread_callbacks(BaseCallback, prof)
    if not extra:
        return callback
    return [*(callback if isinstance(callback, list) else [callback]), *extra]


def main() -> None:  # `python -m microduck_local.machine`
    prof = profile()
    print(prof.describe())
    print(f"  platform={sys.platform} machine={platform.machine()} "
          f"os.cpu_count()={os.cpu_count()} cgroup_quota={_cgroup_quota()}")
    for n in (16, 32, 64):
        print(f"  {n} envs -> {n // prof.envs_per_worker(n)} worker processes "
              f"x {prof.envs_per_worker(n)} envs")


if __name__ == "__main__":
    main()
