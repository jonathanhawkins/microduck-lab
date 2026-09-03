"""What THIS machine can do — usable cores, and the thread policy that suits
them.

**Mac is the default this repo is tuned for and advertised on**, and the
`mac` profile below reproduces the historical numbers exactly: intra-op 8,
inter-op 1, one env per worker process, no thread switching between PPO
phases. Nothing about a Mac run changes because this module exists — that is
what `tests/test_machine.py` locks.

Linux and cloud boxes are a different machine and were measured as one, on a
4-vCPU Xeon (this session's container). Medians of four INTERLEAVED
repetitions — the arms measured back to back with their order rotated per
rep, because this box drifts far too much to compare across windows —
`behavior=run`, fork backend, 40k timed steps per point:

| envs | mac profile | + phase-aware threads |
|-----:|------------:|----------------------:|
|    8 |         892 |     1,420  (+59%)     |
|   16 |       1,228 |     1,727  (+41%)     |
|   32 |       1,764 |     2,155  (+22%)     |

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

The thread split is QUALITY-NEUTRAL by construction, which is what lets it
be a default at all under AGENTS.md's "throughput is not learning speed"
rule: thread counts do not enter the PPO math. The env count — which DOES
set the batch size and therefore the learning dynamics — is deliberately
NOT part of any profile; it stays 32 everywhere.

**Worker packing was measured and REJECTED.** Packing the fleet into one
process per core (32 envs as 4 x 8) looked like a +7% win in a single
unreplicated point, but four interleaved repetitions put it slightly BEHIND
one process per env at every count: -1.7% at 8 envs, -3.9% at 16, -2.6% at
32, never once ahead. Its other claimed advantage, faster startup, was
really the per-worker numba JIT, and `vec_env._warm_jit` now removes that
for every layout (32-env setup 7.5 s unpacked vs 7.2 s packed — a wash).
So no profile packs, and `MICRODUCK_ENVS_PER_WORKER` stays what it was: a
manual escape hatch, worth re-measuring only at env counts far above these.

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
    """The thread policy for one kind of machine.

    ``rollout_threads is None`` means "do not touch torch's thread count
    between PPO phases" — the mac behavior, and the reason a Mac run is
    bit-for-bit what it was before this module existed.
    """

    name: str
    cores: int
    update_threads: int
    rollout_threads: int | None
    interop_threads: int = 1

    @property
    def phase_threads(self) -> bool:
        """Does this profile switch thread counts between rollout and update?"""
        return (self.rollout_threads is not None
                and self.rollout_threads != self.update_threads)

    def describe(self) -> str:
        rollout = ("unchanged" if self.rollout_threads is None
                   else str(self.rollout_threads))
        return (f"{self.name} profile: {self.cores} usable cores, "
                f"torch intra-op {rollout} during rollouts / "
                f"{self.update_threads} during the update, "
                f"inter-op {self.interop_threads}")


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
        # EXACTLY the historical settings. No phase switching.
        profile = Profile(
            name="mac", cores=cores,
            update_threads=min(MAC_INTRA_THREADS, cores),
            rollout_threads=None, interop_threads=1,
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
    packed = os.environ.get("MICRODUCK_ENVS_PER_WORKER")
    print(f"  worker layout: {packed + ' envs per process (MICRODUCK_ENVS_PER_WORKER)' if packed else 'one process per env'}")


if __name__ == "__main__":
    main()
