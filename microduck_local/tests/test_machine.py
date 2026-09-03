"""The per-machine profile — and above all, that a Mac still gets exactly
what it got before `machine.py` existed.

This repo is tuned, measured and advertised on Apple Silicon. Everything the
Linux/cloud profile does (phase-aware torch threads, worker packing) is a
change to a machine whose numbers were measured somewhere else, so the mac
profile is pinned here term by term: same intra-op ceiling, same inter-op,
one worker process per env, and NO callback in the training loop at all.
"""

import os
import sys

import pytest

from microduck_local import machine
from microduck_local.machine import (
    MAC_INTRA_THREADS,
    Profile,
    build_profile,
    detect_platform,
    parse_cpu_max,
    phase_thread_callbacks,
    usable_cores,
)


@pytest.fixture(autouse=True)
def _clean_profile_env(monkeypatch):
    """Every test starts from an unforced, uncached profile."""
    for var in ("MICRODUCK_PROFILE", "MICRODUCK_ROLLOUT_THREADS",
                "MICRODUCK_UPDATE_THREADS"):
        monkeypatch.delenv(var, raising=False)
    machine.reset_cache()
    yield
    machine.reset_cache()


# ------------------------------------------------------------ the mac default


def test_mac_profile_is_the_historical_settings():
    """intra-op min(8, cores), inter-op 1 — what configure_torch_cpu did
    when it read `min(TORCH_INTRA_THREADS, os.cpu_count())` directly."""
    prof = build_profile("mac", cores=18)
    assert prof.update_threads == MAC_INTRA_THREADS == 8
    assert prof.interop_threads == 1
    # A small Mac gets its own core count, not the 8 ceiling.
    assert build_profile("mac", cores=4).update_threads == 4


def test_mac_profile_never_switches_threads_between_phases():
    """`rollout_threads is None` is the promise that nothing re-threads torch
    mid-run: the 21-24% pinning REGRESSION measured on an M5 Max is what
    happens when a Mac's update loses its cores."""
    prof = build_profile("mac", cores=18)
    assert prof.rollout_threads is None
    assert prof.phase_threads is False


def test_mac_profile_adds_no_callback_to_the_training_loop():
    """Not even a no-op: an extra callback costs a per-step call on the
    hottest loop in the repo, and a Mac run must be untouched."""
    from stable_baselines3.common.callbacks import BaseCallback
    assert phase_thread_callbacks(BaseCallback, build_profile("mac", cores=18)) == []


def test_with_phase_callbacks_hands_a_mac_run_the_very_same_object():
    """`with_phase_callbacks` returns its argument IDENTICALLY on the mac
    profile — not a one-element list. A list would make SB3 wrap it in a
    CallbackList, adding a call per step to the hottest loop in the repo for
    a machine that wanted nothing."""
    from stable_baselines3.common.callbacks import BaseCallback

    from microduck_local.machine import with_phase_callbacks

    class Noop(BaseCallback):       # BaseCallback itself is abstract
        def _on_step(self) -> bool:
            return True

    mac, linux = build_profile("mac", cores=18), build_profile("linux", cores=4)
    cb = Noop()
    assert with_phase_callbacks(cb, BaseCallback, mac) is cb
    listed = [cb]
    assert with_phase_callbacks(listed, BaseCallback, mac) is listed

    # On Linux the trainer's callback keeps its place at the front.
    got = with_phase_callbacks(cb, BaseCallback, linux)
    assert isinstance(got, list) and got[0] is cb and len(got) == 2
    got = with_phase_callbacks([cb], BaseCallback, linux)
    assert got[0] is cb and len(got) == 2


def test_mac_profile_keeps_one_worker_process_per_env():
    """Packing is a Linux-profile change. 32 envs on an 18-core Mac is the
    shipped default and stays 32 processes, which is what every number in
    the README was measured with."""
    prof = build_profile("mac", cores=18)
    for n_envs in (4, 16, 32, 48, 64):
        assert prof.envs_per_worker(n_envs) == 1


# ---------------------------------------------------------- the linux profile


def test_linux_profile_gives_the_rollout_one_thread_and_the_update_all_of_them():
    prof = build_profile("linux", cores=4)
    assert prof.rollout_threads == 1
    assert prof.update_threads == 4
    assert prof.phase_threads is True


def test_linux_profile_packs_workers_only_once_envs_outnumber_cores():
    """Packing exists to stop the parent paying two semaphore ops per env
    per vec-step for processes that cannot run in parallel anyway. Below
    that point there is nothing to save."""
    prof = build_profile("linux", cores=4)
    assert prof.envs_per_worker(4) == 1       # one env per core already
    assert prof.envs_per_worker(2) == 1
    assert prof.envs_per_worker(16) == 4      # -> 4 workers
    assert prof.envs_per_worker(32) == 8      # -> 4 workers
    # A big cloud box has a core per env, so it packs nothing.
    assert build_profile("linux", cores=64).envs_per_worker(32) == 1


def test_packing_always_divides_the_fleet_into_at_most_core_count_workers():
    for cores in (1, 2, 3, 4, 8, 12, 16):
        prof = build_profile("linux", cores=cores)
        for n_envs in (1, 4, 7, 16, 32, 48, 64):
            k = prof.envs_per_worker(n_envs)
            workers = -(-n_envs // k)
            assert k >= 1
            assert workers <= max(cores, 1) or n_envs <= cores


def test_linux_callback_rethreads_torch_between_phases():
    """The whole optimization, at its seam: SB3 fires on_rollout_start before
    collection and on_rollout_end after it, so those two hooks ARE the phase
    boundary. Thread counts do not enter the PPO math — this changes
    throughput and nothing else."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    before = torch.get_num_threads()
    try:
        cbs = phase_thread_callbacks(BaseCallback, build_profile("linux", cores=4))
        assert len(cbs) == 1
        cb = cbs[0]
        cb._on_rollout_start()
        assert torch.get_num_threads() == 1
        cb._on_rollout_end()
        assert torch.get_num_threads() == 4
        cb._on_rollout_start()
        assert torch.get_num_threads() == 1
        assert cb._on_step() is True
    finally:
        torch.set_num_threads(before)


# ------------------------------------------------------------ core counting


def test_usable_cores_is_at_least_one_and_never_more_than_the_host():
    cores = usable_cores()
    assert 1 <= cores <= (os.cpu_count() or 1)


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"),
                    reason="affinity is Linux-only")
def test_usable_cores_respects_cpu_affinity():
    """`taskset -c 0,1` must mean two threads, not one per host core — this
    is how a shared CI runner or a per-seed pinned training job appears."""
    assert usable_cores() <= len(os.sched_getaffinity(0))


def test_parse_cpu_max_reads_a_container_quota():
    """A 4-CPU Kubernetes limit on a 64-core node. Getting this wrong is the
    original pathology at full scale: 64 spinning threads on 4 cores."""
    assert parse_cpu_max("400000 100000") == 4
    assert parse_cpu_max("150000 100000") == 1     # 1.5 cores floors to 1
    assert parse_cpu_max("50000 100000") == 1      # half a core still runs
    assert parse_cpu_max("max 100000") is None     # unlimited
    assert parse_cpu_max("") is None
    assert parse_cpu_max("garbage here") is None


# ---------------------------------------------------------------- detection


def test_detect_platform_defaults_to_mac_only_on_darwin():
    expected = "mac" if sys.platform == "darwin" else "linux"
    assert detect_platform() == expected


def test_profile_can_be_forced_from_the_environment(monkeypatch):
    """`MICRODUCK_PROFILE=mac` on a Linux box is how the A/B behind the
    shipped default is reproduced (bench-envs --compare-profiles)."""
    monkeypatch.setenv("MICRODUCK_PROFILE", "mac")
    assert detect_platform() == "mac"
    assert build_profile().name == "mac"
    monkeypatch.setenv("MICRODUCK_PROFILE", "linux")
    assert build_profile().name == "linux"


def test_an_unknown_forced_profile_is_an_error_not_a_silent_default(monkeypatch):
    monkeypatch.setenv("MICRODUCK_PROFILE", "m4-ultra")
    with pytest.raises(ValueError, match="m4-ultra"):
        detect_platform()


def test_thread_counts_can_be_overridden_per_phase(monkeypatch):
    monkeypatch.setenv("MICRODUCK_ROLLOUT_THREADS", "2")
    monkeypatch.setenv("MICRODUCK_UPDATE_THREADS", "6")
    prof = build_profile("linux", cores=4)
    assert (prof.rollout_threads, prof.update_threads) == (2, 6)
    # An override turns phase switching ON even for the mac profile, which is
    # how a Mac owner tries the Linux finding without editing code.
    mac = build_profile("mac", cores=18)
    assert mac.phase_threads is True and mac.rollout_threads == 2


def test_garbage_thread_overrides_are_ignored(monkeypatch):
    monkeypatch.setenv("MICRODUCK_ROLLOUT_THREADS", "not-a-number")
    assert build_profile("mac", cores=18).rollout_threads is None


def test_profile_is_detected_once_per_process(monkeypatch):
    monkeypatch.setenv("MICRODUCK_PROFILE", "mac")
    first = machine.profile()
    monkeypatch.setenv("MICRODUCK_PROFILE", "linux")
    assert machine.profile() is first, "profile() must not re-read the env"
    machine.reset_cache()
    assert machine.profile().name == "linux"


def test_describe_names_the_numbers_a_run_is_about_to_use():
    text = build_profile("linux", cores=4).describe()
    assert "linux" in text and "4 usable cores" in text
    assert "packing on" in text
    assert "unchanged" in build_profile("mac", cores=8).describe()


# ------------------------------------------------------- the wiring downstream


def test_configure_torch_cpu_applies_the_profile(monkeypatch):
    """ppo_hparams is the only place the trainers set threads, and it must
    read the profile rather than os.cpu_count()."""
    import torch

    from microduck_local.ppo_hparams import TORCH_INTRA_THREADS, configure_torch_cpu

    assert TORCH_INTRA_THREADS == MAC_INTRA_THREADS  # back-compat re-export
    before = torch.get_num_threads()
    try:
        monkeypatch.setenv("MICRODUCK_PROFILE", "linux")
        machine.reset_cache()
        configure_torch_cpu(torch)
        assert torch.get_num_threads() == machine.profile().update_threads
    finally:
        torch.set_num_threads(before)


def test_vec_env_packs_workers_per_the_profile(monkeypatch):
    """The integration point: `make_vec_env` must ask the profile, and an
    explicit MICRODUCK_ENVS_PER_WORKER must still win over it."""
    from microduck_local.train_behavior import make_env
    from microduck_local.vec_env import make_vec_env

    monkeypatch.delenv("MICRODUCK_ENVS_PER_WORKER", raising=False)
    fns = [make_env("one_leg", i, 3) for i in range(4)]

    monkeypatch.setenv("MICRODUCK_PROFILE", "mac")
    machine.reset_cache()
    venv = make_vec_env(fns, backend="fork")
    try:
        assert venv.num_workers == 4      # one process per env, as on a Mac
    finally:
        venv.close()

    monkeypatch.setenv("MICRODUCK_PROFILE", "linux")
    monkeypatch.setenv("MICRODUCK_ENVS_PER_WORKER", "2")
    machine.reset_cache()
    venv = make_vec_env(fns, backend="fork")
    try:
        assert venv.num_workers == 2      # the explicit override
    finally:
        venv.close()


def test_a_forced_profile_survives_into_a_bench_point_subprocess():
    """bench-envs measures points in fresh interpreters, so a forced profile
    has to travel in the environment or --compare-profiles measures the same
    arm twice."""
    from microduck_local.bench_envs import _spawn_point

    # Not a real measurement: a 0-step budget still builds the trainer, and
    # the point it returns records which profile the child resolved.
    point = _spawn_point(2, "fork", False, 1, "one_leg", 0, timeout=300.0,
                         profile_name="mac")
    assert point is not None, "point subprocess failed"
    assert point.profile == "mac"
    assert point.per_worker == 1


def test_profile_dataclass_is_frozen():
    """A profile is read all over the trainers; nobody may mutate one."""
    with pytest.raises(Exception):
        build_profile("mac", cores=8).cores = 99  # type: ignore[misc]
    assert isinstance(build_profile("mac", cores=8), Profile)
