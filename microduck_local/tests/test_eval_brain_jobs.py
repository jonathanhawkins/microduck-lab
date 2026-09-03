"""`eval-brain --jobs N`: what it keeps, and the one thing it cannot.

The episodes of one battery are chained — through the sensors' own random
generators (`TofSensor.rng`, `Detector.rng`) and through the duck's leftover
`twist_cmd`, none of which `BrainEnv.reset(seed=)` clears — so they cannot
be split across processes and still come out bit-identical. These tests pin
all three halves of that:

* `--jobs 1` still runs the original serial loop, row for row;
* every `--jobs > 1` gives the SAME answer as every other (a benchmark
  number must never depend on the core count of the machine that took it);
* and the chain itself, so nobody "fixes" `--jobs` by accident and quietly
  re-bases the published follow table.
"""

from __future__ import annotations

import numpy as np
import pytest

from microduck_local.brain import REGISTRY
from microduck_local.brain.brain_env import BrainEnv, FollowTask
from microduck_local.eval_brain import (
    SUMMARY_KEYS,
    _episode,
    _make,
    _run_episodes,
    obs_version_of,
    run,
)

# Short episodes (40 decisions) and a fixed preset: the whole file is a few
# seconds of simulation, and the pool start-ups dominate it.
TASK = FollowTask(episode_s=4.0)
PRESET, SEED, EPISODES = "hostile", 100, 4


def _rows_equal(a: list[dict], b: list[dict]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def _original_serial(brain_kind: str, preset: str | None, episodes: int, seed: int, task: FollowTask) -> list[dict]:
    """`eval_brain.run`'s loop as it stood before `--jobs` existed, copied
    verbatim, so `--jobs 1` is measured against the code it replaced and not
    against itself."""
    brain = REGISTRY.make(brain_kind)
    env = BrainEnv(task, seed=seed, fixed_preset=preset, sense_dr=preset is None,
                   obs_version=obs_version_of(brain))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        brain.reset()
        in_band = err = seen = bumps = contact = 0.0
        n = 0
        ret = 0.0
        falls = 0
        while True:
            s = env.senses()
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
        rows.append({"return": ret, "in_band": in_band / n, "dist_err": err / n,
                     "seen": seen / n, "bumps": bumps, "contact": contact / 10.0, "falls": falls, "decisions": n,
                     "dodges": int(brain.closing.count if hasattr(brain, "closing") else info.get("dodges", 0))})
    return rows


def test_jobs_1_is_the_untouched_serial_path():
    """The default takes no pool and changes no number: the same rows, and
    the same summary means over them, as the pre-`--jobs` loop."""
    res = run("follow", PRESET, EPISODES, SEED, TASK)
    want = _original_serial("follow", PRESET, EPISODES, SEED, TASK)
    assert _rows_equal(res["rows"], want)
    for k in SUMMARY_KEYS:
        assert res[k] == float(np.mean([r[k] for r in want])), k
    assert res["sampling"] == "chained" and res["jobs"] == 1


def test_jobs_1_is_reproducible():
    a = run("follow", PRESET, EPISODES, SEED, TASK)
    b = run("follow", PRESET, EPISODES, SEED, TASK)
    assert _rows_equal(a["rows"], b["rows"])
    assert all(a[k] == b[k] for k in SUMMARY_KEYS)


def test_a_parallel_battery_does_not_depend_on_the_core_count():
    """The property that makes `--jobs` usable at all: 2, 3 and 4 processes
    give one answer, bit for bit — the summary means, the charge count, and
    the episode ORDER of the rows (the shares are dealt round robin, so a
    worker's rows come back interleaved and have to be sorted home)."""
    two = run("follow", PRESET, EPISODES, SEED, TASK, jobs=2)
    three = run("follow", PRESET, EPISODES, SEED, TASK, jobs=3)
    four = run("follow", PRESET, EPISODES, SEED, TASK, jobs=4)
    assert _rows_equal(two["rows"], three["rows"])
    assert _rows_equal(two["rows"], four["rows"])
    for k in SUMMARY_KEYS:
        assert two[k] == three[k] == four[k], k
    assert two["charges"] == three["charges"] == four["charges"]
    assert two["sampling"] == "independent" and two["jobs"] == 2


def test_parallel_is_exactly_its_own_sampling_run_on_one_process():
    """The pool adds nothing of its own: the same episodes run one after
    another in this process, under the same per-episode seeding, are the
    rows the workers hand back."""
    got = run("follow", PRESET, EPISODES, SEED, TASK, jobs=3)
    serial = _run_episodes("follow", PRESET, list(range(EPISODES)), SEED, TASK, False)
    assert _rows_equal(got["rows"], [row for _, row in serial["rows"]])
    assert got["charges"] == serial["charges"]


def test_the_summary_is_the_same_expression_over_the_same_rows():
    """`contact` is a per-row /10 and `dodges` a per-row count — the parent
    must not re-derive either when it aggregates a pool's rows."""
    res = run("follow", PRESET, EPISODES, SEED, TASK, jobs=2)
    for k in SUMMARY_KEYS:
        assert res[k] == float(np.mean([r[k] for r in res["rows"]])), k
    assert len(res["rows"]) == EPISODES == res["episodes"]


def test_every_parallel_episode_is_a_pure_function_of_seed_and_index():
    """What the fresh-env-per-episode worker buys: episode `ep` is the same
    row wherever and whenever it is drawn, so a battery can be cut up any
    way at all."""
    whole = run("follow", PRESET, EPISODES, SEED, TASK, jobs=2)["rows"]
    for ep in range(EPISODES):
        alone = _run_episodes("follow", PRESET, [ep], SEED, TASK, False)["rows"]
        assert alone[0][1] == whole[ep], ep
    backwards = _run_episodes("follow", PRESET, list(reversed(range(EPISODES))), SEED, TASK, False)
    assert _rows_equal([row for _, row in sorted(backwards["rows"])], whole)


def test_episodes_are_chained_so_jobs_cannot_be_bit_identical():
    """WHY `--jobs > 1` is a different number, pinned so it cannot rot.

    A fresh env owns episode 0 and nothing after it. Two things ride across
    `BrainEnv.reset()`: the sensors' generators (neither `TofSensor.reset()`
    nor `Detector.reset()` nor `World.reset()` reseeds them) and the duck's
    last `twist_cmd`, which `World._respawn` leaves standing and which then
    drives the five warm-up steps `reset()` takes before the first decision.
    If this test ever fails because the two agree, the env stopped carrying
    state — and `--jobs > 1` can be made exact and the docstring rewritten.
    """
    chained = run("follow", PRESET, EPISODES, SEED, TASK)["rows"]
    independent = run("follow", PRESET, EPISODES, SEED, TASK, jobs=2)["rows"]
    assert chained[0] == independent[0], "a fresh env must still own episode 0"
    assert chained[1:] != independent[1:], "the env stopped carrying state — see the docstring"

    # Carrier 2, directly: the walk command outlives the reset that is
    # supposed to put the duck back on its spawn.
    brain, env = _make("follow", PRESET, SEED, TASK, False)
    env.reset(seed=SEED)
    brain.reset()
    _episode(env, brain, TASK)
    env.world.reset()
    assert np.any(np.asarray(env.duck.twist_cmd) != 0.0), "twist_cmd is no longer carried"


def test_small_batteries_and_jobs_1_never_start_a_pool(monkeypatch):
    """One episode has nothing to split, and jobs<=1 must not touch a pool
    at all — `--jobs 1` is the original path, not a one-worker pool."""
    import multiprocessing as mp
    monkeypatch.setattr(mp, "get_context", lambda *a, **k: pytest.fail("a pool was started"))
    for jobs in (0, 1):
        assert run("follow", PRESET, 2, SEED, TASK, jobs=jobs)["sampling"] == "chained"
    assert run("follow", PRESET, 1, SEED, TASK, jobs=8)["episodes"] == 1


def test_a_worker_is_single_threaded():
    import os

    from microduck_local.eval_brain import _single_threaded
    _single_threaded()
    assert os.environ["OMP_NUM_THREADS"] == "1" and os.environ["MKL_NUM_THREADS"] == "1"
