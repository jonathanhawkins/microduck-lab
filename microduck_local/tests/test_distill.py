"""The distilled warm start for locomotion recipes.

Why it exists (README, measured): under BAM every policy trained here from
scratch either falls (18/20) or is stable and slow (0.13 m/s), while the
shipped `alpha_walking` is stable AND 0.21 m/s. Cloning it turns "discover a
stable gait from nothing" — which this sample budget cannot do — into "make
an existing gait faster", which it might. Confirmed again in this session:
two 1.5M-step from-scratch runs reached `ep_rew` 2300-2900 and **0.001 m/s**,
i.e. they learned to stand still and not fall.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.distill import default_teacher, ensure_distilled

TEACHER = default_teacher()
pytestmark = pytest.mark.skipif(
    not TEACHER.exists() or not C.MICRODUCK_RL_DIR.exists(),
    reason="upstream checkouts not found")

TINY = dict(episodes=3, epochs=2)


def test_the_clone_is_cached_and_keyed_on_teacher_and_seed(tmp_path):
    """Cloning costs a few hundred episodes plus the fit, which is pure
    overhead to pay on every launch — but a stale cache that ignored a
    swapped teacher would be worse than no cache."""
    a = ensure_distilled(seed=0, cache_dir=tmp_path, **TINY)
    assert (a / "model.zip").exists() and (a / "vecnormalize.pkl").exists()
    meta = json.loads((a / "distill.json").read_text())
    assert meta["seed"] == 0 and meta["teacher"] == str(TEACHER)
    assert meta["mse"] < 0.05, "a clone this far off is not a warm start"

    mtime = (a / "model.zip").stat().st_mtime_ns
    again = ensure_distilled(seed=0, cache_dir=tmp_path, **TINY)
    assert again == a
    assert (a / "model.zip").stat().st_mtime_ns == mtime, "cache hit must not refit"

    b = ensure_distilled(seed=1, cache_dir=tmp_path, **TINY)
    assert b != a, "a different seed is a different clone"


def test_a_missing_teacher_is_an_error_not_a_silent_scratch_start(tmp_path):
    """Silently falling back to from-scratch would hand back a policy that
    stands still, with nothing in the log to say why."""
    with pytest.raises(FileNotFoundError):
        ensure_distilled(tmp_path / "nope.onnx", cache_dir=tmp_path, **TINY)


def test_the_clone_walks_at_roughly_the_teachers_speed(tmp_path):
    """The point of the whole exercise. Scored the same way `select-run`
    scores a checkpoint: deterministic rollouts, mean body-frame forward
    velocity. The teacher reads 0.196 m/s on this harness."""
    from microduck_local.export_onnx import export
    from microduck_local.select_run import _score_one

    d = ensure_distilled(seed=0, cache_dir=tmp_path, episodes=60, epochs=20)
    onnx = d / "policy.onnx"
    export(d, onnx)
    teacher = _score_one((str(TEACHER), "teacher", 123, 3, "run", 0.4, None))
    clone = _score_one((str(onnx), "clone", 123, 3, "run", 0.4, None))
    assert teacher["speed"] > 0.15, "the teacher itself must walk, or the harness is wrong"
    assert clone["speed"] > 0.5 * teacher["speed"], (
        f"clone {clone['speed']:.3f} m/s against teacher {teacher['speed']:.3f} — "
        "the gait did not survive cloning")


def test_distill_runs_after_the_fork_and_the_run_completes(tmp_path):
    """The trap this ordering exists for: `ensure_distilled` imports torch and
    builds an env, and a torch-initialized parent DEADLOCKS the macOS fork.
    Calling it before `make_vec_env` hung four real runs at 0% CPU with only
    behavior.json written. This is the end-to-end guard."""
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path),
               MICRODUCK_VEC_ENV="fork", MICRODUCK_DISTILL="1")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", "d", "--envs", "4", "--steps", "6000", "--snap-steps", "3000",
         "--distill-teacher", str(TEACHER)],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "warm-starting from the distilled walker" in r.stdout
    rows = [json.loads(x) for x in
            (tmp_path / "d" / "progress.jsonl").read_text().splitlines() if x.strip()]
    assert rows and rows[-1].get("done") is True, "the run must reach its end, not hang"
    assert (tmp_path / "d" / "policy.onnx").exists()


def test_a_trick_recipe_declines_the_walker_clone(tmp_path):
    """A cloned walker is the wrong prior for a headstand, and inheriting one
    silently would be worse than ignoring the flag."""
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path),
               MICRODUCK_VEC_ENV="fork", MICRODUCK_DISTILL="1")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "one_leg",
         "--run-name", "t", "--envs", "4", "--steps", "4000", "--snap-steps", "2000"],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "--distill ignored" in r.stdout
    assert "warm-starting from the distilled walker" not in r.stdout
    assert not (Path(tmp_path) / ".distill").exists(), "and it must not even clone"


def _build_cache_entry(cache, q):
    """Module level on purpose: `spawn` pickles the target by qualified name,
    so a function nested in the test cannot be one."""
    try:
        from microduck_local.distill import cache_is_valid, ensure_distilled
        d = ensure_distilled(seed=0, cache_dir=Path(cache), **TINY)
        q.put(("ok", str(d), cache_is_valid(d)))
    except Exception as exc:                       # pragma: no cover
        q.put(("err", f"{type(exc).__name__}: {exc}", False))


def test_a_half_written_cache_entry_is_not_a_hit(tmp_path):
    """`model.zip` EXISTING is not the same as it being usable. A lost race
    leaves a truncated one behind, and SB3 then fails deep inside training
    with "wasn't a zip-file" — AFTER the vec-env workers have forked. It cost
    two real runs. Validating on read means the entry is rebuilt instead."""
    from microduck_local.distill import cache_is_valid

    d = tmp_path / "entry"
    d.mkdir()
    assert cache_is_valid(d) is False, "nothing there"
    (d / "model.zip").write_bytes(b"not a zip at all")
    (d / "vecnormalize.pkl").write_bytes(b"x")
    assert cache_is_valid(d) is False, "present but corrupt must not be a hit"

    good = ensure_distilled(seed=0, cache_dir=tmp_path / "c", **TINY)
    assert cache_is_valid(good) is True


def test_two_concurrent_builds_of_the_same_key_do_not_corrupt_each_other(tmp_path):
    """Both arms of a paired A/B share a cache key — same teacher, same seed —
    and launch together. Before the atomic rename they both missed, both fit
    into the same directory, and interleaved their writes."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_build_cache_entry, args=(str(tmp_path), q))
          for _ in range(2)]
    for p in ps:
        p.start()
    results = [q.get(timeout=1800) for _ in ps]
    for p in ps:
        p.join(timeout=60)

    from microduck_local.distill import cache_is_valid
    assert all(r[0] == "ok" for r in results), results
    assert results[0][1] == results[1][1], "both must end up at the same entry"
    assert all(r[2] for r in results), "and it must be loadable, not half-written"
    assert cache_is_valid(Path(results[0][1]))
    # The loser's private directory must not be left lying around as a hit.
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())


def test_fitting_budget_is_set_by_what_keeps_the_clone_upright():
    """Cloning FIDELITY is what decides whether the clone falls, and the
    default fitting budget exists to buy it. Measured over five budgets on
    the deterministic export, correlation(action MSE, fall rate) = +0.93:

        episodes  epochs   mse rad^2   falls   ep_len
             120      30     0.00030    0.75      289
             250      40     0.00017    0.08      924   <- the old default
             250     120     0.00011    0.00     1000
             600     120     0.00006    0.00     1000

    Below ~0.00011 the falling stops outright. The extra epochs cost 33 s
    once (17 -> 50) and the result is cached, so this is close to free.
    """
    from microduck_local.distill import DISTILL_EPISODES, DISTILL_EPOCHS

    assert DISTILL_EPOCHS >= 120, (
        "40 epochs reaches ~0.00017 rad^2, which measured 8% falls")
    assert DISTILL_EPISODES >= 250


@pytest.mark.slow
def test_a_default_fidelity_clone_matches_the_teacher_and_does_not_fall(tmp_path):
    """The headline: at the default budget the clone walks at the teacher's
    speed and stays up for a full episode. Measured over 8 eval seeds —
    teacher 0.197 m/s / 1000 steps / 0.00 falls, clone 0.202 / 1000 / 0.00.
    Before the budget change it was 0.238 m/s while falling 75% of episodes,
    which is what makes 'fast' the wrong thing to have optimised for."""
    from microduck_local.export_onnx import export
    from microduck_local.select_run import _score_one

    d = ensure_distilled(seed=0, cache_dir=tmp_path)      # DEFAULT budget
    onnx = d / "policy.onnx"
    export(d, onnx)
    rs = [_score_one((str(onnx), "c", sd, 3, "run", 0.4, None)) for sd in (123, 456)]
    falls = float(np.mean([r["falls"] for r in rs]))
    speed = float(np.mean([r["speed"] for r in rs]))
    teach = [_score_one((str(TEACHER), "t", sd, 3, "run", 0.4, None)) for sd in (123, 456)]
    t_speed = float(np.mean([r["speed"] for r in teach]))
    assert falls == 0.0, f"the clone fell {falls:.0%} of episodes at the default budget"
    assert speed > 0.8 * t_speed, f"clone {speed:.3f} against teacher {t_speed:.3f}"
