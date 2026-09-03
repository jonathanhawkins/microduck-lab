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
