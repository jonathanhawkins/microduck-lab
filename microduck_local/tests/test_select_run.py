"""`select-run`: pick a locomotion run's shipping policy by ACHIEVED GROUND SPEED.

This file exists because best-checkpoint selection was tried in this repo
once and reverted. `train_behavior.py` records why: the criterion was
`keep_pace * ep_len`, which is ~90% ep_len, so it collapsed into
"longest-surviving" — and the pace term pays 0.29 at ZERO velocity, so a
motionless duck scored well. Its comment names the condition for bringing
selection back: score ACHIEVED GROUND SPEED, not a reward term.

So the tests below pin the two things that make this criterion different
from the one that failed: a standing policy scores ~0, and survival is a
REJECTION FLOOR rather than a quantity traded against speed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.select_run import METRIC, checkpoints

MJCF = C.MICRODUCK_RL_DIR
pytestmark = pytest.mark.skipif(
    not MJCF.exists(), reason="upstream microduck_rl checkout not found")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    """One short REAL locomotion run with numbered checkpoints."""
    root = tmp_path_factory.mktemp("runs")
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(root), MICRODUCK_VEC_ENV="fork")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", "tiny", "--envs", "4", "--steps", "12000",
         "--snap-steps", "6000", "--checkpoint-every", "6000"],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return root / "tiny"


def test_checkpoints_are_kept_only_when_asked(run_dir, tmp_path):
    """`--checkpoint-every` is off by default, because `live.onnx` and
    `model.zip` being overwritten every snapshot is the long-standing
    behavior and this must not change it silently."""
    cks = checkpoints(run_dir)
    tags = [t for t, _, _ in cks]
    assert "final" in tags and tags[-1] == "final" and len(tags) >= 2
    for _, m, vn in cks:
        assert m.exists() and vn.exists()

    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path), MICRODUCK_VEC_ENV="fork")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", "nockpt", "--envs", "4", "--steps", "6000", "--snap-steps", "3000"],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert not (tmp_path / "nockpt" / "checkpoints").exists()
    assert [t for t, _, _ in checkpoints(tmp_path / "nockpt")] == ["final"]


def _select(run_dir, *extra):
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(run_dir.parent))
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.select_run", "tiny",
         "--episodes", "2", "--seeds", "123", "--jobs", "4", "--json", *extra],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_criterion_is_speed_and_every_candidate_is_scored(run_dir):
    res = _select(run_dir, "--dry-run")
    assert res["metric"] == METRIC == "speed"
    assert {r["tag"] for r in res["table"]} == {t for t, _, _ in checkpoints(run_dir)}
    for row in res["table"]:
        # Speed is a measured body-frame velocity, never a reward term.
        assert np.isfinite(row["speed"]) and "reward" not in row


def test_a_run_where_everything_falls_ships_nothing(run_dir):
    """The failure this tool must not repeat. A 12k-step policy cannot walk,
    so every candidate falls — and among policies that all fall, "fastest"
    picks the one diving forward hardest. It must refuse, not ship that."""
    res = _select(run_dir)                       # NOT --dry-run: it may write
    assert res["viable"] is False
    assert res["shipped"] is None, "a run with no viable policy must ship nothing"
    assert len(res["rejected"]) == len(res["table"])
    assert not (run_dir / "selected.json").exists()


def test_a_high_fall_floor_makes_the_same_run_shippable(run_dir):
    """The floor is the whole mechanism, so moving it has to move the verdict
    — otherwise the guard is decorative."""
    res = _select(run_dir, "--max-falls", "1.0", "--dry-run")
    assert res["viable"] is True and res["rejected"] == []
    assert res["best"]["speed"] == max(r["speed"] for r in res["table"])


def test_a_non_locomotion_recipe_is_refused_rather_than_scored(run_dir):
    """A trick has no ground speed to be judged on, and inventing a fallback
    criterion is exactly how the reverted attempt went wrong."""
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(run_dir.parent))
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.select_run", "tiny",
         "--behavior", "one_leg", "--episodes", "1", "--seeds", "123", "--jobs", "1"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "not a locomotion recipe" in (r.stdout + r.stderr)


def test_scoring_pins_the_command_for_every_episode_and_restores_it():
    """Two bugs in one test, both hit for real.

    `$MICRODUCK_RUN_CMD` is read by the command sampler on EVERY episode
    reset, not at construction. Leaking it broke a command-mix assertion in
    another test file (the scorer is called in-process here); restoring it
    too early un-pinned the command and dropped the shipped walker from
    0.196 to 0.130 m/s. It has to hold for the whole call and be gone after.
    """
    from microduck_local.brain.brain_env import POLICIES_DIR
    from microduck_local.select_run import _score_one

    teacher = POLICIES_DIR / "alpha_walking.onnx"
    if not teacher.exists():
        pytest.skip("shipped policies not found")

    before = os.environ.get("MICRODUCK_RUN_CMD")
    pinned = _score_one((str(teacher), "t", 123, 3, "run", 0.4, None))
    assert os.environ.get("MICRODUCK_RUN_CMD") == before, "the scorer leaked the pin"

    free = _score_one((str(teacher), "t", 123, 3, "run", None, None))
    assert os.environ.get("MICRODUCK_RUN_CMD") == before

    # Pinning has to actually do something, or "restored" is trivially true.
    assert pinned["speed"] > 0.15, "a pinned 0.4 command must produce a real walk"
    assert pinned["speed"] > free["speed"] + 0.02, (
        f"pinned {pinned['speed']:.3f} vs free {free['speed']:.3f} — the pin did nothing")
