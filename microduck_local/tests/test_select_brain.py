"""`select-brain`: pick a run's shipping brain by deterministic benchmark score.

The premise these tests pin is the one AGENTS.md states twice — a reward
curve measures the noise-crutched STOCHASTIC policy while the exported ONNX
is the deterministic mean, so the last rollout's reward cannot choose which
checkpoint ships. `train_behavior` tried reward-term selection and reverted
it; the difference here is that the criterion is an ACHIEVED benchmark
metric (the fraction of decisions actually inside the follow band), measured
on the deterministic export, which is exactly what that reverted comment
asks a future selector to score.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import REGISTRY
from microduck_local.select_brain import METRIC, checkpoints

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"
pytestmark = pytest.mark.skipif(
    not (POLICIES / "alpha_walking.onnx").exists(), reason="upstream checkouts not found")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    """One tiny real run with checkpoints, shared by the module."""
    root = tmp_path_factory.mktemp("brains")
    env = dict(os.environ, MICRODUCK_BRAINS_DIR=str(root))
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_brain", "--run-name", "tiny",
         "--envs", "2", "--steps", "1200", "--n-steps", "32", "--checkpoint-every", "400"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return root / "tiny"


def test_checkpoints_lists_every_candidate_including_the_incumbent(run_dir):
    """The final model MUST be in the candidate set: a selection that cannot
    choose the artifact that would otherwise have shipped is not a comparison."""
    cks = checkpoints(run_dir)
    tags = [t for t, _, _ in cks]
    assert "final" in tags and tags[-1] == "final"
    assert len(tags) >= 3
    for _, model, vn in cks:
        assert model.exists() and vn.exists()
    assert tags[:-1] == sorted(tags[:-1]), "numbered checkpoints must be in step order"


def test_a_checkpoint_is_loadable_and_scoreable_without_being_shipped(run_dir, tmp_path):
    """`select-brain` scores a checkpoint by exporting it to a probe dir and
    loading it BY PATH, so nothing is written over brains/<run>/brain.onnx
    until a winner is chosen."""
    from microduck_local.train_brain import export_brain
    tag, model, vn = checkpoints(run_dir)[0]
    before = (run_dir / "brain.onnx").read_bytes()
    probe = tmp_path / "probe"
    export_brain(run_dir, model_path=model, vn_path=vn, out=probe / "brain.onnx")
    (probe / "brain.json").write_text((run_dir / "brain.json").read_text())

    brain = REGISTRY.make(f"learned:{probe}")
    assert brain.obs_version == json.loads((run_dir / "brain.json").read_text())["obs_version"]
    out = brain.infer(np.zeros(80, np.float32))
    assert out.shape == (3,) and np.isfinite(out).all()
    assert (run_dir / "brain.onnx").read_bytes() == before, "scoring must not ship"


def test_dry_run_scores_every_candidate_and_ships_nothing(run_dir):
    env = dict(os.environ, MICRODUCK_BRAINS_DIR=str(run_dir.parent))
    before = (run_dir / "brain.onnx").read_bytes()
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.select_brain", "tiny", "--episodes", "1",
         "--seeds", "100", "--preset", "datasheet", "--jobs", "4", "--dry-run", "--json"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["metric"] == METRIC
    assert {row["tag"] for row in res["table"]} == {t for t, _, _ in checkpoints(run_dir)}
    # The winner is the best-scoring candidate, by the metric, over the table.
    assert res["best"][METRIC] == max(row[METRIC] for row in res["table"])
    assert res["best"][METRIC] >= res["final"][METRIC], "the winner cannot be worse than the incumbent"
    assert (run_dir / "brain.onnx").read_bytes() == before, "--dry-run must ship nothing"


def test_shipping_replaces_brain_onnx_and_records_why(run_dir):
    env = dict(os.environ, MICRODUCK_BRAINS_DIR=str(run_dir.parent))
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.select_brain", "tiny", "--episodes", "1",
         "--seeds", "100", "--preset", "datasheet", "--jobs", "4", "--json"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    meta = json.loads((run_dir / "brain.json").read_text())
    # The record has to say which checkpoint shipped and what it beat, or a
    # later reader cannot tell a selected brain from a final-checkpoint one.
    assert meta["selected"]["tag"] == res["best"]["tag"]
    assert meta["selected"]["metric"] == METRIC
    assert meta["selected"]["score"] == pytest.approx(res["best"][METRIC], abs=1e-4)
    assert meta["obs_version"] == 2, "the contract fields must survive the rewrite"
    # The shipped file is byte-identical to the winner's probe export.
    winner = run_dir / ".probe" / res["best"]["tag"] / "brain.onnx"
    assert (run_dir / "brain.onnx").read_bytes() == winner.read_bytes()
    # And it still loads as a brain (by path: brains_dir() is read from the
    # environment at call time, and this process is not the one that was
    # pointed at the temporary brains dir).
    reloaded = REGISTRY.make(f"learned:{run_dir}")
    assert reloaded.obs_version == 2
    assert np.isfinite(reloaded.infer(np.zeros(80, np.float32))).all()


def test_the_onnx_export_is_cached_across_scoring_passes(run_dir, tmp_path):
    """Every invocation used to re-export every candidate through
    `torch.onnx.export`, the slowest part of a scoring pass — and re-scoring
    one run under different seeds or presets is the normal way these tools
    are used. Measured 2.25 s cold against 1.17 s warm on five candidates."""
    from microduck_local.select_brain import _cached_export, checkpoints
    from microduck_local.train_brain import export_brain

    # Its OWN output path, not the shared `.probe` dir — an earlier test in
    # this file already scored the run and populated that, so asserting a
    # cold miss against it would depend on test order.
    tag, model, vn = checkpoints(run_dir)[0]
    out = tmp_path / "cache" / tag / "brain.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Copies, so touching them cannot disturb the shared fixture.
    model_c, vn_c = tmp_path / "m.zip", tmp_path / "vn.pkl"
    shutil.copy(model, model_c)
    shutil.copy(vn, vn_c)
    model, vn = model_c, vn_c

    assert _cached_export(export_brain, run_dir, model, vn, out) is True, "cold miss"
    stamp = out.stat().st_mtime_ns
    assert _cached_export(export_brain, run_dir, model, vn, out) is False, "warm hit"
    assert out.stat().st_mtime_ns == stamp, "a hit must not rewrite the file"

    # A newer checkpoint MUST invalidate: a stale ONNX would score the wrong
    # policy and say nothing about it.
    os.utime(model, ns=(stamp + 10**9, stamp + 10**9))
    assert _cached_export(export_brain, run_dir, model, vn, out) is True, (
        "a checkpoint newer than its export must force a re-export")
    # And so must a newer normalizer, which is baked into the ONNX.
    stamp2 = out.stat().st_mtime_ns
    os.utime(vn, ns=(stamp2 + 10**9, stamp2 + 10**9))
    assert _cached_export(export_brain, run_dir, model, vn, out) is True
