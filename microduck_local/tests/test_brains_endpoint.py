"""GET /brains — what the viewer's /train page reads.

train-brain never talks to the lab, so this endpoint is a pure read of the
run directory. The cases that matter are the ragged ones: a run still being
written, one that shipped without its training log (every brain cloned from
the repo), and a progress file with a half-written last line.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from microduck_local import contract as C
from microduck_local import viz_server as V

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


@pytest.fixture
def brains(monkeypatch, tmp_path):
    d = tmp_path / "brains"
    d.mkdir()
    monkeypatch.setenv("MICRODUCK_BRAINS_DIR", str(d))
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return d


@pytest.fixture
def app(brains):
    return V.make_app([])


def age(path, seconds=120.0):
    """Backdate a progress file: `active` is an mtime test, and a file the
    test just wrote is legitimately fresh."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def write_run(root, name, *, rows=0, budget=400_000, onnx=True, card=True):
    d = root / name
    d.mkdir()
    if card:
        (d / "brain.json").write_text(json.dumps(
            {"name": name, "obs_version": 2, "envs": 12, "steps": budget,
             "seed": 0, "variety": True, "fixed_preset": None}))
    if onnx:
        (d / "brain.onnx").write_bytes(b"\0")
    if rows:
        per = budget // rows
        (d / "progress.jsonl").write_text("".join(
            json.dumps({"steps": (i + 1) * per, "ep_rew": 10.0 + i,
                        "ep_len": 200.0, "elapsed_s": (i + 1) * 0.5}) + "\n"
            for i in range(rows)))
    return d


def get(app) -> dict:
    with TestClient(app) as c:
        r = c.get("/brains")
        assert r.status_code == 200
        return {b["name"]: b for b in r.json()["brains"]}


def test_empty_and_missing_dir_are_not_errors(app, brains, monkeypatch):
    assert get(app) == {}
    monkeypatch.setenv("MICRODUCK_BRAINS_DIR", str(brains / "nope"))
    assert get(app) == {}


def test_a_finished_run_reports_its_curve_and_contract(app, brains):
    d = write_run(brains, "follow-v9", rows=8, budget=800_000)
    age(d / "progress.jsonl")
    b = get(app)["follow-v9"]
    assert b["shipped"] is True and b["active"] is False
    assert b["rollouts"] == 8 and len(b["curve"]) == 8
    assert b["obs_version"] == 2 and b["variety"] is True and b["steps"] == 800_000
    assert b["last"]["steps"] == 800_000
    assert b["progress"] == pytest.approx(1.0)
    # Finished: no ETA to offer, but the achieved rate is still meaningful.
    assert b["eta_s"] is None and b["steps_per_s"] > 0


def test_a_brain_from_a_clone_has_no_curve(app, brains):
    # Only brain.onnx + brain.json are committed — progress.jsonl stays local.
    write_run(brains, "follow-v2", rows=0)
    b = get(app)["follow-v2"]
    assert b["shipped"] is True and b["rollouts"] == 0 and b["curve"] == []
    assert b["active"] is False and "last" not in b


def test_a_run_mid_flight_is_active_with_an_eta(app, brains):
    d = write_run(brains, "running", rows=4, budget=1_000_000, onnx=False)
    # write_run's last row is the full budget; rewrite as a half-done run.
    (d / "progress.jsonl").write_text(
        json.dumps({"steps": 500_000, "ep_rew": 1.0, "ep_len": 200.0,
                    "elapsed_s": 250.0}) + "\n")
    b = get(app)["running"]
    assert b["active"] is True and b["shipped"] is False
    assert b["progress"] == pytest.approx(0.5)
    assert b["steps_per_s"] == pytest.approx(2000.0)
    assert b["eta_s"] == 250          # the remaining half at the same rate


def test_a_half_written_last_line_is_skipped_not_fatal(app, brains):
    d = write_run(brains, "torn", rows=3)
    age(d / "progress.jsonl")
    with (d / "progress.jsonl").open("a") as f:
        f.write('{"steps": 999, "ep_rew":')      # the trainer mid-write
    b = get(app)["torn"]
    assert b["rollouts"] == 3 and b["last"]["ep_rew"] == 12.0


def test_a_run_without_brain_json_still_lists(app, brains):
    # The contract is written at launch, but a killed or ancient run may
    # have none; the page must still show the curve rather than 500.
    d = write_run(brains, "bare", rows=2, card=False)
    age(d / "progress.jsonl")
    b = get(app)["bare"]
    assert b["rollouts"] == 2 and b["shipped"] is True
    # No contract on disk, so no budget to divide by — and `progress` must be
    # null rather than a bogus 0 or a crash.
    assert "steps" not in b and b["progress"] is None
    assert b["last"]["steps"] == 400_000


def test_a_long_curve_is_downsampled_but_keeps_its_ends(app, brains):
    n = 1200
    d = write_run(brains, "long", rows=n, budget=1_200_000)
    age(d / "progress.jsonl")
    b = get(app)["long"]
    assert b["rollouts"] == n
    assert len(b["curve"]) <= 401           # BRAIN_MAX_POINTS (+ the kept last)
    assert b["curve"][0]["steps"] == 1000   # first row survives
    assert b["curve"][-1]["steps"] == 1_200_000   # and so does the last
    assert b["last"]["steps"] == 1_200_000        # `last` is the true final row


def test_files_in_the_brains_dir_are_ignored(app, brains):
    (brains / "notes.txt").write_text("not a run")
    write_run(brains, "real", rows=1)
    assert set(get(app)) == {"real"}
