"""describe-brain writes a run's human name and description into brain.json
and touches nothing else — the recipe and `selected` it sits beside are
what the /train matrix and the deploy path read."""

import json

import pytest

from microduck_local.describe_brain import describe


def _brain(tmp_path, **extra):
    d = tmp_path / "p-n256-s31"
    d.mkdir()
    meta = {"name": "p-n256-s31", "task": "follow", "seed": 31, "net_arch": "256,256",
            "selected": {"tag": "000751104", "metric": "in_band", "score": 0.926, "final_score": 0.917},
            **extra}
    (d / "brain.json").write_text(json.dumps(meta))
    return d


def test_writes_both_fields_and_preserves_the_rest(tmp_path):
    d = _brain(tmp_path)
    describe(d, "Capacity sweep: 256-256", "  256-256 against p-n128 on six paired seeds: +0.006, unresolved.  ")
    meta = json.loads((d / "brain.json").read_text())
    assert meta["title"] == "Capacity sweep: 256-256"
    assert meta["description"] == "256-256 against p-n128 on six paired seeds: +0.006, unresolved."
    assert meta["seed"] == 31 and meta["net_arch"] == "256,256"
    assert meta["selected"]["score"] == 0.926


def test_none_leaves_a_field_alone(tmp_path):
    d = _brain(tmp_path, title="keep me", description="and me")
    describe(d, None, "new description")
    meta = json.loads((d / "brain.json").read_text())
    assert meta["title"] == "keep me"
    assert meta["description"] == "new description"


def test_refuses_to_invent_a_brain(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe(tmp_path / "not-a-run", "x", "y")
