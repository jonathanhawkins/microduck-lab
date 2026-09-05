"""Batteries are resumable (`--out` / `--tag`): a 12-seed 3v3 battery is the
best part of an hour, and a machine that reclaims its container mid-run
should cost the seed it was on, not the battery. It cost two whole batteries
before the benchmarks streamed and resumed."""

import json

import pytest

from microduck_local.eval_pitch import load_done


def _row(seed, tag="", per_side=3, seconds=300.0, **kw):
    return {"seed": seed, "tag": tag, "perSide": per_side, "seconds": seconds, **kw}


def test_no_file_means_nothing_measured_yet(tmp_path):
    assert load_done(None, "", 3, 300.0) == {}
    assert load_done(str(tmp_path / "never-written.jsonl"), "", 3, 300.0) == {}


def test_finished_seeds_come_back_keyed_by_seed_and_blank_lines_are_skipped(tmp_path):
    f = tmp_path / "poacher.jsonl"
    f.write_text("\n".join([json.dumps(_row(0, "poacher", left=1)), "",
                            json.dumps(_row(1, "poacher", left=2)), ""]))
    done = load_done(str(f), "poacher", 3, 300.0)
    assert set(done) == {0, 1} and done[1]["left"] == 2


def test_a_file_written_under_other_settings_is_refused_not_silently_mixed(tmp_path):
    """The brain's own parameters never appear in a row, so `--tag` is how a
    caller says which variant a file belongs to; the roster and the run length
    are checked too. Mixing them would fabricate a comparison."""
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps(_row(0, "poacher")) + "\n")
    for tag, per_side, seconds in [("shipped", 3, 300.0), ("poacher", 2, 300.0), ("poacher", 3, 120.0)]:
        with pytest.raises(SystemExit, match="Write a different variant"):
            load_done(str(f), tag, per_side, seconds)
    assert set(load_done(str(f), "poacher", 3, 300.0)) == {0}       # the matching one still loads
