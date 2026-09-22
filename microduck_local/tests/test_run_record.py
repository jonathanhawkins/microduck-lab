"""What a run SAYS it is: run_record.py, describe-run, and the palette.

These exist because of a measured failure, and the tests are written against
it: five G1 imitation runs trained through the lab showed up in the policy
palette as five rows of `teach-g1_imitate-g1_front_kick-<hash>-sN`, and the
chain's own button pointed at the FINAL stage — which was the worst one
(3/8 seeds against 8/8 two stages earlier). So the load-bearing cases here
are "a record reaches the palette", "the pick is unique within a chain", and
"a trainer never writes a judgement about its own run".
"""

from __future__ import annotations

import json

import pytest

from microduck_local import run_record as R
from microduck_local import viz_server as V
from microduck_local.describe_run import main as describe_main


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """An isolated runs/ that describe-run and the palette both look at."""
    d = tmp_path / "runs"
    d.mkdir()
    monkeypatch.setattr(V, "RUNS_DIR", d)
    import microduck_local.describe_run as DR
    monkeypatch.setattr(DR, "RUNS_DIR", d)
    return d


@pytest.fixture
def fake_popen(monkeypatch, tmp_path):
    """Record every trainer launch instead of spawning one (the same trick as
    tests/test_lab.py's fixture of this name, kept local so this file does not
    import another test module)."""
    launches: list[dict] = []

    class FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd, self.kwargs, self.returncode = cmd, kwargs, None

        def poll(self):
            return self.returncode

    def popen(cmd, **kwargs):
        proc = FakeProc(cmd, **kwargs)
        launches.append(proc)
        return proc

    monkeypatch.setattr(V.subprocess, "Popen", popen)
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return launches


def _run(runs, name, **run_json):
    d = runs / name
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"not a real onnx, but the palette lists it")
    meta = {"run_name": name, "robot": "g1", "task": "imitate", "steps": 1_000_000,
            "env_kwargs": {"clip_name": "g1-front-kick"}, "init_from": None}
    meta.update(run_json)
    (d / "run.json").write_text(json.dumps(meta))
    return d


# ------------------------------------------------------------ the record

def test_a_run_with_no_record_reads_as_empty_not_an_error(runs):
    d = _run(runs, "plain")
    assert R.read_record(d) == {}


def test_an_unreadable_record_never_takes_the_run_down_with_it(runs):
    """A label must not be able to hide the policy it labels — the same
    reasoning as load_clips skipping an unparseable clip."""
    d = _run(runs, "broken")
    R.record_path(d).write_text("{ this is not json")
    assert R.read_record(d) == {}
    entry = _entry(V.discover_policies(), "broken")
    assert entry is not None and "title" not in entry


def test_write_record_merges_and_none_leaves_a_field_alone(runs):
    d = _run(runs, "merge")
    R.write_record(d, title="first", description="why")
    R.write_record(d, title=None, note="measured")
    got = R.read_record(d)
    assert got == {"title": "first", "description": "why", "note": "measured"}
    # An empty string is a deliberate clear, not "leave alone".
    R.write_record(d, note="")
    assert R.read_record(d)["note"] == ""


def test_write_record_leaves_no_temp_file_behind(runs):
    d = _run(runs, "atomic")
    R.write_record(d, title="t")
    assert [p.name for p in d.iterdir() if p.name.startswith("record.json.")] == []
    assert json.loads(R.record_path(d).read_text())["title"] == "t"


def test_a_second_measurement_by_the_same_tool_replaces_the_first(runs):
    """The honest current reading is the last one taken: a growing list is
    how a run ends up quoted at its best ever rather than its present."""
    d = _run(runs, "measured")
    R.write_measurement(d, "eval_imitate.py --seeds 8", {"apex_m": 0.10})
    R.write_measurement(d, "eval_imitate.py --seeds 8", {"apex_m": 0.62})
    R.write_measurement(d, "eval_imitate.py --seeds 8 --noise", {"apex_m": 0.65})
    measured = R.read_record(d)["measured"]
    assert measured["eval_imitate.py --seeds 8"] == {"apex_m": 0.62}
    assert len(measured) == 2


# ------------------------------------------------- what a trainer writes

def test_the_default_record_states_facts_and_never_a_verdict(runs):
    rec = R.default_record("teach-g1_imitate-x-s2", json.loads(
        (_run(runs, "facts", init_from="/abs/path/teach-g1_imitate-x-s1") / "run.json").read_text()),
        behavior_title="Perform “g1-front-kick”",
        stage_env={"MICRODUCK_G1_LIFT_MIN": "0.15", "MICRODUCK_CLIP": "g1-front-kick"})
    assert rec["title"] == "Perform “g1-front-kick”"     # not doubled
    assert rec["clip"] == "g1-front-kick"
    assert rec["init_from"] == "teach-g1_imitate-x-s1"   # basename, not an abs path
    # The clip is named on its own; it is not a curriculum rung.
    assert rec["stage_env"] == {"MICRODUCK_G1_LIFT_MIN": "0.15"}
    assert "MICRODUCK_G1_LIFT_MIN=0.15" in rec["description"]
    assert "warm-started from teach-g1_imitate-x-s1" in rec["description"]
    # A trainer does not get to say its own run was good.
    for judgement in ("note", "pick", "measured"):
        assert judgement not in rec


def test_a_title_that_does_not_name_the_clip_gains_it(runs):
    rec = R.default_record("r", json.loads((_run(runs, "clipname") / "run.json").read_text()))
    assert rec["title"] == "imitate (g1) · “g1-front-kick”"


def test_the_lab_hands_the_trainer_the_title_the_watcher_saw(fake_popen):
    """MICRODUCK_RUN_TITLE: the trainer writes the record and cannot know what
    the lab called the job, so the job passes its own display title down."""
    job = V.TrainingJob("g1_stand", steps=1000)
    env = fake_popen[0].kwargs["env"]
    assert env["MICRODUCK_RUN_TITLE"] == job.display_title()


# ------------------------------------------------------------ describe-run

def test_describe_run_writes_the_fields_and_refuses_an_empty_call(runs, capsys):
    _run(runs, "solo")
    assert describe_main(["solo", "--title", "T", "--note", "8/8 seeds"]) == 0
    got = R.read_record(runs / "solo")
    assert got["title"] == "T" and got["note"] == "8/8 seeds"
    capsys.readouterr()
    assert describe_main(["solo"]) == 2            # nothing to write
    assert describe_main(["nosuchrun", "--title", "T"]) == 2


def test_only_one_stage_of_a_chain_can_be_the_pick(runs, capsys):
    """Two picks is the same 'which one do I use?' question the record
    exists to answer — setting one clears the others."""
    for n in ("teach-k-abc-s1", "teach-k-abc-s2", "teach-k-abc-s3"):
        _run(runs, n)
    _run(runs, "teach-k-OTHER-s2")                 # a different chain
    describe_main(["teach-k-abc-s1", "--pick"])
    describe_main(["teach-k-OTHER-s2", "--pick"])
    describe_main(["teach-k-abc-s3", "--pick"])
    capsys.readouterr()
    picks = {n: bool(R.read_record(runs / n).get("pick"))
             for n in ("teach-k-abc-s1", "teach-k-abc-s2", "teach-k-abc-s3")}
    assert picks == {"teach-k-abc-s1": False, "teach-k-abc-s2": False,
                     "teach-k-abc-s3": True}
    # ...and a sibling chain is untouched.
    assert R.read_record(runs / "teach-k-OTHER-s2")["pick"] is True
    describe_main(["teach-k-abc-s3", "--no-pick"])
    assert R.read_record(runs / "teach-k-abc-s3")["pick"] is False


def test_backfill_fills_only_what_is_missing(runs, capsys):
    """Run over hundreds of old directories at once: a hand-written title is
    worth more than a derived one and must survive."""
    _run(runs, "old")
    describe_main(["old", "--title", "Mine", "--note", "measured"])
    capsys.readouterr()
    assert describe_main(["old", "--backfill"]) == 0
    got = R.read_record(runs / "old")
    assert got["title"] == "Mine"                  # not overwritten
    assert got["note"] == "measured"
    assert "imitate on the g1" in got["description"]   # filled, was missing


def test_backfill_reports_a_run_it_cannot_name(runs, capsys):
    d = runs / "norunjson"
    d.mkdir()
    assert describe_main(["norunjson", "--backfill"]) == 2
    assert "nothing to derive" in capsys.readouterr().err


# ------------------------------- the OTHER trainer's file (behavior.json)

def _trick(runs, name, **behavior_json):
    """A duck trick run: train_behavior writes behavior.json, not run.json."""
    d = runs / name
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"onnx")
    meta = {"behavior": "imitate", "steps": 3_000_000, "weights": {}}
    meta.update(behavior_json)
    (d / "behavior.json").write_text(json.dumps(meta))
    return d


def test_a_trick_runs_own_title_reaches_the_palette(runs):
    """behavior.json has carried title/description since the teach panel
    existed, but only the /train page read it — 647 trick runs sat in the
    palette as bare directory names with their titles already on disk."""
    _trick(runs, "teach-backflip-abc", title="Backflip — the pick",
           description="the one that lands")
    assert R.read_label(runs / "teach-backflip-abc")["title"] == "Backflip — the pick"
    assert _entry(V.discover_policies(), "teach-backflip-abc")["title"] == "Backflip — the pick"


def test_an_untitled_trick_run_is_named_by_its_recipe_and_clip(runs):
    _trick(runs, "teach-imitate-xyz", clip="sprint-cycle")
    assert R.read_label(runs / "teach-imitate-xyz")["title"] == "imitate · “sprint-cycle”"
    _trick(runs, "teach-spin-xyz", behavior="spin")
    assert R.read_label(runs / "teach-spin-xyz")["title"] == "spin"


def test_the_record_wins_over_behavior_json(runs):
    """record.json is this module's file and the one describe-run writes, so
    a hand-written title must not be shadowed by the trainer's."""
    d = _trick(runs, "teach-both-abc", title="from behavior.json")
    R.write_record(d, title="from record.json", note="8/8", pick=True)
    label = R.read_label(d)
    assert label == {"title": "from record.json", "note": "8/8", "pick": True}


def test_a_run_with_neither_file_is_simply_unnamed(runs):
    d = runs / "bare"
    d.mkdir()
    (d / "policy.onnx").write_bytes(b"onnx")
    assert R.read_label(d) == {}
    assert "title" not in _entry(V.discover_policies(), "bare")


def test_backfill_names_a_trick_run_from_its_behavior_json(runs, capsys):
    _trick(runs, "teach-crouch-abc", behavior="crouch")
    assert describe_main(["teach-crouch-abc", "--backfill"]) == 0
    capsys.readouterr()
    assert R.read_record(runs / "teach-crouch-abc")["title"] == "crouch"


def test_backfill_still_keeps_the_pick_unique_on_a_trick_chain(runs, capsys):
    """--backfill takes one path through main() whatever file it read from:
    its first draft returned early for trick runs and skipped this."""
    _trick(runs, "teach-flip-abc-s1")
    _trick(runs, "teach-flip-abc-s2")
    describe_main(["teach-flip-abc-s1", "--pick"])
    describe_main(["teach-flip-abc-s2", "--backfill", "--pick"])
    capsys.readouterr()
    assert R.read_record(runs / "teach-flip-abc-s1")["pick"] is False
    assert R.read_record(runs / "teach-flip-abc-s2")["pick"] is True


def test_backfill_says_so_when_there_is_nothing_to_derive_from(runs, capsys):
    d = runs / "nothing"
    d.mkdir()
    assert describe_main(["nothing", "--backfill"]) == 2
    assert "nothing to derive" in capsys.readouterr().err


# ------------------------------------------------------- the roster row

def test_assigning_a_described_run_puts_its_title_on_the_roster(runs, monkeypatch):
    """The roster's policy column is where the user looked first and saw a
    list of hashes; a described run says its name there too, while the event
    line keeps the identifier so the log stays greppable."""
    import asyncio

    R.write_record(_run(runs, "teach-k-xyz-s2"), title="Front kick (G1)")
    monkeypatch.setenv("LAB_STATE_PATH", str(runs.parent / "lab-state.json"))
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    monkeypatch.setattr(V, "policy_robot", lambda p: "microduck")
    monkeypatch.setattr(V, "env_kwargs_for_policy_path", lambda p: {})
    app = V.make_app([])
    duck = V.Duck("d0", "before", V._zero_infer, seed=1, env_kwargs={})
    st = app.state.lab
    st.ducks[:] = [duck]
    asyncio.run(app.state.do_assign("d0", "run:teach-k-xyz-s2"))
    assert duck.label == "Front kick (G1)"
    assert duck.policy_id == "run:teach-k-xyz-s2"
    assert any("teach-k-xyz-s2" in e for e in st.events), st.events


def test_an_undescribed_run_still_shows_its_name_on_the_roster(runs, monkeypatch):
    import asyncio

    _run(runs, "teach-k-plain-s1")
    monkeypatch.setenv("LAB_STATE_PATH", str(runs.parent / "lab-state.json"))
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    monkeypatch.setattr(V, "policy_robot", lambda p: "microduck")
    monkeypatch.setattr(V, "env_kwargs_for_policy_path", lambda p: {})
    app = V.make_app([])
    duck = V.Duck("d0", "before", V._zero_infer, seed=1, env_kwargs={})
    app.state.lab.ducks[:] = [duck]
    asyncio.run(app.state.do_assign("d0", "run:teach-k-plain-s1"))
    assert duck.label == "teach-k-plain-s1"


# --------------------------------------------------------- the palette

def _entry(policies, run_name):
    return next((p for p in policies if p.get("id") == f"run:{run_name}"), None)


def test_the_palette_carries_the_title_note_and_pick(runs):
    d = _run(runs, "teach-k-xyz-s2")
    R.write_record(d, title="Front kick (G1)", note="8/8 seeds hold 20 s", pick=True)
    entry = _entry(V.discover_policies(), "teach-k-xyz-s2")
    assert entry["title"] == "Front kick (G1)"
    assert entry["note"] == "8/8 seeds hold 20 s"
    assert entry["pick"] is True
    # The LABEL stays the run name whatever the title says: it is the
    # identifier --init-from, the docs and the delete endpoint address.
    assert entry["label"] == "teach-k-xyz-s2"
    assert entry["chain"] == "teach-k-xyz" and entry["stage"] == 2


def test_a_described_run_without_a_pick_says_nothing_about_one(runs):
    d = _run(runs, "nopick")
    R.write_record(d, title="T")
    entry = _entry(V.discover_policies(), "nopick")
    assert entry["title"] == "T"
    assert "pick" not in entry and "note" not in entry


def test_a_cleared_pick_is_not_served_as_one(runs):
    d = _run(runs, "cleared")
    R.write_record(d, pick=True)
    R.write_record(d, pick=False)
    assert "pick" not in _entry(V.discover_policies(), "cleared")
