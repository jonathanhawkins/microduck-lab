"""Robot selection in the lab: two bodies on one roster.

The lab was one robot deep — a `Duck` built a `MicroduckWalkEnv` and the
palette handed out policies with no notion of which body they drive. A 99-obs
G1 brain dropped on a 61-obs duck loads happily and emits nonsense, so these
tests pin the plumbing that makes that impossible: policies carry their
robot, roster slots carry theirs, an assignment moves the slot to the
policy's body, and a width mismatch is refused rather than stepped.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from test_lab import fake_popen  # noqa: F401 — fixture: fakes the trainer

from microduck_local import viz_server as V
from microduck_local.robots.g1 import g1_ready
from microduck_local.robots.mars import mars_ready

needs_g1 = pytest.mark.skipif(not g1_ready(), reason="G1 assets missing — uv run fetch-g1")


def _endpoint(app, path: str, method: str):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
            return r.endpoint
    raise AssertionError(f"no {method} {path} route")


def _run(tmp_path, name: str, robot: str | None, obs_dim: int = 61):
    """A minimal run dir the palette will discover."""
    d = tmp_path / "runs" / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "policy.onnx").write_bytes(b"not really onnx")
    meta = {"run_name": name, "envs": 1, "steps": 1, "seed": 0}
    if robot is not None:
        meta["robot"] = robot
    (d / "run.json").write_text(json.dumps(meta))
    return d


# ------------------------------------------------------------- provenance

def test_policy_robot_reads_the_runs_own_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    g1_run = _run(tmp_path, "g1-walk", "g1")
    duck_run = _run(tmp_path, "duck-walk", "microduck")
    assert V.policy_robot(g1_run) == "g1"
    assert V.policy_robot(duck_run) == "microduck"
    assert V.policy_robot(g1_run / "policy.onnx") == "g1"


def test_a_run_without_metadata_is_a_duck(tmp_path, monkeypatch):
    """Every run on disk today predates the field — they are all ducks."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    old = _run(tmp_path, "legacy", None)
    (old / "run.json").unlink()
    assert V.policy_robot(old) == "microduck"


def test_the_palette_tags_every_entry_with_its_robot(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(V, "POLICIES_DIR", tmp_path / "policies")
    _run(tmp_path, "g1-walk", "g1")
    _run(tmp_path, "duck-walk", "microduck")
    by_id = {p["id"]: p for p in V.discover_policies()}
    assert by_id["run:g1-walk"]["robot"] == "g1"
    assert by_id["run:duck-walk"]["robot"] == "microduck"
    assert all("robot" in p for p in by_id.values())


# ------------------------------------------------------------ roster slots

def test_lab_state_remembers_which_body_a_slot_is(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    duck = types.SimpleNamespace(id="d0", label="x", policy_id=None,
                                 onnx_path="/p/walker.onnx", robot="g1",
                                 showcase=False)
    V.save_lab_state([duck])
    data = json.loads((tmp_path / "lab-state.json").read_text())
    assert data["ducks"][0]["robot"] == "g1"


@needs_g1
def test_a_slot_builds_the_env_of_its_own_robot():
    from microduck_local.robots.g1_env import G1WalkEnv
    from microduck_local.walk_env import MicroduckWalkEnv

    duck = V.Duck("d0", "duck", lambda o: np.zeros(14, np.float32), seed=0)
    assert isinstance(duck.env, MicroduckWalkEnv)
    assert duck.env.observation_space.shape == (61,)

    g1d = V.Duck("d1", "g1", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    assert isinstance(g1d.env, G1WalkEnv)
    assert g1d.env.observation_space.shape == (99,)


@needs_g1
def test_set_robot_forces_the_rebuild_a_matching_kwargs_would_skip():
    """`rebuild_env` returns early when the kwargs are unchanged — and a body
    swap usually carries the SAME kwargs, so the slot kept the old env."""
    duck = V.Duck("d0", "duck", lambda o: np.zeros(14, np.float32), seed=0)
    kwargs = dict(duck.env_kwargs)
    duck.set_robot("g1")
    duck.rebuild_env(kwargs)
    assert duck.robot == "g1"
    assert duck.env.observation_space.shape == (99,)
    assert "__robot_swap__" not in duck.env_kwargs


@needs_g1
def test_driving_a_g1_does_not_touch_the_ducks_command_slots():
    """`set_cmd` zeroed head_cmd/body_cmd on every step; the G1's obs has
    no such slots and its env does not carry the arrays."""
    g1d = V.Duck("d0", "g1", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    g1d.set_cmd(np.array([0.3, 0.0, 0.0], np.float32))
    assert float(g1d.env.twist_cmd[0]) == pytest.approx(0.3)


# ----------------------------------------------------------------- guards

def test_onnx_infer_reports_the_obs_width_it_expects(tmp_path):
    """The assign guard reads this; without it a mismatch is only visible as
    garbage motion."""
    torch = pytest.importorskip("torch")
    net = torch.nn.Linear(99, 29)
    p = tmp_path / "g1.onnx"
    torch.onnx.export(net, (torch.zeros(1, 99),), str(p),
                      input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False)
    infer = V._onnx_infer(p)
    assert infer.obs_dim == 99
    assert infer(np.zeros(99, np.float32)).shape == (29,)


def test_scene_endpoint_serves_the_robot_you_ask_for(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    get_scene = _endpoint(app, "/scene", "GET")
    duck = get_scene()
    assert duck["bodies"] and duck["geoms"]
    with pytest.raises(HTTPException) as e:
        get_scene(robot="wombat")
    assert e.value.status_code == 404


@needs_g1
def test_scene_endpoint_serves_the_g1_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    get_scene = _endpoint(app, "/scene", "GET")
    sc = get_scene(robot="g1")
    assert sc["bodies"][1] == "pelvis"
    assert sc["meshes"] and sc["vertScale"] == 0.001
    assert sc != get_scene()


@needs_g1
def test_a_duck_trick_is_never_matched_for_a_g1_trainee(tmp_path, monkeypatch):
    """Duck recipes name duck joints, duck feet and the 61-obs command slots.
    Asking a G1 trainee for one must not reach `train_behavior` — the registry
    is filtered by robot, so the request lands on a G1 task or on nothing."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    trainee = V.Duck("trainee", "g1", lambda o: np.zeros(29, np.float32),
                     seed=0, robot="g1")
    app = V.make_app([trainee])
    teach = _endpoint(app, "/teach", "POST")
    out = asyncio.run(teach(V.TeachReq(text="stand on one leg")))
    if out["matched"]:
        # whatever it matched, it is a task for THIS body
        assert out["job"]["behavior"]["id"].startswith("g1_")
    else:
        assert "one_leg" not in json.dumps(out)


def test_teach_still_works_for_a_duck_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    out = asyncio.run(teach(V.TeachReq(text="a trick that does not exist at all")))
    assert out["matched"] is False
    assert "behaviors" in out          # the "I can teach these" list, not a robot refusal


# --------------------------------------------------- the loop must not die

@needs_g1
def test_spawning_a_g1_policy_spawns_a_g1(tmp_path, monkeypatch):
    """The bug this exists for: `do_spawn_duck` built a DUCK env for a G1
    policy, the ONNX session raised inside the 50 Hz loop, and the loop died
    — the lab then streamed nothing at all and the viewer went blank with no
    error anywhere. The roster slot has to follow the policy's body."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    run = _run(tmp_path, "g1-run", "g1")
    entry = {"id": "run:g1-run", "path": str(run / "policy.onnx"), "robot": "g1"}
    monkeypatch.setattr(V, "discover_policies", lambda: [entry])

    def fake_infer(policy_id):
        f = lambda obs: np.zeros(29, np.float32)      # noqa: E731
        f.obs_dim = 99
        return f

    monkeypatch.setattr(V, "load_policy_infer", fake_infer)
    app = V.make_app([])
    st = app.state.lab
    spawn = app.state.do_spawn_duck
    asyncio.run(spawn("run:g1-run"))
    assert [d.robot for d in st.ducks] == ["g1"]
    d = st.ducks[0]
    assert d.env.observation_space.shape == (99,)
    d.tick()                                          # the loop's own call


def test_a_policy_that_does_not_fit_the_body_is_refused_not_stepped(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    run = _run(tmp_path, "duck-run", "microduck")
    entry = {"id": "run:duck-run", "path": str(run / "policy.onnx"), "robot": "microduck"}
    monkeypatch.setattr(V, "discover_policies", lambda: [entry])

    def fake_infer(policy_id):
        f = lambda obs: np.zeros(29, np.float32)      # noqa: E731
        f.obs_dim = 99                                # a G1 brain mislabelled
        return f

    monkeypatch.setattr(V, "load_policy_infer", fake_infer)
    app = V.make_app([])
    st = app.state.lab
    asyncio.run(app.state.do_spawn_duck("run:duck-run"))
    assert st.ducks == []
    assert any("wants 99 obs" in e for e in st.events)


def test_one_raising_duck_does_not_stop_the_lab(monkeypatch):
    """A duck whose brain raises is parked; the loop keeps streaming."""
    def boom(obs):
        raise RuntimeError("Got: 61 Expected: 99")

    good = V.Duck("d0", "ok", lambda o: np.zeros(14, np.float32), seed=0)
    bad = V.Duck("d1", "bad", boom, seed=1)
    with pytest.raises(RuntimeError):
        bad.tick()
    # …and the loop's guard is what keeps `good` running: the lab removes the
    # raiser rather than letting the exception leave the frame callback.
    good.tick()
    assert good.env.step_count > 0


# ------------------------------------------- the G1's own shipped policies

@needs_g1
def test_the_palette_has_a_shipped_g1_group():
    from microduck_local.robots import g1

    entries = [p for p in V.discover_policies() if p["group"] == "g1"]
    assert entries, "the fetched G1 drop should give the palette a group"
    names = {e["label"] for e in entries}
    assert "walker" in names
    for e in entries:
        assert e["robot"] == "g1"
        assert e["id"] == f"g1:{e['label']}"
        assert Path(e["path"]).is_file()
    # every listed chip is one the lab can actually step
    usable = {e["name"] for e in g1.shipped_policies() if e["usable"]}
    assert names == usable


@needs_g1
def test_policies_the_env_cannot_step_stay_out_of_the_palette():
    """The drop also carries a 101-obs croucher and a 36-obs ARM overlay.
    A chip that can never be assigned is worse than no chip — and the filter
    reads each graph, so a re-export that changes a width changes the list."""
    from microduck_local.robots import g1

    by_name = {e["name"]: e for e in g1.shipped_policies()}
    assert by_name["croucher"]["obs"] == 101 and not by_name["croucher"]["usable"]
    assert by_name["right_reacher"]["actions"] == 7
    assert not by_name["right_reacher"]["usable"]
    listed = {p["label"] for p in V.discover_policies() if p["group"] == "g1"}
    assert "croucher" not in listed and "right_reacher" not in listed


@needs_g1
def test_a_shipped_g1_chip_loads_and_drives_a_g1_slot():
    infer = V.load_policy_infer("g1:walker")
    assert infer.obs_dim == 99
    duck = V.Duck("d0", "walker", infer, seed=0, robot="g1")
    duck.set_cmd(np.array([0.6, 0.0, 0.0], np.float32))
    for _ in range(30):
        duck.obs, *_ = duck.env.step(duck.infer(duck.obs))
    assert float(duck.env._trunk_xpos[2]) > 0.5, "it should still be standing"


@needs_g1
def test_every_shipped_entry_carries_what_it_does():
    """The chip's tooltip is the only place a user learns that the rotator
    falls under a forward command — it is measured, so it must ship."""
    entries = {p["label"]: p for p in V.discover_policies() if p["group"] == "g1"}
    # the walker's note has to answer the two questions people actually ask:
    # how fast does it go, and what happens when you ask for nothing
    assert "dead zone" in entries["walker"]["note"]
    assert "IDLE" in entries["walker"]["note"]
    assert "FALLS" in entries["rotator"]["note"]


# ------------------------------------------------ teaching another body

@needs_g1
def test_teach_trains_the_g1_when_the_roster_is_a_g1(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """The whole point of the panel: ask the roster in front of you to learn
    something and WATCH it. A G1 roster gets G1 tasks, launched through the
    lab (not the CLI) so the job card, the progress curve and the 🎓 trainee
    all work exactly as they do for a duck trick."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    teach = _endpoint(app, "/teach", "POST")

    out = asyncio.run(teach(V.TeachReq(text="stand still")))
    assert out["matched"] is True, out.get("message")
    assert st.job is not None and st.job.behavior.id == "g1_stand"

    argv = fake_popen[-1].cmd
    assert "microduck_local.train" in argv        # train-walk, not train_behavior
    assert argv[argv.index("--robot") + 1] == "g1"
    assert argv[argv.index("--task") + 1] == "stand"
    # …and the trainer snapshots by default, which is what the 🎓 trainee
    # loads while the run is still going (train.py --snap-steps).
    from microduck_local.train import parse_args
    assert parse_args([]).snap_steps > 0

    trainee = st.trainee()
    assert trainee is not None and trainee.robot == "g1"
    assert trainee.env.observation_space.shape == (99,)
    assert trainee.env.action_space.shape == (29,)
    # the untrained trainee steps its own body until the first snapshot
    trainee.obs, *_ = trainee.env.step(trainee.infer(trainee.obs))


@needs_g1
def test_a_g1_roster_is_not_offered_duck_tricks(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    teach = _endpoint(app, "/teach", "POST")
    out = asyncio.run(teach(V.TeachReq(text="a task that does not exist at all")))
    assert out["matched"] is False
    ids = {c["id"] for c in out["behaviors"]}
    # The INVARIANT is "no duck recipes", not a frozen G1 menu — pinning the
    # exact set made adding g1_squat turn this red for no defect.
    assert "g1_stand" in ids, "the G1 menu must not be empty"
    B = V.behaviors_mod
    duck_ids = {b.id for b in B.for_robot("microduck")}
    assert not (ids & duck_ids), "a G1 roster must not be offered duck recipes"
    assert ids == {b.id for b in B.for_robot("g1")}
    assert "g1 task" in out["message"]


def test_a_duck_roster_still_gets_duck_tricks(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    out = asyncio.run(teach(V.TeachReq(text="a trick that does not exist at all")))
    assert out["matched"] is False
    ids = {c["id"] for c in out["behaviors"]}
    assert "g1_stand" not in ids and "one_leg" in ids


# ------------------------------------------------- helpers follow the body

def _g1_infer():
    """A stand-in for the run's live.onnx: 99 obs in, 29 actions out."""
    f = lambda obs: np.zeros(29, np.float32)          # noqa: E731
    f.obs_dim = 99
    return f


def _start_g1_job(app, tmp_path, text="stand still"):
    """Teach a G1 task and take it past the first-snapshot guard."""
    st = app.state.lab
    out = asyncio.run(_endpoint(app, "/teach", "POST")(V.TeachReq(text=text)))
    assert out["matched"] is True, out.get("message")
    assert st.job.behavior.robot == "g1"
    (st.job.dir / "model.zip").write_bytes(b"")   # spawn_helper_error's gate
    assert V.spawn_helper_error(st) is None
    return st.job


@needs_g1
def test_the_plus_button_gives_a_g1_run_a_g1_helper(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """The bug: `do_spawn_helper` built the clone with the DEFAULT body while
    the job's env kwargs carry the G1's `task=`, so `MicroduckWalkEnv` got an
    unexpected keyword and raised — inside an `asyncio.create_task` whose
    try/finally has no except. The + button did nothing and said nothing.
    A helper is a clone of the trainee, so it clones its body too."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    job = _start_g1_job(app, tmp_path)
    monkeypatch.setattr(V, "_onnx_infer", lambda p: _g1_infer())

    asyncio.run(app.state.do_spawn_helper())

    helpers = V.helper_ducks(st.ducks)
    assert len(helpers) == 1, f"the + button did nothing: {list(st.events)}"
    h = helpers[0]
    assert h.robot == "g1"
    assert h.env.observation_space.shape == (99,)
    assert h.env.action_space.shape == (29,)
    assert h.onnx_path == str(job.dir / "live.onnx")
    assert job.helpers == 1
    h.tick()          # the 50 Hz loop's own call, on the real env


@needs_g1
def test_a_helper_idling_before_the_first_snapshot_is_the_right_width(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """live.onnx can lose a race with the guard; the helper then idles on a
    do-nothing brain until the next snapshot. `_zero_infer` emits 14 floats
    for every body, which a 29-joint env rejects outright."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    _start_g1_job(app, tmp_path)

    def no_onnx(p):
        raise FileNotFoundError(p)

    monkeypatch.setattr(V, "_onnx_infer", no_onnx)
    asyncio.run(app.state.do_spawn_helper())

    helpers = V.helper_ducks(st.ducks)
    assert len(helpers) == 1, f"the + button did nothing: {list(st.events)}"
    assert helpers[0].onnx_path is None
    helpers[0].tick()


@needs_g1
def test_a_helper_that_cannot_be_built_says_why(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """Whatever goes wrong, the + button must not be a no-op with no message:
    the raise used to die inside the task and the UI showed nothing at all."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    _start_g1_job(app, tmp_path)
    monkeypatch.setattr(V, "_onnx_infer", lambda p: _g1_infer())

    def boom(*a, **k):
        raise RuntimeError("no MJCF for you")

    monkeypatch.setattr(V, "Duck", boom)
    st.events.clear()
    asyncio.run(app.state.do_spawn_helper())

    assert V.helper_ducks(st.ducks) == []
    assert any("helper" in e and "no MJCF for you" in e for e in st.events), list(st.events)
    assert st.scaling is False          # the guard flag is released either way


@needs_g1
def test_a_brain_that_does_not_fit_the_helper_is_refused_not_stepped(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """Same guard `do_assign`/`do_spawn_duck` carry: a width mismatch is a
    message, not an ONNX raise inside the 50 Hz loop (which stops the loop
    for every duck, not just the bad one)."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    _start_g1_job(app, tmp_path)

    def duck_infer(p):
        f = lambda obs: np.zeros(14, np.float32)      # noqa: E731
        f.obs_dim = 61                                # a duck brain on a G1 run
        return f

    monkeypatch.setattr(V, "_onnx_infer", duck_infer)
    st.events.clear()
    asyncio.run(app.state.do_spawn_helper())

    assert V.helper_ducks(st.ducks) == []
    assert any("61 obs" in e for e in st.events), list(st.events)


@needs_g1
def test_a_new_job_moves_the_helpers_it_keeps_onto_its_own_body(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """`/teach` re-mirrored helpers onto the new run only when the job was a
    duck's, so a helper left over from a duck run stayed a 61-obs duck
    stepping the PREVIOUS run's brain while a G1 job trained beside it — and
    `apply_snapshot` then hands it that job's 99-obs live.onnx. Helpers are
    clones of the trainee: they follow it to the new body, as it does."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    trainee = V.Duck("trainee", "🎓 old", V._zero_infer_for("g1"), seed=97, robot="g1")
    helper = V.Duck("helper1", "🤝 helper 1", V._zero_infer, seed=101,
                    onnx_path="/runs/previous/live.onnx")
    app = V.make_app([trainee, helper])
    st = app.state.lab
    job = _start_g1_job(app, tmp_path)

    h = V.helper_ducks(st.ducks)[0]
    assert h.robot == "g1", "the helper stayed a duck beside a G1 trainee"
    assert h.env.observation_space.shape == (99,)
    assert h.onnx_path == str(job.dir / "live.onnx")
    assert h.label.startswith("🤝"), "a helper keeps its own identity"
    h.tick()


@needs_g1
def test_a_snapshot_is_never_pushed_into_a_body_it_does_not_fit(fake_popen, monkeypatch, tmp_path):  # noqa: F811 — pytest fixture, not a redefinition
    """`apply_snapshot` swapped the run's live.onnx onto every id starting
    with "helper" with no width check at all. A 99-obs brain in a 61-obs duck
    raises inside the 50 Hz loop, which is FATAL — the lab then streams
    nothing and the viewer goes blank with no error anywhere."""
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    g1d = V.Duck("d0", "walker", lambda o: np.zeros(29, np.float32), seed=0, robot="g1")
    app = V.make_app([g1d])
    st = app.state.lab
    _start_g1_job(app, tmp_path)
    # A duck helper that the job did not build (a restored roster, a body the
    # run moved off): the snapshot must skip it rather than arm the loop.
    stray = V.Duck("helper9", "🤝 helper 9", V._zero_infer, seed=109)
    st.ducks.append(stray)
    monkeypatch.setattr(V, "_onnx_infer", lambda p: _g1_infer())
    st.events.clear()

    asyncio.run(app.state.apply_snapshot())

    assert stray.infer is V._zero_infer, "a 99-obs brain landed in a 61-obs duck"
    assert any("helper9" in e for e in st.events), list(st.events)
    stray.tick()
    assert st.trainee().infer.obs_dim == 99   # the trainee still got it


# ------------------------------------------- stage layout: one pitch a body
#
# The lab lays its roster out in a square grid. That grid used to be drawn by
# the VIEWER on one constant, 0.65 m, which is the duck's: six G1 helpers plus
# the trainee — a 1.3 m humanoid is 0.53 m wide — landed 0.65 m apart and stood
# inside each other, one clump you could not read. The pitch is now a property
# of the ROBOT (`RobotSpec.lab_spacing_m`, per-slot), and the layout lives in
# `viz_server.lab_slot_offsets` so there is one definition of it.

# The duck's slot positions as the viewer drew them BEFORE the layout moved:
# taken from duck-viewer/components/Viewer.tsx `gridOffsets(n)` at its shipped
# spacing of 0.65 m, run in node — not re-derived here. The fix is for the G1;
# the duck is not allowed to move.
DUCK_SLOTS_BEFORE = {
    1: [(0.0, 0.0)],
    2: [(-0.325, 0.0), (0.325, 0.0)],
    3: [(-0.325, 0.0), (0.325, 0.0), (-0.325, 0.65)],
    4: [(-0.325, 0.0), (0.325, 0.0), (-0.325, 0.65), (0.325, 0.65)],
    5: [(-0.65, 0.0), (0.0, 0.0), (0.65, 0.0), (-0.65, 0.65), (0.0, 0.65)],
    6: [(-0.65, 0.0), (0.0, 0.0), (0.65, 0.0),
        (-0.65, 0.65), (0.0, 0.65), (0.65, 0.65)],
    7: [(-0.65, 0.0), (0.0, 0.0), (0.65, 0.0),
        (-0.65, 0.65), (0.0, 0.65), (0.65, 0.65), (-0.65, 1.3)],
    8: [(-0.65, 0.0), (0.0, 0.0), (0.65, 0.0),
        (-0.65, 0.65), (0.0, 0.65), (0.65, 0.65), (-0.65, 1.3), (0.0, 1.3)],
    9: [(-0.65, 0.0), (0.0, 0.0), (0.65, 0.0),
        (-0.65, 0.65), (0.0, 0.65), (0.65, 0.65),
        (-0.65, 1.3), (0.0, 1.3), (0.65, 1.3)],
}


def _body_width_m(robot: str) -> float:
    """Widest HORIZONTAL extent of the whole body at its STAND keyframe, in m.

    Measured off the compiled model — the geom AABBs in world axes — so a
    model revision that changes a body's size shows up here instead of quietly
    crowding the stage. The AABB loop is
    `robots/body.widest_horizontal_extent_m`: it was written out here and
    again in `tests/test_body_conformance.py`, and `MjcfBody` needs it a third
    time to measure a body nobody typed a pitch for, so there is one copy.
    What is local to this case is the ENV's model — the walking env's own
    compiled scene, rather than `scene_fn()`'s.
    """
    import mujoco

    from microduck_local.robots import spec as S
    from microduck_local.robots.body import widest_horizontal_extent_m
    from microduck_local.train import env_class

    sp = S.get(robot)
    kw = dict(obs_noise=False, domain_rand=False, action_delay=False,
              random_yaw=False, seed=0)
    if robot != "microduck":
        kw["actuator_force"] = "xml"
    env = env_class(robot, "walk")(**kw)
    env.reset(seed=0)
    m, d = env.model, env.data
    mujoco.mj_resetDataKeyframe(
        m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, sp.stand_keyframe))
    mujoco.mj_forward(m, d)
    return widest_horizontal_extent_m(m, d)


def _min_gap(slots) -> float:
    return min(math.dist(a, b)
               for i, a in enumerate(slots) for b in slots[i + 1:])


def test_the_ducks_slots_are_exactly_where_they_have_always_been():
    """Requirement (b) of the per-robot-spacing change: bit-identical.

    `lab_slot_offsets` generalises the viewer's expression rather than
    replacing it, so for a roster of one robot it must reproduce it to the
    float — including the centring, which is on the columns the FIRST row
    occupies and not on the grid.
    """
    for n, want in DUCK_SLOTS_BEFORE.items():
        got = V.lab_slot_offsets(["microduck"] * n)
        assert got == want, f"the duck's layout moved at n={n}: {got}"
    # A row that predates the field (lab-state restores carry no robot) is a
    # duck, and an id this process cannot resolve falls back to the duck's
    # pitch rather than stopping the 50 Hz loop over a rendering number.
    assert V.lab_slot_offsets([None] * 4) == DUCK_SLOTS_BEFORE[4]
    assert V.lab_spacing_m("nosuchbot") == 0.65


@pytest.mark.parametrize("n", [2, 3, 4, 6, 7])
def test_two_g1_slots_leave_a_whole_g1_of_air_between_them(n):
    """Requirement (a). n=7 is the reported roster: six helpers + the trainee.

    The threshold is a body width of CLEAR AIR (centres >= 2x the width), not
    "centres >= the width" — a G1 is 0.53 m wide, so the duck's 0.65 m pitch
    passes that weaker form while visibly interpenetrating, and a test that
    the bug passes is not a test. The duck gets 0.4655 m of air on 0.1845 m of
    body (2.52x); this asks the G1 for 1.0 of its own.
    """
    from microduck_local.robots.g1 import BODY_WIDTH_M

    gap = _min_gap(V.lab_slot_offsets(["g1"] * n))
    assert gap >= 2 * BODY_WIDTH_M, (
        f"{n} G1s are {gap:.3f} m apart, closer than two body widths "
        f"({2 * BODY_WIDTH_M:.3f} m) — they interpenetrate on the stage")


def test_a_mixed_roster_pitches_each_slot_by_its_own_robot():
    """A duck beside a G1: the gap is the duck's half-pitch plus the G1's, so
    each body contributes its own clearance and neither is spaced as if it
    were the other. (A column that CONTAINS a G1 is G1-pitched throughout —
    a rectangular grid cannot shear — which is what keeps the row above from
    landing on its shoulders.)"""
    from microduck_local.robots import spec as S

    duck = S.get("microduck").lab_spacing_m
    g1 = S.get("g1").lab_spacing_m
    assert g1 > duck

    a, b = V.lab_slot_offsets(["microduck", "g1"])
    assert b[0] - a[0] == duck / 2 + g1 / 2
    assert a[1] == b[1] == 0.0

    # And nothing overlaps: every pair clears both bounding circles plus the
    # smaller body's width of air.
    widths = {"microduck": 0.1845, "g1": 0.5338}   # _body_width_m, above
    roster = ["microduck", "g1", "microduck", "g1", "microduck"]
    slots = V.lab_slot_offsets(roster)
    for i, si in enumerate(slots):
        for j in range(i + 1, len(slots)):
            wi, wj = widths[roster[i]], widths[roster[j]]
            need = (wi + wj) / 2 + min(wi, wj)
            assert math.dist(si, slots[j]) >= need, (
                f"{roster[i]} {i} and {roster[j]} {j} overlap")


def test_the_lab_frame_ships_each_slot_its_own_offset():
    """The layout is only worth having if it reaches the viewer: the per-duck
    row carries it, and the viewer draws that instead of its own grid."""
    ducks = [V.Duck(f"d{i}", "w", V._zero_infer, seed=i, env_kwargs={})
             for i in range(3)]
    app = V.make_app(ducks)
    st = app.state.lab
    frames: list[str] = []

    class Sink:
        async def send_text(self, frame):
            frames.append(frame)

    st.clients.add(Sink())

    async def run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.2)

    asyncio.run(run())
    assert frames, "lab_loop never broadcast"
    rows = json.loads(frames[-1])["ducks"]
    want = V.lab_slot_offsets(["microduck"] * 3)
    assert [tuple(r["offset"]) for r in rows] == want


def test_the_duck_pitch_is_3_5x_the_body_it_was_drawn_for():
    """The constant the whole scaling rests on, measured rather than assumed.

    0.65 m was chosen for a body this wide; `lab_spacing_m` on any other robot
    is that same ratio applied to ITS width. If the duck's MJCF changes shape,
    this is the test that says the ratio moved — see the companion G1 check.
    """
    w = _body_width_m("microduck")
    assert w == pytest.approx(0.1845, abs=0.002), f"the duck is {w:.4f} m wide"
    assert 0.65 / w == pytest.approx(3.52, abs=0.05)


@needs_g1
def test_the_g1s_spacing_is_the_ducks_ratio_on_the_g1s_measured_width():
    """`BODY_WIDTH_M` / `LAB_SPACING_M` are numbers written into g1.py; this
    re-measures the first from the fetched MJCF and re-derives the second, so
    neither can drift from the model it describes."""
    from microduck_local.robots.g1 import BODY_WIDTH_M, LAB_SPACING_M

    w = _body_width_m("g1")
    assert w == pytest.approx(BODY_WIDTH_M, abs=0.002), (
        f"the G1 is {w:.4f} m wide, g1.py says {BODY_WIDTH_M}")
    duck_ratio = 0.65 / _body_width_m("microduck")
    assert LAB_SPACING_M == pytest.approx(duck_ratio * w, abs=0.02)


# --- "is the trainer finished?" must have an authoritative answer ---------

def test_teach_status_reports_whether_a_job_is_running(fake_popen, monkeypatch, tmp_path):  # noqa: F811
    """The lab owns the trainer subprocess, so it is the only thing that knows.

    Without this endpoint the only way to ask from outside was to grep the
    process table, which fails silently in two directions measured on this
    machine: `pgrep -f "microduck_local.train "` matches the very shell
    running the pgrep, and `pgrep -f "Python.*microduck_local"` matches the
    LAB SERVER, whose venv path contains the string. Both answer "a trainer is
    running" forever, and a wait loop built on either never terminates.
    """
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    status = _endpoint(app, "/teach/status", "GET")

    idle = status()
    assert idle["running"] is False and idle["job"] is None

    teach = _endpoint(app, "/teach", "POST")
    out = asyncio.run(teach(V.TeachReq(text="stand still")))
    assert out["matched"] is True

    live = status()
    assert live["running"] is True
    assert live["status"] == "training"
    assert live["job"]["runName"].startswith("teach-")
    assert live["job"]["total"]

    asyncio.run(_endpoint(app, "/teach/stop", "POST")())
    assert status()["running"] is False


def test_a_finished_train_run_leaves_a_done_marker(tmp_path):
    """`train.py` must close progress.jsonl with a terminal record, the way
    train_behavior always has — so "has it finished?" is answerable from the
    artifact alone, with no server and no process table."""
    import inspect

    from microduck_local import train as T

    src = inspect.getsource(T.main)
    assert '"done": True' in src, "train.py never writes a completion record"
    # and it must come AFTER the model is saved, or a reader can see `done`
    # before the artifact it promises exists
    assert src.index('model.save') < src.index('"done": True')


# ======================================================================
# A THIRD BODY, AND A CATALOGUE OF THEM (docs/mars-roadmap.md Phase 1b/2b)
#
# Everything above this line was written with two walkers on the roster, and
# every one of those cases is an `if robot == "g1"` that was replaced by a
# registry lookup. These are the cases that say the replacement is generic:
# MARS is a WHEELED body with an arm env, no drive channel and no shipped
# policy, and a Menagerie model is level 0 — an MJCF and an id, no env at all.
# ======================================================================

from microduck_local.lab import robots as LR  # noqa: E402 — beside its tests
from microduck_local.robots import registry as REG  # noqa: E402

GO2 = "menagerie:unitree_go2"

# REGISTERED is not READY. `mars` is a registry BUILTIN, so `"mars" in
# registry()` is true on a fresh clone that has never downloaded a byte, and
# this guard skipped nothing: CI went red on five cases with
# `FileNotFoundError: Innate MARS assets not in .../innate_mars` (2026-09-22).
# Ask the body whether its assets are on disk, the way every other MARS test
# file does. Verified against the complement — `MICRODUCK_MARS_DIR` pointed at
# an empty directory must SKIP these, not fail them.
needs_mars = pytest.mark.skipif(
    not mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")
needs_go2 = pytest.mark.skipif(
    GO2 not in REG.registry(),
    reason=f"Go2 missing — uv run fetch-robot {GO2}")


@needs_mars
def test_a_mars_slot_builds_the_arm_env_and_idles():
    """The slot's env is the BODY's, for the body's own default task.

    `Duck._make_env` asked for `task="walk"` when nobody said otherwise, and
    "walk" is not a question MARS can be asked — `MarsBody.env_class("walk")`
    rightly raises rather than handing an arm a walker's env. The body
    answers `default_task` now, and for MARS that is `reach`.

    It IDLES on a zero action, and that is a property of the action map
    rather than of zero: MARS's default map is `delta`, where a = 0 is exact
    (docs/mars-roadmap.md 4a-2 — an ABSOLUTE map has no fixed point, which is
    why it could reach a target and never sit still on it). So the arm holds
    where HOME put it.
    """
    from microduck_local.robots.mars_env import MarsArmEnv

    d = V.Duck("d0", "MARS", V._zero_infer_for("mars"), seed=1, robot="mars")
    assert isinstance(d.env, MarsArmEnv)
    assert d.env.task == "reach"
    assert d.env.observation_space.shape == (32,)
    start = np.array(d.env.data.qpos[d.env._arm_qadr], float)
    for _ in range(25):                         # one second of control steps
        d.tick()
    held = np.array(d.env.data.qpos[d.env._arm_qadr], float)
    assert float(np.abs(held - start).max()) < 0.05, (
        "the zero action did not hold the arm — a lab slot with no policy "
        f"drifted {np.abs(held - start).max():.4f} rad")
    assert d.falls == 0


@needs_mars
def test_a_wheeled_slot_is_not_steered_and_reports_no_speed():
    """`set_cmd` and `sample_speed` are asked of the ENV, not of the id.

    An arm env has no `twist_cmd` (its base is disabled for every arm task —
    rolling the whole robot at the target is the cheapest way to satisfy a
    reach and not the thing being taught) and no `heading_lin_vel`. Both are
    read inside the 50 Hz loop, where an AttributeError stops the sim for the
    WHOLE roster, so a body with no drive channel must be a no-op and not a
    raise. A made-up 0.00 m/s would be worse than "—": it is a reading the
    robot never took.
    """
    d = V.Duck("d0", "MARS", V._zero_infer_for("mars"), seed=1, robot="mars")
    assert d.steers() is False
    d.set_cmd(np.array([0.9, 0.0, 0.3], np.float32))    # what lab_loop hands it
    d.tick()
    d.reset()                                            # and the episode edge
    assert d.forward_speed() is None


@needs_go2
def test_a_level_0_slot_idles_kinematically_at_its_keyframe():
    """A body read from an MJCF has no env, and must not be given one.

    Choosing one would mean choosing a policy convention nobody declared
    (`MjcfBody.env_class` raises and names the level that lands it), and
    handing the slot the duck's env is the failure the whole split exists to
    prevent — it arrives as a wrong picture and a wrong policy rather than as
    an error. So the slot holds the pose its author keyframed, and `step()`
    advances nothing: a Go2 stepped at `home` with no controller folds onto
    the floor in about a second, which is every stranger's robot drawn as a
    heap.
    """
    d = V.Duck("d0", "Go2", V._zero_infer_for(GO2), seed=1, robot=GO2)
    assert isinstance(d.env, LR.KinematicIdle)
    assert d.steers() is False
    base_z = float(d.env.data.xpos[1][2])
    for _ in range(100):
        d.tick()
    assert float(d.env.data.xpos[1][2]) == pytest.approx(base_z, abs=1e-9)
    assert base_z > 0.05, "the body is on the floor — that is not its keyframe"
    assert d.falls == 0
    # and the poses it streams line up with the scene the viewer fetches
    assert len(d.pose_payload()) == len(LR.scene_body_names(GO2))


# -------------------------------------------- the guard: a contract, not a width

def _contract_run(tmp_path, name: str, robot: str):
    """A run dir that RECORDS its contract, as `train.py` writes one."""
    d = tmp_path / "runs" / name
    d.mkdir(parents=True)
    (d / "policy.onnx").write_bytes(b"not really onnx")
    contract = REG.get(robot).contract()
    (d / "run.json").write_text(json.dumps(
        {"run_name": name, "robot": robot, "contract": contract.as_dict()}))
    return d


@needs_mars
@pytest.mark.parametrize("policy_robot,slot_robot", [("microduck", "mars"),
                                                     ("mars", "microduck")])
def test_a_recorded_contract_refuses_the_wrong_body_by_ID(tmp_path, policy_robot,
                                                          slot_robot):
    """§6.3: the four width guards become one contract comparison.

    A width is a PROXY — two bodies with the same observation width cross
    silently — and the arrival of a 32-float body makes that a question of
    when rather than whether. Here the widths differ too, so the refusal is
    checked on its MESSAGE: it must name the two contract ids, which is the
    thing a width can never say.
    """
    run = _contract_run(tmp_path, f"{policy_robot}-run", policy_robot)
    want = REG.get(policy_robot).contract()
    mine = REG.get(slot_robot).contract()
    why = LR.policy_refusal(run / "policy.onnx", slot_robot,
                            want_obs=want.obs_dim, have_obs=want.obs_dim)
    assert why and want.id in why and mine.id in why, why
    # …and the SAME file on its own body is waved through.
    assert LR.policy_refusal(run / "policy.onnx", policy_robot,
                             want_obs=want.obs_dim,
                             have_obs=want.obs_dim) is None


def test_a_file_with_nothing_to_say_is_still_refused_by_its_width(tmp_path):
    """The last-ditch check on the FILE, kept exactly as it was (§6.3).

    It is what catches a policy that records no contract at all — every run
    trained before the stamp existed, and every shipped drop that was never
    ours. A shipped G1 `walker.onnx` is the case that makes the ORDER matter:
    it has no metadata and no run directory, so `resolve()` would call it a
    duck and refuse it on its own body; only the self-describing rungs may
    decide, and everything else falls through to the width.
    """
    run = _run(tmp_path, "nameless", None)
    assert LR.policy_refusal(run / "policy.onnx", "microduck", 99, 61)
    assert LR.policy_refusal(run / "policy.onnx", "microduck", 61, 61) is None
    # no width claimed at all (an in-process checkpoint): nothing to compare
    assert LR.policy_refusal(run / "policy.onnx", "microduck", None, 61) is None
    # THE CASE THAT FIXES THE ORDER, and the one a `resolve()` here breaks: a
    # SHIPPED drop — a bare .onnx with no metadata and no run.json anywhere
    # near it, which is what Lucky Robots' G1 walker is. `resolve()`'s last
    # rung says "a file with nothing to say has always been a duck", which is
    # the right answer for naming a policy's body and the WRONG one for
    # judging it: it would refuse the G1's own walker on a G1 slot. Only the
    # self-describing rungs may decide, and this file has none.
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "walker.onnx").write_bytes(b"not really onnx")
    assert LR.policy_refusal(shipped / "walker.onnx", "g1", 99, 99) is None
    assert LR.policy_refusal(shipped / "walker.onnx", "g1", 61, 99)


# ------------------------------------------------------------- the endpoints

@needs_mars
def test_scene_endpoint_serves_the_mars_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    from microduck_local.robots import mars
    get_scene = _endpoint(V.make_app([]), "/scene", "GET")
    sc = get_scene(robot="mars")
    assert len(sc["meshes"]) == 14, (
        "mars.urdf ships nine visual STLs; the tyres come out of base.STL, a "
        "servo case out of each of link2/link3.STL, and the face panel and "
        "camera lenses out of head.STL")
    assert sc["bodies"][1] == "base_link"
    # The colours the SERVER painted (robots/mars.paint_shell): the viewer
    # adds gloss and rubber on top, but the shell itself arrives in the dump.
    # Innate's Blue / White — a blue head bar, and tyres darker than anything
    # else on the robot.
    head = next(g for g in sc["geoms"] if g["name"] == "head")
    assert head["rgba"][2] > head["rgba"][0] + 0.3, "the head is not blue"
    # The tyres are darker than every PAINTED part. Not darker than
    # everything: the servo cases are the same moulded black, and the head's
    # face panel is darker still, so this compares against the shell and the
    # blue rather than against every other geom.
    tyres = next(g for g in sc["geoms"] if g["name"] == mars.WHEEL_MESH)
    painted = [g for g in sc["geoms"]
               if g["name"] not in (*mars.MESH_PAINT, *mars.BLACK_LINKS)]
    assert max(tyres["rgba"][:3]) < min(max(g["rgba"][:3]) for g in painted)


@needs_go2
def test_scene_endpoint_serves_a_discovered_bodys_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    get_scene = _endpoint(V.make_app([]), "/scene", "GET")
    sc = get_scene(robot=GO2)
    assert len(sc["meshes"]) == 16, "the Go2's visual group holds 16 meshes"
    assert sc["geoms"] and sc["bodies"]


def test_an_unfetched_bodys_scene_404s_with_the_command_that_fixes_it(
        tmp_path, monkeypatch):
    """"unknown robot" and "you have not downloaded it yet" are different
    problems, and the palette's ⤓ answers only one of them."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    get_scene = _endpoint(V.make_app([]), "/scene", "GET")
    # (a) the body is in the registry but its ASSETS are gone. `visual_scene`
    # would raise deep inside a mesh read; the gate is `Body.ready()`.
    body = REG.registry().get("mars")
    if body is not None:
        monkeypatch.setattr(type(body), "ready", lambda self: False)
        with pytest.raises(HTTPException) as e:
            get_scene(robot="mars")
        assert e.value.status_code == 404
        assert "fetch-robot mars" in str(e.value.detail), e.value.detail
    # (b) nothing fetched at all — the id is still KNOWN (registry.ids lists a
    # built-in whose assets are absent), so the answer is still the command.
    monkeypatch.setattr(REG, "registry", lambda: {})
    with pytest.raises(HTTPException) as e:
        get_scene(robot="mars")
    assert e.value.status_code == 404
    assert "fetch-robot mars" in str(e.value.detail)
    # (c) and a body nobody has ever heard of is a different sentence
    with pytest.raises(HTTPException) as e:
        get_scene(robot="wombat")
    assert "unknown robot" in str(e.value.detail)


@needs_mars
@needs_go2
def test_get_robots_describes_every_body_the_registry_knows(tmp_path, monkeypatch):
    """It answered `"ready": True if rid == "microduck" else bool(g1_ready())`
    — so a third body inherited the G1's download state — and then
    hand-patched an absent G1 back onto the end of its own list."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    rows = {r["id"]: r for r in _endpoint(V.make_app([]), "/robots", "GET")()["robots"]}
    assert {"microduck", "g1", "mars", GO2} <= set(rows)
    assert rows["microduck"]["kind"] == "legged" and rows["microduck"]["animate"] is True
    # The G1 is a lazy proxy too (`robots/g1._LazyG1Spec`), and recognising a
    # proxy by `_resolve` on its TYPE caught it as well — which answered
    # `animate: false` and took the humanoid out of the 🎬 editor it has
    # always been in. The question is "is this id a CATALOGUE's", not "is
    # this object a proxy".
    assert rows["g1"]["animate"] is True
    # MARS: a wheeled body the 🎬 editor cannot pose — `PoseScratch` wants
    # effectors, soles and a base link, and pose_scratch("mars") died on the
    # last of those the moment a third body was registered.
    assert rows["mars"]["kind"] == "wheeled"
    assert rows["mars"]["animate"] is False
    assert rows["mars"]["noun"] == "MARS"
    # a discovered model is level 0: its own kind, nothing to teach
    assert rows[GO2]["kind"] == "generic"
    assert rows[GO2]["animate"] is False
    assert rows[GO2]["teach"] == []


@needs_mars
def test_the_pose_editor_refuses_a_body_it_cannot_pose(tmp_path, monkeypatch):
    """The door locked as well as unlisted: `pose_scratch("mars")` raised
    `AttributeError: 'MarsBody' object has no attribute 'base_body'`, which
    is a 500 with a traceback for a question that has a good answer."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    get_joints = _endpoint(V.make_app([]), "/joints", "GET")
    assert get_joints()["joints"]                         # the duck still poses
    with pytest.raises(HTTPException) as e:
        get_joints(robot="mars")
    assert e.value.status_code == 404


@needs_mars
def test_the_palette_shows_no_mars_group_but_tags_a_mars_run(tmp_path, monkeypatch):
    """MARS ships no policy this lab can run — Innate's learned skills are ACT
    checkpoints trained from demonstrations, not ONNX, and `innate-os` vendors
    none. So its shipped SECTION disappears rather than showing an empty
    heading; a run trained here is tagged with the body all the same."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    _run(tmp_path, "mars-reach-delta", "mars")
    groups = {g["key"]: g for g in LR.shipped_groups()}
    assert "mars" not in groups
    assert "pollen" in groups and groups["pollen"]["robot"] == "microduck"
    entry = next(p for p in V.discover_policies() if p["id"] == "run:mars-reach-delta")
    assert entry["robot"] == "mars"


@needs_mars
def test_posting_a_fetch_runs_that_bodys_own_download(tmp_path, monkeypatch):
    """One endpoint, any body: it used to 404 for everything but the G1."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    fetch_robot = _endpoint(V.make_app([]), "/robots/{robot}/fetch", "POST")
    called: list[str] = []
    body = REG.get("mars")
    monkeypatch.setattr(type(body), "fetch",
                        lambda self: called.append(self.id) or Path("."))
    assert fetch_robot("mars")["state"] == "fetching"
    for _ in range(200):                       # the download runs in a thread
        if called:
            break
        time.sleep(0.01)
    assert called == ["mars"]
    assert LR.fetch_state("mars")["state"] == "idle"
    # and the state is PER BODY — one shared flag showed a MARS download as
    # a G1 one.
    assert LR.fetch_state("g1")["state"] == "idle"
    with pytest.raises(HTTPException):
        fetch_robot("wombat")


@needs_mars
def test_the_teach_panel_offers_mars_its_own_two_tasks():
    """A recipe with no `suggest` phrase is trainable but INVISIBLE, so a
    MARS roster showed an empty 🎓 panel while both tasks existed."""
    assert [s["behavior"] for s in LR.teach_suggestions("mars")] == [
        "mars_reach", "mars_pick"]


@needs_go2
def test_a_level_0_body_has_nothing_to_teach():
    """And is therefore given no chip at all, rather than the duck's."""
    assert LR.teach_suggestions(GO2) == []


# --------------------------------------------------- a body with no policy

@needs_mars
def test_a_body_that_ships_nothing_can_still_be_put_on_the_stage(tmp_path,
                                                                 monkeypatch):
    """`spawn_duck` needs a palette id, and the two bodies that ship NO
    policy — MARS before anybody trains it, every Menagerie model — are
    exactly the ones most worth looking at first. `spawn_robot` is that
    door; the slot idles."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    app = V.make_app([])
    st = app.state.lab
    asyncio.run(app.state.do_spawn_robot("mars"))
    assert [d.robot for d in st.ducks] == ["mars"]
    st.ducks[0].tick()                                  # the loop's own call
    asyncio.run(app.state.do_spawn_robot("wombat"))     # not a body
    assert len(st.ducks) == 1
    assert any("wombat" in e for e in st.events)
    # a restart brings it back: restore_ducks used to drop every row with no
    # brain recorded, which silently emptied the stage.
    restored = V.restore_ducks(Path(V.lab_state_path()))
    assert [d.robot for d in restored] == ["mars"]
