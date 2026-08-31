"""Farm server units: roster persistence round-trip, helper spawn/remove guard
rules, TrainingJob launch/scale argv construction, staged-curriculum chaining,
stats payload shape, teach initFrom validation, reward-weight plumbing, the
user-chosen practice budget, one compiled mjModel per scene across the roster.
The training subprocess is faked throughout — the real --init-from
continuation lives in test_train_resume.py."""

import asyncio
import json
import types

import numpy as np
import pytest

from microduck_local import behaviors as B
from microduck_local import contract as C
from microduck_local import train_behavior as TB
from microduck_local import viz_server as V

# Above any real macOS pid, so psutil paths deterministically take their
# NoSuchProcess branches instead of touching a live process.
FAKE_PID = 4_194_304


class FakeProc:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = FAKE_PID
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch, tmp_path):
    """Record every trainer launch; keep runs/ and farm-state.json in tmp."""
    launches: list[FakeProc] = []

    def popen(cmd, **kwargs):
        proc = FakeProc(cmd, **kwargs)
        launches.append(proc)
        return proc

    monkeypatch.setattr(V.subprocess, "Popen", popen)
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("FARM_STATE_PATH", str(tmp_path / "farm-state.json"))
    return launches


def _fake_duck(duck_id, policy_id=None, onnx_path=None, label=None):
    return types.SimpleNamespace(id=duck_id, label=label or duck_id,
                                 policy_id=policy_id, onnx_path=onnx_path)


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def _endpoint(app, path: str, method: str):
    """Pull a handler straight off the FastAPI app (same trick as
    test_clips.py — no httpx in the project, so no TestClient)."""
    for r in app.routes:
        if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
            return r.endpoint
    raise AssertionError(f"no {method} {path} route")


# ------------------------------------------------------- TrainingJob argv

def test_training_job_launch_argv(fake_popen):
    job = V.TrainingJob("spin", helpers=1, steps=200_000, snap_steps=5000,
                        weights={"spin_fast": 3.0, "stay_put": -1.0})
    argv = fake_popen[0].cmd
    assert argv[1:4] == ["-m", "microduck_local.train_behavior", "spin"]
    assert _flag(argv, "--run-name") == job.run_name
    # Env count is BASE_ENVS + helpers * ENVS_PER_HELPER. ENVS_PER_HELPER is
    # 0: helpers are viewers, they do not resize the trainer. The test locks
    # the arithmetic so a future non-zero cannot silently drift.
    assert _flag(argv, "--envs") == str(V.BASE_ENVS + V.ENVS_PER_HELPER)
    assert _flag(argv, "--steps") == "200000"
    assert _flag(argv, "--snap-steps") == "5000"
    # Negative weights are clamped before they reach the subprocess — the
    # double-negation sign bug must not be reachable from the wire.
    assert json.loads(_flag(argv, "--weights-json")) == {"spin_fast": 3.0,
                                                         "stay_put": 0.0}
    assert "--init-from" not in argv
    assert job.payload()["weights"]["spin_fast"] == 3.0
    assert job.payload()["weights"]["stay_upright"] == 1.5  # default kept


def test_stop_sweeps_the_worker_tree(fake_popen, monkeypatch):
    """A stop must SIGTERM the SubprocVecEnv workers, not just the trainer.

    Regression: stop() terminated only the trainer, so each stop stranded ~10
    workers (reparented to init, ~322 MB) for the rest of the session. The
    child list must also be snapshotted BEFORE the trainer dies — afterwards
    there is no handle left to find them by.
    """
    killed, order = [], []

    class FakeChild:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            killed.append(self.pid)
            order.append(("child", self.pid))

    class FakeParent:
        def __init__(self, pid):
            assert pid == FAKE_PID

        def children(self, recursive=False):
            order.append(("snapshot", recursive))
            return [FakeChild(i) for i in range(10)]

    monkeypatch.setattr(V.psutil, "Process", FakeParent)
    job = V.TrainingJob("run")
    monkeypatch.setattr(job.proc, "terminate",
                        lambda: order.append(("parent", None)))
    job.stop()

    assert job.status == "stopped"
    assert killed == list(range(10)), "workers were not swept"
    assert order[0] == ("snapshot", True), "snapshot must precede the kill"
    assert order[1] == ("parent", None), "parent dies before its children"


def test_stop_does_not_block_the_event_loop(fake_popen, monkeypatch):
    """/teach/stop is an async handler, so stop() must not wait on the
    trainer — a blocking reap stalls the WebSocket stream for every duck."""
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    job = V.TrainingJob("run")

    def boom(timeout=None):
        raise AssertionError("stop() waited on the trainer")

    monkeypatch.setattr(job.proc, "wait", boom)
    job.stop()
    assert job.status == "stopped"


class FakeWorker:
    """A SubprocVecEnv worker that records its own SIGTERM."""

    def __init__(self, pid, killed, fail=None):
        self.pid = pid
        self._killed = killed
        self._fail = fail

    def terminate(self):
        if self._fail is not None:
            raise self._fail
        self._killed.append(self.pid)


def _fake_worker_tree(monkeypatch, current_proc, workers_for):
    """psutil.Process(pid).children() as the OS really behaves.

    The workers are findable only while the trainer is ALIVE; the moment it
    exits they are reparented to init and its child list is empty. Every
    sweep therefore has to work off a snapshot taken earlier — modelling that
    is the whole point of this fake.
    """

    class FakeParent:
        def __init__(self, pid):
            assert pid == FAKE_PID

        def children(self, recursive=False):
            proc = current_proc()
            return [] if proc.poll() is not None else list(workers_for(proc))

    monkeypatch.setattr(V.psutil, "Process", FakeParent)


def test_poll_sweeps_workers_a_dead_trainer_left_behind(fake_popen, monkeypatch):
    """A trainer the farm did NOT kill — OOM, a stray kill -9, a crash by
    signal — never runs multiprocessing's atexit hook, so it orphans its
    workers exactly as an un-swept stop did, and under the `fork` backend they
    never see EOF on their pipe either. poll() is the only witness left, and
    only because it keeps a snapshot from while the trainer was alive.
    """
    killed = []
    job = V.TrainingJob("run")
    workers = [FakeWorker(i, killed) for i in range(10)]
    _fake_worker_tree(monkeypatch, lambda: job.proc, lambda _p: workers)

    job.poll()               # ~1 Hz tick while training: snapshot, no kills
    assert killed == [], "a live trainer's fleet must be left alone"

    job.proc.returncode = -9  # killed from outside; workers now unfindable
    changed, _ = job.poll()

    assert changed and job.status == "failed"
    assert killed == list(range(10)), "orphans were not swept"


def test_sweep_survives_an_unreadable_worker(fake_popen, monkeypatch):
    """One handle that raises must not strand the rest of the fleet: psutil
    hands back AccessDenied/ZombieProcess as readily as NoSuchProcess."""
    killed = []
    job = V.TrainingJob("run")
    workers = [FakeWorker(0, killed),
               FakeWorker(1, killed, fail=V.psutil.AccessDenied(1)),
               FakeWorker(2, killed)]
    _fake_worker_tree(monkeypatch, lambda: job.proc, lambda _p: workers)

    job.stop()
    assert killed == [0, 2]


def test_stage_handoff_sweeps_only_the_finished_stage(fake_popen, monkeypatch):
    """The sweep must fire BEFORE _advance_stage() rebinds self.proc, or it
    would aim the finished stage's SIGTERMs at the incoming stage's fleet.
    (A clean stage exit already reaped its own workers via atexit, so the
    sweep is a no-op in the happy path — this pins where it points.)"""
    killed = []
    job = V.TrainingJob("backflip", steps=1000)
    s1 = [FakeWorker(i, killed) for i in range(3)]
    s2 = [FakeWorker(10 + i, killed) for i in range(3)]
    _fake_worker_tree(monkeypatch, lambda: job.proc,
                      lambda p: s2 if p is not fake_popen[0] else s1)

    job.poll()                       # snapshot stage 1's fleet
    _finish_stage(job, 1000)
    changed, _ = job.poll()          # stage 1 exits 0 → chain stage 2

    assert changed and job.status == "training" and len(fake_popen) == 2
    assert killed == [0, 1, 2], "stage 1's fleet was not the one swept"

    job.poll()                       # stage 2 alive: snapshot moves with it
    job.stop()
    assert killed == [0, 1, 2, 10, 11, 12]


def test_scale_restarts_warm_with_new_envs(fake_popen):
    job = V.TrainingJob("spin", steps=200_000, weights={"spin_fast": 3.0})
    (job.dir / "model.zip").touch()
    job.scale(2)
    assert fake_popen[0].returncode is not None  # old proc was terminated
    argv = fake_popen[1].cmd
    assert _flag(argv, "--init-from") == str(job.dir)
    two_helpers = V.BASE_ENVS + 2 * V.ENVS_PER_HELPER
    assert _flag(argv, "--envs") == str(two_helpers)
    assert _flag(argv, "--run-name") == job.run_name   # same progress.jsonl
    assert _flag(argv, "--steps") == "200000"          # absolute target kept
    # Weight overrides survive the restart — same scorecard, more envs.
    assert json.loads(_flag(argv, "--weights-json")) == {"spin_fast": 3.0}
    assert (job.envs, job.helpers) == (two_helpers, 2)
    assert job.restarting is False
    assert job.status == "training"

    job.scale(0)
    assert _flag(fake_popen[2].cmd, "--envs") == str(V.BASE_ENVS)


def test_scale_without_snapshot_relaunches_cold(fake_popen):
    job = V.TrainingJob("crouch", steps=100_000)
    job.scale(1)  # no model.zip on disk yet
    assert "--init-from" not in fake_popen[1].cmd
    assert _flag(fake_popen[1].cmd, "--envs") == str(
        V.BASE_ENVS + V.ENVS_PER_HELPER)


def test_poll_survives_scale_gap(fake_popen):
    """A dead proc during restart must not flip the job to failed."""
    job = V.TrainingJob("spin", steps=1000)
    job.proc.terminate()
    job.restarting = True
    job.poll()
    assert job.status == "training"
    job.restarting = False
    job.poll()
    assert job.status == "failed"


# ------------------------------------------------------- spawn/remove guards

def test_spawn_helper_guards(fake_popen):
    st = V.FarmState([_fake_duck("d0")])
    assert "no active training" in V.spawn_helper_error(st)

    st.job = V.TrainingJob("spin", steps=1000)
    assert "first training snapshot" in V.spawn_helper_error(st)

    (st.job.dir / "model.zip").touch()
    assert V.spawn_helper_error(st) is None

    st.scaling = True
    assert "mid-restart" in V.spawn_helper_error(st)
    st.scaling = False

    st.ducks += [_fake_duck(f"helper{i}") for i in range(1, V.MAX_HELPERS)]
    assert V.spawn_helper_error(st) is None  # one below the cap: still allowed
    st.ducks.append(_fake_duck(f"helper{V.MAX_HELPERS}"))
    assert f"helper cap reached ({V.MAX_HELPERS})" in (V.spawn_helper_error(st) or "")

    st.job.status = "done"
    assert "no active training" in V.spawn_helper_error(st)


def test_remove_duck_guards(fake_popen):
    from types import SimpleNamespace
    st = V.FarmState([_fake_duck("d0"), _fake_duck("trainee"),
                      _fake_duck("helper1")])
    # Any duck is removable now (roster declutter) …
    assert V.remove_duck_error(st, "d0") is None
    assert V.remove_duck_error(st, "trainee") is None  # no active run
    assert "no duck helper9" in V.remove_duck_error(st, "helper9")
    assert V.remove_duck_error(st, "helper1") is None
    # … except the trainee while a run is live (it's the run's only window),
    st.job = SimpleNamespace(status="training")
    assert "stop the run" in V.remove_duck_error(st, "trainee")
    st.job = SimpleNamespace(status="done")
    assert V.remove_duck_error(st, "trainee") is None
    # … and helpers still wait out a trainer restart (plain ducks don't).
    st.scaling = True
    assert "mid-restart" in V.remove_duck_error(st, "helper1")
    assert V.remove_duck_error(st, "d0") is None


def test_spawn_duck_guards_and_slots(fake_popen):
    st = V.FarmState([_fake_duck(f"d{i}") for i in range(3)])
    assert "needs a policy" in V.spawn_duck_error(st, None)
    assert V.spawn_duck_error(st, "pollen:alpha_stand") is None
    st.ducks = [_fake_duck(f"d{i}") for i in range(V.MAX_DUCKS)]
    assert "full" in V.spawn_duck_error(st, "pollen:alpha_stand")
    assert V.next_duck_slot(
        [_fake_duck("d0"), _fake_duck("d2"), _fake_duck("helper1")]) == 1


def test_next_helper_slot_reuses_gaps():
    assert V.next_helper_slot([_fake_duck("d0")]) == 1
    assert V.next_helper_slot([_fake_duck("helper1"), _fake_duck("helper3")]) == 2


# ------------------------------------------------------- farm-state.json

def test_farm_state_round_trip(fake_popen, monkeypatch, tmp_path):
    live = tmp_path / "live.onnx"
    ducks = [
        _fake_duck("d0", onnx_path="/policies/alpha_walking.onnx",
                   label="alpha_walking"),
        _fake_duck("d1", policy_id="pollen:alpha_stand", label="alpha_stand"),
        _fake_duck("trainee", onnx_path=str(live), label="🎓 Spin in place @40k"),
    ]
    V.save_farm_state(ducks)
    data = json.loads((tmp_path / "farm-state.json").read_text())
    assert data["version"] == 1
    assert data["ducks"][0] == {"id": "d0", "label": "alpha_walking",
                                "policy": None,
                                "onnxPath": "/policies/alpha_walking.onnx",
                                "showcase": False}
    assert data["ducks"][1]["policy"] == "pollen:alpha_stand"

    loaded_from: list[str] = []
    monkeypatch.setattr(V, "_onnx_infer",
                        lambda p: loaded_from.append(str(p)) or V._zero_infer)
    monkeypatch.setattr(V, "load_policy_infer",
                        lambda pid: loaded_from.append(pid) or V._zero_infer)
    restored = V.restore_ducks(tmp_path / "farm-state.json")
    assert [d.id for d in restored] == ["d0", "d1", "trainee"]
    assert [d.label for d in restored] == ["alpha_walking", "alpha_stand",
                                           "🎓 Spin in place @40k"]
    # Provenance survives the round trip, so the NEXT restart restores too.
    assert restored[1].policy_id == "pollen:alpha_stand"
    assert restored[2].onnx_path == str(live)
    assert loaded_from == ["/policies/alpha_walking.onnx", "pollen:alpha_stand",
                           str(live)]


def test_restore_skips_unloadable_entries(fake_popen, monkeypatch, tmp_path, capsys):
    def only_stand(pid):
        if pid != "pollen:alpha_stand":
            raise KeyError(pid)
        return V._zero_infer

    monkeypatch.setattr(V, "load_policy_infer", only_stand)
    monkeypatch.setattr(V, "_onnx_infer", lambda p: V._zero_infer)
    V.save_farm_state([
        _fake_duck("d0", policy_id="run:deleted-run"),   # registry entry gone
        _fake_duck("trainee"),                            # no brain recorded
        _fake_duck("d2", policy_id="pollen:alpha_stand"),
    ])
    restored = V.restore_ducks(V.farm_state_path())
    assert [d.id for d in restored] == ["d2"]
    out = capsys.readouterr().out
    assert "skipping d0" in out and "skipping trainee" in out


# ------------------------------------------------------- stats payload

def test_stats_payload_shape(fake_popen):
    sampler = V.StatsSampler()
    s = sampler.sample(None)
    assert set(s) == {"cpu", "mem", "farm", "trainer", "trainFps"}
    assert 0.0 <= s["cpu"] <= 100.0 and 0.0 < s["mem"] <= 100.0
    assert set(s["farm"]) == {"cpu", "memMb"} and s["farm"]["memMb"] > 0
    assert s["trainer"] is None and s["trainFps"] is None
    json.dumps(s)

    job = V.TrainingJob("spin", steps=200_000)
    (job.dir / "progress.jsonl").write_text(
        '{"steps": 3072, "total": 200000, "elapsed_s": 2.0}\n'
        '{"steps": 9216, "total": 200000, "elapsed_s": 4.0}\n')
    job.poll()
    s = sampler.sample(job)
    assert s["trainFps"] == pytest.approx(3072.0)   # 6144 steps / 2 s
    assert s["trainer"] is None  # fake pid resolves to no live process


def test_train_fps_none_across_restart_boundary(fake_popen):
    """elapsed_s restarts at 0 in a relaunched trainer — no negative fps."""
    job = V.TrainingJob("spin", steps=200_000)
    (job.dir / "progress.jsonl").write_text(
        '{"steps": 20480, "total": 200000, "elapsed_s": 30.0}\n')
    job.poll()
    assert job.train_fps() is None  # one point is not a rate
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 23552, "total": 200000, "elapsed_s": 1.5}\n')
    job.poll()
    assert job.train_fps() is None  # restart boundary
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 26624, "total": 200000, "elapsed_s": 3.0}\n')
    job.poll()
    assert job.train_fps() == pytest.approx(2048.0)


# ------------------------------------------------------- staged curriculum

def _finish_stage(job, steps):
    """Fake a stage completing: the trainer's final progress line, then a
    clean subprocess exit (poll() advances the chain on the EXIT, after the
    warm-start artifacts are guaranteed on disk)."""
    with open(job.dir / "progress.jsonl", "a") as f:
        f.write(json.dumps({"steps": steps, "total": steps, "done": True}) + "\n")
    job.proc.returncode = 0


def test_curriculum_teach_chains_stages(fake_popen):
    # Follows whatever chain the behavior declares — the recipe's stage list
    # is live tuning surface and must be free to grow/shrink.
    stages = B.BEHAVIORS["backflip"].curriculum
    n = len(stages)
    assert n >= 2
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500)

    # Stage 1: its own run name/dir, no warm start, stage env on the process.
    assert job.run_name.endswith("-s1")
    argv = fake_popen[0].cmd
    assert _flag(argv, "--run-name") == job.run_name
    assert _flag(argv, "--steps") == "1000"
    assert "--init-from" not in argv
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in stages[0].env} == stages[0].env
    p = job.payload()
    assert p["stage"] == {"idx": 1, "count": n, "label": stages[0].label,
                          "detail": stages[0].detail, "start": 1}
    assert p["stage"]["detail"]  # the inspector's story is really there
    # The steps override shrinks EVERY stage; overall totals follow suit.
    assert (p["progress"]["overallSteps"], p["progress"]["overallTotal"]) == (0, 1000 * n)

    # Stage 1 done → stage 2 auto-launches, --init-from stage 1's dir.
    s1_dir = job.dir
    _finish_stage(job, 1000)
    changed, _ = job.poll()
    assert changed and job.status == "training"
    assert len(fake_popen) == 2
    argv2 = fake_popen[1].cmd
    assert _flag(argv2, "--init-from") == str(s1_dir)
    assert _flag(argv2, "--run-name") == job.run_name
    assert job.run_name.endswith("-s2")
    assert job.dir != s1_dir
    env2 = fake_popen[1].kwargs["env"]
    assert {k: env2[k] for k in stages[1].env} == stages[1].env
    p = job.payload()
    assert p["stage"] == {"idx": 2, "count": n, "label": stages[1].label,
                          "detail": stages[1].detail, "start": 1}
    # Per-stage fields reset for the new run; overall progress carries on.
    assert (p["progress"]["steps"], p["progress"]["total"]) == (0, 1000)
    assert p["progress"]["overallSteps"] == 1000

    with open(job.dir / "progress.jsonl", "a") as f:
        f.write('{"steps": 400, "total": 1000}\n')
    job.poll()
    p = job.payload()
    assert (p["progress"]["steps"], p["progress"]["overallSteps"]) == (400, 1400)

    # Chain completes only when the FINAL stage does — with the ✔ run name.
    for i in range(2, n + 1):
        _finish_stage(job, 1000)
        job.poll()
    assert job.status == "done"
    assert job.run_name.endswith(f"-s{n}")  # relabel-on-done names the final stage
    assert len(fake_popen) == n
    assert job.payload()["progress"]["overallSteps"] == 1000 * n


def test_curriculum_scale_keeps_stage_warm_start(fake_popen):
    """A helper join before the current stage's first snapshot must not
    relaunch cold — that would silently drop the previous stage's brain."""
    job = V.TrainingJob("backflip", steps=1000)
    s1_dir = job.dir
    _finish_stage(job, 1000)
    job.poll()  # now on stage 2, no model.zip there yet
    job.scale(1)
    argv = fake_popen[2].cmd
    assert _flag(argv, "--init-from") == str(s1_dir)
    assert _flag(argv, "--envs") == str(V.BASE_ENVS + V.ENVS_PER_HELPER)
    stage2_env = B.BEHAVIORS["backflip"].curriculum[1].env
    env = fake_popen[2].kwargs["env"]
    assert {k: env[k] for k in stage2_env} == stage2_env

    # Once the stage has its own snapshot, scale resumes from its own dir.
    (job.dir / "model.zip").touch()
    job.scale(2)
    assert _flag(fake_popen[3].cmd, "--init-from") == str(job.dir)


def test_teach_stop_mid_chain_prevents_next_stage(fake_popen):
    job = V.TrainingJob("backflip", steps=1000)
    job.stop()
    job.poll()
    assert job.status == "stopped"
    assert len(fake_popen) == 1  # no stage 2

    # Even a stage that finished cleanly right as stop landed stays stopped.
    job2 = V.TrainingJob("backflip", steps=1000)
    _finish_stage(job2, 1000)
    job2.stop()
    job2.poll()
    assert job2.status == "stopped"
    assert len(fake_popen) == 2


def test_non_curriculum_behavior_stays_single_run(fake_popen):
    job = V.TrainingJob("spin", steps=1000)
    assert "-s1" not in job.run_name
    p = job.payload()
    assert p["stage"] is None
    # overall* mirror steps/total so consumers can read one pair everywhere.
    assert (p["progress"]["overallSteps"], p["progress"]["overallTotal"]) == (0, 1000)
    _finish_stage(job, 1000)
    job.poll()
    assert job.status == "done"
    assert len(fake_popen) == 1


def test_init_from_skips_curriculum_uses_final_window(fake_popen):
    """An explicit initFrom is a fine-tune of an existing run: single run,
    trained under the FINAL stage's env (the finished trick's spawn window)."""
    prev = V.RUNS_DIR / "teach-backflip-oldrun"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, init_from=prev)
    assert "-s1" not in job.run_name
    assert job.payload()["stage"] is None
    assert _flag(fake_popen[0].cmd, "--init-from") == str(prev)
    final_env = B.BEHAVIORS["backflip"].curriculum[-1].env
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in final_env} == final_env
    _finish_stage(job, 1000)
    job.poll()
    assert job.status == "done" and len(fake_popen) == 1


def test_behavior_card_exposes_curriculum():
    card = B.behavior_card(B.BEHAVIORS["backflip"])
    declared = B.BEHAVIORS["backflip"].curriculum
    assert card["curriculum"] == [
        {"label": s.label, "steps": s.steps, "detail": s.detail}
        for s in declared]
    assert len(card["curriculum"]) >= 2  # backflip really is staged
    # Every backflip stage tells the inspector its plain-English story.
    assert all(entry["detail"] for entry in card["curriculum"])
    json.dumps(card)  # stays JSON-friendly (env knobs are not in the card)
    assert B.behavior_card(B.BEHAVIORS["spin"])["curriculum"] == []


# ------------------------------------------------------- teach extras

def test_resolve_init_from_validation(fake_popen, monkeypatch):
    with pytest.raises(ValueError, match="does not exist"):
        V.resolve_init_from("no-such-run")
    with pytest.raises(ValueError, match="plain run name"):
        V.resolve_init_from("../../etc")

    run = V.RUNS_DIR / "half-run"
    run.mkdir(parents=True)
    (run / "model.zip").touch()
    with pytest.raises(ValueError, match="vecnormalize.pkl"):
        V.resolve_init_from("half-run")

    (run / "vecnormalize.pkl").touch()
    assert V.resolve_init_from("half-run") == run


def test_weight_overrides_reach_the_env():
    """The --weights-json path: make_env must hand overrides to BehaviorEnv,
    clamped, and the reward must actually be computed with them."""
    base = TB.make_env("spin", 0, 0, None)()
    tuned = TB.make_env("spin", 0, 0, {"spin_fast": 5.0, "stay_put": -2.0})()
    assert tuned.weight_overrides == {"spin_fast": 5.0, "stay_put": 0.0}

    import numpy as np
    base.reset(seed=0)
    tuned.reset(seed=0)
    action = np.full(14, 0.1, np.float32)
    for env in (base, tuned):
        for _ in range(5):
            env.step(action)
    # Same physics (same seed/actions), different scorecard: the tuned term
    # sum must scale by exactly new_weight / default_weight.
    assert tuned.reward_sums["spin_fast"] == pytest.approx(
        base.reward_sums["spin_fast"] * (5.0 / 2.5))
    assert tuned.reward_sums["stay_put_penalty"] == 0.0
    assert base.reward_sums["stay_put_penalty"] <= 0.0

# ------------------------------------------------------- sticky teach weights

def test_teach_weights_sticky_roundtrip(fake_popen):
    """A user's slider crank must survive into the next /teach that arrives
    without explicit weights — 'no weights' means 'same as I had it', not
    'back to defaults' (a fresh chat launch silently wiped a cranked
    legs_over and the user lost track of their setting). Both layers now:
    behavior-level weights AND per-stage overrides round-trip."""
    assert V.load_teach_weights() == {}
    entry = {"weights": {"legs_over": 2.5},
             "stageWeights": {"2": {"neck_kip": 3.0}}}
    V.save_teach_weights({"backflip": entry})
    sticky = V.load_teach_weights()
    # Loading fills in the layers the file predates (practice budget) with
    # "never set" — the panel then falls back to the recipe's own numbers.
    assert sticky == {"backflip": {**V.empty_sticky(), **entry}}
    # The inherit rule the /teach handler applies:
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500,
                        weights=sticky["backflip"]["weights"],
                        stage_weights=sticky["backflip"]["stageWeights"])
    assert job.weights == {"legs_over": 2.5}
    assert job.stage_weights == {2: {"neck_kip": 3.0}}
    assert "--weights-json" in fake_popen[0].cmd


def test_teach_weights_legacy_flat_shape_still_reads(fake_popen):
    """Files written before per-stage weights stored the flat behavior-level
    dict — they must keep loading, as the behavior-level layer."""
    V.teach_weights_path().write_text(json.dumps(
        {"backflip": {"legs_over": 2.5}}))
    sticky = V.load_teach_weights()
    assert sticky == {"backflip": {**V.empty_sticky(),
                                   "weights": {"legs_over": 2.5}}}


# ------------------------------------------------------- per-stage weights

def test_stage_weights_merge_into_each_stage_argv(fake_popen):
    """Stage overrides layer over behavior-level weights (stage wins per key)
    at each stage's LAUNCH; stages without overrides train on the
    behavior-level set alone. Catalog keys in a stage's overrides adopt that
    term like the behavior-level channel does."""
    job = V.TrainingJob("backflip", steps=1000,
                        weights={"legs_over": 2.0},
                        stage_weights={
                            "1": {"legs_over": 4.0, "calm_body": 2.0},
                            "99": {"legs_over": 9.0},   # out of range: dropped
                            "2": {"push_off": -1.0},    # clamped to 0
                        })
    assert json.loads(_flag(fake_popen[0].cmd, "--weights-json")) == {
        "legs_over": 4.0, "calm_body": 2.0}
    p = job.payload()
    assert p["stageWeights"] == {"1": {"legs_over": 4.0, "calm_body": 2.0},
                                 "2": {"push_off": 0.0}}
    # The adopted catalog term shows up as a card row (any layer counts).
    assert "calm_body" in job.extra_keys
    assert any(t["key"] == "calm_body" for t in p["behavior"]["terms"])

    _finish_stage(job, 1000)
    job.poll()  # stage 2: behavior-level + its own override
    assert json.loads(_flag(fake_popen[1].cmd, "--weights-json")) == {
        "legs_over": 2.0, "push_off": 0.0}
    _finish_stage(job, 1000)
    job.poll()  # stage 3 has no overrides -> behavior-level only
    assert json.loads(_flag(fake_popen[2].cmd, "--weights-json")) == {
        "legs_over": 2.0}


def test_set_stage_weights_live_edit(fake_popen):
    """Editing a FUTURE stage only records (no restart needed); editing the
    CURRENT stage reports changed, and the scale() warm-restart the endpoint
    then performs relaunches it under the new merged weights. The handoff
    launch re-reads the map, so recorded future edits land at their stage."""
    job = V.TrainingJob("backflip", steps=1000, weights={"legs_over": 2.0})
    assert job.set_stage_weights({"2": {"neck_kip": 3.0}}) is False
    assert job.stage_weights == {2: {"neck_kip": 3.0}}
    assert job.set_stage_weights({"2": {"neck_kip": 3.0},
                                  "1": {"legs_over": 5.0}}) is True
    (job.dir / "model.zip").touch()
    job.scale(job.helpers)  # the warm-restart /teach/weights performs
    argv = fake_popen[1].cmd
    assert _flag(argv, "--init-from") == str(job.dir)
    assert json.loads(_flag(argv, "--weights-json")) == {"legs_over": 5.0}
    _finish_stage(job, 1000)
    job.poll()  # stage 2 picks up the recorded future edit
    assert json.loads(_flag(fake_popen[2].cmd, "--weights-json")) == {
        "legs_over": 2.0, "neck_kip": 3.0}


def test_stage_weights_ignored_for_single_run_jobs(fake_popen):
    """A fine-tune (initFrom) collapses to a single run — per-stage overrides
    can't half-apply, so they're dropped whole."""
    prev = V.RUNS_DIR / "teach-backflip-oldrun"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, init_from=prev,
                        weights={"legs_over": 2.0},
                        stage_weights={"1": {"legs_over": 9.0}})
    assert job.stage_weights == {}
    assert json.loads(_flag(fake_popen[0].cmd, "--weights-json")) == {
        "legs_over": 2.0}
    assert job.set_stage_weights({"1": {"legs_over": 9.0}}) is False


# ------------------------------------------------------- startStage

def test_resolve_stage_init_picks_newest_trained(fake_popen):
    import os as _os
    with pytest.raises(ValueError, match="train the earlier stages first"):
        V.resolve_stage_init("backflip", 3)
    old = V.RUNS_DIR / "teach-backflip-aaaaaa-s2"
    new = V.RUNS_DIR / "teach-backflip-bbbbbb-s2"
    half = V.RUNS_DIR / "teach-backflip-cccccc-s2"  # no vecnormalize.pkl
    for d in (old, new, half):
        d.mkdir(parents=True)
        (d / "model.zip").touch()
    (old / "vecnormalize.pkl").touch()
    (new / "vecnormalize.pkl").touch()
    _os.utime(old / "model.zip", (1_000_000, 1_000_000))
    _os.utime(new / "model.zip", (2_000_000, 2_000_000))
    _os.utime(half / "model.zip", (3_000_000, 3_000_000))  # newest but unusable
    assert V.resolve_stage_init("backflip", 3) == new
    # A wrong stage number still errors even with s2 runs on disk.
    with pytest.raises(ValueError, match="stage 3"):
        V.resolve_stage_init("backflip", 4)


def test_start_stage_chain_construction(fake_popen):
    """startStage=N skips the earlier stages: the chain opens at -sN warm-
    started from the resolved prev-stage dir, overall totals count only the
    stages actually run, and the chain still walks to the real end."""
    stages = B.BEHAVIORS["backflip"].curriculum
    n = len(stages)
    prev = V.RUNS_DIR / "teach-backflip-oldrun-s2"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", steps=1000, start_stage=3,
                        stage_init_from=prev)
    assert job.run_name.endswith("-s3")
    argv = fake_popen[0].cmd
    assert _flag(argv, "--init-from") == str(prev)
    env = fake_popen[0].kwargs["env"]
    assert {k: env[k] for k in stages[2].env} == stages[2].env
    p = job.payload()
    assert p["stage"] == {"idx": 3, "count": n, "label": stages[2].label,
                          "detail": stages[2].detail, "start": 3}
    assert (p["progress"]["overallSteps"],
            p["progress"]["overallTotal"]) == (0, 1000 * (n - 2))
    # A scale() before the first snapshot keeps the cross-chain warm start.
    job.scale(1)
    assert _flag(fake_popen[1].cmd, "--init-from") == str(prev)
    for _ in range(3, n + 1):
        _finish_stage(job, 1000)
        job.poll()
    assert job.status == "done"
    assert job.run_name.endswith(f"-s{n}")
    assert job.payload()["progress"]["overallSteps"] == 1000 * (n - 2)


# ------------------------------------------------------- one model per scene

def test_roster_shares_one_compiled_model():
    """Every duck on a scene steps the SAME mjModel object.

    A compiled model costs ~470 MB on a process's first compile and ~90-140 MB
    per extra copy; the mjData a duck actually owns is ~0.9 MB. A six-duck farm
    held ~1.4 GB of identical, never-written model — Duck._make_env pins
    domain_rand=False, so there is nothing per-duck in there to keep apart.
    """
    a = V.Duck("m0", "a", V._zero_infer, seed=1, env_kwargs={})
    b = V.Duck("m1", "b", V._zero_infer, seed=2, env_kwargs={})
    assert a.env.model is b.env.model
    assert a.env._model_shared and b.env._model_shared
    assert a.env.data is not b.env.data      # the state still is per duck

    # …and one duck's rollout leaves its neighbour's alone, which is the whole
    # risk of sharing (the frame loop steps them serially, in this order).
    b_qpos = b.env.data.qpos.copy()
    for _ in range(20):
        a.tick()
    np.testing.assert_array_equal(b.env.data.qpos, b_qpos)


def test_scene_swap_gets_its_own_shared_model():
    """rebuild_env onto the full-collision scene: that scene shares a model of
    its own (the cache is keyed by scene), never the walk scene's."""
    walk = V.Duck("m2", "w", V._zero_infer, seed=3, env_kwargs={})
    all_kw = {"scene_xml": str(C.SCENE_ALL_XML), "terminate_on_fall": False}
    a = V.Duck("m3", "a", V._zero_infer, seed=4, env_kwargs=all_kw)
    b = V.Duck("m4", "b", V._zero_infer, seed=5, env_kwargs={})
    b.rebuild_env(all_kw)

    assert b.env.scene_path == str(C.SCENE_ALL_XML)
    assert b.env.terminate_on_fall is False
    assert a.env.model is b.env.model
    assert a.env.model is not walk.env.model
    for _ in range(20):        # the swapped env really runs
        b.tick()
    assert np.isfinite(b.obs).all()


def test_bam_farm_keeps_private_models(monkeypatch):
    """MICRODUCK_ACTUATOR=bam opts out of sharing instead of blowing up: the
    BAM actuator rewrites model.dof_frictionloss every physics substep, so
    siblings on one model would retune each other's servos."""
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "bam")
    duck = V.Duck("m5", "bam", V._zero_infer, seed=6, env_kwargs={})
    assert duck.env.bam is not None
    assert duck.env._model_shared is False
    for _ in range(10):
        duck.tick()


# ------------------------------------------------------- trainee preview (A)

def test_trainee_preview_env_mirrors_stage(fake_popen):
    """The 🎓 preview env must practice what the stage practices — the
    behavior's spawn families under the ACTIVE stage's knobs, read per
    instance (os.environ belongs to the trainer subprocess; the shared farm
    process can't use it). Before this the trainee always spawned STANDING
    and the user watched stand-then-topple during 'learning to land'."""
    b = B.BEHAVIORS["backflip"]
    stage1 = b.curriculum[0]
    duck = V.Duck("trainee", "t", V._zero_infer, seed=97,
                  env_kwargs=V.trainee_env_kwargs(b, stage1.env))
    assert isinstance(duck.env, B.BehaviorEnv)
    ov = duck.env.spawn_overrides
    # Every knob the stage declares is mirrored verbatim — EXCEPT the spawn
    # MIX, which is deliberately LEANED to 85% of the stage's dominant family
    # (the trainer's exact mix includes spawns that look identical to plain
    # standing, and a watcher read that as "the mirroring is broken").
    MIX = "MICRODUCK_SPAWN_FAMILY_PROBS"
    for k, v in stage1.env.items():
        if k != MIX:
            assert ov[k] == v, k
    probs = [float(x) for x in ov[MIX].split(",")]
    declared = [float(x) for x in stage1.env[MIX].split(",")]
    if max(declared) >= 0.85:
        assert probs == declared        # already all-in: never watered down
    else:
        assert max(probs) == 0.85 and abs(sum(probs) - 1.0) < 0.01
    two_pi = 2 * 3.141592653589793
    # …and the resets really rehearse the stage: (almost) every episode is a
    # trick spawn, and any mid-roll drop lands inside the stage's own window
    # (stages without a declared window — the stand-steady one — spawn only
    # the credited standing pose, so there is no window to check).
    lo = float(stage1.env.get("MICRODUCK_BF_SPAWN_LO", 0.0))
    hi = float(stage1.env.get("MICRODUCK_BF_SPAWN_HI", two_pi))
    rots = []
    for seed in range(30):
        duck.env.reset(seed=seed)
        rots.append(duck.env._bf_rot)
    mid = [r for r in rots if 0.0 < r < two_pi - 1e-9]
    assert all(lo <= r <= hi for r in mid)
    assert sum(1 for r in rots if r > 0.0) >= 20
    # A handoff rebuild really swaps the knobs (kwargs differ -> new env).
    windowed = next(st for st in b.curriculum[1:]
                    if "MICRODUCK_BF_SPAWN_HI" in st.env)
    duck.rebuild_env(V.trainee_env_kwargs(b, windowed.env))
    assert (duck.env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == windowed.env["MICRODUCK_BF_SPAWN_HI"])


def test_on_stage_handoff_rebuilds_trainee(fake_popen):
    """The farm-loop hook: a stage advance narrates the handoff and re-mirrors
    the trainee's preview env onto the NEW stage's spawn knobs."""
    rebuilt: list[dict] = []
    trainee = types.SimpleNamespace(
        id="trainee", rebuild_env=lambda kw: rebuilt.append(kw))
    helper = types.SimpleNamespace(
        id="helper1", rebuild_env=lambda kw: rebuilt.append(kw))
    st = V.FarmState([trainee, helper])
    st.job = V.TrainingJob("backflip", steps=1000)
    _finish_stage(st.job, 1000)
    st.job.poll()  # now on stage 2
    V.on_stage_handoff(st)
    assert len(rebuilt) == 2  # trainee AND helpers mirror the new stage
    assert rebuilt[0]["behavior_id"] == "backflip"
    stage2 = B.BEHAVIORS["backflip"].curriculum[1]
    assert (rebuilt[0]["spawn_overrides"]["MICRODUCK_BF_SPAWN_LO"]
            == stage2.env["MICRODUCK_BF_SPAWN_LO"])
    assert any("Training stage 2" in e for e in st.events)


# ------------------------------------------------------- showcase assign


def _mk_teach_run(name, behavior="backflip"):
    """A run dir shaped like a finished teach stage: assignable policy.onnx
    plus the behavior.json sibling env_kwargs/showcase resolution reads."""
    run = V.RUNS_DIR / name
    run.mkdir(parents=True)
    (run / "policy.onnx").touch()
    (run / "behavior.json").write_text(json.dumps({"behavior": behavior}))
    return run


def test_showcase_env_kwargs_uses_final_stage_knobs(fake_popen):
    """The "whole trick" assign rebuilds the duck's env as the behavior's own
    env under the LAST curriculum stage's knobs — the whole-trick stage by
    definition, so the server never learns any trick's knob names."""
    b = B.BEHAVIORS["backflip"]
    run = _mk_teach_run("teach-backflip-abc123-s4")
    kw = V.showcase_env_kwargs(str(run / "policy.onnx"))
    assert kw["behavior_id"] == "backflip"
    final = b.curriculum[-1].env
    ov = kw["spawn_overrides"]
    assert ov["MICRODUCK_BF_SPAWN_LO"] == final["MICRODUCK_BF_SPAWN_LO"]
    assert ov["MICRODUCK_BF_SPAWN_HI"] == final["MICRODUCK_BF_SPAWN_HI"]
    if b.spotter_fn is not None:
        # SPOTTED showcase: standing starts (the whole trick, front to back)
        # with the demo assist carrying the arc the actuators can't.
        assert kw["spotter"] is True
        assert ov["MICRODUCK_SPAWN_FAMILY_PROBS"] == "0.0,0.0"
    else:
        # Unspotted: lean the mix so the viewer sees the trick, not idling.
        probs = [float(x) for x in ov["MICRODUCK_SPAWN_FAMILY_PROBS"].split(",")]
        assert max(probs) == 0.85 and abs(sum(probs) - 1.0) < 0.01
    # A duck rebuilt with these kwargs really runs the behavior env.
    duck = V.Duck("d5", "x", V._zero_infer, seed=1, env_kwargs={})
    duck.rebuild_env(kw)
    assert isinstance(duck.env, B.BehaviorEnv)
    assert (duck.env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == final["MICRODUCK_BF_SPAWN_HI"])
    assert duck.env.spotter == (b.spotter_fn is not None)


def test_plain_assign_env_kwargs_unchanged(fake_popen):
    """A curriculum-stage policy assigned WITHOUT the showcase flag keeps
    exactly today's preview: the behavior's scene/termination physics but
    ordinary standing-start spawns (no behavior env, no overrides)."""
    run = _mk_teach_run("teach-backflip-abc123-s4")
    kw = V.env_kwargs_for_policy_path(str(run / "policy.onnx"))
    assert kw == V.env_kwargs_for_behavior(B.BEHAVIORS["backflip"])
    assert "behavior_id" not in kw and "spawn_overrides" not in kw


def test_showcase_is_noop_without_curriculum(fake_popen):
    """No curriculum (single-run behaviors, pollen policies, plain runs) =
    nothing for showcase to mean — callers fall back to a plain assign."""
    run = _mk_teach_run("teach-spin-abc123", behavior="spin")
    assert V.showcase_env_kwargs(str(run / "policy.onnx")) is None
    plain = V.RUNS_DIR / "plain-run"
    plain.mkdir(parents=True)  # no behavior.json sibling at all
    assert V.showcase_env_kwargs(str(plain / "policy.onnx")) is None
    assert V.showcase_env_kwargs(None) is None


def test_showcase_label_names_the_chain():
    """The roster label reads as the whole trick — the chain's name, not one
    stage's run name — with the ✨ showcase mark."""
    assert V.showcase_label("run:teach-backflip-ef471c-s4") == "backflip-ef471c ✨"
    # Non-chain ids (defensive: showcase only fires with a curriculum, but
    # the label must never crash on odd input) keep their name.
    assert V.showcase_label("run:some-run") == "some-run ✨"


def test_farm_state_showcase_round_trip(fake_popen, monkeypatch):
    """A showcase duck comes back showcasing after a restart — otherwise its
    persisted ✨ label would promise full-arc spawns its restored env no
    longer performs."""
    _mk_teach_run("teach-backflip-abc123-s4")
    monkeypatch.setattr(V, "load_policy_infer", lambda pid: V._zero_infer)
    d = _fake_duck("d0", policy_id="run:teach-backflip-abc123-s4",
                   label="backflip-abc123 ✨")
    d.showcase = True
    V.save_farm_state([d])
    data = json.loads(V.farm_state_path().read_text())
    assert data["ducks"][0]["showcase"] is True
    restored = V.restore_ducks(V.farm_state_path())
    assert restored[0].showcase is True
    assert isinstance(restored[0].env, B.BehaviorEnv)
    final = B.BEHAVIORS["backflip"].curriculum[-1].env
    assert (restored[0].env.spawn_overrides["MICRODUCK_BF_SPAWN_HI"]
            == final["MICRODUCK_BF_SPAWN_HI"])


# ------------------------------------------------------- policies palette

def _mkrun(name, mtime, artifact="policy.onnx"):
    d = V.RUNS_DIR / name
    d.mkdir(parents=True)
    (d / artifact).touch()
    import os as _os
    _os.utime(d / artifact, (mtime, mtime))
    return d


def test_discover_policies_timestamps_sort_and_chains(fake_popen):
    """Run entries carry mtime (epoch seconds), sort newest-first, and stage
    runs are annotated with their chain prefix + 1-based stage so the panel
    can fold a curriculum chain into one family."""
    _mkrun("teach-backflip-aaaaaa-s1", 1_000)
    _mkrun("teach-backflip-aaaaaa-s2", 5_000)
    _mkrun("plain-run", 3_000)
    _mkrun("no-policy-yet", 9_000, artifact="live.onnx")  # not assignable yet
    runs = [p for p in V.discover_policies() if p["group"] == "runs"]
    assert [p["label"] for p in runs] == [
        "teach-backflip-aaaaaa-s2", "plain-run", "teach-backflip-aaaaaa-s1"]
    by = {p["label"]: p for p in runs}
    assert by["plain-run"]["mtime"] == 3_000
    # Non-chain runs carry no chain/stage keys at all.
    assert "chain" not in by["plain-run"] and "stage" not in by["plain-run"]
    assert by["teach-backflip-aaaaaa-s1"]["chain"] == "teach-backflip-aaaaaa"
    assert by["teach-backflip-aaaaaa-s1"]["stage"] == 1
    assert by["teach-backflip-aaaaaa-s2"]["stage"] == 2
    # ids/paths keep the exact assignable shape (drag-to-assign unchanged).
    assert by["teach-backflip-aaaaaa-s2"]["id"] == "run:teach-backflip-aaaaaa-s2"
    assert by["teach-backflip-aaaaaa-s2"]["path"].endswith(
        "teach-backflip-aaaaaa-s2/policy.onnx")
    json.dumps(runs)


def test_run_mtime_fallbacks(fake_popen):
    """policy.onnx first, then live.onnx, then progress.jsonl — a run still
    training (or stopped before export) must still get a timestamp."""
    import os as _os
    run = V.RUNS_DIR / "r"
    run.mkdir(parents=True)
    assert V._run_mtime(run) is None
    (run / "progress.jsonl").touch()
    _os.utime(run / "progress.jsonl", (10, 10))
    assert V._run_mtime(run) == 10
    (run / "live.onnx").touch()
    _os.utime(run / "live.onnx", (20, 20))
    assert V._run_mtime(run) == 20
    (run / "policy.onnx").touch()
    _os.utime(run / "policy.onnx", (30, 30))
    assert V._run_mtime(run) == 30



def test_spotter_is_demo_only_and_self_clearing(fake_popen):
    """The spotter must never leak into training: it is off unless an env is
    built with spotter=True, it releases into the policy's own territory, and
    it never carries an applied force across an episode boundary."""
    plain = B.BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                          action_delay=False, random_yaw=False, seed=0)
    assert plain.spotter is False           # trainers build envs this way
    spotted = B.BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                            action_delay=False, random_yaw=False, seed=0,
                            spotter=True,  # as the spotted showcase builds it:
                            spawn_overrides={"MICRODUCK_SPAWN_FAMILY_PROBS": "0.0,0.0"})
    import numpy as np
    spotted.reset(seed=0)
    for _ in range(40):
        spotted.step(np.zeros(14, np.float32))
    assert spotted.spotter_active            # assisting through the dead zone
    assert spotted.data.qfrc_applied[4] < 0  # backward pitch
    spotted._bf_rot = 5.0                    # past the assist's release point
    spotted.step(np.zeros(14, np.float32))
    assert not spotted.spotter_active and spotted.data.qfrc_applied[4] == 0.0
    spotted.reset(seed=1)                    # and never across episodes
    assert not spotted.spotter_active and float(
        np.max(np.abs(spotted.data.qfrc_applied))) == 0.0


def test_chain_warm_start_from_another_run(fake_popen, tmp_path):
    """initFrom + startStage runs the whole CHAIN warm-started from that run
    (initFrom alone keeps its single-run fine-tune meaning). The backflip
    could not learn to hold a stand on its own; the crouch/one-leg policies
    hold one for a full episode, and inheriting that is the point."""
    donor = V.RUNS_DIR / "teach-crouch-donor"
    donor.mkdir(parents=True)
    (donor / "model.zip").touch()
    (donor / "policy.onnx").touch()
    job = V.TrainingJob("backflip", steps=1000, snap_steps=500,
                        start_stage=1, stage_init_from=donor)
    argv = fake_popen[0].cmd
    assert _flag(argv, "--init-from") == str(donor)   # stage 1 inherits it
    assert job.run_name.endswith("-s1")
    _finish_stage(job, 1000)
    job.poll()
    # …and stage 2 chains from stage 1 as usual, not from the donor again.
    assert _flag(fake_popen[1].cmd, "--init-from") != str(donor)
    assert job.run_name.endswith("-s2")


def test_showcase_duck_hands_off_to_a_standing_brain(fake_popen):
    """A finished trick hot-swaps to a brain that can stand — the robot's own
    pattern. The trick policy lands in a crouch it cannot rise from; the
    handoff turned 0/12 stands into 11/12 with 8.1 s holds."""
    run = _mk_teach_run("teach-backflip-abc123-s4")
    ho = V.handoff_for(str(run / "policy.onnx"))
    assert ho is not None and ho[1] == "alpha_stand"
    duck = V.Duck("d9", "x", V._zero_infer, seed=1, env_kwargs={})
    duck.handoff_infer, duck.handoff_label = ho
    # Before the trick completes, the duck stays on its own brain…
    duck.env._bf_rot = 0.0
    assert duck._handoff_due() is False
    # …and once the roll is done with both feet down, control moves over.
    duck.env._bf_rot = 6.0
    duck.env.foot_contact_state = {"left": True, "right": True}
    assert duck._handoff_due() is True
    duck.tick()
    assert duck.handed is True
    duck.reset()
    assert duck.handed is False   # every episode starts on the trick again


def test_plain_assign_has_no_handoff(fake_popen):
    """Only showcase ducks hand off — a single-stage assign is being watched
    for what THAT policy does, unaided."""
    duck = V.Duck("d8", "x", V._zero_infer, seed=1, env_kwargs={})
    assert duck.handoff_infer is None
    duck.env._bf_rot = 6.0
    duck.env.foot_contact_state = {"left": True, "right": True}
    duck.tick()
    assert duck.handed is False


# --------------------------------------------------- practice-step budget

def _declared(behavior_id):
    """What the recipe itself asks for: per-stage steps, or the single-run
    default. Read from the library, never hardcoded — the budgets are live
    tuning surface."""
    b = B.BEHAVIORS[behavior_id]
    return [s.steps for s in b.curriculum] or [b.default_steps]


def test_split_step_budget_keeps_ratios_and_sums_exactly():
    """A chain's SHAPE is the curriculum's whole point, so a user's total is
    distributed in proportion to the declared stages — and the parts add up
    to exactly the number they typed, because a stage list that doesn't sum
    to the headline figure is the same misreport the control exists to fix."""
    declared = [1_000_000, 2_000_000, 1_000_000, 1_500_000, 1_500_000]
    for total in (500_000, 3_000_000, 7_000_000, 12_345_678):
        parts = V.split_step_budget(declared, total)
        assert sum(parts) == total
        scale = total / sum(declared)
        for want, got in zip(declared, parts):
            assert abs(got - want * scale) <= 1.0   # rounding only
    # The declared total reproduces the declared plan exactly.
    assert V.split_step_budget(declared, sum(declared)) == declared
    # Scaled to nothing, no stage is rounded out of existence.
    assert V.split_step_budget(declared, 5) == [1, 1, 1, 1, 1]
    assert V.split_step_budget([], 1000) == []


def test_no_budget_reproduces_the_declared_defaults(fake_popen):
    """Regression guard: the control is opt-in. Omitting it must train
    EXACTLY what the recipe declares, stage for stage."""
    job = V.TrainingJob("backflip")
    assert job.stage_steps == _declared("backflip")
    assert job.budget is None
    assert job.payload()["stepBudget"] == sum(_declared("backflip"))
    assert _flag(fake_popen[0].cmd, "--steps") == str(_declared("backflip")[0])

    single = V.TrainingJob("spin")
    assert single.stage_steps == [B.BEHAVIORS["spin"].default_steps]


def test_total_budget_splits_across_the_chain(fake_popen):
    """The user's number is the TOTAL for the whole trick. Reading it as
    "this many per stage" would multiply the real cost by the stage count —
    the exact surprise this control removes."""
    declared = _declared("backflip")
    job = V.TrainingJob("backflip", budget=3_500_000)
    assert sum(job.stage_steps) == 3_500_000
    scale = 3_500_000 / sum(declared)
    for want, got in zip(declared, job.stage_steps):
        assert abs(got - want * scale) <= 1.0
    # What the payload advertises is what the trainer is told, stage by
    # stage (the trainer copies --steps into the run's behavior.json, so the
    # provenance on disk follows automatically).
    p = job.payload()
    assert p["stageSteps"] == job.stage_steps
    assert p["stepBudget"] == 3_500_000
    assert p["progress"]["overallTotal"] == 3_500_000
    assert _flag(fake_popen[0].cmd, "--steps") == str(job.stage_steps[0])
    for i in range(1, len(declared)):
        _finish_stage(job, job.stage_steps[i - 1])
        job.poll()
        assert _flag(fake_popen[i].cmd, "--steps") == str(job.stage_steps[i])


def test_total_budget_on_a_single_run_behavior(fake_popen):
    job = V.TrainingJob("spin", budget=1_250_000)
    assert job.stage_steps == [1_250_000]
    assert _flag(fake_popen[0].cmd, "--steps") == "1250000"
    # A fine-tune is a single run too — it gets the whole budget, once.
    prev = V.RUNS_DIR / "prev"
    prev.mkdir(parents=True)
    ft = V.TrainingJob("backflip", init_from=prev, budget=800_000)
    assert ft.stage_steps == [800_000]


def test_budget_is_clamped_to_a_sane_range(fake_popen):
    """A typo in the millions field must not become an overnight run."""
    assert V.TrainingJob("spin", budget=1).stage_steps == [V.MIN_STEP_BUDGET]
    assert (V.TrainingJob("spin", budget=10 ** 12).stage_steps
            == [V.MAX_STEP_BUDGET])


def test_per_stage_budget_wins_over_the_split(fake_popen):
    """Per-stage step counts layer exactly like per-stage weights: explicit
    value wins, everything else keeps its proportional share."""
    declared = _declared("backflip")
    job = V.TrainingJob("backflip", budget=3_500_000,
                        stage_budgets={"2": 900_000, "0": 5, "99": 5,
                                       "nope": 5, "3": 0})
    split = V.split_step_budget(declared, 3_500_000)
    assert job.stage_budgets == {2: 900_000}          # junk keys drop
    assert job.stage_steps == [split[0], 900_000, *split[2:]]
    assert job.payload()["stageBudgets"] == {"2": 900_000}
    # The pinned stage is honest on the wire when it actually launches.
    _finish_stage(job, job.stage_steps[0])
    job.poll()
    assert _flag(fake_popen[1].cmd, "--steps") == "900000"


def test_per_stage_budget_ignored_for_single_run_jobs(fake_popen):
    """A fine-tune has no stages to key — the layer drops rather than
    half-applying one stage's number to the whole run."""
    prev = V.RUNS_DIR / "prev"
    prev.mkdir(parents=True)
    job = V.TrainingJob("backflip", init_from=prev, budget=800_000,
                        stage_budgets={"2": 900_000})
    assert job.stage_budgets == {}
    assert job.stage_steps == [800_000]


def test_steps_override_still_beats_a_chosen_budget(fake_popen):
    """TEACH_STEPS_OVERRIDE keeps its own meaning — that many steps for EVERY
    stage — so a probe runs a tiny job whatever the user last picked, and the
    probe's 1k never leaks back into the sticky budget."""
    job = V.TrainingJob("backflip", steps=1000, budget=8_000_000,
                        stage_budgets={"2": 5_000_000})
    assert job.stage_steps == [1000] * len(_declared("backflip"))
    assert job.budget is None
    assert _flag(fake_popen[0].cmd, "--steps") == "1000"


def test_teach_endpoint_makes_the_budget_sticky(fake_popen, monkeypatch):
    """End to end through the handler: the chosen budget reaches the job,
    lands in the sticky file, and a later ask with no number keeps it — the
    same rule the sliders follow, so 'always 4M' is typed once."""
    # The handler reloads behaviors.py to pick up recipe edits; in-process
    # that swaps class objects other tests hold, so neutralize it here.
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    app = V.make_app([])
    teach = _endpoint(app, "/teach", "POST")
    stop = _endpoint(app, "/teach/stop", "POST")

    out = asyncio.run(teach(V.TeachReq(text="do a backflip", steps=3_500_000)))
    assert out["matched"]
    assert out["job"]["stepBudget"] == 3_500_000
    assert V.load_teach_weights()["backflip"]["steps"] == 3_500_000

    asyncio.run(stop())
    again = asyncio.run(teach(V.TeachReq(text="do a backflip")))
    assert again["job"]["stepBudget"] == 3_500_000
    assert again["job"]["stageSteps"] == out["job"]["stageSteps"]

    # A per-stage pin sticks too, and a new total re-splits around it.
    asyncio.run(stop())
    pinned = asyncio.run(teach(V.TeachReq(text="do a backflip",
                                          stageSteps={"1": 400_000})))
    assert pinned["job"]["stageBudgets"] == {"1": 400_000}
    assert pinned["job"]["stageSteps"][0] == 400_000
    assert V.load_teach_weights()["backflip"]["stageSteps"] == {"1": 400_000}


def test_teach_steps_override_does_not_poison_the_sticky_budget(fake_popen,
                                                                monkeypatch):
    import importlib
    monkeypatch.setattr(importlib, "reload", lambda m: m)
    V.save_teach_weights({"spin": {**V.empty_sticky(), "steps": 4_000_000}})
    monkeypatch.setenv("TEACH_STEPS_OVERRIDE", "1000")
    app = V.make_app([])
    out = asyncio.run(_endpoint(app, "/teach", "POST")(
        V.TeachReq(text="spin in place")))
    assert out["job"]["stageSteps"] == [1000]
    assert V.load_teach_weights()["spin"]["steps"] == 4_000_000


# --------------------------------------------------- forward-speed readout


def _pose(env, quat, world_vel):
    """Impose a known heading and a known WORLD-frame trunk velocity.

    qvel[0:3] of a free joint is global linear velocity, so this is a clean
    "the duck is moving that way, facing this way" fixture; mj_forward
    refreshes the derived rotation matrices heading_lin_vel reads.
    """
    import mujoco
    env.data.qpos[3:7] = quat
    env.data.qvel[:] = 0.0
    env.data.qvel[0:3] = world_vel
    mujoco.mj_forward(env.model, env.data)


def _yaw(deg):
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, 0.0, np.sin(h)]


def _pitch(deg):
    """+deg = nose-down (right-handed about the trunk's +y, as POST /pose)."""
    h = np.radians(deg) / 2
    return [np.cos(h), 0.0, np.sin(h), 0.0]


def _inertial_forward(env) -> float:
    """The WRONG number: mj_objectVelocity(flg_local=1)'s "forward" slot."""
    import mujoco
    out = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                             env.trunk_body_id, out, 1)
    return float(out[3])  # res is rotation:translation, so [3] is local vx


def test_reported_speed_is_the_heading_frame():
    """The readout must project world velocity onto the duck's YAW facing —
    not onto mj_objectVelocity's local frame (which is the trunk's INERTIAL
    frame, whose axes are nowhere near the body's), and not onto the body's
    own x axis (which pays for diving). This test exists to FAIL if either
    substitution is ever made; see Duck.sample_speed."""
    duck = V.Duck("spd0", "x", V._zero_infer, seed=1, env_kwargs={})
    env = duck.env

    # Turned 90° left and moving world-+y at 0.6 m/s: the duck is running
    # straight ahead at 0.6.
    _pose(env, _yaw(90), [0.0, 0.6, 0.0])
    duck.speed_hist.clear()
    duck.sample_speed()
    assert duck.forward_speed() == pytest.approx(0.6, abs=1e-3)
    # World +x is 0 here, so this also pins that the readout is not just
    # qvel[0] with a heading-shaped comment on it.
    assert env.data.qvel[0] == pytest.approx(0.0, abs=1e-9)
    # And the inertial-frame call reads ~0 for this same motion — on this
    # trunk its x axis is close to PERPENDICULAR to forward, so a readout
    # built on it is blind to forward speed entirely.
    assert abs(_inertial_forward(env)) < 0.05

    # Nose-down 45° and falling straight down: no ground is being covered,
    # so forward speed is zero. The body-x projection would score +0.71 for
    # this dive — that is the other half of the trap.
    duck.speed_hist.clear()
    _pose(env, _pitch(45), [0.0, 0.0, -1.0])
    duck.sample_speed()
    assert duck.forward_speed() == pytest.approx(0.0, abs=1e-3)
    body_x = env.data.xmat[env.trunk_body_id].reshape(3, 3) @ np.array([1.0, 0, 0])
    assert float(body_x @ np.array([0.0, 0.0, -1.0])) > 0.5


def test_forward_speed_averages_half_a_second():
    """A stepping gait's instantaneous forward speed swings within each
    stride; the row shows the mean of the last SPEED_WINDOW control steps
    (0.5 s at 50 Hz) so the number is readable, and older samples age out."""
    assert V.SPEED_WINDOW == 25  # 0.5 s at TICK_HZ
    duck = V.Duck("spd1", "x", V._zero_infer, seed=2, env_kwargs={})
    duck.speed_hist.clear()
    assert duck.forward_speed() is None  # nothing sampled yet → "—", not 0.00

    duck.speed_hist.extend([0.0, 0.8] * (V.SPEED_WINDOW // 2))
    assert duck.forward_speed() == pytest.approx(0.4, abs=1e-3)
    # Fully overwriting the window drops every older sample.
    duck.speed_hist.extend([0.2] * V.SPEED_WINDOW)
    assert duck.forward_speed() == pytest.approx(0.2, abs=1e-3)


def test_speed_window_clears_on_reset_and_policy_swap():
    """Speed is per EPISODE and per BRAIN: half a second of the run that just
    ended in a faceplant must not follow the duck into its next episode, and
    the outgoing policy's speed must not be credited to the incoming one."""
    duck = V.Duck("spd2", "x", V._zero_infer, seed=3, env_kwargs={})
    duck.speed_hist.extend([0.3] * V.SPEED_WINDOW)
    duck.reset()
    assert duck.forward_speed() is None

    duck.speed_hist.extend([0.3] * V.SPEED_WINDOW)
    duck.swap_policy("other", V._zero_infer)
    assert duck.forward_speed() is None


def test_tick_records_a_speed_sample():
    """The window is filled by the farm loop's own tick, not only by tests."""
    duck = V.Duck("spd3", "x", V._zero_infer, seed=4, env_kwargs={})
    duck.speed_hist.clear()
    duck.tick()
    assert len(duck.speed_hist) == 1
    assert duck.forward_speed() is not None


# ------------------------------------------------ frame broadcast robustness


def _farm_state(app):
    """The FarmState the handlers close over. make_app keeps it in a closure
    rather than on app.state, and reaching st.clients is the only way to drive
    the broadcast (same "no TestClient in this project" constraint that makes
    _endpoint necessary)."""
    fn = _endpoint(app, "/teach", "POST")
    return fn.__closure__[fn.__code__.co_freevars.index("st")].cell_contents


def test_broadcast_survives_a_client_arriving_mid_send():
    """A browser connecting while the loop is suspended in `await
    send_text` mutates st.clients mid-iteration. That RuntimeError killed
    farm_loop outright — and since nothing ever retrieves that task's
    exception, the farm went on serving HTTP and accepting sockets while
    every duck froze, with no line in the log: the viewer's badge sat green
    over an empty scene. Iterate a snapshot instead.

    The frame payload is asserted here too: this is the only place the real
    per-duck dict is built."""
    walker = V.Duck("d0", "w", V._zero_infer, seed=1, env_kwargs={})
    trainee = V.Duck("trainee", "t", V._zero_infer, seed=2, env_kwargs={})
    app = V.make_app([walker, trainee])
    st = _farm_state(app)

    class Joiner:
        """A client whose send is slow enough for a second tab to connect."""

        def __init__(self):
            self.frames: list[str] = []

        async def send_text(self, frame):
            self.frames.append(frame)
            st.clients.add(object())  # another browser lands mid-broadcast

    joiner = Joiner()
    st.clients.add(joiner)

    async def run():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.4)  # ~10 broadcasts at 25 Hz

    asyncio.run(run())

    # Before the fix this stopped at exactly one — the first send blew the
    # loop up and nothing was ever sent again.
    assert len(joiner.frames) > 2, "farm_loop stopped broadcasting"

    ducks = json.loads(joiner.frames[-1])["ducks"]
    assert [d["id"] for d in ducks] == ["d0", "trainee"]
    assert isinstance(ducks[0]["speed"], float)
    # A steerable duck reports what it was asked for; the trainee runs a
    # pinned-zero twist, so it reports achieved only.
    assert isinstance(ducks[0]["cmdSpeed"], float)
    assert ducks[1]["cmdSpeed"] is None


def test_scale_after_a_stop_does_not_strand_a_trainer(fake_popen, monkeypatch):
    """A stop that lands while a rescale is in flight must win.

    scale() runs on a worker thread while /teach/stop runs on the event loop.
    Without the stop_requested flag, stop() killed the already-dead old
    process and set status="stopped", then scale() launched a BRAND-NEW
    trainer and rebound self.proc — and poll() is gated on
    status == "training", so that trainer plus its 16-32 fork workers ran
    unreachable until the farm process exited.
    """
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    job = V.TrainingJob("run")
    before = len(fake_popen)

    job.stop()                      # the stop lands...
    job.scale(helpers=2)            # ...and the in-flight rescale must yield

    assert job.status == "stopped"
    assert len(fake_popen) == before, "scale() relaunched a trainer after a stop"
    assert not job.restarting, "restarting flag leaked"


# ------------------------------------------------------------------ /scene

def test_scene_geoms_carry_material_colors():
    """Every visual geom streams its MJCF material name + rgba — the viewer
    paints per-part colors (eye ring, mouth, shells) from these instead of
    guessing one color per body, so a missing/renamed field would silently
    regress the duck to a flat palette."""
    scene = V.extract_scene()
    assert scene["geoms"], "no visual geoms extracted"
    for g in scene["geoms"]:
        assert isinstance(g["mat"], str)
        rgba = g["rgba"]
        assert len(rgba) == 4
        assert all(0.0 <= v <= 1.0 for v in rgba)
    mats = {g["mat"] for g in scene["geoms"]}
    # Spot-check parts the viewer art-directs by name (a rename upstream
    # would orphan the overrides).
    assert {"noenoeil_material", "jaw_material", "sole_left_material",
            "top_head_shell_material"} <= mats


def test_preview_ducks_get_deployment_steering_and_drive(fake_popen):
    """The viewer must show DEPLOYED behavior: the policy is compass-blind, so
    an unsteered preview drifts into circles while steered videos run straight
    — the discrepancy the user caught twice. And locomotion trainees must get
    the drive command; zeroing it commanded the run trainee to STAND in the
    preview while the trainer practiced running."""
    duck = V.Duck("d0", "test", V._zero_infer, seed=0)
    # steer: forward drive with no explicit turn -> wz closes on measured yaw
    import mujoco
    yaw = 0.5
    duck.env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(duck.env.model, duck.env.data)
    duck.set_cmd(np.array([0.3, 0.0, 0.0], np.float32))
    # First straight tick ANCHORS to the current heading (no spin-back to
    # world zero — that whip knocked ducks over after turn segments)…
    assert duck.env.twist_cmd[2] == pytest.approx(0.0, abs=1e-6)
    # …then drifting off that anchor steers back.
    yaw2 = yaw + 0.3
    duck.env.data.qpos[3:7] = [np.cos(yaw2 / 2), 0.0, 0.0, np.sin(yaw2 / 2)]
    mujoco.mj_forward(duck.env.model, duck.env.data)
    duck.set_cmd(np.array([0.3, 0.0, 0.0], np.float32))
    # -4 * 0.3 = -1.2, clipped to the ±1.0 turn-rate limit
    assert duck.env.twist_cmd[2] == pytest.approx(-1.0, abs=0.05)
    # an explicit turn command wins over the hold
    duck.set_cmd(np.array([0.3, 0.0, 0.8], np.float32))
    assert duck.env.twist_cmd[2] == pytest.approx(0.8)
    # zero drive -> no steering injected
    duck.set_cmd(np.zeros(3, np.float32))
    assert duck.env.twist_cmd[2] == pytest.approx(0.0)


def test_locomotion_teach_runs_are_not_trick_ducks(fake_popen, tmp_path, monkeypatch):
    """A dragged-in RUN policy must receive drive commands. Name-based
    classification called every teach-* run a trick, so the user's dragged
    runner stood at cmd (0,0,0) — 'how come I don't see it move'."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path)
    for name, behavior, trick in (("teach-run-x", "run", False),
                                  ("teach-backflip-y", "backflip", True)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "behavior.json").write_text(
            json.dumps({"behavior": behavior}))
        d = types.SimpleNamespace(id="d3", policy_id=f"run:{name}")
        assert V.is_trick_duck(d) is trick, name
    # unknown run dir: conservative old rule
    d = types.SimpleNamespace(id="d3", policy_id="run:teach-mystery")
    assert V.is_trick_duck(d) is True


def test_handoff_waits_for_the_spin_to_brake(fake_popen, monkeypatch):
    """Handing off to alpha_stand while still spinning made the stand policy
    absorb the leftover momentum by pivoting — the user's 'it lands then
    turns to the side'. alpha_stand cannot fix heading (yaw unobservable), so
    the flip policy must brake before it lets go of the wheel."""
    monkeypatch.setattr(V.psutil, "Process",
                        lambda pid: types.SimpleNamespace(children=lambda recursive: []))
    duck = V.Duck("d0", "bf", V._zero_infer, seed=0,
                  env_kwargs={"behavior_id": "backflip"})
    duck.env._bf_rot = 6.2
    duck.env.foot_contact_state = {"left": True, "right": True}
    duck.env.data.sensordata[duck.env.gyro_adr] = (0.0, -2.0, 0.0)  # still rolling
    assert not duck._handoff_due(), "handed off mid-spin"
    duck.env.data.sensordata[duck.env.gyro_adr] = (0.1, 0.2, 0.1)   # braked
    assert duck._handoff_due()
