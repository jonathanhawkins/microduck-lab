"""World mode of the lab (roadmap 0.4): scenario CRUD with built-ins read-only,
loading a world, the /ws/sim frame shape with ToF payloads, drive and reset
over the socket, and the same front-door origin rule as /ws."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from microduck_local import contract as C
from microduck_local import viz_server as V
from microduck_local import world_server as W
from microduck_local.sensors.detector import DetectorSpec
from microduck_local.world.compose import scene_model

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")

ORIGIN = {"origin": "http://localhost:63317"}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("MICRODUCK_SCENARIOS_DIR", str(tmp_path / "scenarios"))
    monkeypatch.setenv("MICRODUCK_RECORDINGS_DIR", str(tmp_path / "recordings"))
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return V.make_app([])


def test_builtin_scenarios_validate_and_list(app):
    for sc in W.builtin_scenarios().values():
        assert W.validate_scenario(sc.to_dict()) == sc
    with TestClient(app) as c:
        names = [s["name"] for s in c.get("/scenarios").json()["scenarios"]]
        assert {"empty-floor", "wall-test", "living-room"} <= set(names)
        got = c.get("/scenarios/living-room").json()
        assert got["name"] == "living-room" and len(got["walls"]) == 4 and got["balls"]
        assert c.get("/scenarios/nope").status_code == 404
        assert c.get("/scenarios/..%2Fetc").status_code in (400, 404)


def test_user_scenarios_save_validate_delete(app, tmp_path):
    with TestClient(app) as c:
        raw = c.get("/scenarios/wall-test").json()
        raw["ducks"].append({"id": "d1", "spawn": [0.0, 0.5, 0.0], "policy": None, "tof": "hostile"})
        r = c.put("/scenarios/my-room", json=raw)
        assert r.status_code == 200 and r.json()["name"] == "my-room"
        assert (tmp_path / "scenarios" / "my-room.json").exists()
        listed = {s["name"]: s for s in c.get("/scenarios").json()["scenarios"]}
        assert listed["my-room"]["builtin"] is False and listed["my-room"]["ducks"] == 2
        # Built-ins are read-only; bad content is refused loudly.
        assert c.put("/scenarios/wall-test", json=raw).status_code == 409
        raw["ducks"][0]["tof"] = "lidar"
        assert c.put("/scenarios/my-room", json=raw).status_code == 422
        assert c.delete("/scenarios/my-room").status_code == 200
        assert c.delete("/scenarios/my-room").status_code == 404
        assert c.delete("/scenarios/wall-test").status_code == 409


def test_pitch_2v2_builtin_reports_formation_roles(app):
    """GET /world after loading the lab's `pitch-2v2` carries the stamped
    jobs — the page's inspector and the brains both read `duck_info.role`."""
    with TestClient(app) as c:
        r = c.post("/world/load", json={"scenario": "pitch-2v2"})
        assert r.status_code == 200, r.text
        ducks = {d["id"]: d for d in r.json()["ducks"]}
        assert ducks["d0"]["role"] == "defender" and ducks["d1"]["role"] == "striker"
        assert ducks["d2"]["role"] == "defender" and ducks["d3"]["role"] == "striker"
        world = c.get("/world").json()
        assert {d["id"]: d["role"] for d in world["ducks"]} == {
            "d0": "defender", "d1": "striker", "d2": "defender", "d3": "striker"}


def test_load_world_and_stream_frames(app):
    with TestClient(app) as c:
        assert c.get("/world").json()["scenario"] is None
        r = c.post("/world/load", json={"scenario": "wall-test"})
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["scenario"]["name"] == "wall-test"
        assert [d["id"] for d in info["ducks"]] == ["d0"] and info["ducks"][0]["tof"] == "ideal"
        assert c.post("/world/load", json={"scenario": "nope"}).status_code == 404

        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            # Let a few ticks pass so the ToF has sampled.
            frame = None
            for _ in range(6):
                frame = ws.receive_json()
            assert frame["scenario"] == "wall-test" and frame["tick"] > 0
            d = frame["ducks"][0]
            # One pose per body of the scene the viewer draws from, world
            # first - a count, not a constant, because `split_jaw` adds the
            # `mouth` body to both models at once.
            assert len(d["bodies"]) == scene_model().nbody
            assert d["bodies"][0] == [0, 0, 0, 1, 0, 0, 0]
            tof = d["sensors"]["tof"]
            assert len(tof["mm"]) == 64 and "pts" not in tof
            # The wall is a metre ahead: the middle of the frame reports ~0.94 m.
            mid = tof["mm"][3 * 8 + 3]
            assert 900 < mid < 960
            assert frame["mode"] == "auto" and len(frame["cmd"]) == 3
            # A duck with a ToF drives itself in auto mode.
            assert d["brain"]["kind"] == "wander" and d["brain"]["state"] in ("cruise", "steer", "spin", "blind", "unstick")
            # Drive and reset go through the socket.
            ws.send_text(json.dumps({"cmd": [0.2, 0.0, 0.0]}))
            for _ in range(4):
                frame = ws.receive_json()
            assert frame["mode"] == "manual" and frame["cmd"][0] == 0.2
            assert frame["ducks"][0]["cmdSpeed"] == 0.2
            assert frame["ducks"][0]["brain"]["kind"] == "manual"
            ws.send_text(json.dumps({"noise": {"duck": "d0", "preset": "hostile"}}))
            for _ in range(3):
                frame = ws.receive_json()
            assert frame["ducks"][0]["tof"] == "hostile"
            ws.send_text(json.dumps({"reset": True}))
            for _ in range(2):
                frame = ws.receive_json()
            assert frame["ducks"][0]["step"] < 5
        assert c.get("/world").json()["ducks"][0]["tof"] == "hostile"
        r = c.post("/world/noise", json={"duck": "d0", "preset": "ideal"})
        assert r.status_code == 200 and r.json()["tof"] == "ideal"
        assert c.post("/world/noise", json={"duck": "zz", "preset": "ideal"}).status_code == 404
        assert c.post("/world/noise", json={"duck": "d0", "preset": "x"}).status_code == 422


def test_sim_socket_rejects_foreign_origins(app):
    with TestClient(app) as c:
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/sim", headers={"origin": "http://evil.example"}) as ws:
                ws.receive_json()
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            f = ws.receive_json()
            assert f["ducks"] == [] and f["scenario"] is None   # nothing loaded, still alive


def test_ring_records_without_a_client_and_saves_a_recording(app, tmp_path):
    import gzip
    import time
    with TestClient(app) as c:
        assert c.post("/replay/save", json={"name": "x"}).status_code == 409   # nothing yet
        c.post("/world/load", json={"scenario": "wall-test"})
        time.sleep(0.5)                                     # no socket attached: the ring still fills
        ring = c.get("/replay/ring?last=5").json()
        assert 1 <= ring["count"] <= 5 and len(ring["frames"]) == ring["count"]
        f = ring["frames"][-1]
        assert f["scenario"] == "wall-test" and f["ducks"][0]["id"] == "d0"
        assert c.post("/replay/save", json={"name": "bad name"}).status_code == 400
        h = c.post("/replay/save", json={"name": "take1"}).json()
        assert h["frames"] >= 1 and h["scenario"] == "wall-test"
        p = tmp_path / "recordings" / "take1.jsonl.gz"
        with gzip.open(p, "rt") as fh:
            lines = fh.read().splitlines()
        assert len(lines) == h["frames"] + 1
        assert [r["name"] for r in c.get("/recordings").json()["recordings"]] == ["take1"]
        rec = c.get("/recordings/take1").json()
        assert rec["header"]["name"] == "take1" and len(rec["frames"]) == h["frames"]
        assert rec["frames"][0]["tick"] <= rec["frames"][-1]["tick"]
        assert c.delete("/recordings/take1").status_code == 200
        assert c.get("/recordings/take1").status_code == 404


def test_follow_me_scenario_persons_brains_and_possess(app):
    with TestClient(app) as c:
        info = c.post("/world/load", json={"scenario": "follow-me"}).json()
        assert info["ducks"][0]["detector"] == "datasheet" and "follow" in info["brains"]
        # The scene starts on the SHIPPED follower, so the page's "what the
        # brain sees" panel has something to draw the moment it loads.
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            frame = None
            for _ in range(8):
                frame = ws.receive_json()
            d = frame["ducks"][0]
            assert d["brain"]["kind"] == "learned:follow-v4" and "inputs" in d["brain"]
            assert len(d["brain"]["view"]["obs"]) == 80 and len(d["brain"]["view"]["act"]["clipped"]) == 3
            assert d["brain"]["inputs"]["det"]["max"] > 0 and d["headApplied"] is False
            # The frame carries the detector's output for the page's rays and
            # camera inset: the frustum and each detection's three numbers.
            det = d["sensors"]["det"]
            # From the spec, not a constant: this asserts the payload REPORTS the
            # camera. Hardcoding 62/48 asserted WHICH camera as a side effect, and
            # broke the day the default moved to the fitted 116° × 60° module.
            spec = DetectorSpec()
            assert det["fov"] == [spec.fov_h_deg, spec.fov_v_deg] and det["age"] >= 0
            assert all({"cls", "bearing", "elevation", "width", "range"} <= set(it) for it in det["items"])
            persons = [o for o in frame["objects"] if o["kind"] == "person"]
            assert persons and persons[0]["id"] == "p0" and persons[0]["possessed"] is False
            assert frame["possessed"] is None
            # Possess the person: the manual command drives IT, the duck keeps its brain.
            ws.send_text(json.dumps({"possess": "p0"}))
            ws.send_text(json.dumps({"cmd": [0.4, 0.0, 0.0]}))
            for _ in range(6):
                frame = ws.receive_json()
            assert frame["possessed"] == "p0" and frame["mode"] == "manual"
            assert frame["ducks"][0]["brain"]["kind"] == "learned:follow-v4"
            ws.send_text(json.dumps({"brain": {"duck": "d0", "kind": "wander"}}))
            ws.send_text(json.dumps({"possess": None}))
            ws.send_text(json.dumps({"noise": {"duck": "d0", "preset": "hostile", "sensor": "det"}}))
            for _ in range(4):
                frame = ws.receive_json()
            # Released: the manual command (still held) steers the ducks; the
            # brain behind it is now wander.
            assert frame["possessed"] is None and frame["ducks"][0]["brain"]["kind"] == "manual"
            assert frame["ducks"][0]["brainKind"] == "wander"
            assert frame["ducks"][0]["detector"] == "hostile"
        assert c.post("/world/brain", json={"duck": "d0", "kind": "nope"}).status_code == 422
        assert c.post("/world/brain", json={"duck": "d0", "kind": "follow"}).status_code == 200


def test_playroom_scenario_streams_toys_basket_and_tidy_state(app):
    """Track 12 on the wire: the playroom built-in loads with the tidy brain,
    frames carry the toys (with their in-basket flag), the basket, the tidy
    score and the duck's beak/holding/skill state, and the brain's head
    intents are applied (`wants_head`)."""
    with TestClient(app) as c:
        info = c.post("/world/load", json={"scenario": "playroom"}).json()
        assert info["scenario"]["ducks"][0]["brain"] == "tidy" if "scenario" in info else "tidy" in info["brains"]
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            frame = None
            for _ in range(8):
                frame = ws.receive_json()
            assert frame["tidy"] == {"total": 6, "inBasket": 0, "held": []}
            toys = [o for o in frame["objects"] if o.get("toy")]
            assert len(toys) == 6 and all(o["inBasket"] is False and o["held"] is None for o in toys)
            d = frame["ducks"][0]
            assert d["brainKind"] == "tidy" and d["brain"]["kind"] == "tidy" and d["headApplied"] is True
            assert d["holding"] is None and d["skill"] is None and d["beak"] == "open"
            assert d["brain"]["inputs"]["tidy"]["picked"] == 0


def test_tether_latency_delays_intents_and_maps_stream(app):
    """12.10: with a tether the intent applied now is the one decided
    tether_ms ago; frames say so, and occupancy maps ride every 12th frame."""
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "playroom"})
        assert c.post("/world/tether", json={"ms": 250}).json() == {"tetherMs": 250.0}
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            maps_seen = 0
            frame = None
            for _ in range(30):
                frame = ws.receive_json()
                maps_seen += frame["maps"] is not None
            assert frame["tetherMs"] == 250.0 and maps_seen >= 2
            m = next(iter(frame["maps"].values())) if frame["maps"] else None
            if m is not None:
                assert m["nx"] * m["ny"] == len(m["cells"]) and set(m["cells"]) <= set("012")
            ws.send_text(json.dumps({"tether": 0}))
            for _ in range(4):
                frame = ws.receive_json()
            assert frame["tetherMs"] == 0.0


def test_a_pitch_streams_the_metrics_the_benchmark_judges_by():
    """Goals are ~2.5 a run and resolve nothing, so the page shows what
    eval-pitch actually judges by — the same PitchMetrics class, ticked on
    the server's own step, per team and per minute. A world with no goals
    (the playroom) carries none of it."""
    from microduck_local.world_server import WorldState
    st = WorldState(None)
    st.preload("pitch")
    assert st.metrics is not None
    for _ in range(20):
        st.world.step()
        st.metrics.tick()
    import numpy as np
    soc = st.frame(np.zeros(3), "auto")["soccer"]
    assert soc is not None and set(soc) >= {"left", "right", "kicked", "bumped",
                                            "ballAdvance", "ballProgress", "possession"}
    from microduck_local.world.scenario import PITCH_TEAMS
    assert set(soc["possession"]) == set(PITCH_TEAMS)           # per TEAM (a colorway), as the battery reports it
    assert set(soc["ownGoals"]) == set(PITCH_TEAMS) and soc["goalsUnattributed"] == 0
    assert all(isinstance(v, (int, float)) for v in soc["ballAdvance"].values())
    tidy = WorldState(None)
    tidy.preload("playroom")
    assert tidy.metrics is None and tidy.frame(np.zeros(3), "auto")["soccer"] is None


def test_speed_multiplier_moves_sim_time_per_frame_not_the_frame_rate(app):
    """The speed knob buys SIM TIME, not bandwidth: the wire stays at 25 Hz
    of wall time and each frame carries `SEND_EVERY x speed` more world
    ticks than the last. That arithmetic is exact — the accumulator spends a
    whole tick or none — so this asserts tick deltas, not wall timings, and
    a slow CI box only makes the frames arrive later, never wrong."""
    def deltas(ws, n=6):
        ticks = [ws.receive_json()["tick"] for _ in range(n)]
        return [b - a for a, b in zip(ticks, ticks[1:])]

    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "empty-floor"})
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            assert set(deltas(ws)) == {W.SEND_EVERY}
            assert c.post("/world/speed", json={"x": 4}).json() == {"simSpeed": 4.0}
            for _ in range(4):                       # let the change reach the loop
                ws.receive_json()
            # The SUM over the window, not set-equality per frame: the loop
            # may cut a batch short on STEP_BUDGET_S and carry the unspent
            # credit into the next tick, so a loaded box legitimately shows
            # 6 then 10 where an idle one shows 8 and 8. Uniformity was never
            # promised; the total is.
            d4 = deltas(ws)
            assert sum(d4) == W.SEND_EVERY * 4 * len(d4), d4
            frame = ws.receive_json()
            assert frame["simSpeed"] == 4.0
            # Slow motion is the same accumulator from the other end: half
            # the sim ticks per frame, still one frame per two wall ticks.
            ws.send_text(json.dumps({"speed": 0.5}))
            for _ in range(4):
                ws.receive_json()
            assert set(deltas(ws)) == {W.SEND_EVERY // 2}
            assert ws.receive_json()["simSpeed"] == 0.5


def test_speed_is_clamped_and_a_bad_value_is_refused_not_fatal(app):
    """Out-of-range is clamped rather than rejected (the page's presets are
    a subset of what the loop runs), and junk over the socket is an event,
    not a dead world loop."""
    assert (W.clamp_speed(1e6), W.clamp_speed(0), W.clamp_speed(-3)) == (
        W.SPEED_MAX, W.SPEED_MIN, W.SPEED_MIN)
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "empty-floor"})
        assert c.post("/world/speed", json={"x": 1000}).json() == {"simSpeed": W.SPEED_MAX}
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            ws.send_text(json.dumps({"speed": "fast"}))
            # `events` is cleared on every send, so the complaint lives in
            # exactly one frame — collect them all rather than the last.
            frames = [ws.receive_json() for _ in range(8)]
            said = [e for f in frames for e in f["events"]]
            assert any("speed ignored" in e for e in said), said
            assert frames[-1]["simSpeed"] == W.SPEED_MAX          # unchanged…
            assert frames[-1]["tick"] > frames[0]["tick"]         # …and still stepping
            # A repeat of the speed it already runs at says nothing at all.
            ws.send_text(json.dumps({"speed": W.SPEED_MAX}))
            again = [ws.receive_json() for _ in range(8)]
            assert not [e for f in again for e in f["events"] if e.startswith("speed ")]


def test_speed_changes_the_wall_clock_and_nothing_about_the_world():
    """The whole promise of the knob: 4x is the SAME simulation, watched
    sooner. Everything inside runs off `World.t`, so the only thing the
    batching could get wrong is the demo script, which the loop therefore
    samples per STEP — sampling it per wall tick would hand all four steps of
    a 4x batch the command belonging to the first.

    `per_step_cmd` is what makes this a test rather than a tautology. With it
    False the helper reproduces the WRONG sampling, and the last assertion
    demands that the wrong sampling actually diverges — so if someone moves
    `current_cmd` back out of the loop in world_server.py, this test goes red
    instead of quietly passing on a helper that no longer varies anything.

    Ducks are put on the `script` brain (the only brain a `cmd` reaches) and
    the run straddles the DEMO_SCRIPT's 4.0 s segment boundary, which is the
    one place the two samplings can differ. Commands, not physics, are the
    sensitive readout: it needs no policies, so this runs in a checkout that
    has only the MJCF."""
    import numpy as np

    from microduck_local.world_server import TICK_HZ, WorldState

    def run(per_batch: int, *, per_step_cmd: bool = True,
            ticks: int = 100) -> tuple[np.ndarray, np.ndarray]:
        st = WorldState(None)
        st.preload("empty-floor")
        for d in st.world.ducks:
            st.set_brain(d, "script")     # the script is what `cmd` drives
        st.script_t = 3.9                 # …and 4.0 s is where it changes its mind
        sent, done = [], 0
        while done < ticks:
            cmd, mode = st.current_cmd(0.0)            # once per BATCH
            for _ in range(min(per_batch, ticks - done)):
                if per_step_cmd:
                    cmd, mode = st.current_cmd(0.0)    # …the loop's way: per STEP
                st.script_t += 1.0 / TICK_HZ
                st.drive(cmd, mode)
                st.world.step()
                st.after_step()
                sent.append(np.concatenate([d.twist_cmd for d in st.world.ducks.values()]))
                done += 1
        return np.array(sent), st.world.data.qpos.copy()

    one_cmds, one_qpos = run(1)           # as the loop runs at 1x
    four_cmds, four_qpos = run(4)         # …and as it runs at 4x
    assert np.array_equal(one_cmds, four_cmds), int((one_cmds != four_cmds).any(axis=1).sum())
    assert np.array_equal(one_qpos, four_qpos), float(np.abs(one_qpos - four_qpos).max())

    # The control: the sampling the loop deliberately does NOT do. If this
    # comes out equal too, the comparison above is measuring nothing.
    wrong = run(4, per_step_cmd=False)[0]
    assert not np.array_equal(run(1, per_step_cmd=False)[0], wrong), \
        "the per-wall-tick sampling must diverge, or this test cannot fail"


def test_the_running_loop_samples_the_drive_command_once_per_sim_step(app):
    """The test above pins the ARITHMETIC by re-implementing the batch; this
    one pins the loop that ships. It counts `current_cmd` calls against world
    ticks in the live `world_loop` at 4x, where the two samplings are four
    times apart: per step gives at least one call per tick, per wall tick
    gives about a quarter of one. Move the call back out of the batch in
    world_server.py and this goes red, which the re-implementation cannot."""
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "empty-floor"})
        st = c.app.state.world
        calls = []
        real = st.current_cmd
        st.current_cmd = lambda now: (calls.append(now), real(now))[1]   # noqa: E731
        try:
            assert c.post("/world/speed", json={"x": 4}).json() == {"simSpeed": 4.0}
            with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
                for _ in range(6):
                    ws.receive_json()
                t0, n0 = ws.receive_json()["tick"], len(calls)
                for _ in range(12):
                    frame = ws.receive_json()
                steps, sampled = frame["tick"] - t0, len(calls) - n0
        finally:
            del st.current_cmd
    assert steps > 0
    per_step = sampled / steps
    assert per_step >= 1.0, f"{sampled} samples for {steps} sim steps ({per_step:.2f}/step)"


def test_resending_the_same_speed_does_not_throw_away_banked_sub_tick_credit():
    """The freeze that shipped for a day: `set_speed` zeroed the accumulator
    on EVERY call. At 0.25x the credit needs four wall ticks to buy one step,
    and a held `[` re-sends the same speed every ~30 ms, so the credit never
    reached 1.0 and the world stopped dead while frames kept streaming. A
    no-op must stay a no-op; a real change still starts clean."""
    from microduck_local.world_server import WorldState

    st = WorldState(None)
    st.set_speed(0.25)
    st._step_credit = 0.75                  # three wall ticks of credit banked
    assert st.set_speed(0.25) == 0.25       # the same speed again — a held key
    assert st._step_credit == 0.75, "a no-op speed must not spend the credit"
    st.set_speed(1.0)                       # …but a real change starts clean
    assert st._step_credit == 0.0


def test_a_world_kept_at_the_same_slow_speed_keeps_stepping(app):
    """The end-to-end half of the above, on the running loop: hammer the
    endpoint with the speed it already runs at and the world must still
    advance. Asserts only that it moves at all — the rate is wall-clock
    dependent, the freeze was not."""
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "empty-floor"})
        c.post("/world/speed", json={"x": 0.25})
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(8):
                ws.receive_json()
            t0 = ws.receive_json()["tick"]
            last = t0
            for _ in range(60):             # ~2.4 s of frames, re-asking throughout
                c.post("/world/speed", json={"x": 0.25})
                last = ws.receive_json()["tick"]
    assert last > t0, f"world frozen at 0.25x while the same speed was re-sent ({last} == {t0})"


def test_a_speed_change_drops_the_rtf_window_it_straddles(app):
    """`rtf` is measured over a wall second. One that spans a speed change
    measures neither speed, and the page reads a stale low `rtf` against the
    new speed as a shortfall — amber on every single speed-up. So a change
    zeroes it, and 0 is what the page reads as \"no measurement yet\"."""
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "empty-floor"})
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(70):                    # let a window close at 1x
                frame = ws.receive_json()
            assert frame["rtf"] > 0.5, frame["rtf"]
            c.post("/world/speed", json={"x": 4})
            frame = next(ws.receive_json() for _ in range(1))
            for _ in range(3):
                frame = ws.receive_json()
            assert frame["simSpeed"] == 4.0 and frame["rtf"] == 0.0, frame["rtf"]


def test_junk_speeds_are_refused_by_both_doors_and_neither_dies():
    """NaN is the reason `clamp_speed` refuses instead of clamping: every
    comparison against it is False, so min/max quietly returned SPEED_MIN and
    a bad value read back as a deliberate request for quarter speed. A
    400-digit int is the same class of mistake arriving as OverflowError."""
    import math

    from microduck_local.world_server import SPEED_MAX, SPEED_MIN, clamp_speed

    assert (clamp_speed(1e6), clamp_speed(0), clamp_speed(-3)) == (SPEED_MAX, SPEED_MIN, SPEED_MIN)
    for bad in (math.nan, math.inf, -math.inf, 10 ** 400, "fast", None):
        with pytest.raises((ValueError, TypeError)):
            clamp_speed(bad)


def test_a_reset_drops_everything_keyed_to_the_clock_it_just_restarted(app):
    """World.t goes back to zero on R, and three things used to survive it:

    - PitchMetrics, whose row() scales by 60/w.t — one frame later a carried
      12 s of possession printed as ~7e11 per minute against a 0-0 board.
    - the tether queue, holding senses/intents stamped in the OLD clock: due
      hundreds of seconds ahead, so nothing popped and every brain kept being
      handed the pre-reset frame (whose negative age reads as FRESH).
    - the replay ring, which made /replay/save write two runs under one
      header with t jumping backwards in the middle.
    """
    with TestClient(app) as c:
        c.post("/world/load", json={"scenario": "pitch-2v2"})
        c.post("/world/tether", json={"ms": 200})
        st = c.app.state.world
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(40):
                ws.receive_json()
            before = st.metrics
            assert before is not None and len(st.ring) > 10
            # Identity, not emptiness: `drive()` re-creates a Tether for each
            # duck on the very next tick, so the queue refills at once. What
            # must not survive is the OLD one, holding senses stamped hundreds
            # of seconds ahead of the clock that just restarted.
            stale = dict(st._tether_queue)
            assert stale, "the tether should have queued something to drop"
            ws.send_text(json.dumps({"reset": True}))
            for _ in range(6):
                ws.receive_json()
            assert st.metrics is not before, "metrics must be rebuilt on the new clock"
            assert all(st._tether_queue.get(k) is not v for k, v in stale.items()), \
                "old-clock senses/intents must go"
            ts = [json.loads(f)["t"] for f in st.ring]
            assert all(b >= a for a, b in zip(ts, ts[1:])), "the ring spans two runs"


def test_the_manual_hold_costs_the_same_SIM_time_at_every_speed(app):
    """`drive()` skips brain.step() outright while a manual command holds, so
    the hold is not just "your twist persists" — it is "no brain runs". Flat
    on the wall that was 6 sim seconds at 1x and 48 at 8x: 16% of a 300 s
    pitch run with every brain suspended, and a 48 s jump in senses.t handed
    to each one on resume."""
    from microduck_local.world_server import OVERRIDE_HOLD_S, SPEED_CHOICES, WorldState

    st = WorldState(None)
    for x in SPEED_CHOICES:
        st.speed = x
        wall = st.override_hold_s()
        # BOTH bounds, and as equalities where each is the binding one — an
        # upper bound alone passed just as happily on a hold a quarter the
        # documented length, so it could not tell the two apart.
        assert wall <= OVERRIDE_HOLD_S + 1e-9, f"{x}x holds {wall}s of wall"
        assert wall * x <= OVERRIDE_HOLD_S + 1e-9, f"{x}x costs {wall * x}s of sim"
        # Whichever bound is the tight one equals OVERRIDE_HOLD_S exactly:
        # above 1x that is the SIM cost (wall shrinks as 6/x), below 1x it is
        # the WALL time (sim shrinks as 6x).
        binding = wall * x if x >= 1.0 else wall
        assert binding == pytest.approx(OVERRIDE_HOLD_S)
    st.speed = 1.0
    assert st.override_hold_s() == OVERRIDE_HOLD_S      # unchanged where it was tuned

    # A speed change re-prices a hold already running, or the bound above is
    # only true for whatever speed happened to be set when the key went down.
    now = time.monotonic()
    st.override_until = now + OVERRIDE_HOLD_S
    st.set_speed(8.0)
    assert (st.override_until - time.monotonic()) * 8.0 <= OVERRIDE_HOLD_S + 0.1


def test_the_team_boards_the_brains_hold_are_the_ones_the_world_state_keeps():
    """The regression this exists for: `build` filled the boards through
    `make_brain` and then re-initialised `self.teams` a few lines later, so
    every pitch ran with brains holding live `Team` objects that `WorldState`
    no longer referenced. `after_step` passes `self.teams` to
    `kickoff_brains`/`throw_in_brains`, whose `for tm in teams.values()`
    then iterated nothing: no board reset after a goal, no kickoff stand-off,
    no throw-in belief drop. Nothing caught it — `eval-pitch` builds its own
    teams dict, and the whole suite was green.

    So assert IDENTITY, not merely that the dict is non-empty: the object the
    duck plays on has to be the object the server can reach."""
    from microduck_local.world_server import WorldState

    st = WorldState(None)
    st.preload("pitch-2v2")
    assert st.teams, "the pitch built no team boards at all"
    for did, brain in st.brains.items():
        board = getattr(brain, "team", None)
        if board is None:
            continue                      # not every brain plays for a side
        assert board is st.teams.get(board.name), (
            f"{did} plays on a board WorldState cannot reach: "
            f"{board.name} -> {st.teams.get(board.name)!r}")
    # …and both sides are represented, so a one-sided dict cannot pass either.
    assert sorted(st.teams) == ["cream", "graphite"]


def test_a_reset_clears_the_team_boards_and_resyncs_the_sequence_counters():
    """Deadlines on a board outlive `World.reset()` unless someone drops
    them, and `World.reset()` zeroes `ball_out_seq` but not `goal_seq` — a
    stale counter fires a phantom throw-in or kickoff on the first tick of
    the new run."""
    from microduck_local.world_server import WorldState

    st = WorldState(None)
    st.preload("pitch-2v2")
    board = next(iter(st.teams.values()))
    # The CONCEDING side stands off until `until`; a goal at t=120 sets ~133.
    board.kickoff(ours=False, until=133.0, ball=(0.0, 0.0))
    assert board.waits(0.0), "the board should be standing off before the reset"
    st.out_seq, st.goal_seq = 99, 99      # as if a throw-in and a goal had happened
    st.world.reset()                      # …and now the clock goes back to zero
    st.restart()
    assert not board.waits(0.0), "a board still holds a deadline from the old clock"
    assert (st.goal_seq, st.out_seq) == (st.world.goal_seq, st.world.ball_out_seq)
