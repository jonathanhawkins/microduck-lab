"""World mode of the lab (roadmap 0.4): scenario CRUD with built-ins read-only,
loading a world, the /ws/sim frame shape with ToF payloads, drive and reset
over the socket, and the same front-door origin rule as /ws."""

import json
import math
import time
from pathlib import Path

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


def test_the_listing_says_which_BODIES_a_room_holds(app):
    """`ducks` is a total; `robots` is what the room actually contains.

    The /sim picker read the total and called every entry a duck, so
    `mars-follow` — one MARS and no duck at all — announced itself as
    "1 ducks". The breakdown carries the lab's own noun for each body so a
    menu does not need a table of its own, and it is sorted commonest first
    so the headline body leads.
    """
    with TestClient(app) as c:
        rows = {s["name"]: s for s in c.get("/scenarios").json()["scenarios"]}
        duck_room = rows["living-room"]
        assert duck_room["robots"] == [
            {"id": "microduck", "n": duck_room["ducks"], "noun": "duck"}]
        for row in rows.values():
            assert sum(r["n"] for r in row["robots"]) == row["ducks"]
            counts = [r["n"] for r in row["robots"]]
            assert counts == sorted(counts, reverse=True), row["name"]
            assert all(r["noun"] for r in row["robots"]), row["name"]


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_a_room_of_MARS_is_not_a_room_of_ducks(app, tmp_path):
    """The case the bug was reported on, end to end through the endpoint."""
    with TestClient(app) as c:
        raw = c.get("/scenarios/wall-test").json()
        raw["ducks"][0]["robot"] = "mars"
        raw["ducks"][0]["policy"] = None
        assert c.put("/scenarios/one-mars", json=raw).status_code == 200
        row = {s["name"]: s for s in c.get("/scenarios").json()["scenarios"]}["one-mars"]
        assert row["ducks"] == 1, "the total still counts the entry"
        assert row["robots"] == [{"id": "mars", "n": 1, "noun": "MARS"}]


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


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_a_mars_entrys_frame_block_says_what_it_is_and_what_it_senses(app, tmp_path):
    """The /ws/sim row for a driver-stepped body (`world_server`'s docstring).

    Four things the viewer needs and one it must NOT be given: `robot` so it
    can pick the mesh set, `bodies` in THAT robot's scene order, `steerable`
    so WASD still reaches it, `sensors.lidar` to draw the scan — and no
    `sensors.tof`, because the 8x8 its brains read is adapted from the scan
    and drawing it would put a cone on a head that carries no such sensor.
    """
    from microduck_local.robots import mars

    with TestClient(app) as c:
        # The tracked scenario, through the same door the editor saves by, so
        # this is the file on disk and not a fixture that resembles it.
        raw = json.loads((Path(W.__file__).resolve().parents[2]
                          / "scenarios" / "mars-playroom.json").read_text())
        assert c.put("/scenarios/mars-room", json=raw).status_code == 200
        r = c.post("/world/load", json={"scenario": "mars-room"})
        assert r.status_code == 200, r.text
        info = r.json()["ducks"][0]
        assert info["robot"] == "mars" and info["steerable"] is True
        assert info["tof"] is None and info["lidar"] == "datasheet"
        assert info["falls"] == 0 and info["wallBumps"] == 0

        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            frame = None
            for _ in range(14):          # 6 Hz: a couple of scans in
                frame = ws.receive_json()
            d = frame["ducks"][0]
            assert d["robot"] == "mars" and d["steerable"] is True
            # One pose per body of `GET /scene?robot=mars`, world first — a
            # count off the body's own dump, not a constant.
            assert len(d["bodies"]) == len(mars.visual_scene()["bodies"])
            assert d["bodies"][0] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            lidar = d["sensors"]["lidar"]
            assert len(lidar["mm"]) == 360 and lidar["maxRange"] == 6.0
            assert lidar["da"] == pytest.approx(2 * math.pi / 360, abs=1e-5)
            assert lidar["mount"][0] == pytest.approx(-0.0764, abs=0.001)
            assert 0.0 <= lidar["age"] <= 1 / 6.0 + 0.05
            assert max(lidar["mm"]) > 500                     # it can see the room
            assert "tof" not in d["sensors"], "a wheeled body ships no ToF block"
            # A planar base cannot topple, so the fall count never moves…
            assert d["falls"] == 0
            # …and the gaze is always applied: there is no walker observation
            # for a head pose to disturb (`world_server.head_applied`).
            assert d["headApplied"] is True
            # WASD reaches it through the same command as a duck.
            ws.send_text(json.dumps({"cmd": [0.3, 0.0, 0.0]}))
            for _ in range(6):
                frame = ws.receive_json()
            assert frame["mode"] == "manual"
            assert frame["ducks"][0]["cmdSpeed"] == pytest.approx(0.3)
            assert frame["ducks"][0]["speed"] > 0.1, "the base actually moved"


# -- the arm channels, and the scan's own two numbers -------------------------
#
# The /sim inspector renders ONE BLOCK PER SENSE CHANNEL the frame carries, so
# every case below is about a key being present, absent, or carrying the number
# the instrument is drawn against. Each was shown to FAIL on a planted break
# before it was kept (`AGENTS.md`: "A/B new tests against planted
# regressions"); the table is in the report.

MARS_SCENARIO = "mars-room"


def load_mars(c) -> dict:
    """Put the tracked `mars-playroom` in the world, through the same door the
    editor saves by — so these cases read the file on disk and not a fixture
    that resembles it. Returns `POST /world/load`'s payload."""
    raw = json.loads((Path(W.__file__).resolve().parents[2]
                      / "scenarios" / "mars-playroom.json").read_text())
    assert c.put(f"/scenarios/{MARS_SCENARIO}", json=raw).status_code == 200
    r = c.post("/world/load", json={"scenario": MARS_SCENARIO})
    assert r.status_code == 200, r.text
    return r.json()


def mars_frame(c, ticks: int = 14) -> dict:
    """One /ws/sim frame with a couple of 6 Hz scans in it."""
    with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
        frame = None
        for _ in range(ticks):
            frame = ws.receive_json()
    return frame


def leaf_shape(v, path: str = "") -> list[str]:
    """Every leaf of a payload as `dotted.path:type`, lists as `path[n]:type`.

    A SNAPSHOT instrument, not a pretty-printer: comparing this against a
    pinned list catches a field added anywhere in the tree, one removed, and
    one whose type changed — which is what "unchanged" has to mean for a wire
    format a browser parses positionally.
    """
    if isinstance(v, dict):
        out: list[str] = []
        for k in sorted(v):
            out += leaf_shape(v[k], f"{path}.{k}" if path else k)
        return out
    if isinstance(v, list):
        inner = leaf_shape(v[0], "")[0].split(":")[-1] if v else "empty"
        return [f"{path}[{len(v)}]:{inner}"]
    return [f"{path}:{type(v).__name__}"]


#: Every leaf of `wall-test`'s one duck, as `leaf_shape` reports it. MEASURED
#: against the same probe run on the tree at the commit before the arm
#: channels landed: the duck's whole frame block came back BYTE-IDENTICAL
#: (`json.dumps(..., sort_keys=True)`, 10 steps, seed 0), so this is a pin on
#: a verified baseline and not on today's output. The arm channels are for a
#: body with a hand; a duck must not grow one, and `brain.inputs` must keep
#: saying `tof`, because a duck's range sensor IS its 8x8.
DUCK_FRAME_LEAVES = [
    "beak:str", "bodies[17]:int",
    "brain.beak:NoneType", "brain.cmd[3]:float", "brain.graph:str", "brain.head[4]:float",
    "brain.inputs.det.age:float", "brain.inputs.det.max:float", "brain.inputs.det.n:int",
    "brain.inputs.det.stale:bool",
    "brain.inputs.tof.age:float", "brain.inputs.tof.max:float", "brain.inputs.tof.stale:bool",
    "brain.kind:str", "brain.note:str", "brain.skill:NoneType", "brain.state:str",
    "brainKind:str", "cmdSpeed:float", "detector:str", "falls:int", "headApplied:bool",
    "holding:NoneType", "id:str", "lidar:NoneType", "mouth:float", "name:str",
    "odom:str", "odomEst[3]:float", "policy:NoneType", "rew:float", "robot:str", "role:NoneType",
    "sensors.det.age:float", "sensors.det.cam[7]:float", "sensors.det.fov[2]:float",
    "sensors.det.items[0]:empty", "sensors.det.selfBody:int", "sensors.det.t:float",
    "sensors.tof.age:float", "sensors.tof.mm[64]:int", "sensors.tof.t:float",
    "skill:NoneType", "speed:float", "steerable:bool", "step:int", "team:NoneType",
    "tof:str", "wallBumps:int", "wallTicks:int",
]


def test_a_duck_frame_block_is_unchanged_field_for_field(app):
    """The regression guard for the arm channels: a DUCK's row must be exactly
    what it was. Every leaf, by path and type, against `DUCK_FRAME_LEAVES`."""
    with TestClient(app) as c:
        assert c.post("/world/load", json={"scenario": "wall-test"}).status_code == 200
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            frame = None
            for _ in range(8):
                frame = ws.receive_json()
        d = frame["ducks"][0]
        assert leaf_shape(d) == DUCK_FRAME_LEAVES
        # Named separately from the list above, because these three are the
        # CLAIM and the list is the instrument: a duck has no claw, no arm and
        # no scanner, so it ships none of their channels.
        assert "gripper" not in d["sensors"] and "arm" not in d["sensors"]
        assert "lidar" not in d["sensors"]
        assert "lidar" not in d["brain"]["inputs"], "a duck's range sense is its ToF"
        assert set(d["brain"]["inputs"]) == {"tof", "det"}


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_a_mars_frame_carries_the_claw_as_a_reading(app):
    """`sensors.gripper` — the load, the predicate, and the bar's own scale.

    `holding` is the brain's (`Senses.holding` via `World.sense_grip`), not a
    threshold re-applied in the frame builder, and on an untouched arm at HOME
    it is False with the load at 0.0 N*m — which is the whole reason the flag
    has to be sent: a close on AIR reads the same 0.0, so a bar alone cannot
    answer it (`MarsDriver.gripper_load`'s table).
    """
    from microduck_local.robots import mars

    with TestClient(app) as c:
        load_mars(c)
        g = mars_frame(c)["ducks"][0]["sensors"]["gripper"]
        assert set(g) == {"load", "holding", "limit", "hold"}
        assert g["holding"] is False, "the jaw is empty at ARM_HOME"
        assert abs(g["load"]) < mars.HOLD_LOAD_NM
        # The scale and the mark come off the BODY, so the viewer keeps no
        # copy of another robot's constants.
        assert g["limit"] == mars.GRIPPER_EFFORT_LIMIT == 2.0
        assert g["hold"] == mars.HOLD_LOAD_NM == 1.0


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_a_mars_frame_carries_the_arm_and_the_command_it_is_tracking(app):
    """`sensors.arm` — seven joints of `q`, the same seven of `cmd`, limits.

    The PAIR is the point: the gap between achieved and commanded is the servo
    lag `brain/tidy_arm.py` pre-compensates, so a frame with `q` alone shows
    nothing. Driven through the socket's own command path: an `Intent.arm` is
    not reachable from here, but a gaze is, and `WorldRobot._push_arm` writes
    the head component into the SAME driven set — so a head command moving
    `cmd["joint_head"]` proves `cmd` is the driver's live target table and not
    a copy of `ARM_HOME`.
    """
    from microduck_local.robots import mars

    with TestClient(app) as c:
        load_mars(c)
        arm = mars_frame(c)["ducks"][0]["sensors"]["arm"]
        assert set(arm) == {"q", "cmd", "limits"}
        seven = set(mars.DRIVEN_JOINTS)
        assert len(seven) == 7
        assert set(arm["q"]) == set(arm["cmd"]) == set(arm["limits"]) == seven
        # Rounded to 4 decimals — 1e-4 rad is 0.006 degrees, two orders below
        # the backlash this is drawn to show.
        assert all(round(v, 4) == v for v in arm["q"].values())
        # Limits are THIS model's, which is what the servo clamps to.
        assert arm["limits"]["joint6"] == [-0.085, 0.8727]
        assert arm["limits"]["joint_head"] == [-0.3491, 0.3491]
        for name, (lo, hi) in arm["limits"].items():
            assert lo <= arm["q"][name] <= hi, name
        # `cmd` TRACKS: hold the gaze down and the head's target follows it
        # while the other six stay home.
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(4):
                ws.receive_json()
            for _ in range(20):
                ws.send_text(json.dumps({"cmd": [0.0, 0.0, 0.0]}))
                frame = ws.receive_json()
        moved = frame["ducks"][0]["sensors"]["arm"]
        assert moved["cmd"]["joint1"] == pytest.approx(mars.ARM_HOME["joint1"], abs=1e-3)
        assert moved["cmd"]["joint_head"] == pytest.approx(0.0, abs=1e-3)
        # …and the achieved angle is a MEASUREMENT, not the command echoed:
        # the servo carries Innate's compliance, so the two differ somewhere.
        assert any(abs(moved["q"][j] - moved["cmd"][j]) > 1e-6 for j in seven)


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_the_scan_carries_the_two_numbers_its_plot_cannot_be_drawn_without(app):
    """`sensors.lidar.minRange` and `.footprint`.

    Both were hard-coded per robot id in the viewer before this. `minRange` is
    the device's own floor — a return nearer than it is clipped and marked
    invalid, so it arrives as a 0 and the plot must draw the blind disc.
    `footprint` is how far out a return is the robot looking at its own arm,
    dropped by the adapter every brain here reads; it comes off the BODY, and
    the assertion is against the constants both consumers read.
    """
    from microduck_local.robots import mars
    from microduck_local.sensors.lidar import DEFAULT_MIN_RANGE_M

    with TestClient(app) as c:
        load_mars(c)
        lidar = mars_frame(c)["ducks"][0]["sensors"]["lidar"]
        assert lidar["minRange"] == DEFAULT_MIN_RANGE_M == 0.15
        assert lidar["footprint"] == mars.FOOTPRINT_M == 0.12
        # Still the whole scan beside them.
        assert len(lidar["mm"]) == 360 and lidar["maxRange"] == 6.0


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_the_range_freshness_row_is_named_after_the_device_the_body_has(app):
    """`brain.inputs.lidar` on a MARS, `brain.inputs.tof` on a duck, never both.

    A row labelled `tof` on a robot with no ToF is a claim about the hardware
    (`brain/runtime.age_inputs`). The AGE is checked as the SCAN CLOCK it is
    meant to be — a sawtooth on the control grid that resets when a scan
    lands, never older than one 6 Hz period — and not with a bound like
    `0 <= age <= 1/6`, which a hard-coded 0.0 satisfies and which passed a
    planted break (`AGENTS.md`'s "verify a filter against the complement", in
    its test-shaped form).

    NOT compared field-for-field against the frame's own `sensors.lidar.age`,
    and the reason is a real one worth writing down: they are stamped at
    different instants. The brain is stepped BEFORE the world advances, so its
    row is the age as of `t - CTRL_DT`, and on the tick a scan lands the frame
    reads 0.0 while the brain still holds the previous scan at 0.16 s. An
    equality between them would be a lock on the poll ORDER, which is not what
    this case is about.
    """
    with TestClient(app) as c:
        load_mars(c)
        ages, stale = [], []
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(24):
                d = ws.receive_json()["ducks"][0]
                row = d["brain"]["inputs"].get("lidar")
                if row:
                    ages.append(row["age"])
                    stale.append(row["stale"])
        inputs = d["brain"]["inputs"]
        assert "lidar" in inputs, "a MARS's range sense is the scan"
        assert "tof" not in inputs, "…and it has no ToF to report"
        assert set(inputs["lidar"]) == {"age", "stale", "max"}
        assert len(ages) >= 12, ages
        # A LIVE number and not a constant, on the control grid, never older
        # than one scan period.
        assert max(ages) > 0.01, ages
        assert max(ages) <= 1 / 6.0 + C.CTRL_DT, ages
        assert all(a % C.CTRL_DT < 1e-3 or C.CTRL_DT - a % C.CTRL_DT < 1e-3 for a in ages), ages
        # …and a SAWTOOTH: the age only ever RISES until a scan lands, then
        # drops back inside one frame's worth. Deliberately not "rises by
        # exactly one frame's worth": the send loop is wall-clock paced and
        # the world steps on credit, so a busy box legitimately sends two
        # frames of the same world tick and the age repeats. Pinning the step
        # would make this case a CPU-contention detector — the thing the MARS
        # notes in `docs/mars-roadmap.md` already warn about.
        step = 2 * C.CTRL_DT
        assert any(b < a for a, b in zip(ages, ages[1:])), ("never resets", ages)
        for a, b in zip(ages, ages[1:]):
            assert b >= a - 1e-9 or b < step + 1e-3, ages
        assert not any(stale), "a 6 Hz scan is not stale against a 0.25 s gate"

        # The duck is the other way round, in the same server.
        assert c.post("/world/load", json={"scenario": "wall-test"}).status_code == 200
        duck = mars_frame(c, ticks=8)["ducks"][0]["brain"]["inputs"]
        assert "tof" in duck and "lidar" not in duck


@pytest.mark.skipif(
    not __import__("microduck_local.robots.mars", fromlist=["x"]).mars_ready(),
    reason="MARS assets not fetched")
def test_a_range_preset_reaches_a_mars_lidar_under_either_name(app):
    """`set_noise` on the RANGE channel of a body whose range sensor is a scan.

    `"tof"` is the scenario's field name for "how noisy is this robot's range
    sense", so the panel's select sends it whatever device it is labelled
    after — and before this branch existed the lab answered **409 "has no
    ToF"** and logged "noise ignored", while the scenario's own field set the
    same preset happily at load. `"lidar"` is an alias, and the EVENT names
    whichever device actually changed.
    """
    with TestClient(app) as c:
        load_mars(c)
        r = c.post("/world/noise", json={"duck": "d0", "preset": "hostile"})
        assert r.status_code == 200, r.text
        assert r.json()["lidar"] == "hostile" and r.json()["tof"] is None
        # The alias, and it lands on the same device.
        r = c.post("/world/noise", json={"duck": "d0", "preset": "ideal", "sensor": "lidar"})
        assert r.status_code == 200 and r.json()["lidar"] == "ideal"
        assert c.get("/world").json()["ducks"][0]["lidar"] == "ideal"
        # An unknown preset is still 422, and the detector still has its own
        # branch on the same body.
        assert c.post("/world/noise", json={"duck": "d0", "preset": "x"}).status_code == 422
        assert c.post("/world/noise",
                      json={"duck": "d0", "preset": "hostile", "sensor": "det"}).status_code == 200
        # Over the SOCKET, which is the path the panel uses — and the event
        # log says the DEVICE, not the wire name. Events are drained after
        # every send (the loop clears them once a frame is on the wire), so
        # they are collected across the frames rather than read off the last.
        seen: list[str] = []
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            for _ in range(3):
                ws.receive_json()
            ws.send_text(json.dumps({"noise": {"duck": "d0", "preset": "datasheet"}}))
            frame = None
            for _ in range(5):
                frame = ws.receive_json()
                seen += frame["events"]
        assert frame["ducks"][0]["lidar"] == "datasheet"
        assert any("lidar noise" in e for e in seen), seen
        assert not any("ignored" in e for e in seen), seen


def test_a_duck_has_no_lidar_to_re_noise(app):
    """The complement of the case above: `sensor: "lidar"` on a body that
    carries none is a 409 and not a silent no-op, and a duck's plain `tof`
    request still reaches its ToF."""
    with TestClient(app) as c:
        assert c.post("/world/load", json={"scenario": "wall-test"}).status_code == 200
        r = c.post("/world/noise", json={"duck": "d0", "preset": "hostile", "sensor": "lidar"})
        assert r.status_code == 409 and "lidar" in r.json()["detail"]
        r = c.post("/world/noise", json={"duck": "d0", "preset": "hostile"})
        assert r.status_code == 200 and r.json()["tof"] == "hostile"
        assert c.post("/world/noise",
                      json={"duck": "d0", "preset": "hostile", "sensor": "zz"}).status_code == 422


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
