"""World mode of the lab (roadmap 0.4): scenario CRUD with built-ins read-only,
loading a world, the /ws/sim frame shape with ToF payloads, drive and reset
over the socket, and the same front-door origin rule as /ws."""

import json

import pytest
from fastapi.testclient import TestClient

from microduck_local import contract as C
from microduck_local import viz_server as V
from microduck_local import world_server as W

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
            assert len(d["bodies"]) == 16 and d["bodies"][0] == [0, 0, 0, 1, 0, 0, 0]
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
        with c.websocket_connect("/ws/sim", headers=ORIGIN) as ws:
            frame = None
            for _ in range(8):
                frame = ws.receive_json()
            d = frame["ducks"][0]
            assert d["brain"]["kind"] == "follow" and "inputs" in d["brain"]
            assert d["brain"]["inputs"]["det"]["max"] > 0 and d["headApplied"] is False
            persons = [o for o in frame["objects"] if o["kind"] == "person"]
            assert persons and persons[0]["id"] == "p0" and persons[0]["possessed"] is False
            assert frame["possessed"] is None
            # Possess the person: the manual command drives IT, the duck keeps its brain.
            ws.send_text(json.dumps({"possess": "p0"}))
            ws.send_text(json.dumps({"cmd": [0.4, 0.0, 0.0]}))
            for _ in range(6):
                frame = ws.receive_json()
            assert frame["possessed"] == "p0" and frame["mode"] == "manual"
            assert frame["ducks"][0]["brain"]["kind"] == "follow"
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
