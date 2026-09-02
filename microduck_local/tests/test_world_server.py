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
            assert len(tof["mm"]) == 64 and len(tof["pts"]) == 64
            # The wall is a metre ahead: the middle of the frame reports ~0.94 m,
            # and its points sit on the wall plane at x = 1.0 (thickness 2 cm).
            mid = tof["mm"][3 * 8 + 3]
            assert 900 < mid < 960
            px = tof["pts"][3 * 8 + 3]
            assert px is not None and abs(px[0] - 0.99) < 0.02
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
