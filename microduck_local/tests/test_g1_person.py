"""Unitree G1 as a follow-scene person: MJCF attach + shipped walk ONNX."""

from __future__ import annotations

import json
import math

import pytest

from microduck_local import contract as C
from microduck_local.robots.g1 import g1_ready
from microduck_local.world import Person, Scenario, World, validate_scenario
from microduck_local.world.scenario import Duck

pytestmark = pytest.mark.skipif(
    not g1_ready(), reason="G1 assets missing — uv run fetch-g1")


def _g1_world(**kw) -> World:
    sc = Scenario(
        name="g1-walk",
        floor=(8.0, 8.0),
        ducks=[Duck("d0", (-2.5, 0.0, 0.0), None, None, None)],
        persons=[Person("p0", (0.0, 0.0), 0.0, path=[(2.5, 0.0), (-2.5, 0.0)],
                        speed=0.35, kind="g1", height=1.32, radius=0.25, yield_m=0.80)],
        **kw)
    return World(sc)


def test_g1_kind_round_trips_in_the_scenario_contract():
    raw = {"name": "a", "persons": [{"id": "p0", "pos": [1, 0], "kind": "g1"}]}
    p = validate_scenario(raw).persons[0]
    assert p.kind == "g1" and p.height == 1.32 and p.radius == 0.25 and p.yield_m == 0.80
    again = validate_scenario({"name": "b", "persons": [validate_scenario(raw).to_dict()["persons"][0]]})
    assert again.persons[0].kind == "g1"
    with pytest.raises(Exception):
        validate_scenario({"name": "c", "persons": [{"kind": "spot"}]})


def test_g1_person_is_a_physics_robot_not_a_capsule():
    w = _g1_world()
    p = w.persons["p0"]
    assert p.robot is not None and p.mocap < 0
    assert w.model.body(p.body).name.endswith("pelvis")
    assert p.payload(w.data)["robot"] == "g1"
    assert p.payload(w.data)["kind"] == "person"
    bodies = p.payload(w.data)["bodies"]
    assert len(bodies) == len(p.robot.scene_bodies)
    assert bodies[1][2] > 0.5  # pelvis off the floor


def test_g1_yield_does_not_erase_the_patrol():
    w = _g1_world()
    p = w.persons["p0"]
    n0 = len(p.route)
    p.waiting = 3.0
    # Pretend a duck is in the way for many ticks.
    for _ in range(int(4.0 / C.CTRL_DT)):
        p._step_g1(w.data, C.CTRL_DT, blockers=[(p.x + 0.2, p.y)])
    assert len(p.route) == n0 >= 2


def test_g1_person_walks_along_its_path():
    """The shipped walker, commanded along +x, covers ground without falling."""
    w = _g1_world()
    p = w.persons["p0"]
    x0 = p.x
    for _ in range(int(5.0 / C.CTRL_DT)):
        w.step()
    assert p.x > x0 + 0.4, f"G1 did not walk: x0={x0:.3f} x={p.x:.3f} fallen={p.fallen(w.data)}"
    assert not p.fallen(w.data)
    assert math.hypot(p.x, p.y) < 4.0


def test_g1_scene_endpoint_sends_welded_visual_meshes():
    from fastapi.testclient import TestClient

    from microduck_local import viz_server as V
    app = V.make_app([])
    with TestClient(app) as c:
        r = c.get("/scene/g1")
        assert r.status_code == 200
        sc = r.json()
        assert sc["bodies"][1] == "pelvis"
        # Real welded meshes, not 8-corner AABB boxes.
        assert len(sc["meshes"][0]["v"]) > 24 * 8
        assert len(json.dumps(sc)) < 40_000_000


def test_g1_is_still_the_person_the_detector_finds():
    sc = Scenario(
        name="g1-see",
        floor=(6.0, 6.0),
        ducks=[Duck("d0", (-1.5, 0.0, 0.0), None, None, "ideal")],
        persons=[Person("p0", (1.0, 0.0), math.pi, path=[], speed=0.0, kind="g1")])
    w = World(sc)
    for _ in range(3):
        w.step()
    det = w.ducks["d0"].detector
    assert det is not None and det.last is not None
    assert "person" in {it.cls for it in det.last.detections}


def test_the_g1_follow_room_is_inside_the_scenario_cap():
    """A 1.32 m person needs a taller room than a duck's 30 cm walls, and
    `world_server` raises them to 2.4 m when the G1 is the person. The
    validator's own cap was 2.0, so the BUILT-IN scenario stopped validating
    the moment someone ran `fetch-g1` — three tests failed at once and none
    of them named the G1."""
    from microduck_local.world import validate_scenario
    from microduck_local.world.scenario import MAX_WALL_HEIGHT_M
    from microduck_local.world_server import builtin_scenarios

    follow = builtin_scenarios()["follow-me"]
    assert follow.persons and follow.persons[0].kind == "g1"
    assert max(w.height for w in follow.walls) > 1.32     # taller than the G1
    assert max(w.height for w in follow.walls) <= MAX_WALL_HEIGHT_M
    assert validate_scenario(follow.to_dict()) == follow
