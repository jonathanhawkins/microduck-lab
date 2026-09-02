"""The playroom (roadmap 12.1–12.3): toys and a basket compose, a grasp welds
a toy to the beak and a release drops it, the tidy score counts what is in
the basket, and the shipped ground-pick cycle picks a toy placed where its
beak lands."""


import mujoco
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
from microduck_local.world import (
    Basket,
    Duck,
    Pickable,
    Scenario,
    World,
    make_playroom,
    validate_scenario,
)
from microduck_local.world.arena import GRASP_TOL_XY, PICK_REACH_AHEAD, PICK_REACH_LEFT
from microduck_local.world.scenario import ScenarioError

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def test_playroom_composes_and_validates():
    sc = make_playroom(seed=2, n=5)
    assert len(sc.pickables) == 5 and sc.basket is not None and sc.ducks[0].brain == "tidy"
    raw = sc.to_dict()
    assert validate_scenario(raw) == sc
    raw["pickables"][0]["kind"] = "lego"
    with pytest.raises(ScenarioError):
        validate_scenario(raw)
    raw["pickables"][0]["kind"] = "brick"
    raw["basket"]["rim"] = 0.5
    with pytest.raises(ScenarioError):
        validate_scenario(raw)
    w = World(sc)
    assert w.model.neq == 5 and not w.data.eq_active.any()
    kinds = {o["kind"] for o in w.objects_payload()}
    assert kinds == {"toy"} and w.tidy_score() == {"total": 5, "inBasket": 0, "held": []}
    seen = set()
    for _ in range(int(1.0 / C.CTRL_DT)):
        w.step()
        s = w.sensors_payload("d0")
        if s and "det" in s:
            seen |= {x["cls"] for x in s["det"]["items"]}
    assert "toy" in seen or "basket" in seen


def test_grasp_welds_release_drops_and_basket_counts():
    sc = Scenario(name="g", floor=(4, 4),
                  ducks=[Duck("d0", (0, 0, 0), None, None, None)],
                  pickables=[Pickable("t0", "block", (0.5, 0.0)), Pickable("t1", "brick", (-1.0, 0.0))],
                  basket=Basket((0.0, -1.0), (0.3, 0.3), 0.06))
    # On the walker (a zero-policy duck sags and respawns, which releases).
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")})
    d = w.ducks["d0"]
    # Nothing within reach of the beak: a close is an attempt, not a grasp.
    assert w.grasp(d) is None and d.grasp_attempts == 1 and d.beak_closed
    w.release(d)
    # Teleport the block under the tip and close: welded; it rides along.
    tip = w.mouth_tip(d).copy()
    b = w.pickables["t0"]
    q = w.model.jnt_qposadr[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_JOINT, "t0_free")]
    w.data.qpos[q:q + 3] = [tip[0], tip[1], tip[2]]
    mujoco.mj_forward(w.model, w.data)
    assert w.grasp(d, tol_xy=0.05, tol_z=0.06) == "t0" and d.holding == "t0"
    assert w.data.eq_active.any()
    for _ in range(50):
        d.set_cmd(w.data, [0, 0, 0])
        w.step()
    p = w.data.xpos[b]
    assert abs(p[2] - w.mouth_tip(d)[2]) < 0.04           # still at the beak, not on the floor
    assert w.objects_payload()[0]["held"] == "d0" and w.tidy_score()["held"] == ["t0"]
    # Carry it over the basket and let go: it lands inside.
    # Stand just outside the tray (feet clear of its wall) with the beak over it.
    bx, by = sc.basket.pos
    r = d.adr.root_qpos
    w.data.qpos[r:r + 2] = [bx - 0.19, by]
    mujoco.mj_forward(w.model, w.data)
    # A teleport moves the duck, not what the weld will drag next step: put
    # the block where the beak now is, then let go.
    w.data.qpos[q:q + 3] = w.mouth_tip(d)
    mujoco.mj_forward(w.model, w.data)
    assert w.release(d) == "t0" and d.holding is None and not w.data.eq_active.any()
    for _ in range(75):
        d.set_cmd(w.data, [0, 0, 0])
        w.step()
    assert w.in_basket("t0") and w.tidy_score()["inBasket"] == 1
    # A second duck cannot grasp what another holds; respawn releases.
    w.grasp(d)
    w.reset_duck("d0")
    assert d.holding is None and not d.beak_closed


@pytest.mark.skipif(not (POLICIES_DIR / "alpha_ground_pick.onnx").exists(), reason="upstream policies not checked out")
def test_ground_pick_skill_picks_a_toy_at_the_measured_reach():
    sc = Scenario(name="pick", floor=(4, 4),
                  ducks=[Duck("d0", (0, 0, 0), None, None, None)],
                  pickables=[Pickable("t0", "block", (PICK_REACH_AHEAD, PICK_REACH_LEFT))])
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")})
    d = w.ducks["d0"]
    for _ in range(50):
        d.set_cmd(w.data, [0, 0, 0])
        w.step()
    assert w.start_skill(d, "ground_pick") and d.skill == "ground_pick"
    assert not w.start_skill(d, "ground_pick")            # one cycle at a time
    picked_at = None
    for _ in range(int(3.5 / C.CTRL_DT)):
        w.step()
        if d.holding and picked_at is None:
            picked_at = w.t
    assert d.holding == "t0" and picked_at is not None and 1.3 < picked_at - 1.0 < 1.9
    assert d.skill is None                                # handed back at φ = 0.7
    assert d.falls == 0
    # Standing again with the block at beak height, off the floor.
    for _ in range(50):
        d.set_cmd(w.data, [0, 0, 0])
        w.step()
    assert w.data.xpos[w.pickables["t0"]][2] > 0.15 and d.trunk_pos(w.data)[2] > 0.10
    # …and it walks with it (12.5 check: the shipped walker carries a 20 g block).
    for _ in range(int(3.0 / C.CTRL_DT)):
        d.set_cmd(w.data, [0.3, 0, 0])
        w.step()
    assert d.falls == 0 and d.trunk_pos(w.data)[0] > 0.3 and d.holding == "t0"
    assert abs(GRASP_TOL_XY - 0.03) < 1e-9
