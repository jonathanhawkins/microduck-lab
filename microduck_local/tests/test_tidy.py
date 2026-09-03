"""The playroom (roadmap 12.1–12.3): toys and a basket compose, a grasp welds
a toy to the beak and a release drops it, the tidy score counts what is in
the basket, and the shipped ground-pick cycle picks a toy placed where its
beak lands."""


import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import Senses
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
    assert abs(GRASP_TOL_XY - 0.04) < 1e-9


def test_tidy_brain_geometry_and_gating_without_physics():
    """The brain's own arithmetic (roadmap 12.7): ranging a floor object by
    elevation from the camera pose the frame carries, trusting tracked ids
    rather than confidence, the basket zone, and the cold-gait turn kick."""
    import math

    from microduck_local.brain.tidy import Tidy
    from microduck_local.sensors.detector import Detection, DetectionFrame

    b = Tidy()
    # A toy centre 1.0 m ahead of a camera 0.234 m up looking 0.197 rad down
    # (the standing pose) sits at depression atan((0.234-0.015)/1.0).
    dep = math.atan2(0.234 - 0.015, 1.0)
    det = Detection("toy", "t0", bearing=0.0, elevation=0.197 - dep, width=0.04, range_est=9.0, conf=0.2)
    b._cam = (0.234, 0.197)
    x, y, rng = b._locate((0.0, 0.0, 0.0), det, b.p.toy_z, t=1.0)
    assert abs(rng - (1.0 + b.p.cam_ahead)) < 1e-6 and abs(x - rng) < 1e-9 and abs(y) < 1e-9
    # The walking gait holds the head 0.08 rad higher: a frame stamped with
    # that pose ranges the same elevation reading much farther out — which is
    # exactly the error the stamp exists to remove.
    b._cam = (0.241, 0.112)
    _, _, rng_walk = b._locate((0.0, 0.0, 0.0), det, b.p.toy_z, t=1.0)
    assert rng_walk == b.p.far_range          # 1.78 m by the elevation: beyond far_range it is a direction
    b.p = type(b.p)(far_range=4.0)
    _, _, rng_walk = b._locate((0.0, 0.0, 0.0), det, b.p.toy_z, t=1.0)
    assert rng_walk > 1.6
    b.p = type(b.p)()
    # Tracked ids beat confidence: a faint far toy is worth walking to, a
    # confident ghost (no id) is not.
    assert b._trusted(Detection("toy", "t3", 0.0, 0.0, 0.03, 2.0, conf=0.15))
    assert not b._trusted(Detection("toy", "", 0.0, 0.0, 0.03, 2.0, conf=0.49))
    # A toy that projects into a confirmed basket is one already delivered.
    b._cam = (0.234, 0.197)
    b.basket_mem, b.basket_confirmed = (1.0 + b.p.cam_ahead, 0.0), True
    assert b._in_basket_zone((0.0, 0.0, 0.0), det, 1.0)
    b.basket_mem = (2.0, 2.0)
    assert not b._in_basket_zone((0.0, 0.0, 0.0), det, 1.0)
    # Cold gait: a turn gets the forward kick, either way; warm: none.
    b._gait.cold = True
    assert b._turn(+1.0) == (b.p.turn_kick, 0.0, 1.0) and b._turn(-1.0) == (b.p.turn_kick, 0.0, -1.0)
    b._gait.cold = False
    assert b._turn(-1.0) == (0.0, 0.0, -1.0)
    # The frame contract the stamp rides on.
    fr = DetectionFrame(t=0.0, detections=[det], cam_z=0.234, cam_pitch=0.197)
    assert fr.cam_pitch == 0.197 and DetectionFrame(t=0.0, detections=[]).cam_z == 0.0


def test_tidy_rim_toys_are_staged_from_the_outside_and_routed_round_the_basket():
    """Toys near the basket (12.7): inside `basket_inside` they are left
    alone; out to `basket_zone` they get a staging point beyond them on
    the basket→toy ray, and a via-point when the straight walk to it would
    cross the basket disc. Pure geometry, no physics."""
    import math

    from microduck_local.brain.tidy import Tidy
    b = Tidy()
    p = b.p
    assert p.basket_inside < p.basket_zone
    # Unknown basket: nothing is special.
    assert b._stage_point(0.2, 0.0) is None and not b._point_in_basket_zone(0.0, 0.0)
    b.basket_mem, b.basket_confirmed = (1.0, 0.0), True
    assert b._point_in_basket_zone(1.0 + p.basket_inside - 0.01, 0.0)          # in the tray / hugging the rim
    assert not b._point_in_basket_zone(1.0 + p.basket_inside + 0.01, 0.0)      # rim zone: picked, from outside
    # A toy 0.23 m north of the basket: stage 0.3 m further north.
    st = b._stage_point(1.0, 0.23)
    assert st is not None and abs(st[0] - 1.0) < 1e-9 and abs(st[1] - (0.23 + p.stage_out)) < 1e-9
    assert b._stage_point(1.0, p.basket_zone + 0.01) is None                   # in the open: no staging
    # From (0, -1) the straight walk to that stage passes 0.29 m from the
    # basket centre: a via-point on the near side, `basket_avoid_r` out.
    via = b._via_point((0.0, -1.0), st)
    assert via is not None
    assert abs(math.hypot(via[0] - 1.0, via[1]) - p.basket_avoid_r) < 1e-9 and via[1] > 0 and via[0] < 1.0
    # From (0, 0) it passes 0.47 m out, and from due north nothing is in the way.
    assert b._via_point((0.0, 0.0), st) is None and b._via_point((1.0, 1.5), st) is None
    # Dead through the centre: round the left.
    assert b._via_point((0.0, 0.0), (2.0, 0.0))[1] > 0
    # The route for the toy: via first, then the stage; the estimate is untouched.
    b.est = (1.0, 0.23)
    b._plan_route((0.0, -1.0, 0.0), t=0.0)
    assert b._route == [via, st] and b.est == (1.0, 0.23)
    b._plan_route((0.0, 0.0, 0.0), t=0.0)
    assert b._route == [st]
    b.est = (1.0, 1.0)
    b._plan_route((0.0, 0.0, 0.0), t=0.0)
    assert b._route == []


@pytest.mark.skipif(not (POLICIES_DIR / "alpha_ground_pick.onnx").exists(), reason="upstream policies not checked out")
def test_tidy_picks_a_toy_behind_the_basket_without_touching_it():
    """End to end: a toy 0.23 m past the basket, seen from the spawn with
    the basket in between. The brain notes the basket while scanning,
    routes round it, comes in from the far side and picks the toy — feet
    never inside the rim's reach, no fall (measured before the staging:
    every fall in 8 traced runs was an approach at the rim)."""
    from microduck_local.brain import REGISTRY, Senses
    from microduck_local.brain import tidy as _tidy  # noqa: F401
    sc = Scenario(name="rim", floor=(4, 4),
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal", "ideal", brain="tidy")],
                  pickables=[Pickable("t0", "block", (0.45, 0.23))],
                  basket=Basket((0.45, 0.0), (0.3, 0.3), 0.06))
    w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
    d = w.ducks["d0"]
    brain = REGISTRY.make("tidy")
    states, routed, nearest = [], False, 9.0
    picked_at = None
    for _ in range(int(60.0 / C.CTRL_DT)):
        tof, det = d.tof.last, d.detector.last
        s = Senses(t=w.t, tof=tof, tof_age=None if tof is None else w.t - tof.t,
                   det=det, det_age=None if det is None else w.t - det.t,
                   speed=d.heading_speed(w.data), odom=w.odom(d), holding=d.holding is not None, skill=d.skill)
        intent = brain.step(s)
        w.apply_intent(d, intent)
        if d.skill is None:
            d.set_cmd(w.data, intent.twist, intent.head)
        w.step()
        routed |= bool(brain._route)
        if not states or states[-1] != brain.state:
            states.append(brain.state)
        pos = d.trunk_pos(w.data)
        nearest = min(nearest, float(np.hypot(pos[0] - 0.45, pos[1])))
        if d.holding and picked_at is None:
            picked_at = w.t
        if brain.state == "carry":
            break
    assert brain.basket_confirmed and routed, states
    assert picked_at is not None and d.holding == "t0", states
    assert d.falls == 0
    assert nearest > 0.19, nearest                      # the trunk stayed outside the rim's reach (0.156 + feet)


def test_tidy_blind_end_reaches_with_the_neck_and_walks_the_last_leg_slowly():
    """After the standing look (`aimed`) the last leg to the basket is
    walked at `blind_speed` with the neck pitched back (measured: the tip
    0.095 m ahead instead of 0.080), and the drop keeps that pose; before
    the look the head is level and the neck straight."""
    from microduck_local.brain import Senses
    from microduck_local.brain.tidy import Tidy, TidyParams
    assert TidyParams().neck_reach == 0.0                 # measured: it creeps into the rim at the drop; off
    b = Tidy(TidyParams(neck_reach=-0.6))
    p = b.p
    b.basket_mem, b.basket_confirmed = (1.0, 0.0), True
    b.est, b.goal_kind, b.target_name = (1.0, 0.0), "basket", "basket"
    b.state, b.t_state, b.aimed = "deliver", 0.0, True
    b._gait.cold = False
    leg = b.step(Senses(t=0.02, speed=0.15, odom=(1.0 - p.basket_reach - 0.1, 0.0, 0.0), holding=True))
    assert leg.twist == (p.blind_speed, 0.0, 0.0) and leg.head[0] == p.neck_reach and leg.head[1] == 0.0
    arrive = b.step(Senses(t=0.04, speed=0.15, odom=(1.0 - p.basket_reach + 0.005, 0.0, 0.0), holding=True))
    assert b.state == "drop" and arrive.head[0] == p.neck_reach
    drop = b.step(Senses(t=0.06, speed=0.0, odom=(1.0 - p.basket_reach + 0.005, 0.0, 0.0), holding=True))
    assert drop.beak == "open" and drop.head[0] == p.neck_reach
    # Not yet aimed: level head, straight neck, the normal walk.
    c = Tidy()
    c.basket_mem, c.basket_confirmed = (1.0, 0.0), True
    c.est, c.goal_kind, c.target_name = (1.0, 0.0), "basket", "basket"
    c.state, c.t_state, c.aimed = "deliver", 0.0, False
    c._gait.cold = False
    far = c.step(Senses(t=0.02, speed=0.15, odom=(0.0, 0.0, 0.0), holding=True))
    assert far.head == (0.0, 0.0, 0.0, 0.0) and far.twist[0] == p.approach_speed


def test_tidy_backoff_sidesteps_left_before_turning():
    """After a drop the duck stands with its feet a few centimetres from the
    rim; the back-off sidesteps LEFT first (measured: the one manoeuvre that
    never tripped standing 0.17 m from the basket centre), then turns left,
    then walks."""
    from microduck_local.brain import Senses
    from microduck_local.brain.tidy import Tidy
    b = Tidy()
    p = b.p
    b.state, b.t_state, b.scan_turned = "backoff", 10.0, 0.0
    b._prev_yaw = 0.0
    first = b.step(Senses(t=10.02, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert first.twist == (0.0, p.backoff_side_vy, 0.0) and "sidestep" in first.note
    mid = b.step(Senses(t=10.0 + p.backoff_side_s - 0.02, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert mid.twist[1] == p.backoff_side_vy and mid.twist[0] == 0.0
    after = b.step(Senses(t=10.0 + p.backoff_side_s + 0.02, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert after.twist[2] == 1.0 and after.twist[1] == 0.0                 # the left turn
    # Once turned round, a straight walk, then a scan.
    b.scan_turned = p.backoff_turn
    walk = b.step(Senses(t=10.0 + p.backoff_side_s + 0.5, speed=0.0, odom=(0.0, 0.0, p.backoff_turn)))
    assert walk.twist == (p.approach_speed, 0.0, 0.0) and b.state == "backoff"
    b.step(Senses(t=10.0 + p.backoff_side_s + 0.5 + p.backoff_walk_s + 0.1, speed=0.0, odom=(0.0, 0.0, p.backoff_turn)))
    assert b.state == "scan"


def test_tidy_steers_by_the_loop_closed_pose():
    """The brain keeps its own map and reads the corrected pose (12.7 + 5.5):
    with the map's odometry→map offset set, every estimate it makes is in
    the corrected frame; with loop closure off it steers by raw odometry."""
    from microduck_local.brain import Senses
    from microduck_local.brain.tidy import Tidy, TidyParams
    b = Tidy()
    assert b.map is not None and b.p.loop_closure
    b.map.offset[:] = (0.5, -0.2, 0.0)
    s = Senses(t=0.0, odom=(1.0, 1.0, 0.0), speed=0.0)
    assert b._pose(s) == pytest.approx((1.5, 0.8, 0.0))
    b.step(s)
    assert b.inputs()["tidy"]["loopClosure"]["offset"] == [0.5, -0.2, 0.0]
    raw = Tidy(TidyParams(loop_closure=False))
    assert raw.map is None and raw._pose(s) == (1.0, 1.0, 0.0)


def test_a_tethered_brain_reads_its_latency_off_the_sensor_ages_and_stops_earlier():
    """brain/tether.py delays senses by half the round trip and intents by
    the other half; a tidy brain's ToF ages then floor at the one-way lag,
    it takes twice that as its latency, and the rim stop moves out by its
    speed times that. Onboard (no tether) the margin is zero."""
    from microduck_local.brain.tether import Tether
    from microduck_local.brain.tidy import Tidy, TidyParams
    from microduck_local.sensors.tof import TofFrame

    def senses(t, age):
        f = TofFrame(t=t - age, depth_mm=np.full((8, 8), 2000, np.uint16), valid=np.ones((8, 8), bool))
        return Senses(t=t, tof=f, tof_age=age, speed=0.16, odom=(0.0, 0.0, 0.0))
    onboard = Tidy(TidyParams())
    for k in range(20):
        onboard.step(senses(0.066 * k, 0.02))
    assert onboard.latency < 0.05 and onboard.stop_margin < 0.01
    # Through a 250 ms tether the same 15 Hz frames arrive 125 ms older.
    th = Tether(0.25)
    tethered = Tidy(TidyParams())
    for k in range(30):
        s = th.senses_in(senses(0.066 * k, 0.02))
        if s.tof is not None:
            tethered.step(s)
    assert 0.24 < tethered.latency < 0.32
    assert 0.03 < tethered.stop_margin < 0.06                        # 0.16 m/s x ~0.27 s
    # The intents land half a round trip later: the one applied at t is the one decided at t - 0.125.
    from microduck_local.brain.runtime import Intent
    out = [th.intent_out(Intent(twist=(k * 0.1, 0.0, 0.0)), 0.05 * k) for k in range(8)]
    assert out[0].twist[0] == 0.0 and abs(out[5].twist[0] - 0.2) < 1e-9        # at t=0.25 the one from t=0.10... within a tick
