"""Locks for MARS tidying the playroom (`docs/mars-roadmap.md` Phase 5).

Four seams, and the reason each one is HERE rather than in a phase report:

* **The room's clock.** `Scenario.physics_dt` round-trips, `compose` honours
  it, the arena derives its substeps from it, and every DUCK scenario stays at
  5 ms. That last one is the golden bit: `tests/test_arena.py` locks the
  composed world step for step against the walk env, and a timestep field that
  leaked into a duck room would break the walker's trajectory.
* **The grasp, as a MEASUREMENT turned into a test.** A scripted pick in the
  COMPOSED `mars-playroom` holds at 2 ms and fails at 5 ms. That is the whole
  argument for the field above, so it is asserted rather than written down.
* **`Senses.holding` for a claw.** It flips on a held toy, names WHICH toy,
  and does not flip on a jaw shut on air — the three states 4b measured
  (0.0000 / 0.0000 / ~2 N*m).
* **`TidyArm`'s state machine**, driven on synthetic senses the way
  `tests/test_tidy.py` drives the duck's, plus two end-to-end cases in the
  room: a pick sequence that ends HOLDING, and a place over the rim that ends
  with the toy inside the basket footprint.

Every case here was shown to FAIL on a planted break before it was kept
(`AGENTS.md`: "A/B new tests against planted regressions") — the table is in
the phase report.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import REGISTRY
from microduck_local.brain import tidy_arm as TA
from microduck_local.brain.runtime import Senses, attach_world
from microduck_local.robots import mars
from microduck_local.robots import mars_env as ME
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.world import World
from microduck_local.world.compose import compose
from microduck_local.world.scenario import (
    DEFAULT_PHYSICS_DT,
    PICKABLE_KINDS,
    Basket,
    Duck,
    Pickable,
    Scenario,
    ScenarioError,
    Wall,
    load_scenario,
    make_playroom,
    robot_physics_dt,
    validate_scenario,
)

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"

needs_mars = pytest.mark.skipif(
    not (C.SCENE_WALK_XML.exists() and mars.mars_ready()),
    reason="microduck_rl checkout or MARS assets not present")


# =========================================================== the room's clock

def test_physics_dt_defaults_to_the_worlds_own_and_round_trips():
    sc = Scenario(name="r")
    assert sc.physics_dt == DEFAULT_PHYSICS_DT == C.PHYSICS_DT
    assert sc.to_dict()["physics_dt"] == DEFAULT_PHYSICS_DT
    back = validate_scenario(replace(sc, physics_dt=0.002).to_dict())
    assert back.physics_dt == 0.002


def test_the_control_tick_is_the_worlds_and_the_module_says_so():
    """`world/scenario.py` re-declares the two clock constants (it must not
    import `contract`, which drags a robot package in); they are the same
    numbers or this fails naming which."""
    from microduck_local.world import scenario as S

    assert S.DEFAULT_PHYSICS_DT == C.PHYSICS_DT
    assert S.CONTROL_DT == C.CTRL_DT


def test_a_timestep_that_does_not_divide_the_control_tick_is_refused():
    with pytest.raises(ScenarioError, match="control tick"):
        validate_scenario({"name": "r", "physics_dt": 0.003})
    with pytest.raises(ScenarioError, match="physics_dt"):
        validate_scenario({"name": "r", "physics_dt": 0.05})


def test_compose_honours_the_scenarios_timestep_and_the_arena_derives_substeps():
    if not C.SCENE_WALK_XML.exists():
        pytest.skip("microduck_rl checkout not present")
    sc = Scenario(name="r", ducks=[Duck("d0", (0.0, 0.0, 0.0))])
    assert compose(sc).opt.timestep == pytest.approx(DEFAULT_PHYSICS_DT)
    w = World(sc)
    assert w.substeps == C.DECIMATION
    assert w.substeps * w.model.opt.timestep == pytest.approx(C.CTRL_DT)


@needs_mars
def test_a_mars_room_is_two_milliseconds_and_a_duck_room_is_five():
    """The BODY declares it (`BodyBase.physics_dt`) and the builder asks."""
    assert robot_physics_dt("mars") == mars.GRASP_PHYSICS_DT == 0.002
    assert robot_physics_dt("microduck") == DEFAULT_PHYSICS_DT
    duck = make_playroom(seed=0, n=3)
    marsroom = make_playroom(seed=0, n=3, robot="mars", brain="tidy_arm")
    assert duck.physics_dt == DEFAULT_PHYSICS_DT
    assert marsroom.physics_dt == 0.002
    assert marsroom.ducks[0].robot == "mars" and marsroom.ducks[0].brain == "tidy_arm"
    # ...and the same toy LAYOUT, so the two bodies are measured on one room.
    assert [t.pos for t in duck.pickables] == [t.pos for t in marsroom.pickables]
    w = World(marsroom)
    assert w.model.opt.timestep == pytest.approx(0.002)
    assert w.substeps == 10


@needs_mars
def test_the_shipped_mars_scenarios_carry_the_grasp_timestep():
    for name in ("mars-playroom", "mars-follow"):
        sc = load_scenario(SCENARIOS / f"{name}.json")
        assert sc.physics_dt == pytest.approx(mars.GRASP_PHYSICS_DT), name


def test_every_duck_scenario_on_disk_is_still_at_five_milliseconds():
    """The golden bit. `tests/test_arena.py` locks a composed duck world step
    for step against the walk env, so a saved scene that picked up a finer
    timestep would be a different robot."""
    for path in sorted(SCENARIOS.glob("*.json")):
        sc = load_scenario(path)
        if any(d.robot != "microduck" for d in sc.ducks):
            continue
        assert sc.physics_dt == DEFAULT_PHYSICS_DT, path.name


# ============================================== the grasp, as a MEASUREMENT
#
# Slow (it compiles two worlds and drives an arm through a full pick at 2 ms
# and at 5 ms), and it is the case the whole timestep field exists for.

def _room_with_one_block(dt: float) -> Scenario:
    """The composed playroom, one block, the basket — at `dt`."""
    pts = [(-1.5, -1.25), (1.5, -1.25), (1.5, 1.25), (-1.5, 1.25)]
    return Scenario(
        name="grasp", floor=(3.5, 3.0),
        walls=[Wall(pts[i], pts[(i + 1) % 4], 0.3, 0.02) for i in range(4)],
        ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal", "ideal", None, robot="mars")],
        pickables=[Pickable("t0", "block", (1.4, 1.1), 0.0)],
        basket=Basket((1.15, 0.9), (0.3, 0.3), 0.06),
        physics_dt=dt)


class _Rig:
    """Drive one MARS's arm in a COMPOSED room, slewed as a brain must."""

    SLEW = mars.MAX_TARGET_RATE_RAD_S * C.CTRL_DT

    def __init__(self, dt: float):
        from microduck_local.robots.mars_ik import ArmKinematics

        self.w = World(_room_with_one_block(dt), seed=0)
        self.d = self.w.ducks["d0"]
        self.k = ArmKinematics(seed=0)
        self.cmd = dict(mars.ARM_HOME)

    def to_world(self, p_base):
        x, y, yaw = self.d.driver.pose(self.w.data)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * p_base[0] - s * p_base[1],
                         y + s * p_base[0] + c * p_base[1], p_base[2]])

    def put_toy(self, toy: str, pos) -> None:
        m, data = self.w.model, self.w.data
        b = self.w.pickables[toy]
        j = next(k for k in range(m.njnt) if int(m.jnt_bodyid[k]) == b)
        q, dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        data.qpos[q:q + 3] = pos
        data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(m, data)

    def hold(self, targets: dict, seconds: float) -> None:
        want = {**self.cmd, **targets}
        for _ in range(int(round(seconds / C.CTRL_DT))):
            for j in self.cmd:
                self.cmd[j] += float(np.clip(want[j] - self.cmd[j], -self.SLEW, self.SLEW))
            self.d.set_cmd(self.w.data, (0.0, 0.0, 0.0))
            self.d.set_arm(self.cmd)
            self.w.step()

    def scripted_pick(self) -> bool:
        """IK the open jaw onto a block at the standoff, close, lift, hold 2 s.
        Returns whether the claw still has it."""
        z = PICKABLE_KINDS["block"]["size"][2] / 2 + 0.001
        spot = self.k.shell_point(*ME.PICK_SPOT, z)
        err, q_open = self.k.solve_grasp(spot, TA.OPEN_RAD, restarts=8)
        assert q_open is not None and err < 0.005
        self.hold(dict(zip(mars.ARM_JOINTS, q_open.tolist())), 2.0)
        self.k.place(q_open)
        grasp = self.k.grasp_point().copy()
        self.put_toy("t0", self.to_world(grasp))
        self.hold({}, 0.2)
        self.hold({"joint6": TA.CLOSE_RAD}, 1.5)
        _, q_lift = self.k.solve_grasp(grasp + np.array([0.0, 0.0, 0.12]),
                                       TA.CLOSE_RAD, start=q_open, restarts=3)
        assert q_lift is not None
        lift = dict(zip(mars.ARM_JOINTS, q_lift.tolist()))
        lift["joint6"] = TA.CLOSE_RAD
        self.hold(lift, 1.5)
        self.hold({}, 2.0)
        return self.d.holding == "t0"


@needs_mars
def test_the_scripted_grasp_holds_at_two_milliseconds_in_the_composed_room():
    assert _Rig(0.002).scripted_pick() is True


@needs_mars
def test_the_same_grasp_FAILS_at_the_worlds_five_milliseconds():
    """What fails at 5 ms is EJECTION, not slip (4b: the block leaves at
    0.08-6.6 m of travel in the two seconds after the lift). This is the
    measurement that makes `Scenario.physics_dt` necessary rather than tidy,
    so it is a test: 7/8 spots hold at 2 ms and 3/8 at 5 ms, and this spot
    is one that separates them."""
    assert _Rig(0.005).scripted_pick() is False


# ============================================ Senses.holding, for a gripper

@needs_mars
def test_holding_names_the_toy_and_a_jaw_shut_on_air_does_not():
    rig = _Rig(0.002)
    # (1) EMPTY, jaws open, the toy parked across the room.
    rig.hold({}, 0.5)
    assert rig.d.holding is None
    assert rig.d.held_body(rig.w.data) == -1
    # (2) shut on AIR, at the hard stop: the servo is saturated and the
    #     CONSTRAINT torque is zero — 4b's discriminating case.
    rig.hold({"joint6": TA.CLOSE_RAD}, 1.5)
    assert abs(rig.d.driver.gripper_load(rig.w.data)) < mars.HOLD_LOAD_NM
    assert rig.d.holding is None
    # (3) HOLDING: the claw names WHICH toy, which is what a room with
    #     several needs and what the contact conjunct is there for.
    rig2 = _Rig(0.002)
    assert rig2.scripted_pick() is True
    assert rig2.d.holding == "t0"
    assert abs(rig2.d.driver.gripper_load(rig2.w.data)) >= mars.HOLD_LOAD_NM
    assert rig2.w.sense_grip(rig2.d) == "t0"


@needs_mars
def test_jaws_merely_RESTING_on_a_toy_are_not_holding_it():
    """The state that the LOAD half of the predicate is the whole of.

    Open jaws lowered onto the block make a finger<->toy CONTACT with no
    squeeze, so a predicate that kept only the contact term would call this
    holding. 4b measured the pair apart here (0.0035 N*m resting against
    1.89-2.03 gripping) and this is the case that checks it — 4b's own lesson
    about testing a discriminator only where the candidates AGREE.
    """
    rig = _Rig(0.002)
    z = PICKABLE_KINDS["block"]["size"][2] / 2 + 0.001
    spot = rig.k.shell_point(*ME.PICK_SPOT, z)
    err, q_open = rig.k.solve_grasp(spot, TA.OPEN_RAD, restarts=8)
    assert q_open is not None and err < 0.005
    rig.hold(dict(zip(mars.ARM_JOINTS, q_open.tolist())), 2.0)
    rig.k.place(q_open)
    # Nudged along the JAW's own axis until the block's face overlaps a blade
    # pad: the jaws are 78.8 mm apart at the stop and the block is 40 mm, so a
    # centred block touches nothing (19 mm a side). This is the honest version
    # of the state — the estimate was off and one blade is against the block.
    a, b_ = rig.k._pads()
    axis = (a - b_) / float(np.linalg.norm(a - b_))
    gap = rig.k.jaw_gap()
    half = PICKABLE_KINDS["block"]["size"][0] / 2
    rig.put_toy("t0", rig.to_world(
        rig.k.grasp_point().copy() + axis * (gap / 2 - half + 0.003)))
    rig.hold({}, 0.5)                              # resting, NOT closed
    toy_body = rig.w.pickables["t0"]
    fingers = set(rig.d.driver._finger_bodies)
    touching = False
    for i in range(rig.w.data.ncon):
        con = rig.w.data.contact[i]
        pair = {int(rig.w.model.geom_bodyid[con.geom1]),
                int(rig.w.model.geom_bodyid[con.geom2])}
        if toy_body in pair and pair & fingers:
            touching = True
    assert touching, "the jaws are not on the block — the case is not set up"
    assert rig.d.held_body(rig.w.data) == -1
    assert rig.w.sense_grip(rig.d) is None


@needs_mars
def test_a_close_on_AIR_never_reads_as_holding_at_any_moment_of_it():
    """The state that decides WHICH torque the predicate reads.

    4b: the SERVO torque (`qfrc_applied`, 4a's slot) is saturated at -2 N*m
    for 8 of the 40 control steps of a close on air — so one sample of it
    cannot tell "closing" from "holding" — while the CONSTRAINT torque is
    0.0000 for all 40. Sampling every step of an empty close is the only
    place the two differ.
    """
    rig = _Rig(0.002)
    rig.put_toy("t0", [1.4, 1.1, 0.021])           # far away: the claw is empty
    rig.hold({"joint6": TA.OPEN_RAD}, 1.5)
    loads, servo = [], []
    want = dict(rig.cmd, joint6=TA.CLOSE_RAD)
    for _ in range(40):
        for j in rig.cmd:
            rig.cmd[j] += float(np.clip(want[j] - rig.cmd[j], -rig.SLEW, rig.SLEW))
        rig.d.set_cmd(rig.w.data, (0.0, 0.0, 0.0))
        rig.d.set_arm(rig.cmd)
        rig.w.step()
        loads.append(abs(rig.d.driver.gripper_load(rig.w.data)))
        servo.append(abs(float(rig.w.data.qfrc_applied[rig.d.driver.adr["joint6"][1]])))
        assert rig.d.holding is None
    assert max(loads) < mars.HOLD_LOAD_NM, f"constraint torque peaked at {max(loads)}"
    assert max(servo) >= mars.HOLD_LOAD_NM, (
        "the servo torque never saturated — this case no longer separates the "
        "two candidate signals, so it no longer tests anything")


@needs_mars
def test_the_arm_reports_its_achieved_pose_and_it_is_not_the_command():
    """`Senses.arm` is the robot's own encoders, not the target: Innate's
    compliance model puts the two a few hundredths of a radian apart."""
    rig = _Rig(0.002)
    rig.hold({"joint2": mars.ARM_HOME["joint2"] + 0.6}, 2.0)
    act = rig.d.arm_qpos(rig.w.data)
    assert set(act) == set(mars.DRIVEN_JOINTS)
    assert act["joint2"] != pytest.approx(rig.d.arm_targets()["joint2"], abs=1e-9)
    assert act["joint2"] == pytest.approx(rig.d.arm_targets()["joint2"], abs=0.1)


@needs_mars
def test_the_tidy_score_and_in_basket_are_body_agnostic():
    """`World.in_basket` is pure geometry, so one instrument measures a beak
    and a claw. Drop the block into the tray by hand and the count moves."""
    rig = _Rig(0.002)
    assert rig.w.tidy_score() == {"total": 1, "inBasket": 0, "held": []}
    rig.put_toy("t0", [1.15, 0.9, 0.05])
    assert rig.w.in_basket("t0") is True
    assert rig.w.tidy_score()["inBasket"] == 1
    rig.put_toy("t0", [1.15 + 0.2, 0.9, 0.05])      # outside the footprint
    assert rig.w.in_basket("t0") is False
    # ...and a toy CARRIED over the tray is not in it either: the height rule
    # is what separates "delivered" from "about to be".
    rig.put_toy("t0", [1.15, 0.9, 0.50])
    assert rig.w.in_basket("t0") is False


# ==================================================== the state machine

def _frame(t: float, *dets: Detection, cam_z: float = 0.2438,
           cam_pitch: float = 0.349) -> DetectionFrame:
    return DetectionFrame(t=t, detections=list(dets), cam_z=cam_z,
                          cam_pitch=cam_pitch, cam_yaw=0.0)


def _toy(bearing: float, elevation: float, name: str = "t0") -> Detection:
    return Detection(cls="toy", name=name, bearing=bearing, elevation=elevation,
                     width=0.05, range_est=0.4, conf=0.9)


def _basket(bearing: float, elevation: float) -> Detection:
    return Detection(cls="basket", name="basket", bearing=bearing,
                     elevation=elevation, width=0.2, range_est=0.5, conf=0.9)


def _senses(t: float, frame: DetectionFrame | None = None, odom=(0.0, 0.0, 0.0),
            holding: bool = False, speed: float = 0.0,
            arm: dict | None = None) -> Senses:
    return Senses(t=t, det=frame, det_age=None if frame is None else 0.0,
                  odom=odom, speed=speed, holding=holding,
                  arm=arm if arm is not None else dict(mars.ARM_HOME))


def _elev_for(range_m: float, cam_z: float = 0.2438, cam_pitch: float = 0.349,
              target_z: float = 0.021) -> float:
    """The camera-frame elevation a floor object at this horizontal range
    reports — the inverse of `TidyArm._locate`, so a synthetic frame can put
    a toy exactly where a test wants it."""
    depression = math.atan2(cam_z - target_z, range_m)
    return cam_pitch - depression


def test_a_toy_seen_starts_an_approach_and_the_standoff_is_the_arms():
    b = TA.TidyArm()
    assert b.state == "search"
    b.step(_senses(0.0))                       # nothing in sight: turn
    assert b.state == "search"
    assert b.last[2] > 0.0 and b.last[0] == 0.0
    b.step(_senses(0.1, _frame(0.1, _toy(0.0, _elev_for(1.0)))))
    assert b.state == "approach" and b.target_name == "t0"
    ahead, left = b._standoff()
    assert ahead == pytest.approx(0.3353, abs=0.01)
    assert left == pytest.approx(-0.0528, abs=0.01)


@needs_mars
def test_the_standoff_is_the_ARMS_shell_point_and_not_the_fallback_pair():
    """`_standoff` answers a measured pair before any kinematics exists and
    the ARM's own shell point after — and they have to be the same thing, or
    the fallback is a second definition waiting to drift. The `left` term is
    the one that matters: the shoulder is 53 mm right of the centreline, and a
    standoff that forgot it aims the claw a block's width off."""
    b = TA.TidyArm()
    fallback = b._standoff()
    b.kin()                                        # build the private model
    measured = b._standoff()
    assert measured[0] == pytest.approx(fallback[0], abs=0.002)
    assert measured[1] == pytest.approx(fallback[1], abs=0.002)
    assert measured[1] < -0.04, "the shoulder's lateral offset is missing"


def _toy_at(b, ahead: float, left: float, name: str = "t0") -> Detection:
    """A synthetic detection of a floor toy at (`ahead`, `left`) in the base
    frame — built through the true camera geometry, so the brain's own
    `_locate` puts it back where this test meant it."""
    cam = np.array([b.p.cam_x, b.p.cam_y, 0.2438])
    v = np.array([ahead, left, b.p.toy_z]) - cam
    cp, sp = math.cos(b.p.head_pitch), math.sin(b.p.head_pitch)
    vx, vz = v[0] * cp - v[2] * sp, v[0] * sp + v[2] * cp
    return Detection("toy", name, math.atan2(v[1], vx),
                     math.atan2(vz, math.hypot(vx, v[1])), 0.05, 0.4, 0.9)


def test_at_the_standoff_it_stops_and_settles():
    """Inside the shell, squared up: `approach` -> `settle`. The toy is placed
    exactly at the arm's own standoff, so what is under test is the stop rule
    and not the drive."""
    b = TA.TidyArm()
    ahead, left = b._standoff()
    t = 0.0
    b.step(_senses(t, _frame(t, _toy_at(b, 1.0, 0.0))))
    assert b.state == "approach"
    for k in range(5):
        t = 0.1 + 0.02 * k
        b.step(_senses(t, _frame(t, _toy_at(b, ahead, left))))
    assert b.state == "settle"
    assert b.last == (0.0, 0.0, 0.0)          # standing still for the look


def test_at_the_standoff_but_off_bearing_it_turns_before_it_stands():
    """The stop is TWO conditions and a stand is only legal when both hold.
    Arriving at the right range 0.3 rad off the standoff's bearing must turn,
    not stand — the first version stood there until the 30 s timeout and
    every approach ended that way (0 toys in a 90 s run)."""
    b = TA.TidyArm()
    ahead, left = b._standoff()
    b.est = (math.hypot(ahead, left), 0.0)         # the right RANGE...
    twist, dist, bearing = b._servo((0.0, 0.0, 0.5), math.hypot(ahead, left),
                                    ahead, left)   # ...and 0.5 rad of yaw off
    assert abs(dist - math.hypot(ahead, left)) < b.p.stop_tol
    assert abs(bearing) > b.p.align_tol
    assert twist[0] == 0.0 and abs(twist[2]) > 0.05, "it must square up, not stand"


def test_holding_carries_and_a_basket_in_range_places():
    """`carry` -> `deliver` -> `place`, on synthetic senses. The arm poses are
    stubbed: what is under test is the ORDER, the way `tests/test_tidy.py`
    tests the duck's."""
    b = TA.TidyArm()
    b.state, b.t_state = "lift", 0.0
    b._q_lift = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])
    b._q_open = b._q_lift
    b.target_name = "t0"
    b.step(_senses(b.p.lift_s + 0.1, holding=True))
    assert b.state == "carry" and b.picked == 1 and b.held_name == "t0"
    # The basket, seen from close enough to trust: `carry` -> `deliver`.
    t = 5.0
    b.step(_senses(t, _frame(t, _basket(0.0, _elev_for(0.5, target_z=0.08))),
                   holding=True))
    assert b.state == "deliver"
    # ...and at the standoff, with the arm stubbed, `deliver` -> `place`.
    b.est = (b.p.basket_reach, 0.0)
    b._kin = _StubKin()
    b.step(_senses(t + 0.02, holding=True))
    assert b.state == "place"


def test_the_toy_leaving_the_claw_in_place_counts_as_a_delivery_and_retracts():
    b = TA.TidyArm()
    b.state, b.t_state, b.held_name = "place", 0.0, "t0"
    b.step(_senses(0.1, holding=False))
    assert b.state == "retract"
    assert b.delivered == 1 and b.dropped == 0 and b.lost_in == {"place": 1}


def test_the_toy_leaving_the_claw_on_the_MOVE_is_a_drop_and_starts_over():
    b = TA.TidyArm()
    b.state, b.t_state, b.held_name = "deliver", 0.0, "t0"
    b.step(_senses(0.1, holding=False))
    assert b.state == "search"
    assert b.dropped == 1 and b.delivered == 0 and b.lost_in == {"deliver": 1}


def test_released_goes_back_to_search_after_the_retract():
    b = TA.TidyArm()
    b.state, b.t_state = "drop", 0.0
    b._arm = {j: mars.ARM_HOME[j] for j in mars.ARM_JOINTS}
    b.step(_senses(b.p.drop_s + 0.1, holding=False))
    assert b.state == "retract" and b.delivered == 1
    b.step(_senses(b.p.drop_s + b.p.retract_s + 0.2))
    assert b.state == "search"


def test_asked_to_move_and_not_moving_unsticks_and_keeps_a_held_toy():
    b = TA.TidyArm()
    b.state, b.t_state = "deliver", 0.0
    b.est, b.goal_kind = (3.0, 0.0), "basket"
    b._q_lift = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])
    t = 0.0
    for _ in range(200):
        t += 0.02
        b.step(_senses(t, odom=(0.0, 0.0, 0.0), holding=True, speed=0.0))
        if b.state == "unstick":
            break
    assert b.state == "unstick"
    assert t == pytest.approx(b.p.stuck_s + 0.04, abs=0.08), (
        "the stuck clock must start when the DRIVE starts, not at construction")
    assert b.last[0] < 0.0                      # reversing
    for _ in range(200):
        t += 0.02
        b.step(_senses(t, holding=True, speed=-0.2))
        if b.state != "unstick":
            break
    assert b.state == "carry"                   # a held toy is a trip in progress


def test_the_arm_target_is_slewed_at_the_servos_rate():
    """`MarsDriver.set_arm` applies no limit, so the BRAIN must
    (`mars.MAX_TARGET_RATE_RAD_S`). Without this an `Intent.arm` is a 3 rad
    teleport and the block goes across the room."""
    b = TA.TidyArm()
    b._arm = {j: mars.ARM_HOME[j] for j in mars.ARM_JOINTS}
    b._arm["joint1"] = mars.ARM_HOME["joint1"] - 3.0
    step = mars.MAX_TARGET_RATE_RAD_S * C.CTRL_DT
    out = b.step(_senses(0.02))
    assert out.arm["joint1"] == pytest.approx(mars.ARM_HOME["joint1"] - step, abs=1e-9)
    out = b.step(_senses(0.04))
    assert out.arm["joint1"] == pytest.approx(mars.ARM_HOME["joint1"] - 2 * step, abs=1e-9)


def test_the_head_is_held_at_one_pitch_for_the_whole_run():
    b = TA.TidyArm()
    for t in (0.02, 0.5, 5.0):
        assert b.step(_senses(t)).head == (0.0, b.p.head_pitch, 0.0, 0.0)


def test_locate_places_a_floor_toy_from_a_PITCHED_camera():
    """The measured bug: a pitched lens's azimuth is not a horizontal
    bearing, and the duck's `cam_pitch - elevation` shortcut put the estimate
    11-22 mm to the side of a 40 mm block. Build a frame from a KNOWN
    base-frame point through the true camera geometry and check it comes back.

    **The frame is built at the BLOCK's own resting height**, not at
    `TidyArmParams.toy_z`, so the two sides of the comparison are not the same
    constant (4b: "a test that names the constant it is checking on BOTH
    sides agrees with itself"). That also makes this the case that pins
    `toy_z`: the ray is intersected with the plane the brain BELIEVES, so a
    height 6 mm low stretches every range by 9-10 mm — which is the whole
    margin a 58.8 mm jaw has on a 40 mm block."""
    b = TA.TidyArm()
    true_z = PICKABLE_KINDS["block"]["size"][2] / 2 + 0.001
    cam = np.array([b.p.cam_x, b.p.cam_y, 0.2438])
    pitch = b.p.head_pitch
    for spot in ((0.335, -0.053), (0.30, -0.10), (0.30, 0.10), (0.8, 0.25)):
        target = np.array([spot[0], spot[1], true_z])
        v = target - cam
        # ...into the camera's own PITCHED frame (the inverse of `_locate`).
        cp, sp = math.cos(pitch), math.sin(pitch)
        vx, vz = v[0] * cp - v[2] * sp, v[0] * sp + v[2] * cp
        bearing = math.atan2(v[1], vx)
        elev = math.atan2(vz, math.hypot(vx, v[1]))
        got = b._locate((0.0, 0.0, 0.0),
                        Detection("toy", "t0", bearing, elev, 0.05, 0.4, 0.9),
                        b.p.toy_z, _senses(0.0, _frame(0.0, cam_pitch=pitch)))
        assert got is not None
        assert math.hypot(got[0] - spot[0], got[1] - spot[1]) < 0.002, spot


def test_the_jaw_opens_wide_enough_for_a_block_on_its_diagonal():
    """A 40 mm cube is 56.6 mm across the diagonal and the playroom scatters
    toys at a random yaw, so the open gap has to clear the diagonal — 58.8 mm
    at the scripted pick's 0.60 rad did not, and the blades pushed the block."""
    assert TA.OPEN_RAD == pytest.approx(0.8727, abs=1e-4)
    diagonal = math.hypot(*PICKABLE_KINDS["block"]["size"][:2])
    assert diagonal == pytest.approx(0.0566, abs=0.001)


@needs_mars
def test_the_open_gap_clears_that_diagonal_on_the_model():
    from microduck_local.robots.mars_ik import ArmKinematics

    k = ArmKinematics(seed=0)
    q = k.home.copy()
    q[5] = TA.OPEN_RAD
    k.place(q)
    diagonal = math.hypot(*PICKABLE_KINDS["block"]["size"][:2])
    assert k.jaw_gap() > diagonal + 0.015
    q[5] = 0.60                                  # the scripted pick's opening
    k.place(q)
    assert k.jaw_gap() < diagonal + 0.005        # ...which barely clears it


class _StubKin:
    """Enough of `ArmKinematics` for a state-order test: every solve hits."""

    home = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])

    def solve_grasp(self, target, q6, start=None, restarts=3, point="grasp"):
        q = self.home.copy()
        q[5] = q6
        return 0.0, q

    def place(self, q):
        return np.asarray(q, float)

    def grasp_point(self):
        return np.array([0.3353, -0.0528, 0.021])

    def shell_point(self, radius, yaw, z):
        return np.array([0.3353, -0.0528, z])


# ============================================ the room, end to end

@needs_mars
def test_a_pick_in_the_room_ends_holding_and_a_place_ends_in_the_basket():
    """The two end-to-end cases the phase's bar names. One 300 s run is the
    benchmark (`eval-tidy --robot mars`); this is the shortest thing that
    exercises the whole chain — brain, IK, driver, claw, basket — and it is
    the case that fails first if any link is unwired."""
    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world_server import WorldState

    sc = make_playroom(seed=0, n=6, robot="mars", brain="tidy_arm", kinds=("block",))
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc, seed=0), sc
    w, d, brain = st.world, st.world.ducks["d0"], st.brains["d0"]
    assert brain.kind == "tidy_arm" and brain.world is w and brain.robot_id == "d0"
    cmd = np.zeros(3, np.float32)
    held_once = delivered_once = False
    while w.t < 90.0:
        st.drive(cmd, "auto")
        w.step()
        st.after_step()
        held_once |= d.holding is not None
        delivered_once |= w.tidy_score()["inBasket"] > 0
        if delivered_once:
            break
    assert held_once, "the claw never got hold of a block in 90 s"
    assert delivered_once, "no block reached the basket footprint in 90 s"
    assert d.falls == 0                          # a planar base cannot topple


@needs_mars
def test_the_brain_reads_the_baskets_rim_from_the_room():
    """`rim_z` is the SCENARIO's, not a constant: a deeper tray is a
    different place-over-the-rim target."""
    brain = REGISTRY.make("tidy_arm")
    assert brain.rim_z() == pytest.approx(Basket.rim)
    sc = make_playroom(seed=0, n=1, robot="mars", brain="tidy_arm", kinds=("block",))
    sc.basket = Basket(sc.basket.pos, sc.basket.size, 0.12)
    w = World(sc, seed=0)
    attach_world(brain, w, "d0")
    assert brain.rim_z() == pytest.approx(0.12)
