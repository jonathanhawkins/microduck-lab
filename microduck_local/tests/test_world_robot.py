"""Locks for a DRIVER-stepped body in a /sim room (`world/arena.WorldRobot`).

A MARS in a room is the first body world mode holds that is not a duck: no
walker, no beak, no fall, a controller instead of a policy, and a 360-degree
planar scan instead of an 8x8 ToF on a head. What is pinned here is every
seam that carries it — the spawn, the twist, the arm, the senses (including
the adapter that lets the DUCK's brains drive it), the rate the scanner runs
at, the board-contact instrument, and the body order the viewer will index
positionally.

Every case here was shown to FAIL on a planted break before it was kept
(`AGENTS.md`: "A/B new tests against planted regressions") — the table is in
the phase report.
"""

import math

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.robots import mars
from microduck_local.sensors.lidar import tof_column_bearings, tof_from_lidar
from microduck_local.world import Duck, Person, Scenario, Wall, World, WorldRobot
from microduck_local.world.scenario import load_scenario

pytestmark = pytest.mark.skipif(
    not (C.SCENE_WALK_XML.exists() and mars.mars_ready()),
    reason="microduck_rl checkout or MARS assets not present")

SCENARIOS = __import__("pathlib").Path(__file__).resolve().parents[1] / "scenarios"


def bare_room(hx: float = 2.0, hy: float = 1.5, spawn=(0.0, 0.0, 0.0),
              brain: str | None = None, name: str = "mr") -> Scenario:
    """A walled rectangle with one MARS in it and nothing else — no toys, no
    basket, no furniture, so the walls are the only thing the scan can see."""
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return Scenario(
        name=name, floor=(2 * hx + 0.5, 2 * hy + 0.5),
        walls=[Wall(pts[i], pts[(i + 1) % 4], 0.3, 0.02) for i in range(4)],
        ducks=[Duck("m0", spawn, None, "ideal", None, brain, robot="mars")])


def home_error(w: World, d) -> float:
    """max |q - ARM_HOME| over every driven joint of this MARS."""
    qadr = np.array([d.driver.adr[j][0] for j in mars.DRIVEN_JOINTS])
    home = np.array([mars.ARM_HOME[j] for j in mars.DRIVEN_JOINTS])
    return float(np.max(np.abs(w.data.qpos[qadr] - home)))


# -- the body in the room ----------------------------------------------------

def test_a_mars_spawns_where_the_scenario_puts_it_and_holds_its_arm():
    w = World(bare_room(spawn=(0.4, -0.3, 0.9)))
    d = w.ducks["m0"]
    assert isinstance(d, WorldRobot) and d.robot == "mars"
    x, y, yaw = d.driver.pose(w.data)
    assert (round(x, 6), round(y, 6), round(yaw, 6)) == (0.4, -0.3, 0.9)
    np.testing.assert_allclose(d.trunk_pos(w.data)[:2], [0.4, -0.3], atol=1e-6)
    assert d.yaw(w.data) == pytest.approx(0.9, abs=1e-6)
    # A planar base starts on the floor plane, with nothing to stand up from.
    assert d.trunk_pos(w.data)[2] == pytest.approx(0.0, abs=1e-9)
    assert home_error(w, d) == 0.0
    for _ in range(100):
        w.step()
    # 2 s of holding, uncommanded: the station keeper owns the base and the
    # position servo owns the arm. Both are Innate's, measured in
    # robots/mars_drive.py at 0.0034 rad of arm droop.
    assert home_error(w, d) < 0.01
    np.testing.assert_allclose(d.trunk_pos(w.data)[:2], [0.4, -0.3], atol=0.01)


def test_a_twist_drives_the_base_and_the_driver_runs_every_physics_step():
    """0.3 m/s for 2 s covers 0.6 m, straight, with the arm still home.

    The distance is what pins the CADENCE: the base's velocity loop is
    explicit, and `robots/mars_drive.py` measured that decimating it to the
    50 Hz control tick makes its gain 3.04 and sends the robot BACKWARDS.
    A run that lands on 0.6 m can only have been stepped per substep."""
    w = World(bare_room(hx=3.0, hy=3.0))
    d = w.ducks["m0"]
    for _ in range(int(2.0 / C.CTRL_DT)):
        d.set_cmd(w.data, (0.3, 0.0, 0.0))
        w.step()
    x, y, yaw = d.driver.pose(w.data)
    assert x == pytest.approx(0.6, rel=0.02), x
    assert abs(y) < 0.02 and abs(yaw) < 0.02
    assert d.heading_speed(w.data) == pytest.approx(0.3, abs=0.02)
    assert home_error(w, d) < 0.01
    # …and a turn is the other axis of the same command.
    for _ in range(int(1.0 / C.CTRL_DT)):
        d.set_cmd(w.data, (0.0, 0.0, 1.0))
        w.step()
    assert d.driver.pose(w.data)[2] == pytest.approx(1.0, abs=0.05)


def test_an_uncommanded_base_stops_on_innates_watchdog():
    """A command is a LEASE, not a setting — unlike a duck's `twist_cmd`,
    which persists. `CMD_VEL_TIMEOUT_S` after the last `set_cmd` the base is
    stopped, which is the behaviour of the real driver node."""
    from microduck_local.robots.mars_drive import CMD_VEL_TIMEOUT_S
    w = World(bare_room(hx=3.0, hy=3.0))
    d = w.ducks["m0"]
    d.set_cmd(w.data, (0.5, 0.0, 0.0))
    for _ in range(int(0.3 / C.CTRL_DT)):
        w.step()
    assert d.heading_speed(w.data) > 0.3
    for _ in range(int((CMD_VEL_TIMEOUT_S + 0.2) / C.CTRL_DT)):
        w.step()
    assert abs(d.heading_speed(w.data)) < 0.02


def test_an_arm_intent_moves_a_joint_and_a_gaze_moves_the_head():
    from microduck_local.brain.runtime import Intent
    w = World(bare_room())
    d = w.ducks["m0"]
    want = mars.ARM_HOME["joint4"] + 0.4
    w.apply_intent(d, Intent(arm={"joint4": want}))
    for _ in range(50):
        w.step()
    q4 = w.data.qpos[d.driver.adr["joint4"][0]]
    assert q4 == pytest.approx(want, abs=0.02), q4
    # The gaze: the duck's `head_pitch` is slot 1 and POSITIVE IS DOWN
    # (contract.py's axis table); MARS's `joint_head` has axis "0 -1 0", so
    # positive is UP. A brain asking to look down must move the head down.
    d.set_cmd(w.data, (0.0, 0.0, 0.0), head=(0.0, 0.3, 0.0, 0.0))
    for _ in range(50):
        w.step()
    qh = float(w.data.qpos[d.driver.adr[mars.HEAD_JOINT][0]])
    assert qh == pytest.approx(-0.3, abs=0.02), qh
    # …and the arm target survived the gaze write. Both go through one
    # `set_arm` call, which is absolute over the whole driven set.
    assert w.data.qpos[d.driver.adr["joint4"][0]] == pytest.approx(want, abs=0.02)
    # A duck's look-down of 0.6 rad is past this head's travel, and the
    # driver clamps to the joint's own MJCF range rather than scissoring it.
    d.set_cmd(w.data, (0.0, 0.0, 0.0), head=(0.0, 0.6, 0.0, 0.0))
    lo, hi = w.model.jnt_range[w.model.joint("m0/" + mars.HEAD_JOINT).id]
    assert d.arm_targets()[mars.HEAD_JOINT] == pytest.approx(float(lo), abs=1e-9)
    assert (round(float(lo), 4), round(float(hi), 4)) == (-0.3491, 0.3491)


def test_a_mars_never_falls_and_is_never_respawned():
    """`WorldRobot.fallen` is False by construction (`mars.add_planar_base`:
    a wheeled chassis has no attitude to lose), so the World's fall, get-up
    and respawn machinery never fires on one. Shoved hard, it stays up."""
    w = World(bare_room(hx=3.0, hy=3.0), getup_s=2.0)
    d = w.ducks["m0"]
    for k in range(200):
        if k < 20:
            w.data.xfrc_applied[d.root_body, :3] = (40.0, 25.0, 60.0)
        else:
            w.data.xfrc_applied[d.root_body, :3] = 0.0
        w.step()
        assert not d.fallen(w.data)
    assert d.falls == 0 and d.episodes == 1 and d.down_until == -1.0


# -- the senses --------------------------------------------------------------

def test_a_wall_ahead_is_in_the_scan_and_in_the_adapted_tof_at_the_right_bin():
    """One wall 1 m ahead, read three ways that must agree.

    The scan measures from the LASER, 76.4 mm behind the base origin, so it
    reports the wall FURTHER than it is; the ToF adapter's whole job is to
    hand a base-framed bearing and range to a brain that assumes one.
    """
    sc = Scenario(name="wall1", floor=(8.0, 8.0),
                  walls=[Wall((1.0, -2.0), (1.0, 2.0), 0.4, 0.02)],
                  ducks=[Duck("m0", (0.0, 0.0, 0.0), None, "ideal", None, robot="mars")])
    w = World(sc)
    d = w.ducks["m0"]
    for _ in range(10):
        w.step()
    frame = d.lidar.last
    assert frame is not None and frame.ranges.shape == (360,)
    # Ray 0 is on +x (Innate's CCW convention, `planar_fan(ccw=True)`).
    assert frame.angles[0] == pytest.approx(0.0, abs=1e-6)
    face = 1.0 - 0.02 / 2                       # the wall's inner face
    offset = -float(frame.mount_pos[0])          # the laser sits BEHIND the origin
    assert offset == pytest.approx(0.0764, abs=0.001)
    assert float(frame.ranges[0]) == pytest.approx(face + offset, abs=0.005)

    tof = tof_from_lidar(frame, footprint_m=mars.FOOTPRINT_M)
    assert tof.depth_mm.shape == (8, 8) and tof.valid.all()
    # Every row of a column is the same number: a planar scan has no
    # elevation, which is the adapter's first documented cost.
    for r in range(1, 8):
        np.testing.assert_array_equal(tof.depth_mm[r], tof.depth_mm[0])
    # The two centre columns straddle the nose, so they read the wall's
    # PERPENDICULAR distance from the BASE origin — the offset removed.
    for c in (3, 4):
        assert tof.depth_mm[0, c] / 1000.0 == pytest.approx(face, abs=0.01), c
    # …and an outer column reads the same wall at its own bearing, which is
    # further away by 1/cos(bearing) and NOT the same number.
    edges = tof_column_bearings()
    mid = 0.5 * (edges[0] + edges[1])
    assert tof.depth_mm[0, 0] / 1000.0 == pytest.approx(face / math.cos(mid), rel=0.04)
    assert tof.depth_mm[0, 0] > tof.depth_mm[0, 3] + 20
    # Nothing behind: the columns are the FRONT sector only, and a ToF frame
    # from here carries no mount pose, so `tof_hits_3d` falls back to columns
    # rather than placing a flat scan at eight elevations.
    assert tof.mount_pos is None and tof.mount_rot is None and tof.dirs_local is None


def test_the_scan_arrives_at_the_devices_own_6_hz_with_honest_ages():
    w = World(bare_room())
    d = w.ducks["m0"]
    assert d.tof is None and d.lidar is not None and d.detector is None
    stamps, ages = [], []
    for _ in range(int(2.0 / C.CTRL_DT)):
        w.step()
        f, age = w.senses_tof(d)
        assert f is not None
        stamps.append(round(d.lidar.last.t, 4))
        ages.append(age)
    uniq = sorted(set(stamps))
    # 6 Hz over 2 s is 12 scans, and the age a brain gates on is the SCAN's,
    # up to one period (167 ms) — not zeroed by the adaptation.
    assert 11 <= len(uniq) <= 13, uniq
    gaps = np.diff(uniq)
    # `maybe_scan` schedules off the 1/6 s GRID and is polled on a 20 ms
    # tick, so the gaps alternate 0.16 / 0.18 around the period rather than
    # landing on it — which is the point of scheduling from the grid: they
    # cannot accumulate into a slower rate.
    assert np.all(np.abs(gaps - 1 / 6.0) <= C.CTRL_DT + 1e-9), gaps
    assert float(np.mean(gaps)) == pytest.approx(1 / 6.0, abs=0.005)
    assert max(ages) > 0.1 and max(ages) <= 1 / 6.0 + 1e-9


def test_the_scan_does_not_see_the_turret_it_is_bolted_to():
    """`LidarSensor` excludes the mount's PARENT, which on MARS is
    `base_link` — the body the turret box modelling the scanner's own housing
    belongs to. Without it every ray returns 4-10 cm and the frame is full,
    plausible and useless (`sensors/lidar.py`'s measurement)."""
    w = World(bare_room(hx=3.0, hy=3.0))
    d = w.ducks["m0"]
    assert d.lidar.exclude_body == "m0/base_link"
    for _ in range(10):
        w.step()
    r = d.lidar.last.ranges
    # The only thing inside 0.15 m is the folded arm's own elbow, and it is a
    # handful of rays, not all 360.
    assert int(np.sum(r < 0.2)) < 20, int(np.sum(r < 0.2))
    assert float(np.median(r)) > 1.0


def test_the_camera_sees_a_person_from_inside_its_own_head():
    """The detector mounts on a BODY here, and its housing is that body's
    PARENT. MEASURED before the fix: `head_camera_left` is a marker sphere
    inside `head`'s 113 x 121 x 36 mm collision box, so every ray hit
    `head_body` at 0.0206 m and a `follow` brain sat in `search` for 60 s."""
    sc = Scenario(name="seeme", floor=(8.0, 8.0),
                  ducks=[Duck("m0", (0.0, 0.0, 0.0), None, None, "ideal", robot="mars")],
                  persons=[Person("p0", (1.2, 0.0), math.pi, speed=0.0)])
    w = World(sc)
    d = w.ducks["m0"]
    assert w.model.body(d.detector.exclude_body).name == "m0/head"
    # A non-duck body is NOT a detector target, and the reason is a silent
    # failure rather than a gap: `DETECT_CLASSES` has no class for a wheeled
    # base, and a `Target` built on a body id the model does not have reads
    # `xpos[-1]` — the LAST body in the whole model — and reports a confident
    # detection of something somewhere else entirely.
    assert [t.name for t in d.detector.targets] == ["p0"]
    assert all(t.body >= 0 or t.pos is not None for t in d.detector.targets)
    seen, others = [], []
    for _ in range(int(2.0 / C.CTRL_DT)):
        w.step()
        if d.detector.last is not None:
            seen += [x for x in d.detector.last.detections if x.cls == "person"]
            others += [x for x in d.detector.last.detections if x.cls != "person"]
    assert len(seen) > 5, len(seen)
    assert max(abs(x.bearing) for x in seen) < 0.2
    assert not others, others
    # Innate's calibrated lens, derived from fx/fy and not typed as degrees.
    assert d.detector.spec.fov_h_deg == pytest.approx(115.95, abs=0.05)
    assert d.detector.spec.fov_v_deg == pytest.approx(83.84, abs=0.05)


# -- the brains, unchanged ---------------------------------------------------

def test_the_ducks_wander_brain_turns_a_mars_away_from_a_wall():
    """`brain/controllers.py` is NOT edited for a wheeled body: `wander`
    reads a 64-zone ToF, and the adapter gives it one. Facing a wall 1.2 m
    off at cruise, it must stop cruising and turn within 3 s."""
    w = World(bare_room(hx=1.2, hy=2.0, spawn=(0.2, 0.0, 0.0), brain="wander"))
    d = w.ducks["m0"]
    from microduck_local.brain import REGISTRY
    brain = REGISTRY.make("wander")
    from microduck_local.world_server import WorldState
    st = WorldState(load_infer=None)
    st.world, st.scenario = w, w.scenario
    st.brains = {"m0": brain}
    states, wz, closest = set(), 0.0, 9.0
    for _ in range(int(3.0 / C.CTRL_DT)):
        st.drive(np.zeros(3, np.float32), "auto")
        w.step()
        states.add(brain.state)
        wz = max(wz, abs(float(d.twist_cmd[2])))
        closest = min(closest, 1.19 - float(d.trunk_pos(w.data)[0]))   # to the wall's inner face
    assert "cruise" in states, states
    assert {"steer", "spin"} & states, states
    # A turn was actually COMMANDED, not just a state the brain reported —
    # and it reached the base, which is the seam this test is about.
    assert wz > 0.1, wz
    # …and it kept off the boards, and off the wall, while doing it.
    assert w.wall_bumps["m0"] == 0, w.wall_ticks["m0"]
    assert closest > 0.05, closest


def test_the_board_instrument_counts_walls_and_not_the_floor():
    """`World.wall_bumps` is an instrument, not the `Senses.bumped` channel:
    it counts contacts with STATIC scenery, floor excluded, as separate
    episodes. A duck standing on the floor must read 0."""
    w = World(bare_room(hx=0.6, hy=2.0, spawn=(0.0, 0.0, 0.0)))
    d = w.ducks["m0"]
    assert w.wall_bumps == {"m0": 0} and w.wall_ticks == {"m0": 0}
    # Drive into the wall at 0.58 m and hold there.
    for _ in range(int(4.0 / C.CTRL_DT)):
        d.set_cmd(w.data, (0.5, 0.0, 0.0))
        w.step()
    assert w.wall_bumps["m0"] == 1, w.wall_bumps           # one episode, not one a tick
    assert w.wall_ticks["m0"] > 20, w.wall_ticks           # …which lasted
    assert not w.bumped(d), "a wall is not a body: the brain's bump sense is untouched"
    w.reset()
    assert w.wall_bumps["m0"] == 0 and w.wall_ticks["m0"] == 0

    # A duck alone in a room touches the floor every step and no boards.
    duck = World(Scenario(name="df", floor=(6.0, 6.0),
                          walls=[Wall((2.0, -2.0), (2.0, 2.0), 0.3)],
                          ducks=[Duck("d0", (0.0, 0.0, 0.0), None, None, None)]))
    for _ in range(50):
        duck.step()
    assert duck.wall_bumps == {"d0": 0}


# -- the wire ----------------------------------------------------------------

def test_the_body_list_is_the_viewers_scene_list_name_for_name():
    """The stream maps a robot's poses onto `GET /scene?robot=mars`'s body
    list POSITIONALLY, and `WorldRobot.scene_bodies` derives that order from
    the COMPOSED model rather than parsing 7 MB of STL. That the two agree is
    the claim, so it is a test — `tests/test_arena.py` pins the duck's the
    same way against `scene_model()`."""
    w = World(bare_room())
    d = w.ducks["m0"]
    assert d.scene_bodies() == mars.visual_scene()["bodies"]
    poses = d.bodies_payload(w.data)
    assert len(poses) == len(d.scene_bodies()) and len(poses[0]) == 7
    assert poses[0] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]      # world first
    # And the subtree the World slices is the same set of bodies.
    s = w.duck_bodies["m0"]
    assert s.stop - s.start == len(poses) - 1


def test_the_two_tracked_mars_scenarios_load_and_round_trip():
    from microduck_local.world.scenario import validate_scenario
    for name, brain in (("mars-playroom", "wander"), ("mars-follow", "follow")):
        sc = load_scenario(SCENARIOS / f"{name}.json")
        assert sc.name == name
        assert [d.robot for d in sc.ducks] == ["mars"]
        assert sc.ducks[0].brain == brain and sc.ducks[0].tof == "datasheet"
        assert validate_scenario(sc.to_dict()) == sc
        w = World(sc, seed=0)
        for _ in range(5):
            w.step()
        assert isinstance(w.ducks["d0"], WorldRobot)
    # The follow room's person is a CAPSULE on purpose: every published
    # follow number was measured against one.
    follow = load_scenario(SCENARIOS / "mars-follow.json")
    assert [p.kind for p in follow.persons] == ["capsule"]


def test_the_body_declares_its_frames_and_a_body_that_does_not_says_so():
    """`Body.frames()` (`robots/body.RobotFrames`) is the body's own
    knowledge, not a table in the arena — `docs/mars-roadmap.md` §6.5. Two
    halves: MARS declares the names its own module already carries, and a
    body that has NOT declared them fails at construction naming `frames()`
    rather than resolving -1 and slicing the wrong subtree."""
    from microduck_local.robots.body import BodyBase, RobotFrames, conforms

    f = mars.MARS.frames()
    assert isinstance(f, RobotFrames)
    assert f.base == mars.BASE_BODY == "base_link"
    assert f.head_pitch_joint == mars.HEAD_JOINT == "joint_head"
    # Down is down: the duck's head_pitch is +y (positive = down) and MARS's
    # joint_head is axis "0 -1 0" (positive = up), so the sign must invert.
    assert f.head_pitch_sign == -1.0

    # The generic answer RAISES and names the method, which is `BodyBase`'s
    # own rule for a question with no body-agnostic answer — and it is still
    # a `Body`, because a body that is never DRIVEN never needs frames.
    assert conforms(BodyBase(id="x", joint_names=(), default_pose=None, obs_dim=1)) == ()
    with pytest.raises(NotImplementedError, match="frames"):
        BodyBase(id="x", joint_names=(), default_pose=None, obs_dim=1).frames()

    # …and that is what a room reports when it is asked to drive one.
    class _Mute(mars.MarsBody):
        def frames(self):
            return BodyBase.frames(self)

    body = _Mute(id="mars", joint_names=mars.ARM_JOINTS, default_pose=mars.DEFAULT_POSE,
                 obs_dim=mars.OBS_DIM, scene_fn=mars.scene_xml,
                 stand_keyframe=mars.HOME_KEY)
    from microduck_local.world import arena
    assert not hasattr(arena, "ROBOT_FRAMES"), "the per-robot table is gone"
    with pytest.raises(NotImplementedError, match="frames"):
        arena.WorldRobot("m0", bare_room().ducks[0], body, World(bare_room()).model)


def test_the_camera_housing_is_the_body_the_detector_already_ignores():
    """Which body a viewer must not draw from this robot's own camera.

    MEASURED across the whole head-pitch range: MARS's `head` is 1.5-2.8 cm
    from its lens while inside the frame, and the /sim inset's near plane is
    3 cm — so the shell is clipped while everything is still and swings into
    view the moment the drawn pose lags the captured one. What must NOT be
    hidden is the arm: `link5` is in the same frame at 15-26 cm, and a real
    MARS does see its own claw.

    The index is the detector's OWN `exclude_body`, which is the point of
    the method: the sensor already refuses to detect its housing, and a
    viewer hiding a different body would be a second answer to one question.
    """
    from microduck_local.world_server import resolve_scenario
    w = World(resolve_scenario("mars-follow"))
    robot = next(iter(w._robots))
    idx = robot.camera_housing_body()
    names = robot.scene_bodies()
    assert 0 <= idx < len(names)
    assert names[idx] == "head"
    # …and it IS the detector's own exclusion, not a name written twice.
    excluded = robot.model.body(int(robot.detector.exclude_body)).name
    assert excluded == robot.prefix + names[idx]
    # The arm keeps its place in the list and is never the answer.
    assert "link5" in names and names[idx] != "link5"


def test_a_robot_with_no_detector_asks_the_viewer_to_hide_nothing():
    """-1 is "draw everything" — what an older lab sends and what a blind
    robot means. A viewer that read it as an index would hide body -1."""
    from microduck_local.world_server import resolve_scenario
    w = World(resolve_scenario("mars-follow"))
    robot = next(iter(w._robots))
    robot.detector = None
    assert robot.camera_housing_body() == -1
