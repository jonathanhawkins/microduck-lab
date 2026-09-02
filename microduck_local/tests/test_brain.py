"""The wander controller (roadmap 2.3): decisions from synthetic ToF frames,
dropouts treated as unknown, turn memory, the stuck spin, and one end-to-end
check that it steers a real duck away from a wall it would otherwise walk into."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import Wander, WanderParams, wander_from_tof


def frame(fill_mm: int, **cols) -> np.ndarray:
    f = np.full((8, 8), fill_mm, np.uint16)
    for k, v in cols.items():          # c3=400 → column 3 reads 400 mm in the counted rows
        f[2:7, int(k[1:])] = v
    return f


def test_clear_ahead_cruises():
    assert wander_from_tof(frame(3000)) == (0.3, 0.0, 0.0)
    assert wander_from_tof(np.zeros((8, 8), np.uint16)) == (0.3, 0.0, 0.0)   # nothing reported = nothing known


def test_wall_ahead_slows_and_turns_toward_the_open_side():
    # Right half near, left half far: turn LEFT (+wz), slower.
    f = frame(3000, c4=500, c5=500, c6=500, c7=500)
    vx, vy, wz = wander_from_tof(f)
    assert 0 < vx < 0.3 and wz == pytest.approx(0.8)
    f = frame(3000, c0=500, c1=500, c2=500, c3=500)
    vx, vy, wz = wander_from_tof(f)
    assert 0 < vx < 0.3 and wz == pytest.approx(-0.8)


def test_boxed_in_stops_and_spins():
    vx, vy, wz = wander_from_tof(frame(200))
    assert vx == 0.0 and abs(wz) == 1.0


def test_sky_rows_and_dropouts_are_ignored():
    f = frame(3000)
    f[0:2, :] = 150                 # something "near" only in the top rows: ignored
    assert wander_from_tof(f) == (0.3, 0.0, 0.0)
    f = frame(3000, c3=250)
    valid = np.ones((8, 8), bool)
    valid[:, 3] = False             # that column is all dropouts: unknown, not near
    assert wander_from_tof(f, valid) == (0.3, 0.0, 0.0)
    assert wander_from_tof(f)[0] == 0.0    # …but trusted, it stops


def test_turn_memory_and_stuck_spin():
    w = Wander(WanderParams(), stuck_s=1.0, unstick_s=0.5)
    assert w.decide(None, None, 0.0) == (0.0, 0.0, 0.0) and w.state == "blind"
    f = frame(3000, c4=500, c5=500, c6=500, c7=500)
    assert w.decide(f, None, 0.1)[2] > 0 and w.state == "steer"
    # A nearly symmetric frame keeps the remembered direction.
    sym = frame(3000, c2=500, c3=500, c4=500, c5=500)
    assert w.decide(sym, None, 0.2)[2] > 0
    # Forward command, no progress for > stuck_s → an unstick spin.
    clear = frame(3000)
    for k in range(12):
        out = w.decide(clear, None, 1.0 + k * 0.1, speed=0.0)
    assert w.state == "unstick" and out[0] == 0.0 and abs(out[2]) == 1.0
    assert w.decide(clear, None, 2.7, speed=0.0)[0] == 0.3 and w.state == "cruise"


@pytest.mark.skipif(not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")
def test_wander_keeps_a_walking_duck_off_the_wall():
    import onnxruntime as ort

    from microduck_local.world import Duck, Scenario, Wall, World
    path = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies" / "alpha_walking.onnx"
    if not path.exists():
        pytest.skip("upstream policies not checked out")
    sess = ort.InferenceSession(str(path))
    nm = sess.get_inputs()[0].name
    sc = Scenario(name="wall", floor=(6, 6),
                  walls=[Wall((1.0, -2.0), (1.0, 2.0), 0.6)],
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal")])
    world = World(sc, infer_for={"d0": lambda o: sess.run(None, {nm: o[None]})[0][0].astype(np.float32)})
    d = world.ducks["d0"]
    w = Wander()
    min_gap = 9.0
    for _ in range(int(6.0 / C.CTRL_DT)):
        tof = d.tof.last
        cmd = w.decide(None if tof is None else tof.depth_mm, None if tof is None else tof.valid,
                       world.t, d.heading_speed(world.data))
        d.set_cmd(world.data, cmd)
        world.step()
        min_gap = min(min_gap, 1.0 - float(d.trunk_pos(world.data)[0]))
    assert d.falls == 0
    assert min_gap > 0.15, min_gap            # never got its beak on the wall
    assert w.state in ("steer", "cruise", "spin")


@pytest.mark.skipif(not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")
def test_follow_brain_tracks_a_walking_person():
    import onnxruntime as ort

    from microduck_local.brain import Follow, Senses
    from microduck_local.world import Duck, Person, Scenario, World
    path = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies" / "alpha_walking.onnx"
    if not path.exists():
        pytest.skip("upstream policies not checked out")
    sess = ort.InferenceSession(str(path))
    nm = sess.get_inputs()[0].name
    sc = Scenario(name="fm", floor=(8, 8),
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal", "ideal")],
                  persons=[Person("p0", (1.0, 0.0), 0.0, path=[(2.5, 0.0), (2.5, 1.0)], speed=0.2)])
    world = World(sc, infer_for={"d0": lambda o: sess.run(None, {nm: o[None]})[0][0].astype(np.float32)})
    d, p = world.ducks["d0"], world.persons["p0"]
    brain = Follow()
    gaps, states = [], set()
    for _ in range(int(12.0 / C.CTRL_DT)):
        tof, det = d.tof.last, d.detector.last
        intent = brain.step(Senses(t=world.t, tof=tof, tof_age=None if tof is None else world.t - tof.t,
                                   det=det, det_age=None if det is None else world.t - det.t,
                                   speed=d.heading_speed(world.data)))
        d.set_cmd(world.data, intent.twist)
        world.step()
        states.add(brain.state)
        gaps.append(float(np.hypot(p.x - d.trunk_pos(world.data)[0], p.y - d.trunk_pos(world.data)[1])))
    assert d.falls == 0
    assert "approach" in states
    # The person walked ~2.4 m away and around a corner; the duck kept up.
    assert gaps[-1] < 1.6, gaps[-1]
    assert min(gaps[-100:]) > 0.35            # …without walking into it


def test_chase_brain_walks_at_a_tracked_ball_and_searches_left_without_one():
    """The soccer brain: a tracked ball ahead means walk at it (the walk is
    the kick); none means a left search turn, kicked when the gait is cold."""
    from microduck_local.brain import REGISTRY
    from microduck_local.brain.gait import TURN_KICK
    from microduck_local.brain.runtime import Senses
    from microduck_local.sensors.detector import Detection, DetectionFrame
    b = REGISTRY.make("chase")
    assert b.kind == "chase"
    dip = b.step(Senses(t=0.0, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert dip.twist == (0.0, 0.0, 0.0) and dip.note == "search" and dip.head[1] > 0.3   # a search opens with a look down
    none = b.step(Senses(t=0.7, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert none.twist == (TURN_KICK, 0.0, 1.0) and none.note == "search" and none.head == (0.0, 0.0, 0.0, 0.0)
    fr = DetectionFrame(0.1, [Detection("ball", "ball0", 0.2, -0.3, 0.05, 1.0, 0.9)])
    seen = b.step(Senses(t=0.1, det=fr, det_age=0.0, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert seen.note == "chase" and seen.twist[0] > 0.3 and seen.twist[2] > 0


def test_chase_brain_keeps_off_the_other_duck():
    """Body-aware avoidance on the pitch: a tracked duck close ahead means
    turn AWAY from it, never into it; one touching means stand still;
    one far away changes nothing (measured: 5 of 7 falls over 4 traced runs
    were this duck turning in place against the other's body)."""
    from microduck_local.brain import REGISTRY
    from microduck_local.brain.runtime import Senses
    from microduck_local.sensors.detector import Detection, DetectionFrame

    def duck_at(bearing, rng):
        return Detection("duck", "d1", bearing, 0.0, 2 * 0.1 / rng, rng, 0.9)

    b = REGISTRY.make("chase")
    odom = (0.0, 0.0, 0.0)
    # Far: the search goes on as if it were alone.
    far = b.step(Senses(t=0.0, det=DetectionFrame(0.0, [duck_at(0.2, 1.5)]), det_age=0.0, speed=0.0, odom=odom))
    assert far.note == "search"
    # Close on the LEFT (+bearing): turn right, and no forward creep with it near the nose.
    left = b.step(Senses(t=0.1, det=DetectionFrame(0.1, [duck_at(0.3, 0.35)]), det_age=0.0, speed=0.0, odom=odom))
    assert left.note == "avoid" and left.twist == (0.0, 0.0, -1.0)
    # Close on the right, well off the nose: turn left — cold gait, so the kick comes with it.
    right = b.step(Senses(t=0.2, det=DetectionFrame(0.2, [duck_at(-0.8, 0.35)]), det_age=0.0, speed=0.0, odom=odom))
    assert right.note == "avoid" and right.twist[2] == 1.0 and right.twist[0] > 0
    # Touching: stand.
    touch = b.step(Senses(t=0.3, det=DetectionFrame(0.3, [duck_at(0.1, 0.12)]), det_age=0.0, speed=0.0, odom=odom))
    assert touch.note == "avoid" and touch.twist == (0.0, 0.0, 0.0)
    # A ball in view does not override a duck on top of us — the line-up waits.
    both = DetectionFrame(0.4, [duck_at(0.1, 0.15), Detection("ball", "ball0", 0.0, -0.3, 0.05, 0.5, 0.9)])
    wait = b.step(Senses(t=0.4, det=both, det_age=0.0, speed=0.0, odom=odom))
    assert wait.note == "avoid" and wait.twist == (0.0, 0.0, 0.0) and b.spot is None
    # Gone (the track ages past 0.6 s): back to the ball.
    b.step(Senses(t=0.5, det=DetectionFrame(0.5, [Detection("ball", "ball0", 0.0, -0.3, 0.05, 0.5, 0.9)]),
                  det_age=0.0, speed=0.0, odom=odom))
    resumed = b.step(Senses(t=1.1, det=DetectionFrame(1.1, [Detection("ball", "ball0", 0.0, -0.3, 0.05, 0.5, 0.9)]),
                            det_age=0.0, speed=0.0, odom=odom))
    assert resumed.note in ("lineup", "chase")


def test_follow_lead_term_turns_toward_where_the_target_is_going():
    """`k_lead`: a target whose bearing is drifting left gets more turn
    than its bearing alone asks for; with it off the two are equal."""
    from microduck_local.brain import Follow
    from microduck_local.brain.controllers import FollowParams
    from microduck_local.brain.runtime import Senses
    from microduck_local.sensors.detector import Detection, DetectionFrame

    def frames(brain):
        out = []
        for k in range(4):
            t = 0.1 * k
            det = DetectionFrame(t, [Detection("person", "p0", 0.05 + 0.05 * k, 0.0, 0.3, 1.4, 0.9)])
            out.append(brain.step(Senses(t=t, det=det, det_age=0.0, speed=0.3, odom=(0.0, 0.0, 0.0))))
        return out
    plain = frames(Follow(FollowParams(k_turn=3.0, k_lead=0.0)))
    lead = frames(Follow(FollowParams(k_turn=3.0, k_lead=0.5)))
    assert plain[-1].note == "approach" and lead[-1].note == "approach"
    assert 0 < plain[-1].twist[2] < 1.0                           # bearing ~0.2 rad: not saturated
    assert lead[-1].twist[2] > plain[-1].twist[2] + 0.1           # …moving +0.5 rad/s: turn ahead of it


def test_closing_watch_dodges_what_walks_at_it():
    """The ToF clearance ahead shrinking faster than the duck's own walk is
    something coming at it: a turn toward the freer side, then a walk. A
    wall approached at the walking speed is not."""
    from microduck_local.brain.controllers import ClosingParams, ClosingWatch
    from microduck_local.sensors.tof import TofFrame

    def tof(t, ahead_mm, right_mm=4000, left_mm=4000):
        f = np.full((8, 8), 4000, np.uint16)
        f[2:5, 3:5] = ahead_mm
        f[2:5, 0:3] = left_mm
        f[2:5, 5:8] = right_mm
        return TofFrame(t=t, depth_mm=f, valid=np.ones((8, 8), bool))
    p = ClosingParams()
    # A person at 1.4 m walking in at 0.5 m/s while the duck stands; the right side is blocked.
    w = ClosingWatch(p)
    out = [w.step(tof(0.066 * k, int(1400 - 500 * 0.066 * k), right_mm=300), 0.066 * k, 0.0) for k in range(20)]
    assert out[0] is None and w.closing > 0.4 and w.count == 1
    first = next(i for i, v in enumerate(out) if v is not None)
    assert out[first][2] > 0                                          # turning LEFT, the open side
    assert all(v[2] > 0 for v in out[first:first + 10] if v is not None and v[0] < p.speed)   # the turn phase
    # ...then the walk phase, then nothing until the cooldown is over.
    late = w.step(tof(1.5, 700), 1.5, 0.0)
    assert late == (p.speed, 0.0, 0.0)
    assert w.step(tof(3.0, 700), 3.0, 0.0) is None and w.count == 1
    # Left blocked: the turn goes right.
    w2 = ClosingWatch(p)
    out2 = [w2.step(tof(0.066 * k, int(1400 - 500 * 0.066 * k), left_mm=300), 0.066 * k, 0.0) for k in range(20)]
    assert any(v is not None and v[2] < 0 for v in out2) and not any(v is not None and v[2] > 0 for v in out2)
    # The duck walking at a wall at 0.3 m/s: the clearance shrinks at its own speed - nothing to dodge.
    w3 = ClosingWatch(p)
    out3 = [w3.step(tof(0.066 * k, int(1400 - 300 * 0.066 * k)), 0.066 * k, 0.3) for k in range(20)]
    assert all(v is None for v in out3) and abs(w3.closing) < 0.1 and w3.count == 0


def test_follow_dodges_when_the_person_walks_at_it():
    """The scripted follow holds the band with a stop alone; with the
    person closing on it the closing watch turns that into a dodge."""
    from microduck_local.brain import Follow, FollowParams, Senses
    from microduck_local.sensors.detector import Detection, DetectionFrame
    from microduck_local.sensors.tof import TofFrame

    def run(avoid):
        b = Follow(FollowParams(avoid=avoid))
        notes = []
        for k in range(20):
            t = 0.066 * k
            rng = 1.4 - 0.5 * t
            f = np.full((8, 8), 4000, np.uint16)
            f[2:5, 3:5] = int(rng * 1000)
            det = DetectionFrame(t, [Detection("person", "p0", 0.0, 0.0, 0.3, rng, 0.9)])
            it = b.step(Senses(t=t, tof=TofFrame(t=t, depth_mm=f, valid=np.ones((8, 8), bool)), tof_age=0.0,
                               det=det, det_age=0.0, speed=0.0, odom=(0.0, 0.0, 0.0)))
            notes.append((it.note, it.twist))
        return notes
    with_ = run(True)
    assert any(n == "dodge" and tw[2] != 0.0 for n, tw in with_)
    without = run(False)
    assert not any(n == "dodge" for n, _ in without)
