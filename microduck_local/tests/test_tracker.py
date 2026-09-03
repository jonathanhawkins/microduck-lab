import math
"""The tracker over detection frames (roadmap 1.3) and the gait facts every
brain shares (brain/gait.py): ids persist across frames, a track coasts
through misses with its bearing turning with the body, a one-frame ghost
never confirms, and a cold gait gets the forward kick on a turn."""

import math

from microduck_local.brain.gait import COLD_AFTER_S, TURN_KICK, GaitWatch, turn
from microduck_local.brain.runtime import Senses
from microduck_local.brain.tracker import Tracker
from microduck_local.sensors.detector import Detection, DetectionFrame


def det(cls, bearing, rng, name="", conf=0.8):
    return Detection(cls, name, bearing, 0.0, 2 * math.atan(0.2 / rng), rng, conf)


def test_tracks_persist_coast_with_the_body_and_ignore_ghosts():
    tr = Tracker()
    tr.update(DetectionFrame(0.0, [det("person", 0.30, 1.0, "p0")]), 0.0, yaw=0.0)
    tr.update(DetectionFrame(0.1, [det("person", 0.32, 1.05, "p0"), det("duck", -1.0, 2.0, "")]), 0.1, yaw=0.0)
    person = tr.best("person", 0.1)
    assert person is not None and person.hits == 2 and person.name == "p0" and person.id == 1
    assert tr.best("duck", 0.1) is None                      # one sighting, not confirmed
    # The body turns +0.5 rad with no new frame: the remembered bearing turns the other way.
    tr.update(None, 0.2, yaw=0.5)
    same = tr.best("person", 0.2)
    assert same is not None and same.id == 1 and same.age(0.2) > 0.05
    assert abs(same.bearing - (0.31 - 0.5)) < 0.05
    # A detection near the coasted bearing re-associates with the SAME id.
    tr.update(DetectionFrame(0.3, [det("person", -0.2, 1.1, "p0")]), 0.3, yaw=0.5)
    again = tr.best("person", 0.3)
    assert again is not None and again.id == 1 and again.hits == 3 and again.age(0.3) == 0.0
    # A second person far off in bearing is a new track, not a jump of the old one.
    tr.update(DetectionFrame(0.4, [det("person", -0.2, 1.1, "p0"), det("person", 1.2, 1.5, "p1")]), 0.4, yaw=0.5)
    ids = {t.id: t.name for t in tr.tracks if t.cls == "person"}
    assert ids == {1: "p0", 3: "p1"}                      # 2 was the duck ghost
    # Nothing for longer than the coast window: the track dies.
    tr.update(None, 0.4 + tr.p.coast_s + 0.1, yaw=0.5)
    assert tr.best("person", 3.0) is None and tr.tracks == []


def test_gait_watch_and_turn_kick():
    g = GaitWatch()
    assert g.update(Senses(t=0.0, speed=0.0, odom=(0.0, 0.0, 0.0))) is True
    # Walking: warm. Standing still for COLD_AFTER_S: cold again.
    assert g.update(Senses(t=0.1, speed=0.3, odom=(0.0, 0.0, 0.0))) is False
    assert g.update(Senses(t=0.1 + COLD_AFTER_S / 2, speed=0.0, odom=(0.0, 0.0, 0.0))) is False
    assert g.update(Senses(t=0.1 + COLD_AFTER_S + 0.05, speed=0.0, odom=(0.0, 0.0, 0.0))) is True
    # Turning (yaw moving between steps) also counts as the gait going.
    g.update(Senses(t=1.0, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert g.update(Senses(t=1.02, speed=0.0, odom=(0.0, 0.0, 0.02))) is False
    assert turn(-1.0, cold=True) == (TURN_KICK, 0.0, -1.0) and turn(-1.0, cold=False) == (0.0, 0.0, -1.0)
    assert turn(+1.0, cold=True) == (TURN_KICK, 0.0, 1.0)


def test_tracks_are_kept_in_the_body_frame_whatever_the_head_is_doing():
    """A detection frame carries the camera's yaw relative to the body;
    the track's bearing is the BODY bearing, so a head turned 0.4 rad left
    seeing a person dead ahead of the lens yields a track at +0.4."""
    tr = Tracker()
    fr = DetectionFrame(0.0, [det("person", 0.0, 1.0, "p0")], cam_yaw=0.4)
    tr.update(fr, 0.0, yaw=0.0)
    tr.update(DetectionFrame(0.1, [det("person", 0.02, 1.0, "p0")], cam_yaw=0.4), 0.1, yaw=0.0)
    p = tr.best("person", 0.1)
    assert p is not None and abs(p.bearing - 0.41) < 0.02 and p.hits == 2
    # The head swings back to centre and sees it at +0.4 in the lens: same track, same body bearing.
    tr.update(DetectionFrame(0.2, [det("person", 0.4, 1.0, "p0")], cam_yaw=0.0), 0.2, yaw=0.0)
    q = tr.best("person", 0.2)
    assert q is not None and q.id == p.id and abs(q.bearing - 0.4) < 0.03


def test_a_rolling_ball_gets_a_position_and_a_velocity_and_a_prediction():
    """With the body's odometry position passed in, each hit places the
    track in the odometry frame; consecutive hits give a velocity; predict()
    runs it forward, to a stop under a deceleration."""
    from microduck_local.brain.tracker import Tracker
    from microduck_local.sensors.detector import Detection, DetectionFrame
    tr_ = Tracker()
    # A ball 1.0 m ahead rolling left at 0.5 m/s, seen from a duck at the origin facing +x, 10 Hz.
    for k in range(5):
        t = 0.1 * k
        y = 0.5 * t
        f = DetectionFrame(t, [Detection("ball", "ball0", math.atan2(y, 1.0), 0.0, 0.07, math.hypot(1.0, y), 0.9)])
        tr_.update(f, t, 0.0, (0.0, 0.0))
    tr = tr_.best("ball", 0.4)
    assert tr.xy is not None and abs(tr.xy[0] - 1.0) < 0.05 and abs(tr.xy[1] - 0.2) < 0.05   # smoothed: a few cm behind
    assert tr.vel_hits >= 2 and abs(tr.vel[0]) < 0.05 and 0.35 < tr.vel[1] < 0.6
    # A second later it is 0.5 m further left; with a deceleration it stops short of that.
    px, py = tr.predict(1.4)
    assert abs(px - 1.0) < 0.05 and abs(py - 0.7) < 0.12
    px2, py2 = tr.predict(1.4, decel=0.4)
    assert 0.2 < py2 < py
    # Without the position nothing is placed, and the old call shape still works.
    plain = Tracker()
    plain.update(DetectionFrame(0.0, [Detection("ball", "ball0", 0.0, 0.0, 0.07, 1.0, 0.9)]), 0.0, 0.0)
    assert plain.best("ball", 0.0, min_hits=1).xy is None and plain.best("ball", 0.0, min_hits=1).predict(1.0) is None
