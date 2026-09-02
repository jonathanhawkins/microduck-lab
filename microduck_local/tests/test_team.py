"""Soccer's second form: the team blackboard (brain/team.py), the chase
brain's line-up geometry (behind the ball on the line to the goal, a
walk-round via-point, a push when the goal is far), its head-down ball
tracking and its wall rule, and a pitch with teams."""

import math

import numpy as np
import pytest

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team, brain_kwargs
from microduck_local.brain.tracker import Track
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.sensors.tof import TofFrame


def test_team_roles_hysteresis_and_shared_ball():
    tm = Team("left")
    assert tm.attacker(0.0) is None and tm.role("d0", 0.0) == "attack"      # nobody: everyone attacks
    tm.claim("d0", 1.0, 0.9, (0.5, 0.0))
    tm.claim("d1", 1.0, 0.5, (0.52, 0.01))
    assert tm.attacker(1.0) == "d1" and tm.role("d0", 1.0) == "support" and tm.rank("d0", 1.0) == 0
    # d0 gets a little nearer: not clearly, so d1 keeps the ball.
    tm.claim("d0", 1.1, 0.42, (0.5, 0.0))
    tm.claim("d1", 1.1, 0.5, None)
    assert tm.attacker(1.1) == "d1"
    tm.claim("d0", 1.2, 0.2, (0.5, 0.0))
    assert tm.attacker(1.2) == "d0" and tm.role("d1", 1.2) == "support"
    # The ball position is the freshest sighting; a stale claim drops out.
    assert tm.ball(1.2) == (0.5, 0.0)
    tm.claim("d0", 2.0, 0.3, None)
    assert tm.members(2.5) == ["d0"] and tm.attacker(2.5) == "d0"        # d1's claim went stale
    assert tm.members(9.0) == [] and tm.attacker(9.0) is None
    assert "attacker" in tm.payload(1.2)


def _ball(bearing, rng):
    return Track(1, "ball", bearing, 0.0, 0.1, rng, 0.9, 0.0, 0.0, hits=2)


def test_chase_plans_behind_the_ball_toward_the_goal_and_falls_back_to_the_line_of_sight():
    p = ChaseParams()
    b = Chase(p, goal=(1.5, 0.0))
    # Ball 0.4 m ahead, duck at the origin facing +x, goal at +1.5: behind it already, kick spot.
    spot = b._plan((0.0, 0.0, 0.0), _ball(0.0, 0.4))
    # The body stands rotated by the kick map's deflection so the KICK flies along the line (u = 0).
    defl = {"kick_left": p.kick_deflect_left, "kick_right": p.kick_deflect_right}[spot[2]]
    h = -defl
    side = -p.kick_side if spot[2] == "kick_left" else p.kick_side
    assert spot[4] == "kick" and abs(spot[3] - h) < 1e-9
    assert abs(spot[0] - (0.4 - p.kick_ahead * math.cos(h) - side * math.sin(h))) < 1e-9
    assert abs(spot[1] - (0.0 - p.kick_ahead * math.sin(h) + side * math.cos(h))) < 1e-9
    assert abs(math.hypot(spot[0] - 0.4, spot[1]) - math.hypot(p.kick_ahead, p.kick_side)) < 1e-9
    # The goal is more than `aim_max` off the line of sight: kick along the line of sight instead.
    side_b = Chase(p, goal=(0.0, 1.5))
    spot = side_b._plan((0.0, 0.0, 0.0), _ball(0.0, 0.4))
    assert abs(spot[3] + {"kick_left": p.kick_deflect_left, "kick_right": p.kick_deflect_right}[spot[2]]) < 1e-9
    # Pushing is off by default (measured); switched on, a far goal gives a push spot squarely behind the ball.
    assert ChaseParams().push_beyond == math.inf
    far = Chase(ChaseParams(push_beyond=1.4), goal=(3.0, 0.0))
    spot = far._plan((0.0, 0.0, 0.0), _ball(0.0, 0.4))
    assert spot[4] == "push" and spot[2] is None and abs(spot[0] - (0.4 - p.push_behind)) < 1e-9 and spot[1] == 0.0
    # No goal known: the heading it was placed with is the line.
    free = Chase(p)
    free.attack = 0.5
    spot = free._plan((0.0, 0.0, 0.0), _ball(0.0, 0.4))
    assert abs(spot[3] - (0.5 - {"kick_left": p.kick_deflect_left, "kick_right": p.kick_deflect_right}[spot[2]])) < 1e-9 and spot[4] == "kick"
    assert free._own_goal((0.0, 0.0, 0.0))[0] < 0
    # The foot keeps its choice near the line (hysteresis), flips well off it.
    b.spot = ("x", "y", "kick_left", 0.0, "kick")
    assert b._plan((0.0, 0.01, 0.0), _ball(-0.02, 0.4))[2] == "kick_left"
    assert b._plan((0.0, 0.3, 0.0), _ball(-0.6, 0.4))[2] == "kick_right"


def _senses(t, ball=None, tof=None, odom=(0.0, 0.0, 0.0), speed=0.3):
    det = None if ball is None else DetectionFrame(t, [Detection("ball", "ball0", ball[0], -0.3, 0.12, ball[1], 0.9)])
    return Senses(t=t, det=det, det_age=None if det is None else 0.0, tof=tof, tof_age=None if tof is None else 0.0,
                  speed=speed, odom=odom)


def test_chase_pitches_the_head_down_walking_at_a_near_ball_and_not_when_turning():
    b = Chase(ChaseParams(), goal=(1.5, 0.0))
    b.step(_senses(0.0, (0.05, 1.5)))
    out = b.step(_senses(0.1, (0.05, 1.2)))
    assert out.note == "chase" and out.twist[0] > 0 and out.head == (0.0, 0.0, 0.0, 0.0)   # still far: level
    for k in range(3):                                                      # the track smooths in
        out = b.step(_senses(0.2 + 0.1 * k, (0.05, 0.7)))
    assert out.note in ("chase", "lineup") and out.twist[0] > 0
    assert 0.0 < out.head[1] <= b.p.head_down and abs(out.head[1] - b._gaze(0.7)) < 0.15   # follows the range
    assert b._gaze(0.15) == b.p.head_down and b._gaze(2.0) == 0.0
    # Turning in place toward a ball off to the side: head level (the walker cannot turn with it down).
    c = Chase(ChaseParams(), goal=(1.5, 0.0))
    c.step(_senses(0.0, (1.2, 0.7), speed=0.0))
    out = c.step(_senses(0.1, (1.2, 0.7), speed=0.0))
    assert out.note == "turn" and out.head == (0.0, 0.0, 0.0, 0.0)


def test_chase_wall_rule_turns_away_from_a_wall_beside_it():
    depth = np.full((8, 8), 2000, np.uint16)
    depth[2:5, 0:3] = 150                                  # a wall 15 cm off the LEFT columns
    b = Chase(ChaseParams())
    b.step(_senses(0.0, None, speed=0.0))
    out = b.step(_senses(0.7, None, TofFrame(t=0.7, depth_mm=depth, valid=np.ones((8, 8), bool)), speed=0.0))
    assert out.note == "search" and out.twist[2] == -1.0   # searching = a left turn, but the wall is there
    depth[2:5, 5:8] = 100                                  # a wall nearer on the right: the left turn stands
    out = b.step(_senses(0.8, None, TofFrame(t=0.8, depth_mm=depth, valid=np.ones((8, 8), bool)), speed=0.0))
    assert out.twist[2] == 1.0


def test_pitch_with_teams_and_brain_kwargs():
    from microduck_local import contract as C
    from microduck_local.world import World, make_pitch, validate_scenario
    sc = make_pitch(per_side=2)
    assert len(sc.ducks) == 4 and {d.team for d in sc.ducks} == {"left", "right"}
    assert validate_scenario(sc.to_dict()) == sc
    assert make_pitch(per_side=3).name == "pitch-3v3" and len(make_pitch(per_side=3).ducks) == 6
    if not C.SCENE_WALK_XML.exists():
        pytest.skip("microduck_rl checkout not found")
    w = World(sc)
    teams: dict = {}
    kw = {d.id: brain_kwargs(d, w, teams) for d in sc.ducks}
    assert kw["d0"]["goal"][0] > 0 and kw["d2"]["goal"][0] < 0 and kw["d0"]["goal"] == kw["d1"]["goal"]
    assert kw["d0"]["team"] is kw["d1"]["team"] and kw["d0"]["team"] is not kw["d2"]["team"]
    assert set(teams) == {"left", "right"}
    from microduck_local.world import Duck, Scenario
    plain = Scenario(name="x", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, None, None, "chase")])
    assert brain_kwargs(plain.ducks[0], World(plain), {}) == {}


def test_kickoff_forgets_the_plan_and_keeps_the_tally():
    """A goal: the chase brain drops its spot and manoeuvre but keeps the
    kicks it took (the benchmark counts them); the team board is wiped;
    `kickoff_brains` does both for every brain, falling back to reset()."""
    from microduck_local.brain.team import Team, kickoff_brains
    b = Chase(ChaseParams(), goal=(1.5, 0.0), team=Team("left"), duck_id="d0")
    b.kicks, b.pushes, b.state, b.spot = 3, 1, "retreat", (0.5, 0.0, "kick_left", 0.0, "kick")
    b._poses = [(0.0, 0.0, 0.0, 0.0)]
    b.team.claim("d0", 10.0, 0.4, (0.5, 0.0))
    b.team.claim("d1", 10.0, 0.9, None)
    assert b.team.attacker(10.0) == "d0"
    kickoff_brains({"d0": b}, {"left": b.team})
    assert b.kicks == 3 and b.pushes == 1 and b.goal == (1.5, 0.0)
    assert b.state == "search" and b.spot is None and b._poses == []
    assert b.team.claims == {} and b.team.attacker(10.0) is None

    class Plain:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1
    other = Plain()
    kickoff_brains({"x": other}, {})
    assert other.resets == 1
