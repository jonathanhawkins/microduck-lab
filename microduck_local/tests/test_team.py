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
    b = Chase(ChaseParams(predict_s=3.0, head_yaw_when="always"), goal=(1.5, 0.0))   # the gaze, measured off by default
    b.step(_senses(0.0, (0.05, 1.5)))
    out = b.step(_senses(0.1, (0.05, 1.2)))
    assert out.note == "chase" and out.twist[0] > 0 and out.head[1] == 0.0      # still far: level pitch...
    assert abs(out.head[2] - 0.9 * 0.05) < 0.02                                   # ...but the head yaws toward the ball
    for k in range(3):                                                      # the track smooths in
        out = b.step(_senses(0.2 + 0.1 * k, (0.05, 0.7)))
    assert out.note in ("chase", "lineup") and out.twist[0] > 0
    assert 0.0 < out.head[1] <= b.p.head_down and abs(out.head[1] - b._gaze(0.7)) < 0.15   # follows the range
    assert b._gaze(0.15) == b.p.head_down and b._gaze(2.0) == 0.0
    # Turning in place toward a ball off to the side: head level (the walker cannot turn with it down).
    c = Chase(ChaseParams(predict_s=3.0, head_yaw_when="always"), goal=(1.5, 0.0))
    c.step(_senses(0.0, (1.2, 0.7), speed=0.0))
    out = c.step(_senses(0.1, (1.2, 0.7), speed=0.0))
    assert out.note == "turn" and out.head[1] == 0.0 and out.head[2] != 0.0   # level pitch, the head yawed to the ball


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


def test_a_supporter_keeps_its_spot_inside_the_boards_and_stands_beside_a_teammate():
    """3v3 falls were supporters turning in place against a teammate or the
    boards: the support spot is clamped inside the pitch, and a duck track
    inside `beside_m` - stale or not - means no turn in place."""
    from microduck_local.brain.tracker import Track
    p = ChaseParams()
    tm = Team("left")
    b = Chase(p, goal=(1.5, 0.0), team=tm, duck_id="d1", bounds=(1.5, 1.25))
    tm.claim("d0", 10.0, 0.2, (1.3, 1.1))          # d0 attacks a ball in the corner
    tm.claim("d1", 10.0, 2.0, None)
    b._senses = Senses(t=10.0)
    # Sitting at the clamped spot: the raw spot (0.7 m behind the ball toward our goal) would be
    # inside the boards' margin in y; the clamped one is not, and a supporter there just faces the ball.
    odom = (0.6, 0.9, 0.5)
    b.state = "support"
    vx, wz = b._support(odom, None, False, False)
    assert abs(vx) < 0.3                                             # not a full walk into the boards
    # Beside a teammate (a track at 0.2 m, 90 deg off the nose, seen a second ago): no turn in place.
    b.tracker.tracks.append(Track(id=9, cls="duck", bearing=1.5, elevation=0.0, width=0.5, range=0.2, conf=0.9, born_t=8.0, last_t=9.0))
    b._senses = Senses(t=10.0)
    vx, wz = b._support((0.0, 0.0, 0.0), None, False, True)         # nobody has the ball: it would turn to look
    assert vx == 0.0 and wz == 0.0
    b.tracker.tracks.clear()
    vx, wz = b._support((0.0, 0.0, 0.0), None, False, True)
    assert wz != 0.0


def test_the_line_up_squares_up_behind_the_ball_then_walks_straight_in():
    """Two stages: from the pre-spot (approach_back behind the kick spot on
    the kick line) a duck off the line is sent there and squared up first -
    a turn in place 30 cm from the ball - and only then walks straight in
    along the line with no steering, stopping by the distance left."""
    p = ChaseParams(two_stage=True)
    b = Chase(p, goal=(3.0, 0.0))
    b._senses = Senses(t=1.0)
    b.state, b.t_state = "lineup", 0.0
    b.spot = (0.5, 0.06, "kick_right", 0.0, "kick")            # kick spot, line along +x
    # Well behind and beside: stage one heads for the pre-spot, not the spot.
    it = b.step(Senses(t=1.0, odom=(0.0, 0.3, 0.0)))
    assert not b.lined and it.twist[0] > 0
    # At the pre-spot but facing 0.5 rad off: turn in place, not a step.
    it = b.step(Senses(t=1.1, odom=(0.5 - p.approach_back, 0.06, 0.5)))
    assert not b.lined and it.twist[0] <= 0.2 and it.twist[2] != 0.0
    # At the pre-spot, facing the line: lined; the next decisions walk straight (wz 0) at approach_speed.
    it = b.step(Senses(t=1.2, odom=(0.5 - p.approach_back, 0.06, 0.0)))
    assert b.lined
    it = b.step(Senses(t=1.3, odom=(0.5 - p.approach_back + 0.05, 0.06, 0.0)))
    assert it.twist == (p.approach_speed, 0.0, 0.0) and b.state == "lineup"
    # On the spot: stop and settle.
    it = b.step(Senses(t=1.4, odom=(0.5, 0.06, 0.0)))
    assert it.twist[0] == 0.0 and b.state == "settle"


def test_after_a_kick_the_duck_looks_then_hunts_the_kick_line_then_searches():
    """A kicked ball rolls off along the kick line: after the look finds
    nothing the duck WALKS that line for `hunt_s` (head level, the ball in
    view from 0.3 m out) before the standing search."""
    p = ChaseParams(hunt_s=3.0, seek_s=20.0)
    b = Chase(p, goal=(3.0, 0.0))
    # A settle that fires the kick: on the spot, squared, settled.
    b._senses = Senses(t=0.0)
    b.state, b.t_state, b.lined = "settle", 0.0, True
    b.spot = (0.0, 0.06, "kick_right", 0.3, "kick")
    it = b.step(Senses(t=p.settle_s + 0.01, odom=(0.0, 0.06, 0.3)))
    assert it.skill == "kick_right" and b._hunt_u == 0.3
    # The kick window runs (skill set), then ends: look for look_s...
    b.step(Senses(t=1.0, odom=(0.0, 0.06, 0.3), skill="kick_right"))
    it = b.step(Senses(t=1.1, odom=(0.0, 0.06, 0.3)))
    assert it.note == "look" and it.twist[0] == 0.0
    # ...then hunt: walk the kick line at speed, steering onto its heading.
    it = b.step(Senses(t=1.1 + p.look_s + 0.05, odom=(0.0, 0.06, 0.2)))
    assert it.note == "hunt" and it.twist[0] == p.hunt_speed and it.twist[2] > 0
    # ...and only then the search - which first walks to where the hunted line pointed (the memory).
    it = b.step(Senses(t=1.1 + p.look_s + p.hunt_s + 0.1, odom=(1.0, 0.3, 0.3)))
    assert it.note == "seek" and b.memory is not None
    b.memory = None
    it = b.step(Senses(t=1.1 + p.look_s + p.hunt_s + 0.2, odom=(1.0, 0.3, 0.3)))
    assert it.note == "search"


def test_the_hunt_ends_for_the_tof_a_duck_beside_and_the_boards():
    """The hunt (traced into the boards and into the other duck) is slow,
    turns gently, and ends the moment the ToF has something inside
    hunt_stop, a duck track is beside, or the boards are ahead."""
    from microduck_local.brain.tracker import Track
    p = ChaseParams(hunt_s=3.0)

    def hunting_duck(**kw):
        b = Chase(p, goal=(3.0, 0.0), bounds=(1.5, 1.25), **kw)
        b._senses = Senses(t=0.0)
        b._hunt_u, b._hunt_t0 = 0.0, 0.0
        return b
    # Clear ahead: hunts at hunt_speed with a capped turn toward the line.
    b = hunting_duck()
    it = b.step(Senses(t=0.5, odom=(0.0, 0.0, 0.6)))
    assert it.note == "hunt" and it.twist[0] == p.hunt_speed and abs(it.twist[2]) <= p.hunt_wz
    # Something 0.4 m ahead on the ToF: not a hunt any more.
    depth = np.full((8, 8), 2000, np.uint16)
    depth[2:5, 3:5] = 400
    b = hunting_duck()
    it = b.step(Senses(t=0.5, odom=(0.0, 0.0, 0.0), tof=TofFrame(t=0.5, depth_mm=depth, valid=np.ones((8, 8), bool)), tof_age=0.0))
    assert it.note != "hunt" and b._hunt_u is None
    # A duck track beside: no hunt.
    b = hunting_duck()
    b.tracker.tracks.append(Track(id=3, cls="duck", bearing=1.8, elevation=0.0, width=0.5, range=0.15, conf=0.9, born_t=0.0, last_t=0.4))
    it = b.step(Senses(t=0.5, odom=(0.0, 0.0, 0.0)))
    assert it.note != "hunt"
    # The boards 0.3 m ahead in odometry: no hunt.
    b = hunting_duck()
    it = b.step(Senses(t=0.5, odom=(1.3, 0.0, 0.0)))
    assert it.note != "hunt"


def test_the_search_walks_to_where_the_ball_was_before_circling():
    """A ball memory in odometry: the centre spot at a kickoff, every fresh
    sighting, the end of a hunted line. A search with a memory further
    than seek_min away walks there first ("seek"); arriving with nothing
    seen forgets it, and the circle begins."""
    p = ChaseParams(seek_s=20.0)
    b = Chase(p, goal=(1.5, 0.0), bounds=(1.5, 1.25))
    assert b.memory is not None and b.memory[:2] == (0.0, 0.0)        # a pitch: the centre spot
    # Nothing seen, standing 1 m from the centre spot facing it: seek, walking.
    it = b.step(Senses(t=0.5, odom=(-1.0, 0.0, 0.0)))
    assert it.note == "seek" and it.twist[0] > 0
    # Arrived (inside seek_tol), still nothing: the memory goes, the search circles.
    it = b.step(Senses(t=1.0, odom=(-0.1, 0.0, 0.0)))
    assert b.memory is None
    it = b.step(Senses(t=1.1, odom=(-0.1, 0.0, 0.0)))
    assert it.note == "search"
    # A fresh sighting 0.5 m ahead-left is remembered where it was seen.
    det = DetectionFrame(2.0, [Detection("ball", "ball0", 0.3, 0.0, 0.14, 0.5, 0.9)])
    b.step(Senses(t=2.0, odom=(0.0, 0.0, 0.0), det=det, det_age=0.0))
    assert b.memory is not None and abs(b.memory[0] - 0.5 * math.cos(0.3)) < 1e-6 and abs(b.memory[1] - 0.5 * math.sin(0.3)) < 1e-6
    # Off a pitch there is no centre spot to remember.
    assert Chase(p).memory is None
    # Off by default (measured): a pitch brain with seek_s 0 keeps the memory but never seeks.
    b0 = Chase(ChaseParams(), goal=(1.5, 0.0), bounds=(1.5, 1.25))
    assert b0.step(Senses(t=0.5, odom=(-1.0, 0.0, 0.0))).note == "search"


def test_a_teammate_on_the_board_counts_as_a_duck_beside_or_ahead():
    """Teammates share their poses on the board (brain/team.py `mates`): a
    teammate inside `mate_keepout` that no sensor can see - beside me - means
    no turn in place, and one ahead is avoided like a seen duck."""
    p = ChaseParams(mate_keepout=0.4)                             # measured off by default (3v3: no fewer falls)
    tm = Team("left")
    b = Chase(p, goal=(1.5, 0.0), team=tm, duck_id="d1", bounds=(1.5, 1.25))
    tm.claim("d0", 10.0, 0.2, (1.3, 1.1), pos=(0.0, 0.25, 0.0))   # d0 attacks; it is 25 cm to my left
    assert tm.mates("d1", 10.0) == [("d0", (0.0, 0.25, 0.0))] and tm.mates("d0", 10.0) == []
    b.step(_senses(10.0, None, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert b._beside(10.0)
    vx, wz = b._support((0.0, 0.0, 0.0), None, False, True)          # nobody has the ball: it would turn to look
    assert vx == 0.0 and wz == 0.0
    tm.claim("d0", 10.5, 0.2, (1.3, 1.1), pos=(0.3, 0.0, 0.0))     # now 30 cm straight ahead, unseen
    out = b.step(_senses(10.5, None, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert b.state == "avoid" and out.twist[0] == 0.0             # (the note is a supporter's role)
    tm.claim("d0", 12.5, 0.2, (1.3, 1.1), pos=(1.0, 1.0, 0.0))     # far away: nothing to avoid
    out = b.step(_senses(12.6, None, speed=0.0, odom=(0.0, 0.0, 0.0)))
    assert b.state != "avoid" and not b._beside(12.6)
