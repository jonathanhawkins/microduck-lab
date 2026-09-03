"""Soccer's second form: the team blackboard (brain/team.py), the chase
brain's line-up geometry (behind the ball on the line to the goal, a
walk-round via-point, a push when the goal is far), its head-down ball
tracking and its wall rule, and a pitch with teams."""

import math

import numpy as np
import pytest

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.gait import TURN_KICK
from microduck_local.brain.runtime import Senses
from microduck_local.brain.team import Team, brain_kwargs
from microduck_local.brain.tracker import Track
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.sensors.tof import TofFrame


def test_team_roles_hysteresis_and_shared_ball():
    """The board's shape: the quickest live claim attacks, the rest support
    and rank by id, the ball is the freshest sighting, and a claim nobody
    refreshes falls off after `stale_s`."""
    tm = Team("left")
    assert tm.attacker(0.0) is None and tm.role("d0", 0.0) == "attack"      # nobody: everyone attacks
    tm.claim("d0", 1.0, 0.9, (0.5, 0.0))
    tm.claim("d1", 1.0, 0.5, (0.52, 0.01))
    assert tm.attacker(1.0) == "d1" and tm.role("d0", 1.0) == "support" and tm.rank("d0", 1.0) == 0
    # d0 gets a little nearer: not clearly enough, and not for long enough.
    tm.claim("d0", 1.1, 0.42, (0.5, 0.0))
    tm.claim("d1", 1.1, 0.5, None)
    assert tm.attacker(1.1) == "d1"
    tm.claim("d0", 1.2, 0.2, (0.5, 0.0))
    assert tm.attacker(1.2) == "d1"                                     # 0.67 s quicker, but only just now
    for k in range(1, 15):                                              # …held past `hold_s`: the role moves
        tm.claim("d0", 1.2 + 0.1 * k, 0.2, (0.5, 0.0))
        tm.claim("d1", 1.2 + 0.1 * k, 0.5, None)
        tm.attacker(1.2 + 0.1 * k)                                      # read every tick, as a duck does
    assert tm.attacker(2.6) == "d0" and tm.role("d1", 2.6) == "support"
    # The ball position is the freshest sighting; a stale claim drops out.
    assert tm.ball(2.6) == (0.5, 0.0)
    tm.claim("d0", 2.7, 0.3, None)
    assert tm.members(3.65) == ["d0"] and tm.attacker(3.65) == "d0"      # d1's claim went stale
    assert tm.members(9.0) == [] and tm.attacker(9.0) is None
    assert "attacker" in tm.payload(2.6) and "cost" in tm.payload(2.6)["claims"]["d0"]


def test_the_claim_is_the_time_to_reach_the_ball_not_the_distance_to_it():
    """Measured on this walker (`walker-facts`): it walks at 0.45 m/s
    (`ChaseParams.speed`) and a turn in place runs at ~0.7 rad/s once the
    gait is kicked, 0.25 rad in the first cold second — so turning round
    costs seconds that a straight line charges nothing for. A duck facing
    the ball 0.6 m away is 1.0 s from it; one facing away 0.4 m away is
    4.7 s from it, and it is the first that should be sent."""
    tm = Team("left")
    tm.claim("facing", 1.0, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))         # 0.6 m, nose on the ball
    tm.claim("turned", 1.0, 0.4, (0.0, 0.0), (0.0, 0.4, math.pi / 2))     # 0.4 m, nose the other way
    assert tm.cost("facing", 1.0) == pytest.approx((0.6 - tm.reach) / tm.speed, abs=1e-6)
    assert tm.cost("turned", 1.0) == pytest.approx(
        (math.pi - tm.turn_free) / tm.turn_rate + tm.cold_s + (0.4 - tm.reach) / tm.speed, abs=1e-6)
    assert tm.attacker(1.0) == "facing"                                   # …though "turned" is 0.2 m nearer
    # A claim with no pose to turn from is still the straight-line time.
    tm.claim("blindfold", 1.0, 0.9, None, None)
    assert tm.cost("blindfold", 1.0) == pytest.approx(0.9 / tm.speed, abs=1e-6)


def test_a_duck_that_has_lost_sight_of_the_ball_does_not_resign_the_role():
    """The chase brain claims `inf` the moment its ball track goes cold, and
    the level camera loses a floor ball inside ~0.3 m — exactly where the
    attacker lines up. The board now costs a blind duck off its OWN pose and
    the freshest sighting anyone has, plus `blind_s`: on the ball and blind
    still beats seeing it from a metre away, and a teammate that is really
    quicker still takes over."""
    tm = Team("left")
    tm.claim("watcher", 1.0, 1.2, (0.0, 0.0), (1.2, 0.0, math.pi))        # sees it, 1.2 m out
    tm.claim("onball", 1.0, math.inf, None, (0.2, 0.0, math.pi))          # lost it — it is at its feet
    assert tm.cost("onball", 1.0) == pytest.approx((0.2 - tm.reach) / tm.speed + tm.blind_s, abs=1e-6)
    assert tm.attacker(1.0) == "onball" and tm.role("watcher", 1.0) == "support"
    # Nobody has ever seen the ball: a blind claim has nothing to cost.
    empty = Team("right")
    empty.claim("d0", 1.0, math.inf, None, (0.0, 0.0, 0.0))
    assert empty.cost("d0", 1.0) == math.inf
    # A teammate that really is quicker (0.3 m, facing it) still wins the
    # role — it just has to hold the margin for `hold_s`.
    for k in range(0, 15):
        t = 1.0 + 0.1 * k
        tm.claim("watcher", t, 0.3, (0.0, 0.0), (0.3, 0.0, math.pi))
        tm.claim("onball", t, math.inf, None, (0.2, 0.0, math.pi))
        tm.attacker(t)                                                    # read every tick, as a duck does
    assert tm.attacker(2.4) == "watcher"


def test_a_stale_claim_does_not_beat_a_fresh_one():
    """One message a second over Wi-Fi means half the claims on the board are
    the older one. A claim is worth `age_rate` seconds of cost per second of
    age, so a duck that has not spoken for most of a second has to be that
    much quicker to be believed — and past `stale_s` it stops counting."""
    tm = Team("left")
    tm.claim("fresh", 2.0, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
    tm.claim("old", 1.4, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))            # the same claim, 0.6 s ago
    assert tm.cost("old", 2.0) == pytest.approx(tm.cost("fresh", 2.0) + 0.6 * tm.age_rate, abs=1e-6)
    assert tm.attacker(2.0) == "fresh"
    # "old" is really quicker — 0.5 s of it — but it said so 0.8 s ago, and
    # the age eats the whole margin: the fresh incumbent keeps the role.
    tm.claim("fresh", 3.0, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
    assert tm.attacker(3.0) == "fresh"
    tm.claim("old", 3.2, 0.375, (0.0, 0.0), (0.375, 0.0, math.pi))        # 0.5 s quicker…
    tm.claim("fresh", 4.0, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
    assert tm.cost("old", 4.0) > tm.cost("fresh", 4.0)                    # …but it said so 0.8 s ago
    assert tm.attacker(4.0) == "fresh"
    # Said afresh, and held, the same claim takes the role.
    for k in range(0, 15):
        t = 4.4 + 0.1 * k
        tm.claim("fresh", t, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
        tm.claim("old", t, 0.3, (0.0, 0.0), (0.3, 0.0, math.pi))
        tm.attacker(t)
    assert tm.attacker(5.8) == "old"


def test_the_role_moves_only_when_a_challenger_is_clearly_quicker_for_long_enough():
    """The churn this replaces: over 3 seeds x 300 s of 3v3 the role changed
    hands 11.6 times a duck a run and a quarter of the spells were under a
    second. A challenger must be `switch_s` quicker and STAY that quick for
    `hold_s`; a margin that lapses restarts the clock; `give_up_s` (the
    incumbent out of the play) and a claim gone stale still move it at once."""
    def board():
        tm = Team("left")
        tm.claim("a", 0.0, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
        tm.claim("b", 0.0, 1.2, (0.0, 0.0), (1.2, 0.0, math.pi))
        assert tm.attacker(0.0) == "a"
        return tm

    # b becomes 0.78 s quicker (past `switch_s`) and holds it: the role moves
    # only after `hold_s`, and never sooner.
    tm = board()
    for k in range(1, 20):
        t = 0.1 * k
        tm.claim("a", t, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
        tm.claim("b", t, 0.25, (0.0, 0.0), (0.25, 0.0, math.pi))
        assert tm.attacker(t) == ("a" if t < 0.1 + tm.hold_s - 1e-9 else "b")
    # A margin that lapses restarts the clock: alternating half-ticks never
    # accumulate `hold_s` of pressure.
    tm = board()
    for k in range(1, 40):
        t = 0.1 * k
        tm.claim("a", t, 0.6, (0.0, 0.0), (0.6, 0.0, math.pi))
        near = 0.25 if k % 2 else 0.62
        tm.claim("b", t, near, (0.0, 0.0), (near, 0.0, math.pi))
        assert tm.attacker(t) == "a"
    # `give_up_s`: the incumbent turned away 2 m off is out of the play, and
    # the role moves on the tick, without waiting.
    tm = board()
    tm.claim("a", 0.1, 2.0, (0.0, 0.0), (2.0, 0.0, 0.0))
    tm.claim("b", 0.1, 0.3, (0.0, 0.0), (0.3, 0.0, math.pi))
    assert tm.cost("a", 0.1) - tm.cost("b", 0.1) > tm.give_up_s and tm.attacker(0.1) == "b"
    # A stale incumbent (nothing heard for `stale_s`) is replaced at once.
    tm = board()
    tm.claim("b", 1.5, 1.2, (0.0, 0.0), (1.2, 0.0, math.pi))
    assert tm.members(1.5) == ["b"] and tm.attacker(1.5) == "b"


def test_the_board_carries_the_balls_own_motion_into_the_cost():
    """A kicked ball leaves at 1.4 m/s and slows at 0.04 m/s^2 on this floor,
    so where it IS and where it will BE when a duck arrives are different
    places. The board keeps a velocity from consecutive fixes by the same
    duck (differencing across ducks is noise) and aims at the intercept: of
    two ducks a metre away and both facing the ball, the one it is rolling
    toward is far quicker than the one chasing it, where a straight line
    calls them equal. Off — which is the SHIPPED default, measured: the
    intercept churned the role worse than the straight fix — they are equal
    again."""
    def board(lead):
        tm = Team("left", lead_max_s=lead)
        for t, y in ((0.0, 0.0), (0.2, 0.2), (0.4, 0.4)):                 # a scout, 1.0 m/s along +y
            tm.claim("scout", t, 2.0, (0.0, y), (0.0, -2.0, 0.0))
        tm.claim("ahead", 0.4, math.inf, None, (0.0, 1.4, -math.pi / 2))
        tm.claim("behind", 0.4, math.inf, None, (0.0, -0.6, math.pi / 2))
        return tm

    assert Team("left").lead_max_s == 0.0                                 # shipped off (measured)
    on = board(1.5)
    assert on._vel_hits >= 2 and on._vel[1] == pytest.approx(1.0, abs=1e-6)
    assert on.cost("ahead", 0.4) < on.cost("behind", 0.4) - 2.0
    off = board(0.0)
    assert off.cost("ahead", 0.4) == pytest.approx(off.cost("behind", 0.4), abs=1e-6)
    # A ball nobody has seen move twice, or moving slower than `vel_use`
    # (the walking speed drags a coasting track's fix along with the duck),
    # is aimed at where it is.
    slow = Team("left", lead_max_s=1.5)
    for t, y in ((0.0, 0.0), (0.2, 0.04), (0.4, 0.08)):                   # 0.2 m/s: noise, not a roll
        slow.claim("scout", t, 2.0, (0.0, y), (0.0, -2.0, 0.0))
    slow.claim("ahead", 0.4, math.inf, None, (0.0, 1.08, -math.pi / 2))
    slow.claim("behind", 0.4, math.inf, None, (0.0, -0.92, math.pi / 2))
    assert slow.cost("ahead", 0.4) == pytest.approx(slow.cost("behind", 0.4), abs=1e-6)


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
    assert kw["d0"]["p"].bump_stand_s == ChaseParams().team_bump_stand_s   # a roster with teammates: the bump sense on
    solo = make_pitch(per_side=1)
    assert "p" not in brain_kwargs(solo.ducks[0], World(solo), {})   # a lone attacker keeps the default
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


def test_the_look_after_a_kick_aims_by_the_kick_map_and_the_search_can_sweep_the_head():
    """`look_aim`: the look after a kick yaws the head to the foot's exit
    angle off the kick map (+21.6 deg left, -11 right) near the horizon;
    `search_sweep`: a searching head sweeps side to side. Both inside the
    walker's trained +-1.4 rad."""
    p = ChaseParams(look_aim=True, search_sweep=1.4)
    b = Chase(p, goal=(1.5, 0.0))
    b.step(_senses(0.0, None, speed=0.0))
    b._last_foot, b._look_t0 = "kick_left", 5.0
    out = b.step(_senses(5.1, None, speed=0.0))
    assert out.note == "look" and abs(out.head[2] - 0.9 * math.radians(21.6)) < 1e-6
    assert out.head[1] < b._gaze(0.3)                                  # near the horizon, not the 0.3 m dip
    b._last_foot = "kick_right"
    out = b.step(_senses(5.2, None, speed=0.0))
    assert abs(out.head[2] - 0.9 * math.radians(-11.0)) < 1e-6
    # Searching, no track: the head sweeps; a quarter period in it is at +1.4.
    b._look_t0 = -9.0
    yaws = []
    for k in range(1, 12):
        out = b.step(_senses(6.0 + 0.1 * k, None, speed=0.0))
        yaws.append(out.head[2])
    assert out.note == "search" and max(yaws) > 1.0 and min(yaws) < 0.2 and max(abs(y) for y in yaws) <= 1.4 + 1e-9


def test_a_bumped_duck_stands_instead_of_turning_in_place():
    """Senses.bumped (the body touching another body): for `bump_stand_s`
    after it, a standing turn becomes a stand. Off at 0."""
    # `search` is deliberately NOT a bump-stand state (its circle walks, and
    # freezing it stops the one behaviour that finds the ball), so drive the
    # rule through `support`, which is.
    p = ChaseParams(bump_stand_s=1.0, search_dip_s=0.0, search_vx=0.0)
    lone = Chase(p, goal=(1.5, 0.0))
    lone.step(_senses(0.0, None, speed=0.0))
    out = lone.step(_senses(0.5, None, speed=0.0))
    assert out.note == "search" and out.twist[2] != 0.0                # a standing search turn
    out = lone.step(Senses(t=0.6, speed=0.0, odom=(0.0, 0.0, 0.0), bumped=True))
    assert out.twist[2] != 0.0                                         # searching: the rule leaves it alone
    # A real supporter: a teammate claims the ball from closer, so this duck's role is support.
    tm = Team("left")
    b = Chase(p, goal=(1.5, 0.0), team=tm, duck_id="d1", bounds=(1.5, 1.25))

    def tick(t, bumped):
        tm.claim("d0", t, 0.2, None)                                   # d0 is nearer: it attacks
        return b.step(Senses(t=t, speed=0.0, odom=(0.0, 0.0, 0.0), bumped=bumped))

    out = tick(0.5, False)
    assert b.state == "support" and out.twist[2] != 0.0                # turning to look for the ball
    assert tick(0.7, True).twist == (0.0, 0.0, 0.0)                    # supporting and touching: stand
    assert tick(1.4, True).twist == (0.0, 0.0, 0.0)                    # still inside the window
    # Contact that never let up must NOT extend the window: it is timed from the onset.
    assert tick(1.75, True).twist[2] != 0.0
    # A fresh episode (a gap longer than bump_gap_s) starts a new window.
    assert tick(3.5, True).twist == (0.0, 0.0, 0.0)
    off = Chase(ChaseParams(search_dip_s=0.0, search_vx=0.0), goal=(1.5, 0.0))
    off.step(_senses(0.0, None, speed=0.0))
    out = off.step(Senses(t=0.5, speed=0.0, odom=(0.0, 0.0, 0.0), bumped=True))
    assert out.twist[2] != 0.0
    # `blocked` is an escape, never a stand: 70% of the first version's firing
    # was there, and 6 of 8 traced falls were a stand leaning on the other
    # duck. A wall right ahead blocks the walking search; bumped or not, the
    # turn out of it survives.
    depth = np.full((8, 8), 2000, np.uint16)
    depth[2:5, 3:5] = 150                                              # something 15 cm dead ahead
    bl = Chase(ChaseParams(bump_stand_s=1.0, search_dip_s=0.0), goal=(1.5, 0.0))
    bl.step(_senses(0.0, None, speed=0.0))
    wall = TofFrame(t=0.5, depth_mm=depth, valid=np.ones((8, 8), bool))
    out = bl.step(Senses(t=0.5, tof=wall, tof_age=0.0, speed=0.0, odom=(0.0, 0.0, 0.0), bumped=True))
    assert bl.state == "blocked" and out.twist[2] != 0.0


def test_the_kick_cone_dribbles_a_ball_that_is_too_far_out_and_shoots_from_close():
    """`kick_cone`: the goal mouth subtends a half-angle from the ball; below
    the threshold the plan is a push (dribble it closer), above it a kick.
    A 0.7 m goal subtends 0.35 rad from 0.96 m out."""
    from microduck_local.brain.tracker import Track
    b = Chase(ChaseParams(kick_cone=0.35), goal=(1.5, 0.0), goal_w=0.7)
    assert b.goal_cone(0.7, 0.0) > 0.35 > b.goal_cone(-0.5, 0.0)      # 0.8 m out vs 2.0
    assert b.goal_cone(1.4, 0.0) > 1.0                                # on the line: nearly a right angle
    off = Chase(ChaseParams(), goal=(1.5, 0.0), goal_w=0.0)
    assert off.goal_cone(0.0, 0.0) == math.inf                        # no mouth width: never gated

    def mode_at(brain, duck_xy, ball_xy):
        odom = (duck_xy[0], duck_xy[1], 0.0)
        rng = math.hypot(ball_xy[0] - duck_xy[0], ball_xy[1] - duck_xy[1])
        bearing = math.atan2(ball_xy[1] - duck_xy[1], ball_xy[0] - duck_xy[0])
        tr = Track(id=1, cls="ball", bearing=bearing, elevation=-0.3, width=0.12,
                   range=rng, conf=0.9, born_t=0.0, last_t=0.0)
        return brain._plan(odom, tr)[4]

    assert mode_at(b, (-1.2, 0.0), (-0.9, 0.0)) == "push"             # far out: dribble
    assert mode_at(b, (0.6, 0.0), (0.9, 0.0)) == "kick"               # close in: shoot
    assert mode_at(off, (-1.2, 0.0), (-0.9, 0.0)) == "kick"           # gate off: shoot from anywhere


def test_a_supporter_can_stand_ahead_of_the_ball_instead_of_behind_it():
    """`support_mode`: "back" puts the supporter between the ball and our own
    goal (it defends); "ahead" puts it between the ball and the goal we
    attack (a poacher, in position to walk a loose ball in). Both keep the
    spot inside the boards."""
    p_back = ChaseParams()
    p_ahead = ChaseParams(support_mode="ahead")
    for p, nearer_attack in ((p_back, False), (p_ahead, True)):
        tm = Team("left")
        b = Chase(p, goal=(1.5, 0.0), team=tm, duck_id="d1", bounds=(1.5, 1.25))
        tm.claim("d0", 10.0, 0.2, (0.0, 0.0))            # d0 attacks a ball on the centre spot
        tm.claim("d1", 10.0, 2.0, None)
        b._senses = Senses(t=10.0)
        b._support((-1.0, 0.0, 0.0), None, False, False)
        assert b.spot is None                            # a supporter's spot is not a kick spot
        # Walk the servo target out of _support by reading where it wants to go.
        target_ahead = b.p.support_mode == "ahead"
        assert target_ahead == nearer_attack


def test_chase_params_read_a_variant_off_the_environment():
    """`MICRODUCK_CHASE` is how a battery says which variant it measured
    (`--tag` only names the file). It applies to a brain built WITHOUT
    params — the benchmark's and the lab's path — never over params a
    caller passed, and a name or a value it cannot read raises rather than
    silently measuring the default."""
    import os

    from microduck_local.brain.controllers import ChaseParams as CP
    assert CP.from_env("") == CP() and CP.from_env("  ") == CP()
    p = CP.from_env("two_stage=1, approach_back=0.15 ,head_yaw_when=always")
    assert p.two_stage is True and p.approach_back == 0.15 and p.head_yaw_when == "always"
    assert p.approach_speed == CP().approach_speed                  # untouched knobs keep the shipped value
    assert CP.from_env("two_stage=off").two_stage is False
    for bad in ("nope=1", "two_stage", "two_stage=maybe", "bump_stand_states=lineup"):
        with pytest.raises(ValueError):
            CP.from_env(bad)
    old = os.environ.get("MICRODUCK_CHASE")
    os.environ["MICRODUCK_CHASE"] = "two_stage=1"
    try:
        assert Chase().p.two_stage is True                          # no params: the environment is read
        assert Chase(ChaseParams()).p.two_stage is False            # params given: never overridden
    finally:
        if old is None:
            del os.environ["MICRODUCK_CHASE"]
        else:
            os.environ["MICRODUCK_CHASE"] = old


def test_a_line_up_already_on_the_kick_line_walks_straight_in():
    """`lineup_lat`: stage one's job is to put the duck ON the kick line,
    squared up, short of the spot — so a duck already there starts stage two
    where it stands. With it off (the shipped 0) the very same pose sends it
    back to a pre-spot it has already walked past, which on this walker is a
    turn in place with the ball at its feet, not a step backwards."""
    p = ChaseParams(two_stage=True, lineup_lat=0.06)
    spot = (0.5, 0.06, "kick_right", 0.0, "kick")             # kick spot, line along +x

    def at(brain, odom, t=1.0):
        brain._senses = Senses(t=t)
        brain.state, brain.t_state, brain.lined = "lineup", 0.0, False
        brain.spot = spot
        return brain.step(Senses(t=t, odom=odom))

    b = Chase(p, goal=(3.0, 0.0))
    it = at(b, (0.35, 0.05, 0.0))                             # on the line, 15 cm short, squared
    assert b.lined and it.twist[0] == p.approach_speed and abs(it.twist[2]) < 0.1
    at(b, (0.35, 0.25, 0.0))
    assert not b.lined                                        # 19 cm off the line: stage one takes it there
    at(b, (0.35, 0.05, 0.5))
    assert not b.lined                                        # squared to 0.5 rad: not squared enough
    at(b, (0.15, 0.06, 0.0))
    assert not b.lined                                        # further back than the pre-spot: walk to it first
    at(b, (0.49, 0.06, 0.0))
    assert not b.lined                                        # inside `lineup_tol` of the spot: the settle's job
    off = Chase(ChaseParams(two_stage=True), goal=(3.0, 0.0))      # the knob off: the shipped two-stage path
    it = at(off, (0.35, 0.05, 0.0))
    assert not off.lined and it.twist[0] <= TURN_KICK and abs(it.twist[2]) == 1.0   # turning back to the pre-spot
