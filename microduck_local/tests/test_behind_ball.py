"""Getting behind the ball before touching it (`behind_ball`,
`approach_keepout`; roadmap Track 4 s6 F.2).

The shipped planner caps its walk-round at `aim_max`, so a duck standing
between the ball and the goal it attacks is offered a spot BESIDE the ball
and a kick square across the pitch, and the arrival - 0.10 m from the ball,
inside `duck_touch` - is itself the contact that sends it the wrong way.
`behind_ball` stages squarely behind the ball instead, whenever the best kick
the stance allows would leave the ball more than `behind_ball_cos` off the
goal; `approach_keepout` bends the walk round the ball rather than through
it. Both off, the plan is the old one, to the bit.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, "tests")
from test_ball_out import _seen_ball  # noqa: E402

from microduck_local.brain.controllers import Chase, ChaseParams, _wrap  # noqa: E402
from microduck_local.brain.runtime import Senses  # noqa: E402
from microduck_local.sensors.detector import Detection, DetectionFrame  # noqa: E402

HX, HY = 1.7, 1.425
BALL = (0.0, 0.0)
R = 0.6                      # how far out the duck stands in these scenarios


def _brain(**knobs) -> Chase:
    return Chase(ChaseParams(**knobs), goal=(HX, 0.0), bounds=(HX, HY), goal_w=0.7, duck_id="d0")


def _at(bearing_deg: float, r: float = R):
    """Odometry for a duck standing `bearing_deg` round the ball, facing it.
    0 deg = between the ball and the goal it attacks (the bad side)."""
    a = math.radians(bearing_deg)
    x, y = BALL[0] + r * math.cos(a), BALL[1] + r * math.sin(a)
    return (x, y, math.atan2(BALL[1] - y, BALL[0] - x))


def _plan(b: Chase, odom):
    b.spot, b._spot_ball = None, None
    return b._plan(odom, _seen_ball(b, odom, *BALL))


def _round(spot) -> float:
    """Where a spot sits around the ball, in degrees from the goal direction:
    0 = the goal side, 180 = squarely behind."""
    return abs(math.degrees(math.atan2(spot[1] - BALL[1], spot[0] - BALL[0])))


def _closest(p, q, c) -> tuple[float, float]:
    dx, dy = q[0] - p[0], q[1] - p[1]
    l2 = max(dx * dx + dy * dy, 1e-9)
    f = min(1.0, max(0.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / l2))
    return math.hypot(c[0] - (p[0] + f * dx), c[1] - (p[1] + f * dy)), f


# -- the target half ---------------------------------------------------------

def test_the_shipped_plan_stops_short_of_behind_the_ball():
    """The thing this item is for: from the worst stance the shipped brain
    plans a spot beside the ball, not behind it, because `aim_max` caps the
    walk-round at 60 deg."""
    x, y, foot, h, mode = _plan(_brain(), _at(0))
    assert mode == "kick"
    assert _round((x, y)) < 120.0            # measured 97 deg: beside, not behind


def test_behind_ball_stages_squarely_behind_the_ball():
    b = _brain(behind_ball=0.35)
    x, y, foot, h, mode = _plan(b, _at(0))
    assert mode == "around"
    assert foot is None                      # a staging spot is not a kick spot
    assert _round((x, y)) > 175.0            # on the ball-to-goal line, behind the ball
    assert math.isclose(math.hypot(x - BALL[0], y - BALL[1]), 0.35, abs_tol=1e-6)
    # …and it faces the goal from there: the ball-to-goal line is +x here.
    assert math.isclose(h, 0.0, abs_tol=1e-6)


def test_the_staging_spot_is_outside_the_touch_radius():
    """Arriving at it must not be a contact - that is the whole point."""
    p = ChaseParams(behind_ball=0.35)
    x, y, _, _, mode = _plan(_brain(behind_ball=0.35), _at(0))
    assert mode == "around"
    assert math.hypot(x - BALL[0], y - BALL[1]) > p.duck_touch


def test_it_releases_as_soon_as_the_kick_would_gain_ground():
    """Self-releasing on the OUTCOME, not on where the duck stands: a duck
    already behind the ball plans a kick, never a walk round."""
    x, y, foot, h, mode = _plan(_brain(behind_ball=0.35), _at(180))
    assert mode == "kick"
    assert foot in ("kick_left", "kick_right")


def test_the_threshold_widens_the_staging_zone_monotonically():
    """One brain PER STANCE. `_plan` clears `spot`/`_spot_ball` but the
    walk-round latch (`_around`) is a commitment that deliberately outlives a
    refresh, so a shared brain carried it from one stance into the next and
    the set measured the latch's release geometry instead of the threshold:
    13/16/17 stances staged against 1/7/10 with a fresh brain, and every
    stance from 40 deg on staged even at 0.0, where the knob's own note says
    only the single worst one should."""
    def staged(thr):
        return {d for d in range(-180, 180, 10)
                if _plan(_brain(behind_ball=0.35, behind_ball_cos=thr), _at(d))[4] == "around"}
    loose, mid, tight = staged(0.0), staged(0.5), staged(0.7)
    assert loose < mid < tight               # strict subsets: a stricter bar stages more
    assert 0 in loose                        # the worst stance stages at every setting
    assert len(loose) == 1                   # …and at the loosest bar, ONLY that one
    assert -180 not in tight                 # being behind the ball never stages (180 is
                                             # not in the range, so the old spelling of
                                             # this line could not fail)


def test_a_corner_that_cannot_hold_the_staging_spot_falls_back_to_the_kick():
    """A bad touch beats no touch: when the body cannot stand behind the ball
    the old plan runs rather than the duck standing still."""
    b = _brain(behind_ball=0.35, board_push=0.0)
    # A ball on OUR OWN goal line with the duck up-pitch of it: the stance is
    # the worst there is, and "behind the ball" is 0.25 m outside the board.
    ball = (-HX + 0.10, 0.0)
    odom = (ball[0] + 0.3, 0.0, math.pi)
    b.spot, b._spot_ball = None, None
    assert not b._spot_body_clear(ball[0] - 0.35, 0.0)     # the premise
    mode = b._plan(odom, _seen_ball(b, odom, *ball))[4]
    assert mode != "around"


# -- the path half -----------------------------------------------------------

def test_the_keep_out_bends_a_walk_that_would_cross_the_ball():
    b = _brain(behind_ball=0.35, approach_keepout=0.30)
    odom = _at(0)
    target = (BALL[0] - 0.35, BALL[1])                    # squarely behind: straight through the ball
    straight, _ = _closest(odom[:2], target, BALL)
    assert straight < 0.01                               # the straight line runs over it
    way = b._keep_off(odom, target, BALL, 0.30)
    bent, _ = _closest(odom[:2], way, BALL)
    assert bent >= 0.30 - 1e-6                           # the first leg keeps its distance


def test_the_keep_out_leaves_a_clear_line_alone():
    b = _brain(approach_keepout=0.30)
    odom = _at(180)                                      # already behind: nothing to go round
    target = (BALL[0] - 0.35, BALL[1])
    assert b._keep_off(odom, target, BALL, 0.30) == target


def test_the_keep_out_never_fences_the_duck_off_from_its_own_spot():
    """The effective radius is the target's own distance from the ball, so a
    kick spot 0.10 m from it is always reachable however big the knob."""
    b = _brain(approach_keepout=1.0)
    odom = _at(180)
    target = (BALL[0] - 0.10, BALL[1])                   # a kick spot's distance
    assert b._keep_off(odom, target, BALL, 1.0) == target
    # ...and approached from the wrong side it is bent, but only to 0.10 m.
    way = b._keep_off(_at(0), target, BALL, 1.0)
    bent, _ = _closest(_at(0)[:2], way, BALL)
    assert 0.09 <= bent <= 0.11


def test_the_keep_out_goes_the_short_way_round():
    """The side the duck is already on (`duel_side`'s lesson: the far side
    steers the walk across the ball, not round it)."""
    b = _brain(approach_keepout=0.30)
    target = (BALL[0] - 0.35, BALL[1])
    for side in (+1.0, -1.0):
        odom = _at(side * 20.0)                          # just off the goal line, one side or the other
        way = b._keep_off(odom, target, BALL, 0.30)
        assert side * way[1] > 0.0                       # the waypoint stays on the duck's own side


# -- off is off --------------------------------------------------------------

def test_both_knobs_off_is_the_old_plan():
    """A caller that zeros the pack (and behind_ball) still gets the pre-pack
    kick plan: a kick from every stance, never around, and from the worst
    stance the spot is beside the ball not behind it."""
    off = _brain(behind_ball=0.0, chase_behind=0.0, approach_keepout=0.0, board_push=0.0)
    for d in range(-180, 180, 15):
        x, y, foot, h, mode = _plan(off, _at(d))
        assert mode == "kick"
    assert _round(_plan(off, _at(0))[:2]) < 120.0


def test_the_keep_out_off_is_a_straight_line():
    b = _brain()
    odom = _at(0)
    target = (BALL[0] - 0.35, BALL[1])
    vx, wz, dist, bearing = b._servo(odom, target, True, 0.05, avoid=BALL, avoid_r=0.0)
    plain = b._servo(odom, target, True, 0.05)
    assert (vx, wz, dist, bearing) == plain


# -- the approach, which is where the geometry is actually cheap -------------

def _chase_aim(b: Chase, odom, ball_xy):
    """What the chase branch steers at — THE RUNNING CODE, not a restatement.

    This used to mirror the branch by hand and had already drifted from it:
    the copy applied the full `chase_behind` where the shipped code scales it
    by `0.5 * (1 + cos(ang))`, and it had no `chase_behind_attacker` gate at
    all. Both of those terms are recorded in `controllers.py` as fixes for
    measured harm, so deleting either from the real planner left every test
    here green. `Chase.chase_aim` is now called directly."""
    bearing = _wrap(math.atan2(ball_xy[1] - odom[1], ball_xy[0] - odom[0]) - odom[2])
    return b.chase_aim(odom, ball_xy[0], ball_xy[1], bearing)


def test_the_chase_bias_costs_almost_nothing_far_out_and_a_lot_close_in():
    """Why the approach is the cheap place to fix the side: the same 0.40 m
    offset is a few degrees of turn at 2 m and a big swing at 0.7 m. Taken
    30 deg off the goal line, because ON it the geometry is degenerate — see
    the next test."""
    b = _brain(chase_behind=0.40, chase_behind_upto=1.0)
    turn_at = {}
    for r in (2.0, 1.2, 0.7):
        odom = _at(30, r)
        plain = math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2]
        turn_at[r] = abs(math.degrees(_chase_aim(b, odom, BALL) - plain))
    assert turn_at[2.0] < 15.0                             # nearly free at two metres
    assert turn_at[0.7] > turn_at[1.2] > turn_at[2.0]      # and it grows as the duck closes


def test_the_chase_bias_alone_cannot_help_a_duck_dead_ON_the_goal_line():
    """The degenerate case, written down so nobody re-derives it: a duck
    exactly between the ball and the goal has the behind-point COLLINEAR with
    the ball, so the biased bearing IS the ball's bearing and the run-in goes
    straight through it. `approach_keepout` is what covers this - and it is
    why the two knobs belong together."""
    bias = _brain(chase_behind=0.40, chase_behind_upto=1.0, approach_keepout=0.0)
    odom = _at(0, 1.2)
    plain = math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2]
    assert math.isclose(_chase_aim(bias, odom, BALL), plain, abs_tol=1e-9)
    both = _brain(chase_behind=0.40, chase_behind_upto=1.0, approach_keepout=0.25)
    assert abs(_chase_aim(both, odom, BALL) - plain) > math.radians(5)


def test_the_chase_bias_aims_at_the_goal_side_of_the_ball():
    b = _brain(chase_behind=0.40, chase_behind_upto=1.0)
    odom = _at(0, 1.5)                                     # between the ball and the goal it attacks
    aim = _chase_aim(b, odom, BALL)
    # the aim point is further from the goal than the ball is: the duck is
    # steering to get on the far side of it, not at it
    ax = odom[0] + 3.0 * math.cos(aim + odom[2])
    assert ax < BALL[0]


def test_the_chase_bias_off_is_the_ball_bearing():
    b = _brain(chase_behind=0.0, chase_behind_upto=1.0)
    for d in (0, 45, 120, 180, -90):
        odom = _at(d, 1.2)
        plain = math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2]
        assert math.isclose(_chase_aim(b, odom, BALL), plain, abs_tol=1e-12)


def test_the_chase_keeps_off_the_ball_when_the_run_in_would_cross_it():
    b = _brain(chase_behind=0.40, chase_behind_upto=1.0, approach_keepout=0.25)
    odom = _at(0, 1.2)                                     # the aim point is straight past the ball
    aim = _chase_aim(b, odom, BALL)
    plain = math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2]
    assert abs(aim - plain) > math.radians(5)              # it is steered off the ball, not at it


# -- the walk-round's cost model --------------------------------------------

def test_the_cost_model_needs_a_verdict_to_trade_against():
    """No `last_select`, no trade: the rule must not fire on nothing."""
    b = _brain(behind_ball=0.30, behind_ball_rate=0.5)
    b.last_select = None
    assert b._worth_going_round(_at(0), BALL[0], BALL[1], 0.0) is False


def test_a_dearer_rate_makes_the_duck_go_round_less():
    """Monotone in the knob, which is what makes it a cost: everything the
    expensive rate accepts, the cheap one accepts too."""
    def staged(rate):
        b = _brain(behind_ball=0.30, behind_ball_rate=rate)
        return {d for d in range(-180, 180, 20) if _plan(b, _at(d))[4] == "around"}
    cheap, dear = staged(0.05), staged(5.0)
    assert dear <= cheap


def test_the_chase_bias_fades_out_as_the_duck_comes_round_the_ball():
    """What the `0.5 * (1 + cos(ang))` scaling buys, pinned by its SHAPE.

    Unscaled, the bias fired on every approach including the good ones and
    measured harm there (swings 211 -> 166, whiff 5% -> 12%, p = 0.016). So
    the test is the fade: nearly gone once the duck is mostly behind the
    ball, still there on the wrong side. Measured off-axis both times —
    dead on the goal line (0 deg or 180 deg) the duck, the ball and the aim
    point are collinear, so no offset can turn the bearing at all and the
    term is invisible. Measured here: 0.02 rad at 150 deg against 0.20 at
    30 deg; delete the scaling and the 150 deg figure jumps to 0.67."""
    b = _brain(chase_behind=0.40, chase_behind_upto=1.0)

    def bias(deg):
        odom = _at(deg)
        bearing = _wrap(math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2])
        return abs(_wrap(b.chase_aim(odom, BALL[0], BALL[1], bearing) - bearing))

    mostly_behind, wrong_side = bias(150), bias(30)
    assert mostly_behind < 0.10, f"a duck already coming round is still biased {mostly_behind:.3f} rad"
    assert wrong_side > 0.15, f"the wrong side is barely biased at all ({wrong_side:.3f} rad)"
    assert wrong_side > 5 * mostly_behind          # …and the fade is the point


def test_the_attacker_gate_keeps_the_bias_off_a_supporting_duck():
    """`chase_behind_attacker` is the other term the hand-written mirror had
    lost: with it on, only the duck whose job is the ball is biased."""
    b = _brain(chase_behind=0.40, chase_behind_upto=1.0, chase_behind_attacker=True)
    odom = _at(30)                       # the wrong side, off the degenerate line
    bearing = _wrap(math.atan2(BALL[1] - odom[1], BALL[0] - odom[0]) - odom[2])

    b.role = "support"
    assert b.chase_aim(odom, BALL[0], BALL[1], bearing) == pytest.approx(bearing, abs=1e-9)
    b.role = "attack"
    assert abs(b.chase_aim(odom, BALL[0], BALL[1], bearing) - bearing) > 0.05


# -- the shipped pack, and the line-up veto in our third --------------------

def _restore_chase_env(old: str | None) -> None:
    if old is None:
        os.environ.pop("MICRODUCK_CHASE", None)
    else:
        os.environ["MICRODUCK_CHASE"] = old


def _shipped() -> Chase:
    """The lab path: no kwargs, no MICRODUCK_CHASE, ChaseParams.from_env()."""
    old = os.environ.pop("MICRODUCK_CHASE", None)
    try:
        return Chase(goal=(HX, 0.0), bounds=(HX, HY), goal_w=0.7, duck_id="d0")
    finally:
        _restore_chase_env(old)


def _see(b: Chase, odom, bx: float, by: float, t: float = 1.0):
    """One fresh ball detection at (bx, by). Starts from search (kickoff)."""
    bear = math.atan2(by - odom[1], bx - odom[0]) - odom[2]
    rng = math.hypot(bx - odom[0], by - odom[1])
    det = DetectionFrame(t, [Detection("ball", "ball0", bear, -0.3, 0.05, rng, 0.9)])
    return b.step(Senses(t=t, det=det, det_age=0.0, speed=0.0, odom=odom))


def test_the_shipped_defaults_are_the_own_goal_pack():
    old = os.environ.pop("MICRODUCK_CHASE", None)
    try:
        p = ChaseParams.from_env()
        b = Chase()
        for obj in (p, b.p, ChaseParams()):
            assert obj.board_push == 0.40
            assert obj.chase_behind == 0.40
            assert obj.approach_keepout == 0.20
            assert obj.chase_behind_upto == -0.5
    finally:
        _restore_chase_env(old)


def test_wrong_side_in_our_third_does_not_enter_kick_lineup():
    """Criterion 2: inside lineup range, ball on the nose, between the ball
    and the goal we attack, in the defensive third — keep chasing."""
    b = _shipped()
    bx, by = -1.1, 0.0
    odom = (-0.6, 0.0, math.pi)            # 0.5 m up-pitch of the ball, facing it
    assert math.hypot(bx - odom[0], by - odom[1]) < b.p.lineup_range
    assert b._attack_x(bx) <= b.p.chase_behind_upto
    _see(b, odom, bx, by)
    assert b.state in ("chase", "turn")
    assert b.spot is None
    ball = b.tracker.best("ball", 1.0, min_hits=1)
    assert ball is not None
    aim = b.chase_aim(odom, bx, by, ball.bearing)
    assert abs(aim) > 0.05                 # keep-out bent the collinear run-in
    vx, _, wz = b.last
    assert vx != 0.0 or wz != 0.0
    # a kick spot from here would have been inside duck_touch of the ball
    sx, sy, _, _, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert mode == "kick"
    assert math.hypot(sx - bx, sy - by) < b.p.duck_touch


def test_already_behind_the_ball_in_our_third_still_lines_up():
    """Criterion 3: same third, duck already behind the ball — kick lineup."""
    b = _shipped()
    bx, by = -1.1, 0.0
    odom = (-1.55, 0.0, 0.0)               # 0.45 m behind the ball, facing it
    assert math.hypot(bx - odom[0], by - odom[1]) < b.p.lineup_range
    _see(b, odom, bx, by)
    assert b.state == "lineup"
    assert b.spot is not None and b.spot[4] in ("kick", "push")


def test_attacking_two_thirds_still_lines_up_from_the_wrong_side():
    """Criterion 3: the pack must not orbit up-pitch of chase_behind_upto."""
    b = _shipped()
    bx, by = 0.3, 0.0
    odom = (0.8, 0.0, math.pi)             # 0.5 m on the goal side, facing it
    assert b._attack_x(bx) > b.p.chase_behind_upto
    assert math.hypot(bx - odom[0], by - odom[1]) < b.p.lineup_range
    _see(b, odom, bx, by)
    assert b.state == "lineup"
    assert b.spot is not None and b.spot[4] == "kick"
