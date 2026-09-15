"""Costing the SIDE of the ball on the team board (`Team.side_s`; roadmap
Track 4 s6 F.2).

`Team._cost` answers "seconds to reach the ball" and the board hands the ball
to whoever answers lowest — which says nothing about what that duck can DO
when it arrives. A duck on the goal side of the ball gets there first and can
only knock it backwards; a teammate further away but behind it would arrive
able to shoot. `side_s` prices that: the seconds the walk-round would take if
this duck took the ball, added to its claim.

It is the only fix for the side problem that costs no walking — both ducks go
exactly where they were going, only the assignment changes. Off, the board is
the one it always was.
"""

from __future__ import annotations

import math

from microduck_local.brain.team import Team

HX = 1.7
BALL = (0.0, 0.0)


def _team(side_s: float = 0.0, attack: float = 1.0) -> Team:
    t = Team("cream")
    t.half_x, t.attack_sign = HX, attack
    t.side_s = side_s
    return t


def _at(bearing_deg: float, r: float) -> tuple[float, float, float]:
    """A duck `bearing_deg` round the ball at range `r`, facing it. 0 deg is
    between the ball and the goal its team attacks — the wrong side."""
    a = math.radians(bearing_deg)
    x, y = BALL[0] + r * math.cos(a), BALL[1] + r * math.sin(a)
    return (x, y, math.atan2(BALL[1] - y, BALL[0] - x))


def _claim(t: Team, who: str, pose, now: float = 1.0) -> float:
    r = math.hypot(pose[0] - BALL[0], pose[1] - BALL[1])
    t.claim(who, now, r, BALL, pose, 0.0)
    return t.cost(who, now)


def test_off_the_board_costs_only_the_distance():
    """A knob that changes nothing must change NOTHING: the same two poses
    cost the same with it off, whichever side of the ball they are on."""
    off = _team(0.0)
    near_bad = _claim(off, "d0", _at(0, 0.4))         # close, wrong side
    far_good = _claim(off, "d1", _at(180, 0.8))       # further, behind the ball
    assert near_bad < far_good                        # distance wins, as it always did


def test_the_side_cost_sends_the_better_placed_duck():
    """The whole point: a duck twice as far away but already behind the ball
    is the one that should go."""
    on = _team(6.0)
    near_bad = _claim(on, "d0", _at(0, 0.4))
    far_good = _claim(on, "d1", _at(180, 0.8))
    assert far_good < near_bad
    assert on.attacker(1.0) == "d1"


def test_a_duck_squarely_behind_the_ball_pays_nothing():
    on, off = _team(6.0), _team(0.0)
    pose = _at(180, 0.6)
    assert math.isclose(_claim(on, "d0", pose), _claim(off, "d0", pose), rel_tol=1e-9)


def test_the_penalty_is_worst_dead_on_the_goal_line_and_falls_off_smoothly():
    on = _team(6.0)
    off = _team(0.0)
    pen = {}
    for d in (0, 45, 90, 135, 180):
        pose = _at(d, 0.6)
        pen[d] = _claim(on, f"d{d}", pose) - _claim(off, f"d{d}", pose)
    assert pen[0] > pen[45] > pen[90] > pen[135] > pen[180]
    assert math.isclose(pen[180], 0.0, abs_tol=1e-9)
    assert math.isclose(pen[0], 6.0, rel_tol=1e-6)     # the full knob, dead on the line
    assert math.isclose(pen[90], 3.0, rel_tol=1e-6)    # half of it, abeam


def test_the_side_is_read_in_ATTACK_coordinates():
    """A team attacking -x must find the wrong side on the OTHER side of the
    ball — the bug `_hold_target`'s striker offset already earned once."""
    for attack in (1.0, -1.0):
        on, off = _team(6.0, attack), _team(0.0, attack)
        # `_at(0, …)` is the +x side; the wrong side for a team attacking -x
        # is the -x side, i.e. `_at(180, …)`.
        wrong = _at(0 if attack > 0 else 180, 0.6)
        right = _at(180 if attack > 0 else 0, 0.6)
        pw = _claim(on, "a", wrong) - _claim(off, "a", wrong)
        pr = _claim(on, "b", right) - _claim(off, "b", right)
        assert pw > pr
        assert math.isclose(pr, 0.0, abs_tol=1e-9)


def test_it_cannot_change_anything_with_one_duck_a_side():
    """A 1v1 board has nobody to hand the ball to, so the knob is a no-op
    there by construction — expected, not broken."""
    on = _team(6.0)
    _claim(on, "d0", _at(0, 0.4))
    assert on.attacker(1.0) == "d0"
