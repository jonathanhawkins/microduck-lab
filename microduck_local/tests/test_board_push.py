"""The board push (`board_push`, roadmap 12g): a ball closer than the knob to
a board is walked through along the wall toward the goal from a body-clear
spot behind it, instead of a kick line-up that stands in the wall or times
out; a ball nearer the wall than the body's extent gets the line tilted into
the wall until the spot clears. Off, or in the open, the kick plan as before."""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "tests")
from test_ball_out import _seen_ball  # noqa: E402

from microduck_local.brain.controllers import Chase, ChaseParams  # noqa: E402

HX, HY = 1.5, 1.25
BODY = 0.129


def _brain(**knobs) -> Chase:
    return Chase(ChaseParams(**knobs), goal=(HX, 0.0), bounds=(HX, HY), goal_w=0.7, duck_id="d0")


def _plan(b: Chase, odom, bx, by):
    return b._plan(odom, _seen_ball(b, odom, bx, by))


def test_the_default_is_the_own_goal_pack():
    assert ChaseParams().board_push == 0.40


def test_a_ball_along_the_side_board_is_pushed_up_the_pitch_from_the_open_side():
    b = _brain(board_push=0.25)
    bx, by = 0.0, HY - 0.15
    x, y, foot, h, mode = _plan(b, (-0.5, HY - 0.4, 0.0), bx, by)
    assert mode == "push" and foot is None
    assert abs(h) < 1e-9                                              # up the pitch, along the wall
    assert x < bx and abs(y - by) < 1e-9                              # push_behind behind it on that line
    assert b._spot_body_clear(x, y)


def test_a_ball_nearer_the_wall_than_the_body_gets_a_line_tilted_into_it():
    b = _brain(board_push=0.25)
    bx, by = 0.0, HY - 0.04                                           # 4 cm off the +y wall: no along-wall spot clears
    x, y, foot, h, mode = _plan(b, (-0.5, HY - 0.4, 0.0), bx, by)
    assert mode == "push"
    assert 0.0 < h <= math.radians(45.0) + 1e-9                       # tilted toward +y, the wall, by at most the cap
    assert b._spot_body_clear(x, y) and y < by                        # the spot moved out to a body-clear gap
    assert math.hypot(x - bx, y - by) - b.p.push_behind < 1e-9


def test_the_end_boards_push_away_from_our_mouth_and_across_theirs():
    b = _brain(board_push=0.25)
    x, y, foot, h, mode = _plan(b, (-HX + 0.6, 0.9, math.pi), -HX + 0.15, 0.9)   # our end board, +y side
    assert mode == "push" and abs(h - math.pi / 2) < 1e-9                        # toward the corner, not the mouth
    x, y, foot, h, mode = _plan(b, (HX - 0.6, 0.9, 0.0), HX - 0.15, 0.9)         # their end board, +y side
    assert mode == "push" and abs(h + math.pi / 2) < 1e-9                        # across the mouth: a chance


def test_in_the_open_and_off_the_kick_plan_is_untouched():
    odom = (-0.5, 0.0, 0.0)
    on, off = _brain(board_push=0.25), _brain(board_push=0.0)
    a, c = _plan(on, odom, 0.0, 0.0), _plan(off, odom, 0.0, 0.0)
    assert a[4] == "kick" and np.allclose(a[:2], c[:2]) and a[2:] == c[2:]
    bx, by = 0.0, HY - 0.15                                           # at the board with the knob off: a kick, as before
    assert _plan(_brain(board_push=0.0), (-0.5, HY - 0.4, 0.0), bx, by)[4] == "kick"


def test_the_shipped_pack_still_pushes_a_board_ball():
    assert _plan(_brain(), (-0.5, HY - 0.4, 0.0), 0.0, HY - 0.15)[4] == "push"


def test_a_push_spot_has_its_own_line_up_tolerance():
    """5.5 cm from a push spot, squared: on the spot at `push_tol` 0.06, so
    the line-up settles; a KICK spot 5.5 cm away is not on its spot
    (`lineup_tol` 0.05) and keeps walking."""
    from microduck_local.brain.runtime import Senses
    for mode, foot, expect in (("push", None, "settle"), ("kick", "kick_right", "lineup")):
        b = _brain(gaze_still=True)
        b.state, b.lined, b.t_state = "lineup", True, 5.0
        b.spot = (0.055, 0.0, foot, 0.0, mode)
        b.step(Senses(t=5.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
        assert b.state == expect, (mode, b.state)
    b = _brain(gaze_still=True, push_tol=0.0)                          # off: the kick's tolerance for a push too
    b.state, b.lined, b.t_state = "lineup", True, 5.0
    b.spot = (0.055, 0.0, None, 0.0, "push")
    b.step(Senses(t=5.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
    assert b.state == "lineup"


def test_a_push_spot_has_its_own_aim_tolerance():
    """On a push spot 25 deg off its heading: squared at `push_aim_tol` 0.5,
    so it settles; a kick spot 25 deg off is not (`aim_tol` 0.25) and turns."""
    from microduck_local.brain.runtime import Senses
    off = 0.44                                                         # rad, ~25 deg: past aim_tol, inside push_aim_tol
    for mode, foot, expect in (("push", None, "settle"), ("kick", "kick_right", "lineup")):
        b = _brain(gaze_still=True)
        b.state, b.lined, b.t_state = "lineup", True, 5.0
        b.spot = (0.0, 0.0, foot, off, mode)
        b.step(Senses(t=5.0, det=None, det_age=None, odom=(0.0, 0.0, 0.0), speed=0.0))
        assert b.state == expect, (mode, b.state)
