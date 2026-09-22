"""Locks for outcome-simulated kick selection (brain/kickselect.py, roadmap
Track 4 §6 A.3): the roll-out labels a sample by where it stops, the
selector refuses lines with an own-goal sample and prefers lines that score,
the potential field slopes toward the goal we attack, and with the knob off
`Chase._plan` is the planner as it was before the selector shipped."""

import math

import numpy as np

from microduck_local.brain.controllers import Chase, ChaseParams, _wrap
from microduck_local.brain.kickselect import (
    GOALOPP,
    GOALOWN,
    INFIELD,
    WHIFF,
    KickModel,
    Pitch,
    evaluate,
    potential,
    roll_out,
    select,
)
from microduck_local.brain.runtime import Senses
from microduck_local.sensors.detector import Detection, DetectionFrame

PITCH = Pitch(half_x=1.5, half_y=1.25, goal_w=0.7, attack_sign=1.0)
EXACT = KickModel(speed=1.4, speed_sd=0.0, dir_sd=0.0, decel=0.3)        # no scatter: the geometry alone


def test_roll_out_labels_a_sample_by_where_it_stops():
    rng = np.random.default_rng(0)
    # Straight at the attacked mouth from 0.5 m out: 1.4 m/s over 0.3 m/s^2 rolls 3.3 m, so it crosses.
    (label, end), = roll_out((1.0, 0.0), 0.0, EXACT, PITCH, rng, 1)
    assert label == GOALOPP and abs(end[0] - 1.5) < 1e-9
    # The same, aimed past the post: it reaches the board and stops there, in the field of play.
    (label, end), = roll_out((1.0, 0.0), math.atan2(0.5, 0.5), EXACT, PITCH, rng, 1)
    assert label == INFIELD and abs(end[0] - 1.5) < 1e-9 and end[1] > 0.35
    # Back toward our own mouth from in front of it.
    (label, end), = roll_out((-1.0, 0.0), math.pi, EXACT, PITCH, rng, 1)
    assert label == GOALOWN and abs(end[0] + 1.5) < 1e-9
    # A gentle kick that stops before any board.
    soft = KickModel(speed=0.3, speed_sd=0.0, dir_sd=0.0, decel=0.3)     # rolls 0.15 m
    (label, end), = roll_out((0.0, 0.0), 0.0, soft, PITCH, rng, 1)
    assert label == INFIELD and abs(end[0] - 0.15) < 1e-9


def test_the_potential_slopes_toward_the_goal_we_attack_and_away_from_our_own():
    assert potential(1.2, 0.0, PITCH) > potential(0.0, 0.0, PITCH) > potential(-1.2, 0.0, PITCH)
    assert potential(-1.2, 0.0, PITCH) < potential(-1.2, 1.0, PITCH)    # in front of our own mouth is the worst place
    mirrored = Pitch(1.5, 1.25, 0.7, attack_sign=-1.0)
    assert potential(-1.2, 0.0, mirrored) > potential(1.2, 0.0, mirrored)


def test_select_refuses_an_own_goal_line_and_prefers_a_line_that_scores():
    rng = np.random.default_rng(1)
    model = KickModel(speed=1.4, speed_sd=0.2, dir_sd=0.15, decel=0.3)
    # In front of our own mouth, the line of sight pointing into it: the
    # own-goal line is refused, and the chosen line sends the ball up the pitch.
    ball = (-1.0, 0.0)
    cands = [(_wrap(math.pi - EXACT.exit_left), "kick_left"),          # outcome straight into our net
             (_wrap(math.pi / 2 - EXACT.exit_left), "kick_left"),      # along the mouth: safe, sideways
             (_wrap(0.0 - EXACT.exit_left), "kick_left")]              # up the pitch
    v = select(ball, cands, model, PITCH, rng, n=40)
    assert v is not None and v.p_own == 0.0
    assert abs(_wrap(v.heading - (0.0 - EXACT.exit_left))) < 1e-9     # the up-pitch line wins on the potential
    # In front of THEIR mouth: the straight line scores most and is chosen.
    ball = (1.0, 0.0)
    cands = [(_wrap(0.0 - EXACT.exit_left), "kick_left"),
             (_wrap(0.9 - EXACT.exit_left), "kick_left"),
             (_wrap(-0.9 - EXACT.exit_left), "kick_left")]
    v = select(ball, cands, model, PITCH, rng, n=40)
    assert v is not None and abs(_wrap(v.heading - (0.0 - EXACT.exit_left))) < 1e-9 and v.p_goal > 0.6
    # Every candidate risky: None, so the caller keeps its own line.
    ball = (-1.2, 0.0)
    v = select(ball, [(_wrap(math.pi - EXACT.exit_left), "kick_left")], model, PITCH, rng, n=40)
    assert v is None


def test_evaluate_applies_the_foot_exit_angle_to_the_intended_line():
    rng = np.random.default_rng(2)
    # A line aimed 23.6 deg RIGHT of straight with the LEFT foot leaves straight: the exit angle cancels.
    v = evaluate((1.0, 0.0), -EXACT.exit_left, "kick_left", EXACT, PITCH, rng, 5)
    assert v.p_goal == 1.0
    # The same intended line with the RIGHT foot leaves 52 deg right: past the post, in the field.
    v = evaluate((1.0, 0.0), -EXACT.exit_left, "kick_right", EXACT, PITCH, rng, 5)
    assert v.p_goal == 0.0 and v.p_own == 0.0


def test_whiffs_stay_put_and_the_push_is_an_action_with_the_aim_the_kick_gives_it():
    """A whiffed sample leaves the ball where it is (measured 50-61% of
    swings on this floor). The push (A.4) rolls 0.64 m along the walk with
    no exit angle. With both in the model: mid-pitch, facing their goal, a
    push that always connects beats a kick that whiffs half the time on the
    potential of where the ball ends up; near their mouth the kick that can
    score is chosen; and facing our own mouth from close the push is
    refused for walking the ball in, and a kick sideways is chosen."""
    from microduck_local.brain.kickselect import PUSH, push_model
    rng = np.random.default_rng(3)
    # Whiffs: with p_whiff = 1 every sample stops at the ball.
    whiffy = KickModel(speed=1.4, speed_sd=0.0, dir_sd=0.0, decel=0.3, p_whiff=1.0)
    assert all(label == WHIFF and end == (0.5, 0.0) for label, end in roll_out((0.5, 0.0), 0.0, whiffy, PITCH, rng, 8))
    # The push model: rolls `roll` under `decel`, no exit angle.
    from dataclasses import replace
    pm = push_model(roll=0.64, dir_sd=0.0, decel=0.3)
    (label, end), = roll_out((0.0, 0.0), 0.0, replace(pm, speed_sd=0.0), PITCH, rng, 1)
    assert label == INFIELD and abs(end[0] - 0.64) < 1e-6 and pm.exit(PUSH) == 0.0
    assert 0.0 < pm.speed_sd < 0.1                                    # a little spread in play, not none
    kick = KickModel(speed=1.4, speed_sd=0.3, dir_sd=0.6, decel=0.3, p_whiff=0.5)
    models = {PUSH: push_model(0.64, 0.5, 0.3)}
    # Mid-pitch, facing their goal, under Mellmann's rule (shoot = 0): the
    # kick's few scoring samples outrank the push - which is why that rule
    # measured as the shipped brain (the push was never chosen).
    cands = [(0.0 - kick.exit_left, "kick_left"), (0.0, PUSH)]
    v = select((-0.3, 0.0), cands, kick, PITCH, rng, n=60, t_own=0.1, models=models)
    assert v is not None and v.foot == "kick_left" and 0 < v.p_goal < 0.3
    # With a shooting threshold the safe push is preferred there…
    v = select((-0.3, 0.0), cands, kick, PITCH, rng, n=60, t_own=0.1, models=models, shoot=0.3)
    assert v is not None and v.foot == PUSH
    # …and 0.5 m from their mouth something that SCORES is chosen - here the
    # push itself, which walks the ball in (0.64 m of roll, 0.5 m to the
    # line: 60% of samples), outscoring the kick that clears the threshold.
    v = select((1.0, 0.0), cands, kick, PITCH, rng, n=60, t_own=0.1, models=models, shoot=0.3)
    assert v is not None and v.p_goal >= 0.3
    # From 1.0 m out the push cannot reach and no kick clears 0.3: the push.
    v = select((0.5, 0.0), cands, kick, PITCH, rng, n=60, t_own=0.1, models=models, shoot=0.3)
    assert v is not None and v.foot == PUSH
    # 0.4 m in front of OUR mouth facing it: a push along the line of sight
    # walks the ball into our net and is refused; a kick sideways survives.
    v = select((-1.1, 0.0), [(math.pi, PUSH), (math.pi - 1.05, "kick_right")], kick, PITCH, rng, n=60, t_own=0.1, models=models, shoot=0.3)
    assert v is not None and v.foot == "kick_right"
    v_push_only = select((-1.1, 0.0), [(math.pi, PUSH)], kick, PITCH, rng, n=60, t_own=0.1, models=models, shoot=0.3)
    assert v_push_only is None


def test_the_planner_walks_the_ball_when_the_selector_chooses_the_push():
    on = ChaseParams(kick_select=True, kick_select_push=True, kick_select_p_whiff=0.5)   # this floor's whiff rate; shoot 0.3
    b = Chase(on, goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    odom = (-0.5, 0.0, 0.0)                                             # mid-pitch, facing their goal
    b.step(_senses(0.0, (0.0, 0.5), odom))
    b.step(_senses(0.1, (0.0, 0.5), odom))
    ball = b.tracker.best("ball", 0.1, min_hits=1)
    sx, sy, foot, h, mode = b._plan(odom, ball)
    assert b.last_select is not None and b.last_select.foot == "push"
    assert mode == "push" and foot is None                              # the spot is a push spot behind the ball
    assert abs(sx - (0.0 - on.push_behind)) < 0.05 and abs(sy) < 0.05


def test_a_kick_a_teammate_receives_is_worth_more_and_a_pass_line_is_offered_up_pitch():
    """D.1: a sample that stops within `pass_reach` of a teammate counts as
    received and earns `pass_bonus`; the planner offers a line straight at
    a teammate up-pitch of the ball, never at one behind it."""
    from microduck_local.brain.team import Team
    rng = np.random.default_rng(4)
    # Two identical lines (a 0.8 m/s kick rolls 1.07 m), one with a teammate where the ball stops.
    plain = evaluate((0.0, 0.0), 0.0, "kick_left", KickModel(speed=0.8, speed_sd=0.0, dir_sd=0.0, decel=0.3, exit_left=0.0), PITCH, rng, 10)
    mate_there = evaluate((0.0, 0.0), 0.0, "kick_left", KickModel(speed=0.8, speed_sd=0.0, dir_sd=0.0, decel=0.3, exit_left=0.0), PITCH, rng, 10,
                          mates=[(1.1, 0.0)], pass_reach=0.4, pass_bonus=0.6)
    assert plain.p_pass == 0.0 and mate_there.p_pass == 1.0
    assert abs(mate_there.value - (plain.value + 0.6)) < 1e-9
    # The planner: a teammate 0.8 m up-pitch adds a pass line; one behind the ball does not.
    tm = Team("cream")
    tm.half_x, tm.attack_sign = 1.5, 1.0
    on = ChaseParams(kick_select=True, kick_select_pass=True)
    b = Chase(on, goal=(1.5, 0.0), team=tm, duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
    odom = (-0.7, 0.0, 0.0)
    tm.claim("d0", 0.0, 0.5, (-0.2, 0.0), odom)
    tm.claim("d1", 0.0, 1.0, (-0.2, 0.0), (0.6, 0.5, 0.0))                  # up-pitch and to the left
    tm.claim("d2", 0.0, 2.0, (-0.2, 0.0), (-1.2, 0.0, 0.0))                 # behind the ball: no line for it
    b.step(_senses(0.0, (0.0, 0.5), odom))
    b.step(_senses(0.1, (0.0, 0.5), odom))
    ball = b.tracker.best("ball", 0.1, min_hits=1)
    captured = {}
    from microduck_local.brain import kickselect
    orig = kickselect.select

    def spy(ball_xy, lines, *a, **kw):
        captured["lines"] = list(lines)
        captured["mates"] = kw.get("mates")
        return orig(ball_xy, lines, *a, **kw)
    kickselect.select = spy
    try:
        b._plan(odom, ball)
    finally:
        kickselect.select = orig
    bx, by = b._ball_xy(odom, ball)
    to_d1 = math.atan2(0.5 - by, 0.6 - bx)
    assert any(abs(_wrap(u - to_d1)) < 1e-9 for u, _ in captured["lines"])          # a line at the up-pitch mate
    to_d2 = math.atan2(0.0 - by, -1.2 - bx)
    assert not any(abs(_wrap(u - to_d2)) < 1e-9 for u, _ in captured["lines"])      # none at the one behind
    assert sorted(captured["mates"]) == [(-1.2, 0.0), (0.6, 0.5)]                    # both count as receivers
    assert ChaseParams().kick_select_pass is False


def test_the_back_line_is_not_offered_the_push_so_it_clears_and_returns():
    """D.2's first job: with the push on offer, a defender or keeper is not
    offered it (it kicks clear and returns to its post); a striker is."""
    from microduck_local.brain import kickselect
    from microduck_local.brain.team import Team
    seen = {}
    orig = kickselect.select

    def spy(ball_xy, lines, *a, **kw):
        seen["actions"] = sorted({act for _, act in lines})
        return orig(ball_xy, lines, *a, **kw)
    assert ChaseParams().defender_clears is False                     # measured off (roadmap D.2); a knob
    on = ChaseParams(kick_select=True, kick_select_push=True, kick_select_p_whiff=0.5, defender_clears=True)
    for job, expect_push in (("defender", False), ("keeper", False), ("striker", True), (None, True)):
        tm = Team("cream")
        tm.half_x, tm.attack_sign = 1.5, 1.0
        if job:
            tm.jobs = {"d0": job}
        b = Chase(on, goal=(1.5, 0.0), team=tm, duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7, role=job)
        odom = (-0.9, 0.0, 0.0)
        b.step(_senses(0.0, (0.0, 0.5), odom))
        b.step(_senses(0.1, (0.0, 0.5), odom))
        kickselect.select = spy
        try:
            b._plan(odom, b.tracker.best("ball", 0.1, min_hits=1))
        finally:
            kickselect.select = orig
        assert ("push" in seen["actions"]) is expect_push, (job, seen["actions"])
    # And with the rule off, a defender is offered the push like anyone else.
    tm = Team("cream")
    tm.half_x, tm.attack_sign, tm.jobs = 1.5, 1.0, {"d0": "defender"}
    b = Chase(ChaseParams(kick_select=True, kick_select_push=True, kick_select_p_whiff=0.5, defender_clears=False),
              goal=(1.5, 0.0), team=tm, duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7, role="defender")
    b.step(_senses(0.0, (0.0, 0.5), (-0.9, 0.0, 0.0)))
    b.step(_senses(0.1, (0.0, 0.5), (-0.9, 0.0, 0.0)))
    kickselect.select = spy
    try:
        b._plan((-0.9, 0.0, 0.0), b.tracker.best("ball", 0.1, min_hits=1))
    finally:
        kickselect.select = orig
    assert "push" in seen["actions"]


def _senses(t, ball, odom):
    det = DetectionFrame(t, [Detection("ball", "ball0", ball[0], -0.3, 0.12, ball[1], 0.9)])
    return Senses(t=t, det=det, det_age=0.0, speed=0.3, odom=odom)


def test_with_the_knob_off_the_plan_is_unchanged_and_on_it_stays_inside_the_aim_window():
    """`_plan` with kick_select off is the shipped planner to the bit; on, the
    chosen line is inside `aim_max` of the line of sight (the same walk-round
    the clamp allows), and a duck facing its own goal is sent up the pitch."""
    def plan(p, odom, bearing, rng_range=1.0):
        b = Chase(p, goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7)
        b.step(_senses(0.0, (bearing, rng_range), odom))
        b.step(_senses(0.1, (bearing, rng_range), odom))
        ball = b.tracker.best("ball", 0.1, min_hits=1)
        return b, b._plan(odom, ball)
    assert ChaseParams().kick_select is True                        # ships on since 2026-09-07 (confirmed on fresh seeds)
    off = ChaseParams(kick_select=False, board_push=0.0)              # the pre-2026-09-07 planner
    b0, plan_off = plan(off, (0.0, 0.0, 0.0), 0.2)
    b1, plan_off2 = plan(ChaseParams(kick_select=False, board_push=0.0), (0.0, 0.0, 0.0), 0.2)
    assert plan_off == plan_off2 and b0.last_select is None and b1.last_select is None
    on = ChaseParams(board_push=0.0)                                  # isolate the selector from the board push
    # Facing our OWN mouth from 0.4 m out (the ball 0.5 m ahead of a duck at
    # x = -0.6): straight ahead puts two thirds of kicks in our net, the
    # edge lines of the aim window 7-9% (measured, see the knob). The
    # selector picks inside the window and under the tolerance, never the
    # straight line.
    odom = (-0.6, 0.0, math.pi)
    b, (sx, sy, foot, h, mode) = plan(on, odom, 0.0, 0.5)
    los = math.pi
    v = b.last_select
    assert v is not None and v.p_own <= on.kick_select_t_own
    assert abs(_wrap(v.heading - los)) <= on.aim_max + 1e-9
    assert abs(_wrap(v.heading - los)) > 0.9                            # the edge of the window, not straight at our net
    assert mode == "kick" and foot == v.foot                            # the planner laid the spot for the CHOSEN foot
    # The foot is the lever: the outcome line (exit angle applied) points
    # well away from our mouth, which the planner's own foot on that line
    # could not manage - its exit angle bends the kick back toward the ball's
    # heading (measured: 50-83% own goals with the planner's foot, 7% with
    # the other).
    exit_a = on.kick_exit_left if v.foot == "kick_left" else on.kick_exit_right
    assert abs(_wrap(v.heading + exit_a - los)) > 1.2
    # Mellmann's zero tolerance would refuse every line there and keep the clamp's.
    strict = ChaseParams(kick_select=True, kick_select_t_own=0.0, board_push=0.0)
    b, _ = plan(strict, odom, 0.0, 0.5)
    assert b.last_select is None


def test_a_duck_in_the_way_stops_the_ball_at_its_feet_and_the_selector_turns_away_from_it():
    """Roadmap C.4: an obstacle within `obs_r` of a sample's path stops it
    there, BLOCKED; one beside the path does not; a blocked ball is valued
    where it stops less BLOCK_COST, so of two lines that would otherwise
    score alike the selector takes the one that misses the body."""
    from microduck_local.brain.kickselect import BLOCK_COST, BLOCKED
    rng = np.random.default_rng(0)
    ball = (-0.5, 0.0)
    (label, (x, y)), = roll_out(ball, 0.0, EXACT, PITCH, rng, 1, obstacles=[(0.3, 0.05)], obs_r=0.15)
    assert label == BLOCKED and abs(x - 0.3) < 1e-9 and abs(y) < 1e-9        # stopped at the closest approach
    (label, (x, y)), = roll_out(ball, 0.0, EXACT, PITCH, rng, 1, obstacles=[(0.3, 0.3)], obs_r=0.15)
    assert label == GOALOPP                                                   # beside the line: no effect
    (label, _), = roll_out(ball, 0.0, EXACT, PITCH, rng, 1)                   # nothing in the way
    assert label == GOALOPP
    # Valued: a blocked ball is worth the potential where it stopped, less the cost.
    v = evaluate(ball, 0.0, "kick_left", KickModel(speed=1.0, speed_sd=0.0, dir_sd=0.0, decel=0.3,
                                                   exit_left=0.0), PITCH, rng, 4, obstacles=[(0.3, 0.0)])
    assert v.p_block == 1.0 and abs(v.value - (potential(0.3, 0.0, PITCH) - BLOCK_COST)) < 1e-9
    # Two lines toward the boards, equal but for a duck standing on one of them.
    wide = KickModel(speed=1.0, speed_sd=0.0, dir_sd=0.0, decel=0.3, exit_left=0.0)
    lines = [(math.radians(35), "kick_left"), (math.radians(-35), "kick_left")]
    free = select(ball, lines, wide, PITCH, np.random.default_rng(1), n=8)
    body = select(ball, lines, wide, PITCH, np.random.default_rng(1), n=8,
                  obstacles=[(ball[0] + 0.6 * math.cos(math.radians(35)), ball[1] + 0.6 * math.sin(math.radians(35)))])
    assert free is not None and body is not None
    assert body.heading < 0 < free.heading or (free.heading < 0 and body.heading < 0)   # never onto the body
    assert body.p_block == 0.0


def test_the_brain_feeds_the_selector_the_ducks_the_board_does_not_own():
    """`kick_select_opps`: a duck track that is not a teammate by the
    board's positions (and not our colour) is an obstacle; a track sitting
    on a teammate's claim is not. Off, the selector never hears of them."""
    from microduck_local.brain.team import Team
    from microduck_local.brain.tracker import Track
    tm = Team("cream", half_x=1.5)
    seen = {}
    for on in (True, False):
        b = Chase(ChaseParams(kick_select=True, kick_select_opps=on), goal=(1.5, 0.0), team=tm, duck_id="d0",
                  bounds=(1.5, 1.25), goal_w=0.7)
        b._senses = Senses(t=10.0)
        tm.claim("d1", 10.0, 1.0, None, (0.9, 0.3, 0.0))                     # a teammate, on the board
        b.tracker.tracks.append(Track(id=1, cls="duck", bearing=0.0, elevation=0.0, width=0.3, range=0.9, conf=0.9,
                                      born_t=9.0, last_t=9.9, xy=(0.9, 0.28)))   # that teammate, seen
        b.tracker.tracks.append(Track(id=2, cls="duck", bearing=0.0, elevation=0.0, width=0.3, range=0.6, conf=0.9,
                                      born_t=9.0, last_t=9.9, xy=(0.5, -0.1)))   # a stranger, in the lane
        assert b._opponents(10.0) == [(0.5, -0.1)]
        import microduck_local.brain.kickselect as ks
        calls = []
        real = ks.select

        def spy(*a, **k):
            calls.append(k.get("obstacles"))
            return real(*a, **k)
        ks.select = spy
        try:
            b._select_kick_line((-0.1, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0)
        finally:
            ks.select = real
        seen[on] = calls[-1]
    assert seen[True] == [(0.5, -0.1)] and seen[False] is None
    assert ChaseParams().kick_select_opps is False                            # ships off until measured


def test_a_ball_estimated_on_or_past_our_line_still_refuses_the_own_goal():
    """`roll_out` tests `0 < f` for each wall, so a ball estimate ON or past
    the goal line (a 2-5 cm error on a ball at the boards, the dominant
    dead-ball state) made that wall invisible: a swing straight into our own
    mouth read 0% own goal from x = -1.50 against 99% from -1.49 (code
    review, 2026-09-08). `evaluate` now rolls out from just inside."""
    pitch = Pitch(half_x=1.5, half_y=1.25, goal_w=0.7, attack_sign=1.0)      # our mouth at x = -1.5
    for bx in (-1.49, -1.50, -1.51, -1.60):
        v = evaluate((bx, 0.0), math.pi, "kick_left", KickModel(dir_sd=0.05, speed_sd=0.05), pitch,
                     np.random.default_rng(0), 100)
        assert v.p_own > 0.9, (bx, v.p_own)
    # and the same past the far end for a shot: a goal, not INFIELD at x = 5
    v = evaluate((1.52, 0.0), 0.0, "push", KickModel(dir_sd=0.0, speed_sd=0.0), pitch, np.random.default_rng(0), 10)
    assert v.p_goal == 1.0


def test_a_whiff_is_never_a_received_pass():
    whiffy = KickModel(speed=1.4, speed_sd=0.0, dir_sd=0.0, decel=0.3, p_whiff=1.0)
    v = evaluate((0.5, 0.0), 0.0, "kick_left", whiffy, PITCH, np.random.default_rng(0), 20,
                 mates=[(0.5, 0.3)], pass_reach=0.4, pass_bonus=0.6)
    assert v.p_pass == 0.0 and abs(v.value - potential(0.5, 0.0, PITCH)) < 1e-9      # valued where it lies, no bonus


def test_a_safe_push_that_scores_beats_one_that_only_stands_well():
    """The push-first branch ranked pushes by `value`, which EXCLUDES the
    samples that score: a push into the open mouth (value 0.0) lost to one
    that rolled to a nice spot beside it. Same rule as the kicks now."""
    from microduck_local.brain.kickselect import push_model
    pitch = Pitch(half_x=1.5, half_y=1.25, goal_w=0.7, attack_sign=1.0)
    pm = push_model(dir_sd=0.0)                                  # 0.64 m, dead straight
    v = select((1.2, 0.0), [(0.0, "push"), (math.pi / 2, "push")], KickModel(), pitch,
               np.random.default_rng(0), n=20, shoot=0.3, models={"push": pm})
    assert v is not None and v.foot == "push" and v.p_goal == 1.0 and abs(v.heading) < 1e-9


def test_a_predicted_ball_too_far_ahead_refuses_the_swing_when_the_gate_is_on():
    """Roadmap item 12a: replaying 93 play swings on the bench found the
    whiffs are swings at a ball no longer in front of the foot (inside a
    0.15 x 0.12 m box 8% whiff, outside 76%). `kick_ahead_max` refuses those
    from the fresh predicted ball; ships at 0.15 with `gaze_still` (measured: whiff
    44 -> 31% and 51 -> 41% on two blocks, the ledger flat or better)."""
    assert ChaseParams().kick_ahead_max == 0.15 and ChaseParams().gaze_still is True   # shipped 2026-09-08 (item 12c)
    b = Chase(ChaseParams(kick_ahead_max=0.15), goal=(1.5, 0.0))
    odom = (0.0, 0.0, 0.0)
    b.predicted = None
    assert b._too_far(odom) is False                                  # nothing fresh: the plan's ball stands
    b.predicted = (0.09, 0.05)
    assert b._too_far(odom) is False                                  # on the sweet spot
    b.predicted = (0.25, 0.0)
    assert b._too_far(odom) is True                                   # drifted out of reach
    b.predicted = (-0.20, 0.0)
    assert b._too_far(odom) is False                                  # behind: not this gate's business
    b = Chase(ChaseParams(kick_ahead_max=0.0), goal=(1.5, 0.0))
    b.predicted = (0.5, 0.0)
    assert b._too_far(odom) is False                                  # off


def test_the_post_kick_look_sweeps_the_head_around_the_exit_line_when_asked():
    """Roadmap 12i: with `look_sweep` on, the standing look after a kick
    gazes at `look_sweep_range` and yaws around the predicted line; off, the
    head stays where it was (the shipped brain, to the bit)."""
    from microduck_local.brain.runtime import Senses
    for sweep in (0.0, 0.8):
        b = Chase(ChaseParams(look_sweep=sweep, hunt_s=3.0), goal=(1.5, 0.0))
        b.reset()
        b._hunt_u = 0.5                                        # the kick's predicted line, 0.5 rad left
        b._look_t0 = 10.0
        yaws = []
        for k in range(20):
            t = 10.0 + k * 0.04
            intent = b.step(Senses(t=t, tof=None, tof_age=None, det=None, det_age=None, speed=0.0,
                                   odom=(0.0, 0.0, 0.0), skill=None, bumped=False))
            if b.state == "look":
                yaws.append(float(intent.head[2]))
        assert b.state in ("look", "hunt", "search")
        if sweep == 0.0:
            assert all(abs(y) < 1e-9 for y in yaws)
        else:
            assert yaws and max(yaws) > 0.9 and min(yaws) < 0.1        # sweeps to the left of 0.5 and back across it


def test_the_tof_blob_speaks_only_in_the_lineup_when_the_gate_is_on(monkeypatch):
    """Roadmap 12e: the blob is 97% the ball in `lineup`/`settle` and 85% there
    with the camera blind, against 85%/30% pooled over every state it fires in
    (scripts/probe_tof_ball.py) - so `tof_ball_lineup` restricts it to that
    population. Off, it is offered in every state, which is what the two
    earlier measurements killed."""
    import microduck_local.brain.controllers as ctl
    from microduck_local.brain.runtime import Senses
    from microduck_local.sensors.tof import TofFrame

    assert ChaseParams().tof_ball_lineup is True and ChaseParams().tof_ball_m == 0.0
    monkeypatch.setattr(ctl, "tof_floor_ball", lambda fr, r_max=0.5: (0.1, 0.25))
    # An empty-but-real frame: nothing in view, so only the patched blob speaks.
    frame = TofFrame(t=1.0, depth_mm=np.zeros((8, 8), np.uint16), valid=np.zeros((8, 8), bool),
                     mount_pos=np.array([0.05, 0.0, 0.21]), mount_rot=np.eye(3),
                     dirs_local=np.tile(np.array([1.0, 0.0, 0.0]), (8, 8, 1)))
    got = {}
    for gated in (True, False):
        for state in ("lineup", "search"):
            b = Chase(ChaseParams(tof_ball_m=0.5, tof_ball_lineup=gated), goal=(1.5, 0.0))
            b.state = state
            b.step(Senses(t=1.0, tof=frame, tof_age=0.0, det=None, det_age=None,
                          speed=0.0, odom=(0.0, 0.0, 0.0), skill=None, bumped=False))
            got[(gated, state)] = b.tof_ball is not None
    assert got[(True, "lineup")] and not got[(True, "search")]      # gated: the line-up only
    assert got[(False, "lineup")] and got[(False, "search")]        # off: everywhere, as before
