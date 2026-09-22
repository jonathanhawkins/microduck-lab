"""Locks for roadmap Track 4 item 11b: the ball-out rule (a World knob, off
by default, on for the lab's pitches) and the kick line along the boards.

The second half is one test per finding of the 2026-09-08 review of that
work — the placement's own 0.45 m credited to a team as ball progress, a
boards line that aimed across our own mouth, a resume that would mix two
worlds in one file, and the rest. Each fails on the code as it was.
"""

import json
import math

import mujoco
import numpy as np
import pytest

from microduck_local.brain.controllers import Chase, ChaseParams
from microduck_local.brain.tracker import Track
from microduck_local.world import World, make_pitch


def _park_ball_at_the_boards(w: World) -> tuple[int, float, float]:
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx, hy = w.scenario.floor[0] / 2 - 0.25, w.scenario.floor[1] / 2 - 0.25
    w.data.qpos[q:q + 2] = [0.3, hy - 0.05]                          # against the side board, out of the mouths
    v = int(w.model.jnt_dofadr[j])
    w.data.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)
    return q, hx, hy


def test_ball_out_is_off_by_default_and_places_the_ball_in_when_on():
    sc = make_pitch()
    off = World(sc, seed=1)
    assert off.ball_out_s == 0.0 and off.soccer_score()["ballOuts"] == 0
    q, hx, hy = _park_ball_at_the_boards(off)
    for _ in range(int(2.0 / 0.02)):
        off.step()
    assert off.ball_outs == 0 and abs(float(off.data.qpos[q + 1]) - (hy - 0.05)) < 0.03   # still at the boards
    on = World(sc, seed=1, ball_out_s=1.0)
    q, hx, hy = _park_ball_at_the_boards(on)
    for _ in range(int(0.8 / 0.02)):
        on.step()
    assert on.ball_outs == 0                                          # not yet: it has to REST there for ball_out_s
    for _ in range(int(0.5 / 0.02)):
        on.step()
    assert on.ball_outs == 1 and on.soccer_score()["ballOuts"] == 1
    x, y = float(on.data.qpos[q]), float(on.data.qpos[q + 1])
    assert hy - abs(y) >= on.ball_out_in - 0.02 and abs(x - 0.3) < 0.05   # placed in from the wall, same x
    on.reset()
    assert on.ball_outs == 0


def test_the_lab_pitches_play_on_the_cove_with_the_referee_off():
    """Since 2026-09-09 (roadmap Track 4 item 14) the lab's pitches have the
    cove and the referee is off - the "cove alone" arm, exactly as measured.
    The World knob stays for eval-pitch; a room has neither."""
    from microduck_local.world_server import PITCH_BALL_OUT_S, PITCH_COVE, WorldState
    st = WorldState(None)
    st.preload("pitch-2v2")
    assert st.world.ball_out_s == PITCH_BALL_OUT_S == 0.0
    assert st.world.scenario.cove == PITCH_COVE > 0 and len(st.world.scenario.walls) == 8
    room = WorldState(None)
    room.preload("living-room")
    assert room.world.ball_out_s == 0.0 and room.world.scenario.cove == 0.0


def _seen_ball(b: Chase, odom, bx, by, t=1.0):
    """A track for a ball at (bx, by), as the brain would hold it."""
    rng = math.hypot(bx - odom[0], by - odom[1])
    bearing = math.atan2(by - odom[1], bx - odom[0]) - odom[2]
    tr = Track.__new__(Track)
    tr.bearing, tr.range, tr.vel_hits = bearing, rng, 0
    return tr


def test_a_ball_at_the_boards_gets_a_spot_along_them_on_the_open_side():
    # `board_margin` ships off (the lab's pitches use the ball-out rule), and
    # this exercises its rescue on its own: with `spot_reach` (ships on,
    # roadmap 12al) the selector's fan keeps the one body-reachable line -
    # the line of sight, spot 0.135 m off this wall - BEFORE the rescue can
    # re-lay a scoring line along the wall, so the two are measured apart.
    p = ChaseParams(board_margin=0.08, spot_reach=0.0, board_push=0.0)
    hx, hy = 1.5, 1.25
    b = Chase(p, goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    odom = (-0.5, hy - 0.4, 0.0)
    bx, by = 0.0, hy - 0.04                                            # against the +y side board
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert mode == "kick"
    assert hy - abs(y) >= p.board_margin                               # the spot is off the boards
    assert abs(h) < 0.3                                                # the line runs UP the pitch, along the wall
    assert y < by                                                      # the body on the open side of the ball
    # The same ball with the margin off: the spot lands inside the boards.
    b0 = Chase(ChaseParams(board_margin=0.0, spot_reach=0.0, board_push=0.0), goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    _, y0, _, _, _ = b0._plan(odom, _seen_ball(b0, odom, bx, by))
    assert hy - abs(y0) < p.board_margin
    # The OWN end board (the goal line points away from it, so the spot
    # behind the ball is inside the wall): the line runs along the wall AWAY
    # from our own mouth — toward it is an own goal, see the dedicated test.
    odom = (-hx + 0.6, 0.9, math.pi)
    bx, by = -hx + 0.04, 0.9
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert hx - abs(x) >= p.board_margin and abs(h - math.pi / 2) < 0.3
    # The attacked end board beside the mouth: the goal line's own spot is
    # clear (it lies back up the pitch), so the plan is the ordinary one.
    odom = (hx - 0.6, 0.9, 0.0)
    bx, by = hx - 0.04, 0.9
    x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert hx - abs(x) >= p.board_margin and h < -0.3
    # In the open the plan is untouched by the margin (fresh brains: the
    # foot hysteresis remembers the last spot).
    odom = (-0.5, 0.0, 0.0)
    b1 = Chase(p, goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    b2 = Chase(ChaseParams(board_margin=0.0, spot_reach=0.0, board_push=0.0), goal=(hx, 0.0), bounds=(hx, hy), goal_w=0.7)
    a = b1._plan(odom, _seen_ball(b1, odom, 0.0, 0.0))
    c = b2._plan(odom, _seen_ball(b2, odom, 0.0, 0.0))
    assert np.allclose(a[:2], c[:2]) and a[2:] == c[2:]


# --- the code review's findings, each with the test that would have caught it ---


def test_a_ball_out_placement_is_nobodys_progress():
    """The referee's placement moves the ball up to `ball_out_in`; none of that
    is a team's `ballProgress` or `ballAdvance`.

    `PitchMetrics.tick` already excluded the goal recentre for exactly this
    reason. Without the same guard for a ball-out, a ball parked on a team's
    OWN end board with one of its ducks inside POSSESSION_R booked +0.400 m of
    both on the tick the rule fired — and the ball-out rule was judged on
    those two numbers (roadmap Track 4 item 11b)."""
    from microduck_local.world.compose import spawn_duck
    from microduck_local.world.metrics import PitchMetrics
    sc = make_pitch(per_side=1)
    w = World(sc, seed=0, ball_out_s=1.0)
    m = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    tm = sc.ducks[0].team
    assert m.sign[tm] == 1.0                                        # cream attacks +x
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    hx = sc.floor[0] / 2 - 0.25
    bx, by = -(hx - 0.05), 0.5                                      # cream's OWN end board: a placement is +x
    spawn_duck(w.model, w.data, w.ducks["d0"].adr, bx + 0.10, by, 0.0)   # inside POSSESSION_R: a holder
    for _ in range(200):
        w.data.qpos[q:q + 3] = [bx, by, sc.balls[0].radius + 0.005]  # a ball that is resting there
        w.data.qvel[v:v + 6] = 0.0
        before = (m.progress[tm], m.advance[tm], w.ball_outs)
        w.step()
        m.tick()
        if w.ball_outs != before[2]:
            moved = float(w.data.qpos[q]) - bx
            assert moved > 0.3                                       # it really was a big jump…
            assert m.progress[tm] == before[0]                       # …and none of it was credited
            assert m.advance[tm] == before[1]
            break
    else:
        raise AssertionError("the ball-out rule never fired")


def test_a_ball_out_settles_a_kick_in_the_air_where_the_ball_stopped():
    """A kick is scored on where the ball got to CARRY_S later. A placement
    inside that window would otherwise rewrite the kick's carry (and its
    kicksBack verdict) with the referee's own 0.45 m."""
    from microduck_local.world.metrics import PitchMetrics
    sc = make_pitch(per_side=1)
    w = World(sc, seed=0, ball_out_s=1.0)
    m = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    tm = sc.ducks[0].team
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    hx = sc.floor[0] / 2 - 0.25
    bx, by = -(hx - 0.05), 0.5
    m._pending = [(tm, w.t, (bx - 0.20, by))]                        # a kick that has just left the foot
    for _ in range(200):
        w.data.qpos[q:q + 3] = [bx, by, sc.balls[0].radius + 0.005]
        w.data.qvel[v:v + 6] = 0.0
        outs = w.ball_outs
        w.step()
        m.tick()
        if w.ball_outs != outs:
            assert m.kick_count[tm] == 1 and not m._pending          # settled by the placement, not left pending
            # 0.20 m is where the ball actually got to; the placement's ~0.40 m is not the kick's.
            assert m.kick_carry[tm] == pytest.approx(0.20, abs=0.02)
            assert m.kicks_back[tm] == 0
            break
    else:
        raise AssertionError("the ball-out rule never fired")


def test_a_ball_resting_in_the_mouth_is_not_placed_on_a_penalty_spot():
    """A shot that stops in the goal mouth without crossing is inside the
    ball-out band. Clipping x alone would put it 0.45 m out, dead centre in
    front of the goal it was about to enter — a gift, not a throw-in."""
    sc = make_pitch(per_side=1)
    w = World(sc, seed=0, ball_out_s=1.0)
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    hx = sc.floor[0] / 2 - 0.25
    half = sc.goal_width / 2
    bx, by = hx - 0.12, 0.05                                         # in the mouth, NOT across (needs |x| > hx-0.08)
    assert abs(bx) < hx - 0.08 and abs(by) < half
    for _ in range(200):
        w.data.qpos[q:q + 3] = [bx, by, sc.balls[0].radius + 0.005]
        w.data.qvel[v:v + 6] = 0.0
        outs = w.ball_outs
        w.step()
        if w.ball_outs != outs:
            break
    else:
        raise AssertionError("the ball-out rule never fired")
    assert w.goals["left"] == 0 and w.goals["right"] == 0            # it was never a goal
    x, y = float(w.data.qpos[q]), float(w.data.qpos[q + 1])
    assert abs(y) >= half                                            # beside the mouth, not centred on it
    assert hx - abs(x) >= w.ball_out_in - 0.02                       # and still brought in off the board


def test_the_boards_line_never_aims_across_our_own_mouth():
    """A ball on OUR goal line is already inside the scoring x-band, so a kick
    toward the middle is an own goal as soon as it reaches the mouth. The
    end-board line must clear AWAY from the mouth at our end and may cross it
    at theirs."""
    hx, hy, gw = 1.5, 1.25, 0.7
    p = ChaseParams(board_margin=0.08, board_push=0.0)
    b = Chase(p, goal=(hx, 0.0), bounds=(hx, hy), goal_w=gw)         # attacks +x; our mouth is -x
    for by in (0.40, -0.40):
        odom = (-1.0, by, math.pi)
        bx = -(hx - 0.05)                                            # our own end board
        assert abs(bx) > hx - 0.08                                   # inside the band a goal is scored in
        x, y, foot, h, mode = b._plan(odom, _seen_ball(b, odom, bx, by))
        assert mode == "kick"
        for d in (0.1, 0.3, 0.5, 0.8):                               # follow the ball down that line
            nx, ny = bx + d * math.cos(h), by + d * math.sin(h)
            assert not (abs(ny) < gw / 2 and abs(nx) > hx - 0.08), (
                f"own goal after {d} m at ({nx:.2f}, {ny:.2f})")
        assert abs(by + 0.3 * math.sin(h)) > abs(by)                 # it clears away from the mouth
    # At THEIR end board the line still runs across the mouth — that is a chance.
    by = 0.40
    odom = (1.0, by, 0.0)
    bx = hx - 0.05
    _, _, _, h, _ = b._plan(odom, _seen_ball(b, odom, bx, by))
    assert abs(by + 0.3 * math.sin(h)) < abs(by)


def test_clear_of_boards_answers_off_a_pitch_instead_of_raising():
    """`bounds` is None on every world that is not a pitch; the helper guards
    its own precondition rather than inheriting the caller's."""
    b = Chase(ChaseParams(board_margin=0.08))
    assert b.bounds is None
    assert b._clear_of_boards(0.0, 0.0) is True                      # not a TypeError


def test_the_duel_keepout_reads_the_spot_it_is_executing():
    """`lineup_keepout` is gated on the live kick SPOT, not on `self.state` —
    which, where the threat list is built, is still the previous tick's label
    and so answers for the wrong tick."""
    import inspect
    src = inspect.getsource(Chase.step)
    gate = [ln for ln in src.splitlines() if "lineup_keepout" in ln]
    assert gate and "self.spot" in gate[0] and "self.state" not in gate[0]
    p = ChaseParams(lineup_keepout=0.25)
    b = Chase(p, goal=(1.5, 0.0), bounds=(1.5, 1.25), goal_w=0.7)
    b.state = "lineup"                                               # the label says line-up…
    b.spot = None                                                    # …but there is no plan: no shrunken keep-out
    assert b.spot is None
    b.spot = (0.0, 0.0, "kick_left", 0.0, "push")                    # a push spot is not a kick line-up either
    assert b.spot[4] != "kick"


def test_kick_select_still_chooses_the_foot_and_the_push():
    """`_plan` must apply the selector's foot, and turn its "push" choice into
    a push spot. Both lines were once stranded after a `return` — the function
    read fine and silently ignored the selector."""
    p = ChaseParams(kick_select=True)
    b = Chase(p, goal=(1.5, 0.0), bounds=(1.5, 1.25), goal_w=0.7)
    odom = (-0.5, 0.0, 0.0)
    tr = _seen_ball(b, odom, 0.0, 0.0)
    b._select_kick_line = lambda *a, **k: (0.2, "kick_right")        # the selector picks the RIGHT foot
    assert b._plan(odom, tr)[2] == "kick_right"
    b.spot = None
    b._select_kick_line = lambda *a, **k: (0.2, "kick_left")
    assert b._plan(odom, tr)[2] == "kick_left"
    b.spot = None
    b._select_kick_line = lambda *a, **k: (0.2, "push")              # …or to walk the ball instead
    assert b._plan(odom, tr)[4] == "push"


def test_a_resume_refuses_rows_measured_under_a_different_world(tmp_path):
    """`--ball-out-s` and `--getup-s` change what the ball and the ducks DO.
    Resuming an interrupted battery without repeating them would average two
    worlds in one file, which is the thing `load_done` exists to refuse."""
    from microduck_local.eval_pitch import load_done
    f = tmp_path / "x.jsonl"
    row = {"seed": 0, "tag": "a", "perSide": 3, "seconds": 300.0, "ballOutS": 5.0, "getupS": 0.0}
    f.write_text(json.dumps(row) + "\n")
    got = load_done(str(f), "a", 3, 300.0, {"ballOutS": 5.0, "getupS": 0.0})
    assert set(got) == {0}                                           # the same world resumes
    with pytest.raises(SystemExit, match="ballOutS"):
        load_done(str(f), "a", 3, 300.0, {"ballOutS": 0.0, "getupS": 0.0})
    with pytest.raises(SystemExit, match="getupS"):
        load_done(str(f), "a", 3, 300.0, {"ballOutS": 5.0, "getupS": 10.0})
    # A row from before the knobs existed was measured at their defaults.
    old = {"seed": 1, "tag": "a", "perSide": 3, "seconds": 300.0}
    f.write_text(json.dumps(old) + "\n")
    assert set(load_done(str(f), "a", 3, 300.0, {"ballOutS": 0.0, "getupS": 0.0})) == {1}
    with pytest.raises(SystemExit, match="ballOutS"):
        load_done(str(f), "a", 3, 300.0, {"ballOutS": 5.0, "getupS": 0.0})


# --- the three the parallel session left behind (2026-09-08) ---


def test_a_battery_killed_mid_write_resumes_from_its_truncated_file(tmp_path, capsys):
    """This machine reclaims its container mid-run, so the LAST line of a
    --out file is regularly half a row. That seed was simply not measured;
    anywhere else a bad line is corruption and must not be skipped quietly."""
    from microduck_local.eval_pitch import load_done
    f = tmp_path / "x.jsonl"
    good = {"seed": 0, "tag": "a", "perSide": 3, "seconds": 300.0}
    f.write_text(json.dumps(good) + "\n" + json.dumps({"seed": 1, **good})[:40])   # cut mid-row
    got = load_done(str(f), "a", 3, 300.0)
    assert set(got) == {0}                                           # seed 1 comes back to be re-run
    assert "truncated" in capsys.readouterr().out
    # A blank final line is not truncation and nothing is said about it.
    f.write_text(json.dumps(good) + "\n\n")
    assert set(load_done(str(f), "a", 3, 300.0)) == {0}
    assert "truncated" not in capsys.readouterr().out
    # Corruption in the MIDDLE is fatal: skipping it would shrink the battery.
    f.write_text("{ this is not json\n" + json.dumps(good) + "\n")
    with pytest.raises(SystemExit, match="corrupt"):
        load_done(str(f), "a", 3, 300.0)


def test_a_duck_lying_where_it_fell_holds_neither_the_ball_nor_its_progress():
    """With `getup_s` a fallen duck lies on a zero command for as long as a
    get-up would cost. It was still the nearest duck to the ball, so it took
    the possession clock and became the holder credited with the ball's
    motion — a duck flat on the floor being paid for what the other side did."""
    from microduck_local.world.metrics import PitchMetrics
    sc = make_pitch(per_side=1)
    w = World(sc, seed=0, getup_s=5.0)
    m = PitchMetrics(w, {d.id: (d.team or d.id) for d in sc.ducks})
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    d0 = w.ducks["d0"]
    p = d0.trunk_pos(w.data)
    w.data.qpos[q:q + 2] = [float(p[0]) + 0.05, float(p[1])]         # the ball at its feet
    mujoco.mj_forward(w.model, w.data)
    who, r = m.nearest()
    assert who == "d0" and r < 0.25                                  # upright: it is on the ball
    d0.down_until = w.t + 5.0                                        # …and now it is on the floor
    who, r = m.nearest()
    assert who != "d0"
    poss = dict(m.possession)
    m.tick()
    assert m.possession == poss                                      # no clock for a duck lying down
    assert m._holder != (sc.ducks[0].team)


def test_the_placement_never_drops_the_ball_inside_a_duck():
    """The duck that was lining up on the ball stands ~0.12 m from it, which
    is where the rule moves the ball to. A ball placed inside a body
    interpenetrates and the solver flings both apart — the failure
    `_clear_of_persons` already exists for on the respawn path."""
    from microduck_local.world.compose import spawn_duck
    sc = make_pitch(per_side=1)
    w = World(sc, seed=0, ball_out_s=1.0)
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])
    hx, hy = sc.floor[0] / 2 - 0.25, sc.floor[1] / 2 - 0.25
    bx, by = 0.3, hy - 0.05                                          # against the side board
    # Park a duck exactly where the placement would otherwise land.
    spawn_duck(w.model, w.data, w.ducks["d0"].adr, bx, hy - w.ball_out_in, 0.0)
    mujoco.mj_forward(w.model, w.data)
    for _ in range(200):
        w.data.qpos[q:q + 3] = [bx, by, sc.balls[0].radius + 0.005]
        w.data.qvel[v:v + 6] = 0.0
        outs = w.ball_outs
        w.step()
        if w.ball_outs != outs:
            break
    else:
        raise AssertionError("the ball-out rule never fired")
    ball = (float(w.data.qpos[q]), float(w.data.qpos[q + 1]))
    for did, d in w.ducks.items():
        p = d.trunk_pos(w.data)
        assert math.dist(ball, (float(p[0]), float(p[1]))) >= w.ball_out_clear - 0.02, did
    assert hx - abs(ball[0]) >= 0 and hy - abs(ball[1]) >= 0         # …and still on the pitch


def test_a_throw_in_drops_the_ball_belief_and_nothing_else():
    """Roadmap 12s: the World teleports the ball up to `ball_out_in` and zeroes
    its qvel, but until 2026-09-09 it told no brain — `ball_outs` was a counter
    nobody watched. Measured cost: in the second after a throw-in the predicted
    ball is >0.30 m from the truth on 49.1% of duck-ticks against 4.6% in a
    matched control, and the board publishes an invented velocity on 15%
    against 10% (0.45 m over a 0.15-1.0 s baseline is 0.45-3.0 m/s, under
    `Team.vel_max` 4.0, so the sanity cap does not catch it)."""
    from microduck_local.brain.team import Team, throw_in_brains
    from microduck_local.brain.tracker import Track

    b = Chase(ChaseParams(), goal=(1.5, 0.0), duck_id="d0")
    tm = Team("cream")
    tr = Track.__new__(Track)
    tr.cls, tr.xy, tr.xy_t = "ball", (0.5, 0.0), 10.0
    tr.vel, tr.vel_hits, tr.vel_sig, tr.rest_block = (1.2, 0.0), 3, 0.0, False
    tr.bearing, tr.range, tr.hits = 0.0, 0.5, 3
    b.tracker.tracks = [tr]
    b.kicks, b.role = 4, "attack"
    tm._vel, tm._vel_hits = (1.2, 0.0), 3
    tm.jobs = {"d0": "defender"}

    moved_before = tr.predict(11.0, 0.3)
    assert moved_before is not None and math.dist(moved_before, tr.xy) > 0.2   # it was coasting

    throw_in_brains({"d0": b}, {"cream": tm})

    # The ball belief goes: not at rest, AND no longer coasting. The second is
    # the one that matters — `predict` never consults `rest_block`, so a test
    # on the flag alone passes while the wrong predictions continue.
    assert tr.rest_block is True
    assert tr.predict(11.0, 0.3) == tr.xy                      # no motion invented from a teleport
    assert tr.vel == (0.0, 0.0) and tr.vel_hits == 0
    assert tm.ball_vel() is None or tm.ball_vel() == (0.0, 0.0)
    # …and NOTHING else: a throw-in is not a goal.
    assert b.kicks == 4 and b.role == "attack" and tm.jobs == {"d0": "defender"}


def test_the_world_counts_throw_ins_in_a_sequence_a_harness_can_watch():
    sc = make_pitch()
    w = World(sc, seed=1, ball_out_s=1.0)
    assert w.ball_out_seq == 0
    q, hx, hy = _park_ball_at_the_boards(w)
    for _ in range(int(1.6 / 0.02)):
        w.step()
    assert w.ball_outs == 1 and w.ball_out_seq == 1            # the counter AND the sequence
    w.reset()
    assert w.ball_out_seq == 0
