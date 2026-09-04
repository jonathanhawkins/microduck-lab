"""The soccer benchmark's continuous metrics (`eval_pitch.PitchMetrics`).

Goals are too rare to separate anything in a 300 s run — the same brain
measured twice gave 3 falls and 6 — so the benchmark's real ruler is the ball
progress and possession clock accumulated at the control tick. Those are sums
over 15 000 ticks, which is exactly the shape of thing that can be silently
sign-flipped or double-counted and still look plausible in a battery, so they
are driven here on hand-built states: the ball is put where the test wants it
and the accumulator is ticked directly.
"""

import json

import mujoco
import pytest

from microduck_local import contract as C
from microduck_local.eval_pitch import (
    CARRY_S,
    METRIC_FIELDS,
    POSSESSION_R,
    PitchMetrics,
    _mean_field,
    _seed_line,
    _total,
    load_done,
    run_one,
)
from microduck_local.world import World, make_pitch

FAR = (1.4, 1.1)          # a corner: no duck is anywhere near the ball


@pytest.fixture(scope="module")
def world():
    """A 1v1 pitch, never stepped: these tests own the state."""
    return World(make_pitch(), seed=0)


def _put(w, ball=None, **duck_xy) -> None:
    if ball is not None:
        q = int(w.model.jnt_qposadr[w._ball_joint])
        w.data.qpos[q:q + 2] = ball
    for did, xy in duck_xy.items():
        a = w.ducks[did].adr
        w.data.qpos[a.root_qpos:a.root_qpos + 2] = xy
    mujoco.mj_forward(w.model, w.data)


def _fresh(w, ball, **duck_xy) -> PitchMetrics:
    w.t = 0.0
    w.goal_seq = 0
    _put(w, ball=ball, **duck_xy)
    return PitchMetrics(w, {d.id: d.team for d in w.scenario.ducks})


def _tick(w, m, ball=None, **duck_xy) -> None:
    """One control step of the accumulator, with the world moved first."""
    _put(w, ball=ball, **duck_xy)
    w.t += C.CTRL_DT
    m.tick()


# -- which way is forward -------------------------------------------------------

def test_the_two_sides_attack_opposite_goals(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    assert m.sign == {"left": 1.0, "right": -1.0}      # make_pitch: left attacks +x


def test_the_same_ball_motion_is_progress_for_one_side_and_a_loss_for_the_other(world):
    """The one bug this metric could carry into a published result: a sign
    read off the pitch instead of off the team. The ball going +x is the left
    team's progress and the right team's giveaway, and vice versa."""
    got = {}
    for holder, xy in (("d0", (0.0, 0.0)), ("d1", (0.0, 0.0))):
        m = _fresh(world, (0.0, 0.0), **{holder: xy, ("d1" if holder == "d0" else "d0"): FAR})
        _tick(world, m, ball=(0.0, 0.0), **{holder: xy})       # take possession
        _tick(world, m, ball=(0.30, 0.0), **{holder: xy})      # ball moves +30 cm in x
        got[holder] = (m.progress, m.advance)
    (lp, la), (rp, ra) = got["d0"], got["d1"]
    assert lp["left"] == pytest.approx(0.30) and lp["right"] == 0.0
    assert la["left"] == pytest.approx(0.30)                   # forward for left: it advanced
    assert rp["right"] == pytest.approx(-0.30) and rp["left"] == 0.0
    assert ra["right"] == 0.0                                  # backwards for right: no advance


def test_advance_keeps_only_the_forward_part_where_progress_nets_out(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    for x in (0.0, 0.40, 0.10, 0.25):                          # +0.40, −0.30, +0.15
        _tick(world, m, ball=(x, 0.0), d0=(x, 0.0))
    assert m.progress["left"] == pytest.approx(0.25)           # net: telescopes inside the possession
    assert m.advance["left"] == pytest.approx(0.55)            # 0.40 + 0.15


def test_motion_across_y_and_the_ball_sitting_still_move_nothing(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    for y in (0.0, 0.3, -0.3, 0.0):
        _tick(world, m, ball=(0.0, y), d0=(0.0, y))
    assert m.progress["left"] == pytest.approx(0.0) and m.advance["left"] == pytest.approx(0.0)
    assert m.possession["left"] > 0                            # it was on the ball the whole time


# -- possession ------------------------------------------------------------------

def test_possession_is_the_nearest_ducks_team_inside_the_radius_and_nobody_outside_it(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(POSSESSION_R * 0.5, 0.0), d1=FAR)      # d0 on the ball
    _tick(world, m, ball=(0.0, 0.0), d0=FAR, d1=(0.0, POSSESSION_R * 0.5))      # d1 on the ball
    _tick(world, m, ball=(0.0, 0.0), d0=FAR, d1=FAR)                            # free ball
    assert m.possession["left"] == pytest.approx(C.CTRL_DT)
    assert m.possession["right"] == pytest.approx(C.CTRL_DT)
    assert m.possession_wide["left"] == pytest.approx(C.CTRL_DT)                # the wider clock agrees here


def test_only_the_nearer_duck_holds_the_ball_when_both_are_inside_the_radius(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.20, 0.0), d1=(0.05, 0.0))
    assert m.possession["right"] == pytest.approx(C.CTRL_DT) and m.possession["left"] == 0.0
    assert m.nearest()[0] == "d1"


def test_the_wide_clock_sees_a_duck_the_primary_radius_does_not(world):
    m = _fresh(world, (0.0, 0.0), d0=FAR, d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.33, 0.0), d1=FAR)   # between 0.25 and 0.40
    assert m.possession["left"] == 0.0 and m.possession_wide["left"] == pytest.approx(C.CTRL_DT)


# -- credit for the ball nobody is touching --------------------------------------

def test_a_kicked_ball_running_free_is_credited_to_whoever_last_had_it(world):
    """Most of a kick's distance happens with no duck inside POSSESSION_R. A
    progress metric that only counted contact ticks would score dribbling and
    ignore the shot, which is the thing the benchmark is about."""
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))            # d0 takes possession
    _tick(world, m, ball=(0.5, 0.0), d0=FAR)                   # ball away, nobody near
    _tick(world, m, ball=(0.9, 0.0), d0=FAR)
    assert m.progress["left"] == pytest.approx(0.9)
    assert m.possession["left"] == pytest.approx(C.CTRL_DT)     # the clock, though, stops at contact


def test_credit_lapses_after_carry_s_so_a_ball_rattling_round_the_boards_is_nobodys(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    world.t += CARRY_S + C.CTRL_DT                              # long after the last touch
    _tick(world, m, ball=(0.9, 0.0), d0=FAR)
    assert m.progress["left"] == 0.0 and m.advance["left"] == 0.0


def test_the_other_team_taking_the_ball_takes_the_credit_with_it(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    _tick(world, m, ball=(0.1, 0.0), d0=FAR, d1=(0.1, 0.0))     # d1 wins it: +0.1 still d0's
    _tick(world, m, ball=(0.4, 0.0), d0=FAR, d1=FAR)            # now d1's, and it is going the wrong way
    assert m.progress["left"] == pytest.approx(0.1)
    assert m.progress["right"] == pytest.approx(-0.3)


# -- the goal restart --------------------------------------------------------------

def test_the_ball_teleporting_back_to_the_centre_spot_after_a_goal_is_nobodys_progress(world):
    """`World.kickoff` puts the ball back on the spot. Counted as motion it
    would cancel the goal it followed almost exactly — 1.25 m the wrong way."""
    m = _fresh(world, (1.0, 0.0), d0=(1.0, 0.0), d1=FAR)
    _tick(world, m, ball=(1.0, 0.0), d0=(1.0, 0.0))
    before = dict(m.progress)
    world.goal_seq += 1                                          # a goal, and the World recentres
    _tick(world, m, ball=(0.0, 0.0), d0=FAR)
    assert m.progress == before
    _tick(world, m, ball=(0.6, 0.0), d0=FAR)                     # and the next possession starts clean
    assert m.progress == before


# -- what goes in the row ------------------------------------------------------------

def test_the_row_is_per_minute_of_play_and_carries_every_metric(world):
    m = _fresh(world, (0.0, 0.0), d0=(0.0, 0.0), d1=FAR)
    _tick(world, m, ball=(0.0, 0.0), d0=(0.0, 0.0))
    _tick(world, m, ball=(0.5, 0.0), d0=(0.5, 0.0))
    world.t = 30.0                                               # half a minute of play
    row = m.row()
    assert set(row) == set(METRIC_FIELDS)
    assert row["ballProgress"]["left"] == pytest.approx(1.0, abs=1e-3)      # 0.5 m in 30 s
    assert row["possession"]["left"] == pytest.approx(2 * C.CTRL_DT * 2, abs=1e-3)
    assert _total(row, "ballProgress") == pytest.approx(1.0, abs=1e-3)


def test_run_one_still_returns_every_field_other_tooling_reads():
    r = run_one(0, 1.0)
    for k in ("seed", "perSide", "left", "right", "kickGoals", "bumpGoals",
              "kicks", "pushes", "falls", "simSeconds", "seconds", *METRIC_FIELDS):
        assert k in r, k
    for f in METRIC_FIELDS:
        assert set(r[f]) == {"left", "right"}


# -- resuming a file written before these metrics existed ------------------------------

def _row(seed, tag="", per_side=1, seconds=300.0, **kw):
    return {"seed": seed, "tag": tag, "perSide": per_side, "seconds": seconds,
            "left": 1, "right": 0, "kickGoals": 1, "bumpGoals": 0,
            "kicks": {"d0": 5, "d1": 4}, "pushes": {"d0": 1, "d1": 0},
            "falls": {"d0": 0, "d1": 1}, "simSeconds": 300.0, **kw}


def test_an_old_row_resumes_with_the_new_metrics_missing_not_zero(tmp_path):
    """The alternative — refusing the file — would re-run an hour of seeds to
    recover metrics nobody measured then; the other alternative — filling 0.0 —
    would drag every mean toward zero and never say so."""
    f = tmp_path / "old.jsonl"
    f.write_text(json.dumps(_row(0, "shipped")) + "\n")
    done = load_done(str(f), "shipped", 1, 300.0)
    assert set(done) == {0}
    assert done[0]["kicks"] == {"d0": 5, "d1": 4}                 # what it did measure survives
    for field in METRIC_FIELDS:
        assert done[0][field] is None and _total(done[0], field) is None
    assert "progress —" in _seed_line(done[0])                    # and it prints as unmeasured


def test_a_summary_over_a_half_upgraded_file_averages_only_the_seeds_that_have_it(tmp_path):
    f = tmp_path / "mixed.jsonl"
    new = _row(1, "shipped", ballProgress={"left": 2.0, "right": 1.0},
               ballAdvance={"left": 3.0, "right": 3.0},
               possession={"left": 10.0, "right": 8.0}, possessionWide={"left": 20.0, "right": 18.0})
    f.write_text(json.dumps(_row(0, "shipped")) + "\n" + json.dumps(new) + "\n")
    rows = list(load_done(str(f), "shipped", 1, 300.0).values())
    mean, n = _mean_field(rows, "ballProgress")
    assert (mean, n) == (pytest.approx(3.0), 1)                   # 2.0 + 1.0 from the one row that has it
    assert _mean_field(rows, "possession") == (pytest.approx(18.0), 1)
    assert len(rows) == 2                                         # the old seed is still not re-run
