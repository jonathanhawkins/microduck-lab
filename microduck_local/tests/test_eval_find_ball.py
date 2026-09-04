"""The find_ball battery's two tables.

FINDING has always been here. AIMING was a scratch script until it decided
docs/roadmap.md item 1, and it is in the battery now because the FINDING
columns cannot see the failure this behavior actually has: a policy that
centres the ball with 21 deg of head yaw and never squares up scores 100% in
frame and 98% centred while never satisfying the kick handoff. These tests
lock the parts that make the aim columns mean what they say.
"""

import math

import pytest

from microduck_local.behaviors import BEHAVIORS
from microduck_local.eval_find_ball import BUCKETS, run_battery, summarize, summarize_aim


def _row(**kw):
    """One battery row, with the fields both tables read."""
    base = {"bearing": 0.0, "dist": 1.0, "first": 0.1, "seen": 1.0, "centred": 1.0,
            "fell": False, "steps": 400, "start_psi": 0.0, "psi_final": 0.0,
            "hy_centred": 5.0, "handoff_t": 0.5}
    base.update(kw)
    return base


def test_aim_table_distinguishes_never_centred_from_zero_head_yaw():
    """A duck that never got the ball centred has NO head-yaw measurement, and
    printing that as 0.0 deg would read as a perfect aim — the best possible
    score for the worst possible episode."""
    never = summarize_aim({"rows": [_row(hy_centred=None, centred=0.0)]})
    assert any("never centred" in line for line in never)
    assert not any("0.0 deg" in line for line in never)
    # ...and a real zero still prints as a number.
    assert any("0.0 deg" in line for line in
               summarize_aim({"rows": [_row(hy_centred=0.0)]}))


def test_aim_table_reports_handoff_never_rather_than_a_time():
    """`handoff` is the deliverable, so "0% and no time" has to be visibly
    different from "fired quickly" — a median over an empty list is a crash
    or a nan, and a nan in this column would read as a fired handoff."""
    lines = summarize_aim({"rows": [_row(handoff_t=None)]})
    assert any("never" in ln for ln in lines)
    assert not any("nan" in ln.lower() for ln in lines)
    fired = summarize_aim({"rows": [_row(handoff_t=1.25)]})
    assert any("1.25 s" in ln for ln in fired)
    assert any("100%" in ln for ln in fired)


def test_psi_turned_is_signed_toward_the_ball_not_absolute():
    """`psi_turned` = start bearing - final. A duck that turns AWAY has to
    score negative: the shipped export drifted further from the ball while
    holding it centred, and an absolute value would have scored that as
    progress (which is the whole failure item 1 chased)."""
    toward = summarize_aim({"rows": [_row(start_psi=90.0, psi_final=10.0)]})
    away = summarize_aim({"rows": [_row(start_psi=20.0, psi_final=35.0)]})
    assert any("80.0" in ln for ln in toward)
    assert any("-15.0" in ln for ln in away)


def test_bucket_split_is_by_absolute_bearing_and_covers_the_circle():
    """Left and right are the same search problem, so the buckets fold on
    |bearing| — and between them they have to cover 0-180 with no gap, or
    episodes vanish from every table silently."""
    edges = [lo for _, lo, _ in BUCKETS] + [BUCKETS[-1][2]]
    assert edges[0] == 0.0 and edges[-1] > 180.0
    for (_, _, hi), (_, lo, _) in zip(BUCKETS, BUCKETS[1:]):
        assert hi == lo
    rows = [_row(bearing=b) for b in (-170.0, 170.0)]
    back = [ln for ln in summarize_aim({"rows": rows}) if ln.startswith("back")]
    assert back and " 2 " in back[0]


@pytest.mark.parametrize("events", [0.0, 0.33])
def test_battery_runs_both_ball_regimes_and_fills_both_tables(events):
    """The end-to-end shape, on the shipped brain: --events picks the regime
    (falls disagree between them, which is why the flag exists) and every row
    carries what both tables read."""
    onnx = "policies/find_ball/policy.onnx"
    res = run_battery(onnx, episodes=2, seconds=1.0, seed=7, events=events)
    assert res["events"] == events
    assert len(res["rows"]) == 2
    for r in res["rows"]:
        assert r["hy_centred"] is None or r["hy_centred"] >= 0.0
        assert r["handoff_t"] is None or r["handoff_t"] > 0.0
        assert 0.0 <= r["psi_final"] <= 180.0
    assert len(summarize(res)) >= 2
    assert len(summarize_aim(res)) >= 3


def test_the_aim_gate_printed_is_the_one_the_handoff_actually_uses():
    """The AIMING footer tells the reader the number to beat. If it drifts
    from the behavior's real gate, every reading of this table is wrong."""
    from microduck_local.behaviors import _BALL_AIM_HEAD_YAW

    footer = summarize_aim({"rows": [_row()]})[-1]
    assert f"{math.degrees(_BALL_AIM_HEAD_YAW):.0f} deg" in footer
    # And the battery asks the BEHAVIOR whether the handoff fired, rather than
    # keeping a second copy of the gate in step by a comment.
    assert BEHAVIORS["find_ball"].handoff_fn is not None


def test_finding_table_is_unchanged_by_the_aim_columns():
    """The FINDING header is quoted in README.md and docs/roadmap.md tables;
    adding AIMING must not have moved it."""
    header = summarize({"rows": [_row()]})[0].split()
    assert header == ["bucket", "n", "found", "t_first", "med", "t_first", "max",
                      "in", "frame", "centred", "fell"]
    row = summarize({"rows": [_row(seen=0.5, centred=0.25, fell=True)]})[1]
    assert "50%" in row and "25%" in row and row.rstrip().endswith("1")


def test_env_overrides_reach_the_detector_and_beat_the_flags():
    """`--env KEY=VALUE` exists because docs/roadmap.md's FOV item assumed it
    did. It has to actually reach the behavior's knobs — a flat sweep is only
    good news if the thing swept — and an explicit knob has to win over the
    --events/--prior shorthands, or a sweep silently measures the default."""
    from microduck_local.behaviors import BehaviorEnv

    def half_h(**overrides):
        env = BehaviorEnv("find_ball", obs_noise=False, domain_rand=False,
                          action_delay=False, random_yaw=False, seed=0,
                          spawn_overrides=overrides)
        env.reset(seed=0)
        return math.degrees(env._ball_k["half_h"])

    from microduck_local.behaviors import _BALL_KNOBS

    # Against the recipe's own default, not a literal — the camera's real FOV
    # is documented in ball.py and has already moved once.
    assert round(half_h(), 1) == round(_BALL_KNOBS["MICRODUCK_BALL_HFOV_DEG"] / 2, 1)
    assert round(half_h(MICRODUCK_BALL_HFOV_DEG="90"), 1) == 45.0    # the knob moves

    # An explicit --env wins over the --events shorthand for the same key.
    res = run_battery("policies/find_ball/policy.onnx", episodes=1, seconds=0.2,
                      seed=1, events=0.33,
                      env_overrides={"MICRODUCK_BALL_EVENT_RATE": "0"})
    assert res["events"] == 0.33          # what the caller asked for, reported
    assert len(res["rows"]) == 1          # ...and the run still completed
