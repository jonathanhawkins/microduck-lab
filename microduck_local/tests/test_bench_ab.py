"""`bench-ab`: the matched-STEP comparison AGENTS.md requires before a
training change becomes a default.

The failure this guards against is the one the repo already paid for twice:
an optimization that raised steps/s 25-40% and halved reward per step, which
only a comparison at matched step counts catches. So these tests pin the two
properties that make the tool trustworthy — it never compares an arm against
steps the other arm never ran, and it refuses to call a difference it cannot
resolve.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from microduck_local.bench_ab import compare, read_progress


def write_run(d: Path, steps, vals, elapsed_per_row=1.0, extra_done=False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "progress.jsonl").open("w") as f:
        for i, (s, v) in enumerate(zip(steps, vals)):
            f.write(json.dumps({"steps": int(s), "ep_rew": float(v), "ep_len": 100.0,
                                "elapsed_s": round((i + 1) * elapsed_per_row, 1)}) + "\n")
        if extra_done:
            f.write(json.dumps({"steps": int(steps[-1]), "total": int(steps[-1]), "done": True}) + "\n")
    return d


def test_read_progress_skips_the_done_marker_and_sorts_by_step(tmp_path):
    d = write_run(tmp_path / "r", [300, 100, 200], [3.0, 1.0, 2.0], extra_done=True)
    s, v, elapsed = read_progress(d)
    # Sorted by step, so a warm-restarted run appending to the same file still
    # reads as one curve; the {"done": true} line carries no metric and is skipped.
    np.testing.assert_array_equal(s, [100, 200, 300])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
    assert elapsed > 0


def test_read_progress_rejects_a_run_with_no_such_metric(tmp_path):
    d = write_run(tmp_path / "r", [100, 200], [1.0, 2.0])
    with pytest.raises(SystemExit):
        read_progress(d, metric="not_a_metric")


def test_comparison_stops_at_the_shorter_run(tmp_path):
    """A shorter arm must not win by default: the grid ends where the shorter
    run ends, so nobody is scored against steps the other never ran."""
    short = write_run(tmp_path / "short", np.arange(1, 11) * 100, np.full(10, 5.0))
    long_ = write_run(tmp_path / "long", np.arange(1, 51) * 100, np.arange(1, 51) * 1.0)
    r = compare(short, long_)
    assert r["overlap"] == [100.0, 1000.0]
    assert r["b"]["final_step"] == 5000.0     # reported, but not compared past 1000


def test_a_clearly_better_arm_is_called(tmp_path):
    steps = np.arange(1, 201) * 1000
    base = 100 * (1 - np.exp(-steps / 50_000))
    a = write_run(tmp_path / "a", steps, base)
    b = write_run(tmp_path / "b", steps, base * 1.25)
    r = compare(a, b)
    assert r["verdict"] == "B is better"
    assert r["tail_delta"] > 0 and r["b_leads_frac"] > 0.9
    # And the reverse call is symmetric.
    assert compare(b, a)["verdict"] == "A is better"


def test_two_noisy_copies_of_the_same_recipe_are_called_unresolved(tmp_path):
    """The important negative: a tool that declares a winner between two
    seeds of one recipe would launder noise into a shipped default."""
    rng = np.random.default_rng(0)
    steps = np.arange(1, 201) * 1000
    base = 100 * (1 - np.exp(-steps / 50_000))
    a = write_run(tmp_path / "a", steps, base + rng.normal(0, 3, steps.size))
    b = write_run(tmp_path / "b", steps, base + rng.normal(0, 3, steps.size))
    assert compare(a, b)["verdict"] == "no difference this A/B can resolve"


def test_a_throughput_win_that_costs_reward_per_step_is_not_called_a_win(tmp_path):
    """The exact trap from the README: arm B ran 40% more steps per second
    and learned half as much per step. Wall clock must not enter the verdict."""
    steps = np.arange(1, 201) * 1000
    base = 100 * (1 - np.exp(-steps / 50_000))
    a = write_run(tmp_path / "a", steps, base, elapsed_per_row=1.0)
    b = write_run(tmp_path / "b", steps, base * 0.5, elapsed_per_row=0.6)   # faster, worse
    r = compare(a, b)
    assert r["b"]["steps_per_s"] > r["a"]["steps_per_s"], "arm B is genuinely faster"
    assert r["verdict"] == "A is better", "and is still judged worse, on reward per step"


def test_arms_logged_at_different_rollout_sizes_are_still_comparable(tmp_path):
    """Changing --envs changes the rollout size and so the logging cadence.
    Interpolating onto a common grid is what makes those two arms comparable
    at all; identical curves sampled differently must read as identical."""
    f = lambda s: 100 * (1 - np.exp(-s / 50_000))   # noqa: E731
    a_steps = np.arange(1, 101) * 2000
    b_steps = np.arange(1, 401) * 500
    a = write_run(tmp_path / "a", a_steps, f(a_steps))
    b = write_run(tmp_path / "b", b_steps, f(b_steps))
    r = compare(a, b)
    assert abs(r["tail_delta"]) < 0.5
    assert r["verdict"] == "no difference this A/B can resolve"


def test_non_overlapping_runs_are_refused(tmp_path):
    a = write_run(tmp_path / "a", [100, 200], [1.0, 2.0])
    b = write_run(tmp_path / "b", [900, 1000], [1.0, 2.0])
    with pytest.raises(SystemExit):
        compare(a, b)


# --------------------------------------------------------------------------
# Paired comparison — the fix for the mistake this repo actually made.
# --------------------------------------------------------------------------

def test_one_pair_never_resolves_anything():
    from microduck_local.bench_ab import paired_delta
    r = paired_delta([0.947], [0.929])
    assert r["n"] == 1 and r["lo"] is None
    assert "resolves nothing" in r["verdict"], (
        "a single training run per arm is exactly the comparison that read a "
        "0.018 difference as a verdict when the control showed it was luck")


def test_the_interval_uses_student_t_not_the_normal_value():
    """At n=2 the normal 1.96 understates the interval 6.5-fold. That is not
    academic: the real two-seed measurement below reads 'excludes 0' under
    1.96 and 'unresolved' under t, and the second is correct."""
    from microduck_local.bench_ab import paired_delta, t_critical
    assert t_critical(1) == pytest.approx(12.706)
    assert t_critical(3) == pytest.approx(3.182)
    # The measured two-seed arm: mean -0.0023, sd 0.0009.
    r = paired_delta([0.935, 0.928], [0.932, 0.926])
    assert r["mean"] == pytest.approx(-0.0025, abs=5e-4)
    assert r["lo"] < 0 < r["hi"], "two points cannot exclude zero here"
    assert "unresolved" in r["verdict"]
    # A normal-approximation interval would have been narrow enough to claim
    # a result — the exact error this guards.
    se = np.std(np.array([0.932, 0.926]) - np.array([0.935, 0.928]), ddof=1) / np.sqrt(2)
    assert r["mean"] + 1.96 * se < 0, "the wrong critical value would 'exclude 0'"


def test_pairing_recovers_a_small_consistent_effect_that_means_hide():
    """The point of pairing. Two arms whose per-seed difference is a steady
    -0.002 sit inside a +-0.02 spread of training luck; comparing MEANS at
    one seed each is noise, comparing per-seed DIFFERENCES is not."""
    from microduck_local.bench_ab import paired_delta
    rng = np.random.default_rng(0)
    luck = rng.normal(0.93, 0.02, 6)          # run-to-run variance, shared by the pair
    base = list(luck)
    cand = list(luck - 0.002)
    r = paired_delta(base, cand)
    assert r["candidate_ahead"] == 0
    assert r["verdict"] == "baseline is better"
    assert r["hi"] < 0
    # Unpaired, the same data is hopeless: one seed each is a coin flip.
    assert abs(base[0] - cand[1]) > 0.002, "unpaired, luck swamps the effect"


def test_paired_arms_must_line_up():
    from microduck_local.bench_ab import paired_delta
    with pytest.raises(ValueError):
        paired_delta([0.9, 0.9], [0.9])
    with pytest.raises(ValueError):
        paired_delta([], [])
