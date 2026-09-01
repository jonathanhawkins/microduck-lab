"""bench-envs' selection rule: the recommendation is the KNEE of the measured
throughput curve, not its peak. Pure logic over a table of Points — no MuJoCo,
no PPO, so these run in milliseconds while the sweep itself takes minutes."""

import pytest

from microduck_local.bench_envs import KNEE_TOLERANCE, Point, format_table, recommend, speedup_table
from microduck_local.ppo_hparams import N_MINI_BATCHES, N_STEPS, TARGET_BATCH, ppo_batch_size


def _pts(pairs, vec="subproc", pinned=True):
    """(envs, steps_per_s) -> Points, with a 1 s clock so steps == steps/s."""
    return [Point(envs=n, vec=vec, pinned=pinned, steps=int(rate),
                  seconds=1.0, setup_s=1.0) for n, rate in pairs]


def test_ppo_batch_size_always_divides_the_rollout_buffer():
    """Lab helpers walk the env count through 16, 18, 20, … — 18 * 256 is
    not divisible by 1024, which is the SB3 truncated-minibatch warning."""
    for n_envs in range(1, 65):
        b = ppo_batch_size(N_STEPS, n_envs)
        n = N_STEPS * n_envs
        assert n % b == 0
        assert b >= 1
        if n >= TARGET_BATCH * N_MINI_BATCHES:
            assert n // b == N_MINI_BATCHES
        else:
            assert b <= TARGET_BATCH
    assert ppo_batch_size(256, 16) == 1024   # 4 × 1024, the 16-env default
    assert ppo_batch_size(256, 12) == 1024
    assert ppo_batch_size(256, 18) == 1152   # 4 minibatches, not 6 × 768
    assert ppo_batch_size(256, 24) == 1536
    assert ppo_batch_size(256, 28) == 1792
    assert ppo_batch_size(256, 32) == 2048
    assert ppo_batch_size(256, 1) == 256
    assert N_MINI_BATCHES == 4


def test_recommends_the_knee_not_the_peak():
    """The classic saturating curve: 16 envs wins by 1%, but 12 already has
    98% of it — and the 4 cores that buys back are what keeps the lab server
    and the browser responsive during a run."""
    points = _pts([(4, 3000), (8, 5000), (12, 5900), (16, 6000), (24, 5950)])
    assert recommend(points) == 12


def test_peak_wins_when_scaling_is_genuinely_linear():
    points = _pts([(4, 1000), (8, 2000), (12, 3000), (16, 4000)])
    assert recommend(points) == 16


def test_flat_curve_recommends_the_cheapest_point():
    """More envs not helping is a real result: if every count is within the
    tolerance, take the smallest — extra workers would be pure waste."""
    points = _pts([(4, 5000), (8, 5050), (12, 4980), (16, 5010)])
    assert recommend(points) == 4


def test_curve_that_gets_worse_past_the_knee():
    """Oversubscription: throughput PEAKS mid-sweep and falls off. The pick
    must be the peak's knee, never the widest measured count."""
    points = _pts([(4, 3000), (10, 5850), (14, 6000), (18, 5200), (24, 4100)])
    assert recommend(points) == 10  # 5850 clears 6000 * 0.97 = 5820


def test_tolerance_is_honoured():
    points = _pts([(4, 5700), (8, 6000)])
    assert recommend(points, tolerance=0.10) == 4   # 5700 >= 5400
    assert recommend(points, tolerance=0.01) == 8   # 5700 < 5940
    # The default is the one the shipped defaults were chosen under, and it is
    # deliberately wider than the 5-15% run-to-run noise the sweep measured.
    assert KNEE_TOLERANCE == pytest.approx(0.10)


MEASURED_2026_08_30 = [
    (4, 6836), (6, 8574), (8, 10074), (10, 11072), (12, 12327), (14, 13013),
    (16, 14288), (18, 14529), (20, 15068), (22, 15134), (24, 15626),
    (28, 15949), (32, 16484), (40, 16860), (48, 17031), (64, 17053),
]


def test_the_measured_curve_still_picks_the_shipped_default():
    """The real sweep behind the shipped constants: 2026-08-30, 18-CPU M5 Max,
    behavior=run, fork backend, best of 3 interleaved repeats on a quiet
    machine (points repeated within 2%). viz_server.RECOMMENDED_ENVS is this
    pick; if the rule stops reproducing it, that constant is no longer
    justified."""
    assert recommend(_pts(MEASURED_2026_08_30)) == 24


def test_the_measured_curve_saturates_well_past_cpu_count():
    """The finding that makes the whole exercise worth it: on 18 CPUs the curve
    is STILL RISING at 24 workers, because PPO's rollout and its multi-threaded
    torch update take turns and extra workers fill the update's idle cores. A
    'never exceed cpu_count' rule of thumb would have left 15% on the table."""
    pts = {n: r for n, r in MEASURED_2026_08_30}
    assert pts[24] > pts[18] > pts[12]                    # past 18 CPUs, still up
    assert pts[24] / pts[10] > 1.4                        # vs the old default
    # ...and then it stops: 64 workers buy under 10% over the pick.
    assert pts[64] / pts[24] < 1.10


def test_input_order_does_not_matter():
    shuffled = _pts([(16, 6000), (4, 3000), (12, 5900), (8, 5000)])
    assert recommend(shuffled) == 12


def test_empty_table_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        recommend([])


def test_steps_per_s_and_per_env_derive_from_the_clock():
    p = Point(envs=8, vec="subproc", pinned=True, steps=12_000,
              seconds=2.0, setup_s=3.0)
    assert p.steps_per_s == pytest.approx(6000.0)
    assert p.per_env == pytest.approx(750.0)


def test_speedup_and_efficiency_are_relative_to_the_smallest_count():
    rows = speedup_table(_pts([(4, 3000), (8, 4500), (16, 6000)]))
    assert [r["envs"] for r in rows] == [4, 8, 16]
    assert rows[0]["speedup"] == pytest.approx(1.0)
    assert rows[0]["efficiency"] == pytest.approx(1.0)
    # 8 envs = 2x the workers for 1.5x the throughput -> 75% efficient.
    assert rows[1]["speedup"] == pytest.approx(1.5)
    assert rows[1]["efficiency"] == pytest.approx(0.75)
    # 16 envs = 4x the workers for 2x the throughput -> 50%.
    assert rows[2]["efficiency"] == pytest.approx(0.5)


def test_format_table_renders_every_point():
    text = format_table(_pts([(4, 3000), (12, 5900)]))
    assert "steps/s" in text and "efficiency" in text
    body = [ln for ln in text.splitlines() if ln.strip().startswith(("4 ", "12"))]
    assert len(body) == 2
    assert "3,000" in text and "5,900" in text
