"""Achievement gating for the `run` scorecard.

The defect it addresses, measured with `select-run` (deterministic export,
mean body-frame forward velocity) over one run's checkpoints:

    steps    ep_rew    m/s      air_time  stay_upright  keep_pace
     254k       397   +0.295      0.114       1.381       0.587
    1.00M       708   -0.098      0.374       1.532       0.472

correlation(ep_rew, ground speed) = -0.90. Every positive shaping term gated
on `_run_cmd_norm`, the COMMANDED speed, so the whole shaping budget was
collectable by a duck that ignored the command — and air_time's ceiling
(2.0 x 3.0 = 6.0) sat above keep_pace's entire 1.0 x 4.0, letting the
gait-shaping term outbid the task it was meant to shape.
"""


import numpy as np
import pytest

from microduck_local import contract as C

pytestmark = pytest.mark.skipif(
    not C.MICRODUCK_RL_DIR.exists(), reason="upstream microduck_rl checkout not found")


def _env():
    from microduck_local.behaviors import BehaviorEnv
    return BehaviorEnv("run", seed=0, max_episode_s=2.0,
                       domain_rand=False, obs_noise=False)


def test_the_gate_is_off_by_default_because_it_has_not_earned_one(monkeypatch):
    """Gating air_time ALONE — the shipping variant — measured +0.195 on the
    per-step reward/speed correlation, ahead on 2/4 paired seeds, interval
    [-1.93, +2.32]: unresolved. The resolved +0.366 came from also gating
    `stay_upright`, which a prior recorded measurement says caps local speed
    at 0.27 m/s. So the default stays off until it has seeds behind it."""
    from microduck_local.behaviors.locomotion import _run_gate_on, _run_track_gate
    env = _env()
    env.reset(seed=1)

    monkeypatch.delenv("MICRODUCK_RUN_GATE", raising=False)
    assert _run_gate_on(env) is False
    assert _run_track_gate(env) == 1.0, "off must be an exact no-op, not ~1.0"

    monkeypatch.setenv("MICRODUCK_RUN_GATE", "1")
    env._step_cache.clear()
    assert _run_gate_on(env) is True


def test_stay_upright_is_never_gated(monkeypatch):
    """`test_run_upright_is_gpu_run_std_and_additive` records that
    multiplying upright onto the speed term "is why local speed saturated at
    0.27 m/s". Gating it was tried again here and reverted; this stops a
    third attempt from landing silently."""
    from microduck_local.behaviors import locomotion as L

    monkeypatch.setenv("MICRODUCK_RUN_GATE", "1")
    env = _env()
    env.reset(seed=1)
    env.step(np.zeros(14, np.float32))
    monkeypatch.setattr(L, "_run_speed", lambda e: 0.0)
    env._step_cache.clear()
    gated = L._run_upright(env)
    monkeypatch.setattr(L, "_run_speed", lambda e: 1.0)
    env._step_cache.clear()
    assert L._run_upright(env) == pytest.approx(gated), (
        "stay_upright must not depend on achieved speed")


def test_the_gate_is_exactly_floor_plus_tracking(monkeypatch):
    """Pin the algebra, not just 'it got smaller'."""
    monkeypatch.setenv("MICRODUCK_RUN_GATE", "1")
    from microduck_local.behaviors.locomotion import (
        _RUN_GATE_FLOOR,
        _run_speed,
        _run_track_gate,
    )
    env = _env()
    env.reset(seed=1)
    for _ in range(5):
        env.step(np.zeros(14, np.float32))
        env._step_cache.clear()
        track = _run_speed(env)
        env._step_cache.clear()
        want = _RUN_GATE_FLOOR + (1.0 - _RUN_GATE_FLOOR) * track
        assert _run_track_gate(env) == pytest.approx(want, abs=1e-12)
    assert 0.0 <= _RUN_GATE_FLOOR < 1.0


def test_a_duck_that_does_not_track_cannot_collect_the_shaping_budget(monkeypatch):
    """The substitution the gate exists to stop: with tracking at zero, the
    gated terms fall to the floor share instead of paying in full."""
    from microduck_local.behaviors import locomotion as L

    env = _env()
    env.reset(seed=1)
    env.step(np.zeros(14, np.float32))

    monkeypatch.setattr(L, "_run_speed", lambda e: 0.0)
    monkeypatch.setenv("MICRODUCK_RUN_GATE", "1")
    env._step_cache.clear()
    assert L._run_track_gate(env) == pytest.approx(L._RUN_GATE_FLOOR)

    monkeypatch.setattr(L, "_run_speed", lambda e: 1.0)
    env._step_cache.clear()
    assert L._run_track_gate(env) == pytest.approx(1.0)


def test_air_time_can_no_longer_outbid_keep_pace(monkeypatch):
    """The arithmetic that made the drift profitable. Ungated, a duck
    marching in place collects air_time's full 6.0 while forfeiting
    keep_pace's 4.0 — a net GAIN for not walking. Gated at zero tracking it
    keeps only the floor share, which is below what it gave up."""
    from microduck_local.behaviors import BEHAVIORS
    from microduck_local.behaviors.locomotion import _RUN_GATE_FLOOR

    w = {t.key: t.weight for t in BEHAVIORS["run"].terms}
    air_ceiling = 2.0 * w["air_time"]          # 1.0 per foot, two feet
    pace_ceiling = 1.0 * w["keep_pace"]
    assert air_ceiling > pace_ceiling, "the defect this guards would not reproduce"
    assert air_ceiling * _RUN_GATE_FLOOR < pace_ceiling, (
        "at zero tracking the gait term must be worth less than the task term "
        "it would be traded for")


# --------------------------------------------------------------------------
# The fall penalty.
# --------------------------------------------------------------------------

def test_the_fall_penalty_is_inert_by_default():
    """Weight 0 until an A/B moves it — same rule as every other change to a
    shipped recipe."""
    from microduck_local.behaviors import BEHAVIORS

    fall = {t.key: t for t in BEHAVIORS["run"].terms}["fall"]
    assert fall.weight == 0.0
    assert fall.is_penalty is True, (
        "the trainer's sign guard aborts a run whose *_penalty sum goes positive; "
        "this term has to be under it")


def test_the_penalty_fires_once_on_the_terminal_step(monkeypatch):
    """A one-off cost, not a per-step tax on being near the floor: it must be
    zero on every step the episode survives."""
    from microduck_local.behaviors.locomotion import _run_fall_pen, _run_fell

    env = _env()
    env.reset(seed=1)
    fired = 0
    for _ in range(400):
        _, _, term, trunc, _ = env.step(np.zeros(14, np.float32))
        env._step_cache.clear()
        v = _run_fall_pen(env)
        assert v in (0.0, -1.0)
        fired += v < 0
        if term or trunc:
            # The term's predicate must agree with the env's own decision —
            # `_compute_reward` runs BEFORE `step` sets `terminated`, so this
            # recomputes it rather than reading it.
            assert _run_fell(env) == term
            break
    assert fired == 1, f"fired on {fired} steps; it is a terminal cost, not a per-step one"


def test_the_weight_is_read_from_the_environment(monkeypatch):
    """The trick trainer runs as a subprocess, so the knob has to survive the
    process boundary and land in behavior.json."""
    import importlib

    from microduck_local.behaviors import locomotion as L

    monkeypatch.setenv("MICRODUCK_RUN_FALL_PENALTY", "30")
    assert L._run_fall_weight() == 30.0
    monkeypatch.setenv("MICRODUCK_RUN_FALL_PENALTY", "nonsense")
    assert L._run_fall_weight() == 0.0, "a bad value must not crash a training run"
    monkeypatch.setenv("MICRODUCK_RUN_FALL_PENALTY", "-5")
    assert L._run_fall_weight() == 0.0, "the weight is a magnitude; the term carries the sign"
    importlib.reload  # noqa: B018  (kept explicit: the term list is built at import)


def test_a_recipe_that_never_terminates_on_a_fall_pays_nothing():
    """Tricks like the headstand set `terminate_on_fall=False` on purpose —
    'it can try, crumble, and try again'. Charging them would be a tax on
    every attempt."""
    from microduck_local.behaviors import BehaviorEnv
    from microduck_local.behaviors.locomotion import _run_fall_pen

    env = BehaviorEnv("run", seed=0, max_episode_s=2.0, domain_rand=False,
                      obs_noise=False, terminate_on_fall=False)
    env.reset(seed=1)
    for _ in range(120):
        env.step(np.zeros(14, np.float32))
        env._step_cache.clear()
        assert _run_fall_pen(env) == 0.0
