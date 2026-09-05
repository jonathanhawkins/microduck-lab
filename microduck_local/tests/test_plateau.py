"""Plateau early stopping: the detector, and the clean stop it wires into
train_behavior.

Why it exists: `brains/follow-v4` was trained for 2M decisions and its reward
curve was flat from ~900k — half the wall clock bought nothing. Trick runs
show the same shape.

Most of these tests are about NOT firing. Killing a run that is still
creeping upward is the expensive failure (a lost policy costs far more than a
wasted half hour), so the detector is judged on rising and noisy-rising
series as much as on flat ones. The pinned firing index in
`test_rise_then_flat_fires_at_a_pinned_index` is the regression lock: any
change to the smoothing or the improvement rule moves it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microduck_local.plateau import DEFAULT_PATIENCE, PlateauDetector, env_defaults

PROJECT = Path(__file__).resolve().parents[1]


def _feed(det: PlateauDetector, values, steps_per: int = 1000):
    """Feed a series one value per 'rollout'. Returns the index that fired, or None."""
    for i, v in enumerate(values):
        if det.update(steps_per * (i + 1), v):
            return i
    return None


# --- not firing: the expensive mistake ------------------------------------

def test_rising_series_never_fires():
    det = PlateauDetector(patience=3, min_steps=0, rel=0.01, window=5)
    assert _feed(det, [float(i) for i in range(1, 61)]) is None
    assert not det.fired


def test_slow_creep_below_the_threshold_still_never_fires():
    """0.5% per rollout with a 1% bar. `best` only moves on a SIGNIFICANT
    improvement, so the sub-threshold gains accumulate and eventually clear
    the bar instead of being ratcheted away — a max-tracking `best` would
    stop this run, and it is still learning."""
    det = PlateauDetector(patience=3, min_steps=0, rel=0.01, window=5)
    series = [100.0 * (1.005 ** i) for i in range(80)]
    assert _feed(det, series) is None


def test_noisy_rising_series_does_not_fire():
    """Deterministic sawtooth noise (±3) on a +1/rollout trend: individual
    rollouts fall, the smoothed window keeps climbing."""
    det = PlateauDetector(patience=4, min_steps=0, rel=0.01, window=5)
    series = [float(i) + (3.0 if i % 2 else -3.0) for i in range(1, 41)]
    assert _feed(det, series) is None


# --- firing --------------------------------------------------------------

def test_rise_then_flat_fires_at_a_pinned_index():
    """The regression lock. Series: 1..10 then flat at 10, window 3,
    patience 3, no warmup.

    Hand-checked: the window fills at index 2, the last improvement is index
    11 (window [10,10,10] beats the [9,10,10] mean by more than 1%), so
    indices 12/13/14 are the three stale evaluations and 14 fires.
    """
    det = PlateauDetector(patience=3, min_steps=0, rel=0.01, window=3)
    series = [float(i) for i in range(1, 11)] + [10.0] * 10
    assert _feed(det, series) == 14
    assert det.fired_steps == 15000
    assert det.best == 10.0
    assert det.best_steps == 12000
    assert det.stale == 3


def test_update_is_sticky_once_fired():
    det = PlateauDetector(patience=2, min_steps=0, rel=0.01, window=2)
    _feed(det, [5.0] * 10)
    assert det.fired
    # A late spike does not un-fire a decision the trainer already acted on.
    assert det.update(999_000, 1e6) is True


def test_flat_series_needs_patience_and_warmup():
    """The same flat series fires at index 6 with no warmup and index 12 with
    one — evaluations before `min_steps` may raise `best` but never age the
    staleness counter, because early training legitimately looks flat."""
    hot = PlateauDetector(patience=4, min_steps=0, rel=0.01, window=3)
    assert _feed(hot, [5.0] * 20) == 6

    warm = PlateauDetector(patience=4, min_steps=10_000, rel=0.01, window=3)
    assert _feed(warm, [5.0] * 20) == 12
    assert warm.fired_steps == 13_000
    # Patience alone is never enough: nothing fires below the warmup.
    assert warm.fired_steps >= warm.min_steps


def test_warmup_counts_absolute_steps():
    """A warm restart resumes at a nonzero counter, and the trainer feeds the
    ABSOLUTE step count — so a run already past its warmup may fire on its
    first patience-worth of stale rollouts."""
    det = PlateauDetector(patience=2, min_steps=400_000, rel=0.01, window=2)
    assert _feed(det, [7.0] * 8, steps_per=500_000) == 3


def test_negative_rewards_keep_a_real_bar():
    """rel is relative to |best|, so a run sitting at -10 needs +1 to count as
    improvement rather than any wobble above -10."""
    det = PlateauDetector(patience=2, min_steps=0, rel=0.10, window=1)
    assert _feed(det, [-10.0, -9.5, -9.4]) == 2  # neither beats -10 + 1.0
    up = PlateauDetector(patience=2, min_steps=0, rel=0.10, window=1)
    assert _feed(up, [-10.0, -8.0, -6.0, -4.0]) is None


# --- off by default ------------------------------------------------------

def test_patience_zero_disables_it_entirely():
    det = PlateauDetector(patience=0, min_steps=0, rel=0.01, window=3)
    assert not det.enabled
    assert _feed(det, [5.0] * 500) is None
    assert not det.fired
    # A disabled detector records nothing at all — no window, no best.
    assert det.best is None and det.evals == 0 and det.stale == 0


def test_module_default_patience_is_off():
    assert DEFAULT_PATIENCE == 0
    assert not PlateauDetector().enabled


def test_env_defaults_are_off_and_overridable(monkeypatch):
    for var in ("MICRODUCK_PLATEAU_PATIENCE", "MICRODUCK_PLATEAU_MIN_STEPS",
                "MICRODUCK_PLATEAU_REL", "MICRODUCK_PLATEAU_WINDOW"):
        monkeypatch.delenv(var, raising=False)
    assert env_defaults()["patience"] == 0

    monkeypatch.setenv("MICRODUCK_PLATEAU_PATIENCE", "8")
    monkeypatch.setenv("MICRODUCK_PLATEAU_MIN_STEPS", "900_000")
    monkeypatch.setenv("MICRODUCK_PLATEAU_REL", "0.02")
    monkeypatch.setenv("MICRODUCK_PLATEAU_WINDOW", "12")
    d = env_defaults()
    assert (d["patience"], d["min_steps"], d["rel"], d["window"]) == (
        8, 900_000, 0.02, 12)

    # Junk in the environment must not take a training run down.
    monkeypatch.setenv("MICRODUCK_PLATEAU_PATIENCE", "lots")
    assert env_defaults()["patience"] == 0


def test_cli_default_is_off(monkeypatch):
    """The default path — no flags, no env — must be exactly the old
    behavior: run the full --steps."""
    for var in ("MICRODUCK_PLATEAU_PATIENCE", "MICRODUCK_PLATEAU_MIN_STEPS",
                "MICRODUCK_PLATEAU_REL", "MICRODUCK_PLATEAU_WINDOW"):
        monkeypatch.delenv(var, raising=False)
    from microduck_local.train_behavior import build_parser
    args = build_parser().parse_args(["crouch"])
    assert args.plateau_patience == 0
    assert not PlateauDetector(patience=args.plateau_patience,
                               min_steps=args.plateau_min_steps,
                               rel=args.plateau_rel,
                               window=args.plateau_window).enabled

    # The lab's /teach passes no flags, so the env var is its only way in.
    monkeypatch.setenv("MICRODUCK_PLATEAU_PATIENCE", "6")
    assert build_parser().parse_args(["crouch"]).plateau_patience == 6
    # An explicit flag still wins over the environment.
    assert build_parser().parse_args(
        ["crouch", "--plateau-patience", "0"]).plateau_patience == 0


def test_plateau_module_does_not_import_torch():
    """train_behavior forks its vec-env workers BEFORE importing torch and
    imports this module at the top of the file; a torch-initialized parent
    deadlocks the fork on macOS. Checked in a subprocess so the rest of the
    suite cannot contaminate sys.modules."""
    script = (
        "import sys\n"
        "import microduck_local.plateau\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "assert not bad, bad\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, check=False, cwd=str(PROJECT))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


# --- the callback wiring -------------------------------------------------

def test_callback_feeds_only_real_measurements_and_stops_via_on_step(tmp_path):
    """Two things at once, on the real ProgressCallback with a stub model:

    * A rollout with an empty ep_info_buffer reports ep_rew 0.0 — that is a
      placeholder, not a measurement. Feeding it would pin `best` at 0 on any
      recipe whose early episodes score negative, and stop a run that was
      climbing from -50 toward -10.
    * The verdict is taken in _on_rollout_end but delivered by _on_step
      returning False, which is the only clean SB3 stop (it leaves main()'s
      final snapshot / export / "done" line intact).
    """
    from types import SimpleNamespace

    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from microduck_local.train_behavior import _progress_callback_cls

    det = PlateauDetector(patience=2, min_steps=0, rel=0.01, window=1)
    cb = _progress_callback_cls(BaseCallback)(
        tmp_path, venv=None, total_steps=1000, snap_steps=10 ** 9, plateau=det)
    cb.model = SimpleNamespace(ep_info_buffer=[],
                               policy=SimpleNamespace(log_std=torch.zeros(3)))

    cb.num_timesteps = 100
    cb._on_rollout_end()
    assert det.evals == 0 and det.best is None  # the 0.0 never reached it
    assert cb._on_step() is True

    cb.model.ep_info_buffer = [{"r": -50.0, "l": 40}]
    for step in (200, 300, 400):
        cb.num_timesteps = step
        cb._on_rollout_end()
    assert det.best == -50.0 and det.fired
    assert cb._on_step() is False  # the clean SB3 stop


# --- the real trainer ----------------------------------------------------

def _run(args: list[str], runs_dir: Path) -> subprocess.CompletedProcess:
    """A real (tiny) trainer subprocess, as in tests/test_train_resume.py:
    runs land under MICRODUCK_RUNS_DIR, and the vec-env backend is pinned so
    a developer's exported MICRODUCK_VEC_ENV cannot change what is tested."""
    env = {**os.environ, "MICRODUCK_RUNS_DIR": str(runs_dir),
           "MICRODUCK_VEC_ENV": "fork"}
    res = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", *args],
        env=env, cwd=str(PROJECT), timeout=240, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"train-behavior failed:\n{res.stdout}\n{res.stderr}"
    return res


def test_plateau_stop_still_ships_a_finished_run(tmp_path):
    """A plateau stop is a COMPLETED run, not a crash: the final snapshot,
    the ONNX export and the terminating "done" line all still happen.

    The thresholds are deliberately trigger-happy (a 2-rollout window and a
    500% improvement bar) so the stop path runs in seconds — this test is
    about the plumbing, not about a sane setting.
    """
    runs = tmp_path / "runs"
    _run(["crouch", "--envs", "2", "--steps", "20000", "--run-name", "plateau-test",
          "--snap-steps", "4000", "--plateau-patience", "2",
          "--plateau-window", "2", "--plateau-min-steps", "0",
          "--plateau-rel", "5.0"], runs)
    run_dir = runs / "plateau-test"
    for f in ("live.onnx", "model.zip", "vecnormalize.pkl", "policy.onnx"):
        assert (run_dir / f).exists(), f

    rows = [json.loads(ln)
            for ln in (run_dir / "progress.jsonl").read_text().splitlines()
            if ln.strip()]
    last, rollouts = rows[-1], [r for r in rows if "elapsed_s" in r]
    assert last.get("done") is True
    # It really stopped short, and said why — the point of the field is that
    # nobody reads this run's truncated curve as a crash later.
    assert last.get("stopped") == "plateau"
    assert last["trained_steps"] < 20000
    assert rollouts and rollouts[-1]["steps"] < 20000
    assert last["plateau"]["patience"] == 2
    assert last["plateau"]["at_steps"] <= last["trained_steps"]


def test_default_run_has_no_stop_reason(tmp_path):
    """The same trainer with no plateau flags: full budget, and a final line
    carrying exactly the keys it always carried."""
    runs = tmp_path / "runs"
    _run(["stand", "--envs", "2", "--steps", "8000", "--run-name", "no-plateau",
          "--snap-steps", "4000"], runs)
    rows = [json.loads(ln)
            for ln in (runs / "no-plateau" / "progress.jsonl").read_text().splitlines()
            if ln.strip()]
    assert rows[-1] == {"steps": 8000, "total": 8000, "done": True}
    assert [r for r in rows if "elapsed_s" in r][-1]["steps"] >= 8000


# --------------------------------------------------------------------------
# The measurement that justifies arming this on a brain run.
# --------------------------------------------------------------------------

# `select-brain`'s DETERMINISTIC in-band score at each 250k checkpoint of a
# real 2M-decision follow run (12 envs, --variety, seed 7; 40 benchmark
# episodes per checkpoint, datasheet preset, 2026-09-03). The point of
# keeping the numbers here rather than a synthetic curve: this run's `ep_rew`
# climbed 159 -> 177 over the same span, so a plateau stop watching REWARD
# never fires on it. The benchmark had stopped moving at the first checkpoint.
LEGACY_ARM_PROBE_SCORES = [
    (250_368, 0.938), (500_736, 0.895), (751_104, 0.918), (1_001_472, 0.940),
    (1_250_304, 0.935), (1_500_672, 0.937), (1_751_040, 0.938), (2_001_408, 0.934),
]


def _replay(scores, **kw):
    """Where the detector would have stopped, and what would then have shipped
    (select-brain ships the best checkpoint SEEN, not the last one)."""
    from microduck_local.plateau import PlateauDetector
    d = PlateauDetector(window=1, rel=0.01, min_steps=250_000, label="probe in_band", **kw)
    stop = None
    for steps, v in scores:
        if d.update(steps, v):
            stop = steps
            break
    kept = [s for s in scores if stop is None or s[0] <= stop]
    return stop, max(kept, key=lambda x: x[1])


def test_probe_driven_stop_halves_a_real_brain_run_at_no_measured_cost():
    """Patience 3 over the recorded probe series stops at 1.0M of 2M and
    ships the SAME checkpoint the full run would have — the headline claim,
    pinned so a change to the detector cannot quietly break it."""
    stop, (best_steps, best) = _replay(LEGACY_ARM_PROBE_SCORES, patience=3)
    assert stop == 1_001_472
    full_best = max(v for _, v in LEGACY_ARM_PROBE_SCORES)
    assert best == full_best == 0.940 and best_steps == 1_001_472
    assert stop / LEGACY_ARM_PROBE_SCORES[-1][0] == pytest.approx(0.5, abs=0.01)


def test_a_more_impatient_stop_trades_a_little_quality_for_more_time():
    """Patience 2 saves 62% and costs 0.002 in band — inside the ±0.013
    per-seed spread of the benchmark, so the knob is a real trade and the
    numbers say where it sits."""
    stop, (_, best) = _replay(LEGACY_ARM_PROBE_SCORES, patience=2)
    assert stop == 751_104
    full_best = max(v for _, v in LEGACY_ARM_PROBE_SCORES)
    assert full_best - best == pytest.approx(0.002, abs=1e-9)
    assert full_best - best < 0.013, "the cost must stay under the benchmark's seed spread"


def test_a_reward_curve_would_not_have_stopped_this_run():
    """The negative that makes the probe worth its 1% overhead: the same run's
    reward series rises throughout, so a reward-watching stop never fires."""
    from microduck_local.plateau import PlateauDetector
    # ep_rew sampled from the same run's progress.jsonl, one point per 250k.
    ep_rew = [(250_368, 159.1), (500_736, 165.5), (751_104, 168.6), (1_001_472, 170.4),
              (1_250_304, 172.0), (1_500_672, 174.2), (1_751_040, 172.4), (2_001_408, 176.6)]
    d = PlateauDetector(patience=3, min_steps=250_000, rel=0.01, window=1)
    assert not any(d.update(s, v) for s, v in ep_rew)
    assert not d.fired
