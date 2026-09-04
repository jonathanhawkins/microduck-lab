"""The brain trainer's hyperparameters, and the defects the fixes removed.

Every test here pins something that was measurably wrong or unguarded in
`train_brain.py` before, so a regression is loud rather than silent:

* the rollout buffer was truncated into uneven minibatches at the default
  env count (SB3 warns about it; nothing failed);
* the learning rate was constant, which `train_behavior.py` documents as the
  cause of every run peaking and then coming apart;
* nothing capped the action log_std across a warm-start chain, the trap that
  ships a saturated deterministic mean while the reward curve looks fine.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.ppo_hparams import linear_decay, ppo_batch_size
from microduck_local.train_brain import LOG_STD_MAX, LR_END, LR_START

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"
pytestmark = pytest.mark.skipif(
    not (POLICIES / "alpha_walking.onnx").exists(), reason="upstream checkouts not found")

N_STEPS = 128       # train_brain's --n-steps default


def test_default_env_count_no_longer_truncates_the_rollout_buffer():
    """The bug: 128 steps x 12 envs is a 1536-sample buffer, and the old
    `min(1024, n_steps * envs)` batch does not divide it — SB3 truncates
    every update into a 1024 and a 512 minibatch. Half the gradient steps
    ran at 2/3 the intended batch."""
    old = min(1024, N_STEPS * 12)
    assert (N_STEPS * 12) % old != 0, "the defect this test guards would not reproduce"
    new = ppo_batch_size(N_STEPS, 12)
    assert (N_STEPS * 12) % new == 0
    assert new == 768 and (N_STEPS * 12) // new == 2


@pytest.mark.parametrize("envs", [4, 8, 12, 16, 20, 24, 32])
def test_batch_divides_the_buffer_at_every_env_count(envs):
    """viz_server and helper ducks change the env count at runtime, so this
    has to hold for counts nobody picked by hand."""
    assert (N_STEPS * envs) % ppo_batch_size(N_STEPS, envs) == 0


def test_the_learning_rate_is_constant_by_default_and_decays_when_asked():
    """The decay is AVAILABLE, not a default. `train_behavior` decays because
    every trick run peaked and came apart; the brain runs measured here do
    not show that shape, and a seed-matched 2M A/B of the decay landed inside
    training-seed noise. A tuning change with no measured benefit does not
    get to be a default — a defect fix is a different thing (see the
    minibatch test above)."""
    assert LR_END == LR_START, "the shipped default must be the constant rate"
    flat = linear_decay(LR_START, LR_END)
    assert flat(1.0) == flat(0.5) == flat(0.0) == pytest.approx(LR_START)

    # Asked for, it decays monotonically. SB3 calls it with progress_remaining
    # falling 1 -> 0.
    f = linear_decay(3e-4, 3e-5)
    assert f(1.0) == pytest.approx(3e-4) and f(0.0) == pytest.approx(3e-5)
    ys = np.array([f(x) for x in np.linspace(1.0, 0.0, 25)])
    assert np.all(np.diff(ys) < 0)


def test_the_action_std_cap_does_not_bind_at_initialisation():
    """The brain's `log_std_init` is -0.5. A cap AT -0.5 would bind from step
    0 and forbid the entropy bonus from ever widening exploration. Measured
    over five 2M runs, every policy drove its own log_std down to between
    -0.55 and -1.22, so the cap's whole job is catching the warm-start
    ratchet (log_std 3.2, std 21-26, on a trick chain) — never a healthy
    run."""
    log_std_init = -0.5      # train_brain's PPO construction
    assert LOG_STD_MAX > log_std_init, (
        "a cap at or below log_std_init binds before training starts")
    assert LOG_STD_MAX < 3.2, "and it still has to catch the measured ratchet"


def _train(tmp_path: Path, name: str, extra: list[str], env: dict | None = None) -> Path:
    """A tiny real run, in an isolated brains dir."""
    e = dict(os.environ, MICRODUCK_BRAINS_DIR=str(tmp_path), **(env or {}))
    cmd = [sys.executable, "-m", "microduck_local.train_brain", "--run-name", name,
           "--envs", "2", "--steps", "800", "--n-steps", "32", "--checkpoint-every", "0", *extra]
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp_path / name


def _log_std(run_dir: Path) -> np.ndarray:
    from stable_baselines3 import PPO
    return PPO.load(str(run_dir / "model"), device="cpu").policy.log_std.detach().numpy()


def test_brain_json_records_the_hyperparameters_a_rerun_would_need(tmp_path):
    d = _train(tmp_path, "rec", [])
    meta = json.loads((d / "brain.json").read_text())
    assert meta["batch_size"] == ppo_batch_size(32, 2)
    assert meta["lr"] == LR_START and meta["lr_end"] == LR_END
    assert meta["log_std_max"] == LOG_STD_MAX and meta["legacy_hparams"] is False


def test_warm_start_caps_an_inherited_bang_bang_action_std(tmp_path):
    """The trap: every --init-from reloads the previous run's log_std and the
    entropy bonus ratchets it up each generation, until the CLIPPED sampling
    noise carries the behavior and the exported deterministic MEAN is garbage.
    Poison a checkpoint's log_std by hand and check the warm start pulls it
    back under the cap — and that the legacy arm, which is the A/B baseline,
    does not."""
    import torch
    from stable_baselines3 import PPO

    src = _train(tmp_path, "poison", [])
    model = PPO.load(str(src / "model"), device="cpu")
    with torch.no_grad():
        model.policy.log_std.data.fill_(3.2)      # the measured headstand-chain value
    model.save(str(src / "model"))
    assert _log_std(src).max() == pytest.approx(3.2)

    capped = _train(tmp_path, "capped", ["--init-from", "poison"])
    assert _log_std(capped).max() <= LOG_STD_MAX + 1e-6

    loose = _train(tmp_path, "loose", ["--init-from", "poison"],
                   env={"MICRODUCK_BRAIN_LEGACY": "1"})  # the A/B baseline arm
    assert _log_std(loose).max() > LOG_STD_MAX, (
        "the legacy arm must reproduce the OLD behavior or the A/B is not a baseline")


def test_legacy_env_var_reproduces_the_pre_fix_hyperparameters(tmp_path):
    d = _train(tmp_path, "legacy", [], env={"MICRODUCK_BRAIN_LEGACY": "1"})
    meta = json.loads((d / "brain.json").read_text())
    assert meta["legacy_hparams"] is True
    assert meta["batch_size"] == min(1024, 32 * 2)
    assert meta["log_std_max"] is None


def test_deterministic_probe_scores_the_export_and_can_stop_the_run(tmp_path):
    """The measured reason this exists: over a 2M run the benchmark's in-band
    score was flat from the 250k checkpoint (0.938) to the end (0.939) while
    `ep_rew` climbed 159 -> 177 the whole way. A stop watching REWARD never
    fires on that run. `--probe-every` scores the deterministic export — the
    artifact that ships — and that is the series a plateau stop can use."""
    d = _train(tmp_path, "probed", [
        "--steps", "20000", "--probe-every", "1000", "--probe-seeds", "100",
        "--probe-episodes", "1", "--plateau-patience", "2",
        "--plateau-min-steps", "1500", "--plateau-window", "1", "--plateau-rel", "0.02"])
    rows = [json.loads(ln) for ln in (d / "progress.jsonl").read_text().splitlines() if ln.strip()]
    probes = [r for r in rows if "probe" in r]
    assert probes, "no probe was recorded"
    assert all(0.0 <= r["probe"] <= 1.0 for r in probes), "in-band is a fraction"

    final = rows[-1]
    assert final.get("stopped") == "plateau", "the probe series should have plateaued"
    assert final["plateau"]["signal"] == "probe in_band", (
        "the stop must record WHICH series it watched — a reward-based stop and a "
        "benchmark-based stop are different claims")
    assert final["trained_steps"] < 20000, "an early stop that trains the full budget is not one"
    # The run still completed: the artifact is exported and the record says done.
    assert (d / "brain.onnx").exists() and final["done"] is True


def test_probe_is_off_by_default_and_the_stop_then_watches_reward(tmp_path):
    d = _train(tmp_path, "unprobed", [])
    rows = [json.loads(ln) for ln in (d / "progress.jsonl").read_text().splitlines() if ln.strip()]
    assert not any("probe" in r for r in rows)
    assert not any("stopped" in r for r in rows), "no plateau flags on a default run"


def test_decision_period_schedule_ends_at_the_deployment_period(tmp_path):
    """`brain.json` records ONE decision period and `LearnedBrain` runs at
    whatever it records, so a schedule that ended coarse would ship a brain
    that decides at half the rate it was measured at. The schedule must always
    land on 5."""
    d = _train(tmp_path, "sched", [
        "--steps", "4000", "--decide-every-start", "10", "--decide-every-switch", "0.5"])
    meta = json.loads((d / "brain.json").read_text())
    assert meta["decide_every"] == 5, "the SHIPPED period is the deployment one"
    assert meta["decide_every_start"] == 10, "and the record says where it started"


def test_a_coarser_period_halves_the_decisions_per_episode():
    """The mechanism, and the reason this is not a saving: at period 10 one
    PPO sample covers ten control steps instead of five, so a fixed DECISION
    budget buys twice the physics. What it trades for is a 100-decision
    horizon over the same 20 s episode instead of 200."""
    from microduck_local.brain.brain_env import BrainEnv, FollowTask

    env = BrainEnv(FollowTask(), seed=0, fixed_preset="datasheet")
    assert env.decide_every == 5 and env.max_decisions == 200
    assert env.set_decide_every(10) == 10
    assert env.max_decisions == 100, "the episode stays 20 s, so the count halves"
    env.set_decide_every(5)
    assert env.max_decisions == 200, "and it comes back"


def test_the_schedule_is_off_by_default(tmp_path):
    d = _train(tmp_path, "nosched", [])
    meta = json.loads((d / "brain.json").read_text())
    assert meta["decide_every_start"] is None and meta["decide_every"] == 5


def test_brain_net_arch_is_a_capacity_knob_not_a_throughput_one(tmp_path):
    """Recorded here because the reasoning is the point, not the flag.

    On the TRICK trainer the network is 54% of wall time (12% rollout forward
    + 42% update) and shrinking it is worth ~1.44x. On the BRAIN trainer the
    update is **5.3%** and collection is 94.7% — the brain env is physics, and
    a 128-128 MLP over 80 observations costs almost nothing. Measured on the
    A/B arms themselves: 128-128 and 256-256 ran 113,664 and 112,128 steps in
    the same 85 s. So this flag exists to ask whether the brain is
    CAPACITY-limited, and anyone reaching for it to go faster is at the wrong
    trainer."""
    small = _train(tmp_path, "n64", ["--net-arch", "64,64"])
    big = _train(tmp_path, "n256", ["--net-arch", "256,256"])
    from stable_baselines3 import PPO

    def params(d):
        return sum(p.numel() for p in PPO.load(str(d / "model"), device="cpu")
                   .policy.parameters())

    assert json.loads((small / "brain.json").read_text())["net_arch"] == "64,64"
    assert json.loads((big / "brain.json").read_text())["net_arch"] == "256,256"
    assert params(small) < params(big) / 4, "256-256 should be far larger than 64-64"


def test_brain_net_arch_and_epochs_default_unchanged(tmp_path):
    """Neither knob may move the shipped recipe until an A/B says so."""
    d = _train(tmp_path, "dflt", [])
    meta = json.loads((d / "brain.json").read_text())
    assert meta["net_arch"] == "128,128" and meta["n_epochs"] == 5
    from stable_baselines3 import PPO
    assert PPO.load(str(d / "model"), device="cpu").n_epochs == 5
