"""The KL-targeted learning rate (rsl_rl's `schedule="adaptive"`).

Every microduck_rl task cfg pairs `learning_rate=1.0e-3` with
`schedule="adaptive", desired_kl=0.01`, so that 1e-3 is a STARTING value under
closed-loop control. We had copied the number and left SB3's constant rate in
place; measured approx_kl on a 4M-step `run` was 0.19-0.60 (20-60x target) and
the policy came apart after ~1.2M steps. These lock the controller.
"""

import numpy as np
import pytest
import torch

from microduck_local import contract as C
from microduck_local import symmetry as S

INIT_LR = 1e-3


def _model(desired_kl, seed: int = 0):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _Contract(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (C.OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-4.0, 4.0, (C.NUM_JOINTS,), np.float32)

        def __init__(self):
            self._rng = np.random.default_rng(0)

        def reset(self, *, seed=None, options=None):
            self._t = 0
            return self._rng.normal(size=C.OBS_DIM).astype(np.float32), {}

        def step(self, action):
            self._t += 1
            obs = self._rng.normal(size=C.OBS_DIM).astype(np.float32)
            return obs, float(-np.square(action).sum()), False, self._t >= 20, {}

    return S.SymmetryPPO(
        "MlpPolicy", DummyVecEnv([_Contract]),
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        n_steps=64, batch_size=32, n_epochs=1, device="cpu", seed=seed,
        learning_rate=INIT_LR, symmetry_coef=0.0, desired_kl=desired_kl,
    )


def _lr(model) -> float:
    return float(model.policy.optimizer.param_groups[0]["lr"])


def test_the_controller_is_off_by_default():
    """Upstream's target is recorded, but NOT adopted: at our batch size it
    pins the rate to the 1e-5 floor and cost 13x the episode length in a
    three-arm A/B (see symmetry.py). Off unless asked for."""
    assert S.UPSTREAM_DESIRED_KL == 0.01
    assert S.DEFAULT_DESIRED_KL is None
    assert _model(desired_kl=S.DEFAULT_DESIRED_KL).desired_kl is None


def test_the_rule_is_rsl_rls_rule():
    """Both branches and the deadband, driven directly at chosen KLs."""
    m = _model(desired_kl=0.01)
    m._adaptive_lr = INIT_LR
    m._adapt_learning_rate(0.05)                 # 5x target -> back off
    assert m._adaptive_lr == pytest.approx(INIT_LR / 1.5)
    m._adaptive_lr = INIT_LR
    m._adapt_learning_rate(0.001)                # a tenth of target -> push
    assert m._adaptive_lr == pytest.approx(INIT_LR * 1.5)
    for inside in (0.006, 0.01, 0.019):          # half..twice = leave alone
        m._adaptive_lr = INIT_LR
        m._adapt_learning_rate(inside)
        assert m._adaptive_lr == pytest.approx(INIT_LR)
    m._adaptive_lr = INIT_LR                     # kl == 0 must not push either
    m._adapt_learning_rate(0.0)
    assert m._adaptive_lr == pytest.approx(INIT_LR)


def test_the_rule_clamps_both_ends():
    m = _model(desired_kl=0.01)
    m._adaptive_lr = 1.2e-5
    for _ in range(10):
        m._adapt_learning_rate(1.0)              # always far too big
    assert m._adaptive_lr == pytest.approx(1e-5)
    assert _lr(m) == pytest.approx(1e-5)         # and it reached the optimizer
    m._adaptive_lr = 9e-3
    for _ in range(10):
        m._adapt_learning_rate(1e-9)             # always far too small
    assert m._adaptive_lr == pytest.approx(1e-2)


def test_kl_above_target_lowers_the_rate_in_a_real_run():
    """End to end: an unreachable target means every minibatch overshoots, so
    the rate must come down from its starting value."""
    m = _model(desired_kl=1e-9)
    m.learn(total_timesteps=256)
    assert _lr(m) < INIT_LR / 5, f"rate barely moved: {_lr(m):.2e}"


def test_disabled_controller_keeps_a_constant_rate():
    """`--desired-kl 0` must restore exactly the old behavior."""
    m = _model(desired_kl=None)
    m.learn(total_timesteps=256)
    assert _lr(m) == pytest.approx(INIT_LR)
    assert m._adaptive_lr is None


def test_non_positive_target_disables_it():
    assert _model(desired_kl=0).desired_kl is None
    assert _model(desired_kl=-1).desired_kl is None


def test_the_annealed_rate_survives_a_warm_restart(tmp_path):
    """A resumed run must not jump back to the hot starting rate — that would
    re-break the policy every time viz_server rescales the env count."""
    m = _model(desired_kl=1e-9)
    m.learn(total_timesteps=256)
    annealed = _lr(m)
    assert annealed < INIT_LR
    m.save(str(tmp_path / "model"))

    back = S.SymmetryPPO.load(str(tmp_path / "model"), env=m.get_env(), device="cpu")
    assert back._adaptive_lr == pytest.approx(annealed)
    back.learn(total_timesteps=64)
    assert _lr(back) <= annealed, "warm restart reheated the learning rate"


def test_controller_changes_the_resulting_policy():
    """Sanity: this is a real deviation from stock PPO, not a no-op."""
    off = _model(desired_kl=None, seed=5)
    off.learn(total_timesteps=256)
    on = _model(desired_kl=1e-9, seed=5)
    on.learn(total_timesteps=256)
    diffs = [not torch.equal(a, b) for (_, a), (_, b)
             in zip(off.policy.state_dict().items(), on.policy.state_dict().items())]
    assert any(diffs)


def test_linear_decay_schedule():
    """Plain linear decay, no batch-size calibration needed — unlike the KL
    controller, which is off by default because upstream's 0.01 target is
    unreachable at our batch size. Every run so far peaked and then came
    apart; a rate sized for early exploration overshoots once the policy is
    near a good solution."""
    from microduck_local.train_behavior import LR_END, LR_START, linear_decay

    f = linear_decay(LR_START, LR_END)
    assert f(1.0) == pytest.approx(LR_START)   # start of training
    assert f(0.0) == pytest.approx(LR_END)     # end
    assert f(0.5) == pytest.approx((LR_START + LR_END) / 2)
    assert LR_END < LR_START, "decay must go down"
    # Monotone, so the rate never jumps back up mid-run.
    xs = [f(p / 20) for p in range(21)]
    assert xs == sorted(xs)


def test_entropy_anneals_to_zero_only_when_asked():
    """The exported ONNX is the MEAN policy, and a constant entropy bonus
    keeps the std high forever, so the mean is never forced to work without
    its noise — three policies in one day scored well stochastically and fell
    deterministically in under a second (the best archived 'peak': 0.26 s)."""
    m = _model(desired_kl=None)
    m.ent_coef = 0.01                     # _model uses SB3's default of 0.0
    m.ent_anneal = True
    m.learn(total_timesteps=128)          # fills the buffer, runs train()
    assert m._ent0 == pytest.approx(0.01)
    assert m.ent_coef < 0.01, "ent_coef never decayed"
    m._current_progress_remaining = 0.0
    m.learn(total_timesteps=64, reset_num_timesteps=False)
    assert m.ent_coef <= m._ent0 * 0.5    # well below start by now

    stock = _model(desired_kl=None)
    stock.ent_coef = 0.01
    stock.learn(total_timesteps=128)
    assert stock.ent_coef == pytest.approx(0.01), "default must stay constant"
    assert not stock.ent_anneal
