"""Mechanics of SymmetryPPO's overlapped update (overlap_update=True).

Training QUALITY under the one-update data lag is an empirical question the
A/B run answers (see README); what belongs in unit tests is the machinery:
every collected rollout is trained on exactly once, the step budget is
honored, callbacks fire with settled weights, and a crash in the update
thread surfaces instead of being swallowed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gymnasium as gym  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from microduck_local import contract as C  # noqa: E402
from microduck_local import symmetry as S  # noqa: E402

N_STEPS = 64
N_ENVS = 2


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


def _model(**kwargs):
    return S.SymmetryPPO(
        "MlpPolicy", DummyVecEnv([_Contract] * N_ENVS),
        policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])),
        n_steps=N_STEPS, batch_size=32, n_epochs=1, device="cpu", seed=0,
        symmetry_coef=0.0, verbose=0, **kwargs)


def test_every_rollout_is_trained_on_once_and_budget_is_met():
    model = _model(overlap_update=True)
    trains = []
    orig_train = model.train

    def counting_train():
        trains.append(int(model.num_timesteps))
        orig_train()

    model.train = counting_train
    total = 4 * N_STEPS * N_ENVS
    model.learn(total_timesteps=total, progress_bar=False)
    assert model.num_timesteps == total
    assert len(trains) == 4  # one update per collected rollout, none dropped


def test_callbacks_see_settled_weights_at_rollout_end():
    """The join happens before on_rollout_end: by the time a callback runs
    (train_behavior snapshots ONNX there), no update thread is alive."""
    import threading

    from stable_baselines3.common.callbacks import BaseCallback

    seen = []

    class Probe(BaseCallback):
        def _on_step(self):
            return True

        def _on_rollout_end(self):
            seen.append(any(t.name == "ppo-update" and t.is_alive()
                            for t in threading.enumerate()))

    model = _model(overlap_update=True)
    model.learn(total_timesteps=3 * N_STEPS * N_ENVS, callback=Probe(),
                progress_bar=False)
    assert seen and not any(seen)


def test_update_thread_errors_propagate():
    model = _model(overlap_update=True)

    def boom():
        raise RuntimeError("update exploded")

    model.train = boom
    with pytest.raises(RuntimeError, match="update exploded"):
        model.learn(total_timesteps=3 * N_STEPS * N_ENVS, progress_bar=False)


def test_overlap_off_uses_the_stock_loop():
    """Default stays bit-identical stock SB3: the overlap learn() must defer
    to super().learn(), not vendored code (the zero-coef parity test in
    test_symmetry.py then covers the rest)."""
    model = _model()  # overlap_update defaults to False
    called = []
    orig = S.PPO.learn

    def spy(self, *a, **k):
        called.append(True)
        return orig(self, *a, **k)

    S.PPO.learn = spy
    try:
        model.learn(total_timesteps=N_STEPS * N_ENVS, progress_bar=False)
    finally:
        S.PPO.learn = orig
    assert called


def test_collect_matches_stock_sb3_bitwise():
    """The deferred-add vendored collect must fill the rollout buffer with
    exactly what stock SB3 collection fills it with — same values, same
    slots — since PPO trains on nothing else."""
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    a, b = _model(), _model()  # identical seeds -> identical weights/RNG
    a._setup_learn(N_STEPS * N_ENVS, None)
    b._setup_learn(N_STEPS * N_ENVS, None)
    cb_a = a._init_callback(None)
    cb_b = b._init_callback(None)
    torch.manual_seed(123)
    OnPolicyAlgorithm.collect_rollouts(a, a.env, cb_a, a.rollout_buffer, N_STEPS)
    torch.manual_seed(123)
    b.collect_rollouts(b.env, cb_b, b.rollout_buffer, N_STEPS)
    for field in ("observations", "actions", "rewards", "episode_starts",
                  "values", "log_probs", "advantages", "returns"):
        np.testing.assert_array_equal(
            getattr(a.rollout_buffer, field), getattr(b.rollout_buffer, field),
            err_msg=field)
