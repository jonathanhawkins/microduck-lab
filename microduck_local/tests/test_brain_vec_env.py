"""The brain trainer on the shared-model fork backend.

`ForkVecEnv` used to size its shared-memory buffers from the WALK contract's
constants (61 observations, 14 actions), which silently made the backend
walk-only: `BrainEnv` is 80 and 3, so the brain trainer was stuck on
`SubprocVecEnv` — a private MJCF compile in every worker and a pickled pipe
round-trip per worker per step. The buffers are now sized from the env's own
spaces.

The bar for a throughput change in this repo is that it must not change the
numbers (AGENTS.md, "Performance work"), so the test that matters is the
step-for-step equality below, not the speed.
"""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import BRAIN_ACT_DIM, BRAIN_OBS_DIM
from microduck_local.train_brain import make_env_fn
from microduck_local.vec_env import ForkVecEnv, make_vec_env

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"
pytestmark = pytest.mark.skipif(
    not (POLICIES / "alpha_walking.onnx").exists(), reason="upstream checkouts not found")

N = 3
STEPS = 40


def _rollout(backend: str):
    """A fixed pseudo-random action sequence through N brain envs."""
    fns = [make_env_fn(i, "datasheet", False) for i in range(N)]
    venv = make_vec_env(fns, backend=backend)
    try:
        obs = [venv.reset().copy()]
        rews, dones = [], []
        rng = np.random.default_rng(0)
        for _ in range(STEPS):
            a = rng.uniform(-0.2, 0.5, (N, BRAIN_ACT_DIM)).astype(np.float32)
            o, r, d, _ = venv.step(a)
            obs.append(o.copy())
            rews.append(r.copy())
            dones.append(d.copy())
    finally:
        venv.close()
    return np.array(obs), np.array(rews), np.array(dones)


def test_fork_backend_carries_the_brain_observation_and_action_widths():
    """The regression: buffers sized 61x14 truncate an 80-float observation."""
    fns = [make_env_fn(i, "datasheet", False) for i in range(N)]
    venv = make_vec_env(fns, backend="fork")
    try:
        assert isinstance(venv, ForkVecEnv)
        assert venv._obs_dim == BRAIN_OBS_DIM and venv._act_dim == BRAIN_ACT_DIM
        assert venv._obs_dim != C.OBS_DIM and venv._act_dim != C.NUM_JOINTS, (
            "this test is only meaningful while the brain and walk contracts differ")
        obs = venv.reset()
        assert obs.shape == (N, BRAIN_OBS_DIM)
        o, r, d, _ = venv.step(np.zeros((N, BRAIN_ACT_DIM), np.float32))
        assert o.shape == (N, BRAIN_OBS_DIM) and r.shape == (N,) and d.shape == (N,)
        assert np.isfinite(o).all()
    finally:
        venv.close()


def test_fork_and_subproc_rollouts_are_step_for_step_identical():
    """A throughput change that alters the numbers is not a throughput change.
    Exact equality, not a tolerance: both backends run the same env code on
    the same seeds, so any difference at all is a bug, not float noise."""
    f_obs, f_rew, f_done = _rollout("fork")
    s_obs, s_rew, s_done = _rollout("subproc")
    np.testing.assert_array_equal(f_obs, s_obs)
    np.testing.assert_array_equal(f_rew, s_rew)
    np.testing.assert_array_equal(f_done, s_done)


def test_fork_backend_refuses_a_non_float32_space():
    """The buffers really are float32. An env that declares something else
    must be told, not silently reinterpreted."""
    import gymnasium as gym

    class Float64Env(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (4,), np.float64)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(4), {}

        def step(self, a):
            return np.zeros(4), 0.0, False, False, {}

        def close(self):
            pass

    with pytest.raises(ValueError, match="float32"):
        ForkVecEnv([Float64Env])
