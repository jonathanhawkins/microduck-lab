"""FastActorCriticPolicy must be a pure speedup, not a different policy.

PPO's importance ratio is exp(new_log_prob - rollout_log_prob): the rollout's
log-probs (fast forward) and the update's (stock evaluate_actions) must agree
to float tolerance or every ratio silently drifts from 1. These tests pin the
fast path to the stock machinery on the exact setup the trainers build
("MlpPolicy" shapes, diag Gaussian, shared extractor).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gymnasium import spaces  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402

from microduck_local import contract as C  # noqa: E402
from microduck_local.symmetry import FastActorCriticPolicy  # noqa: E402

OBS_SPACE = spaces.Box(-np.inf, np.inf, (C.OBS_DIM,), np.float32)
ACT_SPACE = spaces.Box(-1.0, 1.0, (C.NUM_JOINTS,), np.float32)
ARCH = dict(net_arch=dict(pi=[64, 32], vf=[64, 32]),
            activation_fn=torch.nn.ELU, log_std_init=-0.3)


def _policy(cls):
    torch.manual_seed(7)
    return cls(OBS_SPACE, ACT_SPACE, lambda _: 3e-4, **ARCH)


def _obs(n=32):
    g = torch.Generator().manual_seed(11)
    return torch.randn(n, C.OBS_DIM, generator=g)


def test_stochastic_forward_matches_evaluate_actions():
    pol = _policy(FastActorCriticPolicy)
    obs = _obs()
    with torch.no_grad():
        actions, values, log_prob = pol(obs)
        ev_values, ev_log_prob, _ = pol.evaluate_actions(obs, actions)
    assert torch.allclose(log_prob, ev_log_prob, atol=1e-5)
    assert torch.allclose(values, ev_values)


def test_deterministic_forward_is_the_stock_forward():
    fast, stock = _policy(FastActorCriticPolicy), _policy(ActorCriticPolicy)
    stock.load_state_dict(fast.state_dict())
    obs = _obs()
    with torch.no_grad():
        fa, fv, flp = fast(obs, deterministic=True)
        sa, sv, slp = stock(obs, deterministic=True)
    assert torch.allclose(fa, sa)
    assert torch.allclose(fv, sv)
    assert torch.allclose(flp, slp, atol=1e-5)


def test_samples_come_from_the_policy_distribution():
    """Mean/std of many fast-path samples match the policy's own Gaussian."""
    pol = _policy(FastActorCriticPolicy)
    obs = _obs(1).repeat(20_000, 1)
    torch.manual_seed(3)
    with torch.no_grad():
        actions, _, _ = pol(obs)
        dist = pol.get_distribution(obs[:1])
    mean = dist.distribution.mean[0]
    std = dist.distribution.stddev[0]
    assert torch.allclose(actions.mean(0), mean, atol=4 * std.max() / np.sqrt(20_000) + 1e-3)
    assert torch.allclose(actions.std(0), std, rtol=0.05)


def test_trainers_build_the_fast_policy():
    """The lab path must actually get the fast class (fresh AND warm start)."""
    import inspect

    from microduck_local import train_behavior
    src = inspect.getsource(train_behavior.main)
    assert "FastActorCriticPolicy" in src
