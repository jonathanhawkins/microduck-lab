"""Network size, the shared trunk, and the epoch count.

Why these three are one file: they are the three knobs on the 54% of wall
time that is the network (12% rollout forward + 42% update, measured at 32
envs). The default 512-256-128 was inherited from the GPU stack, where the
network is free because it runs ~250x more samples; here it is the largest
single cost and had never been swept. Measured on this machine:

    net_arch          fwd (b=32)   update (b=1024)   combined
    512-256-128         178 us         4557 us         1.00x
    256-128              79 us         1988 us         2.29x
    128-128              60 us         1333 us         3.42x
"""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
import torch

from microduck_local import contract as C

pytestmark = pytest.mark.skipif(
    not C.MICRODUCK_RL_DIR.exists(), reason="upstream microduck_rl checkout not found")


def _train(tmp_path, name, extra):
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path), MICRODUCK_VEC_ENV="fork")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", name, "--envs", "4", "--steps", "6000",
         "--snap-steps", "3000", *extra],
        env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    return tmp_path / name, r.stdout


def test_defaults_are_unchanged_and_announce_nothing(tmp_path):
    """The three knobs must be inert until asked for: this repo's rule is that
    a training change earns its default with a paired A/B, and none of these
    has one yet."""
    d, out = _train(tmp_path, "dflt", [])
    meta = json.loads((d / "behavior.json").read_text())
    assert meta["net_arch"] == "512,256,128"
    assert meta["shared_trunk"] is False
    assert meta["n_epochs"] == 5
    assert "net:" not in out, "the default path must not print a net line"


@pytest.mark.parametrize("arch", ["256,128", "128,128"])
def test_a_smaller_net_trains_and_is_recorded(tmp_path, arch):
    d, out = _train(tmp_path, f"a{arch.replace(',', '')}", ["--net-arch", arch])
    meta = json.loads((d / "behavior.json").read_text())
    assert meta["net_arch"] == arch
    assert f"net: {[int(x) for x in arch.split(',')]}" in out
    # The parameter count really did fall.
    from stable_baselines3 import PPO
    model = PPO.load(str(d / "model"), device="cpu")
    n = sum(p.numel() for p in model.policy.parameters())
    assert n < 400_000, f"{arch} should be far smaller than the default, got {n}"


def test_the_shared_trunk_shares_and_still_exports_the_61_to_14_contract(tmp_path):
    """The invariant that matters: policies are hot-swapped on the robot
    behind one 61-obs / 14-action ONNX interface. A topology change that broke
    that would be unusable however fast it trained."""
    import onnxruntime as ort

    from microduck_local.export_onnx import OnnxWalkPolicy
    from microduck_local.symmetry import SharedTrunk

    d, out = _train(tmp_path, "shared", ["--net-arch", "512,256,128", "--shared-trunk"])
    assert "net: shared" in out
    assert json.loads((d / "behavior.json").read_text())["shared_trunk"] is True

    from stable_baselines3 import PPO
    model = PPO.load(str(d / "model"), device="cpu")
    assert isinstance(model.policy.features_extractor, SharedTrunk)
    assert model.policy.features_extractor.features_dim == 256, "all but the last layer"

    sess = ort.InferenceSession(str(d / "policy.onnx"))
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    assert list(i.shape) == [1, C.OBS_DIM] and list(o.shape) == [1, C.NUM_JOINTS]

    # And the exported ONNX is still the torch policy's deterministic mean.
    with open(d / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    wrapper = OnnxWalkPolicy(model.policy, vn.obs_rms.mean, vn.obs_rms.var,
                             vn.clip_obs).eval()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.normal(0, 1, (1, C.OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            want = wrapper(torch.tensor(x)).numpy()
        np.testing.assert_allclose(sess.run(None, {i.name: x})[0], want,
                                   rtol=1e-4, atol=1e-5)


def test_the_shared_trunk_actually_computes_the_early_layers_once():
    """The whole point. If both heads re-ran the trunk this would be a
    rename, so count the forward calls through it."""
    import gymnasium as gym

    from microduck_local.symmetry import SharedTrunk

    space = gym.spaces.Box(-1, 1, (C.OBS_DIM,), np.float32)
    trunk = SharedTrunk(space, arch=(512, 256))
    calls = []
    real = trunk.net.forward
    trunk.net.forward = lambda x: (calls.append(1), real(x))[1]
    out = trunk(torch.zeros(8, C.OBS_DIM))
    assert out.shape == (8, 256) and len(calls) == 1


def test_shared_trunk_needs_two_layers_to_have_something_to_share(tmp_path):
    env = dict(os.environ, MICRODUCK_RUNS_DIR=str(tmp_path), MICRODUCK_VEC_ENV="fork")
    r = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_behavior", "run",
         "--run-name", "bad", "--envs", "4", "--steps", "2000",
         "--net-arch", "128", "--shared-trunk"],
        env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "at least two hidden sizes" in (r.stdout + r.stderr)


def test_fewer_epochs_reaches_the_optimizer(tmp_path):
    d, out = _train(tmp_path, "ep3", ["--n-epochs", "3"])
    assert json.loads((d / "behavior.json").read_text())["n_epochs"] == 3
    from stable_baselines3 import PPO
    assert PPO.load(str(d / "model"), device="cpu").n_epochs == 3
