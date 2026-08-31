"""End-to-end: train a few PPO steps, export ONNX, verify the baked normalizer
and that the artifact matches SB3's deterministic prediction bit-for-bit-ish."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from microduck_local import contract as C
from microduck_local.export_onnx import OnnxWalkPolicy
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    venv = DummyVecEnv([lambda: MicroduckWalkEnv(seed=0)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=100.0)
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1, device="cpu")
    model.learn(total_timesteps=64)
    model.save(str(out / "model"))
    venv.save(str(out / "vecnormalize.pkl"))
    return out


def test_export_matches_sb3_prediction(tiny_run):
    from microduck_local.export_onnx import export
    onnx_path = export(tiny_run, tiny_run / "policy.onnx")  # runs its own allclose check

    sess = ort.InferenceSession(str(onnx_path))
    i, o = sess.get_inputs()[0], sess.get_outputs()[0]
    assert i.name == "obs" and list(i.shape) == [1, C.OBS_DIM]
    assert o.name == "actions" and list(o.shape) == [1, C.NUM_JOINTS]

    # Against model.predict(deterministic=True) with external normalization —
    # the exported graph must reproduce train-time behavior end to end.
    model = PPO.load(str(tiny_run / "model"), device="cpu")
    import pickle
    with open(tiny_run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    rng = np.random.default_rng(7)
    raw = rng.normal(0, 0.5, (1, C.OBS_DIM)).astype(np.float32)
    norm = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
                   -vn.clip_obs, vn.clip_obs).astype(np.float32)
    want, _ = model.predict(norm, deterministic=True)
    got = sess.run(["actions"], {"obs": raw})[0]
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_unnormalized_would_differ(tiny_run):
    """The reason baking matters: raw-vs-normalized obs give different actions."""
    model = PPO.load(str(tiny_run / "model"), device="cpu")
    import pickle
    with open(tiny_run / "vecnormalize.pkl", "rb") as f:
        vn = pickle.load(f)
    wrapper = OnnxWalkPolicy(model.policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs).eval()
    ident = OnnxWalkPolicy(model.policy, np.zeros(C.OBS_DIM), np.ones(C.OBS_DIM), 100.0).eval()
    obs = torch.tensor(np.random.default_rng(3).normal(0, 0.5, (1, C.OBS_DIM)), dtype=torch.float32)
    with torch.no_grad():
        assert not torch.allclose(wrapper(obs), ident(obs))
