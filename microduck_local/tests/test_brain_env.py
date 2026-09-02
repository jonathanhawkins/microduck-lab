"""BrainEnv, the learned-brain export and the registry (roadmap 3.1/3.2):
the observation contract, gym semantics, the scripted baseline scoring the
same episodes, and a tiny PPO round-trip through ONNX into the world."""

import json

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import REGISTRY, Senses
from microduck_local.brain.brain_env import (
    ACT_HIGH,
    ACT_LOW,
    BRAIN_OBS_DIM,
    BrainEnv,
    FollowTask,
    senses_to_obs,
)
from microduck_local.sensors.detector import Detection, DetectionFrame

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"
pytestmark = pytest.mark.skipif(
    not (POLICIES / "alpha_walking.onnx").exists(), reason="upstream checkouts not found")


def test_senses_to_obs_layout():
    det = DetectionFrame(t=1.0, detections=[
        Detection("person", "p0", 0.2, -0.1, 0.3, 1.4, 0.9),
        Detection("person", "p1", -0.5, 0.0, 0.1, 3.0, 0.5),
        Detection("duck", "d1", 0.0, 0.0, 0.2, 0.8, 1.0)])
    s = Senses(t=1.05, tof=None, tof_age=None, det=det, det_age=0.05, speed=0.12)
    o, seen = senses_to_obs(s, "person", np.array([0.1, 0.0, -0.2], np.float32), None)
    assert o.shape == (BRAIN_OBS_DIM,) and o.dtype == np.float32
    assert o[64] == 1.0                              # no ToF: age saturates
    np.testing.assert_allclose(o[65:71], [1, 0.2, -0.1, 0.3, 1.4, 0.9])   # nearest person, not the duck
    assert o[71] == pytest.approx(0.05) and o[72] == 0.0 and seen == 1.05
    np.testing.assert_allclose(o[73:76], [0.1, 0.0, -0.2]) and o[76] == pytest.approx(0.12)
    o2, seen2 = senses_to_obs(Senses(t=3.0, det=None), "person", np.zeros(3, np.float32), seen)
    assert o2[65] == 0.0 and o2[72] == pytest.approx(1.95) and seen2 == 1.05


def test_env_reset_step_and_bounds():
    env = BrainEnv(FollowTask(episode_s=2.0), seed=0)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (BRAIN_OBS_DIM,)
    assert env.action_space.shape == (3,)
    np.testing.assert_allclose(env.action_space.low, ACT_LOW)
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(np.array([5.0, 0.0, 0.0], np.float32))   # clipped to 0.6
        n += 1
        assert np.isfinite(r) and {"dist", "bearing", "seen", "bumped"} <= info.keys()
        if term or trunc:
            break
    assert trunc and n == env.max_decisions == 20
    assert env.world.ducks["d0"].twist_cmd[0] == pytest.approx(ACT_HIGH[0])


def test_scripted_follow_scores_the_same_episodes_deterministically():
    from microduck_local.eval_brain import run
    a = run("follow", "ideal", episodes=1, seed=7, task=FollowTask(episode_s=3.0))
    b = run("follow", "ideal", episodes=1, seed=7, task=FollowTask(episode_s=3.0))
    assert a["rows"][0]["decisions"] == 30 and a["return"] == b["return"]
    assert 0.0 <= a["in_band"] <= 1.0 and a["falls"] == 0


def test_tiny_ppo_round_trips_through_onnx_into_the_world(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from microduck_local.train_brain import export_brain
    monkeypatch.setenv("MICRODUCK_BRAINS_DIR", str(tmp_path))
    venv = VecNormalize(DummyVecEnv([lambda: BrainEnv(FollowTask(episode_s=1.0), seed=1)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=16, n_epochs=1, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[16], vf=[16])))
    model.learn(32)
    d = tmp_path / "tiny"
    d.mkdir()
    model.save(str(d / "model"))
    venv.save(str(d / "vecnormalize.pkl"))
    (d / "brain.json").write_text(json.dumps({"target_cls": "person", "decide_every": 5}))
    out = export_brain(d)
    assert out.exists()
    assert "learned:tiny" in REGISTRY.available()
    brain = REGISTRY.make("learned:tiny")
    assert brain.kind == "learned:tiny"
    intent = brain.step(Senses(t=0.0))
    tw = np.array(intent.twist)
    assert tw.shape == (3,) and (tw >= ACT_LOW - 1e-6).all() and (tw <= ACT_HIGH + 1e-6).all()
    # Holds the decision between ticks, re-decides every 5th.
    assert brain.step(Senses(t=0.02)).twist == intent.twist
    with pytest.raises(ValueError):
        REGISTRY.make("learned:nope")
