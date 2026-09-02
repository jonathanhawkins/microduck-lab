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
    BRAIN_OBS_VERSION,
    BrainEnv,
    FollowTask,
    ObsBuilder,
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


def test_obs_version_2_tracks_and_coasts_while_version_1_is_unchanged():
    """Version 2 fills the target slots from the TRACKER (bearing turning
    with the body through a miss, a coasting flag, confirmation) and the
    reserved slots with the yaw rate; version 1 — what brains before the
    version key were trained on — is bit-for-bit what it was."""
    assert BRAIN_OBS_VERSION == 2
    det = DetectionFrame(t=1.0, detections=[Detection("person", "p0", 0.2, -0.1, 0.3, 1.4, 0.9)])
    a = np.array([0.1, 0.0, -0.2], np.float32)
    v1 = ObsBuilder("person", version=1)
    v2 = ObsBuilder("person", version=2)
    s1 = Senses(t=1.05, det=det, det_age=0.05, speed=0.12, odom=(0.0, 0.0, 0.0))
    o1, o2 = v1(s1, a), v2(s1, a)
    ref, _ = senses_to_obs(s1, "person", a, None)
    np.testing.assert_array_equal(o1, ref)                          # version 1 == the bare function
    assert (o1[77:80] == 0).all()
    np.testing.assert_allclose(o2[65:71], [1, 0.2, -0.1, 0.3, 1.4, 0.9])
    assert o2[77] == 0.0 and o2[79] == 0.0 and o2[78] == 0.0           # hit, one sighting: unconfirmed; no turn yet
    # Same person again, a frame later: confirmed. Then the body turns +0.5
    # rad with no new frame: version 1 forgets it; version 2 coasts it at
    # the bearing it must now be at, flagged as coasting, with the yaw rate.
    det2 = DetectionFrame(t=1.1, detections=[Detection("person", "p0", 0.22, -0.1, 0.3, 1.42, 0.9)])
    o2b = v2(Senses(t=1.15, det=det2, det_age=0.05, speed=0.12, odom=(0.0, 0.0, 0.0)), a)
    assert o2b[79] == 1.0 and o2b[65] == 1.0
    turned = Senses(t=1.35, det=det2, det_age=0.25, speed=0.0, odom=(0.0, 0.0, 0.5))
    o1c, o2c = v1(turned, a), v2(turned, a)
    assert o1c[65] == 1.0 and o1c[66] == pytest.approx(0.22)              # v1: the stale frame, unturned
    assert o2c[65] == 0.0 and o2c[77] == 1.0                              # v2: coasting…
    assert abs(o2c[66] - (0.212 - 0.5)) < 0.05                            # …with the bearing turned with the body
    assert o2c[78] == pytest.approx(0.5 / 0.2, rel=1e-3)                  # yaw rate over the decision interval
    assert o2c[72] == pytest.approx(0.25)                                 # since the last HIT (the frame's time)
    # The env and the world's brain build the same version-2 observation.
    env = BrainEnv(FollowTask(episode_s=1.0), seed=0)
    assert env.obs_version == 2 and env._builder.tracker is not None
    obs, _ = env.reset(seed=1)
    assert obs.shape == (BRAIN_OBS_DIM,)


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
