"""Contract tests: the local env must produce exactly the obs/action semantics
that infer_policy.py (and the runtime) consume. These lock the invariants from
microduck_rl/AGENTS.md that apply here."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(scope="module")
def env():
    # Noise/DR off: contract tests need determinism.
    return MicroduckWalkEnv(obs_noise=False, domain_rand=False, action_delay=False, seed=0)


def test_obs_is_61d_in_contract_order(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (61,)
    assert obs.dtype == np.float32
    # Command block is the tail 13 dims: [twist(3), head(4), body(6)].
    np.testing.assert_array_equal(obs[48:51], env.twist_cmd)
    np.testing.assert_array_equal(obs[51:55], env.head_cmd)
    np.testing.assert_array_equal(obs[55:61], env.body_cmd)
    # last_action starts zero and occupies dims 34:48.
    np.testing.assert_array_equal(obs[34:48], np.zeros(14, np.float32))


def test_joint_slices_match_model_state(env):
    obs, _ = env.reset(seed=0)
    qpos_rel = env.data.qpos[env.joint_qpos_adr] - C.DEFAULT_POSE
    np.testing.assert_allclose(obs[6:20], qpos_rel, atol=1e-6)


def test_action_is_offset_from_default_pose(env):
    env.reset(seed=0)
    action = np.full(14, 0.1, np.float32)
    env.step(action)
    np.testing.assert_allclose(env.data.ctrl, C.DEFAULT_POSE + action, atol=1e-6)


def test_last_action_lands_in_obs(env):
    env.reset(seed=0)
    action = np.linspace(-0.2, 0.2, 14).astype(np.float32)
    obs, *_ = env.step(action)
    np.testing.assert_array_equal(obs[34:48], action)


def test_gravity_upright_is_minus_z(env):
    env.reset(seed=0)
    g = env._projected_gravity()
    assert g[2] < -0.95  # standing → gravity ≈ (0, 0, -1) in body frame


def test_shipped_policy_survives_in_this_env():
    """The strongest contract check available: the alpha_walking.onnx shipped in
    the microduck repo (trained on the official BAM/mjlab stack) must keep the
    robot upright here. Any regression in obs order, frame conventions, action
    application, or timing makes it fall over instantly.

    (A passive-stability settle test is deliberately NOT used: at this env's
    actuator fidelity — XML kp=0.55, kv=0, same as infer_policy.py — the stand
    pose is not a passive equilibrium; policies balance actively.)
    """
    onnx_path = C.MICRODUCK_RL_DIR.parent / "microduck/policies/alpha_walking.onnx"
    if not onnx_path.exists():
        pytest.skip("microduck repo (shipped policies) not checked out next door")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    env = MicroduckWalkEnv(seed=42)  # noise + DR on: the policy must survive them
    for ep in range(3):
        obs, _ = env.reset(seed=42 + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = sess.run(None, {"obs": obs[None]})[0][0].astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
        assert truncated and not terminated, (
            f"shipped policy fell at step {env.step_count} — env contract regression"
        )


def test_penalty_terms_are_never_positive(env):
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(
            rng.uniform(-0.3, 0.3, 14).astype(np.float32)
        )
        if terminated or truncated:
            env.reset(seed=1)
    for name, total in env.reward_sums.items():
        if name.endswith("_penalty"):
            assert total <= 1e-9, f"{name} accumulated {total:+.4f} (> 0): sign bug"


def test_rollout_is_nan_free_and_terminates_on_fall(env):
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    saw_termination = False
    for _ in range(600):
        obs, reward, terminated, truncated, _ = env.step(
            rng.uniform(-2.0, 2.0, 14).astype(np.float32)  # violent → should fall
        )
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated:
            saw_termination = True
            env.reset(seed=3)
    assert saw_termination, "violent random actions never triggered fall termination"


def test_body_lin_vel_is_body_frame_not_inertial():
    """Tracking rewards must use body-x/y, the same frame as the twist command.

    mj_objectVelocity(flg_local=1) returns the COM-inertial frame on this
    trunk (~90° off). A world +x shove at yaw=0 must read as +fwd, not as
    ~0 / sideways — that was the train-walk shuffle bug.
    """
    import mujoco

    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    env.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = 0.5
    mujoco.mj_forward(env.model, env.data)
    body = env.body_lin_vel()
    assert body[0] == pytest.approx(0.5, abs=0.02)
    assert abs(body[1]) < 0.05

    env.data.qvel[0] = 0.0
    env.data.qvel[1] = 0.5
    mujoco.mj_forward(env.model, env.data)
    body = env.body_lin_vel()
    assert abs(body[0]) < 0.05
    assert body[1] == pytest.approx(0.5, abs=0.02)

    # The trap this replaces: inertial-frame x is NOT body-x.
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                             env.trunk_body_id, vel6, 1)
    assert abs(vel6[3] - body[0]) > 0.2


def test_heading_lin_vel_follows_yaw():
    """A 90° yaw makes world +y the heading-forward axis."""
    import mujoco

    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    yaw = np.pi / 2
    env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.data.qvel[:] = 0.0
    env.data.qvel[1] = 0.5  # world +y
    mujoco.mj_forward(env.model, env.data)
    fwd, lat, vz = env.heading_lin_vel()
    assert fwd == pytest.approx(0.5, abs=0.02)
    assert abs(lat) < 0.02
    assert abs(vz) < 1e-9


def test_domain_rand_does_not_accumulate():
    """AGENTS.md: DR must restore-then-apply — 100 resets can't drift the model."""
    env = MicroduckWalkEnv(domain_rand=True, seed=0)
    nominal = env._default_body_mass[env.trunk_body_id]
    for i in range(100):
        env.reset(seed=i)
        assert 0.89 * nominal <= env.model.body_mass[env.trunk_body_id] <= 1.11 * nominal


def test_friction_randomization_actually_varies_grip():
    """The friction DR knob must change the CONTACT, not just an array.

    MuJoCo mixes a contact pair's friction by element-wise max unless one geom
    sets geom_priority. Randomizing the FLOOR while the feet sat at 1.0 with
    equal priority meant draws of 0.3/0.5/0.8/1.0 all produced an effective mu
    of 1.0 — the surface could only ever get grippier than nominal, never
    slipperier, so the knob was ~71% inert and silently disabled the DR it
    looked like it implemented.
    """
    from microduck_local.walk_env import MicroduckWalkEnv

    env = MicroduckWalkEnv(domain_rand=True, obs_noise=False, action_delay=False,
                           random_yaw=False, seed=0)
    for gid in env.foot_geoms.values():
        assert env.model.geom_priority[gid] == 1, "foot friction cannot win the pair"

    seen = []
    for s in range(10):
        env.reset(seed=s)
        for _ in range(40):
            env.step(np.zeros(C.NUM_JOINTS, np.float32))
            c = env.data.contact
            for i in range(env.data.ncon):
                if env.floor_geom in (c.geom1[i], c.geom2[i]):
                    seen.append(float(c.friction[i][0]))
                    break
    seen = np.array(seen)
    assert seen.min() < 0.95, f"floor never got slippery: min mu {seen.min():.3f}"
    assert seen.max() > 1.05, f"floor never got grippier: max mu {seen.max():.3f}"


def test_terminations_match_upstream():
    """Ours were STRICTER than the reference stack on both axes, and the height
    rule had 9 mm of margin over a normal gait's p1 trunk height (0.109 m) —
    enough to cut short any faster, more dynamic stride."""
    from microduck_local.walk_env import MicroduckWalkEnv

    # 70 deg, matching microduck_velocity_env_cfg.py (was 60).
    assert MicroduckWalkEnv.FALL_GRAVITY_Z == pytest.approx(-0.342, abs=1e-3)
    # Upstream has no height rule at all; ours is a folded-crouch guard only,
    # and must sit well clear of a real gait. Run disables it entirely.
    assert MicroduckWalkEnv.FALL_HEIGHT <= 0.08
    assert MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                            action_delay=False, seed=0).height_termination is True


def test_walk_tracking_and_commands_match_gpu_velocity():
    """track_linear_velocity std²=0.1, 15% turn-in-place, 20% forward bucket."""
    from microduck_local.walk_env import MicroduckWalkEnv

    assert MicroduckWalkEnv.TRACK_STD2 == pytest.approx(0.1)
    assert MicroduckWalkEnv.W_TRACK_ANG == pytest.approx(2.0)
    assert MicroduckWalkEnv.W_AIR_TIME == pytest.approx(3.0)
    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False, seed=0)
    assert env.turn_in_place_prob == pytest.approx(0.15)
    assert env.forward_command_prob == pytest.approx(0.2)
    assert env.max_steps == int(round(20.0 / C.CTRL_DT))


def test_last_action_is_the_raw_policy_output_not_the_clipped_one():
    """Unpriced quantities are free resources.

    The action-rate penalty and the obs both read `last_action`. Storing the
    CLIPPED action there let the policy grow its raw output without limit at
    zero cost: a 25M-step run reached mean |a| = 29.0 (max 140.9) with 52% of
    outputs pinned against the ±4 clip, and jerk 12.7 rad/step, while
    alpha_walking — which upstream never clips (clip_actions=None) — sits at
    0.19 and never saturates. The env still clips what it APPLIES; the reward
    must see what was actually asked for.
    """
    from microduck_local.walk_env import MicroduckWalkEnv

    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False, action_delay=False,
                           random_yaw=False, seed=0)
    env.reset(seed=0)
    huge = np.full(C.NUM_JOINTS, 40.0, np.float32)
    env.step(huge)
    assert np.allclose(env.last_action, 40.0), (
        "last_action was clipped — unbounded output growth is unpriced again")
    # ...but the actuator target stays inside the clip.
    assert float(np.max(np.abs(env.data.ctrl - C.DEFAULT_POSE))) <= 4.0 + 1e-6
