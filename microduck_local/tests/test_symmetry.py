"""Locks for the bilateral mirror map and the PPO symmetry loss.

The map is a signed permutation, and a wrong sign or a swapped index is
invisible in every unit test that only checks algebra — it just quietly
teaches the policy a wrong prior. So the load-bearing tests here are the
PHYSICAL ones: they mirror a robot state through the map inside MuJoCo and
check the simulator agrees, both statically (every geom lands at its
reflection) and dynamically (a mirrored rollout tracks the mirror of the
original).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from microduck_local import contract as C
from microduck_local import symmetry as S
from microduck_local.walk_env import MicroduckWalkEnv

REFLECT_Y = np.array([1.0, -1.0, 1.0])


# ---------------------------------------------------------------- the tables


def test_joint_map_matches_upstream_and_our_joint_names():
    """The tables are microduck_rl's, re-derived from our JOINT_NAMES."""
    assert list(S.JOINT_PERM) == [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
    assert list(S.JOINT_SIGN) == [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]
    # ... and the permutation really is "left_X <-> right_X, midline fixed".
    for i, name in enumerate(C.JOINT_NAMES):
        partner = C.JOINT_NAMES[S.JOINT_PERM[i]]
        if name.startswith(("left_", "right_")):
            assert partner == name.replace("left_", "R_").replace("right_", "left_") \
                                  .replace("R_", "right_")
        else:
            assert partner == name
    # Only neck_pitch / head_pitch survive the reflection unsigned.
    kept = {C.JOINT_NAMES[i] for i, s in enumerate(S.JOINT_SIGN) if s > 0}
    assert kept == {"neck_pitch", "head_pitch"}


def test_obs_map_permutes_only_the_joint_blocks():
    """Base and command slots mirror IN PLACE — signs only, no reindexing."""
    ident = np.arange(C.OBS_DIM)
    for lo, hi in [(0, 6), (48, 61)]:
        np.testing.assert_array_equal(S.OBS_PERM[lo:hi], ident[lo:hi])
    for base in (6, 20, 34):  # joint_pos_rel, joint_vel, last_action
        np.testing.assert_array_equal(S.OBS_PERM[base:base + 14], base + S.JOINT_PERM)
        np.testing.assert_array_equal(S.OBS_SIGN[base:base + 14], S.JOINT_SIGN)
    # gyro (pseudovector) negates roll+yaw; gravity (vector) negates y only.
    np.testing.assert_array_equal(S.OBS_SIGN[0:3], [-1, 1, -1])
    np.testing.assert_array_equal(S.OBS_SIGN[3:6], [1, -1, 1])
    np.testing.assert_array_equal(S.OBS_SIGN[48:51], [1, -1, -1])          # twist
    np.testing.assert_array_equal(S.OBS_SIGN[51:55], [1, 1, -1, -1])       # head cmd
    np.testing.assert_array_equal(S.OBS_SIGN[55:61], [1, -1, 1, -1, 1, -1])  # body cmd
    # A signed permutation is invertible: every index used exactly once.
    assert sorted(S.OBS_PERM) == list(range(C.OBS_DIM))


def test_default_pose_is_its_own_mirror():
    """The home pose is bilaterally symmetric, which is WHY one map serves both
    absolute joint angles and joint_pos_rel: mirroring commutes with
    subtracting DEFAULT_POSE only because M(DEFAULT_POSE) == DEFAULT_POSE."""
    np.testing.assert_allclose(S.mirror_action(C.DEFAULT_POSE), C.DEFAULT_POSE, atol=1e-7)


# ------------------------------------------------------------- the operators


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_mirroring_twice_is_identity(backend):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(7, C.OBS_DIM)).astype(np.float32)
    act = rng.normal(size=(7, C.NUM_JOINTS)).astype(np.float32)
    if backend == "torch":
        obs, act = torch.as_tensor(obs), torch.as_tensor(act)
        assert torch.allclose(S.mirror_obs(S.mirror_obs(obs)), obs)
        assert torch.allclose(S.mirror_action(S.mirror_action(act)), act)
    else:
        np.testing.assert_array_equal(S.mirror_obs(S.mirror_obs(obs)), obs)
        np.testing.assert_array_equal(S.mirror_action(S.mirror_action(act)), act)


def test_numpy_and_torch_agree_and_shapes_are_flexible():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(3, 5, C.OBS_DIM)).astype(np.float32)  # arbitrary batch dims
    np.testing.assert_array_equal(
        S.mirror_obs(obs), S.mirror_obs(torch.as_tensor(obs)).numpy()
    )
    flat = rng.normal(size=C.NUM_JOINTS).astype(np.float32)      # unbatched
    np.testing.assert_array_equal(
        S.mirror_action(flat), S.mirror_action(torch.as_tensor(flat)).numpy()
    )
    with pytest.raises(ValueError):
        S.mirror_obs(np.zeros(60, np.float32))
    with pytest.raises(ValueError):
        S.mirror_action(np.zeros(13, np.float32))


def test_mirror_in_normalized_space_equals_mirror_in_raw_space():
    """VecNormalize does not commute with the mirror; mirror_normalized_obs is
    the conjugated operator that does."""
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(16, C.OBS_DIM)).astype(np.float32)
    mean = rng.normal(size=C.OBS_DIM).astype(np.float32)
    std = rng.uniform(0.5, 2.0, C.OBS_DIM).astype(np.float32)
    normed = (raw - mean) / std
    want = (S.mirror_obs(raw) - mean) / std
    got = S.mirror_normalized_obs(normed, mean, std)
    np.testing.assert_allclose(got, want, atol=1e-5)
    # And the naive version (mirroring normalized obs directly) really is wrong,
    # i.e. this test would pass vacuously if it were not.
    assert not np.allclose(S.mirror_obs(normed), want, atol=1e-3)


# ------------------------------------------------------------ physical tests


@pytest.fixture(scope="module")
def envs():
    """Two identical, fully deterministic copies of the walking env."""
    def mk():
        return MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                                action_delay=False, random_yaw=False, seed=0)
    return mk(), mk()


def _mirror_full_state(src: MicroduckWalkEnv, dst: MicroduckWalkEnv) -> None:
    """Put `dst` in the sagittal-plane reflection of `src`'s state.

    Free joint: position reflects in y; the orientation quaternion (w,x,y,z)
    becomes (w,-x,y,-z) — conjugating a rotation by the improper S=diag(1,-1,1)
    gives R' = S R S, i.e. the same angle about the reflected-and-negated axis.
    MuJoCo stores free-joint linear velocity in WORLD frame (reflects in y) and
    angular velocity in the BODY frame, where the pseudovector picks up the
    extra sign: (-wx, wy, -wz).
    """
    qpos, qvel = src.data.qpos.copy(), src.data.qvel.copy()
    qpos[1] = -qpos[1]
    w, x, y, z = qpos[3:7]
    qpos[3:7] = [w, -x, y, -z]
    rel = src.data.qpos[src.joint_qpos_adr] - C.DEFAULT_POSE
    qpos[dst.joint_qpos_adr] = C.DEFAULT_POSE + S.mirror_action(rel)
    qvel[0:3] *= REFLECT_Y
    qvel[3:6] *= [-1.0, 1.0, -1.0]
    qvel[dst.joint_qvel_adr] = S.mirror_action(src.data.qvel[src.joint_qvel_adr])

    dst.data.qpos[:] = qpos
    dst.data.qvel[:] = qvel
    mujoco.mj_forward(dst.model, dst.data)
    # Bookkeeping the observation reads but the physics does not hold.
    dst.prev_joint_vel = S.mirror_action(src.prev_joint_vel).astype(np.float32)
    dst.last_action[:] = S.mirror_action(src.last_action)
    dst.twist_cmd[:] = src.twist_cmd * [1, -1, -1]
    dst.head_cmd[:] = src.head_cmd * [1, 1, -1, -1]
    dst.body_cmd[:] = src.body_cmd * [1, -1, 1, -1, 1, -1]
    dst.step_count = src.step_count


def _random_state(env: MicroduckWalkEnv, rng: np.random.Generator) -> None:
    qpos = env.data.qpos.copy()
    qvel = np.zeros_like(env.data.qvel)
    qpos[0:3] = [0.03, -0.02, 0.24]
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = 0.6
    qpos[3:7] = [np.cos(ang / 2), *(np.sin(ang / 2) * axis)]
    qpos[env.joint_qpos_adr] = C.DEFAULT_POSE + rng.uniform(-0.3, 0.3, C.NUM_JOINTS)
    qvel[0:3] = rng.normal(size=3) * 0.2
    qvel[3:6] = rng.normal(size=3) * 0.8
    qvel[env.joint_qvel_adr] = rng.normal(size=C.NUM_JOINTS)
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)
    env.prev_joint_vel = env._joint_vel().copy()
    env.last_action[:] = rng.uniform(-0.4, 0.4, C.NUM_JOINTS)
    env.twist_cmd[:] = rng.uniform(-0.3, 0.3, 3)
    env.head_cmd[:] = rng.uniform(-0.05, 0.05, 4)
    env.body_cmd[:] = rng.uniform(-0.05, 0.05, 6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mirrored_state_produces_mirrored_observation(envs, seed):
    """THE test for a wrong permutation or sign.

    Independently of the map, put the robot in the physically mirrored state
    (reflect the free joint, mirror the joint angles) and let MuJoCo compute
    the IMU and gravity vectors from scratch. The resulting 61-D observation
    must equal mirror_obs() of the original — exactly, not approximately.
    """
    a, b = envs
    a.reset(seed=seed)
    b.reset(seed=seed)
    _random_state(a, np.random.default_rng(seed))
    _mirror_full_state(a, b)

    obs_a, obs_b = a._get_obs(), b._get_obs()
    err = np.abs(S.mirror_obs(obs_a) - obs_b)
    assert err.max() < 1e-6, (
        "mirror map disagrees with the simulator; worst obs index "
        f"{int(err.argmax())} (err {err.max():.2e})"
    )
    # Guard against a vacuous pass: the two states must actually differ.
    assert np.abs(obs_a - obs_b).max() > 0.1


# Mirror partners in the robot_walk.xml body tree. The names are the CAD
# export's, not ours; they are model constants and a rename would fail loudly.
_BODY_PAIRS = [("yaw2roll", "bearing_roll"), ("hip_l", "hip_l_2"),
               ("upper_leg_left", "upper_leg_right"), ("leg", "leg_2"),
               ("ankle_left", "ankle_right")]
# Midline bodies whose pose DOES change under the mirror (head_yaw/head_roll
# drive them), so their frame must map to its own reflection.
_BODY_MIDLINE = ["trunk_base", "yaw_roll_motion", "jaw_soft"]
# Midline bodies driven ONLY by the two sign-preserving joints (neck_pitch,
# head_pitch): their frames must come out IDENTICAL, not reflected. (Their
# frame origins sit 14.5 mm off the midline — a CAD convention — so "reflected"
# is the wrong predicate for them; "unmoved" is the right one, and it is what
# catches a wrong sign on the two pitch joints.)
_BODY_UNMOVED = ["neck", "neck_pitch"]

_S = np.diag([1.0, -1.0, 1.0])


def _pose_frames(env: MicroduckWalkEnv, rel: np.ndarray):
    """World body frames for joint_pos_rel `rel`, trunk level at the origin."""
    data = mujoco.MjData(env.model)
    data.qpos[0:3] = [0.0, 0.0, 0.25]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE + rel
    mujoco.mj_forward(env.model, data)
    return data


def _frame_reflection_error(env, rel, mirrored_rel, conv) -> tuple[float, float]:
    """(reflection error over mirror partners, motion of the unmoved bodies)."""
    d1, d2 = _pose_frames(env, rel), _pose_frames(env, mirrored_rel)
    reflect = 0.0
    for left, right in _BODY_PAIRS:
        for src, dst in ((left, right), (right, left)):
            i, j = env.model.body(src).id, env.model.body(dst).id
            r1 = d1.xmat[i].reshape(3, 3)
            reflect = max(
                reflect,
                np.abs(d1.xpos[i] * REFLECT_Y - d2.xpos[j]).max(),
                np.abs(_S @ r1 @ _S @ conv[(src, dst)] - d2.xmat[j].reshape(3, 3)).max(),
            )
    for name in _BODY_MIDLINE:
        i = env.model.body(name).id
        reflect = max(
            reflect,
            np.abs(d1.xpos[i] * REFLECT_Y - d2.xpos[i]).max(),
            np.abs(_S @ d1.xmat[i].reshape(3, 3) @ _S @ conv[(name, name)]
                   - d2.xmat[i].reshape(3, 3)).max(),
        )
    unmoved = max(
        max(np.abs(d1.xpos[env.model.body(n).id] - d2.xpos[env.model.body(n).id]).max(),
            np.abs(d1.xmat[env.model.body(n).id] - d2.xmat[env.model.body(n).id]).max())
        for n in _BODY_UNMOVED
    )
    return reflect, unmoved


def test_mirrored_pose_places_every_limb_at_its_reflection(envs):
    """The kinematic ground truth, computed by MuJoCo rather than by our map.

    Feed the mirrored joint angles into forward kinematics and demand that
    every body land at its partner's reflected pose. Orientation is checked
    too, up to a constant per-body frame convention measured at the home pose —
    left and right limbs carry OPPOSITE local axes in this CAD export, which is
    precisely why the leg signs flip, so the raw R' = S R S predicate does not
    hold and the convention has to be divided out. It is a constant, so any
    wrong joint sign still breaks the check away from home, which the
    corruption sweep below proves.
    """
    a, _ = envs
    rng = np.random.default_rng(7)
    rel = rng.uniform(-0.35, 0.35, C.NUM_JOINTS)
    mirrored = S.mirror_action(rel.astype(np.float32)).astype(np.float64)

    # Calibrate the frame conventions at the home pose (which is its own mirror).
    home = _pose_frames(a, np.zeros(C.NUM_JOINTS))
    conv = {}
    for left, right in _BODY_PAIRS:
        for src, dst in ((left, right), (right, left)):
            r_src = home.xmat[a.model.body(src).id].reshape(3, 3)
            r_dst = home.xmat[a.model.body(dst).id].reshape(3, 3)
            conv[(src, dst)] = (_S @ r_src @ _S).T @ r_dst
    for name in _BODY_MIDLINE:
        r0 = home.xmat[a.model.body(name).id].reshape(3, 3)
        conv[(name, name)] = (_S @ r0 @ _S).T @ r0

    reflect, unmoved = _frame_reflection_error(a, rel, mirrored, conv)
    assert reflect < 1e-6, f"mirrored pose is not the reflection (err {reflect:.2e})"
    assert unmoved < 1e-6, f"sagittal-joint bodies moved (err {unmoved:.2e})"

    # ... and the check has teeth: corrupting any single entry of the map — a
    # flipped sign on any of the 14 joints, or a swapped index pair — blows it
    # up by 6+ orders of magnitude.
    for i in range(C.NUM_JOINTS):
        bad = mirrored.copy()
        bad[i] *= -1
        r, u = _frame_reflection_error(a, rel, bad, conv)
        assert max(r, u) > 1e-2, (
            f"flipping the sign on {C.JOINT_NAMES[i]} went unnoticed ({max(r, u):.2e})"
        )
    bad = mirrored.copy()
    bad[[3, 4]] = bad[[4, 3]]                       # left_knee <-> left_ankle
    assert _frame_reflection_error(a, rel, bad, conv)[0] > 1e-2


def _mirrored_rollout(envs, seed: int, steps: int, mirror_the_actions: bool = True):
    """Run a twin pair and yield the per-step |mirror_obs(obs_a) - obs_b|."""
    a, b = envs
    obs_a, _ = a.reset(seed=seed)
    b.reset(seed=seed)
    _mirror_full_state(a, b)
    a.resample_steps = b.resample_steps = 10 ** 9      # freeze the commands
    assert np.abs(S.mirror_obs(obs_a) - b._get_obs()).max() < 1e-6

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        act = rng.uniform(-0.15, 0.15, C.NUM_JOINTS).astype(np.float32)
        twin = S.mirror_action(act).astype(np.float32) if mirror_the_actions else act
        obs_a, *_ = a.step(act)
        obs_b, *_ = b.step(twin)
        out.append(np.abs(S.mirror_obs(obs_a) - obs_b))
    return np.array(out)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_mirrored_rollout_stays_mirrored(envs, seed):
    """Dynamics check: the mirrored twin, driven by mirrored actions, produces
    the mirrored observation — MuJoCo's integrator, contacts and actuators all
    in the loop, nothing derived from our map on the output side.

    Two control steps, because the tracking is exact only to the extent the
    robot's mass distribution is (see the next test); that is long enough for
    ground contact, servo torques and the joint_vel lag to matter, and the
    control below shows the margin is ~300x, not marginal.
    """
    err = _mirrored_rollout(envs, seed, steps=2)
    assert err.max() < 1e-2, f"mirrored rollout diverged by {err.max():.2e}"
    # Teeth: the same rollout driven by UNMIRRORED actions is ~300x worse, so
    # this is not passing because everything is near zero.
    control = _mirrored_rollout(envs, seed, steps=2, mirror_the_actions=False)
    assert control.max() > 0.5 > 50 * err.max()


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_rollout_drift_starts_in_the_head_not_the_legs(envs, seed):
    """The honest caveat, asserted rather than glossed.

    Over a longer horizon the mirrored twin DOES drift, so it matters where it
    starts. The leg chains are exact mirror twins (see the inertia test below),
    but the head/neck subassembly is not — `yaw_roll_motion`'s CoM sits 5.0 mm
    off the midline, `jaw_soft`'s 1.2 mm — so the first channel to go is
    head_roll's VELOCITY, a few control steps in, while the legs are still two
    orders tighter. That is the signature of a model asymmetry, not a wrong
    map: a bad permutation or sign would move a LEG and would show up in the
    slow-integrating channels (joint positions, projected gravity) that this
    test pins.

    Past ~5 control steps the head residual feeds the trunk and contacts and
    the whole state diverges chaotically. That long-horizon divergence is NOT
    attributed here — an attempt to pin it by making the head massless changed
    the numerics too much to be evidence either way — so nothing is asserted
    about it.
    """
    err = _mirrored_rollout(envs, seed, steps=10)
    head_roll = C.JOINT_NAMES.index("head_roll")
    assert err[:, 6:20].max() < 1.5e-2, "joint positions drifted — suspect the map"
    assert err[:, 3:6].max() < 1e-2, "projected gravity drifted — suspect the map"
    early = err[:5, 20:34].max(axis=0)
    assert int(early.argmax()) == head_roll, (
        "expected head_roll's velocity to be the first channel to drift, got "
        f"{C.JOINT_NAMES[int(early.argmax())]} — that would point at the map"
    )
    others = np.delete(early, head_roll)
    assert others.max() < early[head_roll] / 5


def test_mirroring_head_roll_inertia_is_the_only_asymmetry_left(envs):
    """Attribution for the tolerance above: the leg chain's inertial
    properties ARE exact mirror twins, so the residual is not our map."""
    model = envs[0].model
    pairs = [("yaw2roll", "bearing_roll"), ("hip_l", "hip_l_2"),
             ("upper_leg_left", "upper_leg_right"), ("leg", "leg_2"),
             ("ankle_left", "ankle_right")]
    for left, right in pairs:
        bl, br = model.body(left), model.body(right)
        assert abs(bl.mass[0] - br.mass[0]) < 1e-6
        np.testing.assert_allclose(bl.inertia, br.inertia, atol=1e-9)
    # ... while the head links carry genuine lateral CoM offsets.
    assert abs(model.body("yaw_roll_motion").ipos[1]) > 1e-3


# -------------------------------------------------------------- the PPO loss


def _signed_perm_matrix(perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Dense M with (M v)[i] == sign[i] * v[perm[i]]."""
    m = np.zeros((len(perm), len(perm)), dtype=np.float32)
    m[np.arange(len(perm)), perm] = sign
    return m


def _make_model(symmetry_coef: float, seed: int = 0, desired_kl=None):
    """A SymmetryPPO whose policy mean is a single Linear(61, 14).

    `net_arch=[]` makes the actor's latent extractor the identity, so the
    action mean is exactly `W obs + b` — the only architecture in which an
    EXACTLY equivariant policy can be constructed by hand, which is what the
    zero/non-zero loss tests need.
    """
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
        symmetry_coef=symmetry_coef,
        # Off by default here: these tests isolate the MIRROR LOSS, and the
        # KL controller would move the learning rate underneath them.
        desired_kl=desired_kl,
    )


def _symmetrize_action_net(model, norm=None) -> None:
    """Project the linear head onto the mirror-equivariant subspace.

    The policy sees NORMALIZED observations, so the mirror it must commute
    with is the conjugated affine map ``T(x) = A x + c`` from
    ``mirror_normalized_obs`` (``A = diag(1/s) P diag(s)``,
    ``c = (P mu - mu)/s``), not the bare signed permutation ``P``. Both ``A``
    and the action map ``Sa`` are involutions, so ``W <- (W + Sa W A)/2``
    projects onto ``{W : Sa W = W A}``. The bias then needs
    ``Sa b - b = W c``; since ``A c = -c`` the required correction ``-W c / 2``
    lands entirely in ``Sa``'s -1 eigenspace, so it composes cleanly with the
    usual ``(b + Sa b)/2``.
    """
    sa = _signed_perm_matrix(S.JOINT_PERM, S.JOINT_SIGN)
    so = _signed_perm_matrix(S.OBS_PERM, S.OBS_SIGN)
    if norm is None:
        a_mat = so
        c = np.zeros(C.OBS_DIM, dtype=np.float32)
    else:
        mean = np.asarray(norm[0], dtype=np.float32)
        std = np.asarray(norm[1], dtype=np.float32)
        a_mat = so * std[None, :] / std[:, None]
        c = (so @ mean - mean) / std
    with torch.no_grad():
        w = model.policy.action_net.weight.numpy()
        b = model.policy.action_net.bias.numpy()
        w_sym = (w + sa @ w @ a_mat) / 2
        model.policy.action_net.weight.copy_(torch.as_tensor(w_sym))
        model.policy.action_net.bias.copy_(
            torch.as_tensor((b + sa @ b) / 2 - (w_sym @ c) / 2)
        )


def _scale_action_net(model, factor: float) -> None:
    """SB3 initialises the action head with gain 0.01; scale it up so an
    asymmetric policy's mirror error is unambiguously nonzero."""
    with torch.no_grad():
        model.policy.action_net.weight.mul_(factor)
        model.policy.action_net.bias.add_(0.1)


def test_symmetry_loss_is_zero_for_a_symmetric_policy_and_positive_otherwise():
    obs = torch.as_tensor(
        np.random.default_rng(4).normal(size=(64, C.OBS_DIM)).astype(np.float32)
    )
    model = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=3)
    _scale_action_net(model, 50.0)

    asym = model._symmetry_loss(obs, None).item()
    _symmetrize_action_net(model)
    sym = model._symmetry_loss(obs, None).item()

    assert sym < 1e-10, f"equivariant policy should score ~0, got {sym:.3e}"
    assert asym > 1e-2, f"asymmetric policy should score clearly >0, got {asym:.3e}"


def test_symmetry_loss_gradient_actually_symmetrizes_the_policy():
    """The loss is not just a number: descending it makes pi equivariant."""
    obs = torch.as_tensor(
        np.random.default_rng(5).normal(size=(64, C.OBS_DIM)).astype(np.float32)
    )
    model = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=6)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-2)
    before = model._symmetry_loss(obs, None).item()
    for _ in range(200):
        opt.zero_grad()
        loss = model._symmetry_loss(obs, None)
        loss.backward()
        opt.step()
    after = model._symmetry_loss(obs, None).item()
    assert after < before / 10, f"symmetry loss barely moved: {before:.3e} -> {after:.3e}"


def test_symmetry_loss_uses_the_normalizer_when_one_is_present():
    """With VecNormalize in the stack the mirror must be conjugated by it."""
    model = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=8)
    _scale_action_net(model, 50.0)
    rng = np.random.default_rng(9)
    mean = torch.as_tensor(rng.normal(size=C.OBS_DIM).astype(np.float32))
    std = torch.as_tensor(rng.uniform(0.5, 2.0, C.OBS_DIM).astype(np.float32))
    normed = torch.as_tensor(rng.normal(size=(32, C.OBS_DIM)).astype(np.float32))
    _symmetrize_action_net(model, (mean, std))
    # A policy made equivariant w.r.t. the CONJUGATED mirror scores ~0 there,
    # and clearly nonzero under the naive one — i.e. the normalizer really is
    # part of the operator, not a detail.
    assert model._symmetry_loss(normed, (mean, std)).item() < 1e-8
    assert model._symmetry_loss(normed, None).item() > 1e-2


def test_zero_coef_is_bit_identical_to_stock_ppo():
    """The vendored copy of SB3's PPO.train() must not have drifted.

    This is the guard for the one genuinely fragile thing in symmetry.py: if a
    future SB3 changes train() and our copy goes stale, the parameters diverge
    here even at symmetry_coef=0.

    Both deliberate deviations are switched off for the comparison: zero
    mirror coefficient and no KL controller (`desired_kl=None`). With either
    on, differing from stock PPO is the entire point.
    """
    from stable_baselines3 import PPO

    ours = _make_model(0.0, seed=12)
    ours.learn(total_timesteps=256)

    stock = PPO(
        "MlpPolicy", _make_model(0.0, seed=12).get_env(),
        policy_kwargs=dict(net_arch=dict(pi=[], vf=[])),
        n_steps=64, batch_size=32, n_epochs=1, device="cpu", seed=12,
    )
    stock.learn(total_timesteps=256)

    for (kn, ours_p), (_, stock_p) in zip(
        ours.policy.state_dict().items(), stock.policy.state_dict().items()
    ):
        assert torch.equal(ours_p, stock_p), f"vendored train() drifted at {kn}"


def test_evaluate_actions_helper_matches_stock_and_feeds_symmetry_loss():
    """The fused train() path must not change the surrogate or the mirror loss.

    `_evaluate_actions` is evaluate_actions plus the action mean, one feature
    extract. Passing that mean into `_symmetry_loss` must equal the standalone
    `[o ; M o]` forward the tests (and rsl_rl) use.
    """
    model = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=21)
    rng = np.random.default_rng(22)
    obs = torch.as_tensor(rng.normal(size=(64, C.OBS_DIM)).astype(np.float32))
    actions = torch.as_tensor(rng.normal(size=(64, C.NUM_JOINTS)).astype(np.float32))
    v1, lp1, e1 = model.policy.evaluate_actions(obs, actions)
    v2, lp2, e2, mean = model._evaluate_actions(obs, actions)
    assert torch.allclose(v1, v2)
    assert torch.allclose(lp1, lp2)
    assert torch.allclose(e1, e2)
    assert torch.allclose(mean, model._policy_mean(obs))
    cat = model._symmetry_loss(obs, None)
    fused = model._symmetry_loss(obs, None, mean_orig=mean)
    assert torch.allclose(cat, fused, atol=1e-6)


def test_nonzero_coef_changes_the_update():
    """...and with the coefficient on, the update is genuinely different."""
    off = _make_model(0.0, seed=13)
    off.learn(total_timesteps=256)
    on = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=13)
    on.learn(total_timesteps=256)
    w_off = off.policy.action_net.weight
    w_on = on.policy.action_net.weight
    assert not torch.allclose(w_off, w_on), "symmetry_coef had no effect on training"


def test_checkpoint_still_loads_as_a_stock_ppo(tmp_path):
    """export_onnx.py and viz_server.py both call `PPO.load` on model.zip.
    Switching the trainer to SymmetryPPO must not break either."""
    from stable_baselines3 import PPO

    model = _make_model(S.DEFAULT_SYMMETRY_COEF, seed=14)
    model.save(str(tmp_path / "model"))
    reloaded = PPO.load(str(tmp_path / "model"), device="cpu")
    obs = np.zeros((1, C.OBS_DIM), np.float32)
    np.testing.assert_allclose(
        reloaded.predict(obs, deterministic=True)[0],
        model.predict(obs, deterministic=True)[0],
        atol=1e-6,
    )


# ------------------------------------------- which runs get the mirror prior


def _coef(argv: list[str]) -> float:
    """The coefficient a `train-behavior` command line actually trains under.

    Goes through the REAL parser so the default of None — "the user did not
    say" — is part of the lock, not a value the test supplies itself.
    """
    from microduck_local.behaviors import BEHAVIORS
    from microduck_local.train_behavior import build_parser, symmetry_coef_for

    args = build_parser().parse_args(argv)
    return symmetry_coef_for(BEHAVIORS[args.behavior], args.symmetry_coef)


def test_one_sided_behaviors_default_to_no_mirror_loss():
    """viz_server's /teach launches train_behavior WITHOUT --symmetry-coef, so
    this default is what every taught trick trains under. one_leg names the
    right foot as the lifted one; run is an ordinary gait."""
    assert _coef(["one_leg"]) == 0.0
    assert _coef(["imitate"]) == 0.0
    assert _coef(["run"]) == S.DEFAULT_SYMMETRY_COEF
    assert _coef(["spin"]) == S.DEFAULT_SYMMETRY_COEF


def test_an_explicit_coefficient_wins_in_both_directions():
    """The flag is authoritative: it can force the prior ON for a one-sided
    recipe (a warm start from a symmetric policy may still want it) and OFF
    for a symmetric one."""
    assert _coef(["one_leg", "--symmetry-coef", "0.5"]) == 0.5
    assert _coef(["run", "--symmetry-coef", "0"]) == 0.0
    # Explicit 0 must not read as "unset" — that is the whole reason the CLI
    # default is None rather than DEFAULT_SYMMETRY_COEF.
    assert _coef(["one_leg", "--symmetry-coef", "0"]) == 0.0


def test_a_motion_clip_turns_the_prior_off_for_any_behavior(monkeypatch):
    """/teach takes a clip per RUN and hands it over as MICRODUCK_CLIP, for
    whichever behavior was asked for — so 'does this run imitate a clip' is
    not answerable from the recipe alone. Resolved exactly as BehaviorEnv
    resolves it, or the trainer would reason about a different clip than the
    env loads."""
    assert _coef(["spin"]) == S.DEFAULT_SYMMETRY_COEF
    monkeypatch.setenv("MICRODUCK_CLIP", "run")
    assert _coef(["spin"]) == 0.0
    assert _coef(["spin", "--symmetry-coef", "0.5"]) == 0.5  # still overridable


def test_a_clip_makes_the_mirror_a_time_reflection_however_symmetric_it_is():
    """Why `imitate` is asymmetric for EVERY clip, and why deriving the flag
    from the clip's own left/right content would be the wrong test.

    The clip phase rides in body_cmd[4:6] = obs 59/60 as (sin p, cos p), and
    the mirror negates the cos slot: (sin p, -cos p) == (sin(pi-p), cos(pi-p)).
    So M does not merely reflect the robot, it jumps the clip to time pi-p.
    The prior then demands the mirrored pose at an unrelated instant of the
    motion — which the shipped backflip clip fails despite being exactly its
    own mirror frame by frame.
    """
    from microduck_local import motion

    assert (S.OBS_SIGN[59], S.OBS_SIGN[60]) == (1.0, -1.0)
    for p in (0.3, 1.7, 4.0):
        np.testing.assert_allclose(
            [np.sin(p), -np.cos(p)], [np.sin(np.pi - p), np.cos(np.pi - p)],
            atol=1e-12)

    clip = motion.load_clip("backflip")
    # Frame by frame the clip IS its own mirror — a purely sagittal motion.
    np.testing.assert_allclose(S.mirror_action(clip.joints), clip.joints,
                               atol=1e-9)
    # And it still violates what the mirror loss would ask of it.
    worst = 0.0
    for t in range(clip.steps):
        s, c = clip.phase(t)
        reflected = (np.pi - np.arctan2(s, c)) % (2 * np.pi)
        t2 = int(round(reflected / (2 * np.pi) * clip.steps)) % clip.steps
        worst = max(worst, np.abs(S.mirror_action(clip.joints[t])
                                  - clip.joints[t2]).max())
    assert worst > 0.5, (
        f"backflip clip satisfies the phase-reflected mirror to {worst:.4f} rad "
        "— if a clip really can satisfy it, the flat False on `imitate` is "
        "too blunt and should be derived per clip"
    )


def _train_and_capture(tmp_path, monkeypatch, argv: list[str]) -> tuple[float, float]:
    """Run `train-behavior <argv>` far enough to build the model, and report
    (coefficient the optimizer was built with, coefficient on record).

    learn() is stubbed: what is under test is the number SymmetryPPO is
    CONSTRUCTED with, not anything it could learn from a token rollout. Every
    other step is the real main() — the parser, the resolver and the
    behavior.json record all sit on this path.
    """
    from microduck_local import train_behavior as tb

    trained: list[float] = []

    def fake_learn(self, *a, callback=None, **k):
        trained.append(self.symmetry_coef)
        callback.model = self  # what SB3's _setup_learn would have done
        return self

    monkeypatch.setattr(
        "microduck_local.symmetry.SymmetryPPO.learn", fake_learn)
    monkeypatch.setattr(tb, "RUNS_DIR", tmp_path)
    monkeypatch.setenv("MICRODUCK_VEC_ENV", "dummy")  # don't fork the test process
    monkeypatch.setattr("sys.argv", ["train-behavior", *argv])
    tb.main()
    assert len(trained) == 1
    run_name = argv[argv.index("--run-name") + 1]
    import json
    recorded = json.loads((tmp_path / run_name / "behavior.json").read_text())
    return trained[0], recorded["symmetry_coef"]


@pytest.mark.parametrize("behavior_id, expected", [
    ("one_leg", 0.0),
    ("run", S.DEFAULT_SYMMETRY_COEF),
])
def test_the_default_reaches_the_optimizer_and_the_record(
        tmp_path, monkeypatch, behavior_id, expected):
    """The whole wire, end to end: no --symmetry-coef on the command line (how
    viz_server's /teach launches it) and one_leg's PPO is built at 0 while
    run's keeps the mirror loss."""
    built, recorded = _train_and_capture(
        tmp_path, monkeypatch,
        [behavior_id, "--envs", "1", "--steps", "1", "--run-name", behavior_id])
    assert built == expected
    assert recorded == expected
