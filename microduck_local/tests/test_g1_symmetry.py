"""The G1's left/right mirror, checked against physics rather than argued.

The duck's tables are asserted against microduck_rl's own. The G1 has no
upstream table, so the proof here is the only one available and the stronger
one anyway: mirror the STATE, let MuJoCo recompute everything (IMU, gravity,
kinematics), and the observation of the mirrored robot must equal the mirror
of the original observation. A wrong sign on any joint breaks it.
"""

from __future__ import annotations

import numpy as np
import pytest

from microduck_local.robots import g1
from microduck_local.robots import g1_symmetry as S

pytestmark = pytest.mark.skipif(
    not g1.g1_ready(), reason="G1 assets missing — uv run fetch-g1")


def _env(**kw):
    from microduck_local.robots.g1_env import G1WalkEnv
    kw.setdefault("obs_noise", False)
    kw.setdefault("domain_rand", False)
    kw.setdefault("action_delay", False)
    kw.setdefault("random_yaw", False)
    return G1WalkEnv(seed=0, **kw)


def test_the_joint_permutation_pairs_every_limb():
    perm = S.joint_perm()
    names = g1.joint_names()
    assert len(perm) == 29
    for i, j in enumerate(perm):
        a, b = names[i], names[j]
        if a.startswith("left_"):
            assert b == "right_" + a[len("left_"):]
        elif a.startswith("right_"):
            assert b == "left_" + a[len("right_"):]
        else:                       # waist: its own mirror
            assert b == a
    assert np.array_equal(perm[perm], np.arange(29)), "mirroring twice is identity"


def test_the_sign_is_read_off_the_joint_axes():
    """+1 for pitch hinges, -1 for roll and yaw — and the SOURCE is the
    model's own `jnt_axis`, so a re-exported MJCF cannot drift from it."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(g1.g1_scene_xml()))
    sign = S.joint_sign(m)
    for i, name in enumerate(g1.joint_names()):
        axis = np.asarray(m.joint(name).axis, float)
        want = 1.0 if abs(axis[1]) > 0.99 else -1.0
        assert sign[i] == want, name
    # a couple by hand, so a wholesale sign flip cannot pass
    names = list(g1.joint_names())
    assert sign[names.index("left_knee_joint")] == 1.0
    assert sign[names.index("left_hip_roll_joint")] == -1.0
    assert sign[names.index("waist_yaw_joint")] == -1.0


def test_mirroring_the_state_mirrors_the_observation():
    """The load-bearing one: physics, not bookkeeping."""
    env = _env()
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    # Somewhere generic: walk a few steps under a random-ish action so no
    # block is accidentally symmetric (a standing robot mirrors trivially).
    for _ in range(25):
        env.step(rng.normal(0, 0.15, 29).astype(np.float32))
    env.twist_cmd[:] = (0.4, 0.2, -0.3)
    obs = env._get_obs().copy()
    last = env.last_action.copy()

    S.mirror_state(env.model, env.data)
    env.last_action = S.mirror_action(last)
    env.twist_cmd[:] = [0.4, -0.2, 0.3]
    env._refresh_derived()
    mirrored = env._get_obs()

    want = S.mirror_obs(obs)
    worst = float(np.abs(mirrored - want).max())
    assert worst < 2e-5, f"mirror broken, worst block delta {worst:.2e}"


def test_a_planted_sign_flip_breaks_the_physical_check(monkeypatch):
    """The test above is only worth having if it can fail."""
    env = _env()
    env.reset(seed=3)
    for _ in range(10):
        env.step(np.zeros(29, np.float32))
    obs = env._get_obs().copy()
    good = S.joint_sign(env.model)
    bad = good.copy()
    bad[list(g1.joint_names()).index("left_knee_joint")] *= -1
    monkeypatch.setattr(S, "joint_sign", lambda model=None: bad)
    S.mirror_state(env.model, env.data)
    env._refresh_derived()
    assert float(np.abs(env._get_obs() - S.mirror_obs(obs)).max()) > 1e-3


def test_mirroring_an_action_twice_is_the_identity():
    a = np.random.default_rng(1).normal(0, 1, 29).astype(np.float32)
    assert np.allclose(S.mirror_action(S.mirror_action(a)), a, atol=1e-6)


def test_obs_map_covers_every_slot_exactly_once():
    perm, sign = S.obs_perm_sign()
    assert sorted(perm.tolist()) == list(range(g1.OBS_DIM))
    assert set(np.abs(sign).tolist()) == {1.0}
