"""Model sharing: one compiled mjModel behind many envs.

The compiled model is ~150x the size of the mjData it drives, so sharing it is
most of the memory story — but only if sharing is INVISIBLE. These tests lock
the two ways it could stop being invisible: envs bleeding simulation state into
each other, and envs bleeding domain randomization into each other.
"""

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.vec_env import (
    BACKENDS,
    DEFAULT_BACKEND,
    ENV_VAR,
    make_vec_env,
    resolve_backend,
)
from microduck_local.walk_env import (
    MicroduckWalkEnv,
    clear_shared_models,
    shared_model,
    shared_model_scope,
)

ACTION = np.linspace(-0.3, 0.3, C.NUM_JOINTS).astype(np.float32)


@pytest.fixture(scope="module")
def shared_pair():
    """Two envs that share ONE mjModel object, in this process."""
    with shared_model_scope(exclusive=False):
        a = MicroduckWalkEnv(seed=11, obs_noise=False, domain_rand=True)
        b = MicroduckWalkEnv(seed=22, obs_noise=False, domain_rand=True)
    assert a.model is b.model, "the whole point: one compiled model, two envs"
    return a, b


# --------------------------------------------------------------- the cache


def test_shared_model_compiles_once_per_scene():
    clear_shared_models()
    try:
        first = shared_model(C.SCENE_WALK_XML)
        assert shared_model(C.SCENE_WALK_XML) is first
        assert shared_model(C.SCENE_ALL_XML) is not first
    finally:
        clear_shared_models()


def test_default_construction_still_compiles_privately():
    """No scope, no sharing: the pre-existing path must be untouched."""
    a = MicroduckWalkEnv(seed=0)
    b = MicroduckWalkEnv(seed=1)
    assert a.model is not b.model
    assert a._model_shared is False and b._model_shared is False


def test_explicit_model_argument_is_adopted():
    model = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    env = MicroduckWalkEnv(seed=0, model=model)
    assert env.model is model
    # Handed in explicitly = the caller's promise that it is theirs alone, so
    # the per-step re-assert stays off.
    assert env._model_shared is False


def test_fork_scope_is_exclusive_and_in_process_scope_is_not():
    clear_shared_models()
    try:
        with shared_model_scope(exclusive=True):
            forked = MicroduckWalkEnv(seed=0)
        with shared_model_scope(exclusive=False):
            in_proc = MicroduckWalkEnv(seed=0)
        assert forked.model is in_proc.model     # same cache entry
        assert forked._model_shared is False     # one env per process
        assert in_proc._model_shared is True     # siblings step this model
    finally:
        clear_shared_models()


# ------------------------------------------------- independence under sharing


def test_stepping_one_shared_env_does_not_perturb_the_other(shared_pair):
    a, b = shared_pair
    a.reset(seed=101)
    b.reset(seed=202)
    b_qpos = b.data.qpos.copy()
    b_qvel = b.data.qvel.copy()
    b_obs = b._get_obs().copy()

    for _ in range(50):
        a.step(ACTION)

    np.testing.assert_array_equal(b.data.qpos, b_qpos)
    np.testing.assert_array_equal(b.data.qvel, b_qvel)
    # _get_obs advances the joint_vel lag buffer, so compare a fresh read of
    # the state that feeds it rather than the cached array.
    np.testing.assert_array_equal(b._get_obs()[:34], b_obs[:34])


def test_shared_envs_match_privately_compiled_envs_step_for_step():
    """Sharing must not change the physics — same seed, same trajectory."""
    private = MicroduckWalkEnv(seed=7, obs_noise=False, domain_rand=True)
    with shared_model_scope(exclusive=False):
        shared = MicroduckWalkEnv(seed=7, obs_noise=False, domain_rand=True)
        noise = MicroduckWalkEnv(seed=99, obs_noise=False, domain_rand=True)

    o_priv, _ = private.reset(seed=5)
    o_shared, _ = shared.reset(seed=5)
    noise.reset(seed=5)
    np.testing.assert_allclose(o_priv, o_shared, atol=0, rtol=0)

    for _ in range(40):
        noise.step(ACTION)          # a sibling churning the shared model
        o_priv, r_priv, *_ = private.step(ACTION)
        o_shared, r_shared, *_ = shared.step(ACTION)
        np.testing.assert_allclose(o_priv, o_shared, atol=0, rtol=0)
        assert r_priv == r_shared


# ----------------------------------------------- domain randomization sharing


def test_each_shared_env_steps_under_its_own_domain_randomization(shared_pair):
    a, b = shared_pair
    a.reset(seed=1)
    b.reset(seed=2)
    trunk = a.trunk_body_id
    assert a._dr_body_mass[trunk] != b._dr_body_mass[trunk], (
        "different seeds must draw different trunk masses, or this test is blind"
    )
    # Whoever stepped last owns the model right now — that is exactly why the
    # env re-asserts before its own physics rather than trusting the model.
    for _ in range(5):
        a.step(ACTION)
        assert a.model.body_mass[trunk] == a._dr_body_mass[trunk]
        assert a.model.geom_friction[a.floor_geom, 0] == a._dr_geom_friction[a.floor_geom, 0]
        b.step(ACTION)
        assert b.model.body_mass[trunk] == b._dr_body_mass[trunk]
        assert b.model.geom_friction[b.floor_geom, 0] == b._dr_geom_friction[b.floor_geom, 0]


def test_domain_randomization_does_not_accumulate_under_sharing(shared_pair):
    """AGENTS.md's restore-then-apply, with a sibling scribbling in between."""
    a, b = shared_pair
    trunk = a.trunk_body_id
    base = a._default_body_mass[trunk]
    draws = []
    for i in range(6):
        a.reset(seed=1234)              # SAME seed every lap => same draw
        draws.append(float(a._dr_body_mass[trunk]))
        b.reset(seed=2000 + i)          # sibling randomizes the same model
        for _ in range(3):
            b.step(ACTION)
            a.step(ACTION)
    # Accumulation would drift the mass a little further every lap.
    assert len(set(draws)) == 1, f"DR accumulated across resets: {draws}"
    assert 0.9 * base <= draws[0] <= 1.1 * base
    # And the sibling's own draw never crept out of range either.
    assert 0.9 * base <= b._dr_body_mass[trunk] <= 1.1 * base


def test_shared_baseline_survives_a_sibling_randomizing_first():
    """An env built AFTER a sibling randomized must not adopt the randomized
    values as its compile-time baseline."""
    clear_shared_models()
    try:
        with shared_model_scope(exclusive=False):
            first = MicroduckWalkEnv(seed=3, domain_rand=True)
            pristine = first._default_body_mass.copy()
            first.reset(seed=3)
            assert first.model.body_mass[first.trunk_body_id] != pristine[first.trunk_body_id]
            second = MicroduckWalkEnv(seed=4, domain_rand=True)
        np.testing.assert_array_equal(second._default_body_mass, pristine)
    finally:
        clear_shared_models()


def test_bam_and_xml_envs_never_share_a_compiled_model():
    """BAM's setup PERMANENTLY neutralizes the MJCF position servos (it zeroes
    actuator_gainprm) — handing that model to an "xml" env would leave it with
    servos that produce no force at all, and a duck that never moves."""
    clear_shared_models()
    try:
        with shared_model_scope(exclusive=True):
            bam = MicroduckWalkEnv(seed=0, actuator="bam")
            xml = MicroduckWalkEnv(seed=0, actuator="xml")
        assert bam.model is not xml.model
        assert (xml.model.actuator_gainprm != 0).any(), "xml servos got neutered"
        xml.reset(seed=0)
        for _ in range(20):
            xml.step(ACTION)
        assert np.abs(xml.data.qvel).max() > 1e-3, "the duck never moved"
    finally:
        clear_shared_models()


def test_bam_actuator_refuses_an_in_process_shared_model():
    """BAM rewrites model.dof_frictionloss every physics substep — sharing the
    model with a sibling would silently retune the other env's servos."""
    with shared_model_scope(exclusive=False):
        with pytest.raises(ValueError, match="bam"):
            MicroduckWalkEnv(seed=0, actuator="bam")
    # The fork case (one env per process, copy-on-write) is fine.
    with shared_model_scope(exclusive=True):
        env = MicroduckWalkEnv(seed=0, actuator="bam")
    assert env.bam is not None


# ---------------------------------------------------------------- backends


def test_resolve_backend_precedence(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert resolve_backend() == DEFAULT_BACKEND
    monkeypatch.setenv(ENV_VAR, "dummy")
    assert resolve_backend() == "dummy"
    assert resolve_backend("thread") == "thread"    # explicit beats the env var
    with pytest.raises(ValueError, match="unknown vec-env backend"):
        resolve_backend("nonsense")


def test_default_backend_shares_the_model():
    assert DEFAULT_BACKEND == "fork"
    assert set(BACKENDS) == {"fork", "subproc", "dummy", "thread"}


def _rollout(backend: str, n: int = 2, steps: int = 6):
    from microduck_local.train import make_env
    fns = [make_env(i, 0, obs_noise=False, domain_rand=True) for i in range(n)]
    venv = make_vec_env(fns, backend=backend)
    try:
        obs = [venv.reset()]
        for _ in range(steps):
            o, r, d, _ = venv.step(np.tile(ACTION, (n, 1)))
            obs.append(o)
        return np.stack(obs)
    finally:
        venv.close()


@pytest.mark.parametrize("backend", ["fork", "dummy"])
def test_backend_matches_the_unshared_baseline(backend):
    """The shared-model backends must be bit-identical to a private compile."""
    np.testing.assert_array_equal(_rollout(backend), _rollout("subproc"))


def test_fork_backend_keeps_domain_randomization_per_worker():
    """Copy-on-write: a child's write to body_mass must not reach its siblings
    or the parent."""
    from microduck_local.train import make_env
    venv = make_vec_env([make_env(i, 0, domain_rand=True) for i in range(3)],
                        backend="fork")
    # The parent's copy — the one every child forked from — must stay pristine.
    parent = shared_model(C.SCENE_WALK_XML)
    before = parent.body_mass.copy()
    try:
        venv.reset()
        for _ in range(3):
            venv.step(np.tile(ACTION, (3, 1)))
        trunk = venv.get_attr("trunk_body_id")
        masses = [float(m[i]) for m, i in
                  zip(venv.get_attr("_dr_body_mass"), trunk)]
        assert len(set(masses)) == 3, f"workers must draw independently: {masses}"
        # That each worker's live model then carries its own draw is covered
        # in-process above; here the copy-on-write claim is the parent check
        # below.
    finally:
        venv.close()
    np.testing.assert_array_equal(parent.body_mass, before)


def test_fork_workers_inherit_the_model_instead_of_compiling_one():
    """The mechanism itself: fork copies the address space, so a worker that
    inherited the parent's model reports the parent's address, and a worker that
    compiled its own could not."""
    from microduck_local.train import make_env
    venv = make_vec_env([make_env(i, 0) for i in range(3)], backend="fork")
    try:
        parent = shared_model(C.SCENE_WALK_XML)
        assert venv.get_attr("model_id") == [id(parent)] * 3
    finally:
        venv.close()
    # ...and the same factories WITHOUT sharing land somewhere else entirely.
    venv = make_vec_env([make_env(i, 0) for i in range(3)], backend="subproc")
    try:
        assert id(parent) not in venv.get_attr("model_id")
    finally:
        venv.close()


def test_threaded_backend_refuses_domain_randomization():
    from microduck_local.train import make_env
    fns = [make_env(i, 0, domain_rand=True) for i in range(2)]
    with pytest.raises(ValueError, match="domain_rand"):
        make_vec_env(fns, backend="thread")


def test_threaded_backend_runs_and_shares_one_model():
    from microduck_local.train import make_env
    fns = [make_env(i, 0, domain_rand=False, obs_noise=False) for i in range(3)]
    venv = make_vec_env(fns, backend="thread")
    try:
        assert type(venv).__name__ == "ThreadedVecEnv"
        models = venv.get_attr("model")
        assert all(m is models[0] for m in models)
        venv.reset()
        for _ in range(6):
            obs, rew, done, _ = venv.step(np.tile(ACTION, (3, 1)))
        assert obs.shape == (3, C.OBS_DIM)
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
    finally:
        venv.close()


def test_env_var_selects_the_backend(monkeypatch):
    from microduck_local.train import make_env
    monkeypatch.setenv(ENV_VAR, "dummy")
    venv = make_vec_env([make_env(0, 0)])
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
        assert type(venv) is DummyVecEnv   # not the ThreadedVecEnv subclass
    finally:
        venv.close()


def test_train_modules_do_not_import_torch():
    """The trainers fork workers at import-of-make_vec_env time. If importing
    the CLI modules already pulled torch, the parent would be multi-threaded
    before the fork. Locked in a subprocess so this file's other tests (which
    do import torch via SB3) cannot contaminate the check."""
    import subprocess
    import sys
    script = (
        "import sys\n"
        "from microduck_local import train, train_behavior, vec_env\n"
        "torch_mods = [m for m in sys.modules "
        "if m == 'torch' or m.startswith('torch.')]\n"
        "assert not torch_mods, torch_mods\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_fork_backend_is_single_threaded_in_the_parent():
    """A clean interpreter must fork workers without the macOS fork-deadlock
    warning (OpenMP/Accelerate pools inherited into children)."""
    import subprocess
    import sys
    script = r"""
import sys, warnings
warnings.filterwarnings(
    "error",
    message=r".*use of fork\(\) may lead to deadlocks.*",
    category=DeprecationWarning,
)
from microduck_local.vec_env import make_vec_env
from microduck_local.walk_env import MicroduckWalkEnv
assert "torch" not in sys.modules
def fn(rank):
    def _init():
        return MicroduckWalkEnv(seed=rank, obs_noise=False, domain_rand=False)
    return _init
venv = make_vec_env([fn(i) for i in range(2)], backend="fork")
try:
    assert "torch" not in sys.modules
    venv.reset()
    import numpy as np
    venv.step(np.zeros((2, 14), np.float32))
finally:
    venv.close()
print("OK")
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_envs_per_worker_batching_matches_one_per_worker():
    """envs_per_worker=2 must be invisible to the caller: same obs/rew/done
    stream, same per-env attrs, same episode-end infos as the 1:1 layout."""

    from microduck_local.train_behavior import make_env
    from microduck_local.vec_env import ForkVecEnv, _SharedModelEnvFn

    def build(k):
        fns = [_SharedModelEnvFn(make_env("one_leg", i, 7), exclusive=True)
               for i in range(4)]
        return ForkVecEnv(fns, envs_per_worker=k)

    a, b = build(1), build(2)
    assert a.num_workers == 4 and b.num_workers == 2
    obs_a, obs_b = a.reset(), b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    rng = np.random.default_rng(0)
    for _ in range(40):
        act = rng.uniform(-0.3, 0.3, size=(4, 14)).astype(np.float32)
        oa, ra, da, ia = a.step(act)
        ob, rb, db, ib = b.step(act)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(da, db)
        assert [sorted(d) for d in ia] == [sorted(d) for d in ib]
    # per-env control routing: a write to one env index must land on exactly
    # that env, wherever it lives in the worker packing
    before = b.get_attr("domain_rand")
    b.set_attr("domain_rand", not before[2], indices=[2])
    after = b.get_attr("domain_rand")
    assert after[2] != before[2]
    assert after[:2] == before[:2] and after[3:] == before[3:]
    a.close()
    b.close()
