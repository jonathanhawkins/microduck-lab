"""BAM actuator tests: parameter provenance, the torque/friction model, the
command-delay buffer, and the two invariants that matter for the rest of the
harness — the 61-obs/14-action contract still holds under BAM, and selecting
`actuator="xml"` leaves the old physics bit-for-bit untouched."""

import mujoco
import numpy as np
import pytest

from microduck_local import bam_actuator as BA
from microduck_local import contract as C
from microduck_local.bam_actuator import BamXL330Actuator, DelayBuffer
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(autouse=True)
def _no_actuator_env_override(monkeypatch):
    """MICRODUCK_ACTUATOR hard-overrides the constructor, so a shell that has it
    set would otherwise silently retarget every test in this file."""
    monkeypatch.delenv("MICRODUCK_ACTUATOR", raising=False)


@pytest.fixture(scope="module")
def calc():
    """A BAM actuator bound to a throwaway model, used as a pure calculator."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    d = mujoco.MjData(m)
    return BamXL330Actuator(m, d, C.JOINT_NAMES, dt=C.PHYSICS_DT,
                            rng=np.random.default_rng(0), vin_range=None,
                            vin_drop_gain_range=None, delay_max_lag=0)


# ------------------------------------------------------------------ provenance


def test_hardcoded_params_match_the_bam_package_when_installed():
    """The fallback table must not drift from bam/params/xl330/m6.json."""
    params, source = BA.load_bam_params()
    if source != "bam package":
        pytest.skip("better-actuator-models not importable in this venv")
    for k, v in BA.XL330_M6_PARAMS.items():
        assert params[k] == pytest.approx(v, rel=1e-12), k


def test_bam_linearization_reproduces_the_mjcf_actuator_class(calc):
    """The MJCF `chosen_actuator` class IS this BAM model pushed through
    VoltageControlledActuator.to_mujoco() at vin=7.5 — which is exactly why
    swapping in the full model is a fidelity change, not a retune."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    dof = m.joint("left_knee").dofadr[0]
    p, kt, R = calc.p, calc.p["kt"], calc.p["R"]

    kp = calc.error_gain * calc.kp_fw * 7.5 * calc.max_pwm * kt / R
    assert kp == pytest.approx(float(m.actuator_gainprm[0][0]), abs=0.02)      # 0.55
    damping = p["friction_viscous"] + kt**2 / R
    assert damping == pytest.approx(float(m.dof_damping[dof]), abs=0.002)      # 0.053
    assert 7.5 * kt / R == pytest.approx(float(m.actuator_forcerange[0][1]), abs=0.02)
    assert p["friction_base"] == pytest.approx(float(m.dof_frictionloss[dof]), abs=1e-4)
    assert p["armature"] == pytest.approx(float(m.dof_armature[dof]), abs=1e-4)


# ------------------------------------------------------------- torque envelope


def test_motor_torque_is_finite_and_bounded_by_the_voltage_ceiling(calc):
    """Hard envelope for ANY input: the vin*kt/R force limit (BamActuator sets
    exactly this as the MuJoCo forcerange)."""
    rng = np.random.default_rng(0)
    err = rng.uniform(-6.0, 6.0, 4000)
    dq = rng.uniform(-40.0, 40.0, 4000)
    vin = rng.uniform(*BA.DEFAULT_VIN_RANGE, 4000)
    tau = np.clip(calc.compute_motor_torque(np.zeros_like(err), dq, err, vin),
                  -calc.force_limit, calc.force_limit)

    assert np.isfinite(tau).all()
    assert np.abs(tau).max() <= calc.force_limit + 1e-9
    # BamActuator.edit_spec sets forcerange = max(vin_range)*kt/R; this
    # calculator instance is pinned to the nominal 7.5 V.
    assert calc.force_limit == pytest.approx(7.5 * calc.p["kt"] / calc.p["R"], rel=1e-12)
    env_limit = max(BA.DEFAULT_VIN_RANGE) * calc.p["kt"] / calc.p["R"]
    assert env_limit == pytest.approx(1.0676, abs=1e-3)


def test_firmware_current_limit_caps_torque_in_the_operating_range(calc):
    """The firmware bounds the DUTY CYCLE, so |tau| <= kt*I_max holds only while
    the current window still fits inside +-max_pwm, i.e. up to
    |dq| <= (vin - R*I_max)/kt. Past that the battery cannot supply the voltage
    the limiter asks for and the limit is not reached — upstream's documented
    behaviour, and the reason this is modelled in compute_control rather than as
    a torque clamp."""
    kt, R = calc.p["kt"], calc.p["R"]
    ceiling = kt * calc.max_current
    assert ceiling == pytest.approx(0.6405, abs=1e-3)   # vs the XML's +-0.96 Nm

    rng = np.random.default_rng(0)
    vin = 7.5
    dq_max = (vin - R * calc.max_current) / kt
    assert dq_max == pytest.approx(7.05, abs=0.05)
    err = rng.uniform(-6.0, 6.0, 4000)
    dq = rng.uniform(-dq_max, dq_max, 4000)
    tau = calc.compute_motor_torque(np.zeros_like(err), dq, err, vin)
    assert np.abs(tau).max() <= ceiling + 1e-9
    assert np.abs(tau).max() == pytest.approx(ceiling, rel=1e-6)   # and it is reached


def test_torque_sign_and_deadzone(calc):
    z = np.zeros(1)
    assert calc.compute_motor_torque(z, z, z, 7.5)[0] == pytest.approx(0.0)
    assert calc.compute_motor_torque(z, z, np.array([0.3]), 7.5)[0] > 0
    assert calc.compute_motor_torque(z, z, np.array([-0.3]), 7.5)[0] < 0
    # Back-EMF: same command, faster forward motion -> strictly less torque.
    fast = calc.compute_motor_torque(z, np.array([8.0]), np.array([0.3]), 7.5)[0]
    slow = calc.compute_motor_torque(z, np.array([0.0]), np.array([0.3]), 7.5)[0]
    assert fast < slow


def test_lower_supply_voltage_yields_less_torque(calc):
    z, err = np.zeros(1), np.array([2.0])
    lo = calc.compute_motor_torque(z, z, err, BA.DEFAULT_VIN_MIN)[0]
    hi = calc.compute_motor_torque(z, z, err, 8.2)[0]
    assert 0 < lo <= hi


def test_friction_budget_is_positive_finite_and_load_dependent(calc):
    rng = np.random.default_rng(1)
    tau = rng.uniform(-0.65, 0.65, 2000)
    ext = rng.uniform(-1.0, 1.0, 2000)
    dq = rng.uniform(-30.0, 30.0, 2000)
    fl = calc.friction_budget(tau, ext, dq)

    assert np.isfinite(fl).all()
    assert (fl >= calc.p["friction_base"]).all()   # base Coulomb is a floor
    # Load dependence is the term the XML's constant 0.0048 Nm drops entirely:
    # ~27% of the motor torque comes back as friction.
    z = np.zeros(1)
    loaded = calc.friction_budget(np.array([0.6]), z, np.array([10.0]))[0]
    unloaded = calc.friction_budget(z, z, np.array([10.0]))[0]
    assert loaded - unloaded == pytest.approx(0.6 * calc.p["load_friction_motor"], rel=1e-6)


def test_stribeck_only_bites_near_zero_velocity(calc):
    z = np.zeros(1)
    slow = calc.friction_budget(z, z, np.array([0.0]))[0]
    fast = calc.friction_budget(z, z, np.array([20.0]))[0]
    assert slow > fast
    assert fast == pytest.approx(calc.p["friction_base"], abs=1e-9)


def test_friction_scale_multiplies_the_whole_budget(calc):
    z = np.zeros(1)
    base = calc.friction_budget(np.array([0.3]), z, np.array([1.0]))[0]
    calc.friction_scale = 1.1
    try:
        assert calc.friction_budget(np.array([0.3]), z, np.array([1.0]))[0] == \
            pytest.approx(1.1 * base)
    finally:
        calc.friction_scale = 1.0


# ---------------------------------------------------------------- delay buffer


def test_delay_buffer_serves_the_value_from_lag_steps_ago():
    rng = np.random.default_rng(0)
    buf = DelayBuffer(2, 2, 1, rng)          # deterministic lag = 2
    for i in range(6):
        buf.append(np.array([float(i)]))
        got = float(buf.compute()[0])
        assert got == float(max(i - 2, 0))   # clamped while history is short
        assert buf.current_lag == 2


def test_delay_buffer_lag_stays_in_range_and_varies():
    rng = np.random.default_rng(0)
    buf = DelayBuffer(3, 6, 1, rng)
    lags = set()
    for i in range(400):
        buf.append(np.array([float(i)]))
        buf.compute()
        assert 3 <= buf.current_lag <= 6
        lags.add(buf.current_lag)
    assert lags == {3, 4, 5, 6}              # every lag is actually reachable


def test_delay_buffer_reset_drops_history():
    rng = np.random.default_rng(0)
    buf = DelayBuffer(3, 3, 2, rng)
    for i in range(10):
        buf.append(np.full(2, float(i)))
        buf.compute()
    buf.reset()
    buf.append(np.full(2, 99.0))
    np.testing.assert_array_equal(buf.compute(), np.full(2, 99.0))


def test_delay_buffer_rejects_bad_bounds():
    with pytest.raises(ValueError):
        DelayBuffer(5, 2, 1, np.random.default_rng(0))


# ----------------------------------------------------------- env, BAM enabled


@pytest.fixture(scope="module")
def bam_env():
    return MicroduckWalkEnv(actuator="bam", obs_noise=False, domain_rand=False,
                            action_delay=False, random_yaw=False, seed=0)


def test_bam_env_keeps_the_61_obs_14_action_contract(bam_env):
    obs, _ = bam_env.reset(seed=0)
    assert obs.shape == (C.OBS_DIM,) and obs.dtype == np.float32
    assert bam_env.action_space.shape == (C.NUM_JOINTS,)
    action = np.linspace(-0.2, 0.2, C.NUM_JOINTS).astype(np.float32)
    obs, reward, term, trunc, _ = bam_env.step(action)
    assert obs.shape == (C.OBS_DIM,) and np.isfinite(obs).all()
    assert np.isfinite(reward)
    np.testing.assert_array_equal(obs[34:48], action)          # last_action slot
    # ctrl keeps its documented meaning even though it drives nothing.
    np.testing.assert_allclose(bam_env.data.ctrl, C.DEFAULT_POSE + action, atol=1e-6)


def test_bam_env_applies_torque_only_on_the_actuated_dofs(bam_env):
    bam_env.reset(seed=0)
    bam_env.step(np.full(C.NUM_JOINTS, 0.3, np.float32))
    qfrc = bam_env.data.qfrc_applied
    assert np.abs(qfrc[bam_env.joint_qvel_adr]).max() > 0.0
    free_dofs = np.setdiff1d(np.arange(bam_env.model.nv), bam_env.joint_qvel_adr)
    np.testing.assert_array_equal(qfrc[free_dofs], np.zeros(free_dofs.size))


def test_bam_torque_does_not_clobber_a_foreign_qfrc_applied_slot(bam_env):
    """behaviors.py's demo spotter writes qfrc_applied[4] (a free-joint DOF)
    once per control step and expects it to survive the substeps."""
    bam_env.reset(seed=0)
    bam_env.data.qfrc_applied[4] = -0.6
    bam_env.step(np.full(C.NUM_JOINTS, 0.3, np.float32))
    assert bam_env.data.qfrc_applied[4] == -0.6


def test_bam_env_rollout_stays_finite_and_within_the_torque_envelope():
    env = MicroduckWalkEnv(actuator="bam", obs_noise=True, domain_rand=True,
                           action_delay=True, seed=3, terminate_on_fall=False)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(3)
    ceiling = env.bam.force_limit
    for _ in range(120):
        obs, reward, term, trunc, _ = env.step(
            rng.uniform(-1.0, 1.0, C.NUM_JOINTS).astype(np.float32)
        )
        assert np.isfinite(obs).all() and np.isfinite(reward)
        tau = env.bam.applied_torque
        assert np.isfinite(tau).all()
        assert np.abs(tau).max() <= ceiling + 1e-9
        assert np.isfinite(env.bam.last_frictionloss).all()
        assert 3 <= env.bam.current_lag <= 6
        assert BA.DEFAULT_VIN_MIN - 1e-9 <= env.bam.last_vin <= max(BA.DEFAULT_VIN_RANGE)


def test_bam_publishes_its_torque_as_actuator_force(bam_env):
    """Reward terms (walk_env's torque penalty, behaviors.py's torque/stall
    terms) read data.actuator_force; under BAM it must carry the BAM torque,
    not the neutralized MJCF servo's zero."""
    bam_env.reset(seed=0)
    bam_env.step(np.full(C.NUM_JOINTS, 0.5, np.float32))
    np.testing.assert_allclose(bam_env.data.actuator_force,
                               bam_env.bam.applied_torque, atol=1e-12)
    assert np.abs(bam_env.data.actuator_force).max() > 0.0


def test_bam_neutralizes_the_mjcf_position_servos(bam_env):
    assert np.all(bam_env.model.actuator_gainprm[bam_env.bam.actuator_ids] == 0.0)
    assert np.all(bam_env.model.actuator_biasprm[bam_env.bam.actuator_ids] == 0.0)
    dofs = bam_env.bam.dof_adr
    np.testing.assert_allclose(bam_env.model.dof_armature[dofs],
                               bam_env.bam.p["armature"])


def test_bam_env_is_deterministic_under_a_seed():
    def rollout():
        env = MicroduckWalkEnv(actuator="bam", obs_noise=True, domain_rand=True,
                               action_delay=True, seed=7, terminate_on_fall=False)
        env.reset(seed=7)
        rng = np.random.default_rng(11)
        out = []
        for _ in range(25):
            obs, *_ = env.step(rng.uniform(-0.5, 0.5, C.NUM_JOINTS).astype(np.float32))
            out.append(obs)
        return np.array(out)

    np.testing.assert_array_equal(rollout(), rollout())


def test_bam_friction_dr_is_bounded_and_non_accumulating():
    env = MicroduckWalkEnv(actuator="bam", domain_rand=True, seed=0)
    lo, hi = BA.DEFAULT_FRICTION_SCALE_RANGE
    seen = set()
    for s in range(12):
        env.reset(seed=s)
        assert lo <= env.bam.friction_scale <= hi
        seen.add(round(env.bam.friction_scale, 9))
    assert len(seen) > 1                       # it really is randomized
    off = MicroduckWalkEnv(actuator="bam", domain_rand=False, seed=0)
    off.reset(seed=0)
    assert off.bam.friction_scale == 1.0


def test_env_var_overrides_the_constructor(monkeypatch):
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "bam")
    assert MicroduckWalkEnv(seed=0).actuator_model == "bam"
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "xml")
    assert MicroduckWalkEnv(actuator="bam", seed=0).bam is None


def test_unknown_actuator_is_rejected():
    with pytest.raises(ValueError, match="xml"):
        MicroduckWalkEnv(actuator="servo", seed=0)


# ------------------------------------------------- xml path is untouched


def test_xml_is_the_default_and_builds_no_actuator():
    env = MicroduckWalkEnv(seed=0)
    assert env.actuator_model == "xml"
    assert env.bam is None


def test_xml_path_leaves_every_model_field_bam_would_touch_at_its_mjcf_value():
    env = MicroduckWalkEnv(actuator="xml", seed=0)
    ref = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    for field in ("dof_damping", "dof_frictionloss", "dof_armature",
                  "dof_solref", "dof_solimp", "actuator_gainprm",
                  "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(env.model, field),
                                      getattr(ref, field), err_msg=field)


def test_xml_rollout_is_bit_identical_to_stock_mujoco_position_control():
    """The strongest form of "nothing changed": drive a freshly compiled model
    by hand with exactly the XML servo semantics and require identical qpos."""
    env = MicroduckWalkEnv(actuator="xml", obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)

    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    m.opt.timestep = C.PHYSICS_DT
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos
    d.qvel[:] = env.data.qvel
    d.ctrl[:] = env.data.ctrl
    mujoco.mj_forward(m, d)

    rng = np.random.default_rng(5)
    for _ in range(30):
        action = rng.uniform(-0.3, 0.3, C.NUM_JOINTS).astype(np.float32)
        env.step(action)
        d.ctrl[:] = C.DEFAULT_POSE + action
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
        np.testing.assert_array_equal(env.data.qpos, d.qpos)
        np.testing.assert_array_equal(env.data.qvel, d.qvel)


def test_xml_path_never_writes_qfrc_applied():
    env = MicroduckWalkEnv(actuator="xml", seed=0)
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.full(C.NUM_JOINTS, 0.4, np.float32))
        assert float(np.max(np.abs(env.data.qfrc_applied))) == 0.0


def test_vin_pin_knob(monkeypatch):
    """MICRODUCK_BAM_VIN pins the per-robot supply voltage — 'train at the
    battery's limit'. No-load speed scales with vin, so 8.2 V buys ~9% over
    the 7.5 V nominal; sag and the 6.0 V floor stay active (a battery choice,
    not a physics cheat). Malformed values fall back to the sampled draw."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    d = mujoco.MjData(m)

    def mk(seed=0):
        return BamXL330Actuator(m, d, C.JOINT_NAMES, dt=C.PHYSICS_DT,
                                rng=np.random.default_rng(seed))

    monkeypatch.setenv("MICRODUCK_BAM_VIN", "8.2")
    a = mk()
    assert a.vin_nominal == pytest.approx(8.2)
    assert a.vin_drop_gain is not None  # sag DR untouched

    monkeypatch.setenv("MICRODUCK_BAM_VIN", "99")   # clamped, not trusted
    assert mk().vin_nominal == pytest.approx(8.4)

    monkeypatch.setenv("MICRODUCK_BAM_VIN", "abc")  # malformed -> sampled
    assert 6.5 <= mk().vin_nominal <= 8.2

    monkeypatch.delenv("MICRODUCK_BAM_VIN")
    assert 6.5 <= mk().vin_nominal <= 8.2
