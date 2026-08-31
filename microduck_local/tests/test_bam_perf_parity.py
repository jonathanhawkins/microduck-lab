"""Bit-parity regression for the optimized BAM hot path.

bam_actuator.py (and contract.quat_rotate_inverse) were restructured for
per-substep CPU cost — hoisted lookups, `minimum/maximum` in place of the
`np.clip` wrapper, no `full_like`/`astype` churn, slice indexing, unrolled
crosses. Every one of those rewrites is supposed to be BIT-IDENTICAL to the
verbatim ports that were validated against the real `bam` package (max
|delta tau| = 0.0 on a 60-point grid), so this file pins:

1. the pure math (`compute_motor_torque`, `friction_budget`,
   `quat_rotate_inverse`) against verbatim copies of the pre-optimization
   code, over randomized grids of positions / velocities / targets /
   voltages — exact equality, far inside the required 1e-12;
2. the `DelayBuffer` against a verbatim reference, over randomized
   append/compute/reset sequences (same rng stream consumption);
3. a full 50-control-step BAM env rollout against a golden fingerprint
   captured from the pre-optimization implementation (exact float64 bits via
   float.hex), which covers the whole `before_step` pipeline: delay states,
   battery sag, the efc friction scan, and model field writes.

If an intentional model change ever invalidates the golden rollout, recapture
it with the OLD code — never by pasting in whatever the new code produces.
"""

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.bam_actuator import BamXL330Actuator, DelayBuffer
from microduck_local.walk_env import MicroduckWalkEnv


@pytest.fixture(autouse=True)
def _no_actuator_env_override(monkeypatch):
    monkeypatch.delenv("MICRODUCK_ACTUATOR", raising=False)


@pytest.fixture(scope="module")
def calc():
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    d = mujoco.MjData(m)
    return BamXL330Actuator(m, d, C.JOINT_NAMES, dt=C.PHYSICS_DT,
                            rng=np.random.default_rng(0), vin_range=None,
                            vin_drop_gain_range=None, delay_max_lag=0)


# ------------------------------------------------ reference implementations
# Verbatim copies of the pre-optimization code paths (do not "modernize").


def ref_motor_torque(a, q, dq, q_target, vin):
    kt, R = a.p["kt"], a.p["R"]
    duty = (q_target - q) * a.kp_fw * a.error_gain
    if a.max_current is not None:
        duty_center = (kt * dq) / vin
        duty_span = (R * a.max_current) / vin
        duty = np.clip(duty, duty_center - duty_span, duty_center + duty_span)
    duty = np.clip(duty, -a.max_pwm, a.max_pwm)
    volts = vin * duty
    return kt * volts / R - (kt**2) * dq / R


def ref_friction_budget(a, motor_torque, external_torque, dq):
    p = a.p
    stribeck = np.exp(-np.power(np.abs(dq) / p["dtheta_stribeck"], p["alpha"]))
    fl = np.full_like(np.asarray(motor_torque, dtype=np.float64),
                      p["friction_base"])
    fl = fl + stribeck * p["friction_stribeck"]
    gearbox = np.abs(external_torque * p["load_friction_external"]
                     - motor_torque * p["load_friction_motor"])
    fl = fl + gearbox
    gearbox_strib = np.abs(
        external_torque * p["load_friction_external_stribeck"]
        - motor_torque * p["load_friction_motor_stribeck"]
    )
    fl = fl + stribeck * gearbox_strib
    abs_ext = np.abs(external_torque)
    abs_mot = np.abs(motor_torque)
    drive = (abs_mot > abs_ext).astype(np.float64)
    quad = (drive * p["load_friction_external_quad"] * abs_ext**2
            + (1.0 - drive) * p["load_friction_motor_quad"] * abs_mot**2)
    fl = fl + stribeck * quad
    return fl * a.friction_scale


def ref_quat_rotate_inverse(quat_wxyz, vec):
    w = quat_wxyz[0]
    xyz = quat_wxyz[1:4]
    t = np.cross(xyz, vec) * 2
    return (vec - w * t + np.cross(xyz, t)).astype(np.float32)


class RefDelayBuffer:
    """Verbatim pre-optimization DelayBuffer."""

    def __init__(self, min_lag, max_lag, width, rng):
        self.min_lag = int(min_lag)
        self.max_lag = int(max_lag)
        self._rng = rng
        self._size = self.max_lag + 1
        self._buf = np.zeros((self._size, width), dtype=np.float64)
        self._head = 0
        self._len = 0
        self.current_lag = 0

    def reset(self):
        self._len = 0
        self._head = 0
        self.current_lag = 0

    def append(self, value):
        self._head = (self._head + 1) % self._size
        self._buf[self._head] = value
        self._len = min(self._len + 1, self._size)

    def compute(self):
        lag = int(self._rng.integers(self.min_lag, self.max_lag + 1))
        self.current_lag = lag
        lag = min(lag, self._len - 1)
        return self._buf[(self._head - lag) % self._size]


# ----------------------------------------------------------- randomized grids


def test_motor_torque_matches_reference_bitwise(calc):
    rng = np.random.default_rng(2024)
    for _ in range(20):
        n = int(rng.integers(1, 400))
        q = rng.uniform(-2.5, 2.5, n)
        dq = rng.uniform(-40.0, 40.0, n)
        tgt = q + rng.uniform(-3.0, 3.0, n)
        # scalar vin (hot path) and per-element vin arrays both hold
        vin = (float(rng.uniform(6.0, 8.4)) if rng.random() < 0.5
               else rng.uniform(6.0, 8.4, n))
        np.testing.assert_array_equal(
            calc.compute_motor_torque(q, dq, tgt, vin),
            ref_motor_torque(calc, q, dq, tgt, vin))


def test_motor_torque_hot_shape_matches_reference_bitwise(calc):
    rng = np.random.default_rng(7)
    for _ in range(200):
        q = rng.uniform(-2.5, 2.5, 14)
        dq = rng.uniform(-40.0, 40.0, 14)
        tgt = q + rng.uniform(-3.0, 3.0, 14)
        vin = float(rng.uniform(6.0, 8.4))
        np.testing.assert_array_equal(
            calc.compute_motor_torque(q, dq, tgt, vin),
            ref_motor_torque(calc, q, dq, tgt, vin))


def test_friction_budget_matches_reference_bitwise(calc):
    rng = np.random.default_rng(99)
    for _ in range(50):
        n = int(rng.integers(1, 400))
        motor = rng.uniform(-1.1, 1.1, n)
        ext = rng.uniform(-1.5, 1.5, n)
        dq = rng.uniform(-30.0, 30.0, n)
        # Sprinkle the near-zero velocities where Stribeck actually bites,
        # and exact motor/ext ties for the quad-term branch point.
        dq[:: 7] *= 1e-3
        motor[:: 11] = ext[:: 11]
        calc.friction_scale = float(rng.uniform(0.9, 1.1))
        try:
            np.testing.assert_array_equal(
                calc.friction_budget(motor, ext, dq),
                ref_friction_budget(calc, motor, ext, dq))
        finally:
            calc.friction_scale = 1.0


def test_quat_rotate_inverse_matches_np_cross_reference_bitwise():
    rng = np.random.default_rng(31337)
    for _ in range(500):
        quat = rng.standard_normal(4)
        quat /= np.linalg.norm(quat)
        vec = (np.array([0.0, 0.0, -1.0]) if rng.random() < 0.3
               else rng.standard_normal(3))
        np.testing.assert_array_equal(C.quat_rotate_inverse(quat, vec),
                                      ref_quat_rotate_inverse(quat, vec))


def test_delay_buffer_matches_reference_stream():
    """Same rng stream, randomized append/compute/reset sequences, and the
    startup phase where history is shorter than the drawn lag."""
    seq_rng = np.random.default_rng(5)
    for trial in range(10):
        opt = DelayBuffer(3, 6, 14, np.random.default_rng(trial))
        ref = RefDelayBuffer(3, 6, 14, np.random.default_rng(trial))
        for i in range(300):
            v = seq_rng.standard_normal(14)
            opt.append(v)
            ref.append(v)
            np.testing.assert_array_equal(opt.compute(), ref.compute())
            assert opt.current_lag == ref.current_lag
            if seq_rng.random() < 0.02:
                opt.reset()
                ref.reset()


# -------------------------------------------------------- golden env rollout

# Captured from the pre-optimization implementation (commit state with the
# verbatim `bam` port): 50 control steps of the exact rollout below. float.hex
# so the comparison is to the last bit of the float64 state.
GOLDEN_QPOS = [
    "0x1.ad995106063adp-4", "-0x1.7450055b4da66p-3", "-0x1.a7ead23a7f1d9p-4",
    "-0x1.a664d26eed96bp-4", "0x1.7905303229316p-1", "0x1.56524b31eba98p-1",
    "-0x1.5285e732aafaep-7", "-0x1.38392fb21ccc1p-3", "-0x1.83c4d673ee10ap-4",
    "-0x1.19632277d684fp-1", "-0x1.e5a0f6facf573p-3", "0x1.aa17a8f7c5d25p-3",
    "0x1.1b584339c7650p-4", "-0x1.f8a31a9d338dap-5", "-0x1.7f707d7d1e513p-4",
    "0x1.4aa2fe6758b88p-4", "0x1.5675afb6a27e0p-7", "-0x1.9f0232b9eafd4p-8",
    "0x1.81026f04a2f13p-1", "0x1.3a1ee68094e31p-3", "-0x1.35a5304b4b340p-1",
]
GOLDEN_QVEL = [
    "-0x1.8f31a4bdb7334p-2", "0x1.98f0fbc8e5d6cp-3", "0x1.e8e52980eeadcp-3",
    "0x1.0668fb9541ed8p+1", "0x1.712ef56370828p+1", "0x1.02a61a5c37cfep+1",
    "0x1.5d2f41bc5c484p+0", "0x1.a387e369da996p+0", "-0x1.ec1f121dce629p+1",
    "-0x1.67272d6e03564p+1", "-0x1.2bd377d85f729p-1", "-0x1.67145de449559p-3",
    "-0x1.368976a14b30bp-1", "-0x1.46d5c0dff7c56p+0", "-0x1.489a05e61546cp+2",
    "0x1.27dd389488ff4p-1", "0x1.2df3e1533333ap-1", "0x1.aaf3a94fac540p+1",
    "-0x1.296d48f704bebp-2", "0x1.3838764cc3ea7p+2",
]
GOLDEN_TAU_LAST = [
    "0x1.c7328769accb2p-6", "0x1.8534eb8b15106p-2", "-0x1.8c97da86078c0p-4",
    "-0x1.a3740a54d6254p-6", "0x1.d85191dcb0200p-3", "0x1.6f61eb649321cp-3",
    "0x1.44b9895d6db8fp-4", "-0x1.6d22d6fc0dfb6p-4", "-0x1.d28567eea9fdap-3",
    "-0x1.3534d1f289e07p-5", "0x1.7b7a2f55dd119p-2", "0x1.0c81da12a981ep-4",
    "-0x1.40ee34eb9a59ep-3", "0x1.67a89166889f7p-2",
]
GOLDEN_FL_LAST = [
    "0x1.a9a484362a46ep-6", "0x1.00c765c151daap-3", "0x1.01b5968431030p-5",
    "0x1.a986d7dcc9b13p-5", "0x1.ca546bd4da602p-4", "0x1.031cb0acb94b1p-4",
    "0x1.20ce61122b471p-5", "0x1.011e0174ab2d7p-5", "0x1.189ce267e9bffp-4",
    "0x1.4a10b0f334466p-6", "0x1.cd679998a15fdp-4", "0x1.7682256049020p-6",
    "0x1.ab91c6fa84a05p-5", "0x1.9c9dde3e27a1ep-4",
]
GOLDEN_TAU_SUM = "-0x1.be264284807e6p-1"
GOLDEN_VIN_LAST = "0x1.a11962b496cf4p+2"


def _hex(vals):
    return np.array([float.fromhex(h) for h in vals], dtype=np.float64)


def test_full_bam_rollout_matches_pre_optimization_golden_bits():
    env = MicroduckWalkEnv(actuator="bam", obs_noise=True, domain_rand=True,
                           action_delay=True, seed=3, terminate_on_fall=False)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    tau_sum = 0.0
    for _ in range(50):
        env.step(rng.uniform(-1.0, 1.0, C.NUM_JOINTS).astype(np.float32))
        tau_sum += float(np.sum(env.bam.applied_torque))

    np.testing.assert_array_equal(env.data.qpos, _hex(GOLDEN_QPOS))
    np.testing.assert_array_equal(env.data.qvel, _hex(GOLDEN_QVEL))
    np.testing.assert_array_equal(env.bam.applied_torque, _hex(GOLDEN_TAU_LAST))
    np.testing.assert_array_equal(env.bam.last_frictionloss, _hex(GOLDEN_FL_LAST))
    assert env.bam.last_vin == float.fromhex(GOLDEN_VIN_LAST)


def test_full_bam_rollout_torque_sum_matches_golden():
    """Every applied torque over the rollout, not just the final state.

    The sum was accumulated as sum-per-step-then-add in capture order, so it
    is reproducible exactly (same additions in the same order)."""
    env = MicroduckWalkEnv(actuator="bam", obs_noise=True, domain_rand=True,
                           action_delay=True, seed=3, terminate_on_fall=False)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    taus = []
    for _ in range(50):
        env.step(rng.uniform(-1.0, 1.0, C.NUM_JOINTS).astype(np.float32))
        taus.append(env.bam.applied_torque.copy())
    total = float(np.sum(np.asarray(taus)))
    assert total == float.fromhex(GOLDEN_TAU_SUM)
