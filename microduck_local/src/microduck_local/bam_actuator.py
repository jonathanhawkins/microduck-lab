"""BAM (Better Actuator Models) XL330 "m6" actuator — NumPy/CPU port.

The official training stack (microduck_rl / mjlab) does NOT use the MJCF's
`position` actuators. `microduck_constants.py` comments that block out ("Old
actuator") and drives every servo through
`FrictionDRBamActuatorCfg(motor_name="xl330", model="m6", kp_fw=200.0, ...)`.
This module reproduces that actuator on plain CPU MuJoCo so local rollouts see
the same physics.

What BAM does that the XML position servo does not
--------------------------------------------------
The XML class `chosen_actuator` (kp=0.55, forcerange ±0.96, joint damping
0.053, frictionloss 0.0048, armature 0.0018) is *literally* this same BAM model
pushed through `VoltageControlledActuator.to_mujoco()` at vin=7.5 V — every one
of those five numbers reproduces to two digits (see `test_bam_actuator.py`).
It is the small-signal linearization, and it drops four things:

1. **A firmware current limit.** The XL330 clamps at 1.75 A, applied by BAM as a
   duty-cycle window, so |tau| <= kt*I_max = 0.640 Nm. The XML servo's ±0.96 Nm
   forcerange is the *voltage* ceiling (vin*kt/R) with no current limit at all.
2. **Back-EMF is inside the motor, not a passive damper.** MuJoCo applies
   `kp*(target-q)` and the joint damping `-b*qvel` independently, so when the
   joint moves *against* the command the damper ADDS to the actuator force
   (up to ~1.5 Nm). In a real motor the same speed eats voltage headroom, so
   torque can only fall.
3. **Load-dependent gearbox friction.** m6's `load_friction_motor` = 0.267
   charges ~27% of the motor torque back as Coulomb friction, plus Stribeck and
   quadratic terms near zero velocity. The XML has one constant 0.0048 Nm.
4. **Supply-voltage reality.** Per-robot battery voltage (6.5-8.2 V), load-
   dependent sag (V_drop = gain * sum|tau|, gain 0-0.2 V/Nm, floored at 6.0 V),
   and a 3-6 physics-step command lag on the bus.

Parameter provenance
--------------------
Every constant below is either loaded from an importable `bam` package or
hard-coded from these upstream sources (each value carries its citation):
  - `bam/params/xl330/m6.json`           (better-actuator-models 1.0.1)
  - `bam/dynamixel/actuator.py`          (XL330Actuator motor constants)
  - `bam/mjlab.py`                       (BamActuator: pipeline + stiff friction)
  - `bam/model.py`                       (m6 friction-budget formulation)
  - `microduck_rl/.../microduck_constants.py`  (`_BAM_ACTUATOR_KWARGS`)

`better-actuator-models` itself only needs numpy + colorama, so adding it as a
real dependency would be clean; it is kept optional here so this harness stays
self-contained (and so the live venv needn't be re-synced). When it *is*
importable, `load_bam_params()` reads the JSON from the installed package and
the hard-coded table is used only as a fallback — a test asserts they agree.
"""

from __future__ import annotations

import os

import numpy as np

try:  # pragma: no cover - exercised only where the package is installed
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

try:  # optional: fuses the per-substep arithmetic into two compiled kernels
    from numba import njit as _njit
except ImportError:  # pragma: no cover - numpy fallback keeps behavior
    _njit = None


# ------------------------------------------------------------- fused kernels
#
# BAM runs EVERY physics substep, and its cost is ~48 numpy ops on 14-element
# arrays — pure dispatch overhead (~0.7 us/op), 0.14 ms per control step,
# more than mj_step itself. The two kernels below fuse the arithmetic-only
# parts into single compiled loops. They are BITWISE ports, held to the
# golden float64 rollouts in test_bam_perf_parity.py; the rules that make
# that possible:
#
# * Only IEEE-exact ops move into a kernel (+ - * / abs min max compares).
#   np.exp / np.power (the Stribeck factor) STAY in numpy: their vectorized
#   implementations are not bit-reproducible by another libm, so the kernel
#   takes `stribeck` precomputed.
# * Operation ORDER inside each element is the numpy expression's order,
#   including which operands are scalars (kt**2 is folded to a scalar by
#   Python in the numpy path, so it arrives here folded too).
# * numba's default strict FP (no fastmath) keeps each op IEEE-exact.

if _njit is not None:

    @_njit(cache=True)
    def _motor_torque_kernel(q, dq, q_target, vin, kt, kt2, R, kp_fw,
                             error_gain, has_imax, max_current, max_pwm, out):
        for i in range(q.shape[0]):
            duty = (q_target[i] - q[i]) * kp_fw * error_gain
            if has_imax:
                duty_center = (kt * dq[i]) / vin
                duty_span = (R * max_current) / vin
                lo = duty_center - duty_span
                hi = duty_center + duty_span
                duty = min(max(duty, lo), hi)
            duty = min(max(duty, -max_pwm), max_pwm)
            volts = vin * duty
            out[i] = kt * volts / R - kt2 * dq[i] / R

    @_njit(cache=True)
    def _friction_budget_kernel(motor_torque, external_torque, stribeck,
                                fbase, fstrib, lfe, lfm, lfes, lfms, lfeq,
                                lfmq, friction_scale, out):
        for i in range(motor_torque.shape[0]):
            s = stribeck[i]
            mot = motor_torque[i]
            ext = external_torque[i]
            fl = fbase + s * fstrib
            fl = fl + abs(ext * lfe - mot * lfm)
            fl = fl + s * abs(ext * lfes - mot * lfms)
            abs_ext = abs(ext)
            abs_mot = abs(mot)
            if abs_mot > abs_ext:
                quad = lfeq * (abs_ext * abs_ext)
            else:
                quad = lfmq * (abs_mot * abs_mot)
            fl = fl + s * quad
            out[i] = fl * friction_scale
    @_njit(cache=True)
    def _dof_friction_kernel(efc_type, efc_id, efc_force, n, fric_type,
                             dof_slot, out):
        # Accumulates in efc row order — the same order np.bincount visits
        # its input, so the float64 sums are bit-identical to the numpy path.
        for k in range(n):
            if efc_type[k] == fric_type:
                s = dof_slot[efc_id[k]]
                if s >= 0:
                    out[s] += efc_force[k]
else:  # pragma: no cover
    _motor_torque_kernel = None
    _friction_budget_kernel = None
    _dof_friction_kernel = None


# --------------------------------------------------------------------- params

# bam/params/xl330/m6.json, verbatim (better-actuator-models 1.0.1). `q_offset`
# is deliberately absent: it is a testbench mounting-error calibration and
# neither bam.mujoco nor bam.mjlab reads it during simulation.
XL330_M6_PARAMS: dict[str, float] = {
    "kt": 0.36601349688984386,                       # Nm/A (and V/(rad/s))
    "R": 2.8113923539223227,                         # Ohm
    "armature": 0.0018077432831600838,               # kg m^2
    "friction_base": 0.004771183165566,              # Nm
    "friction_stribeck": 0.004676345799486616,       # Nm
    "load_friction_motor": 0.2667860954283698,       # Nm/Nm
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,            # rad/s
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,        # Nm/(rad/s)
}

# bam/dynamixel/actuator.py, XL330Actuator.__init__ / module constants.
XL330_ENCODER_COUNTS_PER_REV = 4096
XL330_KP_DIVISOR = 256   # "Empirically observed for XL330 (manual mentions 128)"
XL330_PWM_LIMIT = 885    # "Default Present PWM limit for XL330"
# error_gain converts (kp_fw * position error [rad]) into a PWM duty cycle.
XL330_ERROR_GAIN = (XL330_ENCODER_COUNTS_PER_REV / (2 * np.pi)) / (
    XL330_KP_DIVISOR * XL330_PWM_LIMIT
)
XL330_MAX_PWM = 1.0      # bam/dynamixel/actuator.py: max_pwm=1.0
XL330_MAX_CURRENT = 1.75 # bam/dynamixel/actuator.py: "Firmware current limit [A]"
# Peak torque the FIRMWARE will ever allow: kt * I_max. This is the honest
# saturation point for a reward that prices motor strain — the MJCF's ±0.96 Nm
# forcerange is the *voltage* ceiling and stays 0.96 even under this model, so
# anything normalizing against the model's forcerange under-charges torque by
# (0.96/0.640)**2 = 2.25x. Consumed by behaviors._tau2_max.
XL330_MAX_TORQUE = 0.36601349688984386 * XL330_MAX_CURRENT  # 0.6405 Nm
XL330_NOMINAL_VIN = 7.5  # bam/dynamixel/actuator.py: vin=7.5 (overridden by vin_range)

# microduck_rl/src/mjlab_microduck/robot/microduck_constants.py, _BAM_ACTUATOR_KWARGS.
DEFAULT_KP_FW = 200.0                  # "microduck's preserved firmware stiffness"
DEFAULT_VIN_RANGE = (6.5, 8.2)         # per-env battery voltage, sampled at startup
DEFAULT_VIN_DROP_GAIN_RANGE = (0.0, 0.2)   # V/Nm, V_drop = gain * sum|tau|
DEFAULT_VIN_MIN = 6.0                  # hard floor after sag
DEFAULT_DELAY_MIN_LAG = 3              # physics steps
DEFAULT_DELAY_MAX_LAG = 6              # physics steps

# microduck_velocity_env_cfg.py: JOINT_FRICTION_RANDOMIZATION_RANGE, applied by
# mdp.randomize_bam_friction through FrictionDRBamActuator.friction_scale.
DEFAULT_FRICTION_SCALE_RANGE = (0.9, 1.1)

# bam/mjlab.py: BamActuator._STIFF_SOLREF_FRICTION / _STIFF_SOLIMP_FRICTION.
# MuJoCo Warp has no noslip solver, so upstream stiffens the per-DOF friction
# constraint directly. CPU MuJoCo defaults to noslip_iterations=0 too, so the
# same substitute applies here.
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def load_bam_params() -> tuple[dict[str, float], str]:
    """Return the xl330/m6 parameters and where they came from.

    Prefers the installed `bam` package (the same JSON microduck_rl trains
    with); falls back to the hard-coded `XL330_M6_PARAMS` table above.
    """
    try:  # pragma: no cover - depends on the environment
        import json

        from bam.model import _resolve_json_path  # type: ignore

        with open(_resolve_json_path(None, "xl330", "m6")) as fh:
            data = json.load(fh)
        return {k: float(data[k]) for k in XL330_M6_PARAMS}, "bam package"
    except Exception:
        return dict(XL330_M6_PARAMS), "hard-coded (bam/params/xl330/m6.json)"


# ----------------------------------------------------------------- delay buffer


class DelayBuffer:
    """Ring buffer serving a target from `lag` steps ago, lag ~ U{min..max}.

    Mirrors `mjlab.utils.buffers.DelayBuffer` as the BAM actuator uses it
    (hold_prob=0, update_period=0): a fresh lag is drawn EVERY physics step and
    clamped to the history actually available, so a just-reset buffer serves the
    newest value rather than a stale one.
    """

    def __init__(self, min_lag: int, max_lag: int, width: int,
                 rng: np.random.Generator):
        if not 0 <= min_lag <= max_lag:
            raise ValueError(f"need 0 <= min_lag <= max_lag, got {min_lag}, {max_lag}")
        self.min_lag = int(min_lag)
        self.max_lag = int(max_lag)
        self._rng = rng
        self._size = self.max_lag + 1
        self._lag_hi = self.max_lag + 1  # exclusive bound for rng.integers
        self._buf = np.zeros((self._size, width), dtype=np.float64)
        self._head = 0      # index of the newest entry
        self._len = 0
        self.current_lag = 0

    def reset(self) -> None:
        self._len = 0
        self._head = 0
        self.current_lag = 0

    def reseed(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def append(self, value: np.ndarray) -> None:
        head = self._head + 1
        if head == self._size:
            head = 0
        self._head = head
        self._buf[head] = value
        if self._len < self._size:
            self._len += 1

    def compute(self) -> np.ndarray:
        n = self._len
        if n == 0:
            raise RuntimeError("DelayBuffer.compute() before any append()")
        # One scalar draw per physics step, by design: the draw ORDER on the
        # shared rng is part of the reproducible stream (reset() interleaves a
        # uniform draw), so lags cannot be batched without changing rollouts.
        lag = int(self._rng.integers(self.min_lag, self._lag_hi))
        self.current_lag = lag
        if lag >= n:                   # clamp to available history
            lag = n - 1
        idx = self._head - lag
        if idx < 0:
            idx += self._size
        return self._buf[idx]


# -------------------------------------------------------------------- actuator


class BamXL330Actuator:
    """BAM xl330/m6 servo driving a set of MuJoCo hinge joints on CPU.

    Structured like `bam.mujoco.MujocoController` (which owns an MjModel/MjData
    pair and is ticked once per physics step) but reproducing the pipeline of
    `bam.mjlab.BamActuator`, which is what microduck_rl actually trains with:
    per-env supply voltage + load sag, a command delay buffer, the firmware
    current limiter, and the m6 friction budget written into MuJoCo's own
    `dof_frictionloss` / `dof_damping` so the solver performs the static-friction
    clipping (BAM Algorithm 1) natively.

    Torque is applied through `data.qfrc_applied` on the actuated DOFs only —
    the other DOF slots (e.g. the free-joint rows a demo "spotter" writes) are
    never touched. The MJCF position actuators are neutralized (zero gain/bias)
    so `data.ctrl` keeps its documented meaning as a position target while
    contributing no force.
    """

    def __init__(
        self,
        model,
        data,
        joint_names,
        *,
        dt: float,
        rng: np.random.Generator | None = None,
        kp_fw: float = DEFAULT_KP_FW,
        vin_range: tuple[float, float] | None = DEFAULT_VIN_RANGE,
        vin_drop_gain_range: tuple[float, float] | None = DEFAULT_VIN_DROP_GAIN_RANGE,
        vin_min: float | None = DEFAULT_VIN_MIN,
        delay_min_lag: int = DEFAULT_DELAY_MIN_LAG,
        delay_max_lag: int = DEFAULT_DELAY_MAX_LAG,
        max_current: float | None = XL330_MAX_CURRENT,
        stiff_frictionloss: bool = True,
        friction_scale_range: tuple[float, float] | None = None,
        params: dict[str, float] | None = None,
        current_scale: float | None = None,
    ):
        self.model = model
        self.data = data
        self.joint_names = tuple(joint_names)
        self.num_joints = len(self.joint_names)
        self.dt = float(dt)
        self._rng = rng if rng is not None else np.random.default_rng(0)

        if params is None:
            params, self.params_source = load_bam_params()
        else:
            self.params_source = "caller-supplied"
        self.p = dict(params)
        # Hot-loop scalar hoists. `self.p` stays the public source of truth for
        # tests/introspection; these cached copies exist because dict lookups
        # inside a 4-substeps-per-control-step loop showed up in the profile.
        # Mutating `self.p` after construction is not a supported operation.
        p = self.p
        self._kt = float(p["kt"])
        self._R = float(p["R"])
        self._fbase = float(p["friction_base"])
        self._fstrib = float(p["friction_stribeck"])
        self._lfm = float(p["load_friction_motor"])
        self._lfe = float(p["load_friction_external"])
        self._lfms = float(p["load_friction_motor_stribeck"])
        self._lfes = float(p["load_friction_external_stribeck"])
        self._lfmq = float(p["load_friction_motor_quad"])
        self._lfeq = float(p["load_friction_external_quad"])
        self._dtheta_strib = float(p["dtheta_stribeck"])
        self._alpha = float(p["alpha"])
        self._fvisc = float(p["friction_viscous"])

        self.kp_fw = float(kp_fw)
        self.error_gain = XL330_ERROR_GAIN
        self.max_pwm = XL330_MAX_PWM
        # MICRODUCK_BAM_CURRENT_SCALE: the servo-strength LADDER knob
        # (default 1.0 = the honest XL330 firmware limit, tau <= 0.640 Nm).
        # The xml->bam jump proved to be one giant physics cliff for
        # from-scratch students (every 2026-08-31 scratch chain re-held at
        # ~0.3 on bam and went flat); curriculum stages set this to 1.4 ->
        # 1.15 -> 1.0 so servo strength descends in rungs like every other
        # difficulty axis. >1.0 is a training device only — never evaluate
        # or ship against a scaled limit.
        # An explicit `current_scale=` beats the env var: envs living INSIDE
        # the lab process (the trainee preview) must be able to mirror a
        # curriculum stage's scale per instance, while the trainer subprocess
        # keeps setting it process-wide through its environment.
        scale = 1.0
        sv = current_scale if current_scale is not None else os.environ.get(
            "MICRODUCK_BAM_CURRENT_SCALE")
        if sv:
            try:
                scale = float(np.clip(float(sv), 0.5, 2.5))
            except (TypeError, ValueError):
                scale = 1.0
        self.max_current = (None if max_current is None
                            else float(max_current) * scale)
        self.vin_min = vin_min
        self.friction_scale_range = friction_scale_range
        self.friction_scale = 1.0

        # Per-robot battery voltage and internal-resistance gain: sampled ONCE
        # at construction and held across resets, exactly as BamActuator does
        # (they are startup randomization, not per-episode events).
        # MICRODUCK_BAM_VIN pins the per-robot supply voltage instead of
        # sampling it — "train at the battery's limit". The XL330's no-load
        # speed scales with vin, so 8.2 V (a fresh 2S pack) buys ~9% more
        # joint speed than the 7.5 V nominal; a policy chasing top speed
        # should train against the voltage the robot will actually be given.
        # Load sag and the 6.0 V floor stay active — pinning the NOMINAL is a
        # battery choice, not a physics cheat.
        pinned = None
        pin = os.environ.get("MICRODUCK_BAM_VIN")
        if pin:
            try:
                pinned = float(np.clip(float(pin), 6.0, 8.4))
            except ValueError:
                pinned = None  # malformed -> fall through to the sampled draw
        if pinned is not None:
            self.vin_nominal = pinned
        else:
            self.vin_nominal = (
                float(self._rng.uniform(*vin_range)) if vin_range is not None
                else XL330_NOMINAL_VIN
            )
        self.vin_drop_gain = (
            float(self._rng.uniform(*vin_drop_gain_range))
            if vin_drop_gain_range is not None else None
        )

        # MuJoCo index resolution — by name, so joint reordering can't bite.
        self.joint_ids = np.array(
            [model.joint(n).id for n in self.joint_names], dtype=np.int32
        )
        self.qpos_adr = np.array(
            [model.joint(n).qposadr[0] for n in self.joint_names], dtype=np.int32
        )
        self.dof_adr = np.array(
            [model.joint(n).dofadr[0] for n in self.joint_names], dtype=np.int32
        )
        self.actuator_ids = np.array(
            [model.actuator(n).id for n in self.joint_names], dtype=np.int32
        )
        # DOF index -> our joint slot, for the per-step efc scan.
        self._dof_slot = np.full(model.nv, -1, dtype=np.int64)
        self._dof_slot[self.dof_adr] = np.arange(self.num_joints)

        # On the shipped MJCF the 14 hinge DOFs sit in one contiguous block
        # after the free joint, so the per-substep gathers/scatters can use
        # slices (views, no fancy-index copy). Falls back to the index arrays
        # for any model where that doesn't hold — values are identical either
        # way, slices are just ~3x cheaper.
        def _as_slice(idx: np.ndarray):
            if idx.size and np.array_equal(
                    idx, np.arange(idx[0], idx[0] + idx.size, dtype=idx.dtype)):
                return slice(int(idx[0]), int(idx[0]) + idx.size)
            return None

        self._qpos_ix = _as_slice(self.qpos_adr) or self.qpos_adr
        self._dof_ix = _as_slice(self.dof_adr) or self.dof_adr
        self._act_ix = _as_slice(self.actuator_ids) or self.actuator_ids
        # Scratch buffer for the per-substep |prev torque| reduction.
        self._abs_buf = np.empty(self.num_joints, dtype=np.float64)
        # Returned when the efc scan finds no DOF-friction rows. Callers only
        # read it (`ext -= zeros`); mutating would poison the next substep.
        self._zero_tau = np.zeros(self.num_joints, dtype=np.float64)
        self._friction_cnstr_type = (
            int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value)
            if mujoco is not None else -1
        )

        # Voltage ceiling on the produced torque. BamActuator.edit_spec sets the
        # MuJoCo forcerange to vin_max*kt/R; we clamp by hand because the torque
        # bypasses the actuator machinery.
        vin_for_limit = max(vin_range) if vin_range is not None else XL330_NOMINAL_VIN
        self.force_limit = vin_for_limit * self.p["kt"] / self.p["R"]

        self._delay = (
            DelayBuffer(delay_min_lag, delay_max_lag, self.num_joints, self._rng)
            if delay_max_lag > 0 else None
        )
        self._target = np.zeros(self.num_joints, dtype=np.float64)
        self._prev_motor_torque = np.zeros(self.num_joints, dtype=np.float64)
        self._applied_torque = np.zeros(self.num_joints, dtype=np.float64)
        self.last_vin = self.vin_nominal
        self.last_frictionloss = np.zeros(self.num_joints, dtype=np.float64)

        self._prepare_model(stiff_frictionloss)

    # ------------------------------------------------------------- model setup

    def _prepare_model(self, stiff_frictionloss: bool) -> None:
        """Retune the model the way `BamActuator.edit_spec` retunes the spec.

        Armature comes from BAM; MuJoCo's own damping/frictionloss are zeroed
        (compute() rewrites them every step); and the position actuators are
        neutralized so nothing but our qfrc_applied drives these joints.
        """
        m = self.model
        m.dof_armature[self.dof_adr] = self.p["armature"]
        m.dof_damping[self.dof_adr] = 0.0
        m.dof_frictionloss[self.dof_adr] = 0.0
        if stiff_frictionloss:
            m.dof_solref[self.dof_adr] = STIFF_SOLREF_FRICTION
            m.dof_solimp[self.dof_adr] = STIFF_SOLIMP_FRICTION

        # Neutralize the MJCF position servos: force = gain*ctrl + bias(q, qvel),
        # so zeroing both gainprm and biasprm makes them produce exactly zero
        # regardless of what anybody writes into data.ctrl.
        m.actuator_gainprm[self.actuator_ids] = 0.0
        m.actuator_biasprm[self.actuator_ids] = 0.0

        # dof_armature feeds derived constants; bam.mujoco does the same.
        if mujoco is not None:
            mujoco.mj_setConst(self.model, self.data)

    # -------------------------------------------------------------- lifecycle

    def reset(self, q_target: np.ndarray | None = None) -> None:
        """Clear per-episode state. Call after `mj_resetData`/`mj_forward`."""
        if self._delay is not None:
            self._delay.reset()
        q = (self.data.qpos[self.qpos_adr] if q_target is None
             else np.asarray(q_target, dtype=np.float64))
        self._target = np.array(q, dtype=np.float64)
        self._prev_motor_torque[:] = 0.0
        self._applied_torque[:] = 0.0
        self.data.qfrc_applied[self.dof_adr] = 0.0
        # Non-accumulating friction DR (restore-then-apply, per AGENTS.md).
        self.friction_scale = 1.0
        if self.friction_scale_range is not None:
            self.friction_scale = float(self._rng.uniform(*self.friction_scale_range))

    def reseed(self, seed: int) -> None:
        """Re-seed the stochastic parts (command lag, friction DR).

        The per-robot battery voltage and internal-resistance gain are NOT
        re-sampled: BamActuator.reset() documents them as startup randomization
        held across resets, and a robot does not swap batteries per episode.
        """
        self._rng = np.random.default_rng(seed)
        if self._delay is not None:
            self._delay.reseed(self._rng)

    def set_target(self, q_target: np.ndarray) -> None:
        """Set the firmware position target [rad] (control rate)."""
        # Copy: callers pass live views (e.g. data.ctrl) that must not alias.
        self._target = np.array(q_target, dtype=np.float64)

    # ----------------------------------------------------------------- physics

    def compute_motor_torque(
        self, q: np.ndarray, dq: np.ndarray, q_target: np.ndarray, vin: float | np.ndarray
    ) -> np.ndarray:
        """Firmware control law + DC-motor equation → motor torque [Nm].

        Port of `VoltageControlledActuator.compute_control` + `compute_torque`.
        Pure function of its arguments — used directly by the validation
        harness and the tests.

        (Bit-compatibility note: the `np.minimum(np.maximum(...))` clamps are
        NumPy's own definition of the `clip` ufunc, so they produce the exact
        bits `np.clip` did — without its Python dispatch overhead, which
        dominated this function's cost at J=14.)
        """
        kt, R = self._kt, self._R
        if _motor_torque_kernel is not None and not isinstance(vin, np.ndarray):
            out = np.empty(len(q), dtype=np.float64)
            _motor_torque_kernel(
                np.asarray(q, dtype=np.float64),
                np.asarray(dq, dtype=np.float64),
                np.asarray(q_target, dtype=np.float64), float(vin),
                kt, kt ** 2, R, self.kp_fw, self.error_gain,
                self.max_current is not None,
                0.0 if self.max_current is None else float(self.max_current),
                self.max_pwm, out)
            return out
        duty = (q_target - q) * self.kp_fw * self.error_gain

        if self.max_current is not None:
            # The firmware can only bound the DUTY CYCLE, not synthesize
            # voltage: solving |I| <= I_max for I = (duty*vin - kt*dq)/R gives
            # this window. Applied BEFORE the physical PWM clamp, so at high
            # back-EMF the limiter saturates without holding I at I_max —
            # exactly how the real firmware behaves.
            duty_center = (kt * dq) / vin
            duty_span = (R * self.max_current) / vin
            duty = np.minimum(np.maximum(duty, duty_center - duty_span),
                              duty_center + duty_span)

        # Battery reality, last.
        duty = np.minimum(np.maximum(duty, -self.max_pwm), self.max_pwm)
        volts = vin * duty
        return kt * volts / R - (kt**2) * dq / R

    def friction_budget(
        self, motor_torque: np.ndarray, external_torque: np.ndarray, dq: np.ndarray
    ) -> np.ndarray:
        """m6 velocity-independent friction budget [Nm] — shape (J,).

        Follows `bam.mjlab.BamActuator._compute_friction_budget` (the path
        microduck_rl trains with). Note it differs slightly from
        `bam.model.Model.compute_frictions`: mjlab drops the latter's
        `sign(ext) != sign(motor)` gate on the quadratic term. mjlab wins here.

        (Bit-compatibility notes for the restructured form: `fbase + x` equals
        the old `full_like(fbase) + x` elementwise; `np.where(m, a, b)` equals
        the old `drive*a + (1-drive)*b` because one addend is exactly +0.0 for
        these non-negative finite terms; the accumulation order of the four
        terms is unchanged. test_bam_perf_parity.py holds this to the bit.)
        """
        motor_torque = np.asarray(motor_torque, dtype=np.float64)
        stribeck = np.exp(-np.power(np.abs(dq) / self._dtheta_strib, self._alpha))

        if _friction_budget_kernel is not None:
            out = np.empty(len(motor_torque), dtype=np.float64)
            _friction_budget_kernel(
                motor_torque, np.asarray(external_torque, dtype=np.float64),
                stribeck, self._fbase, self._fstrib, self._lfe, self._lfm,
                self._lfes, self._lfms, self._lfeq, self._lfmq,
                self.friction_scale, out)
            return out

        fl = self._fbase + stribeck * self._fstrib

        # m5/m6 directional gearbox load: motor-side and external-side torques
        # subtract, so a joint held against gravity is cheaper than one driving.
        gearbox = np.abs(external_torque * self._lfe - motor_torque * self._lfm)
        fl = fl + gearbox

        gearbox_strib = np.abs(external_torque * self._lfes
                               - motor_torque * self._lfms)
        fl = fl + stribeck * gearbox_strib

        # m6 quadratic term, split by which side dominates.
        abs_ext = np.abs(external_torque)
        abs_mot = np.abs(motor_torque)
        quad = np.where(abs_mot > abs_ext,
                        self._lfeq * abs_ext**2, self._lfmq * abs_mot**2)
        fl = fl + stribeck * quad

        return fl * self.friction_scale

    def external_torque(self) -> np.ndarray:
        """Load seen at the gearbox: gravity/Coriolis + constraints [Nm].

        Deliberately excludes the DOF-friction constraint force we ourselves
        injected on the previous solve — otherwise the load-dependent friction
        terms would feed back on themselves (mirrors bam.mjlab / bam.mujoco).
        """
        d = self.data
        ix = self._dof_ix
        # b - a is IEEE-identical to -a + b (subtraction IS addition of the
        # negation), so this keeps the documented sign convention to the bit.
        ext = d.qfrc_constraint[ix] - d.qfrc_bias[ix]
        ext -= self._dof_friction_force()
        return ext

    def _dof_friction_force(self) -> np.ndarray:
        """Force our own dof_frictionloss constraints produced last solve.

        For a `mjCNSTR_FRICTION_DOF` row, `efc_id` is the DOF index (this is
        what bam.mjlab uses; bam.mujoco compares against joint ids instead,
        which only coincides on its single-joint pendulum testbench).
        """
        d = self.data
        n = int(d.nefc)
        if n == 0 or mujoco is None:
            return self._zero_tau
        if _dof_friction_kernel is not None:
            out = np.zeros(self.num_joints, dtype=np.float64)
            _dof_friction_kernel(d.efc_type, d.efc_id, d.efc_force, n,
                                 np.int32(self._friction_cnstr_type),
                                 self._dof_slot, out)
            return out
        is_fric = d.efc_type[:n] == self._friction_cnstr_type
        ids = d.efc_id[:n][is_fric]
        if ids.size == 0:
            return self._zero_tau
        # _dof_slot maps a DOF index to our joint slot (-1 for DOFs we do not
        # actuate), so the scan is three vector ops instead of a per-joint loop.
        slot = self._dof_slot[ids]
        keep = slot >= 0
        # bincount with float weights already returns float64; copy=False makes
        # the astype a no-op instead of a fresh 14-element allocation.
        return np.bincount(slot[keep], weights=d.efc_force[:n][is_fric][keep],
                           minlength=self.num_joints).astype(np.float64,
                                                             copy=False)

    def before_step(self) -> np.ndarray:
        """One physics substep of BAM. Call immediately before `mj_step`.

        Writes `data.qfrc_applied` on the actuated DOFs and rewrites the
        per-DOF friction fields; returns the applied torque [Nm].
        """
        d = self.data
        dix = self._dof_ix
        q = d.qpos[self._qpos_ix]    # views when the DOF block is contiguous —
        dq = d.qvel[dix]             # read-only inside this substep

        # Bus/firmware command lag (3-6 physics steps upstream).
        if self._delay is not None:
            self._delay.append(self._target)
            q_target = self._delay.compute()
        else:
            q_target = self._target

        # Battery sag under load, from the PREVIOUS step's torques.
        vin = self.vin_nominal
        if self.vin_drop_gain is not None:
            np.abs(self._prev_motor_torque, out=self._abs_buf)
            vin = vin - self.vin_drop_gain * float(self._abs_buf.sum())
            if self.vin_min is not None:
                vin = max(vin, self.vin_min)
        self.last_vin = vin

        motor_torque = self.compute_motor_torque(q, dq, q_target, vin)
        # compute_motor_torque returns a fresh array and nothing below mutates
        # it, so holding the reference replaces the old per-substep .copy().
        self._prev_motor_torque = motor_torque

        # Friction budget uses the torque APPLIED on the previous solve as the
        # motor-side load (upstream reads data.qfrc_actuator; we bypass the
        # actuators, so we carry the same quantity ourselves).
        fl = self.friction_budget(self._applied_torque, self.external_torque(), dq)
        self.last_frictionloss = fl
        self.model.dof_frictionloss[dix] = fl
        self.model.dof_damping[dix] = self._fvisc

        torque = np.minimum(np.maximum(motor_torque, -self.force_limit),
                            self.force_limit)
        self._applied_torque = torque
        # Additive-safe: only OUR dof rows are written, never the whole array.
        d.qfrc_applied[dix] = torque
        return torque

    def after_step(self) -> None:
        """Publish the BAM torque as `data.actuator_force`. Call after `mj_step`.

        The neutralized position actuators compute zero, but reward terms
        (walk_env's torque penalty, behaviors.py's torque/stall terms) read
        `data.actuator_force` as "the torque the servos produced" — under BAM
        that is our torque. Purely a readout: MuJoCo recomputes the field from
        scratch inside the next mj_step, so this changes no physics.
        """
        self.data.actuator_force[self._act_ix] = self._applied_torque

    @property
    def applied_torque(self) -> np.ndarray:
        return self._applied_torque

    @property
    def current_lag(self) -> int:
        return 0 if self._delay is None else self._delay.current_lag
