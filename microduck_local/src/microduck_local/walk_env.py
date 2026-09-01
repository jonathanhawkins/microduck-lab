"""Gymnasium walking env for Microduck on plain CPU MuJoCo.

Two actuator models are available (`actuator=`, or `MICRODUCK_ACTUATOR`):

- ``"xml"`` (default) — microduck_rl's deployment-rehearsal fidelity
  (scripts/infer_policy.py): the MJCF position actuators (kp=0.55), dt=0.005,
  decimation 4 → 50 Hz. Policies trained here are prototypes: expect them to run
  in infer_policy.py, but port the env design to an mjlab cfg and retrain on a
  GPU (--hf-jobs) before expecting sim2real transfer.
- ``"bam"`` — the BAM xl330/m6 voltage model the official mjlab stack actually
  trains with (see bam_actuator.py): firmware current limit, real back-EMF,
  load-dependent gearbox friction, battery sag, 3-6 step bus lag. Slower per
  step, but it is the physics the shipped policies were optimized against.

The observation/action contract is exact under both (contract.py), so exported
ONNX is drop-in compatible with infer_policy.py --new-cmd-obs and the runtime.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from . import contract as C
from .bam_actuator import DEFAULT_FRICTION_SCALE_RANGE, BamXL330Actuator

# Shared constant operands for the per-step frame math. quat_rotate_inverse
# and the heading projection only READ them, so one allocation serves every
# step of every env (building them per call measured ~0.5 us each).
_NEG_Z = np.array([0.0, 0.0, -1.0])
_E_FWD = np.array([1.0, 0.0, 0.0])

# ---------------------------------------------------------------- model sharing
#
# Measured on this robot's scene_walk.xml: the mjData that holds the actual
# simulation state costs ~0.9 MB, while the compiled mjModel costs ~138 MB as a
# second copy in a warm process and ~470 MB as the FIRST compile in a fresh one
# (the MJCF compiler's meshes, BVH and textures never come back). MuJoCo is
# built so one read-only mjModel backs many mjData; a worker per env, each
# compiling its own, threw that away — 99% of per-env memory was a private copy
# of an identical, never-written model.
#
# `shared_model()` compiles a scene at most once PER PROCESS. Combined with the
# fork-based vector env in vec_env.py the children inherit the parent's compiled
# model copy-on-write, so the whole fleet costs one model.
#
# Keyed by (scene, actuator): the BAM actuator PERMANENTLY retunes the model it
# is attached to (it zeroes the MJCF position servos' gainprm/biasprm), so a BAM
# env and an "xml" env can never be handed the same compiled model.
_SHARED_MODELS: dict[tuple[str, str], mujoco.MjModel] = {}

# id(model) -> the model's compile-time body_mass/geom_friction, captured the
# moment it was compiled. Domain randomization writes those two arrays, so an
# env that joins a shared model AFTER a sibling has already randomized it would
# otherwise adopt the sibling's draw as its "restore to defaults" baseline and
# quietly accumulate. Only cached models are registered, and the cache holds the
# strong reference that keeps the id valid.
_PRISTINE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

# Set by `shared_model_scope()`: envs constructed inside the scope fetch from
# the cache instead of compiling. A ContextVar rather than a plain global so an
# unrelated env built elsewhere in the process is never silently re-pointed.
# The value is `exclusive`: True when this process hosts exactly ONE env per
# model (the fork case — the child's copy-on-write model is private in every
# way that matters), False when sibling envs step the same mjModel object.
_MODEL_SCOPE: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "microduck_model_scope", default=None
)


def shared_model(scene: str | Path, actuator: str = "xml") -> mujoco.MjModel:
    """Compile `scene` at most once per process; hand back the same MjModel.

    Once any env using this model has reset, the model carries that env's
    domain-randomization draw — `pristine_baselines()` is how a later env
    recovers the compile-time values it must restore to.
    """
    key = (str(scene), actuator)
    model = _SHARED_MODELS.get(key)
    if model is None:
        model = mujoco.MjModel.from_xml_path(key[0])
        _SHARED_MODELS[key] = model
        _PRISTINE[id(model)] = (model.body_mass.copy(),
                                model.geom_friction.copy())
    return model


def pristine_baselines(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """The model's compile-time (body_mass, geom_friction).

    Recorded at compile time for cached models; for a privately compiled or
    caller-supplied model there is nobody else to have touched it, so reading it
    now is the same answer.
    """
    known = _PRISTINE.get(id(model))
    if known is not None:
        return known
    return model.body_mass.copy(), model.geom_friction.copy()


def clear_shared_models() -> None:
    """Drop the per-process model cache (tests; long-lived servers)."""
    _SHARED_MODELS.clear()
    _PRISTINE.clear()


@contextmanager
def shared_model_scope(exclusive: bool = True):
    """Envs built inside this scope adopt the cached model for their scene.

    `exclusive=True` promises this process holds at most one env per model —
    what a fork-based vec env gives you, since each child got its own
    copy-on-write copy. `exclusive=False` means sibling envs in this process
    step the same mjModel object, which costs a per-step re-assert of the
    env's domain-randomization draw (see `_sync_model`) and rules out the BAM
    actuator, which retunes model.dof_frictionloss on every physics substep.
    """
    token = _MODEL_SCOPE.set(bool(exclusive))
    try:
        yield
    finally:
        _MODEL_SCOPE.reset(token)


class MicroduckWalkEnv(gym.Env):
    """Velocity-command walking, rewards distilled from the mjlab velocity recipe."""

    metadata = {"render_modes": []}

    # Reward weights. Convention (mirrors AGENTS.md): *_penalty terms are
    # self-negating (the term function returns <= 0) and carry POSITIVE
    # weights; every logged episode sum for a penalty must come out <= 0.
    W_TRACK_LIN = 2.0
    W_TRACK_ANG = 2.0
    W_UPRIGHT = 2.0
    W_HEAD_POSE = 2.0
    W_POSE = 1.0
    W_AIR_TIME = 3.0
    W_ANG_VEL_XY = 0.05     # body_ang_vel; mjlab-base cost, self-negating here

    TRACK_STD2 = 0.1        # GPU track_linear_velocity std=sqrt(0.1)
    ANG_TRACK_STD2 = 0.5    # GPU track_angular_velocity std=sqrt(0.5)
    UPRIGHT_STD2 = 0.05     # matches velocity cfg's tightened upright
    POSE_STD2 = 0.5
    HEAD_STD = 0.5          # per-joint Gaussian std, as in head_pose_tracking
    AIR_TIME_MIN = 0.125    # s — official walking window; a one-step shuffle
                            # (CTRL_DT=0.02) must not pay
    AIR_TIME_MAX = 0.300    # dense in-window payout, same as GPU feet_air_time
    # GPU ramps action_rate_l2 -0.1 → -1.0 over 1500 iters × 24 steps/env.
    _ACTION_RATE_STAGES = (
        (0, 0.1), (12_000, 0.2), (18_000, 0.4),
        (24_000, 0.6), (30_000, 0.8), (36_000, 1.0),
    )

    # Termination thresholds: walk model strips trunk collisions, so "fell"
    # is orientation/height-based (gravity_z in body frame is ~-1 upright).
    # Matched to upstream (microduck_velocity_env_cfg.py): 70 deg, not 60.
    FALL_GRAVITY_Z = -0.342  # tilted > 70 deg
    # Upstream has NO height termination. Ours exists to catch the folded-crouch
    # failure mode (a policy that shuffles along on bent knees scores as
    # "upright"), but at 0.10 m it sat 9 mm below the measured p1 trunk height
    # of a normal gait — close enough to end any faster, more dynamic stride
    # that dips lower. 0.07 m still catches a genuine collapse (a trunk resting
    # on the floor is 0.02-0.05 m) with real clearance above it.
    FALL_HEIGHT = 0.07      # m

    def __init__(
        self,
        max_episode_s: float = 20.0,
        command_resample_s: float = 5.0,
        # Upstream commands standing in only 2% of envs at the START and
        # ramps it UP to 25% by curriculum — it demands motion first and
        # teaches standing later. We began where they finish (25% of commands
        # involving no forward motion), which rewards the do-nothing policy
        # from step one.
        zero_command_prob: float = 0.02,
        turn_in_place_prob: float = 0.15,   # GPU TURN_IN_PLACE_FRACTION
        forward_command_prob: float = 0.2,  # mjlab rel_forward_envs (silent in velocity cfg)
        obs_noise: bool = True,
        domain_rand: bool = True,
        action_delay: bool = True,
        random_yaw: bool = True,
        seed: int | None = None,
        scene_xml: str | None = None,   # default walk scene; SCENE_ALL_XML for
                                        # tricks needing head/trunk floor contact
        terminate_on_fall: bool = True, # False for deliberately-inverted tricks
        height_termination: bool = True,  # GPU has no z-kill; run turns this off
        actuator: str = "xml",          # "xml" | "bam" — see the module docstring.
                                        # MICRODUCK_ACTUATOR hard-overrides it, so a
                                        # trainer/lab process can switch every env
                                        # it spawns without touching call sites.
        actuator_force: str | None = None,   # ...except here: an explicit
                                        # per-instance choice that BEATS the process
                                        # env. The lab runs with MICRODUCK_ACTUATOR=bam
                                        # for its roster, but a curriculum stage may
                                        # declare xml (the headstand ladder's training
                                        # wheels) — without this the trainee preview
                                        # silently rehearsed the wrong physics while
                                        # the trainer subprocess used the stage's.
        bam_current_scale: float | None = None,  # per-instance servo-strength
                                        # ladder knob; None = read
                                        # MICRODUCK_BAM_CURRENT_SCALE from the env.
        model: mujoco.MjModel | None = None,  # adopt an already-compiled model
                                        # instead of compiling a private ~138 MB
                                        # copy. See `shared_model_scope`.
    ):
        super().__init__()
        scene = Path(scene_xml) if scene_xml else C.SCENE_WALK_XML
        if not scene.exists():
            raise FileNotFoundError(
                f"{scene} not found — clone microduck_rl next to "
                "microduck_local or set MICRODUCK_RL_DIR"
            )
        self.terminate_on_fall = terminate_on_fall
        self.height_termination = height_termination
        self.scene_path = str(scene)
        # Resolved here, before the model is chosen, because the two BAM/xml
        # variants of a scene are different compiled models (see _SHARED_MODELS).
        self.actuator_model = (
            actuator_force if actuator_force is not None
            else os.environ.get("MICRODUCK_ACTUATOR", actuator)
        ).strip().lower()
        if self.actuator_model not in ("xml", "bam"):
            raise ValueError(
                f"actuator must be 'xml' or 'bam', got {self.actuator_model!r}"
            )
        # A model handed in explicitly is the caller's business (they promise it
        # is this scene). Otherwise an enclosing shared_model_scope() decides
        # between the per-process cache and a private compile.
        scope = _MODEL_SCOPE.get()
        if model is not None:
            self._model_shared = False
        elif scope is not None:
            model = shared_model(scene, self.actuator_model)
            # exclusive=True (the fork case) means this process holds one env
            # per model, so nothing has to be re-asserted per step.
            self._model_shared = not scope
        else:
            self._model_shared = False
        self.model = (model if model is not None
                      else mujoco.MjModel.from_xml_path(str(scene)))
        self.model.opt.timestep = C.PHYSICS_DT
        self.data = mujoco.MjData(self.model)

        self.max_steps = int(round(max_episode_s / C.CTRL_DT))
        self.resample_steps = int(round(command_resample_s / C.CTRL_DT))
        self.zero_command_prob = zero_command_prob
        self.turn_in_place_prob = turn_in_place_prob
        self.forward_command_prob = forward_command_prob
        # Lifetime-ramped reward terms count on this. Seeded from
        # MICRODUCK_RAMP_OFFSET (exported by train_behavior BEFORE the vec-env
        # workers fork) so a warm RESTART resumes ramps at strength: without
        # it, every lab helper add/remove reset ramped penalties to their
        # gentle stage-0 value and then slammed them back at full strength a
        # few hundred k steps later — whiplash that collapsed a run from
        # ep_len 396 (the session's best) to 10.
        try:
            self._lifetime_steps = int(float(
                os.environ.get("MICRODUCK_RAMP_OFFSET", "0")))
        except ValueError:
            self._lifetime_steps = 0
        self.obs_noise = obs_noise
        self.domain_rand = domain_rand
        self.action_delay = action_delay
        self.random_yaw = random_yaw

        # Model lookups — resolved by name so joint reordering can't bite.
        self.trunk_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.joint_qpos_adr = np.array([
            self.model.joint(n).qposadr[0] for n in C.JOINT_NAMES
        ])
        self.joint_qvel_adr = np.array([
            self.model.joint(n).dofadr[0] for n in C.JOINT_NAMES
        ])
        gyro = self.model.sensor("imu_ang_vel")
        self.gyro_adr = slice(gyro.adr[0], gyro.adr[0] + 3)
        self.key_stand = self.model.key("STAND").id
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.floor_geom = floor_id
        self.foot_geoms = {
            "left": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"),
            "right": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"),
        }
        # Make the FOOT's friction win the contact pair. MuJoCo mixes friction
        # by element-wise max unless one geom has the higher geom_priority, so
        # without this a foot randomized to 0.7 still contacts at the floor's
        # 1.0 and the DR knob is inert. Upstream sets priority=1 on these pads
        # for exactly this reason. Idempotent, so it is safe to set from every
        # env sharing one mjModel.
        for _gid in self.foot_geoms.values():
            self.model.geom_priority[_gid] = 1
        # ---- hot-path plumbing ----------------------------------------
        # Persistent numpy views into the mjData buffers (the buffers live as
        # long as `self.data`, so a view taken once stays valid; re-fetching
        # `data.sensordata[...]` etc. through the bindings costs ~1 us per
        # access and the reward stack does it dozens of times per step).
        self._gyro = self.data.sensordata[self.gyro_adr]
        self._act_force = self.data.actuator_force
        self._trunk_xpos = self.data.xpos[self.trunk_body_id]
        self._trunk_xquat = self.data.xquat[self.trunk_body_id]
        self._trunk_xmat = self.data.xmat[self.trunk_body_id]
        self._qvel_base = self.data.qvel[0:3]
        # The 14 joint addresses are contiguous in model order on this robot;
        # a slice view then replaces the fancy-index copy (bit-identical — the
        # same elements in the same order). The fancy-index fallback keeps a
        # hypothetical reordered model correct.
        qadr, vadr = self.joint_qpos_adr, self.joint_qvel_adr
        self._qpos_j = (
            self.data.qpos[qadr[0]:qadr[0] + C.NUM_JOINTS]
            if np.array_equal(qadr, np.arange(qadr[0], qadr[0] + C.NUM_JOINTS))
            else None)
        self._qvel_j = (
            self.data.qvel[vadr[0]:vadr[0] + C.NUM_JOINTS]
            if np.array_equal(vadr, np.arange(vadr[0], vadr[0] + C.NUM_JOINTS))
            else None)
        # Step-scoped memo cache: obs, the reward terms and the termination
        # check all re-derive the same quantities (projected gravity, joint
        # pos/vel, heading velocity, contact scans) from one frozen physics
        # state. Active ONLY inside step() — everyone else (tests and tools
        # that re-pose the env and read helpers directly, spawn functions
        # after their own mj_forward) always recomputes from live mjData.
        self._step_cache: dict = {}
        self._cache_active = False

        # Defaults saved for domain randomization restore-then-apply (DR must
        # not accumulate across resets — AGENTS.md). Under a SHARED model these
        # are also the only pristine copy left once a sibling has randomized,
        # which is why every env keeps its own and _sync_model replays it.
        self._default_body_mass, self._default_geom_friction = (
            arr.copy() for arr in pristine_baselines(self.model)
        )
        self._dr_body_mass = self._default_body_mass.copy()
        self._dr_geom_friction = self._default_geom_friction.copy()

        # Nominal standing trunk height, measured off the model itself (never
        # hand-carried across model revisions — AGENTS.md).
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        mujoco.mj_forward(self.model, self.data)
        self.stand_z = float(self.data.xpos[self.trunk_body_id][2])

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (C.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-4.0, 4.0, (C.NUM_JOINTS,), np.float32)

        self._rng = np.random.default_rng(seed)

        self.bam = None
        if self.actuator_model == "bam":
            if self._model_shared:
                # BamXL330Actuator retunes model.dof_frictionloss/dof_damping
                # on EVERY physics substep from this env's own load state, so
                # siblings sharing one mjModel would overwrite each other's
                # servo physics. Fork-based sharing is fine (one env per
                # process, copy-on-write); in-process sharing is not.
                raise ValueError(
                    "actuator='bam' cannot share an mjModel with sibling envs "
                    "in the same process (it rewrites dof_frictionloss every "
                    "substep). Use the fork-based vec env, or drop "
                    "shared_model_scope(exclusive=False)."
                )
            # Own RNG stream so the xml path's draws stay byte-for-byte as they
            # were. The env-level 0/1-ctrl-step action lag is switched OFF under
            # BAM: the actuator models the real 3-6 physics-step bus lag itself,
            # and stacking both would double-count it.
            self.bam = BamXL330Actuator(
                self.model, self.data, C.JOINT_NAMES,
                dt=C.PHYSICS_DT,
                rng=np.random.default_rng(seed),
                delay_min_lag=3 if action_delay else 0,
                delay_max_lag=6 if action_delay else 0,
                friction_scale_range=(
                    DEFAULT_FRICTION_SCALE_RANGE if domain_rand else None
                ),
                current_scale=bam_current_scale,
            )

        self._reset_episode_state()

    # ------------------------------------------------------------------ state

    def _reset_episode_state(self) -> None:
        self.step_count = 0
        self.last_action = np.zeros(C.NUM_JOINTS, dtype=np.float32)
        self.prev_action = np.zeros(C.NUM_JOINTS, dtype=np.float32)
        self.prev_joint_vel = np.zeros(C.NUM_JOINTS, dtype=np.float32)
        self.twist_cmd = np.zeros(3, dtype=np.float32)
        self.head_cmd = np.zeros(4, dtype=np.float32)
        self.body_cmd = np.zeros(6, dtype=np.float32)
        self.air_time = {"left": 0.0, "right": 0.0}
        self.was_contact = {"left": True, "right": True}
        self._action_lag = 0
        self._delayed_action = np.zeros(C.NUM_JOINTS, dtype=np.float32)
        self.reward_sums: dict[str, float] = {}

    def _sample_commands(self) -> None:
        r = self._rng
        u = r.uniform()
        stand = self.zero_command_prob
        turn = stand + self.turn_in_place_prob
        fwd = turn + self.forward_command_prob
        if u < stand:
            self.twist_cmd[:] = 0.0
        elif u < turn:
            self.twist_cmd[:] = (0.0, 0.0, r.uniform(*C.ANG_VEL_Z_RANGE))
        elif u < fwd:
            # mjlab rel_forward_envs: |vx| clamped to >= 0.3, vy = wz = 0.
            vx = abs(float(r.uniform(*C.LIN_VEL_X_RANGE)))
            self.twist_cmd[:] = (max(vx, 0.3), 0.0, 0.0)
        else:
            self.twist_cmd[:] = (
                r.uniform(*C.LIN_VEL_X_RANGE),
                r.uniform(*C.LIN_VEL_Y_RANGE),
                r.uniform(*C.ANG_VEL_Z_RANGE),
            )
        self.head_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.HEAD_CMD_RANGES]
        self.body_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.BODY_CMD_RANGES]

    def _apply_domain_rand(self) -> None:
        # Restore compile-time defaults, then apply — never accumulate.
        # The draw lands in this env's OWN arrays first: with a shared mjModel
        # the model is not a safe place to keep it, because a sibling env's
        # reset would retune this env's physics mid-episode.
        self._dr_body_mass[:] = self._default_body_mass
        self._dr_geom_friction[:] = self._default_geom_friction
        if self.domain_rand:
            r = self._rng
            self._dr_body_mass[self.trunk_body_id] *= r.uniform(0.9, 1.1)
            # Friction DR goes on the FEET, not the floor. MuJoCo mixes a
            # contact pair's friction by element-wise MAX unless one geom sets
            # geom_priority, and neither our floor nor our feet did — so
            # randomizing the floor down to 0.5 did nothing at all (the feet's
            # own 1.0 won) and the surface could only ever get GRIPPIER than
            # nominal. Measured: draws of 0.3/0.5/0.8/1.0 all produced an
            # effective mu of 1.0. Upstream randomizes the foot pads over
            # (0.7, 1.3) with priority=1, which is what actually varies grip.
            mu = r.uniform(0.7, 1.3)
            for gid in self.foot_geoms.values():
                self._dr_geom_friction[gid, 0] = mu
        self._sync_model()

    @property
    def model_id(self) -> int:
        """Address of the compiled model this env steps.

        The only way to observe sharing across a process boundary: fork copies
        the address space, so an inherited model keeps its address in the child
        while a fresh compile lands somewhere else. Shipping the model itself
        down a vec-env pipe to compare would defeat the purpose.
        """
        return id(self.model)

    def _sync_model(self) -> None:
        """Point the (possibly shared) model at THIS env's randomization."""
        self.model.body_mass[:] = self._dr_body_mass
        self.model.geom_friction[:] = self._dr_geom_friction

    # ------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if self.bam is not None:
                self.bam.reseed(seed)
        self._reset_episode_state()
        self._apply_domain_rand()

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        r = self._rng
        # Small pose noise + random yaw so the policy never memorizes one init.
        self.data.qpos[self.joint_qpos_adr] += r.uniform(-0.03, 0.03, C.NUM_JOINTS)
        yaw = r.uniform(-np.pi, np.pi) if self.random_yaw else 0.0
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qpos[2] += r.uniform(0.0, 0.01)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.data.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(self.model, self.data)
        if self.bam is not None:
            # After mj_forward: reset() reads qfrc_bias/qfrc_constraint-backed
            # state and primes the delay buffer at the spawn pose.
            self.bam.reset(self.data.qpos[self.joint_qpos_adr])

        self._sample_commands()
        self._action_lag = (
            int(self._rng.integers(0, 2))
            if (self.action_delay and self.bam is None) else 0
        )
        self.prev_joint_vel = self._joint_vel().copy()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self._model_shared:
            # ~0.55 us: cheaper than a second 138 MB model, and the only thing
            # standing between sibling envs and each other's body mass.
            self._sync_model()
        # last_action is the RAW policy output, NOT the clipped one — upstream
        # never clips at all (clip_actions=None), so there the two are the same
        # thing and the action-rate penalty prices whatever the network emits.
        # Storing the clipped value here decoupled them and made unbounded
        # output growth FREE: measured, a 25M-step policy reached a mean |a| of
        # 29.0 (max 140.9) with 52% of outputs saturated against the +/-4 clip,
        # while alpha_walking sits at 0.19 and never saturates. The env still
        # clips what it APPLIES — the actuator has limits — but the reward and
        # the observation now see what the policy actually asked for.
        raw = np.asarray(action, np.float32)
        self.prev_action = self.last_action
        self.last_action = raw.copy()
        # ndarray.clip is the exact call np.clip dispatches to (fromnumeric's
        # _wrapfunc), minus two wrapper layers.
        action = raw.clip(-4.0, 4.0)

        # Per-episode 0/1-step command delay, as the BAM DR models on the bus.
        # `action` is the clip result — a fresh array no caller holds — so it
        # is stored directly instead of copied.
        applied = self._delayed_action if self._action_lag else action
        self._delayed_action = action
        self.data.ctrl[:] = C.DEFAULT_POSE + applied  # action scale 1.0

        if self.bam is None:
            for _ in range(C.DECIMATION):
                mujoco.mj_step(self.model, self.data)
        else:
            # ctrl above still carries the position target (readable by viz /
            # debug code) but drives nothing: the BAM path neutralizes the MJCF
            # servos and applies its own torque per substep.
            self.bam.set_target(self.data.ctrl)
            for _ in range(C.DECIMATION):
                self.bam.before_step()
                mujoco.mj_step(self.model, self.data)
            self.bam.after_step()

        self.step_count += 1
        if self.step_count % self.resample_steps == 0:
            self._sample_commands()

        # The physics state is frozen from here to the end of the step, so
        # derived quantities memoize (see __init__). try/finally so a raising
        # reward term can never leave a stale cache armed for outside callers.
        self._step_cache.clear()
        self._cache_active = True
        try:
            obs = self._get_obs()
            reward, terms = self._compute_reward()
            sums = self.reward_sums
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v

            fell = self._projected_gravity()[2] > self.FALL_GRAVITY_Z
            if self.height_termination:
                fell = fell or self._trunk_xpos[2] < self.FALL_HEIGHT
        finally:
            self._cache_active = False
        terminated = self.terminate_on_fall and bool(fell)
        if not np.isfinite(obs).all():  # NaN guard: kill the episode, not the run
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True
        truncated = self.step_count >= self.max_steps

        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode_rewards"] = dict(self.reward_sums)
        return obs, reward, terminated, truncated, info

    # ----------------------------------------------------------- observations

    def _projected_gravity(self) -> np.ndarray:
        cache = self._step_cache if self._cache_active else None
        if cache is not None:
            g = cache.get("pgrav")
            if g is not None:
                return g
        g = C.quat_rotate_inverse(self._trunk_xquat, _NEG_Z)
        if cache is not None:
            cache["pgrav"] = g
        return g

    def _joint_qpos(self) -> np.ndarray:
        """The 14 joint angles, raw float64 (a view when contiguous)."""
        q = self._qpos_j
        return q if q is not None else self.data.qpos[self.joint_qpos_adr]

    def _joint_pos_rel(self) -> np.ndarray:
        cache = self._step_cache if self._cache_active else None
        if cache is not None:
            v = cache.get("jpos")
            if v is not None:
                return v
        v = (self._joint_qpos() - C.DEFAULT_POSE).astype(np.float32)
        if cache is not None:
            cache["jpos"] = v
        return v

    def _joint_vel(self) -> np.ndarray:
        cache = self._step_cache if self._cache_active else None
        if cache is not None:
            v = cache.get("jvel")
            if v is not None:
                return v
        q = self._qvel_j
        v = (q if q is not None
             else self.data.qvel[self.joint_qvel_adr]).astype(np.float32)
        if cache is not None:
            cache["jvel"] = v
        return v

    def body_lin_vel(self) -> np.ndarray:
        """Trunk linear velocity in the BODY frame [fwd, lat, up], m/s.

        Do NOT use ``mj_objectVelocity(..., flg_local=1)`` for this. For a
        body, MuJoCo returns that 6-vector in the COM-inertial (``ximat``)
        frame, which on this trunk is ~90° off the body frame: a measured
        world +x motion of 0.5 m/s reads back as v6[3] ≈ 0 while world +y
        of 0.5 reads as v6[3] ≈ 0.5. Tracking rewards that treated v6[3:5]
        as body-xy were therefore paying for SIDEWAYS motion — the same
        shuffle ``behaviors._base_vel`` already documents.

        Commands and ``infer_policy.py`` live in the body frame
        (``quat_rotate_inverse(xquat, qvel[0:3])``). Match them.
        """
        cache = self._step_cache if self._cache_active else None
        if cache is not None:
            v = cache.get("bvel")
            if v is not None:
                return v
        v = C.quat_rotate_inverse(self._trunk_xquat, self._qvel_base)
        if cache is not None:
            cache["bvel"] = v
        return v

    def heading_lin_vel(self) -> tuple[float, float, float]:
        """Trunk speed in the yaw-heading frame: (forward, lateral, world-z).

        Body-x mixes in a vertical component when the trunk pitches; a run
        reward that should pay for covering ground (not diving) wants the
        projection of world velocity onto the yaw-only facing instead.
        """
        cache = self._step_cache if self._cache_active else None
        if cache is not None:
            out = cache.get("hvel")
            if out is not None:
                return out
        v = self._qvel_base
        R = self._trunk_xmat.reshape(3, 3)
        fwd = R @ _E_FWD
        fwd[2] = 0.0
        n = float(np.linalg.norm(fwd))
        fwd = fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        side = np.array([-fwd[1], fwd[0], 0.0])
        out = (float(v @ fwd), float(v @ side), float(v[2]))
        if cache is not None:
            cache["hvel"] = out
        return out

    def _get_obs(self) -> np.ndarray:
        gyro = self._gyro.astype(np.float32)
        gravity = self._projected_gravity()
        joint_pos = self._joint_pos_rel()
        # 1-ctrl-step lag on joint_vel: Dynamixel present_velocity is a
        # trailing moving average (velocity env cfg does the same).
        joint_vel = self.prev_joint_vel
        self.prev_joint_vel = self._joint_vel().copy()

        if self.obs_noise:
            r = self._rng
            gyro = gyro + r.uniform(-C.NOISE_GYRO, C.NOISE_GYRO, 3).astype(np.float32)
            gravity = gravity + r.uniform(-C.NOISE_GRAVITY, C.NOISE_GRAVITY, 3).astype(np.float32)
            joint_pos = joint_pos + r.uniform(-C.NOISE_JOINT_POS, C.NOISE_JOINT_POS, C.NUM_JOINTS).astype(np.float32)
            joint_vel = joint_vel + r.uniform(-C.NOISE_JOINT_VEL, C.NOISE_JOINT_VEL, C.NUM_JOINTS).astype(np.float32)

        # Slice-assembled into one fresh 61-float allocation: concatenate's
        # temporary plus its astype copy measured ~7 us of the old ~22 us obs
        # build. A NEW array every call on purpose — workers and SB3 keep
        # references (terminal_observation) across the next reset.
        obs = np.empty(C.OBS_DIM, np.float32)
        obs[0:3] = gyro
        obs[3:6] = gravity
        obs[6:20] = joint_pos
        obs[20:34] = joint_vel
        obs[34:48] = self.last_action
        obs[48:51] = self.twist_cmd
        obs[51:55] = self.head_cmd
        obs[55:61] = self.body_cmd
        return obs

    # ---------------------------------------------------------------- rewards

    def _foot_contacts(self) -> dict[str, bool]:
        n = int(self.data.ncon)
        if n == 0:
            return {"left": False, "right": False}
        # Plain int loop over tolist'ed geom ids: ncon is small (typically
        # 2-10), where the old six-temporary vectorized masks cost more than
        # the comparisons they saved. Same booleans by construction (a foot
        # counts only when paired with the floor in the SAME contact) — held
        # bit-for-bit by test_step_perf_parity's verbatim reference.
        con = self.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = self.floor_geom
        left, right = self.foot_geoms["left"], self.foot_geoms["right"]
        lc = rc = False
        for i in range(n):
            a = g1[i]
            b = g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            if other == left:
                lc = True
            elif other == right:
                rc = True
        return {"left": lc, "right": rc}

    def _action_rate_weight(self) -> float:
        n = self._lifetime_steps
        w = self._ACTION_RATE_STAGES[0][1]
        for step, wt in self._ACTION_RATE_STAGES:
            if n >= step:
                w = wt
        return w

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        self._lifetime_steps += 1
        # Privileged sim state — fine for rewards, never for actor obs.
        # Body-frame tracking matches the twist command, infer_policy.py, and
        # mjlab's track_linear_velocity (which also folds vz into the same
        # Gaussian). Heading-frame speed is a run-behavior concern only.
        base_v = self.body_lin_vel()
        gyro = self._gyro
        gravity = self._projected_gravity()

        # `x.sum()` in place of `np.sum(x)`: same np.add.reduce, minus the
        # fromnumeric dispatch (bit-identical; pinned by the parity goldens).
        xy_err2 = float(((self.twist_cmd[:2] - base_v[:2]) ** 2).sum())
        lin_err2 = xy_err2 + float(base_v[2] ** 2)
        gyro_xy2 = float(gyro[0] ** 2 + gyro[1] ** 2)
        ang_err2 = float((self.twist_cmd[2] - gyro[2]) ** 2) + gyro_xy2
        track_lin = self.W_TRACK_LIN * np.exp(-lin_err2 / self.TRACK_STD2)
        track_ang = self.W_TRACK_ANG * np.exp(-ang_err2 / self.ANG_TRACK_STD2)

        tilt2 = float((gravity[:2] ** 2).sum())
        upright = self.W_UPRIGHT * np.exp(-tilt2 / self.UPRIGHT_STD2)

        joint_pos_rel = self._joint_pos_rel()
        pose = self.W_POSE * np.exp(
            -float((joint_pos_rel[C.LEG_JOINT_IDS] ** 2).sum()) / self.POSE_STD2
        )
        head_err = joint_pos_rel[C.HEAD_JOINT_IDS] - self.head_cmd
        # e.sum()/4 is np.mean's own reduction and division, minus its wrapper.
        head_pose = self.W_HEAD_POSE * float(
            np.exp(-((head_err / self.HEAD_STD) ** 2)).sum() / 4
        )

        # Dense air-time: GPU feet_air_time pays every step a foot's current
        # air time sits in [min, max], not a lumped touchdown bonus.
        contacts = self._foot_contacts()
        air_reward = 0.0
        moving = (float(np.linalg.norm(self.twist_cmd[:2]))
                  + abs(float(self.twist_cmd[2]))) > 0.01
        for side in ("left", "right"):
            if contacts[side]:
                self.air_time[side] = 0.0
            else:
                self.air_time[side] += C.CTRL_DT
            if moving and self.AIR_TIME_MIN < self.air_time[side] < self.AIR_TIME_MAX:
                air_reward += 1.0
            self.was_contact[side] = contacts[side]
        air_time = self.W_AIR_TIME * air_reward

        # Penalties — each term value is <= 0 by construction.
        # GPU action_rate_l2 is -weight * Σ(Δa²) with no extra 0.02 scale.
        action_rate = self._action_rate_weight() * -float(
            ((self.last_action - self.prev_action) ** 2).sum()
        )
        ang_vel_xy = self.W_ANG_VEL_XY * -gyro_xy2

        terms = {
            "track_lin_vel": track_lin, "track_ang_vel": track_ang,
            "upright": upright, "pose": pose, "head_pose": head_pose,
            "feet_air_time": air_time,
            "action_rate_penalty": action_rate,
            "ang_vel_xy_penalty": ang_vel_xy,
        }
        return float(sum(terms.values())), terms
