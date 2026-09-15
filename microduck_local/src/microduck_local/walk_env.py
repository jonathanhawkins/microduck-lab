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

`train-walk` trains on ``"bam"``; the env's own default stays ``"xml"`` (the
cheap, deployment-rehearsal physics the tricks and the lab run on).

Physics parity with the upstream velocity cfg (2026-09-06 audit): the solver
runs implicitfast / 10 / 20 (UPSTREAM_* below), the IMU blocks of the obs are
refreshed after the substep loop (`_refresh_derived`), and domain
randomization carries upstream's velocity pushes, mass+inertia scaling, CoM
offsets and armature scaling with upstream's ranges (each a constructor knob).
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
from .robots.spec import RobotSpec

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

# The mjModel fields domain randomization writes (restore-then-apply every
# reset — AGENTS.md), and the derived constants `mujoco.mj_setConst` recomputes
# from them (measured on this model: change mass/inertia/ipos/armature, call
# mj_setConst, diff every array of the model — these are the ones that moved).
# `body_subtreemass` matters to anything reading subtree_com; the invweight0
# fields scale the constraint impedance, so a mass draw that skipped them
# would be a slightly different robot from the one the solver thinks it has.
DR_MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "dof_armature",
                   "geom_friction")
SETCONST_FIELDS = ("body_subtreemass", "body_invweight0", "dof_invweight0",
                   "dof_M0", "actuator_acc0", "light_poscom0", "dof_length")

# id(model) -> the model's compile-time copy of every field above, captured
# the moment it was compiled. Domain randomization writes those arrays, so an
# env that joins a shared model AFTER a sibling has already randomized it would
# otherwise adopt the sibling's draw as its "restore to defaults" baseline and
# quietly accumulate. Only cached models are registered, and the cache holds the
# strong reference that keeps the id valid.
_PRISTINE: dict[int, dict[str, np.ndarray]] = {}

# ------------------------------------------------ upstream velocity-cfg values
#
# mjlab's velocity task (mjlab/tasks/velocity/velocity_env_cfg.py, the base of
# upstream microduck_velocity_env_cfg.py) trains on implicitfast with 10
# Newton iterations and 20 line-search iterations; the MJCF default that
# scripts/infer_policy.py (the deployment rehearsal) runs is Euler / 100 / 50.
# Measured here with alpha_walking.onnx, 500 steps with three shoves, xml AND
# bam: implicitfast alone and iterations=10 alone are BIT-IDENTICAL to the
# XML default (the solver converges in <= 6 iterations, and Euler already
# integrates joint damping implicitly on this model). Only ls_iterations=20
# moves anything, and only under BAM: 1.4e-17 in qpos at step 49 — a
# line-search cutoff ULP under the stiff DOF-friction rows — which chaos
# grows to 4e-3 by step 300; the walker falls in neither. Training parity
# wins over deployment parity: these are the numbers the shipped policies
# were optimized against, and the deployment rehearsal is unaffected to the
# bit under xml.
UPSTREAM_INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
UPSTREAM_SOLVER_ITERATIONS = 10
UPSTREAM_LS_ITERATIONS = 20

# Domain randomization ranges, read from upstream microduck_velocity_env_cfg.py
# (the sim2real recipe this env mirrors). Each is a MicroduckWalkEnv knob.
MASS_SCALE_RANGE = (0.95, 1.05)       # dr.pseudo_inertia alpha: mass AND inertia
ARMATURE_SCALE_RANGE = (0.9, 1.1)     # dr.joint_armature, per joint
PUSH_VEL_RANGE = (-0.3, 0.3)          # push_by_setting_velocity, m/s, world xy
PUSH_INTERVAL_S = (3.0, 6.0)          # interval_range_s
# body_ipos offsets, ramped by curriculum in per-env steps (upstream counts
# 24 steps/env per iteration: 500/1000/1500 iterations = 12k/24k/36k).
TRUNK_COM_STAGES = ((0, 0.003), (12_000, 0.005), (24_000, 0.010), (36_000, 0.015))
HEAD_COM_STAGES = ((0, 0.003), (12_000, 0.005), (24_000, 0.010))
# Upstream HEAD_BODY_NAMES, verbatim. Its own comment notes that bearing_roll
# is the right-hip-yaw link, not a head body, "kept only to preserve existing
# DR behavior" — mirrored as-is so the draw distribution is the shipped one.
HEAD_COM_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft",
                   "bearing_roll")


def _need(model: mujoco.MjModel, objtype, name: str, robot: str) -> int:
    """Resolve a model element by name, or say which robot wanted it.

    A missing name used to surface as -1 and then as a contact scan that
    silently never fired (id -1 matches nothing), so a renamed foot pad cost
    the air-time reward with no error anywhere.
    """
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0:
        raise KeyError(f"{robot}: model has no {objtype.name.split('_')[-1].lower()} "
                       f"named {name!r}")
    return i


def _root_adr(model: mujoco.MjModel) -> tuple[int, int]:
    """(qpos, qvel) address of the base free joint."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    raise KeyError("model has no free joint — the base cannot move")


def _staged(stages: tuple[tuple[int, float], ...], n: int) -> float:
    v = stages[0][1]
    for step, val in stages:
        if n >= step:
            v = val
    return v

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
        _PRISTINE[id(model)] = _snapshot_fields(model)
    return model


def _snapshot_fields(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    return {name: getattr(model, name).copy()
            for name in DR_MODEL_FIELDS + SETCONST_FIELDS
            if hasattr(model, name)}


def pristine_baselines(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """The model's compile-time copy of every DR-written / mj_setConst field.

    Recorded at compile time for cached models; for a privately compiled or
    caller-supplied model there is nobody else to have touched it, so reading it
    now is the same answer.
    """
    known = _PRISTINE.get(id(model))
    if known is not None:
        return known
    return _snapshot_fields(model)


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
        # The head-pose command range the policy trains under. The contract's
        # keep-alive range (+-0.05 rad) is why a walker trained here has never
        # seen a head-down command; pass the gaze poses to train one that has
        # (roadmap 4c revisit, 2026-09-07: `train-walk --head-range`).
        head_cmd_ranges: tuple | None = None,
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
        # ---- domain randomization knobs, upstream velocity cfg defaults.
        # All of them are inert when domain_rand=False.
        mass_scale_range: tuple[float, float] = MASS_SCALE_RANGE,
                                        # trunk mass AND inertia, one factor
                                        # (pseudo_inertia alpha), per reset
        armature_scale_range: tuple[float, float] = ARMATURE_SCALE_RANGE,
                                        # per-joint dof_armature factor
        trunk_com_offset_m: float | None = None,  # ±m on trunk body_ipos;
                                        # None = upstream's curriculum ladder
                                        # (TRUNK_COM_STAGES over lifetime steps)
        head_com_offset_m: float | None = None,   # same for HEAD_COM_BODIES
        push_robot: bool | None = None, # add U(push_vel_range) to the base's
                                        # world-xy velocity every
                                        # U(push_interval_s) of episode time;
                                        # None = follow domain_rand
        push_vel_range: tuple[float, float] | None = None,
        push_interval_s: tuple[float, float] = PUSH_INTERVAL_S,
        # WHICH body. None = the duck (contract.MICRODUCK) — the only robot
        # this env had before robots/spec.py, and the goldens prove the spec
        # path is bit-identical to the names it replaced.
        robot: RobotSpec | None = None,
    ):
        super().__init__()
        self.robot = robot if robot is not None else C.MICRODUCK
        self.nj = self.robot.num_joints
        self.default_pose = np.asarray(self.robot.default_pose, np.float32)
        scene = Path(scene_xml) if scene_xml else self.robot.scene_fn()
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
        # Upstream's integrator / solver budget (see UPSTREAM_* above for the
        # measurement). NOTE: scripts/infer_policy.py, the deployment
        # rehearsal, runs the XML default (Euler / 100 / 50) — a deliberate
        # difference: identical to the bit under xml, ULP-level under bam.
        self.model.opt.integrator = UPSTREAM_INTEGRATOR
        self.model.opt.iterations = UPSTREAM_SOLVER_ITERATIONS
        self.model.opt.ls_iterations = UPSTREAM_LS_ITERATIONS
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
        self.head_cmd_ranges = (tuple(tuple(map(float, r)) for r in head_cmd_ranges)
                                if head_cmd_ranges else C.HEAD_CMD_RANGES)
        self.mass_scale_range = tuple(mass_scale_range)
        self.armature_scale_range = tuple(armature_scale_range)
        self.trunk_com_offset_m = trunk_com_offset_m
        self.head_com_offset_m = head_com_offset_m
        self.push_robot = bool(domain_rand if push_robot is None else push_robot)
        # None -> the robot's own range. Defaulting the kwarg to the duck's
        # module constant made `spec.push_vel_range` unreadable: the G1
        # declared +/-0.4 for 34 kg and trained under the 0.8 kg duck's
        # +/-0.3, i.e. a knob that changed nothing.
        self.push_vel_range = tuple(self.robot.push_vel_range
                                    if push_vel_range is None else push_vel_range)
        self.push_interval_s = tuple(push_interval_s)

        # Model lookups — resolved by NAME (from the robot spec) so joint
        # reordering can't bite, and a renamed link fails here rather than
        # silently walking on the wrong geometry.
        spec = self.robot
        self.trunk_body_id = _need(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   spec.base_body, spec.id)
        self.joint_qpos_adr = np.array([
            self.model.joint(n).qposadr[0] for n in spec.joint_names
        ])
        self.joint_qvel_adr = np.array([
            self.model.joint(n).dofadr[0] for n in spec.joint_names
        ])
        # Which mjData.ctrl slots this robot's joints drive. The duck's
        # actuators are the 14 joints in model order (ctrl[:] used to be
        # written whole); anything else — the G1, whose frozen hands leave
        # gaps — is addressed by actuator name.
        self.ctrl_adr = np.array([
            _need(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n, spec.id)
            for n in self._actuator_names()])
        self._ctrl_whole = (self.model.nu == spec.num_joints
                            and np.array_equal(self.ctrl_adr,
                                               np.arange(spec.num_joints)))
        gyro = self.model.sensor(spec.gyro_sensor)
        self.gyro_adr = slice(gyro.adr[0], gyro.adr[0] + 3)
        self.key_stand = self.model.key(spec.stand_keyframe).id
        floor_id = _need(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                         spec.floor_geom, spec.id)
        self.floor_geom = floor_id
        # One id per side for the duck; the G1 carries seven capsules a foot.
        self.foot_geom_ids = {
            side: tuple(_need(self.model, mujoco.mjtObj.mjOBJ_GEOM, n, spec.id)
                        for n in names)
            for side, names in spec.foot_geoms.items()
        }
        self.foot_geoms = {side: ids[0] for side, ids in self.foot_geom_ids.items()}
        # Reverse map for the contact scan: geom id -> side.
        self._foot_side = {g: side for side, ids in self.foot_geom_ids.items()
                           for g in ids}
        # Make the FOOT's friction win the contact pair. MuJoCo mixes friction
        # by element-wise max unless one geom has the higher geom_priority, so
        # without this a foot randomized to 0.7 still contacts at the floor's
        # 1.0 and the DR knob is inert. Upstream sets priority=1 on these pads
        # for exactly this reason. Idempotent, so it is safe to set from every
        # env sharing one mjModel.
        for _ids in self.foot_geom_ids.values():
            for _gid in _ids:
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
        # The base free joint: qpos[0:7] / qvel[0:6] on both robots today,
        # but read off the model rather than assumed.
        self._root_qpos, self._root_qvel = _root_adr(self.model)
        self._qvel_base = self.data.qvel[self._root_qvel:self._root_qvel + 3]
        # The 14 joint addresses are contiguous in model order on this robot;
        # a slice view then replaces the fancy-index copy (bit-identical — the
        # same elements in the same order). The fancy-index fallback keeps a
        # hypothetical reordered model correct.
        qadr, vadr = self.joint_qpos_adr, self.joint_qvel_adr
        nj = self.nj
        self._qpos_j = (
            self.data.qpos[qadr[0]:qadr[0] + nj]
            if np.array_equal(qadr, np.arange(qadr[0], qadr[0] + nj))
            else None)
        self._qvel_j = (
            self.data.qvel[vadr[0]:vadr[0] + nj]
            if np.array_equal(vadr, np.arange(vadr[0], vadr[0] + nj))
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
        # `_dr_fields` holds THIS env's draw (plus the mj_setConst outputs it
        # implies) for every field in DR_MODEL_FIELDS + SETCONST_FIELDS.
        self._defaults = {k: v.copy()
                          for k, v in pristine_baselines(self.model).items()}
        self._dr_fields = {k: v.copy() for k, v in self._defaults.items()}
        # Named views kept for the tests and tools that read them.
        self._default_body_mass = self._defaults["body_mass"]
        self._default_geom_friction = self._defaults["geom_friction"]
        self._dr_body_mass = self._dr_fields["body_mass"]
        self._dr_geom_friction = self._dr_fields["geom_friction"]
        # CoM-offset targets. A name a NON-DUCK spec declares must exist —
        # that is the point of the spec being names rather than indices, and
        # `com_bodies=("torso_link",)` was silently dropping to an empty list
        # (the G1's body is `torso_link_rev_1_0`), so the G1's CoM
        # randomization ran as a no-op with nothing anywhere saying so.
        #
        # The DUCK stays lenient, on the same `is C.MICRODUCK` identity test
        # this file already uses for the fall thresholds. Its list is
        # upstream's head assembly and genuinely spans scenes that do not all
        # carry every body — making it strict (which listing the same five
        # names on its spec quietly did) would newly raise at construction on
        # any trimmed model or custom `scene_xml`.
        if self.robot is C.MICRODUCK:
            ids = [bid for bid in (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in (self.robot.com_bodies or HEAD_COM_BODIES))
                if bid >= 0]
        else:
            ids = [_need(self.model, mujoco.mjtObj.mjOBJ_BODY, n, self.robot.id)
                   for n in self.robot.com_bodies]
        self._head_com_body_ids = np.array(ids, dtype=int)

        # Nominal standing trunk height, measured off the model itself (never
        # hand-carried across model revisions — AGENTS.md).
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_stand)
        mujoco.mj_forward(self.model, self.data)
        self.stand_z = float(self.data.xpos[self.trunk_body_id][2])

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.robot.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(
            -self.robot.action_clip, self.robot.action_clip, (self.nj,), np.float32)

        self._rng = np.random.default_rng(seed)
        # Class constants stay the duck's documented defaults; a spec that
        # names its own wins (a 1.3 m humanoid's floor is not 0.07 m).
        self._fall_gravity_z = (self.robot.fall_gravity_z
                                if self.robot is not C.MICRODUCK else self.FALL_GRAVITY_Z)
        self._fall_height = (self.robot.fall_height
                             if self.robot is not C.MICRODUCK else self.FALL_HEIGHT)
        self._pose_ids = (self.robot.pose_joint_ids
                          if self.robot.pose_joint_ids is not None
                          else np.arange(self.nj))

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
                self.model, self.data, self.robot.joint_names,
                dt=C.PHYSICS_DT,
                rng=np.random.default_rng(seed),
                delay_min_lag=3 if action_delay else 0,
                delay_max_lag=6 if action_delay else 0,
                friction_scale_range=(
                    DEFAULT_FRICTION_SCALE_RANGE if domain_rand else None
                ),
                current_scale=bam_current_scale,
            )
            # BAM retuned dof_armature to its own identified value and ran
            # mj_setConst; that, not the MJCF's number, is what DR restores
            # to. A BAM env never shares a model in-process (raised above),
            # so the model right now IS this env's pristine baseline.
            self._defaults = _snapshot_fields(self.model)
            self._dr_fields = {k: v.copy() for k, v in self._defaults.items()}
            self._default_body_mass = self._defaults["body_mass"]
            self._default_geom_friction = self._defaults["geom_friction"]
            self._dr_body_mass = self._dr_fields["body_mass"]
            self._dr_geom_friction = self._dr_fields["geom_friction"]

        self._reset_episode_state()

    # ------------------------------------------------------------------ state

    def _actuator_names(self) -> tuple[str, ...]:
        """Actuator names driving `robot.joint_names`, in that order.

        Both bodies name an actuator after the joint it drives; a robot that
        does not can override this."""
        return tuple(self.robot.joint_names)

    def _reset_episode_state(self) -> None:
        self.step_count = 0
        self.last_action = np.zeros(self.nj, dtype=np.float32)
        self.prev_action = np.zeros(self.nj, dtype=np.float32)
        self.prev_joint_vel = np.zeros(self.nj, dtype=np.float32)
        self.twist_cmd = np.zeros(3, dtype=np.float32)
        self.head_cmd = np.zeros(4, dtype=np.float32)
        self.body_cmd = np.zeros(6, dtype=np.float32)
        self.air_time = {"left": 0.0, "right": 0.0}
        self.was_contact = {"left": True, "right": True}
        self._action_lag = 0
        self._delayed_action = np.zeros(self.nj, dtype=np.float32)
        self.reward_sums: dict[str, float] = {}
        # Velocity pushes: control steps until the next one (None = off),
        # plus a readout of what landed, for tests and the viewer.
        self._push_countdown: int | None = None
        self.push_count = 0
        self.last_push = np.zeros(2)

    def _sample_commands(self) -> None:
        r = self._rng
        u = r.uniform()
        stand = self.zero_command_prob
        turn = stand + self.turn_in_place_prob
        fwd = turn + self.forward_command_prob
        if u < stand:
            self.twist_cmd[:] = 0.0
        elif u < turn:
            self.twist_cmd[:] = (0.0, 0.0, r.uniform(*self.robot.ang_vel_z_range))
        elif u < fwd:
            # mjlab rel_forward_envs: |vx| clamped to >= 0.3, vy = wz = 0.
            vx = abs(float(r.uniform(*self.robot.lin_vel_x_range)))
            self.twist_cmd[:] = (max(vx, self.robot.min_forward_cmd), 0.0, 0.0)
        else:
            self.twist_cmd[:] = (
                r.uniform(*self.robot.lin_vel_x_range),
                r.uniform(*self.robot.lin_vel_y_range),
                r.uniform(*self.robot.ang_vel_z_range),
            )
        self.head_cmd[:] = [r.uniform(lo, hi) for lo, hi in self.head_cmd_ranges]
        self.body_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.BODY_CMD_RANGES]

    def _apply_domain_rand(self) -> None:
        # Restore compile-time defaults, then apply — never accumulate.
        # The draw lands in this env's OWN arrays first: with a shared mjModel
        # the model is not a safe place to keep it, because a sibling env's
        # reset would retune this env's physics mid-episode.
        for name, base in self._defaults.items():
            self._dr_fields[name][:] = base
        if self.domain_rand:
            r = self._rng
            f = self._dr_fields
            trunk = self.trunk_body_id
            # Trunk mass + inertia together (upstream dr.pseudo_inertia with
            # alpha_range = (ln lo / 2, ln hi / 2): both scale by e^(2 alpha),
            # CoM untouched). Mass-only scaling was a different robot from
            # the one the solver thinks it has — a denser trunk, not a
            # heavier one. Upstream draws this once per env at startup; per
            # reset here is the same distribution with more coverage.
            lo, hi = self.mass_scale_range
            scale = float(np.exp(2.0 * r.uniform(np.log(lo) / 2.0,
                                                 np.log(hi) / 2.0)))
            f["body_mass"][trunk] *= scale
            f["body_inertia"][trunk] *= scale
            # Friction DR goes on the FEET, not the floor. MuJoCo mixes a
            # contact pair's friction by element-wise MAX unless one geom sets
            # geom_priority, and neither our floor nor our feet did — so
            # randomizing the floor down to 0.5 did nothing at all (the feet's
            # own 1.0 won) and the surface could only ever get GRIPPIER than
            # nominal. Measured: draws of 0.3/0.5/0.8/1.0 all produced an
            # effective mu of 1.0. Upstream randomizes the foot pads over
            # (0.7, 1.3) with priority=1, which is what actually varies grip.
            mu = r.uniform(0.7, 1.3)
            for gids in self.foot_geom_ids.values():
                for gid in gids:
                    f["geom_friction"][gid, 0] = mu
            # CoM offsets (dr.body_ipos, operation="add"): the trunk, and the
            # head assembly per body. Range follows upstream's curriculum
            # over lifetime steps unless pinned by the knob.
            rt = (self.trunk_com_offset_m if self.trunk_com_offset_m is not None
                  else _staged(TRUNK_COM_STAGES, self._lifetime_steps))
            f["body_ipos"][trunk] += r.uniform(-rt, rt, 3)
            rh = (self.head_com_offset_m if self.head_com_offset_m is not None
                  else _staged(HEAD_COM_STAGES, self._lifetime_steps))
            ids = self._head_com_body_ids
            f["body_ipos"][ids] += r.uniform(-rh, rh, (len(ids), 3))
            # Reflected rotor inertia (dr.joint_armature, scale, per joint).
            # Under BAM the baseline is BAM's own armature (see __init__).
            lo, hi = self.armature_scale_range
            f["dof_armature"][self.joint_qvel_adr] *= r.uniform(
                lo, hi, self.nj)
            # Land the draw, then let MuJoCo recompute what depends on it
            # (body_subtreemass, the invweight0 impedance scalings, dof_M0
            # ...). mj_setConst poses `data` at qpos0 as scratch; reset()
            # re-poses right after. The recomputed constants join this env's
            # draw so a shared model can be re-pointed at them per step.
            self._sync_model()
            mujoco.mj_setConst(self.model, self.data)
            for name in SETCONST_FIELDS:
                if name in f:
                    f[name][:] = getattr(self.model, name)
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
        model = self.model
        for name, value in self._dr_fields.items():
            getattr(model, name)[:] = value

    # -------------------------------------------------------- perturbations

    def _draw_push_steps(self) -> int:
        lo, hi = self.push_interval_s
        return max(1, int(round(self._rng.uniform(lo, hi) / C.CTRL_DT)))

    def _push(self) -> None:
        """Upstream push_by_setting_velocity: add U(range) to the base's
        world-frame xy velocity (free-joint qvel[0:2] IS world linear
        velocity). Applied after the physics substeps and before the
        observation, where mjlab's interval events fire."""
        lo, hi = self.push_vel_range
        dv = self._rng.uniform(lo, hi, 2)
        self.data.qvel[self._root_qvel:self._root_qvel + 2] += dv
        self.last_push[:] = dv
        self.push_count += 1
        self._push_countdown = self._draw_push_steps()

    def _refresh_derived(self) -> None:
        """Bring the position/velocity-dependent readouts up to the state
        the substep loop actually left.

        `mj_step` integrates AFTER it computed kinematics, cvel and sensors,
        so once the loop ends `xpos`/`xquat`/`sensordata` describe the state
        one substep (5 ms) BEFORE `qpos`/`qvel` — the joint blocks of the
        obs were fresh and the IMU blocks were not (measured on the shipped
        walker: gyro up to 0.77 rad/s off against a ±0.03 noise band,
        projected gravity 0.0098 against ±0.01). Measured in situ, this is
        the cheapest call set that makes gyro, projected gravity and trunk
        height equal a full `mj_forward` to the bit: 3.5 us against 12.8 for
        `mj_step1` and 18.8 for `mj_forward` (the whole step is ~95 us).
        It deliberately touches nothing the actuator reads: `mj_step1` would
        rebuild the constraint rows without solving them, handing the BAM
        friction-budget scan `efc_force` values from a different row layout.
        The contact list (`_foot_contacts`) keeps its substep-old snapshot,
        exactly as before.
        """
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_sensorVel(m, d)

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
        self.data.qpos[self.joint_qpos_adr] += r.uniform(-0.03, 0.03, self.nj)
        yaw = r.uniform(-np.pi, np.pi) if self.random_yaw else 0.0
        rq = self._root_qpos
        self.data.qpos[rq + 3:rq + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qpos[rq + 2] += r.uniform(0.0, 0.01)
        self.data.qvel[:] = 0.0
        self._write_ctrl(self.data.qpos[self.joint_qpos_adr])
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
        # mjlab samples an interval term's first firing at reset, then again
        # after each firing.
        self._push_countdown = self._draw_push_steps() if self.push_robot else None
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
        clip = self.robot.action_clip
        action = raw.clip(-clip, clip)

        # Per-episode 0/1-step command delay, as the BAM DR models on the bus.
        # `action` is the clip result — a fresh array no caller holds — so it
        # is stored directly instead of copied.
        applied = self._delayed_action if self._action_lag else action
        self._delayed_action = action
        # target = default + action * scale (the duck's contract is scale 1.0)
        self._write_ctrl(self.robot.scale_action(applied))

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

        if self._push_countdown is not None:
            self._push_countdown -= 1
            if self._push_countdown <= 0:
                self._push()
        # Obs, rewards and the fall check below all read xquat / sensordata
        # / xpos: make them describe the integrated state (see the method).
        # Before `after_step`, which publishes the BAM torque into
        # actuator_force — a readout no forward call here recomputes.
        self._refresh_derived()
        if self.bam is not None:
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

            fell = self._projected_gravity()[2] > self._fall_gravity_z
            if self.height_termination:
                fell = fell or self._trunk_xpos[2] < self._fall_height
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

    def _write_ctrl(self, target: np.ndarray) -> None:
        """Position targets -> mjData.ctrl, in actuator order.

        `ctrl[:] = ...` whenever the robot's joints ARE every actuator in
        order (the duck: bit-identical to what this used to be), indexed
        otherwise (the G1's frozen hands leave its actuator list shorter
        than its joint list is wide)."""
        if self._ctrl_whole:
            self.data.ctrl[:] = target
        else:
            self.data.ctrl[self.ctrl_adr] = target

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
        v = (self._joint_qpos() - self.default_pose).astype(np.float32)
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
            joint_pos = joint_pos + r.uniform(-C.NOISE_JOINT_POS, C.NOISE_JOINT_POS, self.nj).astype(np.float32)
            joint_vel = joint_vel + r.uniform(-C.NOISE_JOINT_VEL, C.NOISE_JOINT_VEL, self.nj).astype(np.float32)

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
        sides = self._foot_side          # geom id -> "left" / "right"
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
            side = sides.get(other)
            if side == "left":
                lc = True
            elif side == "right":
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
            -float((joint_pos_rel[self._pose_ids] ** 2).sum()) / self.POSE_STD2
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
