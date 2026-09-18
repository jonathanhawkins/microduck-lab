"""The first thing MARS can be TRAINED to do: reach a point with the arm.

`docs/mars-roadmap.md` Phase 4. The design decision behind this file is §1's:
MARS does not walk, so `MicroduckWalkEnv` is not one kwarg away — its fields
are foot geoms, fall heights, air-time windows and a gyro, and a wheeled base
has none of them. `MarsArmEnv` is therefore its OWN `gym.Env`, and what it
shares with the walking env is the trainer's contract rather than its code:
the same constructor kwargs `train.env_kwargs_from_args` produces, the same
`info["episode_rewards"]` the lab's teach panel graphs, the same
`observation_space` / `action_space` / `reset(seed=)` shape SB3 needs.

**The observation is the declared `mars-arm-32-v1` contract, filled.** The
slot table has existed since Phase 2a (`robots/mars.OBS_*`,
`MarsBody.contract()`) precisely so the first env would have to fill a layout
it did not get to choose. `_SLOTS` below is asserted against
`MarsBody.contract().slots` at construction, name by name and width by width:
a builder that drifts from the published table is a policy whose floats mean
something other than what its own file says they mean, which is the whole
failure `robots/policy_contract.py` exists to stop.

**Actuators: Innate's PD for every joint, and the run says so.** Joints 4-6
and the head are XL330s — the servo this repo's BAM model was identified on —
but joints 1-3 are XL430/XC430 with no fit here, so v1 drives the whole arm
with Innate's own position PD (`mars.arm_servo`, KP 50 / KD 1) and `bam` is
REFUSED rather than applied to three joints it does not describe
(`docs/mars-roadmap.md` §6.6: per-joint servo models are the eventual fix).
Pretending one switch covers the arm is the thing to avoid.

**Both halves of the actuation go through `MarsDriver`**, the same object a
`/sim` room steps (`robots/mars_drive.py`): the base's velocity PD through
`xfrc_applied` and the arm/head position PD through `qfrc_applied`, with
Innate's station keeping, watchdog and joint2 guard. The env therefore cannot
disagree with the room about what a command does — which is the one thing that
would make a policy trained here useless the moment it was dropped into the
playroom.

MEASURED before a line of the reward was written (the "check a knob's
reachable set first" rule; `scratchpad/probe_arm*.py`, 2026-09-17):

  * **the arm's HOME is not in the middle of its range.** joint1 sits at
    +1.445 rad with its stop at +1.571, so the arm parks folded across the
    chassis pointing at +83 deg of yaw, and the front arc this task samples
    is 1.4-2.5 rad of joint1 travel AWAY from where every episode starts.
  * **a shell solution costs up to 3.09 rad on its worst joint** (median
    2.01, p95 2.6-3.0 over three windows, 60 targets each, solved by
    coordinate descent over the full ranges with self-colliding poses
    rejected). That is where `ACTION_SCALE_RAD = 3.0` comes from: a smaller
    box provably excludes the poses the task needs, and at 3.0 an action of
    +-1 spans 82-127% of every joint's one-sided headroom from HOME, so
    "+-1 spans most of the range" is true joint by joint.
  * **69-85% of poses in that box self-collide**, dominated by
    `base_link <-> link2 / link5 / link4` — the arm folding down onto its own
    chassis. Innate's joint2 guard is already applied and removes only ~10
    points of it. So the self-collision termination is not a rare safety net,
    it is the environment's dominant early dynamic, and the episode-length
    column is where to read it.
  * **nothing self-collides at HOME** (2 s of `arm_servo` hold: the only
    contacts are floor/chassis and the two wheels), so the termination does
    not fire at reset — the terminal equivalent of AGENTS.md's "check the
    term is not FLAT where the policy starts".
  * **the shell is the arm's**: residual |ee - target| over the planned shell
    is 0.9 mm median, 13 mm p90, 62 mm max, with 87% of draws solved inside
    1 cm and 95% inside 2 cm. The tail is the solver's local minima as much
    as the geometry's, and it is why the acceptance bar is quoted as a count
    rather than assumed to be 8/8 (see the report for this phase).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .. import contract as C
from . import mars
from .mars_drive import MAX_CMD_LINEAR, MAX_CMD_YAW, MarsDriver

#: Physics steps per control step. MARS's contract ticks at 25 Hz (Innate's
#: policy-defined skills do, `mars.CONTROL_HZ`), and the world runs at
#: `C.PHYSICS_DT` = 5 ms, so 8 — where the duck's `C.DECIMATION` is 4 at
#: 50 Hz. Computed, not typed, so a change to either constant cannot leave a
#: policy trained at one rate and stepped at another.
DECIMATION = int(round(1.0 / (mars.CONTROL_HZ * C.PHYSICS_DT)))
CTRL_DT = C.PHYSICS_DT * DECIMATION

#: `target = ARM_HOME + action[0:6] * ACTION_SCALE_RAD`, clipped to each
#: joint's URDF range. 3.0 rad: see the module docstring's measurement — a
#: shell solution costs up to 3.09 rad on its worst joint, and the one-sided
#: headroom from HOME is 3.02 / 2.61 / 3.09 / 2.37 / 1.66 / 0.87 rad for
#: joints 1-6, so +-1 covers 82-127% of it. The clip is what keeps the box
#: legal; a joint whose headroom is smaller than the scale simply saturates,
#: exactly as the duck's +-4 action box saturates against its own ranges.
ACTION_SCALE_RAD = 3.0
#: The action box. +-1, so the scale above reads as "what +-1 means in rad"
#: and the base pair is a fraction of the command envelope.
ACTION_CLIP = 1.0

#: How fast the COMMANDED target may travel, rad/s per joint.
#:
#: This is the physics ladder, and it exists because the first version of this
#: env was unlearnable in a way the reward could never have fixed. MEASURED
#: (`scratchpad/probe_random.py`, a uniform random policy, 60 episodes): with
#: the target applied instantly, **60 of 60 episodes ended in a
#: self-collision inside 1-5 control steps** (mean episode length 1.6 of 200),
#: 53 of them `base_link <-> link2`. The cause is not the policy and not the
#: pay: an absolute joint target applied at 25 Hz is a 3 rad teleport every
#: 40 ms, and the arm's own chassis sits between HOME and the front arc this
#: task samples, so the servo drove straight through it on the first step. No
#: term, weight or ramp teaches a skill that no rollout ever contains
#: (AGENTS.md, "Reward design cannot fix an exploration gap") — the fix is the
#: world.
#:
#: And the world was WRONG, which is what makes this a ladder rung rather
#: than a crutch. Innate's `arm_servo` is KP 50 with a 50 N*m clamp on links
#: whose URDF inertias are 0.001 placeholders, so the sim's arm accelerates
#: far harder than the Dynamixels do. `[datasheet, not measured here]` the
#: slowest joints are the XL430-W250-T shoulder pair at 61 rpm no-load
#: (6.39 rad/s at 12 V); the XL330 wrist and gripper are ~104 rpm
#: (10.9 rad/s). 6.0 rad/s is the slow pair, rounded down, applied to every
#: joint — so one control step may move a target 0.24 rad, which is what the
#: real arm could follow. The ACTION still means an absolute joint target
#: (`mars-arm-32-v1`'s 8 actions are unchanged); what is bounded is how fast
#: the env walks its commanded target there, and `arm_qpos` in the
#: observation is where the arm actually got to.
#:
#: NOT in `MarsDriver`: a brain in a `/sim` room can still command a step
#: target, because Phase 3a's drive numbers are measured against that
#: behaviour and a rate limit there would change them. Moving it down into the
#: driver as an optional default-off knob (the way `RayFan` grew
#: `exclude_body=`) is Phase 3b/4b work, and the day MARS has an `Intent.arm`
#: channel it should be shared rather than copied.
MAX_TARGET_RATE_RAD_S = 6.0

#: The reach task's target shell, in the BASE frame, relative to the SHOULDER
#: (joint1's anchor, measured off the model at HOME — never hand-carried).
#: The plan's numbers, kept because the measurement says the arm owns them:
#: 0.15-0.40 m of the robot's 0.40 m reach, the front arc, and floor to
#: chassis-top in height.
REACH_RADIUS_M = (0.15, 0.40)
REACH_YAW_RAD = (-math.pi / 3.0, math.pi / 3.0)     # +-60 deg of the base's +x
REACH_HEIGHT_M = (0.05, 0.35)
#: A target whose radius is almost all vertical leaves no horizontal offset to
#: aim at, so it is redrawn rather than collapsed onto the shoulder's axis.
REACH_MIN_HORIZ_M = 0.02

#: Success: the effector inside this ball of the target...
SUCCESS_RADIUS_M = 0.02
#: ...for the last second of the episode. A HOLD, not a fly-by: a reach that
#: touches 2 cm on its way past has not done the task, and the proximity
#: bonus below is per-step for the same reason.
SUCCESS_HOLD_S = 1.0
EPISODE_S = 8.0

# ---------------------------------------------------------------- the reward
#
# Progress-pay, per AGENTS.md: "Dense, bounded shaping beats sparse
# windfalls", and a stalled arm must earn NOTHING. So the earner is the CHANGE
# in distance (a telescoping sum: an episode's total is `d_start - d_end`
# whatever route it took, so there is no way to farm it by oscillating) plus a
# bounded per-step bonus for being there.
#: Per metre closed. 40 x the ~0.25 m a typical episode has to close is ~10
#: reward, which is the scale the penalties below are sized against.
W_PROGRESS = 40.0
#: Per step inside the ball: `W_NEAR * exp(-(d/NEAR_SIGMA)^2)`. Smooth, so
#: the last centimetre has a gradient, and effectively confined to the
#: success radius (it pays 0.37 of itself at 2 cm, 0.018 at 4 cm, 1e-4 at
#: 6 cm). Per step, so HOLDING pays: 175 held steps of an 8 s episode are
#: worth ~350, which is what makes an early termination expensive without
#: any explicit penalty having to price it.
W_NEAR = 2.0
NEAR_SIGMA_M = SUCCESS_RADIUS_M
#: Penalties, sized so the task term dominates where the policy STARTS —
#: AGENTS.md's "a ramped penalty can eat its task", which cost the G1 idle a
#: face-plant at ep_rew -92. Neither of these ramps. MEASURED per episode
#: (`scratchpad/probe_random.py`, 60 episodes of a uniform random policy, and
#: the trained v2 export over 8 seeds):
#:
#:                     random      trained
#:     reach_progress   -0.29       +11.16
#:     at_target        +0.00      +109.82
#:     action_rate      -0.72        -3.95
#:     joint_vel        -0.67        -0.80
#:
#: So the two together are 12% of the ~+11.6 a full 0.29 m reach pays and
#: 4% of what a held one earns — present enough to price a thrashing arm,
#: nowhere near enough to make standing still the better deal.
ACTION_RATE_W = 0.002
JOINT_VEL_W = 0.0005
#: Per unit of `sum(action^2)`. The term Phase 4a's FIRST run measured its way
#: into, and it is a control-authority fix rather than a reward preference.
#:
#: MEASURED on `mars-reach-v1` (1.5M steps, the deterministic export over 8
#: seeds): the arm reaches — best distance in an episode 1.1 cm median — and
#: then HOVERS, ending at 2.6 cm median with 0 of 8 seeds holding the 2 cm
#: ball for the required second. The mechanism is on the same line of the
#: probe: **64% of the six arm-action dims sit on the box edge**, the exported
#: mean's magnitude reaches **|a| = 90 inside a +-1 box**, and the effector
#: describes a 9.5 mm limit cycle 3.0 cm from the target. A policy whose mean
#: is saturated has only bang-bang authority: with the target rate-limited it
#: can slew up or slew down and it has no "stay", so it chatters across the
#: ball instead of settling in it.
#:
#: Nothing priced the MAGNITUDE. `ACTION_RATE_W` prices the raw output's
#: change, which is walk_env's rule and right — but once the mean saturates,
#: a 1.4% relative wobble on a magnitude of 87 is a 120% swing in what the
#: arm is ACTUALLY commanded, so the penalty decouples from the behaviour it
#: was meant to price. Pricing the clipped action instead would not fix it
#: either: a constant saturated action has zero rate, so the arm would simply
#: park on a joint stop.
#:
#: A quadratic cost against a bounded benefit puts the optimum at a modest
#: |a|, which is the whole point — it buys back the interior of the box, where
#: "hold still" is expressible. Sized so it is negligible where the task
#: needs full authority and prohibitive where the mean ran to: at |a| = 1 on
#: every dim it costs -0.016/step (-3.2 an episode, 3.6% of the +88 `at_target`
#: this policy already earns), and at |a| = 87 it costs -121/step.
#:
#: **VERDICT, and it is only half a win.** `mars-reach-v2`, the identical
#: recipe plus this term: it DID stop the runaway (the trained magnitude is
#: now |a| ~ 0.84 a dim against 87), and every accuracy number improved —
#: final distance 2.62 -> 2.02 cm median, best-in-episode 1.10 -> 0.75 cm,
#: seeds ending inside 2 cm 1/8 -> 3/8. It did NOT fix the hold: still 0/8 on
#: the task's own rule, with 58% of dims on the box edge (from 64%) and the
#: limit cycle unchanged at 11.8 mm. So the magnitude was A cause and not THE
#: cause, and 4b needs resolution near the target rather than a smaller mean —
#: `docs/mars-roadmap.md` Phase 4a names the three candidates.
ACTION_MAG_W = 0.002
#: How far two of MARS's own bodies must OVERLAP before it counts as the arm
#: hitting itself. Not zero, and the reason is the URDF's, not the policy's.
#:
#: mars.urdf's collision volumes are hand-measured boxes over a printed
#: chassis and five arm links, and they are KNOWN to overlap in poses the real
#: arm allows: `mars.JOINT2_GUARD_MIN` exists because "the simplified
#: collision boxes still overlap ~9 mm there", which is Innate's own note
#: about their own model. So a predicate of "any contact" terminates on the
#: model's error as readily as on the robot's mistake.
#:
#: MEASURED (`scratchpad/probe_depth.py`, 3000 poses a row): HOME itself is
#: clean, and so is everything within +-0.05 rad of it. At +-0.10 rad **15.3%
#: of poses report a contact and 0.0% of them are deeper than 5 mm** — every
#: one is `link1 <-> link3` or `link2 <-> link4`, links two apart in a chain
#: folded tight at HOME, at 2.2 mm median depth. Drive the arm properly into
#: the chassis and the overlap is 20-60 mm. So there are two populations, they
#: are two orders of magnitude apart, and 10 mm — the documented artifact,
#: rounded up — separates them: it fires on nothing within a tenth of a radian
#: of the pose every episode starts in, which is AGENTS.md's "check the term
#: is not flat where the policy starts" asked of a terminal.
SELF_COLLISION_DEPTH_M = 0.010
#: The arm hitting itself ends the episode. The number only has to MARK the
#: event: termination already forfeits the rest of the hold bonus (up to
#: ~350), so the forfeited future is the real price and a large explicit
#: penalty would only teach the policy to stop moving. <= 0 by construction,
#: and it rides in a `*_penalty` key so `train._penalty_sign_callback_cls`
#: watches it.
SELF_COLLISION_PENALTY = -2.0

#: What tasks this env implements. `MarsBody.env_class` is the authority a
#: caller should ask; this is what its refusal names.
TASKS: tuple[str, ...] = ("reach",)


class MarsArmEnv(gym.Env):
    """Reach a sampled point with MARS's gripper. Its own base class.

    One compiled MARS scene, a private `MjData`, one `MarsDriver`, and the
    32-float contract as its observation. See the module docstring for the
    measurements every constant rests on.

    **The model is `mars.model()` — shared, read-only, one compile per
    process.** Nothing in this env writes to an `MjModel`: v1 randomises the
    TARGET and nothing about the robot, so there is no mass, friction or
    armature to restore and no `walk_env.shared_model` bookkeeping to do. That
    is why sharing is safe here where `MicroduckWalkEnv` has to refuse it
    under BAM (the BAM actuator rewrites `dof_frictionloss` every substep, so
    siblings in one process would overwrite each other's servo physics).
    Sharing buys ~0.2 s of STL parsing per env, which is 6 s of a 32-env
    fleet's startup; the fork-based vec env compiles once per worker anyway.
    `own_model=True` compiles a private one for a caller that ever needs to
    mutate the model — Phase 4b's `pick` will, once there is a block in the
    scene whose mass is worth randomising.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "reach",
        *,
        max_episode_s: float = EPISODE_S,
        seed: int | None = None,
        # --- the trainer's shared knobs (train.env_kwargs_from_args) -------
        # Accepted so `train-walk --robot mars` works with the flags every
        # body's trainer already passes. The ones with no meaning here are
        # ignored rather than silently reinterpreted, each with its reason:
        domain_rand: bool = True,
        obs_noise: bool = True,
        action_delay: bool = False,
        random_yaw: bool = False,
        push_robot: bool = False,
        actuator: str | None = None,
        actuator_force: str | None = None,
        # --- this env's own ------------------------------------------------
        own_model: bool = False,
        spawn_yaw: float = 0.0,
        use_base: bool | None = None,
    ):
        if task not in TASKS:
            raise SystemExit(
                f"unknown --task {task!r} for mars (have: {', '.join(TASKS)}) — "
                "pick/place are later rungs of docs/mars-roadmap.md Phase 4")
        for name, value in (("actuator", actuator),
                            ("actuator_force", actuator_force)):
            if value is not None and str(value).lower() == "bam":
                raise ValueError(
                    f"{name}='bam' is this repo's XL330 identification — it "
                    "describes MARS's joints 4-6 and head and NOT its "
                    "XL430/XC430 shoulder (joints 1-3). v1 drives every joint "
                    "with Innate's own position PD (mars.arm_servo); "
                    "docs/mars-roadmap.md §6.6 is the per-joint fix")
        self.task = task
        # Recorded, not applied. `obs_noise` has no v1 meaning: nothing here
        # has measured an encoder sigma for these servos or a detector sigma
        # for the head camera, and inventing one would be the "model both
        # halves of an error" mistake with no number behind it. `action_delay`
        # and `push_robot` are the walking env's DR, off in every task recipe.
        # `domain_rand` has nothing to randomise while DR v1 is the target
        # alone — which is sampled on every reset either way.
        self.obs_noise = bool(obs_noise)
        self.action_delay = bool(action_delay)
        self.push_robot = bool(push_robot)
        self.domain_rand = bool(domain_rand)
        self.random_yaw = bool(random_yaw)
        self.spawn_yaw = float(spawn_yaw)
        # The base is DISABLED for `reach`: it is an arm task, and letting the
        # policy drive would let it solve a reach by rolling the whole robot
        # at the target — the cheapest behaviour that satisfies the terms, and
        # not the one being taught. The contract keeps its 8 actions (a fixed
        # layout, zero-padded, exactly as the duck's 61 floats are), so the
        # base pair is read, forced to zero, and reported in `info`.
        self.use_base = (task != "reach") if use_base is None else bool(use_base)

        self.model = (mujoco.MjModel.from_xml_path(str(mars.scene_xml()))
                      if own_model else mars.model())
        self.data = mujoco.MjData(self.model)
        self.driver = MarsDriver(self.model, "")
        self.dt = CTRL_DT
        self.max_steps = int(round(float(max_episode_s) / CTRL_DT))
        self.hold_steps = int(round(SUCCESS_HOLD_S / CTRL_DT))

        m = self.model
        self._arm_qadr = np.array([m.joint(j).qposadr[0]
                                   for j in mars.ARM_JOINTS], int)
        self._arm_dadr = np.array([m.joint(j).dofadr[0]
                                   for j in mars.ARM_JOINTS], int)
        self._head_qadr = int(m.joint(mars.HEAD_JOINT).qposadr[0])
        self._grip_dadr = int(m.joint("joint6").dofadr[0])
        self._ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                          mars.EFFECTOR_BODY)
        if self._ee_body < 0:                       # never id -1, per spec.py
            raise KeyError(f"no body {mars.EFFECTOR_BODY!r} in the MARS model "
                           "— the reach reward has nothing to measure")
        self._home = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])
        self._jnt_lo = np.array([m.joint(j).range[0] for j in mars.ARM_JOINTS])
        self._jnt_hi = np.array([m.joint(j).range[1] for j in mars.ARM_JOINTS])
        #: rad the commanded target may move in one control step.
        self.target_step_rad = MAX_TARGET_RATE_RAD_S * CTRL_DT
        self._cmd_target = self._home.copy()

        # Every body of THIS robot, for the self-collision scan. A set rather
        # than "not the world", so the same predicate works unchanged when a
        # later task puts a block or a basket in the scene.
        self._robot_bodies = frozenset(
            b for b in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            not in ("world",))
        self._finger_bodies = frozenset(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in mars.FINGER_LINKS)

        # The shoulder, in the BASE frame, measured off the model at HOME.
        # The task's shell is defined from here, so a URDF revision that moves
        # the arm mount moves the shell with it instead of aiming the task at
        # a point in space the arm no longer starts near.
        probe = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, probe, m.key(mars.HOME_KEY).id)
        mujoco.mj_forward(m, probe)
        self.shoulder_base = np.array(probe.xanchor[m.joint("joint1").id]).copy()

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (mars.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(
            -ACTION_CLIP, ACTION_CLIP, (mars.NUM_ACTIONS,), np.float32)

        # The published layout, checked against this builder. `_SLOTS` is the
        # order `_get_obs` writes in; the contract is what every reader of an
        # exported file resolves. Two tables that must agree is exactly the
        # duplication that drifts, so they are compared here rather than in a
        # test alone — a mismatch is a construction error on the first env.
        self._check_contract()

        self._rng = np.random.default_rng(seed)
        self.target = np.zeros(3)
        self.target_base = np.zeros(3)
        self.last_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.reward_sums: dict[str, float] = {}
        self.step_count = 0
        self._near_streak = 0
        self._prev_dist = 0.0
        self._self_hit: tuple[str, str] | None = None
        self.reset(seed=seed)

    # ------------------------------------------------------------- contract

    #: (contract slot name, slice) in the order `_get_obs` fills them.
    _SLOTS: tuple[tuple[str, slice], ...] = (
        ("arm_qpos", mars.OBS_ARM_QPOS),
        ("arm_qvel", mars.OBS_ARM_QVEL),
        ("head_pitch", mars.OBS_HEAD_PITCH),
        ("gripper_load", mars.OBS_GRIPPER_LOAD),
        ("last_action", mars.OBS_LAST_ACTION),
        ("target_base", mars.OBS_TARGET_BASE),
        ("target_seen", mars.OBS_TARGET_SEEN),
        ("base_twist", mars.OBS_BASE_TWIST),
        ("reserved", mars.OBS_RESERVED),
    )

    def _check_contract(self) -> None:
        """This builder tiles `MarsBody.contract().slots`, or nothing runs."""
        want = mars.MARS.contract()
        mine = tuple((n, s.start, s.stop) for n, s in self._SLOTS)
        theirs = tuple((s.name, s.start, s.stop) for s in want.slots)
        if mine != theirs:
            raise ValueError(
                f"MarsArmEnv fills {mine} but {want.id} declares {theirs} — "
                "the contract is what every reader of an exported policy "
                "resolves, so the env may not have its own layout")
        if want.obs_dim != mars.OBS_DIM or want.act_dim != mars.NUM_ACTIONS:
            raise ValueError(
                f"{want.id} is {want.obs_dim}/{want.act_dim}, this env is "
                f"{mars.OBS_DIM}/{mars.NUM_ACTIONS}")
        if abs(want.rate_hz - 1.0 / CTRL_DT) > 1e-9:
            raise ValueError(
                f"{want.id} declares {want.rate_hz} Hz, this env steps at "
                f"{1.0 / CTRL_DT} Hz — a policy run at the wrong rate is a "
                "different controller")

    # --------------------------------------------------------------- the task

    def sample_target(self) -> np.ndarray:
        """A point in the reach shell, in the BASE frame.

        Sampled in the base frame and transformed out, so the observation's
        `target_base` is yaw-invariant and a MARS spawned facing anywhere sees
        the same task. `reset` is the only caller: re-drawing mid-episode
        would make the progress term's telescoping sum meaningless.
        """
        r = self._rng
        while True:
            radius = r.uniform(*REACH_RADIUS_M)
            yaw = r.uniform(*REACH_YAW_RAD)
            z = r.uniform(*REACH_HEIGHT_M)
            dz = z - float(self.shoulder_base[2])
            horiz2 = radius * radius - dz * dz
            if horiz2 > REACH_MIN_HORIZ_M ** 2:
                horiz = math.sqrt(horiz2)
                return self.shoulder_base + np.array(
                    [horiz * math.cos(yaw), horiz * math.sin(yaw), dz])

    def _to_world(self, point_base: np.ndarray) -> np.ndarray:
        x, y, yaw = self.driver.pose(self.data)
        cos, sin = math.cos(yaw), math.sin(yaw)
        return np.array([
            x + point_base[0] * cos - point_base[1] * sin,
            y + point_base[0] * sin + point_base[1] * cos,
            point_base[2],
        ])

    def _to_base(self, point_world: np.ndarray) -> np.ndarray:
        """`point_world` in the base's own frame: rotate by -yaw, translate by
        -base position. The same rotation `MarsDriver.velocity` uses, so what
        the observation sees and what the drive regulates cannot drift."""
        x, y, yaw = self.driver.pose(self.data)
        cos, sin = math.cos(yaw), math.sin(yaw)
        dx, dy = point_world[0] - x, point_world[1] - y
        return np.array([dx * cos + dy * sin, -dx * sin + dy * cos,
                         point_world[2]])

    def effector(self) -> np.ndarray:
        """The gripper's tool point (`ee_link`) in world coordinates."""
        return np.array(self.data.xpos[self._ee_body])

    def distance(self) -> float:
        return float(np.linalg.norm(self.effector() - self.target))

    # -------------------------------------------------------- self-collision

    def self_collision(self) -> tuple[str, str] | None:
        """The DEEPEST pair of MARS's own bodies overlapping, or None.

        Three things are filtered out here and NONE of them is what keeps a
        parked MARS alive — which is only known because each was planted and
        the test did not notice (`AGENTS.md`: a guard nobody can tell is
        broken is a guard nobody should trust). Stated as measured:

        * the FLOOR is excluded by the robot-body check, and at HOME that
          check is redundant: MEASURED over 200 steps, all three
          `world <-> base_link` contacts (the chassis box and the two wheels)
          sit at **0.000 mm** depth, because the planar base pins z and the
          wheels rest exactly on the plane — so the DEPTH threshold would
          have dropped them anyway. The filter earns its place in Phase 4b,
          where a block on the floor makes non-robot pairs ordinary.
        * two geoms of ONE body: MuJoCo excludes same-body pairs itself, so
          this line has never been reached. Kept as a statement of intent.
        * the finger pair, which `mars.tune_contacts` already `add_exclude`s
          (their hub pins overlap ~1 mm at joint6 = 0, and `arm.srdf` disables
          the same pair for MoveIt). The test asserts the MODEL is what
          prevents it, so the redundancy is visible rather than assumed.

        What actually keeps HOME clean is the depth threshold plus the servo
        holding the arm there: `test_home_is_clear_of_self_collision` fails
        when the servo is not stepped, because a limp MARS arm folds into its
        own chassis.

        The deepest rather than the first, so `info["self_collision"]` names
        the pair that actually ended the episode: a real drive into the
        chassis reports several contacts at once and the shallowest of them is
        usually a different pair of links.
        """
        m, d = self.model, self.data
        worst, pair = -SELF_COLLISION_DEPTH_M, None
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = int(m.geom_bodyid[con.geom1])
            b2 = int(m.geom_bodyid[con.geom2])
            if b1 == b2:
                continue
            if b1 not in self._robot_bodies or b2 not in self._robot_bodies:
                continue
            if {b1, b2} == self._finger_bodies:
                continue
            if float(con.dist) >= worst:      # dist < 0 is the overlap
                continue
            worst = float(con.dist)
            pair = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "?",
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "?")
        return pair

    # ---------------------------------------------------------- observations

    def _get_obs(self) -> np.ndarray:
        d = self.data
        obs = np.zeros(mars.OBS_DIM, np.float32)
        obs[mars.OBS_ARM_QPOS] = d.qpos[self._arm_qadr]
        obs[mars.OBS_ARM_QVEL] = d.qvel[self._arm_dadr]
        obs[mars.OBS_HEAD_PITCH] = d.qpos[self._head_qadr]
        # The torque the servo WROTE at joint6 (`mars.arm_servo` ->
        # `qfrc_applied`), clamped by it to +-GRIPPER_EFFORT_LIMIT = 2 N*m, so
        # the slot arrives pre-scaled. It is the closest thing here to the
        # present_load a Dynamixel reports, which is what an Innate code skill
        # would read on the real robot.
        #
        # MEASURED, and it is why Phase 4b needs more than this slot: closing
        # on air drives joint6 into its own hard stop
        # (`GRIPPER_CLOSED_ON_AIR_RAD`), where the position error is zero and
        # this reads 0.0 N*m — identical to an open claw. What distinguishes
        # "holding" from "closed on air" is that an OBJECT stops the blades
        # short of that stop, which shows up in `arm_qpos[5]`, with
        # `qfrc_constraint` as the force behind it. So this slot is the
        # squeeze COMMAND, the qpos slot is the evidence, and `pick` will want
        # the constraint torque as well.
        obs[mars.OBS_GRIPPER_LOAD] = d.qfrc_applied[self._grip_dadr]
        # The CLIPPED action — what the env applied, not what the network
        # asked for. This cost the phase its first eval and it is worth the
        # paragraph, because the bug is invisible during training.
        #
        # SB3 clips a Box action to the space before the env ever sees it
        # (`OnPolicyAlgorithm.collect_rollouts`), so throughout training the
        # raw and the clipped value are the SAME and observing either is
        # identical. At inference nothing clips for you: MEASURED, this
        # policy's exported mean reaches |a| = 90 in a +-1 box, so a consumer
        # that hands the raw ONNX output straight to `step` had 86.6 written
        # into this slot — a float the policy had never once seen — and the
        # arm wandered off. Same policy, clipped: final distance 0.209 m ->
        # 0.026 m and `at_target` +1.5 -> +88 per episode.
        #
        # AGENTS.md's "suspect the harness when train and eval disagree", and
        # the deeper rule it is an instance of: a limit applied at BOTH
        # training and inference must come from ONE place. The place is
        # `action_space`, and this is the env applying it rather than trusting
        # every caller to. The action-rate PENALTY still prices the raw
        # output, which is walk_env's reason for keeping it (a duck reached
        # mean |a| 29 once the reward stopped seeing what it asked for).
        obs[mars.OBS_LAST_ACTION] = self.last_action.clip(-ACTION_CLIP,
                                                          ACTION_CLIP)
        self.target_base = self._to_base(self.target)
        obs[mars.OBS_TARGET_BASE] = self.target_base
        # 1.0 always: `reach`'s target is PRIVILEGED, the way every reward in
        # this repo is. A later task gates it on the head detector actually
        # seeing the thing (AGENTS.md: "a task the robot must SENSE puts its
        # sensing in the command slots, in the robot's own terms") — the slot
        # exists now so that adding the gate does not change the layout.
        obs[mars.OBS_TARGET_SEEN] = 1.0
        v_forward, _v_lateral, wz = self.driver.velocity(d)
        obs[mars.OBS_BASE_TWIST] = (v_forward, wz)
        # OBS_RESERVED stays the zeros `np.zeros` put there.
        return obs

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self._near_streak = 0
        self._self_hit = None
        self.reward_sums = {}
        self.last_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self._cmd_target = self._home.copy()

        yaw = self.spawn_yaw
        if self.random_yaw:
            yaw = float(self._rng.uniform(-math.pi, math.pi))
        if options and "base_yaw" in options:
            yaw = float(options["base_yaw"])
        # `spawn` writes the HOME pose, zeroes this robot's velocities and
        # clears both applied-force channels, then runs mj_forward.
        self.driver.spawn(self.data, 0.0, 0.0, yaw)
        self.data.time = 0.0

        self.target_base_sample = self.sample_target()
        self.target = self._to_world(self.target_base_sample)
        self._prev_dist = self.distance()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # The RAW policy output is what is observed and what the action-rate
        # penalty prices — walk_env's lesson: storing the clipped value made
        # unbounded output growth free, and a 25M-step duck reached mean |a|
        # 29 with half its outputs saturated.
        raw = np.asarray(action, np.float32).reshape(mars.NUM_ACTIONS)
        self.prev_action = self.last_action
        self.last_action = raw.copy()
        act = raw.clip(-ACTION_CLIP, ACTION_CLIP)

        want = np.clip(self._home + act[mars.ACT_ARM] * ACTION_SCALE_RAD,
                       self._jnt_lo, self._jnt_hi)
        # The rate limit (see MAX_TARGET_RATE_RAD_S): the commanded target
        # WALKS to where the action asks, at the speed the servos have. The
        # path stays inside [lo, hi] because it is a convex step from a point
        # inside it toward a point inside it.
        self._cmd_target = self._cmd_target + np.clip(
            want - self._cmd_target, -self.target_step_rad,
            self.target_step_rad)
        self.driver.set_arm(dict(zip(mars.ARM_JOINTS,
                                     self._cmd_target.tolist())))
        if self.use_base:
            vx, wz = act[mars.ACT_BASE_TWIST] * (MAX_CMD_LINEAR, MAX_CMD_YAW)
            self.driver.set_cmd(float(vx), float(wz), float(self.data.time))
        # else: nothing is commanded, so the driver's own `cmd_vel` watchdog
        # holds the base at zero and station keeping pins it where it spawned.

        for _ in range(DECIMATION):
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._self_hit = self.self_collision()
        dist = self.distance()
        reward, terms = self._compute_reward(dist)
        for k, v in terms.items():
            self.reward_sums[k] = self.reward_sums.get(k, 0.0) + v
        self._prev_dist = dist
        self._near_streak = (self._near_streak + 1
                             if dist <= SUCCESS_RADIUS_M else 0)

        obs = self._get_obs()
        terminated = self._self_hit is not None
        if not np.isfinite(obs).all():      # kill the episode, not the run
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True
        truncated = self.step_count >= self.max_steps

        info: dict[str, Any] = {
            "dist": dist,
            # "has held the target for the last second" — at the final step
            # that IS the task's definition of success.
            "success": bool(self._near_streak >= self.hold_steps),
            "near_steps": int(self._near_streak),
            # The base pair of the action is read and forced to zero for
            # `reach`; saying so here is what keeps a silently-ignored knob
            # from looking like a working one (AGENTS.md rule 0).
            "base_enabled": self.use_base,
            "self_collision": self._self_hit,
            "terms": terms,
        }
        if terminated or truncated:
            info["episode_rewards"] = dict(self.reward_sums)
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- rewards

    def _compute_reward(self, dist: float) -> tuple[float, dict[str, float]]:
        """The five terms, every one present every step.

        A stable key set is what the teach panel draws one bar per, and what
        `train._penalty_sign_callback_cls` watches for a sign flip.
        """
        # PROGRESS, not proximity: positive only while the gap is closing, and
        # an episode's total is `d_start - d_end` however it got there, so
        # oscillating earns nothing and parking earns nothing.
        progress = W_PROGRESS * (self._prev_dist - dist)
        near = W_NEAR * math.exp(-(dist / NEAR_SIGMA_M) ** 2)
        action_rate = ACTION_RATE_W * -float(
            ((self.last_action - self.prev_action) ** 2).sum())
        # The RAW magnitude, deliberately: what this prices is the mean
        # running outside its own box, which is invisible once the env
        # clips (see ACTION_MAG_W).
        action_mag = ACTION_MAG_W * -float((self.last_action ** 2).sum())
        joint_vel = JOINT_VEL_W * -float(
            (self.data.qvel[self._arm_dadr] ** 2).sum())
        collision = SELF_COLLISION_PENALTY if self._self_hit else 0.0
        terms = {
            "reach_progress": progress,
            "at_target": near,
            "action_rate_penalty": action_rate,
            "action_mag_penalty": action_mag,
            "joint_vel_penalty": joint_vel,
            "self_collision_penalty": collision,
        }
        return float(sum(terms.values())), terms


def make_mars_env(**kwargs) -> MarsArmEnv:
    """Factory with the training defaults (`train-walk --robot mars`)."""
    kwargs.setdefault("max_episode_s", EPISODE_S)
    return MarsArmEnv(**kwargs)


__all__ = ["ACTION_CLIP", "ACTION_MAG_W", "ACTION_RATE_W", "ACTION_SCALE_RAD",
           "CTRL_DT", "DECIMATION", "EPISODE_S", "JOINT_VEL_W",
           "MAX_TARGET_RATE_RAD_S", "NEAR_SIGMA_M", "REACH_HEIGHT_M",
           "REACH_MIN_HORIZ_M", "REACH_RADIUS_M", "REACH_YAW_RAD",
           "SELF_COLLISION_DEPTH_M", "SELF_COLLISION_PENALTY",
           "SUCCESS_HOLD_S", "SUCCESS_RADIUS_M", "TASKS", "W_NEAR",
           "W_PROGRESS", "MarsArmEnv", "make_mars_env"]
