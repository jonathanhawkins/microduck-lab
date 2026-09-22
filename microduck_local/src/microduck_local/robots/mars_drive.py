"""MARS's base drive: Innate's velocity PD, station keeping and watchdog.

The thing this harness calls "the policy" is a velocity-command walker, and
MARS has no gait to learn — what moves it is a **controller**, not a network.
`docs/mars-roadmap.md` Phase 3 says why that is the cheap half of putting a
wheeled body in a room: for a differential-drive base the reflex tier IS the
base controller, so a MARS drives on the day it is attached.

This is `robots/g1.G1Walker`'s slot in the lab's `Body` contract — construct
one on a compiled model with a prefix, `spawn` it, command it, `step` it —
filled with a port of Innate's own driver rather than an ONNX session:

    ros2_ws/src/mars_bot/mars_sim_driver/core.py    set_cmd_vel, _apply_control,
                                                    _station_keeping, pose, velocity
    ros2_ws/src/mars_bot/mars_sim_driver/world.py   KP_*, HOLD_SETTLE_S, MAX_BASE_*
    ros2_ws/src/mars_bot/mars_sim_driver/drive_limits.py  MAX_LINEAR, MAX_YAW

Every constant below names the function it came from, because the point of
porting instead of tuning is that a drive that is wrong HERE is wrong THERE
too — the same numbers run on Innate's own sim and, through their driver
node, on the real base.

**Two force channels, and this object owns both.** The base is driven by a
velocity PD written into `data.xfrc_applied[base_link]` (the URDF has no
`<actuator>` block; the planar base is pushed, not geared), and the arm and
head by position PD into `data.qfrc_applied` (`mars.arm_servo`). Both are
*assigned* every step, never accumulated: MuJoCo keeps whatever was written
there until something overwrites it, so a `+=` would integrate a force
forever and a skipped step would silently hold the last one.

**One number of Innate's cannot be ported verbatim, and it is measured
here.** `xfrc_applied` is an EXPLICIT force: a first-order velocity loop
`v += (KP*dt/M)(v* - v)` is contractive only while `KP*dt/M < 2`, and this
repo's world runs at `contract.PHYSICS_DT` = 5 ms where Innate's runs at
2 ms. MEASURED apparent inertia at the base DoFs (`mj_fullM`, the base
3x3's Schur complement, arm folded at HOME): 1.315 kg, 1.297 kg,
**0.005761 kg*m^2**. So Innate's forward and lateral gains are fine at both
steps (200 -> 0.76, 40 -> 0.15), and `KP_YAW = 3` is 1.04 at their 2 ms and
**2.60 at ours — past the bound, and it diverges**: 1.0 rad/s commanded for
3 s ends at yaw 7.13 rad with |wz| peaking at 13.3 rad/s, and a straight
0.3 m/s line wanders 229 mm sideways. The empirical boundary is between
KP_YAW 2.0 (gain 1.74, tracks) and 2.5 (gain 2.17, diverges), which is the
2.0 bound to two digits. `_stable_gain` therefore clamps each gain to
`GAIN_LIMIT * M / dt` — nothing at 2 ms but the yaw, nothing at 5 ms but
the yaw, which becomes 1.152 and tracks 3.0000 rad. The station-keeping
gains are a SPRING, not this loop, and are verbatim (`sqrt(K/M)*dt` is
0.075 and 0.161).

MEASURED on this Mac in MARS's own standalone scene (tests/test_mars_drive.py
re-measures all of it):

    0.3 m/s for 5 s      x = 1.4994 m,  y = -6.7 mm,  yaw = -0.0045 rad
    1.0 rad/s for 3 s    yaw = 2.9995 rad,  wz = 1.0000 rad/s
    cmd 2.0 m/s          clamped to 0.8000 m/s at the command envelope
    watchdog             |v| < 0.02 m/s 0.52 s after the last command
    station keeping      a 0.5 m/s shove: 0.9 mm peak, 0.2 mm after 1 s;
                         200 ms of push: 51 mm out, 5.6 mm after 1 s
                         (51 mm and still 51 mm with the hold stubbed out)
    arm while driving    max |q - HOME| = 0.0034 rad
    throughput           66-71 k physics steps/s over four runs, against
                         119-127 k bare: 0.56x bare either way, 6.3-6.6 us a
                         step, about a third of it the drive and two thirds
                         the servo (the 134,700 bare in `mars.py` was a
                         quieter machine). Two driven MARSes in one model,
                         36,400-37,000

**The drive must run every physics step.** MEASURED: decimating `drive()` to
the 50 Hz control tick diverges for the same reason — at a 20 ms tick the
forward loop's gain is `200 * 0.02 / 1.315 = 3.04`, and a 3 s run at
0.3 m/s ends going BACKWARDS at 2.6 m/s having covered nothing; at a 10 ms
tick (gain 1.52) the position is right and the velocity rings. `step()` is
therefore a per-physics-step call, like Innate's own `_apply_control`;
`drive()` and `servo()` are split out so a caller can measure the two
halves, not so it can decimate the first. It would only buy ~1.5 us of a
14 us step anyway (66-71 k -> 77 k steps/s).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import mujoco
import numpy as np

from . import mars

# ------------------------------------------------------- the command envelope
#
# innate drive_limits.MAX_LINEAR / MAX_YAW, applied by their `clamp_cmd_vel`
# in `set_cmd_vel` — their real-robot defaults x2, the "Mad" mode. What a
# brain, a WASD key or a policy is allowed to ASK for; the governor further
# down is what the base is allowed to DO.
MAX_CMD_LINEAR = 0.8              # m/s
MAX_CMD_YAW = 2.0                 # rad/s

# ------------------------------------------------------------- the velocity PD
#
# innate world.KP_FORWARD / KP_LATERAL / KP_YAW. Their note, kept because it
# is the reason the yaw gain looks tiny next to the others: "KP_YAW=6
# resonates with the arm servos on the same axis; 3 is the verified-stable
# value". Forward is stiff (a wheeled base holds its commanded speed),
# lateral is a damper toward zero (frictionless wheels cannot resist a
# sideways slide on their own), yaw is soft.
KP_FORWARD = 200.0                # N per m/s of forward error
KP_LATERAL = 40.0                 # N per m/s of sideways drift
KP_YAW = 3.0                      # N*m per rad/s of yaw error

# --------------------------------------------------------- station keeping
#
# innate world.KP_HOLD_LINEAR / KP_HOLD_YAW / HOLD_SETTLE_S, applied by
# core._station_keeping. The drive above is velocity-only, so with a zero
# command nothing pulls the base back and the arm's reaction torque walks the
# robot — real wheels and gearing do not give that ground away. The pose
# latches only after HOLD_SETTLE_S of quiet so a skill's per-camera-frame
# `cmd_vel` gaps never anchor a base that is still meant to be driving.
KP_HOLD_LINEAR = 300.0            # N per m of position error
KP_HOLD_YAW = 6.0                 # N*m per rad of heading error
HOLD_SETTLE_S = 0.4               # s of quiet before the hold pose latches

# --------------------------------------------------------- the safety governor
#
# innate world.MAX_BASE_LINEAR_SPEED / MAX_BASE_ANGULAR_SPEED, applied at the
# top of core._apply_control. A clamp on the STATE, not on the command: their
# reason is imperfect hull-seam contacts, so that a bad single-step impulse is
# a recoverable thump instead of a NaN. It is the last thing between a room
# whose walls are decomposed hulls and a base that leaves the map.
MAX_BASE_LINEAR_SPEED = 2.0       # m/s
MAX_BASE_ANGULAR_SPEED = 6.0      # rad/s

# ---------------------------------------------------------------- the watchdog
#
# innate core.CMD_VEL_TIMEOUT_S. A command is a lease, not a setting: a
# brain that stops publishing (or a websocket that drops) stops the robot
# rather than leaving it driving at the last thing it was told.
CMD_VEL_TIMEOUT_S = 0.5           # s of sim time

# ------------------------------------------------- the timestep's own limit
#
# NOT Innate's: the step-size guard the module docstring measures. An
# explicit velocity loop is contractive below 2.0 and DEADBEAT at 1.0 — one
# step to the commanded velocity, no overshoot — so 1.0 is both the stiffest
# useful value and the one with a factor of two in hand. The margin is
# needed: the apparent yaw inertia is pose-dependent (MEASURED 0.0035 to
# 0.0155 kg*m^2 over the arm's reachable poses, 0.005761 folded at HOME,
# where the gain is computed), so a gain that is deadbeat at HOME is 1.65 in
# the most-folded pose and still inside the bound.
GAIN_LIMIT = 1.0


def clamp_cmd(vx: float, wz: float) -> tuple[float, float]:
    """The command envelope — innate drive_limits.clamp_cmd_vel.

    Per axis, because the base is differential-drive: a full-speed arc is
    `MAX_CMD_LINEAR` forward AND `MAX_CMD_YAW` of turn at the same time, not
    a vector of length `MAX_CMD_LINEAR`.
    """
    return (
        max(-MAX_CMD_LINEAR, min(MAX_CMD_LINEAR, float(vx))),
        max(-MAX_CMD_YAW, min(MAX_CMD_YAW, float(wz))),
    )


class MarsDriver:
    """One attached MARS: drive the base, hold the arm, read the odometry.

    `model` is the COMPILED model the robot lives in — MARS's own scene
    (`mars.model()`) or a `/sim` world with N bodies in it — and `prefix` is
    what it was attached under (`""` for the standalone scene, `"m0/"` for a
    room). Every address is resolved once, here, by NAME: a 50 Hz world loop
    must not be doing string lookups, and a URDF revision that renames a link
    fails at construction instead of servoing a joint that does not exist.
    """

    def __init__(self, model: mujoco.MjModel, prefix: str = ""):
        self.model = model
        self.prefix = prefix
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         prefix + mars.BASE_BODY)
        if self.base_id < 0:
            raise KeyError(f"no body {prefix + mars.BASE_BODY!r} in the model — "
                           "a MARS must be attached under this prefix")
        # The planar base's (x, y, yaw), in innate's own order.
        self.base_qadr = tuple(int(model.joint(prefix + n).qposadr[0])
                               for n in mars.BASE_JOINTS)
        self.base_dadr = tuple(int(model.joint(prefix + n).dofadr[0])
                               for n in mars.BASE_JOINTS)
        #: {joint: (qpos adr, dof adr)} for the 7 driven joints + the mimic.
        self.adr = mars.servo_addresses(model, prefix)
        # Only THIS robot's rows, so a spawn or a reset in a room with two
        # MARSes and three ducks in it cannot touch anybody else's state.
        self._my_dofs = np.array(
            sorted({*self.base_dadr, *(d for _q, d in self.adr.values())}))
        self._home_adr = np.array(
            sorted({*self.base_qadr, *(q for q, _d in self.adr.values())}))
        # `mars._home_qpos` is the single definition of the HOME pose (it is
        # also what the scene's keyframe is built from, and it is the only
        # place the mimic finger's -joint6 rule is written). It returns a
        # FULL-WIDTH qpos vector though, so `spawn` indexes this robot's own
        # addresses out of it — assigning the whole vector would zero every
        # other body in a composed world.
        self._home_qpos = mars._home_qpos(model, prefix)
        # The step-size guard (see GAIN_LIMIT and the module docstring). Both
        # are properties of THIS model — its timestep and this robot's own
        # inertia — so they are resolved here and the step reads two floats.
        self.dt = float(model.opt.timestep)
        self.base_inertia = self._apparent_inertia(model)
        self.kp_forward = self._stable_gain(KP_FORWARD, self.base_inertia[0])
        self.kp_lateral = self._stable_gain(KP_LATERAL, self.base_inertia[1])
        self.kp_yaw = self._stable_gain(KP_YAW, self.base_inertia[2])
        self._arm: dict[str, float] = dict(mars.ARM_HOME)
        self._cmd_vx = 0.0
        self._cmd_wz = 0.0
        # -inf, so a driver that was never commanded is already expired: an
        # un-driven MARS holds still rather than waiting out the watchdog.
        self._cmd_t = -math.inf
        self._hold: tuple[float, float, float] | None = None
        self._still_since: float | None = None
        # THE CLAW, for `held_body`: the two blades, and this robot's whole
        # subtree (by `body_rootid`, so an attached MARS in a room with five
        # other bodies in it answers about its own parts and nothing else).
        self._finger_bodies = frozenset(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + n)
            for n in mars.FINGER_LINKS)
        if -1 in self._finger_bodies:
            raise KeyError(
                f"no {list(mars.FINGER_LINKS)} bodies under {prefix!r} — "
                "a MARS must be attached with its gripper blades")
        self._my_bodies = frozenset(
            b for b in range(model.nbody)
            if int(model.body_rootid[b]) == int(model.body_rootid[self.base_id]))

    # -------------------------------------------------- the step-size guard

    def _apparent_inertia(self, model: mujoco.MjModel) -> tuple[float, float, float]:
        """(m_x, m_y, I_yaw) the base's three DoFs actually feel, at HOME.

        The Schur complement of the base 3x3 of the mass matrix — the inertia
        an applied force sees when the other two planar DoFs are free and the
        arm is rigid, which is what the position servo makes it. MEASURED
        against the empirical stability boundary (KP_YAW 2.0 tracks, 2.5
        diverges) and it predicts it to two digits, where the fully-free
        complement (0.0039) is too conservative by 50 %.

        Its own `MjData` because a driver is handed a model and no state, and
        the mass matrix needs a pose. The robot's own HOME pose is written
        into it: `M` has no coupling between separate kinematic trees, so
        whatever else a composed world holds cannot change this answer.
        """
        probe = mujoco.MjData(model)
        probe.qpos[self._home_adr] = self._home_qpos[self._home_adr]
        mujoco.mj_forward(model, probe)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, probe, full)
        block = full[np.ix_(self.base_dadr, self.base_dadr)]
        inv = np.linalg.inv(block)
        return tuple(float(1.0 / inv[i, i]) for i in range(3))

    def _stable_gain(self, kp: float, inertia: float) -> float:
        """Innate's gain, or as much of it as this timestep can integrate.

        Returns `min(kp, GAIN_LIMIT * inertia / dt)`. A no-op for every gain
        but the yaw's, at either timestep — which is the point of writing it
        as a limit instead of a second table of constants: the numbers stay
        Innate's, and what changes is what the step can carry.
        """
        return min(float(kp), GAIN_LIMIT * float(inertia) / self.dt)

    def gains(self) -> dict[str, float]:
        """The gains as applied, and the inertia they were limited against.

        For a report or a `/sim` panel: a MARS whose yaw gain is not 3.0 is
        not misconfigured, it is being integrated at 5 ms, and this is where
        that is visible instead of surprising.
        """
        return {"kp_forward": self.kp_forward, "kp_lateral": self.kp_lateral,
                "kp_yaw": self.kp_yaw, "dt": self.dt,
                "mass_x": self.base_inertia[0], "mass_y": self.base_inertia[1],
                "inertia_yaw": self.base_inertia[2]}

    # ------------------------------------------------------------- the pose

    def spawn(self, data: mujoco.MjData, x: float = 0.0, y: float = 0.0,
              yaw: float = 0.0) -> None:
        """Put this MARS at (x, y, yaw) in the HOME pose, at rest.

        `G1Walker.spawn`'s job, and the same contract: it writes state, it
        does not step. The applied-force rows are cleared too — they are this
        object's to own, and a spawn that left the last step's drive force in
        `xfrc_applied` would shove the robot on the next `mj_step` even with
        no command.
        """
        data.qpos[self._home_adr] = self._home_qpos[self._home_adr]
        qx, qy, qyaw = self.base_qadr
        data.qpos[qx] = float(x)
        data.qpos[qy] = float(y)
        data.qpos[qyaw] = float(yaw)
        data.qvel[self._my_dofs] = 0.0
        data.qfrc_applied[self._my_dofs] = 0.0
        data.xfrc_applied[self.base_id] = 0.0
        self._arm = dict(mars.ARM_HOME)
        self._cmd_vx = self._cmd_wz = 0.0
        self._cmd_t = -math.inf
        self._hold = None
        self._still_since = None
        mujoco.mj_forward(self.model, data)

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(x, y, yaw) of the base — innate core.pose.

        Read straight off the planar base's qpos, which IS the pose: no
        quaternion to unwrap and no pitch or roll to lose, because a wheeled
        chassis has neither (`mars.add_planar_base`). `yaw` accumulates, so a
        robot that has turned twice reads past 2*pi — wrap it at the consumer,
        the way `_station_keeping` does with `atan2`.
        """
        qx, qy, qyaw = self.base_qadr
        return (float(data.qpos[qx]), float(data.qpos[qy]),
                float(data.qpos[qyaw]))

    def velocity(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(v_forward, v_lateral, wz) in the base frame — innate core.velocity.

        The body-frame twist a `Senses.odom` channel reports, and the same
        heading rotation the drive PD uses, so what the controller is
        regulating and what an observation sees cannot drift apart.
        """
        dx, dy, dyaw = self.base_dadr
        yaw = float(data.qpos[self.base_qadr[2]])
        cos, sin = math.cos(yaw), math.sin(yaw)
        vx, vy = float(data.qvel[dx]), float(data.qvel[dy])
        return (vx * cos + vy * sin, -vx * sin + vy * cos,
                float(data.qvel[dyaw]))

    # ---------------------------------------------------------- the commands

    def set_cmd(self, vx: float, wz: float, t: float) -> None:
        """Command a body-frame twist, stamped at sim time `t`.

        innate core.set_cmd_vel, with their implicit clock made explicit: they
        read `self.data.time` inside the driver, and here the caller passes
        the sim time it is commanding AT, because a world loop holds one clock
        for every body in the room and a test needs to be able to command in
        the past. `step()` compares it against `data.time`, so the watchdog
        measures SIM seconds — a slow host makes a robot late, never runaway.
        """
        self._cmd_vx, self._cmd_wz = clamp_cmd(vx, wz)
        self._cmd_t = float(t)

    def cmd(self) -> tuple[float, float]:
        """The clamped command as stored — what a `/sim` panel should show."""
        return (self._cmd_vx, self._cmd_wz)

    def set_arm(self, targets: Mapping[str, float]) -> None:
        """Set arm/head position targets; anything absent holds `ARM_HOME`.

        innate core.set_joint_target, including its clamp to the joint's own
        range: the real gripper close is commanded 0.6 rad past the mechanical
        stop, and unclamped in sim that target scissors the two blades through
        each other (`mars.GRIPPER_CLOSED_ON_AIR_RAD` is where the hard stop
        actually is). Clamped, the error still saturates the torque ceiling,
        so the claw still squeezes — it just stops where the metal does.

        An unknown joint name is a `KeyError` rather than a no-op (Innate's
        node drops it silently): here the caller is a brain or a policy action
        block, and a typo that quietly stops driving the gripper would read as
        a broken grasp.
        """
        unknown = set(targets) - set(mars.DRIVEN_JOINTS)
        if unknown:
            raise KeyError(
                f"{sorted(unknown)} are not MARS joints — the driven set is "
                f"{list(mars.DRIVEN_JOINTS)} (joint6M mirrors joint6 and is "
                "not commandable)")
        full = dict(mars.ARM_HOME)
        for name, value in targets.items():
            jid = self.model.joint(self.prefix + name).id
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                value = max(float(lo), min(float(hi), float(value)))
            full[name] = float(value)
        self._arm = full

    def arm_targets(self) -> dict[str, float]:
        """The targets as clamped and stored — innate core.joint_targets."""
        return dict(self._arm)

    # ------------------------------------------------------------- the step

    def step(self, data: mujoco.MjData) -> None:
        """One control pass, to be called before EVERY `mj_step`.

        innate core.step's loop body: the governor, the watchdog, the velocity
        PD, station keeping, then the joint servos. Not decimatable — see the
        module docstring for the gain that says so (`KP_FORWARD * dt / m` is
        2.93 at 50 Hz and 0.73 at the 5 ms physics step).
        """
        self.drive(data)
        self.servo(data)

    def drive(self, data: mujoco.MjData) -> None:
        """The base half — innate core._apply_control's first block.

        Owns `data.xfrc_applied[base_link]`: all six components are assigned,
        including the three this drive never uses, so nothing else's leftover
        z-force or roll torque can ride along on a body whose z is pinned.
        """
        dx, dy, dyaw = self.base_dadr

        # The governor, on the STATE, before anything reads it.
        lin = math.hypot(data.qvel[dx], data.qvel[dy])
        if lin > MAX_BASE_LINEAR_SPEED:
            data.qvel[dx] *= MAX_BASE_LINEAR_SPEED / lin
            data.qvel[dy] *= MAX_BASE_LINEAR_SPEED / lin
        if abs(data.qvel[dyaw]) > MAX_BASE_ANGULAR_SPEED:
            data.qvel[dyaw] = math.copysign(MAX_BASE_ANGULAR_SPEED,
                                            data.qvel[dyaw])

        # The watchdog: a stale command is no command.
        expired = data.time - self._cmd_t > CMD_VEL_TIMEOUT_S
        vx = 0.0 if expired else self._cmd_vx
        wz = 0.0 if expired else self._cmd_wz

        yaw = float(data.qpos[self.base_qadr[2]])
        cos, sin = math.cos(yaw), math.sin(yaw)
        v_forward = data.qvel[dx] * cos + data.qvel[dy] * sin
        v_lateral = -data.qvel[dx] * sin + data.qvel[dy] * cos

        force_forward = self.kp_forward * (vx - v_forward)
        force_lateral = -self.kp_lateral * v_lateral
        torque_yaw = self.kp_yaw * (wz - data.qvel[dyaw])
        hold_x, hold_y, hold_yaw = self._station_keeping(data, vx, wz)
        row = data.xfrc_applied[self.base_id]
        row[0] = force_forward * cos - force_lateral * sin + hold_x
        row[1] = force_forward * sin + force_lateral * cos + hold_y
        row[2] = row[3] = row[4] = 0.0
        row[5] = torque_yaw + hold_yaw

    def servo(self, data: mujoco.MjData) -> None:
        """The arm half — `mars.arm_servo` on the cached addresses.

        Separate from `drive()` because the two halves have different costs
        and different cadences on paper (Innate's arm answers a 25 Hz skill,
        their base a per-step PD). In practice both run every step here: the
        drive because it is unstable slower, the servo because a 7-joint
        position PD is the cheaper of the two anyway.
        """
        mars.arm_servo(self.model, data, self._arm, self.prefix, adr=self.adr)

    def _station_keeping(self, data: mujoco.MjData, vx: float,
                         wz: float) -> tuple[float, float, float]:
        """World-frame (fx, fy, tz) holding a stopped base — innate
        core._station_keeping, verbatim in behaviour.

        Zero while driving, and zero for the first `HOLD_SETTLE_S` of quiet;
        the pose latches once, on the first step past that, and is released
        the instant a non-zero command arrives. The latch is why a shove
        returns the robot to where it stopped instead of to where the shove
        left it.
        """
        if vx or wz:
            self._hold = self._still_since = None
            return 0.0, 0.0, 0.0
        qx, qy, qyaw = self.base_qadr
        if self._still_since is None:
            self._still_since = float(data.time)
        if data.time - self._still_since < HOLD_SETTLE_S:
            return 0.0, 0.0, 0.0
        if self._hold is None:
            self._hold = (float(data.qpos[qx]), float(data.qpos[qy]),
                          float(data.qpos[qyaw]))
        hx, hy, hyaw = self._hold
        d_yaw = hyaw - float(data.qpos[qyaw])
        yaw_err = math.atan2(math.sin(d_yaw), math.cos(d_yaw))
        return (KP_HOLD_LINEAR * (hx - float(data.qpos[qx])),
                KP_HOLD_LINEAR * (hy - float(data.qpos[qy])),
                KP_HOLD_YAW * yaw_err)

    def hold_pose(self) -> tuple[float, float, float] | None:
        """The latched station-keeping pose, or None while driving/settling."""
        return self._hold

    # -------------------------------------------------------------- the claw

    def gripper_load(self, data: mujoco.MjData) -> float:
        """The torque an OBJECT feeds back through joint6 (N*m).

        `qfrc_constraint`, not the servo's `qfrc_applied`, and the table that
        decided it is on `mars.OBS_GRIPPER_LOAD`: the servo torque is
        saturated at -2 N*m for 8 of the 40 control steps of a close on AIR,
        so one sample of it cannot tell "closing" from "holding", while the
        constraint is 0.0000 for all 40 and ~2 N*m with the block in.

        Identical to `MarsArmEnv.gripper_load` by construction — same address,
        same array — so a policy's observation slot and a brain's
        `Senses.holding` cannot disagree about what the claw is doing.
        """
        return float(data.qfrc_constraint[self.adr[mars.ARM_JOINTS[-1]][1]])

    def held_body(self, data: mujoco.MjData) -> int:
        """Which BODY the claw is holding, or -1 — the measured predicate.

        `|gripper_load| >= mars.HOLD_LOAD_NM` **and** a finger-blade contact
        with a body that is not part of this robot. `MarsArmEnv.holding()`
        asks the same two questions of a scene with exactly one toy in it and
        can answer `bool`; a `/sim` room has six toys and a basket, so this
        returns WHICH — the reason 4b kept the contact conjunct at all ("the
        contact adds no discrimination today; it is there so that holding
        names the OBJECT once Phase 5's room has several").

        The robot's own subtree is excluded by `body_rootid`, not by a name
        list: the two blades touch each other every close (`tune_contacts`
        excludes that pair, but a prefix-blind scan would still have to know),
        and 4b's own `_robot_bodies` bug was exactly this — "everything that
        is not the world" counted a TOY as part of the robot and terminated a
        successful grasp as a self-collision.
        """
        if abs(self.gripper_load(data)) < mars.HOLD_LOAD_NM:
            return -1
        m = self.model
        for i in range(data.ncon):
            con = data.contact[i]
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            for finger, other in ((b1, b2), (b2, b1)):
                if finger in self._finger_bodies and other not in self._my_bodies:
                    return other
        return -1


__all__ = ["CMD_VEL_TIMEOUT_S", "GAIN_LIMIT", "HOLD_SETTLE_S", "KP_FORWARD",
           "KP_HOLD_LINEAR", "KP_HOLD_YAW", "KP_LATERAL", "KP_YAW",
           "MAX_BASE_ANGULAR_SPEED", "MAX_BASE_LINEAR_SPEED", "MAX_CMD_LINEAR",
           "MAX_CMD_YAW", "MarsDriver", "clamp_cmd"]
