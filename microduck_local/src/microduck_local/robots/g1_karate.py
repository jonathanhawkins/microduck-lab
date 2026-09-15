"""Karate strikes for the G1: a front kick and a straight punch.

Both are built as HOLDS, exactly like the squat — reach a target pose and
stay there — because that is the shape this harness has already been shown to
learn, and because every lesson the squat cost is reusable:

  * the target pose is SOLVED against the compiled model, never hand-typed
    (the squat's solver bisected an immovable coordinate for an hour);
  * the episode SPAWNS in the target pose half the time, because an unsampled
    state's value is never learned (AGENTS.md, and the squat never once bent a
    knee until it was spawned bent);
  * a `carriage` term prices the limbs that are NOT striking, because a pose
    that meets its numbers can still look wrong — the first clean squat held
    its height with its back twisted 30 degrees and one arm on its thigh.

The one genuinely new thing here is SUPPORT. A squat is a two-foot pose; a
front kick stands on one leg with the other 0.55 m in the air, so:

  * `feet_planted` (both feet down) is replaced by `stance`, which wants the
    support foot down AND the striking foot clear;
  * the strike pose has to be statically balanceable at all, which is a
    property of the model, not of the reward. MEASURED before any training:
    standing, the CoM sits 0.118 m to the side of the left foot, and the foot
    is 0.073 m wide — so a one-leg stance is impossible until the leg adducts
    to bring the foot under the midline. At left_hip_roll -0.20 the CoM offset
    is -0.015 m, inside the foot. The solver below searches that shift jointly
    with the fore-aft counter-lean, because a leg thrown forward moves the CoM
    0.117 m ahead of a foot only 0.203 m long.
"""

from __future__ import annotations

import os

import numpy as np

from .. import contract as C
from .g1_env import G1StandEnv


def _com_offset_from_foot(model, data, mass_col, foot_geom_ids, side):
    """CoM minus the support foot's centre, in xy, plus the foot's footprint.

    Relative to the FOOT, never in world coordinates: the pelvis is the free
    joint's root, so a joint angle moves the foot and leaves the root where it
    was. Solving a stance in the root frame measures nothing, which is the
    same trap the squat's height solver fell into.
    """
    import mujoco

    com = (data.xipos * mass_col).sum(0) / mass_col.sum()
    xs0, xs1, ys0, ys1 = [], [], [], []
    for g in foot_geom_ids[side]:
        r = float(model.geom_size[g][0])
        half = float(model.geom_size[g][1])          # capsule half-length
        axis = data.geom_xmat[g].reshape(3, 3)[:, 2] * half
        p1, p2 = data.geom_xpos[g] - axis, data.geom_xpos[g] + axis
        xs0.append(min(p1[0], p2[0]) - r)
        xs1.append(max(p1[0], p2[0]) + r)
        ys0.append(min(p1[1], p2[1]) - r)
        ys1.append(max(p1[1], p2[1]) + r)
    x0, x1, y0, y1 = min(xs0), max(xs1), min(ys0), max(ys1)
    del mujoco
    return (float(com[0] - (x0 + x1) / 2), float(com[1] - (y0 + y1) / 2),
            x1 - x0, y1 - y0)


class G1StrikeEnv(G1StandEnv):
    """Reach a solved strike pose and hold it.

    Subclasses declare the strike (which limb, how far) and `_solve_strike()`
    returns the joint vector; everything else — spawning into it, paying for
    it, keeping the rest of the body composed — is shared.
    """

    STRIKE_SIDE = "right"                # the limb that strikes
    SUPPORT_SIDE = "left"

    # The pose IS the task, so it is the dominant earner, as `height` was for
    # the squat. std2 is over the joints the strike actually moves.
    W_STRIKE = 6.0
    # The reward has to REACH the policy where it starts, which is standing.
    # MEASURED: the punch pose is 2.715 rad^2 from the default pose (28 deg
    # rms over 11 joints), so at std2 0.35 a standing robot earned 0.00 of 6
    # and the term was perfectly flat across the whole approach. The punch
    # duly gave up on the pose — 27 deg mean joint error at 1.2M steps — and
    # spent its effort on the terms that did have a gradient. This is the
    # same failure, and the same fix, as the squat's first height term.
    # At std2 2.0: standing pays 1.54, 15 deg out pays 4.12, 5 deg out 5.75.
    STRIKE_STD2 = 2.0

    # 2.0 was too cheap to matter: a waist rolled 28 deg costs 0.24 rad^2,
    # which at std2 0.25 pays 0.38 of the term — about 1.2 points, against a
    # `lift` worth 4 and a `strike` worth 6. It stayed twisted.
    W_CARRIAGE = 3.0
    # 0.25 over 22 joints is FLAT where the policy actually sits: the idle
    # posture measured on a trained kick was 0.94 rad^2 from tidy (waist_roll
    # -28.8 deg, waist_pitch +20.5, shoulders ~20), which at 0.25 pays 0.069
    # of 3.0 — so the term was already lost and nothing pulled it back. This
    # is the same flat-reward mistake as the strike width, the height width
    # and the lift earner before it. At 0.80 that posture pays 0.92, a tidy
    # 5 deg/joint pays 2.43 and 3 deg/joint 2.78: a slope the whole way.
    CARRIAGE_STD2 = 0.80
    W_POSE = 0.0                 # the strike pose replaces it; see _compute_reward
    # RETARGETED, not deleted. Setting this to 0 on the reasoning that "a
    # strike is not defined by pelvis height" left nothing at all paying to
    # stay up, and the punch simply sagged: measured at 1.2M steps, the hips
    # were 23-44 deg off the pose and the pelvis sank from 0.735 to 0.479-0.493
    # until it tripped the 0.50 m fall floor on every seed. The strike term
    # does include the leg joints, but one Gaussian over 11 joints is far too
    # loose to hold a body up on its own. The target is the SOLVED pose's own
    # pelvis height, so the term says "stand as tall as this strike stands".
    # (This is the second time in this file's history that deleting a term
    # whose TARGET was wrong removed the only thing describing the task —
    # see G1SquatEnv.W_POSE for the first.)
    W_HEIGHT = 3.0
    # The idle's 0.004 is a HOLDING width — 6 cm of sag costs half — and it is
    # flat long before the sag that actually happens. Measured with it: the
    # punch settled 0.24 m low (pelvis 0.476-0.496 against a 0.733 target),
    # where exp(-0.0576/0.004) is e^-14, i.e. no gradient back up at all. The
    # squat needed the same widening for the same reason. At 0.03: 24 cm low
    # pays 0.15 of the term, 6 cm low pays 0.89.
    HEIGHT_STD2 = 0.03
    W_PLANTED = 0.0              # replaced by `stance`
    W_STANCE = 2.0
    # Weight over the support foot. This is the quantity that decides whether
    # a one-leg pose survives, and nothing else in the reward measures it:
    # `still`, `upright` and `stay_home` are all proxies that a slow topple
    # satisfies right up until the moment it does not. MEASURED on the first
    # kick chain, which reached the pose within 3-6 deg and stood one-legged
    # 90-100% of frames on every seed, then fell at 1.4-1.9 s — SIDEWAYS on
    # 5 of 6, which is the axis with 0.044 m of margin against 0.100 m
    # fore-aft. Scored in units of the foot's own half-extent so the term
    # means the same thing on any body: 1.0 at the centre, ~0.37 at the edge.
    W_BALANCE = 3.0
    BALANCE_STD2 = 1.0
    # Pay for the FOOT BEING UP, which is the task, not for a pose that
    # implies it. `strike` spreads one Gaussian over 7 joints and is a proxy:
    # measured on a 6M chain, the policy stood through the whole kick window
    # collecting `strike` 4.87 of 6 on the idle phases alone, and its foot
    # never left the floor (0.001-0.002 m at every rung). Standing cannot
    # satisfy this one at all while the strike is extended.
    W_LIFT = 0.0                 # a punch lifts nothing; the kick turns it on
    # SPIN, priced on the body's ACTUAL angular momentum rather than on arm
    # angles that would cancel it in theory. The feed-forward version failed
    # because it assumed the policy tracks the reference and it does not
    # (-21% of the commanded arm swing); this asks for the OUTCOME and lets
    # the robot find its own counter-motion.
    #
    # LINEAR, not a Gaussian. Measured on the shipped kick: |Lz| is 0.020
    # mean but spikes to 0.895 during the strike — a Gaussian narrow enough
    # to see the mean is flat at the spikes, which are the thing worth
    # pricing. A capped ramp has gradient the whole way.
    W_MOMENTUM = 0.0             # the kick turns it on
    MOMENTUM_REF = 0.5           # kg m^2/s at which the penalty saturates
    # A kick that ARRIVES decelerates at the top; a ballistic throw flies
    # through it. Nothing distinguished them: `lift` saturates AT the target,
    # so overshoot was free, and the policy duly threw the foot to 190% of a
    # 0.545 m target and toppled at the peak on every seed. Making `lift`
    # peak at the target instead was measured WORSE (it lowers the expected
    # value of attempting a kick at all, so it stops kicking). Pricing the
    # SPEED at full extension asks for control without taking away the pay
    # for height.
    W_ARRIVE = 0.0               # the kick turns it on
    ARRIVE_REF = 2.0             # m/s of foot speed at which it saturates
    # These three widths are INHERITED FROM THE IDLE, where the robot really
    # does stand still — and on a robot that kicks they are all flat at zero,
    # which is why it spun 44 deg and wandered 1.34 m for free. MEASURED on
    # the trained policy: gyro^2 7.5 against a 0.10 width (e^-75), drift^2
    # 1.80 against 0.0025 (e^-720), body speed^2 0.078 against 0.02. A term
    # that pays ~0 both where the policy IS and where you want it teaches
    # nothing. Widened so each pays a real fraction now and approaches its
    # maximum as the behaviour improves.
    W_STILL = 1.0                # a strike holds a limb out; it is not an idle
    STILL_STD2 = 0.08            # was 0.02: pays 0.38 now, 0.91 at 10x stiller
    W_SPIN = 2.0                 # a visible defect; worth more than the idle's 1.0
    SPIN_STD2 = 4.0              # was 0.10: pays 0.15 now, 0.83 at 10x steadier
    W_ANCHOR = 1.0
    ANCHOR_STD2 = 1.0            # was 0.0025 (5 cm): pays 0.17 at the 1.34 m
                                 # drift measured, 0.91 at 0.3 m

    SPAWN_IN_POSE_PROB = 0.5
    FALL_HEIGHT = 0.45

    # 0 -> the strike is a HOLD (the punch). > 0 -> it is a repeating
    # strike-and-recover CYCLE of this many seconds: idle, throw it, hold it
    # briefly, come back to the idle stance. The phase goes into the three
    # command slots of the observation, because a memoryless policy cannot
    # produce a limit cycle out of nothing — AGENTS.md, "anything the policy
    # must do over time needs time in the obs". Everything downstream reads
    # `strike_amplitude()`, so a hold is just the constant-1 case.
    CYCLE_S = 0.0
    # Fractions of one cycle: idle, throwing, extended, recovering, then idle
    # again for the remainder.
    # A SNAP: the amplitude peaks and comes straight back, with no hold at
    # full extension. The hold was the hard part and it was never asked for —
    # a kick does not need to stand on one leg, it needs to go out and come
    # back. Measured with a 0.75 s hold in the cycle: across a 9M five-rung
    # gravity ladder the policy traded the kick away for survival on every
    # rung (lift 3.21 -> 2.47 of 4 as balance climbed 1.34 -> 1.96), ending at
    # 0.010 m of foot clearance. Snapping through cuts the time on one leg
    # roughly in half.
    CYCLE_RISE = (0.25, 0.45)      # amplitude 0 -> 1
    CYCLE_FALL = (0.45, 0.65)      # amplitude 1 -> 0, immediately

    def __init__(self, *args, spawn_in_pose_prob: float | None = None, **kwargs):
        # A curriculum stage passes its knobs through the trainer's
        # ENVIRONMENT (the lab's stage machinery does not rewrite argv), so
        # the env var wins when the kwarg was left at its default — the same
        # contract MICRODUCK_G1_COMMAND_MIX uses.
        import os
        if spawn_in_pose_prob is None:
            env = os.environ.get("MICRODUCK_G1_SPAWN_IN_POSE")
            spawn_in_pose_prob = float(env) if env else None
        self.spawn_in_pose_prob = (self.SPAWN_IN_POSE_PROB
                                   if spawn_in_pose_prob is None
                                   else float(spawn_in_pose_prob))
        super().__init__(*args, **kwargs)
        self._fall_height = self.FALL_HEIGHT
        self._jname = {n: i for i, n in enumerate(self.robot.joint_names)}
        self._strike_joints = self._solve_strike()
        self._strike_rel = self._strike_joints - self.default_pose
        # Only the joints the strike MOVES are scored by the strike term; the
        # rest are the carriage's business. Scoring all 29 in one Gaussian is
        # what made the squat's `pose` term hold nothing.
        self._strike_ids = np.array(
            [i for i in range(self.nj) if abs(self._strike_rel[i]) > 1e-6],
            dtype=int)
        self._phase0 = 0
        # A DRILL rung LIGHTENS THE WORLD, it does not switch the floor off.
        #
        # The first attempt at this drill set `terminate_on_fall=False` so that
        # trying the kick was free. It taught the wrong skill: with nothing
        # ending the episode the robot spent most of it ON THE GROUND, and the
        # measurement that gives it away is `balance` at 0.04 of 3 for the
        # whole 2M-step stage — it learned to kick while falling, and the next
        # stage, which asks for the kick AND the balance together, could not
        # use any of it (episode length collapsed 40 s -> 1.85 s).
        #
        # Reduced gravity keeps the fall rule live — so staying upright is
        # still the task — while making the weight transfer onto one foot
        # survivable at all. This is the same shape as the headstand's
        # strong-servo drill: make the WORLD easier, never the judging, and
        # step it back down over the rungs.
        scale = float(os.environ.get("MICRODUCK_G1_GRAVITY", "1.0") or 1.0)
        if scale != 1.0:
            self.model.opt.gravity[2] *= scale
        env = os.environ.get(getattr(self, "CYCLE_ENV", "") or "_none_")
        if env:
            self.CYCLE_S = float(env)
        self._support_geoms = set(
            g for side in self.support_sides() for g in self.foot_geom_ids[side])
        self._strike_height = self._solved_pelvis_height()
        # The carriage must not fight the strike. `_carriage_ids` is every
        # joint above the hips, which for a PUNCH is exactly the joints doing
        # the punching — measured before this line existed: carriage paid
        # 0.00 at the punch pose and 1.70 at the kick (the kick leans the
        # waist), i.e. the tidiness term was charging the robot for striking.
        # A joint the strike moves is the strike term's business; every other
        # joint is still held composed.
        # EVERY joint the strike does not move. Defining this as "above the
        # hips" left a hole: the strike moves 7 leg joints, carriage covered
        # the 16 waist-and-arm joints, and the 6 in between — both hip yaws
        # and both ankles — were scored by NOTHING (W_POSE is 0 for a strike,
        # so there was no general pose term to catch them either). Measured on
        # the idle phase of a kick cycle: waist_roll 28.4 deg off and hip_yaw
        # 23.2 deg off, a body visibly twisted between kicks while the pelvis
        # was within 2 cm of standing height and the trunk perfectly vertical.
        # The rule is simply: strike owns what it moves, carriage owns the
        # rest, and nothing falls between them.
        # EVERY joint, scored against the SAME phase target the strike uses.
        # The two terms are a coarse/fine pair, not rivals: `strike` is wide
        # so it reaches the policy from standing, `carriage` is tight so the
        # pose it settles into is clean. Splitting them by joint instead left
        # the strike's own joints with only the wide term — which is how the
        # waist ended up pitched 20 deg forward through the idle phase with
        # nothing objecting.
        self._carriage_ids = np.arange(self.nj, dtype=int)
        names = list(self.robot.joint_names)
        self._arm_slots = np.array(
            [k for k, i in enumerate(self._carriage_ids)
             if any(x in names[i] for x in ("shoulder", "elbow", "wrist"))],
            dtype=int)

    # --- the pose ---------------------------------------------------------

    def _solve_strike(self) -> np.ndarray:
        raise NotImplementedError

    def _balance_cost(self, q) -> tuple[float, float, float, float]:
        import mujoco
        d = self._solve_data
        mujoco.mj_resetDataKeyframe(self.model, d, self.key_stand)
        d.qpos[self.joint_qpos_adr] = q
        mujoco.mj_forward(self.model, d)
        return _com_offset_from_foot(self.model, d, self._mass_col,
                                     self.foot_geom_ids, self.SUPPORT_SIDE)

    @property
    def _solve_data(self):
        import mujoco
        if not hasattr(self, "_sd"):
            self._sd = mujoco.MjData(self.model)
            self._mass_col = self.model.body_mass[:, None]
        return self._sd

    # --- reward -----------------------------------------------------------

    # --- the cycle ---------------------------------------------------------

    def cycle_phase(self) -> float:
        """Where in the strike cycle we are, in [0, 1)."""
        if self.CYCLE_S <= 0.0:
            return 0.0
        n = max(int(round(self.CYCLE_S / C.CTRL_DT)), 1)
        return ((self.step_count + self._phase0) % n) / n

    def strike_amplitude(self) -> float:
        """How extended the strike should be RIGHT NOW, in [0, 1].

        A hold is the constant-1 case, so every term below is written once.
        """
        if self.CYCLE_S <= 0.0:
            return 1.0
        p = self.cycle_phase()
        r0, r1 = self.CYCLE_RISE
        f0, f1 = self.CYCLE_FALL
        if p < r0 or p >= f1:
            return 0.0
        if p < r1:
            u = (p - r0) / (r1 - r0)
        elif p < f0:
            return 1.0
        else:
            u = 1.0 - (p - f0) / (f1 - f0)
        return float(u * u * (3.0 - 2.0 * u))      # smoothstep: no velocity step

    def _get_obs(self) -> np.ndarray:
        # The command slots carry the clock for a cycling strike. They are
        # free here: these tasks are paid to IGNORE a drive command, so the
        # slots would otherwise be pinned at zero.
        if self.CYCLE_S > 0.0:
            import math
            p = self.cycle_phase()
            self.twist_cmd[:] = (math.sin(2.0 * math.pi * p),
                                 math.cos(2.0 * math.pi * p),
                                 self.strike_amplitude())
        return super()._get_obs()

    def _sample_commands(self) -> None:
        if self.CYCLE_S > 0.0:
            return                      # the clock owns these slots
        super()._sample_commands()

    def carriage_target_rel(self):
        return self.strike_amplitude() * self._strike_rel[self._carriage_ids]

    def carriage_weights(self):
        """Per-joint grip. Full at idle; the ARMS are released as the strike
        extends, because paying for low angular momentum while pinning the
        only limbs that could cancel it asks for something unreachable. They
        are gripped again through the idle stretch, which is where the
        composed look actually matters."""
        w = np.ones(len(self._carriage_ids))
        if self._arm_slots.size:
            w[self._arm_slots] = 1.0 - self.strike_amplitude()
        return w

    def pose_now(self) -> np.ndarray:
        """The joint target for this instant: default, the strike, or between."""
        return self.default_pose + self.strike_amplitude() * self._strike_rel

    def support_sides(self) -> tuple[str, ...]:
        """Which feet carry the weight. A punch stands on both."""
        # Mid-cycle the strike is not yet committed, so the body is still over
        # both feet; scoring it against one foot would charge it for standing
        # normally during the 55% of the cycle it is meant to be idle.
        if self.CYCLE_S > 0.0 and self.strike_amplitude() < 0.6:
            return ("left", "right")
        return (self.SUPPORT_SIDE,)

    def _solved_pelvis_height(self) -> float:
        """Pelvis height of the solved pose with the support sole on the floor."""
        import mujoco
        d = self._solve_data
        mujoco.mj_resetDataKeyframe(self.model, d, self.key_stand)
        d.qpos[self.joint_qpos_adr] = self._strike_joints
        mujoco.mj_forward(self.model, d)
        low = min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0])
                  for g in self._support_geoms)
        return float(d.xpos[self.trunk_body_id][2]) - low

    def target_height(self) -> float:
        """Blends back to standing height as the strike recovers."""
        amp = self.strike_amplitude()
        return self.stand_z + amp * (self._strike_height - self.stand_z)

    def com_over_support(self) -> tuple[float, float]:
        """CoM offset from the support foot, in units of its half-extent.

        `data.subtree_com[0]` is the whole-body CoM and mj_forward has already
        computed it, so this costs a footprint scan and no dynamics.
        """
        com = self.data.subtree_com[0]
        x0, x1, y0, y1 = self._support_box()
        hx, hy = max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)
        return (float(com[0] - (x0 + x1) / 2) / hx,
                float(com[1] - (y0 + y1) / 2) / hy)

    def _support_box(self) -> tuple[float, float, float, float]:
        d, m = self.data, self.model
        geoms = (self._support_geoms if self.CYCLE_S <= 0.0 else
                 [g for side in self.support_sides()
                  for g in self.foot_geom_ids[side]])
        xs0 = xs1 = ys0 = ys1 = None
        for g in geoms:
            r = float(m.geom_size[g][0])
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * float(m.geom_size[g][1])
            p1, p2 = d.geom_xpos[g] - axis, d.geom_xpos[g] + axis
            a0, a1 = min(p1[0], p2[0]) - r, max(p1[0], p2[0]) + r
            b0, b1 = min(p1[1], p2[1]) - r, max(p1[1], p2[1]) + r
            xs0 = a0 if xs0 is None else min(xs0, a0)
            xs1 = a1 if xs1 is None else max(xs1, a1)
            ys0 = b0 if ys0 is None else min(ys0, b0)
            ys1 = b1 if ys1 is None else max(ys1, b1)
        return float(xs0), float(xs1), float(ys0), float(ys1)

    def _strike_foot_clearance(self) -> float:
        """How far the striking sole is above the support sole, right now."""
        d, m = self.data, self.model

        def sole(side):
            return min(float(d.geom_xpos[g][2] - m.geom_size[g][0])
                       for g in self.foot_geom_ids[side])

        return sole(self.STRIKE_SIDE) - sole(self.SUPPORT_SIDE)

    def stance_ok(self) -> float:
        """Support foot down, striking foot clear. 1.0 when both hold.

        On a CYCLE the expectation inverts as the strike recovers: the point
        of the task is that it comes back down and stands on both feet, so
        scoring "one foot up" for the whole episode would pay it to stay on
        one leg — the opposite of what was asked for.
        """
        c = self._foot_contacts()
        if self.CYCLE_S > 0.0 and self.strike_amplitude() < 0.6:
            return float(c["left"] and c["right"])
        return float(c[self.SUPPORT_SIDE] and not c[self.STRIKE_SIDE])

    def _compute_reward(self) -> tuple[float, float]:
        total, terms = super()._compute_reward()
        # `still` and `stay_home` come from the idle. A cycling strike is
        # motion by construction, so charging it for moving while it throws
        # the kick prices the task itself. They fade out with the strike and
        # come back for the idle stretch, which is where stillness is meant.
        if self.CYCLE_S > 0.0:
            quiet = 1.0 - self.strike_amplitude()
            for k in ("still", "joint_vel_penalty"):
                if k in terms:
                    total -= terms[k] * (1.0 - quiet)
                    terms[k] *= quiet
        rel = self._joint_pos_rel()
        amp = self.strike_amplitude()
        want = amp * self._strike_rel[self._strike_ids]
        err = rel[self._strike_ids] - want
        strike = self.W_STRIKE * float(
            np.exp(-float((err ** 2).sum()) / self.STRIKE_STD2))
        stance = self.W_STANCE * self.stance_ok()
        bx, by = self.com_over_support()
        balance = self.W_BALANCE * float(
            np.exp(-(bx * bx + by * by) / self.BALANCE_STD2))
        # `feet_planted` and `flat_feet` are two-foot terms: the first is
        # switched off by W_PLANTED, and the second would pay a kicking foot
        # for being level in mid-air, so it is scored on the support only.
        total = total - terms.pop("flat_feet", 0.0)
        spin_cost = 0.0
        if self.W_MOMENTUM:
            import mujoco
            mujoco.mj_subtreeVel(self.model, self.data)   # mj_forward does NOT
            lz = abs(float(self.data.subtree_angmom[0][2]))
            spin_cost = -self.W_MOMENTUM * min(lz / self.MOMENTUM_REF, 1.0)

        arrive = 0.0
        if self.W_ARRIVE and amp > 0.9:
            g = self.foot_geom_ids[self.STRIKE_SIDE][0]
            bid = int(self.model.geom_bodyid[g])
            v = float(np.linalg.norm(self.data.cvel[bid][3:6]))
            arrive = -self.W_ARRIVE * min(v / self.ARRIVE_REF, 1.0)

        lift = 0.0
        if self.W_LIFT:
            # PROGRESS PAY, not a Gaussian around the target. A Gaussian at
            # std2 0.02 m^2 against a 0.48 m target pays e^-9 to a robot that
            # has lifted its foot 5 cm — i.e. it is FLAT exactly where a
            # policy first tries, so there is nothing pulling the attempt
            # bigger. Measured on the chain that used it: the policy really
            # was tracking the clock (probing the ONNX, the hip action swings
            # +0.385 idle -> -0.232 at full extension) but committed only
            # ~11% of the hip travel, and the foot never left the floor.
            # Linear in how far up the foot actually is: every centimetre
            # pays, and it saturates at the height the phase asks for.
            want = amp * self.strike_foot_clear
            got = self._strike_foot_clearance()
            # MONOTONE and capped, deliberately. Making it peak AT the target
            # and fall off on both sides looked like the right way to ask for
            # a controlled kick instead of a ballistic one — and measured
            # worse on both counts: kick height fell back from 105% to 13%,
            # survival stayed 0/5, kicks per episode dropped 1.0 -> 0.2 and
            # yaw drift doubled. Punishing overshoot lowers the expected value
            # of ATTEMPTING a kick, because a policy whose kick is still
            # variable is then penalised on both sides of it, and the safe
            # move is not to kick. Keep the one-sided ramp; control has to
            # come from the terms that price falling, not from taking the pay
            # away when it tries too hard.
            frac = 1.0 if want <= 1e-6 else min(max(got / want, 0.0), 1.0)
            lift = self.W_LIFT * frac
        terms["strike"] = strike
        terms["stance"] = stance
        terms["balance"] = balance
        terms["lift"] = lift
        terms["spin_cost"] = spin_cost          # <= 0, a penalty (AGENTS.md)
        terms["arrive"] = arrive                # <= 0, a penalty
        total += strike + stance + balance + lift + spin_cost + arrive
        return float(total), terms

    # --- spawn ------------------------------------------------------------

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.last_spawn = "standing"
        if self.CYCLE_S > 0.0:
            # Start at a random point in the cycle and pose the body to match,
            # so every phase is sampled from step one. Same reason the squat
            # spawns squatting: an unsampled state's value is never learned,
            # and a policy that only ever sees the cycle from phase 0 has to
            # discover the extended phases through exploration alone.
            import mujoco
            n = max(int(round(self.CYCLE_S / C.CTRL_DT)), 1)
            # Half the episodes start INSIDE the strike window rather than
            # uniformly over the cycle. Uniform sampling spends most of its
            # starts in the idle stretch — with a 5 s cycle only about a
            # quarter land near full extension — and the state whose value
            # decides whether the kick is ever attempted is the extended one.
            if self._rng.random() < 0.5:
                lo, hi = self.CYCLE_RISE[0], self.CYCLE_FALL[1]
                self._phase0 = int(self._rng.integers(int(lo * n), int(hi * n)))
            else:
                self._phase0 = int(self._rng.integers(0, n))
            amp = self.strike_amplitude()
            if amp > 0.01:
                d = self.data
                d.qpos[self.joint_qpos_adr] = self.pose_now()
                mujoco.mj_forward(self.model, d)
                low = min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0])
                          for side in self.support_sides()
                          for g in self.foot_geom_ids[side])
                d.qpos[self._root_qpos + 2] += 0.002 - low
                mujoco.mj_forward(self.model, d)
                self._refresh_derived()
                self.last_spawn = "mid-kick"
            return self._get_obs(), info
        if self._rng.random() < self.spawn_in_pose_prob:
            import mujoco
            d = self.data
            d.qpos[self.joint_qpos_adr] = self._strike_joints
            # stand the SUPPORT sole on the floor: the strike shortens one leg
            # and lifts the other, so the nominal standing height is wrong.
            mujoco.mj_forward(self.model, d)
            low = min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0])
                      for g in self._support_geoms)
            d.qpos[self._root_qpos + 2] += 0.002 - low
            mujoco.mj_forward(self.model, d)
            self._refresh_derived()
            self.last_spawn = "strike"
            obs = self._get_obs()
        return obs, info


class G1FrontKickEnv(G1StrikeEnv):
    """Mae geri: stand on one leg, the other extended forward and up.

    The pose is solved, not typed. The knee and hip of the kicking leg are
    fixed by the strike itself (leg straight, thigh up); what the solver finds
    is the SUPPORT configuration that makes it standable — the hip adduction
    that brings the support foot under the body's midline, and the fore-aft
    lean that answers the CoM being thrown forward by the raised leg.
    """

    # Negative swings the leg forward (measured). -1.1 gave 0.545 m of foot
    # clearance; the balance solver finds a centred pose at EVERY height it
    # was swept over (0.415 m at -0.9 through 1.040 m at -1.9), because the
    # counter-lean compensates — so the old height was a conservative guess,
    # not a limit. -1.5 puts the foot 0.806 m up, about waist height.
    # OFF, measured. Solving the arm swing that cancels the kick's angular
    # momentum is correct arithmetic (residual exactly 0 on both axes, for a
    # subtle 12 deg / 7 deg motion) and it made the robot WORSE on the very
    # thing it targets: yaw drift 27 -> 139 deg clean, 23 -> 234 under noise,
    # survival 5/5 -> 2/5, recovery 97% -> 65%.
    #
    # The reason is the assumption underneath it. A feed-forward cancellation
    # is only valid if the policy TRACKS the reference, and this one does not:
    # measured at full extension it performs -21% of the commanded arm swing
    # (it moves them the wrong way) and 28% of the kick. Cancelling momentum
    # that is not being generated, with a motion that is not being followed,
    # is a disturbance rather than a correction.
    #
    # The closed-loop form is the one worth trying: score the robot's ACTUAL
    # angular momentum and let it discover its own counter-motion, instead of
    # prescribing the arm angles that would work if everything else were
    # perfect. Kept here, off, because the solver and its numbers are the
    # evidence for that.
    ARM_COUNTERSWING = False
    W_MOMENTUM = 1.5             # the closed-loop replacement for it
    W_ARRIVE = 2.0               # price the SPEED at the top, not the height
    # -1.5 (0.806 m, waist height) is a target the cycle cannot recover from:
    # the policy that reached it threw the foot to 1.086 m and toppled AT the
    # peak on every seed. -1.1 gives 0.545 m, which 13.2's held kick sustained
    # for 20 s on 4/5 seeds — a height this robot is DEMONSTRABLY able to
    # carry, rather than one it reaches badly. Ask for what the machine can
    # do well; the lift term then saturates somewhere reachable instead of
    # paying 13% forever.
    # -0.8 (0.36 m, a low front kick). The height ladder has been walked
    # down twice on measurement, not preference: -1.5 (0.806 m) produced a
    # policy that threw the foot to 1.086 m and toppled AT the peak on every
    # seed; -1.1 (0.545 m, the height 13.2's HELD kick sustained) still gave
    # 0/5 from a standing start. Every rung above this one buys height with
    # survival. Ask for a kick the machine can finish and recover from.
    KICK_HIP_PITCH = -1.1
    MIN_FOOT_CLEAR = 0.25        # m above the support sole
    # A curriculum stage ladders the HEIGHT (physics/difficulty), never the
    # reward — AGENTS.md. Read from the environment because the lab's stage
    # machinery passes knobs that way, not through argv.
    HEIGHT_ENV = "MICRODUCK_G1_KICK_HIP"

    # Throw it, hold it briefly, come back to the idle stance, repeat. A held
    # one-leg pose was the wrong shape for a karate kick: it does not need to
    # stand on one leg, it needs to RECOVER.
    # 2.0 s was measured to be too fast to LEARN, even at the lowest rung:
    # a 0.3 s throw means shifting the whole body onto one foot and swinging
    # the other leg up inside 15 control steps, and because falling ends the
    # episode the policy correctly settled on standing through the kick
    # window and eating the loss — foot clearance 0.001-0.002 m at every rung
    # of a 6M chain, i.e. it never once kicked. The cycle is now a ladder
    # rung in its own right (slow and low, to fast and high); at 4 s the
    # throw takes 0.6 s.
    CYCLE_S = 4.0
    CYCLE_ENV = "MICRODUCK_G1_KICK_CYCLE"
    # 4.0, and it stays there. MEASURED both ways from a standing start:
    # at 4.0 the policy holds 5/5 for 20 s and kicks 0.104 m; doubling it to
    # 8.0 bought height (0.208 m) and cost everything else — 0/5 surviving,
    # recovery to both feet halved, yaw drift 27 -> 141 deg. Height bought by
    # outbidding survival is not height worth having. The arm counter-swing
    # is the attempt to raise the ceiling instead of raising the bid.
    W_LIFT = 4.0
    FALL_HEIGHT = 0.45

    def _solve_strike(self) -> np.ndarray:
        jid = self._jname
        base = self.default_pose.astype(np.float64)
        s, sup = self.STRIKE_SIDE, self.SUPPORT_SIDE
        env = os.environ.get(self.HEIGHT_ENV)
        hip = float(env) if env else self.KICK_HIP_PITCH

        def build(hip_roll, hip_pitch, waist_pitch):
            q = base.copy()
            q[jid[f"{s}_hip_pitch_joint"]] += hip
            q[jid[f"{s}_knee_joint"]] = 0.0                  # leg straight
            q[jid[f"{s}_hip_roll_joint"]] += -0.10
            q[jid[f"{sup}_hip_roll_joint"]] += hip_roll
            q[jid[f"{sup}_hip_pitch_joint"]] += hip_pitch
            q[jid[f"{sup}_knee_joint"]] += 0.25
            q[jid["waist_pitch_joint"]] += waist_pitch
            return q

        best = None
        for hr in np.arange(-0.35, -0.05, 0.025):
            for hp in np.arange(-0.60, 0.10, 0.05):
                for wp in np.arange(-0.10, 0.45, 0.05):
                    q = build(hr, hp, wp)
                    dx, dy, lx, ly = self._balance_cost(q)
                    clear = self._foot_clearance(q)
                    if clear < self.MIN_FOOT_CLEAR:
                        continue
                    cost = (dx / (lx / 2)) ** 2 + (dy / (ly / 2)) ** 2
                    if best is None or cost < best[0]:
                        best = (cost, q, dx, dy, lx, ly, clear)
        if best is None:
            raise ValueError("no front-kick pose clears the support foot")
        cost, q, dx, dy, lx, ly, clear = best
        if abs(dx) > lx / 2 or abs(dy) > ly / 2:
            raise ValueError(
                f"the solved front kick is not statically balanced: CoM "
                f"({dx:+.3f},{dy:+.3f}) outside a {lx:.3f}x{ly:.3f} m foot")
        self.strike_foot_clear = clear
        if self.ARM_COUNTERSWING:
            q = self._add_arm_counterswing(q)
        else:
            self.arm_counterswing = (0.0, 0.0)
            self.arm_residual = (0.0, 0.0)
        # Re-measure AFTER the arms move: the balance solve ran on a
        # leg-only pose, and recording its number would describe a pose that
        # is not the one being trained.
        dx, dy, lx, ly = self._balance_cost(q)
        self.strike_com_offset = (dx, dy)
        if abs(dx) > lx / 2 or abs(dy) > ly / 2:
            raise ValueError(
                f"the arm counter-swing pushed the kick off balance: CoM "
                f"({dx:+.3f},{dy:+.3f}) outside a {lx:.3f}x{ly:.3f} m foot")
        return q

    def _add_arm_counterswing(self, q) -> np.ndarray:
        """Arm motion that cancels the angular momentum the kick generates.

        This is SOLVED, not posed. A leg swung forward 0.145 m off the
        centreline throws the body about the vertical axis, and nothing else
        in the reference opposes it — MEASURED on the trained kick: the swing
        makes Lz = -0.174 kg m^2/s and the robot could only absorb it by
        yawing, which is exactly the 27-141 deg of drift per 20 s that showed
        up in every evaluation. Worse, `carriage` was actively pinning all 14
        arm joints at their default while this happened, so the one set of
        limbs that could have cancelled it was being held still ON PURPOSE.

        Two arm patterns give independent authority over the two axes that
        matter, so it is a 2x2 solve rather than a guess:
          antisymmetric shoulder PITCH -> mostly yaw   (Lz -0.789 per rad)
          same-sign shoulder ROLL      -> mostly roll  (Lx -0.756 per rad)
        Solving both to zero costs 12 deg of pitch and 7 deg of roll — a
        subtle motion, not a flail, which matters because the carriage term
        then holds the arms to THIS instead of to the default pose.
        """
        import mujoco

        names = list(self.robot.joint_names)
        d = self._solve_data
        rise = (self.CYCLE_RISE[1] - self.CYCLE_RISE[0]) * max(self.CYCLE_S, 1e-6)
        rel = q - self.default_pose

        def momentum(qd):
            mujoco.mj_resetDataKeyframe(self.model, d, self.key_stand)
            d.qpos[self.joint_qpos_adr] = self.default_pose + 0.5 * rel
            d.qvel[self.joint_qvel_adr] = qd
            mujoco.mj_forward(self.model, d)
            out = np.zeros(3)
            for b in range(self.model.nbody):
                out += self.model.body_mass[b] * np.cross(
                    d.xipos[b] - d.subtree_com[0], d.cvel[b][3:6])
            return out

        def pattern(**kw):
            v = np.zeros(self.nj)
            for k, val in kw.items():
                v[names.index(k)] = val
            return v

        rate = 1.5 / rise                    # smoothstep's peak slope
        leg = momentum(rel * rate)
        p_pitch = pattern(left_shoulder_pitch_joint=-1.0,
                          right_shoulder_pitch_joint=+1.0)
        p_roll = pattern(left_shoulder_roll_joint=+1.0,
                         right_shoulder_roll_joint=+1.0)
        a_pitch, a_roll = momentum(p_pitch * rate), momentum(p_roll * rate)
        M = np.array([[a_pitch[0], a_roll[0]], [a_pitch[2], a_roll[2]]])
        try:
            k_pitch, k_roll = np.linalg.solve(M, -np.array([leg[0], leg[2]]))
        except np.linalg.LinAlgError:        # no authority: leave the arms be
            self.arm_counterswing = (0.0, 0.0)
            return q
        # A counter-swing big enough to be a flail is a solve gone wrong, not
        # a kick — clamp and record it rather than shipping a windmill.
        lim = 0.8
        k_pitch = float(np.clip(k_pitch, -lim, lim))
        k_roll = float(np.clip(k_roll, -lim, lim))
        self.arm_counterswing = (k_pitch, k_roll)
        self.arm_residual = tuple(
            float(v) for v in (leg + k_pitch * a_pitch + k_roll * a_roll)[[0, 2]])
        return q + k_pitch * p_pitch + k_roll * p_roll

    def _foot_clearance(self, q) -> float:
        import mujoco
        d = self._solve_data
        mujoco.mj_resetDataKeyframe(self.model, d, self.key_stand)
        d.qpos[self.joint_qpos_adr] = q
        mujoco.mj_forward(self.model, d)
        def sole(side):
            return min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0])
                       for g in self.foot_geom_ids[side])
        return sole(self.STRIKE_SIDE) - sole(self.SUPPORT_SIDE)


class G1PunchEnv(G1StrikeEnv):
    """A straight punch: one arm driven forward to full extension.

    Both feet stay down, so this is much easier than the kick — the balance
    problem is a lean, not a one-leg stance. `stance` is redefined to want
    BOTH feet, and the fall floor goes back to the idle's.
    """

    FALL_HEIGHT = 0.50
    W_STANCE = 1.0

    def support_sides(self) -> tuple[str, ...]:
        return ("left", "right")

    def _solve_level_punch(self, q) -> float:
        """Shoulder pitch that puts the fist level with the shoulder."""
        import mujoco

        d = self._solve_data
        s = self.STRIKE_SIDE

        def body(sub):
            for i in range(self.model.nbody):
                nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if nm and sub in nm:
                    return i
            raise KeyError(f"no body matching {sub!r}")

        fist, shoulder = body(f"{s}_wrist_yaw"), body(f"{s}_shoulder_pitch")
        adr = self._jname[f"{s}_shoulder_pitch_joint"]

        def rise(angle: float) -> float:
            probe = np.asarray(q, np.float64).copy()
            probe[adr] = angle
            mujoco.mj_resetDataKeyframe(self.model, d, self.key_stand)
            d.qpos[self.joint_qpos_adr] = probe
            mujoco.mj_forward(self.model, d)
            return float(d.xpos[fist][2] - d.xpos[shoulder][2])

        lo, hi = -1.8, 0.0          # more negative = higher
        if rise(lo) < 0 or rise(hi) > 0:
            raise ValueError("no shoulder angle puts the fist level")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if rise(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def stance_ok(self) -> float:
        c = self._foot_contacts()
        return float(c["left"] and c["right"])

    def _solve_strike(self) -> np.ndarray:
        jid = self._jname
        q = self.default_pose.astype(np.float64).copy()
        s = self.STRIKE_SIDE
        other = "left" if s == "right" else "right"
        # Punching arm: elbow straight, no roll out, and the shoulder angle
        # SOLVED so the fist ends up level with the shoulder. A hand-typed
        # -1.35 rad looked reasonable and rendered as a 36 deg uppercut with
        # 12.8 cm of reach; the solved angle is both level and further out
        # (28.4 cm), because past the level point the arm is rotating up
        # rather than forward.
        q[jid[f"{s}_shoulder_roll_joint"]] = 0.0
        q[jid[f"{s}_elbow_joint"]] = 0.0
        q[jid[f"{s}_shoulder_pitch_joint"]] = self._solve_level_punch(q)
        # the other arm chambers back at the hip, as a karate guard does
        q[jid[f"{other}_shoulder_pitch_joint"]] += 0.35
        q[jid[f"{other}_elbow_joint"]] += 0.50
        # a small braced stance: knees soft, hips square
        for side in ("left", "right"):
            q[jid[f"{side}_knee_joint"]] += 0.20
            q[jid[f"{side}_hip_pitch_joint"]] -= 0.10
            q[jid[f"{side}_ankle_pitch_joint"]] -= 0.10
        return q
