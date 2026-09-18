"""The tidy brain for a body with an ARM — MARS picks the toys up and drops
them in the basket (`docs/mars-roadmap.md` Phase 5).

    search ──▶ approach ──▶ settle ──▶ reach ──▶ close ──▶ lift ─┐
      ▲         (servo on    (let the   (jaws     (squeeze)  │ holding?
      │          detection)   base come  open,               ▼
      │                       to rest)   IK onto     carry ──▶ deliver ──▶ place ──▶ drop ──▶ retract
      │                       the toy)              (find the  (servo to   (arm     (open   (arm
      └──────────────────────────────────────────────  basket)  the rim)    over    the     back
                                                                            it)     jaw)   HOME)
                                                                                             │
      explore (nothing in sight) ─┘        unstick (asked to move, did not) ──────────────────┘

**A sibling of `brain/tidy.py`, not a subclass.** The duck's loop is
beak-shaped in thirteen places — `skill="ground_pick"`, `Intent.beak`, the
mouth tip's `PICK_REACH_AHEAD`, the rim-topple geometry, `gait.TURN_KICK` for
a cold walker, the blind legs — and every one of those is either absent on a
wheeled base or means something else. Inheriting them is the mistake
`AGENTS.md` names most often ("retarget a term, don't delete it" applies to
rewards; for a whole brain the honest move is a second class). What IS shared
is the state vocabulary, so the `/sim` inspector reads the same picture.

**Five things are structurally simpler here than on the duck, and each was
measured rather than assumed.**

* **No blind leg.** MARS's head camera is 0.2585 m up; a toy at the pick
  standoff (0.335 m ahead of the base origin) sits 35 deg below the optical
  axis against an 84 deg vertical field — so the toy is still in frame when
  the robot stops to grasp it, and stays in frame to 0.264 m. The duck loses
  a floor toy at ~0.5 m and dead-reckons the rest, which is where its
  `blind` / `reach_pad` / `stale_fix` machinery comes from. None of that is
  needed. With the head pitched to its own limit (`HEAD_PITCH`, 0.349 rad)
  the near edge moves in to 0.126 m and the basket marker stays visible from
  0.14 m out, so the head is held at ONE angle for the whole run and there is
  no head-settle window either.
* **No falls and no rim topple.** `world/arena.WorldRobot.fallen()` is always
  False by construction (a planar base has no attitude to lose), so the
  duck's `basket_reach` knife-edge — 0.22 m, where a centimetre either way
  trades toys for topples over 64 paired seeds — has no analogue. The
  standoff here is chosen by the ARM's reach, not by where the toes land.
* **No gait.** A twist command is a twist: `turn` is `(0, 0, wz)` with no
  forward kick, and a turn in place tracks to 0.0005 rad
  (`robots/mars_drive.py`'s measured block). The duck cannot turn in place
  with its head down and cannot turn right from a standstill at all.
* **A command is a LEASE.** `MarsDriver` carries Innate's 0.5 s `cmd_vel`
  watchdog, so this brain must emit a twist every tick — including the zero
  twists, which it does — and a brain that stops talking leaves the base
  still rather than coasting.
* **The grasp is PHYSICS.** A duck's is a weld the World switches on; MARS's
  two blades really hold the block, so `Senses.holding` is a reading of the
  constraint torque at joint6 and not a modelled event
  (`world/arena.World.sense_grip`). There is nothing to retry against a
  probability curve: a close either loads the blade or it does not.

**What the claw can pick, MEASURED in the composed room at 2 ms** (6 spots in
this brain's own standoff band, `scratchpad/probe_room_grasp.py`):

    block  4.0 x 4.0 x 4.0 cm, 20 g    6/6 held
    sock   6.0 x 3.5 x 2.5 cm, 20 g    3/6 held   (the 6 cm axis against a
                                                   58.8 mm open jaw: it holds
                                                   when the yaw happens to
                                                   present the 3.5 cm side)
    brick  3.2 x 1.6 x 0.96 cm, 2.5 g  0/6 held   (a 9.6 mm-tall object is
                                                   below what two blades can
                                                   pinch off a floor; the IK
                                                   residual at that height is
                                                   3-9 mm and the pads close
                                                   above it)

That is a property of a parallel jaw, not of this brain, and it is why
`eval-tidy --robot mars` scatters blocks: the duck's soft bill takes all three
and a 4 cm gripper does not. The mixed `mars-playroom` is still the honest
picture and the per-seed table in the roadmap reports it as one.

**IK is a planning pause, and it is stated rather than hidden.** `reach`,
`lift` and `place` each solve `robots/mars_ik.ArmKinematics.solve_grasp` —
coordinate descent on the robot's OWN private model, never the live world —
which costs 40-120 ms inside one 50 Hz tick. In sim that is a tick that took
longer than 20 ms of wall time; on the robot it would be a real pause before
the arm moves, which is what a planner does. Three solves per toy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .controllers import WanderParams, _column_clearance, wander_from_tof
from .runtime import REGISTRY, Intent, Senses, age_inputs

#: Innate's close command: 0.6 rad past the mechanical stop, which
#: `MarsDriver.set_arm` clamps to `mars.GRIPPER_CLOSED_ON_AIR_RAD`. The error
#: still saturates the 2 N*m ceiling, so the blades squeeze — they just stop
#: where the metal does. `scripts/probe_mars_pick.py`'s `CLOSE_TARGET_RAD`.
CLOSE_RAD = -0.60
#: How wide the jaws open for the approach — the jaw's own STOP, not the 0.60
#: the scripted pick used, and the difference is the measured lever of this
#: whole phase.
#:
#: MEASURED gap between the blade pads: 33.4 mm at 0.30 rad, 58.8 at 0.60,
#: 78.8 at the +0.8727 stop. A 40 mm cube is 40 mm across a FACE and
#: **56.6 mm across its diagonal**, and the playroom scatters toys at a
#: uniform random yaw (`make_playroom`) while the IK constrains the grasp
#: POINT and not the jaw's heading — so the angle between the jaw and the
#: block is effectively random, and at 0.60 rad a block presenting its
#: diagonal has 1 mm of clearance a side. That is what the descent was
#: catching: the blades pushed the block 16-30 mm sideways before they could
#: close on it, 5 picks in 6. At the stop the same block has 11 mm a side.
#:
#: 4b's scripted pick holds 14/16 at 0.60 rad because it PLACES the block at
#: the grasp point with an identity quaternion — axis-aligned with the jaw,
#: which a room is not. Exactly the shape of `AGENTS.md`'s "suspect the
#: harness when train and eval disagree".
OPEN_RAD = 0.8727


@dataclass(frozen=True)
class TidyArmParams:
    # -- the pick standoff ---------------------------------------------------
    #: Where the toy has to be, in the base frame, for the arm to grasp it:
    #: `(shell radius, shell yaw)` about the SHOULDER, so a URDF revision that
    #: moves the arm mount moves the standoff with it rather than aiming the
    #: robot at a point in space. `robots/mars_env.PICK_SPOT` — MEASURED there
    #: as the spot the scripted pick holds at, and re-measured here: in a
    #: +-0.15 rad / 0.22-0.28 m band about it the grasp holds 7/8 at 2 ms
    #: against 3/8 at 5 ms, while a band over the WHOLE +-60 deg arc holds
    #: 5/8. The edges of the shell are where 4b's two misses of 16 were, and a
    #: robot with wheels never has to pick from the edge — that is the whole
    #: point of a mobile base, and it is why this is a point and not an arc.
    spot_radius: float = 0.25
    spot_yaw: float = 0.0
    #: A toy's centre height, class-blind (the duck's `TidyParams.toy_z`) —
    #: and it is the BLOCK's 21 mm, not a compromise across the three kinds.
    #:
    #: The standoff moves by under a millimetre across 5.8-21.0 mm, so this
    #: looks free and is not: `_locate` intersects the sight ray with the
    #: plane `toy_z`, so a target height that is 6 mm low stretches the range
    #: by `6/223` of it — MEASURED as a systematic **+9 to +10 mm of range on
    #: every spot** with an ideal detector and `toy_z` at the old 15 mm
    #: compromise. Against a 40 mm block in a 58.8 mm jaw the margin is
    #: 9.4 mm a side, so that one number was the whole of it. A flatter toy is
    #: now over-ranged instead, which costs nothing because a 9.6 mm brick is
    #: not graspable by this claw at all (the module docstring's table).
    toy_z: float = 0.021
    #: Fresh detection frames averaged while STANDING STILL in `settle`, and
    #: the mean is what the grasp is solved against.
    #:
    #: The duck's `aim` state, for the duck's reason: "walking rocks the head
    #: a few hundredths of a radian, which at 0.4 m is centimetres of range —
    #: more than the release geometry has to spare", and the mean of the
    #: standing looks beats the last one. Here the noise is the detector's own
    #: (`datasheet`: 1 deg of bearing, and an elevation that moves the range
    #: by tens of millimetres), MEASURED at 0.6-14.7 mm of lateral and
    #: 2-54 mm of range error on a single frame against 0.7-2.9 / 9-10 mm
    #: with an ideal one — so after the pitched-bearing fix above, the
    #: detector IS the binding constraint and this is the cheapest filter
    #: against it. 0.8 s at the camera's 10 Hz is ~8 frames.
    settle_fixes: int = 8
    #: Stop when the toy's estimate is within this of the standoff point (m).
    #: The grasp band measured above is +-3 cm of radius, so half of it.
    stop_tol: float = 0.03
    #: ...and squared up to within this of the standoff's bearing (rad).
    align_tol: float = 0.08
    approach_speed: float = 0.30        # m/s; `MAX_CMD_LINEAR` is 0.8
    #: ...and the speed and turn rate used while CARRYING a toy. **Both ship
    #: at the approach's own values — the knob exists and is REFUTED.**
    #:
    #: The grip is two blades at 2 N*m, not a weld, so a carried block is held
    #: by friction against whatever the base does, and one loss had a clear
    #: mechanism: in the mixed `mars-playroom` (seed 0) the one block carried
    #: the length of the room — 2.3 m, `t1` from (-1.16, 0.60) to the basket
    #: at (1.15, 0.90) — was dropped 12.5 s into `deliver`, at full speed with
    #: the obstacle guard turning at the full 1.0 rad/s.
    #:
    #: So it was A/B'd, PAIRED on 8 seeds x 300 s (`scratchpad/ab_carry.py`),
    #: 0.20 m/s + 0.5 rad/s against 0.30 + 1.0:
    #:
    #:     arm      tidied   per seed              picked   lost in
    #:     gentle   0.750    5 3 6 4 5 5 5 3         48     place 34, deliver 5
    #:     full     0.792    6 5 5 5 5 4 4 4         56     place 33, deliver 10
    #:
    #:     paired gentle-full: -0.250 toys, SE 0.412, better 3 / worse 4 / tied 1
    #:
    #: It does exactly what it was built for — **deliver losses halve, 10 to
    #: 5** — and it does not buy a toy, because the slower carry costs 8 picks
    #: over the battery and a time-bound benchmark pays for that. The
    #: difference is unresolved at this size (a real effect up to ~0.8 toys
    #: would not have been seen), so the simpler configuration ships and the
    #: knob keeps the table. `place` is where the toy is lost either way, 33
    #: or 34 times of ~50 picks, and that is the shape the next phase owes.
    carry_speed: float = 0.30
    carry_wz: float = 1.0
    creep_speed: float = 0.12           # the last 0.3 m, so a late stop lands nearer
    creep_at: float = 0.30
    k_turn: float = 2.0                 # rad/s per rad of bearing error
    turn_wz: float = 1.0                # rad/s for a turn in place (tracks to 5e-4)
    search_wz: float = 0.8              # rad/s while scanning
    settle_s: float = 0.8               # the base comes to rest (station keeping latches at 0.4 s)

    # -- the arm -------------------------------------------------------------
    #: How far ABOVE the toy the jaws are opened first, before descending.
    #:
    #: Two jobs, and both were measured as failures without it. (1) The
    #: approach path: slewing from ARM_HOME straight to a pose whose claw is
    #: AROUND the toy sweeps the blades in sideways and **pushed the block
    #: 16-22 mm** before the jaw could close — 4b saw the same shape in the
    #: env ("what misses is the APPROACH, not the grasp") and named an
    #: above-the-block rung as the fix. (2) It is where the SAG is measured:
    #: a hover pose 0.08 m up is close enough to the grasp pose that the
    #: gravity torque, and so the compliance error, is nearly the same.
    hover_m: float = 0.08
    lift_m: float = 0.12                # how far the GRASP POINT rises after the close
    #: A sag correction larger than this is not trusted (m) — something other
    #: than compliance is wrong (the arm on the floor, a blade on the toy),
    #: and pre-compensating by it would aim the claw somewhere arbitrary.
    sag_max_m: float = 0.08
    hover_s: float = 1.8
    reach_s: float = 1.6                # dwell in each arm phase — 6 rad/s over ~3 rad of travel
    close_s: float = 1.2
    lift_s: float = 1.2
    place_s: float = 2.0
    drop_s: float = 0.8
    retract_s: float = 1.2
    #: Restarts for the grasp / place solves. MEASURED: at 3 restarts the
    #: residual at the standoff is 0.1-0.5 mm but the solver falls into a
    #: 20-30 mm local minimum on maybe a fifth of targets; at 8 it is 0.1-0.4 mm
    #: everywhere inside the shell at the heights this brain uses, for 60-120 ms.
    ik_restarts: int = 8
    #: A solve worse than this is not attempted — the toy is given up rather
    #: than groped at (`robots/mars_ik.IK_TOLERANCE_M` is the same line drawn
    #: from 4b's two populations: every spot the claw was PUT on held, and the
    #: two misses of 16 were 13.7 and 15.1 mm off).
    ik_tol: float = 0.012
    #: The commanded arm target is SLEWED at this rate (rad/s), because
    #: `MarsDriver.set_arm` applies none and the servo it models cannot
    #: teleport. `robots/mars.MAX_TARGET_RATE_RAD_S`, which also records what
    #: leaving it out did: the first `tidy_arm` run wrote each IK solution as
    #: one `Intent.arm` and threw the 20 g block out of the room on 6 of 6
    #: picks — a failure that reads exactly like the 5 ms ejection and is a
    #: different bug. The brain's own control tick is 50 Hz, so one tick is
    #: `rate * 0.02` = 0.12 rad.
    slew_rad_s: float = 6.0

    # -- the basket ----------------------------------------------------------
    #: Base origin to basket CENTRE at the drop (m). MEASURED: the grasp point
    #: reaches 0.42 m ahead at the drop height with a 0.4 mm residual, and
    #: 0.45 - `drop_inset` = 0.37 m is comfortably inside that. It also keeps
    #: the chassis front face (0.17 m ahead of the origin) 0.13 m clear of the
    #: rim's outer face, which is the whole of the duck's rim problem removed
    #: by having an arm.
    basket_reach: float = 0.45
    #: The drop point is pulled this far back from the basket centre along the
    #: approach line. The tray is 0.3 m square, so 0.08 m short of the centre
    #: is still 0.07 m inside the near rim — and it buys 8 cm of the arm's
    #: forward reach, which is what makes a 0.45 m standoff possible at all.
    drop_inset: float = 0.08
    #: Height of the grasp point at the drop (m). The rim is 0.06 and
    #: `World.in_basket` wants the toy under rim + 0.05, so the toy falls
    #: 6-10 cm into the tray and settles. Read off the scenario's `Basket`
    #: through `rim_z`, which is why this is a CLEARANCE and not a height.
    drop_clear: float = 0.06
    basket_z: float = 0.08              # the marker's height (rim + 0.02, `compose`)
    basket_confirm_range: float = 0.8   # a fix from closer than this is trustworthy
    #: Inside this of the basket centre a toy is in the tray or against the
    #: rim: leave it (the duck's `basket_inside`).
    basket_inside: float = 0.21
    #: ...and the base keeps this far from the centre on any leg that is not a
    #: delivery. The chassis front is 0.17 m ahead of the origin and the tray's
    #: half-width is 0.15, so 0.37 m is where the bumper would touch the rim;
    #: Phase 3b measured `wander` PINNED on that rim for 53 of 60 s, because a
    #: 6 cm rim is 11 cm below the lidar's scan plane and the scanner cannot
    #: see it at all. This is the keep-out that replaces seeing it.
    basket_keepout: float = 0.40

    # -- driving -------------------------------------------------------------
    #: Turn away when the adapted ToF's centre columns report closer than this
    #: (m, in the LASER's frame — `tof_from_lidar` re-reads each ray as a
    #: base-frame range, so this is from the base origin). The chassis front
    #: face is 0.17 m out, so 0.40 leaves 0.23 m. Phase 3b's note is the
    #: reason it is not the duck's 0.30: that leaves a duck 0.21 m of
    #: clearance and MARS 0.014 m.
    guard_m: float = 0.40
    #: ...and the guard is OFF inside this of the current target, because the
    #: last leg of an approach drives deliberately at something.
    guard_off_m: float = 0.60
    explore_s: float = 5.0
    #: After the guard has fired, drive STRAIGHT for this long once the way is
    #: clear, whatever the servo wants.
    #:
    #: The duck's `TidyParams.detour_s`, ported for the reason it exists
    #: there: "it wanted to turn back into it — measured: a left/right
    #: ping-pong at a toy, 4 min". MEASURED here in `mars-playroom` seed 1:
    #: the `deliver` leg ran **45 s without arriving** with the distance
    #: bouncing 1.49 -> 1.32 -> 1.38 m and the note alternating on
    #: "· blocked", and the sock was dropped where it stood. The guard turns
    #: away, the servo turns back, and neither ever wins.
    detour_s: float = 1.0
    #: ToF returns closer than this are the ROBOT'S OWN ARM while it carries
    #: something, not the room.
    #:
    #: MEASURED in the carry pose (`_clearance`'s trace): the adapted 8x8's
    #: right-hand columns read **0.12-0.17 m** and the raw scan's nearest
    #: return is 0.196-0.204 m at -8 to -17 deg — the extended arm crossing
    #: the scanner's 0.17 m plane. `robots/mars.FOOTPRINT_M` is 0.12 and
    #: measured for the arm FOLDED at ARM_HOME, which is Phase 3b's own
    #: warning ("the shadow moves with the arm, so ignoring returns inside the
    #: footprint belongs to the consumer") coming due. The centre columns the
    #: forward guard reads are clean, so this matters to `wander_from_tof` in
    #: `carry_explore`, which reads all eight and would steer away from its
    #: own elbow for the whole leg. The duck has the same rule for a toy in
    #: its beak (`TidyParams.hold_blind_m`, 0.12 m).
    hold_blind_m: float = 0.25
    #: Asked to move, not moving: `|speed|` under this while a forward command
    #: stands, for this long, is stuck (`Wander`'s own `unstick`, on the
    #: quantity the drive PD regulates).
    stuck_speed: float = 0.04
    stuck_s: float = 1.2
    unstick_s: float = 1.5
    unstick_speed: float = -0.25        # the base reverses as readily as it drives
    unstick_wz: float = 1.0

    # -- the camera ----------------------------------------------------------
    #: Head pitch held for the WHOLE run, rad, positive = down (the duck's
    #: convention; `MarsBody.frames()` flips the sign for `joint_head`). 0.349
    #: is the joint's own limit. MEASURED as strictly better than level: with
    #: the optical axis 20 deg down the visible band for a floor object is
    #: 0.126 m to the horizon instead of 0.264 m, the basket marker is visible
    #: from 0.14 m, and nothing a floor robot needs is above +22 deg. Holding
    #: it FIXED is what removes the duck's `head_settle_s` window — every
    #: detection in a run is taken through the same lens pose.
    head_pitch: float = 0.349
    #: Camera x/y in the base frame at the HOME head pose (m), for placing a
    #: detection. Not the duck's single `cam_ahead`: MARS's left eye sits
    #: 29.5 mm to the LEFT of the centreline, which is three quarters of a
    #: block, and the duck's model has no term for it because a duck's camera
    #: is on its centreline. Read off the model by `ArmKinematics` when one is
    #: built; these are the fallback and the measured values.
    cam_x: float = 0.0025
    cam_y: float = 0.0295
    cam_z_fallback: float = 0.2585
    #: Beyond this a floor object's elevation says almost nothing about range
    #: (the duck's `far_range`, same geometry and the same lesson: at 2.3 m
    #: the map is 34 m per radian). A far sighting is a DIRECTION.
    far_range: float = 1.5
    min_conf: float = 0.1
    max_retries: int = 2
    done_after_scans: int = 6


class TidyArm:
    """Find toys, pick them up with the arm, drop them in the basket."""

    kind = "tidy_arm"
    wants_head = True
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.35          # a 6 Hz scanner's frame is up to 167 ms old

    def __init__(self, p: TidyArmParams = TidyArmParams(), truth: bool = False,
                 seed: int = 0):
        """`truth=True` reads the world's own toy pose instead of the
        detector's — the CONTROL ARM, and `eval-tidy` does NOT use it.

        Everything the benchmark reports is the sensed loop: the toy's
        position comes from one detection's bearing and elevation through
        `_locate`, exactly as the duck's does. The truth option exists so that
        "the pick missed" can be split into "the estimate was wrong" and "the
        arm was wrong" without changing anything else, which is the split that
        `AGENTS.md`'s kick-error entry says to measure before believing either
        (`--truth` on `scripts/probe_tidy_arm.py`). A brain constructed with
        it has no honest number to report.
        """
        self.p = p
        self.truth = bool(truth)
        self.seed = int(seed)
        self._kin = None            # lazy: see `kin()`
        #: The harness hands the brain its world and its own id — the world for
        #: `rim_z` (the basket's geometry is a fact about the ROOM, not a
        #: constant a brain should carry) and, under `truth`, for the control
        #: arm's toy pose. Both are None for a brain stepped on synthetic
        #: senses, which is how `tests/test_tidy_arm.py` drives the machine.
        self.world = None
        self.robot_id: str | None = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def kin(self):
        """The arm's kinematics, built on first use.

        LAZY on purpose. `ArmKinematics` compiles a MARS (0.15 s, 7 MB of STL)
        and raises with the fetch hint when the assets are not downloaded, and
        a brain KIND is listed in the `/sim` menu long before anybody picks it
        — so a machine with no MARS assets must be able to construct this
        object, list it and refuse at the pick rather than at import.
        """
        if self._kin is None:
            from ..robots.mars_ik import ArmKinematics
            self._kin = ArmKinematics(seed=self.seed)
        return self._kin

    def reset(self) -> None:
        self.state = "search"
        self.t_state = 0.0
        self.est: tuple[float, float] | None = None   # odom-frame target
        self.goal_kind: str | None = None             # "toy" | "basket"
        self.target_name: str | None = None
        self.t_seen = -9.0
        self.search_turned = 0.0
        self.scans_empty = 0
        self.retries: dict[str, int] = {}
        self.given_up: set[str] = set()
        self.picked = 0
        self.delivered = 0
        self.dropped = 0            # the claw let go somewhere that is not the basket
        #: Where the toy was lost, by STATE — the diagnosis, per run.
        self.lost_in: dict[str, int] = {}
        self.memory: dict[str, tuple[float, float, float]] = {}
        self.basket_mem: tuple[float, float] | None = None
        self.basket_confirmed = False
        self.held_name: str | None = None
        self.last = (0.0, 0.0, 0.0)
        self._arm: dict[str, float] | None = None     # the arm GOAL this phase wants
        self._cmd_arm: dict[str, float] | None = None  # ...and the slewed target actually sent
        self._t_prev: float | None = None
        self._q_open = None
        self._q_hover = None
        self._q_lift = None
        self._target_base = None
        self._sag = np.zeros(3)
        self._fixes: list[tuple[float, float]] = []   # standing looks, in `settle`
        self._fix_t = -9.0
        self._prev_yaw: float | None = None
        self._senses: Senses | None = None
        self._moving_since = -9.0
        self._blocked_t = -9.0
        self._was_driving = False
        self._ik_mm: float | None = None
        self._ik_fail = 0
        self._note = ""

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.est is None else {
            "bearing": 0.0, "range": None,
            "since": round(self._senses.t - self.t_seen, 2),
            "goal": [round(v, 3) for v in self.est],
            "kind": self.goal_kind, "name": self.target_name}
        out["tidy"] = {"picked": self.picked, "delivered": self.delivered,
                       "dropped": self.dropped, "lostIn": dict(self.lost_in),
                       "givenUp": sorted(self.given_up),
                       "retries": dict(self.retries),
                       "ikMm": None if self._ik_mm is None else round(self._ik_mm, 1),
                       "ikFail": self._ik_fail,
                       "holding": self.held_name}
        return out

    # ---------------------------------------------------------------- helpers
    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self.t_state = t

    def _pose(self, senses: Senses) -> tuple[float, float, float]:
        """The (x, y, yaw) this brain steers by.

        Raw odometry, and NO loop closure. The duck's `Tidy` folds each ToF
        frame into its own occupancy grid and steers by the corrected pose
        (`brain/mapping.py`), which it needs because its odometry is
        contact-anchored dead reckoning off a walking gait. A wheeled base's
        is wheel odometry, the room is 3.5 x 3.0 m, and the trip lengths here
        are ~1 m — so loop closure is a thing to ADD with a measurement
        (a `datasheet` / `hostile` odom preset A/B), not to inherit untested.
        """
        return senses.odom or (0.0, 0.0, 0.0)

    def _cam(self, senses: Senses) -> tuple[float, float, float, float, float]:
        """(x, y, z, pitch, yaw) of the lens in the BASE frame at capture.

        Height, pitch and yaw come off the detection frame — "the frame says
        where the camera was", the duck's rule, and the one thing that makes
        an elevation-to-range map honest while the head moves. x and y are the
        model's, read once from the private kinematics if one has been built
        and from the measured constants otherwise.
        """
        p = self.p
        fr = senses.fresh_det(self.DET_MAX_AGE)
        z = fr.cam_z if (fr is not None and fr.cam_z > 0.0) else p.cam_z_fallback
        pitch = fr.cam_pitch if fr is not None else p.head_pitch
        yaw = getattr(fr, "cam_yaw", 0.0) if fr is not None else 0.0
        return (p.cam_x, p.cam_y, float(z), float(pitch), float(yaw))

    def _locate(self, odom, det, target_z: float,
                senses: Senses) -> tuple[float, float, float] | None:
        """One detection -> (x, y, horizontal range from the base) in odom.

        The same idea as the duck's (`brain/tidy._locate`) — a floor object
        seen from a known camera height and pitch gives range far better than
        its apparent width does — done as a RAY rather than as a depression
        angle, and that difference is a measured bug fix rather than a tidy-up.

        **A pitched camera's azimuth is not a horizontal bearing.** The
        detector reports `bearing` and `elevation` in the CAMERA's own frame
        (`sensors/detector.py`), and with the lens pitched down by `cam_pitch`
        that azimuth is compressed toward the optical axis. MEASURED with an
        IDEAL detector (no noise at all), head at MARS's 0.349 rad, a block at
        four known base-frame spots — reported bearing against the true one:

            spot (base)        reported   true      ratio
            (0.335, -0.053)    -0.2077   -0.2413    0.861
            (0.335,  0.000)    -0.0752   -0.0878    0.857
            (0.300, -0.100)    -0.3465   -0.4073    0.851
            (0.300, +0.100)    +0.1941   +0.2307    0.841

        A gain of ~0.85, which is `AGENTS.md`'s "uncalibrated lens is a
        bearing GAIN" in a second form: not the lens model this time but the
        MOUNT's pitch. Fed through the duck's `cam_pitch - elevation`
        shortcut it put the estimate **11-22 mm to the side** of a 40 mm block
        and the claw closed on the block's edge — 1 pick in 6 over a 120 s
        run. Rebuilt as a ray (below) the same four spots land within
        **0.3-1.5 mm**.

        The duck's `_locate` has the same shape and is NOT touched: its
        `reach_pad` and `reach_left` were FITTED against this bias over 64
        paired seeds (that parameter's own block says so, and says the two
        must move together), and its targets sit near the optical axis anyway
        because it steers them onto its nose. MARS's toy is 0.21 rad off axis
        BY DESIGN — the shoulder is 53 mm to the right of the centreline — so
        the same approximation is 20 mm here and 3 mm there.

        Two smaller differences, both because this camera is not the duck's:
        the lens's own x/y OFFSET in the base frame is applied (MARS's left
        eye is 29.5 mm off the centreline, three quarters of a block), and
        there is no head-settle gate because the head never moves.
        """
        p = self.p
        cx, cy, cz, cpitch, cyaw = self._cam(senses)
        # The ray, in the camera's own frame: azimuth `bearing`, elevation up.
        ce, se = math.cos(det.elevation), math.sin(det.elevation)
        ux, uy, uz = ce * math.cos(det.bearing), ce * math.sin(det.bearing), se
        # ...un-pitched into the base's level frame (rotation about +y by
        # `cam_pitch`, positive = the lens looks DOWN). VERIFIED against the
        # true direction to a known target: y is untouched and x/z match to
        # four decimals.
        cp, sp = math.cos(cpitch), math.sin(cpitch)
        rx, rz = ux * cp + uz * sp, -ux * sp + uz * cp
        ry = uy
        if rz >= -1e-4:
            return None                 # the ray is level or rising: no floor hit
        # Where that ray meets the plane the target's centre sits on.
        scale = (cz - target_z) / (-rz)
        horiz = float(np.clip(math.hypot(rx, ry) * scale, 0.0, 6.0))
        # Beyond `far_range` the geometry is noise: keep the BEARING and take
        # the range as "at least this far" (the duck's lesson, measured there
        # as a duck that spun for four minutes on a hopping estimate).
        horiz = min(horiz, p.far_range)
        a = math.atan2(ry, rx) + cyaw
        xb, yb = cx + horiz * math.cos(a), cy + horiz * math.sin(a)
        x, y, yaw = odom
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * xb - s * yb, y + s * xb + c * yb,
                float(math.hypot(xb, yb)))

    def _trusted(self, det) -> bool:
        """A TRACKED detection, not a confident one — the duck's rule and the
        same honest boundary: ghosts have no id, and on the robot the id comes
        from a tracker."""
        return bool(det.name) and det.conf >= self.p.min_conf

    def _candidates(self, senses: Senses, cls: str):
        det = senses.fresh_det(self.DET_MAX_AGE)
        if det is None:
            return []
        out = [d for d in det.detections
               if d.cls == cls and self._trusted(d) and d.name not in self.given_up]
        if cls == "toy":
            # Never chase the toy in your own claw. It is 0.36 m ahead in the
            # carry pose and dead in frame, so without this the brain would
            # approach what it is holding.
            out = [d for d in out if d.name != self.held_name]
        return out

    def _nearest(self, senses: Senses, cls: str, odom=None):
        cands = self._candidates(senses, cls)
        if cls == "toy" and odom is not None:
            cands = [d for d in cands if not self._in_basket_zone(odom, d, senses)]
        return min(cands, key=lambda d: d.range_est) if cands else None

    def _in_basket_zone(self, odom, det, senses: Senses) -> bool:
        if self.basket_mem is None or not self.basket_confirmed:
            return False
        loc = self._locate(odom, det, self.p.toy_z, senses)
        return loc is not None and self._point_in_basket(loc[0], loc[1])

    def _point_in_basket(self, x: float, y: float) -> bool:
        return (self.basket_mem is not None and self.basket_confirmed
                and math.hypot(x - self.basket_mem[0], y - self.basket_mem[1])
                < self.p.basket_inside)

    def _note_basket(self, senses: Senses, odom) -> None:
        """A close look at the basket while NOT carrying still fixes where it
        is, so the first trip already knows the keep-out (the duck's
        `_note_basket`, halved: no staging, because there is no rim to trip
        on — only a keep-out to respect)."""
        det = self._nearest(senses, "basket")
        if det is None:
            return
        loc = self._locate(odom, det, self.p.basket_z, senses)
        if loc is None or loc[2] >= self.p.basket_confirm_range:
            return
        if self.basket_mem is None or not self.basket_confirmed:
            self.basket_mem = (loc[0], loc[1])
        else:
            self.basket_mem = (self.basket_mem[0] + 0.5 * (loc[0] - self.basket_mem[0]),
                               self.basket_mem[1] + 0.5 * (loc[1] - self.basket_mem[1]))
        self.basket_confirmed = True

    def _update_estimate(self, odom, det, target_z: float,
                         senses: Senses) -> float | None:
        """Fold one detection into the odom-frame estimate; returns its range.

        A far fix REPLACES (it is a direction), a close one is averaged in
        with a gain that grows as the range shrinks — the duck's rule, and the
        reason is the same steep elevation-to-range map.
        """
        p = self.p
        loc = self._locate(odom, det, target_z, senses)
        if loc is None:
            return None
        tx, ty, rng = loc
        if self.est is None or rng >= p.far_range:
            self.est = (tx, ty)
        else:
            k = 0.7 if rng < 0.6 else 0.3
            self.est = (self.est[0] + k * (tx - self.est[0]),
                        self.est[1] + k * (ty - self.est[1]))
        if det.cls == "toy" and det.name:
            self.memory[det.name] = (self.est[0], self.est[1], senses.t)
        elif det.cls == "basket":
            self.basket_mem = self.est
            if rng < p.basket_confirm_range:
                self.basket_confirmed = True
        return rng

    # -- driving -------------------------------------------------------------
    def _turn(self, sign: float) -> tuple[float, float, float]:
        """Turn in place. No forward kick: a wheeled base turns from rest."""
        return (0.0, 0.0, math.copysign(self.p.turn_wz, sign))

    def _servo(self, odom, stop_at: float, ahead: float = 0.0,
               left: float = 0.0, cruise: float | None = None,
               wz_cap: float | None = None
               ) -> tuple[tuple[float, float, float], float, float]:
        """Drive so the estimate ends up `ahead`/`left` of the base origin.

        `ahead`/`left` are the STANDOFF in the base frame — the pick spot for
        a toy, `basket_reach` for the basket — so the aim is offset by the
        same geometry the arm will use, and the bearing this returns is the
        error against that offset rather than against the nose.
        """
        p = self.p
        cruise = p.approach_speed if cruise is None else cruise
        wz_cap = p.turn_wz if wz_cap is None else wz_cap
        x, y, yaw = odom
        dx, dy = self.est[0] - x, self.est[1] - y
        dist = math.hypot(dx, dy)
        want = math.atan2(left, max(ahead, 0.05))       # where the target should sit
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw - want),
                             math.cos(math.atan2(dy, dx) - yaw - want))
        err = dist - stop_at
        if abs(err) <= p.stop_tol:
            # AT the standoff range. Square up if the bearing is still off —
            # and this ORDER is the whole of it. The first version returned a
            # zero twist here and left the caller's stop rule wanting both
            # tolerances at once, so a robot that arrived 0.2 rad off stood
            # still forever and every approach ended on its 30 s timeout (0
            # toys in a 90 s run, and the sheet showed the robot parked beside
            # a block). A stop is two conditions; a STAND is only legal when
            # both hold.
            if abs(bearing) > p.align_tol:
                return ((0.0, 0.0, float(np.clip(p.k_turn * bearing, -wz_cap, wz_cap))),
                        dist, bearing)
            return (0.0, 0.0, 0.0), dist, bearing
        if err < 0.0:
            # PAST it. A duck would have to turn round and come back; this
            # base reverses at the same rate it drives (`gait.back_up` is a
            # walker's problem), so overshooting the standoff costs a second
            # rather than a trip.
            return (-p.creep_speed, 0.0, 0.0), dist, bearing
        if abs(bearing) > 0.35:
            return (0.0, 0.0, math.copysign(wz_cap, bearing)), dist, bearing
        speed = p.creep_speed if err < p.creep_at else cruise
        return ((speed, 0.0, float(np.clip(p.k_turn * bearing, -wz_cap, wz_cap))),
                dist, bearing)

    def _tof_view(self, senses: Senses):
        """(depth_mm, valid) for obstacle logic, with the robot's own arm cut.

        The duck's `_tof_view`, for the duck's reason and a measured MARS one:
        while carrying, returns closer than `hold_blind_m` are the extended
        arm and the toy in the claw (0.12-0.20 m, measured), not the room.
        """
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        if tof is None:
            return None, None
        if senses.holding:
            near = tof.depth_mm < int(self.p.hold_blind_m * 1000)
            return tof.depth_mm, (tof.valid & ~near) if tof.valid is not None else ~near
        return tof.depth_mm, tof.valid

    def _clearance(self, senses: Senses) -> float:
        """Metres to the nearest thing in the ToF's centre columns.

        The 8x8 frame a wheeled body gets is `sensors.lidar.tof_from_lidar`'s
        adaptation of the planar scan, so this is the same reduction
        `wander_from_tof` performs and the same two centre columns the duck's
        guard reads. **What it cannot see**: the adapter keeps only the front
        +-22.5 deg, so an obstacle beside the robot is thrown away — Phase 3b
        traced `wander`'s two wall grazes to the arm's ELBOW against a wall it
        was driving parallel to, with metres clear ahead. `unstick` is what
        catches that here; the full 360 scan rides on `Senses.lidar` for a
        brain that wants to do better.
        """
        depth, valid = self._tof_view(senses)
        if depth is None:
            return math.inf
        cols = _column_clearance(depth, valid, WanderParams())
        return float(cols[3:5].min())

    def _keepout(self, twist, odom, note: str) -> tuple[tuple[float, float, float], str]:
        """Turn away from a confirmed basket while not delivering.

        The rim is 0.06 m and the scan plane is 0.17 m, so the lidar has
        nothing to say about it — Phase 3b measured `wander` parked on that rim
        for 53 of 60 s. A keep-out on the REMEMBERED basket is the cheapest
        instrument that sees what the scanner cannot.
        """
        p = self.p
        # ...and ONLY while exploring or scanning, which is where the duck
        # applies its own (`brain/tidy.py`'s rule reads `state in ("explore",
        # "carry_explore")`). MEASURED why it must not apply to an APPROACH:
        # in `mars-playroom` seed 1 the brain sat in `approach 0.56 m` for
        # 80 s beside the basket — the servo drove at a toy on the rim and the
        # keep-out turned it away, forever. A radial approach from outside is
        # geometrically safe (the standoff leaves the chassis front 0.165 m
        # short of the toy), so an approach is allowed through and `unstick`
        # is what catches a mistake.
        if self.state not in ("search", "explore", "carry_explore"):
            return twist, note
        if twist[0] <= 0.0 or self.basket_mem is None or not self.basket_confirmed:
            return twist, note
        bdx, bdy = self.basket_mem[0] - odom[0], self.basket_mem[1] - odom[1]
        bdist = math.hypot(bdx, bdy)
        bb = math.atan2(math.sin(math.atan2(bdy, bdx) - odom[2]),
                        math.cos(math.atan2(bdy, bdx) - odom[2]))
        if bdist < p.basket_keepout and abs(bb) < 1.2:
            return self._turn(+1.0), note + " · basket keep-out"
        return twist, note

    # -- the arm -------------------------------------------------------------
    def _standoff(self) -> tuple[float, float]:
        """(ahead, left) of the pick standoff in the base frame.

        Derived from the ARM (the shell point at `spot_radius` / `spot_yaw` on
        the toy's plane) so that "where the robot stops" and "where the arm
        reaches" are one number. Falls back to the measured pair before any
        kinematics has been built, so a brain that never picks costs no model.
        """
        p = self.p
        if self._kin is None:
            return (0.3353, -0.0528)
        pt = self.kin().shell_point(p.spot_radius, p.spot_yaw, p.toy_z)
        return (float(pt[0]), float(pt[1]))

    def _toy_target_base(self, odom, senses: Senses) -> np.ndarray | None:
        """Where to put the claw, in the BASE frame.

        From the DETECTOR by default: the odom-frame estimate `_update_estimate`
        has been folding, rotated back into the base frame. With `truth=True`
        it is the world's own toy pose — the control arm.
        """
        p = self.p
        z = p.toy_z
        if self.truth and self.world is not None and self.target_name:
            w = self.world
            bid = w.pickables.get(self.target_name)
            robot = w.ducks.get(self.robot_id or "") or next(iter(w.ducks.values()))
            if bid is None:
                return None
            pw = np.array(w.data.xpos[bid])
            bx, by, byaw = robot.driver.pose(w.data)
            c, s = math.cos(byaw), math.sin(byaw)
            dx, dy = pw[0] - bx, pw[1] - by
            return np.array([c * dx + s * dy, -s * dx + c * dy, float(pw[2])])
        if self.est is None:
            return None
        x, y, yaw = odom
        dx, dy = self.est[0] - x, self.est[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy, z])

    def _solve(self, target_base, q6: float, start=None, restarts: int | None = None):
        """One IK solve, with the residual recorded for the inspector."""
        p = self.p
        err, q = self.kin().solve_grasp(
            target_base, q6, start=start,
            restarts=p.ik_restarts if restarts is None else restarts)
        self._ik_mm = None if q is None else err * 1000.0
        if q is None or err > p.ik_tol:
            self._ik_fail += 1
            return None
        return q

    def _hold_arm(self, q, jaw: float) -> None:
        """Set the arm GOAL (`_arm`); `_slew` is what actually gets commanded."""
        from ..robots import mars
        self._arm = {j: float(v) for j, v in zip(mars.ARM_JOINTS, np.asarray(q, float))}
        self._arm["joint6"] = float(jaw)

    def _home_arm(self) -> None:
        from ..robots import mars
        self._arm = {j: float(mars.ARM_HOME[j]) for j in mars.ARM_JOINTS}

    def _slew(self, dt: float) -> dict[str, float] | None:
        """Move the COMMANDED target toward the goal at the servo's own rate.

        `MarsDriver.set_arm` takes an absolute target and applies no limit, so
        this is where the servo's speed lives on the brain's side
        (`robots/mars.MAX_TARGET_RATE_RAD_S` has the measurement and what
        omitting it did). Without it an `Intent.arm` is a 3 rad teleport every
        20 ms and the arm goes through the chassis — or, with a block in the
        claw, throws it across the room.
        """
        if self._arm is None:
            return None
        step = float(self.p.slew_rad_s) * float(dt)
        if self._cmd_arm is None:
            from ..robots import mars
            self._cmd_arm = {j: float(mars.ARM_HOME[j]) for j in mars.ARM_JOINTS}
            self._cmd_arm["joint6"] = float(mars.ARM_HOME["joint6"])
        for j, want in self._arm.items():
            now = self._cmd_arm.get(j, want)
            self._cmd_arm[j] = now + float(np.clip(want - now, -step, step))
        return dict(self._cmd_arm)

    def _measure_sag(self, senses: Senses, q_cmd) -> np.ndarray:
        """Where the claw REALLY is, minus where the solve put it (m, base).

        `Senses.arm` is the achieved joint vector — the robot's own encoders,
        `/mars/arm/state` — and forward kinematics on the private model turns
        it into a claw position. The difference from the commanded pose's claw
        is Innate's structural compliance and backlash, which `mars.arm_servo`
        models and which nothing in an IK solution knows about. Subtracting it
        from the next target is feed-forward compensation from a MEASUREMENT,
        which is the only kind available: the stiffness is a URDF-era constant
        on placeholder inertias, so computing the sag would be computing it
        from numbers Innate labels provisional.

        Zero when the body reports no arm, and clamped at `sag_max_m`: a
        correction that big is not compliance (the arm is on the floor, or a
        blade is already on the toy) and using it would aim the claw
        somewhere arbitrary.
        """
        from ..robots import mars
        if not senses.arm:
            return np.zeros(3)
        k = self.kin()
        k.place(q_cmd)
        want = k.grasp_point().copy()
        q_act = np.array([float(senses.arm.get(j, c))
                          for j, c in zip(mars.ARM_JOINTS, np.asarray(q_cmd, float))])
        k.place(q_act)
        sag = k.grasp_point().copy() - want
        n = float(np.linalg.norm(sag))
        if n > self.p.sag_max_m:
            sag *= self.p.sag_max_m / n
        return sag

    def _give_up(self, name: str | None) -> None:
        n = self.retries.get(name or "", 0) + 1
        self.retries[name or ""] = n
        if n > self.p.max_retries:
            self.given_up.add(name or "")

    # ------------------------------------------------------------ the machine
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p, t = self.p, senses.t
        odom = self._pose(senses)
        twist = (0.0, 0.0, 0.0)
        note = self.state
        ahead, left = self._standoff()
        if self.state in ("search", "explore", "approach"):
            self._note_basket(senses, odom)
        # The claw is read, never remembered: `Senses.holding` is the
        # constraint torque at joint6 (`World.sense_grip`), so a toy that
        # slips out is gone the tick it leaves.
        if not senses.holding:
            self.held_name = None

        if self.state == "search":
            toy = self._nearest(senses, "toy", odom)
            if toy is not None and self._update_estimate(odom, toy, p.toy_z, senses) is not None:
                self.goal_kind, self.target_name = "toy", toy.name
                self.t_seen = t
                self.scans_empty = 0
                self._enter("approach", t)
            else:
                for dd in self._candidates(senses, "toy"):
                    saved = self.est
                    self.est = None
                    self._update_estimate(odom, dd, p.toy_z, senses)
                    self.est = saved
                twist = (0.0, 0.0, p.search_wz)
                if self._prev_yaw is not None:
                    d = odom[2] - self._prev_yaw
                    self.search_turned += math.atan2(math.sin(d), math.cos(d))
                if self.search_turned >= 2 * math.pi:
                    self.search_turned = 0.0
                    remembered = [(n, m) for n, m in self.memory.items()
                                  if n not in self.given_up
                                  and not self._point_in_basket(m[0], m[1])]
                    if remembered:
                        x, y, _ = odom
                        name, (mx, my, _) = min(
                            remembered, key=lambda nm: math.hypot(nm[1][0] - x, nm[1][1] - y))
                        self.est, self.goal_kind, self.target_name = (mx, my), "toy", name
                        self.t_seen = t
                        self.scans_empty = 0
                        self._enter("approach", t)
                    else:
                        self.scans_empty += 1
                        self._enter("done" if self.scans_empty >= p.done_after_scans
                                    else "explore", t)
                note = f"search {self.scans_empty}/{p.done_after_scans}"

        elif self.state == "explore":
            if self._nearest(senses, "toy", odom) is not None:
                self._enter("search", t)
            else:
                depth, valid = self._tof_view(senses)
                twist = ((0.0, 0.0, 0.0) if depth is None
                         else wander_from_tof(depth, valid,
                                              WanderParams(stop_at=p.guard_m,
                                                           cruise=p.approach_speed)))
                if t - self.t_state > p.explore_s:
                    self._enter("search", t)

        elif self.state == "approach":
            toy = self._nearest(senses, "toy", odom)
            if toy is not None and toy.name == self.target_name:
                if self._update_estimate(odom, toy, p.toy_z, senses) is not None:
                    self.t_seen = t
            twist, dist, bearing = self._servo(odom, math.hypot(ahead, left), ahead, left)
            note = f"approach {dist:.2f} m"
            if self._point_in_basket(*self.est):
                self.est = None
                self._enter("search", t)
            elif (abs(dist - math.hypot(ahead, left)) <= p.stop_tol
                    and abs(bearing) <= p.align_tol):
                self._fixes, self._fix_t = [], -9.0
                self._enter("settle", t)
            elif t - self.t_seen > 6.0 or t - self.t_state > 30.0:
                self._give_up(self.target_name)
                self.est = None
                self._enter("search", t)

        elif self.state == "settle":
            # STANDING LOOKS, averaged. One frame of the `datasheet` detector
            # puts the estimate 2-54 mm out; the mean of ~8 taken with the
            # base at rest is what the IK is solved against (`settle_fixes`).
            fr = senses.fresh_det(self.DET_MAX_AGE)
            toy = self._nearest(senses, "toy", odom)
            if (toy is not None and toy.name == self.target_name
                    and fr is not None and fr.t > self._fix_t):
                loc = self._locate(odom, toy, p.toy_z, senses)
                if loc is not None:
                    self._fix_t = fr.t
                    self._fixes.append((loc[0], loc[1]))
            if t - self.t_state >= p.settle_s or len(self._fixes) >= p.settle_fixes:
                if self._fixes:
                    xs = self._fixes[-p.settle_fixes:]
                    self.est = (float(np.mean([f[0] for f in xs])),
                                float(np.mean([f[1] for f in xs])))
                    self.t_seen = t
                self._fixes = []
                self._fix_t = -9.0
                tb = self._toy_target_base(odom, senses)
                # HOVER first: the jaws open `hover_m` above the toy. The
                # descent is then vertical, so the blades never sweep the toy
                # sideways, and this pose is where the arm's compliance is
                # measured (see `hover`).
                self._target_base = tb
                q = None if tb is None else self._solve(
                    tb + np.array([0.0, 0.0, p.hover_m]), OPEN_RAD)
                if q is None:
                    self._give_up(self.target_name)
                    self.est = None
                    self._enter("search", t)
                else:
                    self._q_hover = q
                    self._hold_arm(q, OPEN_RAD)
                    self._enter("hover", t)

        elif self.state == "hover":
            self._hold_arm(self._q_hover, OPEN_RAD)
            if t - self.t_state >= p.hover_s:
                # SAG, measured rather than modelled. `Senses.arm` is the
                # ACHIEVED joint vector (Innate's `/mars/arm/state`); forward
                # kinematics on the private model turns it into where the
                # claw really is, and the difference from where the solve put
                # it is the compliance error. MEASURED at this pose: joint2
                # -0.067 rad, joint3 -0.048 rad, which is 28.5 mm of claw
                # height and 11.5 mm of reach — three times the 9.4 mm margin
                # a 58.8 mm jaw has on a 40 mm block, and it is why the first
                # version of this brain grasped 1 pick in 6 with a 1-8 mm
                # ESTIMATE error: the estimate was never the problem.
                self._sag = self._measure_sag(senses, self._q_hover)
                tb = self._target_base
                q = None if tb is None else self._solve(
                    tb - self._sag, OPEN_RAD, start=self._q_hover, restarts=6)
                if q is None and tb is not None:
                    # The corrected target is a few millimetres off the one
                    # that solved at hover height, and a descent from a local
                    # start can miss its basin — so fall back to the
                    # UNCORRECTED target with fresh restarts rather than
                    # giving the toy up. MEASURED: the correction is 0.5-3 mm
                    # (the sag table on `_measure_sag`), so this loses almost
                    # nothing, while giving up here cost three toys in a 120 s
                    # run before the fallback existed.
                    q = self._solve(tb, OPEN_RAD)
                    self._sag = np.zeros(3)
                if q is None:
                    self._give_up(self.target_name)
                    self.est = None
                    self._home_arm()
                    self._enter("search", t)
                else:
                    self._q_open = q
                    self._hold_arm(q, OPEN_RAD)
                    self._enter("reach", t)

        elif self.state == "reach":
            self._hold_arm(self._q_open, OPEN_RAD)
            if t - self.t_state >= p.reach_s:
                self._enter("close", t)

        elif self.state == "close":
            self._hold_arm(self._q_open, CLOSE_RAD)
            if t - self.t_state >= p.close_s:
                # The lift target is the grasp point RAISED, so the wrist keeps
                # the orientation it closed in. MEASURED: lifting the tool
                # frame instead ends in a different wrist pose and the transit
                # to a fixed carry pose then lost the block 4 times in 5, while
                # the raised grasp point holds 6/6.
                self.kin().place(self._q_open)
                grasp = self.kin().grasp_point().copy() + self._sag
                q = self._solve(grasp + np.array([0.0, 0.0, p.lift_m]) - self._sag,
                                CLOSE_RAD, start=self._q_open, restarts=3)
                self._q_lift = q if q is not None else self._q_open
                self._enter("lift", t)

        elif self.state == "lift":
            self._hold_arm(self._q_lift, CLOSE_RAD)
            if t - self.t_state >= p.lift_s:
                if senses.holding:
                    self.picked += 1
                    self.held_name = self.target_name
                    self.memory.pop(self.target_name or "", None)
                    self.est = None
                    self.goal_kind = None
                    self._enter("carry", t)
                else:
                    self._give_up(self.target_name)
                    self.est = None
                    self._home_arm()
                    self._enter("search", t)

        elif self.state in ("carry", "carry_explore", "deliver", "place") and not senses.holding:
            # It left the claw. WHERE decides what it means, and the counters
            # are split by state because that is the whole diagnosis: a loss
            # in `carry`/`deliver` is a grip that let go on the move, and one
            # in `place` is the arm extending over the rim — which usually
            # still lands the toy IN the tray (the arm is already over it), so
            # it counts as a delivery rather than a drop. The geometric score
            # (`World.in_basket`) is the truth either way; this is what the
            # inspector and the sheet read.
            self.lost_in[self.state] = self.lost_in.get(self.state, 0) + 1
            if self.state == "place":
                self.delivered += 1
            else:
                self.dropped += 1
            self.est, self.goal_kind = None, None
            self._home_arm()
            self._enter("retract" if self.state == "place" else "search", t)

        elif self.state == "carry":
            self._hold_arm(self._q_lift, CLOSE_RAD)
            basket = self._nearest(senses, "basket")
            if self.basket_mem is not None and self.basket_confirmed:
                self.est, self.goal_kind, self.target_name = self.basket_mem, "basket", "basket"
                self.t_seen = t
                self._enter("deliver", t)
            elif basket is not None and self._update_estimate(
                    odom, basket, p.basket_z, senses) is not None:
                self.goal_kind, self.target_name = "basket", "basket"
                self.t_seen = t
                self._enter("deliver", t)
            else:
                twist = (0.0, 0.0, p.carry_wz)
                if t - self.t_state > 12.0:
                    self._enter("carry_explore", t)

        elif self.state == "carry_explore":
            # Holding a toy and the basket has never been seen from anywhere
            # in this turn: the room is 3.5 x 3.0 m and the lens is 116 deg
            # wide, so this is rare — but a MARS parked in a corner facing a
            # wall can turn a full circle and see nothing. Drive somewhere
            # else and turn again (the duck's `carry_explore`, same job).
            self._hold_arm(self._q_lift, CLOSE_RAD)
            if self._nearest(senses, "basket") is not None:
                self._enter("carry", t)
            else:
                depth, valid = self._tof_view(senses)
                twist = ((0.0, 0.0, 0.0) if depth is None
                         else wander_from_tof(depth, valid,
                                              WanderParams(stop_at=p.guard_m,
                                                           cruise=p.carry_speed,
                                                           turn=p.carry_wz,
                                                           spin=p.carry_wz)))
                if t - self.t_state > p.explore_s:
                    self._enter("carry", t)

        elif self.state == "deliver":
            self._hold_arm(self._q_lift, CLOSE_RAD)
            basket = self._nearest(senses, "basket")
            if basket is not None and self._update_estimate(
                    odom, basket, p.basket_z, senses) is not None:
                self.t_seen = t
            twist, dist, bearing = self._servo(odom, p.basket_reach, p.basket_reach,
                                               cruise=p.carry_speed, wz_cap=p.carry_wz)
            note = f"deliver {dist:.2f} m"
            if abs(dist - p.basket_reach) <= p.stop_tol and abs(bearing) <= p.align_tol:
                tgt = np.array([p.basket_reach - p.drop_inset, 0.0,
                                self.rim_z() + p.drop_clear]) - self._sag
                q = self._solve(tgt, CLOSE_RAD)
                if q is None:
                    self._enter("retract", t)
                else:
                    self._hold_arm(q, CLOSE_RAD)
                    self._enter("place", t)
            elif t - self.t_state > 40.0:
                self.basket_mem, self.basket_confirmed = None, False
                self._enter("carry", t)

        elif self.state == "place":
            if t - self.t_state >= p.place_s:
                self._enter("drop", t)

        elif self.state == "drop":
            if self._arm is not None:
                self._arm = {**self._arm, "joint6": OPEN_RAD}
            if t - self.t_state >= p.drop_s and not senses.holding:
                self.delivered += 1
                self.held_name = None
                self.search_turned = 0.0
                self._enter("retract", t)
            elif t - self.t_state >= 3.0:
                self._enter("retract", t)          # the jaw will not let go: leave

        elif self.state == "retract":
            self._home_arm()
            if t - self.t_state >= p.retract_s:
                self.est, self.goal_kind = None, None
                self.search_turned = 0.0
                self._enter("search", t)

        elif self.state == "unstick":
            twist = (p.unstick_speed, 0.0, p.unstick_wz)
            if t - self.t_state >= p.unstick_s:
                self.est, self.goal_kind = None, None
                self.search_turned = 0.0
                # A toy still in the claw is a trip in progress: back to the
                # basket, not back to searching. Losing the delivery because
                # the base bumped something would count the work twice.
                self._enter("carry" if senses.holding else "search", t)

        elif self.state == "done":
            twist = (0.0, 0.0, 0.0)

        # -- guards, in the order they must apply ----------------------------
        twist, note = self._keepout(twist, odom, note)
        driving = twist[0] > 0.0
        near = (self.est is not None
                and math.hypot(self.est[0] - odom[0], self.est[1] - odom[1]) < p.guard_off_m)
        clear = self._clearance(senses) if driving and not near else math.inf
        if driving and not near and t - self._blocked_t < p.detour_s and clear >= p.guard_m:
            # Just cleared a blocker: one straight leg past it WHATEVER the
            # servo wants (`detour_s` — the alternative is measured, 45 s of
            # deliver that never arrives).
            twist = (p.carry_speed if senses.holding else p.approach_speed, 0.0, 0.0)
            note += " · detour"
        elif driving and not near and clear < p.guard_m:
            # Gentler while HOLDING: a full-rate turn in place is what the
            # carried block is slung off by (`carry_wz`).
            twist = (0.0, 0.0, p.carry_wz if senses.holding else p.turn_wz)
            self._blocked_t = t
            note += " · blocked"
            driving = False
        # Asked to move and not moving: the rim, a wall the 45-degree adapter
        # threw away, or a toy under the chassis. Measured on the quantity the
        # drive PD regulates (`Senses.speed`), which is the only one that
        # cannot be fooled by a command that never took effect.
        #
        # The clock resets whenever the robot is NOT being asked to drive, and
        # that is not cosmetic: the first version seeded it at -9.0 and only
        # stamped it while moving, so the very first tick of the very first
        # approach was already 9 s "stuck" and the run was an unstick loop
        # (measured: 0 toys, `unstick` at t = 1.0 s).
        if (not driving or not self._was_driving
                or abs(float(senses.speed or 0.0)) >= p.stuck_speed):
            self._moving_since = t
        self._was_driving = driving
        if driving and t - self._moving_since > p.stuck_s and self.state != "unstick":
            if not senses.holding:
                self._home_arm()
            self._enter("unstick", t)
            twist = (p.unstick_speed, 0.0, p.unstick_wz)
            note = "unstick"

        self._prev_yaw = odom[2]
        self.last = twist
        self._note = note
        # The tick's own length, measured off the sense clock rather than
        # assumed: a tethered brain (`brain/tether.py`) and `eval-tidy` step
        # this at 50 Hz, and a test steps it at whatever it likes.
        dt = 0.02 if self._t_prev is None else max(1e-4, min(0.2, t - self._t_prev))
        self._t_prev = t
        return Intent(twist=twist, head=(0.0, p.head_pitch, 0.0, 0.0),
                      note=note, arm=self._slew(dt))

    # ------------------------------------------------------------------ world
    def rim_z(self) -> float:
        """The basket rim's height (m).

        From the scenario when the harness has handed this brain the world
        (`world` is set by `world_server.make_brain` / `eval-tidy`), and the
        `Basket` default otherwise. The place target is the rim plus
        `drop_clear`, so a deeper tray is delivered into rather than dropped
        beside.
        """
        w = self.world
        if w is not None and getattr(w, "basket", None) is not None:
            return float(w.basket.rim)
        from ..world.scenario import Basket
        return float(Basket.rim)


REGISTRY.register("tidy_arm", TidyArm)
__all__ = ["CLOSE_RAD", "OPEN_RAD", "TidyArm", "TidyArmParams"]
