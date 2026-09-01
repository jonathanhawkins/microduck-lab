from .headstand import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------------------------ backflip
# A backward roll-over: lean back, roll across the back with the neck tucked,
# carry the legs over the top, land on the feet, STAND. XL330s can't launch
# 800 g airborne, so "backflip" here means the floor-contact somersault — the
# same trick family as the headstand's front flip, plus the genuinely new
# part: the maneuver is a TRAJECTORY (0 -> 360° of rotation), not a pose, so
# the reward needs memory (Behavior.state_fn).

_BF_FULL = 2.0 * np.pi
_BF_DONE = 5.2  # rad (~298°) — "flip complete": the roll has crossed the
                # crown and the feet are coming under; the landing terms
                # grade the rest of the rise.


def _bf_update(env) -> None:
    """Integrate cumulative BACKWARD pitch rotation (verified sign: gyro_y is
    negative while the trunk pitches backward). Clipped to [0, one full flip]:
    forward wobble can't bank negative credit, and a second flip earns nothing
    — so after landing, holding the stand strictly beats re-flipping (the
    headstand's cycling lesson, priced in from day one)."""
    w = env._gyro
    # C.CTRL_DT is the same product of the same doubles: __init__ pins
    # model.opt.timestep = C.PHYSICS_DT and nothing retunes it afterwards
    # (reading it back through the bindings cost ~1 us per step).
    env._bf_rot = min(max(env._bf_rot - C.CTRL_DT * float(w[1]), 0.0), _BF_FULL)


def _bf_progress(env) -> float:
    """Rotation progress, CUBIC: shallow early so lying mid-roll earns pocket
    change (no well-paid halfway point — the headstand's comfy-bow lesson),
    but with enough interior slope that the 70°→180° bridge is worth crossing.
    The first cut was quartic and the bridge paid ~nothing."""
    return float((env._bf_rot / _BF_FULL) ** 3)


def _bf_no_jaw_parking_pen(env) -> float:
    """Penalty for resting the jaw on the floor before the flip (<= 0).

    gentle_head charges head-contact IMPACT, not occupancy — so when the
    attempt-taxes briefly made flipping unprofitable, the policy discovered a
    rent-free rest: lean forward, place the jaw softly, park for the whole
    episode (measured: 10 s at pitch -64, rot +3). Occupancy is now priced,
    but ONLY pre-flip (rot < 0.3) — mid-roll head contact is the neck-kip,
    the mechanism the whole trick runs on. Bounded at one unit.
    """
    rot = getattr(env, "_bf_rot", 0.0)
    if rot >= 0.3:
        return 0.0
    z = float(env.data.xpos[_jaw_bid(env)][2])
    if z > 0.12:                 # jaw well clear of the floor: free
        return 0.0
    return -min((0.12 - z) / 0.06, 1.0)


def _bf_brake_pen(env) -> float:
    """Penalty for STILL rolling backward once the flip is complete (<= 0).

    _bf_rot caps at 2*pi, so over-rolling banks no more progress — but nothing
    charged it either, and unpriced quantities are free resources: the policy
    happily carried extra momentum through the landing, which is exactly the
    user-observed cluster — rolling a second time, or sitting down hard on
    its butt because the rotation never got braked before touchdown.
    Observable: attitude (gravity says upright-ish again) plus the duck's own
    gyro wy. Gated to the completed-rotation regime; bounded at one unit
    (saturates at ~4.5 rad/s of residual back-spin).
    """
    if env._bf_rot < 0.95 * _BF_FULL:
        return 0.0
    wy = float(env._gyro[1])
    if wy >= 0.0:          # not back-rolling: nothing to brake
        return 0.0
    return -min(0.05 * wy * wy, 1.0)


def _bf_lean_back(env) -> float:
    """Discovery slope for the first wiggle: pays for leaning back — but ONLY
    while still ROTATING backward. The first cut paid for the lean itself,
    and the very first policy parked at a 70° recline collecting it forever
    (rent-free basin right below the gate). Motion-gating makes a parked lean
    worth zero at any angle; only carrying the roll onward earns."""
    if env._bf_rot >= 1.4:
        return 0.0
    wy = float(env._gyro[1])
    if wy > -0.3:
        return 0.0  # not rotating backward right now -> no pay
    return max(0.0, min(1.0, -float(env._projected_gravity()[0])))


def _bf_landed_hold(env) -> float:
    """THE money: flip complete, feet planted, nothing else touching — and
    GRADED by uprightness rather than gated on it: a landed CROUCH earns 25%,
    rising to full pay standing tall. The all-or-nothing version left the
    roll's actual arrival state (a deep forward crouch at ~260°) worth zero,
    so the stand-up — itself a real sub-skill — had no slope to climb; v5
    coasted to 260° and stopped, 0/20 even from its own spawns."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]) or _body_floor_contacts(env) > 0:
        return 0.0
    # Two-layer uprightness (the zero-gradient law, fourth appearance): the
    # tight-only layer pays ~0 anywhere shy of vertical, so rising from the
    # arrival crouch (~60° forward) to 30° earned nothing more than the flat
    # base — stages completed the roll 10/12 and stood up 0/12. The wide
    # layer slopes the whole way up from the crouch.
    g = env._projected_gravity()
    d2 = float(g[0] ** 2 + g[1] ** 2)
    upright = (0.25 + 0.45 * float(np.exp(-d2 / 0.5))
               + 0.30 * float(np.exp(-d2 / 0.05)))
    # ...and HEIGHT, because orientation is not elevation (the headstand
    # learned this the hard way with gz=1.0 while the trunk lay on the floor;
    # here a folded crouch — trunk 5.5 cm vs 12 cm standing, head 9 cm vs
    # 24 cm, neck tucked — collected the full landed payout and looked like
    # "20/20 stands" to a metric that only asked which way was up. The user
    # watching said it never got up, twice, and was right both times).
    # Graded, not gated: a landed crouch still earns something (catching the
    # landing must stay worth doing) but far less than standing. At 0.3+0.7
    # the crouch was worth ~half a stand, and two-thirds of a stage's episodes
    # ending in one TRAINED AWAY an inherited stand — 7.95 s of hold became
    # 0.14 s in a single stage. A 6:1 ratio keeps the landing worth catching
    # without letting the crouch out-vote the stand.
    dz2 = (float(env._trunk_xpos[2]) - env.stand_z) ** 2
    tall = (0.5 * float(np.exp(-dz2 / 0.08 ** 2))
            + 0.5 * float(np.exp(-dz2 / 0.03 ** 2)))
    return upright * (0.15 + 0.85 * tall)


def _bf_neck_targets(rot: float) -> tuple[float, float]:
    """Phase-dependent neck technique, refined by a probe matrix (all
    variants measured to their max rotation): FULL tuck (-1.5, pulled tight
    against the body — the user's 'roll on the curved part') rolls to ~183°
    where the headstand-style -0.6 tuck shelfed at 161°; the ARCH plants the
    crown and carries 160°→283°; between them sits the KIP — a fast
    tuck→arch snap pressing the head into the floor. This schedule is the
    measured best-known sequence; the terms below pay its pieces so the
    panel's live bars show which parts the policy has adopted."""
    if rot < 2.3:
        return (-1.5, -1.5)
    if rot < 2.8:
        t = (rot - 2.3) / 0.5
        return (-1.5 + 2.5 * t, -1.5 + 2.8 * t)
    if rot < 4.6:
        return (1.0, 1.3)
    t = min(1.0, (rot - 4.6) / 0.5)  # ease home for the landing
    return (1.0 - 0.65 * t, 1.3 - 0.95 * t)


def _bf_push_off(env) -> float:
    """The committed entry (rot < ~50°): pays the backward rotation RATE the
    legs manage to generate — a real push-off arrives on the back at 5-6
    rad/s, a polite sit arrives at ~1 and stalls (probe-measured). Rate pay,
    not a pose: how the legs make the speed is the policy's business. The
    user called this gap watching the trainee: 'I'm not seeing it really
    give an initial push off with the legs.'"""
    if env._bf_rot >= 0.9:
        return 0.0
    wy = float(env._gyro[1])
    rate = max(0.0, -wy)
    # BAND, not monotone. The old min(1, rate/6) paid full marks for ANY spin
    # >= 6 rad/s and never charged excess — over-rotation was manufactured
    # right here at launch, then taxed three stages later at the landing
    # (user: "isn't it the sum of all the parts... how much momentum it's
    # giving to the flip?" — yes). A clean entry probe-measures ~5 rad/s, so
    # the bell targets that; the small monotone floor keeps discovery slope
    # alive from zero (the zero-gradient trap otherwise).
    bell = float(np.exp(-((rate - 5.0) ** 2) / (2.0 ** 2)))
    return 0.6 * bell + 0.4 * min(1.0, rate / 5.0)


def _bf_rolling(env, thresh: float = -0.2) -> bool:
    """Still rotating backward — the gate on every mid-roll SHAPE term.
    Ungated shape pay re-creates the v1 basin: lying on the back holding a
    pretty tuck out-earned attempting the flip (the user spotted the gentle
    sit-and-park entries this bred)."""
    return float(env._gyro[1]) < thresh


def _bf_tuck_ball(env) -> float:
    """Ball up from the moment the fall starts (rot ~20-140°): legs folded,
    neck pulled FULLY against the body so the back rolls on a curved profile
    instead of slamming the head shelf down. Starts at 20° — the fall itself
    must be tucked (the neck used to be unpriced until 50°, and the duck
    fell with it extended) — but pays only WHILE rolling backward."""
    rot = env._bf_rot
    if not (0.35 < rot < 2.4) or not _bf_rolling(env):
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[2] + 1.0) ** 2 + (q[11] - 1.0) ** 2
               + (q[3] + 1.0) ** 2 + (q[12] - 1.0) ** 2
               + (q[5] + 1.5) ** 2 + (q[6] + 1.5) ** 2)
    return 0.5 * float(np.exp(-d2 / 2.5 ** 2)) + 0.5 * float(np.exp(-d2 / 1.0 ** 2))


def _bf_neck_kip(env) -> float:
    """The press-off at the crux (rot ~132-172°): pays neck EXTENSION SPEED
    while the head is grounded — snapping tuck→arch presses the head into
    the floor and levers the trunk over the back-shell edge (wrestler's kip).
    A velocity term on purpose: the probe showed the static poses on either
    side both stall; the SNAP itself is the move. Bounded ≤1/step."""
    rot = env._bf_rot
    if not (2.3 < rot < 3.0):
        return 0.0
    if not _head_on_floor(env):
        return 0.0  # nothing to press against
    qv = env._joint_vel()
    v = max(0.0, float(qv[5])) + max(0.0, float(qv[6]))
    return min(1.0, v / 10.0)


def _bf_arch_over(env) -> float:
    """The crown-bridge (rot ~160-265°): neck arched so the body carries
    over the planted crown — measured to run 160°→283° on momentum alone."""
    rot = env._bf_rot
    if not (2.8 < rot < 4.6) or not _bf_rolling(env, thresh=-0.1):
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[5] - 1.0) ** 2 + (q[6] - 1.3) ** 2)
    return 0.5 * float(np.exp(-d2 / 2.0 ** 2)) + 0.5 * float(np.exp(-d2 / 0.7 ** 2))


def _bf_legs_over(env) -> float:
    """Feet overhead through the middle of the roll — the leg WHIP. Kicking
    the legs up and over while the back is on the floor is what generates the
    angular momentum that carries the body across the crown (the user's read
    of the stall, and the reaction-torque physics agree). Gated so neither
    the entry nor the landing pays for high feet."""
    if not (1.4 < env._bf_rot < 3.5) or not _bf_rolling(env):
        return 0.0
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    z = (zl + zr) / 2
    return float(np.exp(-((z - 0.30) ** 2) / 0.12 ** 2))


def _bf_straight_pen(env) -> float:
    """Keep the flip in the sagittal plane (<= 0): roll/yaw rates are the
    free resources here — an unpriced corkscrew reads as rotation progress
    while twisting the landing off-axis."""
    w = env._gyro
    # Bounded (run-recipe lesson), and PHASE-SCOPED: measured per-phase yaw
    # attribution (24 flips) put ~98 deg of wobble in the head-pivot arc and
    # ~114 in the carry, vs 3.5 at launch. The head JOINTS are already still
    # (0.16 rad max — still_head works); the slew is the body spinning on the
    # head contact like a top, steered by the legs. Off-axis rate costs 3x
    # only in the pivot/carry arc — priced where the damage happens, no
    # attempt-tax on the launch.
    rot = getattr(env, "_bf_rot", 0.0)
    k = 3.0 if 0.9 <= rot <= 5.2 else 1.0
    return -min(k * 0.1 * float(w[0] ** 2 + w[2] ** 2), 1.0)


_HEAD_YAW_I, _HEAD_ROLL_I = 7, 8   # C.JOINT_NAMES indices


def _bf_still_head_pen(env) -> float:
    """Penalty for turning/rolling the HEAD mid-flip (<= 0).

    User-observed failure mode: flips where the head yaws mid-roll dump
    angular momentum off the sagittal plane and the duck flops sideways; the
    clean flips keep the head still. straight_flip prices BODY twist rate but
    head-joint motion is nearly free — the heavy head is ~30% of total mass,
    so a head_yaw swing is a big off-axis impulse. Gated to the flip itself
    (rotation banked but not complete) so landing recovery and the stand
    aren't taxed. Head joint pos/vel are in the obs — observable, so priced.
    Bounded at one unit (~0.7 rad of combined deflection).
    """
    rot = getattr(env, "_bf_rot", 0.0)
    if not (0.05 < rot < 2.0 * np.pi - 0.3):
        return 0.0
    q = env._joint_qpos()
    dq = env._joint_vel()
    dev = float(q[_HEAD_YAW_I] ** 2 + q[_HEAD_ROLL_I] ** 2)
    vel = float(dq[_HEAD_YAW_I] ** 2 + dq[_HEAD_ROLL_I] ** 2)
    return -min(2.0 * dev + 0.02 * vel, 1.0)


def _bf_calm_landed_pen(env) -> float:
    """Stillness once landed (<= 0), the flip itself untaxed — the mirror of
    the headstand's calm_up_top, against flip-stagger-flop cycling."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    return _upright(env) * -0.06 * float((env._gyro ** 2).sum())


def _bf_feet_under(env) -> float:
    """Get-up mechanics, part 1 (landing window): feet PLANTED and pulled in
    UNDER the body — you can't rise from feet sprawled out in front. Pays per
    planted foot, scaled by how close the feet sit to under the trunk."""
    if env._bf_rot < 4.6:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] or c["right"]):
        return 0.0
    tx, ty = env._trunk_xpos[:2]
    total = 0.0
    for side in ("left", "right"):
        if not c[side]:
            continue
        fx, fy = env.data.geom_xpos[env.foot_geoms[side]][:2]
        d2 = (fx - tx) ** 2 + (fy - ty) ** 2
        total += 0.5 * float(np.exp(-d2 / 0.08 ** 2))
    return total


def _bf_neck_pushup(env) -> float:
    """Get-up mechanics, part 2 (landing window): with 38% of the mass in the
    head, pressing it into the floor to lever the torso up is this body's
    substitute for arms — the kip's physics pointed at the rise. Pays neck
    extension SPEED while the head is grounded, same shape as neck_kip."""
    if env._bf_rot < 4.6 or not _head_on_floor(env):
        return 0.0
    qv = env._joint_vel()
    v = max(0.0, float(qv[5])) + max(0.0, float(qv[6]))
    return min(1.0, v / 10.0)


def _bf_head_rise(env) -> float:
    """Get-up mechanics, part 3 — the relay's last runner: with feet planted
    in the landing window, pay the HEAD'S HEIGHT toward standing (~22 cm).
    Without this the get-up stalled exactly between the press-off and the
    standing pay: feet_under earned, neck_pushup earned, and then the head
    simply stayed down — the head-down tripod collected enough base income
    to be a comfy rest stop (the user watched it park there). Two-layer so
    the slope reaches all the way down to a head resting on the floor."""
    if env._bf_rot < 4.6:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    z = float(env.data.xpos[_jaw_bid(env)][2])
    d2 = (z - 0.22) ** 2
    return 0.5 * float(np.exp(-d2 / 0.12 ** 2)) + 0.5 * float(np.exp(-d2 / 0.05 ** 2))


def _bf_stand_pose(env) -> float:
    """Finish IN the canonical stand (the user's suggestion, with a sim2real
    bonus they intuited: DEFAULT_POSE is the shared home pose every shipped
    policy starts from, so ending the trick there means the robot can
    hot-swap straight back to its walk policy). Gated to the landed phase;
    two-layer over leg+neck joints so the pull reaches a crouch."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    q = env._joint_qpos()
    d2 = float(((q[C.LEG_JOINT_IDS] - C.DEFAULT_POSE[C.LEG_JOINT_IDS]) ** 2).sum())
    d2 += float((q[5] - C.DEFAULT_POSE[5]) ** 2 + (q[6] - C.DEFAULT_POSE[6]) ** 2)
    # Widths measured against reality, not guessed: the landed stand sits
    # ~3.9 rad from DEFAULT_POSE (a splayed survival crouch), where the first
    # cut of this term paid 0.0006 — invisible, and a 1M-step polish run
    # moved the pose the WRONG way (3.92 -> 4.37). Fifth zero-gradient bug of
    # this project; the wide layer now pays ~0.19 out there and slopes in.
    return 0.5 * float(np.exp(-d2 / 4.0 ** 2)) + 0.5 * float(np.exp(-d2 / 1.5 ** 2))


def _bf_spotter(env) -> bool:
    """Gymnastics-coach assist for SHOWCASE previews only (never training).

    The trained policy owns everything past ~170° — carry, land, stand, hold
    (19-20/20) — but the 90-170° arc is blocked by this actuator model, so
    from a standing start it rationally never begins, and a viewer watching
    the finished trick sees... a duck standing there. A gentle backward pitch
    torque through the dead zone, released the moment the policy's own
    territory begins, shows the real trick honestly: probe-measured 4/5 full
    360° flips with 8-13 s stands, and the duck's label says 'spotted'."""
    # Release at 4.6 rad (264 deg), not 4.0: after the momentum band removed
    # EXCESS spin, marginal episodes arrived at the old release with too
    # little and stalled face-down in the 230-300 deg arc (user: "it ends up
    # on its face more"). Measured sweep: release 4.0 -> 2 face-downs, 46/48
    # stands; 4.6 -> 0 face-downs, 48/48 flips AND stands, over-flip tail
    # unchanged; 5.0 -> tail creeps back. 4.6 is the plateau's edge.
    assisting = (15 < env.step_count < 250) and env._bf_rot < 4.6
    env.data.qfrc_applied[4] = -0.6 if assisting else 0.0
    # Plane stabilization, same assist window: the policy CANNOT learn to
    # land pointing straight — final yaw is unobservable (no compass in the
    # 61 obs), and the off-axis RATE terms can't price a clean-looking roll
    # about a slightly tilted axis. The coach's other hand: a small
    # yaw/roll-damping torque while spotting. Measured (16 spawns, b18a5c):
    # landing yaw mean 90 deg -> 33, stands 16/16. Stronger gains or a wider
    # window measured WORSE (53/46 deg) — fighting the policy destabilizes.
    if assisting:
        q = env.data.qpos[3:7]
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        wz = float(env.data.qvel[5])
        wx = float(env.data.qvel[3])
        env.data.qfrc_applied[5] = float(np.clip(-0.4 * yaw - 0.06 * wz, -0.35, 0.35))
        env.data.qfrc_applied[3] = float(np.clip(-0.06 * wx, -0.2, 0.2))
    else:
        env.data.qfrc_applied[5] = 0.0
        env.data.qfrc_applied[3] = 0.0
        # The coach's CATCH: flip essentially complete but still back-spinning
        # hard — counter-torque against the over-roll. Fires only on the
        # violent tail (wy < -2.5), because braking every landing measurably
        # worsened heading (torque on the body-frame pitch dof of a yawed
        # trunk leaks off-plane): measured over 48 spawns, overshoot p90
        # 174 deg -> 31, bad (>45 deg) episodes 9 -> 4, landing yaw and
        # stand rate unchanged (48/48). A world-frame projection of the same
        # brake measured WORSE (frame conventions bite; see the velocity-trap
        # memory) — the body-frame dof with a tail-only gate is what works.
        wy = float(env._gyro[1])
        if env._bf_rot >= 0.95 * _BF_FULL and wy < -2.5:
            env.data.qfrc_applied[4] = float(min(0.10 * (-wy), 0.5))
    return assisting


def _bf_land_tall(env) -> float:
    """Extend the legs on the way DOWN (the last quarter-turn before the roll
    completes) so touchdown happens on near-straight legs.

    Measured cliff: the stand policy rescues a landing whose trunk is at
    12.8 cm and fails at 11.7 — and nothing, including the shipped walking
    policy, rises from the 5.5 cm crouch this trick used to land in. These
    servos can HOLD a stand but cannot lift the body from below it, so
    'land, then get up' is not a thing this robot can do. Landing tall is
    the only version that works, and that is decided in the air."""
    if not (4.4 < env._bf_rot < _BF_DONE):
        return 0.0
    q = env._joint_qpos()
    d2 = float(((q[C.LEG_JOINT_IDS] - C.DEFAULT_POSE[C.LEG_JOINT_IDS]) ** 2).sum())
    return (0.5 * float(np.exp(-d2 / 4.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 1.5 ** 2)))


def _bf_spawn_landed(env):
    """Reverse-curriculum end-state spawn: the ordinary upright pose with the
    flip already CREDITED — practices the landed hold, and the value of being
    'done and standing' propagates back to make finishing worth it."""
    env._bf_rot = _BF_FULL
    return env._get_obs()


def _bf_spawn_mid_roll(env):
    """Mid-maneuver spawn: anywhere from just past the commit point (~75°,
    where upright episodes used to park) to most of the way over (~245° —
    v2 proved the whole 150-300° carry-over needs on-policy data too, not
    just the on-back stretch), WITH backward angular momentum. Static
    spawns taught a dead-stop kip that never worked; a real roll arrives
    at every attitude still rotating, and conserving that momentum is the
    technique. Legs tuck through the middle and extend past inverted;
    rotation credit matches the posed attitude."""
    r = env._rng
    d, m = env.data, env.model
    # Spawn window, overridable for STAGED reverse curriculum (train the
    # landing zone first with a concentrated window, then march the window
    # back toward the entry): the fixed full-range mix gave each hard zone
    # only ~10% of episodes and none of the sub-skills ever converged.
    lo = float(_spawn_knob(env, "MICRODUCK_BF_SPAWN_LO", "1.3"))
    hi = float(_spawn_knob(env, "MICRODUCK_BF_SPAWN_HI", "4.3"))
    rot = r.uniform(lo, hi)
    pitch = -rot
    d.qpos[:] = 0.0
    d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    q = d.qpos
    # Tucked legs through the middle of the roll — and STILL flexed on the
    # way down: the first unfold schedule straightened the legs past ~250°,
    # which posed the landing-zone spawns as a stiff plank tipped nose-down
    # (un-landable — every stage face-planted its landing practice, 0/36
    # stand-ups). The real arrival is a deep crouch, feet under the body.
    fold = 1.0 if rot < 3.8 else 0.85
    hip = fold * r.uniform(0.7, 1.2)
    knee = fold * r.uniform(0.7, 1.2)
    for adr in env.joint_qpos_adr:
        q[adr] = r.uniform(-0.1, 0.1)
    q[env.joint_qpos_adr[2]] = -hip    # left hip_pitch folded to chest
    q[env.joint_qpos_adr[11]] = hip    # right hip_pitch (mirrored sign)
    q[env.joint_qpos_adr[3]] = -knee
    q[env.joint_qpos_adr[12]] = knee
    # Neck follows the phase schedule (tuck in, arch over) so spawned states
    # demonstrate the technique the reward asks for.
    n1, n2 = _bf_neck_targets(rot)
    q[env.joint_qpos_adr[5]] = n1 + r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[6]] = n2 + r.uniform(-0.15, 0.15)
    # Attitude-aware drop height: a fixed spawn z WEDGED the head into the
    # floor at high rotations — the solver ejected it, absorbing all the
    # spawn momentum, and the whole carry-over stretch trained on corrupted
    # states (the invisible wall of v3). Pose high, measure the lowest
    # geom bound, settle to true clearance.
    d.qpos[2] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = [float(d.geom_xpos[g][2]) - float(m.geom_rbound[g])
            for g in range(m.ngeom) if g != env.floor_geom]
    d.qpos[2] = 0.6 - min(lows) + 0.005 + r.uniform(0.0, 0.01)
    d.qvel[4] = -r.uniform(0.5, 4.0)   # arriving mid-roll, still rolling
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    env.prev_joint_vel = env._joint_vel().copy()
    env._bf_rot = rot
    return env._get_obs()


_register(Behavior(
    id="backflip",
    emoji="🤸",
    title="Do a backflip",
    description=(
        "Lean back, roll backwards over the back with the neck tucked, "
        "carry the feet over the top, and land standing."
    ),
    how_it_learns=(
        "The duck earns a little for leaning back at all, more as its total "
        "backward rotation adds up, and the big payout only once it has rolled "
        "a full turn AND is back on its feet — so 'finish the roll and stand "
        "up' is the only real money. The technique terms pay the measured "
        "best-known sequence: ball up tight and roll on the curved back, "
        "whip the feet overhead, then SNAP the neck from tucked to arched — "
        "pressing the head into the floor to lever the body over the crown. "
        "Some episodes start mid-roll (still rolling) or already landed, so "
        "the hard moments get practiced from the first minute of training."
    ),
    keywords=("backflip", "back flip", "backwards flip", "flip backwards",
              "somersault", "backward roll", "roll backwards", "flip"),
    terms=(
        RewardTerm("landed_hold", "Big points for standing on both feet after a full roll",
                   5.0, _bf_landed_hold),
        RewardTerm("flip_progress", "Points as the backward rotation adds up", 1.5,
                   _bf_progress),
        RewardTerm("lean_back", "Points for tipping backwards to start the roll", 1.0,
                   _bf_lean_back),
        RewardTerm("push_off", "Points for a committed backward push-off with the legs", 2.0,
                   _bf_push_off),
        RewardTerm("tuck_ball", "Points for balling up tight — neck pulled in, rolling on the curve", 1.5,
                   _bf_tuck_ball),
        RewardTerm("neck_kip", "Points for the head press-off that levers the body over", 2.0,
                   _bf_neck_kip),
        RewardTerm("arch_over", "Points for arching over the crown once past the press", 1.5,
                   _bf_arch_over),
        RewardTerm("legs_over", "Points for whipping the feet up and over mid-roll", 1.5,
                   _bf_legs_over),
        RewardTerm("land_tall", "Points for straightening the legs before touchdown", 2.5,
                   _bf_land_tall),
        RewardTerm("feet_under", "Points for planting the feet back under the body to rise", 1.5,
                   _bf_feet_under),
        RewardTerm("neck_pushup", "Points for pressing the head down to push itself up", 1.5,
                   _bf_neck_pushup),
        RewardTerm("head_rise", "Points for lifting the head up off the floor to standing height", 2.0,
                   _bf_head_rise),
        RewardTerm("stand_pose", "Points for finishing in the normal standing pose (walk-ready)", 1.5,
                   _bf_stand_pose),
        # 1.0 (was 0.5): landing crooked is accumulated off-axis rotation, and
        # rate is the observable handle on it (final yaw itself is not in the
        # obs — observability law).
        # Attempt-tax lesson, learned AGAIN here: raising this to 1.0 (plus
        # two new penalties) priced every noisy flip attempt above its
        # expected payoff, and stage 5 converged to never flipping — it
        # parked jaw-down on the floor for entire episodes instead. Back to
        # 0.5; the new terms below run at half strength for the same reason.
        RewardTerm("straight_flip", "Penalty for twisting sideways out of the roll", 0.5,
                   _bf_straight_pen, is_penalty=True),
        RewardTerm("still_head", "Penalty for turning the head mid-flip (keeps the roll on-plane)",
                   0.5, _bf_still_head_pen, is_penalty=True),
        RewardTerm("stick_it", "Penalty for still rolling once the flip is done — brake, then stand",
                   0.75, _bf_brake_pen, is_penalty=True),
        RewardTerm("no_jaw_parking", "Penalty for resting the jaw on the floor instead of flipping",
                   1.0, _bf_no_jaw_parking_pen, is_penalty=True),
        RewardTerm("calm_landed", "Penalty for wobbling after the landing (the roll stays free)",
                   1.0, _bf_calm_landed_pen, is_penalty=True),
        RewardTerm("gentle_head", "Penalty each time the head touches down, scaled by impact",
                   0.75, _gentle_plant_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 0.4,
                   _stay_home_pen, is_penalty=True),
        # Half the headstand's weight: rolling onto the back IS the trick —
        # at 1.5 this term was part of the tax wall that kept the first run
        # parked at a 70° recline, afraid to commit. Head impacts stay priced
        # separately (gentle_head).
        RewardTerm("soft_landings", "Penalty for slamming down hard", 0.75,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 1.0,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 1.0,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="full backward rolls that end standing on both feet",
    episode_s=15.0,  # ~2 s of roll, the rest is the landed hold
    scene="all",
    terminate_on_fall=False,
    state_fn=_bf_update,
    spotter_fn=_bf_spotter,
    spawn_families=((0.20, _bf_spawn_landed), (0.35, _bf_spawn_mid_roll)),
    # Staged reverse curriculum over _bf_spawn_mid_roll's window: master the
    # landing zone first, then march the window back toward the entry — the
    # fixed full-range mix gave each hard zone only ~10% of episodes and none
    # of the sub-skills ever converged. Each stage fine-tunes the previous.
    # Windows chosen from measured zone batteries: landing and bridge
    # converge quickly once their spawn poses are physically right, but the
    # arrival-to-carry transition (~92-126° — on the back, committing across
    # the crown) got ~4% of episodes under a uniform full window and stayed
    # the chain's weak link — it gets its own concentrated stage.
    # Each focused stage is ~90% its own practice (probs are landed,mid-roll —
    # the declared 0.20/0.35 mix is only right for the final integration
    # stage, where the point IS mostly-from-standing attempts).
    # Each stage's `detail` narrates its actual spawn window/mix in plain
    # words — the viewer's stage inspector shows it, so keep these in step
    # with the env knobs when tuning.
    curriculum=(
        # 2M, not 1M: at 1M the get-up plateaued around half the landings
        # (stands bounced 0-6/12 across chains — noise around "not enough");
        # a 2M concentrated run went 20/20 with 9.7 s holds. The rise is the
        # hardest sub-skill and gets the budget to match.
        # Stage 0 exists because a per-term audit found the policy leaving
        # 7.3 points/step on the table: standing pays +9.28/step vs the
        # crouch's +2.01, yet it collapsed out of a STANDING spawn in ~1 s.
        # Not an incentive problem — it had never learned to balance (its
        # whole life was spent tumbling). So: learn to stand still first,
        # then learn the trick on top of a body that knows how to stand.
        CurriculumStage("learning to stand steady", 1_000_000,
                        {"MICRODUCK_SPAWN_FAMILY_PROBS": "1.0,0.0",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Every practice run starts already standing, with "
                            "the flip counted as done — the duck's only job is "
                            "to stay up: knees under it, head high, steady. "
                            "It cannot learn to stand up from a fall until it "
                            "knows how to stand at all.")),
        CurriculumStage("learning to land and stand up", 2_000_000,
                        {"MICRODUCK_BF_SPAWN_LO": "4.2",
                         "MICRODUCK_BF_SPAWN_HI": "5.6",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.35,0.60",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "The duck is dropped mid-roll or near the end of "
                            "the flip (~90% of practice runs) and learns to "
                            "catch the landing in a crouch and rise to "
                            "standing.")),
        CurriculumStage("carrying the roll over the top", 1_000_000,
                        {"MICRODUCK_BF_SPAWN_LO": "2.6",
                         "MICRODUCK_BF_SPAWN_HI": "5.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.30,0.65",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Most runs (~75%) drop the duck somewhere between "
                            "on-its-back and nearly landed, still rolling; it "
                            "learns to conserve that momentum across the "
                            "crown and into the landing it already knows.")),
        CurriculumStage("punching through the middle", 1_500_000,
                        {"MICRODUCK_BF_SPAWN_LO": "1.4",
                         "MICRODUCK_BF_SPAWN_HI": "3.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.25,0.70",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Most runs (~80%) start just past the commit "
                            "point, on the back with the roll underway; the "
                            "duck drills the tuck, the leg whip, and the neck "
                            "press-off that carry the roll through its "
                            "hardest stretch.")),
        CurriculumStage("the whole flip from standing", 1_500_000,
                        {"MICRODUCK_BF_SPAWN_LO": "1.3",
                         "MICRODUCK_BF_SPAWN_HI": "5.5",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.25,0.30"},
                        detail=(
                            "More than half the runs now start standing "
                            "upright, the rest sprinkled across the whole "
                            "roll; the duck stitches the push-off entry onto "
                            "the mid-roll and landing skills it already "
                            "has.")),
    ),
))




# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
