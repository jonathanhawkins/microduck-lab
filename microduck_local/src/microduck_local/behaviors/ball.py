import math as _math

from .locomotion import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------------------ find the ball
#
# The eyes for soccer and fetch: a whole-body policy that SCANS for a ball it
# cannot see, turns to it, and keeps it centred in the head camera. The
# shipped kick policies are ball-blind ("the operator aims the robot at the
# ball" — microduck_rl's ball_kick cfg); this brain is the aiming.
#
# There is no ball in the physics. The ball is a point the env tracks and
# projects through the robot's own `head_camera` (the MJCF element, whose +z
# is the optical axis, -x image-up and -y image-right — the sensor sits
# rotated 90 degrees, which is also why the daemon rotates its frames). What
# the policy sees is exactly what a detector on the robot would hand it, and
# it rides the four HEAD command slots (obs[51:55]) — a task-specific meaning
# for the slots, the same way the imitation clip's phase rides two body
# slots. The layout stays the 61-dim contract; the daemon fills the slots:
#
#   [51] bx    horizontal bearing across the frame, -1 hard left .. +1 hard
#              right (duck_detect::Detection::bearing), 0 when not seen
#   [52] by    vertical bearing, -1 bottom edge .. +1 top edge, 0 when not seen
#   [53] seen  1.0 while the detector reports the ball, else 0.0
#   [54] belief  where the daemon BELIEVES the ball is, in the duck's own yaw
#              frame: bearing/pi (-1..+1, + = to the left) times a
#              confidence. 1.0 while seen; while lost, the last bearing
#              dead-reckoned by the gyro's yaw rate, confidence fading as
#              exp(-t_lost / MEM_TAU) down to a FLOOR (a weak "it went that
#              way" never becomes silence). At the start of an episode it is
#              seeded the way the daemon would seed it: usually a noisy prior
#              from before this brain took over (the ball was in view when
#              the kick happened, then rolled off), otherwise the fixed
#              convention +0.15 — "nothing known, sweep left first".
#              A memoryless policy needs someone to remember which side the
#              ball went, and it needs the tie broken when nobody knows:
#              with the ball equally likely on either side and no cue in the
#              obs, turning left and turning right earn the same advantage
#              and the mean action stays at zero while the exploration noise
#              does the finding — the stage-1 export stood and stared at a
#              ball 42 deg off while the stochastic trainer saw it half the
#              time. On the robot the slot is one gyro integral plus a
#              default in the daemon.
#
# And a SCAN CLOCK in two body slots (obs[59], obs[60] = sin, cos of a phase
# that runs at SCAN_PERIOD while the ball is lost and restarts at zero on
# every loss; (0, 1) while seen). The imitation recipe's phase trick: a
# memoryless policy cannot sweep on its own — a sweep is a limit cycle in
# head yaw, and two million steps of PPO produced a static gaze-as-a-
# function-of-belief instead — but with a clock the sweep is a static
# mapping from phase to head yaw and pitch, and the head's +-170 deg range
# means one sinusoidal sweep covers nearly the whole circle. The daemon
# runs the same clock. Period 4 s is the pace of the search; a stage may
# tighten it.
#
# Detector realism: updates every DETECT_EVERY control steps (a 15-30 Hz NPU
# detector against the 50 Hz loop), a small bearing jitter under obs_noise,
# and an optional per-update dropout.
#
# Every knob below is a physics/spawn knob a curriculum stage may ladder
# (AGENTS.md: stages ladder the world, never the pay). The camera sits
# ~25 cm up, so a ball closer than ~0.5 m is BELOW a level gaze — finding it
# means nodding down as well as sweeping, which the pitch bands of the
# coverage pay make explicit.

BALL_RADIUS = 0.035           # the 70 mm / 15 g kick ball (microduck_rl ball.xml)
_BALL_KNOBS = {
    # Camera field of view as the detector sees it, in the ROBOT's frame.
    #
    # From the camera's own datasheet (2026-09-04): EFL 2.9 mm giving
    # H 116 deg, V 60 deg, D 142.2 deg on a 1920x1080 / 2.75 um / 1/2.9 in
    # sensor. Those are the SENSOR's own axes; the part is mounted rotated
    # 90 deg (which is why the daemon rotates its frames), so in the robot's
    # frame they SWAP: 60 deg across the frame, 116 deg up it. Tall and
    # narrow, which is the right shape for a duck hunting a ball on the floor.
    #
    # The previous values here (48 x 62) were placeholders written for a 4:3
    # IMX219 that this robot does not carry. They were wrong in both axes and,
    # worse, wrong in RATIO: 1.29 against the real 1.93.
    #
    # A 116 deg lens is a fisheye, so mind the projection — but the model is
    # already right: `_ball_sense` computes an ANGLE (atan2) and divides by the
    # half-FOV angle, which is an equidistant f-theta mapping, exactly what a
    # fisheye does. A rectilinear model (tan/tan) would be badly wrong out at
    # 58 deg. Nothing to rework; these numbers were the whole gap.
    #
    # Measured (docs/roadmap.md section 2, 60-episode batteries at
    # --events 0.33): found rate is flat across HFOV 24-90 deg, because the
    # bearing is reported NORMALIZED by the FOV and the geometry cancels.
    # VFOV is the axis that bites — in-frame share runs 58% at 40 deg to 75%
    # at 116 — and ORIENTATION matters most: mounted the other way round
    # (116 across, 60 up) the duck loses 10 points of in-frame share and takes
    # 40% longer to reach the kick handoff. **If the camera is ever remounted
    # landscape, swap these back and retrain.**
    "MICRODUCK_BALL_HFOV_DEG": 60.0,
    "MICRODUCK_BALL_VFOV_DEG": 116.0,
    "MICRODUCK_BALL_MAX_RANGE": 3.0,     # m — beyond this the detector has no box
    # Spawn: distance window and how far around the duck (yaw, rad) the
    # ball may start. pi = anywhere; 1.2 = in the front 140 deg.
    "MICRODUCK_BALL_DIST_LO": 0.3,
    "MICRODUCK_BALL_DIST_HI": 1.5,
    "MICRODUCK_BALL_BEARING_MAX": np.pi,
    # Mid-episode events (per second): the ball teleports (a new search) or
    # rolls away (a track, then a re-acquisition using the memory slot).
    "MICRODUCK_BALL_EVENT_RATE": 0.33,
    "MICRODUCK_BALL_ROLL_PROB": 0.5,
    "MICRODUCK_BALL_ROLL_SPEED_LO": 0.3,
    "MICRODUCK_BALL_ROLL_SPEED_HI": 0.9,
    # Detector cadence (control steps), jitter (normalized bearing units),
    # dropout (probability an update reports nothing while the ball is in
    # frame) and the memory's fade time constant (s).
    #
    # 2 steps = 25 Hz against the 50 Hz control loop. The bottleneck this
    # models is the NPU, NOT the camera: the sensor does 1920x1080 at up to
    # 90 fps, which is ~9x more than this pipeline can consume, so frame rate
    # is not where compute should go. **The real requirement is >= 10 Hz
    # detection**, measured (docs/roadmap.md section 2): 50/25/17/10 Hz are
    # indistinguishable, and between 10 and 6 Hz the behavior falls off a
    # cliff — centred share 60% -> 23%, kick handoff 75% -> 13%, falls 1 -> 4,
    # and at 4 Hz the handoff never fires at all. Note the trap: `found` stays
    # 97% at EVERY rate down to 4 Hz, because "ever saw it" is satisfied
    # eventually. Only the AIMING columns see the collapse.
    "MICRODUCK_BALL_DETECT_EVERY": 2.0,
    "MICRODUCK_BALL_JITTER": 0.02,
    "MICRODUCK_BALL_DROPOUT": 0.0,
    "MICRODUCK_BALL_MEM_TAU": 4.0,
    # The belief slot's floor confidence, and how episodes start: with
    # PRIOR_PROB the daemon "remembers" the ball's bearing +- PRIOR_NOISE
    # (rad) at PRIOR_CONF; otherwise it knows nothing and the slot carries
    # the sweep-left-first convention (+0.15).
    "MICRODUCK_BALL_SCAN_PERIOD": 4.0,   # s per sweep cycle of the scan clock
    "MICRODUCK_BALL_MEM_FLOOR": 0.15,
    "MICRODUCK_BALL_PRIOR_PROB": 0.7,
    "MICRODUCK_BALL_PRIOR_NOISE": 0.6,
    "MICRODUCK_BALL_PRIOR_CONF": 0.5,
}
# "Nothing known": the belief slot reads +0.15 — bearing +pi/2 (left) at 0.3
# confidence. A convention the daemon and the policy share, so a duck with
# no idea always starts its sweep the same way instead of not at all.
_BALL_NO_PRIOR_MEM = _math.pi / 2
_BALL_NO_PRIOR_CONF = 0.3
# Handoff gate — "squared up on the ball", the state a kick policy wants
# handed to it. Both halves are things the ROBOT can evaluate (detector
# report + joint encoders), never the privileged truth: the daemon has to be
# able to run this same test. Ball centred in the frame AND the head pointing
# straight ahead means the BODY is pointing at the ball, which is the
# precondition ball_kick_* was trained under (it is ball-blind and expects to
# be aimed). Held for half a second so a sweep passing across the ball does
# not trip it.
# In DEGREES off the optical axis, not in normalized-bearing units. The
# detector's bx/by are divided by the half-FOV, so a tolerance expressed in
# them silently rescales with the camera: swapping the 48x62 placeholder for
# the real 60x116 lens moved "centred" from +-7.8 deg to +-14.5 deg vertically
# with no reward edited, and moved this gate with it (docs/roadmap.md section
# 2). Degrees are the units the kick actually cares about. The daemon can
# still run this test — it knows its own FOV, which it needs anyway to produce
# a normalized bearing at all.
_BALL_AIM_DEG = 7.0           # ball this many degrees off the axis, or closer
_BALL_AIM_HEAD_YAW = 0.25     # rad (~14 deg) — head aligned with the body
_BALL_AIM_STEPS = 25          # control steps (0.5 s at 50 Hz)
# Centring pay, same units: a wide pull and a tight polish (`_ball_eyes_on`).
# Chosen to reproduce what the 48x62 placeholder happened to mean, which is
# what the working brains trained under: its 0.25/0.6 normalized stds worked
# out to 6.0/14.4 deg across the frame and 7.8/18.6 deg up it.
_BALL_EYES_TIGHT_DEG = 7.0
_BALL_EYES_WIDE_DEG = 16.0
# Resolved from the contract so a joint reordering cannot silently point this
# at the wrong servo (the reward stack looks joints up by name for the same
# reason). head_yaw's DEFAULT_POSE entry is 0, so raw angle == relative here.
_BALL_HEAD_YAW_ID = C.JOINT_NAMES.index("head_yaw")

# `face_the_ball`'s tight layer, in rad. 0.2, not the 0.4 this recipe shipped
# with: at 0.4 the term still paid ~2/3 at 19 deg off, so the last 20 deg of
# the turn had almost no gradient behind it and the duck finished the job with
# its neck. docs/roadmap.md item 1 A/B'd 0.4 vs 0.2 against a seed-matched
# control, and tightening it nearly doubles the kick handoff (38% -> 68% of
# episodes) and takes head yaw from 25 to 19 deg for ONE extra fall in sixty.
# The feared failure — a steeper Gaussian producing a policy that fights to
# hold an exact pose — did not appear: 0 reversals, and it renders as a clean
# square-up. `body_aimed` buys more aim than this and costs ten falls in sixty;
# the veto on falls is why this is the shipped fix and that one is at weight 0.
_BALL_FACE_TIGHT_STD = 0.2

# `turn_to_belief` pays only while the ball is OUT of frame. docs/roadmap.md
# item 1 A/B'd ungating it (fix 3) and it is the worse recipe on every axis —
# a yaw-rate pay that never switches off makes ARRIVING worth nothing, so the
# duck turns in to 15 deg and then drifts back out to 28 deg to have something
# left to turn toward. Kept as a named constant because that is the experiment.
_BALL_TURN_GATED_TO_LOST = True

_BALL_KEEPOUT = 0.12          # m — a rolling ball stops at the duck's feet
_BALL_ARENA = 2.0             # m — a rolling ball stops at the arena edge
# Gaze coverage bins for the sweep pay: 10 deg of camera yaw x two pitch
# bands (nose down past -25 deg: the near floor; everything else). No "up"
# band: a floor ball is never above the horizon, and with one the stage-4
# export spent sweep time looking at the ceiling for the coverage pay.
_BALL_YAW_BINS = 36
_BALL_PITCH_DOWN = -0.436           # rad: below this is the near-floor band
_BALL_PITCH_BANDS = 2


def _ball_knob(env, key: str) -> float:
    v = _spawn_knob(env, key)
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    return float(_BALL_KNOBS[key])


def _ball_knobs(env) -> dict[str, float]:
    """Every knob, resolved ONCE per episode (an os.environ lookup per knob
    per step was a third of the hook's cost). Knobs are stage-level
    settings, so per-episode is as fresh as they can ever change."""
    k = {key: _ball_knob(env, key) for key in _BALL_KNOBS}
    k["half_h"] = _math.radians(k["MICRODUCK_BALL_HFOV_DEG"]) / 2
    k["half_v"] = _math.radians(k["MICRODUCK_BALL_VFOV_DEG"]) / 2
    k["detect_every"] = max(1, int(k["MICRODUCK_BALL_DETECT_EVERY"]))
    k["mem_decay"] = _math.exp(-C.CTRL_DT / max(k["MICRODUCK_BALL_MEM_TAU"], 1e-3))
    k["scan_rate"] = 2 * _math.pi / max(k["MICRODUCK_BALL_SCAN_PERIOD"], 0.1)
    env._ball_k = k
    return k


def _ball_camera(env):
    """(position, forward, image-right, image-up) of the head camera, world
    frame. The MJCF `head_camera` element's +z is the optical axis; its -x
    is image-up and its -y image-right (measured off the STAND keyframe:
    level gaze straight down trunk +x, 0.248 m up)."""
    cid = getattr(env, "_ball_cam_id", None)
    if cid is None:
        cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if cid < 0:
            raise RuntimeError("scene has no head_camera element")
        env._ball_cam_id = cid
        env._ball_cam_xpos = env.data.cam_xpos[cid]
        env._ball_cam_xmat = env.data.cam_xmat[cid]
    R = env._ball_cam_xmat
    fwd = R[2::3]          # column 2
    right = -R[1::3]       # -column 1
    up = -R[0::3]          # -column 0
    return env._ball_cam_xpos, fwd, right, up


def _ball_place(env, dist: float, bearing: float) -> None:
    """Put the ball `dist` m from the trunk at `bearing` rad in the duck's
    yaw frame (+ = to the left), on the floor."""
    yaw = _trunk_yaw(env) + bearing
    t = env._trunk_xpos
    env.ball_pos[0] = t[0] + dist * _math.cos(yaw)
    env.ball_pos[1] = t[1] + dist * _math.sin(yaw)
    env.ball_pos[2] = BALL_RADIUS
    env.ball_vel[:] = 0.0
    env._ball_roll_left = 0.0


def _ball_spawn(env) -> tuple[float, float]:
    r = env._rng
    k = env._ball_k
    dist = float(r.uniform(k["MICRODUCK_BALL_DIST_LO"], k["MICRODUCK_BALL_DIST_HI"]))
    span = min(abs(k["MICRODUCK_BALL_BEARING_MAX"]), _math.pi)
    bearing = float(r.uniform(-span, span))
    _ball_place(env, dist, bearing)
    return dist, bearing


def _ball_spawn_leaning(env):
    """Reverse-curriculum spawn: already tipped 20-40 deg and still tipping.

    docs/roadmap.md's falls/aim frontier: every reward lever tried trades the
    aim away to buy stability, because the skill actually missing is RECOVERING
    from a committed lean and no episode ever starts in one. The duck only
    reaches 30-40 deg of tilt on its way to the floor — states the value
    function sees once, terminally, and can learn nothing from. This is the
    backflip/headstand pattern applied to that gap: change the world, not the
    pay (AGENTS.md).

    Tilt about a random horizontal axis so recovery is not a one-sided trick,
    and carry a little angular velocity, because the real failure arrives
    rotating — the backflip spawn learned that the hard way ("static spawns
    taught a dead-stop kip that never worked"). Well inside FALL_GRAVITY_Z's
    70 deg, so the episode is always recoverable rather than pre-lost.
    """
    r = env._rng
    d, m = env.data, env.model
    lo = _math.radians(float(_spawn_knob(env, "MICRODUCK_BALL_LEAN_LO_DEG", "20")))
    hi = _math.radians(float(_spawn_knob(env, "MICRODUCK_BALL_LEAN_HI_DEG", "40")))
    rate = float(_spawn_knob(env, "MICRODUCK_BALL_LEAN_RATE", "0.8"))
    tilt = r.uniform(lo, hi)
    axis = r.uniform(-_math.pi, _math.pi)      # which way it is falling
    ax, ay = _math.cos(axis), _math.sin(axis)
    d.qpos[:] = 0.0
    half = tilt / 2
    d.qpos[3:7] = [_math.cos(half), ax * _math.sin(half), ay * _math.sin(half), 0.0]
    q = d.qpos
    for i, adr in enumerate(env.joint_qpos_adr):
        q[adr] = C.DEFAULT_POSE[i] + r.uniform(-0.05, 0.05)
    # Attitude-aware drop height, the backflip spawn's trick: pose high, find
    # the lowest geom bound, settle to true clearance. A fixed z wedges a
    # tilted duck's foot into the floor and the solver ejects it, spending the
    # spawn state on a launch nobody asked for.
    d.qpos[2] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = [float(d.geom_xpos[g][2]) - float(m.geom_rbound[g])
            for g in range(m.ngeom) if g != env.floor_geom]
    d.qpos[2] = 0.6 - min(lows) + 0.005
    # Still tipping the way it is already leaning, plus a little yaw: the duck
    # falls while TURNING toward a ball behind it, so practise that state.
    d.qvel[3] = -ay * rate * r.uniform(0.5, 1.0)
    d.qvel[4] = ax * rate * r.uniform(0.5, 1.0)
    d.qvel[5] = r.uniform(-1.0, 1.0) * rate
    mujoco.mj_forward(m, d)
    return env._get_obs()


def _ball_reset(env) -> None:
    """Behavior.reset_fn: a fresh ball, fresh detector/memory/coverage state."""
    if not hasattr(env, "ball_pos"):
        env.ball_pos = np.zeros(3)
        env.ball_vel = np.zeros(3)
    _ball_knobs(env)
    dist, bearing = _ball_spawn(env)
    env.last_spawn = f"ball {_math.degrees(bearing):+.0f}° {dist:.1f}m"
    env._ball_episode = env.episode_id
    env._ball_step_done = -1
    env._ball_events = 0
    env._ball_cover = set()
    env._ball_new_bins = 0
    # Detector output as the policy sees it (held between updates).
    env._ball_det = np.zeros(3, np.float32)      # bx, by, seen
    env._ball_det_step = -10 ** 9
    # Truth for the rewards.
    env._ball_seen = False
    env._ball_bx = env._ball_by = 0.0
    env._ball_dist = dist
    env._ball_psi = bearing
    env._ball_lost_s = 0.0
    env._ball_aim_steps = 0
    env._ball_scan_phase = 0.0
    env._ball_first_seen_t = None
    env._ball_seen_steps = 0
    env._ball_centred_steps = 0
    env._ball_losses = 0
    # Seed the belief slot (see the module header): a noisy prior from
    # "before this brain took over", or the sweep-left-first convention.
    k = env._ball_k
    if env._rng.uniform() < k["MICRODUCK_BALL_PRIOR_PROB"]:
        noise = k["MICRODUCK_BALL_PRIOR_NOISE"]
        m = bearing + float(env._rng.uniform(-noise, noise))
        env._ball_mem = _math.atan2(_math.sin(m), _math.cos(m))
        env._ball_mem_conf = k["MICRODUCK_BALL_PRIOR_CONF"]
        env._ball_prior = "prior"
    else:
        env._ball_mem = _BALL_NO_PRIOR_MEM
        env._ball_mem_conf = _BALL_NO_PRIOR_CONF
        env._ball_prior = "blind"
    env.last_spawn += f" {env._ball_prior}"
    _ball_sense(env, force=True)


def _ball_advance(env) -> None:
    """One control step of ball kinematics: rare events, rolling, stops."""
    r = env._rng
    k = env._ball_k
    dt = C.CTRL_DT
    rate = k["MICRODUCK_BALL_EVENT_RATE"]
    if rate > 0.0 and r.uniform() < rate * dt:
        env._ball_events += 1
        if r.uniform() < k["MICRODUCK_BALL_ROLL_PROB"]:
            ang = float(r.uniform(-_math.pi, _math.pi))
            speed = float(r.uniform(k["MICRODUCK_BALL_ROLL_SPEED_LO"],
                                    k["MICRODUCK_BALL_ROLL_SPEED_HI"]))
            env.ball_vel[0] = speed * _math.cos(ang)
            env.ball_vel[1] = speed * _math.sin(ang)
            env._ball_roll_left = float(r.uniform(0.5, 2.0))
        else:
            _ball_spawn(env)
    if env._ball_roll_left > 0.0:
        env.ball_pos[0] += env.ball_vel[0] * dt
        env.ball_pos[1] += env.ball_vel[1] * dt
        env._ball_roll_left -= dt
        t = env._trunk_xpos
        dx, dy = env.ball_pos[0] - t[0], env.ball_pos[1] - t[1]
        d2 = dx * dx + dy * dy
        if env._ball_roll_left <= 0.0 or d2 < _BALL_KEEPOUT ** 2 or d2 > _BALL_ARENA ** 2:
            env.ball_vel[:] = 0.0
            env._ball_roll_left = 0.0


def _ball_sense(env, force: bool = False) -> None:
    """Project the ball through the head camera, run the 'detector', keep
    the memory and the sweep coverage, and write the four head slots.

    Scalar `math` throughout: this runs every control step, and numpy's
    per-call dispatch on 3-vectors costs more than the arithmetic."""
    k = env._ball_k
    cam, fwd, right, up = _ball_camera(env)
    bp = env.ball_pos
    vx, vy, vz = bp[0] - cam[0], bp[1] - cam[1], bp[2] - cam[2]
    dist = _math.sqrt(vx * vx + vy * vy + vz * vz)
    f = vx * fwd[0] + vy * fwd[1] + vz * fwd[2]
    if f > 1e-6:
        bx = _math.atan2(vx * right[0] + vy * right[1] + vz * right[2], f) / k["half_h"]
        by = _math.atan2(vx * up[0] + vy * up[1] + vz * up[2], f) / k["half_v"]
        seen = (-1.0 < bx < 1.0 and -1.0 < by < 1.0
                and dist < k["MICRODUCK_BALL_MAX_RANGE"])
    else:
        bx = by = 0.0
        seen = False
    # Body-frame bearing (the memory's truth, and face_the_ball's).
    t = env._trunk_xpos
    psi = _math.atan2(bp[1] - t[1], bp[0] - t[0]) - _trunk_yaw(env)
    psi = _math.atan2(_math.sin(psi), _math.cos(psi))
    was_seen = env._ball_seen
    env._ball_seen, env._ball_bx, env._ball_by = seen, bx, by
    env._ball_dist, env._ball_psi = dist, psi

    # Sweep coverage: which (yaw, pitch band) cells the gaze has visited.
    yaw = _math.atan2(fwd[1], fwd[0])
    pitch = _math.asin(max(-1.0, min(1.0, fwd[2])))
    band = 0 if pitch < _BALL_PITCH_DOWN else 1
    cell = (int((yaw + _math.pi) / (2 * _math.pi) * _BALL_YAW_BINS) % _BALL_YAW_BINS, band)
    cover = env._ball_cover
    if force:
        cover.add(cell)
        env._ball_new_bins = 0
    elif cell in cover:
        env._ball_new_bins = 0
    else:
        cover.add(cell)
        env._ball_new_bins = 1
        if len(cover) >= _BALL_YAW_BINS * _BALL_PITCH_BANDS:
            cover.clear()       # a complete sweep: start counting afresh

    # Detector cadence + noise + dropout, held between updates.
    det = env._ball_det
    fresh_seen = False          # a detector report THIS step, with the ball in it
    if force or env.step_count - env._ball_det_step >= k["detect_every"]:
        env._ball_det_step = env.step_count
        r = env._rng
        seen_det = seen
        if seen and env.obs_noise and k["MICRODUCK_BALL_DROPOUT"] > 0.0 \
                and r.uniform() < k["MICRODUCK_BALL_DROPOUT"]:
            seen_det = False
        if seen_det:
            jit = k["MICRODUCK_BALL_JITTER"] if env.obs_noise else 0.0
            det[0] = bx + (float(r.uniform(-jit, jit)) if jit else 0.0)
            det[1] = by + (float(r.uniform(-jit, jit)) if jit else 0.0)
            det[2] = 1.0
            fresh_seen = True
        else:
            det[:] = 0.0

    # Bookkeeping for the reports, and the memory slot.
    dt = C.CTRL_DT
    if seen:
        env._ball_seen_steps += 1
        # Degrees, like every other aim tolerance here: a "centred" share
        # measured in normalized bearing is not comparable across cameras,
        # which is exactly the trap that hid the FOV rescale.
        if (bx * _math.degrees(k["half_h"])) ** 2 + \
                (by * _math.degrees(k["half_v"])) ** 2 < _BALL_AIM_DEG ** 2:
            env._ball_centred_steps += 1
        if env._ball_first_seen_t is None:
            env._ball_first_seen_t = env.step_count * dt
        env._ball_lost_s = 0.0
    else:
        if was_seen:
            env._ball_losses += 1
        env._ball_lost_s += dt
    # Aim streak for the handoff gate — consecutive steps the DAEMON could
    # call this squared up (see _BALL_AIM_*). Counted here, not in the
    # predicate, so it stays right whether or not anyone is asking: the lab
    # only calls the predicate while a handoff is armed.
    aim2 = ((det[0] * _math.degrees(k["half_h"])) ** 2
            + (det[1] * _math.degrees(k["half_v"])) ** 2)
    if (det[2] > 0.5 and aim2 < _BALL_AIM_DEG ** 2
            and abs(float(env._joint_qpos()[_BALL_HEAD_YAW_ID])) < _BALL_AIM_HEAD_YAW):
        env._ball_aim_steps += 1
    else:
        env._ball_aim_steps = 0
    # The scan clock runs on the DETECTOR's view (what the daemon has), from
    # zero at every loss; it parks at phase 0 while the report says seen.
    if det[2] > 0.5:
        env._ball_scan_phase = 0.0
    elif not force:
        env._ball_scan_phase = (env._ball_scan_phase + k["scan_rate"] * dt) % (2 * _math.pi)
    env.body_cmd[4] = _math.sin(env._ball_scan_phase)
    env.body_cmd[5] = _math.cos(env._ball_scan_phase)
    if fresh_seen:
        # The memory is refreshed from a detector REPORT, never from the
        # truth: between reports (and while a held report goes stale) it
        # dead-reckons like the daemon's would.
        env._ball_mem = psi
        env._ball_mem_conf = 1.0
    elif not force:
        # Dead-reckon: a target fixed in the world drifts through the body
        # frame at minus the body's yaw rate. Confidence fades only once
        # the report in force says the ball is gone.
        m = env._ball_mem - float(env._gyro[2]) * dt
        env._ball_mem = _math.atan2(_math.sin(m), _math.cos(m))
        if det[2] < 0.5:
            env._ball_mem_conf = max(env._ball_mem_conf * k["mem_decay"],
                                     k["MICRODUCK_BALL_MEM_FLOOR"])
    hc = env.head_cmd
    hc[0] = det[0]
    hc[1] = det[1]
    hc[2] = det[2]
    hc[3] = max(-1.0, min(1.0, env._ball_mem / _math.pi)) * env._ball_mem_conf


def _ball_obs(env) -> None:
    """Behavior.obs_fn: advance the ball and sense it once per control step
    (the obs can be rebuilt more than once per step; the world moves once)."""
    if getattr(env, "_ball_episode", None) != env.episode_id:
        return                      # reset in progress; reset_fn seeds first
    if env._ball_step_done == env.step_count:
        return
    env._ball_step_done = env.step_count
    _ball_advance(env)
    _ball_sense(env)


# ------------------------------------------------------------- reward terms

def _ball_off_axis_deg2(env) -> float:
    """Squared angular distance of the ball from the optical axis, in deg^2.

    The detector reports bx/by already divided by the half-FOV, so multiplying
    back by it recovers the angle — the unit every aim tolerance here is
    written in, so that changing the camera cannot silently change what
    "centred" means. Exact, not an approximation: `_ball_sense` builds bx/by
    as atan2(...)/half, an equidistant mapping."""
    k = env._ball_k
    ax = env._ball_bx * _math.degrees(k["half_h"])
    ay = env._ball_by * _math.degrees(k["half_v"])
    return ax * ax + ay * ay


def _ball_eyes_on(env) -> float:
    """Ball centred in the frame — wide pull + tight polish, seen only."""
    if not env._ball_seen:
        return 0.0
    e2 = _ball_off_axis_deg2(env)
    return (0.5 * _math.exp(-e2 / _BALL_EYES_WIDE_DEG ** 2)
            + 0.5 * _math.exp(-e2 / _BALL_EYES_TIGHT_DEG ** 2))


def _ball_in_view(env) -> float:
    return 1.0 if env._ball_seen else 0.0


def _ball_face(env) -> float:
    """Body pointed at the ball. Paid in full while seen and by belief
    confidence while lost (the policy sees the same estimate in slot 54).

    Wide layer = raised cosine, not a Gaussian: a std-1.5 rad Gaussian paid
    0.04 with the ball straight behind and sloped nowhere the policy was —
    the stage-2 export dithered +-25 deg around "ball directly behind" for
    8 s and never committed to the turn (the foot-flatness lesson again:
    price the escapable part). (1 + cos psi) / 2 slopes all the way round;
    the tight layer squares the body up for the kick."""
    gate = 1.0 if env._ball_seen else env._ball_mem_conf
    if gate < 0.05:
        return 0.0
    psi = env._ball_psi
    return gate * (0.25 * (1.0 + _math.cos(psi))
                   + 0.5 * _math.exp(-psi * psi / _BALL_FACE_TIGHT_STD ** 2))


def _ball_new_ground(env) -> float:
    """While the ball is lost: points for pointing the camera somewhere it
    has not looked yet this sweep — a wiggle re-covers the same cells and
    earns nothing, a steady sweep (and a nod through the pitch bands) pays
    every step. The cells are not in the obs, but the strategy that
    collects them (keep turning the way you were turning) is a function of
    head_yaw position and velocity, which are."""
    if env._ball_seen:
        return 0.0
    return float(min(env._ball_new_bins, 1))


def _ball_turn_to_belief(env) -> float:
    """SIGNED body-yaw-rate pay toward the belief while the ball is out of
    frame (the spin recipe's signed pay, aimed by slot 54): turning the way
    the belief points earns up to +1 at 2 rad/s, turning away charges the
    same, scaled by the belief's confidence. The stage-4 export glimpsed
    balls behind it with the head at its limit but never committed the body
    to the turn — the facing term's slope there is shallow, and this is the
    direct price on the missing motion. Zero while seen: the facing and
    centring terms own that regime."""
    if _BALL_TURN_GATED_TO_LOST and env._ball_det[2] > 0.5:
        return 0.0
    conf = env._ball_mem_conf
    if conf < 0.1:
        return 0.0
    d = 1.0 if env._ball_mem > 0.0 else -1.0
    return conf * max(-1.0, min(1.0, d * float(env._gyro[2]) / 2.0))


def _ball_body_aimed(env) -> float:
    """The HANDOFF STATE itself, priced: ball centred in the frame AND the
    head straight ahead — which together mean the body is what is pointing
    at the ball.

    `face_the_ball` prices the body bearing and hopes squaring up falls out
    of it; it does not. Traced over 8 s with the ball 15 deg off, the stage-5
    export held the camera dead centre using 21 deg of HEAD yaw while the
    body bearing sat at 18-20 deg and drifted further away — the head does
    the eyes-on job alone and for free, while turning the body costs steps,
    smoothness and fall risk. So pay the conjunction directly instead: with
    the ball centred, the only way to earn this is to bring the body round
    under the head.

    Both factors are things the ROBOT sees — the detector's bearing (head
    slots 52/53) and one joint encoder — so this is the same test
    `_ball_handoff_due` runs, without the privileged truth `face_the_ball`
    uses. The head-yaw std is 0.3 rad against the gate's 0.25: the pay has
    to still slope where the gate is not yet satisfied."""
    if not env._ball_seen:
        return 0.0
    e2 = _ball_off_axis_deg2(env)
    hy = float(env._joint_qpos()[_BALL_HEAD_YAW_ID])
    return (_math.exp(-e2 / _BALL_EYES_TIGHT_DEG ** 2)
            * _math.exp(-hy * hy / 0.3 ** 2))


def _ball_handoff_due(env) -> bool:
    """Squared up on the ball for half a second — hand over to the kick.

    Behavior.handoff_fn, so render-rollout's `--handoff` and the lab's
    showcase duck ask the identical question. What it does NOT test is
    RANGE: the fake detector reports a bearing, not a box size, so "aimed"
    is all this can honestly assert. ball_kick_* wants the ball ~9 cm in
    front of the kicking foot, so a render that hands off cleanly and then
    whiffs is the expected first result and the argument for an approach
    behavior — see docs/roadmap.md.
    """
    return getattr(env, "_ball_aim_steps", 0) >= _BALL_AIM_STEPS


# ------------------------------------------------------------- read-side

def _ball_caption(env) -> str:
    """One contact-sheet line, built to fit the 34-column tile budget:
    x/y = detector bearing across/up the frame (-1..1), m = the memory slot,
    d = range in m, p = the ball's true bearing in the body frame (deg,
    + left) — what the body still has to turn through."""
    # At most 29 columns: a 320 px tile holds ~31, and a clipped "p+143"
    # read as "p+14" once (the misread the render skill exists to prevent).
    p = _math.degrees(env._ball_psi)
    if env._ball_seen:
        return (f"SEEN x{env._ball_bx:+.2f} y{env._ball_by:+.2f} "
                f"d{env._ball_dist:.1f} p{p:+.0f}")
    return (f"LOST {env._ball_lost_s:.1f}s m{env.head_cmd[3]:+.2f} "
            f"d{env._ball_dist:.1f} p{p:+.0f}")   # lost time = the scan clock


def _ball_markers(env):
    cam, fwd, _, _ = _ball_camera(env)
    return [
        (env.ball_pos.copy(), BALL_RADIUS, (1.0, 0.55, 0.0, 1.0)),
        # a gaze dot 30 cm down the optical axis — cyan when the ball is in frame
        (cam + 0.3 * fwd, 0.012,
         (0.2, 0.9, 1.0, 1.0) if env._ball_seen else (0.9, 0.2, 0.2, 1.0)),
    ]


def _ball_report(env) -> list[str]:
    n = max(1, env.step_count)
    first = ("never" if env._ball_first_seen_t is None
             else f"t={env._ball_first_seen_t:.2f} s")
    return [
        f"ball: first seen {first}; in frame {env._ball_seen_steps / n:.0%} of steps, "
        f"centred (<{_BALL_AIM_DEG:.0f} deg off axis) {env._ball_centred_steps / n:.0%}; "
        f"lost {env._ball_losses}x; {env._ball_events} ball events",
        f"aim streak at the end: {env._ball_aim_steps} steps "
        f"({env._ball_aim_steps * C.CTRL_DT:.2f} s of "
        f"{_BALL_AIM_STEPS * C.CTRL_DT:.2f} s needed to hand off to a kick)",
    ]


def ball_marker_payload(env):
    """What the lab streams for the viewer: [x, y, z, radius] or None."""
    pos = getattr(env, "ball_pos", None)
    if pos is None or getattr(env, "_ball_episode", None) != env.episode_id:
        return None
    return [round(float(v), 4) for v in pos] + [BALL_RADIUS]


_register(Behavior(
    id="find_ball",
    emoji="🔎",
    title="Find the ball",
    description=(
        "Scan for the ball, turn to face it, and keep it centred in the "
        "camera — even when it rolls off or reappears somewhere else."
    ),
    how_it_learns=(
        "The duck is paid every step the ball is in its camera, more for "
        "having it dead centre, and for squaring its body up to it. While the "
        "ball is out of frame the only income is for aiming the camera "
        "somewhere it has not looked yet, so a steady sweep beats a "
        "wiggle, and nodding down finds the near ones. Balls start anywhere "
        "around it and sometimes roll away or jump elsewhere mid-episode, so "
        "it also learns to chase with its eyes and to remember which side "
        "the ball went. The kick and pick-up policies are ball-blind; this "
        "brain is what aims them."
    ),
    keywords=("find the ball", "look for the ball", "where is the ball",
              "scan for", "look around", "track the ball", "watch the ball",
              "find ball", "the ball", "soccer", "fetch"),
    terms=(
        RewardTerm("eyes_on_ball", "Big points for holding the ball dead centre in the camera",
                   3.0, _ball_eyes_on),
        RewardTerm("ball_in_view", "Points every step the ball is anywhere in the camera",
                   1.0, _ball_in_view),
        RewardTerm("face_the_ball", "Points for squaring the body up to the ball",
                   1.5, _ball_face),
        # Weight 0 = measured but NOT priced by default. docs/roadmap.md item 1
        # A/B'd it at 1.0 and 2.0: it is by far the strongest lever on the aim
        # (head yaw 25 -> 8 deg, handoff 38% -> 83%) and it also triples the
        # falls, which this project treats as a veto. Shipping that trade is a
        # call for a human; the term, its test and these numbers are here so
        # the call can be made in one edit.
        RewardTerm("body_aimed", "Points for having the ball centred with the head straight — the body doing the aiming",
                   0.0, _ball_body_aimed),
        RewardTerm("new_ground", "While the ball is lost: points for looking somewhere new",
                   1.0, _ball_new_ground),
        RewardTerm("turn_to_belief", "While the ball is lost: points for turning the body the way it went",
                   1.0, _ball_turn_to_belief),
        # Narrow, deliberately. The two-layer `wide=True` shape was built and
        # A/B'd for this recipe's remaining falls and is Pareto-DOMINATED —
        # docs/roadmap.md has the frontier.
        _upright_term(1.5),
        RewardTerm("step_dont_skid", "Points for lifting the feet to turn (no skid-steering)",
                   1.0, _step_dont_skid),
        RewardTerm("flat_feet", "Points for keeping the feet flat on the floor",
                   0.5, _flat_feet),
        RewardTerm("stay_home", "Penalty for wandering off the spot (turning is fine)",
                   1.0, _stay_home_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for cranking joints to their end stops",
                   1.0, _limit_parking_pen, is_penalty=True),
        *_BASE_REGULARIZERS,
    ),
    default_steps=3_000_000,
    success_metric="seconds to first sight, then the share of steps with the ball centred",
    episode_s=10.0,
    # One-sided on purpose: from a symmetric start (ball unseen, memory
    # empty) a mirror-consistent policy must output a zero yaw sweep — it
    # cannot choose a side to look first. The exported deterministic mean
    # would sit and stare. Locked by tests/test_behaviors.py.
    symmetric=False,
    # A quarter of episodes start already tipping. This teaches real lean
    # recovery — survival from a 20-25 deg lean goes 20% -> 47%, from 30-35 deg
    # 0% -> 17% — and it is what makes the recipe land its kick handoff: 92% of
    # episodes against 85% without, with head yaw 9.5 deg against 15.9, and a
    # third of the falls (2.7 vs 8.0 per 60). It costs in-frame SHARE (68% vs
    # 81%), which is the trade docs/roadmap.md calls the falls/aim frontier —
    # taken deliberately, because the handoff is the deliverable and holding a
    # ball in frame while never squaring up is what this brain used to do.
    spawn_families=((0.25, _ball_spawn_leaning),),
    reset_fn=_ball_reset,
    obs_fn=_ball_obs,
    handoff_fn=_ball_handoff_due,
    # The kick this brain exists to aim. Right-footed by convention (the
    # shipped pair is ball_kick_left/right and upstream's own cfg trains one
    # foot per policy); it is ball-blind, which is the whole point.
    handoff_policy="pollen:ball_kick_right",
    # The turn toward the ball is the DELIVERABLE, not landing drift: the
    # lab's post-handoff yaw correction would spin it straight back.
    handoff_recenter=False,
    caption_fn=_ball_caption,
    markers_fn=_ball_markers,
    report_fn=_ball_report,
    curriculum=(
        CurriculumStage(
            "finding a ball in front", 1_000_000,
            env={"MICRODUCK_BALL_BEARING_MAX": "1.2",
                 "MICRODUCK_BALL_EVENT_RATE": "0.15"},
            detail=("The ball starts somewhere in the front 140 degrees, "
                    "near or far, and rarely moves: head sweeps and nods, "
                    "centring, and squaring up without a big turn."),
        ),
        CurriculumStage(
            "turning around to find it", 2_000_000,
            env={"MICRODUCK_BALL_BEARING_MAX": "3.1416",
                 "MICRODUCK_BALL_EVENT_RATE": "0.33"},
            detail=("Anywhere around the duck, and every few seconds it "
                    "jumps or rolls: stepping the body round to the "
                    "ball, then finding it again after it goes."),
        ),
        CurriculumStage(
            "keeping up with a rolling ball", 1_000_000,
            env={"MICRODUCK_BALL_BEARING_MAX": "3.1416",
                 "MICRODUCK_BALL_EVENT_RATE": "0.5",
                 "MICRODUCK_BALL_ROLL_PROB": "0.8"},
            detail=("Mostly rolling balls, often: tracking with the eyes "
                    "and re-acquiring from the memory slot when one rolls "
                    "out of frame."),
        ),
    ),
))


# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
