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
# is the optical axis, -x image-up and -y image-right — the IMX219 sits
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
    # Camera field of view as the detector sees it (portrait after the
    # daemon's 90 deg rotation of the 4:3 IMX219: ~48 deg across, ~62 deg tall).
    "MICRODUCK_BALL_HFOV_DEG": 48.0,
    "MICRODUCK_BALL_VFOV_DEG": 62.0,
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
_BALL_KEEPOUT = 0.12          # m — a rolling ball stops at the duck's feet
_BALL_ARENA = 2.0             # m — a rolling ball stops at the arena edge
# Gaze coverage bins for the sweep pay: 10 deg of camera yaw x three pitch
# bands (nose down past -25 deg: the near floor; level; up).
_BALL_YAW_BINS = 36
_BALL_PITCH_EDGES = (-0.436, 0.0)   # rad, band boundaries


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
    band = 0 if pitch < _BALL_PITCH_EDGES[0] else 1 if pitch < _BALL_PITCH_EDGES[1] else 2
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
        if len(cover) >= _BALL_YAW_BINS * 3:
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
        if -0.25 < bx < 0.25 and -0.25 < by < 0.25:
            env._ball_centred_steps += 1
        if env._ball_first_seen_t is None:
            env._ball_first_seen_t = env.step_count * dt
        env._ball_lost_s = 0.0
    else:
        if was_seen:
            env._ball_losses += 1
        env._ball_lost_s += dt
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

def _ball_eyes_on(env) -> float:
    """Ball centred in the frame — wide pull + tight polish, seen only."""
    if not env._ball_seen:
        return 0.0
    e2 = env._ball_bx ** 2 + env._ball_by ** 2
    return 0.5 * _math.exp(-e2 / 0.6 ** 2) + 0.5 * _math.exp(-e2 / 0.25 ** 2)


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
    return gate * (0.25 * (1.0 + _math.cos(psi)) + 0.5 * _math.exp(-psi * psi / 0.4 ** 2))


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
        f"centred (|b|<0.25) {env._ball_centred_steps / n:.0%}; "
        f"lost {env._ball_losses}x; {env._ball_events} ball events",
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
        RewardTerm("new_ground", "While the ball is lost: points for looking somewhere new",
                   1.0, _ball_new_ground),
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
    reset_fn=_ball_reset,
    obs_fn=_ball_obs,
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
