from .imitate import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------------------------- running
# Port of microduck_rl's Mjlab-Run task (microduck_run_env_cfg.py), not of
# the velocity WALK recipe. The walk stack prices running out of the
# optimum (dead running-pose std, upright tuned to erase lean, air_time
# outweighing speed). GPU run inverts that: speed is the largest term,
# the pose running-regime actually fires, lean is cheap, smoothness caps
# at -0.5. Local "run" used to copy walk weights and then invent extra
# terms; those policies are not a predictor of a GPU run.

# GPU stage boundaries are iterations x 24 per-env steps, saturating at
# 3000 x 24 = 72k. A local 20M-step/16-env run is 1.25M per-env steps, so the
# ported units saturated at 5.8% of the run — the smoothness tax landed
# during discovery (the exact attempt-tax AGENTS.md forbids) and the speed
# ceiling stopped being a curriculum. _RUN_STAGE_SCALE stretches every stage
# boundary so saturation lands at ~60% of the default run (audit finding #2).
_RUN_NUM_STEPS_PER_ENV = 24
_RUN_STAGE_SCALE = float(os.environ.get("MICRODUCK_STAGE_SCALE",
                                        750_000 / (3000 * 24)))
_RUN_FORWARD_FRAC = 0.55
_RUN_FORWARD_CLAMP = 0.3
_RUN_LIN_VEL_Y = (-0.3, 0.3)
_RUN_ANG_VEL_Z = (-1.0, 1.0)
_RUN_SPEED_STAGES = (
    (0, 0.4),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.45),
    (2000 * _RUN_NUM_STEPS_PER_ENV, 0.5),
    (2500 * _RUN_NUM_STEPS_PER_ENV, 0.55),
    (3000 * _RUN_NUM_STEPS_PER_ENV, 0.6),
    # Extended past the GPU cfg's 0.6 ceiling: this robot delivers ~66% of
    # the commanded speed (0.295 measured at cmd 0.45), so a 0.6 command
    # ceiling caps ACHIEVED speed near ~0.4 by construction — the 0.6 m/s
    # goal was structurally unreachable. 0.9 remains inside the measured
    # kinematic envelope (~1.4 m/s under BAM); whether the gait can cash it
    # is what the late curriculum exists to find out.
    (4000 * _RUN_NUM_STEPS_PER_ENV, 0.7),
    (4500 * _RUN_NUM_STEPS_PER_ENV, 0.8),
    (5000 * _RUN_NUM_STEPS_PER_ENV, 0.9),
    # Second extension after 0.61 m/s was banked at cmd 0.9: envelope ~1.4,
    # delivery ~68%, so 1.1 commanded targets ~0.75 achieved.
    (5500 * _RUN_NUM_STEPS_PER_ENV, 1.0),
    (6000 * _RUN_NUM_STEPS_PER_ENV, 1.1),
)
_RUN_STANDING_STAGES = (
    (0, 0.02),
    (500 * _RUN_NUM_STEPS_PER_ENV, 0.05),
    (1000 * _RUN_NUM_STEPS_PER_ENV, 0.08),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.10),
)
# GPU run action_rate_l2: -0.1 → -0.5 by iter 3000 (not velocity's -1.0).
_RUN_ACTION_RATE_STAGES = (
    (0, 0.1),
    (1000 * _RUN_NUM_STEPS_PER_ENV, 0.2),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.3),
    (2250 * _RUN_NUM_STEPS_PER_ENV, 0.4),
    (3000 * _RUN_NUM_STEPS_PER_ENV, 0.5),
)
_RUN_AIR_MIN = 0.15
_RUN_AIR_MAX = 0.35
_RUN_FOOT_TARGET = 0.03
_RUN_TRACK_STD2 = 0.1
_RUN_ANG_STD2 = 0.5
_RUN_UPRIGHT_STD2 = 0.12
_RUN_WALKING_THRESHOLD = 0.01
_RUN_RUNNING_THRESHOLD = 0.40
# variable_posture per-joint stds, matching GPU run's sagittal loosening.
# Order is LEG_JOINT_IDS: L(yaw,roll,pitch,knee,ankle) R(...).
_RUN_STD_STAND = np.array([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
_RUN_STD_WALK = np.array([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
_RUN_STD_RUN = np.array([0.4, 0.08, 0.6, 0.6, 0.4] * 2)


def _stage_value(n: int, stages: tuple) -> float:
    n = n / _RUN_STAGE_SCALE  # rescale local lifetime steps into GPU stage units
    value = stages[0][1]
    for step, v in stages:
        if n >= step:
            value = v
    return float(value)


def _run_cmd_speed(n: int) -> float:
    return _stage_value(n, _RUN_SPEED_STAGES)


def _run_standing_frac(n: int) -> float:
    return _stage_value(n, _RUN_STANDING_STAGES)


def _run_cmd_norm(env) -> float:
    # Four run terms gate on this each step; memoized in the step cache
    # (commands are frozen for the whole obs+reward phase of a step).
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("cmd_norm")
        if v is not None:
            return v
    c = env.twist_cmd
    v = float(np.linalg.norm(c[:2])) + abs(float(c[2]))
    if cache is not None:
        cache["cmd_norm"] = v
    return v


def _run_speed(env) -> float:
    """GPU track_linear_velocity: body-frame xy vs the twist command, plus
    vz² in the same Gaussian (std²=0.1). Not heading-frame, and not gated
    on upright — those two local inventions paid for dives or taxed the
    lean a run needs."""
    v = env.body_lin_vel()
    c = env.twist_cmd
    err2 = float((c[0] - v[0]) ** 2 + (c[1] - v[1]) ** 2 + v[2] ** 2)
    raw = float(np.exp(-err2 / _RUN_TRACK_STD2))
    # DELIBERATE deviation from the GPU stack: standing pays exactly zero.
    # Upstream's bare Gaussian has height at v=0 whenever the command is low
    # (cmd 0.15 -> 0.80 for immobility), and with upright/pose terms on top a
    # from-scratch policy at OUR sample budget parked itself and collected
    # rent — the user watched it stand perfectly still while this term's bar
    # climbed. Upstream escapes that basin with ~1e9 samples and 4096 envs;
    # we do not. Subtracting the standstill baseline keeps the full gradient
    # from v=0 toward the command while pricing immobility at 0.
    cmd2 = float(c[0] ** 2 + c[1] ** 2)
    if cmd2 < 1e-4:
        # Standing BUCKET (cmd == 0): the baseline subtraction would zero this
        # bucket entirely — no reward for stillness, no charge for wandering —
        # killing the deployment-idle training the bucket exists for. The raw
        # GPU bell is correct here: stillness pays 1, drifting decays it.
        return raw
    base = float(np.exp(-cmd2 / _RUN_TRACK_STD2))
    return max(0.0, raw - base) / max(1.0 - base, 1e-6)


def _run_track_ang(env) -> float:
    """GPU track_angular_velocity: yaw-rate vs cmd[2], plus ω_xy in the
    same Gaussian (std²=0.5)."""
    w = env._gyro
    z_err2 = (float(env.twist_cmd[2]) - float(w[2])) ** 2
    xy_err2 = float(w[0] ** 2 + w[1] ** 2)
    return float(np.exp(-(z_err2 + xy_err2) / _RUN_ANG_STD2))


def _run_air_time(env) -> float:
    """GPU feet_air_time: 1.0 per foot whose CURRENT air time is inside
    [min, max], every step, gated on a live command. 0, 1 or 2.

    The old local version paid a lumped touchdown bonus, which is a
    different function (and ~4× less reward mass for the same stride).
    """
    air = getattr(env, "_run_air", None)
    if air is None:
        air = env._run_air = {"left": 0.0, "right": 0.0}
    paid = 0.0
    moving = _run_cmd_norm(env) > 0.01
    for side in ("left", "right"):
        if env.foot_contact_state[side]:
            air[side] = 0.0
        else:
            air[side] += C.CTRL_DT
        if moving and _RUN_AIR_MIN < air[side] < _RUN_AIR_MAX:
            paid += 1.0
    return paid


def _run_clearance_pen(env) -> float:
    """GPU feet_clearance: -Σ |h − 0.03| · |v_xy|, gated on command.

    A COST, not a Gaussian reward around 4.5 cm (which was above what this
    robot can lift)."""
    if _run_cmd_norm(env) <= 0.01:
        return 0.0
    v6 = _v6_buf(env)
    tot = 0.0
    # Height ABOVE GROUND, not raw geom-center z: the geom center sits at
    # 0.0086 m when the foot is planted, so measuring it raw silently lowered
    # the intended 3 cm swing target to ~2.1 cm of true lift (audit #5).
    # Self-calibrated once, same pattern as foot_flat_ref: the planted-stance
    # center height IS the zero point. GPU measures via a ray sensor, ~0 when
    # planted, so this restores its semantics.
    z0 = getattr(env, "_run_foot_z0", None)
    if z0 is None:
        both_down = all(env.foot_contact_state.values())
        z0 = (float(np.mean([env.data.geom_xpos[g][2]
                             for g in env.foot_geoms.values()]))
              if both_down else 0.0086)
        env._run_foot_z0 = z0
    for gid in env.foot_geoms.values():
        z = float(env.data.geom_xpos[gid][2]) - z0
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 gid, v6, 0)
        vel = float(np.hypot(v6[3], v6[4]))
        tot += abs(z - _RUN_FOOT_TARGET) * vel
    return -tot


def _run_upright(env) -> float:
    """GPU run upright: isotropic Gaussian, std²=0.12 (walk uses 0.05 to
    erase a 2–4° lean; a run needs that lean). Additive — not a multiplier
    on speed."""
    g = env._projected_gravity()
    return float(np.exp(-(g[0] ** 2 + g[1] ** 2) / _RUN_UPRIGHT_STD2))


def _run_pose(env) -> float:
    """GPU variable_posture on LEG joints, with a reachable running_threshold
    of 0.40 m/s (mjlab's default 1.5 is dead code on this robot)."""
    speed = _run_cmd_norm(env)
    if speed < _RUN_WALKING_THRESHOLD:
        std = _RUN_STD_STAND
    elif speed < _RUN_RUNNING_THRESHOLD:
        std = _RUN_STD_WALK
    else:
        std = _RUN_STD_RUN
    q = env._joint_pos_rel()[C.LEG_JOINT_IDS]
    # t.sum()/10 is np.mean's own reduction and division, minus the wrapper.
    t = (q / std) ** 2
    return float(np.exp(-float(t.sum() / 10)))


def _run_head_pose(env) -> float:
    """GPU head_pose_tracking: per-joint Gaussian on (q − head_cmd), std=0.5."""
    err = env._joint_pos_rel()[C.HEAD_JOINT_IDS] - env.head_cmd
    return float(np.exp(-((err / 0.5) ** 2)).sum() / 4)


def _run_slip_pen(env) -> float:
    """GPU feet_slip: -Σ |v_xy|² on contacting feet, gated on command.

    Touchdown is charged (GPU does). The local 'skip first contact' guard
    was for a 10×-too-heavy weight; at GPU's -0.1 it is the wrong prior."""
    if _run_cmd_norm(env) <= 0.01:
        return 0.0
    v6 = _v6_buf(env)
    tot = 0.0
    for side, gid in env.foot_geoms.items():
        if not env.foot_contact_state[side]:
            continue
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 gid, v6, 0)
        tot += float(v6[3] ** 2 + v6[4] ** 2)
    return -tot


def _run_body_ang_pen(env) -> float:
    """GPU body_ang_vel: -ω_xy² on the trunk (run weight -0.025)."""
    w = env._gyro
    return -float(w[0] ** 2 + w[1] ** 2)


def _run_action_rate_weight(n: int) -> float:
    return _stage_value(n, _RUN_ACTION_RATE_STAGES)


def _run_action_rate_pen(env) -> float:
    """GPU action_rate_l2: -weight × Σ(Δa²), weight -0.1 → -0.5.

    No extra 0.02 scale — that made this term ~50× weaker than GPU while
    comments claimed they matched. Lifetime counter lives on the env
    (incremented in BehaviorEnv._compute_reward), so the ramp tracks
    training progress across episode resets."""
    n = getattr(env, "_lifetime_steps", 0)
    da2 = float(((env.last_action - env.prev_action) ** 2).sum())
    return -_run_action_rate_weight(n) * da2


def _run_lateral_pen(env) -> float:
    """Bounded penalty for sliding off the run's LINE (<= 0).

    Heading-relative: the displacement since spawn is projected onto the
    lateral axis of the spawn heading, so a straight run at any start yaw is
    free (world-y alone charged a full unit for running straight under a
    random yaw — measured before this was first fixed). Saturates one unit at
    0.5 m off the line. Not in the default run recipe (the GPU-port reward
    tracks commanded twist and needs no line) — offered as the
    `stay_on_line` catalog slider for anyone training an autonomous
    straight run; spec locked by test_penalty_scale.py.
    """
    home = getattr(env, "home_xy", None)
    if home is None:
        return 0.0
    cmd = getattr(env, "twist_cmd", None)  # see _face_home_pen: commanded
    if cmd is not None and (abs(float(cmd[1])) > 0.05 or abs(float(cmd[2])) > 0.1):
        return 0.0                          # turns/sidesteps own the line
    dx = float(env._trunk_xpos[0]) - float(home[0])
    dy = float(env._trunk_xpos[1]) - float(home[1])
    yaw = getattr(env, "home_yaw", None)
    if yaw is not None:
        dy = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return -min((dy / 0.5) ** 2, 1.0)


CATALOG["stay_on_line"] = RewardTerm(
    "stay_on_line",
    "Penalty for drifting sideways off the straight line it started on",
    1.0, _run_lateral_pen, is_penalty=True)


_register(Behavior(
    id="run",
    emoji="🏃",
    title="Run forward",
    description=(
        "Run forward at a steady clip: quick alternating steps, feet clearing "
        "the floor, body upright and travelling in a straight line."
    ),
    how_it_learns=(
        "The scorecard is the official GPU run task, not a walk retune: the "
        "duck is paid most for matching the commanded body-frame speed, then "
        "for a longer aerial phase and a reachable running pose. A fifth of "
        "practice is still omnidirectional walking, and a little is standing "
        "still — the idle command the robot actually sends at rest. Falling "
        "ends the attempt."
    ),
    keywords=("run", "running", "run forward", "run fast", "sprint", "jog"),
    terms=(
        RewardTerm("keep_pace", "Big points for matching the commanded body-frame speed",
                   4.0, _run_speed),
        RewardTerm("track_turn", "Points for matching the commanded yaw rate",
                   2.0, _run_track_ang),
        RewardTerm("air_time", "Points each step a foot is in a running-length flight",
                   3.0, _run_air_time),
        RewardTerm("stay_upright", "Points for keeping the body upright (lean is cheap)",
                   2.0, _run_upright),
        RewardTerm("pose", "Points for a speed-appropriate leg pose",
                   1.0, _run_pose),
        # 3.5 (was 2.0): at speed the policy ducks its heavy head toward the
        # floor to shift weight forward until it face-plants — measured: a
        # 1-rad slump only cost ~1.0 of the 2.0, cheap enough to be worth it.
        # Head posture is fully observable (neck/head joints + pitch).
        RewardTerm("head_up", "Points for tracking the head-pose command",
                   3.5, _run_head_pose),
        RewardTerm("foot_clearance", "Points lost when a moving foot is at the wrong height",
                   2.0, _run_clearance_pen, is_penalty=True),
        RewardTerm("plant_the_foot", "Points lost when a planted foot skids",
                   0.1, _run_slip_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky movements (ramps in as the gait appears)",
                   1.0, _run_action_rate_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops",
                   1.0, _limit_parking_pen, is_penalty=True),
        # Straightness anchors — deliberately NOT in the GPU stack, which
        # tracks commanded twist and lets a teleoperator own the heading. We
        # ask for a straight autonomous run, and rate terms cannot stop slow
        # accumulated drift (session law, learned twice): track_turn charges
        # turning FAST, so a lazy arc was free and the user watched the duck
        # curve away. These anchor ABSOLUTE heading and the lateral line.
        # NOT PORTED from the GPU run cfg, documented for honesty (audit #4):
        # foot_swing_height (-0.25, needs per-foot peak-height tracking),
        # angular_momentum (-0.01 there; calm_roll covers the same axis here),
        # self_collisions (-1.0; no local contact sensor for it).
        # NO absolute-heading terms, by hard-won design: the 61-obs contract
        # carries gyro+gravity+joints+actions+commands — pitch and roll but NO
        # yaw. A memoryless policy cannot perceive its heading error, so terms
        # anchored to it (face_home, hold_the_line, an anchor-frame keep_pace
        # — all three were tried) are unlearnable noise; the duck circled for
        # 30M+ steps regardless. Straightness is the COMMANDER'S job: a
        # 3-line heading-hold (cmd wz = -4*yaw_err) on top of the trained
        # yaw-rate tracking measured 0.293 m/s, 4 deg err, 0 falls where a
        # day of reward surgery never went below 64 deg.
        RewardTerm("calm_roll", "Penalty for rolling/pitching the trunk too fast",
                   0.025, _run_body_ang_pen, is_penalty=True),
    ),
    default_steps=20_000_000,
    success_metric="metres per second sustained without falling",
    episode_s=20.0,
    forward_cmd=0.6,
))




# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
