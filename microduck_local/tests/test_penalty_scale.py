"""Penalty terms are priced in the same currency as the rewards.

Positive terms are bounded [0,1] and multiplied by their weight, so a weight
reads as "worth N units". Penalties used hand-picked coefficients instead, and
`save_energy` landed ~77x too small to influence anything — the reason raising
its weight during the backflip safety work never changed behavior. These lock
the physical normalizers so that can't drift back.
"""

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors import (
    _RUN_AIR_MAX,
    _RUN_AIR_MIN,
    _RUN_STAGE_SCALE,  # noqa: F401
    _RUN_TRACK_STD2,
    _RUN_UPRIGHT_STD2,
    QVEL2_MAX,
    TAU2_MAX,
    BehaviorEnv,
    _joint_vel_pen,
    _run_action_rate_pen,
    _run_action_rate_weight,
    _run_air_time,
    _run_cmd_speed,
    _run_speed,
    _run_standing_frac,
    _tau2_max,
    _torque_pen,
)


def _env(behavior="run"):
    env = BehaviorEnv(behavior, obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    return env


def test_tau2_max_matches_the_actuators():
    """The constant is the model's own saturation, not a guess — if the
    actuator changes, this fails instead of silently mispricing motor strain."""
    env = _env()
    fr = env.model.actuator_forcerange
    assert np.allclose(fr[:, 1], 0.96), "unexpected force limit"
    assert TAU2_MAX == pytest.approx(float(np.sum(fr[:, 1] ** 2)), rel=1e-3)


def test_torque_penalty_is_minus_one_at_saturation():
    """A weight of 1.0 must mean 'all motors pinned at their limit costs one
    full unit of reward per step' — the same units the positive terms use."""
    env = _env()
    env.data.actuator_force[:] = env.model.actuator_forcerange[:, 1]
    assert _torque_pen(env) == pytest.approx(-1.0, rel=1e-3)
    env.data.actuator_force[:] = 0.0
    assert _torque_pen(env) == 0.0


def test_torque_penalty_is_no_longer_negligible():
    """Regression on the actual bug: under the old 1e-3 coefficient a hard-
    working duck paid ~0.003 against a per-step reward near 4.5."""
    env = _env()
    env.data.actuator_force[:] = 0.46  # ~half the limit on every joint
    old = -1e-3 * float(np.sum(env.data.actuator_force ** 2))
    new = _torque_pen(env)
    assert abs(old) < 0.01, "old scale was not negligible — premise changed"
    assert abs(new) > 0.2
    # Against the normalizer actually in force, not the rounded module
    # constant — _tau2_max reads the model (and the BAM current clamp).
    assert abs(new / old) == pytest.approx(1.0 / _tau2_max(env) / 1e-3, rel=1e-6)


def test_joint_vel_penalty_bounded_by_physical_peak():
    """-1.0 when every joint slews at the peak speed the hardware reaches."""
    env = _env()
    peak = np.sqrt(QVEL2_MAX / C.NUM_JOINTS)
    env.data.qvel[env.joint_qvel_adr] = peak
    assert _joint_vel_pen(env) == pytest.approx(-1.0, rel=1e-3)


def test_run_speed_tracks_the_commanded_pace_not_the_top_speed():
    """The policy is SHOWN a per-episode command in obs[48]; it must be PAID
    for hitting that command. Scoring against the recipe's constant top speed
    made the observation actively misleading (it varied, the target did not),
    so the only optimal policy was to sprint flat out on every episode."""
    env = _env("run")
    top = env.behavior.forward_cmd
    for cmd in (0.15, 0.30, top):
        env.reset(seed=1)
        env.twist_cmd[:] = (cmd, 0.0, 0.0)
        # Yaw is not randomized, so world +x is body-x. GPU tracks body frame.
        env.data.qvel[0:3] = (cmd, 0.0, 0.0)
        mujoco.mj_forward(env.model, env.data)
        score = _run_speed(env)
        assert score > 0.9, f"cmd {cmd}: matched pace scored only {score:.3f}"


def test_run_speed_penalises_sprinting_past_a_slow_command():
    """The converse: at a slow command, running flat out must NOT score full
    marks — that is exactly the behavior the old bug rewarded."""
    env = _env("run")
    env.reset(seed=1)
    env.twist_cmd[:] = (0.15, 0.0, 0.0)
    env.data.qvel[0:3] = (env.behavior.forward_cmd, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    assert _run_speed(env) < 0.5


def test_run_speed_is_body_frame_not_heading():
    """GPU track_linear_velocity uses root_link_lin_vel_b. A pitched trunk
    with world-+x velocity must NOT score as a perfect match to cmd_vx —
    heading-frame speed would, and that was the local/GPU frame split."""
    env = _env("run")
    env.reset(seed=1)
    pitch = np.deg2rad(30)
    env.data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    env.data.qvel[0:3] = (0.4, 0.0, 0.0)
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    body = env.body_lin_vel()
    heading, _, _ = env.heading_lin_vel()
    assert abs(heading - 0.4) < 0.02
    assert abs(body[0] - 0.4) > 0.05
    score = _run_speed(env)
    # Body frame for ALL commands (GPU semantics): the anchor-frame variant
    # was removed as unlearnable — no yaw in the obs. Perfect heading match
    # at 0.4 would score ~1; body-frame error must bite.
    assert score < 0.9


def test_run_air_time_does_not_pay_a_shuffle():
    """A one-control-step unweight (0.02 s) is skating, not a stride."""
    env = _env("run")
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    env._run_air = {"left": C.CTRL_DT, "right": 0.0}
    env.foot_contact_state = {"left": False, "right": True}
    assert C.CTRL_DT < _RUN_AIR_MIN
    assert _run_air_time(env) == 0.0


def test_run_air_time_pays_inside_the_stride_window():
    """Dense GPU form: 1.0 this step while current air time is in-window."""
    env = _env("run")
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    # Already at 0.20 s; the fn adds CTRL_DT then checks the window.
    env._run_air = {"left": 0.20, "right": 0.0}
    env.foot_contact_state = {"left": False, "right": True}
    assert _RUN_AIR_MIN < 0.20 + C.CTRL_DT < _RUN_AIR_MAX
    assert _run_air_time(env) == pytest.approx(1.0)


def test_run_air_time_silent_when_standing():
    env = _env("run")
    env.twist_cmd[:] = 0.0
    env._run_air = {"left": 0.20, "right": 0.20}
    env.foot_contact_state = {"left": False, "right": False}
    assert _run_air_time(env) == 0.0


def test_air_time_state_does_not_leak_across_episodes():
    """Air banked during a terminal fall must not be paid out at the next
    episode's first in-window step."""
    env = _env("run")
    env.step(np.zeros(14, np.float32))
    env._run_air["left"] = 5.0
    env.reset(seed=7)
    assert env._run_air == {"left": 0.0, "right": 0.0}


def test_run_speed_keeps_a_gradient_far_from_the_command():
    """GPU std²=0.1 still slopes when the duck is well below a fast command."""
    env = _env("run")
    env.reset(seed=1)
    env.twist_cmd[:] = (0.8, 0.0, 0.0)
    scores = []
    for v in (0.20, 0.30, 0.40):
        env.data.qvel[0:3] = (v, 0.0, 0.0)
        mujoco.mj_forward(env.model, env.data)
        scores.append(_run_speed(env))
    assert scores[0] < scores[1] < scores[2]
    assert scores[2] - scores[0] > 0.02, f"slope too flat: {scores}"


def test_run_speed_still_peaks_at_the_commanded_pace():
    env = _env("run")
    env.reset(seed=1)
    env.twist_cmd[:] = (0.45, 0.0, 0.0)
    out = {}
    for v in (0.25, 0.45, 0.65):
        env.data.qvel[0:3] = (v, 0.0, 0.0)
        mujoco.mj_forward(env.model, env.data)
        out[v] = _run_speed(env)
    assert out[0.45] > out[0.25] and out[0.45] > out[0.65]


def test_run_upright_is_gpu_run_std_and_additive():
    """GPU run upright is std²=0.12 (lean is cheap) and is NOT multiplied
    onto keep_pace — that gate is why local speed saturated at 0.27 m/s."""
    from microduck_local.behaviors import _run_upright

    env = _env("run")
    env.reset(seed=1)
    pitch = np.deg2rad(10)
    env.data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    mujoco.mj_forward(env.model, env.data)
    u = _run_upright(env)
    assert u > 0.6, f"10 deg lean scored {u:.3f} — too tight for a run"
    assert _RUN_UPRIGHT_STD2 == pytest.approx(0.12)

    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = 0.4
    mujoco.mj_forward(env.model, env.data)
    v = env.body_lin_vel()
    err2 = float((0.4 - v[0]) ** 2 + v[1] ** 2 + v[2] ** 2)
    # One deliberate deviation from the raw GPU bell: the standstill baseline
    # is subtracted and renormalized so immobility pays 0 (a from-scratch
    # policy parked itself and collected rent at our sample budget — see
    # test_standing_still_pays_exactly_zero_pace). Perfect tracking still
    # scores what the normalized bell says it should.
    v = env.body_lin_vel()
    err2 = float((0.4 - v[0]) ** 2 + v[1] ** 2 + v[2] ** 2)
    raw = np.exp(-err2 / _RUN_TRACK_STD2)
    base = np.exp(-0.4 ** 2 / _RUN_TRACK_STD2)
    assert _run_speed(env) == pytest.approx((raw - base) / (1 - base), rel=1e-6)
    assert _run_speed(env) != pytest.approx(_run_speed(env) * u)


def test_torque_normalizer_follows_the_actuator_model():
    """The MJCF forcerange is a VOLTAGE ceiling and reads 0.96 N·m under both
    actuators, but the BAM model clamps to the XL330's firmware current limit
    (kt × 1.75 A = 0.640 N·m). Normalizing against the forcerange therefore
    under-charged motor strain by 2.25× on exactly the actuator that models
    the real robot."""
    from microduck_local.bam_actuator import XL330_MAX_TORQUE

    xml = _env("run")
    assert _tau2_max(xml) == pytest.approx(
        float(np.sum(xml.model.actuator_forcerange[:, 1] ** 2)))

    bam = BehaviorEnv("run", actuator="bam", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    bam.reset(seed=0)
    assert _tau2_max(bam) == pytest.approx(C.NUM_JOINTS * XL330_MAX_TORQUE ** 2)
    assert _tau2_max(bam) < _tau2_max(xml) / 2, "BAM must charge strain harder"


def test_slip_penalty_matches_gpu_from_first_contact():
    """GPU feet_slip charges any contacting foot. The local 'skip first
    contact' guard was for a 10×-too-heavy weight; at GPU's -0.1, skip is
    the wrong prior."""
    from microduck_local.behaviors import _run_slip_pen

    env = _env("run")
    env.reset(seed=0)
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    env.foot_contact_state["left"] = True
    env.foot_contact_state["right"] = True
    env.data.qvel[0:3] = (1.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    first = _run_slip_pen(env)
    assert first < 0.0, "GPU slip charges touchdown"
    env.twist_cmd[:] = 0.0
    assert _run_slip_pen(env) == 0.0, "slip is gated on a live command"


def test_action_rate_ramps_like_gpu_run():
    """GPU run action_rate_l2 is -0.1 → -0.5 over 3000 iters, as
    -weight × Σ(Δa²) with no extra 0.02 scale."""
    env = _env("run")
    env.reset(seed=0)
    env.last_action = np.full(C.NUM_JOINTS, 0.5, np.float32)
    env.prev_action = np.zeros(C.NUM_JOINTS, np.float32)
    da2 = float(np.sum((env.last_action - env.prev_action) ** 2))

    env._lifetime_steps = 0
    first = _run_action_rate_pen(env)
    assert first == pytest.approx(-0.1 * da2, rel=1e-6)

    env._lifetime_steps = int(3000 * 24 * _RUN_STAGE_SCALE)
    late = _run_action_rate_pen(env)
    assert late == pytest.approx(-0.5 * da2, rel=1e-6)
    assert _run_action_rate_weight(int(3000 * 24 * 50 * _RUN_STAGE_SCALE)) == pytest.approx(0.5)


def test_action_rate_ramp_survives_episode_resets():
    """The ramp tracks TRAINING progress, so it must not reset with each
    episode — otherwise it would sit at 1x forever."""
    env = _env("run")
    env.reset(seed=0)
    for _ in range(20):
        env.step(np.zeros(C.NUM_JOINTS, np.float32))
    before = env._lifetime_steps
    assert before > 0
    env.reset(seed=1)
    assert env._lifetime_steps >= before, "lifetime counter was reset"


def test_run_command_mix_has_forward_bucket_and_idle():
    """GPU run: 55% straight-forward (vx>=0.3, vy=wz=0), standing >0, rest
    omni. A uniform [0.15, 0.6] forward-only sampler never trained idle or
    turning."""
    env = _env("run")
    env._lifetime_steps = 0
    n = 400
    standing = forward = omni = 0
    vxs = []
    for i in range(n):
        env._sample_commands()
        c = env.twist_cmd
        vxs.append(float(c[0]))
        if np.allclose(c, 0.0):
            standing += 1
        elif abs(c[1]) < 1e-9 and abs(c[2]) < 1e-9 and c[0] > 0:
            forward += 1
            assert c[0] + 1e-9 >= 0.3
        else:
            omni += 1
    assert 0.01 * n < standing < 0.08 * n, standing
    assert 0.40 * n < forward < 0.70 * n, forward
    assert omni > 0.20 * n, omni
    assert max(vxs) <= _run_cmd_speed(0) + 1e-9
    # Ceiling opens after the gait exists, not from step 0.
    assert _run_cmd_speed(0) == pytest.approx(0.4)
    assert _run_cmd_speed(int(3000 * 24 * _RUN_STAGE_SCALE)) == pytest.approx(0.6)
    # Extended ceiling: 0.6 commanded caps achieved speed at ~0.4 (66%
    # delivery measured), so the goal speed needs headroom to be commandable.
    assert _run_cmd_speed(int(5000 * 24 * _RUN_STAGE_SCALE)) == pytest.approx(0.9)
    assert _run_cmd_speed(int(6000 * 24 * _RUN_STAGE_SCALE)) == pytest.approx(1.1)
    assert _run_standing_frac(0) == pytest.approx(0.02)
    assert _run_standing_frac(int(1500 * 24 * _RUN_STAGE_SCALE)) == pytest.approx(0.10)


def test_run_has_no_height_termination():
    """GPU locomotion terminates on tilt only. A 0.07 m z-kill cuts a
    bouncing stride short."""
    env = _env("run")
    assert env.height_termination is False


def test_pinned_run_cmd_overrides_the_mix(monkeypatch):
    env = _env("run")
    monkeypatch.setenv("MICRODUCK_RUN_CMD", "0.4")
    env._sample_commands()
    np.testing.assert_allclose(env.twist_cmd, (0.4, 0.0, 0.0), atol=1e-6)


def test_ramp_offset_survives_a_warm_restart(monkeypatch):
    """The lifetime ramp must resume at strength after a trainer restart.

    The counter lives on env objects, so every warm restart (the lab does one
    per helper add/remove) silently reset ramped penalties to stage-0: the
    policy loosened into jerk, then the ramp came back at full strength and
    crushed it — ep_len 396 (the session's best) collapsed to 10 and the peak
    policy was overwritten. MICRODUCK_RAMP_OFFSET, exported before the workers
    fork, seeds the counter with the run's prior step count.
    """
    monkeypatch.setenv("MICRODUCK_RAMP_OFFSET", "50000")
    env = _env("run")
    env.reset(seed=0)
    env.step(np.zeros(C.NUM_JOINTS, np.float32))
    assert env._lifetime_steps > 50000, "offset ignored — restart whiplash is back"

    monkeypatch.setenv("MICRODUCK_RAMP_OFFSET", "bogus")
    env2 = BehaviorEnv("run", obs_noise=False, domain_rand=False,
                       action_delay=False, random_yaw=False, seed=0)
    env2.reset(seed=0)
    env2.step(np.zeros(C.NUM_JOINTS, np.float32))
    assert 0 < env2._lifetime_steps < 100


def test_standing_still_pays_exactly_zero_pace():
    """A from-scratch policy parked itself and collected keep_pace rent at
    zero velocity (the raw bell pays ~0.3-0.5 for immobility at low commands,
    and standing also dodges every penalty). Immobility must earn exactly 0,
    while the gradient from v=0 toward the command must survive intact."""
    import mujoco
    env = _env("run")
    for cmd in (0.15, 0.3, 0.45, 0.8):
        env.reset(seed=1)
        env.twist_cmd[0] = cmd
        env.data.qvel[0:3] = (0.0, 0.0, 0.0)
        mujoco.mj_forward(env.model, env.data)
        assert _run_speed(env) == pytest.approx(0.0, abs=1e-9), f"paid at cmd {cmd}"
        # gradient: quarter-speed must beat standing, target must beat both
        scores = []
        for v in (cmd * 0.25, cmd * 0.6, cmd):
            env.data.qvel[0:3] = (v, 0.0, 0.0)
            mujoco.mj_forward(env.model, env.data)
            scores.append(_run_speed(env))
        assert 0 < scores[0] < scores[1] < scores[2], f"gradient broken at cmd {cmd}: {scores}"


def test_run_recipe_has_no_unobservable_terms():
    """The 61-obs contract has NO yaw signal, so any term conditioned on
    absolute heading or the spawn line is unlearnable noise — the duck
    circled for 30M+ steps under three variants of them. Straightness is the
    commander's job (heading-hold over trained yaw-rate tracking: 0.293 m/s,
    4 deg, 0/10 falls). This locks those terms OUT of the run recipe."""
    from microduck_local.behaviors import BEHAVIORS

    keys = {t.key for t in BEHAVIORS["run"].terms}
    assert "face_home" not in keys
    assert "hold_the_line" not in keys


def test_anchors_yield_to_steering_commands_and_reanchor():
    """Obeying a commanded turn must not be charged: unconditional anchors
    made tracking a turn net ~0 (face_home saturated in 0.63 s) where the GPU
    stack pays +1.73 — the same bug family as pay-to-stand, opposite sign.
    And after any resample the line must re-anchor HERE, or one obedient turn
    leaves ~15 s charged against a stale heading (measured ~-1500/episode)."""
    from microduck_local.behaviors import _face_home_pen, _run_lateral_pen

    env = _env("run")
    env.reset(seed=0)
    env.home_yaw = 0.0
    env.home_xy = (0.0, 0.0)
    # 90 deg off heading, 2 m off the line — worst case…
    yaw = np.pi / 2
    env.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.data.xpos[env.trunk_body_id][:2] = (0.0, 2.0)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env.data.xpos[env.trunk_body_id][:2] = (0.0, 2.0)
    # …charged in full while commanded STRAIGHT:
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    assert _face_home_pen(env) == pytest.approx(-1.0)
    assert _run_lateral_pen(env) == pytest.approx(-1.0)
    # …free while commanded to TURN or SIDESTEP (the commander owns heading):
    env.twist_cmd[:] = (0.3, 0.0, 1.0)
    assert _face_home_pen(env) == 0.0 and _run_lateral_pen(env) == 0.0
    env.twist_cmd[:] = (0.3, 0.3, 0.0)
    assert _face_home_pen(env) == 0.0 and _run_lateral_pen(env) == 0.0
    # …and a resample re-anchors to the current pose.
    env._sample_commands()
    assert env.home_yaw == pytest.approx(yaw, abs=0.05)
    assert env.home_xy[1] == pytest.approx(float(env.data.xpos[env.trunk_body_id][1]), abs=1e-6)


def test_standing_bucket_keeps_the_idle_signal():
    """cmd == 0 uses the raw GPU bell: stillness pays ~1, wandering decays it.
    The baseline subtraction (right for nonzero commands) zeroed this bucket
    entirely — no reward for stillness, no charge for wandering (audit #3)."""
    import mujoco
    env = _env("run")
    env.reset(seed=0)
    env.twist_cmd[:] = 0.0
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    still = _run_speed(env)
    env.data.qvel[0:3] = (0.3, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    wander = _run_speed(env)
    assert still > 0.9, f"stillness at cmd 0 pays {still:.3f}"
    assert wander < still * 0.6, f"wandering at cmd 0 pays {wander:.3f}"


def test_clearance_measures_height_above_ground_not_geom_center():
    """The foot geom CENTER is 0.0086 m when planted, so raw z silently cut
    the 3 cm swing target to ~2.1 cm of true lift (audit #5). Planted feet at
    stance height must contribute (near) zero height error."""
    from microduck_local.behaviors import _run_clearance_pen

    env = _env("run")
    env.reset(seed=0)
    for _ in range(20):  # settle onto both feet
        env.step(np.zeros(C.NUM_JOINTS, np.float32))
        if all(env.foot_contact_state.values()):
            break
    env.twist_cmd[:] = (0.4, 0.0, 0.0)
    assert hasattr(env, "_run_foot_z0") or _run_clearance_pen(env) is not None
    _run_clearance_pen(env)                      # triggers calibration
    z0 = env._run_foot_z0
    assert 0.004 < z0 < 0.02, f"stance calibration implausible: {z0}"
    # a planted foot's height-above-ground now reads ~0
    g = next(iter(env.foot_geoms.values()))
    assert abs(float(env.data.geom_xpos[g][2]) - z0) < 0.01
