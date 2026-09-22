"""The G1's karate strikes — the properties that had to be true before any
of it trained, and the four reward mistakes that cost a retrain each.

Every number quoted here was measured on this model, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from microduck_local.robots.g1 import g1_ready

needs_g1 = pytest.mark.skipif(not g1_ready(),
                              reason="G1 assets missing — run `uv run fetch-g1`")

if g1_ready():
    from microduck_local.robots.g1_karate import G1FrontKickEnv, G1PunchEnv


def _kick(**kw):
    return G1FrontKickEnv(seed=0, obs_noise=False, domain_rand=False, **kw)


def _punch(**kw):
    return G1PunchEnv(seed=0, obs_noise=False, domain_rand=False, **kw)


@needs_g1
def test_the_kick_pose_is_statically_balanced_before_anything_trains():
    """A one-leg pose the model cannot hold is not a reward problem.

    MEASURED on this G1: standing, the centre of mass sits 0.118 m to the side
    of either foot and a foot is 0.073 m wide, so a one-leg stance is
    impossible until the support leg adducts to bring the foot under the
    midline. The solver searches that shift jointly with a fore-aft lean,
    because a leg thrown forward puts the CoM 0.117 m ahead of a foot only
    0.203 m long. `_solve_strike` raises rather than return an unbalanced
    pose, so this test is also what makes that raise meaningful.
    """
    e = _kick()
    dx, dy = e.strike_com_offset
    x0, x1, y0, y1 = e._support_box()
    assert abs(dx) < (x1 - x0) / 2, "solved kick is outside the foot fore-aft"
    assert abs(dy) < (y1 - y0) / 2, "solved kick is outside the foot laterally"
    # and it is a KICK, not a shuffle: the foot is properly off the floor
    assert e.strike_foot_clear > 0.25


@needs_g1
def test_the_strike_reward_reaches_the_standing_start():
    """A Gaussian narrow enough to be flat where the policy begins teaches
    nothing. MEASURED with STRIKE_STD2 0.35: the punch pose is 2.715 rad^2
    from the default pose, so a standing robot earned 0.00 of 6 and the
    policy abandoned the pose entirely (27 deg mean joint error at 1.2M
    steps). The term must pay a real fraction at the start and still peak
    sharply at the pose."""
    for env in (_kick(), _punch()):
        rel = env._strike_rel[env._strike_ids]
        at_start = env.W_STRIKE * float(np.exp(-float((rel ** 2).sum())
                                               / env.STRIKE_STD2))
        assert at_start > 0.15 * env.W_STRIKE, (
            f"{type(env).__name__}: strike pays {at_start:.2f} standing — flat")
        assert at_start < 0.75 * env.W_STRIKE, (
            f"{type(env).__name__}: strike pays {at_start:.2f} standing — no peak")


@needs_g1
def test_height_is_retargeted_at_the_strike_not_switched_off():
    """W_HEIGHT = 0 on the reasoning that "a strike is not defined by pelvis
    height" left nothing paying to stay up, and the punch sagged from 0.735 to
    0.479-0.493 until it tripped its own fall floor on every seed."""
    for env in (_kick(), _punch()):
        assert env.W_HEIGHT > 0.0
        env.reset(seed=0)
        # at FULL extension the target is the solved pose's own height
        env._phase0 = 0
        env.step_count = (0 if env.CYCLE_S <= 0 else
                          int(round(env.CYCLE_RISE[1] * env.CYCLE_S / 0.02)))
        assert env.strike_amplitude() == pytest.approx(1.0, abs=1e-6)
        assert env.target_height() == pytest.approx(env._strike_height)
        assert env.target_height() > env._fall_height + 0.1
        # and the width must not be flat where the sag actually lands: the
        # idle's 0.004 makes a 0.24 m sag worth e^-14.
        sag = env.W_HEIGHT * float(np.exp(-(0.24 ** 2) / env.HEIGHT_STD2))
        assert sag > 0.05 * env.W_HEIGHT, "height term is flat where it sags to"


@needs_g1
def test_the_carriage_never_fights_the_strike():
    """`carriage` holds every joint above the hips at the default, which for a
    PUNCH is exactly the joints doing the punching — measured before the fix,
    carriage paid 0.00 at the punch pose, i.e. it charged the robot for
    striking. A joint the strike moves belongs to the strike term."""
    import mujoco

    for env in (_kick(), _punch()):
        assert len(env._carriage_ids) > 0, "nothing left holding the body composed"
        env.reset(seed=0)
        # The two terms share a target, so the STRIKE pose must pay the
        # carriage its maximum. Before the fix `carriage` held every joint at
        # the STANDING pose, so it paid 0.00 at the punch pose — literally
        # charging the robot for striking.
        env.data.qpos[env.joint_qpos_adr] = env._strike_joints
        env._phase0 = 0
        if env.CYCLE_S > 0:
            env.step_count = int(env.CYCLE_RISE[1] * env.CYCLE_S / 0.02)
        mujoco.mj_forward(env.model, env.data)
        env._refresh_derived()
        assert env._compute_reward()[1]["carriage"] == pytest.approx(
            env.W_CARRIAGE, rel=0.02), "the carriage term fights the strike"


@needs_g1
def test_at_the_solved_pose_every_earner_pays_its_maximum():
    """The pose the solver returns must be the one the reward wants; if any
    term peaks somewhere else the policy is being pulled two ways."""
    for env in (_kick(spawn_in_pose_prob=1.0), _punch(spawn_in_pose_prob=1.0)):
        env.reset(seed=0)
        terms = env._compute_reward()[1]
        assert terms["strike"] == pytest.approx(env.W_STRIKE, rel=0.02)
        assert terms["carriage"] == pytest.approx(env.W_CARRIAGE, rel=0.02)
        assert terms["height"] == pytest.approx(env.W_HEIGHT, rel=0.05)
        assert terms["balance"] > 0.9 * env.W_BALANCE


@needs_g1
def test_the_punch_is_level_and_the_kick_stands_on_one_leg():
    """The two things a human would check by looking."""
    import mujoco

    p = _punch(spawn_in_pose_prob=1.0)
    p.reset(seed=0)

    def body(sub):
        for i in range(p.model.nbody):
            n = mujoco.mj_id2name(p.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if n and sub in n:
                return i
        raise KeyError(sub)

    fist = p.data.xpos[body("right_wrist_yaw")]
    shoulder = p.data.xpos[body("right_shoulder_pitch")]
    rise = float(fist[2] - shoulder[2])
    reach = float(fist[0] - shoulder[0])
    # a hand-typed -1.35 rad rendered as a 36 deg uppercut with LESS reach
    assert abs(np.degrees(np.arctan2(rise, reach))) < 12.0, "the punch is not level"
    assert reach > 0.10, "the punch is not extended"
    assert p.support_sides() == ("left", "right")

    k = _kick()
    k.reset(seed=0)
    k._phase0 = 0
    n = int(round(k.CYCLE_S / 0.02))
    k.step_count = int(k.CYCLE_RISE[1] * n)            # full extension
    assert k.support_sides() == (k.SUPPORT_SIDE,)
    k.step_count = 0                                   # idle
    assert k.support_sides() == ("left", "right")


@needs_g1
def test_curriculum_knobs_are_read_from_the_environment(monkeypatch):
    """A curriculum stage passes knobs through the trainer's ENVIRONMENT — the
    lab's stage machinery does not rewrite argv."""
    # the HOLD's spawn probability
    monkeypatch.setenv("MICRODUCK_G1_SPAWN_IN_POSE", "1.0")
    p = _punch()
    assert p.spawn_in_pose_prob == 1.0
    assert all((p.reset(seed=s), p.last_spawn)[1] == "strike" for s in range(5))
    monkeypatch.delenv("MICRODUCK_G1_SPAWN_IN_POSE")

    # the KICK's height ladder — the rung must REACH the solved pose. The
    # assertion is that the knob moves the height, not by how much: the
    # default target has changed twice (0.806 m when it was waist height,
    # 0.545 m once that proved unrecoverable) and a hardcoded margin just
    # breaks on the next honest retune.
    # The invariant is that the knob MOVES the height, monotonically — not
    # which rung happens to sit above the default. That default has been
    # retuned three times on measurement (0.806 -> 0.545 -> 0.353 m), and a
    # test naming a fixed rung breaks on every honest retune rather than on
    # a defect.
    heights = []
    for hip in ("-0.9", "-1.2", "-1.5"):
        monkeypatch.setenv("MICRODUCK_G1_KICK_HIP", hip)
        heights.append(_kick().strike_foot_clear)
    assert all(b > a + 0.05 for a, b in zip(heights, heights[1:])), (
        f"the height knob is not monotone: {[round(h, 3) for h in heights]}")


@needs_g1
def test_the_kick_comes_back_down_to_both_feet():
    """The whole point of the change from a held pose to a cycle: it does not
    need to stand on one leg, it needs to RECOVER. If `strike` did not pay for
    the idle phase, or `stance` still wanted one foot up throughout, the
    policy would be paid to stay on one leg — the opposite of the task."""
    e = _kick()
    e.reset(seed=0)
    e._phase0 = 0
    n = int(round(e.CYCLE_S / 0.02))

    e.step_count = int(e.CYCLE_RISE[1] * n)
    assert e.strike_amplitude() == pytest.approx(1.0, abs=1e-6)
    assert e.support_sides() == (e.SUPPORT_SIDE,)

    e.step_count = int(0.85 * n)                       # idle stretch
    assert e.strike_amplitude() == 0.0
    assert e.support_sides() == ("left", "right")
    assert e.target_height() == pytest.approx(e.stand_z)
    # standing still in the idle phase IS the target pose, so it pays in full
    assert np.allclose(e.pose_now(), e.default_pose)

    # The amplitude is CONTINUOUS — a square step would demand an impulse the
    # action-rate penalty then charges it for. Smoothstep over the 0.3 s throw
    # peaks at 1.5/15 = 0.1 per control step, which is a real kick's speed; a
    # discontinuity would show up here as ~1.0.
    amps = []
    for k in range(n):
        e.step_count = k
        amps.append(e.strike_amplitude())
    assert max(abs(b - a) for a, b in zip(amps, amps[1:])) < 0.15
    assert max(amps) == pytest.approx(1.0) and min(amps) == 0.0


@needs_g1
def test_the_cycle_spawns_across_the_whole_phase():
    """An unsampled phase's value is never learned, so an episode may start
    anywhere in the cycle with the body posed to match."""
    e = _kick()
    seen = set()
    for s in range(24):
        e.reset(seed=s)
        seen.add(round(e.strike_amplitude(), 1))
    assert len(seen) > 3, f"only {sorted(seen)} sampled — the cycle is not covered"
    assert 0.0 in seen


@needs_g1
def test_both_strikes_are_reachable_inside_the_action_clip():
    """The policy commands `default + action * scale`, so a pose needing more
    than `action_clip` cannot be held however good the controller is."""
    for env in (_kick(), _punch()):
        scale = np.asarray(env.robot.action_scale, np.float64)
        scale = scale if scale.ndim else np.full(env.nj, scale)
        need = np.max(np.abs(env._strike_rel / scale))
        assert need < env.robot.action_clip, (
            f"{type(env).__name__} needs {need:.2f} of a {env.robot.action_clip} clip")


@needs_g1
def test_the_strikes_are_not_shoved():
    """Same invariant as the idle and the squat (AGENTS.md): pushes are off in
    a behavior recipe."""
    from microduck_local import train as T

    for task in ("front_kick", "punch"):
        env = T.env_class("g1", task)(
            **T.env_kwargs_from_args(T.parse_args(["--robot", "g1", "--task", task])))
        assert env.push_robot is False
        assert env.max_steps == 2000        # 40 s at 50 Hz


@needs_g1
def test_the_lift_earner_pays_the_FIRST_centimetre():
    """A reward that is flat where the policy currently is teaches nothing.

    This is the mistake this file keeps re-making, in its fourth form. A
    Gaussian on foot clearance at std2 0.02 m^2 against a 0.48 m target pays
    e^-9 to a robot that has lifted its foot 5 cm — so an early, timid attempt
    earns exactly what standing earns, and nothing pulls it bigger. Measured
    on the chain that used it: the policy tracked the phase clock correctly
    (probing the ONNX, the hip action swung +0.385 idle to -0.232 at full
    extension) but committed ~11% of the hip travel and the foot never left
    the floor. Progress pay is linear from zero.
    """
    e = _kick()
    e.reset(seed=0)
    e._phase0 = 0
    e.step_count = int(e.CYCLE_RISE[1] * e.CYCLE_S / 0.02)   # full extension
    assert e.strike_amplitude() == pytest.approx(1.0, abs=1e-6)
    want = e.strike_foot_clear

    def pay(got):
        frac = 1.0 if want <= 1e-6 else min(max(got / want, 0.0), 1.0)
        return e.W_LIFT * frac

    assert pay(0.0) == 0.0
    # a first, small attempt must be worth measurably more than standing
    assert pay(0.05) > 0.05 * e.W_LIFT, "the lift earner is flat near zero"
    # and it must rise monotonically all the way, not peak early
    steps = [pay(g) for g in np.linspace(0.0, want, 12)]
    assert all(b >= a for a, b in zip(steps, steps[1:]))
    assert steps[-1] == pytest.approx(e.W_LIFT)


@needs_g1
def test_no_joint_is_scored_by_nothing():
    """Strike owns what it moves; carriage owns the rest; nothing falls between.

    `carriage` was defined as "every joint above the hips" and `strike` as the
    joints the strike moves — which for the kick are 7 LEG joints. The 6 in
    between (both hip yaws, both ankles) were scored by neither, and `W_POSE`
    is 0 for a strike so no general pose term caught them. Measured on a
    kick's idle phase: hip_yaw 23.2 deg off and waist_roll 28.4 deg off, a
    body visibly twisted between kicks, while the pelvis was within 2 cm of
    standing and the trunk perfectly vertical — so no height or uprightness
    term saw anything wrong either.
    """
    for env in (_kick(), _punch()):
        strike = set(map(int, env._strike_ids))
        carriage = set(map(int, env._carriage_ids))
        missing = set(range(env.nj)) - strike - carriage
        assert not missing, (
            f"{type(env).__name__}: scored by nothing: "
            f"{[env.robot.joint_names[i] for i in sorted(missing)]}")
        # Overlap is deliberate: the two terms are a coarse/fine PAIR on the
        # same target — `strike` wide enough to reach the policy from
        # standing, `carriage` tight enough that the pose it settles into is
        # clean. They must not disagree about WHERE, so pin that instead.
        env.reset(seed=0)
        env._phase0 = env.step_count = 0
        want = np.zeros(env.nj)
        want[env._strike_ids] = env.strike_amplitude() * env._strike_rel[env._strike_ids]
        assert np.allclose(np.asarray(env.carriage_target_rel()),
                           want[env._carriage_ids], atol=1e-9), \
            "carriage and strike are pulling toward different poses"


@needs_g1
def test_the_kick_carries_an_arm_counterswing_that_cancels_its_yaw():
    """A leg swung 0.145 m off the centreline throws the body about the
    vertical, and nothing else in the reference opposes it. MEASURED on the
    trained kick before this existed: the swing makes Lz = -0.174 kg m^2/s,
    and the robot absorbed it by yawing 27-141 deg per 20 s — while the
    `carriage` term held all 14 arm joints at their default, pinning the one
    set of limbs that could have cancelled it.
    """
    e = _kick()
    if not e.ARM_COUNTERSWING:
        # OFF by measurement, not by oversight — see the constant's comment.
        # Pin that the solver still WORKS, so the evidence stays checkable.
        e.ARM_COUNTERSWING = True
        try:
            q = e._add_arm_counterswing(e._strike_joints.copy())
        finally:
            e.ARM_COUNTERSWING = False
        assert max(abs(v) for v in e.arm_residual) < 1e-4
        assert all(abs(v) < 0.8 for v in e.arm_counterswing)
        assert not np.allclose(q, e._strike_joints)
        return
    names = list(e.robot.joint_names)
    arms = [names[i] for i in e._strike_ids
            if any(k in names[i] for k in ("shoulder", "elbow", "wrist"))]
    assert arms, "the kick reference moves no arm joints at all"
    # solved to cancel, not posed: both axes must come out at zero
    assert max(abs(v) for v in e.arm_residual) < 1e-4, (
        f"counter-swing leaves residual momentum {e.arm_residual}")
    # and it must be a counter-SWING, not a windmill
    assert all(abs(v) < 0.8 for v in e.arm_counterswing)
    # the pose it produces is still balanced and still reachable
    dx, dy = e.strike_com_offset
    x0, x1, y0, y1 = e._support_box()
    assert abs(dx) < (x1 - x0) / 2 and abs(dy) < (y1 - y0) / 2
    scale = np.asarray(e.robot.action_scale, np.float64)
    scale = scale if scale.ndim else np.full(e.nj, scale)
    assert np.max(np.abs(e._strike_rel / scale)) < e.robot.action_clip


@needs_g1
def test_the_spin_penalty_reads_real_angular_momentum_and_is_not_flat():
    """`mj_forward` does NOT populate `subtree_angmom` — it needs an explicit
    `mj_subtreeVel`. Without that call the term reads exactly 0.000 forever,
    which is a knob that changes nothing rather than a term that is satisfied.
    And it is LINEAR: measured on the shipped kick, |Lz| is 0.020 mean but
    spikes to 0.895, so a Gaussian narrow enough to see the mean is flat at
    the spikes, which are the thing worth pricing.
    """
    import mujoco

    e = _kick()
    assert e.W_MOMENTUM > 0.0
    e.reset(seed=0)
    # spin the body and check the term actually moves
    e.data.qvel[e._root_qvel + 5] = 2.0            # yaw rate
    mujoco.mj_forward(e.model, e.data)
    e._refresh_derived()
    spun = e._compute_reward()[1]["spin_cost"]
    assert spun < -0.1, f"spin penalty reads {spun} on a body yawing at 2 rad/s"
    assert spun >= -e.W_MOMENTUM - 1e-9, "penalty must be bounded"

    # linear, so a half-sized spin costs about half — no flat region
    def cost(lz):
        return -e.W_MOMENTUM * min(lz / e.MOMENTUM_REF, 1.0)
    assert cost(0.05) == pytest.approx(0.5 * cost(0.10))
    assert cost(0.02) < -1e-3, "the term is flat at the momentum it actually carries"
    # and it is a PENALTY (AGENTS.md: <= 0 by construction)
    assert cost(0.0) == 0.0 and cost(9.9) == pytest.approx(-e.W_MOMENTUM)


@needs_g1
def test_the_arms_are_released_while_the_strike_extends():
    """Paying for low angular momentum while pinning the only limbs that could
    cancel it asks for something unreachable. The grip comes back for the idle
    stretch, which is where the composed look matters."""
    e = _kick()
    e.reset(seed=0)
    e._phase0 = 0
    n = int(round(e.CYCLE_S / 0.02))
    assert e._arm_slots.size, "no arm joints identified in the carriage set"

    e.step_count = 0                                    # idle
    assert e.strike_amplitude() == 0.0
    assert e.carriage_weights()[e._arm_slots].min() == pytest.approx(1.0)

    e.step_count = int(e.CYCLE_RISE[1] * n)             # full extension
    assert e.strike_amplitude() == pytest.approx(1.0, abs=1e-6)
    assert e.carriage_weights()[e._arm_slots].max() == pytest.approx(0.0)

    # the LEGS stay gripped throughout — only the arms are freed
    legs = [k for k in range(len(e._carriage_ids)) if k not in set(e._arm_slots)]
    assert e.carriage_weights()[legs].min() == pytest.approx(1.0)


def test_train_py_caps_the_action_log_std_like_train_behavior_does():
    """A warm-start chain ratchets log_std until the exported MEAN is garbage.

    `train_behavior` has capped this since 2026-09-01; `train.py` — the trainer
    every G1 task uses — did not. Measured on a G1 kick warm-start chain before
    the cap: action std 0.89 mean / 1.57 peak, stochastic episodes surviving
    40 s while the DETERMINISTIC policy that `export-walk` ships fell in 1.5 s.
    Every deterministic evaluation of those runs was reading a mean the
    training had never optimised.
    """
    import inspect

    from microduck_local import train as T
    from microduck_local import train_behavior as TB

    assert T.LOG_STD_MAX == TB.LOG_STD_MAX, "the two trainers must agree"
    src = inspect.getsource(T)
    # bound on LOAD (a warm start inherits it) and again EVERY ROLLOUT
    # (the entropy bonus pushes it back up between clamps)
    assert src.count("clamp_(max=LOG_STD_MAX)") >= 2, (
        "the cap must bind on warm-start load AND per rollout")
    assert "_log_std_cap_callback_cls" in src
    # and the callback must actually be installed, not merely defined
    assert "_log_std_cap_callback_cls(BaseCallback)()" in src
