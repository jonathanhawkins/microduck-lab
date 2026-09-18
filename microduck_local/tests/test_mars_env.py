"""Training MARS's arm: the contract it fills, the physics, the CLI seam.

Phase 4a of `docs/mars-roadmap.md`. The load-bearing tests here are the ones
that read the OBSERVATION back off live state and the ones that pin the
physics constants the reward rests on, because both are places this repo has
been fooled before:

  * a slot table that tiles perfectly can still have two same-width slots
    swapped, and only an env can tell them apart
    (`tests/test_policy_contract.py` learned that the hard way), so
    `arm_qpos`, `last_action` and `target_base` are each compared against the
    thing that produced them rather than against a literal;
  * every constant in `mars_env` was chosen from a measurement, and a test
    that fixes a number it does not NAME is a hostage rather than a
    characterisation (`AGENTS.md`, the camera-geometry lesson). So the
    self-collision depth, the target rate limit and the action scale are
    asserted with their own module constants in the expression, and what is
    pinned against literals is the MEASURED behaviour they produce.

Every positive case below was shown to fail on a planted break; the table is
in the phase report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from microduck_local import train as T
from microduck_local.robots import mars
from microduck_local.robots import mars_env as ME
from microduck_local.robots.mars_env import MarsArmEnv

pytestmark = pytest.mark.skipif(
    not mars.mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")

#: A constant action that drives the arm into its own HEAD, found by search
#: and pinned: `link5 <-> head` at control step 6 of an 8 s episode. Named so
#: that a URDF revision which moves the head fails HERE, with a sentence, and
#: not as a mysteriously surviving episode somewhere downstream.
INTO_THE_HEAD = (0.73, 0.08, -0.40, -0.15, -0.94, -0.75)
#: ...and one that is safe for the whole episode and moves the gripper 19.5 cm
#: from HOME. MEASURED: the effector settles at (0.2343, -0.0146, 0.1782) and
#: has drifted 0.0 mm two seconds later, which is what makes it usable as a
#: "the servo reaches the commanded pose" fixture.
SAFE_AND_MOVING = (-0.40, 0.34, -0.60, 0.88, -0.27, -0.79)
SAFE_EE = np.array([0.2343, -0.0146, 0.1782])


def _env(**kw) -> MarsArmEnv:
    kw.setdefault("seed", 0)
    return MarsArmEnv(**kw)


def _hold(env, arm, steps: int, base=(0.0, 0.0)):
    """Hold one action for `steps` control steps. Returns the last info."""
    a = np.zeros(mars.NUM_ACTIONS, np.float32)
    a[mars.ACT_ARM] = arm
    a[mars.ACT_BASE_TWIST] = base
    info: dict = {}
    for _ in range(steps):
        _obs, _rew, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return info


# ---------------------------------------------------------------- the contract

def test_the_observation_tiles_the_published_contract():
    """The env fills `mars-arm-32-v1`, slot for slot.

    Not "the widths add up" — the NAMES and the offsets, against
    `MarsBody.contract()`, which is what every reader of an exported policy
    resolves. A builder with its own layout is a policy whose floats mean
    something other than what its own file says.
    """
    env = _env()
    contract = mars.MARS.contract()
    assert contract.id == "mars-arm-32-v1"
    assert contract.tiles() == ()
    assert [(s.name, s.start, s.stop) for s in contract.slots] == \
           [(n, s.start, s.stop) for n, s in env._SLOTS]
    assert env.observation_space.shape == (contract.obs_dim,) == (32,)
    assert env.action_space.shape == (contract.act_dim,) == (8,)
    # ...and the RATE is part of the contract: a policy trained at 25 Hz and
    # run at 50 is a different controller.
    assert contract.rate_hz == pytest.approx(1.0 / env.dt) == 25.0
    assert ME.DECIMATION == 8


def test_a_layout_that_disagrees_with_the_contract_is_a_construction_error(
        monkeypatch):
    """The check above runs in `__init__`, not only in this file.

    `monkeypatch`, not a try/finally: the first draft restored the attribute
    from `env._SLOTS`, which by then RESOLVED TO THE PLANTED VALUE, and left
    every later test in the process building a broken env.
    """
    original = MarsArmEnv._SLOTS
    bad = tuple(original[:-1])            # drop `reserved`: a 4-float hole
    monkeypatch.setattr(MarsArmEnv, "_SLOTS", bad)
    with pytest.raises(ValueError, match="may not have its own layout"):
        MarsArmEnv(seed=0)
    monkeypatch.undo()
    assert MarsArmEnv._SLOTS is original
    MarsArmEnv(seed=0)                    # and it builds again


def test_arm_qpos_and_qvel_are_read_off_the_live_state():
    env = _env()
    _hold(env, SAFE_AND_MOVING, 30)
    obs = env._get_obs()
    qadr = [env.model.joint(j).qposadr[0] for j in mars.ARM_JOINTS]
    dadr = [env.model.joint(j).dofadr[0] for j in mars.ARM_JOINTS]
    assert np.allclose(obs[mars.OBS_ARM_QPOS], env.data.qpos[qadr], atol=1e-6)
    assert np.allclose(obs[mars.OBS_ARM_QVEL], env.data.qvel[dadr], atol=1e-6)
    # ABSOLUTE radians, not relative to HOME — the contract says so, and the
    # difference is invisible in a tiling check.
    assert not np.allclose(obs[mars.OBS_ARM_QPOS], 0.0, atol=0.2)
    assert obs[mars.OBS_HEAD_PITCH][0] == pytest.approx(
        float(env.data.qpos[env._head_qadr]), abs=1e-6)


def test_last_action_echoes_the_previous_action():
    """All eight slots, including the base pair `reach` forces to zero."""
    env = _env()
    asked = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], np.float32)
    obs, _r, _t, _tr, _i = env.step(asked)
    assert np.allclose(obs[mars.OBS_LAST_ACTION], asked, atol=1e-6)
    again = np.zeros(8, np.float32)
    obs, _r, _t, _tr, _i = env.step(again)
    assert np.allclose(obs[mars.OBS_LAST_ACTION], again, atol=1e-6)
    assert np.allclose(env.prev_action, asked, atol=1e-6)


def test_the_observed_last_action_is_CLIPPED_to_the_box():
    """The bug that cost Phase 4a its first eval.

    SB3 clips a Box action before the env sees it, so in training the raw and
    the clipped value coincide and this slot is unambiguous. At inference
    nothing clips: the first exported MARS policy's mean reached |a| = 90 in
    a +-1 box, and a probe that handed the raw output over wrote 86.6 into
    this slot — a float the policy had never seen — scoring the same policy
    at 0.209 m instead of 0.026 m. The limit has to come from ONE place, and
    it is `action_space`.

    The action-rate PENALTY still prices the raw output, which is walk_env's
    rule: growth outside the box must not be free.
    """
    env = _env()
    wild = np.full(8, 90.0, np.float32)
    obs, _r, _t, _tr, info = env.step(wild)
    assert np.all(np.abs(obs[mars.OBS_LAST_ACTION]) <= ME.ACTION_CLIP + 1e-6)
    assert np.allclose(obs[mars.OBS_LAST_ACTION], ME.ACTION_CLIP)
    # ...and the penalties saw the real magnitude
    assert np.allclose(env.last_action, wild)
    assert info["terms"]["action_mag_penalty"] == pytest.approx(
        -ME.ACTION_MAG_W * 8 * 90.0 ** 2)
    assert info["terms"]["action_rate_penalty"] < -1.0


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.9, math.pi])
def test_target_base_is_the_target_rotated_into_the_base_frame(yaw):
    """Spawn the base facing somewhere and the observation is unchanged.

    The target is DRAWN in the base frame and pushed out to the world, so a
    MARS spawned at any heading sees the same task — and the slot is the
    inverse transform, computed against `MarsDriver.pose`'s own (x, y, yaw)
    rather than re-derived from a quaternion.
    """
    env = _env(spawn_yaw=yaw)
    obs, _ = env.reset(seed=7)
    x, y, base_yaw = env.driver.pose(env.data)
    assert base_yaw == pytest.approx(yaw, abs=1e-9)
    # the slot IS the sampled base-frame point, to the float32 round trip
    assert np.allclose(obs[mars.OBS_TARGET_BASE], env.target_base_sample,
                       atol=1e-6)
    # ...and that point, rotated out by hand, is where the world target is
    cos, sin = math.cos(yaw), math.sin(yaw)
    p = env.target_base_sample
    want = np.array([x + p[0] * cos - p[1] * sin,
                     y + p[0] * sin + p[1] * cos, p[2]])
    assert np.allclose(env.target, want, atol=1e-9)
    # A DIFFERENT yaw must move the world target but not the observation.
    other = _env(spawn_yaw=yaw + 1.0)
    obs2, _ = other.reset(seed=7)
    assert np.allclose(obs2[mars.OBS_TARGET_BASE],
                       obs[mars.OBS_TARGET_BASE], atol=1e-6)
    assert not np.allclose(other.target, env.target, atol=1e-3)


def test_target_seen_is_privileged_and_the_reserved_slot_is_zeros():
    env = _env()
    obs, _ = env.reset(seed=3)
    assert obs[mars.OBS_TARGET_SEEN][0] == 1.0
    assert np.all(obs[mars.OBS_RESERVED] == 0.0)
    obs = _hold(env, SAFE_AND_MOVING, 20) and env._get_obs()
    assert obs[mars.OBS_TARGET_SEEN][0] == 1.0, "reach's target is privileged"
    assert np.all(obs[mars.OBS_RESERVED] == 0.0)


def test_base_twist_is_the_measured_twist_not_a_command():
    """With the base ENABLED, driving shows up in the slot — which is what
    makes it a measurement.

    Finding a case that could TELL THE TWO APART took two attempts, which is
    the point worth recording. Innate's forward gain is deadbeat at this
    timestep — MEASURED, the base reads 0.4799 of a 0.4800 command after a
    single control step — so in ordinary driving the command and the
    measurement agree to four decimals and a slot echoing `driver.cmd()`
    passes any assertion about either. Even the first step is settled.

    What separates them is motion NOBODY ASKED FOR, which is also the case a
    policy most needs this slot for: the base shoved while `reach` has the
    drive switched off. Then `cmd()` is (0, 0) and the measurement is not.
    """
    # The discriminating case: no command, real motion.
    env = _env()                            # reach -> use_base False
    env.reset(seed=0)
    assert env.driver.cmd() == (0.0, 0.0)
    env.data.qvel[env.driver.base_dadr[0]] = 0.7      # shoved along +x
    env.data.qvel[env.driver.base_dadr[2]] = -0.4     # ...and spun
    obs = env._get_obs()
    v_fwd, wz = obs[mars.OBS_BASE_TWIST]
    assert v_fwd == pytest.approx(0.7, abs=1e-6), (
        f"the base is moving at 0.7 m/s and the slot says {v_fwd} — it is "
        "echoing the command, not measuring the body")
    assert wz == pytest.approx(-0.4, abs=1e-6)
    assert (v_fwd, wz) != env.driver.cmd()

    # ...and with the base ENABLED, driving shows up in the slot too.
    live = _env(use_base=True)
    live.reset(seed=0)
    a = np.zeros(8, np.float32)
    a[6] = 0.6                              # 0.6 x MAX_CMD_LINEAR = 0.48 m/s
    for _ in range(25):                     # 1 s
        obs, _r, _t, _tr, _i = live.step(a)
    v_fwd, wz = obs[mars.OBS_BASE_TWIST]
    assert v_fwd > 0.2, f"the base is not moving: {v_fwd}"
    assert abs(wz) < 0.2
    measured = live.driver.velocity(live.data)
    assert v_fwd == pytest.approx(measured[0], abs=1e-5)
    assert wz == pytest.approx(measured[2], abs=1e-5)


# ------------------------------------------------------------------ the arm

def test_a_constant_action_drives_the_gripper_to_the_pose_it_names():
    """The servo works THROUGH the env: hold one action and the effector
    arrives where that action's joint targets put it.

    The target is the env's own measurement of where the action lands, so this
    is a closed loop on the whole chain — action -> scale -> clip -> rate
    limit -> `MarsDriver.set_arm` -> `mars.arm_servo` -> physics -> `ee_link`.
    """
    env = _env()
    env.reset(seed=0)
    env.target = SAFE_EE.copy()             # a reachable point, 19.5 cm out
    env._prev_dist = env.distance()
    info = _hold(env, SAFE_AND_MOVING, 50)  # 2 s
    assert info["self_collision"] is None
    assert env.distance() < 0.01, f"settled {env.distance() * 100:.1f} cm away"
    assert np.allclose(env.effector(), SAFE_EE, atol=0.01)


def test_the_commanded_target_is_rate_limited_to_what_the_servos_have():
    """One control step may move a joint target `MAX_TARGET_RATE_RAD_S * dt`.

    This is the physics-ladder rung the phase turns on (60 of 60 random
    episodes self-collided inside 1-5 steps without it), so it gets a test
    that names the constant rather than the number it happens to produce.
    """
    env = _env()
    env.reset(seed=0)
    per_step = ME.MAX_TARGET_RATE_RAD_S * ME.CTRL_DT
    assert per_step == pytest.approx(0.24)
    before = env._cmd_target.copy()
    env.step(np.full(8, -1.0, np.float32))   # ask for the far corner
    moved = np.abs(env._cmd_target - before)
    assert moved.max() <= per_step + 1e-9
    assert moved.max() == pytest.approx(per_step, rel=1e-6), \
        "a full-scale action should saturate the limit"
    # ...and it keeps walking: 20 steps of the same ask move 20x as far.
    for _ in range(19):
        env.step(np.full(8, -1.0, np.float32))
    assert np.abs(env._cmd_target - before).max() > 10 * per_step


def test_the_action_scale_and_clip_span_the_joint_ranges():
    """A full-scale action lands on each joint's own stop, not past it.

    `ACTION_SCALE_RAD` was chosen because a shell solution costs up to
    3.09 rad on its worst joint, so the assertion is that +-1 covers most of
    each joint's one-sided headroom AND that the clip keeps it legal.
    """
    env = _env()
    lo, hi = env._jnt_lo, env._jnt_hi
    home = env._home
    for sign in (+1.0, -1.0):
        want = np.clip(home + sign * ME.ACTION_SCALE_RAD, lo, hi)
        assert np.all(want >= lo - 1e-12) and np.all(want <= hi + 1e-12)
    headroom = np.maximum(home - lo, hi - home)
    assert ME.ACTION_SCALE_RAD / headroom.min() > 3.0     # joint6, saturated
    # every joint's WIDER side is at least 78% covered by a full action
    assert (ME.ACTION_SCALE_RAD / headroom).min() > 0.78


# --------------------------------------------------------- the base is off

def test_the_base_action_is_ignored_for_reach():
    """`reach` is an arm task, and rolling the robot at the target would be
    the cheapest way to satisfy the score. So the pair is read, forced to
    zero, and `info` SAYS so — a knob that is silently discarded looks
    exactly like one that works (AGENTS.md rule 0)."""
    env = _env()
    env.reset(seed=0)
    assert env.use_base is False
    start = env.driver.pose(env.data)
    info = _hold(env, (0.0,) * 6, 50, base=(1.0, 1.0))   # full-ahead, full-turn
    assert info["base_enabled"] is False
    end = env.driver.pose(env.data)
    assert abs(end[0] - start[0]) < 0.002
    assert abs(end[1] - start[1]) < 0.002
    assert abs(end[2] - start[2]) < 0.01
    # ...and the same env with the base enabled DOES move, so the assertion
    # above is about the gate and not about a base that cannot drive.
    live = _env(use_base=True)
    live.reset(seed=0)
    _hold(live, (0.0,) * 6, 50, base=(1.0, 0.0))
    assert live.driver.pose(live.data)[0] > 0.2


# --------------------------------------------------------- self-collision

def test_home_is_clear_of_self_collision():
    """The terminal must not fire where every episode starts — AGENTS.md's
    "check the term is not FLAT where the policy starts", asked of a
    termination. MEASURED: the only contacts at HOME are floor/chassis and
    the two wheels."""
    env = _env()
    env.reset(seed=0)
    assert env.self_collision() is None
    info = _hold(env, (0.0,) * 6, 200)
    assert info["self_collision"] is None
    assert env.step_count == env.max_steps, "the parked arm ended early"


def test_driving_the_arm_into_the_head_terminates():
    env = _env()
    env.reset(seed=0)
    info = _hold(env, INTO_THE_HEAD, 200)
    assert info["self_collision"] is not None
    assert set(info["self_collision"]) == {"link5", "head"}
    assert env.step_count < 20, f"took {env.step_count} steps to notice"
    assert info["terms"]["self_collision_penalty"] == ME.SELF_COLLISION_PENALTY
    assert info["episode_rewards"]["self_collision_penalty"] < 0


def test_a_shallow_box_overlap_is_not_a_collision():
    """The DEPTH threshold is what does the work, and this is the case that
    proves it rather than a case that could not fail.

    MEASURED: within +-0.1 rad of HOME, 15.3% of poses report a contact and
    none of them is deeper than 5 mm — they are `link1 <-> link3` and
    `link2 <-> link4`, links two apart in a chain folded tight at HOME, in a
    URDF whose boxes are documented to overlap ~9 mm (`JOINT2_GUARD_MIN`). So
    a pose that CONTACTS but does not overlap past the threshold must survive,
    and the same pose must terminate if the threshold is dropped to zero.
    """
    import mujoco

    env = _env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    shallow = None
    for _ in range(600):
        env.reset(seed=0)
        q = env._home + rng.uniform(-0.1, 0.1, 6)
        env.data.qpos[env._arm_qadr] = np.clip(q, env._jnt_lo, env._jnt_hi)
        mujoco.mj_forward(env.model, env.data)
        depths = [float(env.data.contact[i].dist)
                  for i in range(env.data.ncon)
                  if int(env.model.geom_bodyid[env.data.contact[i].geom1])
                  != int(env.model.geom_bodyid[env.data.contact[i].geom2])
                  and 0 not in (int(env.model.geom_bodyid[env.data.contact[i].geom1]),
                                int(env.model.geom_bodyid[env.data.contact[i].geom2]))]
        if depths and min(depths) > -ME.SELF_COLLISION_DEPTH_M:
            shallow = (q, min(depths))
            break
    assert shallow is not None, "no shallow robot-robot contact found near HOME"
    q, depth = shallow
    assert -ME.SELF_COLLISION_DEPTH_M < depth < 0.0
    assert env.self_collision() is None, \
        f"a {-depth * 1000:.1f} mm box overlap counted as the arm hitting itself"
    # the SAME pose, with no tolerance: now it is a collision
    keep = ME.SELF_COLLISION_DEPTH_M
    try:
        ME.SELF_COLLISION_DEPTH_M = 0.0
        assert env.self_collision() is not None
    finally:
        ME.SELF_COLLISION_DEPTH_M = keep


def test_the_finger_pair_is_excluded_by_the_MODEL_not_by_the_env():
    """Which of the two guards is doing the work.

    `mars.tune_contacts` calls `add_exclude` on the finger pair (their hub
    pins overlap ~1 mm at joint6 = 0, and `arm.srdf` disables the same pair
    for MoveIt); `self_collision` also skips it explicitly. A guard that is
    never exercised is a guard nobody can tell is broken, so: close the claw
    hard and assert the model produces NO such contact at all.
    """
    env = _env()
    env.reset(seed=0)
    a = np.zeros(8, np.float32)
    a[5] = -1.0                              # close, hard, into the hard stop
    for _ in range(50):
        env.step(a)
    fingers = env._finger_bodies
    pairs = [{int(env.model.geom_bodyid[env.data.contact[i].geom1]),
              int(env.model.geom_bodyid[env.data.contact[i].geom2])}
             for i in range(env.data.ncon)]
    assert fingers not in pairs, "the model's add_exclude is gone"
    assert len(fingers) == 2


# ---------------------------------------------------------------- the reward

def test_progress_pays_for_closing_and_nothing_for_parking():
    """The task term is a CHANGE in distance, so a stalled arm earns zero and
    an episode's total is only how much ground it made up."""
    env = _env()
    env.reset(seed=0)
    env.target = SAFE_EE.copy()
    env._prev_dist = start = env.distance()
    closing = []
    a = np.zeros(8, np.float32)
    a[mars.ACT_ARM] = SAFE_AND_MOVING
    for _ in range(20):
        _o, _r, _t, _tr, info = env.step(a)
        closing.append(info["terms"]["reach_progress"])
    assert sum(closing) > 1.0, f"closing {start * 100:.0f} cm paid {sum(closing):.3f}"
    assert env.distance() < 0.05
    # It TELESCOPES: the total is W x (d_start - d_now) whatever route it
    # took, so oscillating cannot farm it.
    assert sum(closing) == pytest.approx(
        ME.W_PROGRESS * (start - env.distance()), abs=1e-3)
    # ...and now PARKED at that pose: the term goes to zero, not to a
    # proximity payout that a stalled arm could farm.
    parked = [env.step(a)[4]["terms"]["reach_progress"] for _ in range(20)]
    assert max(abs(p) for p in parked) < 0.02, f"parked earned {parked}"


def test_the_proximity_bonus_is_bounded_per_step_and_pays_for_holding():
    env = _env()
    env.reset(seed=0)
    env.target = env.effector().copy()      # sitting exactly on it
    env._prev_dist = 0.0
    info = _hold(env, (0.0,) * 6, 40)
    per_step = info["terms"]["at_target"]
    assert per_step == pytest.approx(ME.W_NEAR, rel=0.05)
    # 40 steps of holding are worth 40x one step: the bonus is PER STEP, so
    # a fly-by cannot collect what a hold does.
    assert env.reward_sums["at_target"] > 40 * ME.W_NEAR * 0.9
    assert info["success"] is True
    assert info["near_steps"] >= env.hold_steps == 25
    # bounded: 2 cm pays e^-1 of it, 6 cm pays essentially nothing
    env.reset(seed=0)
    env.target = env.effector() + np.array([0.0, 0.0, 0.02])
    env._prev_dist = 0.02
    at2 = _hold(env, (0.0,) * 6, 2)["terms"]["at_target"]
    assert at2 == pytest.approx(ME.W_NEAR * math.exp(-1.0), rel=0.15)


def test_the_penalties_do_not_eat_the_task_at_the_start():
    """AGENTS.md's "a ramped penalty can eat its task": the two penalties must
    be small where the policy STARTS, and neither may ramp.

    MEASURED on a uniform random policy (60 episodes): progress swings a few
    tenths while action_rate is -0.72 and joint_vel -0.67 per episode, against
    the +11 a full 0.29 m reach pays and the +350 a held one does. Here: the
    same comparison in one episode of the worst case the action box allows.
    """
    env = _env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    worst = {"action_rate_penalty": 0.0, "action_mag_penalty": 0.0,
             "joint_vel_penalty": 0.0}
    for _ in range(40):
        _o, _r, term, _t, info = env.step(
            rng.uniform(-1, 1, 8).astype(np.float32))
        for k in worst:
            worst[k] += info["terms"][k]
        if term:
            break
    reach_ceiling = ME.W_PROGRESS * 0.29
    assert all(v <= 0.0 for v in worst.values())       # penalties are <= 0
    assert -sum(worst.values()) < 0.25 * reach_ceiling, worst
    # not ramped: the weights are module constants, not a function of steps
    assert not hasattr(env, "_action_rate_weight")


def test_every_term_is_present_every_step_and_the_signs_are_right():
    """A stable key set is what the teach panel draws one bar per, and what
    `train._penalty_sign_callback_cls` watches for a flip."""
    env = _env()
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(np.zeros(8, np.float32))
    assert set(info["terms"]) == {
        "reach_progress", "at_target", "action_rate_penalty",
        "action_mag_penalty", "joint_vel_penalty", "self_collision_penalty"}
    for key, value in info["terms"].items():
        if key.endswith("_penalty"):
            assert value <= 0.0, f"{key} is {value:+.4f}"
    assert "episode_rewards" not in info           # only at the end
    info = _hold(env, (0.0,) * 6, 200)
    assert set(info["episode_rewards"]) == set(info["terms"])


# ----------------------------------------------------------------- the task

def test_reset_is_a_pure_function_of_the_seed():
    a = _env()
    first, _ = a.reset(seed=0)
    target_a = a.target.copy()
    second, _ = a.reset(seed=0)
    assert np.allclose(first, second)
    assert np.allclose(target_a, a.target)
    other, _ = a.reset(seed=1)
    assert not np.allclose(target_a, a.target)
    # a fresh env with the same seed agrees too (no process state carried)
    assert np.allclose(_env().reset(seed=0)[0], first)


def test_the_shell_is_the_one_the_roadmap_declares():
    """The WINDOW itself, against literals.

    `test_the_target_is_drawn_from_the_declared_shell` reads the module's own
    constants, so it says the draws respect whatever the shell currently is —
    and a planted widening to the whole circle passed it. That is the
    camera-geometry lesson (`AGENTS.md`: a test that fixes a number it does
    not NAME is a hostage) pointed at my own test. So the numbers
    `docs/mars-roadmap.md` Phase 4 specifies are pinned here, once, and a
    deliberate change to the shell fails exactly this case with a sentence
    rather than scattering.
    """
    assert ME.REACH_RADIUS_M == (0.15, 0.40)
    assert ME.REACH_YAW_RAD == pytest.approx((-math.pi / 3, math.pi / 3))
    assert ME.REACH_HEIGHT_M == (0.05, 0.35)
    assert ME.SUCCESS_RADIUS_M == 0.02
    assert ME.SUCCESS_HOLD_S == 1.0
    assert ME.EPISODE_S == 8.0


def test_the_target_is_drawn_from_the_declared_shell():
    """Radius, yaw and height, measured against the SHOULDER the env read off
    the model — so a URDF revision that moves the arm mount moves the shell
    with it instead of aiming the task at a point in space."""
    env = _env()
    assert np.allclose(env.shoulder_base, [0.086, -0.0528, 0.0402], atol=1e-3)
    for seed in range(200):
        env.reset(seed=seed)
        rel = env.target_base_sample - env.shoulder_base
        r = float(np.linalg.norm(rel))
        yaw = math.atan2(rel[1], rel[0])
        assert ME.REACH_RADIUS_M[0] - 1e-9 <= r <= ME.REACH_RADIUS_M[1] + 1e-9
        assert ME.REACH_YAW_RAD[0] - 1e-9 <= yaw <= ME.REACH_YAW_RAD[1] + 1e-9
        assert (ME.REACH_HEIGHT_M[0] - 1e-9 <= env.target_base_sample[2]
                <= ME.REACH_HEIGHT_M[1] + 1e-9)
        assert math.hypot(*rel[:2]) > ME.REACH_MIN_HORIZ_M


def test_the_target_is_not_redrawn_mid_episode():
    """The progress term telescopes to `d_start - d_end`, which is only a
    measure of the task if the target stands still."""
    env = _env()
    env.reset(seed=4)
    fixed = env.target.copy()
    _hold(env, SAFE_AND_MOVING, 60)
    assert np.allclose(env.target, fixed)


def test_success_needs_the_LAST_second_not_a_fly_by():
    env = _env()
    env.reset(seed=0)
    env.target = env.effector().copy()
    env._prev_dist = 0.0
    # 10 steps inside the ball is 0.4 s — not enough
    info = _hold(env, (0.0,) * 6, 10)
    assert info["near_steps"] == 10 and info["success"] is False
    # leave the ball and the streak resets
    env.target = env.effector() + np.array([0.5, 0.0, 0.0])
    info = _hold(env, (0.0,) * 6, 1)
    assert info["near_steps"] == 0 and info["success"] is False


# ------------------------------------------------------------ the CLI seam

def test_env_class_picks_the_mars_arm_env():
    assert T.env_class("mars", "reach") is MarsArmEnv
    with pytest.raises(SystemExit, match="unknown --task"):
        T.env_class("mars", "walk")
    with pytest.raises(SystemExit, match="unknown --task"):
        T.env_class("mars", "pick")


def test_the_task_vocabulary_has_one_definition():
    """`train.MARS_TASKS`, `mars_env.TASKS` and the recipe registry must
    agree. The tuple in `train.py` is a literal so that `--help` does not
    import the behavior library before the fork; this is what stops the three
    from drifting."""
    from microduck_local import behaviors as B

    assert T.MARS_TASKS == ME.TASKS == ("reach",)
    assert tuple(b.task for b in B.for_robot("mars")) == T.MARS_TASKS


def test_the_env_accepts_every_kwarg_the_trainer_produces():
    """Built from `train.parse_args`, not from a hand-written dict: the whole
    point is that whatever the trainer passes, the env takes."""
    args = T.parse_args(["--robot", "mars", "--task", "reach", "--envs", "8",
                         "--steps", "20000", "--run-name", "t", "--seed", "3"])
    kw = T.env_kwargs_from_args(args)
    assert kw == {"domain_rand": True, "obs_noise": True}
    env = MarsArmEnv(task=args.task, seed=args.seed, **kw)
    assert env.obs_noise is True and env.domain_rand is True
    # `make_env` is the actual construction path the vec env uses
    made = T.make_env(0, args.seed, robot="mars", task="reach", **kw)()
    assert isinstance(made, MarsArmEnv)
    assert made.observation_space.shape == (32,)
    # ...and the walker knobs that reach a non-duck body are ignored, not
    # reinterpreted
    quiet = MarsArmEnv(seed=0, obs_noise=False, action_delay=True,
                       random_yaw=False, push_robot=True, domain_rand=False)
    assert quiet.action_delay is True and quiet.push_robot is True


def test_bam_is_refused_for_mars_at_the_cli_and_in_the_env():
    """BAM is an XL330 identification. It describes joints 4-6 and the head
    and NOT the XL430/XC430 shoulder, so it is refused rather than applied to
    three joints it does not model — and refused rather than DROPPED, which
    is the shape of AGENTS.md rule 0."""
    args = T.parse_args(["--robot", "mars", "--task", "reach",
                         "--actuator", "bam"])
    with pytest.raises(SystemExit, match="XL430"):
        T.env_kwargs_from_args(args)
    with pytest.raises(ValueError, match="XL430"):
        MarsArmEnv(seed=0, actuator="bam")
    with pytest.raises(ValueError, match="XL430"):
        MarsArmEnv(seed=0, actuator_force="bam")
    # the default path asks for nothing and gets nothing
    assert mars.MARS.train_env_kwargs(
        T.parse_args(["--robot", "mars", "--task", "reach"])) == {}


def test_head_range_is_still_refused_and_the_duck_is_untouched():
    args = T.parse_args(["--robot", "mars", "--head-range=0,0,0,0,0,0,0,0"])
    with pytest.raises(SystemExit, match="head-pose command"):
        T.env_kwargs_from_args(args)
    duck = T.env_kwargs_from_args(T.parse_args([]))
    assert duck["actuator"] == T.DEFAULT_ACTUATOR == "bam"


def test_a_reach_run_is_recorded_as_command_pinned():
    """The lab must not drive a MARS whose base pair was forced to zero
    throughout training — a drive command was never in its distribution."""
    assert T.is_pinned_command("reach") is True
    assert T.is_pinned_command("stand") is True     # the G1's, unchanged
    assert T.is_pinned_command("imitate") is True
    assert T.is_pinned_command("walk") is False
    assert T.is_pinned_command("squat") is False


def test_the_recipe_the_teach_panel_shows_matches_the_env():
    """The panel's rows are DISPLAY rows (the env owns the reward), so the
    weights are copied — and two tables that must agree are exactly the
    duplication that drifts."""
    from microduck_local import behaviors as B

    recipes = B.for_robot("mars")
    assert [b.id for b in recipes] == ["mars_reach"]
    reach = recipes[0]
    assert reach.robot == "mars"
    assert reach.trainer == ("-m", "microduck_local.train", "--robot", "mars",
                            "--task", "reach")
    assert reach.episode_s == ME.EPISODE_S == 8.0
    weights = {t.key: t.weight for t in reach.terms}
    assert weights == {
        "reach_progress": ME.W_PROGRESS,
        "at_target": ME.W_NEAR,
        "action_rate_penalty": ME.ACTION_RATE_W,
        "action_mag_penalty": ME.ACTION_MAG_W,
        "joint_vel_penalty": ME.JOINT_VEL_W,
        "self_collision_penalty": -ME.SELF_COLLISION_PENALTY,
    }
    penalties = {t.key for t in reach.terms if t.is_penalty}
    assert penalties == {k for k in weights if k.endswith("_penalty")}
    # every row the env emits has a panel row, and vice versa
    env = _env()
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(np.zeros(8, np.float32))
    assert set(info["terms"]) == set(weights)


def test_eval_walk_refuses_a_wheeled_body(monkeypatch, tmp_path):
    """Every number `eval-walk` prints is a gait's. Building an arm env and
    printing three meaningless columns off it would read like a measurement,
    which is worse than no measurement (AGENTS.md rule 6)."""
    import os
    import sys

    from microduck_local import eval_onnx as EV

    monkeypatch.delenv("MICRODUCK_RUN_CMD", raising=False)
    onnx = tmp_path / "policy.onnx"
    onnx.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv",
                        ["eval-walk", str(onnx), "--robot", "mars"])
    with pytest.raises(SystemExit, match="probe_mars_reach"):
        EV.main()
    assert "MICRODUCK_RUN_CMD" not in os.environ


# --------------------------------------------------------- train and export

def test_a_short_ppo_run_trains_and_the_export_round_trips(tmp_path):
    """The whole pipeline in miniature: SB3 on this env, then `export-walk`'s
    own function, then onnxruntime — and the file leaves stamped with
    `mars-arm-32-v1`, so `resolve()` knows its body with no run directory."""
    pytest.importorskip("torch")
    import json

    import onnxruntime as ort
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from microduck_local.export_onnx import export
    from microduck_local.robots.policy_contract import recorded, resolve

    venv = VecNormalize(DummyVecEnv([lambda: MarsArmEnv(seed=0)]),
                        norm_obs=True, norm_reward=False, clip_obs=100.0)
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=16, n_epochs=1,
                device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=64)

    run = tmp_path / "mars-run"
    run.mkdir()
    model.save(str(run / "model"))
    venv.save(str(run / "vecnormalize.pkl"))
    (run / "run.json").write_text(json.dumps({
        "robot": "mars", "task": "reach",
        "contract": mars.MARS.contract().as_dict()}))

    out = export(run, run / "policy.onnx")
    sess = ort.InferenceSession(str(out))
    assert sess.get_inputs()[0].shape == [1, 32]
    assert sess.get_outputs()[0].shape == [1, 8]
    stamped = recorded(out)
    assert stamped is not None and stamped.id == "mars-arm-32-v1"
    assert stamped.rate_hz == 25.0
    assert "untested on hardware" in stamped.deploy
    assert resolve(out).robot == "mars"
    # ...and the policy drives the env it was trained in
    env = MarsArmEnv(seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        action = sess.run(None, {"obs": obs[None]})[0][0]
        obs, _r, term, trunc, _i = env.step(action.astype(np.float32))
        if term or trunc:
            break
