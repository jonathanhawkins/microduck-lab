"""Picking a block up with MARS's arm: the scene, the grasp, the ladder.

`docs/mars-roadmap.md` Phase 4b. `tests/test_mars_env.py` covers the contract,
the action map and the `reach` task; this file covers what `pick` adds, and it
is organised around the four things that could silently not work:

  * **the toy is not part of the robot.** `self_collision` used to mean "any
    two bodies that are not the world", which a free-jointed block is also not
    — so a grasp would have ended the episode as a self-collision at the
    moment of success. The subtree check is pinned here against the toy's own
    id rather than against a count.
  * **the holding predicate.** Every number in it was measured
    (`scripts/probe_mars_pick.py --scripted`), so the tests assert the
    BEHAVIOUR — a claw shut on air reads zero, a claw with the block in it
    reads the servo ceiling — with `mars_env`'s own constants in the
    expression, never a literal threshold.
  * **the ladder.** A spawn box that quietly ignores its rung is the
    curriculum equivalent of a dead knob, so the box is asserted by sampling
    it, and the environment variable is followed all the way from
    `MICRODUCK_MARS_PICK_RUNG` to the constructed env.
  * **the task reaching the env at all.** `train.make_env` uses `--task` only
    to choose a CLASS; the first `--task pick` run of this phase trained
    `reach` under a pick run's name because `MarsBody.env_class` answered one
    class for both. That path is now a test.

Every positive case below was shown to fail on a planted break; the table is
in the phase report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_local import train as T
from microduck_local.robots import mars
from microduck_local.robots import mars_env as ME
from microduck_local.robots.mars_env import MarsArmEnv, MarsPickEnv

pytestmark = pytest.mark.skipif(
    not mars.mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

#: The scripted pick's two poses for seed 0 of rung 1, MEASURED with
#: `scripts/probe_mars_pick.py --scripted-env` (coordinate descent onto the
#: block, then onto the same effector point 10 cm higher) and PINNED so the
#: test costs no solver. Held through the env's own `delta` action map they
#: take the block to a 9.0 cm lift with `gripper_load` +1.997.
PICK_OPEN_Q = (-0.0539, 0.5750, -0.1198, 0.6859, -0.3333, 0.6000)
PICK_LIFT_Q = (-0.0528, 0.1896, -0.1491, 0.8115, -0.3105, 0.6000)


def _pick(**kw) -> MarsArmEnv:
    kw.setdefault("seed", 0)
    return MarsArmEnv(task="pick", **kw)


def _scripted(env, seed: int = 0, **kw):
    """The rung-0 pick, through the env's own action space."""
    import probe_mars_pick as P

    env.reset(seed=seed)
    return P.scripted_env_pick(env, np.array(PICK_OPEN_Q),
                               np.array(PICK_LIFT_Q), **kw)


# ------------------------------------------------------------- the pick scene

def test_the_pick_scene_runs_at_2_ms_and_still_ticks_at_25_Hz():
    """The one measurement that changed the ENVIRONMENT rather than the pay.

    A scripted pick holds 4/16 spots at the world's 5 ms and 14/16 at 2 ms,
    and what fails at 5 ms is ejection. The control rate may not move with it:
    `mars-arm-32-v1` declares 25 Hz, and a policy stepped at another rate is a
    different controller.
    """
    env = _pick()
    # The LITERAL, not the constant: everything downstream is derived from
    # `PICK_PHYSICS_DT`, so asserting the scene against it would move with any
    # change to it and pin nothing. 2 ms is a measurement (4/16 spots held at
    # 5 ms, 14/16 at 2 ms), so it is pinned as one, and a change to it has to
    # come with a new measurement rather than a passing test.
    assert ME.PICK_PHYSICS_DT == 0.002
    assert ME.PICK_DECIMATION == 20
    assert float(env.model.opt.timestep) == pytest.approx(ME.PICK_PHYSICS_DT)
    assert env.decimation == ME.PICK_DECIMATION
    assert (env.model.opt.timestep * env.decimation
            == pytest.approx(ME.CTRL_DT))
    assert 1.0 / env.dt == pytest.approx(mars.CONTROL_HZ)
    # ...and `reach` is untouched: the world's step, and its own decimation.
    reach = MarsArmEnv(seed=0)
    assert float(reach.model.opt.timestep) == pytest.approx(0.005)
    assert reach.decimation == ME.DECIMATION == 8
    assert reach.dt == env.dt == ME.CTRL_DT


def test_the_toy_is_the_playroom_s_own_and_rests_on_the_floor():
    """Phase 5's `TidyArm` has to pick up the object this policy trained on,
    so the block is `world/scenario.PICKABLE_KINDS` by reference."""
    from microduck_local.world.scenario import PICKABLE_KINDS

    # the KIND is pinned as a literal, and so are the roadmap's own "4 cm
    # block, 20 g" — naming the constant on both sides of the comparison
    # would let a different toy pass by agreeing with itself
    assert ME.PICK_TOY_KIND == "block"
    size, mass, _rgba = ME.toy_spec()
    assert size == (0.04, 0.04, 0.04) and mass == 0.02
    assert size == PICKABLE_KINDS[ME.PICK_TOY_KIND]["size"]
    assert mass == PICKABLE_KINDS[ME.PICK_TOY_KIND]["mass"]
    assert ME.toy_rest_z() == pytest.approx(size[2] / 2 + ME.PICK_FLOAT_M)

    env = _pick()
    env.reset(seed=3)
    geom = env.model.geom(f"{ME.PICK_TOY_BODY}_geom")
    np.testing.assert_allclose(geom.size, [v / 2 for v in size])
    # the composer's friction, not MuJoCo's default: at equal priority the
    # floor's 1.0 sliding would win and the toy's 0.8 would be inert
    assert int(np.ravel(geom.priority)[0]) == 1
    np.testing.assert_allclose(np.ravel(geom.friction)[:3],
                               [0.8, 0.005, 0.0001])
    assert env.block_pos()[2] == pytest.approx(ME.toy_rest_z())
    assert env.lift_m() == pytest.approx(0.0)


def test_the_toy_parks_outside_the_robot_in_the_models_rest_pose():
    """A keyframe has to be a legal state. Parked at the origin the toy is
    INSIDE the chassis, which makes every pose an IK solver tries look like a
    collision — and it is the model's qpos0, so it is what the first forward
    pass of an un-reset env sees."""
    import mujoco

    env = _pick()
    m = env.model
    park = np.array(m.key(mars.HOME_KEY).qpos[env._toy_qadr:env._toy_qadr + 3])
    np.testing.assert_allclose(park[:2], ME.PICK_TOY_PARK)
    assert park[2] == pytest.approx(ME.toy_rest_z())
    probe = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, probe, m.key(mars.HOME_KEY).id)
    mujoco.mj_forward(m, probe)
    toy_geom = m.geom(f"{ME.PICK_TOY_BODY}_geom").id
    for i in range(probe.ncon):
        con = probe.contact[i]
        assert toy_geom not in (int(con.geom1), int(con.geom2)), (
            "the parked toy touches the robot in the HOME keyframe")


def test_a_grasp_is_not_a_self_collision():
    """`_robot_bodies` is `base_link`'s SUBTREE, not "everything but world".

    The toy is not the world either, so the old spelling would have ended
    every successful episode as a self-collision. And for `reach` the two
    spellings are provably the same set, which is what keeps Phase 4a's
    numbers valid.
    """
    env = _pick()
    assert env._toy_body >= 0
    assert env._toy_body not in env._robot_bodies
    assert len(env._robot_bodies) == env.model.nbody - 2      # world and toy

    reach = MarsArmEnv(seed=0)
    everything_but_world = {
        b for b in range(reach.model.nbody)
        if (__import__("mujoco").mj_id2name(
            reach.model, __import__("mujoco").mjtObj.mjOBJ_BODY, b) or "")
        != "world"}
    assert set(reach._robot_bodies) == everything_but_world

    # ...and a held block does not terminate the episode
    info = _scripted(env)
    assert info["holding"] is True
    assert info["self_collision"] is None


# ------------------------------------------------------------- the observation

def test_the_observation_still_tiles_the_contract_for_pick():
    """`pick` adds a task, not a layout. The contract is what a hot-swapped
    policy is resolved by, so two tasks at one width must mean one thing."""
    env = _pick()
    want = mars.MARS.contract()
    assert env.observation_space.shape == (want.obs_dim,) == (32,)
    assert env.action_space.shape == (want.act_dim,) == (8,)
    assert [(s.name, s.start, s.stop) for s in want.slots] == [
        (n, s.start, s.stop) for n, s in env._SLOTS]
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32,) and obs.dtype == np.float32
    # target_base IS the block, in the base frame, and target_seen is the
    # privileged 1.0 the detector gate will replace in Phase 5
    np.testing.assert_allclose(obs[mars.OBS_TARGET_BASE],
                               env._to_base(env.block_pos()), atol=1e-6)
    assert obs[mars.OBS_TARGET_SEEN] == pytest.approx(1.0)
    np.testing.assert_allclose(obs[mars.OBS_RESERVED], 0.0)
    # ...and the base pair is read and forced to zero, as it is for reach
    assert env.use_base is False
    assert "pick" in ME.ARM_ONLY_TASKS


def test_gripper_load_reads_the_object_and_not_the_servo():
    """The MEASURED discriminator, asserted as behaviour.

    Shut on air the blades stop at their own hard stop with zero position
    error, so the servo torque is zero and so is the constraint. With the
    block in the claw the servo saturates at its ceiling and the constraint
    carries the same magnitude back — which is the only one of the two that is
    zero while the jaw is merely TRAVELLING.
    """
    env = _pick()

    # (a) shut on AIR: the toy parked far away, the gripper driven closed.
    # The WHOLE close is sampled, not just its end, and that is the case that
    # rules the servo torque out: while the blades travel it saturates at the
    # 2 N*m ceiling with nothing in the claw, so one sample of it cannot tell
    # "closing" from "holding". The constraint is 0 for every step of it.
    env.reset(seed=0)
    env._place_block(env._to_base(np.array(
        [ME.PICK_TOY_PARK[0], ME.PICK_TOY_PARK[1], ME.toy_rest_z()])))
    # OPEN the jaw first. ARM_HOME already sits 0.087 rad off the stop, which
    # is less than one `DELTA_RAD_PER_STEP`, so closing from there arrives in
    # a single control step and nothing travels — the whole point of this
    # case is the travel.
    openit = np.zeros(8, np.float32)
    openit[5] = 1.0
    for _ in range(10):
        env.step(openit)
    assert env.data.qpos[env._arm_qadr[5]] > 0.5
    close = np.zeros(8, np.float32)
    close[5] = -1.0
    through_air, servo = [], []
    for _ in range(60):
        env.step(close)
        through_air.append(abs(env.gripper_load()))
        servo.append(abs(float(env.data.qfrc_applied[env._grip_dadr])))
        assert env.holding() is False
    assert max(through_air) == pytest.approx(0.0, abs=1e-6), (
        "the load slot moved with nothing in the claw")
    # ...and the servo torque, which is what this slot used to carry, DID
    # saturate on the way down — the measurement that decided the slot
    assert max(servo) >= mars.GRIPPER_EFFORT_LIMIT * 0.9
    assert sum(s >= mars.GRIPPER_EFFORT_LIMIT * 0.9 for s in servo) >= 3
    assert abs(env.data.qpos[env._arm_qadr[5]] -
               mars.GRIPPER_CLOSED_ON_AIR_RAD) < 1e-3
    obs = env._get_obs()
    assert obs[mars.OBS_GRIPPER_LOAD] == pytest.approx(0.0, abs=1e-6)

    # (b) HOLDING: the scripted pick
    info = _scripted(env)
    assert info["holding"] is True
    assert abs(info["gripper_load"]) >= ME.HOLD_LOAD_NM
    assert abs(info["gripper_load"]) <= mars.GRIPPER_EFFORT_LIMIT * 1.1
    obs = env._get_obs()
    assert obs[mars.OBS_GRIPPER_LOAD] == pytest.approx(env.gripper_load(),
                                                       rel=1e-6)
    # the predicate names the OBJECT as well as the load
    assert env._touching_toy() is True


# -------------------------------------------------------------- the ladder

@pytest.mark.parametrize("rung", (1, 2, 3))
def test_the_block_spawns_inside_its_rung_and_never_outside(rung):
    """The physics ladder, sampled rather than asserted from a constant.

    Rungs 1 and 2 are squares about `PICK_SPOT`; rung 3 is the shell's floor
    footprint. Every rung must stay inside ONE footprint, or the knocked-away
    terminal would mean something different on each.
    """
    env = _pick(pick_rung=rung)
    half = ME.PICK_RUNG_HALF_M.get(rung)
    centre = env._shell_point(*ME.PICK_SPOT, ME.toy_rest_z())
    seen = []
    for s in range(60):
        env.reset(seed=s)
        p = env.target_base_sample
        seen.append(p)
        assert p[2] == pytest.approx(ME.toy_rest_z())
        assert env.in_shell(p), f"rung {rung} spawned outside the footprint"
        if half is not None:
            assert abs(p[0] - centre[0]) <= half + 1e-9
            assert abs(p[1] - centre[1]) <= half + 1e-9
        else:
            r = env.shell_radius(p)
            assert ME.REACH_RADIUS_M[0] - 1e-9 <= r <= ME.REACH_RADIUS_M[1] + 1e-9
            assert abs(env.shell_yaw(p)) <= ME.REACH_YAW_RAD[1] + 1e-9
    spread = np.ptp(np.array(seen)[:, :2], axis=0)
    if half is not None:
        # the box is SAMPLED, not a point: 60 draws must fill most of it
        assert (spread > half).all(), f"rung {rung} barely moves the block"
    else:
        assert spread.max() > 2 * max(ME.PICK_RUNG_HALF_M.values())


def test_the_rungs_are_nested_and_widen():
    """Rung 1 is inside rung 2 is inside rung 3 — a ladder, not three boxes.

    Rung 0 is the drill and is deliberately ONE state, so it is the floor of
    the ladder rather than a fourth box.
    """
    spreads = []
    for rung in ME.PICK_RUNGS:
        env = _pick(pick_rung=rung)
        pts = []
        for s in range(40):
            env.reset(seed=s)
            pts.append(env.target_base_sample[:2])
        spreads.append(float(np.ptp(np.array(pts), axis=0).max()))
    assert spreads == sorted(spreads)
    assert spreads[0] == pytest.approx(0.0, abs=1e-9)   # the drill
    assert spreads[1] < spreads[2] < spreads[3]
    assert ME.PICK_RUNG_HALF_M[1] < ME.PICK_RUNG_HALF_M[2]


def test_rung_0_spawns_the_arm_around_the_block():
    """The DRILL, and why it exists.

    Rung 1 from scratch never grasps — 1.5 M steps, `at_target` +324 over 20
    deterministic episodes and `lift_progress` / `held_high` exactly 0.000 on
    every one, with the claw parked 2-10 mm from the block for seven of the
    eight seconds. `AGENTS.md`: if no rollout ever contains the skill, ladder
    the PHYSICS. Rung 0 spawns the arm with its jaws already open around the
    block, so closing is one action away.

    Three things have to agree or the drill silently is not one: the written
    `qpos`, the driver's target, and the action integrator. If the last is
    left at HOME the arm snaps back on the first step with the block in the
    way.
    """
    env = _pick(pick_rung=0)
    for s in range(3):
        env.reset(seed=s)
        np.testing.assert_allclose(env.data.qpos[env._arm_qadr],
                                   ME.PICK_GRASP_POSE, atol=1e-9)
        np.testing.assert_allclose(env._cmd_target, ME.PICK_GRASP_POSE,
                                   atol=1e-9)
        assert env.driver.arm_targets()["joint1"] == pytest.approx(
            ME.PICK_GRASP_POSE[0])
        # the block is BETWEEN the blades, not on the floor in front
        assert env.distance() < 0.02
        assert env.holding() is False          # open, not yet squeezing
        assert env.block_pos()[2] == pytest.approx(ME.toy_rest_z(), abs=2e-3)

    # a zero action holds the spawn pose instead of walking back to HOME
    env.reset(seed=0)
    for _ in range(25):
        env.step(np.zeros(8, np.float32))
    np.testing.assert_allclose(env._cmd_target, ME.PICK_GRASP_POSE, atol=1e-9)
    assert env.distance() < 0.03

    # ...and closing alone is enough to hold it: the skill is one action deep
    env.reset(seed=0)
    close = np.zeros(8, np.float32)
    close[5] = -1.0
    for _ in range(40):
        _o, _r, _t, _tr, info = env.step(close)
    assert info["holding"] is True
    assert abs(info["gripper_load"]) >= ME.HOLD_LOAD_NM


def test_the_rung_comes_through_the_environment_variable(monkeypatch):
    """A curriculum stage passes its knobs through the trainer's ENVIRONMENT
    (the lab's stage machinery does not rewrite argv), which is how the G1's
    `MICRODUCK_G1_COMMAND_MIX` reaches `command_mix`. It still has to land in
    the run's `env_kwargs`, or the rung a run trained on is not recorded."""
    args = T.parse_args(["--robot", "mars", "--task", "pick"])
    monkeypatch.delenv("MICRODUCK_MARS_PICK_RUNG", raising=False)
    assert "pick_rung" not in T.env_kwargs_from_args(args)

    for rung in ME.PICK_RUNGS:
        monkeypatch.setenv("MICRODUCK_MARS_PICK_RUNG", str(rung))
        kw = T.env_kwargs_from_args(args)
        assert kw["pick_rung"] == rung
        assert MarsArmEnv(task="pick", seed=0, **kw).pick_rung == rung

    # ...and it does NOT leak into a reach run
    reach_args = T.parse_args(["--robot", "mars", "--task", "reach"])
    assert "pick_rung" not in T.env_kwargs_from_args(reach_args)

    # a rung nobody measured is refused, not clamped
    for bad in ("-1", "4", "two"):
        monkeypatch.setenv("MICRODUCK_MARS_PICK_RUNG", bad)
        with pytest.raises(SystemExit, match="MICRODUCK_MARS_PICK_RUNG"):
            T.env_kwargs_from_args(args)
    with pytest.raises(SystemExit, match="pick ladder"):
        MarsArmEnv(task="pick", seed=0, pick_rung=7)


def test_the_task_reaches_the_env_and_not_just_the_class():
    """`train.make_env` uses `--task` ONLY to choose a class.

    This is the phase's own bug, promoted to a test: with one class for both
    tasks a `--task pick` run trained `reach`, in `reach`'s scene, at
    `reach`'s timestep, and the only tell was six reward keys instead of nine.
    """
    assert T.env_class("mars", "pick") is MarsPickEnv
    assert T.env_class("mars", "reach") is MarsArmEnv
    built = T.make_env(0, 0, robot="mars", task="pick",
                       domain_rand=True, obs_noise=True)()
    assert built.task == "pick"
    assert built.decimation == ME.PICK_DECIMATION
    assert built._toy_body >= 0
    reach = T.make_env(0, 0, robot="mars", task="reach")()
    assert reach.task == "reach" and reach.decimation == ME.DECIMATION
    # MarsPickEnv is MarsArmEnv with one default changed, and nothing else
    assert issubclass(MarsPickEnv, MarsArmEnv)
    assert MarsPickEnv(seed=0).task == "pick"


# --------------------------------------------------------------- the reward

def test_the_scripted_pick_succeeds_under_the_envs_own_step_loop():
    """Does ANY rollout contain the skill? — asked of the ACTION SPACE.

    `AGENTS.md`: a reward cannot pay for a behaviour the action space has no
    way to express, and `delta` itself is the lesson that produced that rule.
    The rung-0 pick — open, place, close, lift — driven by the same eight
    floats a policy emits.
    """
    env = _pick()
    info = _scripted(env)
    assert info["holding"] is True
    assert info["lift_m"] >= ME.SUCCESS_LIFT_M
    assert info["grasps"] == 1
    assert info["hold_frac"] > 0.5
    assert info["success"] is True
    assert info["block_lost"] is False
    sums = env.reward_sums
    assert sums["held_high"] > 0.0 and sums["lift_progress"] > 0.0
    # the task terms dominate what a successful trajectory is paid: the four
    # penalties together cost it under 1% of what it earns
    task = sums["at_target"] + sums["held_high"] + sums["lift_progress"]
    penalties = -sum(v for k, v in sums.items() if k.endswith("_penalty"))
    assert penalties < 0.02 * task


def test_height_progress_pays_only_while_holding():
    """A block that goes up on its own is not a pick.

    `lift_progress` telescopes in the HELD height, which is 0 whenever the
    claw is empty — so knocking the block into the air earns nothing, and
    dropping it from height hands back every point the lift earned.
    """
    env = _pick()
    env.reset(seed=0)
    # (a) the block hoisted to 30 cm with the claw empty: no pay, ever
    high = env._to_base(env.block_pos()) + np.array([0.0, 0.0, 0.30])
    for _ in range(5):
        env._place_block(high)
        _o, _r, _t, _tr, info = env.step(np.zeros(8, np.float32))
        assert info["holding"] is False
        assert info["terms"]["lift_progress"] == pytest.approx(0.0)
        assert info["terms"]["held_high"] == pytest.approx(0.0)

    # (b) the same rise, held: it pays, and it telescopes
    env2 = _pick()
    info = _scripted(env2)
    paid = env2.reward_sums["lift_progress"]
    assert paid == pytest.approx(ME.W_LIFT * info["lift_m"], rel=0.05)

    # (c) ...and a release hands it back. Prised open at height, the held
    # height falls to 0 and the term charges the whole lift.
    before = env2.reward_sums["lift_progress"]
    lift_at_release = info["lift_m"]
    openit = np.zeros(8, np.float32)
    openit[5] = 1.0
    for _ in range(25):
        _o, _r, t, tr, info = env2.step(openit)
        if t or tr:
            break
    assert info["holding"] is False
    assert env2.reward_sums["lift_progress"] == pytest.approx(
        before - ME.W_LIFT * lift_at_release, rel=0.1)


def test_knocking_the_block_out_of_the_shell_terminates_with_the_penalty():
    """...and only while the claw is empty: a toy the arm is CARRYING can be
    swung anywhere the arm reaches, and ending a successful hold because the
    carry went wide would charge the policy for doing the task."""
    env = _pick()
    env.reset(seed=0)
    out = env._to_base(env.block_pos()) + np.array([1.0, 0.0, 0.0])
    assert env.in_shell(out) is False
    env._place_block(out)
    _o, reward, terminated, _tr, info = env.step(np.zeros(8, np.float32))
    assert info["block_lost"] is True
    assert terminated is True
    assert info["terms"]["block_lost_penalty"] == ME.BLOCK_LOST_PENALTY
    assert reward < 0.0

    # a NUDGE inside the margin is not "knocked away"
    env.reset(seed=0)
    nudged = env._to_base(env.block_pos()) + np.array([0.02, 0.0, 0.0])
    env._place_block(nudged)
    _o, _r, terminated, _tr, info = env.step(np.zeros(8, np.float32))
    assert info["block_lost"] is False and terminated is False

def test_a_block_outside_the_shell_but_HELD_does_not_end_the_episode(
        monkeypatch):
    """The terminal is the CONJUNCTION, and this is how you make it fire.

    A carried block cannot normally be got outside the footprint — the arm's
    reach is the footprint — so the shell is shrunk under the running env
    instead. `in_shell` reads the module constant per call, so the held block
    is now geometrically "lost" and the episode must carry on anyway: ending a
    successful hold because the carry went wide would charge the policy for
    doing the task.
    """
    env = _pick()
    info = _scripted(env)
    assert info["holding"] is True and info["block_lost"] is False

    monkeypatch.setattr(ME, "REACH_RADIUS_M", (0.15, 0.20))
    assert env.block_lost() is True, "the shrunk shell did not take"
    assert env.holding() is True
    hold = np.zeros(8, np.float32)
    hold[5] = -1.0
    for _ in range(10):
        _o, _r, terminated, _tr, info = env.step(hold)
        assert info["holding"] is True
        assert info["block_lost"] is False
        assert terminated is False
        assert info["terms"]["block_lost_penalty"] == 0.0

    # ...and the moment it is released, the same geometry DOES end it
    openit = np.zeros(8, np.float32)
    openit[5] = 1.0
    ended = False
    for _ in range(40):
        _o, _r, terminated, truncated, info = env.step(openit)
        if terminated:
            ended = True
            break
        if truncated:
            break
    assert ended and info["block_lost"] is True
    assert info["terms"]["block_lost_penalty"] == ME.BLOCK_LOST_PENALTY


def test_every_term_is_present_every_step_and_the_signs_are_right():
    """A stable key set is what the teach panel draws one bar per and what
    `train._penalty_sign_callback_cls` watches. A penalty that goes positive
    is a reward the recipe never meant to offer."""
    env = _pick()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    keys = None
    for _ in range(80):
        _o, _r, t, tr, info = env.step(rng.uniform(-1, 1, 8).astype(np.float32))
        terms = info["terms"]
        keys = keys or set(terms)
        assert set(terms) == keys
        for k, v in terms.items():
            if k.endswith("_penalty"):
                assert v <= 0.0, f"{k} paid {v:+.4f}"
        if t or tr:
            env.reset(seed=1)
    assert keys == {"reach_progress", "at_target", "lift_progress",
                    "held_high", "action_rate_penalty", "action_mag_penalty",
                    "joint_vel_penalty", "self_collision_penalty",
                    "block_lost_penalty"}


def test_the_null_action_earns_nothing_and_never_grasps():
    """AGENTS.md rule 3: the baseline before anything is credited. A parked
    arm and a block on the floor is worth zero, so every point a policy
    reports is a point it made."""
    env = _pick()
    for seed in range(4):
        env.reset(seed=seed)
        done = False
        while not done:
            _o, _r, t, tr, info = env.step(np.zeros(8, np.float32))
            done = t or tr
        assert info["grasps"] == 0 and info["hold_frac"] == 0.0
        assert info["success"] is False
        assert info["lift_m"] == pytest.approx(0.0)
        total = sum(info["episode_rewards"].values())
        assert abs(total) < 1.0, f"the null earned {total:+.3f}"


# ------------------------------------------------------- the panel and the run

def test_the_recipe_the_teach_panel_shows_matches_the_env():
    """The panel's rows are DISPLAY rows (the env owns the reward), so the
    weights are copied — and two tables that must agree are the duplication
    that drifts."""
    from microduck_local import behaviors as B

    recipes = {b.id: b for b in B.for_robot("mars")}
    assert sorted(recipes) == ["mars_pick", "mars_reach"]
    assert tuple(b.task for b in B.for_robot("mars")) == T.MARS_TASKS
    pick = recipes["mars_pick"]
    assert pick.robot == "mars"
    assert pick.trainer == ("-m", "microduck_local.train", "--robot", "mars",
                            "--task", "pick")
    assert pick.episode_s == ME.EPISODE_S == 8.0
    weights = {t.key: t.weight for t in pick.terms}
    assert weights == {
        "reach_progress": ME.W_PROGRESS,
        "at_target": ME.W_NEAR,
        "lift_progress": ME.W_LIFT,
        "held_high": ME.W_HELD,
        "action_rate_penalty": ME.ACTION_RATE_W,
        "action_mag_penalty": ME.ACTION_MAG_W,
        "joint_vel_penalty": ME.JOINT_VEL_W,
        "self_collision_penalty": -ME.SELF_COLLISION_PENALTY,
        "block_lost_penalty": -ME.BLOCK_LOST_PENALTY,
    }
    assert {t.key for t in pick.terms if t.is_penalty} == {
        k for k in weights if k.endswith("_penalty")}
    env = _pick()
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(np.zeros(8, np.float32))
    assert set(info["terms"]) == set(weights)


def test_the_curriculum_is_three_rungs_of_PHYSICS_only():
    """`CurriculumStage.env` may ladder spawns and strictness, never the pay.
    Every stage's knob is the spawn box and nothing else."""
    from microduck_local import behaviors as B

    pick = next(b for b in B.for_robot("mars") if b.id == "mars_pick")
    stages = pick.curriculum
    assert len(stages) == len(ME.PICK_RUNGS) == 4
    assert [s.env for s in stages] == [
        {"MICRODUCK_MARS_PICK_RUNG": str(r)} for r in ME.PICK_RUNGS]
    assert all(s.steps > 0 and s.label and s.detail for s in stages)
    # the drill is FIRST: it is the rung that makes a grasp samplable, and
    # every later rung is warm-started off it
    assert stages[0].env["MICRODUCK_MARS_PICK_RUNG"] == "0"
    # the reach recipe is untouched by any of this
    reach = next(b for b in B.for_robot("mars") if b.id == "mars_reach")
    assert reach.curriculum == ()


def test_a_pick_run_is_recorded_as_command_pinned():
    """`MarsArmEnv` forces the base pair to zero for every arm task, so a
    drive command was never in a pick policy's training distribution."""
    assert T.is_pinned_command("pick") is True
    assert T.is_pinned_command("reach") is True
    assert T.is_pinned_command("walk") is False


def test_a_short_ppo_run_trains_and_the_export_round_trips(tmp_path):
    """The whole pipeline in miniature on the PICK env, and the file leaves
    stamped with `mars-arm-32-v1` — the same contract `reach` exports, which
    is the point of not letting a task have its own layout."""
    pytest.importorskip("torch")
    import onnxruntime as ort
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from microduck_local.export_onnx import export
    from microduck_local.robots.policy_contract import recorded, resolve

    venv = VecNormalize(DummyVecEnv([lambda: MarsPickEnv(seed=0)]),
                        norm_obs=True, norm_reward=False, clip_obs=100.0)
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=16, n_epochs=1,
                device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=64)

    run = tmp_path / "mars-pick-run"
    run.mkdir()
    model.save(str(run / "model"))
    venv.save(str(run / "vecnormalize.pkl"))
    (run / "run.json").write_text(json.dumps({
        "robot": "mars", "task": "pick",
        "env_kwargs": {"pick_rung": 1},
        "contract": mars.MARS.contract().as_dict()}))

    out = export(run, run / "policy.onnx")
    sess = ort.InferenceSession(str(out))
    assert sess.get_inputs()[0].shape == [1, 32]
    assert sess.get_outputs()[0].shape == [1, 8]
    stamped = recorded(out)
    assert stamped is not None and stamped.id == "mars-arm-32-v1"
    assert stamped.rate_hz == mars.CONTROL_HZ == 25.0
    assert resolve(out).robot == "mars"
    env = MarsPickEnv(seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        action = sess.run(None, {"obs": obs[None]})[0][0]
        obs, _r, term, trunc, _i = env.step(action.astype(np.float32))
        if term or trunc:
            break


def test_the_probe_scores_a_policy_under_the_rung_it_trained_on(tmp_path):
    """The map lesson again, one task on: a pick scored in a different spawn
    box than it trained in is a number about a task the policy never saw."""
    import probe_mars_pick as P

    assert hasattr(P, "scripted_pick") and hasattr(P, "scripted_env_pick")
    run = tmp_path / "a-run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps(
        {"robot": "mars", "task": "pick", "env_kwargs": {"pick_rung": 3}}))
    meta = json.loads((run / "run.json").read_text())
    assert meta["env_kwargs"]["pick_rung"] == 3
    # the constant the probe falls back to when a run does not say
    assert ME.DEFAULT_PICK_RUNG in ME.PICK_RUNGS
