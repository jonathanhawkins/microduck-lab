"""Locks for the scenario contract and world composition (roadmap 0.1/0.3):
a one-duck world carries the same 14-joint contract as the walk scene,
N ducks step in one model, and bad scenarios fail loudly."""

import json
import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain.brain_env import POLICIES_DIR
from microduck_local.world import (
    Ball,
    Box,
    Duck,
    Scenario,
    Wall,
    compose,
    load_scenario,
    make_room,
    validate_scenario,
)
from microduck_local.world.compose import DuckAddress, spawn_duck
from microduck_local.world.scenario import ScenarioError

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")


def test_single_duck_world_matches_the_walk_contract():
    m = compose(Scenario(name="one", ducks=[Duck("d0", (0, 0, 0))]))
    adr = DuckAddress.resolve(m, "d0")
    ref = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    # Same joint count, same contract order, same actuator order, same limits.
    assert m.nq == ref.nq and m.nv == ref.nv and m.nu == ref.nu
    for k, name in enumerate(C.JOINT_NAMES):
        j = m.joint("d0/" + name)
        r = ref.joint(name)
        assert int(j.qposadr[0]) == int(r.qposadr[0]) == adr.joint_qpos[k]
        np.testing.assert_allclose(j.range, r.range)
        assert m.actuator("d0/" + name).trnid[0] == j.id
    assert m.opt.timestep == C.PHYSICS_DT
    assert adr.tof_site >= 0 and adr.gyro_adr >= 0 and min(adr.foot_geoms) >= 0
    assert m.geom_priority[adr.foot_geoms[0]] == 1


def test_spawn_puts_duck_in_stand_pose_at_rest():
    m = compose(Scenario(name="one", ducks=[Duck("d0", (0.3, -0.2, 1.0))]))
    d = mujoco.MjData(m)
    adr = DuckAddress.resolve(m, "d0")
    spawn_duck(m, d, adr, 0.3, -0.2, 1.0)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.qpos[adr.joint_qpos], C.DEFAULT_POSE, atol=1e-6)
    np.testing.assert_allclose(d.xpos[adr.trunk_body][:2], [0.3, -0.2], atol=1e-6)
    fwd = d.xmat[adr.trunk_body].reshape(3, 3)[:, 0]
    assert fwd @ [math.cos(1.0), math.sin(1.0), 0] > 0.999
    # Held by the servos alone (no policy), the duck stands for a while and
    # then sags — the SAME behaviour as the reference walk env under a zero
    # action (it terminates at ~1.0 s there too). Lock the shared truth: up
    # at 0.4 s, and the sag matches the reference env to the millimetre.
    z_world = []
    for k in range(int(0.8 / C.PHYSICS_DT)):
        mujoco.mj_step(m, d)
        if (k + 1) % int(0.2 / C.PHYSICS_DT) == 0:
            z_world.append(float(d.xpos[adr.trunk_body][2]))
    assert z_world[1] > 0.10
    from microduck_local.walk_env import MicroduckWalkEnv
    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False, action_delay=False,
                           random_yaw=False, seed=0)
    env.reset(seed=0)
    # The reference reset adds ±0.03 rad pose noise and up to 1 cm of height;
    # re-pose it exactly as spawn_duck did so the two trajectories are comparable.
    env.data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
    env.data.qpos[0:3] = [0.3, -0.2, 0.12]
    env.data.qpos[3:7] = [math.cos(0.5), 0, 0, math.sin(0.5)]
    env.data.qvel[:] = 0
    env.data.ctrl[:] = C.DEFAULT_POSE
    mujoco.mj_forward(env.model, env.data)
    z_ref = []
    for k in range(int(0.8 / C.PHYSICS_DT)):
        mujoco.mj_step(env.model, env.data)
        if (k + 1) % int(0.2 / C.PHYSICS_DT) == 0:
            z_ref.append(float(env.data.xpos[env.trunk_body_id][2]))
    np.testing.assert_allclose(z_world, z_ref, atol=1e-3)


def test_three_ducks_and_objects_step_in_one_model():
    sc = Scenario(name="three", floor=(5, 5),
                  walls=[Wall((-2, -2), (2, -2))],
                  boxes=[Box((1, 1, 0.1), (0.2, 0.2, 0.2)), Box((-1, 1, 0.3), (0.1, 0.1, 0.1), mass=0.2)],
                  balls=[Ball((0.6, 0.0))],
                  ducks=[Duck(f"d{i}", (0.0, 0.5 * i, 0.0)) for i in range(3)])
    m = compose(sc)
    d = mujoco.MjData(m)
    adrs = [DuckAddress.resolve(m, f"d{i}") for i in range(3)]
    for i, a in enumerate(adrs):
        spawn_duck(m, d, a, 0.0, 0.5 * i, 0.0)
    mujoco.mj_forward(m, d)
    # 3 free ducks + 1 free box + 1 ball: 5 freejoints + 42 hinges.
    assert m.nq == 5 * 7 + 3 * 14 and m.nu == 3 * 14
    # Distinct, non-overlapping addresses.
    qs = np.concatenate([a.joint_qpos for a in adrs])
    assert len(set(qs.tolist())) == 42
    for _ in range(100):
        mujoco.mj_step(m, d)
    for a in adrs:
        assert d.xpos[a.trunk_body][2] > 0.10
    ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball0")
    assert abs(d.xpos[ball][2] - 0.035) < 0.01     # resting on the floor
    assert m.body_mass[ball] == pytest.approx(0.015)


def test_scenario_roundtrip_and_validation(tmp_path):
    sc = make_room(seed=7, n_boxes=3, n_ducks=2)
    assert len(sc.walls) == 4 and len(sc.ducks) == 2 and sc.name == "room-7"
    assert make_room(seed=7, n_boxes=3, n_ducks=2) == sc      # deterministic
    p = tmp_path / "room.json"
    sc.save(p)
    back = load_scenario(p)
    assert back == sc
    raw = json.loads(p.read_text())
    assert raw["version"] == 1 and raw["walls"][0]["from"] == list(sc.walls[0].start)

    def bad(mutate):
        r = json.loads(p.read_text())
        mutate(r)
        with pytest.raises(ScenarioError):
            validate_scenario(r)

    bad(lambda r: r.update(name="../etc"))
    bad(lambda r: r["ducks"].append({"id": "d0", "spawn": [0, 0, 0]}))   # duplicate
    bad(lambda r: r["ducks"].append({"id": "Bad Id", "spawn": [0, 0, 0]}))
    bad(lambda r: r["ducks"][0].update(tof="lidar"))
    bad(lambda r: r["walls"][0].update(height=float("nan")))
    bad(lambda r: r["boxes"][0].update(size=[0, 0, 0]))
    bad(lambda r: r.update(collision="rollers"))
    bad(lambda r: r.update(floor={"size": [100, 1]}))
    bad(lambda r: r["ducks"].extend({"id": f"x{i}", "spawn": [0, 0, 0]} for i in range(20)))


def test_all_collision_robot_variant_composes():
    m = compose(Scenario(name="all", collision="all", ducks=[Duck("d0", (0, 0, 0))]))
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "d0/mouth_tip") >= 0


def test_pitch_counts_goals_and_recentres_the_ball():
    """Soccer, first form: a ball across a short wall's line inside the goal
    width is a goal for that side; the ball comes back to the centre."""
    import mujoco

    from microduck_local.world import World, make_pitch
    from microduck_local.world.scenario import validate_scenario
    sc = make_pitch()
    assert validate_scenario(sc.to_dict()) == sc and sc.goal_width == 0.7
    w = World(sc)
    assert w.soccer_score() == {"left": 0, "right": 0, "ball": [0.0, 0.0], "lastGoal": None, "kickoff": 0.0}
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx = sc.floor[0] / 2 - 0.25
    w.data.qpos[q:q + 2] = [hx - 0.03, 0.1]                  # on the right goal line, inside the posts
    mujoco.mj_forward(w.model, w.data)
    w.step()
    s = w.soccer_score()
    assert s["left"] == 0 and s["right"] == 1 and abs(s["ball"][0]) < 0.06 and abs(s["ball"][1]) < 0.06
    assert s["lastGoal"] == "right" and w.goal_seq == 1
    w.kickoff_until = -1.0                                   # skip the hold for the second probe
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.6]               # left line but outside the posts: no goal
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.soccer_score()["left"] == 0 and w.goal_seq == 1


def test_a_goal_restarts_play_from_a_kickoff():
    """After a goal every duck is back on its spawn, standing on a zero
    command for the hold however hard its brain pushes, and play resumes
    when the hold ends; the ball sits on the centre spot within its nudge."""
    import mujoco

    from microduck_local.world import World, make_pitch
    sc = make_pitch()
    w = World(sc, seed=3)
    d0, d1 = (w.ducks[d.id] for d in sc.ducks)
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    hx = sc.floor[0] / 2 - 0.25
    # Walk both ducks somewhere else first.
    for d in (d0, d1):
        x, y, yaw = d.spawn
        from microduck_local.world.compose import spawn_duck
        spawn_duck(w.model, w.data, d.adr, x + 0.4, y - 0.3, yaw + 1.0)
    w.data.qpos[q:q + 2] = [-(hx - 0.03), 0.0]
    mujoco.mj_forward(w.model, w.data)
    w.step()
    assert w.soccer_score()["left"] == 1 and w.in_kickoff
    for d in (d0, d1):
        pos = d.trunk_pos(w.data)
        assert abs(pos[0] - d.spawn[0]) < 0.03 and abs(pos[1] - d.spawn[1]) < 0.03
        assert abs(w.odom(d)[0] - d.spawn[0]) < 1e-6          # the odometry frame is the pitch again
    bx, by = w.soccer_score()["ball"]
    assert abs(bx) <= 0.05 and abs(by) <= 0.05
    # The hold: the walker sees a zero twist even though the caller asks for full speed.
    n = 0
    while w.in_kickoff:
        d0.set_cmd(w.data, (0.6, 0.0, 0.0))
        w.step()
        n += 1
        assert float(d0.twist_cmd[0]) == 0.0
    assert abs(n * C.CTRL_DT - w.kickoff_hold_s) <= 2 * C.CTRL_DT and w.goal_seq == 1
    d0.set_cmd(w.data, (0.6, 0.0, 0.0))
    w.step()
    assert not w.in_kickoff and abs(float(d0.twist_cmd[0]) - 0.6) < 1e-6
    # reset() clears the board.
    w.reset()
    assert w.soccer_score()["left"] == 0 and w.soccer_score()["lastGoal"] is None


@pytest.mark.skipif(not (POLICIES_DIR / "ball_kick_left.onnx").exists(), reason="upstream policies not checked out")
def test_shipped_kick_skill_sends_the_ball_flying():
    """The kicks run as a 0.5 s window with an all-zero command, like robotd.
    Measured: a ball 8 cm ahead and 6 cm to the foot's side flies over a
    metre; on the wrong side it is not touched."""
    import mujoco

    from microduck_local.brain.brain_env import onnx_infer
    from microduck_local.world import Ball, Duck, Scenario, World
    for side, by, expect in (("kick_left", 0.06, 1.0), ("kick_right", -0.06, 1.0), ("kick_right", 0.06, 0.02)):
        sc = Scenario(name="k", floor=(4, 4), ducks=[Duck("d0", (0, 0, 0), None, None, None)], balls=[Ball((0.08, by))])
        w = World(sc, infer_for={"d0": onnx_infer(POLICIES_DIR / "alpha_walking.onnx")}, seed=0)
        d = w.ducks["d0"]
        for _ in range(50):
            d.set_cmd(w.data, [0, 0, 0])
            w.step()
        b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, "ball0")
        p0 = w.data.xpos[b].copy()
        assert w.start_skill(d, side) and d.skill == side
        for _ in range(int(1.5 / C.CTRL_DT)):
            w.step()
        moved = float(np.hypot(*(w.data.xpos[b] - p0)[:2]))
        assert d.skill is None and d.falls == 0
        if expect >= 1.0:
            assert moved > expect, (side, by, moved)
        else:
            assert moved < expect, (side, by, moved)


@pytest.mark.skipif(not (POLICIES_DIR / "ball_kick_left.onnx").exists(), reason="upstream policies not checked out")
def test_kick_window_runs_at_the_standing_gain_and_hands_it_back():
    """robotd runs a kick at the standing tuning: the walking Kp softened by
    `standing_gain_ratio` (control.rs). Here the duck's actuators drop to
    that ratio for the 0.5 s window — this duck's only — and come back."""
    from microduck_local.brain.brain_env import onnx_infer
    from microduck_local.world import Ball, Duck, Scenario, World
    from microduck_local.world.arena import KICK_S, STANDING_GAIN_RATIO
    sc = Scenario(name="k2", floor=(4, 4), balls=[Ball((0.08, 0.06))],
                  ducks=[Duck("d0", (0, 0, 0), None, None, None), Duck("d1", (1.5, 0, 0), None, None, None)])
    infer = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
    w = World(sc, infer_for={"d0": infer, "d1": infer}, seed=0)
    d0, d1 = w.ducks["d0"], w.ducks["d1"]
    kp0 = w.model.actuator_gainprm[d0.adr.actuators, 0].copy()
    kp1 = w.model.actuator_gainprm[d1.adr.actuators, 0].copy()
    assert (kp0 > 0).all()
    for _ in range(50):
        for d in (d0, d1):
            d.set_cmd(w.data, [0, 0, 0])
        w.step()
    assert w.start_skill(d0, "kick_left")
    np.testing.assert_allclose(w.model.actuator_gainprm[d0.adr.actuators, 0], STANDING_GAIN_RATIO * kp0)
    np.testing.assert_allclose(w.model.actuator_biasprm[d0.adr.actuators, 1], -STANDING_GAIN_RATIO * kp0)
    np.testing.assert_array_equal(w.model.actuator_gainprm[d1.adr.actuators, 0], kp1)   # the other duck: untouched
    steps = 0
    while d0.skill is not None:
        w.step()
        steps += 1
    assert abs(steps * C.CTRL_DT - KICK_S) <= C.CTRL_DT + 1e-9
    np.testing.assert_array_equal(w.model.actuator_gainprm[d0.adr.actuators, 0], kp0)
    np.testing.assert_array_equal(w.model.actuator_biasprm[d0.adr.actuators, 1], -kp0)
    # A respawn mid-window (a fall) restores it too.
    assert w.start_skill(d0, "kick_right")
    w.reset_duck("d0")
    assert d0.skill is None and d0.gain_ratio == 1.0
    np.testing.assert_array_equal(w.model.actuator_gainprm[d0.adr.actuators, 0], kp0)
