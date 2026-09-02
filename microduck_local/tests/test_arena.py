"""World stepping locks (roadmap 0.2): a one-duck world reproduces the walk
env's inference-time behaviour step for step, N ducks share one mjData, a
fallen duck respawns and counts, and the shipped walker stays up in a room."""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv
from microduck_local.world import Ball, Duck, Scenario, Wall, World, make_room

pytestmark = pytest.mark.skipif(
    not C.SCENE_WALK_XML.exists(), reason="microduck_rl checkout not found")

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"


def onnx_infer(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path))
    name = sess.get_inputs()[0].name
    return lambda obs: sess.run(None, {name: obs[None]})[0][0].astype(np.float32)


def sync_env_to_world(env: MicroduckWalkEnv, world: World, duck_id: str) -> None:
    """Put the reference env in the world duck's exact state."""
    d = world.ducks[duck_id]
    q, v = d.adr.root_qpos, d.adr.root_qvel
    env.data.qpos[0:7] = world.data.qpos[q:q + 7]
    env.data.qpos[env.joint_qpos_adr] = world.data.qpos[d.adr.joint_qpos]
    env.data.qvel[:] = 0.0
    env.data.qvel[0:6] = world.data.qvel[v:v + 6]
    env.data.qvel[env.joint_qvel_adr] = world.data.qvel[d.adr.joint_qvel]
    env.data.ctrl[:] = world.data.ctrl[d.adr.actuators]
    mujoco.mj_forward(env.model, env.data)
    env.prev_joint_vel = env._joint_vel().copy()
    env.last_action = d.last_action.copy()
    env.twist_cmd[:] = d.twist_cmd
    env.head_cmd[:] = d.head_cmd
    env.body_cmd[:] = d.body_cmd


def test_one_duck_world_matches_walk_env_step_for_step():
    world = World(Scenario(name="one", ducks=[Duck("d0", (0.2, -0.1, 0.7), None, None)]))
    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False, action_delay=False,
                           random_yaw=False, seed=0)
    env.reset(seed=0)
    sync_env_to_world(env, world, "d0")
    d = world.ducks["d0"]
    rng = np.random.default_rng(3)
    actions = rng.uniform(-0.3, 0.3, (40, C.NUM_JOINTS)).astype(np.float32)
    seen: list[np.ndarray] = []      # what the world's policy was handed
    k = [0]

    def scripted(obs):
        seen.append(obs.copy())
        a = actions[k[0]]
        k[0] += 1
        return a
    d.infer = scripted
    d.set_cmd(world.data, [0.0, 0.0, 0.0])
    obs_env = None
    for i in range(40):
        world.step()
        if obs_env is not None:
            # The obs the world's policy saw at step i is the obs the env
            # RETURNED from step i-1 (the one its policy would act on).
            np.testing.assert_allclose(seen[i], obs_env, atol=1e-6, err_msg=f"obs at step {i}")
        obs_env, *_ = env.step(actions[i])
        np.testing.assert_allclose(world.data.qpos[d.adr.joint_qpos], env.data.qpos[env.joint_qpos_adr],
                                   atol=1e-9, err_msg=f"qpos at step {i}")
        np.testing.assert_allclose(d.trunk_pos(world.data), env.data.xpos[env.trunk_body_id], atol=1e-9)
    # …and the lag is real: the joint_vel block is one step behind the truth.
    assert not np.allclose(seen[-1][20:34], d.joint_vel(world.data), atol=1e-4)
    assert d.step_count == 40 and d.falls == 0


def test_fallen_duck_respawns_and_counts():
    world = World(Scenario(name="fall", ducks=[Duck("d0", (0.0, 0.0, 0.0), None, None)]))
    d = world.ducks["d0"]
    # Shove it over: a big sideways velocity on the root.
    world.data.qvel[d.adr.root_qvel:d.adr.root_qvel + 3] = [0.0, 3.0, 0.0]
    for _ in range(100):
        world.step()
        if d.falls:
            break
    assert d.falls == 1 and d.episodes == 2
    np.testing.assert_allclose(d.trunk_pos(world.data)[:2], [0.0, 0.0], atol=0.02)
    assert d.step_count < 100 and d.tof is None


def test_episode_timeout_respawns_without_a_fall():
    world = World(Scenario(name="t", ducks=[Duck("d0", (0, 0, 0), None, None)]), max_episode_s=0.2)
    d = world.ducks["d0"]
    for _ in range(12):
        world.step()
    assert d.episodes == 2 and d.falls == 0 and d.step_count == 2


def test_duck_bodies_are_one_contiguous_slice_in_scene_order():
    world = World(Scenario(name="two", ducks=[Duck("a", (0, 0, 0), None, None),
                                              Duck("b", (0.5, 0, 0), None, None)]))
    ref = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    ref_names = [ref.body(b).name for b in range(1, ref.nbody)]
    for did in ("a", "b"):
        s = world.duck_bodies[did]
        names = [world.model.body(b).name for b in range(s.start, s.stop)]
        assert names == [f"{did}/{n}" for n in ref_names]
        pose = world.duck_pose(did)
        assert len(pose) == ref.nbody - 1 and len(pose[0]) == 7


def test_payloads_carry_objects_and_tof():
    sc = Scenario(name="p", floor=(5, 5), walls=[Wall((1.0, -2), (1.0, 2), 0.6)],
                  balls=[Ball((0.5, 0.3))],
                  ducks=[Duck("d0", (0, 0, 0), None, "ideal")])
    world = World(sc)
    objs = world.objects_payload()
    assert [o["id"] for o in objs] == ["ball0"] and objs[0]["kind"] == "ball"
    assert abs(objs[0]["pose"][2] - 0.035) < 1e-3
    assert world.sensors_payload("d0") is None      # nothing sampled yet
    world.step()
    s = world.sensors_payload("d0")
    # Sampled right after the first control step: taken at t = 0.02, age 0.
    assert s is not None and len(s["tof"]["mm"]) == 64
    assert s["tof"]["t"] == pytest.approx(0.02) and s["tof"]["age"] == 0.0
    assert max(s["tof"]["mm"]) > 0
    world.step()
    assert world.sensors_payload("d0")["tof"]["age"] == pytest.approx(0.02)
    # 15 Hz: the next frame is due at 1/15 s and lands on the first tick after.
    for _ in range(3):
        world.step()
    assert world.sensors_payload("d0")["tof"]["t"] == pytest.approx(0.08)


def test_heading_hold_command_closes_the_loop_on_yaw():
    world = World(Scenario(name="h", ducks=[Duck("d0", (0, 0, 0.3), None, None)]))
    d = world.ducks["d0"]
    d.set_cmd(world.data, [0.2, 0.0, 0.0])
    assert d.twist_cmd[2] == 0.0                     # anchored at the current yaw
    # Pretend the duck yawed +0.1 rad since: the hold steers back.
    q = d.adr.root_qpos
    yaw = 0.4
    world.data.qpos[q + 3:q + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(world.model, world.data)
    d.set_cmd(world.data, [0.2, 0.0, 0.0])
    assert d.twist_cmd[2] == pytest.approx(-0.4, abs=1e-4)
    d.set_cmd(world.data, [0.2, 0.0, 0.5])          # an explicit turn wins
    assert d.twist_cmd[2] == pytest.approx(0.5) and d._hold_yaw is None


@pytest.mark.skipif(not (POLICIES / "alpha_walking.onnx").exists(),
                    reason="upstream microduck/policies not checked out")
def test_shipped_walker_walks_in_a_world():
    infer = onnx_infer(POLICIES / "alpha_walking.onnx")
    # Open floor, three ducks abreast, all asked for a 0.3 m/s walk (alpha treats 0.15 as "stand"): they must
    # stay up and cover ground (alpha delivers about half its command).
    sc = Scenario(name="abreast", floor=(8, 8),
                  ducks=[Duck(f"d{i}", (0.0, 0.6 * i - 0.6, 0.0), None, None) for i in range(3)])
    world = World(sc, infer_for={d.id: infer for d in sc.ducks})
    for d in world.ducks.values():
        d.set_cmd(world.data, [0.3, 0.0, 0.0])
    speeds = {d.id: [] for d in world.ducks.values()}
    for _ in range(int(4.0 / C.CTRL_DT)):
        world.step()
        for d in world.ducks.values():
            speeds[d.id].append(d.heading_speed(world.data))
    for d in world.ducks.values():
        assert d.falls == 0, d.id
        assert np.mean(speeds[d.id][50:]) > 0.03, (d.id, np.mean(speeds[d.id][50:]))
    # And in a cluttered room they may bump into things, but they do not fall.
    room = make_room(seed=3, n_boxes=2, n_ducks=3)
    world = World(room, infer_for={d.id: infer for d in room.ducks})
    for d in world.ducks.values():
        d.set_cmd(world.data, [0.3, 0.0, 0.0])
    for _ in range(int(3.0 / C.CTRL_DT)):
        world.step()
    assert sum(d.falls for d in world.ducks.values()) == 0


def test_persons_walk_their_path_and_ducks_detect_them():
    from microduck_local.world import Person
    sc = Scenario(name="pp", floor=(8, 8),
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal", "ideal")],
                  persons=[Person("p0", (1.0, 0.0), 0.0, path=[(1.0, 0.6), (1.0, -0.6)], speed=0.5)])
    world = World(sc)
    p = world.persons["p0"]
    assert world.objects == [] and world.persons_payload()[0]["kind"] == "person"
    ys, seen = [], []
    for _ in range(int(3.0 / C.CTRL_DT)):
        world.step()
        ys.append(p.y)
        s = world.sensors_payload("d0")
        if s and "det" in s:
            seen += [x for x in s["det"]["items"] if x["cls"] == "person"]
    assert max(ys) > 0.5 and min(ys) < 0.4          # went up to the first waypoint and came back
    # Seen while it crossed the field of view, at a bearing that tracked it.
    assert len(seen) > 5 and max(abs(x["bearing"]) for x in seen) < 0.6
    assert all(0.5 < x["range"] < 1.6 for x in seen)
    # Possess: the path stops and the twist drives it in its own heading frame.
    world.possess("p0")
    p.yaw = 0.0                                   # face +x: the twist is in the person's own frame
    p.cmd = np.array([0.5, 0.0, 0.0])
    x0 = p.x
    for _ in range(50):
        world.step()
    assert p.possessed and p.x > x0 + 0.4
    world.possess(None)
    assert not p.possessed and p.cmd is None
    world.reset()
    assert (p.x, p.y) == (1.0, 0.0) and world.ducks["d0"].detector.last is None
