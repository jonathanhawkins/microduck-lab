"""World stepping locks (roadmap 0.2): a one-duck world reproduces the walk
env's inference-time behaviour step for step, N ducks share one mjData, a
fallen duck respawns and counts, and the shipped walker stays up in a room."""

import math

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.sensors.detector import DetectorSpec
from microduck_local.walk_env import MicroduckWalkEnv
from microduck_local.world import Ball, Duck, Scenario, Wall, World, make_room
from microduck_local.world.compose import scene_model

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
    ref = scene_model()   # the model GET /scene serves, mouth body and all
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
    #
    # The bound comes from the SPEC's own half-FOV rather than a constant: it was
    # 0.6 rad, which was silently the 62 deg camera's half-angle, and it failed
    # the day the default became the robot's real 116 deg lens — where a person
    # at 0.90 rad is correctly seen, not wrongly. A test that pins a detection
    # geometry to a number pins the camera too.
    half_h = math.radians(DetectorSpec().fov_h_deg) / 2
    assert len(seen) > 5 and max(abs(x["bearing"]) for x in seen) < half_h
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


def test_the_walker_is_bit_identical_under_all_and_walk_on_a_flat_floor():
    """`collision="all"` (the default) changes what a duck can TOUCH, not how
    it walks: the shipped walker at 0.3 m/s with a shove, every qpos equal
    to the bit between the two robot variants (10 s x 3 seeds and a turn,
    measured). Two things had to be pinned for that - the variants' hand-
    rounded `<inertial>` values (`compose._pin_mass_properties_to_walk`)
    and the shoe shell's floor contact - or they drifted 0.2-0.5 rad
    apart from a 1e-9 seed."""
    infer = onnx_infer(POLICIES / "alpha_walking.onnx")

    def run(collision, seconds=4.0):
        sc = Scenario(name="w", floor=(20, 20), ducks=[Duck("d0", (0, 0, 0), None, None, None)],
                      collision=collision)
        w = World(sc, infer_for={"d0": infer}, seed=0)
        d = w.ducks["d0"]
        d.set_cmd(w.data, (0.3, 0.0, 0.0))
        qs = []
        for k in range(int(seconds / C.CTRL_DT)):
            w.data.xfrc_applied[d.adr.trunk_body, :2] = (1.2, -0.8) if 50 <= k < 60 else (0.0, 0.0)
            w.step()
            qs.append(w.data.qpos.copy())
        return w.model, np.array(qs), d.falls

    ma, qa, fa = run("walk")
    mb, qb, fb = run("all")
    assert fa == fb == 0 and mb.ngeom > ma.ngeom
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(ma, f), getattr(mb, f), err_msg=f)
    np.testing.assert_array_equal(qa, qb)


def test_a_bump_that_lasts_one_substep_is_sensed():
    """`World._sense_bumps` reads the contact list after EVERY substep of a
    tick. Two ducks stood just touching, one flung sideways at 2 m/s (1 cm
    a substep): the contact exists in the tick's first substep only - the
    tick ends with no duck-duck pair in the list - and both are `bumped`.
    (Reading the last substep alone, as before, missed it.)"""
    sc = Scenario(name="bb", floor=(4, 4), ducks=[Duck("d0", (0.0, 0.0, 0.0), None, None, None),
                                                   Duck("d1", (0.0, 0.2, 0.0), None, None, None)])
    w = World(sc)
    m, d = w.model, w.data
    d0, d1 = w.ducks["d0"], w.ducks["d1"]

    def duck_duck_pairs() -> int:
        n = 0
        for c in range(d.ncon):
            b1 = m.body(m.geom_bodyid[d.contact.geom1[c]]).name
            b2 = m.body(m.geom_bodyid[d.contact.geom2[c]]).name
            n += b1.startswith("d0/") and b2.startswith("d1/") or b1.startswith("d1/") and b2.startswith("d0/")
        return n

    # Slide d1 in beside d0 until the first contact appears.
    q1 = d1.adr.root_qpos
    y = 0.2
    while y > 0.0 and duck_duck_pairs() == 0:
        y -= 0.001
        d.qpos[q1 + 1] = y
        mujoco.mj_forward(m, d)
    assert 0.02 < y < 0.2 and duck_duck_pairs() > 0, y
    assert not w.bumped(d0) and not w.bumped(d1)
    d.qvel[d1.adr.root_qvel + 1] = 2.0
    w.step()
    assert duck_duck_pairs() == 0                     # gone by the tick's last substep
    assert w.bumped(d0) and w.bumped(d1)


def test_the_mouth_is_shut_at_spawn_and_opens_only_for_the_work():
    """The 15th servo (roadmap 12.13). A duck's beak is SHUT at rest — the
    first cut armed the drop-open window inside `release`, which spawning
    calls on every duck to clear its hands, so a fresh room was full of ducks
    gaping for MOUTH_DROP_S."""
    from microduck_local.world.arena import MOUTH_DROP_S
    from microduck_local.world.compose import mouth_frac_for_gape
    from microduck_local.world.scenario import PICKABLE_KINDS, Pickable

    sc = Scenario(name="m", ducks=[Duck("d0", (0, 0, 0), None, None)],
                  pickables=[Pickable("t0", "brick", (0.25, 0.0)),
                             Pickable("t1", "block", (0.0, 0.25))])
    w = World(sc)
    d = w.ducks["d0"]
    for _ in range(20):
        w.step()
    assert d.mouth == 0.0, "a duck at rest has its beak shut"

    # Releasing nothing is not a drop, so it must not open the beak.
    assert w.release(d) is None and d.mouth_open_until == 0.0

    # A held toy closes the bill as far as the toy allows: a 40 mm block is
    # wider than the 36 mm gape and holds it wide, a 10 mm brick nearly shuts.
    for toy, kind in (("t1", "block"), ("t0", "brick")):
        d.holding = toy
        w._mouth(d)
        assert d.mouth == pytest.approx(
            mouth_frac_for_gape(min(PICKABLE_KINDS[kind]["size"])), abs=1e-9)
    assert w.ducks["d0"].mouth < 0.5, "the brick lets the bill close"
    d.holding = None

    # A real drop opens it, and only for as long as the window.
    d.holding, d.beak_closed = "t0", True
    w.data.eq_active[w._eq_id(d, "t0")] = 1
    assert w.release(d) == "t0"
    w._mouth(d)
    assert d.mouth == 1.0
    w.t += MOUTH_DROP_S + 1e-6
    w._mouth(d)
    assert d.mouth == 0.0
