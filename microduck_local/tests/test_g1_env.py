"""Training the Unitree G1 in this harness: the scene, the env, the goldens.

The load-bearing test here is `test_shipped_walker_walks_in_the_env`. The G1
env speaks the 99-d LuckyRobots observation on purpose, so the policy that
layout was trained for is a GOLDEN: if `walker.onnx` walks in our scene, the
scene, the obs assembly and the action scaling are right. If it does not,
every reward number measured here would have been noise.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from microduck_local.robots import g1
from microduck_local.robots.g1_env import G1SquatEnv, G1StandEnv, G1WalkEnv

pytestmark = pytest.mark.skipif(
    not g1.g1_ready(), reason="G1 assets missing — uv run fetch-g1")


# --------------------------------------------------------------- the model

def _model(freeze: bool = True):
    import mujoco
    return mujoco.MjModel.from_xml_path(str(g1.g1_scene_xml(freeze)))


def test_freezing_removes_the_finger_dofs_and_nothing_else():
    """Freeze, don't delete: the fingers stop costing solver time, and the
    robot keeps its mass, its inertia and its meshes (measured: deleting
    them costs 0.65 kg of forearm and buys 0.03x — see robots/g1.py)."""
    frozen, free = _model(True), _model(False)
    assert free.nv - frozen.nv == g1.NUM_HAND_JOINTS == 14
    assert free.nu - frozen.nu == 14
    assert frozen.nv == 35 and frozen.nu == g1.NUM_JOINTS == 29
    # kept: bodies, meshes, total mass, per-body mass
    assert frozen.nbody == free.nbody
    assert frozen.nmesh == free.nmesh
    assert frozen.ngeom == free.ngeom
    assert abs(float(frozen.body_mass.sum()) - float(free.body_mass.sum())) < 1e-9
    assert np.allclose(frozen.body_mass, free.body_mass)


def test_the_stand_keyframe_stands_on_the_floor():
    """Measured off the model, never hand-carried: the lowest foot capsule
    sits just above the floor at the default pose."""
    import mujoco
    m = _model()
    d = mujoco.MjData(m)
    assert m.nkey == 1
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
           for names in g1.FOOT_GEOMS.values() for n in names]
    low = min(float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in ids)
    assert 0.0 < low < 0.005, f"lowest foot capsule at {low:.4f} m"
    assert 0.70 < float(d.qpos[2]) < 0.80
    # upright, and the joints at the config's default pose
    assert np.allclose(d.qpos[3:7], [1, 0, 0, 0])
    qadr = [m.joint(n).qposadr[0] for n in g1.joint_names()]
    assert np.allclose(d.qpos[qadr], g1.default_pose(), atol=1e-6)


# ------------------------------------------------------------------- env

def _env(**kw) -> G1WalkEnv:
    kw.setdefault("obs_noise", False)
    kw.setdefault("domain_rand", False)
    kw.setdefault("action_delay", False)
    kw.setdefault("random_yaw", False)
    kw.setdefault("max_episode_s", 60.0)
    return G1WalkEnv(seed=0, **kw)


def _stand_env(**kw):
    """The idle's env (robots/g1_env.G1StandEnv), randomizers off."""
    from microduck_local.robots.g1_env import G1StandEnv
    kw.setdefault("obs_noise", False)
    kw.setdefault("domain_rand", False)
    kw.setdefault("action_delay", False)
    kw.setdefault("random_yaw", False)
    return G1StandEnv(seed=0, **kw)


def _walker():
    s = ort.InferenceSession(str(g1.walker_onnx()), providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def _drive(env, cmd, steps, policy=None):
    """Run `steps` control steps at a pinned command; return (path, fell)."""
    sess_in = policy or _walker()
    sess, inp = sess_in
    env.twist_cmd[:] = cmd
    obs = env._get_obs()
    prev = env._trunk_xpos[:2].copy()
    path = 0.0
    for i in range(steps):
        act = sess.run(None, {inp: obs.reshape(1, -1)})[0][0].astype(np.float32)
        obs, _r, term, _tr, _info = env.step(act)
        env.twist_cmd[:] = cmd          # pin: the env resamples every 5 s
        now = env._trunk_xpos[:2].copy()
        path += float(np.linalg.norm(now - prev))
        prev = now
        if term:
            return path, True, i + 1
    return path, False, steps


def test_shipped_walker_walks_in_the_env():
    """GOLDEN. 0.6 m/s commanded for 8 s: it walks, it tracks, it stays up."""
    env = _env()
    env.reset(seed=0)
    path, fell, steps = _drive(env, (0.6, 0.0, 0.0), 400)
    assert not fell, "the shipped walker fell in our scene"
    speed = path / (steps * 0.02)
    assert path > 2.0, f"walked only {path:.2f} m"
    assert 0.4 < speed < 0.8, f"tracked {speed:.2f} m/s against a 0.6 command"
    assert float(env._trunk_xpos[2]) > g1.FALL_HEIGHT


def test_the_walker_tracks_the_command_it_is_given():
    env = _env()
    speeds = {}
    for cmd in (0.5, 0.8):
        env.reset(seed=0)
        path, fell, steps = _drive(env, (cmd, 0.0, 0.0), 300)
        assert not fell
        speeds[cmd] = path / (steps * 0.02)
    assert speeds[0.8] > speeds[0.5] + 0.15, speeds


def test_the_shipped_walker_has_a_dead_zone_below_the_command_floor():
    """Measured, and the reason `min_forward_cmd` exists: under ~0.4 m/s the
    shipped policy stands. A local policy has to learn what it never did —
    the trainer's *forward* orders stay inside the regime a gait exists in."""
    env = _env()
    env.reset(seed=0)
    slow, fell, _ = _drive(env, (0.25, 0.0, 0.0), 300)
    assert not fell
    assert slow < 0.5, f"expected the stall, walked {slow:.2f} m"
    assert g1.G1_SPEC.min_forward_cmd >= 0.4


def test_freezing_the_hands_keeps_the_walk():
    """The fidelity half of the freeze decision: the same policy, the same
    command, the frozen body and the fingers-free one walk the same."""
    paths = {}
    for freeze in (True, False):
        env = _env(scene_xml=str(g1.g1_scene_xml(freeze)))
        env.reset(seed=0)
        p, fell, steps = _drive(env, (0.6, 0.0, 0.0), 300)
        assert not fell, f"fell with freeze={freeze}"
        paths[freeze] = p
    rel = abs(paths[True] - paths[False]) / paths[False]
    assert rel < 0.15, f"frozen {paths[True]:.2f} m vs free {paths[False]:.2f} m"


# ------------------------------------------------------------- obs / action

def test_obs_is_the_luckyrobots_layout():
    """Re-derived independently from mjData — a reordered block here would
    feed the policy someone else's joints."""
    env = _env()
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(29, np.float32))
    obs = env._get_obs()
    m, d = env.model, env.data
    rq, rv = env._root_qpos, env._root_qvel
    quat = d.qpos[rq + 3:rq + 7]

    def qinv(q, v):
        w, xyz = q[0], q[1:4]
        t = np.cross(xyz, v) * 2
        return v - w * t + np.cross(xyz, t)

    qadr = [m.joint(n).qposadr[0] for n in g1.joint_names()]
    vadr = [m.joint(n).dofadr[0] for n in g1.joint_names()]
    assert obs.shape == (99,) == (g1.OBS_DIM,)
    assert np.allclose(obs[0:3], qinv(quat, d.qvel[rv:rv + 3]), atol=1e-6)
    assert np.allclose(obs[3:6], d.qvel[rv + 3:rv + 6], atol=1e-6)
    assert np.allclose(obs[6:9], qinv(quat, np.array([0.0, 0.0, -1.0])), atol=1e-6)
    assert np.allclose(obs[9:38], d.qpos[qadr] - g1.default_pose(), atol=1e-6)
    assert np.allclose(obs[38:67], d.qvel[vadr], atol=1e-6)
    assert np.allclose(obs[67:96], env.last_action, atol=1e-7)
    assert np.allclose(obs[96:99], env.twist_cmd, atol=1e-7)


def test_actions_reach_the_joint_they_name():
    """The 29 actuators are addressed BY NAME. The G1 XML carries finger DoFs
    the walk policy does not own, so a positional `7+i` mapping fed the right
    arm the left hand (robots/g1.py) — this is that class of bug, in ctrl."""
    import mujoco
    env = _env()
    env.reset(seed=0)
    a = np.zeros(29, np.float32)
    j = g1.joint_names().index("left_elbow_joint")
    a[j] = 0.5
    env.step(a)
    target = g1.default_pose() + a * g1.action_scales()
    aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_joint")
    assert abs(float(env.data.ctrl[aid]) - float(target[j])) < 1e-6
    # and every other actuator sits at its own default
    for k, n in enumerate(g1.joint_names()):
        if k == j:
            continue
        aidk = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        assert abs(float(env.data.ctrl[aidk]) - float(g1.default_pose()[k])) < 1e-6


def test_action_scale_is_the_shipped_per_joint_one():
    s = g1.G1_SPEC
    assert s.action_scale is not None and s.action_scale.shape == (29,)
    a = np.full(29, 0.3, np.float32)
    assert np.allclose(s.scale_action(a), g1.default_pose() + 0.3 * g1.action_scales())


# ------------------------------------------------------- rewards / episodes

def test_reward_has_no_head_term_and_penalties_are_negative():
    env = _env()
    env.reset(seed=0)
    for _ in range(30):
        env.step(np.zeros(29, np.float32))
    sums = env.reward_sums
    assert "head_pose" not in sums
    for k, v in sums.items():
        if k.endswith("_penalty"):
            assert v <= 0.0, f"{k} paid {v}"


def test_a_fallen_g1_terminates():
    import mujoco
    env = _env()
    env.reset(seed=0)
    env.data.qpos[env._root_qpos + 2] = 0.30      # pelvis on the floor
    mujoco.mj_forward(env.model, env.data)
    _o, _r, term, _tr, _i = env.step(np.zeros(29, np.float32))
    assert term


def test_bam_is_refused_for_the_g1():
    """BAM is an XL330 identification — it does not describe this robot."""
    with pytest.raises(ValueError, match="XL330"):
        G1WalkEnv(seed=0, actuator="bam")


def test_env_uses_the_frozen_scene_and_the_spec_by_default():
    env = _env()
    assert env.robot.id == "g1"
    assert env.model.nv == 35 and env.model.nu == 29
    assert env.observation_space.shape == (99,)
    assert env.action_space.shape == (29,)
    assert env.nj == 29


def test_the_air_time_window_pays_on_real_swings():
    """A reward term that can never fire is worse than no term: the window
    this env started with was GUESSED from the duck's ([0.25, 0.60] s) and
    paid on 0.0% of the shipped walker's swings — a 1.3 m humanoid swings
    FASTER than the duck's numbers assume (median 0.16 s at 0.5 m/s, 0.20 s
    at 0.8). This pins the window against the gait it is supposed to price."""
    import numpy as np

    env = _env()
    env.reset(seed=0)
    sess, inp = _walker()
    env.twist_cmd[:] = (0.6, 0.0, 0.0)
    obs = env._get_obs()
    air = {"left": 0.0, "right": 0.0}
    swings: list[float] = []
    for _ in range(800):
        act = sess.run(None, {inp: obs.reshape(1, -1)})[0][0].astype(np.float32)
        obs, _r, term, _tr, _i = env.step(act)
        env.twist_cmd[:] = (0.6, 0.0, 0.0)
        assert not term
        contacts = env._foot_contacts()
        for side in ("left", "right"):
            if contacts[side]:
                if air[side] > 0.0:
                    swings.append(air[side])
                air[side] = 0.0
            else:
                air[side] += 0.02
    assert len(swings) > 40, f"only {len(swings)} swings — it is not walking"
    paid = [s for s in swings if env.AIR_TIME_MIN < s < env.AIR_TIME_MAX]
    assert len(paid) / len(swings) > 0.5, (
        f"only {100 * len(paid) / len(swings):.0f}% of swings fall in "
        f"[{env.AIR_TIME_MIN}, {env.AIR_TIME_MAX}] — median swing "
        f"{float(np.median(swings)):.3f} s")
    # …and the one-control-step blips must NOT pay (the duck's rule).
    assert env.AIR_TIME_MIN > 0.02


def test_the_idle_prices_rocking_and_wandering_not_just_speed():
    """The two terms that killed the sway.

    Scoring only body VELOCITY leaves a sway that averages out free, and
    nothing at all priced the ankles a standing biped rocks on. Measured:
    adding these cut the settled path from 26.3 cm per 10 s to 1.9 cm and
    took survival under noise+DR from 6.5 s to 40 s."""
    import mujoco

    env = _stand_env()
    env.reset(seed=0)
    _r, terms = env._compute_reward()
    assert "flat_feet" in terms and "stay_home" in terms
    # standing square on the spot: both terms are at (or near) full value
    assert terms["flat_feet"] > 0.9 * env.W_FLAT_FEET
    assert terms["stay_home"] > 0.9 * env.W_ANCHOR

    # …and they are EARNERS: bounded above by their weight, never negative,
    # so neither can take the reward over the way a ramped penalty did.
    for key in ("flat_feet", "stay_home"):
        assert 0.0 <= terms[key] <= env.W_FLAT_FEET + 1e-9

    # stay_home is measured from where the EPISODE started, not the origin
    env.reset(seed=1)
    env.data.qpos[env._root_qpos] += 0.30          # shove it 30 cm downrange
    mujoco.mj_forward(env.model, env.data)
    env._refresh_derived()
    moved = env._compute_reward()[1]["stay_home"]
    assert moved < 0.2 * env.W_ANCHOR, "wandering off should cost nearly all of it"
    env.reset(seed=1)
    assert abs(float(env._spawn_xy[0]) - float(env._trunk_xpos[0])) < 1e-9


def test_flat_feet_notices_a_rolled_ankle():
    """A sole tipped onto its edge must score less than one lying flat."""
    import mujoco
    import numpy as np

    env = _stand_env()
    env.reset(seed=0)
    flat = env._compute_reward()[1]["flat_feet"]
    roll = env.model.joint("left_ankle_roll_joint")
    env.data.qpos[roll.qposadr[0]] = 0.25          # ~14 deg onto the edge
    mujoco.mj_forward(env.model, env.data)
    env._refresh_derived()
    rolled = env._compute_reward()[1]["flat_feet"]
    assert rolled < flat, f"rolled {rolled:.3f} should score below flat {flat:.3f}"
    assert np.isfinite(rolled)


# --- the squat: the three things that were wrong, pinned ------------------

def test_the_squat_spawn_pose_shortens_the_LEGS_not_the_root():
    """The solver must bisect on leg clearance, not on pelvis height.

    The pelvis IS the free-joint root, so `data.xpos[trunk][2]` equals
    `qpos[2]` and no joint angle can move it. Bisecting on it solved against
    a constant, saturated at the 2.0 rad search bound, and handed back a fold
    the robot could not stand up from.
    """
    e = G1SquatEnv(seed=0, obs_noise=False, domain_rand=False)
    assert 0.0 < e.squat_knee_rad < 1.99, "solver saturated at the search bound"
    # the solved pose really is `squat_drop_m` shorter in the legs
    assert e.squat_clearance == pytest.approx(
        e.stand_z - e.squat_drop_m, abs=0.01)
    # and the spawn stands the soles ON the floor at that clearance
    for seed in range(12):
        e.reset(seed=seed)
        if e.last_spawn == "squat":
            assert float(e._trunk_xpos[2]) == pytest.approx(
                e.target_height(), abs=0.02)
            break
    else:
        pytest.fail("no squat spawn in 12 resets at prob 0.5")


def test_the_squat_pose_term_is_retargeted_not_deleted():
    """W_POSE=0 left NOTHING specifying the configuration, only "be lower" —
    and the cheapest way to drop the pelvis 20 cm is to pitch over the
    ankles. Measured: 2 of 3 deterministic seeds ended face-down inside 2 s
    with the trunk up-axis at 0.19-0.27 (1.0 is vertical), while the height
    term was paying 5.42 of 6.0. The term must pay MORE at the squat pose
    than at the standing pose."""
    import mujoco
    e = G1SquatEnv(seed=0, obs_noise=False, domain_rand=False)
    assert e.W_POSE > 0.0, "the pose term is the only thing pricing the SHAPE"

    def pose_at(q):
        e.reset(seed=0)
        e.data.qpos[e.joint_qpos_adr] = q
        mujoco.mj_forward(e.model, e.data)
        e._refresh_derived()
        return e._compute_reward()[1]["pose"]

    assert pose_at(e._squat_joints) > pose_at(e.default_pose) + 0.3

    # the STAND env keeps paying for the standing pose — not retargeted
    s = G1StandEnv(seed=0, obs_noise=False, domain_rand=False)
    assert float(np.asarray(s.pose_target_rel()).sum()) == 0.0


def test_the_g1_idle_and_squat_are_not_shoved():
    """AGENTS.md: "Velocity pushes are part of the walk env's DR and OFF in
    every behavior recipe. Turning them on for a trick is an experiment to
    name and measure, not a fix." These two reach the env through train.py
    rather than BehaviorEnv, so they were inheriting the WALK env's DR and
    being shoved while their reward scored `still` and `stay_home`."""
    from microduck_local import train as T
    for task in ("stand", "squat"):
        env = T.env_class("g1", task)(
            **T.env_kwargs_from_args(T.parse_args(["--robot", "g1", "--task", task])))
        assert env.push_robot is False, f"the g1 {task} recipe is being shoved"
    walk = T.env_class("g1", "walk")(
        **T.env_kwargs_from_args(T.parse_args(["--robot", "g1", "--task", "walk"])))
    assert walk.push_robot is True, "the WALK env keeps its push DR"


def test_the_squat_pays_for_a_TIDY_CARRIAGE_not_just_a_low_pelvis():
    """`pose` spreads one Gaussian over 29 joints at std2 4.0 and holds
    nothing. The first squat that met its height and uprightness targets was
    measured at waist_roll +29.8 deg, waist_pitch +29.9, right_shoulder_roll
    -29.1 against left -9.5 — it squatted correctly and looked wrong: back
    twisted, one arm folded onto the thigh. `carriage` prices the waist and
    arms tightly enough to matter, without pinning the legs it balances with.
    """
    import mujoco
    e = G1SquatEnv(seed=0, obs_noise=False, domain_rand=False)
    assert e.W_CARRIAGE > 0.0
    # only joints ABOVE the hips — the legs stay free to balance
    names = [e.robot.joint_names[i] for i in e._carriage_ids]
    assert names and all(
        any(k in n for k in ("waist", "shoulder", "elbow", "wrist")) for n in names)
    assert not any("hip" in n or "knee" in n or "ankle" in n for n in names)

    def carriage_at(q):
        e.reset(seed=0)
        e.data.qpos[e.joint_qpos_adr] = q
        mujoco.mj_forward(e.model, e.data)
        e._refresh_derived()
        return e._compute_reward()[1]["carriage"]

    clean = carriage_at(e._squat_joints)
    twisted = e._squat_joints.copy()
    n = list(e.robot.joint_names)
    for nm, d in (("waist_roll_joint", 0.52), ("waist_pitch_joint", 0.52),
                  ("right_shoulder_roll_joint", -0.51)):      # the measured pose
        twisted[n.index(nm)] += d
    assert clean == pytest.approx(e.W_CARRIAGE, abs=1e-3)
    assert carriage_at(twisted) < 0.15 * clean, "the twisted carriage barely loses anything"


def test_the_idle_keeps_its_carriage_weight_off():
    """The shipped idle was trained before the term existed, so turning it on
    here would silently stop its recipe describing its policy."""
    s = G1StandEnv(seed=0, obs_noise=False, domain_rand=False)
    assert s.W_CARRIAGE == 0.0
    s.reset(seed=0)
    assert s._compute_reward()[1]["carriage"] == 0.0
