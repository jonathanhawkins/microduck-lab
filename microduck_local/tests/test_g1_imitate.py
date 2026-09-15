"""Motion imitation on the G1: robots/g1_imitate.G1ImitateEnv.

The clip under test is AUTHORED HERE with the IK solver (a weight shift,
a chamber, a kick) into a temporary clips dir, so the tests own their
fixture and also exercise the clip → reference → reward path end to end.
Every reward test is a retarget check in the AGENTS.md sense: the term pays
MORE at the clip's own pose than away from it, and pays SOMETHING away from
it (a flat term teaches nothing).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from microduck_local import motion
from microduck_local.pose import IkTarget, pose_scratch
from microduck_local.robots import g1

pytestmark = pytest.mark.skipif(
    not g1.g1_ready(), reason="G1 assets missing — uv run fetch-g1")


@pytest.fixture(scope="module")
def kick_clip(tmp_path_factory):
    """A three-key kick authored with the IK solver, saved as a G1 clip."""
    ps = pose_scratch("g1")
    q0 = ps.spec.default_pose.astype(np.float64)
    ps.solve(q0, 0.0)
    left = np.array(ps.effector_point("left_foot"))
    right = np.array(ps.effector_point("right_foot"))
    shift = ps.solve_ik(q0, 0.0, {"com": IkTarget(pos=(left[0], left[1], 0.0), weight=2.0)},
                        pins=("left_foot", "right_foot"))
    assert shift.converged, shift.residual
    ps.solve(shift.joints, 0.0)
    left = np.array(ps.effector_point("left_foot"))
    kick = ps.solve_ik(shift.joints, 0.0, {
        "right_foot": IkTarget(pos=(right[0] + 0.45, right[1], right[2] + 0.5)),
        "com": IkTarget(pos=(left[0], left[1], 0.0), weight=2.0),
    }, pins=("left_foot",))
    assert kick.converged, kick.residual
    keys = [(0.0, q0), (0.4, shift.joints), (0.8, kick.joints), (1.2, shift.joints), (1.6, q0)]
    d = tmp_path_factory.mktemp("clips")
    (d / "test-kick.json").write_text(json.dumps({
        "version": 1, "name": "test-kick", "robot": "g1", "duration": 2.0, "loop": True,
        "keys": [{"t": t, "joints": [float(v) for v in q], "rootPitch": 0.0} for t, q in keys],
    }))
    (d / "duck-hop.json").write_text(json.dumps({
        "version": 1, "name": "duck-hop", "duration": 1.0, "loop": False,
        "keys": [{"t": 0.0, "joints": [0.0] * 14, "rootPitch": 0.0}],
    }))
    return d


@pytest.fixture
def clips(kick_clip, monkeypatch):
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", str(kick_clip))
    return kick_clip


def _env(**kw):
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    kw.setdefault("obs_noise", False)
    kw.setdefault("domain_rand", False)
    kw.setdefault("max_episode_s", 5.0)
    return G1ImitateEnv(clip_name="test-kick", **kw)


# ------------------------------------------------------------- the clip

def test_a_clip_carries_its_robot_and_the_right_joint_count(clips):
    c = motion.load_clip("test-kick")
    assert c.robot == "g1" and c.num_joints == 29 and c.loop
    assert c.steps == 100                       # 2.0 s at 50 Hz
    d = motion.load_clip("duck-hop")
    assert d.robot == "microduck" and d.num_joints == 14


def test_a_clip_with_the_wrong_joint_count_for_its_robot_is_refused(clips, tmp_path):
    (clips / "bad.json").write_text(json.dumps({
        "version": 1, "name": "bad", "robot": "g1", "duration": 1.0,
        "keys": [{"t": 0.0, "joints": [0.0] * 14, "rootPitch": 0.0}]}))
    with pytest.raises(ValueError, match="29 joints for the g1"):
        motion.load_clip("bad")


def test_the_env_refuses_a_duck_clip_and_needs_a_clip_at_all(clips, monkeypatch):
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    with pytest.raises(ValueError, match="poses the microduck"):
        G1ImitateEnv(clip_name="duck-hop")
    monkeypatch.delenv("MICRODUCK_CLIP", raising=False)
    with pytest.raises(ValueError, match="needs a clip"):
        G1ImitateEnv()


def test_the_clip_can_come_through_the_environment_like_the_ducks_do(clips, monkeypatch):
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    monkeypatch.setenv("MICRODUCK_CLIP", "test-kick")
    env = G1ImitateEnv(obs_noise=False, domain_rand=False)
    assert env.clip.name == "test-kick"


# -------------------------------------------------------- the reference

def test_reference_is_grounded_and_reads_the_lifted_foot(clips):
    env = _env()
    # Standing frames: pelvis at its standing clearance, both feet planted.
    assert env.ref_height[0] == pytest.approx(env.stand_z, abs=0.02)
    assert env.ref_planted["left"].all()
    assert env.ref_planted["right"][0] and not env.ref_planted["right"][40]
    # The kick's apex lifts the right foot about half a metre.
    assert env.ref_foot_air["right"].max() > 0.4
    assert env.ref_foot_air["left"].max() < 0.02
    # Foot positions are in the PELVIS frame: the lifted foot is forward+up.
    apex = int(np.argmax(env.ref_foot_air["right"]))
    rel = env.ref_foot_rel["right"][apex]
    assert rel[0] > 0.3 and rel[2] > -0.4
    # Zero rootPitch clips lean nothing: reference gravity points down.
    assert np.allclose(env.ref_gravity[:, 2], -1.0, atol=1e-6)


# --------------------------------------------------------------- the obs

def test_the_command_slots_carry_the_clip_phase(clips):
    env = _env()
    obs, _ = env.reset(seed=0)
    s, c = env.clip.phase(env.clip_step())
    assert obs[-3:] == pytest.approx([s, c, 0.0], abs=1e-6)
    assert s * s + c * c == pytest.approx(1.0)
    obs, *_ = env.step(np.zeros(29, np.float32))
    s2, c2 = env.clip.phase(env.clip_step())
    assert obs[-3:] == pytest.approx([s2, c2, 0.0], abs=1e-6)
    assert (s2, c2) != (s, c)


def test_spawns_inside_the_clip_are_posed_to_the_clip_with_soles_on_the_floor(clips):
    env = _env()
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        seen.add(env.last_spawn)
        if env.last_spawn != "in-clip":
            continue
        q, _ = env.clip.at(env._phase0)
        assert np.allclose(env._joint_qpos(), q, atol=1e-6)
        low = min(float(env.data.geom_xpos[g][2] - env.model.geom_size[g][0])
                  for ids in env.foot_geom_ids.values() for g in ids)
        assert 0.0 < low < 0.01                # on the floor, not through it
    assert seen == {"standing", "in-clip"}


# ------------------------------------------------------------ the reward

def _terms_at(env, frame: int, joints=None):
    """Reward terms with the body TELEPORTED to `joints` at clip frame
    `frame` (velocity zero), measured without stepping."""
    import mujoco
    env.reset(seed=1)
    env._phase0 = frame
    env.step_count = 0
    q, _ = env.clip.at(frame) if joints is None else (joints, 0.0)
    d = env.data
    d.qpos[env.joint_qpos_adr] = q
    mujoco.mj_forward(env.model, d)
    low = min(float(d.geom_xpos[g][2] - env.model.geom_size[g][0])
              for ids in env.foot_geom_ids.values() for g in ids)
    # A hair INTO the floor: a sole floating 2 mm above it makes no contact,
    # and the contact half of `feet` would read as if the foot were lifted.
    d.qpos[env._root_qpos + 2] += -0.0005 - low
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    env._refresh_derived()
    env._spawn_xy = np.asarray(env._trunk_xpos[:2], float).copy()
    _, terms = env._compute_reward()
    return terms


def test_every_earner_pays_more_at_the_clip_pose_than_standing_through_the_kick(clips):
    """The retarget check: at the kick's apex, the clip's own pose beats the
    standing pose on every pose-shaped term, and standing still pays
    SOMETHING (not flat) on each of them."""
    env = _env()
    apex = int(np.argmax(env.ref_foot_air["right"]))
    on = _terms_at(env, apex)
    off = _terms_at(env, apex, joints=env.default_pose.astype(np.float64))
    for k in ("pose", "carriage", "foot_track", "feet"):
        assert on[k] > off[k], (k, on[k], off[k])
        assert off[k] > 0.05, f"{k} is flat at the standing pose ({off[k]})"
    # At the apex the clip lifts the right foot: standing earns only the
    # planted half of `feet`; the posed body earns nearly all of it.
    assert on["feet"] > 0.9 * env.W_FEET
    assert off["feet"] < 0.6 * env.W_FEET
    assert on["foot_track"] > 0.9 * env.W_FOOT_TRACK
    # ...and the retargeted terms are near-full at the pose they ask for.
    assert on["height"] > 0.9 * env.W_HEIGHT
    assert on["upright"] > 0.9 * env.W_UPRIGHT


def test_lift_is_progress_pay_not_a_gaussian(clips):
    """Half the lift earns about half the credit — a gradient from zero,
    where the Gaussian this harness kept reaching for is flat (roadmap 13.3)."""
    env = _env()
    apex = int(np.argmax(env.ref_foot_air["right"]))
    ps = pose_scratch("g1")
    ps.solve(env.default_pose.astype(np.float64), 0.0)
    r = np.array(ps.effector_point("right_foot"))
    want = float(env.ref_foot_air["right"][apex])
    half = ps.solve_ik(env.default_pose.astype(np.float64), 0.0,
                       {"right_foot": IkTarget(pos=(r[0] + 0.1, r[1], r[2] + want / 2))},
                       pins=("left_foot",))
    assert half.converged
    t_half = _terms_at(env, apex, joints=half.joints)
    t_none = _terms_at(env, apex, joints=env.default_pose.astype(np.float64))
    t_full = _terms_at(env, apex)
    assert t_none["feet"] < t_half["feet"] < t_full["feet"]
    # left foot planted (0.5 + flat 0.5) + right lift fraction, over 2 feet
    lift_half = (t_half["feet"] / env.W_FEET) * 2 - 1.0
    assert 0.3 < lift_half < 0.7


def test_a_standing_frame_of_the_clip_is_the_idle_reward(clips):
    """Frame 0 is the standing pose: the retargeted terms reduce to the idle's
    — both feet planted and flat, pelvis at standing height."""
    env = _env()
    t = _terms_at(env, 0)
    assert t["feet"] > 0.95 * env.W_FEET
    assert t["height"] > 0.95 * env.W_HEIGHT
    assert t["carriage"] > 0.95 * env.W_CARRIAGE
    assert t["foot_track"] > 0.95 * env.W_FOOT_TRACK
    assert "feet_planted" not in t and "flat_feet" not in t


def test_episodes_run_and_a_fall_ends_them(clips):
    env = _env(max_episode_s=3.0)
    env.reset(seed=4)
    n = 0
    done = False
    while not done and n < 200:
        _, r, term, trunc, _ = env.step(np.zeros(29, np.float32))
        assert np.isfinite(r)
        done = term or trunc
        n += 1
    assert done


def test_standing_through_the_kick_ends_the_episode_like_a_fall(clips, monkeypatch):
    """DeepMimic's early termination: a body that does not lift when the clip
    lifts is cut off after LIFT_MISS_STEPS such frames, so standing still is
    no longer the safe policy (the first run stood through every kick, apex
    0.000 m). Judged on a TELEPORTED standing body (zero actions collapse
    this robot on their own, which would end the episode as a fall instead):
    the rule counts only frames that ask for a lift, an achieved lift resets
    it, and a LOOSE env never applies it."""
    env = _env(max_episode_s=10.0)
    apex = int(np.argmax(env.ref_foot_air["right"]))
    lift_frames = [i for i in range(env.clip.steps)
                   if env.ref_foot_air["right"][i] >= env.LIFT_MIN_REF]
    assert len(lift_frames) >= 5
    _terms_at(env, 0)                        # standing, both soles down
    env._miss = 0
    # Frames asking for no lift: nothing counted.
    for _ in range(5):
        env._phase0, env.step_count = 0, 0
        assert not env.not_tracking() and env._miss == 0
    # Standing through lift frames counts, and trips after LIFT_MISS_STEPS.
    fired = None
    for k in range(env.LIFT_MISS_STEPS + 3):
        env._phase0, env.step_count = lift_frames[k % len(lift_frames)], 0
        if env.not_tracking():
            fired = k
            break
    assert fired == env.LIFT_MISS_STEPS - 1
    # A frame asking for nothing in between neither counts nor forgives.
    env._miss = 3
    env._phase0, env.step_count = 0, 0
    env.not_tracking()
    assert env._miss == 3
    # An achieved lift forgives: pose the body AT the apex frame.
    _terms_at(env, apex)
    env._miss = 7
    env._phase0, env.step_count = apex, 0
    assert not env.not_tracking() and env._miss == 0
    # And `step` reports it as a termination of its own kind: one step from
    # a standing teleport at a lift frame with the count one short.
    _terms_at(env, lift_frames[0])
    env.data.qpos[env.joint_qpos_adr] = env.default_pose
    env.data.qvel[:] = 0.0
    env._write_ctrl(env.default_pose)
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env._refresh_derived()
    env._phase0, env.step_count = lift_frames[0], 0
    env._miss = env.LIFT_MISS_STEPS - 1
    _, _, term, _, info = env.step(np.zeros(29, np.float32))
    assert term and env.last_termination == "not-tracking"
    assert "episode_rewards" in info
    # The loose env keeps the old rule.
    monkeypatch.setenv("MICRODUCK_G1_IMITATE_LOOSE", "1")
    loose = _env(max_episode_s=10.0)
    assert not loose.track_termination
    _terms_at(loose, lift_frames[0])
    loose.data.qpos[loose.joint_qpos_adr] = loose.default_pose
    mujoco.mj_forward(loose.model, loose.data)
    loose._refresh_derived()
    loose._phase0, loose.step_count = lift_frames[0], 0
    loose._miss = loose.LIFT_MISS_STEPS + 5
    _, _, term, _, _ = loose.step(np.zeros(29, np.float32))
    assert not term


def test_the_lift_rung_comes_from_the_stage_env_and_reaches_the_preview(clips, monkeypatch):
    """A curriculum stage raises LIFT_MIN_GOT through MICRODUCK_G1_LIFT_MIN;
    the lab's trainee preview gets the same rung as a kwarg."""
    from microduck_local import behaviors as B
    from microduck_local import viz_server as V
    assert _env().LIFT_MIN_GOT == 0.05
    assert _env(lift_min_got=0.3).LIFT_MIN_GOT == 0.3
    monkeypatch.setenv("MICRODUCK_G1_LIFT_MIN", "0.15")
    assert _env().LIFT_MIN_GOT == 0.15
    b = B.BEHAVIORS["g1_imitate"]
    rungs = [float(st.env["MICRODUCK_G1_LIFT_MIN"]) for st in b.curriculum]
    assert rungs == sorted(rungs) and rungs[0] == 0.05 and rungs[-1] >= 0.3
    kw = V.trainee_env_kwargs(b, {"MICRODUCK_CLIP": "test-kick",
                                  "MICRODUCK_G1_LIFT_MIN": "0.15"})
    assert kw["lift_min_got"] == 0.15


def test_a_frame_with_no_lift_asked_never_counts_as_missed(clips):
    env = _env()
    env.reset(seed=3)
    env._phase0 = 0                  # standing frames: nothing to lift
    env.step_count = 0
    for _ in range(30):
        _, _, term, trunc, _ = env.step(np.zeros(29, np.float32))
        assert not term
    assert env._miss == 0


# ---------------------------------------------------------- the plumbing

def test_the_trainer_and_the_teach_panel_know_the_task(clips):
    from microduck_local import behaviors as B
    from microduck_local import viz_server as V
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    from microduck_local.train import G1_TASKS, HOLD_TASKS, env_class
    assert "imitate" in G1_TASKS and "imitate" in HOLD_TASKS
    assert env_class("g1", "imitate") is G1ImitateEnv
    b = B.BEHAVIORS["g1_imitate"]
    assert b.robot == "g1" and b.task == "imitate"
    assert B.match_behavior("copy the animation", "g1") is b
    assert B.match_behavior("copy the animation", "microduck").id == "imitate"
    assert V.imitation_behavior("g1") is b
    assert V.match_teach_text('Perform "test-kick"', "g1") == (b, "test-kick")


def test_the_trainee_preview_tracks_the_same_clip(clips):
    from microduck_local import behaviors as B
    from microduck_local import viz_server as V
    kw = V.trainee_env_kwargs(B.BEHAVIORS["g1_imitate"], {"MICRODUCK_CLIP": "test-kick"})
    assert kw["task"] == "imitate" and kw["clip_name"] == "test-kick"
    assert "clip_name" not in V.trainee_env_kwargs(B.BEHAVIORS["g1_imitate"], {})


def test_train_args_carry_the_clip_into_the_env(clips, monkeypatch):
    from microduck_local.train import env_kwargs_from_args, parse_args
    kw = env_kwargs_from_args(parse_args(
        ["--robot", "g1", "--task", "imitate", "--clip", "test-kick"]))
    assert kw["clip_name"] == "test-kick"
    monkeypatch.setenv("MICRODUCK_CLIP", "test-kick")
    kw = env_kwargs_from_args(parse_args(["--robot", "g1", "--task", "imitate"]))
    assert kw["clip_name"] == "test-kick"
    monkeypatch.delenv("MICRODUCK_CLIP")
    with pytest.raises(SystemExit, match="needs --clip"):
        env_kwargs_from_args(parse_args(["--robot", "g1", "--task", "imitate"]))
