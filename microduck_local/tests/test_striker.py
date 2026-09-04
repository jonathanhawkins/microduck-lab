"""The learned striker (roadmap 4.4): the observation contract, the reflex
tier's kick option, the reward's sign, the spawn ladder, and the head-to-head
harness — including that it reproduces `eval-pitch` exactly when both sides
run the scripted `chase`, which is what makes the baseline arm the published
baseline and not a re-implementation of it."""

import json
import math

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import REGISTRY
from microduck_local.brain.brain_env import BRAIN_OBS_DIM, ObsBuilder
from microduck_local.brain.runtime import Senses
from microduck_local.brain.striker import (
    KICK_AHEAD,
    KICK_COOLDOWN_S,
    KICK_GIVE_UP_S,
    KICK_ON,
    KICK_SIDE,
    S_ACT_HIGH,
    S_ACT_LOW,
    STRIKER_ACT_DIM,
    STRIKER_OBS_DIM,
    LearnedStriker,
    StrikerEnv,
    StrikerSenses,
    StrikerTask,
    kick_from,
    striker_obs,
)
from microduck_local.sensors.detector import Detection, DetectionFrame

POLICIES = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies"
pytestmark = pytest.mark.skipif(
    not (POLICIES / "alpha_walking.onnx").exists(), reason="upstream checkouts not found")


# --- the observation ----------------------------------------------------------
def test_striker_obs_is_the_brain_contract_plus_the_goal_geometry():
    """The first 80 floats ARE the shared brain observation, untouched; the
    eight after them are the body-frame geometry a brain reads off odometry.
    A reward for taking the ball to the goal is only legitimate because
    these exist (AGENTS.md: never pay for what the policy cannot see)."""
    base = np.arange(BRAIN_OBS_DIM, dtype=np.float32)
    # Standing at the origin facing +x; goal 1.5 m ahead; ball 0.5 m ahead,
    # 0.2 m to the left.
    o = striker_obs(base, (0.0, 0.0, 0.0), (1.5, 0.0), (0.5, 0.2), False)
    assert o.shape == (STRIKER_OBS_DIM,) and o.dtype == np.float32
    np.testing.assert_array_equal(o[:BRAIN_OBS_DIM], base)
    assert o[80] == pytest.approx(0.0, abs=1e-6) and o[81] == pytest.approx(1.0)   # goal dead ahead
    assert o[82] == pytest.approx(1.5 / 4.0)
    assert o[85] == pytest.approx(0.5) and o[86] == pytest.approx(0.2)             # ball, body frame
    # The ball -> goal line points ahead and to the right of the body.
    assert o[83] < 0 and o[84] > 0
    assert o[87] == 0.0

    # Same scene from a body yawed 90 deg: the goal is now to the RIGHT.
    o2 = striker_obs(base, (0.0, 0.0, math.pi / 2), (1.5, 0.0), (0.5, 0.2), True)
    assert o2[80] == pytest.approx(-1.0) and o2[81] == pytest.approx(0.0, abs=1e-6)
    assert o2[85] == pytest.approx(0.2) and o2[86] == pytest.approx(-0.5)
    assert o2[87] == 1.0

    # No track at all: the ball slots are zero, the goal slots are not.
    o3 = striker_obs(base, (0.0, 0.0, 0.0), (1.5, 0.0), None, False)
    assert (o3[83:87] == 0).all() and o3[81] == pytest.approx(1.0)


def test_the_ball_slots_are_dead_reckoned_and_survive_the_ball_leaving_the_camera():
    """The last 0.3 m onto the ball is blind — a floor ball is under the
    camera — so slots 85/86 come from the track's ODOMETRY position and the
    duck's own, not from a stale bearing. Walk 0.4 m forward with no new
    detection and the ball must be 0.4 m nearer, not unchanged."""
    ss = StrikerSenses((1.5, 0.0))
    det = DetectionFrame(t=1.0, detections=[Detection("ball", "ball0", 0.0, -0.3, 0.05, 0.6, 0.9)])
    a = np.zeros(3, np.float32)
    o1 = ss.observe(Senses(t=1.0, det=det, det_age=0.0, speed=0.2, odom=(0.0, 0.0, 0.0)), a)
    assert o1[85] == pytest.approx(0.6, abs=0.05) and abs(o1[86]) < 0.05
    o2 = ss.observe(Senses(t=1.4, det=det, det_age=0.4, speed=0.2, odom=(0.4, 0.0, 0.0)), a)
    assert o2[85] == pytest.approx(0.2, abs=0.05)          # 0.4 m nearer, with no new frame
    assert o2[65] == 0.0 and o2[77] == 1.0                 # …and the track says it is coasting


def test_the_brain_observation_is_unchanged_by_the_trackers_odometry_placement():
    """`senses_to_obs` now hands the tracker the body position so tracks
    carry `xy`. That must not move any of the 80 floats a follow brain
    reads — it only adds fields nothing there looks at."""
    det = DetectionFrame(t=1.0, detections=[Detection("person", "p0", 0.2, -0.1, 0.3, 1.4, 0.9)])
    a = np.array([0.1, 0.0, -0.2], np.float32)
    s = Senses(t=1.05, det=det, det_age=0.05, speed=0.12, odom=(2.0, -1.0, 0.5))
    b = ObsBuilder("person", version=2)
    o = b(s, a)
    np.testing.assert_allclose(o[65:71], [1, 0.2, -0.1, 0.3, 1.4, 0.9])
    assert b.tracker.best("person", 1.05, min_hits=1).xy is not None    # the new field is populated
    assert o[78] == 0.0 and o[79] == 0.0                    # first frame: no yaw rate, unconfirmed


# --- the hierarchical head ----------------------------------------------------
def test_kick_from_picks_the_larger_logit_over_the_threshold():
    assert kick_from([0, 0, 0, 0.0, 0.0]) is None
    assert kick_from([0, 0, 0, KICK_ON - 0.01, KICK_ON - 0.01]) is None
    assert kick_from([0, 0, 0, 1.0, 0.6]) == "kick_left"
    assert kick_from([0, 0, 0, 0.6, 1.0]) == "kick_right"


def test_the_kick_is_an_option_the_reflex_tier_executes_not_an_instant_action():
    """A kick request LATCHES: the tier zeroes the twist, waits for the body
    to stand, then fires. That is what makes a kick reachable by exploration
    (a brain that had to hold a still command by chance never kicked), and
    it is why a kick is not fired mid-stride, which is the fall mode."""
    ss = StrikerSenses((1.5, 0.0))
    walk_and_kick = np.array([0.5, 0.0, 0.0, 1.0, -1.0], np.float32)
    twist, skill = ss.act(walk_and_kick, Senses(t=0.0, speed=0.2))
    assert skill is None and twist == (0.0, 0.0, 0.0)       # latched: the tier stops the body
    assert ss.pending == "kick_left"
    # Still moving at the settle time: no swap.
    assert ss.act(walk_and_kick, Senses(t=0.35, speed=0.2))[1] is None
    # Stopped and settled: it fires, once, and the latch clears.
    assert ss.act(walk_and_kick, Senses(t=0.7, speed=0.01))[1] == "kick_left"
    assert ss.pending is None
    # While the skill runs the tier owns the body and takes no new request.
    assert ss.act(walk_and_kick, Senses(t=0.8, speed=0.0, skill="kick_left")) == ((0.0, 0.0, 0.0), None)


def test_the_tier_refuses_to_swap_again_inside_the_cooldown():
    """The gate that broke the kick-spam lock. Without it a brain that always
    asks for a kick is STUCK: settle 0.3 s + kick 0.5 s back to back, never a
    step walked (measured at 75k decisions: 514 kicks over 4 x 120 s, zero
    possession, zero ball advance). The cooldown is a reflex-tier knob, not a
    reward term — the scripted brain kicks 0.03 times a second and never
    touches it."""
    ss = StrikerSenses((1.5, 0.0))
    a = np.array([0.0, 0.0, 0.0, 1.0, -1.0], np.float32)
    ss.act(a, Senses(t=0.0, speed=0.0))
    assert ss.act(a, Senses(t=0.4, speed=0.0))[1] == "kick_left"
    ss.act(a, Senses(t=0.5, speed=0.0, skill="kick_left"))       # the cycle runs
    # Straight after the cycle: the request is refused and the body is FREE.
    twist, skill = ss.act(np.array([0.5, 0.0, 0.0, 1.0, -1.0], np.float32), Senses(t=0.6, speed=0.0))
    assert skill is None and ss.pending is None and twist[0] == pytest.approx(0.5)
    # …and once the cooldown is up it takes one again.
    t = 0.5 + KICK_COOLDOWN_S + 0.05
    ss.act(a, Senses(t=t, speed=0.0))
    assert ss.pending == "kick_left"
    assert ss.act(a, Senses(t=t + 0.4, speed=0.0))[1] == "kick_left"


def test_a_latched_kick_that_cannot_settle_is_dropped():
    ss = StrikerSenses((1.5, 0.0))
    a = np.array([0.0, 0.0, 0.0, 1.0, 0.0], np.float32)
    ss.act(a, Senses(t=0.0, speed=0.5))
    assert ss.pending == "kick_left"
    ss.act(a, Senses(t=KICK_GIVE_UP_S + 0.2, speed=0.5))    # never stopped
    assert ss.pending is None


def test_the_reflex_tier_refuses_to_walk_into_something():
    """`ChaseParams.tof_stop` as a reflex: a forward command with a
    body-height return inside the stop distance becomes a stand."""
    from microduck_local.sensors.tof import TofFrame
    ss = StrikerSenses((1.5, 0.0))
    depth = np.full((8, 8), 2000.0)
    frame = TofFrame(t=1.0, depth_mm=depth.copy(), valid=np.ones((8, 8), bool))
    fwd = np.array([0.5, 0.0, 0.0, -1.0, -1.0], np.float32)
    assert ss.act(fwd, Senses(t=1.0, tof=frame, tof_age=0.0, speed=0.2))[0][0] == pytest.approx(0.5)
    depth[2:6, 3:5] = 150.0                                  # a board right there
    close = TofFrame(t=1.1, depth_mm=depth, valid=np.ones((8, 8), bool))
    assert ss.act(fwd, Senses(t=1.1, tof=close, tof_age=0.0, speed=0.2))[0][0] == 0.0


# --- the env ------------------------------------------------------------------
def test_env_reset_step_and_bounds():
    env = StrikerEnv(StrikerTask(episode_s=2.0), seed=0)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (STRIKER_OBS_DIM,)
    assert env.action_space.shape == (STRIKER_ACT_DIM,)
    np.testing.assert_array_equal(env.action_space.low, S_ACT_LOW)
    np.testing.assert_array_equal(env.action_space.high, S_ACT_HIGH)
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(STRIKER_ACT_DIM, np.float32))
        n += 1
        assert np.isfinite(obs).all() and math.isfinite(r)
        if term or trunc:
            break
    assert n == env.max_decisions == round(2.0 / C.CTRL_DT / 5)


def test_an_episode_is_a_pure_function_of_its_seed():
    """The battery rests on it, exactly as the follow env's does: nothing
    rides in from the episode before."""
    env = StrikerEnv(StrikerTask(episode_s=1.5), seed=0)
    a = np.array([0.3, 0.0, 0.2, -1.0, -1.0], np.float32)

    def roll(seed):
        env.reset(seed=seed)
        return [float(env.step(a)[1]) for _ in range(env.max_decisions)]

    first = roll(11)
    roll(12)
    assert roll(11) == first


def test_the_reward_is_signed_ball_progress_toward_the_goal():
    """Pay the ball's displacement toward the goal, and CHARGE the other
    way. The signed form is the point: `ballAdvance` keeps only the forward
    part and is inflated by churn, so a reward shaped like it would be
    farmed by knocking the ball about."""
    env = StrikerEnv(StrikerTask(episode_s=2.0, w_approach=0.0, jerk_penalty=0.0), seed=0)
    env.reset(seed=5)
    stand = np.zeros(STRIKER_ACT_DIM, np.float32)
    w = env.world
    j = w._ball_joint
    q, v = int(w.model.jnt_qposadr[j]), int(w.model.jnt_dofadr[j])

    def shove(dx):
        w.data.qpos[q] += dx
        w.data.qvel[v:v + 6] = 0.0
        env._prev_ball = (w.data.qpos[q] - dx, w.data.qpos[q + 1])
        return env.step(stand)[1]

    forward = shove(0.10)
    back = shove(-0.10)
    assert forward > 0 and back < 0
    assert forward == pytest.approx(-back, rel=0.2)
    assert forward == pytest.approx(env.task.w_progress * 0.10, rel=0.35)


def test_the_spawn_ladder_puts_the_ball_on_a_kicking_foots_sweet_spot():
    """The physics ladder, not a reward term: with `spot_prob` 1 every
    episode starts on the measured sweet spot (0.08 m ahead, 0.06 m to a
    foot's side), which is what makes the value of a kick learnable at all.
    `spot_prob` 0 never does."""
    env = StrikerEnv(StrikerTask(spot_prob=1.0, near_prob=0.0), seed=1)
    for ep in range(6):
        env.reset(seed=200 + ep)
        p = env.duck.trunk_pos(env.world.data)
        bx, by = env.world.ball_xy()
        yaw = env.duck.yaw(env.world.data)
        ahead = (bx - p[0]) * math.cos(yaw) + (by - p[1]) * math.sin(yaw)
        side = -(bx - p[0]) * math.sin(yaw) + (by - p[1]) * math.cos(yaw)
        assert ahead == pytest.approx(KICK_AHEAD, abs=0.05)
        assert abs(side) == pytest.approx(KICK_SIDE, abs=0.04)

    far = StrikerEnv(StrikerTask(spot_prob=0.0, near_prob=0.0), seed=1)
    dists = []
    for ep in range(6):
        far.reset(seed=300 + ep)
        dists.append(far._ball_dist())
    assert min(dists) > 0.3


def test_the_scripted_chase_reaches_the_ball_and_outscores_a_standing_duck():
    """A sanity floor for the reward: the brain this one has to beat is paid
    by it, and doing nothing is not. Read as a BATTERY, not per episode —
    over eight episodes the scripted chase's returns here ran from +9.7 to
    -6.4 (it kicks the ball the wrong way often enough), which is the same
    lesson AGENTS.md's "How much can the benchmark actually resolve?"
    teaches about the pitch: quote a sum over seeds, never one run."""
    from microduck_local.render_striker import action_of
    env = StrikerEnv(StrikerTask(spot_prob=0.0, near_prob=1.0, episode_s=12.0), seed=7)
    chase = REGISTRY.make("chase", goal=env.goal, duck_id="d0", bounds=env.bounds, goal_w=0.7)

    def roll(step_fn, seed):
        env.reset(seed=seed)
        chase.reset()
        total, closest = 0.0, env._ball_dist()
        while True:
            _, r, term, trunc, info = env.step(step_fn())
            total += r
            closest = min(closest, info["ball_dist"])
            if term or trunc:
                break
        return total, closest

    scored, idle, reach, stay = 0.0, 0.0, [], []
    for ep in range(8):
        r, d = roll(lambda: action_of(chase.step(env.senses())), 400 + ep)
        scored += r
        reach.append(d)
        r, d = roll(lambda: np.zeros(STRIKER_ACT_DIM, np.float32), 400 + ep)
        idle += r
        stay.append(d)
    # `reach` is the CLOSEST the duck came to the ball, not where it ended:
    # after a kick the ball is 2 m away through no fault of the brain.
    assert np.mean(reach) < 0.3 < np.mean(stay)      # it walks to the ball; a standing duck does not
    assert scored > idle


# --- the head-to-head harness -------------------------------------------------
def test_the_chase_arm_reproduces_eval_pitch_exactly():
    """The scripted arm of a striker battery must BE `eval-pitch`, field for
    field, or the comparison is against a different benchmark."""
    from microduck_local.eval_pitch import run_one as pitch_run
    from microduck_local.eval_striker import run_one as striker_run
    want = pitch_run(0, 12.0, 1)
    got = striker_run(0, 12.0, "chase", "chase", 1, False)
    for k, v in want.items():
        assert got[k] == v, k


def test_solo_drops_the_opponent_but_keeps_the_pitch():
    from microduck_local.eval_striker import pitch_scenario
    full = pitch_scenario(1, solo=False)
    solo = pitch_scenario(1, solo=True)
    assert [d.id for d in full.ducks] == ["d0", "d1"]
    assert [d.id for d in solo.ducks] == ["d0"]
    assert solo.floor == full.floor and solo.goal_width == full.goal_width
    assert [(w.start, w.end) for w in solo.walls] == [(w.start, w.end) for w in full.walls]


def test_side_reading_quotes_advance_progress_and_advance_per_kick_together():
    """`ballAdvance` alone is inflatable by churn (eval_pitch's docstring):
    the reader must always get the signed progress and the per-kick number
    beside it, and the event counts under them."""
    from microduck_local.eval_striker import side_reading
    rows = [{"ballAdvance": {"left": 1.0, "right": 0.5}, "ballProgress": {"left": -0.2, "right": 0.1},
             "possession": {"left": 10.0, "right": 5.0}, "kicks": {"d0": 4, "d1": 2},
             "falls": {"d0": 1, "d1": 0}, "left": 3, "right": 2, "simSeconds": 60.0,
             "team": {"d0": "left", "d1": "right"}}]
    s = side_reading(rows, "left")
    assert s["advance"] == 1.0 and s["progress"] == -0.2
    assert s["kicks"] == 4 and s["falls"] == 1
    assert s["goals"] == 2                       # the LEFT team attacks +x, whose mouth is "right"
    assert s["advPerKick"] == pytest.approx(1.0 / 4)


# --- the round trip -----------------------------------------------------------
def test_a_tiny_ppo_round_trips_through_onnx_into_the_pitch(tmp_path, monkeypatch):
    """train-brain --task striker → brain.onnx → a LearnedStriker playing on
    the pitch, kicking through `Intent.skill`."""
    import subprocess
    import sys
    monkeypatch.setenv("MICRODUCK_BRAINS_DIR", str(tmp_path))
    out = subprocess.run(
        [sys.executable, "-m", "microduck_local.train_brain", "--task", "striker",
         "--run-name", "t", "--envs", "2", "--steps", "128", "--n-steps", "16",
         "--episode-s", "3"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    meta = json.loads((tmp_path / "t" / "brain.json").read_text())
    assert meta["task"] == "striker" and meta["obs_dim"] == STRIKER_OBS_DIM
    assert len(meta["act_low"]) == STRIKER_ACT_DIM

    b = LearnedStriker("t", goal=(1.5, 0.0), duck_id="d0", bounds=(1.5, 1.25), goal_w=0.7,
                       brains_root=tmp_path)
    env = StrikerEnv(StrikerTask(episode_s=3.0), seed=0)
    env.reset(seed=1)
    b.reset()
    intents = []
    for _ in range(env.max_decisions):
        i = b.step(env.senses())
        intents.append(i)
        _, _, term, trunc, _ = env.step(np.array([*i.twist, 1.0 if i.skill == "kick_left" else -1.0,
                                                  1.0 if i.skill == "kick_right" else -1.0], np.float32))
        if term or trunc:
            break
    assert all(S_ACT_LOW[k] - 1e-5 <= i.twist[k] <= S_ACT_HIGH[k] + 1e-5
               for i in intents for k in range(3))
    assert all(i.skill in (None, "kick_left", "kick_right") for i in intents)
