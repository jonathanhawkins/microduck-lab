"""Behavior library: envs build and step cleanly, recipes obey the sign rules,
the chat matcher finds the right trick."""

import numpy as np
import pytest

from microduck_local.behaviors import _RUN_STAGE_SCALE  # noqa: F401
from microduck_local.behaviors import BEHAVIORS, BehaviorEnv, behavior_card, match_behavior


@pytest.mark.parametrize("behavior_id", sorted(BEHAVIORS))
def test_behavior_env_steps_clean(behavior_id):
    env = BehaviorEnv(behavior_id, obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (61,)
    # Command slots: a TRICK pins them to zero, while a locomotion behavior
    # legitimately carries its commanded forward speed there (that command is
    # the whole difference between imitating a gait and being asked to run).
    cmd = BEHAVIORS[behavior_id].forward_cmd
    if cmd:
        # GPU run mix: standing (vx=vy=wz=0), 55% forward (vy=wz=0, vx>=0.3),
        # remainder omni (vy in ±0.3, wz in ±1). Ceiling starts at 0.4.
        assert abs(float(obs[48])) <= cmd + 1e-6
        assert abs(float(obs[49])) <= 0.3 + 1e-6
        assert abs(float(obs[50])) <= 1.0 + 1e-6
    else:
        np.testing.assert_allclose(obs[48:51], np.zeros(3, np.float32), atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(
            rng.uniform(-0.3, 0.3, 14).astype(np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=1)


@pytest.mark.parametrize("behavior_id", sorted(BEHAVIORS))
def test_penalty_terms_stay_negative(behavior_id):
    env = BehaviorEnv(behavior_id, obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        _, _, terminated, truncated, _ = env.step(
            rng.uniform(-0.5, 0.5, 14).astype(np.float32))
        if terminated or truncated:
            env.reset(seed=3)
    for name, total in env.reward_sums.items():
        if name.endswith("_penalty"):
            assert total <= 1e-9, f"{name} accumulated {total:+.4f} (> 0)"


def test_positive_terms_are_bounded_per_step():
    """No jackpots: every non-penalty term pays at most its weight per step."""
    env = BehaviorEnv("one_leg", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(np.zeros(14, np.float32))
        if terminated or truncated:
            env.reset(seed=0)
    weights = {t.key: t.weight for t in BEHAVIORS["one_leg"].terms if not t.is_penalty}
    steps = 30
    for key, w in weights.items():
        assert env.reward_sums.get(key, 0.0) <= w * steps + 1e-6


def test_matcher():
    assert match_behavior("please stand on 1 leg").id == "one_leg"
    assert match_behavior("can you balance on one foot?").id == "one_leg"
    assert match_behavior("crouch down").id == "crouch"
    assert match_behavior("do a spin!").id == "spin"
    assert match_behavior("do a back flip").id == "backflip"
    # "stand still" must not be stolen by — or steal from — one_leg.
    assert match_behavior("stand still").id == "stand"
    assert match_behavior("just stand there and hold still").id == "stand"
    assert match_behavior("stand on one leg").id == "one_leg"
    assert match_behavior("balance on one foot").id == "one_leg"
    assert match_behavior("flip backwards for me").id == "backflip"
    # The raw jumping flip must not be confused with the staged floor roll.
    assert match_behavior("do a jump backflip").id == "airflip"
    assert match_behavior("flip in the air").id == "airflip"
    assert match_behavior("do a back flip").id == "backflip"
    assert match_behavior("wave hello") is None


def test_cards_are_json_friendly():
    import json
    for b in BEHAVIORS.values():
        card = behavior_card(b)
        json.dumps(card)
        assert card["terms"] and card["howItLearns"] and card["emoji"]


def _backflip_env():
    env = BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    return env


def test_backflip_rotation_integrates_backward_only():
    """The state hook credits BACKWARD pitch (negative gyro_y) and clips:
    forward rotation can't bank negative progress."""
    import mujoco
    env = _backflip_env()
    env._bf_rot = 0.0
    env.data.qvel[:] = 0.0
    env.data.qvel[4] = -3.0  # backward spin (verified sign convention)
    mujoco.mj_forward(env.model, env.data)
    BEHAVIORS["backflip"].state_fn(env)
    assert env._bf_rot > 0.03
    env._bf_rot = 0.0
    env.data.qvel[4] = +3.0  # forward spin banks nothing
    mujoco.mj_forward(env.model, env.data)
    BEHAVIORS["backflip"].state_fn(env)
    assert env._bf_rot == 0.0


def test_backflip_landed_hold_needs_a_completed_flip():
    """No free money at spawn: the upright start pays landed_hold only once
    the full rotation has actually been credited."""
    from microduck_local.behaviors import _bf_landed_hold
    env = _backflip_env()
    for seed in range(20):  # find an ordinary UPRIGHT spawn (families are random)
        env.reset(seed=seed)
        if env._bf_rot == 0.0:
            break
    else:
        pytest.fail("no upright spawn in 20 seeds")
    for _ in range(10):  # settle the drop-in until both feet touch
        env.step(np.zeros(14, np.float32))
        if env.foot_contact_state["left"] and env.foot_contact_state["right"]:
            break
    env._bf_rot = 0.0
    assert _bf_landed_hold(env) == 0.0
    env._bf_rot = 2.0 * np.pi
    assert _bf_landed_hold(env) > 0.3  # standing + credited flip pays


def test_backflip_mid_roll_spawn_credits_rotation():
    """A body posed halfway through the flip has, by definition, already
    rotated halfway — the spawn presets the accumulator to match."""
    from microduck_local.behaviors import _bf_spawn_mid_roll
    env = _backflip_env()
    obs = _bf_spawn_mid_roll(env)
    assert obs.shape == (61,)
    assert 1.2 < env._bf_rot < 4.4
    g = env._projected_gravity()
    # Attitude must match the credited rotation (gx = -sin(rot) for a pure
    # backward pitch), and the spawn arrives still rolling backward.
    assert abs(float(g[0]) - (-np.sin(env._bf_rot))) < 0.05
    assert env.data.qvel[4] < 0.0


def test_backflip_reset_clears_rotation():
    env = _backflip_env()
    env._bf_rot = 5.0
    env.reset(seed=11)
    valid = (env._bf_rot == 0.0 or abs(env._bf_rot - 2.0 * np.pi) < 1e-9
             or 1.2 < env._bf_rot < 4.4)  # fresh, landed-spawn, or mid-roll
    assert valid


def test_spawn_family_probs_env_override(monkeypatch):
    """A curriculum stage can override the spawn MIX, not just the window —
    'learning to land' must actually be mostly landing spawns."""
    env = _backflip_env()
    monkeypatch.setenv("MICRODUCK_SPAWN_FAMILY_PROBS", "1.0,0.0")
    env.reset(seed=3)
    assert abs(env._bf_rot - 2.0 * np.pi) < 1e-9  # always the landed spawn
    monkeypatch.setenv("MICRODUCK_SPAWN_FAMILY_PROBS", "0.0,1.0")
    env.reset(seed=3)
    assert 1.2 < env._bf_rot < 4.4                # always mid-roll
    monkeypatch.setenv("MICRODUCK_SPAWN_FAMILY_PROBS", "bogus")
    env.reset(seed=3)                             # malformed -> declared mix


def test_spawn_overrides_beat_environ(monkeypatch):
    """Per-instance stage knobs (BehaviorEnv(spawn_overrides=...)) win over
    os.environ — the farm process hosts many preview envs at once, and each
    must carry ITS stage; os.environ stays the trainer-subprocess channel."""
    monkeypatch.setenv("MICRODUCK_BF_SPAWN_LO", "1.3")
    monkeypatch.setenv("MICRODUCK_BF_SPAWN_HI", "1.35")
    monkeypatch.setenv("MICRODUCK_SPAWN_FAMILY_PROBS", "1.0,0.0")
    env = BehaviorEnv("backflip", spawn_overrides={
        "MICRODUCK_BF_SPAWN_LO": "4.2", "MICRODUCK_BF_SPAWN_HI": "5.6",
        "MICRODUCK_SPAWN_FAMILY_PROBS": "0.0,1.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    for seed in range(8):
        env.reset(seed=seed)
        # Always the mid-roll family (override mix), inside the OVERRIDE
        # window — not the environment's landed-only mix or tiny window.
        assert 4.2 <= env._bf_rot <= 5.6
    # Without instance overrides the env vars still rule (the trainer path).
    env2 = BehaviorEnv("backflip", obs_noise=False, domain_rand=False,
                       action_delay=False, random_yaw=False, seed=0)
    env2.reset(seed=0)
    assert abs(env2._bf_rot - 2 * np.pi) < 1e-9  # probs say always landed


# ------------------------------------------------- the bilateral mirror prior


def test_run_training_path_uses_bam_and_domain_rand(monkeypatch):
    """The trainer, not the unit-test env, is where BAM/DR have to be on —
    otherwise local run is a different plant from GPU run."""
    monkeypatch.delenv("MICRODUCK_ACTUATOR", raising=False)
    from microduck_local.train_behavior import make_env

    trick = make_env("stand", 0, 0)()
    assert trick.actuator_model == "xml"
    assert trick.domain_rand is False
    assert trick.random_yaw is False

    run = make_env("run", 0, 0)()
    assert run.actuator_model == "bam"
    assert run.domain_rand is True
    assert run.random_yaw is True
    assert run.height_termination is False


def test_run_recipe_matches_gpu_run_not_walk():
    """The local run recipe is a port of microduck_run_env_cfg, not velocity.

    Walk weights (air_time 3 > speed 2, pose running-std dead, extra
    face_home/hold_the_line) priced running out of the optimum. GPU run
    makes speed the largest term, wakes the running pose, and drops the
    local-only heading/crab taxes."""
    terms = {t.key: t for t in BEHAVIORS["run"].terms}
    assert terms["keep_pace"].weight == pytest.approx(4.0)
    assert terms["air_time"].weight == pytest.approx(3.0)
    assert terms["stay_upright"].weight == pytest.approx(2.0)
    assert terms["track_turn"].weight == pytest.approx(2.0)
    assert terms["pose"].weight == pytest.approx(1.0)
    assert terms["head_up"].weight == pytest.approx(3.5)  # head-drop priced (see recipe)
    assert "natural_pose" not in terms
    assert "gentle_joints" not in terms
    assert "go_straight" not in terms
    assert "save_energy" not in terms
    # face_home / hold_the_line are BACK by explicit user request ("it's not
    # walking straight — we don't have a reward for keeping its initial
    # orientation"): track_turn prices yaw RATE only, so a lazy arc was free.
    # What hurt before was the UNBOUNDED face_home at weight 3 (-74/step
    # worst); both anchors are now bounded to one unit at weight 1, small
    # against the ~9-point positive budget. Locked the other way by
    # test_penalty_scale.test_run_recipe_anchors_absolute_heading_and_line.
    # Anchors OUT again — not as "GPU purity" but because the policy is
    # compass-blind (no yaw in the 61 obs); see
    # test_run_recipe_has_no_unobservable_terms. Steering owns the heading.
    assert "face_home" not in terms
    assert "hold_the_line" not in terms
    # Smoothness weight is 1.0: the 0.1→0.5 GPU schedule lives inside the fn.
    assert terms["smooth_moves"].weight == pytest.approx(1.0)
    from microduck_local.behaviors import _run_action_rate_weight
    assert _run_action_rate_weight(0) == pytest.approx(0.1)
    assert _run_action_rate_weight(int(3000 * 24 * _RUN_STAGE_SCALE)) == pytest.approx(0.5)
    assert BEHAVIORS["run"].episode_s == pytest.approx(20.0)
    assert BEHAVIORS["run"].forward_cmd == pytest.approx(0.6)


def test_only_the_one_sided_recipes_opt_out_of_the_mirror_prior():
    """The audit, as a lock. `Behavior.symmetric` decides whether the trainer
    adds symmetry.py's mirror loss, so a new recipe that names a side (or
    imitates a clip) has to be listed here — the default is True and silence
    would train it under a wrong prior."""
    asymmetric = {b.id for b in BEHAVIORS.values() if not b.symmetric}
    assert asymmetric == {"one_leg", "imitate"}
    # spin is genuinely direction-agnostic (_spin_rate reads abs(wz)); the
    # rest are sagittal or two-footed.
    for bid in ("spin", "run", "stand", "crouch", "backflip", "airflip",
                "headstand"):
        assert BEHAVIORS[bid].symmetric, bid


def _side_scores(behavior_id: str, down: str, up: str) -> float:
    """Total recipe score with `down` in contact and `up` in the air, holding
    the BODY fixed — so the only thing that varies is which foot is named."""
    env = BehaviorEnv(behavior_id, obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    for _ in range(20):  # settle the drop-in so the pose is a real stance
        env.step(np.zeros(14, np.float32))
    env.foot_contact_state = {down: True, up: False}
    return sum(t.weight * (-1.0 if t.is_penalty else 1.0) * t.fn(env)
               for t in BEHAVIORS[behavior_id].terms)


def test_one_leg_is_asymmetric_in_its_reward_not_just_its_flag():
    """`symmetric=False` is a claim about the recipe, so check the recipe.
    Swapping which foot is down — the same body, only the side renamed —
    moves one_leg's score hard (~3.0 of reward), because one_leg_hold,
    foot_in_air and planted_foot all read a foot by NAME."""
    left_down = _side_scores("one_leg", "left", "right")
    right_down = _side_scores("one_leg", "right", "left")
    assert left_down - right_down > 1.0, (
        f"one_leg scored {left_down:.3f} on the left foot vs {right_down:.3f} "
        "on the right — if these matched, the mirror loss would cost nothing"
    )
    # A symmetric recipe is indifferent to the same swap: nothing to fight.
    assert abs(_side_scores("stand", "left", "right")
               - _side_scores("stand", "right", "left")) < 1e-6


def test_backflip_brake_charges_only_completed_overroll():
    """Over-rolling was FREE (progress caps at 2*pi but nothing charged the
    leftover momentum) — the user watched second rolls and butt-landings.
    The brake must charge residual back-spin ONLY once rotation is complete,
    never mid-flip where that spin IS the trick."""
    import numpy as np
    from microduck_local.behaviors import _BF_FULL, _bf_brake_pen

    env = _backflip_env()
    env._gyro[:] = (0.0, -4.0, 0.0)     # hard backward spin
    env._bf_rot = 0.5 * _BF_FULL        # mid-flip: spin is the trick
    assert _bf_brake_pen(env) == 0.0
    env._bf_rot = _BF_FULL              # flip complete: brake it
    pen = _bf_brake_pen(env)
    assert -1.0 <= pen < 0.0
    env._gyro[:] = (0.0, 4.0, 0.0)      # forward correction is fine
    assert _bf_brake_pen(env) == 0.0


def test_jaw_parking_is_priced_but_the_neck_kip_is_free():
    """Stage 5 once converged to resting its jaw on the floor for whole
    episodes (rent-free under impact-only gentle_head) instead of flipping.
    Occupancy is charged pre-flip only — mid-roll head contact IS the trick."""
    import numpy as np
    from microduck_local.behaviors import _bf_no_jaw_parking_pen, _jaw_bid

    env = _backflip_env()
    env._bf_rot = 0.0
    env.data.xpos[_jaw_bid(env)][2] = 0.07   # parked
    assert -1.0 <= _bf_no_jaw_parking_pen(env) < 0.0
    env._bf_rot = 1.5                        # mid-roll: kip is free
    assert _bf_no_jaw_parking_pen(env) == 0.0
    env._bf_rot = 0.0
    env.data.xpos[_jaw_bid(env)][2] = 0.20   # standing headroom: free
    assert _bf_no_jaw_parking_pen(env) == 0.0
