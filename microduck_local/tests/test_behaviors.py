"""Behavior library: envs build and step cleanly, recipes obey the sign rules,
the chat matcher finds the right trick."""

import numpy as np
import pytest

from microduck_local.behaviors import (
    _RUN_STAGE_SCALE,  # noqa: F401
    BEHAVIORS,
    BehaviorEnv,
    behavior_card,
    match_behavior,
)


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
    if behavior_id == "spin":
        # Spin carries a per-episode DIRECTION command in the wz slot — the
        # signed spin_fast pay needs it, and it makes the trick steerable.
        # vx/vy stay zero like any trick.
        np.testing.assert_allclose(obs[48:50], np.zeros(2, np.float32), atol=1e-6)
        assert abs(float(obs[50])) == pytest.approx(1.0)
    elif cmd:
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
    os.environ — the lab process hosts many preview envs at once, and each
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


def test_headstand_mid_flip_window_knob():
    """A headstand curriculum stage can march the tripod-spawn pitch window,
    same pattern as the backflip's MICRODUCK_BF_SPAWN_LO/HI."""
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "0.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "1.0",
        "MICRODUCK_MF_PITCH_LO": "2.35", "MICRODUCK_MF_PITCH_HI": "2.5"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    for seed in range(6):
        env.reset(seed=seed)
        assert env.last_spawn == "mid-flip"
        q = env.data.qpos[3:7]  # pure-pitch spawn quat [cos p/2, 0, sin p/2, 0]
        pitch = 2.0 * np.arctan2(float(q[2]), float(q[0]))
        assert 2.35 <= pitch <= 2.5


def test_headstand_mid_flip_slump_variant():
    """MICRODUCK_MF_SLUMP_PROB spawns the measured dive-arrival slump —
    deep pitch, trunk low, legs folded under — instead of the feet-planted
    launch pad, so recovering from a botched dive is on-policy."""
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "0.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "1.0",
        "MICRODUCK_MF_SLUMP_PROB": "1.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    for seed in range(6):
        env.reset(seed=seed)
        assert env.last_spawn == "mid-flip"
        q = env.data.qpos[3:7]
        pitch = 2.0 * np.arctan2(float(q[2]), float(q[0]))
        assert 2.4 <= pitch <= 2.7          # ~137-155 deg nose-down
        assert env.data.qpos[2] <= 0.09     # trunk LOW — a slump, not a pad


def test_headstand_hold_pays_the_stack_not_the_tuck():
    """User-specified success test: head on the ground, body ABOVE the head,
    feet ABOVE that. The clean inverted-spawn stack must out-pay a folded
    tuck of the same orientation — orientation alone used to pay a low tuck
    with feet dangling at head height in full."""
    import mujoco

    # Raw (streak-free) pricing: this test grades POSES against each
    # other; the persistence ramp is locked separately.
    from microduck_local.behaviors import _headstand_hold_raw
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    from microduck_local.behaviors import _head_on_floor
    env.reset(seed=1)  # clean near-vertical stack, legs ~straight up
    # The spawn hovers a hair above the floor — lower it until the head
    # actually touches (kinematic, no dynamics), as a settled hold would.
    for _ in range(40):
        if _head_on_floor(env):
            break
        env.data.qpos[2] -= 0.003
        mujoco.mj_forward(env.model, env.data)
    assert _head_on_floor(env), "could not settle the spawn into head contact"
    # foot_contact_state is a per-step hoisted snapshot — refresh it after
    # kinematic pushes (a real step does this itself).
    env.foot_contact_state = env._foot_contacts()
    v_clean = _headstand_hold_raw(env)
    assert v_clean > 0.1, f"clean stack should pay, got {v_clean}"
    # Fold hips+knees hard (tuck) at the same trunk attitude so the feet
    # drop toward head height — try both fold directions and grade the one
    # that actually lowers the feet (sign conventions differ per joint).
    q = env.data.qpos
    # Real flops also TIP: at ~146 deg (the measured dive-trap attitude)
    # the folded legs hang at/below the trunk — an ORDERING violation
    # (straight-up kinematics can't get feet under the trunk on this robot;
    # a 160-deg bent-leg pose keeps feet above trunk and legitimately pays).
    p = 2.55  # rad, ~146 deg — the dive-trap flop family
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    lo_feet, v_tuck = None, None
    for s in (1.0, -1.0):
        for li, ri, val in ((2, 11, 1.6), (3, 12, 1.9), (4, 13, 1.2)):
            q[env.joint_qpos_adr[li]] = -s * val
            q[env.joint_qpos_adr[ri]] = s * val
        # The observed tuck also carries the trunk LOW (renders: ~0.11 vs the
        # clean stack's 0.165) — drop the root to match the real pose family.
        q[2] = 0.115
        mujoco.mj_forward(env.model, env.data)
        env.foot_contact_state = env._foot_contacts()
        zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
        zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
        fz = (zl + zr) / 2
        if lo_feet is None or fz < lo_feet:
            lo_feet, v_tuck = fz, _headstand_hold_raw(env)
    # A two-joint kinematic fold only reaches feet ~0.14 above the head
    # (real crumples bend everything and get to ~0.06, paying 3.5x less);
    # lock the gradient direction and the ratio this pose can prove.
    assert v_clean > 1.3 * v_tuck, (
        f"tuck must pay under the stack: clean {v_clean}, tuck {v_tuck} "
        f"(feet_z {lo_feet:.3f})")
    # The toe-press pike (head down, hips folded, feet only ~22 cm up)
    # used to collect almost full hold. Straight stack must clearly win.
    from microduck_local.behaviors import _HS_HEAD_TUCK, _HS_NECK_TUCK
    env.reset(seed=1)
    q = env.data.qpos
    p = np.deg2rad(152.0)
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    q[2] = 0.12
    q[env.joint_qpos_adr[5]] = _HS_NECK_TUCK
    q[env.joint_qpos_adr[6]] = _HS_HEAD_TUCK
    q[env.joint_qpos_adr[2]] = -1.2
    q[env.joint_qpos_adr[11]] = 1.2
    q[env.joint_qpos_adr[3]] = -1.4
    q[env.joint_qpos_adr[12]] = 1.4
    mujoco.mj_forward(env.model, env.data)
    env.foot_contact_state = {"left": False, "right": False}
    v_pike = _headstand_hold_raw(env)
    assert v_clean > 1.5 * v_pike, (
        f"toe-press pike must not collect hold: clean {v_clean}, pike {v_pike}")


def test_headstand_slope_terms_pay_progress_not_state():
    """The anti-parking contract: a duck HOLDING a pose earns nothing from
    the slope terms (they pay only new episode-best progress), and a spawn
    handout banks no slope money either (baseline anchors to the spawn pose).
    Three graduates in a row parked at ~146° living off these terms when
    they were per-step state pay."""
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=0)
    hold = np.zeros(14, np.float32)  # hold the spawn pose: no progress
    for _ in range(25):
        env.step(hold)
    sums = env.reward_sums
    # body_lifted, feet_up and neck_tuck are deliberately excluded: they are
    # per-step SALARIES for the stacked pose (see the recipe), and the
    # anti-farming contract covers only the shepherds, which pay progress.
    #
    # sums[k], not sums.get(k, 0.0): this summed "head_low" and "nose_down"
    # too, which are _hs_update POTENTIALS that were never registered as
    # reward terms. get() scored both 0.0 forever, so the assertion silently
    # covered two fewer terms than it named while reading as if it covered
    # five — and it also carried neck_tuck, a salary whose legitimate
    # per-step income ate the budget meant for detecting an annuity. Indexing
    # makes a rename fail loudly instead of hollowing the test out again.
    shepherds = ("upside_down", "feet_rise")
    missing = [k for k in shepherds if k not in sums]
    assert not missing, f"shepherd terms renamed or dropped: {missing}"
    slope = sum(sums[k] for k in shepherds)
    # Tiny settle-in drift is fine; a per-step annuity over 25 steps is not
    # (state pay measured ~2.9/step here — 25 steps would be ~70). Parking
    # measures about -49: symmetric shaping makes a settling duck GIVE BACK
    # potential, so only the positive direction is the farming signal.
    assert slope < 12.0, f"slope terms paid {slope:.1f} for sitting still"


def test_reload_library_rolls_back_a_failing_recipe_edit(monkeypatch):
    """A recipe edit that COMPILES but raises at import must leave the library
    exactly as it was. Restoring only the failing module is not enough: core
    reloads first and rebinds BEHAVIORS/CATALOG, so the modules after the
    failure never re-register — leaving `match_behavior` denying tricks the
    package still lists. Simulated by making the reload of a LATE submodule
    raise, which is what a bad edit does at exec time."""
    import importlib

    import microduck_local.behaviors as B

    before_behaviors = dict(B.BEHAVIORS)   # same objects, for identity checks
    before_catalog_keys = sorted(B.CATALOG)
    core_registry = B._core.BEHAVIORS

    real_reload = importlib.reload
    target = B._backflip.__name__

    def reload_but_fail_on_backflip(m):
        if m.__name__ == target:
            raise NameError("_A_TYPO_IN_MY_EDIT")
        return real_reload(m)

    monkeypatch.setattr(B._importlib, "reload", reload_but_fail_on_backflip)
    with pytest.raises(NameError):
        B.reload_library()
    monkeypatch.undo()

    # Untouched: the rollback restores the very objects, not equal copies.
    assert B.BEHAVIORS == before_behaviors
    assert sorted(B.CATALOG) == before_catalog_keys
    # The registry every core helper closes over must be the SAME object, or
    # match_behavior reads a truncated dict while the package shows all nine.
    assert B._core.BEHAVIORS is core_registry
    assert B.BEHAVIORS is B._core.BEHAVIORS
    assert B.match_behavior("do a backflip").id == "backflip"
    assert B.match_behavior("do a headstand").id == "headstand"
    # Deliberately NO real reload_library() here: a successful reload rebinds
    # every Behavior and BehaviorEnv, and the tests defined after this one bind
    # their names at import. The rollback restoring the ORIGINAL objects is
    # exactly what makes this test safe to run mid-suite — assert that.
    assert B.BEHAVIORS["headstand"] is before_behaviors["headstand"]
    assert sorted(B.CATALOG) == before_catalog_keys


def test_reload_modules_restores_a_module_that_raises_mid_exec(tmp_path,
                                                              monkeypatch):
    """`importlib.reload` re-executes a module in its LIVE dict, so an edit
    that compiles but raises part-way leaves it HALF-NEW: names bound before
    the raise hold the new values while everything after is stale. The lab hit
    this on motion.py, which /teach reloads before the behaviors package —
    it then served a Frankenstein clip module the trainer subprocess (always a
    fresh import) would never run. Uses a throwaway module rather than the
    real motion so no live class is rebound mid-suite."""
    import importlib
    import sys

    from microduck_local import motion

    src = tmp_path / "torn_reload_probe.py"
    src.write_text('BEFORE = "v1"\nDELETED = "v1"\n'
                   'def fn():\n    return "v1"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    mod = importlib.import_module("torn_reload_probe")
    try:
        # The edit: rebinds BEFORE, drops DELETED, would rewrite fn — and
        # raises in between, the way a NameError in a fresh recipe line does.
        src.write_text('BEFORE = "v2"\nraise RuntimeError("half-applied edit")\n'
                       'def fn():\n    return "v2"\n')
        importlib.invalidate_caches()
        with pytest.raises(RuntimeError):
            motion.reload_modules([mod])
        assert mod.BEFORE == "v1", "kept a value from the edit that failed"
        assert mod.fn() == "v1"
        # The rollback has to REFILL the namespace it emptied, not just stop.
        assert mod.DELETED == "v1"

        # The tear itself, so this guarantee cannot quietly degrade back into
        # a plain reload: same module, same edit, torn.
        with pytest.raises(RuntimeError):
            importlib.reload(mod)
        assert mod.BEFORE == "v2" and mod.fn() == "v1"
    finally:
        sys.modules.pop("torn_reload_probe", None)


def test_motion_reload_self_targets_the_live_motion_module(monkeypatch):
    """viz_server's /teach reloads motion before the behaviors package; the
    fix is for it to call motion.reload_self() instead of importlib.reload.
    Deliberately does NOT reload for real — that rebinds Clip/load_clip on
    fresh objects for every test that imported them at module scope, the same
    hazard test_reload_library_rolls_back_a_failing_recipe_edit avoids."""
    import sys

    from microduck_local import motion

    seen = []
    monkeypatch.setattr(motion, "reload_modules",
                        lambda mods, after=None: seen.append(list(mods)))
    motion.reload_self()
    assert seen == [[sys.modules["microduck_local.motion"]]]


def test_flatten_reports_a_recipe_name_that_collides_with_the_package():
    """`from . import core as _core` pins the BARE name `core` on the package
    too, so all eight submodule names sit in _RESERVED beside the real
    machinery — and _flatten used to skip a reserved name in silence. A recipe
    author adding a module-level `env = ...` (or `core`, or `locomotion`) to
    headstand.py had it dropped with no error and no log, and
    `from microduck_local.behaviors import env` answered with the SUBMODULE.
    Now it says so, and says nothing else has changed."""
    import microduck_local.behaviors as B

    env_module, machinery = B.env, B._SUBMODULES
    headstand = B.BEHAVIORS["headstand"]

    # An ordinary new recipe constant still flattens, and still un-flattens.
    B._headstand._A_BRAND_NEW_KNOB = 1.23
    B._flatten()
    assert B._A_BRAND_NEW_KNOB == 1.23
    del B._headstand._A_BRAND_NEW_KNOB
    B._flatten()
    assert not hasattr(B, "_A_BRAND_NEW_KNOB")

    for module, name in ((B._headstand, "env"),        # a submodule's name
                         (B._poses, "_SUBMODULES"),    # package machinery
                         (B._poses, "vars")):          # a builtin _flatten calls
        setattr(module, name, 0.3)
        try:
            with pytest.raises(NameError, match=name):
                B._flatten()
        finally:
            delattr(module, name)

    # Nothing was mutated on the way to any of those errors.
    assert B.env is env_module and B._SUBMODULES is machinery
    assert B.BEHAVIORS is B._core.BEHAVIORS
    assert B.BEHAVIORS["headstand"] is headstand
    B._flatten()          # leave the package as the rest of the suite found it
    assert B.match_behavior("do a headstand").id == "headstand"


def test_reload_library_rolls_back_when_a_reloaded_recipe_shadows_the_package(
        monkeypatch):
    """The re-flatten is part of "the reload succeeded", so a name clash found
    there has to roll the MODULES back as well. Left reloaded, core's dict —
    which every helper closes over — would hold the new registry while the
    package namespace still showed the old one: `match_behavior` denying
    tricks the catalog lists, the split brain the all-module rollback exists
    to prevent. Same no-live-reload discipline as the rollback test above: the
    rollback is what makes this safe to run mid-suite, so assert it."""
    import importlib

    import microduck_local.behaviors as B

    before_behaviors = dict(B.BEHAVIORS)
    core_registry = B._core.BEHAVIORS
    real_reload = importlib.reload

    def reload_and_shadow(m):
        new = real_reload(m)
        if new.__name__ == B._headstand.__name__:
            new.env = 0.3     # the plausible recipe constant, post-exec
        return new

    monkeypatch.setattr(B._importlib, "reload", reload_and_shadow)
    with pytest.raises(NameError, match="env"):
        B.reload_library()
    monkeypatch.undo()

    assert B.BEHAVIORS["headstand"] is before_behaviors["headstand"]
    assert B.BEHAVIORS["backflip"] is before_behaviors["backflip"]
    assert B._core.BEHAVIORS is core_registry
    assert B.BEHAVIORS is B._core.BEHAVIORS
    assert not hasattr(B._headstand, "env")
    assert B.match_behavior("do a headstand").id == "headstand"
    assert B.match_behavior("do a backflip").id == "backflip"


def test_headstand_ladder_carries_no_reward_edits():
    """The ladder contract (user decision, 2026-09-01, reinstating the
    champion's servo ladder): the headstand trains as a CHAIN whose stages
    may ladder only PHYSICS, SPAWNS, and DEPTH — never the reward. The
    sealed term set (locked by the tests around this one) is identical in
    every stage, so no rung can grow its own leak — the failure mode of the
    first staged era. Stage 1 must be the XML training-wheels drill (BAM
    from scratch never ignites: ad85a8-s1, re-proven across five one-move
    configs on 2026-09-01), and the finisher must keep standing reps ≤ 20%
    (standing-heavy finals measurably destroyed working holds)."""
    b = BEHAVIORS["headstand"]
    assert len(b.curriculum) >= 4
    ALLOWED = {"MICRODUCK_ACTUATOR", "MICRODUCK_BAM_CURRENT_SCALE",
               "MICRODUCK_INVERTED_SPAWN_PROB", "MICRODUCK_MID_FLIP_SPAWN_PROB",
               "MICRODUCK_EPISODE_S", "MICRODUCK_HS_GATE",
               "MICRODUCK_INV_SPAWN_KICK", "MICRODUCK_MF_PITCH_LO",
               "MICRODUCK_MF_PITCH_HI", "MICRODUCK_MF_SLUMP_PROB",
               "MICRODUCK_MF_SPIN_MAX"}
    for st in b.curriculum:
        assert set(st.env) <= ALLOWED, f"stage '{st.label}' smuggles a knob"
        assert st.env.get("MICRODUCK_EPISODE_S") == "8"  # 8s ignites, 20s doesn't
    assert b.curriculum[0].env.get("MICRODUCK_ACTUATOR") == "xml"
    assert b.curriculum[-1].env.get("MICRODUCK_ACTUATOR") == "bam"
    last = b.curriculum[-1]
    standing = 1.0 - float(last.env["MICRODUCK_INVERTED_SPAWN_PROB"]) \
                   - float(last.env["MICRODUCK_MID_FLIP_SPAWN_PROB"])
    assert standing <= 0.20 + 1e-9
    keys = {t.key for t in b.terms}
    assert "headstand_hold" in keys
    assert "body_lifted" in keys
    assert "legs_straight" in keys
    assert "neck_tuck" in keys
    assert "feet_up" in keys
    # calm_up_top must stay OUT of the scratch recipe: -0.06*|w|^2 taxed the
    # correction thrash a BAM learner needs (-1.08/step fleet-wide) while the
    # motionless side-prop sat untaxed (removed 2026-09-01).
    assert "calm_up_top" not in keys
    # Same lesson: smoothness taxed catch-corrections harder than the hold
    # paid (4d93a6: −1.9 vs +1.3). Polish-only, once a hold exists.
    assert "smooth_moves" not in keys
    assert "gentle_joints" not in keys
    assert "save_energy" not in keys


def test_headstand_inverted_spawn_plants_the_head():
    """The drop-in spawn used to hover ~6 mm (hold = 0 on frame 0), so the
    one start that's supposed to pay the stack never did and from-scratch
    brains learned to crumple. After reset the crown must already be in
    contact and both stack salaries must be live."""
    from microduck_local.behaviors import (
        _body_lifted,
        _head_on_floor,
        _headstand_hold,
        _headstand_hold_raw,
        _legs_straight_up,
    )
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    for seed in range(8):
        env.reset(seed=seed)
        env.foot_contact_state = env._foot_contacts()
        assert _head_on_floor(env), f"seed {seed} still hovering"
        assert _headstand_hold_raw(env) > 0.1, (
            f"seed {seed} planted stack should pay raw hold, got "
            f"{_headstand_hold_raw(env)}")
        # Persistence ramp (2026-09-01): at streak 0 the term pays the 0.3
        # floor of raw — transits stay paid (ignition), dwelling pays 3.3x
        # more (two capped-std graduates flickered at ~0.3 s under flat pay).
        assert _headstand_hold(env) == pytest.approx(
            0.3 * _headstand_hold_raw(env)), "streak-0 hold must be 0.3*raw"
        assert _body_lifted(env) > 0.5, (
            f"seed {seed} planted stack should pay tallness, got "
            f"{_body_lifted(env)}")
        assert _legs_straight_up(env) > 0.5, (
            f"seed {seed} planted stack should pay straight legs, got "
            f"{_legs_straight_up(env)}")


def test_headstand_tallness_salary_zeros_the_heap():
    """body_lifted is the anti-crumple salary: the measured parked heap
    (feet down, trunk ~7 cm, still 'inverted') must pay ~0, or orientation
    shaping's blind spot comes back."""
    import mujoco

    from microduck_local.behaviors import _body_lifted, _head_on_floor
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    env.foot_contact_state = env._foot_contacts()
    v_stack = _body_lifted(env)
    assert v_stack > 0.5, f"planted stack should pay tallness, got {v_stack}"
    # Measured crumple (f2b99e / 45b073 inverted drops): trunk ~0.07, feet
    # planted, jaw on the floor, still nearly inverted.
    q = env.data.qpos
    q[2] = 0.072
    p = np.deg2rad(168.0)
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    for li, ri, val in ((2, 11, 1.2), (3, 12, 1.4), (4, 13, 0.8)):
        q[env.joint_qpos_adr[li]] = -val
        q[env.joint_qpos_adr[ri]] = val
    mujoco.mj_forward(env.model, env.data)
    # The live crumple plants both feet; a two-joint kinematic fold can
    # leave them hovering, so name the contacts the term actually reads.
    env.foot_contact_state = {"left": True, "right": True}
    v_heap = _body_lifted(env)
    assert v_heap < 0.05, (
        f"feet-down heap must not collect tallness: stack {v_stack:.3f} "
        f"heap {v_heap:.3f} head={_head_on_floor(env)} "
        f"feet={env.foot_contact_state}")


def _min_z_by_mat(env, substr: str) -> float:
    import mujoco
    m, d = env.model, env.data
    zs = []
    for i in range(m.ngeom):
        matid = int(m.geom_matid[i])
        mat = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
               if matid >= 0 else "") or ""
        if substr in mat:
            zs.append(float(d.geom_xpos[i][2]))
    assert zs, f"no geom with material containing {substr!r}"
    return min(zs)


def test_headstand_inverted_spawn_plants_the_crown():
    """Same-sign neck/head (−0.6, −0.6) planted the FACE (eye/lens below
    the crown). Opposite-sign tuck must put the skull on the floor and
    the face several centimetres above it."""
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    crown = _min_z_by_mat(env, "top_head_shell")
    face = _min_z_by_mat(env, "face_part")
    eye = _min_z_by_mat(env, "noenoeil")
    assert face - crown > 0.02, (
        f"face should sit above the crown: crown={crown:.3f} face={face:.3f}")
    assert eye - crown > 0.02, (
        f"eyes should sit above the crown: crown={crown:.3f} eye={eye:.3f}")
    bottom = _min_z_by_mat(env, "bottom_head_shell")
    assert bottom - crown > 0.004, (
        f"must rest on the rounded crown, not the underside: "
        f"crown={crown:.3f} bottom={bottom:.3f}")
    gx = float(env._projected_gravity()[0])
    assert gx > 0.25, (
        f"must be a forward / nose-down roll on the crown, not a backbend "
        f"(gx={gx:.2f})")


def test_headstand_neck_tuck_pays_crown_not_home():
    """Action 0 is DEFAULT_POSE (neck +0.35) — the standing 'head upright'
    look, which on an inverted body plants the FACE. Spawn already tucks
    (neck−, head+); untucking or the old same-sign 'tuck' must lose the
    salary."""
    import mujoco

    from microduck_local import contract as C
    from microduck_local.behaviors import _neck_tuck
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    env.foot_contact_state = env._foot_contacts()
    v_tuck = _neck_tuck(env)
    q = env.data.qpos
    q[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5]
    q[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6]
    mujoco.mj_forward(env.model, env.data)
    env.foot_contact_state = env._foot_contacts()
    v_home = _neck_tuck(env)
    q[env.joint_qpos_adr[5]] = -0.6
    q[env.joint_qpos_adr[6]] = -0.6
    mujoco.mj_forward(env.model, env.data)
    env.foot_contact_state = env._foot_contacts()
    v_face = _neck_tuck(env)
    assert v_tuck > 0.5, f"spawn tuck should pay, got {v_tuck}"
    assert v_tuck > 2.0 * v_home, (
        f"upright neck must not collect tuck: tucked {v_tuck:.3f} "
        f"home {v_home:.3f}")
    assert v_tuck > 2.0 * v_face, (
        f"same-sign face-plant must not collect tuck: tucked {v_tuck:.3f} "
        f"face {v_face:.3f}")


def test_headstand_salaries_zero_the_backward_rest():
    """gz is symmetric: a back-of-head rest is just as 'inverted' as the
    front crown-roll. Without the hold's g[0] < -0.12 gate on every
    per-step salary, that camp collected ~6/step for free (body_lifted +
    legs_straight + neck_tuck + feet_up) against a tiny wrong_way charge.
    Front spawn must still pay; the backbend must not."""
    import mujoco

    from microduck_local.behaviors import (
        _body_lifted,
        _feet_up,
        _legs_straight_up,
        _neck_tuck,
    )
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    env.foot_contact_state = env._foot_contacts()
    front = {
        "neck_tuck": _neck_tuck(env),
        "legs_straight": _legs_straight_up(env),
        "feet_up": _feet_up(env),
        "body_lifted": _body_lifted(env),
    }
    assert front["neck_tuck"] > 0.3, front
    assert front["legs_straight"] > 0.3, front
    assert front["feet_up"] > 0.1, front
    # Mirror the spawn through vertical: same stack, gx flipped (backbend).
    q = env.data.qpos
    p = 2.0 * np.pi - 2.0 * np.arctan2(float(q[5]), float(q[3]))  # 360° - pitch
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    mujoco.mj_forward(env.model, env.data)
    env.foot_contact_state = {"left": False, "right": False}
    gx = float(env._projected_gravity()[0])
    assert gx < -0.12, f"backbend pose should have gx<0, got {gx:.2f}"
    back = {
        "neck_tuck": _neck_tuck(env),
        "legs_straight": _legs_straight_up(env),
        "feet_up": _feet_up(env),
        "body_lifted": _body_lifted(env),
    }
    for k, v in back.items():
        assert v < 0.05, f"{k} still pays the backward rest: {v:.3f} (gx={gx:.2f})"


def test_headstand_feet_up_requires_both_feet_off():
    """A planted-foot bow used to collect feet_up from one waving foot /
    still-high kinematics. Both feet must be off the floor."""
    from microduck_local.behaviors import _feet_up
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    env.foot_contact_state = env._foot_contacts()
    v_air = _feet_up(env)
    env.foot_contact_state = {"left": True, "right": True}
    v_down = _feet_up(env)
    assert v_air > 0.2, f"spawn feet-up should pay, got {v_air}"
    assert v_down < 0.05, f"feet-down bow must not collect feet_up, got {v_down}"


def test_headstand_face_plant_pen_zeros_on_crown_only():
    """Crown-only spawn is free; a face-lean (HOME neck, ~146°) is charged."""
    import mujoco

    from microduck_local import contract as C
    from microduck_local.behaviors import _face_on_floor_pen
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    assert _face_on_floor_pen(env) == 0.0, "crown spawn must not be charged"
    q = env.data.qpos
    p = np.deg2rad(146.0)
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    q[2] = 0.08
    q[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5]
    q[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6]
    mujoco.mj_forward(env.model, env.data)
    assert _face_on_floor_pen(env) < 0.0, "face-lean must be charged"


def test_headstand_overshoot_does_not_end_the_episode():
    """The covenant's contract: falling NEVER ends a headstand episode —
    "it can try, crumble, and try again." The gx < -0.2 overshoot terminal
    (removed 2026-09-01) priced every unfold attempt at catastrophe: from
    the balanced tuck-ball the likeliest failure of extending is a backward
    topple, and ending the episode forfeited all remaining income — so
    ball-converged brains never risked the unfold (943186: feet_up income
    3x'd, extension frozen). The backward route is priced by the per-step
    wrong_way penalty instead."""
    import mujoco
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)
    env.reset(seed=1)
    a = np.zeros(14, np.float32)
    _, _, term, _, _ = env.step(a)
    assert not term, "crown spawn must not be terminal"
    p = np.deg2rad(200.0)  # past vertical, gx negative
    q = env.data.qpos
    q[3:7] = [np.cos(p / 2), 0.0, np.sin(p / 2), 0.0]
    mujoco.mj_forward(env.model, env.data)
    _, _, term, _, _ = env.step(a)
    assert not term, "overshoot must NOT end the episode (attempt tax)"


def test_every_headstand_salary_carries_the_roll_gate():
    """The side-prop kickstand (a ~37 deg lean on the head-shell edge, gz 0.75)
    is statically stable and pays nothing to hold — so EVERY positive
    headstand term must consult _hs_too_rolled, catalog terms included. It was
    added to five of six; feet_on_top (a teach-panel slider) was the gap."""
    import mujoco

    from microduck_local.behaviors import (
        _HS_HEAD_TUCK,
        _HS_NECK_TUCK,
        _body_lifted,
        _feet_on_top,
        _feet_up,
        _head_on_floor,
        _headstand_hold_raw,
        _legs_straight_up,
        _neck_tuck,
    )
    env = BehaviorEnv("headstand", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    d = env.data
    pitch, roll = np.pi - 0.35, 0.65          # the measured kickstand pose
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    d.qpos[:] = 0.0
    d.qpos[3:7] = [cp * cr, cp * sr, sp * cr, sp * sr]
    d.qpos[2] = 0.165
    for adr in env.joint_qpos_adr:
        d.qpos[adr] = 0.0
    d.qpos[env.joint_qpos_adr[5]] = _HS_NECK_TUCK
    d.qpos[env.joint_qpos_adr[6]] = _HS_HEAD_TUCK
    d.qvel[:] = 0.0
    mujoco.mj_forward(env.model, d)
    for _ in range(90):
        if _head_on_floor(env):
            break
        d.qpos[2] -= 0.002
        mujoco.mj_forward(env.model, d)
    env.foot_contact_state = env._foot_contacts()
    assert abs(float(env._projected_gravity()[1])) > 0.35   # genuinely rolled
    for fn in (_headstand_hold_raw, _body_lifted, _legs_straight_up,
               _feet_up, _neck_tuck, _feet_on_top):
        assert fn(env) == 0.0, f"{fn.__name__} pays the side-prop kickstand"


def test_headstand_roll_gate_allows_catch_wobble_kills_kickstand():
    """|g[1]| > 0.35 (≈20°) zeroed a BAM catch's salaries. 25° must still
    pay; the 37° side-prop (g[1]≈0.60) must not."""
    import mujoco

    from microduck_local.behaviors import _legs_straight_up
    env = BehaviorEnv("headstand", spawn_overrides={
        "MICRODUCK_INVERTED_SPAWN_PROB": "1.0",
        "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.0",
        "MICRODUCK_INV_SPAWN_KICK": "0.0"},
        obs_noise=False, domain_rand=False, action_delay=False,
        random_yaw=False, seed=0)

    def roll(deg):
        env.reset(seed=1)
        p = 2.0 * np.arctan2(float(env.data.qpos[5]), float(env.data.qpos[3]))
        r = np.deg2rad(deg)
        cp, sp = np.cos(p / 2), np.sin(p / 2)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        env.data.qpos[3:7] = [cp * cr, cp * sr, sp * cr, sp * sr]
        mujoco.mj_forward(env.model, env.data)
        env.foot_contact_state = {"left": False, "right": False}
        return _legs_straight_up(env), abs(float(env._projected_gravity()[1]))

    v25, g25 = roll(25)
    v37, g37 = roll(37)
    assert g25 < 0.50 < g37, f"g1 25°={g25:.2f} 37°={g37:.2f}"
    assert v25 > 0.3, f"25° catch wobble must still pay, got {v25}"
    assert v37 < 0.05, f"37° kickstand must not pay, got {v37}"


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
    # spin stays mirror-safe: the direction COMMAND rides the wz slot, and
    # the mirror map negates that slot and the gyro together, so a mirrored
    # episode is just the opposite commanded direction. The rest are sagittal
    # or two-footed.
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


def test_push_off_pays_the_right_momentum_not_the_most():
    """min(1, rate/6) paid full marks for ANY spin >= 6 rad/s — over-rotation
    was manufactured at launch and then taxed at the landing. The band pays
    most at the ~5 rad/s a clean entry needs, less for excess, and keeps a
    discovery slope from zero."""
    from microduck_local.behaviors import _bf_push_off

    env = _backflip_env()
    env._bf_rot = 0.1
    def pay(rate):
        env._gyro[:] = (0.0, -rate, 0.0)
        return _bf_push_off(env)
    assert pay(5.0) > pay(8.0), "excess momentum must pay less than the target"
    assert pay(5.0) > pay(2.0)
    assert pay(2.0) > pay(0.5) > pay(0.0) >= 0.0   # discovery slope alive
    assert pay(5.0) == pytest.approx(1.0, abs=0.01)


def test_straight_flip_prices_the_pivot_arc_hardest():
    """Measured: ~98 deg of yaw wobble accrues while the body pivots on the
    head and ~114 in the carry, vs 3.5 at launch — the body slews on the head
    contact like a top (the head JOINTS stay still; still_head already
    works). Off-axis rate costs 3x only in that arc."""
    from microduck_local.behaviors import _bf_straight_pen

    env = _backflip_env()
    env.data.sensordata[env.gyro_adr] = (1.0, 0.0, 1.0)
    env._bf_rot = 0.3                      # launch: gentle
    launch = _bf_straight_pen(env)
    env._bf_rot = 2.0                      # head-pivot: 3x
    pivot = _bf_straight_pen(env)
    assert pivot == pytest.approx(3.0 * launch, rel=1e-6)
    assert -1.0 <= pivot < 0.0


def test_spin_anchors_position_not_just_velocity():
    """stay_put prices drift SPEED — a slow shuffle beats it. The spin now
    also carries the bounded position anchor (user request: stay at the
    origin), the same mechanism that keeps stand/one_leg planted."""
    from microduck_local.behaviors import BEHAVIORS, _stay_home_pen

    terms = {t.key: t for t in BEHAVIORS["spin"].terms}
    assert "stay_home" in terms and terms["stay_home"].is_penalty
    env = BehaviorEnv("spin", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    env.home_xy = (0.0, 0.0)
    env._trunk_xpos[0:2] = (1.0, 1.0)   # far from home: saturated, bounded
    assert _stay_home_pen(env) == pytest.approx(-1.0)


def test_step_dont_skid_pays_stepping_not_scooting():
    """The spin was yawing by pivoting planted feet (skid-steer scooting).
    The catalog term pays only while ROTATING (observable |wz|) for feet in a
    stride-like flight window — planted-feet scooting earns zero."""
    from microduck_local.behaviors import _step_dont_skid

    env = BehaviorEnv("spin", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    env._gyro[:] = (0.0, 0.0, 3.0)          # spinning
    env.foot_contact_state = {"left": True, "right": True}
    assert _step_dont_skid(env) == 0.0       # scooting: planted feet earn nothing
    env.foot_contact_state = {"left": False, "right": True}
    for _ in range(4):                        # 0.08 s airborne -> in window
        pay = _step_dont_skid(env)
    assert pay == pytest.approx(0.5)
    env._gyro[:] = (0.0, 0.0, 0.0)           # not rotating: no rent for hopping
    assert _step_dont_skid(env) == 0.0
    env.reset(seed=1)
    assert env._skid_air == {"left": 0.0, "right": 0.0}


def test_spin_pays_the_commanded_direction_not_a_wiggle():
    """abs(wz) paid a pelvis wiggle almost as well as a true spin (zero net
    rotation — the user watched it). Signed pay: commanded direction earns,
    against-command charges, so oscillation nets ~zero."""
    from microduck_local.behaviors import _spin_rate

    env = BehaviorEnv("spin", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=0)
    env.reset(seed=0)
    env._spin_dir = 1.0
    env._gyro[:] = (0.0, 0.0, 3.0)
    fwd = _spin_rate(env)
    env._gyro[:] = (0.0, 0.0, -3.0)
    back = _spin_rate(env)
    assert fwd == pytest.approx(0.6) and back == pytest.approx(-0.6)
    assert fwd + back == pytest.approx(0.0)   # a wiggle nets zero
    # the direction command is in the OBSERVABLE wz slot every observation
    obs = env._get_obs()
    assert abs(float(obs[50])) == pytest.approx(1.0)
