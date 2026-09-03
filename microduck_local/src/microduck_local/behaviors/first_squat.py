"""My first custom behavior — a calm deep squat (深蹲并保持).

Why this exists
---------------
This is a deliberately small "hello world" for reward engineering. The goal is
NOT to invent a hard trick; it is to *see* how a reward is built:

  * one positive shaping term  -> what we want the duck to DO  (_squat_z)
  * a stack of penalty terms   -> what we DON'T want          (thrash / drift / jitter)

Change the target depth (see ``_squat_target_z``) and the weights, re-run, and
watch the behavior change. That loop — edit reward, watch policy — IS physical-AI
software work. Everything here reuses the catalog helpers in ``core.py`` so it
runs on the CPU trainer with no GPU.

How to run
----------
* Official trainer (Linux/macOS, fork-based vec-env):
    uv run train-behavior deep_squat
    uv run eval-walk runs/deep_squat/policy.onnx
    uv run render-rollout runs/deep_squat/policy.onnx   # read the contact sheet!

* Windows-native CPU fallback (no GPU needed):
    set MICRODUCK_RL_DIR=<path-to-microduck_rl>
    python train_deep_squat.py          # PPO + plots + comparison report
"""
import numpy as np

from .core import (
    Behavior,
    RewardTerm,
    _register,
    _head_up,
    _flat_feet,
    _upright_term,
    _still_body_pen,
    _stay_home_pen,
    _action_rate_pen,
    _joint_vel_pen,
    _torque_pen,
    _limit_parking_pen,
)


def _squat_target_z(env) -> float:
    """Trunk height the squat aims for — the deepest the EPISODE allows.

    First draft used ``stand_z * 0.55`` (= 0.066 m here) and it was silently
    unlearnable: ``WalkEnv.FALL_HEIGHT`` is 0.07 m, so the duck got terminated
    the instant it got *close* to the target. The reward was pointing at a
    state the environment kills on sight.

    Lesson (the single most transferable thing in this file): a shaping target
    must live in the region the env does not terminate. So the target is an
    OFFSET below standing, floored by the fall threshold plus a margin wide
    enough for the wobble of a real policy.

    For reference poses.py's `crouch` uses a flat ``stand_z - 0.035``; this is
    deliberately a touch deeper, which is what makes it a *deep* squat.
    """
    fall_z = float(getattr(env, "FALL_HEIGHT", 0.07))
    return max(env.stand_z - 0.040, fall_z + 0.012)


def _squat_z(env) -> float:
    """Trunk near the squat target height.

    Two-layer Gaussian (wide + tight) exactly like ``_stand_tall`` in poses.py.
    Returns >= 0 (a shaping term).

    Sigmas are set against the REAL stand->target distance, which is only
    ~3.8 cm on this robot (not 10 cm — see `_squat_target_z`). Measured payout:
    standing ~0.49, halfway ~0.80, target 1.00. That is the shape you want —
    a gradient the policy can climb from where it starts, but with the target
    still the strict argmax, and collapsing past it paying less, so
    overshooting down is never a jackpot.
    """
    z = float(env._trunk_xpos[2])
    d2 = (z - _squat_target_z(env)) ** 2
    return (0.5 * float(np.exp(-d2 / 0.075 ** 2))
            + 0.5 * float(np.exp(-d2 / 0.030 ** 2)))


_register(Behavior(
    id="deep_squat",
    emoji="\U0001F986",  # 🦆
    title="Calm deep squat",
    description=(
        "Lower the body as deep as the balance allows and hold it there, calmly, "
        "feet flat, head up — then rise back when the command releases."
    ),
    how_it_learns=(
        "Random wiggles at first; episodes score the trunk being near the target "
        "height and penalize thrashing. PPO nudges toward the calm-low pose. This is "
        "your first reward-engineering exercise: change the depth in "
        "`_squat_target_z` (but never below FALL_HEIGHT, or the env kills the "
        "episode on arrival) and the weights below, re-run, and watch the behavior "
        "change."
    ),
    # NOTE: keep these distinct from poses.py's `crouch` recipe, which owns the
    # bare words "crouch"/"squat" (test_matcher asserts "crouch down" -> crouch).
    # `match_behavior` scores by total keyword length, so the longer, more
    # specific "deep squat" (1.10) still outranks crouch's "squat" (1.05).
    keywords=("deep squat", "squat deep", "squat low", "深蹲", "下蹲", "蹲下去"),
    terms=(
        # --- what we WANT (positive shaping, returns >= 0) ---
        RewardTerm("squat_height", "Big points for trunk near ~68% stand", 3.0, _squat_z),
        RewardTerm("head_up", "Points for holding the head up in its natural pose", 0.8, _head_up),
        RewardTerm("flat_feet", "Points for keeping both feet flat on the floor", 1.0, _flat_feet),
        _upright_term(1.0),  # returns a RewardTerm directly

        # --- what we DON'T want (penalties, fn returns <= 0, is_penalty=True) ---
        RewardTerm("calm_body", "Penalty for wobbling/thrashing the body", 1.0,
                   _still_body_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the start spot", 0.8,
                   _stay_home_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 2.0,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing joints fast", 2.5,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 0.8,
                   _torque_pen, is_penalty=True),
        RewardTerm("off_limits", "Penalty for parking joints at their hard stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
    ),
    default_steps=2_000_000,
    success_metric="average time held calmly near target height per episode",
    episode_s=12.0,        # a slow hold pose -> long episodes so it finds a real equilibrium
    scene="walk",          # 'walk' strips head/trunk floor contacts (falling is cheap)
    terminate_on_fall=True,
    symmetric=True,        # squat is left/right symmetric -> mirror loss helps sample efficiency
))
