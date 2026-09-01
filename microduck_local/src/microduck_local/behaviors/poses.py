from .core import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

_register(Behavior(
    id="one_leg",
    emoji="🦩",
    title="Stand on one leg",
    description=(
        "Balance flamingo-style: right foot in the air, all weight on the left "
        "leg — calmly, stance foot flat, no flailing."
    ),
    how_it_learns=(
        "The duck tries random wiggles at first and mostly falls over. Every 20 ms it "
        "gets a score from the recipe below; moves that kept it upright on one foot "
        "score high, falls score nothing. The learning algorithm (PPO) nudges the "
        "brain toward the high-scoring moves — millions of tries later, balance. "
        "v2 of this recipe adds calm-and-flat terms: v1 paid only for 'upright on one "
        "foot', so the duck balanced by flailing — the letter of the score, not the "
        "spirit. Now thrash costs points and a flat stance foot earns them."
    ),
    keywords=("one leg", "1 leg", "one foot", "1 foot", "flamingo", "balance on"),
    terms=(
        RewardTerm("one_leg_hold", "Big points for being upright with ONLY the left foot down", 3.0, _hold_L),
        RewardTerm("foot_in_air", "Points for holding the right foot ~5 cm off the ground", 1.5, _lift_up_L),
        RewardTerm("flat_stance_foot", "Points for keeping the left foot flat on the floor", 1.2,
                   _stance_flat("left")),
        RewardTerm("head_up", "Points for holding the head up in its natural pose", 0.8, _head_up),
        RewardTerm("head_up_pull", "Points for lifting the head toward its natural pose", 0.7,
                   _head_up_pull),
        RewardTerm("planted_foot", "Points for keeping the left foot on the ground", 0.5, _stance_L),
        _upright_term(1.5),
        RewardTerm("calm_body", "Penalty for wobbling and thrashing the body", 1.0,
                   _still_body_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 1.0,
                   _stay_home_pen, is_penalty=True),
        RewardTerm("face_home", "Penalty for twisting away from the starting direction", 1.0,
                   _face_home_pen, is_penalty=True),
        RewardTerm("stay_put", "Penalty for drifting away from the spot", 1.0, _still_penalty, is_penalty=True),
        # Heavier smoothness than the default: this is a slow, careful trick —
        # jitter is pure loss here (for a dynamic trick these weights would
        # smother discovery; see AGENTS.md on attempt-taxes).
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 2.5,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 3.0,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=2_500_000,
    success_metric="average time balanced calmly on one foot per episode",
    episode_s=20.0,
    # One-sided by construction: _one_leg("right", "left") names the RIGHT
    # foot as the one in the air, and half these terms read that name
    # (one_leg_hold, planted_foot, flat_stance_foot). The mirror loss would
    # demand the reflected state score the reflected action — i.e. that the
    # duck ALSO stand on its right foot — so it pulls against the recipe
    # instead of shaping it. Locked by tests/test_behaviors.py.
    symmetric=False,
))

def _stand_tall(env) -> float:
    """Trunk at full standing height — two-layer, wide enough to slope up
    from a collapsed crouch (the whole point: something must PAY for the
    climb, not just for being up)."""
    z = float(env._trunk_xpos[2])
    d2 = (z - env.stand_z) ** 2
    return (0.5 * float(np.exp(-d2 / 0.08 ** 2))
            + 0.5 * float(np.exp(-d2 / 0.03 ** 2)))


def _pose_home(env) -> float:
    """Joints near the canonical DEFAULT_POSE — the pose every shipped policy
    starts from, so a duck that stands HERE can hand straight off to the walk
    policy. Wide first layer: measured stands can sit ~4 rad away, where a
    tight-only Gaussian pays nothing (that exact mistake shipped once)."""
    q = env._joint_qpos()
    d2 = float(((q[C.LEG_JOINT_IDS] - C.DEFAULT_POSE[C.LEG_JOINT_IDS]) ** 2).sum())
    return (0.5 * float(np.exp(-d2 / 4.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 1.5 ** 2)))


def _stand_spawn_ground(env):
    """Start DOWN — folded in a crouch, sometimes tipped — so the recipe
    teaches getting UP, not just staying up. The first stand policy trained
    only from standing spawns and learned a hold it could never enter: it
    could not rise from a crouch, and I mistook that for the robot being
    unable to (the shipped alpha_stand rises from the very same pose, as the
    user said it would). A hold-only stand is half a skill."""
    r = env._rng
    d, m = env.data, env.model
    d.qpos[:] = 0.0
    pitch = r.uniform(-0.5, 0.5)
    d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    fold = r.uniform(0.5, 1.0)   # how deeply collapsed this start is
    q = d.qpos
    for adr in env.joint_qpos_adr:
        q[adr] = r.uniform(-0.1, 0.1)
    q[env.joint_qpos_adr[2]] = 1.2 * fold
    q[env.joint_qpos_adr[11]] = -1.2 * fold
    q[env.joint_qpos_adr[3]] = 1.3 * fold
    q[env.joint_qpos_adr[12]] = -1.4 * fold
    q[env.joint_qpos_adr[4]] = -1.4 * fold
    q[env.joint_qpos_adr[13]] = 0.4 * fold
    d.qpos[2] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = [float(d.geom_xpos[g][2]) - float(m.geom_rbound[g])
            for g in range(m.ngeom) if g != env.floor_geom]
    d.qpos[2] = 0.6 - min(lows) + 0.005
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    env.prev_joint_vel = env._joint_vel().copy()
    return env._get_obs()


_register(Behavior(
    id="stand",
    emoji="🧍",
    title="Stand still",
    description=(
        "Just stand: both feet flat, body at full height, head up, holding "
        "still in the normal standing pose."
    ),
    how_it_learns=(
        "The simplest trick in the book, and the one everything else is built "
        "on. It pays for being TALL (not merely upright — a duck can face the "
        "sky in a collapsed heap), for both feet planted flat, for a level "
        "head, and for holding the normal standing pose rather than some "
        "invented stance. Because it is a hold, episodes are long: surviving "
        "a few seconds is not the same as finding a posture you can keep."
    ),
    keywords=("stand still", "stand steady", "standing still", "just stand",
              "hold still", "stand up straight", "both feet", "stand on both"),
    terms=(
        RewardTerm("stand_tall", "Big points for standing at full height", 3.0, _stand_tall),
        RewardTerm("stay_upright", "Points for keeping the body upright", 2.0, _upright),
        RewardTerm("both_feet", "Points for both feet on the ground", 1.5, _both_feet_down),
        RewardTerm("flat_feet", "Points for keeping the feet flat on the floor", 1.0, _flat_feet),
        RewardTerm("head_up", "Points for holding the head level and up", 1.5, _head_up_blend),
        RewardTerm("normal_pose", "Points for standing in the normal ready pose", 1.5, _pose_home),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 1.0,
                   _stay_home_pen, is_penalty=True),
        RewardTerm("face_home", "Penalty for turning away from the starting direction", 0.5,
                   _face_home_pen, is_penalty=True),
        RewardTerm("hold_still", "Penalty for swaying and fidgeting", 1.0,
                   _still_body_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.5,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 1.0,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 1.0,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=2_000_000,
    success_metric="unbroken seconds at full standing height on both feet",
    episode_s=20.0,  # a hold task: short clips never demand a real equilibrium
    # Half the episodes start collapsed on the floor: standing up IS the
    # skill, and a policy that only ever starts upright never learns it.
    spawn_families=((0.5, _stand_spawn_ground),),
))


_register(Behavior(
    id="crouch",
    emoji="🐥",
    title="Crouch down low",
    description="Bend the knees and hold a steady squat about 3.5 cm lower than normal standing.",
    how_it_learns=(
        "The recipe pays the duck for having its body at the crouch height while staying "
        "level with both feet planted. At first it just stands (wrong height, few points) "
        "or collapses (not level, no points); PPO gradually finds the knee bend that "
        "collects all the points at once."
    ),
    keywords=("crouch", "squat", "duck down", "get low", "kneel", "bend"),
    terms=(
        RewardTerm("crouch_height", "Big points for holding the body ~3.5 cm lower than standing", 3.0, _crouch_height),
        RewardTerm("feet_planted", "Points for keeping both feet on the ground", 1.0, _both_feet_down),
        RewardTerm("head_up", "Points for holding the head up in its natural pose", 1.0, _head_up_blend),
        RewardTerm("flat_feet", "Points for keeping the feet flat on the floor", 0.8, _flat_feet),
        RewardTerm("face_home", "Penalty for twisting away from the starting direction", 0.8,
                   _face_home_pen, is_penalty=True),
        _upright_term(1.5),
        RewardTerm("stay_put", "Penalty for drifting away from the spot", 1.0, _still_penalty, is_penalty=True),
        *_BASE_REGULARIZERS,
    ),
    default_steps=1_500_000,
    success_metric="how close the body height sits to the crouch target",
))

_register(Behavior(
    id="spin",
    emoji="🌀",
    title="Spin in place",
    description="Turn on the spot as fast as it can without falling or walking away.",
    how_it_learns=(
        "Points flow for yaw speed — but only while upright, and drifting away from the "
        "spot costs points. Early attempts are wild lunges that fall instantly; the ones "
        "that rotate AND survive score more, so the policy converges on a stable pirouette."
    ),
    keywords=("spin", "turn around", "pirouette", "rotate", "twirl"),
    terms=(
        RewardTerm("spin_fast", "Points for yaw speed (capped — no points for violence)", 2.5, _spin_rate),
        RewardTerm("head_up", "Points for holding the head up in its natural pose", 0.6, _head_up_blend),
        _upright_term(1.5),
        RewardTerm("stay_put", "Penalty for drifting away from the spot", 1.5, _still_penalty, is_penalty=True),
        # POSITION anchor, not just velocity: stay_put prices drift SPEED and
        # a slow shuffle beats it (its sibling's own docstring says so) — the
        # user asked for "stays in place rather than moving away from the
        # origin". Position is not observable, but spin drift is a SYSTEMATIC
        # gait asymmetry, so the anchor's pressure transfers to the
        # (observable) gait choice — same mechanism that keeps stand/one_leg
        # planted.
        RewardTerm("stay_home", "Penalty for ending up away from where the spin started",
                   1.0, _stay_home_pen, is_penalty=True),
        RewardTerm("step_dont_skid", "Points for lifting the feet to step around (no skid-steering)",
                   1.5, _step_dont_skid),
        *_BASE_REGULARIZERS,
    ),
    default_steps=1_500_000,
    success_metric="average turning speed while upright",
))




# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
