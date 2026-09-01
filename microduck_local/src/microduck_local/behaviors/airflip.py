from .backflip import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------- airflip (raw, no scaffolds)
# Deliberately the OPPOSITE of the staged backflip: no curriculum, no
# mid-trick spawns, no spotter. Every episode starts standing and the duck
# either jumps and throws itself over or it doesn't. If this fails it fails
# honestly, and the null control below says so.

def _af_airborne(env) -> float:
    """Both feet off the floor, paid by clearance — a jump, not a shuffle."""
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    return float(np.clip(min(zl, zr) / 0.06, 0.0, 1.0))


def _af_head_throw(env) -> float:
    """Head thrown BACK during the launch (first quarter-turn): with 38% of
    the mass in the head, slinging it backward is this body's main source of
    angular momentum — the user's 'jump and move its head back at the same
    time'. Pays the throw's SPEED, so it rewards the whip, not a pose."""
    if env._bf_rot > 1.6:
        return 0.0
    qv = env._joint_vel()
    return min(1.0, (max(0.0, float(qv[5])) + max(0.0, float(qv[6]))) / 8.0)


def _af_head_straight(env) -> float:
    """Keep the flip in the sagittal plane — the same vector it started on.

    A pure backward flip rotates about the body's y-axis, so gravity's
    SIDEWAYS component in the trunk frame stays ~0 throughout; any of it is
    the flip corkscrewing off-axis. Priced together with the head's own yaw
    and roll joints, because the duck was throwing its head back and then
    turning it to the side, which spends the throw sideways instead of into
    the rotation (the user watched it happen)."""
    g = env._projected_gravity()
    q = env._joint_qpos()
    d2 = 4.0 * float(g[1] ** 2) + float(q[7] ** 2 + q[8] ** 2)
    return (0.5 * float(np.exp(-d2 / 1.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 0.35 ** 2)))


def _af_tuck_spin(env) -> float:
    """Ball up while rotating — pays LOW moment of inertia about the flip
    axis, which is the physical reason a tuck spins faster.

    Measured on this model: extended 4.46 g·m², feet-over-head-with-straight-
    legs 3.77, tight tuck (knees to chest) 2.71 — so a tuck spins ~1.6× faster
    for the same launch. Reference footage of a comparable robot shows exactly
    this: compact through the entire rotation, opening only to land. Note the
    correction this encodes — asking for feet HIGH above the head with
    straight legs (an inverted layout) makes the body slower to rotate, the
    opposite of what a flip needs."""
    if float(env._gyro[1]) > -0.3:
        return 0.0   # only while actually rotating backward
    com = env.data.subtree_com[0]
    inertia = 0.0
    for b in range(1, env.model.nbody):
        r = env.data.xipos[b] - com
        inertia += float(env.model.body_mass[b]) * float(r[0] ** 2 + r[2] ** 2)
    return float(np.clip((4.6e-3 - inertia) / (4.6e-3 - 2.7e-3), 0.0, 1.0))


def _af_legs_over_head(env) -> float:
    """Feet coming up over the head as the body comes around — the user's
    kick-over, but targeted at a height a TUCKED body can actually reach
    (~10 cm above the head, not the 20 cm an extended layout would need).
    Shape and inertia are priced separately: this says where, _af_tuck_spin
    says how compact."""
    if float(env._gyro[1]) > -0.3:
        return 0.0
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    zh = float(env.data.xpos[_jaw_bid(env)][2])
    d2 = ((zl + zr) / 2.0 - zh - 0.10) ** 2
    return (0.5 * float(np.exp(-d2 / 0.25 ** 2))
            + 0.5 * float(np.exp(-d2 / 0.10 ** 2)))


def _af_spin_rate(env) -> float:
    """Backward pitch RATE — the thing a flip actually needs. Paid whenever
    it is rotating backward, so leaning, launching and tumbling all get
    credit for the same underlying quantity."""
    wy = float(env._gyro[1])
    return min(1.0, max(0.0, -wy) / 8.0)


_register(Behavior(
    id="airflip",
    emoji="🤾",
    title="Jump backflip",
    description=(
        "Jump and throw the head back at the same time, rotate backwards in "
        "the air, and land on both feet."
    ),
    how_it_learns=(
        "No stages, no help: every attempt starts from standing. It earns for "
        "getting both feet off the floor, for slinging its heavy head "
        "backwards as it launches, for backward spin however it is produced, "
        "and — the real prize — for coming down on both feet after most of a "
        "turn. Whether this body can generate enough air and spin to get "
        "around is exactly what the run answers."
    ),
    keywords=("jump backflip", "jumping backflip", "jump flip", "air flip",
              "aerial flip", "flip in the air", "jump and flip"),
    terms=(
        RewardTerm("land_it", "Big points for landing on both feet after a full turn",
                   6.0, _bf_landed_hold),
        RewardTerm("get_air", "Points for jumping — both feet off the floor", 2.5, _af_airborne),
        RewardTerm("head_throw", "Points for slinging the head back as it launches", 2.5,
                   _af_head_throw),
        RewardTerm("tuck_spin", "Points for balling up tight so the spin is fast", 3.0,
                   _af_tuck_spin),
        RewardTerm("legs_over_head", "Points for the feet coming up over the head", 2.0,
                   _af_legs_over_head),
        RewardTerm("head_straight", "Points for flipping straight back, not turning to the side",
                   2.0, _af_head_straight),
        RewardTerm("spin_back", "Points for backward rotation speed", 2.0, _af_spin_rate),
        RewardTerm("flip_progress", "Points as the backward rotation adds up", 2.0, _bf_progress),
        RewardTerm("straight_flip", "Penalty for twisting sideways out of the flip", 0.5,
                   _bf_straight_pen, is_penalty=True),
        RewardTerm("gentle_head", "Penalty each time the head hits the floor, scaled by impact",
                   1.0, _gentle_plant_pen, is_penalty=True),
        RewardTerm("soft_landings", "Penalty for slamming down hard", 0.75,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 0.4,
                   _stay_home_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 0.5,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="full backward turns in the air that end on both feet",
    episode_s=8.0,
    scene="all",
    terminate_on_fall=False,
    state_fn=_bf_update,
))




# ------------------------------------------------------- the shared library
# Terms general enough to help ANY trick, promoted out of the recipe that
# first needed them so the viewer's "＋ add a term" picker can offer them
# everywhere. An audit found 62 distinct terms in use but only 9 in the
# catalog — save_energy alone had been re-declared in eight recipes — so the
# picker was showing a fraction of what we had actually built.
#
# Promotion criterion: the term must be meaningful for a behavior that is not
# a flip. Terms gated on trick-specific progress (the roll accumulator, the
# clip phase) stay private to their recipes, because elsewhere they would sit
# at a permanent 0.00 and read as "not felt".
CATALOG.update({t.key: t for t in (
    RewardTerm("stay_upright", "Points for keeping the body upright", 1.5, _upright),
    RewardTerm("stand_tall", "Points for standing at full height, not crouching",
               2.0, _stand_tall),
    RewardTerm("normal_pose", "Points for holding the normal ready pose (walk-ready)",
               1.0, _pose_home),
    RewardTerm("both_feet", "Points for keeping both feet on the ground", 1.0,
               _both_feet_down),
    RewardTerm("head_straight", "Points for facing straight ahead, not turning to the side",
               1.5, _af_head_straight),
    RewardTerm("tuck_spin", "Points for balling up tight (spins faster, lands softer)",
               1.5, _af_tuck_spin),
    RewardTerm("get_air", "Points for getting both feet off the floor", 1.5, _af_airborne),
    RewardTerm("spin_back", "Points for rotating backwards", 1.5, _af_spin_rate),
    RewardTerm("stay_put", "Penalty for drifting off the spot", 1.0,
               _still_penalty, is_penalty=True),
    RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 1.0,
               _action_rate_pen, is_penalty=True),
    RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 1.0,
               _joint_vel_pen, is_penalty=True),
    RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
               _torque_pen, is_penalty=True),
    RewardTerm("gentle_head", "Penalty each time the head hits the floor, scaled by impact",
               1.0, _gentle_plant_pen, is_penalty=True),
)})




# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
