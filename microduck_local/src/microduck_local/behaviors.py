"""Teachable behaviors: small reward recipes on top of the walking env.

Each behavior is what a user asks for in plain words ("stand on one leg") plus
the machine version of that sentence: a handful of reward terms. The friendly
strings are shown in the viewer's teach panel, so a non-ML user can see exactly
what the duck is being scored on.

Design rules follow microduck_rl/AGENTS.md: penalties are self-negating
(<= 0) with positive weights; positive terms are Gaussians (no jackpots);
commands stay zero-but-alive so the 61-obs contract holds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Callable

import mujoco
import numpy as np

from . import contract as C
from . import motion
from .walk_env import MicroduckWalkEnv


@dataclass(frozen=True)
class RewardTerm:
    key: str
    friendly: str          # one plain-English sentence, shown in the UI
    weight: float
    fn: Callable           # fn(env: BehaviorEnv) -> float (unweighted)
    is_penalty: bool = False


@dataclass(frozen=True)
class CurriculumStage:
    """One stage of a staged curriculum: a separate training run that
    fine-tunes from the stage before it. `env` is extra environment variables
    for the stage's trainer subprocess — how per-stage knobs (e.g. spawn
    windows) reach the env without this schema learning any trick's
    specifics."""
    label: str             # plain English — the viewer narrates "stage 2 of 3 · <label>"
    steps: int
    env: dict[str, str] = field(default_factory=dict)
    # One or two plain sentences on what the stage actually rehearses (spawn
    # window/mix in words) — the viewer's stage inspector shows this. The
    # label says WHERE the chain is; detail says WHAT the practice looks like.
    detail: str = ""


@dataclass(frozen=True)
class Behavior:
    id: str
    emoji: str
    title: str
    description: str       # what the duck will learn, one sentence
    how_it_learns: str     # 2-3 plain sentences for the explainer card
    keywords: tuple[str, ...]
    terms: tuple[RewardTerm, ...] = field(default=())
    default_steps: int = 2_000_000
    success_metric: str = ""
    # Is the mirror loss (symmetry.py) a VALID prior for this recipe? It pays
    # for pi(M o) == M pi(o) — left/right consistency for free, and the one
    # lever that buys sample EFFICIENCY rather than more samples, which is the
    # only currency this CPU trainer has. But it is a WRONG prior for a recipe
    # that names a side: it then fights the reward instead of shaping it.
    # False here defaults --symmetry-coef to 0 (train_behavior.symmetry_coef_for);
    # an explicit --symmetry-coef still wins, in both directions.
    symmetric: bool = True
    # Training episode length. Static hold-a-pose tricks want LONG episodes:
    # with 8 s clips + gamma 0.99 (~2 s effective horizon) a policy is never
    # economically pressured toward a true stationary equilibrium — it just
    # survives the clip.
    episode_s: float = 8.0
    # "walk" strips head/trunk floor contacts (falling is cheap — right for
    # standing tricks); "all" lets the body rest on the ground (headstand).
    scene: str = "walk"
    # Inverted tricks ARE what the walk env calls "fallen" — disable for them.
    terminate_on_fall: bool = True
    # Reverse curriculum: fraction of episodes spawned already IN the trick's
    # end state (playbook: the reliable fix for "learns the start, never the
    # last mile" — the goal state otherwise gets no on-policy data).
    inverted_spawn_prob: float = 0.0
    # ... and spawned MID-maneuver (face-plant tripod, feet still down, the
    # kick-over's launch position) — the bridge between start and goal
    # otherwise never gets practiced (a scratch run held perfect spawned
    # headstands but went 0/4 flipping from upright).
    mid_flip_spawn_prob: float = 0.0
    # Called once per control step before the terms — for behaviors whose
    # reward needs MEMORY of the episode so far (the backflip's cumulative
    # rotation). A plain term can't own this: term fns may run under a zero
    # weight or in any order, so integration lives in one dedicated hook.
    state_fn: Callable | None = None
    # Generalized reverse-curriculum spawns: (probability, fn(env) -> obs)
    # pairs, tried in order against one uniform draw. The two prob fields
    # above are the headstand's legacy spelling of the same idea; new
    # behaviors bring their own spawn poses here.
    spawn_families: tuple = ()
    # Reference motion this behavior imitates (a clip authored in the
    # viewer's timeline editor, resolved from the clips dir at env build).
    clip_name: str | None = None
    # Forward speed (m/s) written into the twist command every step. Non-zero
    # turns a behavior into a LOCOMOTION task: the policy sees the command in
    # its observation and is paid for achieving it, exactly as the shipped
    # walking policy was trained.
    forward_cmd: float = 0.0
    # DEMO-ONLY assist: fn(env) -> bool, called before each control step when
    # an env is built with spotter=True (showcase previews only — training
    # never sets it, or the policy would learn to lean on a hand that isn't
    # there on the robot). Returns whether it is currently assisting.
    spotter_fn: Callable | None = None
    # Staged curriculum (CurriculumStage per stage): complex tricks train as a
    # CHAIN of runs, each fine-tuning from the previous under its own env
    # knobs. Empty = ordinary single-run training. The farm server
    # orchestrates the chain on /teach; the env dicts keep the mechanism
    # generic — behaviors own their knobs, the server just exports them into
    # each stage's subprocess environment.
    curriculum: tuple = ()


def resolve_clip_name(behavior: Behavior, clip_name: str | None = None) -> str | None:
    """The reference clip a run of `behavior` will actually imitate.

    Explicit kwarg wins, then MICRODUCK_CLIP (how viz_server hands a clip to
    the trainer subprocess), then the recipe's default. Shared by BehaviorEnv
    and by train_behavior's symmetry decision so the trainer cannot reason
    about a different clip than the one the env loads.
    """
    return clip_name or os.environ.get("MICRODUCK_CLIP") or behavior.clip_name


def is_symmetric(behavior: Behavior, clip_name: str | None = None) -> bool:
    """Is the bilateral mirror loss a valid prior for THIS run?

    Two ways to lose it: the recipe declares itself one-sided
    (``Behavior.symmetric``), or the run carries a reference clip. A clip puts
    its phase in two command slots and the mirror map negates one of them, so
    mirroring an observation also time-reflects the clip — see the note on the
    `imitate` recipe. Any behavior can be handed a clip (viz_server's /teach
    takes one per run), so this is a per-RUN question, not a per-recipe one.
    """
    return behavior.symmetric and not resolve_clip_name(behavior, clip_name)


def _spawn_knob(env, key: str, default: str | None = None) -> str | None:
    """Per-stage spawn knob lookup: the env instance's overrides first, then
    the process environment, then the default. The trainer subprocess gets
    stage knobs via its environment (one process, one stage), but the farm
    hosts MANY preview envs in one process — its trainee must carry the
    active stage's knobs per instance or it silently previews the wrong
    curriculum (the user watched stage 1 and saw only stand-then-topple)."""
    ov = getattr(env, "spawn_overrides", None)
    if ov and key in ov:
        return ov[key]
    return os.environ.get(key, default)


# ---------------------------------------------------------------- reward fns
# Each takes the env; env exposes .data, .model, foot helpers, etc.

def _base_vel(env) -> tuple[float, float, float]:
    """(forward, lateral, vertical) speed of the trunk, in the HEADING frame.

    Thin wrapper around ``MicroduckWalkEnv.heading_lin_vel`` — that is also
    where the ``mj_objectVelocity(flg_local=1)`` trap is documented. Run
    rewards want ground-plane forward (a dive has body-x speed); walk_env
    tracking uses the body frame instead, matching the twist command.
    """
    return env.heading_lin_vel()


def _foot_z(env, side: str) -> float:
    gid = env.foot_geoms[side]
    return float(env.data.geom_xpos[gid][2])


def _v6_buf(env) -> np.ndarray:
    """Reusable 6-vector for mj_objectVelocity (which overwrites all six
    slots), instead of a fresh np.zeros(6) per foot per term per step."""
    v6 = getattr(env, "_v6_scratch", None)
    if v6 is None:
        v6 = env._v6_scratch = np.zeros(6)
    return v6


def _upright(env) -> float:
    g = env._projected_gravity()
    return float(np.exp(-(g[0] ** 2 + g[1] ** 2) / 0.05))


def _still_penalty(env) -> float:
    """Discourage drifting away: penalize horizontal base speed (<= 0)."""
    fwd, lat, _ = _base_vel(env)
    return -0.5 * (fwd * fwd + lat * lat)


def _one_leg(lift: str, stance: str):
    def lifted_foot_up(env) -> float:
        contacts = env.foot_contact_state
        if contacts[lift]:
            return 0.0  # no credit while the "lifted" foot still touches
        h = _foot_z(env, lift) - _foot_z(env, stance)
        # Target 8 cm (was 5): a user cranked this term's WEIGHT trying to get
        # the leg higher — but the weight only pays more for the same target;
        # "higher" lives HERE. std 4 cm keeps gradient alive for a warm start
        # currently holding ~5 cm.
        return float(np.exp(-((h - 0.08) ** 2) / 0.04 ** 2))

    def stance_planted(env) -> float:
        return 1.0 if env.foot_contact_state[stance] else 0.0

    def balance_hold(env) -> float:
        # The jackpot-free main dish: upright AND on one foot, paid per step.
        contacts = env.foot_contact_state
        one_foot = contacts[stance] and not contacts[lift]
        return _upright(env) if one_foot else 0.0

    return lifted_foot_up, stance_planted, balance_hold


def _crouch_height(env) -> float:
    """Two-layer height target (wide pull + tight polish). The tight-only
    version paid ~0 at STANDING height (35 mm off, 12 mm std), so once the
    posture terms (head_up/flat_feet) made standing lucrative, the policy
    stopped crouching entirely — the third zero-gradient-where-the-policy-is
    bug (after head_up and flat_stance_foot). The wide layer slopes all the
    way from standing down to the target."""
    z = float(env._trunk_xpos[2])
    d2 = (z - (env.stand_z - 0.035)) ** 2  # target ~3.5 cm below standing
    return 0.5 * float(np.exp(-d2 / 0.035 ** 2)) + 0.5 * float(np.exp(-d2 / 0.012 ** 2))


def _both_feet_down(env) -> float:
    c = env.foot_contact_state
    return 1.0 if (c["left"] and c["right"]) else 0.0


def _stay_home_pen(env) -> float:
    """POSITION anchor (<= 0): distance² from where the episode started.
    The velocity-based stay_put penalty cannot stop a slow shuffle — creeping
    at 0.2 m/s costs ~0.02/step while covering half the floor (a user watched
    a 2.5M run drift away despite maxed smoothness taxes). Charging the
    accumulated DISPLACEMENT makes drifting expensive no matter how gently
    it's done."""
    d = env._trunk_xpos
    home = getattr(env, "home_xy", None)
    if home is None:
        return 0.0
    dist2 = float((d[0] - home[0]) ** 2 + (d[1] - home[1]) ** 2)
    return -6.0 * dist2


def _trunk_yaw(env) -> float:
    qw, qx, qy, qz = env._trunk_xquat
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _face_home_pen(env) -> float:
    """HEADING anchor (<= 0): yaw² away from the episode's starting direction
    — the rotational sibling of stay_home. Slow creeping spin during a static
    trick is nearly free under velocity/wobble penalties, exactly like the
    positional shuffle was."""
    home = getattr(env, "home_yaw", None)
    if home is None:
        return 0.0
    # Anchor ONLY while the command is straight-ahead (or zero — tricks).
    # Unconditional, this charged the run policy for OBEYING a commanded
    # turn/sidestep: with the omni bucket in ~43% of episodes, obeying a turn
    # saturated the bound within 0.63 s and tracking a turn netted ~0 where
    # the GPU stack pays up to +1.73 (audit finding #1). A steering command
    # hands the heading to the commander; the anchor owns it only when the
    # command says "straight".
    cmd = getattr(env, "twist_cmd", None)
    if cmd is not None and (abs(float(cmd[1])) > 0.05 or abs(float(cmd[2])) > 0.1):
        return 0.0
    dyaw = _trunk_yaw(env) - home
    dyaw = float(np.arctan2(np.sin(dyaw), np.cos(dyaw)))  # wrap to [-pi, pi]
    # Bounded like every other penalty: saturates one unit at ~36 deg off
    # heading (was unbounded, worst measured -74/step at recipe weight 3).
    return -min(2.5 * dyaw * dyaw, 1.0)


def _still_body_pen(env) -> float:
    """Whole-body angular-velocity penalty (<= 0). A motion-blocker — safe for
    STATIC tasks only (AGENTS.md: never tax rotation a dynamic trick needs)."""
    w = env._gyro
    return -0.03 * float(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)


def _stance_flat(side: str):
    """Foot-flatness: gravity seen in the foot's own frame must match what it
    looks like in a flat stand. Self-calibrating — the reference is measured
    off the STAND keyframe at env init, so no axis conventions are assumed.

    std 0.45, not 0.15: at 0.15 a mildly rolled foot scored ~0, so the term
    never produced a gradient and finished runs earned 0.01 of its 0.8 max
    (v2/22f079 postmortem). Price the escapable part, not perfection.
    """
    def flat(env) -> float:
        ref = env.foot_flat_ref[side]
        # -row 2 of xmat IS R.T @ [0,0,-1]: the products with the two zeros
        # are exact, so only a zero's sign can differ — and it dies in d2.
        g_local = -env.data.geom_xmat[env.foot_geoms[side]][6:9]
        d2 = float(((g_local - ref) ** 2).sum())
        return float(np.exp(-d2 / 0.45 ** 2))
    return flat


def _head_up(env) -> float:
    """Head/neck joints at their natural pose — the tight "polish" layer.
    Without a head term a drooped head is FREE, and actively useful as a
    counterweight (38% of body mass), so every balance policy slumps."""
    d = env._joint_pos_rel()[C.HEAD_JOINT_IDS]
    return float(np.exp(-float((d * d).sum()) / 0.3 ** 2))


def _head_up_pull(env) -> float:
    """The wide layer of the two-layer head reward. std 1.2 rad: the v3
    postmortem measured the slumped head ~59° (≈1 rad RMS) off pose, where the
    tight layer pays exp(-47) ≈ 0 — zero gradient, zero learning (same failure
    as the v2 foot-flatness term). This layer still pays ~0.05 out there and
    slopes all the way home; the tight layer takes over for the last degrees."""
    d = env._joint_pos_rel()[C.HEAD_JOINT_IDS]
    return float(np.exp(-float((d * d).sum()) / 1.2 ** 2))


def _spin_rate(env) -> float:
    wz = float(env._gyro[2])
    # Pay for yaw speed up to ~4 rad/s (physically natural for a 25 cm robot),
    # saturating instead of exploding — no reward for violence beyond it.
    return float(np.clip(abs(wz) / 4.0, 0.0, 1.0))


# Penalty normalizers. Positive terms are all bounded [0, 1] and scaled by
# their weight, so "weight 2.0" means "worth two units". Penalties had no such
# convention: each carried a hand-picked coefficient, and `save_energy` came
# out ~77x too small to matter — under a trained policy it contributed 0.002
# against a per-step reward of ~4.5, i.e. it was priced at zero. That is why
# raising its WEIGHT during the backflip safety work never changed behavior:
# the weight was multiplying a number that was already negligible.
#
# Both constants are measured against a PHYSICAL maximum, not against observed
# flailing (an observed-worst-case denominator makes a penalty saturate at 1.0
# for merely-typical motion, which is a different bug):
#   TAU2_MAX   all 14 joints pinned at the XL330's 0.96 N.m force limit.
#   QVEL2_MAX  peak per-joint speed under full-amplitude random drive
#              (~8-9 rad/s), summed in quadrature.
# 14 * 0.96**2 — the XML servo's ±0.96 Nm forcerange, which is a VOLTAGE
# ceiling with no current limit. Only correct for actuator="xml"; see
# _tau2_max, which picks the right saturation for the actuator in use.
TAU2_MAX = 12.90
QVEL2_MAX = 990.0    # measured; see tests/test_penalty_scale.py


def _action_rate_pen(env) -> float:
    """Jerk penalty. Deliberately NOT renormalized like the two below.

    Actions are joint-target offsets in radians at scale 1.0, over a +/-4 rad
    action space, so the algebraic maximum (sum = 896) is far outside anything
    reachable and dividing by it would silence the term. The physically
    meaningful ceiling is the opposite extreme -- a joint can only travel
    vmax*dt ~= 0.18 rad per control step, so sum = 0.45 -- and normalizing
    THERE would make ordinary stepping cost ~86 per step, an attempt tax that
    smothers discovery (AGENTS.md). The existing 0.02 already puts this at
    -0.3..-0.8 per step, the second-largest term in the run recipe, so it is
    doing its job and is left alone until the learning-rate fix lands: most of
    the chatter it is currently measuring looks like an optimizer artifact,
    not a reward-shaping failure."""
    return -0.02 * float(((env.last_action - env.prev_action) ** 2).sum())


def _joint_vel_pen(env) -> float:
    """-1.0 when every joint is slewing at its peak observed speed."""
    return -float((env._joint_vel() ** 2).sum()) / QVEL2_MAX


def _tau2_max(env) -> float:
    """Sum of squared actuator forces at TRUE saturation, per actuator model.

    A module constant was wrong here. `model.actuator_forcerange` reads 0.96
    Nm under BOTH actuators, but the BAM model clamps to the XL330's real
    firmware current limit, kt * 1.75 A = 0.640 Nm — so normalizing against
    12.90 under-charged motor strain by (0.96/0.640)**2 = 2.25x on exactly the
    actuator that models the real robot.
    """
    v = getattr(env, "_tau2_max_cache", None)
    if v is None:
        if getattr(env, "actuator_model", "xml") == "bam":
            from .bam_actuator import XL330_MAX_TORQUE
            v = float(C.NUM_JOINTS * XL330_MAX_TORQUE ** 2)
        else:
            v = float(np.sum(env.model.actuator_forcerange[:, 1] ** 2))
        env._tau2_max_cache = v
    return v


def _torque_pen(env) -> float:
    """-1.0 exactly when all 14 motors are pinned at their force limit."""
    return -float((env._act_force ** 2).sum()) / _tau2_max(env)


_BASE_REGULARIZERS = (
    RewardTerm("smooth_moves", "Small penalty for jerky, twitchy movements", 1.0,
               _action_rate_pen, is_penalty=True),
    RewardTerm("gentle_joints", "Small penalty for flailing the joints fast", 1.0,
               _joint_vel_pen, is_penalty=True),
    RewardTerm("save_energy", "Small penalty for straining the motors", 1.0,
               _torque_pen, is_penalty=True),
)


def _upright_term(w=1.5):
    return RewardTerm("stay_upright", "Points for keeping the body level", w,
                      _upright)


_lift_up_L, _stance_L, _hold_L = _one_leg("right", "left")

BEHAVIORS: dict[str, Behavior] = {}


# ------------------------------------------------------------- term catalog
# Optional, composable terms any behavior can adopt — mirrors the reward
# vocabulary of the official microduck_rl stack (head/pose holds, limit
# guards, torque-rate smoothness, impact softness). The teach UI lists these
# under "＋ add a term"; /teach passes {"extraTerms": {key: weight}} and
# BehaviorEnv composes them into the recipe at train time. One slider each —
# multi-layer terms (head_up) blend their layers internally.

def _head_up_blend(env) -> float:
    """Two-layer head-at-natural-pose (wide pull + tight polish, blended) —
    the wide layer keeps gradient alive even ~1 rad off (v3 postmortem)."""
    d2 = float((env._joint_pos_rel()[C.HEAD_JOINT_IDS] ** 2).sum())
    return 0.5 * float(np.exp(-d2 / 1.2 ** 2)) + 0.5 * float(np.exp(-d2 / 0.3 ** 2))


def _flat_feet(env) -> float:
    """Both feet flat (mean of the per-foot self-calibrated flatness)."""
    total = 0.0
    for side in ("left", "right"):
        ref = env.foot_flat_ref[side]
        # See _stance_flat: -xmat row 2 == R.T @ [0,0,-1], bit-held in d2.
        g_local = -env.data.geom_xmat[env.foot_geoms[side]][6:9]
        total += float(np.exp(-float(((g_local - ref) ** 2).sum()) / 0.45 ** 2))
    return total / 2.0


def _limit_parking_pen(env) -> float:
    """Penalty for joints camping near their hard stops (<= 0). The stock
    dof_pos_limits only fires in the last ~7.5% of range (AGENTS.md); this
    charges the outer 12% quadratically."""
    # (mid, half) are compile-time constants of the model, but this used to
    # rebuild them per STEP via 14 model.joint(name) lookups — measured
    # 40-50 us, the single most expensive line of the whole reward stack.
    ranges = getattr(env, "_limit_park_ranges", None)
    if ranges is None:
        m = env.model
        lo = m.jnt_range[:, 0][[m.joint(n).id for n in C.JOINT_NAMES]]
        hi = m.jnt_range[:, 1][[m.joint(n).id for n in C.JOINT_NAMES]]
        ranges = ((lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6))
        env._limit_park_ranges = ranges
    mid, half = ranges
    q = env._joint_qpos()
    frac = np.abs(q - mid) / half  # 0 center → 1 at the stop
    over = np.maximum(0.0, frac - 0.88)
    # Coefficient 25 (was 3): at 3 a fully-parked joint cost only ~0.04/step —
    # invisible next to pose income, and a headstand policy happily kept five
    # joints cranked (hip yaw/roll splay + head_roll) for cheap stability.
    return -25.0 * float((over ** 2).sum())


def _torque_rate_pen(env) -> float:
    """Penalty for fast FORCE changes (<= 0) — damps shudder that position
    action_rate misses. Uses the actuator forces MuJoCo computed this step."""
    tau = env._act_force  # float64 view, exactly what the old asarray returned
    prev = getattr(env, "_prev_tau", None)
    env._prev_tau = tau.copy()
    if prev is None:
        return 0.0
    return -2e-3 * float(((tau - prev) ** 2).sum())


def _soft_landing_pen(env) -> float:
    """Penalty for vertical-velocity spikes of the trunk (<= 0) — the |a_z|
    anti-slam pressure AGENTS.md recommends instead of rotation-speed caps."""
    # True vertical speed (see _base_vel on why the local-frame 6-vector's
    # components are not the axes they look like).
    vz = _base_vel(env)[2]
    prev = getattr(env, "_prev_vz", None)
    env._prev_vz = vz
    if prev is None:
        return 0.0
    az = (vz - prev) / C.CTRL_DT
    return -6e-4 * float(az * az)


def _stall_pen(env) -> float:
    """Sustained-stall penalty (<= 0): torque² counted ONLY on joints that
    aren't moving — torque doing work is legitimate, torque against a wall is
    pure servo heat. Added when a headstand policy braced sideways with
    head_roll at max torque, 97.5% of hold time (XL330 overheat/strip risk);
    plain torque² can't target this without also taxing the load path."""
    tau = env._act_force
    qv = np.abs(env._joint_vel())
    stalled = (np.abs(tau) > 0.6) & (qv < 0.5)
    return -1.2 * float((tau[stalled] ** 2).sum())


CATALOG: dict[str, RewardTerm] = {
    t.key: t for t in (
        RewardTerm("head_up", "Points for holding the head up in its natural pose",
                   1.0, _head_up_blend),
        RewardTerm("flat_feet", "Points for keeping the feet flat on the floor",
                   0.8, _flat_feet),
        RewardTerm("calm_body", "Penalty for wobbling and thrashing the body",
                   1.0, _still_body_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for cranking joints to their end stops",
                   1.0, _limit_parking_pen, is_penalty=True),
        RewardTerm("smooth_torque", "Penalty for shuddering motor forces",
                   1.0, _torque_rate_pen, is_penalty=True),
        RewardTerm("soft_landings", "Penalty for slamming down hard",
                   1.0, _soft_landing_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot",
                   1.0, _stay_home_pen, is_penalty=True),
        # NOTE: don't add face_home to spin — its whole job is rotating.
        RewardTerm("face_home", "Penalty for twisting away from the starting direction",
                   1.0, _face_home_pen, is_penalty=True),
        RewardTerm("no_stall", "Penalty for straining a motor that isn't moving (pure heat)",
                   1.0, _stall_pen, is_penalty=True),
    )
}


def catalog_cards(exclude: tuple[str, ...] = ()) -> list[dict]:
    """UI list for "＋ add a term" — catalog entries not already in a recipe."""
    return [
        {"key": t.key, "friendly": t.friendly, "weight": t.weight,
         "isPenalty": t.is_penalty}
        for t in CATALOG.values() if t.key not in exclude
    ]


def _register(b: Behavior) -> None:
    BEHAVIORS[b.id] = b


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
        *_BASE_REGULARIZERS,
    ),
    default_steps=1_500_000,
    success_metric="average turning speed while upright",
))


# ------------------------------------------------------------- headstand
# Inverted-pose trick on the full-collision scene. Recipe lessons applied
# preemptively: every positive term is two-layer or wide (gradient at the
# CURRENT behavior, i.e. standing upright), the "doing it" pay is per-step and
# state-gated (no jackpots), anchors mild during discovery, motion-blockers
# LOW (this needs a dynamic pitch-over the calm taxes would smother).

def _head_bodies(env):
    ids = getattr(env, "_head_contact_bodies", None)
    if ids is None:
        ids = {mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")}
        env._head_contact_bodies = ids
    return ids


def _head_on_floor(env) -> bool:
    # Consulted by up to four backflip/headstand terms per step; memoized in
    # the step-scoped cache (walk_env) so the contact scan runs once. The old
    # loop also re-materialized data.contact.geom1/geom2 arrays through the
    # bindings on EVERY contact row — hoisted, like _foot_contacts.
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("head_floor")
        if v is not None:
            return v
    heads = _head_bodies(env)
    v = False
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = env.floor_geom
        bodyid = env.model.geom_bodyid
        for i in range(n):
            a = g1[i]
            b = g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            if int(bodyid[other]) in heads:
                v = True
                break
    if cache is not None:
        cache["head_floor"] = v
    return v


def _feet_body_ids(env):
    ids = getattr(env, "_feet_body_ids", None)
    if ids is None:
        ids = {mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("ankle_left", "ankle_right")}
        env._feet_body_ids = ids
    return ids


def _body_floor_contacts(env) -> int:
    """Floor contacts from anything that ISN'T the head or the feet — i.e.,
    the trunk/hips/legs dragging on the ground. A 'headstand' that reads
    perfect on orientation can still be a chest-slump (trunk z 0.033 with 584
    trunk-floor contacts, caught by the user's eye): orientation and
    ELEVATION are independent, and both must be priced."""
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("body_floor")
        if v is not None:
            return v
    heads, feet = _head_bodies(env), _feet_body_ids(env)
    v = 0
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = env.floor_geom
        bodyid = env.model.geom_bodyid
        for i in range(n):
            a = g1[i]
            b = g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            bid = int(bodyid[other])
            if bid not in heads and bid not in feet:
                v += 1
    if cache is not None:
        cache["body_floor"] = v
    return v


def _inverted(env) -> float:
    """Two-layer upside-down-ness: gravity_z in the trunk frame is -1 upright,
    +1 in a headstand. Wide layer pays ~0.13 even fully upright — the slope
    that makes pitching over discoverable at all."""
    gz = float(env._projected_gravity()[2])
    d2 = (gz - 1.0) ** 2  # 0 when inverted, 4 when upright
    return 0.6 * float(np.exp(-d2 / 1.4 ** 2)) + 0.4 * float(np.exp(-d2 / 0.35 ** 2))


def _jaw_bid(env) -> int:
    """jaw_soft's body id, resolved by name ONCE per env (a per-step
    mj_name2id lookup measured ~2 us in three different terms)."""
    bid = getattr(env, "_jaw_body_id", None)
    if bid is None:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
        env._jaw_body_id = bid
    return bid


def _head_low(env) -> float:
    """Head shell near the floor (target measured off the model: resting jaw
    center ~6 cm; standing is 23 cm — wide std keeps the slope alive)."""
    z = float(env.data.xpos[_jaw_bid(env)][2])
    return float(np.exp(-((z - 0.06) ** 2) / 0.12 ** 2))


def _feet_up(env) -> float:
    """Feet high overhead. Target 30 cm (head base ~6 + trunk + EXTENDED legs)
    — the original 18 cm target was reachable by a crumpled tuck, and the
    policy delivered exactly that crumple. Wide std keeps the slope alive from
    tuck height."""
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    z = (zl + zr) / 2
    return float(np.exp(-((z - 0.30) ** 2) / 0.12 ** 2))


def _legs_straight_up(env) -> float:
    """Straight, extended legs — but ONLY while inverted (gated on gravity):
    the standing start keeps its bent STAND pose unpunished. Raw joint angles
    toward zero = a straight vertical line, like the reference headstand."""
    gz = float(env._projected_gravity()[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    q = env._joint_qpos()[C.LEG_JOINT_IDS]
    d2 = float((q * q).sum())
    return gate * (0.5 * float(np.exp(-d2 / 2.0 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.7 ** 2)))


def _gentle_plant_pen(env) -> float:
    """One-time charge at the instant the head TOUCHES DOWN, scaled by impact
    speed² (<= 0). Serves two goals at once: the plant itself must be gentle
    (camera + ToF live in that shell), and every RETRY costs its own plant —
    so a single clean first-try attempt beats flip-drop-reflip cycling."""
    head_now = _head_on_floor(env)
    prev = getattr(env, "_gp_prev_head", False)
    env._gp_prev_head = head_now
    if not head_now or prev:
        return 0.0
    _, _, vz = _base_vel(env)
    # Flat fee + impact scale — MODERATE on purpose. History: impact-only →
    # free dabbing; flat-heavy → fast plants; then 0.6+20v² (at weight 2)
    # crossed the attempt-tax line and the policy simply STOPPED doing the
    # trick from upright (gorgeous 0.15 m/s taps, zero headstands). Entry
    # must stay clearly profitable; this pricing nudges toward few, soft
    # touches without ever making "don't try" the argmax.
    return -(0.3 + 10.0 * vz * vz)


def _calm_inverted_pen(env) -> float:
    """Stillness ONLY while inverted (<= 0): wobble/thrash charges once the
    duck is up, but the dynamic flip itself (gz < 0.5) stays untaxed. Without
    this the recipe paid a flip-teeter-drop-reflip cycle exactly as well as a
    real balance — longest unbroken hold was 0.9 s with 14 drops/episode."""
    gz = float(env._projected_gravity()[2])
    gate = max(0.0, min(1.0, (gz - 0.5) / 0.4))
    if gate == 0.0:
        return 0.0
    w = env._gyro
    return gate * -0.06 * float(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)


def _body_lifted(env) -> float:
    """Trunk ELEVATED to the clean-stack height while inverted (gated) —
    two-layer toward the measured spawn-pose trunk z of ~0.165. The missing
    complement to orientation: pays for head-body-feet actually stacking."""
    gz = float(env._projected_gravity()[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    z = float(env._trunk_xpos[2])
    d2 = (z - 0.165) ** 2
    return gate * (0.5 * float(np.exp(-d2 / 0.10 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.04 ** 2)))


def _body_drag_pen(env) -> float:
    """Penalty (<= 0) whenever the trunk/hips/legs touch the floor — the
    direct price on chest-slumping (head and feet contacts stay free)."""
    return -1.0 if _body_floor_contacts(env) > 0 else 0.0


def _nose_down(env) -> float:
    """Direction shaping: the trick is a FRONT flip — face-plant first, then
    legs over the head. Pays for leaning nose-DOWN (gravity acquiring +x in
    the trunk frame). gz is symmetric and can't tell a front headstand from a
    backbend, so without this the policy went over backwards (user caught it
    on video comparison)."""
    return max(0.0, min(1.0, float(env._projected_gravity()[0])))


def _wrong_way_pen(env) -> float:
    """Penalty (<= 0) for the backbend route: gravity tipping toward -x
    (head craning back under the body). Zero while upright or nose-down."""
    gx = float(env._projected_gravity()[0])
    return -2.0 * max(0.0, -gx - 0.15)


def _neck_tuck(env) -> float:
    """Chin-to-chest while inverted (gated on gravity). The reference
    technique (real microduck footage): face-plant, TUCK the neck — which
    rolls the contact point from the beak to the crown and pulls the pivot
    under the spine — then kick over. Untucked, the duck cranes its head back,
    rests on its face, and vertical alignment is geometrically impossible.
    Tuck target ≈ -0.6 rad on neck_pitch+head_pitch (HOME is +0.35 each)."""
    gz = float(env._projected_gravity()[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[5] + 0.6) ** 2 + (q[6] + 0.6) ** 2)
    return gate * (0.5 * float(np.exp(-d2 / 0.8 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.3 ** 2)))


def _headstand_hold(env) -> float:
    """The per-step trick pay: head planted + both feet airborne → pays STEEPLY
    with full inversion. The first version paid `gz` from a 0.3 gate, and the
    policy parked in a comfy 60° face-bow collecting most of every term — the
    playbook's compromise basin, verbatim. Now a bow (gz~0.5) earns ~2% of what
    crown-vertical (gz~1) earns, so completing the flip is the only real money."""
    if not _head_on_floor(env):
        return 0.0
    if _body_floor_contacts(env) > 0:
        return 0.0  # ONLY the head may touch — a chest-slump earns nothing
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest doesn't count — front-flip entries only
    gz = float(g[2])
    x = max(0.0, (gz - 0.5) / 0.5)
    return x ** 2


_register(Behavior(
    id="headstand",
    emoji="🙃",
    title="Do a headstand",
    description=(
        "Tip forward, plant the head on the floor, and balance upside down "
        "with the feet in the air."
    ),
    how_it_learns=(
        "The recipe pays a slope, not a destination: being MORE upside down, the "
        "head LOWER, the feet HIGHER each pay a little more, so tipping forward is "
        "profitable from the very first wiggle. The big per-step payout only flows "
        "in the real pose — head planted, feet airborne, body inverted. Falling "
        "over doesn't end the episode here (being 'fallen' is the whole point), "
        "so the duck can experiment freely. Its heavy head is an advantage for "
        "once: 38% of its mass becomes the base of the stand."
    ),
    keywords=("headstand", "head stand", "handstand", "upside down",
              "invert", "on its head"),
    terms=(
        RewardTerm("headstand_hold", "Big points for balancing inverted on the head, feet in the air",
                   4.0, _headstand_hold),
        RewardTerm("upside_down", "Points for being more upside down", 1.5, _inverted),
        RewardTerm("head_low", "Points for getting the head down to the floor", 1.2, _head_low),
        RewardTerm("feet_up", "Points for lifting the feet high overhead", 1.5, _feet_up),
        RewardTerm("legs_straight", "Points for straight, extended legs while inverted", 1.5,
                   _legs_straight_up),
        RewardTerm("neck_tuck", "Points for tucking the chin in while inverted (crown down)", 1.5,
                   _neck_tuck),
        RewardTerm("nose_down", "Points for tipping face-first (it's a FRONT flip)", 1.0,
                   _nose_down),
        RewardTerm("body_lifted", "Points for stacking the body up off the floor", 2.0,
                   _body_lifted),
        RewardTerm("wrong_way", "Penalty for going over backwards instead", 1.0,
                   _wrong_way_pen, is_penalty=True),
        RewardTerm("body_drag", "Penalty for resting the body on the ground", 1.0,
                   _body_drag_pen, is_penalty=True),
        RewardTerm("calm_up_top", "Penalty for wobbling while balanced (the flip stays free)", 1.0,
                   _calm_inverted_pen, is_penalty=True),
        RewardTerm("gentle_plant", "Penalty each time the head touches down, scaled by impact", 1.5,
                   _gentle_plant_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 0.4,
                   _stay_home_pen, is_penalty=True),
        RewardTerm("soft_landings", "Penalty for slamming down hard", 1.5,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 1.0,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 1.0,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="time spent inverted on the head with feet airborne",
    episode_s=20.0,  # long holds are the task — short clips never demand them
    scene="all",
    terminate_on_fall=False,
    inverted_spawn_prob=0.20,
    mid_flip_spawn_prob=0.25,
))


# ------------------------------------------------------------------ backflip
# A backward roll-over: lean back, roll across the back with the neck tucked,
# carry the legs over the top, land on the feet, STAND. XL330s can't launch
# 800 g airborne, so "backflip" here means the floor-contact somersault — the
# same trick family as the headstand's front flip, plus the genuinely new
# part: the maneuver is a TRAJECTORY (0 -> 360° of rotation), not a pose, so
# the reward needs memory (Behavior.state_fn).

_BF_FULL = 2.0 * np.pi
_BF_DONE = 5.2  # rad (~298°) — "flip complete": the roll has crossed the
                # crown and the feet are coming under; the landing terms
                # grade the rest of the rise.


def _bf_update(env) -> None:
    """Integrate cumulative BACKWARD pitch rotation (verified sign: gyro_y is
    negative while the trunk pitches backward). Clipped to [0, one full flip]:
    forward wobble can't bank negative credit, and a second flip earns nothing
    — so after landing, holding the stand strictly beats re-flipping (the
    headstand's cycling lesson, priced in from day one)."""
    w = env._gyro
    # C.CTRL_DT is the same product of the same doubles: __init__ pins
    # model.opt.timestep = C.PHYSICS_DT and nothing retunes it afterwards
    # (reading it back through the bindings cost ~1 us per step).
    env._bf_rot = min(max(env._bf_rot - C.CTRL_DT * float(w[1]), 0.0), _BF_FULL)


def _bf_progress(env) -> float:
    """Rotation progress, CUBIC: shallow early so lying mid-roll earns pocket
    change (no well-paid halfway point — the headstand's comfy-bow lesson),
    but with enough interior slope that the 70°→180° bridge is worth crossing.
    The first cut was quartic and the bridge paid ~nothing."""
    return float((env._bf_rot / _BF_FULL) ** 3)


def _bf_no_jaw_parking_pen(env) -> float:
    """Penalty for resting the jaw on the floor before the flip (<= 0).

    gentle_head charges head-contact IMPACT, not occupancy — so when the
    attempt-taxes briefly made flipping unprofitable, the policy discovered a
    rent-free rest: lean forward, place the jaw softly, park for the whole
    episode (measured: 10 s at pitch -64, rot +3). Occupancy is now priced,
    but ONLY pre-flip (rot < 0.3) — mid-roll head contact is the neck-kip,
    the mechanism the whole trick runs on. Bounded at one unit.
    """
    rot = getattr(env, "_bf_rot", 0.0)
    if rot >= 0.3:
        return 0.0
    z = float(env.data.xpos[_jaw_bid(env)][2])
    if z > 0.12:                 # jaw well clear of the floor: free
        return 0.0
    return -min((0.12 - z) / 0.06, 1.0)


def _bf_brake_pen(env) -> float:
    """Penalty for STILL rolling backward once the flip is complete (<= 0).

    _bf_rot caps at 2*pi, so over-rolling banks no more progress — but nothing
    charged it either, and unpriced quantities are free resources: the policy
    happily carried extra momentum through the landing, which is exactly the
    user-observed cluster — rolling a second time, or sitting down hard on
    its butt because the rotation never got braked before touchdown.
    Observable: attitude (gravity says upright-ish again) plus the duck's own
    gyro wy. Gated to the completed-rotation regime; bounded at one unit
    (saturates at ~4.5 rad/s of residual back-spin).
    """
    if env._bf_rot < 0.95 * _BF_FULL:
        return 0.0
    wy = float(env._gyro[1])
    if wy >= 0.0:          # not back-rolling: nothing to brake
        return 0.0
    return -min(0.05 * wy * wy, 1.0)


def _bf_lean_back(env) -> float:
    """Discovery slope for the first wiggle: pays for leaning back — but ONLY
    while still ROTATING backward. The first cut paid for the lean itself,
    and the very first policy parked at a 70° recline collecting it forever
    (rent-free basin right below the gate). Motion-gating makes a parked lean
    worth zero at any angle; only carrying the roll onward earns."""
    if env._bf_rot >= 1.4:
        return 0.0
    wy = float(env._gyro[1])
    if wy > -0.3:
        return 0.0  # not rotating backward right now -> no pay
    return max(0.0, min(1.0, -float(env._projected_gravity()[0])))


def _bf_landed_hold(env) -> float:
    """THE money: flip complete, feet planted, nothing else touching — and
    GRADED by uprightness rather than gated on it: a landed CROUCH earns 25%,
    rising to full pay standing tall. The all-or-nothing version left the
    roll's actual arrival state (a deep forward crouch at ~260°) worth zero,
    so the stand-up — itself a real sub-skill — had no slope to climb; v5
    coasted to 260° and stopped, 0/20 even from its own spawns."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]) or _body_floor_contacts(env) > 0:
        return 0.0
    # Two-layer uprightness (the zero-gradient law, fourth appearance): the
    # tight-only layer pays ~0 anywhere shy of vertical, so rising from the
    # arrival crouch (~60° forward) to 30° earned nothing more than the flat
    # base — stages completed the roll 10/12 and stood up 0/12. The wide
    # layer slopes the whole way up from the crouch.
    g = env._projected_gravity()
    d2 = float(g[0] ** 2 + g[1] ** 2)
    upright = (0.25 + 0.45 * float(np.exp(-d2 / 0.5))
               + 0.30 * float(np.exp(-d2 / 0.05)))
    # ...and HEIGHT, because orientation is not elevation (the headstand
    # learned this the hard way with gz=1.0 while the trunk lay on the floor;
    # here a folded crouch — trunk 5.5 cm vs 12 cm standing, head 9 cm vs
    # 24 cm, neck tucked — collected the full landed payout and looked like
    # "20/20 stands" to a metric that only asked which way was up. The user
    # watching said it never got up, twice, and was right both times).
    # Graded, not gated: a landed crouch still earns something (catching the
    # landing must stay worth doing) but far less than standing. At 0.3+0.7
    # the crouch was worth ~half a stand, and two-thirds of a stage's episodes
    # ending in one TRAINED AWAY an inherited stand — 7.95 s of hold became
    # 0.14 s in a single stage. A 6:1 ratio keeps the landing worth catching
    # without letting the crouch out-vote the stand.
    dz2 = (float(env._trunk_xpos[2]) - env.stand_z) ** 2
    tall = (0.5 * float(np.exp(-dz2 / 0.08 ** 2))
            + 0.5 * float(np.exp(-dz2 / 0.03 ** 2)))
    return upright * (0.15 + 0.85 * tall)


def _bf_neck_targets(rot: float) -> tuple[float, float]:
    """Phase-dependent neck technique, refined by a probe matrix (all
    variants measured to their max rotation): FULL tuck (-1.5, pulled tight
    against the body — the user's 'roll on the curved part') rolls to ~183°
    where the headstand-style -0.6 tuck shelfed at 161°; the ARCH plants the
    crown and carries 160°→283°; between them sits the KIP — a fast
    tuck→arch snap pressing the head into the floor. This schedule is the
    measured best-known sequence; the terms below pay its pieces so the
    panel's live bars show which parts the policy has adopted."""
    if rot < 2.3:
        return (-1.5, -1.5)
    if rot < 2.8:
        t = (rot - 2.3) / 0.5
        return (-1.5 + 2.5 * t, -1.5 + 2.8 * t)
    if rot < 4.6:
        return (1.0, 1.3)
    t = min(1.0, (rot - 4.6) / 0.5)  # ease home for the landing
    return (1.0 - 0.65 * t, 1.3 - 0.95 * t)


def _bf_push_off(env) -> float:
    """The committed entry (rot < ~50°): pays the backward rotation RATE the
    legs manage to generate — a real push-off arrives on the back at 5-6
    rad/s, a polite sit arrives at ~1 and stalls (probe-measured). Rate pay,
    not a pose: how the legs make the speed is the policy's business. The
    user called this gap watching the trainee: 'I'm not seeing it really
    give an initial push off with the legs.'"""
    if env._bf_rot >= 0.9:
        return 0.0
    wy = float(env._gyro[1])
    return min(1.0, max(0.0, -wy) / 6.0)


def _bf_rolling(env, thresh: float = -0.2) -> bool:
    """Still rotating backward — the gate on every mid-roll SHAPE term.
    Ungated shape pay re-creates the v1 basin: lying on the back holding a
    pretty tuck out-earned attempting the flip (the user spotted the gentle
    sit-and-park entries this bred)."""
    return float(env._gyro[1]) < thresh


def _bf_tuck_ball(env) -> float:
    """Ball up from the moment the fall starts (rot ~20-140°): legs folded,
    neck pulled FULLY against the body so the back rolls on a curved profile
    instead of slamming the head shelf down. Starts at 20° — the fall itself
    must be tucked (the neck used to be unpriced until 50°, and the duck
    fell with it extended) — but pays only WHILE rolling backward."""
    rot = env._bf_rot
    if not (0.35 < rot < 2.4) or not _bf_rolling(env):
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[2] + 1.0) ** 2 + (q[11] - 1.0) ** 2
               + (q[3] + 1.0) ** 2 + (q[12] - 1.0) ** 2
               + (q[5] + 1.5) ** 2 + (q[6] + 1.5) ** 2)
    return 0.5 * float(np.exp(-d2 / 2.5 ** 2)) + 0.5 * float(np.exp(-d2 / 1.0 ** 2))


def _bf_neck_kip(env) -> float:
    """The press-off at the crux (rot ~132-172°): pays neck EXTENSION SPEED
    while the head is grounded — snapping tuck→arch presses the head into
    the floor and levers the trunk over the back-shell edge (wrestler's kip).
    A velocity term on purpose: the probe showed the static poses on either
    side both stall; the SNAP itself is the move. Bounded ≤1/step."""
    rot = env._bf_rot
    if not (2.3 < rot < 3.0):
        return 0.0
    if not _head_on_floor(env):
        return 0.0  # nothing to press against
    qv = env._joint_vel()
    v = max(0.0, float(qv[5])) + max(0.0, float(qv[6]))
    return min(1.0, v / 10.0)


def _bf_arch_over(env) -> float:
    """The crown-bridge (rot ~160-265°): neck arched so the body carries
    over the planted crown — measured to run 160°→283° on momentum alone."""
    rot = env._bf_rot
    if not (2.8 < rot < 4.6) or not _bf_rolling(env, thresh=-0.1):
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[5] - 1.0) ** 2 + (q[6] - 1.3) ** 2)
    return 0.5 * float(np.exp(-d2 / 2.0 ** 2)) + 0.5 * float(np.exp(-d2 / 0.7 ** 2))


def _bf_legs_over(env) -> float:
    """Feet overhead through the middle of the roll — the leg WHIP. Kicking
    the legs up and over while the back is on the floor is what generates the
    angular momentum that carries the body across the crown (the user's read
    of the stall, and the reaction-torque physics agree). Gated so neither
    the entry nor the landing pays for high feet."""
    if not (1.4 < env._bf_rot < 3.5) or not _bf_rolling(env):
        return 0.0
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    z = (zl + zr) / 2
    return float(np.exp(-((z - 0.30) ** 2) / 0.12 ** 2))


def _bf_straight_pen(env) -> float:
    """Keep the flip in the sagittal plane (<= 0): roll/yaw rates are the
    free resources here — an unpriced corkscrew reads as rotation progress
    while twisting the landing off-axis."""
    w = env._gyro
    # Bounded (run-recipe lesson: unbounded rate penalties become the
    # dominant term under a flailing policy). Saturates at ~3.2 rad/s of
    # combined off-axis rate.
    return -min(0.1 * float(w[0] ** 2 + w[2] ** 2), 1.0)


_HEAD_YAW_I, _HEAD_ROLL_I = 7, 8   # C.JOINT_NAMES indices


def _bf_still_head_pen(env) -> float:
    """Penalty for turning/rolling the HEAD mid-flip (<= 0).

    User-observed failure mode: flips where the head yaws mid-roll dump
    angular momentum off the sagittal plane and the duck flops sideways; the
    clean flips keep the head still. straight_flip prices BODY twist rate but
    head-joint motion is nearly free — the heavy head is ~30% of total mass,
    so a head_yaw swing is a big off-axis impulse. Gated to the flip itself
    (rotation banked but not complete) so landing recovery and the stand
    aren't taxed. Head joint pos/vel are in the obs — observable, so priced.
    Bounded at one unit (~0.7 rad of combined deflection).
    """
    rot = getattr(env, "_bf_rot", 0.0)
    if not (0.05 < rot < 2.0 * np.pi - 0.3):
        return 0.0
    q = env._joint_qpos()
    dq = env._joint_vel()
    dev = float(q[_HEAD_YAW_I] ** 2 + q[_HEAD_ROLL_I] ** 2)
    vel = float(dq[_HEAD_YAW_I] ** 2 + dq[_HEAD_ROLL_I] ** 2)
    return -min(2.0 * dev + 0.02 * vel, 1.0)


def _bf_calm_landed_pen(env) -> float:
    """Stillness once landed (<= 0), the flip itself untaxed — the mirror of
    the headstand's calm_up_top, against flip-stagger-flop cycling."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    return _upright(env) * -0.06 * float((env._gyro ** 2).sum())


def _bf_feet_under(env) -> float:
    """Get-up mechanics, part 1 (landing window): feet PLANTED and pulled in
    UNDER the body — you can't rise from feet sprawled out in front. Pays per
    planted foot, scaled by how close the feet sit to under the trunk."""
    if env._bf_rot < 4.6:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] or c["right"]):
        return 0.0
    tx, ty = env._trunk_xpos[:2]
    total = 0.0
    for side in ("left", "right"):
        if not c[side]:
            continue
        fx, fy = env.data.geom_xpos[env.foot_geoms[side]][:2]
        d2 = (fx - tx) ** 2 + (fy - ty) ** 2
        total += 0.5 * float(np.exp(-d2 / 0.08 ** 2))
    return total


def _bf_neck_pushup(env) -> float:
    """Get-up mechanics, part 2 (landing window): with 38% of the mass in the
    head, pressing it into the floor to lever the torso up is this body's
    substitute for arms — the kip's physics pointed at the rise. Pays neck
    extension SPEED while the head is grounded, same shape as neck_kip."""
    if env._bf_rot < 4.6 or not _head_on_floor(env):
        return 0.0
    qv = env._joint_vel()
    v = max(0.0, float(qv[5])) + max(0.0, float(qv[6]))
    return min(1.0, v / 10.0)


def _bf_head_rise(env) -> float:
    """Get-up mechanics, part 3 — the relay's last runner: with feet planted
    in the landing window, pay the HEAD'S HEIGHT toward standing (~22 cm).
    Without this the get-up stalled exactly between the press-off and the
    standing pay: feet_under earned, neck_pushup earned, and then the head
    simply stayed down — the head-down tripod collected enough base income
    to be a comfy rest stop (the user watched it park there). Two-layer so
    the slope reaches all the way down to a head resting on the floor."""
    if env._bf_rot < 4.6:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    z = float(env.data.xpos[_jaw_bid(env)][2])
    d2 = (z - 0.22) ** 2
    return 0.5 * float(np.exp(-d2 / 0.12 ** 2)) + 0.5 * float(np.exp(-d2 / 0.05 ** 2))


def _bf_stand_pose(env) -> float:
    """Finish IN the canonical stand (the user's suggestion, with a sim2real
    bonus they intuited: DEFAULT_POSE is the shared home pose every shipped
    policy starts from, so ending the trick there means the robot can
    hot-swap straight back to its walk policy). Gated to the landed phase;
    two-layer over leg+neck joints so the pull reaches a crouch."""
    if env._bf_rot < _BF_DONE:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    q = env._joint_qpos()
    d2 = float(((q[C.LEG_JOINT_IDS] - C.DEFAULT_POSE[C.LEG_JOINT_IDS]) ** 2).sum())
    d2 += float((q[5] - C.DEFAULT_POSE[5]) ** 2 + (q[6] - C.DEFAULT_POSE[6]) ** 2)
    # Widths measured against reality, not guessed: the landed stand sits
    # ~3.9 rad from DEFAULT_POSE (a splayed survival crouch), where the first
    # cut of this term paid 0.0006 — invisible, and a 1M-step polish run
    # moved the pose the WRONG way (3.92 -> 4.37). Fifth zero-gradient bug of
    # this project; the wide layer now pays ~0.19 out there and slopes in.
    return 0.5 * float(np.exp(-d2 / 4.0 ** 2)) + 0.5 * float(np.exp(-d2 / 1.5 ** 2))


def _bf_spotter(env) -> bool:
    """Gymnastics-coach assist for SHOWCASE previews only (never training).

    The trained policy owns everything past ~170° — carry, land, stand, hold
    (19-20/20) — but the 90-170° arc is blocked by this actuator model, so
    from a standing start it rationally never begins, and a viewer watching
    the finished trick sees... a duck standing there. A gentle backward pitch
    torque through the dead zone, released the moment the policy's own
    territory begins, shows the real trick honestly: probe-measured 4/5 full
    360° flips with 8-13 s stands, and the duck's label says 'spotted'."""
    assisting = (15 < env.step_count < 250) and env._bf_rot < 4.0
    env.data.qfrc_applied[4] = -0.6 if assisting else 0.0
    # Plane stabilization, same assist window: the policy CANNOT learn to
    # land pointing straight — final yaw is unobservable (no compass in the
    # 61 obs), and the off-axis RATE terms can't price a clean-looking roll
    # about a slightly tilted axis. The coach's other hand: a small
    # yaw/roll-damping torque while spotting. Measured (16 spawns, b18a5c):
    # landing yaw mean 90 deg -> 33, stands 16/16. Stronger gains or a wider
    # window measured WORSE (53/46 deg) — fighting the policy destabilizes.
    if assisting:
        q = env.data.qpos[3:7]
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        wz = float(env.data.qvel[5])
        wx = float(env.data.qvel[3])
        env.data.qfrc_applied[5] = float(np.clip(-0.4 * yaw - 0.06 * wz, -0.35, 0.35))
        env.data.qfrc_applied[3] = float(np.clip(-0.06 * wx, -0.2, 0.2))
    else:
        env.data.qfrc_applied[5] = 0.0
        env.data.qfrc_applied[3] = 0.0
    return assisting


def _bf_land_tall(env) -> float:
    """Extend the legs on the way DOWN (the last quarter-turn before the roll
    completes) so touchdown happens on near-straight legs.

    Measured cliff: the stand policy rescues a landing whose trunk is at
    12.8 cm and fails at 11.7 — and nothing, including the shipped walking
    policy, rises from the 5.5 cm crouch this trick used to land in. These
    servos can HOLD a stand but cannot lift the body from below it, so
    'land, then get up' is not a thing this robot can do. Landing tall is
    the only version that works, and that is decided in the air."""
    if not (4.4 < env._bf_rot < _BF_DONE):
        return 0.0
    q = env._joint_qpos()
    d2 = float(((q[C.LEG_JOINT_IDS] - C.DEFAULT_POSE[C.LEG_JOINT_IDS]) ** 2).sum())
    return (0.5 * float(np.exp(-d2 / 4.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 1.5 ** 2)))


def _bf_spawn_landed(env):
    """Reverse-curriculum end-state spawn: the ordinary upright pose with the
    flip already CREDITED — practices the landed hold, and the value of being
    'done and standing' propagates back to make finishing worth it."""
    env._bf_rot = _BF_FULL
    return env._get_obs()


def _bf_spawn_mid_roll(env):
    """Mid-maneuver spawn: anywhere from just past the commit point (~75°,
    where upright episodes used to park) to most of the way over (~245° —
    v2 proved the whole 150-300° carry-over needs on-policy data too, not
    just the on-back stretch), WITH backward angular momentum. Static
    spawns taught a dead-stop kip that never worked; a real roll arrives
    at every attitude still rotating, and conserving that momentum is the
    technique. Legs tuck through the middle and extend past inverted;
    rotation credit matches the posed attitude."""
    r = env._rng
    d, m = env.data, env.model
    # Spawn window, overridable for STAGED reverse curriculum (train the
    # landing zone first with a concentrated window, then march the window
    # back toward the entry): the fixed full-range mix gave each hard zone
    # only ~10% of episodes and none of the sub-skills ever converged.
    lo = float(_spawn_knob(env, "MICRODUCK_BF_SPAWN_LO", "1.3"))
    hi = float(_spawn_knob(env, "MICRODUCK_BF_SPAWN_HI", "4.3"))
    rot = r.uniform(lo, hi)
    pitch = -rot
    d.qpos[:] = 0.0
    d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    q = d.qpos
    # Tucked legs through the middle of the roll — and STILL flexed on the
    # way down: the first unfold schedule straightened the legs past ~250°,
    # which posed the landing-zone spawns as a stiff plank tipped nose-down
    # (un-landable — every stage face-planted its landing practice, 0/36
    # stand-ups). The real arrival is a deep crouch, feet under the body.
    fold = 1.0 if rot < 3.8 else 0.85
    hip = fold * r.uniform(0.7, 1.2)
    knee = fold * r.uniform(0.7, 1.2)
    for adr in env.joint_qpos_adr:
        q[adr] = r.uniform(-0.1, 0.1)
    q[env.joint_qpos_adr[2]] = -hip    # left hip_pitch folded to chest
    q[env.joint_qpos_adr[11]] = hip    # right hip_pitch (mirrored sign)
    q[env.joint_qpos_adr[3]] = -knee
    q[env.joint_qpos_adr[12]] = knee
    # Neck follows the phase schedule (tuck in, arch over) so spawned states
    # demonstrate the technique the reward asks for.
    n1, n2 = _bf_neck_targets(rot)
    q[env.joint_qpos_adr[5]] = n1 + r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[6]] = n2 + r.uniform(-0.15, 0.15)
    # Attitude-aware drop height: a fixed spawn z WEDGED the head into the
    # floor at high rotations — the solver ejected it, absorbing all the
    # spawn momentum, and the whole carry-over stretch trained on corrupted
    # states (the invisible wall of v3). Pose high, measure the lowest
    # geom bound, settle to true clearance.
    d.qpos[2] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = [float(d.geom_xpos[g][2]) - float(m.geom_rbound[g])
            for g in range(m.ngeom) if g != env.floor_geom]
    d.qpos[2] = 0.6 - min(lows) + 0.005 + r.uniform(0.0, 0.01)
    d.qvel[4] = -r.uniform(0.5, 4.0)   # arriving mid-roll, still rolling
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    env.prev_joint_vel = env._joint_vel().copy()
    env._bf_rot = rot
    return env._get_obs()


_register(Behavior(
    id="backflip",
    emoji="🤸",
    title="Do a backflip",
    description=(
        "Lean back, roll backwards over the back with the neck tucked, "
        "carry the feet over the top, and land standing."
    ),
    how_it_learns=(
        "The duck earns a little for leaning back at all, more as its total "
        "backward rotation adds up, and the big payout only once it has rolled "
        "a full turn AND is back on its feet — so 'finish the roll and stand "
        "up' is the only real money. The technique terms pay the measured "
        "best-known sequence: ball up tight and roll on the curved back, "
        "whip the feet overhead, then SNAP the neck from tucked to arched — "
        "pressing the head into the floor to lever the body over the crown. "
        "Some episodes start mid-roll (still rolling) or already landed, so "
        "the hard moments get practiced from the first minute of training."
    ),
    keywords=("backflip", "back flip", "backwards flip", "flip backwards",
              "somersault", "backward roll", "roll backwards", "flip"),
    terms=(
        RewardTerm("landed_hold", "Big points for standing on both feet after a full roll",
                   5.0, _bf_landed_hold),
        RewardTerm("flip_progress", "Points as the backward rotation adds up", 1.5,
                   _bf_progress),
        RewardTerm("lean_back", "Points for tipping backwards to start the roll", 1.0,
                   _bf_lean_back),
        RewardTerm("push_off", "Points for a committed backward push-off with the legs", 2.0,
                   _bf_push_off),
        RewardTerm("tuck_ball", "Points for balling up tight — neck pulled in, rolling on the curve", 1.5,
                   _bf_tuck_ball),
        RewardTerm("neck_kip", "Points for the head press-off that levers the body over", 2.0,
                   _bf_neck_kip),
        RewardTerm("arch_over", "Points for arching over the crown once past the press", 1.5,
                   _bf_arch_over),
        RewardTerm("legs_over", "Points for whipping the feet up and over mid-roll", 1.5,
                   _bf_legs_over),
        RewardTerm("land_tall", "Points for straightening the legs before touchdown", 2.5,
                   _bf_land_tall),
        RewardTerm("feet_under", "Points for planting the feet back under the body to rise", 1.5,
                   _bf_feet_under),
        RewardTerm("neck_pushup", "Points for pressing the head down to push itself up", 1.5,
                   _bf_neck_pushup),
        RewardTerm("head_rise", "Points for lifting the head up off the floor to standing height", 2.0,
                   _bf_head_rise),
        RewardTerm("stand_pose", "Points for finishing in the normal standing pose (walk-ready)", 1.5,
                   _bf_stand_pose),
        # 1.0 (was 0.5): landing crooked is accumulated off-axis rotation, and
        # rate is the observable handle on it (final yaw itself is not in the
        # obs — observability law).
        # Attempt-tax lesson, learned AGAIN here: raising this to 1.0 (plus
        # two new penalties) priced every noisy flip attempt above its
        # expected payoff, and stage 5 converged to never flipping — it
        # parked jaw-down on the floor for entire episodes instead. Back to
        # 0.5; the new terms below run at half strength for the same reason.
        RewardTerm("straight_flip", "Penalty for twisting sideways out of the roll", 0.5,
                   _bf_straight_pen, is_penalty=True),
        RewardTerm("still_head", "Penalty for turning the head mid-flip (keeps the roll on-plane)",
                   0.5, _bf_still_head_pen, is_penalty=True),
        RewardTerm("stick_it", "Penalty for still rolling once the flip is done — brake, then stand",
                   0.75, _bf_brake_pen, is_penalty=True),
        RewardTerm("no_jaw_parking", "Penalty for resting the jaw on the floor instead of flipping",
                   1.0, _bf_no_jaw_parking_pen, is_penalty=True),
        RewardTerm("calm_landed", "Penalty for wobbling after the landing (the roll stays free)",
                   1.0, _bf_calm_landed_pen, is_penalty=True),
        RewardTerm("gentle_head", "Penalty each time the head touches down, scaled by impact",
                   0.75, _gentle_plant_pen, is_penalty=True),
        RewardTerm("stay_home", "Penalty for wandering away from the starting spot", 0.4,
                   _stay_home_pen, is_penalty=True),
        # Half the headstand's weight: rolling onto the back IS the trick —
        # at 1.5 this term was part of the tax wall that kept the first run
        # parked at a 70° recline, afraid to commit. Head impacts stay priced
        # separately (gentle_head).
        RewardTerm("soft_landings", "Penalty for slamming down hard", 0.75,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky, twitchy movements", 1.0,
                   _action_rate_pen, is_penalty=True),
        RewardTerm("gentle_joints", "Penalty for flailing the joints fast", 1.0,
                   _joint_vel_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 1.0,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="full backward rolls that end standing on both feet",
    episode_s=15.0,  # ~2 s of roll, the rest is the landed hold
    scene="all",
    terminate_on_fall=False,
    state_fn=_bf_update,
    spotter_fn=_bf_spotter,
    spawn_families=((0.20, _bf_spawn_landed), (0.35, _bf_spawn_mid_roll)),
    # Staged reverse curriculum over _bf_spawn_mid_roll's window: master the
    # landing zone first, then march the window back toward the entry — the
    # fixed full-range mix gave each hard zone only ~10% of episodes and none
    # of the sub-skills ever converged. Each stage fine-tunes the previous.
    # Windows chosen from measured zone batteries: landing and bridge
    # converge quickly once their spawn poses are physically right, but the
    # arrival-to-carry transition (~92-126° — on the back, committing across
    # the crown) got ~4% of episodes under a uniform full window and stayed
    # the chain's weak link — it gets its own concentrated stage.
    # Each focused stage is ~90% its own practice (probs are landed,mid-roll —
    # the declared 0.20/0.35 mix is only right for the final integration
    # stage, where the point IS mostly-from-standing attempts).
    # Each stage's `detail` narrates its actual spawn window/mix in plain
    # words — the viewer's stage inspector shows it, so keep these in step
    # with the env knobs when tuning.
    curriculum=(
        # 2M, not 1M: at 1M the get-up plateaued around half the landings
        # (stands bounced 0-6/12 across chains — noise around "not enough");
        # a 2M concentrated run went 20/20 with 9.7 s holds. The rise is the
        # hardest sub-skill and gets the budget to match.
        # Stage 0 exists because a per-term audit found the policy leaving
        # 7.3 points/step on the table: standing pays +9.28/step vs the
        # crouch's +2.01, yet it collapsed out of a STANDING spawn in ~1 s.
        # Not an incentive problem — it had never learned to balance (its
        # whole life was spent tumbling). So: learn to stand still first,
        # then learn the trick on top of a body that knows how to stand.
        CurriculumStage("learning to stand steady", 1_000_000,
                        {"MICRODUCK_SPAWN_FAMILY_PROBS": "1.0,0.0",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Every practice run starts already standing, with "
                            "the flip counted as done — the duck's only job is "
                            "to stay up: knees under it, head high, steady. "
                            "It cannot learn to stand up from a fall until it "
                            "knows how to stand at all.")),
        CurriculumStage("learning to land and stand up", 2_000_000,
                        {"MICRODUCK_BF_SPAWN_LO": "4.2",
                         "MICRODUCK_BF_SPAWN_HI": "5.6",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.35,0.60",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "The duck is dropped mid-roll or near the end of "
                            "the flip (~90% of practice runs) and learns to "
                            "catch the landing in a crouch and rise to "
                            "standing.")),
        CurriculumStage("carrying the roll over the top", 1_000_000,
                        {"MICRODUCK_BF_SPAWN_LO": "2.6",
                         "MICRODUCK_BF_SPAWN_HI": "5.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.30,0.65",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Most runs (~75%) drop the duck somewhere between "
                            "on-its-back and nearly landed, still rolling; it "
                            "learns to conserve that momentum across the "
                            "crown and into the landing it already knows.")),
        CurriculumStage("punching through the middle", 1_500_000,
                        {"MICRODUCK_BF_SPAWN_LO": "1.4",
                         "MICRODUCK_BF_SPAWN_HI": "3.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.25,0.70",
                         "MICRODUCK_EPISODE_S": "8"},
                        detail=(
                            "Most runs (~80%) start just past the commit "
                            "point, on the back with the roll underway; the "
                            "duck drills the tuck, the leg whip, and the neck "
                            "press-off that carry the roll through its "
                            "hardest stretch.")),
        CurriculumStage("the whole flip from standing", 1_500_000,
                        {"MICRODUCK_BF_SPAWN_LO": "1.3",
                         "MICRODUCK_BF_SPAWN_HI": "5.5",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.25,0.30"},
                        detail=(
                            "More than half the runs now start standing "
                            "upright, the rest sprinkled across the whole "
                            "roll; the duck stitches the push-off entry onto "
                            "the mid-roll and landing skills it already "
                            "has.")),
    ),
))


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


# ------------------------------------------------ imitation (authored motion)
# Instead of discovering a maneuver, TRACK one an animator keyframed. The
# reward stops being a landscape to explore and becomes "be where the clip
# says you should be right now" — the standard fix when a trick is too hard
# to stumble into, and the reason most published robot flips exist.

def _mi_pose_match(env) -> float:
    """Joints matching the reference pose for this instant. Two-layer: the
    wide layer keeps a gradient when the body is nowhere near the clip
    (which is where every run starts), the tight layer pays for precision."""
    ref, _ = env.clip.at(env.step_count)
    q = env._joint_qpos()
    d2 = float(((q - ref) ** 2).sum())
    return (0.5 * float(np.exp(-d2 / 6.0 ** 2))
            + 0.5 * float(np.exp(-d2 / 2.0 ** 2)))


def _mi_rotation_match(env) -> float:
    """Body rotation matching the clip's timeline.

    Two regimes, because one measure cannot serve both. A one-shot FLIP needs
    the accumulator (rotation past 180° must keep counting). A LOOPING gait
    needs the instantaneous attitude — and using the accumulator there was a
    silent disaster: it only counts BACKWARD rotation and clips at zero, so a
    duck face-planting FORWARD scored a constant ~80% of this term while
    lying on the floor. Nothing in the recipe noticed it had fallen."""
    _, ref_pitch = env.clip.at(env.step_count)
    if env.clip.loop:
        g = env._projected_gravity()
        pitch_now = float(np.arctan2(g[0], -g[2]))   # 0 upright, + = nose-down
        d = pitch_now - ref_pitch
    else:
        # Clip pitch is sim/editor convention (negative = lean back); the
        # accumulator counts backward rotation as positive. Negate to compare.
        d = float(env._bf_rot) - (-ref_pitch)
    return (0.5 * float(np.exp(-(d ** 2) / 2.0 ** 2))
            + 0.5 * float(np.exp(-(d ** 2) / 0.7 ** 2)))


def _mi_on_feet(env) -> float:
    """Still standing on your feet, for a LOOPING gait: upright AND at
    standing height AND actually touching the floor with a foot. A pose-match
    reward alone is blind to this — the joint angles of a run cycle are just
    as matchable lying face-down on the floor, which is exactly what the
    first run policy learned to do."""
    if not env.clip.loop:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] or c["right"]):
        return 0.0
    z = float(env._trunk_xpos[2])
    tall = float(np.exp(-((z - env.stand_z) ** 2) / 0.05 ** 2))
    return float(_upright(env)) * tall


def _mi_no_slip(env) -> float:
    """Penalty for a planted foot SLIDING (<= 0). Upstream's walking task
    prices this at -0.1 for a reason: without it the cheapest way to satisfy
    a gait's poses is to skate — feet moving through the cycle while stuck to
    the floor — which looks like walking and travels nowhere."""
    if not env.clip.loop:
        return 0.0
    total = 0.0
    for side in ("left", "right"):
        if not env.foot_contact_state[side]:
            continue
        v6 = _v6_buf(env)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 env.foot_geoms[side], v6, 0)
        total += float(v6[3] ** 2 + v6[4] ** 2)
    return -total


def _mi_no_spin(env) -> float:
    """Penalty for yawing (<= 0). Upstream commands zero turn rate and pays
    for tracking it; we have no command channel here, so price the spin
    directly — the duck was corkscrewing instead of running straight."""
    if not env.clip.loop:
        return 0.0
    return -float(env._gyro[2] ** 2)


def _mi_travel(env) -> float:
    """For a LOOPING clip (a gait): actually go somewhere. Matching a run
    cycle's poses is satisfied just as well by running on the spot, which is
    the classic way an imitation reward gets gamed — so forward travel is
    priced on its own, and only for loops."""
    if not env.clip.loop:
        return 0.0
    fwd, _, _ = _base_vel(env)
    return float(np.clip(fwd / 0.3, 0.0, 1.0)) * float(_upright(env))


def _mi_end_upright(env) -> float:
    """Once a ONE-SHOT clip is over, be standing — the landing is the part a
    reference clip cannot specify, because it depends on how the physics
    actually went. A loop has no end, so this stays out of its way."""
    if env.clip.loop or env.step_count < env.clip.steps:
        return 0.0
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    z = float(env._trunk_xpos[2])
    tall = float(np.exp(-((z - env.stand_z) ** 2) / 0.06 ** 2))
    return float(_upright(env)) * tall


_register(Behavior(
    id="imitate",
    emoji="🎬",
    title="Copy the animation",
    description=(
        "Physically perform a keyframed motion clip: match the authored pose "
        "and rotation at every instant, then land and stand."
    ),
    how_it_learns=(
        "This one is not asked to invent anything. An animation clip says "
        "where every joint should be at each moment, and the duck earns for "
        "being there — so the hard search for 'how do I flip' becomes the far "
        "easier problem of 'follow this'. What it still has to solve is the "
        "physics: momentum, contact and balance are not in the clip, and no "
        "keyframe can fake them."
    ),
    keywords=("copy the animation", "imitate", "follow the clip",
              "animation clip", "motion clip", "do the animation"),
    terms=(
        RewardTerm("pose_match", "Points for matching the animation's pose right now",
                   4.0, _mi_pose_match),
        RewardTerm("rotation_match", "Points for matching the animation's rotation timing",
                   4.0, _mi_rotation_match),
        RewardTerm("stick_it", "Big points for standing on both feet when a trick clip ends",
                   5.0, _mi_end_upright),
        RewardTerm("on_feet", "Points for staying on its feet at full height (looping clips)",
                   5.0, _mi_on_feet),
        RewardTerm("no_slip", "Penalty for planted feet skating along the floor", 0.5,
                   _mi_no_slip, is_penalty=True),
        RewardTerm("no_spin", "Penalty for turning instead of going straight", 0.3,
                   _mi_no_spin, is_penalty=True),
        RewardTerm("travel", "Points for actually moving forward (looping clips like a run)",
                   3.0, _mi_travel),
        RewardTerm("gentle_head", "Penalty each time the head hits the floor, scaled by impact",
                   1.0, _gentle_plant_pen, is_penalty=True),
        RewardTerm("soft_landings", "Penalty for slamming down hard", 0.75,
                   _soft_landing_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops", 1.0,
                   _limit_parking_pen, is_penalty=True),
        RewardTerm("save_energy", "Penalty for straining the motors", 0.5,
                   _torque_pen, is_penalty=True),
    ),
    default_steps=3_000_000,
    success_metric="how closely the authored motion is reproduced, and whether it lands",
    episode_s=4.0,        # the clip plus time to settle on the landing
    scene="all",
    terminate_on_fall=False,
    state_fn=_bf_update,  # the rotation accumulator the clip is compared against
    clip_name="backflip",
    # Asymmetric because it carries a CLIP, not because of any clip's own
    # left/right content. The clip phase rides in body_cmd[4:6] (obs 59/60,
    # see _get_obs) and the mirror map negates the cos slot, so M sends phase
    # p to pi - p: it TIME-REFLECTS the clip. The prior then asks for the
    # mirrored action at an unrelated instant of the motion, which no clip
    # satisfies — the shipped backflip clip is exactly mirror-symmetric
    # frame by frame and still fails it. That is why this is a flat False and
    # not derived from the clip; tests/test_symmetry.py carries the numbers.
    symmetric=False,
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


# ------------------------------------------------------------------- running
# Port of microduck_rl's Mjlab-Run task (microduck_run_env_cfg.py), not of
# the velocity WALK recipe. The walk stack prices running out of the
# optimum (dead running-pose std, upright tuned to erase lean, air_time
# outweighing speed). GPU run inverts that: speed is the largest term,
# the pose running-regime actually fires, lean is cheap, smoothness caps
# at -0.5. Local "run" used to copy walk weights and then invent extra
# terms; those policies are not a predictor of a GPU run.

# GPU stage boundaries are iterations x 24 per-env steps, saturating at
# 3000 x 24 = 72k. A local 20M-step/16-env run is 1.25M per-env steps, so the
# ported units saturated at 5.8% of the run — the smoothness tax landed
# during discovery (the exact attempt-tax AGENTS.md forbids) and the speed
# ceiling stopped being a curriculum. _RUN_STAGE_SCALE stretches every stage
# boundary so saturation lands at ~60% of the default run (audit finding #2).
_RUN_NUM_STEPS_PER_ENV = 24
_RUN_STAGE_SCALE = float(os.environ.get("MICRODUCK_STAGE_SCALE",
                                        750_000 / (3000 * 24)))
_RUN_FORWARD_FRAC = 0.55
_RUN_FORWARD_CLAMP = 0.3
_RUN_LIN_VEL_Y = (-0.3, 0.3)
_RUN_ANG_VEL_Z = (-1.0, 1.0)
_RUN_SPEED_STAGES = (
    (0, 0.4),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.45),
    (2000 * _RUN_NUM_STEPS_PER_ENV, 0.5),
    (2500 * _RUN_NUM_STEPS_PER_ENV, 0.55),
    (3000 * _RUN_NUM_STEPS_PER_ENV, 0.6),
    # Extended past the GPU cfg's 0.6 ceiling: this robot delivers ~66% of
    # the commanded speed (0.295 measured at cmd 0.45), so a 0.6 command
    # ceiling caps ACHIEVED speed near ~0.4 by construction — the 0.6 m/s
    # goal was structurally unreachable. 0.9 remains inside the measured
    # kinematic envelope (~1.4 m/s under BAM); whether the gait can cash it
    # is what the late curriculum exists to find out.
    (4000 * _RUN_NUM_STEPS_PER_ENV, 0.7),
    (4500 * _RUN_NUM_STEPS_PER_ENV, 0.8),
    (5000 * _RUN_NUM_STEPS_PER_ENV, 0.9),
    # Second extension after 0.61 m/s was banked at cmd 0.9: envelope ~1.4,
    # delivery ~68%, so 1.1 commanded targets ~0.75 achieved.
    (5500 * _RUN_NUM_STEPS_PER_ENV, 1.0),
    (6000 * _RUN_NUM_STEPS_PER_ENV, 1.1),
)
_RUN_STANDING_STAGES = (
    (0, 0.02),
    (500 * _RUN_NUM_STEPS_PER_ENV, 0.05),
    (1000 * _RUN_NUM_STEPS_PER_ENV, 0.08),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.10),
)
# GPU run action_rate_l2: -0.1 → -0.5 by iter 3000 (not velocity's -1.0).
_RUN_ACTION_RATE_STAGES = (
    (0, 0.1),
    (1000 * _RUN_NUM_STEPS_PER_ENV, 0.2),
    (1500 * _RUN_NUM_STEPS_PER_ENV, 0.3),
    (2250 * _RUN_NUM_STEPS_PER_ENV, 0.4),
    (3000 * _RUN_NUM_STEPS_PER_ENV, 0.5),
)
_RUN_AIR_MIN = 0.15
_RUN_AIR_MAX = 0.35
_RUN_FOOT_TARGET = 0.03
_RUN_TRACK_STD2 = 0.1
_RUN_ANG_STD2 = 0.5
_RUN_UPRIGHT_STD2 = 0.12
_RUN_WALKING_THRESHOLD = 0.01
_RUN_RUNNING_THRESHOLD = 0.40
# variable_posture per-joint stds, matching GPU run's sagittal loosening.
# Order is LEG_JOINT_IDS: L(yaw,roll,pitch,knee,ankle) R(...).
_RUN_STD_STAND = np.array([0.1, 0.05, 0.15, 0.15, 0.1] * 2)
_RUN_STD_WALK = np.array([0.3, 0.05, 0.4, 0.4, 0.25] * 2)
_RUN_STD_RUN = np.array([0.4, 0.08, 0.6, 0.6, 0.4] * 2)


def _stage_value(n: int, stages: tuple) -> float:
    n = n / _RUN_STAGE_SCALE  # rescale local lifetime steps into GPU stage units
    value = stages[0][1]
    for step, v in stages:
        if n >= step:
            value = v
    return float(value)


def _run_cmd_speed(n: int) -> float:
    return _stage_value(n, _RUN_SPEED_STAGES)


def _run_standing_frac(n: int) -> float:
    return _stage_value(n, _RUN_STANDING_STAGES)


def _run_cmd_norm(env) -> float:
    # Four run terms gate on this each step; memoized in the step cache
    # (commands are frozen for the whole obs+reward phase of a step).
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("cmd_norm")
        if v is not None:
            return v
    c = env.twist_cmd
    v = float(np.linalg.norm(c[:2])) + abs(float(c[2]))
    if cache is not None:
        cache["cmd_norm"] = v
    return v


def _run_speed(env) -> float:
    """GPU track_linear_velocity: body-frame xy vs the twist command, plus
    vz² in the same Gaussian (std²=0.1). Not heading-frame, and not gated
    on upright — those two local inventions paid for dives or taxed the
    lean a run needs."""
    v = env.body_lin_vel()
    c = env.twist_cmd
    err2 = float((c[0] - v[0]) ** 2 + (c[1] - v[1]) ** 2 + v[2] ** 2)
    raw = float(np.exp(-err2 / _RUN_TRACK_STD2))
    # DELIBERATE deviation from the GPU stack: standing pays exactly zero.
    # Upstream's bare Gaussian has height at v=0 whenever the command is low
    # (cmd 0.15 -> 0.80 for immobility), and with upright/pose terms on top a
    # from-scratch policy at OUR sample budget parked itself and collected
    # rent — the user watched it stand perfectly still while this term's bar
    # climbed. Upstream escapes that basin with ~1e9 samples and 4096 envs;
    # we do not. Subtracting the standstill baseline keeps the full gradient
    # from v=0 toward the command while pricing immobility at 0.
    cmd2 = float(c[0] ** 2 + c[1] ** 2)
    if cmd2 < 1e-4:
        # Standing BUCKET (cmd == 0): the baseline subtraction would zero this
        # bucket entirely — no reward for stillness, no charge for wandering —
        # killing the deployment-idle training the bucket exists for. The raw
        # GPU bell is correct here: stillness pays 1, drifting decays it.
        return raw
    base = float(np.exp(-cmd2 / _RUN_TRACK_STD2))
    return max(0.0, raw - base) / max(1.0 - base, 1e-6)


def _run_track_ang(env) -> float:
    """GPU track_angular_velocity: yaw-rate vs cmd[2], plus ω_xy in the
    same Gaussian (std²=0.5)."""
    w = env._gyro
    z_err2 = (float(env.twist_cmd[2]) - float(w[2])) ** 2
    xy_err2 = float(w[0] ** 2 + w[1] ** 2)
    return float(np.exp(-(z_err2 + xy_err2) / _RUN_ANG_STD2))


def _run_air_time(env) -> float:
    """GPU feet_air_time: 1.0 per foot whose CURRENT air time is inside
    [min, max], every step, gated on a live command. 0, 1 or 2.

    The old local version paid a lumped touchdown bonus, which is a
    different function (and ~4× less reward mass for the same stride).
    """
    air = getattr(env, "_run_air", None)
    if air is None:
        air = env._run_air = {"left": 0.0, "right": 0.0}
    paid = 0.0
    moving = _run_cmd_norm(env) > 0.01
    for side in ("left", "right"):
        if env.foot_contact_state[side]:
            air[side] = 0.0
        else:
            air[side] += C.CTRL_DT
        if moving and _RUN_AIR_MIN < air[side] < _RUN_AIR_MAX:
            paid += 1.0
    return paid


def _run_clearance_pen(env) -> float:
    """GPU feet_clearance: -Σ |h − 0.03| · |v_xy|, gated on command.

    A COST, not a Gaussian reward around 4.5 cm (which was above what this
    robot can lift)."""
    if _run_cmd_norm(env) <= 0.01:
        return 0.0
    v6 = _v6_buf(env)
    tot = 0.0
    # Height ABOVE GROUND, not raw geom-center z: the geom center sits at
    # 0.0086 m when the foot is planted, so measuring it raw silently lowered
    # the intended 3 cm swing target to ~2.1 cm of true lift (audit #5).
    # Self-calibrated once, same pattern as foot_flat_ref: the planted-stance
    # center height IS the zero point. GPU measures via a ray sensor, ~0 when
    # planted, so this restores its semantics.
    z0 = getattr(env, "_run_foot_z0", None)
    if z0 is None:
        both_down = all(env.foot_contact_state.values())
        z0 = (float(np.mean([env.data.geom_xpos[g][2]
                             for g in env.foot_geoms.values()]))
              if both_down else 0.0086)
        env._run_foot_z0 = z0
    for gid in env.foot_geoms.values():
        z = float(env.data.geom_xpos[gid][2]) - z0
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 gid, v6, 0)
        vel = float(np.hypot(v6[3], v6[4]))
        tot += abs(z - _RUN_FOOT_TARGET) * vel
    return -tot


def _run_upright(env) -> float:
    """GPU run upright: isotropic Gaussian, std²=0.12 (walk uses 0.05 to
    erase a 2–4° lean; a run needs that lean). Additive — not a multiplier
    on speed."""
    g = env._projected_gravity()
    return float(np.exp(-(g[0] ** 2 + g[1] ** 2) / _RUN_UPRIGHT_STD2))


def _run_pose(env) -> float:
    """GPU variable_posture on LEG joints, with a reachable running_threshold
    of 0.40 m/s (mjlab's default 1.5 is dead code on this robot)."""
    speed = _run_cmd_norm(env)
    if speed < _RUN_WALKING_THRESHOLD:
        std = _RUN_STD_STAND
    elif speed < _RUN_RUNNING_THRESHOLD:
        std = _RUN_STD_WALK
    else:
        std = _RUN_STD_RUN
    q = env._joint_pos_rel()[C.LEG_JOINT_IDS]
    # t.sum()/10 is np.mean's own reduction and division, minus the wrapper.
    t = (q / std) ** 2
    return float(np.exp(-float(t.sum() / 10)))


def _run_head_pose(env) -> float:
    """GPU head_pose_tracking: per-joint Gaussian on (q − head_cmd), std=0.5."""
    err = env._joint_pos_rel()[C.HEAD_JOINT_IDS] - env.head_cmd
    return float(np.exp(-((err / 0.5) ** 2)).sum() / 4)


def _run_slip_pen(env) -> float:
    """GPU feet_slip: -Σ |v_xy|² on contacting feet, gated on command.

    Touchdown is charged (GPU does). The local 'skip first contact' guard
    was for a 10×-too-heavy weight; at GPU's -0.1 it is the wrong prior."""
    if _run_cmd_norm(env) <= 0.01:
        return 0.0
    v6 = _v6_buf(env)
    tot = 0.0
    for side, gid in env.foot_geoms.items():
        if not env.foot_contact_state[side]:
            continue
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_GEOM,
                                 gid, v6, 0)
        tot += float(v6[3] ** 2 + v6[4] ** 2)
    return -tot


def _run_body_ang_pen(env) -> float:
    """GPU body_ang_vel: -ω_xy² on the trunk (run weight -0.025)."""
    w = env._gyro
    return -float(w[0] ** 2 + w[1] ** 2)


def _run_action_rate_weight(n: int) -> float:
    return _stage_value(n, _RUN_ACTION_RATE_STAGES)


def _run_action_rate_pen(env) -> float:
    """GPU action_rate_l2: -weight × Σ(Δa²), weight -0.1 → -0.5.

    No extra 0.02 scale — that made this term ~50× weaker than GPU while
    comments claimed they matched. Lifetime counter lives on the env
    (incremented in BehaviorEnv._compute_reward), so the ramp tracks
    training progress across episode resets."""
    n = getattr(env, "_lifetime_steps", 0)
    da2 = float(((env.last_action - env.prev_action) ** 2).sum())
    return -_run_action_rate_weight(n) * da2


def _run_lateral_pen(env) -> float:
    """Bounded penalty for sliding off the run's LINE (<= 0).

    Heading-relative: the displacement since spawn is projected onto the
    lateral axis of the spawn heading, so a straight run at any start yaw is
    free (world-y alone charged a full unit for running straight under a
    random yaw — measured before this was first fixed). Saturates one unit at
    0.5 m off the line. Re-created after the GPU-port rewrite dropped it: the
    port tracks commanded twist and never needs a line, but our goal is an
    autonomous straight run.
    """
    home = getattr(env, "home_xy", None)
    if home is None:
        return 0.0
    cmd = getattr(env, "twist_cmd", None)  # see _face_home_pen: commanded
    if cmd is not None and (abs(float(cmd[1])) > 0.05 or abs(float(cmd[2])) > 0.1):
        return 0.0                          # turns/sidesteps own the line
    dx = float(env._trunk_xpos[0]) - float(home[0])
    dy = float(env._trunk_xpos[1]) - float(home[1])
    yaw = getattr(env, "home_yaw", None)
    if yaw is not None:
        dy = -np.sin(yaw) * dx + np.cos(yaw) * dy
    return -min((dy / 0.5) ** 2, 1.0)


_register(Behavior(
    id="run",
    emoji="🏃",
    title="Run forward",
    description=(
        "Run forward at a steady clip: quick alternating steps, feet clearing "
        "the floor, body upright and travelling in a straight line."
    ),
    how_it_learns=(
        "The scorecard is the official GPU run task, not a walk retune: the "
        "duck is paid most for matching the commanded body-frame speed, then "
        "for a longer aerial phase and a reachable running pose. A fifth of "
        "practice is still omnidirectional walking, and a little is standing "
        "still — the idle command the robot actually sends at rest. Falling "
        "ends the attempt."
    ),
    keywords=("run", "running", "run forward", "run fast", "sprint", "jog"),
    terms=(
        RewardTerm("keep_pace", "Big points for matching the commanded body-frame speed",
                   4.0, _run_speed),
        RewardTerm("track_turn", "Points for matching the commanded yaw rate",
                   2.0, _run_track_ang),
        RewardTerm("air_time", "Points each step a foot is in a running-length flight",
                   3.0, _run_air_time),
        RewardTerm("stay_upright", "Points for keeping the body upright (lean is cheap)",
                   2.0, _run_upright),
        RewardTerm("pose", "Points for a speed-appropriate leg pose",
                   1.0, _run_pose),
        # 3.5 (was 2.0): at speed the policy ducks its heavy head toward the
        # floor to shift weight forward until it face-plants — measured: a
        # 1-rad slump only cost ~1.0 of the 2.0, cheap enough to be worth it.
        # Head posture is fully observable (neck/head joints + pitch).
        RewardTerm("head_up", "Points for tracking the head-pose command",
                   3.5, _run_head_pose),
        RewardTerm("foot_clearance", "Points lost when a moving foot is at the wrong height",
                   2.0, _run_clearance_pen, is_penalty=True),
        RewardTerm("plant_the_foot", "Points lost when a planted foot skids",
                   0.1, _run_slip_pen, is_penalty=True),
        RewardTerm("smooth_moves", "Penalty for jerky movements (ramps in as the gait appears)",
                   1.0, _run_action_rate_pen, is_penalty=True),
        RewardTerm("no_limit_parking", "Penalty for grinding joints against their end stops",
                   1.0, _limit_parking_pen, is_penalty=True),
        # Straightness anchors — deliberately NOT in the GPU stack, which
        # tracks commanded twist and lets a teleoperator own the heading. We
        # ask for a straight autonomous run, and rate terms cannot stop slow
        # accumulated drift (session law, learned twice): track_turn charges
        # turning FAST, so a lazy arc was free and the user watched the duck
        # curve away. These anchor ABSOLUTE heading and the lateral line.
        # NOT PORTED from the GPU run cfg, documented for honesty (audit #4):
        # foot_swing_height (-0.25, needs per-foot peak-height tracking),
        # angular_momentum (-0.01 there; calm_roll covers the same axis here),
        # self_collisions (-1.0; no local contact sensor for it).
        # NO absolute-heading terms, by hard-won design: the 61-obs contract
        # carries gyro+gravity+joints+actions+commands — pitch and roll but NO
        # yaw. A memoryless policy cannot perceive its heading error, so terms
        # anchored to it (face_home, hold_the_line, an anchor-frame keep_pace
        # — all three were tried) are unlearnable noise; the duck circled for
        # 30M+ steps regardless. Straightness is the COMMANDER'S job: a
        # 3-line heading-hold (cmd wz = -4*yaw_err) on top of the trained
        # yaw-rate tracking measured 0.293 m/s, 4 deg err, 0 falls where a
        # day of reward surgery never went below 64 deg.
        RewardTerm("calm_roll", "Penalty for rolling/pitching the trunk too fast",
                   0.025, _run_body_ang_pen, is_penalty=True),
    ),
    default_steps=20_000_000,
    success_metric="metres per second sustained without falling",
    episode_s=20.0,
    forward_cmd=0.6,
))


def match_behavior(text: str) -> Behavior | None:
    """Cheap keyword matcher from a chat message to a behavior."""
    t = " " + text.lower().replace("1", "one").strip() + " "
    best, best_score = None, 0.0
    for b in BEHAVIORS.values():
        # Score by how much of the message a keyword actually explains, so a
        # SPECIFIC phrase outranks a generic substring of itself: "jump
        # backflip" must reach the jumping flip, not the floor roll that owns
        # the word "backflip".
        score = sum(1 + len(k) / 100.0 for k in b.keywords if k in t)
        if b.title.lower() in t:
            score += 2
        if score > best_score:
            best, best_score = b, score
    return best


def behavior_card(b: Behavior, extra_keys: tuple[str, ...] = ()) -> dict:
    """JSON shape the teach panel renders. `extra_keys` are adopted CATALOG
    terms (a run's weights entries outside the recipe) — they render as
    ordinary slider rows; `availableTerms` is what "＋ add a term" offers."""
    recipe = tuple(b.terms) + tuple(
        CATALOG[k] for k in extra_keys if k in CATALOG)
    return {
        "id": b.id,
        "emoji": b.emoji,
        "title": b.title,
        "description": b.description,
        "howItLearns": b.how_it_learns,
        "successMetric": b.success_metric,
        "defaultSteps": b.default_steps,
        # Stage labels/steps/details — env knobs are trainer plumbing, not
        # story (detail is the knobs' story, told in plain words).
        "curriculum": [{"label": s.label, "steps": s.steps, "detail": s.detail}
                       for s in b.curriculum],
        "availableTerms": catalog_cards(
            exclude=tuple(t.key for t in recipe)),
        "terms": [
            {"key": t.key, "friendly": t.friendly, "weight": t.weight,
             "isPenalty": t.is_penalty}
            for t in recipe
        ],
    }


class BehaviorEnv(MicroduckWalkEnv):
    """Walking env with the reward stack replaced by a behavior's recipe.

    Commands are pinned to zero (tiny keep-alive noise stays via reset
    sampling being overridden) so the 61-obs contract is preserved and the
    exported policy hot-swaps like any other.
    """

    def __init__(self, behavior_id: str,
                 weight_overrides: dict[str, float] | None = None,
                 spawn_overrides: dict[str, str] | None = None,
                 spotter: bool = False, clip_name: str | None = None, **kwargs):
        self.behavior = BEHAVIORS[behavior_id]
        kwargs.setdefault("max_episode_s", self.behavior.episode_s)
        # Per-stage episode length (MICRODUCK_EPISODE_S, instance override or
        # trainer env var): a landing rehearsal is decided in ~4 s — clipping
        # those stages' episodes multiplies rehearsals per wall-clock and
        # spares the viewer 10 s of a duck lying still. Hard override: the
        # stage knob must win even when a caller passes max_episode_s.
        ep = ((spawn_overrides or {}).get("MICRODUCK_EPISODE_S")
              or os.environ.get("MICRODUCK_EPISODE_S"))
        if ep:
            try:
                kwargs["max_episode_s"] = float(ep)
            except ValueError:
                pass
        if self.behavior.scene == "all":
            kwargs.setdefault("scene_xml", str(C.SCENE_ALL_XML))
        kwargs.setdefault("terminate_on_fall", self.behavior.terminate_on_fall)
        # GPU locomotion has no height termination; a bouncing stride can dip
        # the trunk through 0.07 m without having fallen.
        if self.behavior.forward_cmd:
            kwargs.setdefault("height_termination", False)
        self.foot_contact_state = {"left": True, "right": True}
        # Reference motion, if this behavior imitates one. The clip is
        # selectable per run (a user authors several in the timeline editor):
        # explicit kwarg wins, then MICRODUCK_CLIP for the trainer subprocess,
        # then the recipe's default.
        name = resolve_clip_name(self.behavior, clip_name)
        self.clip = motion.load_clip(name) if name else None
        if self.clip is not None and self.clip.loop:
            # Locomotion rules, as every walking task uses (and as our own
            # one_leg/crouch/stand behaviors already do): a fall ends the
            # episode. Without this the duck face-plants at 0.7 s and spends
            # the next 3 s earning pose matches from the floor.
            kwargs["terminate_on_fall"] = True
            kwargs.pop("scene_xml", None)          # the walk scene, not "all"
            self.behavior = replace(self.behavior, scene="walk",
                                    terminate_on_fall=True)
        # Demo assist (see Behavior.spotter_fn) — showcase previews only.
        self.spotter = bool(spotter) and self.behavior.spotter_fn is not None
        self.spotter_active = False
        # Per-instance stage knobs (spawn windows/mix), consulted BEFORE
        # os.environ by _spawn_knob: the trainer subprocess can keep riding
        # its environment, but envs living inside the farm process (the
        # trainee preview) need per-instance values — os.environ there is
        # shared across every duck and never carries the active stage.
        self.spawn_overrides = dict(spawn_overrides or {})
        # Clamped to >= 0: a negative weight on a self-negating penalty would
        # double-negate into a reward for the violation (AGENTS.md's four-env
        # sign bug) — the UI's sliders must not be able to reintroduce it.
        self.weight_overrides = {
            k: max(0.0, float(v)) for k, v in (weight_overrides or {}).items()
        }
        # Overrides for keys OUTSIDE the recipe adopt that CATALOG term at the
        # given weight — the "＋ add a term" channel, riding the same weights
        # pipe as the sliders (train_behavior/scale restarts need no changes).
        recipe_keys = {t.key for t in self.behavior.terms}
        self._terms = tuple(self.behavior.terms) + tuple(
            CATALOG[k] for k in self.weight_overrides
            if k not in recipe_keys and k in CATALOG
        )
        # Hot-loop view of the recipe: (key, output name, default weight, fn)
        # per term, resolved once — _compute_reward runs it every step.
        self._term_rows = tuple(
            (t.key, t.key if not t.is_penalty else t.key + "_penalty",
             t.weight, t.fn)
            for t in self._terms)
        super().__init__(**kwargs)
        # Flat-foot reference: super().__init__ leaves the model posed at the
        # STAND keyframe (that's how stand_z is measured), so each foot's
        # foot-frame gravity right now IS what "flat on the floor" looks like.
        self.foot_flat_ref = {}
        for side, gid in self.foot_geoms.items():
            R = self.data.geom_xmat[gid].reshape(3, 3)
            self.foot_flat_ref[side] = R.T @ np.array([0.0, 0.0, -1.0])

    def step(self, action):
        if self.spotter:
            self.spotter_active = bool(self.behavior.spotter_fn(self))
        return super().step(action)

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        self.data.qfrc_applied[:] = 0.0   # never carry an assist across episodes
        self.spotter_active = False
        # Per-episode reward state (see Behavior.state_fn). Zeroed before the
        # spawn families run so a mid-maneuver spawn can PRESET it to match
        # the attitude it poses (a body spawned halfway through a flip has,
        # by definition, already rotated halfway).
        self._bf_rot = 0.0
        # Air-time bookkeeping is per-episode state too. _run_air_time banks
        # time while a foot is off the ground and pays it out at touchdown, so
        # time accrued during a terminal FALL would otherwise be paid on the
        # next episode's first contact -- free money at every spawn.
        self._run_air = {"left": 0.0, "right": 0.0}
        self._run_was = {"left": True, "right": True}
        self._run_contact_age = {"left": 0, "right": 0}
        # Audit item: these lazy-init memories leaked across episodes — one
        # spurious soft-landing / torque-rate / head-contact charge on the
        # first step after a violent episode end.
        self._prev_tau = None
        self._prev_vz = None
        self._gp_prev_head = False
        # What kind of start this episode got — surfaced on the viewer's duck
        # label so a watcher can tell a landing rehearsal from a plain
        # standing start (visually near-identical for some spawn families).
        self.last_spawn = "standing"
        u = self._rng.uniform()
        if self.behavior.spawn_families:
            # Per-stage spawn MIX override (comma-separated probs, positional):
            # a curriculum stage labeled "learning to land" must actually be
            # mostly landing spawns — the declared mix (tuned for the final
            # integration stage) left 55% of a focused stage's episodes as
            # plain upright starts, rehearsing nothing the stage is for (a
            # user watched stage 1 and rightly asked what it was doing).
            fams = self.behavior.spawn_families
            probs_env = _spawn_knob(self, "MICRODUCK_SPAWN_FAMILY_PROBS")
            if probs_env:
                try:
                    probs = [float(x) for x in probs_env.split(",")]
                except ValueError:
                    probs = []
                if len(probs) == len(fams):
                    fams = tuple((p, fn) for p, (_, fn) in zip(probs, fams))
            acc = 0.0
            for prob, fn in fams:
                acc += prob
                if u < acc:
                    out = (fn(self), out[1])
                    kind = fn.__name__.split("_spawn_")[-1].replace("_", "-")
                    if 0.0 < self._bf_rot < 2.0 * np.pi:
                        kind += f" {np.degrees(self._bf_rot):.0f}°"
                    self.last_spawn = kind
                    break
        elif u < self.behavior.inverted_spawn_prob:
            out = (self._spawn_inverted(), out[1])
            self.last_spawn = "inverted"
        elif u < (self.behavior.inverted_spawn_prob
                  + self.behavior.mid_flip_spawn_prob):
            out = (self._spawn_mid_flip(), out[1])
            self.last_spawn = "mid-flip"
        # Anchors for stay_home / face_home: where this episode began.
        self.home_xy = (float(self.data.xpos[self.trunk_body_id][0]),
                        float(self.data.xpos[self.trunk_body_id][1]))
        self.home_yaw = _trunk_yaw(self)
        return out

    def _spawn_inverted(self):
        """Reverse-curriculum spawn: drop into a noisy near-headstand — trunk
        pitched ~170° with the head at the floor and legs roughly straight up.
        Imperfect on purpose: crumbling spawns teach recovery, clean ones teach
        the hold, and the value of BEING inverted propagates back to make the
        kick-over worth attempting from upright spawns."""
        import mujoco
        r = self._rng
        d, m = self.data, self.model
        pitch = np.pi - 0.17 + r.uniform(-0.12, 0.12)
        d.qpos[:] = 0.0
        d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        d.qpos[2] = 0.165 + r.uniform(-0.01, 0.01)  # head shell near the floor
        q = d.qpos  # leg joints ~straight (the target pose), light noise
        for i, adr in enumerate(self.joint_qpos_adr):
            q[adr] = r.uniform(-0.15, 0.15)
        # Neck TUCKED (chin to chest) — the reference technique's key: the
        # crown, not the beak, meets the floor.
        q[self.joint_qpos_adr[5]] = -0.6 + r.uniform(-0.15, 0.15)
        q[self.joint_qpos_adr[6]] = -0.6 + r.uniform(-0.15, 0.15)
        d.qvel[:] = 0.0
        d.ctrl[:] = d.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(m, d)
        self.prev_joint_vel = self._joint_vel().copy()
        return self._get_obs()

    def _spawn_mid_flip(self):
        """Mid-maneuver spawn: the face-plant tripod — nose down past
        vertical, crown on the floor, neck tucked, legs folded with feet
        still planted, butt at a random height. The kick-over's launch pad."""
        import mujoco
        r = self._rng
        d, m = self.data, self.model
        pitch = r.uniform(1.7, 2.5)  # nose-down, between face-plant and vertical
        d.qpos[:] = 0.0
        d.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        d.qpos[2] = 0.13 + r.uniform(-0.015, 0.015)
        q = d.qpos
        hip = r.uniform(0.3, 0.9)
        knee = r.uniform(0.3, 1.0)
        for i, adr in enumerate(self.joint_qpos_adr):
            q[adr] = r.uniform(-0.1, 0.1)
        q[self.joint_qpos_adr[2]] = -hip   # left hip_pitch (folded under)
        q[self.joint_qpos_adr[11]] = hip   # right hip_pitch (mirrored sign)
        q[self.joint_qpos_adr[3]] = -knee
        q[self.joint_qpos_adr[12]] = knee
        q[self.joint_qpos_adr[5]] = -0.6 + r.uniform(-0.15, 0.15)  # neck tucked
        q[self.joint_qpos_adr[6]] = -0.6 + r.uniform(-0.15, 0.15)
        d.qvel[:] = 0.0
        d.ctrl[:] = d.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(m, d)
        self.prev_joint_vel = self._joint_vel().copy()
        return self._get_obs()

    def _sample_commands(self) -> None:
        # Tricks: zero twist, keep-alive noise on head/body slots.
        # Locomotion: GPU run command mix — standing bucket, 55% straight
        # forward with vx clamped ≥ 0.3, remainder omnidirectional. Speed
        # ceiling and standing fraction follow the GPU curricula.
        r = self._rng
        self.twist_cmd[:] = 0.0
        if self.behavior.forward_cmd:
            pinned = _spawn_knob(self, "MICRODUCK_RUN_CMD")
            if pinned:
                try:
                    self.twist_cmd[0] = float(pinned)
                except ValueError:
                    pinned = None
            if not pinned:
                n = getattr(self, "_lifetime_steps", 0)
                speed = _run_cmd_speed(n)
                stand_p = _run_standing_frac(n)
                u = r.uniform()
                if u < stand_p:
                    self.twist_cmd[:] = 0.0
                elif u < stand_p + _RUN_FORWARD_FRAC:
                    vx = abs(float(r.uniform(-speed, speed)))
                    vx = max(vx, min(_RUN_FORWARD_CLAMP, speed))
                    self.twist_cmd[:] = (vx, 0.0, 0.0)
                else:
                    self.twist_cmd[:] = (
                        r.uniform(-speed, speed),
                        r.uniform(*_RUN_LIN_VEL_Y),
                        r.uniform(*_RUN_ANG_VEL_Z),
                    )
        self.head_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.HEAD_CMD_RANGES]
        self.body_cmd[:] = [r.uniform(lo, hi) for lo, hi in C.BODY_CMD_RANGES]
        # Re-anchor the straightness terms to HERE: after an obedient turn
        # segment, the old spawn heading/line is ancient history — measured
        # up to ~-1500/episode charged for perfectly tracking the NEW straight
        # command against the OLD line (audit finding #1).
        if getattr(self, "trunk_body_id", None) is not None:
            self.home_xy = (float(self.data.xpos[self.trunk_body_id][0]),
                            float(self.data.xpos[self.trunk_body_id][1]))
            self.home_yaw = _trunk_yaw(self)

    def _get_obs(self):
        # Imitation needs a sense of TIME: the policy is memoryless and the
        # same body pose means different things at different points in a clip.
        # The phase rides in two body-command slots (noise otherwise), so the
        # 61-dim contract is untouched — see motion.phase_signal.
        if self.clip is not None:
            s, c = self.clip.phase(self.step_count)
            self.body_cmd[4], self.body_cmd[5] = s, c
        return super()._get_obs()

    def _compute_reward(self):
        # Counter is owned and seeded by MicroduckWalkEnv.__init__ (which
        # reads MICRODUCK_RAMP_OFFSET so warm restarts resume ramps at
        # strength); here it only advances.
        self._lifetime_steps += 1
        self.foot_contact_state = self._foot_contacts()
        if self.behavior.state_fn is not None:
            self.behavior.state_fn(self)
        # _term_rows precomputes the "<key>_penalty" output names (an f-string
        # per penalty per step adds up at ~20 terms x 10k steps/s). The total
        # MUST stay builtin sum(): since 3.12 it runs Neumaier-compensated
        # summation over floats, so a plain `total += v` loop differs by an
        # ulp (found the hard way by the parity goldens).
        terms = {}
        wo = self.weight_overrides
        if wo:
            for key, out_key, w, fn in self._term_rows:
                terms[out_key] = wo.get(key, w) * fn(self)
        else:
            for key, out_key, w, fn in self._term_rows:
                terms[out_key] = w * fn(self)
        return float(sum(terms.values())), terms
