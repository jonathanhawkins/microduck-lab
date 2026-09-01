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
from dataclasses import (  # noqa: F401 — `replace` rides the cascade (weight overrides downstream)
    dataclass,
    field,
    replace,
)
from typing import Callable

import mujoco  # noqa: F401 — cascade re-export: trick modules and env.py use it
import numpy as np

from .. import contract as C

# The next two are unused HERE but re-exported through the package's
# star-import cascade (ruff once stripped them and env.py lost its base
# class): motion feeds the clip machinery, MicroduckWalkEnv is
# BehaviorEnv's base in env.py.
from .. import motion  # noqa: F401
from ..walk_env import MicroduckWalkEnv  # noqa: F401


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
    # knobs. Empty = ordinary single-run training. The lab server
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
    stage knobs via its environment (one process, one stage), but the lab
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
    # Bounded like every other penalty (the run-recipe lesson: unbounded rate
    # penalties dominate under a flailing policy — this was the last one).
    # Saturates at ~1.4 m/s of drift speed.
    return -min(0.5 * (fwd * fwd + lat * lat), 1.0)


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
    # Bounded (the last stragglers of the run-recipe lesson): saturates one
    # unit at ~41 cm from home, so a single wander can't swamp the return.
    return -min(6.0 * dist2, 1.0)


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
    # SIGNED pay in the COMMANDED direction (cap 5 rad/s, no pay for
    # violence). The old abs(wz) paid a pelvis WIGGLE almost as well as a
    # true spin — oscillating +-3 rad/s collects |wz| pay with zero net
    # rotation ("the pelvis is spinning but it really doesn't go around").
    # The direction rides the observable wz command slot (per episode,
    # reasserted every obs); rotation AGAINST the command charges, so a
    # wiggle nets ~zero while a steady spin collects in full. Mirror-safe:
    # the symmetry map negates the wz command and the gyro together.
    d = float(np.sign(getattr(env, "_spin_dir", 1.0)) or 1.0)
    return float(np.clip((wz * d) / 5.0, -1.0, 1.0))


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
            from ..bam_actuator import XL330_MAX_TORQUE
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


def _step_dont_skid(env) -> float:
    """Pay for LIFTING feet while rotating — the anti-skid-steer term (0..1).

    A spinning duck can yaw by pivoting planted feet (scooting), which looks
    wrong and grinds the real robot's soles. This is the walk's air_time idea
    made drive-command-free: gate on the OBSERVABLE rotation itself (|wz|),
    pay 0.5 per foot whose current airborne time is inside a stride-like
    window (0.06-0.35 s). Catalog term — any behavior can adopt it from the
    panel's "+ add a term".
    """
    if abs(float(env._gyro[2])) < 0.5:      # not rotating: nothing to step
        return 0.0
    air = getattr(env, "_skid_air", None)
    if air is None:
        air = env._skid_air = {"left": 0.0, "right": 0.0}
    pay = 0.0
    for side in ("left", "right"):
        if env.foot_contact_state[side]:
            air[side] = 0.0
        else:
            air[side] += C.CTRL_DT
            if 0.06 <= air[side] <= 0.35:
                pay += 0.5
    return pay


CATALOG: dict[str, RewardTerm] = {
    t.key: t for t in (
        RewardTerm("step_dont_skid", "Points for lifting the feet to step around (no skid-steering)",
                   1.5, _step_dont_skid),
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



# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
