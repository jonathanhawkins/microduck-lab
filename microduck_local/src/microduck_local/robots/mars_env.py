"""What MARS can be TRAINED to do: reach a point, and pick a block up.

`docs/mars-roadmap.md` Phase 4. The design decision behind this file is §1's:
MARS does not walk, so `MicroduckWalkEnv` is not one kwarg away — its fields
are foot geoms, fall heights, air-time windows and a gyro, and a wheeled base
has none of them. `MarsArmEnv` is therefore its OWN `gym.Env`, and what it
shares with the walking env is the trainer's contract rather than its code:
the same constructor kwargs `train.env_kwargs_from_args` produces, the same
`info["episode_rewards"]` the lab's teach panel graphs, the same
`observation_space` / `action_space` / `reset(seed=)` shape SB3 needs.

**The observation is the declared `mars-arm-32-v1` contract, filled.** The
slot table has existed since Phase 2a (`robots/mars.OBS_*`,
`MarsBody.contract()`) precisely so the first env would have to fill a layout
it did not get to choose. `_SLOTS` below is asserted against
`MarsBody.contract().slots` at construction, name by name and width by width:
a builder that drifts from the published table is a policy whose floats mean
something other than what its own file says they mean, which is the whole
failure `robots/policy_contract.py` exists to stop.

**Actuators: Innate's PD for every joint, and the run says so.** Joints 4-6
and the head are XL330s — the servo this repo's BAM model was identified on —
but joints 1-3 are XL430/XC430 with no fit here, so v1 drives the whole arm
with Innate's own position PD (`mars.arm_servo`, KP 50 / KD 1) and `bam` is
REFUSED rather than applied to three joints it does not describe
(`docs/mars-roadmap.md` §6.6: per-joint servo models are the eventual fix).
Pretending one switch covers the arm is the thing to avoid.

**Both halves of the actuation go through `MarsDriver`**, the same object a
`/sim` room steps (`robots/mars_drive.py`): the base's velocity PD through
`xfrc_applied` and the arm/head position PD through `qfrc_applied`, with
Innate's station keeping, watchdog and joint2 guard. The env therefore cannot
disagree with the room about what a command does — which is the one thing that
would make a policy trained here useless the moment it was dropped into the
playroom.

MEASURED before a line of the reward was written (the "check a knob's
reachable set first" rule; `scratchpad/probe_arm*.py`, 2026-09-17):

  * **the arm's HOME is not in the middle of its range.** joint1 sits at
    +1.445 rad with its stop at +1.571, so the arm parks folded across the
    chassis pointing at +83 deg of yaw, and the front arc this task samples
    is 1.4-2.5 rad of joint1 travel AWAY from where every episode starts.
  * **a shell solution costs up to 3.09 rad on its worst joint** (median
    2.01, p95 2.6-3.0 over three windows, 60 targets each, solved by
    coordinate descent over the full ranges with self-colliding poses
    rejected). That is where `ACTION_SCALE_RAD = 3.0` comes from: a smaller
    box provably excludes the poses the task needs, and at 3.0 an action of
    +-1 spans 82-127% of every joint's one-sided headroom from HOME, so
    "+-1 spans most of the range" is true joint by joint.
  * **69-85% of poses in that box self-collide**, dominated by
    `base_link <-> link2 / link5 / link4` — the arm folding down onto its own
    chassis. Innate's joint2 guard is already applied and removes only ~10
    points of it. So the self-collision termination is not a rare safety net,
    it is the environment's dominant early dynamic, and the episode-length
    column is where to read it.
  * **nothing self-collides at HOME** (2 s of `arm_servo` hold: the only
    contacts are floor/chassis and the two wheels), so the termination does
    not fire at reset — the terminal equivalent of AGENTS.md's "check the
    term is not FLAT where the policy starts".
  * **the shell is the arm's**: residual |ee - target| over the planned shell
    is 0.9 mm median, 13 mm p90, 62 mm max, with 87% of draws solved inside
    1 cm and 95% inside 2 cm. The tail is the solver's local minima as much
    as the geometry's, and it is why the acceptance bar is quoted as a count
    rather than assumed to be 8/8 (see the report for this phase).

And MEASURED before a line of `pick`'s reward was written, which is the same
rule applied to a second task (`scripts/probe_mars_pick.py --scripted`,
2026-09-18):

  * **the claw holds the playroom's block, but only at 2 ms.** A scripted
    pick — arm to an open-jaw pose, the 4 cm / 20 g `block` toy placed between
    the blades, joint6 commanded 0.6 rad past its stop, lift 10 cm, hold 2 s —
    at **16 spots drawn from the shell**: MARS's own option block holds
    **4/16 at the lab's 5 ms step and 14/16 at Innate's 2 ms**. The failure
    mode is not slip, it is EJECTION: at 5 ms the block leaves at 0.08-6.6 m
    of travel in the two seconds after the lift, which is a contact impulse
    the coarse step cannot integrate. **Both 2 ms misses are the solver's,
    not the grasp's** — they are the two shell-EDGE spots (+-57 and +-60 deg
    of the +-60 deg arc) whose open-jaw pose lands 13.7 and 15.1 mm off the
    block, the worst two IK residuals in the set, and every spot the claw was
    actually put on held. So `pick` compiles its own scene at
    `PICK_PHYSICS_DT` with decimation 20 and the contract's 25 Hz is unmoved;
    `reach`, which closes on nothing, keeps the world's 5 ms and every number
    Phase 4a measured.
  * **the elliptic cone and `impratio 10` are NOT the lever, and that closes
    Phase 3b's open question.** The same 16 spots under the world's own option
    block (pyramidal cone, impratio 1, the spec's Euler integrator):
    **1/16 at 5 ms and 14/16 at 2 ms** — the same 14 at the step that
    matters. (And the G1's variant of the world's block — `implicitfast`,
    iterations 10, ls_iterations 20 — was bit-identical to the plain world's
    in all 24 cells of an earlier sweep, so the integrator and the solver
    budgets change nothing here either; the cone is the only part of MARS's
    own block that does anything at all, and at 2 ms it does not do this.)
    Innate set those two to stop a held object creeping out of a closed claw,
    and at 2 ms nothing creeps under either cone. What a MARS in a ROOM needs
    for a grasp is therefore the smaller TIMESTEP — which is the whole
    world's, and a `/sim` decision rather than a per-body one — and not a cone
    the composer can set for one robot.
  * **`gripper_load` is the constraint torque, and the servo torque could
    not have been it.** The table is in `robots/mars.OBS_GRIPPER_LOAD`'s
    block: `qfrc_constraint` at joint6 reads 0.0000 open, 0.0000 shut on air,
    0.0000 with the shut claw driven into the floor, 0.0035 with the jaws
    resting on the block and 1.97 holding it, while `qfrc_applied` is
    saturated at -2 N*m for 8 of the 40 control steps of a close on AIR and
    so cannot tell "closing" from "holding" from one sample.
  * **the pick is expressible in the eight floats a policy emits**, which is
    the exploration question `AGENTS.md` asks before any reward
    (`--scripted-env`): the same open / place / close / lift, driven by
    `delta` ACTIONS through this env's own `step`, ends `holding=True` at a
    **9.0 cm lift with load +1.997, one grasp, hold_frac 0.722 and the task's
    own `success` rule satisfied**. A skill the action space cannot express
    is one no weight can buy — that is the lesson `delta` itself came from.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .. import contract as C
from . import mars
from .mars_drive import MAX_CMD_LINEAR, MAX_CMD_YAW, MarsDriver

#: Physics steps per control step. MARS's contract ticks at 25 Hz (Innate's
#: policy-defined skills do, `mars.CONTROL_HZ`), and the world runs at
#: `C.PHYSICS_DT` = 5 ms, so 8 — where the duck's `C.DECIMATION` is 4 at
#: 50 Hz. Computed, not typed, so a change to either constant cannot leave a
#: policy trained at one rate and stepped at another.
DECIMATION = int(round(1.0 / (mars.CONTROL_HZ * C.PHYSICS_DT)))
CTRL_DT = C.PHYSICS_DT * DECIMATION

#: The physics step `pick` runs at, and the one number Phase 4b measured that
#: changed the ENVIRONMENT rather than the reward.
#:
#: The module docstring has the table: a scripted pick of the playroom block
#: holds **4 of 16 spots at 5 ms and 14 of 16 at 2 ms**, and what fails at
#: 5 ms is ejection (0.08-6.6 m of block travel in the two seconds after the
#: lift), not slip. Innate's own world runs at 2 ms and their finger contact
#: model — condim 6, friction 2.0/0.05/0.02, `solref (0.005, 1)`, armature
#: 1e-4 on a 2e-5 inertia blade — was tuned there; Phase 2 checked that the
#: coarser step survives a HOLD (0.00344 rad, identical to five decimals) and
#: that check simply does not cover a contact.
#:
#: The CONTROL rate is untouched: `PICK_DECIMATION` is computed so that
#: `PICK_PHYSICS_DT * PICK_DECIMATION` is the same 40 ms `CTRL_DT` the
#: contract declares. A policy trained under `pick` and one trained under
#: `reach` tick the same clock and mean the same thing; what differs is how
#: finely the world underneath is integrated.
#:
#: Phase 5 moved the FLOAT to `mars.GRASP_PHYSICS_DT` and kept this name: a
#: `/sim` room needs the same number (`world/scenario.Scenario.physics_dt`,
#: read from `MarsBody.physics_dt`), and `world/scenario.py` cannot import
#: this module — it is the on-disk contract and this one pulls gymnasium in.
#: Same value, one definition, and `tests/test_mars_pick.py`'s planted breaks
#: still read this name.
PICK_PHYSICS_DT = mars.GRASP_PHYSICS_DT
PICK_DECIMATION = int(round(1.0 / (mars.CONTROL_HZ * PICK_PHYSICS_DT)))

#: `target = ARM_HOME + action[0:6] * ACTION_SCALE_RAD`, clipped to each
#: joint's URDF range. 3.0 rad: see the module docstring's measurement — a
#: shell solution costs up to 3.09 rad on its worst joint, and the one-sided
#: headroom from HOME is 3.02 / 2.61 / 3.09 / 2.37 / 1.66 / 0.87 rad for
#: joints 1-6, so +-1 covers 82-127% of it. The clip is what keeps the box
#: legal; a joint whose headroom is smaller than the scale simply saturates,
#: exactly as the duck's +-4 action box saturates against its own ranges.
ACTION_SCALE_RAD = 3.0
#: The action box. +-1, so the scale above reads as "what +-1 means in rad"
#: and the base pair is a fraction of the command envelope.
ACTION_CLIP = 1.0

#: How fast the COMMANDED target may travel, rad/s per joint.
#:
#: This is the physics ladder, and it exists because the first version of this
#: env was unlearnable in a way the reward could never have fixed. MEASURED
#: (`scratchpad/probe_random.py`, a uniform random policy, 60 episodes): with
#: the target applied instantly, **60 of 60 episodes ended in a
#: self-collision inside 1-5 control steps** (mean episode length 1.6 of 200),
#: 53 of them `base_link <-> link2`. The cause is not the policy and not the
#: pay: an absolute joint target applied at 25 Hz is a 3 rad teleport every
#: 40 ms, and the arm's own chassis sits between HOME and the front arc this
#: task samples, so the servo drove straight through it on the first step. No
#: term, weight or ramp teaches a skill that no rollout ever contains
#: (AGENTS.md, "Reward design cannot fix an exploration gap") — the fix is the
#: world.
#:
#: And the world was WRONG, which is what makes this a ladder rung rather
#: than a crutch. Innate's `arm_servo` is KP 50 with a 50 N*m clamp on links
#: whose URDF inertias are 0.001 placeholders, so the sim's arm accelerates
#: far harder than the Dynamixels do. `[datasheet, not measured here]` the
#: slowest joints are the XL430-W250-T shoulder pair at 61 rpm no-load
#: (6.39 rad/s at 12 V); the XL330 wrist and gripper are ~104 rpm
#: (10.9 rad/s). 6.0 rad/s is the slow pair, rounded down, applied to every
#: joint — so one control step may move a target 0.24 rad, which is what the
#: real arm could follow. The ACTION still means an absolute joint target
#: (`mars-arm-32-v1`'s 8 actions are unchanged); what is bounded is how fast
#: the env walks its commanded target there, and `arm_qpos` in the
#: observation is where the arm actually got to.
#:
#: NOT in `MarsDriver`: a brain in a `/sim` room can still command a step
#: target, because Phase 3a's drive numbers are measured against that
#: behaviour and a rate limit there would change them. Moving it down into the
#: driver as an optional default-off knob (the way `RayFan` grew
#: `exclude_body=`) is Phase 3b/4b work, and the day MARS has an `Intent.arm`
#: channel it should be shared rather than copied.
#:
#: Phase 5 moved the FLOAT to `mars.MAX_TARGET_RATE_RAD_S` and kept this name,
#: for the third time and the same reason: `brain/tidy_arm.py` writes arm
#: targets through `Intent.arm`, which `MarsDriver.set_arm` applies with no
#: limit of its own, so a brain has to carry the servo's speed too — and a
#: limit with two definitions is 4a's bug back again. The block on `mars.py`
#: records what forgetting it did in a room.
MAX_TARGET_RATE_RAD_S = mars.MAX_TARGET_RATE_RAD_S

# ------------------------------------------------------- the action map
#
# WHAT AN ACTION MEANS. Phase 4a shipped one map — a rate-limited ABSOLUTE
# joint target, `HOME + a * ACTION_SCALE_RAD` — and its acceptance measurement
# is what put these three here. `mars-reach-v2` reached the target inside 1 s
# and then limit-cycled between 1.1 and 2.3 cm forever: 3/8 seeds ended inside
# 2 cm, 0/8 held it for the second the task asks for, with 58% of the six arm
# dims sitting on the +-1 box edge.
#
# The mechanism is in the map rather than in the pay, which is why no reward
# term appears below. A rate-limited absolute target has NO FIXED POINT the
# policy can name: `a` says where to go, the target walks there at 6 rad/s,
# and holding still requires `a` to keep pointing exactly at wherever the
# target already is — a moving quantity the policy has to re-hit every 40 ms.
# A saturated mean cannot even try: on the box edge it can only slew up or
# slew down, so it alternates. AGENTS.md's "reward design cannot fix an
# exploration gap" has a sibling here — a reward cannot pay for a behaviour
# the action space has no way to express.
#
#: The maps `MarsArmEnv(action_mode=...)` implements.
#:
#:   * `"delta"` — `target += a * DELTA_RAD_PER_STEP` from the CURRENT
#:     commanded target. `a = 0` is an exact fixed point, so "stay" exists.
#:   * `"cubic"` — `target = HOME + sign(a)|a|^3 * scale`. Absolute, and odd
#:     and monotone so +-1 still spans the box, but the interior is stretched:
#:     millimetres of effector travel per unit action near zero against the
#:     full reach at the edge.
#:   * `"absolute"` — Phase 4a's, kept so its numbers stay reproducible and so
#:     a narrower box (`action_scale_rad`) remains expressible.
ACTION_MODES: tuple[str, ...] = ("absolute", "delta", "cubic")

#: rad the commanded target moves per unit of action in `"delta"` mode.
#:
#: DERIVED from the rate limit, not typed, and the equality is the point:
#: `MAX_TARGET_RATE_RAD_S * CTRL_DT` is exactly what one control step is
#: allowed to move a target, so a full-scale delta asks for precisely the
#: fastest legal step and every value in between is a real command.
#:
#: Both directions of getting this wrong are failures this repo has a rule
#: for. LARGER, and the rate limiter silently truncates the top of the action
#: box: a band of actions would all produce the identical motion and
#: `last_action` would report a number the arm never followed — a knob that
#: changes nothing is broken, not null (AGENTS.md rule 0). SMALLER, and the
#: arm can no longer slew as fast as the servos it is modelling: MEASURED, a
#: shell solution's worst joint travels 2.62 rad at p95 (`scratchpad/
#: probe_box_reach2.py`, the solver `ACTION_SCALE_RAD` was chosen with), which
#: is 11 control steps of 0.24 rad — 0.44 s of an 8 s episode — and halving
#: the step doubles that for nothing.
#:
#: The reachable set (AGENTS.md, "check a knob's reachable set first"): 200
#: control steps x 0.24 rad is 48 rad of travel per joint against a 2.62 rad
#: p95 requirement, so `"delta"` can reach every pose `"absolute"` can. It
#: costs travel TIME, not reach.
DELTA_RAD_PER_STEP = MAX_TARGET_RATE_RAD_S * CTRL_DT

#: `"absolute"`'s box narrowed to what Phase 4a's plan called a second
#: curriculum rung — and it is here as a REFUTED option, with the measurement
#: that refuted it, rather than as a recommended setting.
#:
#: The idea was the physics ladder: warm-start v2 into a 1.0 rad box so the
#: same +-1 action buys 3x the resolution near the target. MEASURED FIRST
#: (`scratchpad/probe_box_reach2.py`, the coordinate-descent solver from the
#: probe that chose 3.0, 40 draws from this env's own shell):
#:
#:     box            residual median   p90      <=2cm   the 8 eval targets
#:     FULL range          0.14 cm    3.67 cm     72%    -
#:     HOME +- 3.0         0.15 cm    3.15 cm     78%    8/8 inside 0.3 mm
#:     HOME +- 2.0         2.30 cm    6.91 cm     48%    -
#:     HOME +- 1.0        12.96 cm   24.96 cm      0%    0/8 (5.0-27.8 cm)
#:
#: HOME parks the arm folded at joint1 = +1.445 rad pointing 83 deg to the
#: robot's LEFT while the task samples the front arc, so a 1.0 rad box cannot
#: get the gripper to a single one of the eight targets the A/B scores. Its
#: run would have reported ~13 cm and said nothing about the hold: a
#: structural null, expected rather than informative, and the reason this
#: phase spent its third run on a second training SEED instead. Narrowing the
#: box only helps if its centre follows the arm — which is `"delta"`.
RUNG_SCALE_RAD = 1.0

# ------------------------------------------- the A/B, and what it decided
#
# **VERDICT (Phase 4a-2, 2026-09-18).** Matched runs: 1.5 M steps, 8 envs,
# seed 0, `--robot mars --task reach`, nothing else changed. Scored on the
# DETERMINISTIC export over 8 seeds x 8 s with `scripts/probe_mars_reach.py`,
# which builds its env from the run's own map because the same six floats
# mean an increment under one and a position under another.
#
#                              final    best    <=2cm  held  sat   tail   hits
#     null (zero action)       30.2 cm  30.1 cm  0/8   0/8    0%   0.0 mm  0/8
#     absolute (4a v2)          2.02 cm  0.75 cm 3/8   0/8   58%  11.8 mm  0/8
#     cubic                     7.62 cm  5.33 cm 0/8   0/8   58%  22.9 mm  2/8
#     delta, train seed 0       1.13 cm  0.42 cm 5/8   5/8   58%   5.3 mm  0/8
#     delta, train seed 1       2.38 cm  1.23 cm 3/8   2/8   55%   7.0 mm  0/8
#
# **`"delta"` wins and is the default.** It is the first map under which the
# task's own success rule fires at all — 0/8 held becomes 5/8 and 2/8 — and
# the contact sheets are qualitatively different, not just better: v2 crossed
# the 2 cm ball for 1-5 control steps at a time on a ~2 s cycle, while the
# delta sheet sits at 0.5-0.7 cm from t = 2 s to the end of the episode with
# the near-streak climbing past 150 of the 25 it needs. `at_target` doubles
# (+110 -> +212) because holding is what that term pays for. The tail spread
# halves, 11.8 -> 5.3 mm.
#
# The 8/8 BAR IS STILL OPEN, and two honest caveats come with the win:
#
#   * **Training-seed variance is large here.** Seed 1 of the same recipe
#     holds 2/8 against seed 0's 5/8 — wider than the 8-seed eval spread,
#     which is `AGENTS.md`'s "eval seeds do not measure training runs" landing
#     exactly as advertised. So "delta holds about half the seeds" is the
#     claim, and any future change to this task needs two training seeds
#     before it is credited. `absolute` has only ONE training seed here (4a's
#     v2), so the comparison rests on delta's 0/8 -> {5,2}/8 being a
#     structural change rather than on a paired statistic: an absolute map's
#     0 is not variance, it is the absence of a fixed point.
#   * **What still misses is not the hold.** On seed 0's run the three failures
#     are the two shell EDGES and one drift: seed 4's target sits at 0.386 m
#     of the 0.40 m shell AND 0.343 m of its 0.35 m ceiling (the arm reaches
#     it stretched out and stalls 2.7-3.4 cm short, though the solver gets
#     within 3 mm, so it is the policy and not the geometry); seed 1 is at
#     +54 deg of the +-60 deg arc; seed 6 reaches 0.2 cm and then drifts out
#     to 2.1 cm. Two edges and a drift is a PHYSICS-LADDER shape — ladder the
#     shell, which `behaviors/mars_tasks.py` already names as the rung to add
#     — and not a reward one.
#
# **`"cubic"` LOSES, and it loses for a reason worth keeping.** It is the
# worst of the three: 0/8 inside 2 cm, 7.62 cm median, and the only variant
# that self-collides (2/8). Its sheet shows it PARKING — 2.9-3.2 cm for six
# straight seconds, stable and wrong — so the map did buy a settled pose; it
# just cannot get the pose close. The mechanism is the same saturation
# statistic read the other way: `|a|^3` means an action must run to the box
# edge to travel at all (|0.5|^3 * 3.0 rad is 0.375 rad, an eighth of what
# `absolute` gives at the same output), so the policy lives at |a| ~ 1 — where
# the cubic slope is 9.0 rad per unit action against `absolute`'s 3.0. It sold
# the interior resolution it was bought for in exchange for THREE TIMES the
# coarseness where the policy actually operates. `at_target` collapses to
# +5.9. The lesson generalises past this arm: stretching an action map's
# interior only helps if the policy's operating point is IN the interior, and
# 58% of dims on the box edge was the measurement saying it is not.
#
# **What the saturation number stopped being.** All three maps sit at 55-58%
# of arm dims on the box edge, and delta holds anyway — so edge-sitting is
# not itself the fault. Under `delta` a saturated action means "slew at the
# fastest legal rate", which is the right command while travelling and simply
# is not what the policy emits once it arrives. Phase 4a read 58% as the
# mechanism; it was a SYMPTOM of the absent fixed point, and the fixed point
# was the mechanism. (`AGENTS.md`'s "KL was a symptom, not the cause", again.)
#
# The trained per-dim action std says the same thing from the optimizer's
# side. `train.LOG_STD_MAX` caps std at 0.6065, and under `absolute` it BINDS
# on 6 of 8 dims (v2: 0.584-0.612) and under `cubic` on 4 of 8 — but `delta`
# pulls the three shoulder dims down on its own, to 0.451 / 0.530 / 0.489.
# Only under delta does noise cost anything: it integrates into target drift,
# so there is gradient pressure to be quiet near the target. Under an
# absolute map noise is re-decided every step and costs nothing to hold.

#: The reach task's target shell, in the BASE frame, relative to the SHOULDER
#: (joint1's anchor, measured off the model at HOME — never hand-carried).
#: The plan's numbers, kept because the measurement says the arm owns them:
#: 0.15-0.40 m of the robot's 0.40 m reach, the front arc, and floor to
#: chassis-top in height.
REACH_RADIUS_M = (0.15, 0.40)
REACH_YAW_RAD = (-math.pi / 3.0, math.pi / 3.0)     # +-60 deg of the base's +x
REACH_HEIGHT_M = (0.05, 0.35)
#: A target whose radius is almost all vertical leaves no horizontal offset to
#: aim at, so it is redrawn rather than collapsed onto the shoulder's axis.
REACH_MIN_HORIZ_M = 0.02

#: Success: the effector inside this ball of the target...
SUCCESS_RADIUS_M = 0.02
#: ...for the last second of the episode. A HOLD, not a fly-by: a reach that
#: touches 2 cm on its way past has not done the task, and the proximity
#: bonus below is per-step for the same reason.
SUCCESS_HOLD_S = 1.0
EPISODE_S = 8.0

# ---------------------------------------------------------------- the reward
#
# Progress-pay, per AGENTS.md: "Dense, bounded shaping beats sparse
# windfalls", and a stalled arm must earn NOTHING. So the earner is the CHANGE
# in distance (a telescoping sum: an episode's total is `d_start - d_end`
# whatever route it took, so there is no way to farm it by oscillating) plus a
# bounded per-step bonus for being there.
#: Per metre closed. 40 x the ~0.25 m a typical episode has to close is ~10
#: reward, which is the scale the penalties below are sized against.
W_PROGRESS = 40.0
#: Per step inside the ball: `W_NEAR * exp(-(d/NEAR_SIGMA)^2)`. Smooth, so
#: the last centimetre has a gradient, and effectively confined to the
#: success radius (it pays 0.37 of itself at 2 cm, 0.018 at 4 cm, 1e-4 at
#: 6 cm). Per step, so HOLDING pays: 175 held steps of an 8 s episode are
#: worth ~350, which is what makes an early termination expensive without
#: any explicit penalty having to price it.
W_NEAR = 2.0
NEAR_SIGMA_M = SUCCESS_RADIUS_M
#: Penalties, sized so the task term dominates where the policy STARTS —
#: AGENTS.md's "a ramped penalty can eat its task", which cost the G1 idle a
#: face-plant at ep_rew -92. Neither of these ramps. MEASURED per episode
#: (`scratchpad/probe_random.py`, 60 episodes of a uniform random policy, and
#: the trained v2 export over 8 seeds):
#:
#:                     random      trained
#:     reach_progress   -0.29       +11.16
#:     at_target        +0.00      +109.82
#:     action_rate      -0.72        -3.95
#:     joint_vel        -0.67        -0.80
#:
#: So the two together are 12% of the ~+11.6 a full 0.29 m reach pays and
#: 4% of what a held one earns — present enough to price a thrashing arm,
#: nowhere near enough to make standing still the better deal.
ACTION_RATE_W = 0.002
JOINT_VEL_W = 0.0005
#: Per unit of `sum(action^2)`. The term Phase 4a's FIRST run measured its way
#: into, and it is a control-authority fix rather than a reward preference.
#:
#: MEASURED on `mars-reach-v1` (1.5M steps, the deterministic export over 8
#: seeds): the arm reaches — best distance in an episode 1.1 cm median — and
#: then HOVERS, ending at 2.6 cm median with 0 of 8 seeds holding the 2 cm
#: ball for the required second. The mechanism is on the same line of the
#: probe: **64% of the six arm-action dims sit on the box edge**, the exported
#: mean's magnitude reaches **|a| = 90 inside a +-1 box**, and the effector
#: describes a 9.5 mm limit cycle 3.0 cm from the target. A policy whose mean
#: is saturated has only bang-bang authority: with the target rate-limited it
#: can slew up or slew down and it has no "stay", so it chatters across the
#: ball instead of settling in it.
#:
#: Nothing priced the MAGNITUDE. `ACTION_RATE_W` prices the raw output's
#: change, which is walk_env's rule and right — but once the mean saturates,
#: a 1.4% relative wobble on a magnitude of 87 is a 120% swing in what the
#: arm is ACTUALLY commanded, so the penalty decouples from the behaviour it
#: was meant to price. Pricing the clipped action instead would not fix it
#: either: a constant saturated action has zero rate, so the arm would simply
#: park on a joint stop.
#:
#: A quadratic cost against a bounded benefit puts the optimum at a modest
#: |a|, which is the whole point — it buys back the interior of the box, where
#: "hold still" is expressible. Sized so it is negligible where the task
#: needs full authority and prohibitive where the mean ran to: at |a| = 1 on
#: every dim it costs -0.016/step (-3.2 an episode, 3.6% of the +88 `at_target`
#: this policy already earns), and at |a| = 87 it costs -121/step.
#:
#: **VERDICT, and it is only half a win.** `mars-reach-v2`, the identical
#: recipe plus this term: it DID stop the runaway (the trained magnitude is
#: now |a| ~ 0.84 a dim against 87), and every accuracy number improved —
#: final distance 2.62 -> 2.02 cm median, best-in-episode 1.10 -> 0.75 cm,
#: seeds ending inside 2 cm 1/8 -> 3/8. It did NOT fix the hold: still 0/8 on
#: the task's own rule, with 58% of dims on the box edge (from 64%) and the
#: limit cycle unchanged at 11.8 mm. So the magnitude was A cause and not THE
#: cause, and 4b needs resolution near the target rather than a smaller mean —
#: `docs/mars-roadmap.md` Phase 4a names the three candidates.
ACTION_MAG_W = 0.002
#: How far two of MARS's own bodies must OVERLAP before it counts as the arm
#: hitting itself. Not zero, and the reason is the URDF's, not the policy's.
#:
#: mars.urdf's collision volumes are hand-measured boxes over a printed
#: chassis and five arm links, and they are KNOWN to overlap in poses the real
#: arm allows: `mars.JOINT2_GUARD_MIN` exists because "the simplified
#: collision boxes still overlap ~9 mm there", which is Innate's own note
#: about their own model. So a predicate of "any contact" terminates on the
#: model's error as readily as on the robot's mistake.
#:
#: MEASURED (`scratchpad/probe_depth.py`, 3000 poses a row): HOME itself is
#: clean, and so is everything within +-0.05 rad of it. At +-0.10 rad **15.3%
#: of poses report a contact and 0.0% of them are deeper than 5 mm** — every
#: one is `link1 <-> link3` or `link2 <-> link4`, links two apart in a chain
#: folded tight at HOME, at 2.2 mm median depth. Drive the arm properly into
#: the chassis and the overlap is 20-60 mm. So there are two populations, they
#: are two orders of magnitude apart, and 10 mm — the documented artifact,
#: rounded up — separates them: it fires on nothing within a tenth of a radian
#: of the pose every episode starts in, which is AGENTS.md's "check the term
#: is not flat where the policy starts" asked of a terminal.
SELF_COLLISION_DEPTH_M = 0.010
#: The arm hitting itself ends the episode. The number only has to MARK the
#: event: termination already forfeits the rest of the hold bonus (up to
#: ~350), so the forfeited future is the real price and a large explicit
#: penalty would only teach the policy to stop moving. <= 0 by construction,
#: and it rides in a `*_penalty` key so `train._penalty_sign_callback_cls`
#: watches it.
SELF_COLLISION_PENALTY = -2.0

# ============================================================= the pick task
#
# Phase 4b. `reach` puts the gripper on a point that is not in the physics;
# `pick` puts a BODY in the scene and asks for it off the floor. Everything
# below was measured before a weight was chosen — the scene, the timestep,
# the holding predicate — because a grasp is the first thing this body does
# that the contact solver has an opinion about.

#: What `pick` puts on the floor: the PLAYROOM's own toy, by name.
#:
#: `world/scenario.PICKABLE_KINDS["block"]` is a 4 cm cube of 20 g, and it is
#: imported rather than re-typed so that Phase 5's `TidyArm` picks up the
#: object this policy was trained on. A 4 cm block re-declared here at 4.1 cm
#: would be a policy trained on a thing the room does not contain.
PICK_TOY_KIND = "block"
#: The body and free joint the toy is attached under, in the pick scene.
PICK_TOY_BODY = "toy"
PICK_TOY_JOINT = "toy_free"
PICK_SCENE_NAME = "scene_mars_pick.xml"
#: Where the toy sits in the model's own qpos0 and in the HOME keyframe: well
#: outside the shell, on the floor. NOT the origin, which is INSIDE the
#: chassis — a keyframe has to be a legal state, and a model whose rest pose
#: buries the toy in the robot makes every pose the IK solver tries look like
#: a collision. `reset` writes the sampled spawn over it immediately.
PICK_TOY_PARK = (1.5, 1.5)

#: Where the toy's CENTRE sits at rest: half its height, plus the 1 mm
#: `world/compose.py` floats every pickable by so it starts out of contact.
#: Derived at import from the toy's own size for the reason above.
PICK_FLOAT_M = 0.001

#: The spawn ladder — the PHYSICS curriculum, and the only thing that changes
#: between rungs (AGENTS.md: "curriculum stages may ladder only physics,
#: spawns and strictness, never the reward").
#:
#:   0. the DRILL: the arm spawns with its jaws already open around the block,
#:      so the only thing left to do is close and lift,
#:   1. a 6 x 6 cm box at one spot that a scripted pick demonstrably grasps,
#:   2. 15 x 15 cm about the same spot,
#:   3. the reach shell's whole floor footprint (+-60 deg, 0.15-0.40 m).
#:
#: Rungs 1 and 2 are squares in the BASE frame centred on `PICK_SPOT`, which
#: is itself a (radius, yaw) of the reach shell rather than a point in space —
#: so a URDF revision that moves the arm mount moves the box with it, exactly
#: as `sample_target` moves the reach shell. Draws that fall outside the shell
#: footprint are redrawn, so every rung spawns inside ONE footprint and the
#: "knocked out of the shell" terminal means the same thing on all three.
#:
#: **Rung 0 exists because rung 1 from scratch does not grasp, MEASURED.**
#: 1.5 M steps at rung 1, 8 envs, seed 0: `at_target` climbs to +223 of the
#: +267 a scripted success earns and `lift_progress` and `held_high` stay at
#: 0.000 for the whole run — the policy parks the (shut) claw on the block and
#: collects the proximity income forever. That is `AGENTS.md`'s "does ANY
#: rollout ever do the thing?" answered NO, and the rule is then explicit:
#: ladder the physics, do not re-price the reward. A drill rung that spawns
#: the arm in the grasp-ready pose makes the close-and-lift samplable from
#: step one, and rung 1 warm-starts from it — the same shape as the
#: headstand's `xml` drill stage and of `inverted_spawn_prob` generally.
PICK_RUNGS: tuple[int, ...] = (0, 1, 2, 3)
DEFAULT_PICK_RUNG = 1
#: The arm pose rung 0 spawns in: the jaws open at `OPEN_RAD` around a block
#: resting at `PICK_SPOT`. Solved by the same coordinate descent that chose
#: `ACTION_SCALE_RAD` (`scripts/probe_mars_pick.py --scripted-env` prints it)
#: and PINNED, so the drill costs no solver and a URDF revision that moves the
#: arm fails a test rather than quietly drilling a different pose. MEASURED
#: held through the env's own `delta` map: a 9.0 cm lift, `gripper_load`
#: +1.997, the task's own `success` rule satisfied.
PICK_GRASP_POSE = (-0.0539, 0.5750, -0.1198, 0.6859, -0.3333, 0.6000)
#: (radius from the shoulder, yaw) of rungs 1 and 2's centre. MEASURED: this
#: is the spot the scripted pick holds at both 5 ms (2/8 overall) and 2 ms,
#: so rung 1 is a box around a grasp that is known to exist.
PICK_SPOT = (0.25, 0.0)
#: Half-width of each rung's square, m.
PICK_RUNG_HALF_M: dict[int, float] = {1: 0.03, 2: 0.075}

#: How far outside the shell footprint the toy may be nudged before the
#: episode ends as "knocked away".
#:
#: Not zero: a block shoved 2 cm by a clumsy approach is still pickable, and
#: terminating on it would charge the policy for the contact it has to make.
#: Not large either — the numbers that matter are the failures, and MEASURED
#: at 5 ms an ejected block travels 0.24-7.9 m, three orders of magnitude past
#: this. The margin is generous precisely because the two populations are.
SHELL_EXIT_MARGIN_M = 0.05
SHELL_EXIT_MARGIN_RAD = math.radians(15.0)

#: |`qfrc_constraint` at joint6| above which the claw is HOLDING something.
#:
#: **What it actually separates is EMPTY from LOADED**, and the margin there
#: is enormous. MEASURED (`robots/mars.OBS_GRIPPER_LOAD`'s table) across every
#: empty state there is — open, shut on air at the hard stop, and the shut
#: claw driven 0.25 rad down into the FLOOR — the slot reads **0.0000**, to
#: four decimals, in all of them. With the block in the claw it reads
#: 1.89-2.03: the servo's own 2 N*m ceiling fed back through the blade.
#:
#: It does NOT separate "gripped" from "resting against". That reading is
#: pose-dependent — 0.0035 where the open jaws merely touch the block and
#: 1.82 where the approach has pressed them onto it — and it is the right
#: answer either way, because a blade carrying 1.8 N*m of an object's reaction
#: IS loaded by it. So the threshold sits at 1.0 to be two orders clear of the
#: empty population and a factor of ~2 below the loaded one, and what the
#: predicate promises is "something is in the claw", not "the grasp is good".
#:
#: The finger<->toy CONTACT conjunct adds no discrimination today (every empty
#: case reads exactly 0), and it is there to name the OBJECT: "holding" must
#: mean holding the thing the task is about, and Phase 5's room has a basket
#: and several toys in it.
#:
#: Phase 5 moved the FLOAT to `mars.HOLD_LOAD_NM` for the same reason
#: `PICK_PHYSICS_DT` moved: `robots/mars_drive.MarsDriver.held_body` fills
#: `Senses.holding` for a MARS in a room off the identical predicate, and the
#: driver cannot import this module. Same value, one definition.
HOLD_LOAD_NM = mars.HOLD_LOAD_NM

#: How far the toy must rise ABOVE ITS RESTING HEIGHT to count as lifted.
#: "Lift 5 cm" read literally, so the number is the lift and not a height
#: above the floor that the toy already has 2.1 cm of for free.
SUCCESS_LIFT_M = 0.05
#: ...and held there for the last second of the episode, exactly as `reach`
#: asks for its ball. A pick that drops the block at t = 7.9 s has not done
#: the task.

# ------------------------------------------------------------ pick's reward
#
# Same shape as `reach`'s and for the same reason: PROGRESS pays, presence
# does not. Two progress terms, because a pick is two movements.
#
#: Per metre of held HEIGHT gained. Sized against `at_target` below, which is
#: the income a policy that merely parks its claw on the block collects: over
#: a 200-step episode that is ~300, so the lift terms have to beat it. They
#: do — `held_high` alone pays 4.0 a step, so lifting by t = 3 s is worth
#: ~500 — and this term is the GRADIENT that gets there (a 0.1 m/s lift is
#: +0.4 a control step, the same scale as the proximity bonus).
W_LIFT = 100.0
#: Per step while HOLDING the block above `SUCCESS_LIFT_M`. Bounded and
#: per-step, so an early termination forfeits it and the reward for doing the
#: task is the reward for KEEPING doing it.
W_HELD = 4.0
#: The toy knocked out of the shell ends the episode. Like the self-collision
#: penalty this number only has to MARK the event — what it really costs is
#: the rest of the episode's hold bonus — and it rides in a `*_penalty` key
#: so `train._penalty_sign_callback_cls` watches its sign.
BLOCK_LOST_PENALTY = -2.0

#: What tasks this env implements. `MarsBody.env_class` is the authority a
#: caller should ask; this is what its refusal names.
TASKS: tuple[str, ...] = ("reach", "pick")
#: The tasks whose base pair is forced to zero. BOTH of them, and for one
#: reason: they are ARM tasks, and a policy allowed to drive would solve
#: either by rolling the whole robot at the thing — the cheapest behaviour
#: that satisfies the terms and not the one being taught. The contract keeps
#: its 8 actions either way (a fixed layout, zero-padded), and `info`
#: reports that the pair was read and ignored.
ARM_ONLY_TASKS: tuple[str, ...] = ("reach", "pick")


def toy_spec() -> tuple[tuple[float, float, float], float, tuple[float, ...]]:
    """(full extents, mass, rgba) of the toy `pick` uses — the playroom's.

    A LAZY import of `world/scenario.py`: `world` imports `robots` (the G1's
    spec, MARS's own `attach`), so a module-level import here would be a
    cycle. It is the same lazy-import rule `MarsBody.env_class` follows.
    """
    from ..world.scenario import PICKABLE_KINDS

    k = PICKABLE_KINDS[PICK_TOY_KIND]
    return tuple(k["size"]), float(k["mass"]), tuple(k["rgba"])


def toy_rest_z() -> float:
    """The toy's centre height at rest, off its own size."""
    return toy_spec()[0][2] / 2.0 + PICK_FLOAT_M


def pick_scene_spec() -> mujoco.MjSpec:
    """MARS's standalone scene at 2 ms, plus one free-jointed toy.

    The toy's geom is `world/compose.py`'s, parameter for parameter — the
    same `priority=1` and `friction=[0.8, 0.005, 0.0001]`, whose reason is
    written out there (at equal priority MuJoCo takes the element-wise max
    and the floor's 1.0 sliding would win, making the toy's 0.8 inert). A
    pick trained against a different friction than the room's would be a
    policy for a block that does not exist.

    The HOME keyframe has to be WIDENED, not replaced: `mars._scene_spec`
    sizes it from a compile of the robot alone, and a free joint adds seven
    qpos. The toy's keyframe pose is its park position; `reset` writes the
    sampled one.
    """
    size, mass, rgba = toy_spec()
    park = [PICK_TOY_PARK[0], PICK_TOY_PARK[1], toy_rest_z()]
    spec = mars.scene_spec(timestep=PICK_PHYSICS_DT)
    body = spec.worldbody.add_body(name=PICK_TOY_BODY, pos=park)
    body.add_freejoint(name=PICK_TOY_JOINT)
    body.add_geom(name=f"{PICK_TOY_BODY}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[v / 2 for v in size], mass=mass, rgba=list(rgba),
                  priority=1, friction=[0.8, 0.005, 0.0001])
    key = spec.key(mars.HOME_KEY)
    key.qpos = np.concatenate([np.asarray(key.qpos),
                               park, [1.0, 0.0, 0.0, 0.0]])
    key.qvel = np.zeros(len(np.asarray(key.qvel)) + 6)
    return spec


def pick_scene_xml() -> Path:
    """Path to the generated pick scene, written beside the assets.

    Same properties as `mars.scene_xml`: atomic, content-addressed, and next
    to `meshes/` so the STL paths resolve.
    """
    mars.require_mars()
    return mars.write_scene_xml(pick_scene_spec(),
                                mars.CACHE_DIR / PICK_SCENE_NAME)

#: The map `reach` trains under unless a caller says otherwise.
#:
#: `"delta"` since Phase 4a-2 (was `"absolute"`, Phase 4a): it is the only map
#: measured to hold the target at all — 5/8 and 2/8 seeds on two training
#: seeds against 0/8 for both absolute maps. The full A/B table and both
#: caveats are in the verdict block under `ACTION_MODES`. Changing this
#: changes what every float a MARS policy emits MEANS, so it also changes
#: `MarsBody.contract().deploy` — which is why that sentence is built from
#: this constant rather than typed.
DEFAULT_ACTION_MODE = "delta"


class MarsArmEnv(gym.Env):
    """Reach a sampled point, or pick a block up, with MARS's arm.

    One compiled MARS scene, a private `MjData`, one `MarsDriver`, and the
    32-float contract as its observation. See the module docstring for the
    measurements every constant rests on.

    **Two tasks, one observation layout, and the task decides the SCENE.**
    `reach` compiles the standalone scene at the world's 5 ms;  `pick`
    compiles its own at `PICK_PHYSICS_DT` with a toy in it, because a grasp
    is ejected at 5 ms (2/8 spots) and held at 2 ms (7/8). Both step at
    `CTRL_DT`, which is the contract's 25 Hz, so the policy's clock is the
    same and only the integration underneath differs.

    **The model is `mars.model()` — shared, read-only, one compile per
    process.** Nothing in this env writes to an `MjModel`: v1 randomises the
    TARGET and nothing about the robot, so there is no mass, friction or
    armature to restore and no `walk_env.shared_model` bookkeeping to do. That
    is why sharing is safe here where `MicroduckWalkEnv` has to refuse it
    under BAM (the BAM actuator rewrites `dof_frictionloss` every substep, so
    siblings in one process would overwrite each other's servo physics).
    Sharing buys ~0.2 s of STL parsing per env, which is 6 s of a 32-env
    fleet's startup; the fork-based vec env compiles once per worker anyway.
    `own_model=True` compiles a private one for a caller that ever needs to
    mutate the model — Phase 4b's `pick` will, once there is a block in the
    scene whose mass is worth randomising.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "reach",
        *,
        max_episode_s: float = EPISODE_S,
        seed: int | None = None,
        # --- the trainer's shared knobs (train.env_kwargs_from_args) -------
        # Accepted so `train-walk --robot mars` works with the flags every
        # body's trainer already passes. The ones with no meaning here are
        # ignored rather than silently reinterpreted, each with its reason:
        domain_rand: bool = True,
        obs_noise: bool = True,
        action_delay: bool = False,
        random_yaw: bool = False,
        push_robot: bool = False,
        actuator: str | None = None,
        actuator_force: str | None = None,
        # --- this env's own ------------------------------------------------
        own_model: bool = False,
        spawn_yaw: float = 0.0,
        use_base: bool | None = None,
        action_mode: str = DEFAULT_ACTION_MODE,
        action_scale_rad: float | None = None,
        pick_rung: int = DEFAULT_PICK_RUNG,
    ):
        if task not in TASKS:
            raise SystemExit(
                f"unknown --task {task!r} for mars (have: {', '.join(TASKS)}) — "
                "`place` is a later rung of docs/mars-roadmap.md Phase 4")
        if int(pick_rung) not in PICK_RUNGS:
            raise SystemExit(
                f"pick_rung={pick_rung!r} is not a rung of the pick ladder "
                f"({', '.join(str(r) for r in PICK_RUNGS)}) — a spawn box "
                "nobody measured is not a curriculum")
        if action_mode not in ACTION_MODES:
            raise SystemExit(
                f"unknown action_mode {action_mode!r} for mars (have: "
                f"{', '.join(ACTION_MODES)}) — what an action MEANS is this "
                "env's half of the contract, so a name it does not implement "
                "is refused rather than quietly defaulted")
        self.action_mode = str(action_mode)
        # The box's WIDTH in rad, orthogonal to the MAP above. `"delta"` has no
        # use for it — its step is `DELTA_RAD_PER_STEP` and its box is not
        # centred on HOME at all — so passing one is REFUSED rather than
        # ignored: a knob that changes nothing is broken, not null (AGENTS.md
        # rule 0), and accepting a width here would let a caller believe it had
        # narrowed a box it had not.
        if action_scale_rad is not None and self.action_mode == "delta":
            raise SystemExit(
                "action_scale_rad has no meaning in action_mode='delta': the "
                "step is DELTA_RAD_PER_STEP, derived from "
                "MAX_TARGET_RATE_RAD_S. Narrow a box with "
                "action_mode='absolute' or 'cubic'")
        self.action_scale_rad = float(ACTION_SCALE_RAD if action_scale_rad
                                      is None else action_scale_rad)
        if self.action_scale_rad <= 0.0:
            raise SystemExit("action_scale_rad must be positive rad, got "
                             f"{self.action_scale_rad}")
        for name, value in (("actuator", actuator),
                            ("actuator_force", actuator_force)):
            if value is not None and str(value).lower() == "bam":
                raise ValueError(
                    f"{name}='bam' is this repo's XL330 identification — it "
                    "describes MARS's joints 4-6 and head and NOT its "
                    "XL430/XC430 shoulder (joints 1-3). v1 drives every joint "
                    "with Innate's own position PD (mars.arm_servo); "
                    "docs/mars-roadmap.md §6.6 is the per-joint fix")
        self.task = task
        # Recorded, not applied. `obs_noise` has no v1 meaning: nothing here
        # has measured an encoder sigma for these servos or a detector sigma
        # for the head camera, and inventing one would be the "model both
        # halves of an error" mistake with no number behind it. `action_delay`
        # and `push_robot` are the walking env's DR, off in every task recipe.
        # `domain_rand` has nothing to randomise while DR v1 is the target
        # alone — which is sampled on every reset either way.
        self.obs_noise = bool(obs_noise)
        self.action_delay = bool(action_delay)
        self.push_robot = bool(push_robot)
        self.domain_rand = bool(domain_rand)
        self.random_yaw = bool(random_yaw)
        self.spawn_yaw = float(spawn_yaw)
        self.pick_rung = int(pick_rung)
        # The base is DISABLED for every arm task (`ARM_ONLY_TASKS`): letting
        # the policy drive would let it solve a reach — or a pick — by rolling
        # the whole robot at the thing, the cheapest behaviour that satisfies
        # the terms and not the one being taught. The contract keeps its 8
        # actions (a fixed layout, zero-padded, exactly as the duck's 61
        # floats are), so the base pair is read, forced to zero, and reported
        # in `info`.
        self.use_base = ((task not in ARM_ONLY_TASKS) if use_base is None
                         else bool(use_base))

        # `pick` always gets a PRIVATE model: its scene has a free body in it
        # whose mass is the first thing about this robot worth randomising,
        # and a shared `MjModel` is the one thing `walk_env.shared_model` has
        # already paid to learn about. v1 writes nothing to the model, so the
        # cost is the honest part of this: MEASURED 0.20 s of STL parsing per
        # env, 1.6 s once for an 8-env fleet, against a 7-minute run.
        if task == "pick":
            self.model = mujoco.MjModel.from_xml_path(str(pick_scene_xml()))
            self.decimation = PICK_DECIMATION
        else:
            self.model = (mujoco.MjModel.from_xml_path(str(mars.scene_xml()))
                          if own_model else mars.model())
            self.decimation = DECIMATION
        self.own_model = bool(own_model) or task == "pick"
        self.data = mujoco.MjData(self.model)
        self.driver = MarsDriver(self.model, "")
        self.dt = CTRL_DT
        self.max_steps = int(round(float(max_episode_s) / CTRL_DT))
        self.hold_steps = int(round(SUCCESS_HOLD_S / CTRL_DT))

        m = self.model
        self._arm_qadr = np.array([m.joint(j).qposadr[0]
                                   for j in mars.ARM_JOINTS], int)
        self._arm_dadr = np.array([m.joint(j).dofadr[0]
                                   for j in mars.ARM_JOINTS], int)
        self._head_qadr = int(m.joint(mars.HEAD_JOINT).qposadr[0])
        self._grip_dadr = int(m.joint("joint6").dofadr[0])
        self._ee_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                          mars.EFFECTOR_BODY)
        if self._ee_body < 0:                       # never id -1, per spec.py
            raise KeyError(f"no body {mars.EFFECTOR_BODY!r} in the MARS model "
                           "— the reach reward has nothing to measure")
        self._home = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])
        self._jnt_lo = np.array([m.joint(j).range[0] for j in mars.ARM_JOINTS])
        self._jnt_hi = np.array([m.joint(j).range[1] for j in mars.ARM_JOINTS])
        #: rad the commanded target may move in one control step.
        self.target_step_rad = MAX_TARGET_RATE_RAD_S * CTRL_DT
        self._cmd_target = self._home.copy()

        # Every body of THIS robot, for the self-collision scan: the SUBTREE
        # hanging off `base_link`, walked through `body_parentid`.
        #
        # It used to be "every body that is not the world", and Phase 4b is
        # the task that breaks that: a free-jointed toy is not the world
        # either, so a grasp would have read as the arm hitting itself and
        # ended the episode at the moment of success. The subtree is what the
        # set always MEANT, and it is also what a composed `/sim` world would
        # need. For `reach` it is provably the same set — the scene holds
        # nothing but MARS and the floor — which a test pins.
        self._robot_bodies = self._subtree(m, mars.BASE_BODY)
        self._finger_bodies = frozenset(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in mars.FINGER_LINKS)
        # The blade PADS — the inner faces, named `link6*_pad1..4` in
        # mars.urdf — for `grasp_point`. Resolved by name suffix and asserted
        # non-empty, because a URDF revision that renames them would silently
        # leave the pinch point at the origin.
        self._pad_geoms = tuple(
            np.array([g for g in range(m.ngeom)
                      if m.geom_bodyid[g] == mujoco.mj_name2id(
                          m, mujoco.mjtObj.mjOBJ_BODY, link)
                      and "_pad" in (mujoco.mj_id2name(
                          m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")], int)
            for link in mars.FINGER_LINKS)
        if any(len(g) == 0 for g in self._pad_geoms):
            raise KeyError(
                f"no `_pad` geoms on {list(mars.FINGER_LINKS)} in mars.urdf — "
                "the blade pads are where a grasp happens; check ASSETS' "
                "revision")

        # `pick`'s toy: the free body, its qpos/qvel rows, and its geom.
        self._toy_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           PICK_TOY_BODY)
        self._mimic_qadr = int(m.joint(mars.MIMIC_JOINT[0]).qposadr[0])
        if task == "pick" and self._toy_body < 0:
            raise KeyError(f"no body {PICK_TOY_BODY!r} in the pick scene — "
                           "there is nothing to pick up")
        if self._toy_body >= 0:
            self._toy_qadr = int(m.joint(PICK_TOY_JOINT).qposadr[0])
            self._toy_dadr = int(m.joint(PICK_TOY_JOINT).dofadr[0])
            self.toy_rest_z = toy_rest_z()
        self._rung_half = PICK_RUNG_HALF_M.get(self.pick_rung)

        # The shoulder, in the BASE frame, measured off the model at HOME.
        # The task's shell is defined from here, so a URDF revision that moves
        # the arm mount moves the shell with it instead of aiming the task at
        # a point in space the arm no longer starts near.
        probe = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, probe, m.key(mars.HOME_KEY).id)
        mujoco.mj_forward(m, probe)
        self.shoulder_base = np.array(probe.xanchor[m.joint("joint1").id]).copy()

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (mars.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(
            -ACTION_CLIP, ACTION_CLIP, (mars.NUM_ACTIONS,), np.float32)

        # The published layout, checked against this builder. `_SLOTS` is the
        # order `_get_obs` writes in; the contract is what every reader of an
        # exported file resolves. Two tables that must agree is exactly the
        # duplication that drifts, so they are compared here rather than in a
        # test alone — a mismatch is a construction error on the first env.
        self._check_contract()

        self._rng = np.random.default_rng(seed)
        self.target = np.zeros(3)
        self.target_base = np.zeros(3)
        self.target_base_sample = np.zeros(3)
        self.last_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.reward_sums: dict[str, float] = {}
        self.step_count = 0
        self._near_streak = 0
        self._prev_dist = 0.0
        self._prev_lift = 0.0
        self._hold_steps_seen = 0
        self._grasps = 0
        self._was_holding = False
        self._lost = False
        self._held = False
        self._lift = 0.0
        self._self_hit: tuple[str, str] | None = None
        self.reset(seed=seed)

    @staticmethod
    def _subtree(m: mujoco.MjModel, root: str) -> frozenset[int]:
        """Every body id at or under `root`, by `body_parentid`."""
        rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
        if rid < 0:
            raise KeyError(f"no body {root!r} in the MARS model")
        out = {rid}
        for b in range(rid + 1, m.nbody):      # children always follow parents
            if int(m.body_parentid[b]) in out:
                out.add(b)
        return frozenset(out)

    # ------------------------------------------------------------- contract

    #: (contract slot name, slice) in the order `_get_obs` fills them.
    _SLOTS: tuple[tuple[str, slice], ...] = (
        ("arm_qpos", mars.OBS_ARM_QPOS),
        ("arm_qvel", mars.OBS_ARM_QVEL),
        ("head_pitch", mars.OBS_HEAD_PITCH),
        ("gripper_load", mars.OBS_GRIPPER_LOAD),
        ("last_action", mars.OBS_LAST_ACTION),
        ("target_base", mars.OBS_TARGET_BASE),
        ("target_seen", mars.OBS_TARGET_SEEN),
        ("base_twist", mars.OBS_BASE_TWIST),
        ("reserved", mars.OBS_RESERVED),
    )

    def _check_contract(self) -> None:
        """This builder tiles `MarsBody.contract().slots`, or nothing runs."""
        want = mars.MARS.contract()
        mine = tuple((n, s.start, s.stop) for n, s in self._SLOTS)
        theirs = tuple((s.name, s.start, s.stop) for s in want.slots)
        if mine != theirs:
            raise ValueError(
                f"MarsArmEnv fills {mine} but {want.id} declares {theirs} — "
                "the contract is what every reader of an exported policy "
                "resolves, so the env may not have its own layout")
        if want.obs_dim != mars.OBS_DIM or want.act_dim != mars.NUM_ACTIONS:
            raise ValueError(
                f"{want.id} is {want.obs_dim}/{want.act_dim}, this env is "
                f"{mars.OBS_DIM}/{mars.NUM_ACTIONS}")
        if abs(want.rate_hz - 1.0 / CTRL_DT) > 1e-9:
            raise ValueError(
                f"{want.id} declares {want.rate_hz} Hz, this env steps at "
                f"{1.0 / CTRL_DT} Hz — a policy run at the wrong rate is a "
                "different controller")

    # --------------------------------------------------------------- the task

    def sample_target(self) -> np.ndarray:
        """A point in the reach shell, in the BASE frame.

        Sampled in the base frame and transformed out, so the observation's
        `target_base` is yaw-invariant and a MARS spawned facing anywhere sees
        the same task. `reset` is the only caller: re-drawing mid-episode
        would make the progress term's telescoping sum meaningless.
        """
        r = self._rng
        while True:
            point = self._shell_point(r.uniform(*REACH_RADIUS_M),
                                      r.uniform(*REACH_YAW_RAD),
                                      r.uniform(*REACH_HEIGHT_M))
            if point is not None:
                return point

    # ------------------------------------------------------------ the shell

    def shell_radius(self, point_base: np.ndarray) -> float:
        """A base-frame point's distance from the SHOULDER — the shell's own
        coordinate, so `sample_target`, `sample_block` and the knocked-away
        terminal all measure the same thing."""
        return float(np.linalg.norm(np.asarray(point_base) - self.shoulder_base))

    def shell_yaw(self, point_base: np.ndarray) -> float:
        """...and its bearing off the base's +x, about the shoulder."""
        d = np.asarray(point_base) - self.shoulder_base
        return float(math.atan2(d[1], d[0]))

    def in_shell(self, point_base: np.ndarray, margin_m: float = 0.0,
                 margin_rad: float = 0.0) -> bool:
        lo, hi = REACH_RADIUS_M
        r = self.shell_radius(point_base)
        return (lo - margin_m <= r <= hi + margin_m
                and abs(self.shell_yaw(point_base))
                <= REACH_YAW_RAD[1] + margin_rad)

    def sample_block(self) -> np.ndarray:
        """Where the toy spawns this episode, in the BASE frame.

        The PHYSICS ladder, and the only thing `pick_rung` changes. Rungs 1
        and 2 are squares about `PICK_SPOT`; rung 3 is the shell's floor
        footprint, sampled the way `sample_target` samples the shell so that
        "the whole footprint" means the same set the reach task draws from.

        Every rung rejects a draw that leaves the footprint, so the toy always
        starts somewhere the terminal below would not immediately fire.
        """
        r = self._rng
        z = self.toy_rest_z
        if self.pick_rung == 0:
            # `reset` poses the arm first and places the toy at the pinned
            # pose's own grasp point, so this rung has nothing to sample.
            return self._shell_point(*PICK_SPOT, z)
        if self._rung_half is not None:
            centre = self._shell_point(*PICK_SPOT, z)
            while True:
                p = centre + np.array(
                    [r.uniform(-self._rung_half, self._rung_half),
                     r.uniform(-self._rung_half, self._rung_half), 0.0])
                if self.in_shell(p):
                    return p
        while True:
            p = self._shell_point(r.uniform(*REACH_RADIUS_M),
                                  r.uniform(*REACH_YAW_RAD), z)
            if p is not None:
                return p

    def _shell_point(self, radius: float, yaw: float,
                     z: float) -> np.ndarray | None:
        """The shell point at (`radius`, `yaw`) on the plane `z`, base frame.

        `radius` is the 3-D distance from the shoulder, as `sample_target`
        means it, so the horizontal offset shrinks with the height difference
        — the shell is a shell and not a cylinder.
        """
        dz = z - float(self.shoulder_base[2])
        horiz2 = radius * radius - dz * dz
        if horiz2 <= REACH_MIN_HORIZ_M ** 2:
            return None
        horiz = math.sqrt(horiz2)
        return self.shoulder_base + np.array(
            [horiz * math.cos(yaw), horiz * math.sin(yaw), dz])

    # -------------------------------------------------------------- the toy

    def block_pos(self) -> np.ndarray:
        """The toy's centre, in world coordinates."""
        return np.array(self.data.qpos[self._toy_qadr:self._toy_qadr + 3])

    def lift_m(self) -> float:
        """How far the toy is above its RESTING height (m, >= 0 in practice).

        Its resting height rather than the floor: "lift 5 cm" is a lift, and
        a block sitting on the floor already has 2.1 cm of centre height it
        did nothing to earn.
        """
        return float(self.block_pos()[2]) - self.toy_rest_z

    def gripper_load(self) -> float:
        """The torque an OBJECT feeds back through joint6 (N*m).

        `qfrc_constraint`, not the servo's `qfrc_applied`: the module
        docstring has the table that decided it. This is the observation's
        `gripper_load` slot and the holding predicate's first half, and it is
        the same float for both — a gate the reward uses and the policy
        cannot see is a gate the policy has to guess at.
        """
        return float(self.data.qfrc_constraint[self._grip_dadr])

    def _touching_toy(self) -> bool:
        m, d = self.model, self.data
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])}
            if self._toy_body in pair and pair & self._finger_bodies:
                return True
        return False

    def holding(self) -> bool:
        """Is the toy in the claw? The MEASURED predicate.

        Load AND contact. The load alone separates every state measured here
        (0.0000 open / shut on air / shut against the floor, 0.3465 resting on
        the toy, 1.97-2.03 holding it), so the contact term adds no
        discrimination TODAY — it is there because "holding" has to name the
        object, and Phase 5 puts a basket and several toys in the room.
        """
        if self._toy_body < 0:
            return False
        return (abs(self.gripper_load()) >= HOLD_LOAD_NM
                and self._touching_toy())

    def block_lost(self) -> bool:
        """Has the toy been knocked out of the shell's footprint?

        Asked only while NOT holding, by `step`: a toy the arm is carrying can
        be swung anywhere the arm reaches, and ending a successful hold
        because the carry went wide would charge the policy for doing the
        task.
        """
        if self._toy_body < 0:
            return False
        return not self.in_shell(self._to_base(self.block_pos()),
                                 SHELL_EXIT_MARGIN_M, SHELL_EXIT_MARGIN_RAD)

    def _place_block(self, point_base: np.ndarray) -> None:
        q, dofs = self._toy_qadr, self._toy_dadr
        self.data.qpos[q:q + 3] = self._to_world(point_base)
        self.data.qpos[q + 3:q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[dofs:dofs + 6] = 0.0

    def _to_world(self, point_base: np.ndarray) -> np.ndarray:
        x, y, yaw = self.driver.pose(self.data)
        cos, sin = math.cos(yaw), math.sin(yaw)
        return np.array([
            x + point_base[0] * cos - point_base[1] * sin,
            y + point_base[0] * sin + point_base[1] * cos,
            point_base[2],
        ])

    def _to_base(self, point_world: np.ndarray) -> np.ndarray:
        """`point_world` in the base's own frame: rotate by -yaw, translate by
        -base position. The same rotation `MarsDriver.velocity` uses, so what
        the observation sees and what the drive regulates cannot drift."""
        x, y, yaw = self.driver.pose(self.data)
        cos, sin = math.cos(yaw), math.sin(yaw)
        dx, dy = point_world[0] - x, point_world[1] - y
        return np.array([dx * cos + dy * sin, -dx * sin + dy * cos,
                         point_world[2]])

    def joint_target(self, act_arm: np.ndarray) -> np.ndarray:
        """Where the six arm actions ASK the commanded target to go, in rad.

        The map, and only the map — the rate limit is applied by `step`, in
        one place, for every mode. Clipped to the URDF ranges here so that
        "where the action asks" is always a legal pose and the rate-limited
        walk toward it stays legal the whole way (a convex step from a point
        inside the box toward a point inside it).

        `ACTION_MODES` documents the three and why they exist.
        """
        if self.action_mode == "delta":
            # INCREMENTAL, so `a = 0` is an exact fixed point and the policy
            # has a "stay". The integrator is the commanded target rather than
            # the measured `arm_qpos`: integrating the measurement would let a
            # servo lagging under load drag the command with it, and the
            # observation already carries where the arm actually got to.
            want = self._cmd_target + act_arm * DELTA_RAD_PER_STEP
        elif self.action_mode == "cubic":
            # Odd and strictly monotone, so +-1 still spans exactly what
            # `"absolute"` spans and the sign of an action still means the
            # direction it always meant — only the interior is stretched.
            want = self._home + (np.sign(act_arm) * np.abs(act_arm) ** 3
                                 * self.action_scale_rad)
        else:
            want = self._home + act_arm * self.action_scale_rad
        return np.clip(want, self._jnt_lo, self._jnt_hi)

    def effector(self) -> np.ndarray:
        """The gripper's tool point (`ee_link`) in world coordinates."""
        return np.array(self.data.xpos[self._ee_body])

    def grasp_point(self) -> np.ndarray:
        """Midway between the two blade PADS — where an object is pinched.

        Not `ee_link`, and the difference is small but real: MEASURED 7.9 mm,
        along the finger rather than across the jaw, so a block centred on
        `ee_link` still sits between the blades. `distance()` stays on
        `ee_link` because that is what `reach` measures and what the contract
        names, and this exists for the two callers that need the physical
        pinch point: rung 0's kinematic placement, and the probe.
        """
        a, b = (self.data.geom_xpos[g].mean(axis=0) for g in self._pad_geoms)
        return (a + b) / 2.0

    def distance(self) -> float:
        return float(np.linalg.norm(self.effector() - self.target))

    # -------------------------------------------------------- self-collision

    def self_collision(self) -> tuple[str, str] | None:
        """The DEEPEST pair of MARS's own bodies overlapping, or None.

        Three things are filtered out here and NONE of them is what keeps a
        parked MARS alive — which is only known because each was planted and
        the test did not notice (`AGENTS.md`: a guard nobody can tell is
        broken is a guard nobody should trust). Stated as measured:

        * the FLOOR is excluded by the robot-body check, and at HOME that
          check is redundant: MEASURED over 200 steps, all three
          `world <-> base_link` contacts (the chassis box and the two wheels)
          sit at **0.000 mm** depth, because the planar base pins z and the
          wheels rest exactly on the plane — so the DEPTH threshold would
          have dropped them anyway. The filter earns its place in `pick`,
          where a block on the floor makes non-robot pairs ordinary — and
          where "not the world" was the WRONG spelling of it, because a
          free-jointed toy is not the world either and a grasp would have
          terminated as a self-collision at the moment of success.
          `_robot_bodies` is now `base_link`'s subtree, which is what it
          always meant.
        * two geoms of ONE body: MuJoCo excludes same-body pairs itself, so
          this line has never been reached. Kept as a statement of intent.
        * the finger pair, which `mars.tune_contacts` already `add_exclude`s
          (their hub pins overlap ~1 mm at joint6 = 0, and `arm.srdf` disables
          the same pair for MoveIt). The test asserts the MODEL is what
          prevents it, so the redundancy is visible rather than assumed.

        What actually keeps HOME clean is the depth threshold plus the servo
        holding the arm there: `test_home_is_clear_of_self_collision` fails
        when the servo is not stepped, because a limp MARS arm folds into its
        own chassis.

        The deepest rather than the first, so `info["self_collision"]` names
        the pair that actually ended the episode: a real drive into the
        chassis reports several contacts at once and the shallowest of them is
        usually a different pair of links.
        """
        m, d = self.model, self.data
        worst, pair = -SELF_COLLISION_DEPTH_M, None
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = int(m.geom_bodyid[con.geom1])
            b2 = int(m.geom_bodyid[con.geom2])
            if b1 == b2:
                continue
            if b1 not in self._robot_bodies or b2 not in self._robot_bodies:
                continue
            if {b1, b2} == self._finger_bodies:
                continue
            if float(con.dist) >= worst:      # dist < 0 is the overlap
                continue
            worst = float(con.dist)
            pair = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "?",
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "?")
        return pair

    # ---------------------------------------------------------- observations

    def _get_obs(self) -> np.ndarray:
        d = self.data
        obs = np.zeros(mars.OBS_DIM, np.float32)
        obs[mars.OBS_ARM_QPOS] = d.qpos[self._arm_qadr]
        obs[mars.OBS_ARM_QVEL] = d.qvel[self._arm_dadr]
        obs[mars.OBS_HEAD_PITCH] = d.qpos[self._head_qadr]
        # The torque an OBJECT feeds back through joint6 (`qfrc_constraint`),
        # bounded by the servo's own 2 N*m ceiling so the slot arrives
        # pre-scaled. `gripper_load()` is the one definition, and it is also
        # the holding predicate's first half — the gate the reward uses is a
        # float the policy can see.
        #
        # Phase 4a filled this with the torque the servo WROTE
        # (`qfrc_applied`) and Phase 4b measured that out (the table is in
        # `robots/mars.OBS_GRIPPER_LOAD`): `set_arm` clamps the close target
        # to the hard stop, so shutting on air ends at zero position error and
        # zero torque, and while the blades TRAVEL through air the same torque
        # is saturated at -2 N*m — so one sample of it cannot tell "closing"
        # from "holding". `qfrc_constraint` is 0.0000 in every empty state
        # measured and 1.97-2.03 with the block in the claw.
        obs[mars.OBS_GRIPPER_LOAD] = self.gripper_load()
        # The CLIPPED action — what the env applied, not what the network
        # asked for. This cost the phase its first eval and it is worth the
        # paragraph, because the bug is invisible during training.
        #
        # SB3 clips a Box action to the space before the env ever sees it
        # (`OnPolicyAlgorithm.collect_rollouts`), so throughout training the
        # raw and the clipped value are the SAME and observing either is
        # identical. At inference nothing clips for you: MEASURED, this
        # policy's exported mean reaches |a| = 90 in a +-1 box, so a consumer
        # that hands the raw ONNX output straight to `step` had 86.6 written
        # into this slot — a float the policy had never once seen — and the
        # arm wandered off. Same policy, clipped: final distance 0.209 m ->
        # 0.026 m and `at_target` +1.5 -> +88 per episode.
        #
        # AGENTS.md's "suspect the harness when train and eval disagree", and
        # the deeper rule it is an instance of: a limit applied at BOTH
        # training and inference must come from ONE place. The place is
        # `action_space`, and this is the env applying it rather than trusting
        # every caller to. The action-rate PENALTY still prices the raw
        # output, which is walk_env's reason for keeping it (a duck reached
        # mean |a| 29 once the reward stopped seeing what it asked for).
        #
        # WHAT THE SIX ARM FLOATS MEAN depends on `action_mode`, and the slot
        # is the same +-1 either way — which is exactly why it has to be
        # written down. Under `"absolute"`/`"cubic"` they are a POSITION
        # (`joint_target` maps them to a target about HOME), so the slot tells
        # the policy where it last pointed. Under `"delta"` they are a
        # VELOCITY — the increment applied to the commanded target — so the
        # slot tells it how fast it was last moving, and WHERE the arm is
        # comes from `arm_qpos` instead. The layout does not change (the
        # contract's 32 floats are unmoved); the units do, and
        # `MarsBody.contract().deploy` carries the sentence so a code skill
        # reading only the .onnx applies the same map.
        obs[mars.OBS_LAST_ACTION] = self.last_action.clip(-ACTION_CLIP,
                                                          ACTION_CLIP)
        self.target_base = self._to_base(self.target)
        obs[mars.OBS_TARGET_BASE] = self.target_base
        # 1.0 always: the target is PRIVILEGED for both tasks, the way every
        # reward in this repo is — `reach`'s point is not in the physics at
        # all, and `pick` reads the toy's pose off `qpos`. Phase 5 gates it on
        # the head detector actually seeing the thing (AGENTS.md: "a task the
        # robot must SENSE puts its sensing in the command slots, in the
        # robot's own terms") — the slot exists now so that adding the gate
        # does not change the layout.
        obs[mars.OBS_TARGET_SEEN] = 1.0
        v_forward, _v_lateral, wz = self.driver.velocity(d)
        obs[mars.OBS_BASE_TWIST] = (v_forward, wz)
        # OBS_RESERVED stays the zeros `np.zeros` put there.
        return obs

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self._near_streak = 0
        self._self_hit = None
        self._hold_steps_seen = 0
        self._grasps = 0
        self._was_holding = False
        self._lost = False
        self._held = False
        self._lift = 0.0
        self._prev_lift = 0.0
        self.reward_sums = {}
        self.last_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self.prev_action = np.zeros(mars.NUM_ACTIONS, np.float32)
        self._cmd_target = self._home.copy()

        yaw = self.spawn_yaw
        if self.random_yaw:
            yaw = float(self._rng.uniform(-math.pi, math.pi))
        if options and "base_yaw" in options:
            yaw = float(options["base_yaw"])
        # `spawn` writes the HOME pose, zeroes this robot's velocities and
        # clears both applied-force channels, then runs mj_forward.
        self.driver.spawn(self.data, 0.0, 0.0, yaw)
        self.data.time = 0.0

        if self.task == "pick":
            if self.pick_rung == 0:
                # The DRILL. `spawn` has just written ARM_HOME, where the jaws
                # are shut; rung 0 overwrites it with the grasp-ready pose and
                # tells the driver AND the action integrator about it, so that
                # a zero delta holds the pose the episode starts in. Getting
                # only one of those three right leaves the arm snapping back
                # to HOME on the first step with the block still in its way.
                q = np.array(PICK_GRASP_POSE, float)
                self.data.qpos[self._arm_qadr] = q
                self.data.qpos[self._mimic_qadr] = -q[5]
                self._cmd_target = q.copy()
                self.driver.set_arm(dict(zip(mars.ARM_JOINTS, q.tolist())))
                mujoco.mj_forward(self.model, self.data)
                self.target_base_sample = self._to_base(self.grasp_point())
            else:
                self.target_base_sample = self.sample_block()
            # The toy is placed BEFORE the forward pass, so the first
            # observation sees it where the episode actually starts. The
            # target is then the toy itself and moves with it.
            self._place_block(self.target_base_sample)
            mujoco.mj_forward(self.model, self.data)
            self.target = self.block_pos()
            self._prev_lift = self.lift_m() if self.holding() else 0.0
        else:
            self.target_base_sample = self.sample_target()
            self.target = self._to_world(self.target_base_sample)
        self._prev_dist = self.distance()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # The RAW policy output is what is observed and what the action-rate
        # penalty prices — walk_env's lesson: storing the clipped value made
        # unbounded output growth free, and a 25M-step duck reached mean |a|
        # 29 with half its outputs saturated.
        raw = np.asarray(action, np.float32).reshape(mars.NUM_ACTIONS)
        self.prev_action = self.last_action
        self.last_action = raw.copy()
        act = raw.clip(-ACTION_CLIP, ACTION_CLIP)

        want = self.joint_target(act[mars.ACT_ARM])
        # The rate limit (see MAX_TARGET_RATE_RAD_S): the commanded target
        # WALKS to where the action asks, at the speed the servos have. The
        # path stays inside [lo, hi] because it is a convex step from a point
        # inside it toward a point inside it.
        #
        # Applied for EVERY action mode, from this one place, which is the
        # `_get_obs` lesson generalised: a limit that two modes each enforced
        # their own way is a limit two modes can disagree about. In `"delta"`
        # it is provably non-binding — `DELTA_RAD_PER_STEP` IS
        # `self.target_step_rad`, so a full-scale delta asks for exactly the
        # fastest legal step — and it is kept rather than branched around so
        # that changing `MAX_TARGET_RATE_RAD_S` cannot leave one mode behind.
        self._cmd_target = self._cmd_target + np.clip(
            want - self._cmd_target, -self.target_step_rad,
            self.target_step_rad)
        self.driver.set_arm(dict(zip(mars.ARM_JOINTS,
                                     self._cmd_target.tolist())))
        if self.use_base:
            vx, wz = act[mars.ACT_BASE_TWIST] * (MAX_CMD_LINEAR, MAX_CMD_YAW)
            self.driver.set_cmd(float(vx), float(wz), float(self.data.time))
        # else: nothing is commanded, so the driver's own `cmd_vel` watchdog
        # holds the base at zero and station keeping pins it where it spawned.

        for _ in range(self.decimation):
            self.driver.step(self.data)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._self_hit = self.self_collision()
        if self.task == "pick":
            # The target IS the toy, so it is re-read every step rather than
            # sampled once. `reach`'s target may not be redrawn mid-episode
            # (it would break the progress term's telescoping sum); here the
            # thing itself moved, and the distance to where it USED to be is
            # not a quantity the task cares about.
            self.target = self.block_pos()
            # Read the three task predicates ONCE, before the reward, because
            # `holding()` walks the contact list and `_pick_terms`, the
            # streak, the terminal and `info` all need the same answer. Two
            # calls on one state would agree; two calls that could not are
            # how a reward and a terminal disagree about the same step.
            self._held = self.holding()
            self._lift = self.lift_m() if self._held else 0.0
            self._lost = (not self._held) and self.block_lost()
        dist = self.distance()
        reward, terms = self._compute_reward(dist)
        for k, v in terms.items():
            self.reward_sums[k] = self.reward_sums.get(k, 0.0) + v
        self._prev_dist = dist
        if self.task == "pick":
            self._grasps += int(self._held and not self._was_holding)
            self._hold_steps_seen += int(self._held)
            self._was_holding = self._held
            self._prev_lift = self._lift
            self._near_streak = (self._near_streak + 1
                                 if self._held and self._lift >= SUCCESS_LIFT_M
                                 else 0)
        else:
            self._near_streak = (self._near_streak + 1
                                 if dist <= SUCCESS_RADIUS_M else 0)

        obs = self._get_obs()
        terminated = self._self_hit is not None or self._lost
        if not np.isfinite(obs).all():      # kill the episode, not the run
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated = True
        truncated = self.step_count >= self.max_steps

        info: dict[str, Any] = {
            "dist": dist,
            # "has held the target for the last second" — at the final step
            # that IS the task's definition of success. For `pick` the streak
            # counts steps HOLDING the toy above `SUCCESS_LIFT_M`, so one
            # rule serves both: reach's ball and pick's lift are both "and
            # keep doing it".
            "success": bool(self._near_streak >= self.hold_steps),
            "near_steps": int(self._near_streak),
            # The base pair of the action is read and forced to zero for
            # `reach`; saying so here is what keeps a silently-ignored knob
            # from looking like a working one (AGENTS.md rule 0).
            "base_enabled": self.use_base,
            # What an action MEANT this episode, beside the numbers it
            # produced. A probe or a sheet that reports a distance without it
            # is comparing two different controllers by name only.
            "action_mode": self.action_mode,
            "self_collision": self._self_hit,
            "terms": terms,
        }
        if self.task == "pick":
            # `hold_frac` is the instrument AGENTS.md's "does ANY rollout
            # ever do the thing?" asks for, per rollout and while training:
            # if the ladder is wrong it is 0.000 for a whole run and no amount
            # of reward work will move it. `grasps` separates "never got hold
            # of it" from "got hold of it and dropped it", which are two
            # different fixes.
            info.update({
                "holding": bool(self._held),
                "lift_m": float(self._lift),
                "block_z": float(self.block_pos()[2]),
                "block_base": self._to_base(self.block_pos()).tolist(),
                "block_lost": bool(self._lost),
                "grasps": int(self._grasps),
                "hold_frac": self._hold_steps_seen / max(1, self.step_count),
                "gripper_load": self.gripper_load(),
                "pick_rung": self.pick_rung,
            })
        if terminated or truncated:
            info["episode_rewards"] = dict(self.reward_sums)
        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------- rewards

    def _compute_reward(self, dist: float) -> tuple[float, dict[str, float]]:
        """The task's terms, every one present every step.

        A stable key set is what the teach panel draws one bar per, and what
        `train._penalty_sign_callback_cls` watches for a sign flip — so each
        task emits a FIXED set and the two sets share the four penalties
        rather than each computing their own.
        """
        terms = {"reach_progress": 0.0, "at_target": 0.0}
        # PROGRESS, not proximity: positive only while the gap is closing, and
        # an episode's total is `d_start - d_end` however it got there, so
        # oscillating earns nothing and parking earns nothing.
        progress = W_PROGRESS * (self._prev_dist - dist)
        near = W_NEAR * math.exp(-(dist / NEAR_SIGMA_M) ** 2)
        if self.task == "pick":
            # Not paid while HOLDING, which is the plan's rule and is also
            # what keeps the two progress terms from double-counting: with the
            # toy in the claw this distance is pinned at the ~8 mm offset
            # between `ee_link` and the blades and has nothing left to say.
            # `_prev_dist` is still updated every step by `step`, so a release
            # cannot produce a phantom jump on the step after it.
            terms["reach_progress"] = 0.0 if self._held else progress
            # ...but `at_target` is NOT gated. Gating it would make grasping
            # cost 2.0 a step against a lift worth 5.0 in total, i.e. it would
            # pay the policy to hover next to the block forever — AGENTS.md's
            # "what else satisfies the terms that remain?" answered wrong.
            # Ungated, holding keeps the proximity income (the claw IS on the
            # block) and lifting adds `held_high` on top, so the ordering
            # hover < hold < lift is monotone with no cliff in it.
            terms["at_target"] = near
            # The second progress term, and the telescoping is the whole
            # design. `h` is the HELD height — the lift while the claw has the
            # toy and exactly 0 whenever it does not — so the episode's total
            # is `W_LIFT * (h_end - h_start)` however it got there. Dropping
            # the toy from 10 cm charges back every point the lift earned, and
            # KICKING it into the air earns nothing at all, because `h` is 0
            # when nothing is held. No jackpot to farm and no release to
            # exploit.
            terms["lift_progress"] = W_LIFT * (self._lift - self._prev_lift)
            # The income. Bounded, per step, and gated on the task's own
            # success height, so an early termination forfeits the rest of it
            # — the same way `at_target` makes a reach worth keeping.
            terms["held_high"] = (W_HELD if self._held
                                  and self._lift >= SUCCESS_LIFT_M else 0.0)
        else:
            terms["reach_progress"] = progress
            terms["at_target"] = near
        terms.update(self._penalty_terms())
        if self.task == "pick":
            terms["block_lost_penalty"] = (BLOCK_LOST_PENALTY if self._lost
                                           else 0.0)
        return float(sum(terms.values())), terms

    def _penalty_terms(self) -> dict[str, float]:
        """The four both tasks share, computed in one place."""
        action_rate = ACTION_RATE_W * -float(
            ((self.last_action - self.prev_action) ** 2).sum())
        # The RAW magnitude, deliberately: what this prices is the mean
        # running outside its own box, which is invisible once the env
        # clips (see ACTION_MAG_W).
        action_mag = ACTION_MAG_W * -float((self.last_action ** 2).sum())
        joint_vel = JOINT_VEL_W * -float(
            (self.data.qvel[self._arm_dadr] ** 2).sum())
        return {
            "action_rate_penalty": action_rate,
            "action_mag_penalty": action_mag,
            "joint_vel_penalty": joint_vel,
            "self_collision_penalty": (SELF_COLLISION_PENALTY
                                       if self._self_hit else 0.0),
        }


class MarsPickEnv(MarsArmEnv):
    """`MarsArmEnv` with `pick` as its default task — and it has to exist.

    **`train.make_env` uses `--task` only to CHOOSE a class**, then constructs
    it with the kwargs every body shares; the task string itself never reaches
    the constructor. The G1 has one env class per task for exactly this
    reason, and Phase 4b found out what a body with two tasks behind one class
    does: the first `--task pick` run trained `reach` under a pick run's name,
    at `reach`'s 5 ms with no block in the scene, and the only tell was that
    `progress.jsonl` carried six reward keys instead of nine. A knob that
    changes nothing is broken, not null (AGENTS.md rule 0) — and the smallest
    thing that cannot be got wrong is a subclass whose only content is its
    default.

    `MarsArmEnv(task="pick")` remains exactly equivalent, and is what a test
    or a probe that names the task explicitly should keep using.
    """

    def __init__(self, task: str = "pick", **kwargs):
        super().__init__(task=task, **kwargs)


def action_map_sentence(mode: str) -> str:
    """What `actions[0:6]` MEAN under `mode`, in one clause, from the constants.

    `MarsBody.contract().deploy` embeds this, so the caveat leaves the
    building with the .onnx instead of living in a docstring the person
    holding the file will never open. Built here because the numbers are
    here: a sentence typed into the contract would be a second copy of
    `DELTA_RAD_PER_STEP` to keep in step.
    """
    if mode == "delta":
        return (f"an INCREMENT on the commanded joint target: add "
                f"a * {DELTA_RAD_PER_STEP:g} rad per {1.0 / CTRL_DT:g} Hz "
                "step to the target you last sent, clamp to the URDF joint "
                "ranges, and send that (a = 0 means hold). The consumer "
                "keeps the integrator; arm_qpos[0:6] is where the arm got to")
    if mode == "cubic":
        return (f"an ABSOLUTE joint target, ARM_HOME + sign(a)*|a|^3 * "
                f"{ACTION_SCALE_RAD:g} rad, clamped to the URDF joint ranges")
    return (f"an ABSOLUTE joint target, ARM_HOME + a * {ACTION_SCALE_RAD:g} "
            "rad, clamped to the URDF joint ranges")


def make_mars_env(**kwargs) -> MarsArmEnv:
    """Factory with the training defaults (`train-walk --robot mars`)."""
    kwargs.setdefault("max_episode_s", EPISODE_S)
    return MarsArmEnv(**kwargs)


__all__ = ["ACTION_CLIP", "ACTION_MAG_W", "ACTION_MODES", "ACTION_RATE_W",
           "ACTION_SCALE_RAD", "ARM_ONLY_TASKS", "BLOCK_LOST_PENALTY",
           "CTRL_DT", "DECIMATION", "DEFAULT_ACTION_MODE", "DEFAULT_PICK_RUNG",
           "DELTA_RAD_PER_STEP", "EPISODE_S", "HOLD_LOAD_NM", "JOINT_VEL_W",
           "MAX_TARGET_RATE_RAD_S", "NEAR_SIGMA_M", "PICK_DECIMATION",
           "PICK_FLOAT_M", "PICK_PHYSICS_DT", "PICK_RUNGS", "PICK_RUNG_HALF_M",
           "PICK_GRASP_POSE", "PICK_SPOT", "PICK_TOY_BODY", "PICK_TOY_JOINT",
           "PICK_TOY_KIND",
           "REACH_HEIGHT_M", "REACH_MIN_HORIZ_M", "REACH_RADIUS_M",
           "REACH_YAW_RAD", "RUNG_SCALE_RAD", "SELF_COLLISION_DEPTH_M",
           "SELF_COLLISION_PENALTY", "SHELL_EXIT_MARGIN_M",
           "SHELL_EXIT_MARGIN_RAD", "SUCCESS_HOLD_S", "SUCCESS_LIFT_M",
           "SUCCESS_RADIUS_M", "TASKS", "W_HELD", "W_LIFT", "W_NEAR",
           "W_PROGRESS", "MarsArmEnv", "MarsPickEnv", "PICK_TOY_PARK",
           "action_map_sentence", "make_mars_env",
           "pick_scene_spec", "pick_scene_xml", "toy_rest_z", "toy_spec"]
