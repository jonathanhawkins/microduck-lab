"""Compose a scenario into ONE compiled MuJoCo model with `MjSpec`.

Each duck is the upstream robot MJCF attached under its own prefix
(`"<id>/"`), so `"d1/left_hip_yaw"`, `"d1/imu_ang_vel"`, `"d1/tof"`,
`"d1/left_foot_collision"` all resolve by name and per-duck code never has
to know where in qpos a duck landed (`DuckAddress` looks it up once).

Verified on the pinned MuJoCo 3.10: attaching the same robot file N times
prefixes joints, actuators, sensors, sites, geoms and meshes, and a 2-duck
world steps in one `mj_step`. The compiled model has NO keyframes (they live
in the upstream scene file, not the robot file), so spawning sets qpos from
`contract.DEFAULT_POSE` explicitly — see `spawn_duck`.

Objects: walls are static boxes, boxes with mass are free bodies, balls
match upstream's 70 mm / 15 g kick ball. Scenery is geom group 0 so range
sensors (`sensors.ray.DEFAULT_GROUPS`) see it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import contract as C
from ..sensors.ray import UNSENSED_GROUP
from .scenario import PICKABLE_KINDS, TEAM_COLORWAYS, Scenario, Wall

# Toys live in their own geom group: range sensors see them (they are
# obstacles and pick targets), but the detector's line-of-sight test looks
# through them — a 4 cm block in the beak, 2 cm in front of the camera,
# once hid a basket for a whole run, and a real detector would look at a
# 30 cm basket, not at a point its centre.
PICKABLE_GROUP = 4
ROBOT_DIR = C.MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck"
ROBOT_XML = {
    "walk": ROBOT_DIR / "robot_walk.xml",
    "all": ROBOT_DIR / "robot_allcollisions.xml",
}


def duck_prefix(duck_id: str) -> str:
    return f"{duck_id}/"


# Which of the robot's ~38 materials a colorway owns (roadmap Track 4.2.2,
# corrected in 4.2.3). EVERY printed part takes one of the colorway's two
# colours: the shells are the head, trunk, legs and hips, the trim is the beak,
# feet, ankles and soles. What is left out is what is not printed and is the
# same on every duck as it is on the real robot — servos, PCBs, bearings, the
# lens, the camera mount, the neck and yaw brackets.
#
# The lists are long because the CAD export is not tidy. Four printed parts
# carried colours no colorway ever claimed — a teal thigh plate and shoe rim
# (`upper_leg_rigidity_plate`, `sole_*`), a pale-blue hip (`yaw_roll_motion`)
# and a pink soft mouth (`jaw_soft`, `soft_mouth_top`, the same pink on all
# four colorways) — so a duck that was meant to be one colour rendered as five.
# Painting only the four body shells hid it; painting the legs a second shade
# made it worse. If an upstream re-export adds a printed part, it belongs in
# one of these two lists, and `paint_team`'s return count is what catches a
# name that moved.
SHELL_MATERIALS = ("left_shell_material", "right_shell_material",
                   "top_head_shell_material", "bottom_head_shell_material",
                   "leg_material", "upper_leg_left_material", "upper_leg_right_material",
                   "hip_l_material", "upper_leg_rigidity_plate_material",
                   "yaw_roll_motion_material", "jaw_soft_material",
                   "soft_mouth_top_material")
TRIM_MATERIALS = ("jaw_material", "foot_left_material", "foot_right_material",
                  "ankle_left_material", "ankle_right_material",
                  "sole_left_material", "sole_right_material")


def paint_team(model: mujoco.MjModel, duck_id: str, colorway: str) -> int:
    """Give one duck its team's colours, in the compiled model.

    `MjSpec.attach` prefixes materials per duck (`d0/left_shell_material`), so
    this is a write to that duck's own materials and no other duck's. Returns
    how many it painted — 0 means the names moved in an upstream CAD re-export
    and the paint silently did nothing, which is worth a caller's assert.

    Colour is not mass: nothing here touches physics, and
    `tests/test_arena.py` still locks a world step-for-step against the walk
    env. What it does change is every MuJoCo render of a world
    (`render-striker`, a viewer opened on one). The browser viewer draws from
    the single-robot scene and tints client-side."""
    look = TEAM_COLORWAYS.get(colorway)
    if look is None:
        return 0
    n = 0
    for names, rgb in ((SHELL_MATERIALS, look["shell"]), (TRIM_MATERIALS, look["trim"])):
        for name in names:
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, duck_prefix(duck_id) + name)
            if mid >= 0:
                model.mat_rgba[mid] = [*rgb, 1.0]
                n += 1
    return n


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _key(pt: tuple[float, float]) -> tuple[float, float]:
    return (round(float(pt[0]), 6), round(float(pt[1]), 6))


def _away(wall: Wall, pt: tuple[float, float]) -> tuple[float, float]:
    """Unit direction along `wall` away from its endpoint `pt`."""
    (ax, ay), (bx, by) = wall.start, wall.end
    length = math.dist(wall.start, wall.end)
    dx, dy = (bx - ax) / length, (by - ay) / length
    return (dx, dy) if _key(pt) == _key(wall.start) else (-dx, -dy)


def _add_cove(w, scenario: Scenario) -> None:
    """The cove (`Scenario.cove`, a radius R): a quarter-round along the base
    of every wall, from the floor one R in from the wall's inner face up to
    the face one R above the floor, built from N tangent boxes - flat facets
    on the arc's midpoints that meet at concave creases, which a ball rolls
    across without a bump (the facet's sagitta is 0.7 mm at N = 8, R =
    0.15; the ball is 35 mm). Where two walls share an endpoint each facet
    stops short of the corner by its own inward offset over tan(half the
    interior angle) - a mitre - so the two coves meet on the bisector. On a
    pitch the cove is cut at the goal mouths: a goal needs the ball 8 cm
    from the end line, and a cove there would roll a slow shot back out. The
    cut ends are the posts. Static geoms never collide with each other, so
    the facets may sit in the floor and in one another."""
    R = float(scenario.cove)
    n_seg = max(4, int(round(R / 0.02)))
    dphi = (math.pi / 2) / n_seg
    hw = R * math.tan(dphi / 2)             # half-width: tangent planes meet exactly at the creases
    t = 0.01                                # half-thickness, hung below the tangent plane
    walls = scenario.walls
    ends: dict[tuple[float, float], list[int]] = {}
    for i, wl in enumerate(walls):
        ends.setdefault(_key(wl.start), []).append(i)
        ends.setdefault(_key(wl.end), []).append(i)
    gw = scenario.goal_width / 2
    for i, wl in enumerate(walls):
        p0, p1 = wl.start, wl.end
        length = math.dist(p0, p1)
        if length < 1e-6:
            continue
        ux, uy = (p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length
        nx, ny = -uy, ux                                     # the wall's left-hand normal
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        if (-mx) * nx + (-my) * ny < 0:                      # the floor's centre is on the right: flip
            p0, p1, ux, uy, nx, ny = p1, p0, -ux, -uy, -nx, -ny
        yaw = math.atan2(uy, ux)
        # Mitres: per unit of inward offset, how far a facet stops short of each end.
        cut = [0.0, 0.0]
        for k, (pt, bx, by) in enumerate(((p0, ux, uy), (p1, -ux, -uy))):
            for j in ends.get(_key(pt), []):
                if j == i or math.dist(walls[j].start, walls[j].end) < 1e-6:
                    continue
                ax, ay = _away(walls[j], pt)
                alpha = math.acos(max(-1.0, min(1.0, ax * bx + ay * by)))    # the interior angle
                if alpha > 1e-3:
                    cut[k] = 1.0 / math.tan(alpha / 2)
        # Runs along the wall: all of it, minus the goal mouth on an END wall.
        runs = [(0.0, length, cut[0], cut[1])]
        if gw > 0 and abs(nx) > 0.9 and abs(uy) > 1e-9:
            lam_a, lam_b = sorted(((-gw - p0[1]) / uy, (gw - p0[1]) / uy))
            if lam_a < length and lam_b > 0:
                runs = []
                if lam_a > 0:
                    runs.append((0.0, lam_a, cut[0], 0.0))
                if lam_b < length:
                    runs.append((lam_b, length, 0.0, cut[1]))
        fx, fy = mx + nx * wl.thickness / 2, my + ny * wl.thickness / 2     # a point on the inner face
        tilt, quat = np.zeros(4), np.zeros(4)
        for r_i, (lam0, lam1, c0, c1) in enumerate(runs):
            for k in range(n_seg):
                phi = (k + 0.5) * dphi
                s = R * (1.0 - math.sin(phi))               # the facet midpoint: in from the face…
                z = R * (1.0 - math.cos(phi))               # …and up from the floor
                a, b = lam0 + s * c0, lam1 - s * c1
                if b - a < 1e-3:
                    continue
                lam_m = (a + b) / 2 - length / 2
                mujoco.mju_axisAngle2Quat(tilt, np.array([1.0, 0.0, 0.0]), -phi)
                mujoco.mju_mulQuat(quat, np.array(_yaw_quat(yaw)), tilt)
                w.add_geom(name=f"cove{i}_{r_i}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[(b - a) / 2, hw, t],
                           pos=[fx + ux * lam_m + nx * (s - t * math.sin(phi)),
                                fy + uy * lam_m + ny * (s - t * math.sin(phi)),
                                z - t * math.cos(phi)],
                           quat=list(quat), group=0, rgba=[0.74, 0.72, 0.68, 1.0])


def _pin_mass_properties_to_walk(robot: mujoco.MjSpec) -> None:
    """Give a non-walk robot variant the walk file's `<inertial>` values.

    Upstream exports each variant separately and rounds the inertials by
    hand: `jaw_soft` is `fullinertia="0.000320811 ..."` in robot_walk.xml
    and `0.000320812` in robot_allcollisions.xml (three bodies differ, in
    the seventh digit; every mass is identical). Nothing physical - and
    enough to seed chaos: the shipped walker's joint angles under "all"
    drifted 0.2-0.4 rad from "walk" over 10 s on a flat floor with no
    contact but the soles', and tests/test_arena.py's step-for-step lock
    against the walk env failed at 3e-9 on step 0. A variant is the walk
    robot plus collision meshes, so its mass properties are the walk
    robot's, to the bit."""
    walk = mujoco.MjSpec.from_file(str(ROBOT_XML["walk"]))
    ref = {b.name: b for b in walk.bodies}
    for b in robot.bodies:
        r = ref.get(b.name)
        if r is None or not b.name:
            continue
        b.mass, b.ipos, b.iquat = r.mass, r.ipos, r.iquat
        b.inertia, b.fullinertia = r.inertia, r.fullinertia
        b.explicitinertial = r.explicitinertial


# --- the mouth ------------------------------------------------------------
#
# The robot has FIFTEEN servos: the extra one is the mouth (Dynamixel id 32,
# index 9 of `duck-control`'s joint table), and every alpha policy skips it -
# the 61-obs / 14-action contract has no mouth slot, so the beak belongs to
# the app, not to the network. Upstream's MJCF has no mouth joint at all:
# onshape-to-robot cannot emit a closed loop into an MJCF tree, so the jaw's
# linkage was collapsed and the lower bill (`jaw`) became a static geom
# inside `jaw_soft`. The beak is welded shut in every render.
#
# Split the bill onto a hinge, and take the cheap half of the trade: only the
# VISUAL geom moves. The collision geom stays on `jaw_soft`, so the bill never
# takes a contact force, and the new body's mass is a picogram - MuJoCo
# refuses an exactly-zero mass on a moving body, so this is as close to
# nothing as the compiler allows. The DOF's whole inertia is the joint's
# `armature`, which adds to that DOF's diagonal of M and to nothing else.
#
# What that buys, measured (scratch A/B on a ONE-DUCK empty floor, the same
# scripted actions, with and without this function): nothing PHYSICAL changes
# - same masses, same inertias, same contacts, same sensing - but the
# trajectories are not bit-identical. Carrying one more DOF through the
# constraint solver moves the result ~1.6e-9 per step, and this walker is
# chaotic enough to amplify that (the note in `_pin_mass_properties_to_walk`
# is the same effect from a seventh-digit inertia). So a SEEDED world no
# longer replays the exact trajectory it did before the mouth existed.
#
# What that costs is NOT yet measured, and must not be read as untouched:
# `docs/roadmap.md` keeps it open ("Not measured: whether `eval-tidy`'s mean
# moved"). A per-step divergence this size compounds, so any battery banked
# under a `--tag` before the mouth and resumed after it is mixing two
# populations - re-run those from scratch rather than appending, and re-quote
# the README's tidy and pitch bands only once someone has re-measured them.
# `tests/test_arena.py`'s step-for-step lock
# against the walk env still holds at its 1e-9 / 1e-6 tolerances. What the
# beak MEETS is what it met before; what it LOOKS like now moves. A bill that
# bore contacts would be a different robot and a re-measure.
#
# The pivot is MEASURED, not sourced: no shipped file has it (upstream's
# MJCF, the onboard `kinematics` crate and the mjlab scenes are all 14
# joints). `jaw.stl` carries a 6 mm through-bore at (+0.0030, +/-0.042,
# -0.0181) in the `jaw_soft` frame - found by fitting circles with an
# empty-interior test, the only full circle in the mesh - and the
# `seeed_bearing` part inside the head sits at (+0.0032, -0.0385, -0.0180).
# A bearing concentric with the bill's bore is the pivot: two independent
# measurements agreeing to 0.2 mm. Worth an eye against the real robot.
#
# `bottom_head_shell` does NOT move. It reaches the bill tip but runs back to
# world x -0.030, the whole underside of the head; `jaw` runs +0.017 -> +0.0858
# and ends exactly at the `mouth_tip` site. Upstream's own commented-out
# `<exclude body1="jaw" body2="bottom_head_shell"/>` says the same: they were
# separate bodies before the collapse.
MOUTH_BODY = "mouth"
# The bill renders but is never SENSED. Every range sensor excludes its own
# mount body (`sensors/ray.py`), and until the split the bill was IN that body
# - `jaw_soft` was a leaf - so the ToF never saw it. A child body is not
# excluded, and the first cut had the duck's downward rays hitting its own
# beak instead of the floor (tests/test_sensors.py, test_detector.py caught
# it). `sensors.ray.UNSENSED_GROUP` is the group for exactly this, and MuJoCo
# renders it by default (mjvOption.geomgroup = [1,1,1,0,0,0]), so the bill
# still shows up in `record-world` and `render-rollout`. The sensed twin is
# the collision geom, which stays on `jaw_soft` and stays excluded.
MOUTH_GROUP = UNSENSED_GROUP
MOUTH_MESH = "jaw"
MOUTH_PIVOT = (0.0030, 0.0, -0.0181)   # in the `jaw_soft` frame
MOUTH_AXIS = (0.0, 1.0, 0.0)           # the head's left-right axis; +ve drops the bill
MOUTH_CLOSED = math.radians(-5.0)      # duck-control/src/model.rs, the v1.6 travel
MOUTH_OPEN = math.radians(30.0)
MOUTH_KP = 12.0
# The bill has no mass of its own, so the servo's whole second-order response
# is set here: w = sqrt(KP / ARMATURE) = 49 rad/s (w*dt = 0.25 at the 5 ms
# step) and zeta = DAMPING / (2 sqrt(KP * ARMATURE)) = 0.82, which opens the
# beak in about 0.1 s without ringing. The first cut ran zeta = 0.065 and the
# bill oscillated 50 degrees past its own limit. Armature and damping are
# diagonal terms on the mouth DOF and touch nothing else, so tuning this
# costs the rest of the robot nothing.
MOUTH_ARMATURE = 5e-3
MOUTH_DAMPING = 0.02   # a touch of joint damping; the servo's own kv does the work
MOUTH_KV = 0.40
# 1e-12, not 1e-9: measured, the divergence from the unsplit model scales with
# this mass down to ~1e-12 and then hits a floor (1.6e-9 over 60 steps) that is
# the extra DOF in the constraint solver, not the link. Ten times quieter for
# nothing. MuJoCo refuses an exactly-zero mass on a moving body.
MOUTH_MASS = 1e-12


# Pivot to the `mouth_tip` site: the bill's lever, and so the arithmetic that
# turns "how thick is this toy" into "how far can the beak close on it".
MOUTH_LEVER = 0.0606
MOUTH_TRAVEL = MOUTH_OPEN - MOUTH_CLOSED


def mouth_gape(open_frac: float) -> float:
    """Tip-to-tip opening, metres, at an opening fraction. Wide is 36 mm - so
    a 40 mm block does NOT fit between the bills, which is why a held toy is
    an attachment (`World.grasp`) and not a pinch."""
    return 2.0 * MOUTH_LEVER * math.sin(0.5 * min(max(open_frac, 0.0), 1.0) * MOUTH_TRAVEL)


def mouth_frac_for_gape(gap_m: float) -> float:
    """The inverse: how far the bill closes on something `gap_m` thick. Wider
    than the beak can open saturates at 1 - the bill goes AROUND the toy
    rather than through it."""
    half = min(max(0.5 * gap_m / MOUTH_LEVER, 0.0), 1.0)
    return min(max(2.0 * math.asin(half) / MOUTH_TRAVEL, 0.0), 1.0)


def mouth_target(open_frac: float) -> float:
    """Joint angle for an opening fraction - 0 closed, 1 wide. The sim half of
    `duck_control::model::mouth_target`, clamped the same way (anything outside
    0..1, NaN included, is clamped rather than fed to a servo)."""
    f = open_frac if math.isfinite(open_frac) else 0.0
    return MOUTH_CLOSED + min(max(f, 0.0), 1.0) * (MOUTH_OPEN - MOUTH_CLOSED)


def split_jaw(robot: mujoco.MjSpec) -> None:
    """Re-parent the bill's visual geom onto a hinged `mouth` body under
    `jaw_soft`, and add the 15th actuator. Idempotent-ish: a spec that
    already has the body is left alone.

    The new body carries no mass and no contacts, so nothing physical about
    the rest of the robot changes - but it is one more DOF in the solver, and
    that is not free at the last few digits. See the note above."""
    head = next((b for b in robot.bodies if b.name == "jaw_soft"), None)
    if head is None or any(b.name == MOUTH_BODY for b in robot.bodies):
        return
    bill = [g for g in head.geoms
            if g.meshname == MOUTH_MESH and g.classname and g.classname.name == "visual"]
    if not bill:
        return
    jaw = head.add_body(name=MOUTH_BODY, pos=MOUTH_PIVOT, quat=[1.0, 0.0, 0.0, 0.0])
    jaw.mass, jaw.inertia, jaw.explicitinertial = MOUTH_MASS, [1e-12] * 3, True
    j = jaw.add_joint(name=MOUTH_BODY, type=mujoco.mjtJoint.mjJNT_HINGE,
                      axis=list(MOUTH_AXIS), range=[MOUTH_CLOSED, MOUTH_OPEN])
    j.limited = int(mujoco.mjtLimited.mjLIMITED_TRUE)
    j.armature = MOUTH_ARMATURE   # the whole of the DOF's inertia: the link has none
    j.damping[0] = MOUTH_DAMPING
    # The `microduck` childclass hands every joint the XL330's 0.1 Nm of
    # stiction, which on a bill with no mass is a half-degree dead band and
    # nothing else. A cosmetic joint tracks its target.
    j.frictionloss = 0.0
    for i, g in enumerate(bill):
        ng = jaw.add_geom(name=f"{MOUTH_BODY}_bill_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=g.meshname,
                          pos=[p - c for p, c in zip(g.pos, MOUTH_PIVOT)], quat=g.quat,
                          material=g.material, rgba=g.rgba, group=MOUTH_GROUP,
                          contype=g.contype, conaffinity=g.conaffinity)
        ng.classname = g.classname
        robot.delete(g)
    a = robot.add_actuator(name=MOUTH_BODY, target=MOUTH_BODY,
                           trntype=mujoco.mjtTrn.mjTRN_JOINT)
    # What `<position kp=...>` expands to. `add_actuator` defaults to
    # biastype NONE, which is a CONSTANT-FORCE actuator, not a servo: the
    # first cut drove the bill into whichever limit the sign of `ctrl`
    # pointed at and sat there, 1.5 degrees past it, for every target.
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0], a.biasprm[1], a.biasprm[2] = MOUTH_KP, -MOUTH_KP, -MOUTH_KV
    a.ctrlrange = [MOUTH_CLOSED, MOUTH_OPEN]
    a.ctrllimited = int(mujoco.mjtLimited.mjLIMITED_TRUE)


def _reseat_keyframes(model: mujoco.MjModel) -> None:
    """Put the mouth's slot back into keyframes authored without it.

    MuJoCo pads a `<key qpos>` that is one short with a zero at the END, but
    the mouth takes a slot in the MIDDLE of the chain (it hangs off `jaw_soft`,
    and the right leg comes after), so every upstream keyframe landed the whole
    right leg one slot late. The STAND pose came back with a scrambled right
    leg and the trunk 3 mm high, which the /pose grounding test caught.

    Only for a model whose keyframes predate `split_jaw` - which is every one
    we compile, since no upstream file has ever had a mouth joint."""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MOUTH_BODY)
    if j < 0 or model.nkey == 0:
        return
    q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MOUTH_BODY)
    for k in range(model.nkey):
        row = model.key_qpos[k].copy()
        model.key_qpos[k, q] = MOUTH_CLOSED       # a keyed pose has its beak shut
        model.key_qpos[k, q + 1:] = row[q:-1]
        row = model.key_qvel[k].copy()
        model.key_qvel[k, v] = 0.0
        model.key_qvel[k, v + 1:] = row[v:-1]
        if a >= 0 and model.nu:
            row = model.key_ctrl[k].copy()
            model.key_ctrl[k, a] = MOUTH_CLOSED
            model.key_ctrl[k, a + 1:] = row[a:-1]


def scene_model() -> mujoco.MjModel:
    """The single-robot scene the viewer draws every duck from, with the same
    `split_jaw` surgery the composed world gets.

    The stream's per-duck body list is mapped onto `GET /scene`'s body list
    POSITIONALLY (world_server's `bodies`), so the two models have to agree
    body for body - `tests/test_arena.py` locks exactly that."""
    spec = mujoco.MjSpec.from_file(str(C.SCENE_WALK_XML))
    split_jaw(spec)
    model = spec.compile()
    _reseat_keyframes(model)
    return model


def compose(scenario: Scenario) -> mujoco.MjModel:
    """Compile the scenario. Raises FileNotFoundError if microduck_rl is not
    checked out (same message as the walk env)."""
    robot_xml = ROBOT_XML[scenario.collision]
    if not robot_xml.exists():
        raise FileNotFoundError(
            f"{robot_xml} not found — clone microduck_rl next to microduck_local "
            "or set MICRODUCK_RL_DIR")
    spec = mujoco.MjSpec()
    spec.modelname = f"world:{scenario.name}"
    spec.option.timestep = C.PHYSICS_DT
    # The G1 XML wants implicitfast / 10 / 20; attach keeps the PARENT, so
    # set it here before the robot is attached. Capsule-only worlds stay on
    # the spec default — test_arena's duck lock does not load a G1.
    if any(p.kind == "g1" for p in scenario.persons):
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.iterations = 10
        spec.option.ls_iterations = 20
    w = spec.worldbody
    w.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1],
                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    hx, hy = scenario.floor[0] / 2, scenario.floor[1] / 2
    # A finite plane: size = (half x, half y, spacing) — rays and contacts
    # treat a plane as infinite regardless, the extents only draw it.
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[hx, hy, 0.05], group=0, rgba=[0.32, 0.36, 0.40, 1.0])
    for i, wall in enumerate(scenario.walls):
        (x0, y0), (x1, y1) = wall.start, wall.end
        length = math.dist(wall.start, wall.end)
        yaw = math.atan2(y1 - y0, x1 - x0)
        w.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[length / 2, wall.thickness / 2, wall.height / 2],
                   pos=[(x0 + x1) / 2, (y0 + y1) / 2, wall.height / 2],
                   quat=_yaw_quat(yaw), group=0, rgba=[0.82, 0.80, 0.76, 1.0])
    if scenario.cove > 0:
        _add_cove(w, scenario)
    for i, box in enumerate(scenario.boxes):
        half = [s / 2 for s in box.size]
        if box.mass > 0:
            body = w.add_body(name=f"box{i}", pos=list(box.pos), quat=_yaw_quat(box.yaw))
            body.add_freejoint(name=f"box{i}_free")
            body.add_geom(name=f"box{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=half, mass=box.mass, group=0, rgba=list(box.rgba))
        else:
            w.add_geom(name=f"box{i}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=half, pos=list(box.pos), quat=_yaw_quat(box.yaw),
                       group=0, rgba=list(box.rgba))
    for i, ball in enumerate(scenario.balls):
        body = w.add_body(name=f"ball{i}", pos=[ball.pos[0], ball.pos[1], ball.radius])
        body.add_freejoint(name=f"ball{i}_free")
        # Thin hollow sphere, as upstream's ball.xml: I = 2/3 m r².
        inertia = (2.0 / 3.0) * ball.mass * ball.radius ** 2
        body.mass = ball.mass
        body.ipos = [0, 0, 0]
        body.inertia = [inertia, inertia, inertia]
        body.explicitinertial = True
        # condim 6: MuJoCo only applies the torsional and rolling
        # coefficients on a 6-dim contact; on the default 3-dim contact
        # (sliding only) a rolling ball has NOTHING to slow it, and it rolls
        # until a wall does. Upstream's ball.xml carries the same
        # `friction="0.5 0.005 0.0001"` on a condim-3 geom, so its rolling
        # value is silently ignored too. The floor keeps priority parity, so
        # the pair takes the larger of each coefficient (sliding stays the
        # floor's 1.0; rolling is the ball's, `Ball.rolling`), and the feet
        # keep their priority-1 contact with the ball unchanged.
        #
        # Restitution is NOT modelled, and a stiffer contact does not buy it
        # (measured 2026-09-06, a 1.4 m/s roll into a board): the default
        # solref (0.02, 1) rebounds at e = 0.15; (0.02, 0.2) 0.20;
        # (0.01, 0.2) 0.20; (0.01, 0.1) - a time constant at the 2 x timestep
        # floor and a fifth of critical damping - 0.22, and that setting also
        # shortens the roll-outs above by a fifth (0.26 -> 0.21 m, 3.5 ->
        # 3.4 m). A hollow ball is e ~ 0.5-0.7, so a kicked ball here sits
        # at the wall it hits, whatever the solref; left at the default.
        body.add_geom(name=f"ball{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[ball.radius, 0, 0], group=0, rgba=[1.0, 0.55, 0.0, 1.0],
                      condim=6, friction=[0.5, 0.005, ball.rolling])
    for t in scenario.pickables:
        k = PICKABLE_KINDS[t.kind]
        half = [v / 2 for v in k["size"]]
        body = w.add_body(name=t.id, pos=[t.pos[0], t.pos[1], half[2] + 0.001], quat=_yaw_quat(t.yaw))
        body.add_freejoint(name=f"{t.id}_free")
        # priority 1, so the 0.8 sliding friction is the one that applies:
        # MuJoCo takes the HIGHER-priority geom's friction, and at equal
        # priority the element-wise max - and the floor's sliding is 1.0, so
        # at parity a toy's 0.8 was inert (measured, a 0.3 m/s nudge on the
        # floor: 0.41 cm, the mu = 1.0 prediction 0.46; at priority 1,
        # 0.52 cm, the mu = 0.8 prediction 0.57). The feet are priority 1
        # too, so a foot on a toy stays the max of the two (1.0). Only the
        # sliding value does anything on this condim-3 geom; the torsional
        # and rolling entries are ignored (they need condim 4 / 6, as the
        # ball's rolling does above) and are left at MuJoCo's defaults.
        body.add_geom(name=f"{t.id}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                      mass=k["mass"], group=PICKABLE_GROUP, rgba=list(k["rgba"]),
                      priority=1, friction=[0.8, 0.005, 0.0001])
    if scenario.basket is not None:
        b = scenario.basket
        bx, by = b.pos
        sx, sy = b.size[0] / 2, b.size[1] / 2
        th = 0.006
        w.add_geom(name="basket_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, th],
                   pos=[bx, by, th], group=0, rgba=[0.55, 0.42, 0.25, 1.0])
        for i, (px, py, hx_, hy_) in enumerate((
                (bx, by - sy, sx, th), (bx, by + sy, sx, th), (bx - sx, by, th, sy), (bx + sx, by, th, sy))):
            w.add_geom(name=f"basket_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx_, hy_, b.rim / 2],
                       pos=[px, py, b.rim / 2], group=0, rgba=[0.6, 0.47, 0.3, 1.0])
        # A marker the detector can find (the "basket" class): a small
        # non-colliding body at the rim's centre, so re-acquiring the basket
        # across the room is a detection, not dead reckoning.
        mk = w.add_body(name="basket_marker", pos=[bx, by, b.rim + 0.02])
        mk.add_geom(name="basket_marker_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                    contype=0, conaffinity=0, group=1, rgba=[0.2, 0.9, 0.5, 1.0])
    for i, person in enumerate(scenario.persons):
        if person.kind == "g1":
            from ..robots.g1 import g1_spec
            robot = g1_spec()
            frame = w.add_frame(pos=[person.pos[0], person.pos[1], 0.0],
                                quat=_yaw_quat(person.yaw))
            spec.attach(robot, prefix=duck_prefix(person.id), frame=frame)
            continue
        # A mocap body: the world moves it kinematically (data.mocap_pos /
        # mocap_quat); ducks collide with it like a wall that walks. A capsule
        # standing on the floor, its "chest" at duck-head height.
        body = w.add_body(name=person.id, mocap=True,
                          pos=[person.pos[0], person.pos[1], person.height / 2],
                          quat=_yaw_quat(person.yaw))
        half = max(person.height / 2 - person.radius, 0.01)
        body.add_geom(name=f"{person.id}_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[person.radius, half, 0], group=0,
                      rgba=[0.35, 0.55, 0.85, 1.0])
    for duck in scenario.ducks:
        robot = mujoco.MjSpec.from_file(str(robot_xml))
        if scenario.collision != "walk":
            _pin_mass_properties_to_walk(robot)
        split_jaw(robot)
        x, y, yaw = duck.spawn
        frame = w.add_frame(pos=[x, y, 0.0], quat=_yaw_quat(yaw))
        spec.attach(robot, prefix=duck_prefix(duck.id), frame=frame)
    # Grasp = attachment: one INACTIVE weld per (duck, pickable). The world
    # sets its relative pose and switches it on when a beak closes on a toy.
    # And one contact EXCLUDE per pair: the beak is a gripper whose soft
    # mouth closes AROUND a toy, which a rigid convex hull cannot - under
    # "all" the jaw's hull sat on top of a 4 cm block and held the mouth tip
    # 2 cm above it, the grasp missed by 3.5 cm against a 4 cm tolerance,
    # and the toy-behind-the-basket pick went from 8/8 seeds to 3/8
    # (measured, tests/test_tidy.py). The jaw still meets the floor, the
    # boards, the ball, the basket, persons and other ducks.
    for duck in scenario.ducks:
        for t in scenario.pickables:
            spec.add_equality(name=f"{duck.id}/hold/{t.id}", type=mujoco.mjtEq.mjEQ_WELD,
                              objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=f"{duck_prefix(duck.id)}jaw_soft", name2=t.id, active=False)
            ex = spec.add_exclude(name=f"{duck.id}/mouth/{t.id}")
            ex.bodyname1, ex.bodyname2 = f"{duck_prefix(duck.id)}jaw_soft", t.id
    model = spec.compile()
    for duck in scenario.ducks:
        if duck.team:
            paint_team(model, duck.id, duck.team)
    # Feet win the friction pair, as the walk env sets for every model.
    for duck in scenario.ducks:
        for side in ("left", "right"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{duck_prefix(duck.id)}{side}_foot_collision")
            if gid >= 0:
                model.geom_priority[gid] = 1
            # The SOLE is the ground contact the walker was trained on. Under
            # "all" the shoe shell (`foot_left` / `foot_right`, in the same
            # ankle body) is a collision mesh too, and its convex hull comes
            # to 2 mm above the sole's underside - so a foot that tilts or
            # sinks its millimetre into the soft floor stands on the shell
            # as well, and the walker's flat-floor trajectory drifted 16 cm
            # in 10 s from the walk model's (measured, seeded shoves; and
            # tests/test_arena.py's step-for-step lock against the walk env
            # broke at 3e-9). The shell keeps upstream's self-collision bits
            # (contype/conaffinity 2, `robot_walk.xml`'s
            # `self_collision_only` class): it meets other ducks' shells and
            # legs, never the floor, the boards, the ball or a toy - those
            # meet the sole, exactly as in the walk model. With this the
            # walker is bit-identical under "walk" and "all" on a flat floor
            # (tests/test_world.py).
            ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      f"{duck_prefix(duck.id)}ankle_{side}")
            for g in range(model.ngeom):
                if g != gid and model.geom_bodyid[g] == ankle and (model.geom_contype[g] or model.geom_conaffinity[g]):
                    model.geom_contype[g] = model.geom_conaffinity[g] = 2
    return model


@dataclass(frozen=True)
class DuckAddress:
    """Where one duck lives inside a composed model: the addresses the obs
    builder, the actuator write and the sensors need, resolved by name once."""

    id: str
    prefix: str
    trunk_body: int
    root_qpos: int          # freejoint qpos start: [x y z qw qx qy qz]
    root_qvel: int          # freejoint dof start: [vx vy vz wx wy wz]
    joint_qpos: np.ndarray  # (14,) in contract order
    joint_qvel: np.ndarray  # (14,)
    actuators: np.ndarray   # (14,) ctrl indices in contract order
    gyro_adr: int           # sensordata start of imu_ang_vel (3)
    foot_geoms: tuple[int, int]
    tof_site: int
    # The 15th servo. -1 on a model built before `split_jaw`, so a caller
    # that wants the beak checks rather than indexing ctrl with -1.
    mouth_act: int = -1
    mouth_qpos: int = -1

    @classmethod
    def resolve(cls, model: mujoco.MjModel, duck_id: str) -> "DuckAddress":
        p = duck_prefix(duck_id)

        def jid(name: str) -> int:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p + name)
            if j < 0:
                raise KeyError(f"duck {duck_id!r}: joint {p + name!r} not in model")
            return j

        root = jid("trunk_base_freejoint")
        joints = [jid(n) for n in C.JOINT_NAMES]
        acts = []
        for n in C.JOINT_NAMES:
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, p + n)
            if a < 0:
                raise KeyError(f"duck {duck_id!r}: actuator {p + n!r} not in model")
            acts.append(a)
        gyro = model.sensor(p + "imu_ang_vel")
        return cls(
            id=duck_id, prefix=p,
            trunk_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p + "trunk_base"),
            root_qpos=int(model.jnt_qposadr[root]),
            root_qvel=int(model.jnt_dofadr[root]),
            joint_qpos=np.array([model.jnt_qposadr[j] for j in joints]),
            joint_qvel=np.array([model.jnt_dofadr[j] for j in joints]),
            actuators=np.array(acts),
            gyro_adr=int(gyro.adr[0]),
            foot_geoms=(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p + "left_foot_collision"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, p + "right_foot_collision"),
            ),
            tof_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, p + "tof"),
            mouth_act=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, p + MOUTH_BODY),
            mouth_qpos=(int(model.jnt_qposadr[mouth_j])
                        if (mouth_j := mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, p + MOUTH_BODY)) >= 0 else -1),
        )


def spawn_duck(model: mujoco.MjModel, data: mujoco.MjData, adr: DuckAddress,
               x: float, y: float, yaw: float, z: float = 0.12,
               pose: np.ndarray | None = None) -> None:
    """Put a duck at (x, y, yaw) in the STAND pose, at rest, servos holding.
    Mirrors what the walk env's reset does from its keyframe. The caller runs
    mj_forward when it is done spawning."""
    q = adr.root_qpos
    data.qpos[q:q + 3] = [x, y, z]
    data.qpos[q + 3:q + 7] = _yaw_quat(yaw)
    pose = C.DEFAULT_POSE if pose is None else pose
    data.qpos[adr.joint_qpos] = pose
    data.qvel[adr.root_qvel:adr.root_qvel + 6] = 0.0
    data.qvel[adr.joint_qvel] = 0.0
    data.ctrl[adr.actuators] = pose
    if adr.mouth_qpos >= 0:
        # A duck spawns with its beak SHUT. Zero is 5 degrees open on this
        # joint (the travel runs -5..+30), which is a visibly ajar bill in
        # every first frame until the servo pulls it closed.
        data.qpos[adr.mouth_qpos] = MOUTH_CLOSED
        data.ctrl[adr.mouth_act] = MOUTH_CLOSED
