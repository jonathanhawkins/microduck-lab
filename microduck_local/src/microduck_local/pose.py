"""Posing a robot for the 🎬 animation editor: forward kinematics, balance,
and inverse kinematics — for ANY body the training stack knows (robots/spec).

`PoseScratch` is a model/data pair used only to answer the editor's
requests. It is deliberately not any live duck's env: the editor previews an
arbitrary authored pose on every slider tick, and writing qpos into a duck
mid-episode would corrupt the very rollout the viewer is streaming. One
extra MjData per robot costs nothing, and `mj_forward` here measures ~0.03 ms
on the duck's 17-body model, ~0.1 ms on the G1's 45.

Three questions it answers, one method each:

- `solve`     — where is every body for these joint angles (POST /pose)
- `balance`   — where does the centre of mass sit against the soles
- `solve_ik`  — which joint angles put a foot / hand THERE (POST /ik): a
                damped-least-squares solve on the limb's own chain, joint
                limits respected, the other feet held where they are.

This grew out of `viz_server.PoseScratch`, which was the duck: it resolved
`left_foot_collision` by name, required it to be a mesh, and labelled the
joints "left leg / head + neck / right leg" by position. Everything here is
read off the `RobotSpec` instead, which is how the G1 (seven foot capsules a
side, no head joints, 29 joints in four groups) gets the same editor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from .robots.spec import RobotSpec

# --------------------------------------------------------------- geometry

_CIRCLE = 12   # samples around a capsule / sphere end when tracing its footprint


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain on (n, 2) points: counter-clockwise, no repeated
    endpoint. Fewer than three distinct points come back as they are."""
    pts = np.unique(pts, axis=0)          # sorted by x, then y
    if len(pts) < 3:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def chain(seq):
        out: list[np.ndarray] = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = chain(pts), chain(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _signed_distance(p: np.ndarray, hull: np.ndarray) -> float:
    """Distance from `p` to the nearest edge of a counter-clockwise convex
    polygon, positive inside. A hull of fewer than three points has no
    inside, so the distance to it is always negative."""
    a = hull
    if len(hull) == 0:
        return -math.inf
    e = np.roll(hull, -1, axis=0) - a                # edge vectors a -> b
    ap = p - a
    inside = len(hull) >= 3 and bool(np.all(e[:, 0] * ap[:, 1] - e[:, 1] * ap[:, 0] >= 0))
    ee = np.einsum("ij,ij->i", e, e)
    t = np.clip(np.einsum("ij,ij->i", ap, e) / np.where(ee > 0, ee, 1.0), 0.0, 1.0)
    best = float(np.linalg.norm(ap - t[:, None] * e, axis=1).min())
    return best if inside else -best


def _over(feet: dict) -> str | None:
    """The grounded foot whose footprint holds the CoM, the deeper one if
    both do, else None. A foot in the air cannot be stood on however well
    the CoM lines up with it."""
    standing = [s for s, f in feet.items() if f["grounded"] and f["marginMm"] > 0]
    return max(standing, key=lambda s: feet[s]["marginMm"]) if standing else None


def _outline(hull: np.ndarray) -> list[list[float]]:
    """A hull as the wire wants it: world xy in metres, counter-clockwise,
    the last point NOT repeated (the viewer closes the loop)."""
    return [[round(float(x), 4), round(float(y), 4)] for x, y in hull]


def _sole_points_world(model, data, g: int, sole_tol: float) -> np.ndarray:
    """World-frame points on the UNDERSIDE of one collision geom, with the
    data posed at STAND — what a flat floor would touch.

    A mesh sole (the duck) is the vertices within `sole_tol` of its lowest
    point. A capsule lying flat (the G1's seven a foot) prints a stadium: a
    circle of its radius under each end. Spheres print a circle, boxes their
    four lowest corners; anything else falls back to its AABB's underside,
    which only ever over-reads the footprint, never under-reads it.
    """
    import mujoco

    t = int(model.geom_type[g])
    R = data.geom_xmat[g].reshape(3, 3)
    c = data.geom_xpos[g]
    size = model.geom_size[g]
    ring = np.array([[math.cos(a), math.sin(a), 0.0]
                     for a in np.linspace(0, 2 * math.pi, _CIRCLE, endpoint=False)])
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[g]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = model.mesh_vert[adr:adr + num] @ R.T + c
        flat = world[:, 2] <= world[:, 2].min() + sole_tol
        return world[flat]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, half = float(size[0]), float(size[1])
        axis = R[:, 2] * half
        pts = []
        for end in (c - axis, c + axis):
            pts.append(end + ring * r + np.array([0.0, 0.0, -r]))
        return np.concatenate(pts)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
        return c + ring * r + np.array([0.0, 0.0, -r])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)]) * size[:3]
        world = corners @ R.T + c
        return world[world[:, 2] <= world[:, 2].min() + sole_tol]
    aabb = model.geom_aabb[g]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)]) * aabb[3:] + aabb[:3]
    world = corners @ R.T + c
    return world[world[:, 2] <= world[:, 2].min() + sole_tol]


# ------------------------------------------------------------------ IK types

@dataclass(frozen=True)
class IkTarget:
    """One effector's goal for `solve_ik`, in the same world frame the
    `bodies` payload of `solve` reports (grounded, root as the STAND
    keyframe put it)."""
    pos: tuple[float, float, float]
    # Keep the effector body as level as it stands at STAND (a foot flat on
    # the floor): its standing up-vector is held pointing at world +z.
    level: bool = False
    weight: float = 1.0


@dataclass
class IkResult:
    joints: np.ndarray
    iterations: int
    # Residual per effector after the solve, metres (position error).
    residual: dict[str, float] = field(default_factory=dict)
    # True when every target was reached within `tol`.
    converged: bool = False


class PoseScratch:
    """See the module docstring."""

    # --- IK tuning ---------------------------------------------------------
    IK_ITERS = 80
    IK_TOL = 1e-3            # m — reached; scaled by the body's size below
    # Damping in metres-per-radian, scaled by the body's size below: too
    # small and a stretched limb whips through the singular pose, too large
    # and every drag lags behind the pointer.
    IK_DAMPING = 0.02
    # Largest joint move per iteration, rad. Bounds the linearisation.
    IK_MAX_STEP = 0.25
    # How readily the free root translates when several limbs are goals
    # (see solve_ik): 1.0 = as cheaply as a joint, 0 = never.
    IK_ROOT_WEIGHT = 0.3
    # Weight of a `level` row against a position row (rad of tilt vs m of
    # gap). SOFT on purpose: a foot pinned level under a body leaning its
    # weight over it runs the ankle roll into its ±0.26 rad limit, and at
    # equal weight the least-squares compromise SLID the pinned foot 5 cm to
    # buy back tilt it could not have. Position is the promise; level is the
    # preference.
    IK_LEVEL_WEIGHT = 0.2

    def __init__(self, spec: RobotSpec | None = None) -> None:
        import mujoco

        self.mj = mujoco
        self.spec = spec = spec or C.MICRODUCK
        if spec.id == "microduck":
            # The viewer's own scene (with the split jaw), because /pose
            # hands back one pose PER SCENE BODY and the viewer zips the two
            # lists positionally — the raw walk scene is one body short.
            from .world.compose import scene_model
            self.model = scene_model()
        else:
            self.model = mujoco.MjModel.from_xml_path(str(spec.scene_fn()))
        m = self.model
        self.data = mujoco.MjData(m)
        self.joint_ids = np.array([m.joint(n).id for n in spec.joint_names])
        self.joint_qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.joint_dof_adr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.limits = np.array([m.jnt_range[j] for j in self.joint_ids], dtype=np.float64)
        self.base_body = int(m.body(spec.base_body).id)
        self.robot_geoms = np.array(
            [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0])
        # How big this body is against the duck the editor was built around:
        # the lab-stage pitch is a measured width ratio (RobotSpec).
        self.size_scale = float(spec.lab_spacing_m / 0.65)
        mujoco.mj_resetDataKeyframe(m, self.data, m.key(spec.stand_keyframe).id)
        mujoco.mj_forward(m, self.data)
        self.base_qpos = self.data.qpos.copy()
        self.stand_height = float(self.data.xpos[self.base_body][2])
        # Grounding is measured RELATIVE to the standing pose, so the AABB
        # bound's conservatism cancels exactly at the default pose instead of
        # leaving the preview robot hovering a few mm off the floor.
        self.stand_low_z = self._lowest_z()
        self.stand_anchor_xy: np.ndarray | None = None      # set below
        # The footprint of each foot: every collision geom's underside at
        # STAND, kept in its geom frame and transformed per call, so however
        # the foot is turned the projected outline's corners are among them.
        self.soles: dict[str, list[tuple[int, np.ndarray]]] = {}
        # The foot body per side (for `flat`/`level` and the sole effector).
        self.foot_bodies: dict[str, int] = {}
        for side, names in spec.foot_geoms.items():
            parts = []
            for name in names:
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                if g < 0:
                    raise ValueError(f"{spec.id}: no geom named {name!r}")
                world = _sole_points_world(m, self.data, g, spec.sole_tol)
                R = self.data.geom_xmat[g].reshape(3, 3)
                local = (world - self.data.geom_xpos[g]) @ R
                parts.append((g, local))
                self.foot_bodies[side] = int(m.geom_bodyid[g])
            self.soles[side] = parts
        # Effectors: body id, chain (joint indices root-ward), local point.
        self.effectors: dict[str, dict] = {}
        for e in spec.effectors:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, e.body)
            if b < 0:
                raise ValueError(f"{spec.id}: effector {e.id!r} names no body {e.body!r}")
            if e.point == "sole":
                side = next((s for s, fb in self.foot_bodies.items() if fb == b), None)
                if side is None:
                    raise ValueError(f"{spec.id}: effector {e.id!r} asks for a sole "
                                     f"but {e.body!r} carries no foot geom")
                world = np.concatenate([
                    pts @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]
                    for g, pts in self.soles[side]])
                hull = _convex_hull(world[:, :2])
                centre = np.array([*hull.mean(axis=0), float(world[:, 2].min())])
                Rb = self.data.xmat[b].reshape(3, 3)
                point = (centre - self.data.xpos[b]) @ Rb
            elif e.point is None:
                point = np.zeros(3)
            else:
                point = np.asarray(e.point, dtype=np.float64)
            # The body-frame direction that points UP at STAND: what `level`
            # holds. A foot body's own +z is not world-up on every model (the
            # duck's ankle frame is tilted), and asking for that instead
            # dragged a pinned foot 8 cm across the floor trying to obey.
            up_local = self.data.xmat[b].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
            self.effectors[e.id] = {
                "id": e.id, "label": e.label, "kind": e.kind, "body": int(b),
                "bodyName": e.body, "point": point, "chain": self._chain(int(b)),
                "up": up_local,
            }

        # Where the feet stand at STAND: `solve` keeps the grounded soles'
        # centroid HERE (see _anchor), so the frame the editor draws in is
        # anchored to the floor contact rather than to the free-joint root.
        self.stand_anchor_xy = self._grounded_centroid_xy()
        # The centre of mass as a draggable point: an IK target on it is how
        # an animator asks for BALANCE ("weight over the left foot") without
        # guessing hip angles. Every joint may serve it; only its ground
        # projection (xy) is solved for.
        self.effectors["com"] = {
            "id": "com", "label": "centre of mass", "kind": "com",
            "body": self.base_body, "bodyName": spec.base_body,
            "point": np.zeros(3), "chain": list(range(spec.num_joints)),
            "up": np.array([0.0, 0.0, 1.0]),
        }

    # --- kinematics helpers ------------------------------------------------

    def _chain(self, body: int) -> list[int]:
        """Joint indices (spec order) on the path from `body` up to the base
        body, root-most first — the joints a target on `body` may move."""
        m = self.model
        by_body: dict[int, list[int]] = {}
        for i, j in enumerate(self.joint_ids):
            by_body.setdefault(int(m.jnt_bodyid[j]), []).append(i)
        out: list[int] = []
        b = body
        while b > 0 and b != self.base_body:
            out = by_body.get(b, []) + out
            b = int(m.body_parentid[b])
        return out

    def _lowest_z(self) -> float:
        """World z of the lowest point of the robot's geom AABBs (conservative
        — see stand_low_z for why that is fine)."""
        m, d = self.model, self.data
        g = self.robot_geoms
        rot = d.geom_xmat[g].reshape(-1, 3, 3)
        aabb = m.geom_aabb[g]                       # [cx cy cz hx hy hz], geom frame
        zrow = rot[:, 2, :]                         # world-z row of each geom's basis
        centers = d.geom_xpos[g, 2] + np.einsum("ij,ij->i", zrow, aabb[:, :3])
        extents = np.einsum("ij,ij->i", np.abs(zrow), aabb[:, 3:])
        return float(np.min(centers - extents))

    def clamp(self, joints: np.ndarray) -> np.ndarray:
        return np.clip(joints, self.limits[:, 0], self.limits[:, 1])

    def _pose(self, joints: np.ndarray, root_pitch: float) -> None:
        """Write a pose into `self.data` (root at the STAND keyframe's spot,
        pitched about +Y) and run forward kinematics. No grounding."""
        d = self.data
        d.qpos[:] = self.base_qpos
        d.qvel[:] = 0.0
        d.qpos[self.joint_qpos_adr] = joints
        half = float(root_pitch) / 2.0
        # Rotation about +Y (wxyz) — the sign convention the editor documents:
        # NEGATIVE = lean back.
        d.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
        self.mj.mj_forward(self.model, d)

    def _grounded_centroid_xy(self) -> np.ndarray:
        """Centroid of every grounded sole's footprint, world xy, for the
        pose in `self.data`."""
        d = self.data
        outlines, lows = {}, {}
        for side, parts in self.soles.items():
            world = np.concatenate([
                pts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, pts in parts])
            outlines[side] = world[:, :2]
            lows[side] = float(world[:, 2].min())
        floor = min(lows.values())
        down = [s for s in outlines if lows[s] <= floor + self.spec.ground_tol]
        return np.concatenate([outlines[s] for s in down]).mean(axis=0)

    def _anchor(self) -> None:
        """Slide the root in xy so the grounded soles' centroid sits where it
        does at STAND.

        The pelvis is the free joint, so a pose is stored relative to the
        pelvis — and an IK solve that shifts the weight over one foot moves
        the PELVIS, which drawn root-fixed looks like the feet sliding across
        the floor the other way. Anchoring the stance instead draws what a
        standing body does: the feet stay put and the hips move. A foot
        lifting off does re-anchor onto the other (a step-sized shift at the
        moment of lift), which is the stance really changing.
        """
        if self.stand_anchor_xy is None:
            return
        shift = self._grounded_centroid_xy() - self.stand_anchor_xy
        if float(np.abs(shift).max()) > 1e-6:
            self.data.qpos[0:2] -= shift
            self.mj.mj_forward(self.model, self.data)

    def _ground(self) -> float:
        """Drop the posed root so its lowest point sits where the standing
        pose's does. Returns the drop applied (m, positive = moved down)."""
        drop = self._lowest_z() - self.stand_low_z
        if abs(drop) > 1e-6:
            self.data.qpos[2] -= drop
            self.mj.mj_forward(self.model, self.data)
            return drop
        return 0.0

    def _bodies(self) -> list[list[float]]:
        d = self.data
        return [[round(float(v), 4) for v in (*d.xpos[b], *d.xquat[b])]
                for b in range(self.model.nbody)]

    def solve(self, joints: np.ndarray, root_pitch: float = 0.0,
              ground: bool = True) -> list[list[float]]:
        """Forward kinematics for an authored pose → the WS frame's body payload."""
        self._pose(joints, root_pitch)
        if ground:
            self._ground()
            self._anchor()
        return self._bodies()

    def effector_point(self, eid: str) -> np.ndarray:
        """World position of an effector for the pose in `self.data`."""
        e = self.effectors[eid]
        b = e["body"]
        if e["kind"] == "com":
            return self.data.subtree_com[b].copy()
        return self.data.xmat[b].reshape(3, 3) @ e["point"] + self.data.xpos[b]

    def effector_positions(self) -> dict[str, list[float]]:
        return {eid: [round(float(v), 4) for v in self.effector_point(eid)]
                for eid in self.effectors}

    # --- balance -----------------------------------------------------------

    def balance(self) -> dict:
        """Where the CoM sits relative to each sole, for the POSED state in
        `self.data` (call straight after `solve`).

        Per foot, `marginMm` is the signed distance from the CoM's ground
        projection to the edge of that sole's footprint: positive inside it,
        negative outside. The footprint is the flat of the sole projected onto
        the world ground plane (its convex hull), so a yawed foot keeps its
        true outline. `grounded` is false for a foot held clear of the floor,
        and `over` names the grounded foot whose footprint contains the CoM.

        `support` is the same read against the SUPPORT POLYGON: the convex
        hull of every grounded sole's footprint — the region a body stands in
        statically. A STATIC check, no velocity, no momentum, no ankle
        torque: read it as "how hard is this pose to hold", never as a
        verdict.
        """
        d = self.data
        com = d.subtree_com[0]
        soles = {}
        for side, parts in self.soles.items():
            world = np.concatenate([
                pts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, pts in parts])
            soles[side] = (float(world[:, 2].min()), _convex_hull(world[:, :2]))
        floor = min(low for low, _ in soles.values())
        feet = {}
        for side, (low, hull) in soles.items():
            feet[side] = {
                "grounded": bool(low <= floor + self.spec.ground_tol),
                "marginMm": round(_signed_distance(com[:2], hull) * 1000, 1),
                "outline": _outline(hull),
            }
        down = [side for side in soles if feet[side]["grounded"]]
        stance = _convex_hull(np.concatenate([soles[side][1] for side in down]))
        support = {
            "feet": down,
            "marginMm": round(_signed_distance(com[:2], stance) * 1000, 1),
            "outline": _outline(stance),
        }
        return {"com": [round(float(v), 4) for v in com], "feet": feet,
                "over": _over(feet), "support": support}

    # --- inverse kinematics ------------------------------------------------

    def solve_ik(self, joints: np.ndarray, root_pitch: float,
                 targets: dict[str, IkTarget], pins: tuple[str, ...] = (),
                 ground: bool = True, iters: int | None = None) -> IkResult:
        """Joint angles that put each effector at its target.

        Damped least squares (Levenberg–Marquardt on the task Jacobian): each
        iteration stacks every target's position error (and, for `level`
        targets, the tilt of the body's +z away from world up) against the
        point Jacobian from `mj_jac`, restricted to the joints on the
        effectors' own chains, and steps

            dq = Jᵀ (J Jᵀ + λ² I)⁻¹ e

        clamped per joint to IK_MAX_STEP and to the MJCF limits. The root
        starts where `solve` grounded and anchored it for the INPUT pose, so a
        target is given in the frame the previous /pose answered in; the
        result is re-grounded and re-anchored on the way out (see _anchor),
        which can shift the whole body when the pose changes which foot is
        down — that is the stance changing, not the solver missing.

        `pins` name effectors to hold at their current world position (the
        planted foot, while the other one is dragged). Every joint not on a
        moving chain is left exactly as it was.
        """
        mj = self.mj
        m, d = self.model, self.data
        q = self.clamp(np.asarray(joints, dtype=np.float64).copy())
        self._pose(q, root_pitch)
        if ground:
            self._ground()
            self._anchor()
        root = d.qpos[:7].copy()
        goals: dict[str, IkTarget] = {}
        for eid in pins:
            if eid in self.effectors and eid not in targets:
                p = self.effector_point(eid)
                goals[eid] = IkTarget(pos=(float(p[0]), float(p[1]), float(p[2])),
                                      level=self.effectors[eid]["kind"] == "foot")
        goals.update(targets)
        unknown = [eid for eid in goals if eid not in self.effectors]
        if unknown:
            raise KeyError(f"{self.spec.id}: no effector named {unknown[0]!r} "
                           f"(have {sorted(self.effectors)})")
        cols = sorted({i for eid in goals for i in self.effectors[eid]["chain"]})
        result = IkResult(joints=q, iterations=0)
        if not cols or not goals:
            return result
        dof = self.joint_dof_adr[cols]
        # The pelvis IS the free joint. With two feet pinned and the root
        # held, the legs are a closed chain and nothing below the waist can
        # move — a "weight over the left foot" request then stalls 6 mm short,
        # served by the arms alone (the root trap, met again). So whenever the
        # goals span more than one limb, the root's TRANSLATION is a variable
        # too: the pelvis shifts, the pinned legs re-bend under it, and the
        # solved joints describe the same pose relative to the feet. A single
        # unpinned target keeps the root still, or a dragged foot would slide
        # the whole body instead of bending the leg.
        root_free = (len(goals) > 1 and m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE)
        colw = np.ones(len(dof))
        if root_free:
            dof = np.concatenate([np.array([0, 1, 2]), dof])
            ncols = len(cols)
            # Root motion is the EXPENSIVE way to reach: weighting its columns
            # down makes the minimum-norm step prefer bending a leg to sliding
            # the pelvis, which is what keeps a far target from dragging the
            # planted foot along (measured: 6 cm of pinned-foot drift at full
            # weight on a 0.6 m kick, 0 at this one).
            colw = np.concatenate([np.full(3, self.IK_ROOT_WEIGHT), np.ones(ncols)])
        lam2 = (self.IK_DAMPING * self.size_scale) ** 2
        tol = self.IK_TOL * self.size_scale
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        n_it = self.IK_ITERS if iters is None else int(iters)
        it = 0
        for it in range(1, n_it + 1):
            rows, errs = [], []
            pos_errs = []
            for eid, t in goals.items():
                e = self.effectors[eid]
                b = e["body"]
                p = self.effector_point(eid)
                w = float(t.weight)
                gap = np.asarray(t.pos, dtype=np.float64) - p
                if e["kind"] == "com":
                    mj.mj_jacSubtreeCom(m, d, jacp, b)
                    rows.append(w * jacp[:2, dof])
                    errs.append(w * gap[:2])
                    pos_errs.append(float(np.linalg.norm(gap[:2])))
                    continue
                mj.mj_jac(m, d, jacp, jacr, p, b)
                rows.append(w * jacp[:, dof])
                errs.append(w * gap)
                pos_errs.append(float(np.linalg.norm(gap)))
                if t.level:
                    up = d.xmat[b].reshape(3, 3) @ e["up"]
                    # d(up)/dq = ω × up  →  −[up]× · Jr ; only the xy tilt matters.
                    skew = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]],
                                     [-up[1], up[0], 0]])
                    wl = w * self.IK_LEVEL_WEIGHT
                    rows.append(wl * (-skew @ jacr)[:2, dof])
                    errs.append(wl * (np.array([0.0, 0.0, 1.0]) - up)[:2])
            J = np.concatenate(rows)
            err = np.concatenate(errs)
            # Convergence is judged on POSITION: `level` rows are soft (an
            # ankle at its limit cannot always oblige) and must not keep the
            # loop grinding once every point is where it was asked to be.
            worst = max(pos_errs)
            if worst < tol:
                break
            # Levenberg-style damping: full when the target is far (a
            # stretched limb must not whip through its singular pose), easing
            # off over the last few centimetres so the solve finishes instead
            # of crawling — measured, a two-foot-pinned weight shift stalled
            # 6 mm short at fixed damping, and reaches 1 mm with this.
            ease = 0.05 + 0.95 * min(1.0, worst / (0.05 * self.size_scale))
            Jw = J * colw
            dq = colw * (Jw.T @ np.linalg.solve(Jw @ Jw.T + lam2 * ease * np.eye(len(err)), err))
            # Active limits: a joint parked on a stop and asked to go through
            # it contributes nothing, so drop its column and solve again —
            # otherwise the step it cannot take is charged to the others as
            # position error (the same 5 cm slide as above).
            jd = dq[3:] if root_free else dq
            lo, hi = self.limits[cols, 0], self.limits[cols, 1]
            stuck = ((q[cols] <= lo + 1e-9) & (jd < 0)) | ((q[cols] >= hi - 1e-9) & (jd > 0))
            if stuck.any():
                colw2 = colw.copy()
                colw2[(3 if root_free else 0) + np.flatnonzero(stuck)] = 0.0
                Jw = J * colw2
                dq = colw2 * (Jw.T @ np.linalg.solve(
                    Jw @ Jw.T + lam2 * ease * np.eye(len(err)), err))
            biggest = float(np.abs(dq).max())
            if biggest > self.IK_MAX_STEP:
                dq *= self.IK_MAX_STEP / biggest
            if root_free:
                root[:3] += dq[:3]
                dq = dq[3:]
                assert len(dq) == ncols
            q[cols] += dq
            q = self.clamp(q)
            d.qpos[:7] = root
            d.qpos[self.joint_qpos_adr] = q
            mj.mj_forward(m, d)
        residual = {}
        for eid, t in goals.items():
            p = self.effector_point(eid)
            gap = np.asarray(t.pos, dtype=np.float64) - p
            if self.effectors[eid]["kind"] == "com":
                gap = gap[:2]
            residual[eid] = round(float(np.linalg.norm(gap)), 5)
        result.joints = q
        result.iterations = it
        result.residual = residual
        result.converged = all(r <= tol for r in residual.values())
        return result

    # --- metadata ----------------------------------------------------------

    def meta(self) -> dict:
        """Everything the editor needs to build clamped controls, map a
        clicked body back to the joint that moves it, and offer this body's
        rig controls and IK handles."""
        m = self.model
        spec = self.spec
        groups = spec.joint_groups or ("joints",) * spec.num_joints
        joints = []
        for i, name in enumerate(spec.joint_names):
            j = m.joint(name)
            body = int(m.jnt_bodyid[j.id])
            joints.append({
                "index": i,
                "name": name,
                "group": groups[i],
                "min": round(float(self.limits[i, 0]), 6),
                "max": round(float(self.limits[i, 1]), 6),
                "default": round(float(spec.default_pose[i]), 6),
                "body": body,
                "bodyName": m.body(body).name,
                # Hinge axis and anchor in the BODY frame — the viewer turns a
                # screen drag into a joint delta with these.
                "axis": [round(float(v), 6) for v in m.jnt_axis[j.id]],
                "pos": [round(float(v), 6) for v in m.jnt_pos[j.id]],
            })
        seen: list[str] = []
        for g in groups:
            if g not in seen:
                seen.append(g)
        return {
            "robot": spec.id,
            "title": spec.title or spec.id,
            "numJoints": spec.num_joints,
            "joints": joints,
            "groups": seen,
            "bodies": [m.body(b).name for b in range(m.nbody)],
            "trunkBody": self.base_body,
            # Standing height of the base body (m) and the body's size against
            # the duck (a width ratio): the viewer scales its gizmos by these.
            "standHeight": round(self.stand_height, 4),
            "sizeScale": round(self.size_scale, 4),
            # An EDITOR hint, not a validation bound: a flip is a continuous
            # rotation, so a clip may legitimately carry a full turn (the
            # backflip recipe runs to ±2π) and the clip contract never clamps
            # rootPitch. Two turns of slider travel covers either direction.
            "rootPitchRange": [-round(float(2 * np.pi), 6), round(float(2 * np.pi), 6)],
            # Restated here so a client never has to guess (docs own the why).
            "rootPitchSign": "negative = lean back (gravity gains -x in trunk frame)",
            "effectors": [
                {"id": e["id"], "label": e["label"], "kind": e["kind"],
                 "body": e["body"], "bodyName": e["bodyName"],
                 "point": [round(float(v), 5) for v in e["point"]],
                 "chain": list(e["chain"])}
                for e in self.effectors.values()],
            "rig": [dict(c) for c in spec.rig_controls],
        }


_scratch: dict[str, PoseScratch] = {}


def pose_scratch(robot: str = "microduck") -> PoseScratch:
    """Lazily built, one per robot — the editor is optional, so a lab that
    never opens it never pays for the extra model, and a lab without the G1's
    assets only fails when the G1 is actually asked for."""
    key = robot or "microduck"
    if key in ("duck", ""):
        key = "microduck"
    if key not in _scratch:
        from .robots import spec as S
        body = S.get(key)
        if not hasattr(body, "base_body"):
            # The editor is a WALKER's tool: it reads a base body, effectors,
            # soles and a stand height, none of which a wheeled body declares.
            # The lab's `/robots` list offers every registry entry, and a
            # non-walker (MARS, docs/mars-roadmap.md Phase 2b) would otherwise
            # arrive here and die on `spec.base_body` as a 500. KeyError is
            # what `viz_server.scratch_for` turns into a 404 with the text.
            # Asked as a capability, not `isinstance(body, RobotSpec)`: the
            # G1 arrives as a lazy proxy that IS a RobotSpec once resolved
            # and fails isinstance until then (robots/body.py's docstring).
            raise KeyError(
                f"{key!r} has no animate support yet — the 🎬 editor poses "
                "walkers (a base body, effectors, soles); a wheeled body's "
                "arm editor is docs/mars-roadmap.md Phase 2b")
        _scratch[key] = PoseScratch(body)
    return _scratch[key]
