"""MARS's arm kinematics, on a PRIVATE model — where does the claw have to be?

Phase 4b's scripted pick answered "can the claw hold the block" with an IK
routine living inside `scripts/probe_mars_pick.py`'s `Rig`: coordinate descent
with restarts, over the five positioning joints, scoring the distance from the
blade-pad midpoint to a target and rejecting self-colliding poses. Phase 5's
`brain/tidy_arm.py` needs exactly that routine in a room, so the SOLVER lives
here and both callers use it — `solve_arm` below is the loop lifted out of
`Rig.solve` line for line (the probe now calls it), because two coordinate
descents with slightly different acceptance rules would be two different arms
and the 4b table would stop meaning anything.

**Two things about a brain doing IK, and both are constraints the probe never
had.**

* A brain gets `Senses` and returns `Intent` — it never touches physics
  (`brain/runtime.py`). Writing `qpos` into the live world's `MjData` and
  calling `mj_forward` to score a pose would corrupt the very simulation the
  answer is for. So `ArmKinematics` owns its OWN compiled MARS
  (`mars.model()`) and its own `MjData`, and solves there. That is not a
  workaround: the arm's kinematics are a property of the robot and not of the
  room, and on the real machine a planner reads the robot's own URDF exactly
  this way.
* In that private model the robot stands at the origin, so its BASE frame and
  its world frame coincide. Every target here is therefore in the base frame
  — which is also the frame the policy contract's `target_base` slot speaks
  (`robots/mars.OBS_TARGET_BASE`), so a scripted pick and a learned one aim
  at the same numbers.

MEASURED on this Mac (`tests/test_tidy_arm.py` re-measures the first two, so a
URDF revision that moves the arm mount fails there rather than quietly aiming
a brain at a point the arm no longer reaches):

    build (mars.model() + MjData)     0.10-0.13 s, once per brain
    shoulder, base frame              (0.0860, -0.0528, 0.0402) m
    grasp point at ARM_HOME           (0.0332, 0.1707, 0.2159) m  -- 79 deg
                                      to the LEFT of the nose, which is why
                                      ARM_HOME is a legal CARRY pose: a toy
                                      held there is outside the head camera's
                                      +-58 deg half-field (`mars.CAMERA_*`).
    solve_grasp at the pick standoff  0.3-0.5 mm residual, 60-90 ms with
                                      `restarts=3` (the brain's setting)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from . import mars

#: How far the blade pads must be from a target to call the solve a hit (m).
#: 4b's two misses of 16 were IK residuals of 13.7 and 15.1 mm and every spot
#: the claw was actually PUT on held, so the line sits between those two
#: populations rather than at a tolerance anybody chose.
IK_TOLERANCE_M = 0.010


def solve_arm(cost: Callable[[np.ndarray], tuple[float, float]],
              accept: Callable[[], bool],
              lo: np.ndarray, hi: np.ndarray, q6: float,
              rng: np.random.Generator,
              home: np.ndarray | None = None,
              start: Sequence[float] | None = None,
              restarts: int = 6, iters: int = 80,
              step0: float = 0.6) -> tuple[float, np.ndarray | None]:
    """Coordinate descent with restarts over the five positioning joints.

    `scratchpad/probe_box_reach2`'s solver — the one `ACTION_SCALE_RAD` was
    chosen with — so residuals from here are comparable with the shell numbers
    in `robots/mars_env.py`'s docstring and with 4b's scripted table.

    `cost(q)` must POSE the arm at `q` and return `(penalised, distance)`;
    `accept()` is then asked about the state `cost` left behind (the probe's
    "no bad contact"). `q6` is the jaw, held fixed: it is the grip, not a
    degree of freedom the reach may spend.

    Returns `(best distance, best q)` — `(1e9, None)` when every restart
    ended somewhere `accept()` refused.
    """
    best, best_q = 1e9, None
    for k in range(restarts):
        if k == 0:
            q = (np.array(home, float) if start is None else np.array(start, float))
        elif start is not None:
            q = np.array(start, float) + rng.normal(0.0, 0.25, 6)
        else:
            q = rng.uniform(lo, hi)
        q[5] = q6
        c, dist = cost(q)
        step = step0
        for _ in range(iters):
            improved = False
            for i in range(5):           # joint6 is the jaw, held fixed
                for s in (+step, -step):
                    t = q.copy()
                    t[i] = np.clip(t[i] + s, lo[i], hi[i])
                    c2, d2 = cost(t)
                    if c2 < c - 1e-6:
                        q, c, dist, improved = t, c2, d2, True
            if not improved:
                step *= 0.5
                if step < 2e-3:
                    break
        cost(q)
        if accept() and dist < best:
            best, best_q = dist, q.copy()
    return best, best_q


class ArmKinematics:
    """MARS's arm on a private model: where do the joints go for this point?

    Construct one per brain (it compiles a MARS), then ask it for joint
    vectors. It never reads or writes the caller's world.
    """

    def __init__(self, seed: int = 0):
        m = mars.model()
        self.m = m
        self.d = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)
        self.qadr = [int(m.joint(j).qposadr[0]) for j in mars.ARM_JOINTS]
        self.mimic_q = int(m.joint(mars.MIMIC_JOINT[0]).qposadr[0])
        self.lo = np.array([m.joint(j).range[0] for j in mars.ARM_JOINTS])
        self.hi = np.array([m.joint(j).range[1] for j in mars.ARM_JOINTS])
        self.home = np.array([mars.ARM_HOME[j] for j in mars.ARM_JOINTS])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mars.BASE_BODY)
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mars.EFFECTOR_BODY)
        self.floor_g = int(m.geom(mars.FLOOR_GEOM).id)
        self.wheels = {int(m.geom(g).id) for g in mars.WHEEL_GEOMS}
        self.fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                        for n in mars.FINGER_LINKS}
        # The blade PADS: the inner faces an object is actually pinched
        # between, and the reason the grasp point is not `ee_link` (they are
        # 7.9 mm apart, along the finger rather than across the jaw — 4b).
        self.pads: dict[str, list[int]] = {}
        for link in mars.FINGER_LINKS:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            self.pads[link] = [
                g for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == bid
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     ).rsplit("_", 1)[-1].startswith("pad")]
        mujoco.mj_resetDataKeyframe(m, self.d, m.key(mars.HOME_KEY).id)
        mujoco.mj_forward(m, self.d)
        #: The shoulder in the base frame — the shell's own origin, the same
        #: quantity `MarsArmEnv.shoulder_base` measures off the same model.
        self.shoulder = np.array(self.d.xanchor[m.joint("joint1").id]).copy()

    # ------------------------------------------------------------- geometry

    def _pads(self) -> list[np.ndarray]:
        return [np.mean([self.d.geom_xpos[g] for g in self.pads[link]], axis=0)
                for link in mars.FINGER_LINKS]

    def grasp_point(self) -> np.ndarray:
        """The midpoint between the blade pads — where an object is pinched."""
        a, b = self._pads()
        return (a + b) / 2.0

    def jaw_gap(self) -> float:
        a, b = self._pads()
        return float(np.linalg.norm(a - b))

    def place(self, q) -> np.ndarray:
        """Pose the private arm at `q` (clamped, joint2 guard applied)."""
        q = np.clip(np.asarray(q, float), self.lo, self.hi)
        q[1] = max(q[1], mars.joint2_min_target(q[0], float(self.lo[1])))
        for adr, v in zip(self.qadr, q):
            self.d.qpos[adr] = v
        self.d.qpos[self.mimic_q] = -q[5]
        mujoco.mj_forward(self.m, self.d)
        return q

    def bad_contact(self) -> bool:
        """The arm into itself or the floor — the wheels and the base may rest.

        The probe's predicate with the toy branch dropped: there is no object
        in this private model, so "is this pose legal for the ROBOT" is the
        whole question. The 10 mm depth is 4a's measurement — within +-0.1 rad
        of HOME 15.3 % of poses report a contact and 0.0 % are deeper than
        5 mm, because mars.urdf's `link1`/`link3` boxes overlap ~9 mm.
        """
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == b2 or {b1, b2} == self.fingers:
                continue
            if self.floor_g in (g1, g2):
                other = g2 if g1 == self.floor_g else g1
                if other in self.wheels or int(m.geom_bodyid[other]) == self.base:
                    continue
                return True
            if float(c.dist) < -0.010:
                return True
        return False

    # ------------------------------------------------------------- the shell

    def shell_radius(self, point_base) -> float:
        return float(np.linalg.norm(np.asarray(point_base, float) - self.shoulder))

    def shell_yaw(self, point_base) -> float:
        v = np.asarray(point_base, float) - self.shoulder
        return float(math.atan2(v[1], v[0]))

    def shell_point(self, radius: float, yaw: float, z: float) -> np.ndarray:
        """The shell point at (`radius`, `yaw`) on the plane `z`, base frame.

        `MarsArmEnv._shell_point`'s geometry: `radius` is the 3-D distance
        from the shoulder, so the horizontal offset shrinks with height — the
        shell is a shell and not a cylinder.
        """
        dz = z - float(self.shoulder[2])
        horiz = math.sqrt(max(radius * radius - dz * dz, 1e-6))
        return self.shoulder + np.array(
            [horiz * math.cos(yaw), horiz * math.sin(yaw), dz])

    # ------------------------------------------------------------------- IK

    def solve_grasp(self, target_base, q6: float, start=None,
                    restarts: int = 3, point: str = "grasp"
                    ) -> tuple[float, np.ndarray | None]:
        """Joints that put the GRASP POINT (or `point="ee"`, the tool frame)
        on a base-frame target with the jaw at `q6`.

        `restarts` defaults to 3 rather than the probe's 6: MEASURED on the
        pick standoff the residual is the same 0.3-0.5 mm and the solve is
        60-90 ms instead of 150-200, and a brain pays this inside one 50 Hz
        tick (a planning pause the real robot would have too, and which
        `brain/tidy_arm.py` states in its docstring rather than hides).
        """
        target = np.asarray(target_base, float)

        def read() -> np.ndarray:
            return (self.grasp_point() if point == "grasp"
                    else np.array(self.d.xpos[self.ee]))

        def cost(q) -> tuple[float, float]:
            self.place(q)
            dist = float(np.linalg.norm(read() - target))
            return dist + (1.0 if self.bad_contact() else 0.0), dist

        return solve_arm(cost, lambda: not self.bad_contact(), self.lo, self.hi,
                         q6, self.rng, home=self.home, start=start,
                         restarts=restarts)

    def targets(self, q) -> dict[str, float]:
        """A joint vector as an `Intent.arm` mapping (`mars.ARM_JOINTS`)."""
        return {j: float(v) for j, v in zip(mars.ARM_JOINTS, np.asarray(q, float))}


__all__ = ["ArmKinematics", "IK_TOLERANCE_M", "solve_arm"]
