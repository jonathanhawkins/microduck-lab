"""LOOK at what a MARS pick policy does — and MEASURE the grasp under it.

    # the physics, before any reward exists (Phase 4b step 1)
    uv run python scripts/probe_mars_pick.py --scripted

    # the skill is expressible in the env's own action space
    uv run python scripts/probe_mars_pick.py --scripted-env

    # a trained policy, deterministically, with a contact sheet
    uv run python scripts/probe_mars_pick.py runs/mars-pick-r1/policy.onnx \
        --seeds 20 --out /tmp/mars-pick
    uv run python scripts/probe_mars_pick.py --null --seeds 20

`scripts/probe_mars_reach.py` is the same instrument for `reach`, and this
file borrows its sheet builder and camera rather than copying them (both live
in `scripts/`, so a plain import works when either is run as a script).

What a pick needs that a reach does not is a SCRIPTED mode. `AGENTS.md` says
"if rollouts never contain the skill you're paying for, fix the physics
curriculum, not the reward" — and the way to know whether the physics can
contain it at all is to do the thing by hand, with an IK solver instead of a
policy, before a weight is chosen. `--scripted` is that: it places the
playroom's block between the open blades, closes on Innate's own
0.6-rad-past-the-stop target, lifts 10 cm and holds 2 s, at several spots,
several timesteps and several `<option>` blocks. It answers three questions
that decided the env:

  * what timestep `pick` must run at (2/8 spots held at 5 ms, 7/8 at 2 ms),
  * whether Innate's elliptic cone + `impratio 10` are the lever (they are
    not: the world's own block scores the same 7/8 at 2 ms), and
  * which float `gripper_load` should carry (`qfrc_constraint`, which is
    0.0000 in every empty state and ~2 N*m with the block in the claw).

Offscreen rendering uses `mujoco.Renderer`, which on macOS picks the bundled
CGL backend with no `MUJOCO_GL` setting and no display. Set `MUJOCO_GL=egl`
(or `osmesa`) on Linux.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_mars_reach import add_sphere, build_sheet, make_camera  # noqa: E402

BLOCK_RGBA = (0.2, 1.0, 0.4, 0.55)
EE_RGBA = (1.0, 0.45, 0.1, 0.9)

#: Innate's close command: 0.6 rad past the mechanical stop. `set_arm` clamps
#: it to `GRIPPER_CLOSED_ON_AIR_RAD`, which still saturates the torque ceiling
#: — the blades squeeze, they just stop where the metal does.
CLOSE_TARGET_RAD = -0.60
#: How wide the jaws are opened for the scripted approach. MEASURED: joint6
#: 0.60 rad is a 58.8 mm gap between the blade pads, against the 40 mm block
#: and the 82 mm the jaws reach at their +0.873 stop.
OPEN_RAD = 0.60
SCRIPTED_LIFT_M = 0.10

#: The `<option>` blocks the scripted sweep compares. "mars" is
#: `mars._scene_spec`'s (Innate's own); "world" is what `world/compose.py`
#: gives a MARS in a room; "world+g1" is the same with the block that a G1 in
#: the room forces on it.
OPTION_BLOCKS = ("mars", "world", "world+g1")


# ------------------------------------------------------------- the scripted rig

def _build_scene(dt: float, option_block: str):
    """One MARS, a floor and the playroom block, under a named option block.

    The probe compiles its OWN model here rather than asking `mars_env`,
    because the whole question is what happens at a timestep and under a cone
    the env does not offer — a measurement tool that can only build the
    shipped configuration cannot say why the shipped configuration was chosen.
    """
    import mujoco

    from microduck_local.robots import mars as M
    from microduck_local.robots import mars_env as ME

    size, mass, rgba = ME.toy_spec()
    spec = M.robot_spec()
    spec.option.timestep = dt
    if option_block in ("mars", "world+g1"):
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if option_block == "mars":
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
    if option_block == "world+g1":
        spec.option.iterations = 10
        spec.option.ls_iterations = 20
    w = spec.worldbody
    light = w.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    floor = w.add_geom()
    floor.name = M.FLOOR_GEOM
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10.0, 10.0, 0.1]
    park = [ME.PICK_TOY_PARK[0], ME.PICK_TOY_PARK[1], ME.toy_rest_z()]
    body = w.add_body(name=ME.PICK_TOY_BODY, pos=park)
    body.add_freejoint(name=ME.PICK_TOY_JOINT)
    body.add_geom(name=f"{ME.PICK_TOY_BODY}_geom",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[v / 2 for v in size], mass=mass, rgba=list(rgba),
                  priority=1, friction=[0.8, 0.005, 0.0001])
    probe = M.robot_spec().compile()
    key = spec.add_key()
    key.name = M.HOME_KEY
    key.qpos = np.concatenate([M._home_qpos(probe), park, [1.0, 0, 0, 0]])
    key.qvel = np.zeros(probe.nv + 6)
    return spec.compile()


class Rig:
    """A MARS, a block and an IK solver — the hand that does the pick."""

    def __init__(self, dt: float, option_block: str = "mars"):
        import mujoco

        from microduck_local.robots import mars as M
        from microduck_local.robots.mars_drive import MarsDriver

        self.mj = mujoco
        self.M = M
        self.m = _build_scene(dt, option_block)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = [m.joint(j).qposadr[0] for j in M.ARM_JOINTS]
        self.mimic_q = m.joint(M.MIMIC_JOINT[0]).qposadr[0]
        self.lo = np.array([m.joint(j).range[0] for j in M.ARM_JOINTS])
        self.hi = np.array([m.joint(j).range[1] for j in M.ARM_JOINTS])
        self.home = np.array([M.ARM_HOME[j] for j in M.ARM_JOINTS])
        self.ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, M.EFFECTOR_BODY)
        self.toy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "toy")
        self.toy_q = int(m.joint("toy_free").qposadr[0])
        self.toy_d = int(m.joint("toy_free").dofadr[0])
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, M.BASE_BODY)
        self.wheels = {m.geom(g).id for g in M.WHEEL_GEOMS}
        self.floor_g = m.geom(M.FLOOR_GEOM).id
        self.fingers = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                        for n in M.FINGER_LINKS}
        # The blade PADS: the inner faces an object is actually pinched
        # between, and the reason the grasp point is not `ee_link` (they are
        # 7.9 mm apart, along the finger rather than across the jaw).
        self.pads = {}
        for link in M.FINGER_LINKS:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
            self.pads[link] = [
                g for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     ).rsplit("_", 1)[-1].startswith("pad")]
        self.j6_d = int(m.joint("joint6").dofadr[0])
        self.j6_q = int(m.joint("joint6").qposadr[0])
        self.driver = MarsDriver(m, "")
        mujoco.mj_resetDataKeyframe(m, self.d, m.key(M.HOME_KEY).id)
        mujoco.mj_forward(m, self.d)
        self.shoulder = np.array(self.d.xanchor[m.joint("joint1").id]).copy()
        self._cmd = dict(M.ARM_HOME)

    # -------------------------------------------------------------- geometry

    def _pads(self):
        return [np.mean([self.d.geom_xpos[g] for g in self.pads[link]], axis=0)
                for link in self.M.FINGER_LINKS]

    def grasp_point(self) -> np.ndarray:
        a, b = self._pads()
        return (a + b) / 2.0

    def jaw_gap(self) -> float:
        a, b = self._pads()
        return float(np.linalg.norm(a - b))

    def spot(self, radius: float, yaw: float, z: float) -> np.ndarray:
        dz = z - float(self.shoulder[2])
        horiz = float(math.sqrt(max(radius * radius - dz * dz, 1e-6)))
        return self.shoulder + np.array(
            [horiz * math.cos(yaw), horiz * math.sin(yaw), dz])

    # ------------------------------------------------------------------- IK

    def place(self, q):
        q = np.clip(q, self.lo, self.hi)
        q[1] = max(q[1], self.M.joint2_min_target(q[0], float(self.lo[1])))
        for adr, v in zip(self.qadr, q):
            self.d.qpos[adr] = v
        self.d.qpos[self.mimic_q] = -q[5]
        self.mj.mj_forward(self.m, self.d)
        return q

    def bad_contact(self, ignore_toy: bool = False) -> bool:
        """The arm into itself, the floor or the toy — the wheels may rest."""
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == b2 or {b1, b2} == self.fingers:
                continue
            if self.toy in (b1, b2):
                if ignore_toy:
                    continue
                return True
            if self.floor_g in (g1, g2):
                other = g2 if g1 == self.floor_g else g1
                if other in self.wheels or m.geom_bodyid[other] == self.base:
                    continue
                return True
            if float(c.dist) < -0.010:
                return True
        return False

    def solve(self, target, q6, rng, start=None, restarts=6, iters=80,
              step0=0.6, point="grasp", ignore_toy=False):
        """Coordinate descent with restarts — `scratchpad/probe_box_reach2`'s
        solver, which is the one `ACTION_SCALE_RAD` was chosen with, so the
        residuals here are comparable with the shell numbers in
        `robots/mars_env.py`'s docstring.

        The loop itself now lives in `robots/mars_ik.solve_arm`, because
        Phase 5's `brain/tidy_arm.py` needs the same routine in a room and two
        coordinate descents with slightly different acceptance rules would be
        two different arms (4b's table would stop meaning anything). This
        wrapper is what supplies the probe's own scene, its grasp point and
        its toy-aware contact rule. VERIFIED unchanged: the 2 ms / mars cell
        of `--scripted` reports the same residuals and the same HELD verdict
        before and after the move.
        """
        from microduck_local.robots.mars_ik import solve_arm

        def read():
            return (self.grasp_point() if point == "grasp"
                    else np.array(self.d.xpos[self.ee]))

        def cost(q):
            self.place(q)
            dist = float(np.linalg.norm(read() - target))
            return dist + (1.0 if self.bad_contact(ignore_toy) else 0.0), dist

        return solve_arm(cost, lambda: not self.bad_contact(ignore_toy),
                         self.lo, self.hi, q6, rng, home=self.home,
                         start=start, restarts=restarts, iters=iters,
                         step0=step0)

    # ---------------------------------------------------------------- motion

    def reset(self):
        self.mj.mj_resetDataKeyframe(self.m, self.d,
                                     self.m.key(self.M.HOME_KEY).id)
        self.driver.spawn(self.d, 0.0, 0.0, 0.0)
        self.d.time = 0.0
        self._cmd = dict(self.M.ARM_HOME)

    def put_toy(self, pos):
        self.d.qpos[self.toy_q:self.toy_q + 3] = pos
        self.d.qpos[self.toy_q + 3:self.toy_q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[self.toy_d:self.toy_d + 6] = 0.0
        self.mj.mj_forward(self.m, self.d)

    def hold(self, targets: dict, seconds: float):
        """The env's control loop: rate-limit the commanded target at 25 Hz,
        then step the DRIVER (station keeping + servo) every physics step."""
        from microduck_local.robots import mars_env as ME

        dt = float(self.m.opt.timestep)
        dec = int(round(ME.CTRL_DT / dt))
        cur = dict(self._cmd)
        want = {**cur, **targets}
        for _ in range(int(round(seconds / ME.CTRL_DT))):
            for k in cur:
                cur[k] += float(np.clip(want[k] - cur[k],
                                        -ME.DELTA_RAD_PER_STEP,
                                        ME.DELTA_RAD_PER_STEP))
            self.driver.set_arm(cur)
            for _ in range(dec):
                self.driver.step(self.d)
                self.mj.mj_step(self.m, self.d)
        self._cmd = cur

    # ----------------------------------------------------------- the readings

    def pinch(self) -> float:
        m, d = self.m, self.d
        total, buf = 0.0, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if self.toy in pair and pair & self.fingers:
                self.mj.mj_contactForce(m, d, i, buf)
                total += abs(float(buf[0]))
        return total

    def finger_contacts(self) -> dict[str, int]:
        out = {}
        for link in self.M.FINGER_LINKS:
            bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, link)
            out[link] = sum(
                1 for i in range(self.d.ncon)
                if {int(self.m.geom_bodyid[self.d.contact[i].geom1]),
                    int(self.m.geom_bodyid[self.d.contact[i].geom2])}
                == {bid, self.toy})
        return out

    def readings(self) -> dict:
        return {"q6": float(self.d.qpos[self.j6_q]),
                "servo": float(self.d.qfrc_applied[self.j6_d]),
                "constraint": float(self.d.qfrc_constraint[self.j6_d]),
                "pinch": self.pinch(), "gap_mm": self.jaw_gap() * 1000,
                "toy_z_mm": float(self.d.qpos[self.toy_q + 2]) * 1000,
                "contacts": self.finger_contacts()}


# ------------------------------------------------------------ the scripted pick

def scripted_pick(dt_ms: float, option_block: str, radius: float, yaw: float,
                  seed: int = 3, verbose: bool = False) -> dict | None:
    """Open, place the block between the blades, close, lift 10 cm, hold 2 s.

    The block is placed KINEMATICALLY — this is the rung-0 the ladder falls
    back to, not an approach — so what is being measured is the GRASP and
    nothing about finding the block.
    """
    from microduck_local.robots import mars as M
    from microduck_local.robots import mars_env as ME

    rig = Rig(dt_ms / 1000.0, option_block)
    rng = np.random.default_rng(seed)
    target = rig.spot(radius, yaw, ME.toy_rest_z())
    err, q_open = rig.solve(target, OPEN_RAD, rng, point="grasp")
    if q_open is None:
        return None

    rig.reset()
    approach = dict(zip(M.ARM_JOINTS, q_open.tolist()))
    rig.hold(approach, 2.0)
    grasp = rig.grasp_point().copy()
    ee_open = np.array(rig.d.xpos[rig.ee]).copy()
    open_empty = rig.readings()
    rig.put_toy(grasp)
    rig.hold({}, 0.2)
    resting = rig.readings()
    rig.hold({"joint6": CLOSE_TARGET_RAD}, 1.5)
    held = rig.readings()

    # the AIR control: the identical pose and close, with no block
    air = Rig(dt_ms / 1000.0, option_block)
    air.reset()
    air.hold(approach, 2.0)
    air.hold({"joint6": CLOSE_TARGET_RAD}, 1.5)
    shut_on_air = air.readings()
    # ...and the shut claw driven 0.25 rad down into the FLOOR, which is the
    # other way a load reading could be a false positive
    q_down = q_open.copy()
    q_down[1] -= 0.25
    floor_targets = dict(zip(M.ARM_JOINTS, q_down.tolist()))
    floor_targets["joint6"] = CLOSE_TARGET_RAD
    air.hold(floor_targets, 1.5)
    into_floor = air.readings()

    ee_closed = np.array(rig.d.xpos[rig.ee]).copy()
    q_now = np.array([rig.d.qpos[a] for a in rig.qadr])
    save = (rig.d.qpos.copy(), rig.d.qvel.copy())
    lift_err, q_lift = rig.solve(ee_closed + np.array([0.0, 0.0, SCRIPTED_LIFT_M]),
                                 float(q_now[5]), rng, start=q_now, restarts=5,
                                 iters=80, step0=0.25, point="ee",
                                 ignore_toy=True)
    rig.d.qpos[:], rig.d.qvel[:] = save
    rig.mj.mj_forward(rig.m, rig.d)
    if q_lift is None:
        return None
    lift_targets = dict(zip(M.ARM_JOINTS, q_lift.tolist()))
    lift_targets["joint6"] = CLOSE_TARGET_RAD
    rig.hold(lift_targets, 1.5)
    lifted = rig.readings()
    rig.hold({}, 2.0)
    after = rig.readings()
    drift = float(np.linalg.norm(
        rig.d.qpos[rig.toy_q:rig.toy_q + 2] - grasp[:2])) * 1000
    ok = (after["toy_z_mm"] - ME.toy_rest_z() * 1000 >= ME.SUCCESS_LIFT_M * 1000
          and after["pinch"] > 0.0)
    out = {"dt_ms": dt_ms, "block": option_block, "radius": radius, "yaw": yaw,
           "solve_mm": err * 1000, "lift_solve_mm": lift_err * 1000,
           "gap_open_mm": open_empty["gap_mm"],
           "ee_to_grasp_mm": float(np.linalg.norm(ee_open - grasp)) * 1000,
           "open_empty": open_empty, "resting": resting, "held": held,
           "shut_on_air": shut_on_air, "into_floor": into_floor,
           "lifted": lifted, "after": after, "drift_mm": drift, "ok": ok,
           "q_open": q_open.tolist(), "q_lift": q_lift.tolist()}
    if verbose:
        _load_table(out)
    return out


def _load_table(r: dict) -> None:
    print("\n  what `gripper_load` must discriminate "
          f"({r['dt_ms']:g} ms, {r['block']} block):")
    print(f"  {'state':34s} {'qfrc_constraint':>15} {'qfrc_applied':>13} "
          f"{'finger contacts':>17}")
    for label, key in (("open, empty", "open_empty"),
                       ("shut on AIR, at the stop", "shut_on_air"),
                       ("shut claw driven into the FLOOR", "into_floor"),
                       ("jaws resting open ON the block", "resting"),
                       ("HOLDING the block", "held")):
        s = r[key]
        c = s["contacts"]
        print(f"  {label:34s} {abs(s['constraint']):15.4f} {s['servo']:13.4f} "
              f"{c['link61']:8d} /{c['link62']:4d}")


def scripted_matrix(dt_list, blocks, spots) -> None:
    print("=== the scripted pick: does the claw lift the 4 cm / 20 g block "
          "and hold it 2 s?\n")
    print(f"{'dt':>4} {'option':>9}  {'spot':>14}  {'ik_mm':>6} {'z_lift':>7} "
          f"{'z_hold':>7} {'drift':>8} {'pinch':>7}  verdict")
    tally: dict[tuple, list[int]] = {}
    # The load table is printed for the LAST cell run, so `--dt-ms 2` alone
    # prints the shipped configuration's numbers rather than a 5 ms sweep's.
    last = None
    for dt_ms in dt_list:
        for block in blocks:
            for radius, yaw in spots:
                r = scripted_pick(dt_ms, block, radius, yaw)
                key = (dt_ms, block)
                tally.setdefault(key, [0, 0])
                if r is None:
                    print(f"{dt_ms:4g} {block:>9}  r{radius:.2f} y{yaw:+.2f}"
                          "   (no pose found)")
                    continue
                last = r
                tally[key][0] += int(r["ok"])
                tally[key][1] += 1
                print(f"{dt_ms:4g} {block:>9}  r{radius:.2f} y{yaw:+.2f}  "
                      f"{r['solve_mm']:6.1f} {r['lifted']['toy_z_mm']:7.1f} "
                      f"{r['after']['toy_z_mm']:7.1f} {r['drift_mm']:8.1f} "
                      f"{r['held']['pinch']:7.1f}  "
                      f"{'HELD' if r['ok'] else 'DROPPED'}")
    print(f"\n{'dt':>4} {'option':>9}  held")
    for (dt_ms, block), (ok, n) in tally.items():
        print(f"{dt_ms:4g} {block:>9}  {ok}/{n}")
    if last is not None:
        _load_table(last)


# ----------------------------------------------- the skill in the ACTION space

def scripted_env_pick(env, q_open, q_lift, place_in_claw: bool = True,
                      settle: int = 50, close: int = 40, lift: int = 40,
                      hold: int = 50):
    """Drive `MarsArmEnv(task="pick")` through a pick with delta ACTIONS.

    The point is not that a script can pick — `--scripted` already measured
    that — but that the pick is expressible in the 8 floats a POLICY emits.
    A skill the action space cannot express is one no reward can buy
    (`robots/mars_env.py`'s `ACTION_MODES` block, the lesson `delta` came
    from), so this is the exploration question asked of the action map.

    `place_in_claw` is rung 0: the toy is teleported between the open blades
    instead of being approached. Returns the final `info`.
    """
    import mujoco

    from microduck_local.robots import mars_env as ME

    def toward(target_q, gripper=None):
        want = np.array(target_q, float)
        if gripper is not None:
            want[5] = gripper
        step = (want - env._cmd_target) / ME.DELTA_RAD_PER_STEP
        a = np.zeros(8, np.float32)
        a[:6] = np.clip(step, -1.0, 1.0)
        return a

    info: dict = {}
    for _ in range(settle):
        _o, _r, _t, _tr, info = env.step(toward(q_open))
    if place_in_claw:
        env._place_block(env._to_base(env.grasp_point()))
        mujoco.mj_forward(env.model, env.data)
    for _ in range(close):
        _o, _r, _t, _tr, info = env.step(toward(q_open, gripper=-1.0))
    for _ in range(lift):
        _o, _r, _t, _tr, info = env.step(toward(q_lift, gripper=-1.0))
    for _ in range(hold):
        _o, _r, _t, _tr, info = env.step(toward(q_lift, gripper=-1.0))
        if _t or _tr:
            break
    return info


# ------------------------------------------------------------------- scoring

def rollout(env, act_fn, seed: int, renderer=None, cam=None,
            frame_every: int | None = None):
    """One deterministic episode. Returns (record, tiles, captions)."""
    obs, _ = env.reset(seed=seed)
    tiles, captions = [], []
    lifts, dists = [0.0], [env.distance()]
    info: dict = {"dist": dists[0], "success": False, "near_steps": 0,
                  "self_collision": None, "holding": False, "lift_m": 0.0,
                  "grasps": 0, "hold_frac": 0.0, "block_lost": False,
                  "gripper_load": 0.0}
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        if renderer is not None and frame_every and step % frame_every == 0:
            renderer.update_scene(env.data, camera=cam)
            add_sphere(renderer.scene, env.block_pos(), 0.035, BLOCK_RGBA)
            add_sphere(renderer.scene, env.effector(), 0.012, EE_RGBA)
            tiles.append(renderer.render().copy())
            captions.append([
                f"t={step * env.dt:4.1f}s d={info['dist'] * 100:5.1f}cm",
                f"lift {info['lift_m'] * 100:5.1f}cm "
                f"{'HOLD' if info['holding'] else '----'}",
                f"load {info['gripper_load']:+5.2f} "
                f"held {info['near_steps']:3d}/{env.hold_steps}",
            ])
        obs, _rew, terminated, truncated, info = env.step(act_fn(obs))
        step += 1
        dists.append(info["dist"])
        lifts.append(info["lift_m"])
    record = {
        "seed": seed, "steps": step, "final_dist": info["dist"],
        "best_dist": min(dists), "final_lift": info["lift_m"],
        "best_lift": max(lifts), "success": bool(info["success"]),
        "near_steps": int(info["near_steps"]), "grasps": int(info["grasps"]),
        "hold_frac": float(info["hold_frac"]), "holding": bool(info["holding"]),
        "block_lost": bool(info["block_lost"]),
        "self_collision": info["self_collision"],
        "terminated": bool(terminated),
        "terms": dict(info.get("episode_rewards", {})),
        "block_base": np.round(env.target_base_sample, 3).tolist(),
    }
    return record, tiles, captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=None,
                    help="an exported policy.onnx (omit with --null)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rung", type=int, default=None,
                    help="the spawn box to score under (default: the run's, "
                         "else mars_env.DEFAULT_PICK_RUNG)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mars-pick"))
    ap.add_argument("--sheet-seed", type=int, default=None)
    ap.add_argument("--null", action="store_true",
                    help="score the ZERO action — the arm parked at ARM_HOME "
                         "with the block on the floor, which is what any "
                         "number must beat before it is credited")
    ap.add_argument("--scripted", action="store_true",
                    help="the PHYSICS measurement: an IK-driven pick over "
                         "timesteps, option blocks and spots")
    ap.add_argument("--scripted-env", action="store_true",
                    help="the same pick driven by delta ACTIONS through "
                         "MarsArmEnv, which is the exploration question")
    ap.add_argument("--dt-ms", default="5,4,3,2")
    ap.add_argument("--option-blocks", default="mars,world,world+g1")
    ap.add_argument("--spots", type=int, default=8)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--cam-distance", type=float, default=1.05)
    ap.add_argument("--cam-azimuth", type=float, default=130.0)
    ap.add_argument("--cam-elevation", type=float, default=-18.0)
    args = ap.parse_args()

    import mujoco

    from microduck_local.robots import mars_env as ME
    from microduck_local.robots.mars_env import MarsArmEnv

    if args.scripted:
        lo, hi = ME.REACH_RADIUS_M
        yaw_hi = ME.REACH_YAW_RAD[1]
        rng = np.random.default_rng(0)
        spots = [(float(r), float(y)) for r, y in zip(
            rng.uniform(lo + 0.03, hi - 0.05, args.spots),
            rng.uniform(-yaw_hi, yaw_hi, args.spots))]
        scripted_matrix([float(v) for v in args.dt_ms.split(",")],
                        [b for b in args.option_blocks.split(",")], spots)
        return

    if args.scripted_env:
        env = MarsArmEnv(task="pick", seed=0,
                         pick_rung=args.rung or ME.DEFAULT_PICK_RUNG)
        rig = Rig(ME.PICK_PHYSICS_DT, "mars")
        rng = np.random.default_rng(3)
        env.reset(seed=args.seed0)
        target = env.block_pos()
        _e, q_open = rig.solve(target, OPEN_RAD, rng, point="grasp")
        rig.place(q_open)
        _e2, q_lift = rig.solve(np.array(rig.d.xpos[rig.ee])
                                + np.array([0.0, 0.0, SCRIPTED_LIFT_M]),
                                OPEN_RAD, rng, start=q_open, restarts=5,
                                step0=0.25, point="ee", ignore_toy=True)
        info = scripted_env_pick(env, q_open, q_lift)
        print("scripted pick through MarsArmEnv's own action space:")
        print(f"  q_open  {np.round(q_open, 4).tolist()}")
        print(f"  q_lift  {np.round(q_lift, 4).tolist()}")
        print(f"  holding {info['holding']}  lift {info['lift_m'] * 100:.1f} cm"
              f"  load {info['gripper_load']:+.3f}  grasps {info['grasps']}"
              f"  hold_frac {info['hold_frac']:.3f}"
              f"  success {info['success']}")
        return

    rung = args.rung
    if rung is None and args.policy:
        import json

        meta = Path(args.policy).resolve().parent / "run.json"
        if meta.exists():
            try:
                rung = (json.loads(meta.read_text()).get("env_kwargs")
                        or {}).get("pick_rung")
            except (OSError, ValueError):
                rung = None
    # `is None`, not `or`: rung 0 is a real rung and `0 or 1` is 1. Scoring a
    # drill policy in the next rung's spawn box is the map lesson wearing a
    # falsy zero.
    rung = ME.DEFAULT_PICK_RUNG if rung is None else int(rung)
    env = MarsArmEnv(task="pick", seed=args.seed0, pick_rung=rung)

    if args.null:
        label = "NULL (zero action: the arm parked at ARM_HOME)"

        def act_fn(_obs):
            return np.zeros(env.action_space.shape, np.float32)
    else:
        if not args.policy:
            ap.error("a policy path, or --null")
        import onnxruntime as ort

        sess = ort.InferenceSession(args.policy)
        in_name = sess.get_inputs()[0].name
        label = str(args.policy)

        def act_fn(obs):
            # CLIP, exactly as SB3 does in training and as a deployed code
            # skill must — see `mars_env._get_obs` for what an unclipped
            # action cost Phase 4a.
            raw = sess.run(None, {in_name: obs[None]})[0][0]
            return np.clip(raw, env.action_space.low,
                           env.action_space.high).astype(np.float32)

    lookat = np.array([0.20, 0.0, 0.12])
    cam = make_camera(args.cam_distance, args.cam_azimuth,
                      args.cam_elevation, lookat)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    frame_every = max(1, env.max_steps // 8)
    sheet_seed = (args.sheet_seed if args.sheet_seed is not None
                  else args.seed0)

    records, sheet = [], None
    for i in range(args.seeds):
        seed = args.seed0 + i
        want = seed == sheet_seed
        rec, tiles, caps = rollout(env, act_fn, seed,
                                   renderer if want else None, cam,
                                   frame_every if want else None)
        records.append(rec)
        if want:
            sheet = (tiles, caps, rec)
        hit = rec["self_collision"]
        print("seed %3d  d %.3f  best lift %5.1f cm  final lift %5.1f cm  "
              "grasps %d  hold %.2f  held %3d/%d  %-7s %s"
              % (seed, rec["final_dist"], rec["best_lift"] * 100,
                 rec["final_lift"] * 100, rec["grasps"], rec["hold_frac"],
                 rec["near_steps"], env.hold_steps,
                 "SUCCESS" if rec["success"] else "-",
                 ("lost" if rec["block_lost"] else "")
                 + (f" self-collision {hit[0]}<->{hit[1]}" if hit else "")))

    n = len(records)
    lifted = sum(r["best_lift"] >= ME.SUCCESS_LIFT_M for r in records)
    grasped = sum(r["grasps"] > 0 for r in records)
    success = sum(r["success"] for r in records)
    lost = sum(r["block_lost"] for r in records)
    hits = sum(r["self_collision"] is not None for r in records)
    terms: dict[str, float] = {}
    for r in records:
        for k, v in r["terms"].items():
            terms[k] = terms.get(k, 0.0) + v / n
    half = ME.PICK_RUNG_HALF_M.get(rung)
    if rung == 0:
        rung_line = ("rung 0: the DRILL — the arm spawns with its jaws open "
                     "around the block, so this is ONE state and the seeds "
                     "below are the same episode")
    elif half is not None:
        rung_line = (f"rung {rung}: a {half * 200:.0f} x {half * 200:.0f} cm "
                     f"spawn box about the shell spot {ME.PICK_SPOT}")
    else:
        rung_line = f"rung {rung}: the shell's whole floor footprint"
    summary = [
        f"policy: {label}",
        rung_line,
        f"seeds {args.seed0}..{args.seed0 + n - 1}",
        f"EVER GRASPED: {grasped}/{n}      LIFTED >= "
        f"{ME.SUCCESS_LIFT_M * 100:.0f} cm: {lifted}/{n}      "
        f"HELD 1 s (the task's rule): {success}/{n}",
        f"final lift: median {np.median([r['final_lift'] for r in records]) * 100:.1f} cm"
        f"   best-in-episode: median "
        f"{np.median([r['best_lift'] for r in records]) * 100:.1f} cm",
        f"holding fraction of an episode: mean "
        f"{np.mean([r['hold_frac'] for r in records]):.3f}   grasp attempts: "
        f"mean {np.mean([r['grasps'] for r in records]):.2f}",
        f"block knocked out of the shell: {lost}/{n}      "
        f"self-collision terminations: {hits}/{n}",
        "per-episode terms: " + "  ".join(f"{k} {v:+.2f}"
                                          for k, v in terms.items()),
    ]
    print()
    for line in summary:
        print(line)

    if sheet is not None:
        tiles, caps, rec = sheet
        header = [f"MARS pick — {Path(label).name if not args.null else label}",
                  f"seed {rec['seed']}  block(base) {rec['block_base']}  "
                  f"lift {rec['final_lift'] * 100:.1f} cm  "
                  f"{'SUCCESS' if rec['success'] else 'not held'}"]
        out = args.out / "sheet.png"
        build_sheet(tiles, caps, header, summary, out)
        print(f"\nsheet: {out}  ({len(tiles)} tiles at "
              f"{frame_every * env.dt:.1f} s)")


if __name__ == "__main__":
    main()
