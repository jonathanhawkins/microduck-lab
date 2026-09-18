"""Phase 0 of docs/mars-roadmap.md: does Innate's MARS load into this lab?

Loads mars.urdf into MuJoCo 3.10 the way innate-os does (MjSpec.from_string +
the discardvisual compiler override + package:// rewrite + a planar base +
their contact tuning), attaches it into a floor world with MjSpec.attach under
a prefix (the lab's world/compose.py pattern), drives the arm to ARM_HOME with
the same PD servo innate uses, and times raw physics steps against the duck.

    # the assets: one directory of innate-os at the pinned sha (7.2 MB)
    SHA=0ca73670ae20940e381de8b4356ac9855f66e25a
    RAW=https://raw.githubusercontent.com/innate-inc/innate-os/$SHA/ros2_ws/src/mars_bot/mars_description
    mkdir -p /tmp/mars/mars_description/{urdf,meshes}
    curl -sL $RAW/urdf/mars.urdf -o /tmp/mars/mars_description/urdf/mars.urdf
    for m in base head link1 link2 link3 link4 link5 link61 link62; do
      curl -sL $RAW/meshes/$m.STL -o /tmp/mars/mars_description/meshes/$m.STL; done
    uv run python scripts/probe_mars.py /tmp/mars

Measured 2026-09-17 on an Apple M-series Mac:
    nq=11 nv=11 nbody=18 ngeom=59 nmesh=9, 1.365 kg
    after 2 s at ARM_HOME: max |q - home| = 0.0034 rad
    mars 92,700 physics steps/s   duck 72,700   (single env, 2 ms step)
"""
import sys, time, math
from pathlib import Path
import mujoco, numpy as np

S = Path(sys.argv[1])
urdf = S / "mars_description/urdf/mars.urdf"
robot_dir = str((S / "mars_description").resolve()) + "/"
txt = urdf.read_text().replace("package://mars_description/", robot_dir)
txt = txt.replace('<robot name="mars_bot">',
                  '<robot name="mars_bot"><mujoco><compiler discardvisual="false"/></mujoco>')
robot = mujoco.MjSpec.from_string(txt)
print("robot name:", robot.modelname)
base = robot.body("base_link")
for name, jt, ax in (("base_x", mujoco.mjtJoint.mjJNT_SLIDE, (1,0,0)),
                     ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, (0,1,0)),
                     ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, (0,0,1))):
    j = base.add_joint(); j.name = name; j.type = jt; j.axis = ax
for name in ("base_wheel_left", "base_wheel_right"):
    g = robot.geom(name); g.condim = 1; g.priority = 1
robot.add_exclude(bodyname1="link61", bodyname2="link62")

# --- the lab's pattern: attach under a prefix into a world with a floor ---
world = mujoco.MjSpec()
world.option.timestep = 0.002
w = world.worldbody
w.add_light(pos=[0,0,3], dir=[0,0,-1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3,3,0.05])
frame = w.add_frame(pos=[0,0,0.0])
world.attach(robot, prefix="m0/", frame=frame)
model = world.compile()
data = mujoco.MjData(model)
print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")
jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
print("joints:", jn)
mass = float(sum(model.body_mass))
print(f"total mass {mass:.3f} kg")
# AABB of the whole body at qpos0 (lab spacing needs this)
mujoco.mj_forward(model, data)
lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
for g in range(model.ngeom):
    if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
        continue
    c = data.geom_xpos[g]; r = model.geom_rbound[g]
    lo = np.minimum(lo, c - r); hi = np.maximum(hi, c + r)
print("collision AABB extent (x,y,z) ~", np.round(hi - lo, 3), " (rbound-based, conservative)")

# --- PD servo to ARM_HOME, innate's gains, driven through qfrc_applied ---
ARM_HOME = {"joint1": 1.445, "joint2": -1.388, "joint3": 1.517, "joint4": 0.446,
            "joint5": -0.089, "joint6": 0.0015, "joint_head": 0.0}
KP, KD, EFF = 50.0, 1.0, 50.0
adr = {}
for n in list(ARM_HOME) + ["joint6M"]:
    jid = model.joint("m0/" + n).id
    adr[n] = (model.jnt_qposadr[jid], model.jnt_dofadr[jid])
def servo():
    for n, tgt in ARM_HOME.items():
        q, d = adr[n]
        tau = KP * (tgt - data.qpos[q]) - KD * data.qvel[d]
        data.qfrc_applied[d] = max(-EFF, min(EFF, tau))
    # mimic finger
    q6, d6 = adr["joint6"]; qm, dm = adr["joint6M"]
    tau = KP * (-data.qpos[q6] - data.qpos[qm]) - KD * data.qvel[dm]
    data.qfrc_applied[dm] = max(-2.0, min(2.0, tau))
# settle 2 s, then time 5 s of physics
for _ in range(1000):
    servo(); mujoco.mj_step(model, data)
err = max(abs(data.qpos[adr[n][0]] - t) for n, t in ARM_HOME.items())
print(f"after 2 s: max |q - home| = {err:.4f} rad; base z drift = {data.qpos[adr['joint1'][0]-3+0]:.4f} (base_x)")
n = 2500
t0 = time.perf_counter()
for _ in range(n):
    servo(); mujoco.mj_step(model, data)
dt = time.perf_counter() - t0
print(f"mars: {n/dt:,.0f} physics steps/s (single env, 2 ms step, PD in python)")
print("contacts now:", data.ncon, " base pose:", np.round(data.qpos[:3], 4))

# --- the duck for scale, same harness ---
from microduck_local import contract as C
duck = mujoco.MjModel.from_xml_path(str(C.MICRODUCK.scene_fn()))
dd = mujoco.MjData(duck)
mujoco.mj_resetDataKeyframe(duck, dd, duck.key("STAND").id)
t0 = time.perf_counter()
for _ in range(n):
    mujoco.mj_step(duck, dd)
dt2 = time.perf_counter() - t0
print(f"duck: {n/dt2:,.0f} physics steps/s (nv={duck.nv}, same loop, no servo python)")
