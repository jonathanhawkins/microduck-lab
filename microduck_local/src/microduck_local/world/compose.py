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
from .scenario import Scenario

ROBOT_DIR = C.MICRODUCK_RL_DIR / "src/mjlab_microduck/robot/microduck"
ROBOT_XML = {
    "walk": ROBOT_DIR / "robot_walk.xml",
    "all": ROBOT_DIR / "robot_allcollisions.xml",
}


def duck_prefix(duck_id: str) -> str:
    return f"{duck_id}/"


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


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
        body.add_geom(name=f"ball{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[ball.radius, 0, 0], group=0, rgba=[1.0, 0.55, 0.0, 1.0],
                      friction=[0.5, 0.005, 0.0001])
    for i, person in enumerate(scenario.persons):
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
        x, y, yaw = duck.spawn
        frame = w.add_frame(pos=[x, y, 0.0], quat=_yaw_quat(yaw))
        spec.attach(robot, prefix=duck_prefix(duck.id), frame=frame)
    model = spec.compile()
    # Feet win the friction pair, as the walk env sets for every model.
    for duck in scenario.ducks:
        for side in ("left", "right"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{duck_prefix(duck.id)}{side}_foot_collision")
            if gid >= 0:
                model.geom_priority[gid] = 1
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
