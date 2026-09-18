"""Unitree G1 as a walking person in a world.

The body is the Lucky Robots G1 MJCF + shipped `walker.onnx` (99-d obs → 29
joint targets at 50 Hz). Joint addresses are resolved by name: this XML has
finger DoFs the walk policy does not own, so the luckyrobots `7+i` layout
would feed the right arm the left hand.

Assets live in `.cache/unitree_g1/` (gitignored, ~140 MB of meshes). Fetch
with `uv run fetch-g1`. Compose raises FileNotFoundError if they are missing.

The hands are FROZEN, not deleted (`freeze_hands`): the 14 finger DoFs are
removed while the finger bodies, meshes and mass stay. Measured on this Mac,
driving the shipped `walker.onnx` at 0.3 m/s for 30 s:

    variant                 nv   mass      physics ctrl steps/s   30 s path
    full (fingers free)     49   33.99 kg  5 271   (1.00x)        2.97 m
    hands FROZEN            35   33.99 kg  6 737   (1.28x)        2.96 m
    hands deleted           35   33.34 kg  6 900   (1.31x)        1.09 m

Deleting buys 0.03x more than freezing and costs 0.65 kg of forearm, and its
walk is the one result outside the noise band (a 1e-6 rad hip nudge alone
moves that path between 2.32 and 3.11 m, so only the deleted row resolves).
An earlier 1.76x figure for deleting was measured at qpos0 in free fall,
where DoF count dominates because there are no contacts — under load the
constraint solver does, and 1.28x is the honest number.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .. import contract as C
from ..world.compose import duck_prefix
from .policy_contract import PolicyContract, Slot, declare
from .spec import RobotSpec

_PKG = Path(__file__).resolve().parent / "unitree_g1"
_ROOT = Path(__file__).resolve().parents[3]  # microduck_local/
CACHE_DIR = Path(os.environ.get("MICRODUCK_G1_DIR") or _ROOT / ".cache" / "unitree_g1")
G1_REPO = "https://github.com/luckyrobots/g1-manipulation-challenge.git"

STAND_Z = 0.79
FALL_Z = 0.40
# Widest HORIZONTAL extent of the whole body at the STAND keyframe (m), and
# the lab-stage pitch derived from it. Measured from the fetched MJCF, not
# quoted from a datasheet: the geom AABBs at STAND span 0.3675 (x) x 0.5338
# (y) x 1.3071 (z). The duck is 0.1845 m wide and the lab gives it 0.65 m
# (3.52x); the same ratio on 0.5338 m is 1.88 m, which leaves 1.35 m of air
# between two G1s instead of the 0.12 m the duck's pitch left. See
# RobotSpec.lab_spacing_m; tests/test_lab_robots.py re-measures BODY_WIDTH_M.
BODY_WIDTH_M = 0.5338
LAB_SPACING_M = 1.88
NUM_JOINTS = 29
NUM_HAND_JOINTS = 14          # the finger DoFs `freeze_hands` removes
OBS_DIM = 99
# Obs layout (99D, LuckyRobots run.py order — the layout walker.onnx was
# trained against, so the shipped policy is both a golden test for this env
# and a teacher to distil from):
#   [ base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
#     joint_pos_rel(29), joint_vel(29), last_action(29), twist_cmd(3) ]
# NOTE base_lin_vel: no real humanoid observes it without state estimation.
# This is a lab contract, not a sim2real one — see AGENTS.md "sim2real honesty".
OBS_LIN_VEL = slice(0, 3)
OBS_ANG_VEL = slice(3, 6)
OBS_GRAVITY = slice(6, 9)
CMD_DIM = 3
#: The G1's policy contract id (`robots/policy_contract.py`). "lucky" names
#: whose layout it is: LuckyRobots' 99-d order, kept verbatim so their
#: `walker.onnx` is both a golden test and a teacher. A local re-layout would
#: be `g1-lucky-99-v2` and would cost us both of those, which is why the
#: provenance is in the id rather than only in a comment.
CONTRACT_ID = "g1-lucky-99-v1"
SCENE_NAME = "scene_walk_g1.xml"
SCENE_NAME_HANDS = "scene_walk_g1_hands.xml"   # control arm: fingers not frozen
STAND_KEY = "STAND"
FLOOR_GEOM = "floor"
FOOT_CLEARANCE = 0.002        # m of air under the lowest capsule at STAND
BASE_BODY = "pelvis"
GYRO_SENSOR = "imu-pelvis-angular-velocity"
# Seven collision capsules a foot (the duck has one pad a side).
FOOT_GEOMS = {
    "left": tuple(f"left_foot{i}_collision" for i in range(1, 8)),
    "right": tuple(f"right_foot{i}_collision" for i in range(1, 8)),
}
# Velocity commands for a 1.3 m humanoid. walker.onnx was trained on a
# forward-biased range; these are the ranges the LOCAL trainer samples.
# MEASURED on the shipped walker in G1WalkEnv (600 steps a command, spawned
# at the STAND keyframe, no noise/DR):
#     cmd vx  0.2   0.3   0.4   0.5   0.6   0.8   1.0
#     m/s     0.01  0.01  0.30  0.43  0.53  0.73  0.92
# Below ~0.4 m/s the shipped policy STANDS — a dead zone, not a failure of
# this env (obs are bit-identical to the standalone driver, and a spawn
# transient alone is enough to kick it out of the stall at 0.3). Above it,
# tracking is good. `min_forward_cmd` keeps the trainer's "walk forward"
# orders inside the regime a gait exists in; the full range below still
# samples slower commands, which is where a LOCAL policy has to learn what
# the shipped one never did.
MIN_FORWARD_CMD = 0.4
LIN_VEL_X_RANGE = (-0.5, 0.8)
LIN_VEL_Y_RANGE = (-0.3, 0.3)
ANG_VEL_Z_RANGE = (-1.0, 1.0)
# Termination: pelvis below this, or tilted past 70 deg.
FALL_HEIGHT = 0.50
FALL_GRAVITY_Z = -0.342
# Pushes on 34 kg, scaled from the duck's 0.3 m/s on 0.8 kg by nothing more
# than "a shove a person could give" — a knob, not a ported constant.
PUSH_VEL_RANGE = (-0.4, 0.4)

# 🎬 editor data (pose.PoseScratch.meta()). Coefficient signs come from the
# WORLD hinge axes at STAND, measured off the fetched MJCF (x forward, y left,
# z up): both legs' hip_pitch / knee / ankle_pitch are +y (a positive angle
# swings the limb BACK), hip_roll is +x-ish (positive swings the leg LEFT),
# hip_yaw is +z-ish (positive turns the leg left), waist yaw/roll/pitch are
# +z/+x/+y, shoulder_pitch is +y (positive swings the arm back). The squat
# coupling (hip -t/2, knee +t, ankle -t/2) is the one g1_env solves with,
# and it keeps each leg's world-pitch sum at zero, so the feet stay flat.
JOINT_GROUPS = (("left leg",) * 6 + ("right leg",) * 6 + ("waist",) * 3
                + ("left arm",) * 7 + ("right arm",) * 7)


def _obs_slots(num_joints: int) -> tuple[Slot, ...]:
    """The 99-float layout, as `robots/g1_env.G1WalkEnv._get_obs` writes it.

    Read off that method's slice assignments — which are themselves written
    in terms of `nj` — and anchored on the three named slices above, so the
    table and the code that fills it move together:

        obs[0:3]  lin        obs[9:38]  joint_pos   obs[67:96] last_action
        obs[3:6]  gyro       obs[38:67] joint_vel   obs[96:99] twist_cmd
        obs[6:9]  gravity

    `tests/test_policy_contract.py` pins two of these against a live G1 env.
    """
    n = int(num_joints)
    return (
        Slot("base_lin_vel", OBS_LIN_VEL.start, OBS_LIN_VEL.stop),
        Slot("base_ang_vel", OBS_ANG_VEL.start, OBS_ANG_VEL.stop),
        Slot("projected_gravity", OBS_GRAVITY.start, OBS_GRAVITY.stop),
        Slot("joint_pos_rel", 9, 9 + n),
        Slot("joint_vel", 9 + n, 9 + 2 * n),
        Slot("last_action", 9 + 2 * n, 9 + 3 * n),
        Slot("twist_cmd", 9 + 3 * n, 9 + 3 * n + CMD_DIM),
    )


def _effectors():
    from .spec import Effector
    return (
        Effector("left_foot", "left foot", "left_ankle_roll_link", "foot", "sole"),
        Effector("right_foot", "right foot", "right_ankle_roll_link", "foot", "sole"),
        Effector("left_hand", "left hand", "left_wrist_yaw_link", "hand", (0.06, 0.0, 0.0)),
        Effector("right_hand", "right hand", "right_wrist_yaw_link", "hand", (0.06, 0.0, 0.0)),
    )


def _legs(**per_leg):
    out = {}
    for side in ("left", "right"):
        for j, c in per_leg.items():
            out[f"{side}_{j}_joint"] = c
    return out


RIG_CONTROLS = (
    {"id": "squat", "label": "squat", "hint": "+ crouch",
     "title": "fold both legs symmetrically, feet flat, trunk upright — the ⇕ "
              "handle drags this when no other control is selected",
     "parts": _legs(hip_pitch=-1, knee=2, ankle_pitch=-1),
     "pick": ["left_knee_joint", "right_knee_joint"],
     "handle": {"joint": "root", "offset": [-0.3, 0, 0.1]}},
    {"id": "lean", "label": "lean", "hint": "+ fwd",
     "title": "the pelvis pitches while the legs counterbalance, feet flat",
     "parts": {"root": 1, **_legs(hip_pitch=-1 / 3, knee=-1 / 3, ankle_pitch=-1 / 3)},
     "pick": ["root"],
     "handle": {"joint": "root", "offset": [-0.3, 0, 0.35]}},
    {"id": "swingL", "label": "L swing", "hint": "+ fwd",
     "title": "swing the whole left leg forward/back about the hip, foot kept level",
     "parts": {"left_hip_pitch_joint": -1, "left_ankle_pitch_joint": 1},
     "pick": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"],
     "handle": {"joint": "left_hip_pitch_joint", "offset": [0, 0.2, 0]}},
    {"id": "swingR", "label": "R swing", "hint": "+ fwd",
     "title": "swing the whole right leg forward/back about the hip, foot kept level",
     "parts": {"right_hip_pitch_joint": -1, "right_ankle_pitch_joint": 1},
     "pick": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"],
     "handle": {"joint": "right_hip_pitch_joint", "offset": [0, -0.2, 0]}},
    {"id": "sway", "label": "sway", "hint": "hips ±",
     "title": "both hip rolls together — swing the legs sideways under the pelvis",
     "parts": {"left_hip_roll_joint": 1, "right_hip_roll_joint": 1},
     "pick": [],
     "handle": {"joint": "root", "offset": [0, 0.33, 0.03]}},
    {"id": "stance", "label": "stance", "hint": "+ wide",
     "title": "hip rolls apart — widen or narrow the stance",
     "parts": {"left_hip_roll_joint": 1, "right_hip_roll_joint": -1},
     "pick": [],
     "handle": {"joint": "root", "offset": [0, -0.33, 0.03]}},
    {"id": "twist", "label": "twist", "hint": "hips ±",
     "title": "both hip yaws together — pivot the pelvis against the feet",
     "parts": {"left_hip_yaw_joint": 1, "right_hip_yaw_joint": 1},
     "pick": [],
     "handle": {"joint": "root", "offset": [-0.42, 0, -0.07]}},
    {"id": "toes", "label": "toes", "hint": "+ out",
     "title": "hip yaws apart — turn the feet out or in",
     "parts": {"left_hip_yaw_joint": 1, "right_hip_yaw_joint": -1},
     "pick": ["left_ankle_pitch_joint", "left_ankle_roll_joint",
              "right_ankle_pitch_joint", "right_ankle_roll_joint"],
     "handle": {"joint": "left_ankle_roll_joint", "offset": [0.2, 0, 0.04]}},
    {"id": "turn", "label": "turn", "hint": "+ left",
     "title": "the waist yaws — turn the torso against the pelvis",
     "parts": {"waist_yaw_joint": 1},
     "pick": ["waist_yaw_joint", "waist_roll_joint"],
     "handle": {"joint": "waist_yaw_joint", "offset": [-0.3, 0, 0.3]}},
    {"id": "bend", "label": "bend", "hint": "+ fwd",
     "title": "the waist pitches — bow the torso forward or arch it back",
     "parts": {"waist_pitch_joint": 1},
     "pick": ["waist_pitch_joint"],
     "handle": {"joint": "waist_pitch_joint", "offset": [0.3, 0, 0.3]}},
    {"id": "arms", "label": "arms", "hint": "+ fwd",
     "title": "both shoulders pitch together — raise the arms in front or swing them back",
     "parts": {"left_shoulder_pitch_joint": -1, "right_shoulder_pitch_joint": -1},
     "pick": ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
     "handle": {"joint": "left_shoulder_pitch_joint", "offset": [0, 0.2, 0]}},
    {"id": "elbows", "label": "elbows", "hint": "+ flex",
     "title": "both elbows together — the guard",
     "parts": {"left_elbow_joint": 1, "right_elbow_joint": 1},
     "pick": ["left_elbow_joint", "right_elbow_joint"],
     "handle": {"joint": "left_elbow_joint", "offset": [0, 0.15, 0]}},
)

# Luckyrobots armature (run.py set_armature), applied by joint name.
_ARM = {
    "elbow": 0.00360972, "shoulder": 0.00360972, "wrist_roll": 0.00360972,
    "hip_pitch": 0.01017752, "hip_yaw": 0.01017752, "waist_yaw": 0.01017752,
    "hip_roll": 0.02510192, "knee": 0.02510192,
    "wrist_pitch": 0.00425, "wrist_yaw": 0.00425,
    "ankle": 0.00721945, "waist_pitch": 0.00721945, "waist_roll": 0.00721945,
}


def cache_dir() -> Path:
    return CACHE_DIR


@lru_cache(maxsize=1)
def joint_names() -> tuple[str, ...]:
    """The 29 actuated joints, in policy order (model_config.json)."""
    names = tuple(load_config()["joint_names"])
    if len(names) != NUM_JOINTS:
        raise RuntimeError(f"G1 config has {len(names)} joints, want {NUM_JOINTS}")
    return names


@lru_cache(maxsize=1)
def default_pose() -> np.ndarray:
    cfg = load_config()
    return np.array([cfg["default_joint_pos"][n] for n in joint_names()], np.float32)


@lru_cache(maxsize=1)
def action_scales() -> np.ndarray:
    cfg = load_config()
    return np.array([cfg["action_scales"][n] for n in joint_names()], np.float32)


def g1_ready(d: Path | None = None) -> bool:
    d = d or CACHE_DIR
    return (d / "g1.xml").is_file() and (d / "walker.onnx").is_file() and (d / "assets").is_dir()


def require_g1() -> Path:
    if not g1_ready():
        raise FileNotFoundError(
            f"Unitree G1 assets not in {CACHE_DIR} — run `uv run fetch-g1` "
            "(Lucky Robots MJCF + walker.onnx, ~140 MB)")
    return CACHE_DIR


def g1_xml() -> Path:
    return require_g1() / "g1.xml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    for p in (CACHE_DIR / "model_config.json", _PKG / "model_config.json"):
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError("G1 model_config.json missing — run `uv run fetch-g1`")


# The three finger chains per hand, by the link each one hangs from.
_HAND_MARK = "_hand_"


def freeze_hands(spec: mujoco.MjSpec) -> int:
    """Delete the finger JOINTS and their actuators, keeping the bodies.

    The walk policy owns 29 joints; the 14 finger DoFs are free weight in
    every sense. Freezing keeps the mass, the inertia and the meshes — the
    robot looks and weighs the same, and `walker.onnx` walks it the same
    (see the module docstring) — while the solver stops carrying them.
    Returns how many joints were frozen, so a model revision that renames
    the fingers cannot silently freeze nothing.
    """
    n = 0
    for j in list(spec.joints):
        if _HAND_MARK in j.name:
            spec.delete(j)
            n += 1
    for a in list(spec.actuators):
        if _HAND_MARK in a.name:
            spec.delete(a)
    return n


def g1_spec(freeze: bool = True) -> mujoco.MjSpec:
    """MJCF ready to `MjSpec.attach` under a person prefix.

    `freeze=True` (the default everywhere: training AND the world's person,
    so the two are the same robot) removes the finger DoFs.
    """
    xml = g1_xml()
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.meshdir = str(xml.parent / "assets")
    spec.option.timestep = C.PHYSICS_DT
    if freeze:
        frozen = freeze_hands(spec)
        if frozen != NUM_HAND_JOINTS:
            raise RuntimeError(
                f"expected {NUM_HAND_JOINTS} finger joints to freeze, found {frozen} "
                "— the G1 MJCF changed; check robots/g1.py")
    return spec


def _scene_spec(freeze: bool = True) -> mujoco.MjSpec:
    """The training scene: one G1, a floor, a light, a STAND keyframe."""
    spec = g1_spec(freeze=freeze)
    spec.option.timestep = C.PHYSICS_DT
    w = spec.worldbody
    light = w.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    floor = w.add_geom()
    floor.name = FLOOR_GEOM
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [25.0, 25.0, 0.1]
    floor.friction = [1.0, 0.005, 0.0001]
    model = spec.compile()
    # Stand height MEASURED off the model, never hand-carried: drop the
    # default pose until the lowest foot capsule kisses the floor.
    data = mujoco.MjData(model)
    names = joint_names()
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    default = default_pose().astype(np.float64)
    data.qpos[qadr] = default
    data.qpos[2] = STAND_Z
    mujoco.mj_forward(model, data)
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for names in FOOT_GEOMS.values() for n in names]
    low = min(float(data.geom_xpos[g][2] - model.geom_size[g][0]) for g in foot_ids)
    stand_z = STAND_Z - low + FOOT_CLEARANCE
    key = spec.add_key()
    key.name = STAND_KEY
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, stand_z]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[qadr] = default
    key.qpos = qpos
    key.qvel = np.zeros(model.nv)
    key.ctrl = default.copy()
    return spec


def g1_scene_xml(freeze: bool = True) -> Path:
    """Path to the generated training scene, written under `.cache`.

    Generated beside the vendored assets so `meshdir` still resolves, and
    never into the clone's own files. Atomic (temp + os.replace): a vec-env
    worker must never import a half-written scene.

    `freeze=False` writes the fingers-free variant — the control arm for the
    freeze measurement (tests/test_g1_env.py), never the training scene.
    """
    require_g1()
    out = CACHE_DIR / (SCENE_NAME if freeze else SCENE_NAME_HANDS)
    xml = _scene_spec(freeze=freeze).to_xml()
    if not out.exists() or out.read_text() != xml:
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, out)
    return out


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], np.float64)


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def extract_visual_scene(model: mujoco.MjModel, prefix: str = "",
                         group: int = 2) -> dict:
    """The real G1 visual meshes.

    Verts are millimetres (ints) so the JSON is ~21 MB instead of ~78 MB of
    floats; the viewer multiplies by `vertScale`. No voxel weld — that turned
    every thin CAD shell (chest, thighs, shins) into craters.
    Same `{bodies, meshes, geoms}` shape as viz_server.extract_scene.

    `group` is which geom group holds the VISUALS. 2 is the Lucky Robots
    MJCF's convention and the default, so nothing about the G1 changes;
    MuJoCo's URDF importer uses 1, which is what `robots/mars.visual_scene`
    passes. Everything else here is body-agnostic — PHASE 1B: this dump
    belongs in a shared module rather than on one robot
    (`docs/mars-roadmap.md` §6.4).
    """
    body_index: dict[int, int] = {}
    bodies: list[str] = []
    for b in range(model.nbody):
        name = model.body(b).name
        if prefix and not (name == "world" or name.startswith(prefix)):
            continue
        body_index[b] = len(bodies)
        bodies.append(name[len(prefix):] if name.startswith(prefix) else name)
    mesh_ids: dict[int, int] = {}
    meshes: list[dict] = []
    geoms: list[dict] = []
    for i in range(model.ngeom):
        if int(model.geom_group[i]) != group:
            continue
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        bid = int(model.geom_bodyid[i])
        if bid not in body_index:
            continue
        mid = int(model.geom_dataid[i])
        if mid not in mesh_ids:
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            if vn == 0 or fn == 0:
                continue
            v = np.round(model.mesh_vert[va:va + vn] * 1000.0).astype(np.int32)
            f = model.mesh_face[fa:fa + fn]
            mesh_ids[mid] = len(meshes)
            meshes.append({
                "v": v.reshape(-1).tolist(),
                "f": f.reshape(-1).tolist(),
            })
        mat_id = int(model.geom_matid[i])
        rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[i]
        geoms.append({
            "mesh": mesh_ids[mid],
            "body": body_index[bid],
            "pos": [round(float(x), 5) for x in model.geom_pos[i]],
            "quat": [round(float(x), 5) for x in model.geom_quat[i]],
            "mat": model.material(mat_id).name if mat_id >= 0 else "",
            "name": model.mesh(mid).name,
            "rgba": [round(float(x), 4) for x in rgba],
        })
    return {"bodies": bodies, "meshes": meshes, "geoms": geoms, "vertScale": 0.001}


@lru_cache(maxsize=1)
def visual_scene() -> dict:
    """Standalone G1 scene for the viewer (no world prefix). Cached: the mesh dump is heavy."""
    m = mujoco.MjModel.from_xml_path(str(g1_xml()))
    return extract_visual_scene(m)


@lru_cache(maxsize=8)
def _walker_session(path: str | None = None) -> ort.InferenceSession:
    """The walking policy the world's G1 runs.

    Defaults to the shipped `walker.onnx`; `MICRODUCK_G1_WALKER` (or an
    explicit path) swaps in a policy trained here — `train-walk --robot g1`
    exports the same 99 -> 29 graph, so the person in the /sim world can be
    driven by a LOCAL brain exactly as the ducks are.
    """
    require_g1()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(path or walker_onnx()), sess_options=opts,
        providers=["CPUExecutionProvider"])
    shape = sess.get_inputs()[0].shape
    if len(shape) != 2 or int(shape[1]) != OBS_DIM:
        raise ValueError(
            f"G1 walker policy {path} takes obs{shape}, this robot speaks "
            f"{OBS_DIM}-d — see robots/g1_env.py for the layout")
    return sess


class G1Walker:
    """50 Hz velocity-commanded walk on one attached G1."""

    def __init__(self, model: mujoco.MjModel, person_id: str,
                 policy: str | None = None):
        self.model = model
        self.prefix = duck_prefix(person_id)
        cfg = load_config()
        self.joint_names: list[str] = list(cfg["joint_names"])
        if len(self.joint_names) != NUM_JOINTS:
            raise RuntimeError(f"G1 config has {len(self.joint_names)} joints, want {NUM_JOINTS}")
        self.default = np.array(
            [cfg["default_joint_pos"][n] for n in self.joint_names], np.float32)
        self.scale = np.array(
            [cfg["action_scales"][n] for n in self.joint_names], np.float32)
        self.qpos_adr = np.array(
            [int(model.joint(self.prefix + n).qposadr[0]) for n in self.joint_names])
        self.qvel_adr = np.array(
            [int(model.joint(self.prefix + n).dofadr[0]) for n in self.joint_names])
        self.act_id = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.prefix + n)
            for n in self.joint_names])
        if np.any(self.act_id < 0):
            raise KeyError(f"G1 {person_id}: missing actuators for {self.joint_names}")
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 self.prefix + "floating_base_joint")
        if root < 0:
            raise KeyError(f"G1 {person_id}: no {self.prefix}floating_base_joint")
        self.root_qpos = int(model.jnt_qposadr[root])
        self.root_qvel = int(model.jnt_dofadr[root])
        self.pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "pelvis")
        self.body_ids = [
            b for b in range(model.nbody)
            if model.body(b).name.startswith(self.prefix)]
        self.last_action = np.zeros(NUM_JOINTS, np.float32)
        self.cmd = np.zeros(3, np.float32)
        self._sess = _walker_session(policy)
        self._in = self._sess.get_inputs()[0].name
        # Same order GET /scene/g1 lists (world first, then the attached tree).
        self.scene_bodies = ["world"] + [
            model.body(b).name[len(self.prefix):] for b in self.body_ids]
        self._set_armature(model)

    def _set_armature(self, model: mujoco.MjModel) -> None:
        for name, dof in zip(self.joint_names, self.qvel_adr):
            val = next((v for k, v in _ARM.items() if k in name), 0.00360972)
            model.dof_armature[dof] = val

    def spawn(self, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
        q = self.root_qpos
        data.qpos[q:q + 3] = [x, y, STAND_Z]
        data.qpos[q + 3:q + 7] = _yaw_quat(yaw)
        data.qpos[self.qpos_adr] = self.default
        data.qvel[self.root_qvel:self.root_qvel + 6] = 0.0
        data.qvel[self.qvel_adr] = 0.0
        data.ctrl[self.act_id] = self.default
        self.last_action[:] = 0.0
        self.cmd[:] = 0.0

    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        p = data.xpos[self.pelvis]
        q = data.xquat[self.pelvis]
        # heading from the pelvis quaternion about +Z
        yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] * q[2] + q[3] * q[3])))
        return float(p[0]), float(p[1]), yaw

    def fallen(self, data: mujoco.MjData) -> bool:
        return float(data.xpos[self.pelvis][2]) < FALL_Z

    def bodies_payload(self, data: mujoco.MjData, scene_bodies: list[str]) -> list[list[float]]:
        out = []
        for name in scene_bodies:
            if name in ("", "world"):
                out.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                continue
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + name)
            if b < 0:
                out.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                continue
            p, q = data.xpos[b], data.xquat[b]
            out.append([round(float(v), 4) for v in (*p, *q)])
        return out

    def control(self, data: mujoco.MjData) -> None:
        q = data.qpos
        v = data.qvel
        lin_world = v[self.root_qvel:self.root_qvel + 3]
        ang = v[self.root_qvel + 3:self.root_qvel + 6]
        quat = q[self.root_qpos + 3:self.root_qpos + 7]
        lin = _quat_apply_inverse(quat, lin_world)
        grav = _quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0]))
        jpos = (q[self.qpos_adr] - self.default).astype(np.float32)
        jvel = v[self.qvel_adr].astype(np.float32)
        obs = np.concatenate([
            lin.astype(np.float32), ang.astype(np.float32), grav.astype(np.float32),
            jpos, jvel, self.last_action, self.cmd,
        ]).astype(np.float32)
        if obs.shape[0] != OBS_DIM:
            raise RuntimeError(f"G1 obs {obs.shape[0]}d, policy wants {OBS_DIM}")
        action = self._sess.run(None, {self._in: obs.reshape(1, -1)})[0][0]
        action = np.asarray(action, np.float32).reshape(NUM_JOINTS)
        target = self.default + action * self.scale
        data.ctrl[self.act_id] = target
        self.last_action = action


# --------------------------------------------------------------- robot spec
#
# What `walk_env.MicroduckWalkEnv` needs to walk this body (robots/spec.py).
# Built lazily: `load_config()` reads the fetched cache, so importing this
# module on a machine that has never run `fetch-g1` must not explode — the
# spec is only materialized when something asks for the G1 by name.


class G1Body(RobotSpec):
    """The G1's half of the lab's `Body` contract.

    A subclass, the sibling of `robots/microduck.MicroduckBody`: everything
    here already existed and was reached by an `if robot == "g1"` somewhere
    else, so this is the same code with the branch removed.

    Not a `@dataclass` of its own — it adds no fields, only answers, and
    re-emitting `__init__` would invite a field to be added here instead of
    in the spec construction below where it belongs.

    Module level, and importing this module still reads nothing: the
    laziness that matters is `load_config()` touching the fetched cache, not
    importing `robots/spec.py` (which imports only `body.py`, and nothing in
    the spec/registry chain imports this module at module level). A class
    built inside a function is also unpicklable by name —
    `pickle.dumps` of the spec failed with "Can't get local object
    '_g1_body_cls.<locals>.G1Body'" — and a vec-env worker ships specs.
    `tests/test_registry.py` pins that the import stays cache-free.
    """

    def contract(self) -> PolicyContract:
        """A LAB contract, and the file says so.

        The 99 floats are the LuckyRobots layout `walker.onnx` was trained
        against, and their first three are base LINEAR velocity — which no
        real humanoid observes without state estimation. That is the whole
        `deploy` sentence, and it is the reason this record exists in the
        file rather than only in `g1_env.py`'s docstring: a `.onnx` someone
        is handed is exactly where the caveat used to get lost.

        The slot table is derived from `self.num_joints` (29 with the hands
        frozen), so a config revision that changes the joint count fails at
        construction against `obs_dim` instead of quietly mislabelling the
        table — `declare()` explains why the dims come from the body.
        """
        return declare(
            self,
            id=CONTRACT_ID,
            # `G1WalkEnv` subclasses `MicroduckWalkEnv`, so it inherits the
            # duck's DECIMATION x PHYSICS_DT clock: 50 Hz. Read from there
            # rather than written as 50 because that coupling is real, and a
            # change to it would silently change what this file promises.
            rate_hz=round(1.0 / C.CTRL_DT, 6),
            slots=_obs_slots(self.num_joints),
            deploy="lab contract: base linear velocity is not observed on "
                   "hardware without state estimation")

    def ready(self) -> bool:
        """The MJCF, the meshes and the walker ONNX, not just the scene:
        `g1_scene_xml()` GENERATES a scene, so the base class's "is the
        scene file there" would answer for a file that does not exist
        until something asks for it."""
        return g1_ready()

    def fetch(self):
        from ..fetch_g1 import fetch
        return fetch()

    def look(self) -> str:
        return "g1"

    def visual_scene(self) -> dict:
        return visual_scene()

    def shipped_policies(self) -> tuple[dict, ...]:
        """The Lucky Robots drop, as palette entries.

        Only the ones that speak this robot's observation are listed —
        `shipped_policies()` reads each graph, so a 101-obs croucher or a
        36-obs arm overlay stays out of the palette instead of becoming a
        chip that can never be assigned. Empty, and the group disappears,
        on a machine that never fetched the assets.

        `groupTitle` is the palette's section heading, carried on the entry
        so the viewer needs no table of its own (`lab/robots.shipped_groups`).
        """
        return tuple(
            {"id": f"g1:{e['name']}", "label": e["name"], "group": "g1",
             "groupTitle": f"{self.title} (shipped)",
             "path": e["path"], "robot": self.id, "note": e.get("note", "")}
            for e in shipped_policies() if e["usable"])

    def env_class(self, task: str = "walk") -> type:
        """The env that trains `task` on the G1.

        The G1 is not a `MicroduckWalkEnv` kwarg away: it speaks a
        different observation (99-d, robots/g1_env.py). Imported lazily so
        a machine without the fetched assets can still train the duck.
        """
        from ..train import G1_TASKS
        if task in (None, "", "walk"):
            from .g1_env import G1WalkEnv
            return G1WalkEnv
        if task == "stand":
            from .g1_env import G1StandEnv
            return G1StandEnv
        if task == "squat":
            from .g1_env import G1SquatEnv
            return G1SquatEnv
        if task in ("front_kick", "punch"):
            from .g1_karate import G1FrontKickEnv, G1PunchEnv
            return G1FrontKickEnv if task == "front_kick" else G1PunchEnv
        if task == "imitate":
            # Tracks the clip MICRODUCK_CLIP / --clip names (g1_imitate).
            from .g1_imitate import G1ImitateEnv
            return G1ImitateEnv
        raise SystemExit(
            f"unknown --task {task!r} for {self.id} "
            f"(have: {', '.join(G1_TASKS)})")

    def train_env_kwargs(self, args) -> dict:
        """`train-walk`'s per-body knobs for the G1.

        `HOLD_TASKS` and `G1_TASKS` stay in `train.py`: `--task` is a
        trainer flag and its vocabulary has one definition. §6.1 of
        `docs/mars-roadmap.md` collapses the two task systems and this
        import with them.
        """
        from ..train import HOLD_TASKS
        kw: dict = {}
        # BAM is an XL330 identification; the G1 runs MJCF position servos.
        if getattr(args, "actuator", None) == "bam":
            raise SystemExit(
                "--actuator bam is the duck's XL330 servo model — "
                "the G1 trains on its MJCF position actuators")
        kw["actuator_force"] = "xml"
        task = getattr(args, "task", "walk")
        if task in HOLD_TASKS:
            # Practise LONG holds. The env default is 20 s, and the sway
            # this task is trying to remove damps out at ~20 s — so a
            # default episode is almost entirely the transient, and the
            # settled regime is a tail the policy barely experiences.
            kw["max_episode_s"] = 40.0
            # Fraction of episodes that SHOW a drive command the idle must
            # ignore. Default 0: a clone that has only seen zeros collapses
            # on a commanded episode, so mixing them in from step one feeds
            # PPO a run of one-second disasters. Raise it once the policy
            # holds its ground (see --command-mix).
            # A curriculum stage passes its knobs through the trainer's
            # ENVIRONMENT (the lab's stage machinery does not rewrite
            # argv), so the env var wins when the flag was left at its
            # default.
            mix = getattr(args, "command_mix", 0.0) or 0.0
            if not mix:
                mix = float(os.environ.get("MICRODUCK_G1_COMMAND_MIX", 0.0) or 0.0)
            kw["command_mix"] = float(mix)
        if task == "imitate":
            clip = getattr(args, "clip", None) or os.environ.get("MICRODUCK_CLIP")
            if not clip:
                raise SystemExit("--task imitate needs --clip <name> (or "
                                 "MICRODUCK_CLIP): a clip saved in the 🎬 panel")
            kw["clip_name"] = clip
        return kw

    def attach(self, spec: mujoco.MjSpec, prefix: str, frame,
               collision: str = "walk") -> None:
        """Put one G1 in a `/sim` world. `collision` is the duck's knob
        and has no meaning here — the G1 has one MJCF."""
        spec.attach(g1_spec(), prefix=prefix, frame=frame)


class _LazyG1Spec:
    """`G1_SPEC` without paying `load_config()` at import time."""

    _spec = None

    def _resolve(self):
        if _LazyG1Spec._spec is None:
            _LazyG1Spec._spec = G1Body(
                id="g1",
                title="Unitree G1",
                noun="G1",
                joint_names=joint_names(),
                default_pose=default_pose(),
                action_scale=action_scales(),
                obs_dim=OBS_DIM,
                base_body=BASE_BODY,
                gyro_sensor=GYRO_SENSOR,
                foot_geoms=FOOT_GEOMS,
                scene_fn=g1_scene_xml,
                floor_geom=FLOOR_GEOM,
                stand_keyframe=STAND_KEY,
                pose_joint_ids=None,          # hold every joint near default
                fall_gravity_z=FALL_GRAVITY_Z,
                fall_height=FALL_HEIGHT,
                com_bodies=("torso_link_rev_1_0",),
                lin_vel_x_range=LIN_VEL_X_RANGE,
                lin_vel_y_range=LIN_VEL_Y_RANGE,
                ang_vel_z_range=ANG_VEL_Z_RANGE,
                min_forward_cmd=MIN_FORWARD_CMD,
                twist_obs_slice=(OBS_DIM - CMD_DIM, OBS_DIM),
                push_vel_range=PUSH_VEL_RANGE,
                lab_spacing_m=LAB_SPACING_M,
                joint_groups=JOINT_GROUPS,
                effectors=_effectors(),
                rig_controls=RIG_CONTROLS,
                # Seven 1 cm capsules a foot, no fillet to pick a flat from;
                # 2 mm gathers every capsule's underside. Grounding is judged
                # at 1 cm on a 1.3 m body (the duck's 5 mm on 25 cm).
                sole_tol=0.002,
                ground_tol=0.01,
            )
        return _LazyG1Spec._spec

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self):
        return "RobotSpec(id='g1', lazy)" if _LazyG1Spec._spec is None else repr(self._spec)


G1_SPEC = _LazyG1Spec()


# What each shipped policy DOES, measured in G1WalkEnv (400 steps a command,
# no noise/DR, spawned at the STAND keyframe) rather than guessed from its
# name. The lab shows these as the chip's tooltip.
_SHIPPED_NOTES = {
    "walker": "velocity-commanded gait — 0.6 m/s forward walks 4.2 m in 8 s. "
              "It is also the IDLE: below ~0.4 m/s (its own dead zone) it "
              "plants both feet and "
              "holds still (measured at zero command, 60 s, two seeds: 1.2-1.7 "
              "cm drift, both feet down 100% of steps, no falls), so there is "
              "no separate stand policy to assign",
    "rotator": "turns on the spot under a yaw command (+14 deg in 8 s at "
               "0.8 rad/s) — but FALLS within a second under a forward "
               "command, so drive it with turns only",
}


@lru_cache(maxsize=1)
def shipped_policies() -> tuple[dict, ...]:
    """Every policy that ships in the fetched G1 cache, with MEASURED dims.

    The Lucky Robots drop carries four, and only two of them speak this
    robot's observation (`OBS_DIM` -> `NUM_JOINTS`):

        walker.onnx         99 -> 29   velocity-commanded gait  (usable)
        rotator.onnx        99 -> 29   turning                  (usable)
        croucher.onnx      101 -> 29   two observations we do NOT have a
                                       meaning for — upstream's own run.py
                                       loads it and never drives it, so
                                       guessing what they are would be a
                                       fabrication (not usable)
        right_reacher.onnx  36 ->  7   a right-ARM overlay that rides on top
                                       of the walker, not a body policy
                                       (not usable)

    `usable` is decided by reading the graph, not by a hard-coded list: a
    re-export that changes a width flips the flag instead of handing the lab
    a brain it cannot step.
    """
    d = CACHE_DIR
    if not d.is_dir():
        return ()
    out: list[dict] = []
    for p in sorted(d.glob("*.onnx")):
        try:
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            i, o = sess.get_inputs()[0], sess.get_outputs()[0]
            obs = int(i.shape[1]) if len(i.shape) == 2 and isinstance(i.shape[1], int) else -1
            act = int(o.shape[1]) if len(o.shape) == 2 and isinstance(o.shape[1], int) else -1
        except Exception as e:                      # a corrupt/partial download
            print(f"[g1] skipping {p.name}: {type(e).__name__}: {e}")
            continue
        out.append({
            "name": p.stem, "path": str(p), "obs": obs, "actions": act,
            "usable": obs == OBS_DIM and act == NUM_JOINTS,
            "note": _SHIPPED_NOTES.get(p.stem, ""),
        })
    return tuple(out)


def walker_onnx() -> Path:
    """The 99-obs -> 29-action walking policy the world's G1 runs.

    `MICRODUCK_G1_WALKER` points it at a locally trained export instead of
    the shipped one (`train-walk --robot g1` -> `export-walk`).
    """
    override = os.environ.get("MICRODUCK_G1_WALKER")
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"MICRODUCK_G1_WALKER={override} is not a file")
        return p
    return require_g1() / "walker.onnx"
