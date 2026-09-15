"""Left/right symmetry for the Unitree G1.

`symmetry.py` derives the duck's mirror from `contract.JOINT_NAMES` and then
asserts it against microduck_rl's own tables. The G1 has no upstream table to
check against, and its sign rule is NOT the duck's: the duck negates almost
everything because its left and right home frames are sign-flipped twins,
while the G1's joints are plain hinges whose behaviour under a sagittal
reflection is decided by their AXIS.

So the sign is derived from the compiled model's `jnt_axis`, not from names:

    axis ±y (pitch: hip_pitch, knee, ankle_pitch, shoulder_pitch, elbow,
             waist_pitch, wrist_pitch)   -> +1, a reflection preserves it
    axis ±x (roll) or ±z (yaw)           -> -1, a reflection reverses it

and then it is CHECKED against physics rather than argued about: mirror a
state, let MuJoCo recompute, and the observation of the mirrored state must
equal the mirror of the observation (tests/test_g1_symmetry.py).
"""

from __future__ import annotations

import numpy as np

from . import g1

# Which obs block is where, for the 99-d layout (robots/g1_env.py).
_LIN, _ANG, _GRAV = slice(0, 3), slice(3, 6), slice(6, 9)


def joint_perm() -> np.ndarray:
    """left_X <-> right_X by name (robots/spec.py)."""
    return g1.G1_SPEC.mirror_joint_perm()


def joint_sign(model=None) -> np.ndarray:
    """+1 for hinges about the lateral axis, -1 for roll and yaw hinges.

    Read off the model when one is given (the honest source), else off the
    generated training scene.
    """
    import mujoco

    if model is None:
        model = mujoco.MjModel.from_xml_path(str(g1.g1_scene_xml()))
    sign = np.empty(g1.NUM_JOINTS, np.float32)
    for i, name in enumerate(g1.joint_names()):
        axis = np.asarray(model.joint(name).axis, float)
        if abs(axis[1]) > 0.99:              # pitch: rotation about +-y
            sign[i] = 1.0
        elif abs(axis[0]) > 0.99 or abs(axis[2]) > 0.99:
            sign[i] = -1.0
        else:
            raise ValueError(
                f"G1 joint {name} has axis {axis} — neither a pure pitch nor a "
                "pure roll/yaw hinge, so its mirror sign is not derivable here")
    return sign


def obs_perm_sign(model=None) -> tuple[np.ndarray, np.ndarray]:
    """(perm, sign) for the full 99-d observation, in G1WalkEnv order."""
    jp = joint_perm()
    js = joint_sign(model)
    nj = g1.NUM_JOINTS
    perm = np.concatenate([
        np.arange(0, 9),          # lin_vel, ang_vel, gravity mirror in place
        9 + jp,                   # joint_pos_rel
        9 + nj + jp,              # joint_vel
        9 + 2 * nj + jp,          # last_action
        np.arange(9 + 3 * nj, 9 + 3 * nj + 3),   # twist command
    ]).astype(np.int64)
    sign = np.concatenate([
        [1.0, -1.0, 1.0],         # base linear velocity: negate vy
        [-1.0, 1.0, -1.0],        # base angular velocity: negate roll, yaw
        [1.0, -1.0, 1.0],         # projected gravity: negate gy
        js, js, js,
        [1.0, -1.0, -1.0],        # twist: negate vy, wz
    ]).astype(np.float32)
    assert perm.shape == (g1.OBS_DIM,) and sign.shape == (g1.OBS_DIM,)
    return perm, sign


def mirror_obs(obs: np.ndarray, model=None) -> np.ndarray:
    perm, sign = obs_perm_sign(model)
    return (sign * np.asarray(obs)[..., perm]).astype(np.float32)


def mirror_action(action: np.ndarray, model=None) -> np.ndarray:
    js = joint_sign(model)
    return (js * np.asarray(action)[..., joint_perm()]).astype(np.float32)


def mirror_state(model, data) -> None:
    """Reflect the whole physical state across the robot's sagittal plane.

    In place, and the caller forwards. Used to CHECK the tables above against
    physics: MuJoCo recomputes the IMU, the contacts and the kinematics of
    the mirrored body, and the resulting observation must be the mirror of
    the original's.
    """
    import mujoco

    jp, js = joint_perm(), joint_sign(model)
    qadr = np.array([model.joint(n).qposadr[0] for n in g1.joint_names()])
    vadr = np.array([model.joint(n).dofadr[0] for n in g1.joint_names()])
    q, v = data.qpos, data.qvel
    # base: y flips, and so do the rotation components about x and z
    q[1] = -q[1]
    w, x, y, z = q[3], q[4], q[5], q[6]
    q[3:7] = [w, -x, y, -z]
    v[1] = -v[1]                       # linear velocity, world frame
    v[3], v[5] = -v[3], -v[5]          # angular velocity, body frame
    q[qadr] = js * q[qadr][jp]
    v[vadr] = js * v[vadr][jp]
    mujoco.mj_forward(model, data)
