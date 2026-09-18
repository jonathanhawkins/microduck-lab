"""The robot seam: `walk_env` used to resolve one body by hard-coded name.

These tests pin that the spec path resolves to EXACTLY those names and ids,
that a spec naming something the model does not have fails loudly (a missing
name used to surface as id -1 and a contact scan that silently never fired),
and that the duck's spec is derived from the contract rather than retyped.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local import symmetry
from microduck_local.robots import spec as spec_mod
from microduck_local.walk_env import MicroduckWalkEnv


def test_duck_spec_is_the_contract_not_a_second_copy():
    s = C.MICRODUCK
    assert s.id == "microduck"
    assert s.joint_names == C.JOINT_NAMES
    assert np.array_equal(s.default_pose, C.DEFAULT_POSE)
    assert s.obs_dim == C.OBS_DIM
    assert s.num_joints == C.NUM_JOINTS
    assert np.array_equal(s.pose_joint_ids, C.LEG_JOINT_IDS)
    assert s.lin_vel_x_range == C.LIN_VEL_X_RANGE
    assert s.ang_vel_z_range == C.ANG_VEL_Z_RANGE
    assert s.action_scale is None          # contract: target = DEFAULT + action


def test_duck_spec_names_are_the_ones_the_env_hard_coded():
    """The literal strings walk_env carried before robots/spec.py."""
    s = C.MICRODUCK
    assert s.base_body == "trunk_base"
    assert s.gyro_sensor == "imu_ang_vel"
    assert s.floor_geom == "floor"
    assert s.stand_keyframe == "STAND"
    assert s.foot_geoms == {"left": ("left_foot_collision",),
                            "right": ("right_foot_collision",)}
    assert s.fall_gravity_z == MicroduckWalkEnv.FALL_GRAVITY_Z
    assert s.fall_height == MicroduckWalkEnv.FALL_HEIGHT


def test_env_resolves_the_same_ids_the_hard_coded_lookups_did():
    env = MicroduckWalkEnv(seed=0)
    m = env.model
    assert env.trunk_body_id == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert env.floor_geom == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert env.foot_geoms["left"] == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    assert env.foot_geoms["right"] == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    assert np.array_equal(env.joint_qpos_adr,
                          [m.joint(n).qposadr[0] for n in C.JOINT_NAMES])
    assert env.key_stand == m.key("STAND").id
    # The duck's actuators ARE its joints in order: the ctrl write stays the
    # whole-array assignment it was.
    assert env._ctrl_whole and np.array_equal(env.ctrl_adr, np.arange(C.NUM_JOINTS))


def test_a_spec_naming_a_missing_geom_fails_at_construction():
    """A wrong foot pad used to be id -1: no contact ever matched, the
    air-time reward silently paid nothing, and nothing raised."""
    bad = dataclasses.replace(
        C.MICRODUCK,
        foot_geoms={"left": ("left_foot_collision",), "right": ("no_such_pad",)})
    with pytest.raises(KeyError, match="no_such_pad"):
        MicroduckWalkEnv(seed=0, robot=bad)


def test_a_spec_naming_a_missing_body_fails_at_construction():
    bad = dataclasses.replace(C.MICRODUCK, base_body="no_such_link")
    with pytest.raises(KeyError, match="no_such_link"):
        MicroduckWalkEnv(seed=0, robot=bad)


def test_foot_contacts_see_every_pad_of_a_multi_geom_foot():
    """The G1 carries seven capsules a foot. The scan keys on a geom->side
    map, so a contact on ANY of them counts as that foot down."""
    env = MicroduckWalkEnv(seed=0)
    assert set(env._foot_side.values()) == {"left", "right"}
    for side, ids in env.foot_geom_ids.items():
        for gid in ids:
            assert env._foot_side[gid] == side
            assert env.model.geom_priority[gid] == 1   # foot friction wins


def test_mirror_permutation_agrees_with_the_symmetry_module():
    assert np.array_equal(C.MICRODUCK.mirror_joint_perm(), symmetry.JOINT_PERM)


def test_a_walker_with_no_scene_is_still_refused_at_construction():
    """`scene_fn` and `stand_keyframe` moved up to `BodyBase`.

    Every body has a model and a pose to spawn in, walker or not, and a
    non-walker (Innate's MARS) needs both — but a field on the base class
    needs a DEFAULT, and inheriting one would have turned "you forgot the
    scene" from a TypeError at construction into a NotImplementedError
    somewhere inside the trainer. `RobotSpec.__post_init__` restates the
    requirement, so this is the planted negative for that guard.
    """
    fields = {f.name: getattr(C.MICRODUCK, f.name)
              for f in dataclasses.fields(C.MICRODUCK)}
    fields.pop("scene_fn")
    with pytest.raises(TypeError, match="no scene_fn"):
        spec_mod.RobotSpec(**fields)
    # And the keyframe still defaults to STAND for a walker that omits it.
    assert spec_mod.RobotSpec(**fields, scene_fn=C.MICRODUCK.scene_fn
                              ).stand_keyframe == "STAND"


def test_a_body_that_declares_no_scene_says_so_when_asked():
    """The `BodyBase.scene_fn` placeholder, and the trap it sidesteps.

    A plain function held as a dataclass DEFAULT is a class attribute, and
    functions are descriptors — so `self.scene_fn()` would have called it
    with `self` and raised a TypeError about arity instead of saying what is
    missing. MEASURED: the generated `__init__` copies the default into the
    instance, so the call arrives with no arguments and the message is the
    useful one. (A `staticmethod` default or a lambda would each behave
    differently again; this pins which of them is in play.)
    """
    from microduck_local.robots.body import BodyBase

    @dataclasses.dataclass(frozen=True, eq=False, kw_only=True)
    class Bare(BodyBase):
        pass

    bare = Bare(id="bare", joint_names=("a",),
                default_pose=np.zeros(1, np.float32), obs_dim=3)
    assert "scene_fn" in bare.__dict__, "the default did not reach the instance"
    with pytest.raises(NotImplementedError, match="declares no scene_fn"):
        bare.scene_fn()


def test_registry_knows_the_duck_and_rejects_nonsense():
    reg = spec_mod.registry()
    assert "microduck" in reg
    assert spec_mod.get("microduck") is C.MICRODUCK
    with pytest.raises(KeyError, match="unknown robot"):
        spec_mod.get("wombat")


def test_scale_action_is_the_duck_contract():
    a = np.full(C.NUM_JOINTS, 0.25, np.float32)
    assert np.allclose(C.MICRODUCK.scale_action(a), C.DEFAULT_POSE + a)
