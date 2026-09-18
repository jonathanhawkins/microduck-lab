"""The posing engine behind the 🎬 editor, for BOTH bodies: pose.PoseScratch.

Forward kinematics + balance are what the editor shows; inverse kinematics
is what a dragged foot or hand asks for. The IK tests check real reach
(the solved joints put the point THERE, verified by forward kinematics, not by
the solver's own report), that a limb only ever moves its own chain, that a
pinned foot stays put, that the centre of mass can be asked for, and that an
unreachable target degrades to "closest" with an honest residual rather
than an error. G1 cases skip on a machine without the fetched assets.
"""

from __future__ import annotations

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.pose import IkTarget, PoseScratch, _sole_points_world, pose_scratch
from microduck_local.robots import g1

ROBOTS = ["microduck", pytest.param("g1", marks=pytest.mark.skipif(
    not g1.g1_ready(), reason="G1 assets missing — uv run fetch-g1"))]


def _default(ps: PoseScratch) -> np.ndarray:
    return ps.spec.default_pose.astype(np.float64)


# ---------------------------------------------------------------- metadata

@pytest.mark.parametrize("robot", ROBOTS)
def test_meta_describes_the_body_it_was_asked_for(robot):
    ps = pose_scratch(robot)
    meta = ps.meta()
    assert meta["robot"] == robot
    assert meta["numJoints"] == ps.spec.num_joints == len(meta["joints"])
    assert [j["name"] for j in meta["joints"]] == list(ps.spec.joint_names)
    # Every joint's group is one of the ordered section labels, in order.
    assert set(j["group"] for j in meta["joints"]) == set(meta["groups"])
    assert meta["groups"] == list(dict.fromkeys(ps.spec.joint_groups))
    assert meta["bodies"][meta["trunkBody"]] == ps.spec.base_body
    assert meta["standHeight"] == pytest.approx(ps.stand_height, abs=1e-4)
    # The rig is the spec's, verbatim, and every part names a real joint.
    assert [r["id"] for r in meta["rig"]] == [r["id"] for r in ps.spec.rig_controls]
    names = set(ps.spec.joint_names) | {"root"}
    for r in meta["rig"]:
        assert set(r["parts"]) <= names, r["id"]
        assert set(r["pick"]) <= names, r["id"]
        assert r["handle"]["joint"] in names, r["id"]
    # Effectors: the spec's plus the centre of mass, each with a chain.
    ids = [e["id"] for e in meta["effectors"]]
    assert ids == [e.id for e in ps.spec.effectors] + ["com"]
    for e in meta["effectors"]:
        assert all(0 <= i < ps.spec.num_joints for i in e["chain"])


def test_the_duck_meta_is_what_the_editor_always_got():
    """The generic engine must reproduce the duck's old hard-coded meta."""
    meta = pose_scratch("microduck").meta()
    assert meta["groups"] == ["left leg", "head + neck", "right leg"]
    assert [j["group"] for j in meta["joints"]] == list(
        ("left leg",) * 5 + ("head + neck",) * 4 + ("right leg",) * 5)
    assert meta["bodies"][meta["trunkBody"]] == "trunk_base"
    assert meta["sizeScale"] == 1.0
    assert [r["id"] for r in meta["rig"]] == [
        "squat", "lean", "swingL", "swingR", "sway", "stance", "twist", "toes", "look"]


@pytest.mark.parametrize("robot", ROBOTS)
def test_every_rig_control_pair_is_orthogonal_and_keeps_the_feet_flat(robot):
    """Two invariants the rig header comment promises: controls read 0 until
    used (pairwise orthogonal in joint space), and every leg-pitch coupling
    preserves the per-leg world-pitch sum so the feet stay level."""
    ps = pose_scratch(robot)
    ctrls = ps.spec.rig_controls
    names = list(ps.spec.joint_names) + ["root"]

    def vec(c):
        v = np.zeros(len(names))
        for k, coeff in c["parts"].items():
            v[names.index(k)] = coeff
        return v

    vs = {c["id"]: vec(c) for c in ctrls}
    for a in vs:
        for b in vs:
            if a < b:
                assert abs(float(vs[a] @ vs[b])) < 1e-9, (a, b)
    # Feet flat: the sagittal (pitch) joints of each leg, with the sign of
    # their WORLD hinge axis, plus the root, must sum to zero per control.
    import mujoco
    m, d = ps.model, ps.data
    ps.solve(_default(ps), 0.0)
    for side in ("left", "right"):
        total = {}
        for c in ctrls:
            s = 0.0
            for k, coeff in c["parts"].items():
                if k == "root":
                    s += coeff
                    continue
                if not k.startswith(side):
                    continue
                if not any(t in k for t in ("hip_pitch", "knee", "ankle_pitch", "ankle")) \
                        or "roll" in k or "yaw" in k:
                    continue
                j = m.joint(k).id
                ax = d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]
                s += coeff * float(np.sign(ax[1]))          # world +y = pitch
            total[c["id"]] = s
        for cid, s in total.items():
            assert abs(s) < 1e-9, (side, cid, s)
    del mujoco


# ------------------------------------------------------------ FK + balance

@pytest.mark.parametrize("robot", ROBOTS)
def test_standing_pose_is_grounded_on_both_soles_and_balanced(robot):
    ps = pose_scratch(robot)
    ps.solve(_default(ps), 0.0)
    bal = ps.balance()
    assert bal["feet"]["left"]["grounded"] and bal["feet"]["right"]["grounded"]
    assert bal["support"]["feet"] == ["left", "right"]
    assert bal["support"]["marginMm"] > 0          # standing square STANDS
    assert bal["over"] is None                     # ...and over neither sole
    # The support polygon contains both soles' outlines.
    for side in ("left", "right"):
        assert len(bal["feet"][side]["outline"]) >= 3


def test_capsule_soles_print_a_stadium_the_size_of_the_capsule():
    """The G1's foot is seven capsules: each prints a stadium 2r wide and
    2(h + r) long, at the capsule's lowest point."""
    if not g1.g1_ready():
        pytest.skip("G1 assets missing")
    ps = pose_scratch("g1")
    ps.solve(_default(ps), 0.0)
    m, d = ps.model, ps.data
    g = m.geom("left_foot4_collision").id
    r, h = float(m.geom_size[g][0]), float(m.geom_size[g][1])
    pts = _sole_points_world(m, d, g, ps.spec.sole_tol)
    # A ring under each end, a radius below that end (the capsule lies a
    # hair off horizontal, so the two rings sit ~1 mm apart in z).
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * h
    ends_z = sorted([d.geom_xpos[g][2] - axis[2], d.geom_xpos[g][2] + axis[2]])
    assert pts[:, 2].min() == pytest.approx(ends_z[0] - r, abs=1e-6)
    assert pts[:, 2].max() == pytest.approx(ends_z[1] - r, abs=1e-6)
    ext = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    assert max(ext) == pytest.approx(2 * (h + r), abs=1e-3)
    assert min(ext) == pytest.approx(2 * r, abs=1e-3)


@pytest.mark.parametrize("robot", ROBOTS)
def test_lifting_a_foot_reads_as_in_the_air(robot):
    ps = pose_scratch(robot)
    q = _default(ps)
    ps.solve(q, 0.0)
    right = np.array(ps.effector_point("right_foot"))
    k = 0.05 * ps.size_scale
    res = ps.solve_ik(q, 0.0, {"right_foot": IkTarget(pos=(right[0], right[1], right[2] + k))},
                      pins=("left_foot",))
    ps.solve(res.joints, 0.0)
    bal = ps.balance()
    assert bal["feet"]["left"]["grounded"]
    assert not bal["feet"]["right"]["grounded"]
    assert bal["support"]["feet"] == ["left"]


# ----------------------------------------------------------------------- IK

@pytest.mark.parametrize("robot", ROBOTS)
def test_ik_reaches_a_foot_target_and_moves_only_that_leg(robot):
    ps = pose_scratch(robot)
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    p = np.array(ps.effector_point("right_foot"))
    k = 0.06 * ps.size_scale
    target = (p[0] + k, p[1], p[2] + k)
    res = ps.solve_ik(q0, 0.0, {"right_foot": IkTarget(pos=target)}, pins=())
    assert res.converged, res.residual
    # Verified by FORWARD kinematics, not by the solver's own residual.
    ps._pose(res.joints, 0.0)
    got = ps.effector_point("right_foot")
    assert np.linalg.norm(got - np.array(target)) < 2e-3 * ps.size_scale
    chain = set(ps.effectors["right_foot"]["chain"])
    moved = {i for i in range(len(q0)) if abs(res.joints[i] - q0[i]) > 1e-9}
    assert moved <= chain, "IK moved a joint outside the limb's chain"
    assert moved, "nothing moved"
    assert np.all(res.joints >= ps.limits[:, 0] - 1e-12)
    assert np.all(res.joints <= ps.limits[:, 1] + 1e-12)


@pytest.mark.parametrize("robot", ROBOTS)
def test_a_pinned_foot_stays_where_it_was(robot):
    ps = pose_scratch(robot)
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    left0 = np.array(ps.effector_point("left_foot"))
    right = np.array(ps.effector_point("right_foot"))
    k = 0.06 * ps.size_scale
    res = ps.solve_ik(q0, 0.0, {"right_foot": IkTarget(pos=(right[0] + k, right[1], right[2] + k))},
                      pins=("left_foot",))
    assert res.converged, res.residual
    assert res.residual["left_foot"] < 2e-3 * ps.size_scale
    # ...by forward kinematics too, not only by the solver's own account:
    # `data` holds the solve's final state (root shift included — with two
    # limbs in play the pelvis is free to move, see solve_ik).
    assert np.linalg.norm(ps.effector_point("left_foot") - left0) < 3e-3 * ps.size_scale


def test_the_centre_of_mass_can_be_put_over_a_foot():
    """The balance request that used to take a 3-parameter grid search
    (g1_karate._solve_strike): CoM over the left sole, both feet planted."""
    if not g1.g1_ready():
        pytest.skip("G1 assets missing")
    ps = pose_scratch("g1")
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    left = np.array(ps.effector_point("left_foot"))
    assert ps.balance()["over"] is None
    res = ps.solve_ik(q0, 0.0, {"com": IkTarget(pos=(left[0], left[1], 0.0), weight=2.0)},
                      pins=("left_foot", "right_foot"))
    assert res.converged, res.residual
    ps.solve(res.joints, 0.0)
    bal = ps.balance()
    assert bal["over"] == "left"
    assert bal["feet"]["left"]["marginMm"] > 20
    assert bal["feet"]["right"]["grounded"]


def test_a_high_kick_is_solved_balanced_over_the_support_foot():
    """The pose the karate env solved by grid search, from the IK: right foot
    half a metre up and forward, weight over the left sole."""
    if not g1.g1_ready():
        pytest.skip("G1 assets missing")
    ps = pose_scratch("g1")
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    left = np.array(ps.effector_point("left_foot"))
    right = np.array(ps.effector_point("right_foot"))
    res = ps.solve_ik(q0, 0.0, {
        "right_foot": IkTarget(pos=(right[0] + 0.5, right[1], right[2] + 0.55)),
        "com": IkTarget(pos=(left[0], left[1], 0.0), weight=2.0),
    }, pins=("left_foot",))
    assert res.converged, res.residual
    ps.solve(res.joints, 0.0)
    bal = ps.balance()
    assert bal["over"] == "left"
    l_, r_ = ps.effector_point("left_foot"), ps.effector_point("right_foot")
    assert r_[2] - l_[2] > 0.5


@pytest.mark.parametrize("robot", ROBOTS)
def test_an_unreachable_target_degrades_to_closest_with_an_honest_residual(robot):
    ps = pose_scratch(robot)
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    p = np.array(ps.effector_point("right_foot"))
    far = (p[0] + 10.0, p[1], p[2])
    res = ps.solve_ik(q0, 0.0, {"right_foot": IkTarget(pos=far)}, pins=())
    assert not res.converged
    assert res.residual["right_foot"] > 5.0
    assert np.isfinite(res.joints).all()
    assert np.all(res.joints >= ps.limits[:, 0] - 1e-12)
    assert np.all(res.joints <= ps.limits[:, 1] + 1e-12)
    # ...and the leg did reach as far as it could in that direction.
    ps._pose(res.joints, 0.0)
    assert ps.effector_point("right_foot")[0] > p[0] + 0.05 * ps.size_scale


def test_an_unknown_effector_is_refused_by_name():
    ps = pose_scratch("microduck")
    with pytest.raises(KeyError, match="no effector named 'tail'"):
        ps.solve_ik(_default(ps), 0.0, {"tail": IkTarget(pos=(0, 0, 0))})


def test_solve_ik_is_fast_enough_for_a_drag():
    """A pointer drag sends one solve per move; ~1 ms is what keeps it live."""
    import time
    ps = pose_scratch("microduck")
    q0 = _default(ps)
    ps.solve(q0, 0.0)
    p = np.array(ps.effector_point("right_foot"))
    t0 = time.perf_counter()
    for _ in range(20):
        ps.solve_ik(q0, 0.0, {"right_foot": IkTarget(pos=(p[0] + 0.03, p[1], p[2] + 0.03))},
                    pins=("left_foot",))
    assert (time.perf_counter() - t0) / 20 < 0.05


def test_pose_scratch_registry_is_one_model_per_robot():
    assert pose_scratch("microduck") is pose_scratch("duck") is pose_scratch("")
    if g1.g1_ready():
        assert pose_scratch("g1") is not pose_scratch("microduck")
        assert pose_scratch("g1").spec.num_joints == 29
    with pytest.raises(KeyError):
        pose_scratch("spot")


def test_the_duck_spec_still_names_the_contract_joints():
    assert C.MICRODUCK.joint_groups is not None
    assert len(C.MICRODUCK.joint_groups) == C.NUM_JOINTS
    assert {e.id for e in C.MICRODUCK.effectors} == {"left_foot", "right_foot", "head"}


def test_a_non_walker_is_refused_by_the_editor_with_a_404_not_a_500(monkeypatch):
    """A third body that does not walk is in the registry now, and the lab's
    `/robots` list hands every registry entry to the 🎬 editor. `PoseScratch`
    reads `base_body`, effectors and a stand height — a walker's things — so
    without this guard a click on MARS died as `AttributeError` (a 500). A
    `KeyError` is what `viz_server.scratch_for` reports as a 404, with a
    sentence that says where the arm editor is planned."""
    from microduck_local import pose
    from microduck_local.robots import registry as R
    from microduck_local.robots.body import BodyBase

    class Wheelie(BodyBase):
        def ready(self):
            return True

    body = Wheelie(id="wheelie", joint_names=("j1",), default_pose=[0.0],
                   obs_dim=4, kind="wheeled")
    monkeypatch.setitem(R._EXTRA, "wheelie", body)
    monkeypatch.setattr(pose, "_scratch", {})
    with pytest.raises(KeyError, match="no animate support"):
        pose.pose_scratch("wheelie")
    assert "wheelie" not in pose._scratch

