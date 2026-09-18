"""The Innate MARS body: the download, the URDF rewrites, the model, the servo.

`tests/test_body_conformance.py` is the definition of "supported" and holds
MARS to everything a body must satisfy (its scene compiles, its names
resolve, its numbers agree, it attaches beside a duck, its stage pitch
matches its measured width, the viewer can see it, and it holds its arm at
HOME). This file is what that suite does NOT cover, and it is mostly the
things `robots/mars.py` inherited from somebody else's repo:

* the asset manifest and its sha256s — the only thing between a moved
  revision or a captive-portal HTML page and a compiled robot;
* the three URDF rewrites, each of which is invisible when it works and
  silent when it does not (no meshes; no `ee_link`; 31 g of missing markers);
* Innate's contact tuning, where every constant is load-bearing for a grasp;
* the arm servo, and the joint2 guard that keeps the arm out of the head;
* the answers that are deliberately refusals — no env, no tasks, no shipped
  policies — so a phase that lands one has to come here and say so (Phase 3a
  landed `driver()`, and this file says so in two places rather than one).

Network: none. `mars._download` is the seam; every fetch case replaces it,
and the cases that need real bytes are skipped without the assets.
"""

from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.robots import mars
from microduck_local.robots import registry as R
from microduck_local.robots.body import conforms
from microduck_local.robots.spec import RobotSpec

needs_mars = pytest.mark.skipif(
    not mars.mars_ready(), reason="MARS assets missing — uv run fetch-robot mars")


# ------------------------------------------------------------- the manifest

def test_the_manifest_is_eleven_pinned_files_under_one_revision():
    """Eleven paths, eleven distinct sha256s, one sha in the URL.

    The manifest is the whole trust story of a body whose assets live in
    someone else's repository: a tag that moves, a `main` that advances or a
    proxy that answers with a login page are all the same event to a
    downloader, and only the hash tells them from the robot.
    """
    assert len(mars.ASSETS) == 11
    paths = [p for p, _ in mars.ASSETS]
    hashes = [h for _, h in mars.ASSETS]
    assert len(set(paths)) == 11, "a path is listed twice"
    assert len(set(hashes)) == 11, "two files share a hash"
    assert sorted(paths) == sorted(
        ["urdf/mars.urdf", "urdf/arm.srdf"]
        + [f"meshes/{m}.STL" for m in ("base", "head", "link1", "link2",
                                       "link3", "link4", "link5", "link61",
                                       "link62")])
    for h in hashes:
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert len(mars.INNATE_OS_SHA) == 40
    assert mars.INNATE_OS_SHA in mars.RAW_BASE, (
        "the download URL must carry the pinned revision, or the manifest "
        "describes a different tree than the one being fetched")
    assert "Apache-2.0" in mars.INNATE_OS_LICENCE


@needs_mars
def test_every_manifest_hash_matches_the_file_on_disk():
    """The bytes in the cache ARE the pinned revision's.

    Re-hashed here rather than trusted from the download, because the file
    on disk is what `load_robot_spec` reads and the rewrites are applied to
    a COPY in memory — so nothing else in the tree would notice an edited
    STL.
    """
    d = mars.asset_dir()
    for rel, want in mars.ASSETS:
        assert (d / rel).is_file(), rel
        assert mars._sha256(d / rel) == want, f"{rel} is not the pinned file"


# ------------------------------------------------------------- the download

_FAKE = (("urdf/mars.urdf", hashlib.sha256(b"<robot/>").hexdigest()),
         ("meshes/base.STL", hashlib.sha256(b"solid\n").hexdigest()))
_FAKE_BYTES = {"urdf/mars.urdf": b"<robot/>", "meshes/base.STL": b"solid\n"}


def _fake_downloader(calls: list[str], payload=None):
    """A `_download` that writes known bytes and records what was asked for."""
    def download(url: str, dest):
        rel = url[len(mars.RAW_BASE) + 1:]
        calls.append(rel)
        dest.write_bytes((payload or _FAKE_BYTES)[rel])
    return download


def test_fetch_downloads_every_missing_file_and_verifies_it(tmp_path, monkeypatch,
                                                           capsys):
    """One download per file, each checked against the manifest."""
    calls: list[str] = []
    monkeypatch.setattr(mars, "ASSETS", _FAKE)
    monkeypatch.setattr(mars, "_download", _fake_downloader(calls))
    out = mars.fetch(tmp_path)
    assert out == tmp_path
    assert sorted(calls) == sorted(p for p, _ in _FAKE)
    assert mars.mars_ready(tmp_path)
    assert (mars.asset_dir(tmp_path) / "urdf/mars.urdf").read_bytes() == b"<robot/>"
    assert "2 file(s) downloaded" in capsys.readouterr().out


def test_a_second_fetch_downloads_nothing(tmp_path, monkeypatch):
    """Idempotence, which is what makes `fetch-robot mars` safe to re-run and
    what `scripts/setup.sh` will lean on: a present file that hashes right is
    skipped, so the second call costs 11 hashes and no bytes."""
    calls: list[str] = []
    monkeypatch.setattr(mars, "ASSETS", _FAKE)
    monkeypatch.setattr(mars, "_download", _fake_downloader(calls))
    mars.fetch(tmp_path)
    assert len(calls) == 2
    mars.fetch(tmp_path)
    assert len(calls) == 2, f"a second fetch re-downloaded {calls[2:]}"


def test_fetch_refuses_a_file_whose_hash_is_not_the_manifests(tmp_path,
                                                              monkeypatch):
    """The planted regression for the manifest: wrong bytes, refused.

    And refused WITHOUT leaving anything behind — the temp file is cleaned
    and the target never appears, so a failed fetch cannot be followed by a
    `ready()` that says yes.
    """
    wrong = {"urdf/mars.urdf": b"<html>sign in</html>", "meshes/base.STL": b"solid\n"}
    monkeypatch.setattr(mars, "ASSETS", _FAKE)
    monkeypatch.setattr(mars, "_download", _fake_downloader([], wrong))
    with pytest.raises(RuntimeError, match="hashes .* the manifest says"):
        mars.fetch(tmp_path)
    assert not (mars.asset_dir(tmp_path) / "urdf/mars.urdf").exists()
    assert not mars.mars_ready(tmp_path)
    assert not list(mars.asset_dir(tmp_path).glob("urdf/.*")), "a temp file was left"


def test_fetch_replaces_a_file_that_was_corrupted_after_it_arrived(tmp_path,
                                                                   monkeypatch):
    """A cached file that no longer hashes right is re-downloaded, not kept.

    `ready()` only checks presence (it is asked per roster change), so this
    is the case that makes that cheap answer safe: the next fetch repairs
    what a truncated write, a half-finished copy or an edit left behind.
    """
    calls: list[str] = []
    monkeypatch.setattr(mars, "ASSETS", _FAKE)
    monkeypatch.setattr(mars, "_download", _fake_downloader(calls))
    mars.fetch(tmp_path)
    victim = mars.asset_dir(tmp_path) / "meshes/base.STL"
    victim.write_bytes(b"truncated")
    mars.fetch(tmp_path)
    assert calls[-1] == "meshes/base.STL"
    assert victim.read_bytes() == b"solid\n"


def test_ready_is_false_on_an_empty_cache_and_the_hint_says_what_to_run(
        tmp_path, monkeypatch):
    """The palette's "not set up yet — ⤓" affordance rests on this answer."""
    monkeypatch.setattr(mars, "CACHE_DIR", tmp_path / "nothing-here")
    assert mars.mars_ready() is False
    assert mars.MARS.ready() is False
    assert mars.MARS.setup_hint() == "uv run fetch-robot mars"
    with pytest.raises(FileNotFoundError, match="uv run fetch-robot mars"):
        mars.require_mars()


def test_the_registry_lists_mars_on_a_machine_that_never_fetched_it(monkeypatch):
    """`--robot mars` must be ACCEPTED and then answered with the download
    command, exactly as `--robot g1` is: an argparse "invalid choice" would
    tell somebody their robot does not exist when it is one command away.
    `ids()` therefore reads the DECLARATION, not the loaded body."""
    monkeypatch.setattr(R, "_load_builtin", lambda b: None)
    assert R.ids() == ("microduck", "g1", "mars")
    assert R.setup_hint("mars") == "uv run fetch-robot mars"
    with pytest.raises(KeyError, match="uv run fetch-robot mars"):
        R.get("mars")


def test_fetch_robot_with_no_argument_lists_mars(capsys):
    """`uv run fetch-robot` is the inventory, and a body absent from it is a
    body nobody knows how to install."""
    from microduck_local.fetch_robot import main

    main([])
    out = capsys.readouterr().out
    assert "mars" in out
    assert "uv run fetch-robot mars" in out


# ---------------------------------------------------------------- the body

def test_mars_is_a_body_and_deliberately_not_a_robot_spec():
    """The design decision, as a test.

    `RobotSpec` is "what the WALKING env needs to know about a robot" — foot
    geoms, fall thresholds, air-time windows, a gyro. MARS has none of them
    and a wheeled base cannot fall, so it is a `BodyBase`. If somebody ever
    makes it a `RobotSpec` to reuse a reward or an env, this fails and says
    why: the walker's fields would have to be invented, and an invented fall
    height is a threshold a policy can satisfy by doing nothing.
    """
    assert conforms(mars.MARS) == ()
    assert not isinstance(mars.MARS, RobotSpec)
    assert mars.MARS.kind == "wheeled"
    assert (mars.MARS.id, mars.MARS.noun, mars.MARS.title) == (
        "mars", "MARS", "Innate MARS")
    assert mars.MARS.look() == "mars"
    assert R.get("mars") is mars.MARS


#: Every constant this module carries from Innate's simulator, as a LITERAL.
#: Written out here rather than read from `mars` so that the test below
#: compares two independent statements of the number. Asserting
#: `model.geom_friction[i] == mars.FINGER_FRICTION` is a tautology the moment
#: the question is "did the ported value drift" — and it was: the first draft
#: of this file did exactly that, and a planted `FINGER_ARMATURE = 0.0` and a
#: planted `GRIPPER_CLOSED_ON_AIR_RAD = 0.0` both passed it.
#:
#: Source (innate-os at the pinned sha):
#:   world.py   FINGER_*, WHEEL_GEOMS, GRIPPER_CLOSED_ON_AIR_RAD, ARM_HOME,
#:              STRUCT_STIFFNESS, ARM_BACKLASH_RAD, BACKLASH_TANH_NM
#:   core.py    KP_JOINT, KD_JOINT, EFFORT_LIMIT, GRIPPER_EFFORT_LIMIT,
#:              KD_GRIPPER, JOINT2_GUARD_MIN
INNATE_CONSTANTS = {
    "KP_JOINT": 50.0,
    "KD_JOINT": 1.0,
    "EFFORT_LIMIT": 50.0,
    "GRIPPER_EFFORT_LIMIT": 2.0,
    "KD_GRIPPER": 0.0,
    "JOINT2_GUARD_MIN": -0.25,
    "GRIPPER_CLOSED_ON_AIR_RAD": -0.085,
    "FINGER_CONDIM": 6,
    "FINGER_FRICTION": (2.0, 0.05, 0.02),
    "FINGER_SOLREF": (0.005, 1.0),
    "FINGER_SOLIMP": (0.95, 0.99, 0.001, 0.5, 2),
    "FINGER_DAMPING": 1.0,
    "FINGER_ARMATURE": 1e-4,
    "STRUCT_STIFFNESS": 25.0,
    "ARM_BACKLASH_RAD": 0.055,
    "BACKLASH_TANH_NM": 0.05,
    "CONTROL_HZ": 25.0,
}


def test_every_constant_carried_from_innate_is_still_theirs():
    """The port has not drifted.

    Each of these was measured by Innate against something failing — a
    grasped object creeping out of the claw, 1.1 kN of pinch ejecting it,
    12 g blades sinking into whatever they touch, an arm sweeping through
    the head. None of them is a tuning knob for this harness, and a number
    changed here is a robot that behaves differently in this lab than on
    their machine while claiming to be the same recipe.
    """
    for name, want in INNATE_CONSTANTS.items():
        got = getattr(mars, name)
        if isinstance(want, tuple):
            assert tuple(got) == want, name
        else:
            assert got == pytest.approx(want), name
    assert mars.ARM_HOME["joint1"] == 1.445009902188274
    assert mars.ARM_HOME["joint6"] == 0.0015339807878856412
    assert mars.WHEEL_GEOMS == ("base_wheel_left", "base_wheel_right")
    assert mars.FINGER_LINKS == ("link61", "link62")


def test_the_spawn_keyframe_is_named_home():
    """The keyframe's NAME is a contract, not an implementation detail.

    `docs/mars-roadmap.md` Phase 2 names it, the lab and `render-rollout`
    spawn a body by `stand_keyframe`, and a scenario or a test elsewhere may
    hard-code the string. Renaming it inside this module is self-consistent
    — MEASURED: every other test in this file and the whole conformance
    suite still pass with the keyframe called NOT_A_KEY — so the name needs
    its own assertion or nothing holds it.
    """
    assert mars.HOME_KEY == "HOME"
    assert mars.MARS.stand_keyframe == "HOME"


def test_the_policy_contract_is_six_joints_eight_actions_and_thirty_two_floats():
    """The widths the lab, the exporter and Phase 4's env all read.

    `num_actions != num_joints` is the point: six joint targets plus the
    base twist. `BodyBase`'s default is one action per joint, which would
    have handed a MARS policy 6 outputs and silently dropped the wheels.
    """
    assert mars.MARS.joint_names == ("joint1", "joint2", "joint3", "joint4",
                                     "joint5", "joint6")
    assert mars.MARS.num_joints == 6
    assert mars.MARS.num_actions == 8 == mars.NUM_ACTIONS
    assert mars.MARS.obs_dim == 32 == mars.OBS_DIM
    assert len(mars.MARS.default_pose) == 6
    assert len(mars.MARS.joint_groups) == 6


def test_the_observation_layout_tiles_the_contract_exactly():
    """The v1 obs layout is a set of slices, and nothing fills it yet.

    So this is the only thing standing between a documented layout and one
    that overlaps or leaves a hole: Phase 4 will index these constants, and
    two slices sharing a float is a task reading another task's channel with
    no error anywhere. Checked as a tiling — contiguous, in order, ending at
    `OBS_DIM` — rather than field by field.
    """
    layout = [mars.OBS_ARM_QPOS, mars.OBS_ARM_QVEL, mars.OBS_HEAD_PITCH,
              mars.OBS_GRIPPER_LOAD, mars.OBS_LAST_ACTION,
              mars.OBS_TARGET_BASE, mars.OBS_TARGET_SEEN,
              mars.OBS_BASE_TWIST, mars.OBS_RESERVED]
    at = 0
    for s in layout:
        assert s.start == at, f"{s} does not start where the last one ended ({at})"
        assert s.stop > s.start
        at = s.stop
    assert at == mars.OBS_DIM
    # The slots whose width is a body fact, not a choice.
    assert mars.OBS_ARM_QPOS.stop - mars.OBS_ARM_QPOS.start == mars.MARS.num_joints
    assert mars.OBS_ARM_QVEL.stop - mars.OBS_ARM_QVEL.start == mars.MARS.num_joints
    assert (mars.OBS_LAST_ACTION.stop - mars.OBS_LAST_ACTION.start
            == mars.MARS.num_actions)
    assert mars.ACT_ARM.stop == mars.ACT_BASE_TWIST.start
    assert mars.ACT_BASE_TWIST.stop == mars.NUM_ACTIONS


def test_an_overlapping_obs_slot_is_caught():
    """The planted regression for the tiling above: two slots sharing a
    float must not read as a valid layout."""
    layout = [slice(0, 6), slice(5, 12)]          # one float over
    at = 0
    ok = True
    for s in layout:
        ok = ok and s.start == at
        at = s.stop
    assert not ok


def test_the_head_is_a_command_slot_not_a_policy_joint():
    """`joint_head` and `joint6M` are real joints and neither is an action.

    The head is the duck's neck all over again — a command the operator or a
    brain sets, not something a task policy is scored on — and `joint6M` is
    a `<mimic>` in the URDF driven by the gripper's geartrain. Putting
    either in `joint_names` would widen the contract by two and give a
    policy two outputs that fight the servo.
    """
    assert mars.HEAD_JOINT == "joint_head"
    assert mars.HEAD_JOINT not in mars.MARS.joint_names
    assert mars.MIMIC_JOINT[0] not in mars.MARS.joint_names
    assert mars.MIMIC_JOINT == ("joint6M", "joint6", -1.0)
    assert mars.DRIVEN_JOINTS == mars.MARS.joint_names + (mars.HEAD_JOINT,)


def test_the_default_pose_is_innates_arm_home_in_joint_order():
    """`default_pose` is what a zero action means and where the lab parks an
    unpolicied slot, so it has to be Innate's own home to the digit — their
    webapp's ARM_HOME_POSITIONS, which is the pose the real arm folds to."""
    assert np.allclose(mars.MARS.default_pose,
                       [mars.ARM_HOME[j] for j in mars.MARS.joint_names])
    assert mars.ARM_HOME["joint_head"] == 0.0
    assert set(mars.ARM_HOME) == set(mars.DRIVEN_JOINTS)


def test_the_answers_that_are_refusals_name_the_phase_that_lands_them():
    """A body with no env, no tasks and no shipped policy.

    Each of these could return something plausible — the duck's env, an
    empty walker, a duck recipe — and each would be a wrong answer that only
    shows up as bad behaviour. Raising with the phase number is also how the
    next person finds out where the work goes.

    `driver()` was on this list until Phase 3a landed
    `robots/mars_drive.py`; it is now a real answer and
    `tests/test_mars_drive.py` measures it, which is what a refusal turning
    into an implementation is supposed to look like from here.
    """
    with pytest.raises(NotImplementedError, match="Phase 4"):
        mars.MARS.env_class("reach")
    assert mars.MARS.tasks() == ()
    assert mars.MARS.shipped_policies() == ()
    assert mars.MARS.train_env_kwargs(None) == {}


@needs_mars
def test_the_body_hands_out_a_driver_for_the_prefix_it_is_asked_about():
    """`Body.driver(model, prefix)` — the seam `/sim` steps a MARS through.

    Here rather than in `tests/test_mars_drive.py` because what is being
    checked is the BODY's answer: that the registry's MARS returns a driver
    bound to the model and prefix it was given, so a room with two of them
    gets two independent controllers. The drive itself is measured next door.
    """
    m = mars.model()
    drv = mars.MARS.driver(m, "")
    assert type(drv).__name__ == "MarsDriver"
    assert drv.model is m and drv.prefix == ""
    assert drv.base_id == mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_BODY, mars.BASE_BODY)
    with pytest.raises(KeyError, match="attached under this prefix"):
        mars.MARS.driver(m, "m0/")     # nothing is attached under m0/ here


# ------------------------------------------------------- the URDF rewrites

@needs_mars
def test_the_discardvisual_rewrite_keeps_all_nine_meshes():
    """MuJoCo's URDF importer DISCARDS `<visual>` geometry by default.

    Without Innate's embedded compiler override the robot still simulates —
    it just has no shell, which reaches a person as an invisible robot on the
    stage and nothing anywhere as an error.
    """
    m = mars.robot_spec().compile()
    assert m.nmesh == 9
    visual = [i for i in range(m.ngeom)
              if int(m.geom_group[i]) == mars.VISUAL_GROUP]
    collision = [i for i in range(m.ngeom) if int(m.geom_contype[i]) == 1]
    assert len(visual) == 12, "9 meshes + the 3 marker spheres"
    assert len(collision) == 46
    assert set(visual) & set(collision) == set()


@needs_mars
def test_without_the_rewrites_the_meshes_and_the_frames_are_gone():
    """The planted regression for both of the rewrites that matter.

    Loading the same URDF with the importer's own defaults must lose the
    meshes (discardvisual) and the static frames (fusestatic) — if it ever
    stops doing so, the rewrites are dead code and the test above is
    passing for a reason that has nothing to do with them.
    """
    path = mars.urdf_path()
    plain = path.read_text().replace(
        "package://mars_description/", str(path.parent.parent.resolve()) + "/")
    m = mujoco.MjSpec.from_string(plain).compile()
    assert m.nmesh == 0, "discardvisual no longer discards"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_link") < 0, (
        "fusestatic no longer fuses")


@needs_mars
def test_the_static_frames_survive_and_an_attach_does_not_lose_their_mass():
    """`fusestatic="false"`, and the 31 g that proves why it is there.

    MEASURED on this URDF: with the importer's default (fuse ON), compiling
    the robot alone gives 10 bodies and 1.3650 kg, and attaching that
    already-compiled spec into a world gives 10 bodies and 1.3340 kg — the
    marker links' mass is silently dropped the second time. The lab builds
    the viewer's dump from one model and streams poses from another, indexed
    positionally, so two compiles that disagree on the body list draw a robot
    with its parts on the wrong joints.

    The names also have to exist for their own sake: Phase 3 mounts the lidar
    on `base_laser` and the camera on `head_camera_left`, and Phase 4's
    reward reads `ee_link`.
    """
    alone = mars.robot_spec().compile()
    assert alone.nbody == 18
    assert float(sum(alone.body_mass)) == pytest.approx(1.365, abs=5e-4)
    for name in (mars.BASE_BODY, mars.LIDAR_SITE, mars.CAMERA_BODY,
                 mars.EFFECTOR_BODY, "base_footprint"):
        assert mujoco.mj_name2id(alone, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name

    world = mujoco.MjSpec()
    # The lab's timestep, like `compose()` sets: `attach` keeps the PARENT's
    # options, and a bare MjSpec's 2 ms would warn about overriding the
    # scene's 5 ms — noise that says nothing about the mass.
    world.option.timestep = C.PHYSICS_DT
    frame = world.worldbody.add_frame(pos=[0, 0, 0])
    mars.MARS.attach(world, prefix="m0/", frame=frame)
    attached = world.compile()
    assert attached.nbody == alone.nbody
    assert float(sum(attached.body_mass)) == pytest.approx(
        float(sum(alone.body_mass)), abs=1e-6), (
        "the attach lost mass — see fusestatic in mars.load_robot_spec")


# ---------------------------------------------------- innate's model recipe

@needs_mars
def test_the_planar_base_has_three_dofs_and_no_way_to_fall():
    """(x, y, yaw) on `base_link` — and nothing else.

    A free joint lets the arm's reaction torque tip the 0.89 kg base over
    (Innate's own reason), which would make MARS a body that falls and put
    every walker question back on the table. So the base's DoFs are pinned
    here by TYPE, not just by name: two slides on x and y, one hinge on z.
    """
    m = mars.model()
    want = (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_SLIDE,
            mujoco.mjtJoint.mjJNT_HINGE)
    axes = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    for name, jtype, axis in zip(mars.BASE_JOINTS, want, axes):
        j = m.joint(name)
        assert int(j.type[0]) == int(jtype), name
        assert np.allclose(j.axis, axis), name
    assert m.nq == 11 and m.nv == 11, "3 planar + 6 arm + the mimic + the head"
    # No free joint anywhere: a 7-qpos root is exactly what was excluded.
    assert not any(int(m.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)
                   for i in range(m.njnt))


@needs_mars
def test_the_drive_wheels_are_frictionless_and_win_their_contact_pairs():
    """condim 1 AND priority 1, both of Innate's, both necessary.

    The planar base pins z, so a tangent wheel answers every step with ~50 N
    of spurious normal force whose friction cone glues the base. condim 1
    drops the friction; priority 1 is what stops the floor's condim 3 from
    winning the pair and putting it straight back.
    """
    m = mars.model()
    for name in ("base_wheel_left", "base_wheel_right"):
        g = m.geom(name)
        assert int(g.condim[0]) == 1, name
        assert int(g.priority[0]) == 1, name
    assert int(m.geom(mars.FLOOR_GEOM).condim[0]) == 3, (
        "the floor is condim 3 — which is why the wheels need priority")


@needs_mars
def test_the_finger_blades_carry_the_grasp_contact_model():
    """Innate's grasp tuning, on the geoms it belongs to.

    Every number here was measured against an object slipping out of the
    claw, so a revision that drops one is a gripper that looks right and
    cannot hold anything. Priority 2 is what makes the finger's parameters
    govern every pair it is in — which is also why its condim must be 6.
    """
    m = mars.model()
    blades = [i for i in range(m.ngeom)
              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                    m.geom_bodyid[i]) in mars.FINGER_LINKS
                  and int(m.geom_contype[i]) == 1)]
    assert len(blades) == 26, "13 collision shapes a blade"
    # Against the LITERALS (`INNATE_CONSTANTS`), not against the module's own
    # constants: comparing the model to the value it was built from cannot
    # catch the value changing, and a planted `FINGER_ARMATURE = 0.0` passed
    # exactly that version of this test.
    for i in blades:
        assert int(m.geom_priority[i]) == 2
        assert int(m.geom_condim[i]) == 6
        assert np.allclose(m.geom_friction[i], (2.0, 0.05, 0.02))
        assert np.allclose(m.geom_solref[i], (0.005, 1.0))
        assert np.allclose(m.geom_solimp[i], (0.95, 0.99, 0.001, 0.5, 2))
    for name in (mars.MIMIC_JOINT[0], mars.MIMIC_JOINT[1]):
        dof = int(m.joint(name).dofadr[0])
        assert m.dof_damping[dof] == pytest.approx(1.0)
        assert m.dof_armature[dof] == pytest.approx(1e-4)
    # And the arm's joints are NOT re-tuned: the URDF's damping=5 stands
    # everywhere the fingers are not (the fingers' 1.0 is what sets the
    # ~0.45 s close, and applying it to the arm would triple its speed).
    arm_dof = int(m.joint("joint1").dofadr[0])
    assert m.dof_damping[arm_dof] == pytest.approx(5.0)


@needs_mars
def test_the_two_fingers_are_excluded_from_colliding_with_each_other():
    """Their hub pins overlap by ~1 mm at joint6 = 0.

    Without the exclude the claw fights itself at the closed pose — and
    `arm.srdf` disables the same pair for MoveIt, so this is the robot's own
    statement about its geometry, not a convenience.
    """
    m = mars.model()
    b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mars.FINGER_LINKS[0])
    b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mars.FINGER_LINKS[1])
    pairs = {(int(m.exclude_signature[i]) >> 16,
              int(m.exclude_signature[i]) & 0xFFFF) for i in range(m.nexclude)}
    assert (b1, b2) in pairs or (b2, b1) in pairs, (
        f"link61/link62 not excluded (exclusions: {pairs})")


@needs_mars
def test_the_gripper_range_is_clamped_to_the_real_hard_stop():
    """Closed on air the encoder reads -0.085 rad, past nominal zero.

    The URDF's canonical range starts at 0; unclamped, a -0.6 close target
    scissors the blades through each other. The mimic finger's range is the
    negation, or the geartrain has a reachable pose its partner does not.
    """
    m = mars.model()
    j6 = m.joint(mars.MIMIC_JOINT[1]).range
    j6m = m.joint(mars.MIMIC_JOINT[0]).range
    # PAST nominal zero, asserted as the property and not just against the
    # constant: `j6[0] == mars.GRIPPER_CLOSED_ON_AIR_RAD` alone passes when
    # the constant is edited to 0.0, which is the clamp not existing.
    assert j6[0] < 0.0, "the closed-on-air stop is past zero, or it is not a clamp"
    assert j6[0] == pytest.approx(-0.085)
    assert j6[1] == pytest.approx(0.8727, abs=1e-4)
    assert j6m[1] == pytest.approx(0.085)
    assert j6m[0] == pytest.approx(-j6[1], abs=1e-4)
    # The URDF's own canonical range starts at 0, so the clamp is this
    # module's edit and not something the file already said.
    assert '<limit lower="0" upper="0.8727"' in mars.urdf_path().read_text()


@needs_mars
def test_the_home_keyframe_folds_the_arm_and_mirrors_the_mimic_finger():
    """The keyframe every MARS spawns at, addressed by joint name.

    The mimic finger is the one that cannot be typed in: it has to be
    `-joint6` at spawn or the claw starts with one blade open, which reads
    as a broken gripper the moment anything is put in it.
    """
    m = mars.model()
    key = m.key(mars.HOME_KEY)
    assert key.qpos.shape[0] == m.nq
    for name, want in mars.ARM_HOME.items():
        adr = int(m.joint(name).qposadr[0])
        # 1e-5, not exact: the scene is written as XML text, which keeps ~6
        # significant digits (joint2 lands at -1.38825 for Innate's
        # -1.3882526130365052). 2.6e-6 rad is 0.00015 deg, three orders
        # below the robot's own 2 mm repeatability — but it does mean
        # ARM_HOME is the source of truth and the keyframe is a rounding of
        # it, which is why `arm_servo` reads the dict and not the keyframe.
        assert key.qpos[adr] == pytest.approx(want, abs=1e-5), name
    mimic, source, mult = mars.MIMIC_JOINT
    assert key.qpos[int(m.joint(mimic).qposadr[0])] == pytest.approx(
        mult * mars.ARM_HOME[source], abs=1e-5)
    # The base spawns at the origin, on the floor, with nothing moving.
    for name in mars.BASE_JOINTS:
        assert key.qpos[int(m.joint(name).qposadr[0])] == 0.0
    assert np.all(key.qvel == 0.0)


@needs_mars
def test_two_mars_attach_under_two_prefixes_in_one_model():
    """A roster is N bodies in ONE model, and MARS's names are all bare.

    `tests/test_body_conformance.py` attaches one beside a duck; this is the
    harder case — two of the SAME body, where every mesh, material, joint and
    the `link61/link62` exclude has to survive being named twice.
    """
    world = mujoco.MjSpec()
    world.option.timestep = C.PHYSICS_DT
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                             size=[5, 5, 0.05])
    for i, y in enumerate((1.0, -1.0)):
        frame = world.worldbody.add_frame(pos=[0, y, 0])
        mars.MARS.attach(world, prefix=f"m{i}/", frame=frame)
    m = world.compile()
    assert m.nexclude == 2, "each MARS keeps its own finger exclude"
    for i in range(2):
        for name in mars.DRIVEN_JOINTS + mars.BASE_JOINTS:
            assert mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_JOINT, f"m{i}/{name}") >= 0, name
    data = mujoco.MjData(m)
    mujoco.mj_step(m, data)
    assert np.isfinite(data.qpos).all()


@needs_mars
def test_the_scene_is_rewritten_only_when_its_content_changes():
    """`scene_fn` is called per env build, per lab slot and per test.

    Rewriting the file each time would be a write under a vec-env worker
    that may be importing it (the repo's atomic-write rule), so the content
    is compared first and the write is temp + `os.replace`.
    """
    path = mars.scene_xml()
    assert path.is_file() and path.name == mars.SCENE_NAME
    before = path.stat().st_mtime_ns
    assert mars.scene_xml() == path
    assert path.stat().st_mtime_ns == before, "the scene was rewritten"
    path.write_text(path.read_text() + "<!-- edited -->")
    assert mars.scene_xml() == path
    assert "edited" not in path.read_text(), "a changed scene was not regenerated"
    assert not list(path.parent.glob(f".{path.name}.*.tmp")), "a temp file was left"


# ---------------------------------------------------------------- the servo

@needs_mars
def test_there_are_no_actuators_and_the_servo_drives_qfrc_applied():
    """`nu == 0` is the model being faithful, not broken.

    mars.urdf has no `<actuator>` block because Innate's driver commands
    positions through `qfrc_applied`. Anything in this tree that reaches for
    `data.ctrl` on a MARS is writing into a zero-length array, so the fact
    is pinned where somebody will read it.
    """
    m = mars.model()
    assert m.nu == 0
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, m.key(mars.HOME_KEY).id)
    mujoco.mj_forward(m, data)
    assert np.all(data.qfrc_applied == 0.0)
    mars.arm_servo(m, data, {**mars.ARM_HOME, "joint1": mars.ARM_HOME["joint1"] - 0.5})
    driven = [mars.servo_addresses(m)[n][1] for n in mars.DRIVEN_JOINTS]
    assert np.any(data.qfrc_applied[driven] != 0.0)
    assert np.all(np.abs(data.qfrc_applied) <= mars.EFFORT_LIMIT + 1e-9)


@needs_mars
def test_the_gripper_runs_on_its_own_two_newton_metre_clamp():
    """50 N*m on a 45 mm finger is 1.1 kN of pinch.

    It ejects whatever it grabs and shakes the contact solver, so the two
    finger DoFs are clamped at the real servo's rating while the arm keeps
    the generic 50. A single clamp for the whole body is the mistake this
    asserts against.
    """
    m = mars.model()
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, m.key(mars.HOME_KEY).id)
    mujoco.mj_forward(m, data)
    adr = mars.servo_addresses(m)
    # Ask for the gripper wide open and the shoulder far away: both servos
    # saturate, and the two limits must be different.
    mars.arm_servo(m, data, {"joint6": 0.87, "joint1": -1.5}, adr=adr)
    grip = abs(float(data.qfrc_applied[adr["joint6"][1]]))
    mimic = abs(float(data.qfrc_applied[adr["joint6M"][1]]))
    arm = abs(float(data.qfrc_applied[adr["joint1"][1]]))
    assert grip == pytest.approx(mars.GRIPPER_EFFORT_LIMIT)
    assert mimic == pytest.approx(mars.GRIPPER_EFFORT_LIMIT)
    assert arm == pytest.approx(mars.EFFORT_LIMIT)


@needs_mars
def test_the_servo_refuses_a_joint_the_model_does_not_have():
    """The planted regression for every name in the servo.

    A renamed link in a URDF revision has to fail here. The alternative is
    `mj_name2id` answering -1 and the servo writing into `qfrc_applied[-1]`
    — the last DoF in the model, which on this body is the head.
    """
    m = mars.model()
    with pytest.raises(KeyError):
        mars.servo_addresses(m, prefix="m0/")      # nothing is attached here
    with pytest.raises(KeyError):
        mars.arm_servo(m, mujoco.MjData(m), mars.ARM_HOME, prefix="nope/")


def test_the_joint2_guard_ramps_from_the_head_to_the_full_range():
    """arm_control.cpp's "intelligent joint limits", ported.

    The arm must duck UNDER the head rather than sweep through it when
    joint1 crosses the front arc, so joint2's floor is a function of joint1.
    Pinned as the shape of the ramp — a no-op outside the arc, the guard
    value inside it, monotonic in between — because a sign error here is an
    arm that is free to hit the head exactly where the guard was meant to
    apply. It is a no-op at HOME (joint1 = 1.445), which is why the hold test
    cannot be passing because of it.
    """
    full = -1.5708
    assert mars.joint2_min_target(1.445, full) == full          # HOME
    assert mars.joint2_min_target(-1.4, full) == full
    assert mars.joint2_min_target(1.30, full) == full
    assert mars.joint2_min_target(0.0, full) == mars.JOINT2_GUARD_MIN
    assert mars.joint2_min_target(0.99, full) == mars.JOINT2_GUARD_MIN
    # Inside the arc the guard RESTRICTS: a higher floor than the joint's own.
    assert mars.joint2_min_target(0.0, full) > full
    ramp = [mars.joint2_min_target(x / 100.0, full) for x in range(100, 126)]
    assert ramp == sorted(ramp, reverse=True), "the ramp back to full is not monotonic"
    assert ramp[0] == mars.JOINT2_GUARD_MIN and ramp[-1] == pytest.approx(full)


@needs_mars
def test_the_sag_model_is_off_by_default_and_moves_the_arm_when_on():
    """Innate's structural sag, and why this harness leaves it off.

    On the real robot the sag lives PAST the encoders and `/joint_states`
    reports the encoder side, so `qpos` would disagree with the reported arm
    by up to the backlash. Until the reporting layer exists (Phase 4's obs),
    switching it on would make the observation less honest, not more — so
    the default is off, and turning it on must visibly move the arm or the
    port is dead code.
    """
    m = mars.model()
    errs = {}
    for sag in (False, True):
        data = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, data, m.key(mars.HOME_KEY).id)
        adr = mars.servo_addresses(m)
        for _ in range(400):
            mars.arm_servo(m, data, mars.ARM_HOME, adr=adr, sag=sag)
            mujoco.mj_step(m, data)
        errs[sag] = max(abs(float(data.qpos[adr[n][0]]) - t)
                        for n, t in mars.ARM_HOME.items())
    assert errs[False] < 0.01, f"the default hold drifted {errs[False]:.4f} rad"
    assert errs[True] > errs[False] * 3, (
        f"sag=True changed nothing ({errs}) — the port is dead code")


# --------------------------------------------------------------- the viewer

@needs_mars
def test_the_visual_dump_paints_innates_colours_and_leaves_out_the_rest():
    """`style_robot_geoms`' intent, applied to the dump the browser draws.

    The lab ships a colour per geom, so this is how a MARS arrives orange
    without the viewer knowing anything about MARS. What must NOT be in the
    dump is as important: 46 collision boxes and 3 frame-marker spheres,
    which would draw as a robot inside a pile of blocks.
    """
    scene = mars.MARS.visual_scene()
    by_name = {g["name"]: g for g in scene["geoms"]}
    assert len(scene["geoms"]) == len(scene["meshes"]) == 9
    assert set(by_name) == {"base", "head", "link1", "link2", "link3",
                            "link4", "link5", "link61", "link62"}
    for link in ("link1", "link3", "link5"):
        assert by_name[link]["rgba"] == list(mars.BRIGHT_ORANGE), link
    for link in ("base", "head", "link2", "link4"):
        rgb = by_name[link]["rgba"][:3]
        assert rgb == [mars.CHARCOAL] * 3, f"{link} is {rgb}, not charcoal"
        assert by_name[link]["rgba"][3] == 1.0
    # Every geom names a body in the dump's own list, and the mm-int verts
    # are what the viewer multiplies by vertScale.
    assert scene["vertScale"] == 0.001
    for g in scene["geoms"]:
        assert scene["bodies"][g["body"]] in ("base_link", "head") or \
            scene["bodies"][g["body"]].startswith("link")
    assert all(isinstance(v, int) for v in scene["meshes"][0]["v"][:10])


@needs_mars
def test_the_dump_is_json_serialisable_and_carries_the_whole_robot():
    """The lab serves it over HTTP, so a numpy scalar that survived the dump
    would be a 500 at `GET /scene?robot=mars` and a blank stage — which has
    already cost one debugging session that went looking in the viewer."""
    blob = json.dumps(mars.MARS.visual_scene())
    assert len(blob) > 100_000
    assert json.loads(blob)["bodies"][1] == "base_link"
