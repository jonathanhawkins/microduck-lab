"""`MjcfBody`: a body read out of an MJCF, and the proof that it is generic.

`docs/mars-roadmap.md` §7 says the harness is generic when a robot reaches
level 0 — on the stage, posable, in a room as a body — from nothing but an
MJCF file and an id. The settling number is a Menagerie robot passing the
conformance suite with zero lines of its own in `src/`, and
`tests/test_menagerie.py` plus the suite's own `menagerie:*` parameters are
where that is measured against a real download.

This file measures the claim OFFLINE, against the two bodies this repo already
has, and that is the sharper test of the two: `MjcfBody` is built on the
duck's own `robot_walk.xml` and the G1's `g1.xml` and asked to rediscover what
`contract.py` and `robots/g1.py` spell out by hand. If it reads the duck's 14
joints in the duck's order, the duck's default pose and the duck's 0.65 m
stage pitch off the file alone, then nothing it read was declared — which is
the whole property, and it needs no network to state.

**The generic conformance cases are IMPORTED and called**, not re-written:
`tests/test_body_conformance.py` parameterises over the registry at COLLECTION
time, so a body registered in a fixture is invisible to it. Calling its case
functions directly is what makes "the suite is green for this body" a claim
about the suite rather than a copy of it.

Three MEASURED corrections to the plan are recorded in the cases below:

* the duck's default pose comes back to **4e-5**, not 1e-6. Upstream's STAND
  keyframe in `scene_walk.xml` is `contract.DEFAULT_POSE` rounded to five
  decimals, so 4e-5 is the *file's* precision and a tighter band would be a
  test of nothing.
* the duck's keyframes live in its SCENE, not its robot file
  (`robot_walk.xml` has `nkey == 0`), and its first keyframe is INIT, which is
  0.458 rad from the default pose. Hence `PREFERRED_KEYS` and the
  `pose_source` record.
* `g1.g1_xml()` and `g1.g1_scene_xml()` cannot be paired: the scene freezes
  the 14 finger joints, so nq is 36 against the robot's 50. That refusal is a
  case here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest
import test_body_conformance as CONF

from microduck_local import contract as C
from microduck_local.robots import g1
from microduck_local.robots import registry as R
from microduck_local.robots.body import LAB_PITCH_RATIO, conforms
from microduck_local.robots.mjcf_body import (
    GENERATED_KEY,
    MjcfBody,
    compile_robot,
    load_robot_spec,
    pick_keyframe,
    scalar_joint_names,
    visual_geom_group,
)
from microduck_local.world.compose import ROBOT_XML

DUCK_ID = "mjcf:duck-test"
G1_ID = "mjcf:g1-test"

#: The duck's STAND keyframe is `contract.DEFAULT_POSE` rounded to five
#: decimals in upstream's `scene_walk.xml`. MEASURED: 4.0e-5. The plan asked
#: for 1e-6, which the FILE cannot supply.
POSE_TOL = 5e-5

needs_g1 = pytest.mark.skipif(
    not g1.g1_ready(), reason="G1 assets missing — uv run fetch-robot g1")


# ------------------------------------------------------------- the fixtures

@pytest.fixture(scope="module")
def cache(tmp_path_factory) -> Path:
    """Where a generated scene is written. Module-scoped: building a body
    compiles its model, and the G1's costs ~0.7 s."""
    return tmp_path_factory.mktemp("mjcf-cache")


@pytest.fixture(scope="module")
def duck_body(cache) -> MjcfBody:
    """The DUCK, read as a stranger's MJCF.

    Robot file + scene file, both the upstream ones, and nothing else told to
    it: not the joint names, not the pose, not the pitch, not which group the
    meshes are in.
    """
    return MjcfBody.from_mjcf(ROBOT_XML["walk"], id=DUCK_ID,
                              scene_xml=C.MICRODUCK.scene_fn(), cache_dir=cache)


@pytest.fixture(scope="module")
def g1_body(cache) -> MjcfBody:
    """The G1's bare MJCF with NO scene, so one is generated.

    The fingers are not frozen here (that is `g1_spec`'s doing, not the
    file's), so this body has 43 joints where `robots/g1.py` declares 29 —
    which is the honest answer for the file it was handed and is asserted as
    such below.
    """
    return MjcfBody.from_mjcf(g1.g1_xml(), id=G1_ID, cache_dir=cache)


@pytest.fixture
def registered(duck_body):
    """`mjcf:duck-test` in the process registry, and out again.

    `register()` is module state (`robots/registry.py`), so this restores it
    the way `test_registry.clean_registry` does — a fake body left behind
    would join every later test's roster.
    """
    before = dict(R._EXTRA)
    R.register(duck_body)
    try:
        yield duck_body
    finally:
        R._EXTRA.clear()
        R._EXTRA.update(before)
        CONF._model.cache_clear()


# --------------------------------------------- 1. the duck, rediscovered

def test_the_ducks_own_mjcf_yields_the_ducks_own_joint_table(duck_body):
    """The joints, in the ORDER the deployment contract lists them.

    Order is the load-bearing half: `contract.JOINT_NAMES` is the order the
    robot's 14 servos appear in an action vector, and a reader that returned
    the right names in the wrong order would produce a body whose every
    action lands on the wrong joint. MuJoCo numbers joints in tree order, so
    this is the file's own order — and the free root is out, exactly as
    `JOINT_NAMES` has it.
    """
    assert duck_body.joint_names == C.JOINT_NAMES
    assert duck_body.num_joints == C.NUM_JOINTS == 14
    assert duck_body.num_actions == 14
    assert duck_body.extra["skipped_joints"] == ("trunk_base_freejoint",)


def test_the_ducks_own_mjcf_yields_the_ducks_own_default_pose(duck_body):
    """The pose, to the precision the FILE has.

    MEASURED: 4.0e-5, because upstream's STAND keyframe is `DEFAULT_POSE`
    written out to five decimals. That is also why the plan's 1e-6 is not the
    band — a tighter one would fail on a file that is right.
    """
    assert duck_body.default_pose.shape == C.DEFAULT_POSE.shape
    err = float(np.abs(duck_body.default_pose - C.DEFAULT_POSE).max())
    assert err < POSE_TOL, f"the read pose is {err:.2g} off DEFAULT_POSE"
    assert err > 0, ("it matched to the bit — upstream rounded its keyframe "
                     "no longer, so tighten POSE_TOL and say so")


def test_the_ducks_own_mjcf_yields_the_ducks_own_stage_pitch(duck_body):
    """0.65 m of stage, MEASURED rather than inherited.

    The one number the viewer has always drawn, recovered from the model by
    `body.measure_lab_spacing_m` with nothing but the ratio. MEASURED:
    0.64956 m, 0.07 % off the declared 0.65 — inside the 1 % the plan asked
    for and inside the 5 % the conformance suite allows.
    """
    assert duck_body.lab_spacing_m == pytest.approx(0.65, rel=0.01)
    assert duck_body.extra["measured_width_m"] == pytest.approx(0.1845, abs=0.002)
    assert (duck_body.lab_spacing_m
            == pytest.approx(LAB_PITCH_RATIO * duck_body.extra["measured_width_m"]))


def test_the_pose_it_read_is_recorded_and_not_implied(duck_body):
    """Which keyframe fed the pose is DATA, because the answer is surprising.

    The duck's robot file carries no keyframe at all and its scene carries
    four, of which the FIRST (INIT) is 0.458 rad from the default pose. A
    reader that silently took "the first one" would hand back a duck folded
    differently and nothing would say so, which is why `PREFERRED_KEYS` names
    STAND and why the choice is written into `extra`.
    """
    assert duck_body.stand_keyframe == "STAND"
    assert duck_body.extra["pose_source"] == "keyframe 'STAND'"
    assert duck_body.extra["scene_source"] == "given"
    robot = compile_robot(ROBOT_XML["walk"])
    assert int(robot.nkey) == 0, "the duck's robot file grew keyframes"
    scene = mujoco.MjModel.from_xml_path(str(C.MICRODUCK.scene_fn()))
    assert [scene.key(i).name for i in range(scene.nkey)][0] == "INIT"
    init = CONF.np.asarray(scene.key("INIT").qpos)
    adr = [scene.joint(n).qposadr[0] for n in C.JOINT_NAMES]
    assert float(np.abs(init[adr] - C.DEFAULT_POSE).max()) > 0.4, (
        "INIT is no longer far from the default pose — the keyframe ORDER "
        "would then be a safe rule and PREFERRED_KEYS could go")


# ------------------------------------------- 2. the suite's generic cases

GENERIC_CASES = (
    CONF.test_the_bodys_scene_compiles_and_every_declared_name_resolves,
    CONF.test_the_name_scan_catches_a_wrong_name_of_every_kind,
    CONF.test_the_bodys_own_numbers_agree_with_each_other,
    CONF.test_a_planted_dimension_mismatch_is_caught,
    CONF.test_the_body_attaches_under_a_prefix_beside_a_duck_in_one_model,
    CONF.test_two_bodies_in_one_model_need_the_prefix,
    CONF.test_the_lab_pitch_is_the_ducks_ratio_on_this_bodys_measured_width,
    CONF.test_a_pitch_that_drifted_from_the_measured_width_is_caught,
    CONF.test_the_visual_scene_serves_every_mesh_the_model_has,
    CONF.test_a_doctored_visual_scene_is_caught,
)


@pytest.mark.parametrize("case", GENERIC_CASES,
                         ids=lambda c: c.__name__[len("test_"):])
def test_the_suites_generic_cases_pass_for_a_body_read_from_an_mjcf(
        case, registered):
    """Every ROBOTS case in the conformance suite, on a body nobody declared.

    This is the claim of the whole phase, and it is stated by CALLING the
    suite rather than by copying it: the suite parameterises over the registry
    at collection time, so a fixture-registered body cannot appear in its own
    parameter list, and a re-written copy of ten cases would prove only that
    two files agree.

    What the suite needed to accept a fourth kind is two things and they are
    both listed in its docstring: a `GENERIC` roster (so the kind-roster case
    does not report an unclassified body) and one branch in `_visual_scene`
    for where a generic body's mesh count comes from. No case body changed.
    """
    case(registered.id)


def test_the_generic_roster_is_not_empty_when_a_generic_body_is_registered(
        registered):
    """The guard on the case above: it must be running SOMETHING.

    A `kind` the roster filter did not recognise would leave `GENERIC` empty
    and the kind-roster case would report the body as unclassified. Rebuilding
    the roster with the body registered is how that is checked without
    re-running collection.
    """
    generic = CONF._params(lambda b: b.kind == "generic")
    assert registered.id in {p.values[0] for p in generic}
    assert conforms(registered) == ()
    CONF.test_every_body_is_on_exactly_one_of_the_kind_rosters()


def test_the_suites_generic_cases_fail_on_a_body_that_is_broken(registered):
    """The planted negative for the import above: the cases must BITE here.

    A suite function called with a body it cannot see would pass vacuously —
    `pytest.skip` on missing assets, an empty parameter list, a helper that
    silently returned. So each of three plants must make its own case fail:
    a wrong joint name, a pitch 1.5x the measured width, and an obs width of
    zero.
    """
    body = registered
    plants = (
        ("joint name", dict(joint_names=("no_such_joint",) + body.joint_names[1:]),
         CONF.test_the_bodys_scene_compiles_and_every_declared_name_resolves),
        ("stage pitch", dict(lab_spacing_m=body.lab_spacing_m * 1.5),
         CONF.test_the_lab_pitch_is_the_ducks_ratio_on_this_bodys_measured_width),
        ("obs width", dict(obs_dim=0),
         CONF.test_the_bodys_own_numbers_agree_with_each_other),
    )
    for label, override, case in plants:
        broken = dataclasses.replace(body, **override)
        R._EXTRA[body.id] = broken
        CONF._model.cache_clear()
        try:
            with pytest.raises(Exception):
                case(body.id)
        finally:
            R._EXTRA[body.id] = body
            CONF._model.cache_clear()
        assert True, label


# --------------------------------------------------- 3. keyframe selection

def _two_joint_xml(keys: str = "", extra_body: str = "") -> str:
    """A minimal robot: one link, two hinges. The substrate for the plants.

    Hand-written rather than derived from a real robot because these cases
    are about the READER's rules — which keyframe it picks, what it refuses —
    and a 38-mesh duck would make each one a 0.3 s compile for no extra
    coverage.
    """
    return f"""<mujoco model="two">
      <worldbody>
        <body name="link0" pos="0 0 0.5">
          <geom name="g0" type="box" size="0.1 0.1 0.1"/>
          <joint name="ja" type="hinge" axis="0 1 0" range="-1 1"/>
          <body name="link1" pos="0 0 0.2">
            <geom name="g1" type="box" size="0.05 0.05 0.05"/>
            <joint name="jb" type="slide" axis="0 0 1" range="-0.1 0.1"/>
          </body>
        </body>
        {extra_body}
      </worldbody>
      {keys}
    </mujoco>"""


def _write(tmp_path: Path, name: str, xml: str) -> Path:
    p = tmp_path / name
    p.write_text(xml)
    return p


@pytest.mark.parametrize("keys,want,pose", [
    ("""<keyframe><key name="first" qpos="0.1 0.01"/>
         <key name="home" qpos="0.2 0.02"/>
         <key name="STAND" qpos="0.3 0.03"/></keyframe>""", "home", [0.2, 0.02]),
    ("""<keyframe><key name="first" qpos="0.1 0.01"/>
         <key name="STAND" qpos="0.3 0.03"/></keyframe>""", "STAND", [0.3, 0.03]),
    ("""<keyframe><key name="first" qpos="0.1 0.01"/>
         <key name="second" qpos="0.4 0.04"/></keyframe>""", "first", [0.1, 0.01]),
    ("", GENERATED_KEY, [0.0, 0.0]),
])
def test_the_keyframe_it_spawns_from_follows_a_stated_order(
        tmp_path, keys, want, pose):
    """`home` beats this repo's names, which beat "the first one", which
    beats `qpos0`.

    The order is the whole of what a generic reader can know: `home` is
    Menagerie's convention across all 71 models, `HOME`/`STAND` are this
    repo's, and after that the file's own first keyframe is the best guess
    available. A body with none gets a written `HOME` at `qpos0` rather than
    no spawn pose, because the conformance suite looks the keyframe up by
    name and a body with no keyframe cannot be posed at all.
    """
    robot = _write(tmp_path, "two.xml", _two_joint_xml(keys))
    body = MjcfBody.from_mjcf(robot, id="mjcf:two", cache_dir=tmp_path)
    assert body.stand_keyframe == want
    np.testing.assert_allclose(body.default_pose, pose, atol=1e-9)
    assert body.joint_names == ("ja", "jb")
    # The generated scene must actually CARRY it, or the body cannot spawn.
    scene = mujoco.MjModel.from_xml_path(str(body.scene_fn()))
    assert mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY, want) >= 0


def test_a_keyframe_named_in_the_manifest_overrides_the_order(duck_body, cache):
    """A catalogue that KNOWS better may say so, and only then.

    The duck is the case: its own `default_pose` is STAND, and the generic
    rule happens to agree only because `PREFERRED_KEYS` lists STAND. A body
    whose right pose is called something else needs a way to say it that is
    not a patch to this reader.
    """
    body = MjcfBody.from_mjcf(ROBOT_XML["walk"], id="mjcf:duck-init",
                              scene_xml=C.MICRODUCK.scene_fn(), cache_dir=cache,
                              manifest={"keyframe": "INIT"})
    assert body.stand_keyframe == "INIT"
    assert float(np.abs(body.default_pose - duck_body.default_pose).max()) > 0.4


def test_an_unknown_keyframe_name_is_refused_rather_than_ignored(cache):
    """The planted negative for the override: a typo must not fall back.

    Falling back to `pick_keyframe` would be the silent version — the body
    would spawn in a pose the catalogue did not ask for and every number read
    at that pose (the pitch, the default) would be someone else's.
    """
    with pytest.raises(ValueError, match="keyframe called 'NOPE'"):
        MjcfBody.from_mjcf(ROBOT_XML["walk"], id="mjcf:x",
                           scene_xml=C.MICRODUCK.scene_fn(), cache_dir=cache,
                           manifest={"keyframe": "NOPE"})


def test_a_given_scene_with_no_keyframe_anywhere_gets_a_generated_one(tmp_path):
    """Branch 4 of `_resolve_scene`, and the one shape that could have LIED.

    A given scene is somebody else's file and is never written to, while every
    reader in this repo looks the spawn pose up by NAME on `scene_fn()` — so
    keeping the given scene when nothing carries a keyframe would leave
    `stand_keyframe` naming `HOME` in a file that has no `HOME`. The
    conformance suite's very first case (`model.key(body.stand_keyframe)`)
    would then be the thing that discovered it, three layers from the cause.
    """
    robot = _write(tmp_path, "nokey.xml", _two_joint_xml())
    scene = _write(tmp_path, "nokey_scene.xml", """<mujoco model="s">
      <include file="nokey.xml"/>
      <worldbody><geom name="floor" size="0 0 0.05" type="plane"/></worldbody>
    </mujoco>""")
    body = MjcfBody.from_mjcf(robot, id="mjcf:nokey", scene_xml=scene,
                              cache_dir=tmp_path)
    assert body.extra["scene_source"] == "generated"
    assert body.stand_keyframe == GENERATED_KEY
    assert Path(body.scene_fn()) != scene
    compiled = mujoco.MjModel.from_xml_path(str(body.scene_fn()))
    assert mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_KEY,
                             GENERATED_KEY) >= 0
    # The given scene is untouched, and still has nothing to spawn from.
    assert int(mujoco.MjModel.from_xml_path(str(scene)).nkey) == 0
    CONF._model.cache_clear()


def test_pick_keyframe_reads_the_model_and_not_a_name_it_hoped_for(tmp_path):
    """The helper on its own, so the order above is not only end-to-end."""
    with_home = compile_robot(_write(tmp_path, "h.xml", _two_joint_xml(
        """<keyframe><key name="a" qpos="0 0"/>
            <key name="home" qpos="0 0"/></keyframe>""")))
    assert pick_keyframe(with_home) == "home"
    bare = compile_robot(_write(tmp_path, "b.xml", _two_joint_xml()))
    assert pick_keyframe(bare) is None


# ------------------------------------------------ 4. it fails loudly

def test_a_model_with_no_scalar_joints_is_refused(tmp_path):
    """A body with nothing to pose or act on is not a body.

    The quiet version is worse than it sounds: `joint_names == ()` makes
    `obs_dim` 0, which `PolicyContract` then refuses with a message about
    widths, three layers away from the file that is actually wrong.
    """
    xml = """<mujoco model="static"><worldbody>
      <body name="b"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
      </worldbody></mujoco>"""
    with pytest.raises(ValueError, match="no hinge or slide joints"):
        MjcfBody.from_mjcf(_write(tmp_path, "static.xml", xml), id="mjcf:static",
                           cache_dir=tmp_path)


@needs_g1
def test_a_scene_from_a_different_variant_of_the_robot_is_refused(cache):
    """MEASURED, on a pair this repo already ships.

    `g1.g1_xml()` has 43 joints; `g1.g1_scene_xml()` freezes the 14 finger
    DoFs, so its nq is 36 against the robot's 50. Reading a keyframe off one
    and applying it to the other puts every joint target on the wrong joint —
    a robot spawned folded, and a whole session spent blaming the policy.
    """
    with pytest.raises(ValueError, match="not the same robot"):
        MjcfBody.from_mjcf(g1.g1_xml(), id="mjcf:g1-mixed",
                           scene_xml=g1.g1_scene_xml(), cache_dir=cache)


def test_a_missing_mesh_names_the_model_and_not_only_the_file(tmp_path):
    """The commonest way somebody else's MJCF arrives broken.

    A partial download, a case-sensitive filesystem, an `assets/` directory
    that was not copied. MuJoCo says which FILE; at the point a registry is
    building four bodies, the model is the part that identifies the robot.
    """
    xml = """<mujoco model="m">
      <asset><mesh name="ghost" file="nowhere.stl"/></asset>
      <worldbody><body name="b">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="mesh" mesh="ghost"/>
      </body></worldbody></mujoco>"""
    p = _write(tmp_path, "ghost.xml", xml)
    with pytest.raises(ValueError, match=r"ghost\.xml does not compile"):
        MjcfBody.from_mjcf(p, id="mjcf:ghost", cache_dir=tmp_path)


def test_a_file_that_is_not_there_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="no MJCF at"):
        MjcfBody.from_mjcf(tmp_path / "absent.xml", id="mjcf:absent",
                           cache_dir=tmp_path)


def test_a_model_with_no_geoms_cannot_be_measured_and_says_so(tmp_path):
    """The planted negative for the stage-pitch measurement.

    A body of zero width would take `lab_spacing_m` to 0, which every stage
    layout divides by. The message names what is missing rather than the
    arithmetic that failed.
    """
    xml = """<mujoco model="bare"><worldbody>
      <body name="b" pos="0 0 1"><inertial pos="0 0 0" mass="1"
        diaginertia="0.01 0.01 0.01"/>
        <joint name="j" type="hinge" axis="0 1 0"/>
      </body></worldbody></mujoco>"""
    with pytest.raises(ValueError, match="measures 0 m across"):
        MjcfBody.from_mjcf(_write(tmp_path, "bare.xml", xml), id="mjcf:bare",
                           cache_dir=tmp_path)


# ------------------------------------------- 5. the visual group, detected

@needs_g1
def test_the_visual_group_is_read_off_the_model_for_both_real_bodies(
        duck_body, g1_body):
    """MEASURED: the duck's visuals are group 2 and so are the G1's.

    Both agree with the MJCF convention, which is exactly why this has to be
    READ: a constant 2 would pass here and then produce an empty mesh dump on
    MARS, whose visuals the URDF importer puts in group 1
    (`robots/mars.VISUAL_GROUP`). An empty dump is a body nobody can see, and
    "blank stage" has already cost a debugging session in the viewer.
    """
    assert duck_body.visual_group == 2
    assert g1_body.visual_group == 2
    from microduck_local.robots import mars
    if mars.mars_ready():
        assert visual_geom_group(mars.robot_spec().compile()) == mars.VISUAL_GROUP


def test_a_model_whose_visuals_are_group_one_is_detected_as_group_one(tmp_path):
    """The planted regression for the detection: MARS's shape, minimal.

    Mesh geoms with `contype="0"` in group 1 and colliding boxes in group 3 —
    the pattern MuJoCo's URDF importer emits. A reader that returned the
    fallback would dump nothing for this model.
    """
    stl = tmp_path / "tri.stl"
    _write_tri_stl(stl)
    xml = f"""<mujoco model="urdfish">
      <asset><mesh name="tri" file="{stl.name}"/></asset>
      <worldbody><body name="b" pos="0 0 0.5">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="mesh" mesh="tri" group="1" contype="0" conaffinity="0"/>
        <geom type="box" size="0.1 0.1 0.1" group="3"/>
      </body></worldbody></mujoco>"""
    body = MjcfBody.from_mjcf(_write(tmp_path, "urdfish.xml", xml),
                              id="mjcf:urdfish", cache_dir=tmp_path)
    assert body.visual_group == 1
    assert len(body.visual_scene()["meshes"]) == 1
    # And the fallback is only for a model with nothing to go on.
    boxes = """<mujoco model="boxes"><worldbody><body name="b" pos="0 0 .5">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>"""
    assert visual_geom_group(compile_robot(_write(tmp_path, "boxes.xml", boxes))) == 2


def _write_tri_stl(path: Path) -> None:
    """A tetrahedron as BINARY STL — the smallest mesh MuJoCo will take.

    Two MEASURED constraints, both found the hard way: the decoder reads the
    four bytes after an 80-byte header as a face count and refuses an ASCII
    file ("perhaps this is an ASCII file?"), and a single triangle is refused
    with "at least 4 vertices required" because a mesh has to be a solid it
    can hull.
    """
    import struct

    v = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    body = b"".join(
        struct.pack("<12fH", 0.0, 0.0, 0.0, *v[a], *v[b], *v[c], 0)
        for a, b, c in faces)
    path.write_bytes(b"microduck test stl".ljust(80, b"\0")
                     + struct.pack("<I", len(faces)) + body)


# ------------------------------------------------------- 6. two in a model

@needs_g1
def test_two_of_these_attach_under_two_prefixes_beside_a_duck(duck_body,
                                                              g1_body):
    """The lab and `/sim` put N robots in ONE compiled model.

    The conformance case does this for one generic body and a duck; what this
    adds is TWO generic bodies at once, which is where a shared asset name
    would collide — both of these carry a mesh called `left_hip_yaw`-ish
    geometry from the same upstream tree, and the duck body below is the
    third. If a prefix did not isolate them the compile fails on a repeated
    name.
    """
    world = mujoco.MjSpec()
    world.modelname = "two-generics"
    world.option.timestep = C.PHYSICS_DT
    w = world.worldbody
    w.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
               size=[10, 10, 0.05], group=0)
    for i, body in enumerate((duck_body, duck_body, g1_body)):
        body.attach(world, prefix=f"r{i}/",
                    frame=w.add_frame(pos=[0, 2.0 * i, 0.0]))
    model = world.compile()
    for i, body in enumerate((duck_body, duck_body, g1_body)):
        for j in body.joint_names:
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"r{i}/{j}") >= 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_the_attach_carries_absolute_asset_paths(duck_body, tmp_path):
    """Why `load_robot_spec` rewrites `meshdir`, as a test.

    MEASURED: `MjSpec.from_file` keeps `meshdir` exactly as the file spells it
    ("assets"), `to_xml()` writes that back out, and a scene saved anywhere
    else then dies with `Error opening file 'assets/...'`. So the rewrite is
    load-bearing for both halves of this class — the generated scene and the
    attach into a world compiled from another directory.
    """
    spec = load_robot_spec(duck_body.robot_xml)
    assert Path(spec.meshdir).is_absolute()
    assert Path(spec.meshdir).is_dir()
    # A scene written into an unrelated directory still compiles.
    generated = MjcfBody.from_mjcf(duck_body.robot_xml, id="mjcf:elsewhere",
                                   cache_dir=tmp_path)
    assert Path(generated.scene_fn()).parent == tmp_path
    model = mujoco.MjModel.from_xml_path(str(generated.scene_fn()))
    assert int(model.nmesh) == 38


def test_a_relative_meshdir_would_not_survive_the_move(duck_body, tmp_path):
    """The planted negative for the rewrite above.

    The same spec with `meshdir` put back to its relative spelling, written
    to the same place, must FAIL to compile — otherwise the rewrite is
    decoration and the test above passes for the wrong reason.
    """
    spec = mujoco.MjSpec.from_file(str(duck_body.robot_xml))
    assert not Path(spec.meshdir).is_absolute(), "the file changed"
    out = tmp_path / "relative.xml"
    out.write_text(spec.to_xml())
    with pytest.raises(ValueError, match="Error opening file"):
        mujoco.MjModel.from_xml_path(str(out))


# ------------------------------------------------------- 7. what it says

@needs_g1
def test_the_contract_tiles_and_names_the_level(duck_body, g1_body):
    """A level-0 body speaks ONE contract from the day it is listed.

    Three terms, no gaps, no overlaps — the smallest honest layout for a body
    with no declared sensor, no command channel and no task. `v0` is the level
    and the `deploy` sentence says so out loud, because the caveat is what
    gets lost when a `.onnx` is passed along.
    """
    for body, n in ((duck_body, 14), (g1_body, 43)):
        c = body.contract()
        assert c.tiles() == ()
        assert c.obs_dim == 3 * n == body.obs_dim
        assert c.act_dim == n == body.num_actions
        assert [s.name for s in c.slots] == ["joint_pos", "joint_vel",
                                             "last_action"]
        assert c.rate_hz == 50.0
        assert str(body.obs_dim) in c.id, c.id
        assert c.id.startswith("mjcf-") and c.id.endswith("-v0")
        assert "level 0" in c.deploy and "no hardware claim" in c.deploy


def test_a_contract_whose_slots_do_not_tile_is_refused(duck_body):
    """The planted negative: `declare()` validates against the BODY's width.

    A slot table that does not add up to `obs_dim` is a construction error the
    first time anything asks the body for its contract — not a mislabelled
    file discovered later. This plants a body whose `obs_dim` no longer
    matches its joint count and checks that asking for the contract raises.
    """
    bad = dataclasses.replace(duck_body, obs_dim=duck_body.obs_dim + 1)
    with pytest.raises(ValueError, match="does not describe"):
        bad.contract()


def test_the_answers_that_are_refusals_name_the_level(duck_body):
    """`env_class`, `driver` and `frames` raise, and each says which level.

    Not a shrug: a body that quietly returned the duck's env would train a
    quadruped against foot-contact rewards on the duck's geometry, and a
    `frames()` that guessed a base link would slice the wrong subtree out of a
    composed model and stream a robot drawn with its parts on another robot's
    joints (`robots/body.RobotFrames`).
    """
    for call, needle in (
            (lambda: duck_body.env_class("walk"), "level-0"),
            (lambda: duck_body.driver(None, "a/"), "level 2"),
            (duck_body.frames, "Level 2"),
    ):
        with pytest.raises(NotImplementedError, match=needle):
            call()


def test_no_sensors_is_an_answer_and_not_a_refusal(duck_body):
    """MEASURED against `robots/body.py`'s own rule, which this obeys.

    `Body.make_sensors`' docstring says the empty dict is the TRUE answer for
    "a body at level 0 of §7.1's ladder has declared no apertures" — so this
    is the one contract member `MjcfBody` inherits rather than overriding.
    Raising instead would take a kinematic body out of the room that level 0
    promises it, and the arena would stop composing rooms that contain one.
    """
    assert duck_body.make_sensors(None, "a/", presets={"tof": "datasheet"},
                                  targets=(), seed=lambda: 0) == {}
    assert duck_body.tasks() == ()
    assert duck_body.shipped_policies() == ()
    assert duck_body.look() == "generic"
    assert duck_body.kind == "generic"


def test_ready_and_fetch_describe_files_that_are_already_here(duck_body,
                                                              tmp_path):
    """A level-0 body has nothing to download: it IS a file somebody has.

    `ready()` is what the palette shows "not set up yet — ⤓" by, so it has to
    answer for a GENERATED scene too — `RobotSpec.ready()`'s "is the scene
    file there" would say False until something asked for it and True
    thereafter.
    """
    assert duck_body.ready() is True
    assert duck_body.fetch() == Path(duck_body.robot_xml).parent
    gone = dataclasses.replace(duck_body, robot_xml=tmp_path / "absent.xml")
    assert gone.ready() is False


def test_the_title_is_a_label_and_the_id_keeps_its_namespace(duck_body):
    """The id is the wire name and never changes; the title is what a chip
    shows (`AGENTS.md`, "Name and describe every run")."""
    assert duck_body.id == DUCK_ID
    assert duck_body.title == duck_body.noun == "Duck Test"
    named = dataclasses.replace(duck_body, title="Microduck (read from MJCF)")
    assert named.title == "Microduck (read from MJCF)"


# ------------------------------------------------- 8. the G1, rediscovered

@needs_g1
def test_the_g1s_bare_mjcf_reads_its_unfrozen_joint_set(g1_body):
    """43 joints, not 29 — the honest answer for the file it was handed.

    `robots/g1.py` declares 29 because `g1_spec()` DELETES the 14 finger
    joints before the walker ever sees them; the file itself has all 43. A
    reader that reported 29 would be reading `robots/g1.py`, not the MJCF,
    which is the one thing this class must not do.
    """
    declared = set(g1.joint_names())
    read = set(g1_body.joint_names)
    assert declared <= read
    assert len(read) == len(declared) + g1.NUM_HAND_JOINTS == 43
    assert all("_hand_" in n for n in read - declared)
    assert g1_body.extra["skipped_joints"] == ("floating_base_joint",)


@needs_g1
def test_the_g1s_generated_scene_carries_a_written_home_keyframe(g1_body,
                                                                 duck_body):
    """`g1.xml` has no keyframe of its own, so one is written from `qpos0`.

    `robots/g1.py` MEASURES its STAND height by dropping the default pose
    until the lowest foot capsule kisses the floor — a body-specific
    measurement a generic reader cannot make. `qpos0` is what the file says
    instead, and the record says that is where the pose came from.
    """
    assert g1_body.extra["scene_source"] == "generated"
    assert g1_body.stand_keyframe == GENERATED_KEY
    # The record distinguishes a keyframe somebody AUTHORED from a `HOME`
    # written out of `qpos0` — the name alone cannot, and a stage pitch
    # measured at one is not the same statement as one measured at the other.
    assert g1_body.extra["pose_source"] == (
        f"qpos0, written as keyframe '{GENERATED_KEY}'")
    assert duck_body.extra["pose_source"] == "keyframe 'STAND'"
    robot = compile_robot(g1.g1_xml())
    assert int(robot.nkey) == 0
    scene = mujoco.MjModel.from_xml_path(str(g1_body.scene_fn()))
    assert mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY,
                             GENERATED_KEY) >= 0
    np.testing.assert_allclose(
        g1_body.default_pose,
        [robot.qpos0[robot.joint(n).qposadr[0]] for n in g1_body.joint_names],
        atol=1e-9)


@needs_g1
def test_the_generated_scene_is_written_atomically_and_only_when_it_changes(
        g1_body):
    """`AGENTS.md`, "Atomic writes and live imports": there is no safe window.

    A parallel worker can read a half-written file, so the scene lands on a
    temp name and is `os.replace`d. And an unchanged rewrite must not touch
    the file at all, or every `scene_fn()` call invalidates somebody's open
    handle.
    """
    path = Path(g1_body.scene_fn())
    before = path.stat().st_mtime_ns
    assert Path(g1_body.scene_fn()) == path
    assert path.stat().st_mtime_ns == before, "an unchanged scene was rewritten"
    path.unlink()
    assert Path(g1_body.scene_fn()).is_file(), "a deleted scene was not rebuilt"
    assert not list(path.parent.glob(f".{path.name}.*")), "a temp file was left"


# --------------------------------------------------- 9. the helpers alone

def test_scalar_joint_names_keeps_tree_order_and_drops_the_free_root(tmp_path):
    """The order is the contract; a set would lose it.

    Asserted against a model whose joints are deliberately NOT in alphabetical
    order, because a reader that sorted them would pass a name-set check and
    still put every action on the wrong joint.
    """
    xml = """<mujoco model="order"><worldbody>
      <body name="b0" pos="0 0 1"><freejoint name="root"/>
        <geom type="box" size="0.1 0.1 0.1"/>
        <body name="b1" pos="0 0 0.3">
          <geom type="box" size="0.05 0.05 0.05"/>
          <joint name="zulu" type="hinge" axis="0 1 0"/>
          <body name="b2" pos="0 0 0.2">
            <geom type="box" size="0.03 0.03 0.03"/>
            <joint name="alpha" type="slide" axis="0 0 1"/>
          </body>
          <body name="b3" pos="0.2 0 0">
            <geom type="box" size="0.03 0.03 0.03"/>
            <joint name="ball" type="ball"/>
          </body>
        </body>
      </body></worldbody></mujoco>"""
    model = compile_robot(_write(tmp_path, "order.xml", xml))
    assert scalar_joint_names(model) == ("zulu", "alpha")
    # ...and a body built on it reports what it left out, by name.
    body = MjcfBody.from_mjcf(tmp_path / "order.xml", id="mjcf:order",
                              cache_dir=tmp_path)
    assert body.joint_names == ("zulu", "alpha")
    assert set(body.extra["skipped_joints"]) == {"root", "ball"}
