"""A body from nothing but an MJCF file and an id — level 0 of the ladder.

`docs/mars-roadmap.md` §7.1 grades a body by what it declares. The duck is
level 3 (servo models, measured sensors, a deployment contract), the G1 and
MARS are level 2 (a driver and sense channels). **Level 0 is "on the stage,
posable, in a room as a body", and what it costs is an MJCF and an id.** This
file is that level, as one class, and §7's settling number is that a MuJoCo
Menagerie robot reaches it with *zero* lines of its own in `src/`.

Everything here is READ off the compiled model rather than declared, because a
declaration is the thing a stranger's robot cannot supply:

    joint_names     every hinge/slide joint in tree order, free/ball roots out
    default_pose    the `home` keyframe's joint subset (else the first
                    keyframe, else qpos0) — `extra["pose_source"]` says which
    stand_keyframe  that keyframe's name, or a `HOME` written from qpos0
    lab_spacing_m   MEASURED: 3.52x the widest horizontal extent at that
                    keyframe, the same arithmetic the conformance suite
                    re-derives (`body.measure_lab_spacing_m`)
    visual_group    the geom group holding this model's VISUAL meshes, found
                    by `contype == 0` rather than assumed to be 2

`kind = "generic"` — a fourth kind beside `legged`, `wheeled` and `arm`. It is
not a shrug: it is the statement that nothing about this body's PHYSICS has
been measured here, so the conformance suite must run its generic cases and
skip every walker and wheeled one. A Menagerie quadruped is mechanically a
walker and would fail the walker's cases (no declared feet, no gyro name, no
fall height) — claiming `legged` for it would be the architecture-scale
version of the mistake `AGENTS.md` warns about most, keeping an inherited term
whose target is wrong for the new body. It climbs to level 1 when someone
declares those names, not when a string is edited.

**What this class deliberately does NOT answer.** `env_class`, `driver` and
`frames` raise, each naming the level that would land them: there is no task
at level 0, no controller, and no `/sim` root link to drive. `make_sensors` is
the exception and is INHERITED — `robots/body.py`'s own docstring says the
empty dict is the true answer for "a body at level 0 of §7.1's ladder (an MJCF
and an id) has declared no apertures", and raising there would take a
kinematic body out of the room that level 0 promises it.

Construction fails LOUDLY, never quietly: a file with no joints, a keyframe of
the wrong nq, a scene whose nq disagrees with the robot's, or a mesh that does
not resolve are all errors at construction. A body that half-loaded would
arrive as a wrong picture and a wrong pose rather than as an error, which is
the failure mode this whole registry exists to avoid.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple

import mujoco
import numpy as np

from .body import BodyBase, measure_lab_spacing_m
from .policy_contract import PolicyContract, Slot, declare

#: The keyframe name a body with none is given, written from `qpos0`. Upper
#: case because `BodyBase.stand_keyframe` defaults to "STAND" and MARS's is
#: "HOME": the generated one joins that family rather than inventing a third
#: spelling. Menagerie's own is lower-case `home` and is used AS IT IS — the
#: name travels from the file, because the conformance suite looks it up by
#: name and a "helpful" rename would be a keyframe nobody can find.
GENERATED_KEY = "HOME"

#: Keyframe names tried in order before falling back to `qpos0`. `home` is
#: Menagerie's convention (all 71 models carry one); `HOME` and `STAND` are
#: this repo's, so a body generated here round-trips through its own reader.
PREFERRED_KEYS: tuple[str, ...] = ("home", "HOME", "STAND")

#: The joint types a policy can hold a target for. A free or ball root is a
#: pose, not a scalar servo, and every body in this repo excludes its own the
#: same way (`contract.JOINT_NAMES` has no `trunk_base_freejoint`).
_SCALAR_JOINTS = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)

#: Fallback visual geom group when `contype == 0` finds nothing to go on. 2 is
#: the MJCF convention (Menagerie's `class="visual"`, the duck's, the G1's);
#: MuJoCo's URDF importer uses 1, which is why this is DETECTED and not fixed.
DEFAULT_VISUAL_GROUP = 2

#: The level-0 contract's control rate. Not measured — declared, and the
#: `deploy` sentence says as much: there is no policy at this level, so the
#: number is the lab's own clock (`contract.CTRL_DT`) and a body that trains
#: one day states its real rate then.
DEFAULT_RATE_HZ = 50.0


# --------------------------------------------------------------- the scene


def _static_scene(path: Path) -> Path:
    """`scene_fn` for a body whose scene is a file that already exists.

    A module-level function behind `functools.partial` rather than a lambda:
    `BodyBase.scene_fn` is a FIELD, so whatever goes in it is carried by every
    `dataclasses.replace` (the conformance suite's planted negatives are all
    `replace`) and would be pickled by a vec-env worker. A lambda is
    unpicklable and a closure no better — `robots/g1.py` paid for that lesson
    with `G1Body` at module level.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"the scene {path} is gone — re-fetch this body's assets")
    return path


def _generated_scene(robot_xml: Path, out: Path, add_key: str) -> Path:
    """Write `out`: the robot, a floor, a light, and a keyframe. Return it.

    Atomic (temp + `os.replace`) and rewritten only when the content differs —
    `g1.g1_scene_xml`'s pattern for its reason: a parallel worker must never
    import a half-written scene (`AGENTS.md`, "Atomic writes and live
    imports").

    `add_key` is the keyframe name to write from `qpos0`, or "" when the robot
    file already carries one (which then survives into the scene, because this
    ADDS to the robot's own spec rather than attaching it into a fresh world).
    """
    spec = load_robot_spec(robot_xml)
    w = spec.worldbody
    light = w.add_light()
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    floor = w.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [10.0, 10.0, 0.05]
    if add_key:
        # Sized from a SEPARATE compile of the robot, the way
        # `mars._scene_spec` does: `MjSpec.compile()` normalises the spec it
        # is called on, and this one still has to survive `to_xml()`.
        probe = load_robot_spec(robot_xml).compile()
        key = spec.add_key()
        key.name = add_key
        key.qpos = np.asarray(probe.qpos0, np.float64).copy()
        key.qvel = np.zeros(probe.nv)
    xml = spec.to_xml()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text() != xml:
        tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, out)
    return out


def load_robot_spec(robot_xml: Path) -> mujoco.MjSpec:
    """`robot_xml` as an `MjSpec`, with its asset paths made ABSOLUTE.

    The one rewrite a stranger's MJCF needs, and it is not optional: MEASURED,
    `MjSpec.from_file` keeps `meshdir` exactly as the file spells it
    (`"assets"` for both the duck and every Menagerie model), `to_xml()`
    writes that relative path straight back out, and the generated scene then
    fails to compile from anywhere but the model's own directory — `Error
    opening file 'assets/banana_pcb_locker.stl'`. Resolving the three asset
    directories against the file they were read from is what lets a generated
    scene live under `.cache` while the meshes stay where they were
    downloaded, and what lets `attach()` put this robot in a world compiled
    from somewhere else entirely.

    Absent directories are left alone rather than defaulted, so a model that
    keeps its meshes beside the XML is untouched.
    """
    if not Path(robot_xml).is_file():
        raise FileNotFoundError(f"no MJCF at {robot_xml}")
    try:
        spec = mujoco.MjSpec.from_file(str(robot_xml))
    except ValueError as exc:
        raise ValueError(
            f"{robot_xml} is not a model MuJoCo can read: {exc}") from exc
    base = Path(robot_xml).resolve().parent
    for attr in ("meshdir", "texturedir", "assetdir"):
        rel = getattr(spec, attr, "") or ""
        if rel and not os.path.isabs(rel):
            setattr(spec, attr, str((base / rel).resolve()))
    return spec


def compile_robot(robot_xml: Path) -> mujoco.MjModel:
    """The robot alone, compiled, with a mesh failure named.

    A missing mesh is the single most common way somebody else's MJCF arrives
    broken (a partial download, a case-sensitive filesystem, an `assets/`
    directory that was not copied). MuJoCo says which file; this says which
    MODEL, because at the point a registry is building four bodies the file
    name alone does not identify the robot.
    """
    try:
        return load_robot_spec(robot_xml).compile()
    except ValueError as exc:
        raise ValueError(f"{robot_xml} does not compile: {exc}") from exc


# ------------------------------------------------------------- reading it


def scalar_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Every hinge/slide joint, in the model's own (tree) order.

    MuJoCo numbers joints in tree order, so this is the order the file
    declares them in — which is the order a person reading the MJCF expects
    the 🎬 panel's sliders and an action vector to be in.
    """
    return tuple(model.joint(j).name
                 for j in range(model.njnt)
                 if model.jnt_type[j] in _SCALAR_JOINTS)


def visual_geom_group(model: mujoco.MjModel,
                      default: int = DEFAULT_VISUAL_GROUP) -> int:
    """Which geom group holds this model's visual meshes — READ, not assumed.

    `contype == 0` is the definition of a geom that exists to be looked at:
    it collides with nothing. So the visual group is the one with the most
    non-colliding MESH geoms in it, and the fallback is only for a model that
    has no such geom at all (every mesh collides, or there are no meshes).

    Assuming 2 would be right for Menagerie and wrong for MARS (MuJoCo's URDF
    importer emits 1, `robots/mars.VISUAL_GROUP`), and the failure is the
    silent kind: an empty mesh dump is a body nobody can see, and "blank
    stage" has already cost this repo a debugging session that went looking
    in the viewer (`AGENTS.md`).

    MEASURED, at the groups this returns: duck 2 (70 geoms), G1 2, go2 2,
    so_arm100 2, MARS 1.
    """
    counts: dict[int, int] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_contype[g]) != 0:
            continue
        group = int(model.geom_group[g])
        counts[group] = counts.get(group, 0) + 1
    if not counts:
        return int(default)
    return max(counts, key=lambda g: (counts[g], -g))


def pick_keyframe(model: mujoco.MjModel) -> str | None:
    """The keyframe this body should spawn from, or None if it has none.

    `home` first because that is Menagerie's convention across all 71 models
    and the one name a stranger's file is likely to carry; then this repo's
    own two; then the FIRST keyframe, on the argument that a file's first
    keyframe is the one its author meant as the default.

    "The first one" is deliberately not "the only sensible one": the duck's
    scene lists INIT, STAND, SIT, FOLD and its own `default_pose` is STAND,
    not INIT (MEASURED: INIT is 0.458 rad away from `contract.DEFAULT_POSE`,
    STAND is 4e-5). A generic reader cannot know that, which is exactly why
    `MjcfBody.from_mjcf` takes a `keyframe` override and why the one it
    picked is recorded in `extra["pose_source"]` instead of being implied.
    """
    names = [model.key(i).name for i in range(model.nkey)]
    for preferred in PREFERRED_KEYS:
        if preferred in names:
            return preferred
    return names[0] if names else None


def joint_subset(model: mujoco.MjModel, qpos: np.ndarray,
                 joint_names: tuple[str, ...]) -> np.ndarray:
    """The scalar joints' values out of a full `qpos` vector, by NAME.

    By name and never by a `qpos[7:]` slice: which address a joint occupies
    depends on whether the root is free, and a slice would be an assumption a
    model revision could break in silence.
    """
    adr = [int(model.joint(n).qposadr[0]) for n in joint_names]
    return np.asarray(qpos, np.float64)[adr].astype(np.float32)


class _Scene(NamedTuple):
    """What `_resolve_scene` decided: the scene, the keyframe, and the pose.

    A named tuple rather than five positional values, because the caller has
    to thread them into a dataclass construction and `pose_model is not None
    and int(pose_model.nkey)` inlined there was the kind of expression nobody
    can read twice the same way.

    `keyframe` is never empty and never None: every branch either found a
    keyframe or WROTE one. `pose_source` is the sentence recorded in
    `extra` — it distinguishes a keyframe somebody authored from a `HOME`
    written out of `qpos0`, which is the one thing a reader of a stage pitch
    needs to know and cannot see from the keyframe's name.
    """

    path: Path
    #: "given" (somebody else's file, untouched) or "generated" (ours, under
    #: the cache). It decides whether `scene_fn` may rebuild the file.
    source: str
    keyframe: str
    pose_source: str
    pose: Any                                   # np.ndarray, rad
    #: Did WE write `keyframe`, out of `qpos0`? A flag and not a sniff at
    #: `pose_source`'s wording: `scene_fn` has to rebuild the scene the same
    #: way it was built the first time, and deciding that by parsing a
    #: sentence meant for a person is how the two drift apart.
    wrote_keyframe: bool = False

    def fn(self, robot_xml: Path) -> Callable[[], Path]:
        """The `scene_fn` for this scene, picklable (see `_static_scene`).

        A given scene is somebody else's file and is only ever checked; a
        generated one is rebuilt on demand from the robot, writing the
        keyframe again only if we wrote it the first time.
        """
        if self.source == "given":
            return partial(_static_scene, self.path)
        return partial(_generated_scene, robot_xml, self.path,
                       self.keyframe if self.wrote_keyframe else "")


# ---------------------------------------------------------------- the body


@dataclass(frozen=True, eq=False, kw_only=True)   # eq=False: ndarray fields
class MjcfBody(BodyBase):
    """A `Body` read out of an MJCF file. Build one with `from_mjcf`.

    A dataclass with FIELDS and a classmethod factory, rather than an
    `__init__` that reads the model: `dataclasses.replace` is how the
    conformance suite plants every one of its negatives (a wrong joint name, a
    short default pose, a drifted stage pitch), and a constructor that
    re-derived its fields from disk would quietly undo the plant and turn ten
    planted regressions into ten green no-ops. So all the reading happens once
    in `from_mjcf`, and a `replace` copy is exactly as broken as it was asked
    to be.
    """

    #: The model this body IS: what `attach()` puts in a world and what
    #: `visual_scene()` dumps. Never the scene — a scene carries a floor and a
    #: light, and attaching those into a room would give it two floors.
    robot_xml: Path
    #: Which geom group `visual_scene()` reads. Detected, see
    #: `visual_geom_group`.
    visual_group: int = DEFAULT_VISUAL_GROUP
    #: Free-form, and the place the reading records ITSELF: `pose_source`
    #: (which keyframe the default pose came from, or `qpos0`), `scene_source`
    #: ("given" / "generated"), `measured_width_m`, `skipped_joints` (the
    #: free/ball roots left out of `joint_names`). `RobotSpec.extra` is the
    #: same field for the same reason.
    extra: Mapping[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------- building

    @classmethod
    def from_mjcf(cls, robot_xml: Path | str, *, id: str,
                  scene_xml: Path | str | None = None,
                  manifest: Mapping[str, Any] | None = None,
                  cache_dir: Path | str | None = None) -> MjcfBody:
        """Read `robot_xml` and answer the whole `Body` contract from it.

        `scene_xml` is the model the body SPAWNS in — the shipped `scene.xml`
        of a Menagerie model, or the duck's `scene_walk.xml`. Left out, one is
        generated under `cache_dir` (the robot, a floor, a light, a keyframe).
        Passing one is not free: its nq must equal the robot's, or the two
        files describe different robots and every keyframe read from the scene
        would land on the wrong joints. MEASURED, that is not hypothetical —
        `g1.g1_xml()` (43 joints) and `g1.g1_scene_xml()` (29, the fingers
        frozen) are exactly such a pair, and this refuses it.

        `manifest` is the small dict a catalogue can override the reading
        with, all optional: `title`, `noun`, `keyframe` (name a keyframe
        instead of taking `pick_keyframe`'s), `visual_group`, `joint_groups`.
        Nothing in it is required and nothing in it is invented — an absent
        key means "read it".
        """
        man = dict(manifest or {})
        robot_xml = Path(robot_xml).resolve()
        model = compile_robot(robot_xml)

        joint_names = scalar_joint_names(model)
        if not joint_names:
            raise ValueError(
                f"{robot_xml} has no hinge or slide joints — there is nothing "
                f"to pose or to act on, so it cannot be a body (it has "
                f"{model.njnt} joint(s), none of them scalar)")
        skipped = tuple(model.joint(j).name for j in range(model.njnt)
                        if model.jnt_type[j] not in _SCALAR_JOINTS)

        scene = cls._resolve_scene(robot_xml, model, joint_names, scene_xml,
                                   man, cache_dir, id)

        scene_model = mujoco.MjModel.from_xml_path(str(scene.path))
        # Every reader in this repo looks the spawn pose up BY NAME on
        # `scene_fn()`, so a keyframe that did not survive into the scene is a
        # body that cannot spawn — and the failure would otherwise surface in
        # the conformance suite's first case, a layer away from the cause.
        if mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY,
                             scene.keyframe) < 0:
            raise ValueError(
                f"{id}: the keyframe {scene.keyframe!r} is not in "
                f"{scene.path}")
        spacing, width = measure_lab_spacing_m(scene_model, scene.keyframe)

        group = int(man.get("visual_group", visual_geom_group(model)))
        groups = man.get("joint_groups")
        n = len(joint_names)
        return cls(
            id=str(id),
            title=str(man.get("title") or title_of(id)),
            noun=str(man.get("noun") or title_of(id)),
            kind="generic",
            joint_names=joint_names,
            joint_groups=tuple(groups) if groups else None,
            default_pose=scene.pose,
            # joint_pos + joint_vel + last_action. See `contract()`.
            obs_dim=3 * n,
            lab_spacing_m=spacing,
            scene_fn=scene.fn(robot_xml),
            stand_keyframe=scene.keyframe,
            robot_xml=robot_xml,
            visual_group=group,
            extra={
                "pose_source": scene.pose_source,
                "scene_source": scene.source,
                "scene_path": str(scene.path),
                "measured_width_m": round(width, 6),
                "skipped_joints": skipped,
                **{k: v for k, v in man.items()
                   if k not in ("title", "noun", "keyframe", "visual_group",
                                "joint_groups")},
            },
        )

    @staticmethod
    def _resolve_scene(robot_xml: Path, model: mujoco.MjModel,
                       joint_names: tuple[str, ...],
                       scene_xml: Path | str | None,
                       man: Mapping[str, Any],
                       cache_dir: Path | str | None,
                       body_id: str):
        """Pick the scene and the keyframe, and read the default pose.

        Split out of `from_mjcf` because it is the one part with a decision in
        it, and the decision has four branches worth naming:

        1. a scene was GIVEN and the keyframes live in it (the duck: its
           robot file carries none at all, its scene carries four);
        2. a scene was given and the keyframes live in the ROBOT (every
           Menagerie model: `home` is in `go2.xml`, and `scene.xml` includes
           it);
        3. no scene, so one is generated — and if nothing anywhere carries a
           keyframe, a `HOME` is written from `qpos0`;
        4. a scene was given and NOTHING carries a keyframe, which falls
           through to (3). A given scene cannot be handed a keyframe (it is
           somebody else's file and is never written to), and every reader in
           this repo looks the spawn pose up by NAME on `scene_fn()` — so
           keeping the given scene here would leave `stand_keyframe` naming a
           keyframe that does not exist anywhere. That is the one shape of
           this function that is a silent lie rather than an error.

        Returns a `_Scene`, whose `keyframe` is NEVER None — every branch
        either found one or wrote one, which is the invariant that lets
        `stand_keyframe` be a name the scene really carries.
        """
        want = man.get("keyframe")
        if scene_xml is not None:
            scene_path = Path(scene_xml).resolve()
            scene_model = mujoco.MjModel.from_xml_path(str(scene_path))
            if int(scene_model.nq) != int(model.nq):
                raise ValueError(
                    f"{body_id}: the scene {scene_path.name} has nq "
                    f"{scene_model.nq} and the robot {robot_xml.name} has "
                    f"{model.nq} — they are not the same robot, so any "
                    "keyframe read from one would land on the other's joints")
            # The robot's own keyframes win: that is where a model author
            # puts `home`, and a scene that merely includes the robot has the
            # same ones anyway. The scene is consulted for the case the duck
            # is (keyframes only in the scene).
            for source_model in (model, scene_model):
                key = _keyframe_named(source_model, want)
                if key is not None:
                    name, qpos = key
                    _check_nq(body_id, name, qpos, source_model)
                    return _Scene(scene_path, "given", name,
                                  f"keyframe {name!r}",
                                  joint_subset(source_model, qpos, joint_names))
            if want:
                raise ValueError(
                    f"{body_id}: neither {robot_xml.name} nor "
                    f"{scene_path.name} has a keyframe called {want!r}")
            # Branch 4: fall through and generate one, so the keyframe this
            # body names is a keyframe its scene HAS.

        cache = Path(cache_dir) if cache_dir else robot_xml.parent
        out = cache / f"scene_{_slug(body_id)}.xml"
        key = _keyframe_named(model, want)
        if key is None and want:
            raise ValueError(f"{body_id}: {robot_xml.name} has no keyframe "
                             f"called {want!r}")
        if key is None:
            # Nothing to spawn from, so the scene gets one. `qpos0` is the
            # model's own rest pose, which is the only honest default — and
            # `pose_source` says so, because "HOME" alone would read as a
            # keyframe somebody authored.
            _generated_scene(robot_xml, out, GENERATED_KEY)
            return _Scene(out, "generated", GENERATED_KEY,
                          f"qpos0, written as keyframe {GENERATED_KEY!r}",
                          joint_subset(model, model.qpos0, joint_names),
                          wrote_keyframe=True)
        name, qpos = key
        _check_nq(body_id, name, qpos, model)
        _generated_scene(robot_xml, out, "")
        return _Scene(out, "generated", name, f"keyframe {name!r}",
                      joint_subset(model, qpos, joint_names))

    # ------------------------------------------------------- the contract

    def contract(self) -> PolicyContract:
        """The level-0 contract: proprioception in, joint targets out.

        Declared even though nothing trains against it, for the reason
        `robots/policy_contract.py` gives: a body has to speak ONE contract
        from the day it is listed, or the first policy anyone exports for it
        is a file with a width and no meaning. The layout is the smallest
        honest one — what the body can observe about itself with no declared
        sensor, no command channel and no task — and the `v0` says the level:
        the day this body declares feet or an effector, the env that fills its
        observation defines a real layout and the version bumps with it.

        The id carries the WIDTH (`mjcf-<id>-<obs_dim>-v0`) because that is
        this repo's contract-id convention and `tests/test_policy_contract.py`
        pins every body to it: a mismatch is then legible in a log line
        without looking anything up.
        """
        n = self.num_joints
        return declare(
            self,
            id=f"mjcf-{self.id}-{self.obs_dim}-v0",
            rate_hz=DEFAULT_RATE_HZ,
            slots=(Slot("joint_pos", 0, n),
                   Slot("joint_vel", n, 2 * n),
                   Slot("last_action", 2 * n, 3 * n)),
            deploy="level 0: a generic MJCF body — on the stage and in a "
                   "room; no task, no policy, no hardware claim")

    # ---------------------------------------------------------------- assets

    def ready(self) -> bool:
        """The MJCF is on disk, and so is a scene to spawn in.

        Not `BodyBase`'s "is the scene file there": a generated scene does not
        exist until something asks for it, so the generic answer would be
        False on a body whose assets are all present.
        """
        try:
            return Path(self.robot_xml).is_file() and Path(self.scene_fn()).is_file()
        except Exception:
            return False

    def fetch(self) -> Path:
        """Nothing to download: this body IS a file somebody already has.

        The directory it lives in, so the palette's one download path needs no
        special case. What puts a Menagerie model there is
        `robots/menagerie.fetch`, which runs before a body exists to be asked.
        """
        return Path(self.robot_xml).parent

    # ---------------------------------------------------------------- viewer

    def robot_model(self) -> mujoco.MjModel:
        """The robot alone, compiled — no floor, no light.

        Public because it is the only honest source for "how many meshes
        should the viewer have got", which the conformance suite has no seam
        for and reads per body.
        """
        return compile_robot(self.robot_xml)

    def visual_scene(self) -> dict:
        """The viewer's mesh dump, from the detected visual group.

        `g1.extract_visual_scene` is the body-agnostic dump (its own docstring
        marks the move to a shared module as PHASE 1B); the only per-body
        thing in it is the group, which this body measured.
        """
        from .g1 import extract_visual_scene
        return extract_visual_scene(self.robot_model(), group=self.visual_group)

    def look(self) -> str:
        """One material table for every body at level 0.

        Not `self.id` (`BodyBase`'s default): a stranger's robot has no
        material set in the viewer, and `generic` is the name of the table
        that paints an unknown body by its own MJCF rgba instead of leaving it
        black or borrowing a duck's colours.
        """
        return "generic"

    # --------------------------------------------------------------- training

    def env_class(self, task: str = "walk") -> type:
        raise NotImplementedError(
            f"{self.id!r} is a level-0 body (docs/mars-roadmap.md §7.1): an "
            f"MJCF and an id buy the stage, the pose editor and a room — not "
            f"a task. There is no env for {task!r} until this body declares "
            "its kind and that kind's names (feet + base + gyro for a "
            "walker; an effector for an arm)")

    def tasks(self) -> tuple:
        """No recipes. `BodyBase` would read `behaviors.for_robot(self.id)`
        and get the same empty answer; this says so without dragging the
        recipe registry in for a body that has no env to run one."""
        return ()

    def shipped_policies(self) -> tuple[dict, ...]:
        """An MJCF ships no policy. Menagerie ships none for any of its 71
        models, and a body whose contract is `v0` has nothing to run one
        against anyway."""
        return ()

    # ---------------------------------------------------------- /sim world

    def attach(self, spec: Any, prefix: str, frame: Any) -> None:
        """Put one of these in a `/sim` world model under `prefix`.

        The ROBOT spec, with its asset paths already absolute
        (`load_robot_spec`), so the world it joins can be compiled from
        anywhere. `world/compose.py`'s pattern, and the reason level 0
        includes "a kinematic body in a room".
        """
        spec.attach(load_robot_spec(self.robot_xml), prefix=prefix, frame=frame)

    def driver(self, model: Any, prefix: str) -> Any:
        raise NotImplementedError(
            f"{self.id!r} is a level-0 body: it can STAND in a room, not be "
            "driven in one. A driver is level 2 (docs/mars-roadmap.md §7.1) "
            "and needs an ONNX with a contract id or a scripted controller — "
            "robots/g1.G1Walker and robots/mars_drive.MarsDriver are the two "
            "shapes")

    def frames(self) -> Any:
        """No root link and no gaze joint — and both are NAMES, not guesses.

        `mj_name2id` answers -1 for a base link a body does not have, and a
        `WorldRobot` built on -1 slices the wrong subtree out of a composed
        model (`robots/body.RobotFrames`' own docstring). So this raises
        rather than picking body 1 and hoping.
        """
        raise NotImplementedError(
            f"{self.id!r} does not declare its /sim frames: which link is the "
            "root and which joint is the gaze. Level 2, and only a "
            "driver()-stepped body is ever asked (world/arena.WorldRobot) — "
            "a level-0 body stands where it was put")

    # `make_sensors` is INHERITED on purpose — see the module docstring. An
    # empty dict is `robots/body.py`'s own documented answer for a body at
    # level 0, and raising would take a kinematic body out of the room.


def title_of(body_id: str) -> str:
    """"menagerie:unitree_go2" -> "Unitree Go2". A label, not an id.

    The namespace is dropped and underscores become spaces because this is
    what a palette chip and a `--robot` listing show. `from_mjcf`'s manifest
    can override it, which is how a catalogue that knows the real product
    name says so.
    """
    name = str(body_id).split(":", 1)[-1]
    return " ".join(part.capitalize() for part in name.replace("-", "_").split("_"))


def _slug(body_id: str) -> str:
    """A filename-safe form of an id, for the generated scene's name."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(body_id))


def _keyframe_named(model: mujoco.MjModel, want: str | None):
    """`(name, qpos)` of the chosen keyframe, or None if there is none.

    `want` names one explicitly; without it `pick_keyframe` decides.
    """
    if want:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, str(want)) < 0:
            return None
        key = model.key(str(want))
        return str(want), np.asarray(key.qpos, np.float64)
    name = pick_keyframe(model)
    if name is None:
        return None
    key = model.key(name)
    return name, np.asarray(key.qpos, np.float64)


def _check_nq(body_id: str, name: str, qpos: np.ndarray,
              model: mujoco.MjModel) -> None:
    """A keyframe from a different model is a wrong pose, not an error later.

    MuJoCo itself refuses a `<key qpos>` of the wrong length at COMPILE time,
    so this catches the other way in: a keyframe read off one model and
    applied to another (a frozen-joint variant, a revision that added a DoF).
    Loud, because the silent version is a robot spawned folded into the floor
    and a whole session spent blaming the policy.
    """
    if int(np.asarray(qpos).size) != int(model.nq):
        raise ValueError(
            f"{body_id}: keyframe {name!r} is {np.asarray(qpos).size} floats "
            f"and the model has nq {model.nq} — that keyframe belongs to a "
            "different robot")


__all__ = ["DEFAULT_RATE_HZ", "DEFAULT_VISUAL_GROUP", "GENERATED_KEY",
           "MjcfBody", "compile_robot", "joint_subset", "load_robot_spec",
           "pick_keyframe", "scalar_joint_names", "title_of",
           "visual_geom_group"]
