"""What the LAB needs to know about a robot — walker or not.

`robots/spec.py`'s `RobotSpec` is honestly named in its own docstring: "what
the **walking env** needs to know about a robot". Its fields are foot geoms,
fall thresholds, air-time windows, twist-command ranges, a gyro sensor. Every
body so far entered through it because every body so far walked.

The third body planned (Innate's MARS, `docs/mars-roadmap.md` §1) is a wheeled
base with an arm. It has no feet, cannot fall, and the thing this harness calls
"the policy" — a velocity-command walker — does not exist for it. Threading it
through `RobotSpec` would be the architecture-scale version of the mistake
`AGENTS.md` warns about most: keeping an inherited term whose default target is
wrong for the new body, and letting the robot sag into the proxy.

So the lab's contract is split from the walker's:

    Body       what the LAB, the VIEWER and /sim need — this file
    RobotSpec  Body + the walking env's names and windows — spec.py

`RobotSpec` inherits `BodyBase`, so the duck and the G1 keep every field and
every byte of behaviour (the golden-bit tests are the proof, as they were when
the G1 arrived). A non-walker implements `Body` directly and never sees
`foot_geoms`, `fall_gravity_z` or an air-time window.

`Body` is a Protocol, not a base class to inherit: a body that arrives from a
pip-installed plugin through the `microduck_local.bodies` entry point
(`robots/registry.py`) conforms by shape, not by importing this package's
class.

**`conforms()`, not `isinstance()`, is the check** — measured, because the
obvious one is wrong here. Python 3.12 resolves a runtime-checkable Protocol's
members with `inspect.getattr_static`, which deliberately does not run
`__getattr__`; the G1's spec is a lazy proxy built on `__getattr__`
(`robots/g1._LazyG1Spec`, so that importing the module does not read the
fetched config), so `isinstance(G1_SPEC, Body)` is **False** while the G1 is a
perfectly good body. `conforms()` asks `hasattr`, which the proxy answers.
Keep the Protocol for the type checker and the documentation; use `conforms()`
when the answer decides anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


def _no_scene_fn() -> Path:
    """The `scene_fn` placeholder: a body that never declared one.

    A default that RAISES rather than a `None` to be checked at every call
    site, and rather than one robot's scene — which is the whole reason the
    duck's assets moved off `RobotSpec` (`robots/microduck.py`'s docstring).
    `RobotSpec.__post_init__` refuses it outright, so a walker still fails at
    CONSTRUCTION as it did when the field was declared there.
    """
    raise NotImplementedError(
        "this body declares no scene_fn — a Body must be able to hand out a "
        "compiled-scene path (robots/g1.g1_scene_xml and "
        "robots/mars.scene_xml both generate theirs under .cache)")


@runtime_checkable
class Body(Protocol):
    """One robot the lab can list, draw, teach and put in a room.

    Everything here is something a NON-walker can answer. The walking env's
    questions live on `RobotSpec`, which is a `Body` plus those.
    """

    # --- identity ---------------------------------------------------------
    id: str                      # "microduck" | "g1" | "mars" — the wire name
    title: str                   # "Unitree G1" — what a chip or menu shows
    noun: str                    # "duck" | "G1" — what a SENTENCE calls one
    kind: str                    # "legged" | "wheeled"; "arm" reserved
    # --- the policy contract this body speaks -----------------------------
    joint_names: tuple[str, ...]
    joint_groups: tuple[str, ...] | None
    default_pose: Any            # np.ndarray, rad, len == num_joints. `Any`
    #                              so conforming does not drag numpy in.
    obs_dim: int
    lab_spacing_m: float         # floor one slot on the lab stage gets, m
    # --- the model, and the pose it starts in -----------------------------
    # Both were `RobotSpec` fields until a non-walker needed them: EVERY body
    # has a compiled scene and a keyframe to spawn at, walker or not, and
    # `tests/test_body_conformance.py`'s generic cases read exactly these two
    # to compile a body and measure it. A body that cannot answer them is one
    # the suite cannot check, so the contract says so here.
    scene_fn: Callable[[], Path]
    stand_keyframe: str

    @property
    def num_actions(self) -> int: ...

    def contract(self) -> Any: ...   # robots/policy_contract.PolicyContract
    def ready(self) -> bool: ...
    def fetch(self) -> Path: ...
    def setup_hint(self) -> str: ...
    def visual_scene(self) -> dict: ...
    def look(self) -> str: ...
    def env_class(self, task: str = "walk") -> type: ...
    def tasks(self) -> tuple[Any, ...]: ...
    def shipped_policies(self) -> tuple[dict, ...]: ...
    def train_env_kwargs(self, args: Any) -> dict: ...
    def attach(self, spec: Any, prefix: str, frame: Any) -> None: ...
    def driver(self, model: Any, prefix: str) -> Any: ...


@dataclass(frozen=True, eq=False, kw_only=True)   # eq=False: ndarray fields
class BodyBase:
    """The `Body` fields, as data, plus the answers that are body-agnostic.

    Keyword-only because `RobotSpec` adds REQUIRED fields after these
    defaulted ones; a dataclass cannot order that positionally, and every
    construction in the tree was already by keyword.

    A method here either has one true generic answer (`num_actions`,
    `tasks`, `look`, `setup_hint`) or raises with the name of the thing to
    override. It never guesses: a body that inherited another robot's meshes
    or another robot's env would fail as a wrong picture and a wrong policy
    rather than as an error, and this harness has paid for that class of
    silence more than once (`AGENTS.md`, "Verification discipline").
    """

    id: str
    joint_names: tuple[str, ...]
    default_pose: Any                          # np.ndarray, rad
    obs_dim: int
    title: str = ""
    # What the UI calls one of these in a sentence: "teach the {noun} a
    # trick". Empty falls back to the title.
    noun: str = ""
    # What SHAPE of robot this is, so a feature can ask instead of listing
    # ids: "legged" (duck, G1), and "wheeled" / "arm" reserved for MARS.
    # The walking env's questions are on RobotSpec, not behind this string —
    # `kind` is for the UI and for picking a base env, never for physics.
    kind: str = "legged"
    # Panel section label per joint, in joint_names order (None: one group).
    joint_groups: tuple[str, ...] | None = None
    # How much floor ONE SLOT on the lab stage gets, centre to centre (m).
    # The lab lays its roster out in a square grid and each slot is pitched by
    # its own robot (viz_server.lab_slot_offsets), because one duck-sized
    # constant put six 1.3 m G1 helpers inside each other.
    # Measured at each robot's own STAND keyframe — the widest HORIZONTAL
    # extent of the whole body's geom AABBs, which is what must not overlap:
    #     microduck  0.1845 m wide  ->  0.65 m   (3.52x, the pitch the viewer
    #                                             has always drawn: this default)
    #     g1         0.5338 m wide  ->  1.88 m   (the same 3.52x; robots/g1.py)
    # tests/test_lab_robots.py re-measures both widths from the models, so a
    # model revision that changes a body's size fails here rather than quietly
    # crowding the stage. (Scaling on standing height instead — 0.277 m vs
    # 1.307 m — would give the G1 3.07 m: also defensible, simply further apart
    # than the stage camera can frame.)
    lab_spacing_m: float = 0.65
    # The compiled scene ONE of these stands in, as a callable so a body
    # whose scene is GENERATED (the G1's, MARS's) writes it on demand rather
    # than at import. See `_no_scene_fn` for why the default raises instead
    # of being None or one robot's scene.
    scene_fn: Callable[[], Path] = _no_scene_fn
    # The keyframe in that scene a body spawns from. Named for the walkers
    # that came first — MARS's is "HOME", an arm pose with nothing to stand
    # on — and kept under that name because the walking env, the 🎬 pose
    # editor, `render-rollout` and the conformance suite all read it already.
    stand_keyframe: str = "STAND"

    # ------------------------------------------------------------------ data

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def num_actions(self) -> int:
        """One action per joint. A body whose action vector is not its joint
        vector overrides this — MARS's is 6 joints plus a base twist."""
        return len(self.joint_names)

    # -------------------------------------------------------- the contract

    def contract(self) -> Any:
        """What this body's policies SAY THEY ARE: a `PolicyContract`.

        No generic answer, and deliberately not a synthesised one. An id and
        a width could be spelled from `self.id` and `self.obs_dim`, but the
        two things that make a contract worth carrying cannot be guessed: the
        control RATE (the walkers tick at 50 Hz, MARS's skills at 25, and a
        policy run at the wrong one is a different controller) and the
        DEPLOY sentence (whether putting this on hardware is the robot's own
        contract, a lab convenience, or an untested code skill). A guessed id
        over a wrong rate is precisely the silent cross that
        `robots/policy_contract.py` exists to stop, so a body that has not
        declared its contract says so.

        The slot table is the other half: it must be read off the env that
        FILLS the observation, which only the body's own module knows.
        """
        raise NotImplementedError(
            f"{self.id!r} does not declare its policy contract — implement "
            "contract() returning a robots/policy_contract.PolicyContract "
            "(robots/microduck.py, robots/g1.py and robots/mars.py are the "
            "three shapes: a deployment contract, a lab contract and a code "
            "skill)")

    # ---------------------------------------------------------------- assets

    def ready(self) -> bool:
        """Are this body's assets on disk?

        The lab asks before offering a chip, so the palette can show "not set
        up yet — ⤓" instead of a chip that throws when clicked.
        """
        raise NotImplementedError(
            f"{self.id!r} does not say whether its assets are present — "
            "implement ready() (robots/g1.py checks for the MJCF, the meshes "
            "and the walker ONNX)")

    def fetch(self) -> Path:
        """Download the assets and return the directory holding them.

        One click in the palette and `uv run fetch-robot <id>` both land here,
        so there is one download path per body instead of a script per body.
        """
        raise NotImplementedError(
            f"{self.id!r} has no fetch() — a body whose assets ship with the "
            "checkout returns its asset directory unchanged")

    def setup_hint(self) -> str:
        """The command that makes `ready()` true, for an error message."""
        return f"uv run fetch-robot {self.id}"

    # ---------------------------------------------------------------- viewer

    def visual_scene(self) -> dict:
        """The viewer's mesh dump: {bodies, meshes, geoms, vertScale}."""
        raise NotImplementedError(
            f"{self.id!r} has no visual_scene() — robots/g1.extract_visual_scene "
            "is the mm-int dump every body's should produce")

    def look(self) -> str:
        """Which material set the viewer paints this body with.

        An id by default, so a new body draws with its own look the moment it
        has one and never silently borrows another robot's colours.
        """
        return self.id

    # --------------------------------------------------------------- training

    def env_class(self, task: str = "walk") -> type:
        """The gymnasium env that trains `task` on this body.

        This was `train.env_class`'s if-chain. It moved here so a third body
        is a registry entry rather than a third branch in the trainer.
        """
        raise NotImplementedError(
            f"{self.id!r} has no env for task {task!r} — implement env_class()")

    def tasks(self) -> tuple[Any, ...]:
        """The `Behavior` recipes the 🎓 panel offers for this body.

        Delegated, not duplicated: `behaviors/` is the one place a recipe is
        declared, and it already tags each one with the robot it belongs to.
        """
        from ..behaviors.core import for_robot
        return tuple(for_robot(self.id))

    def train_env_kwargs(self, args: Any) -> dict:
        """The env kwargs a `train-walk` invocation applies for this body.

        `train.env_kwargs_from_args` carried these as an `if robot == "g1"`
        block — the actuator model it refuses, the episode length its held
        poses need. Same reason as `env_class`: a body's trainer knobs belong
        to the body.
        """
        return {}

    def shipped_policies(self) -> tuple[dict, ...]:
        """Policies that came WITH this body, as palette entries.

        The shape is the one `viz_server.discover_policies` appends:
        `{id, label, group, path, robot}` plus an optional measured `note`.
        Empty is the normal answer — most bodies ship nothing.
        """
        return ()

    # ------------------------------------------------------------- /sim world

    def attach(self, spec: Any, prefix: str, frame: Any) -> None:
        """Attach one of these into a `/sim` world model under `prefix`.

        `world/compose.py` puts N bodies in one `MjSpec` this way. Phase 3
        makes `compose()` call this; until then it is the declared seam and
        `compose()` still holds its own copy of both paths.
        """
        raise NotImplementedError(
            f"{self.id!r} has no attach() — world/compose.py shows the pattern "
            "(MjSpec.attach under a per-robot prefix and frame)")

    def driver(self, model: Any, prefix: str) -> Any:
        """A controller that STEPS one of these in a `/sim` world.

        Deliberately not built in this phase. The two that exist are
        `robots/g1.G1Walker` (a 99-obs ONNX at 50 Hz) and the duck's walker +
        command block inside `world/arena.WorldDuck`, which is still sized by
        the 61-obs contract (`docs/mars-roadmap.md` §6.5). Generalising them
        is Phase 3's job, and doing it here would mean designing the sense and
        intent channels against one body — the way `WorldDuck` already was.
        """
        raise NotImplementedError(
            f"{self.id!r} has no driver() — see robots/g1.G1Walker and "
            "world/arena.WorldDuck (docs/mars-roadmap.md Phase 3)")


def conforms(body: object) -> tuple[str, ...]:
    """Which `Body` names `body` is missing — empty when it conforms.

    The registry's gate, for two reasons. It SAYS what is wrong, because a
    plugin author reading "not a Body" learns nothing. And it sees through a
    lazy proxy, which `isinstance(x, Body)` does not — see the module
    docstring; `G1_SPEC` is exactly such a proxy.

    The names are spelled out rather than read from `Body.__protocol_attrs__`
    so the ORDER of the message is the order the contract is documented in,
    and so this keeps working if that private attribute moves. Two lists that
    must agree is exactly the kind of duplication that drifts, so
    `tests/test_registry.py` pins them against each other.
    """
    return tuple(n for n in WANTED if not hasattr(body, n))


#: Every name a `Body` must have, in the order `Body` declares them.
WANTED: Sequence[str] = (
    "id", "title", "noun", "kind", "joint_names", "joint_groups",
    "default_pose", "obs_dim", "num_actions", "lab_spacing_m",
    "scene_fn", "stand_keyframe",
    "contract",
    "ready", "fetch", "setup_hint", "visual_scene", "look",
    "env_class", "tasks", "shipped_policies", "train_env_kwargs",
    "attach", "driver",
)


__all__ = ["WANTED", "Body", "BodyBase", "conforms"]
