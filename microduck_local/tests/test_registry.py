"""The body registry: the seam a THIRD robot enters through.

`docs/mars-roadmap.md` §1 counts what the old answer cost — 45 hand-written
`"g1"` literals, one of which (`/robots`) had to patch an absent G1 back into
a list the registry had dropped. The point of a registry is that the next body
is an entry, not a pass over the tree, so most of this file is proven by a
body that is NOT the G1: a fake third one, registered at runtime and arriving
through an entry point, because a seam demonstrated only by the robot it was
extracted from has not been demonstrated at all.

The rest pins that nothing moved: `train.env_class` returns the same classes
it returned when it was an if-chain, compared against the classes themselves.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local import train as T
from microduck_local.robots import registry as R
from microduck_local.robots.body import BodyBase, conforms
from microduck_local.robots.g1 import g1_ready
from microduck_local.walk_env import MicroduckWalkEnv

needs_g1 = pytest.mark.skipif(
    not g1_ready(), reason="G1 assets missing — uv run fetch-robot g1")


# --------------------------------------------------------- a fake third body

class _FakeEnv:
    """Stands in for a trainer env class. Never instantiated."""


@dataclass(frozen=True, eq=False, kw_only=True)
class FakeBody(BodyBase):
    """A wheeled body with no feet, no gyro and no fall threshold.

    Deliberately a `BodyBase`, not a `RobotSpec`: the whole claim of
    `robots/body.py` is that a non-walker can be a lab citizen without the
    walking env's fields, and a fake that subclassed `RobotSpec` would have to
    invent a `base_body`, a `gyro_sensor` and two foot geoms to exist — which
    would prove the opposite.
    """

    def ready(self) -> bool:
        return True

    def fetch(self) -> Path:
        return Path("/nowhere")

    def visual_scene(self) -> dict:
        return {"bodies": [], "meshes": [], "geoms": [], "vertScale": 0.001}

    def env_class(self, task: str = "walk") -> type:
        if task != "roll":
            raise SystemExit(f"unknown --task {task!r} for {self.id} (have: roll)")
        return _FakeEnv

    def attach(self, spec: Any, prefix: str, frame: Any) -> None:
        return None


def fake_body(body_id: str = "wheelie") -> FakeBody:
    return FakeBody(
        id=body_id,
        title="Wheelie Mk I",
        noun="Wheelie",
        kind="wheeled",
        joint_names=("arm1", "arm2", "grip"),
        joint_groups=("arm", "arm", "gripper"),
        default_pose=np.zeros(3, np.float32),
        obs_dim=21,
        lab_spacing_m=1.09,
    )


@pytest.fixture
def clean_registry():
    """Restore the process-wide registry, whatever a test did to it.

    `_load_entry_points` is memoised (one `entry_points()` scan of every
    installed distribution per process), so a test that monkeypatches
    `entry_points` has to clear it on the way IN or it reads the real scan,
    and on the way OUT or it leaves its fake bodies in the cache for every
    later test.
    """
    before = dict(R._EXTRA)
    R._load_entry_points.cache_clear()
    try:
        yield
    finally:
        R._EXTRA.clear()
        R._EXTRA.update(before)
        R._load_entry_points.cache_clear()


# ------------------------------------------------------- registration by hand

def test_a_third_body_registered_at_runtime_is_a_first_class_robot(clean_registry):
    body = R.register(fake_body())
    assert "wheelie" in R.ids()
    assert "wheelie" in R.registry()
    assert R.get("wheelie") is body
    # And it DISPATCHES: the registry is only worth having if the thing it
    # returns answers the questions the trainer asks.
    assert R.get("wheelie").env_class("roll") is _FakeEnv
    assert R.get("wheelie").noun == "Wheelie"
    assert R.get("wheelie").kind == "wheeled"
    assert R.get("wheelie").num_actions == 3


def test_a_non_walker_needs_none_of_the_walking_env_fields(clean_registry):
    """The reason `Body` was split off `RobotSpec` at all."""
    body = R.register(fake_body())
    for walker_only in ("foot_geoms", "gyro_sensor", "base_body",
                        "fall_gravity_z", "ground_tol", "min_forward_cmd"):
        assert not hasattr(body, walker_only), walker_only
    assert conforms(body) == ()


def test_a_plugin_may_not_replace_a_builtin(clean_registry):
    """The duck's goldens and the G1's measurements name specific bodies."""
    with pytest.raises(ValueError, match="built-in"):
        R.register(fake_body("microduck"))
    with pytest.raises(ValueError, match="built-in"):
        R.register(fake_body("g1"))


def test_register_refuses_something_that_is_not_a_body(clean_registry):
    class Half:
        id = "half"

    with pytest.raises(TypeError, match="not a Body"):
        R.register(Half())


# ------------------------------------------------- discovery by entry point

class _FakeEntryPoint:
    """What `importlib.metadata.entry_points(group=...)` yields."""

    def __init__(self, name: str, value: str, obj):
        self.name, self.value, self._obj = name, value, obj

    def load(self):
        return self._obj


def test_an_installed_package_adds_a_body_through_the_entry_point(monkeypatch,
                                                                  clean_registry):
    """`pip install someone-elses-robot` and it is in the lab — no fork."""
    seen: dict[str, object] = {}

    def fake_entry_points(*, group):
        seen["group"] = group
        return [_FakeEntryPoint("wheelie", "plugin.robots:BODY", fake_body())]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    assert "wheelie" in R.ids()
    assert R.get("wheelie").env_class("roll") is _FakeEnv
    # The group name is the published contract; a typo here is a plugin
    # ecosystem that silently finds nothing.
    assert seen["group"] == "microduck_local.bodies"
    assert R.ENTRY_POINT_GROUP == "microduck_local.bodies"


def test_an_entry_point_may_be_a_zero_arg_factory(monkeypatch, clean_registry):
    """So a plugin can defer its own heavy imports, as `g1.py` does."""
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: [_FakeEntryPoint("w", "plugin:make", fake_body)])
    assert R.get("wheelie").title == "Wheelie Mk I"


def test_a_broken_plugin_is_an_absence_not_a_crash(monkeypatch, clean_registry):
    """One bad distribution must not take the duck down with it."""
    def boom():
        raise RuntimeError("no assets")

    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: [_FakeEntryPoint("bad", "plugin:boom", boom)])
    assert "microduck" in R.registry()
    assert "bad" not in R.ids()


def test_an_entry_point_cannot_shadow_a_builtin(monkeypatch, clean_registry):
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda *, group: [_FakeEntryPoint(
            "evil", "plugin:duck", fake_body("microduck"))])
    # The plugin's body must never enter the dict in the first place. Asserting
    # only on `get()` was TOOTHLESS: `registry()` merges entry points with
    # `setdefault`, so the built-in won on ordering even with the guard
    # deleted, and the test could not tell the two apart.
    assert "microduck" not in R._load_entry_points()
    assert R.get("microduck") is C.MICRODUCK


# ------------------------------------------------------- the built-in bodies

def test_the_duck_is_the_contract_object_itself():
    """`walk_env` and the goldens read `contract.MICRODUCK`; a registry that
    handed out a COPY would make the two describe different robots."""
    assert R.get("microduck") is C.MICRODUCK
    assert conforms(C.MICRODUCK) == ()


def test_both_builtins_answer_the_whole_body_contract():
    for body_id in ("microduck", "g1"):
        body = R.get(body_id)
        assert conforms(body) == (), body_id
        assert body.id == body_id
        assert body.noun and body.title
        assert body.kind == "legged"
        assert body.num_actions == len(body.joint_names)
        assert body.lab_spacing_m > 0
        assert isinstance(body.ready(), bool)
        assert isinstance(body.look(), str) and body.look()


def test_the_entry_point_scan_happens_once_per_process(clean_registry):
    """`registry()` runs per roster change and per policy load in the lab, and
    `entry_points()` walks the metadata of every installed distribution. What
    is installed cannot change inside a process, so the scan is memoised."""
    calls = {"n": 0}

    def counting_entry_points(*, group):
        calls["n"] += 1
        return []

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(importlib.metadata, "entry_points", counting_entry_points)
        R._load_entry_points.cache_clear()
        for _ in range(20):
            R.registry()
            R.ids()
        assert calls["n"] == 1, f"scanned {calls['n']} times, want 1"
    finally:
        monkey.undo()
        R._load_entry_points.cache_clear()


def test_a_registered_body_is_not_hidden_by_the_memoised_scan(clean_registry):
    """The cache must not swallow `register()`: the SCAN is memoised, the
    registry dict is rebuilt per call.

    Asserting only `"wheelie" in R.registry()` after a register was TOOTHLESS
    — with the whole of `registry()` cached it still passed, because an
    earlier test had already warmed that cache with the same fake body. So
    pin the property itself: a dict taken BEFORE the register must not
    contain the body, and must not be the same object as the one after.
    """
    before = R.registry()
    R.register(fake_body())
    after = R.registry()
    assert after is not before, "registry() handed back a cached dict"
    assert "wheelie" not in before
    assert "wheelie" in after
    assert "wheelie" in R.ids()


def test_the_conforms_list_is_the_protocols_own_members():
    """Two hand-kept lists that must agree: `Body`'s declared members and
    `body.WANTED`. A name added to the Protocol and forgotten here would be a
    contract the registry never checks."""
    from microduck_local.robots import body as B

    declared = getattr(B.Body, "__protocol_attrs__", None)
    if declared is None:                      # a Python that renamed it
        pytest.skip("Protocol members are not introspectable on this Python")
    assert set(B.WANTED) == set(declared)


def test_robotspec_itself_does_not_know_the_duck():
    """A generic walker must not INHERIT the duck's assets.

    For one draft these answers sat on `RobotSpec` behind an id check, which
    is the shape of the mistake the Body/RobotSpec split exists to avoid: a
    second walker would have been served the duck's meshes, the duck's
    shipped policies and the duck's env, and the failure would have arrived
    as a wrong picture and a wrong policy rather than as an error.
    """
    from microduck_local.robots import microduck as M
    from microduck_local.robots.spec import RobotSpec

    for duck_answer in ("visual_scene", "shipped_policies", "env_class",
                        "train_env_kwargs", "attach", "look", "setup_hint",
                        "fetch"):
        assert duck_answer in vars(M.MicroduckBody), duck_answer
        assert duck_answer not in vars(RobotSpec), duck_answer
    # What a walker CAN answer for itself stays generic.
    assert "ready" in vars(RobotSpec)
    assert not hasattr(RobotSpec, "REFERENCE_ID")
    # A bare RobotSpec therefore raises rather than answering as the duck.
    bare = RobotSpec(
        id="walker2", joint_names=("a",), default_pose=np.zeros(1, np.float32),
        obs_dim=7, base_body="b", gyro_sensor="g",
        foot_geoms={"left": ("l",), "right": ("r",)},
        scene_fn=lambda: Path("/nowhere/scene.xml"))
    assert bare.look() == "walker2"          # its own id, never "duck"
    assert bare.ready() is False             # no scene on disk
    for raises in ("visual_scene", "fetch"):
        with pytest.raises(NotImplementedError):
            getattr(bare, raises)()
    with pytest.raises(NotImplementedError):
        bare.env_class("walk")
    assert bare.shipped_policies() == ()
    assert bare.train_env_kwargs(None) == {}


def test_the_duck_is_a_microduckbody_and_the_g1_a_g1body():
    from microduck_local.robots.g1 import G1Body
    from microduck_local.robots.microduck import MicroduckBody

    assert isinstance(C.MICRODUCK, MicroduckBody)
    assert type(R.get("g1")._resolve()) is G1Body


def test_importing_the_g1_module_reads_nothing_from_the_cache():
    """`G1Body` moved to module level. The laziness that matters is that
    `load_config()` — which reads the fetched cache — is NOT called at import,
    so a machine that never ran the download can still import and train the
    duck. `load_config` is `lru_cache`d, so its cache_info is a direct count
    of whether anything touched it. Measured in a FRESH interpreter, because
    this one resolved the spec long ago."""
    code = (
        "import microduck_local.robots.g1 as g1\n"
        "ci = g1.load_config.cache_info()\n"
        "assert (ci.hits, ci.misses) == (0, 0), ci\n"
        "assert g1._LazyG1Spec._spec is None\n"
        "g1.load_config = lambda: 1 / 0\n"
        "try:\n"
        "    g1.G1_SPEC.id\n"
        "except ZeroDivisionError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('resolving the spec did not read the config')\n"
        "print('ok')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=Path(__file__).resolve().parents[1])
    assert p.returncode == 0, p.stderr
    assert "ok" in p.stdout


def test_the_g1_spec_pickles_by_name():
    """A vec-env worker ships specs. A class built inside a function is
    unpicklable — plain `pickle.dumps` failed with "Can't get local object
    '_g1_body_cls.<locals>.G1Body'" — which is why `G1Body` is module level."""
    import pickle

    from microduck_local.robots.g1 import G1Body

    assert pickle.loads(pickle.dumps(G1Body)) is G1Body
    revived = pickle.loads(pickle.dumps(R.get("g1")._resolve()))
    assert type(revived) is G1Body
    assert revived.id == "g1" and revived.obs_dim == 99


def test_the_nouns_are_what_a_sentence_calls_one():
    assert R.get("microduck").noun == "duck"
    assert R.get("g1").noun == "G1"


def test_the_looks_are_the_viewers_material_sets():
    """The duck's look is "duck", not its id: the viewer has painted `duck`
    since before a second body existed."""
    assert R.get("microduck").look() == "duck"
    assert R.get("g1").look() == "g1"


def test_ids_lists_a_known_body_whether_or_not_it_loads(monkeypatch):
    """`--robot g1` on a fresh checkout has always been ACCEPTED and then
    answered with the command that fetches it. An argparse "invalid choice"
    would be a worse answer, so the choices must not depend on the download —
    and `ids()` must therefore read the DECLARATION, not the loaded body.

    The three built-ins LEAD the list, in their declared order, and the
    assertion is a prefix rather than an equality because a fourth source
    exists now: a discovered body (`menagerie:<name>`, `robots/menagerie.py`)
    follows whatever is in this machine's cache. That is the opposite
    property to the one under test here — a discovered id is only listed once
    it HAS been downloaded — so it cannot be enumerated and must not be able
    to break this case.
    """
    monkeypatch.setattr(R, "_load_builtin", lambda b: None)
    assert R.ids()[:3] == ("microduck", "g1", "mars")
    assert all(":" in i for i in R.ids()[3:]), R.ids()
    assert T.parse_args(["--robot", "g1"]).robot == "g1"


def test_an_unfetched_g1_is_still_in_the_registry_and_says_so_via_ready(
        monkeypatch):
    """MEASURED, because the obvious assumption is wrong: `g1_ready()` False
    does NOT remove the G1 from `registry()`. Its `model_config.json` ships in
    the repo (`robots/unitree_g1/`) and `load_config()` falls back to it, so
    the spec resolves with no download at all — only the MJCF, the meshes and
    the walker ONNX are missing. "Is it in the registry" and "are its assets
    here" are two questions, which is why `Body.ready()` exists."""
    from microduck_local.robots import g1

    monkeypatch.setattr(g1, "g1_ready", lambda *a, **k: False)
    assert "g1" in R.registry()
    assert R.get("g1").ready() is False
    assert R.get("g1").obs_dim == 99


def test_an_unknown_id_is_refused_with_the_ids_listed():
    with pytest.raises(KeyError, match="unknown robot") as e:
        R.get("wombat")
    assert "microduck" in e.value.args[0]


def test_a_known_but_unloadable_body_names_its_fetch_command(monkeypatch):
    """The difference between "no such robot" and "you have not downloaded it
    yet" is the whole message. Driven by making the G1 fail to LOAD (a missing
    dependency, a corrupt config) rather than by `g1_ready`, for the reason the
    test above measures."""
    real_load = R._load_builtin
    monkeypatch.setattr(
        R, "_load_builtin",
        lambda b: None if b.id == "g1" else real_load(b))
    with pytest.raises(KeyError, match="unknown robot") as e:
        R.get("g1")
    assert "fetch-robot g1" in e.value.args[0]
    assert R.setup_hint("g1") == "uv run fetch-robot g1"
    # The duck has nothing to fetch, and must not be given a fake command.
    assert R.setup_hint("microduck") == ""
    with pytest.raises(SystemExit, match="fetch-robot g1"):
        T.env_class("g1")


def test_the_spec_module_aliases_still_work():
    """`viz_server`, `export_onnx`, `motion`, `pose`, `test_robot_spec` and
    `test_pose` all import these from `robots/spec.py`."""
    from microduck_local.robots import spec as spec_mod

    assert spec_mod.get("microduck") is C.MICRODUCK
    assert sorted(spec_mod.registry()) == sorted(R.registry())
    with pytest.raises(KeyError, match="unknown robot"):
        spec_mod.get("wombat")


# ----------------------------------------------- nothing moved: train.env_class

def test_env_class_still_returns_exactly_the_classes_it_used_to():
    """The if-chain's answers, compared against the classes themselves."""
    assert T.env_class("microduck") is MicroduckWalkEnv
    assert T.env_class("microduck", "walk") is MicroduckWalkEnv
    # The aliases the if-chain accepted, kept at the CLI edge.
    assert T.env_class(None) is MicroduckWalkEnv
    assert T.env_class("") is MicroduckWalkEnv
    assert T.env_class("duck") is MicroduckWalkEnv


@needs_g1
def test_env_class_still_returns_the_g1_envs_it_used_to():
    from microduck_local.robots.g1_env import G1SquatEnv, G1StandEnv, G1WalkEnv
    from microduck_local.robots.g1_imitate import G1ImitateEnv
    from microduck_local.robots.g1_karate import G1FrontKickEnv, G1PunchEnv

    assert T.env_class("g1") is G1WalkEnv
    assert T.env_class("g1", "walk") is G1WalkEnv
    assert T.env_class("g1", "stand") is G1StandEnv
    assert T.env_class("g1", "squat") is G1SquatEnv
    assert T.env_class("g1", "front_kick") is G1FrontKickEnv
    assert T.env_class("g1", "punch") is G1PunchEnv
    assert T.env_class("g1", "imitate") is G1ImitateEnv


def test_env_class_refuses_an_unknown_robot_at_the_cli():
    with pytest.raises(SystemExit, match="unknown --robot"):
        T.env_class("wombat")


def test_a_duck_task_is_still_the_category_error_it_was():
    with pytest.raises(SystemExit, match="train-behavior"):
        T.env_class("microduck", "stand")


def test_env_class_dispatches_to_a_registered_third_body(clean_registry):
    """The claim under test: a body the trainer has never heard of trains."""
    R.register(fake_body())
    assert T.env_class("wheelie", "roll") is _FakeEnv
    with pytest.raises(SystemExit, match="unknown --task"):
        T.env_class("wheelie", "walk")


# ------------------------------------------------------------- fetch-robot

def test_fetch_robot_calls_the_bodys_own_download(clean_registry, capsys):
    """One download path per body: `fetch-robot <id>`, the palette's ⤓ and
    `setup.sh` all end up in `Body.fetch`."""
    from microduck_local import fetch_robot

    R.register(fake_body())
    fetch_robot.main(["wheelie"])
    assert "/nowhere" in capsys.readouterr().out


def test_fetch_robot_with_no_body_lists_what_this_install_knows(capsys):
    from microduck_local import fetch_robot

    fetch_robot.main([])
    out = capsys.readouterr().out
    assert "microduck" in out and "g1" in out
    assert "uv run fetch-robot g1" in out


def test_fetch_robot_refuses_an_unknown_body_without_a_traceback(capsys):
    """An unknown id is a user error with a list, not a KeyError dump."""
    from microduck_local import fetch_robot

    with pytest.raises(SystemExit) as e:
        fetch_robot.main(["wombat"])
    assert e.value.code == 1
    assert "unknown robot" in capsys.readouterr().err


def test_fetch_g1_stays_an_alias_for_the_same_download():
    """The README, scripts/setup.sh and the memory index all say `fetch-g1`."""
    import tomllib

    from microduck_local import fetch_g1

    assert callable(fetch_g1.main)
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert scripts["fetch-g1"] == "microduck_local.fetch_g1:main"
    assert scripts["fetch-robot"] == "microduck_local.fetch_robot:main"
