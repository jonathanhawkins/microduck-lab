"""id -> `Body`: the one place that knows which robots exist.

Before this file the answer was written out by hand at 45 sites — 15 of them
in `viz_server.py` alone (`docs/mars-roadmap.md` §1 counts them). Every one
was an `if robot == "g1"`, which is a pattern that costs a full pass over the
tree per robot and silently forgets a site: the `/robots` endpoint still has
to hand-patch an absent G1 back into its own list because `registry()` had
dropped it.

Four ways a body gets in:

1. **Built in** — `_BUILTINS`, below. Declared as (id, module, attribute) so
   `ids()` can name a body whose assets are NOT fetched. That matters for the
   CLI: `--robot g1`'s choices must be the same on every machine, or the error
   for a missing download becomes argparse's "invalid choice" instead of the
   line that says which command fetches it.
2. **A pip-installed plugin** — a Python entry point in the group
   `microduck_local.bodies`, resolving to a `Body` instance or a zero-arg
   factory (`docs/mars-roadmap.md` §7.2 item 1). Somebody else's robot is then
   a package, not a fork of this repo.
3. **At runtime** — `register(body)`, for a test or a notebook.
4. **DISCOVERED in a cache** — `_NAMESPACES`, and today that is MuJoCo
   Menagerie's 71 models (`robots/menagerie.py`). The first three all require
   somebody to WRITE the body down; a catalogue cannot, because a table of 71
   entries is exactly the maintenance the generic seam exists to remove. So a
   namespace's bodies are whatever its cache holds, found by scanning it, and
   an id from a namespace that has never been fetched is still answered — by
   `get()` raising with `uv run fetch-robot menagerie:<name>`.

A plugin may NOT shadow a built-in id. The duck's 61-obs contract and the
G1's assets are what the goldens and the deployment path are measured against;
an installed package quietly replacing either would make every measurement in
this repo describe a different robot than the one named on the run.

Resolution failures are absences, never exceptions: importing the G1 on a
machine that never fetched it, or a broken plugin, leaves that body out of
`registry()` with a printed reason — exactly what `spec.registry()` did for
the G1 before this file existed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from functools import lru_cache

from .body import Body, conforms

#: The entry-point group a distribution advertises a body in:
#:
#:     [project.entry-points."microduck_local.bodies"]
#:     go2 = "my_robots.go2:BODY"
ENTRY_POINT_GROUP = "microduck_local.bodies"


@dataclass(frozen=True)
class _Builtin:
    """A body that ships with this repo, declared without importing it."""

    id: str
    module: str          # relative to `microduck_local.robots`
    attr: str
    setup_hint: str      # the command that fetches its assets ("" if none)


# Order is the order every `--robot` choices= and every menu shows. The duck
# is first because it is the default and the reference body.
_BUILTINS: tuple[_Builtin, ...] = (
    _Builtin(id="microduck", module="..contract", attr="MICRODUCK",
             setup_hint=""),
    _Builtin(id="g1", module=".g1", attr="G1_SPEC",
             setup_hint="uv run fetch-robot g1"),
    _Builtin(id="mars", module=".mars", attr="MARS",
             setup_hint="uv run fetch-robot mars"),
)

#: Id namespaces whose bodies are DISCOVERED from a cache: `{prefix: module}`,
#: the module relative to `microduck_local.robots`. A namespace module answers
#: three calls — `ids()` (cheap, no model compiled), `registry()` (the built
#: bodies, absences printed) and `fetch(name)` — and `fetch_robot.py` routes
#: `fetch-robot <prefix>:<name>` to the last of those, because at that moment
#: there is no body yet to ask for its own download.
#:
#: A namespaced id can never collide with a built-in or a plugin: the prefix
#: and the colon are not legal in the others, which is why the ids carry one.
_NAMESPACES: dict[str, str] = {"menagerie": ".menagerie"}


def namespace_of(body_id: str) -> str:
    """The discovery namespace `body_id` belongs to, or "".

    One definition of "this id is a catalogue's", read by `setup_hint`, by
    `get`'s error and by `fetch_robot.py`'s routing branch.
    """
    prefix = str(body_id).split(":", 1)[0]
    return prefix if ":" in str(body_id) and prefix in _NAMESPACES else ""


def namespace_module(prefix: str):
    """The module behind a namespace prefix. Raises `KeyError` if unknown."""
    return importlib.import_module(_NAMESPACES[prefix], __package__)


def _load_namespaces() -> dict[str, Body]:
    """Every body every namespace's cache holds right now.

    NOT memoised, unlike the entry-point scan: what is installed cannot change
    inside a process, but what is DOWNLOADED can — `fetch-robot` and the
    lab's ⤓ button both add a body mid-process, and a cached listing would
    hide it until a restart. The cost is a directory scan; the expensive part
    (compiling the model, measuring its stage pitch) is memoised inside the
    namespace module.

    MEASURED with a Go2 and an SO-ARM100 in the cache, in a FRESH interpreter
    each time (a warm one is dominated by the built-ins' own import and hides
    this), two samples each:

        eager   first registry()  629 / 672 ms   warm 0.19 ms   2 models built
        lazy    first registry()  334 / 351 ms   warm 0.65 ms   0 models built

    Building a discovered body compiles its MJCF twice and measures its stage
    pitch: **~155 ms per model**, so the eager version put 310 ms on
    whichever call happened to be first with two in the cache, and would put
    ~3.1 s there with twenty. The ~340 ms that remains is the BUILT-INS
    (importing `g1` and `mars`, now including `mars_env` and the `mars_reach`
    recipe, and resolving the G1's own lazy spec) and is unchanged by any of
    this — it is also why the floor moved from 225 ms when this was first
    measured: a built-in growing an env is exactly the cost a catalogue must
    not multiply. The warm call goes the other way by half a millisecond,
    which is the directory scan and two manifest reads that let a
    `fetch-robot` in another terminal show up without a restart.

    **The conformance gate had to move for that to be true, and this is the
    one subtlety worth stating.** `conforms()` sees through a lazy proxy
    (`robots/body.py`: it asks `hasattr`, which `__getattr__` answers, where
    `isinstance` would say no) — but *seeing through* it means RESOLVING it,
    so calling it here would have compiled every model and bought nothing. A
    proxy is recognised by `_resolve` on its CLASS, which is a lookup that
    does not go through `__getattr__`, and its conformance is checked on the
    real body inside `_resolve()` instead. A namespace that yields a plain
    body is still gated here exactly as before.
    """
    out: dict[str, Body] = {}
    for prefix, module in _NAMESPACES.items():
        try:
            found = namespace_module(prefix).registry()
        except Exception as exc:              # a broken cache / import error
            _report(f"[robots] namespace {prefix!r} unavailable: "
                    f"{type(exc).__name__}: {exc}")
            continue
        for body_id, body in found.items():
            if hasattr(type(body), "_resolve"):      # a lazy body: see above
                out[str(body_id)] = body
                continue
            missing = conforms(body)
            if missing:
                _report(f"[robots] {body_id} skipped: not a Body — missing "
                        f"{', '.join(missing)}")
                continue
            out[str(body_id)] = body
    return out


def _namespace_ids() -> tuple[str, ...]:
    """The ids every namespace's cache holds, without building any body."""
    out: list[str] = []
    for prefix in _NAMESPACES:
        try:
            out.extend(str(i) for i in namespace_module(prefix).ids())
        except Exception as exc:
            _report(f"[robots] namespace {prefix!r} could not be listed: "
                    f"{type(exc).__name__}: {exc}")
    return tuple(out)


# Bodies handed in by `register()`. Module state on purpose: a plugin
# registering at import time and a test registering in a fixture want the same
# door, and there is one registry per process either way.
_EXTRA: dict[str, Body] = {}

# Reasons already printed. `registry()` is called per roster change and per
# policy load, and `spec.registry()` was SILENT about an absent body — a line
# per call would be new noise in the lab's log for a condition that does not
# change within a process.
_REPORTED: set[str] = set()


def _report(msg: str) -> None:
    if msg not in _REPORTED:
        _REPORTED.add(msg)
        print(msg)


def _load_builtin(b: _Builtin) -> Body | None:
    """The body `b` names, or None with a printed reason.

    The G1's spec is lazy (`robots/g1._LazyG1Spec`): touching an attribute is
    what reads its config, so this is also where "the assets are not there"
    surfaces.
    """
    try:
        mod = importlib.import_module(b.module, __package__)
        body = getattr(mod, b.attr)
        _ = body.id                      # force a lazy spec to resolve
    except Exception as exc:             # assets missing / import error
        _report(f"[robots] {b.id} unavailable: {type(exc).__name__}: {exc}")
        return None
    return body


@lru_cache(maxsize=1)
def _load_entry_points() -> dict[str, Body]:
    """Every `microduck_local.bodies` entry point that resolves to a Body.

    Memoised per process. `entry_points()` walks the metadata of every
    installed distribution, and `registry()` runs per roster change and per
    policy load in the lab — a scan of site-packages on each of those is a
    cost nobody asked for. What is installed cannot change inside a process,
    so once is right.

    Callers must NOT mutate the returned dict; it is the cache. `registry()`
    and `ids()` only read it. A test that monkeypatches `entry_points` calls
    `_load_entry_points.cache_clear()` first — `tests/test_registry.py`'s
    `clean_registry` fixture does it on the way in and out.
    """
    out: dict[str, Body] = {}
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:             # a broken installed distribution
        _report(f"[robots] entry-point discovery failed: "
                f"{type(exc).__name__}: {exc}")
        return out
    builtin_ids = {b.id for b in _BUILTINS}
    for ep in eps:
        try:
            obj = ep.load()
            # A Body instance is a dataclass, not callable; a factory or a
            # class is. Both are allowed so a plugin can defer its own imports.
            body = obj() if callable(obj) else obj
            missing = conforms(body)
            if missing:
                raise TypeError(f"not a Body — missing {', '.join(missing)}")
            body_id = str(body.id)
        except Exception as exc:
            _report(f"[robots] plugin {ep.name!r} ({ep.value}) skipped: "
                    f"{type(exc).__name__}: {exc}")
            continue
        if body_id in builtin_ids:
            _report(f"[robots] plugin {ep.name!r} tried to replace the "
                    f"built-in {body_id!r} — ignored")
            continue
        out[body_id] = body
    return out


def register(body: Body) -> Body:
    """Add `body` to this process's registry, and return it.

    Refuses a built-in id for the reason in the module docstring. `_EXTRA` is
    read live by `registry()` and `ids()`, so a body registered here is
    visible to the very next call — it does not go through, and is not hidden
    by, the memoised entry-point scan.
    """
    missing = conforms(body)
    if missing:
        raise TypeError(
            f"{body!r} is not a Body — missing {', '.join(missing)} "
            "(robots/body.py lists the contract)")
    body_id = str(body.id)
    if any(b.id == body_id for b in _BUILTINS):
        raise ValueError(
            f"{body_id!r} is a built-in body and cannot be replaced — the "
            "goldens and the deployment contract are measured against it")
    _EXTRA[body_id] = body
    return body


def registry() -> dict[str, Body]:
    """Every body that is actually usable right now, by id.

    A body whose assets are missing or whose import failed is ABSENT, not
    broken — `spec.registry()`'s behaviour since the G1 arrived, and what lets
    a machine that never ran a download still train the duck.

    The dict is rebuilt per call so that `register()` shows up immediately,
    but nothing expensive happens twice: the entry-point scan is memoised
    (`_load_entry_points`) and a lazy spec's config read caches itself.
    """
    out: dict[str, Body] = {}
    for b in _BUILTINS:
        body = _load_builtin(b)
        if body is not None:
            out[b.id] = body
    for body_id, body in _load_entry_points().items():
        out.setdefault(body_id, body)
    for body_id, body in _load_namespaces().items():
        out.setdefault(body_id, body)
    for body_id, body in _EXTRA.items():
        out[body_id] = body
    return out


def ids() -> tuple[str, ...]:
    """Every body id this install KNOWS, fetched or not.

    This is what a `--robot` flag offers. It deliberately includes a built-in
    whose assets are absent: `--robot g1` on a fresh checkout has always been
    accepted and then answered with the command that fetches it, and an
    argparse "invalid choice" would be a worse answer.

    The BUILT-INS come first and in their declared order, because that order
    is what every menu and every `choices=` shows and the duck is the default.
    A discovered body (`menagerie:<name>`) follows: it is only in this list at
    all once its assets are in the cache, so unlike a built-in it never names
    a download that has not happened.
    """
    out = [b.id for b in _BUILTINS]
    for body_id in (list(_load_entry_points()) + list(_namespace_ids())
                    + list(_EXTRA)):
        if body_id not in out:
            out.append(body_id)
    return tuple(out)


def setup_hint(body_id: str) -> str:
    """The command that would make `body_id` available, or "".

    Answerable for a body that is NOT loadable, which is the only time it is
    needed — so it reads the declaration, not the body.

    A namespaced id is answerable even though nothing knows whether that
    model EXISTS: `menagerie:wombat` is offered `uv run fetch-robot
    menagerie:wombat`, which then fails with Menagerie's own "no model
    directory 'wombat'". That is the right division — a catalogue of 71 models
    cannot be enumerated here without a network call, and "try the download"
    is a better answer than "no such robot" for the 71 that are real.
    """
    for b in _BUILTINS:
        if b.id == body_id:
            return b.setup_hint
    prefix = namespace_of(body_id)
    if prefix:
        try:
            return str(namespace_module(prefix).setup_hint(
                str(body_id).split(":", 1)[1]))
        except Exception:
            return f"uv run fetch-robot {body_id}"
    body = _EXTRA.get(body_id)
    if body is not None:
        try:
            return str(body.setup_hint())
        except Exception:
            return ""
    return ""


def get(body_id: str) -> Body:
    """The body `body_id` names.

    Raises `KeyError` naming the setup command when the id is one this install
    knows but cannot load — the difference between "no such robot" and "you
    have not downloaded it yet" is the whole message.
    """
    reg = registry()
    if body_id in reg:
        return reg[body_id]
    hint = setup_hint(body_id)
    raise KeyError(
        f"unknown robot {body_id!r} — have {sorted(reg)}"
        + (f"; `{hint}` adds it" if hint else ""))
