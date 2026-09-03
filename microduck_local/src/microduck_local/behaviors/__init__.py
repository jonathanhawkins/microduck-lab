"""Behavior library — one module per trick family, split from the original
flat behaviors.py (3.4k lines) with ZERO surface change: every name the flat
module exposed (private helpers included — the tests import them) is
re-exported here, so `from microduck_local.behaviors import X` is untouched.

Layout: core.py (dataclasses, registry, catalog, shared reward helpers) →
poses / headstand / backflip / airflip / imitate / locomotion (each registers
its behaviors on import, same order as the flat file) → env.py (BehaviorEnv).
"""
import builtins as _builtins

# Unused by name, but it is the seam the rollback regression test neutralizes
# reload through (`monkeypatch.setattr(B._importlib, "reload", ...)`) — the
# same module object `motion.reload_modules` resolves `reload` on.
import importlib as _importlib  # noqa: F401

from ..motion import reload_modules as _reload_modules
from . import airflip as _airflip
from . import backflip as _backflip
from . import core as _core
from . import env as _envm
from . import headstand as _headstand
from . import imitate as _imitate
from . import locomotion as _locomotion
from . import poses as _poses

_SUBMODULES = [_core, _poses, _headstand, _backflip, _airflip, _imitate,
               _locomotion, _envm]

# The BARE submodule names. `from . import core as _core` binds `_core` here
# AND pins `core` on the package (the import system does that for every
# submodule), so these land in _RESERVED next to the real machinery even
# though they are not machinery at all. Named separately so a collision with
# them can be reported as what it is — see _flatten.
_SUBMODULE_NAMES = frozenset(m.__name__.rpartition(".")[2] for m in _SUBMODULES)


# Names this module copied out of the submodules last time. Tracked so a
# reload can REMOVE what the submodules no longer define instead of leaving
# it behind (see reload_library).
_FLATTENED: set[str] = set()

# This package's OWN machinery, plus the bare submodule names the import
# system pins alongside it. A submodule that defines one of these must not
# overwrite it: clobbering _SUBMODULES (say) replaces the module list with a
# trick's constant, and the next reload_library() dies in vars() and stays
# dead for the life of the process — while the trainer subprocess, importing
# fresh, works fine. The submodules are trick recipes people are invited to
# edit, so the collision is a question of when. _flatten REPORTS such a name
# rather than skipping it; skipping was silent, and a silent drop of a
# legitimate recipe constant is the expensive half of this bug.
# Computed, not hand-listed: everything bound above this line IS the
# machinery, by construction, so adding a submodule or a helper can never
# leave a gap for a trick constant to slip through.
# Filled in at the bottom of this module, once every name it must protect
# exists. Computing it here would capture only the names bound ABOVE, leaving
# any helper defined below silently unreserved — the gap this set exists to
# close. Builtins are included because _flatten resolves `vars`, `globals`
# and `hasattr` through THESE globals: a submodule defining `vars = 1` would
# be flattened over the builtin, and _flatten would then die inside its own
# machinery and stay dead even after the offending line is removed.
_RESERVED: frozenset[str] = frozenset()


def _flatten():
    """Re-copy every submodule's namespace onto the package.

    A reserved name is REPORTED, not skipped. Skipping was silent, and the
    reserved set is not only machinery: it contains the eight bare submodule
    names, and `env`, `core` and `locomotion` are all plausible module-level
    names in a trick recipe. A recipe author who added `env = ...` to
    headstand.py had it dropped with no error and no log, and
    `from microduck_local.behaviors import env` then handed back the SUBMODULE
    — a wrong value that looks like a working import.

    Nothing is mutated until the whole scan is clean, so a clash leaves the
    package exactly as it was and reload_library can roll the modules back
    around it.
    """
    fresh: dict[str, object] = {}
    clashes: list[str] = []
    g = globals()
    for _m in _SUBMODULES:
        for _k, _v in vars(_m).items():
            if _k.startswith("__"):
                continue
            if _k in _RESERVED:
                # Re-exporting the package's own object under its own name
                # (a submodule doing `from . import env`) is a no-op, not a
                # clash — only a DIFFERENT value would be shadowed.
                if _v is not g.get(_k):
                    kind = ("submodule" if _k in _SUBMODULE_NAMES
                            else "builtin" if hasattr(_builtins, _k)
                            else "package machinery")
                    clashes.append(f"{_m.__name__}.{_k} ({kind} {_k!r})")
                continue
            fresh[_k] = _v
    if clashes:
        raise NameError(
            "behaviors recipe names collide with names the package already "
            "owns: " + "; ".join(clashes) + ". Rename them in the recipe: "
            "flattening them would overwrite the package's own machinery, and "
            "dropping them (what this used to do, in silence) hands "
            "`from microduck_local.behaviors import <name>` the wrong object.")
    for _k in _FLATTENED:
        g.pop(_k, None)
    _FLATTENED.clear()
    g.update(fresh)
    _FLATTENED.update(fresh.keys())


def reload_library():
    """Hot-reload every submodule from disk, in registration order.

    The lab's /teach used importlib.reload on the flat module to pick up
    recipe edits without a server restart; reloading a PACKAGE only re-runs
    __init__ and would keep every submodule stale — so viz_server calls this
    instead.

    The compile-first / snapshot-all / restore-all machinery this needs is
    `motion.reload_modules` — the lab reloads motion the same way and for the
    same reasons, so there is one implementation, not two that drift.

    Re-flattening rides INSIDE that protection (`after=_flatten`) because the
    reload is not finished until the package namespace matches the fresh
    modules. If `_flatten` rejects the edit (a recipe name colliding with the
    package's own) and the modules stayed reloaded, the package would keep the
    OLD BEHAVIORS while core's dict — which every helper closes over — holds
    the new one: `match_behavior` denying tricks the catalog still lists, the
    exact split brain the all-module rollback exists to prevent."""
    try:
        _reload_modules(_SUBMODULES, after=_flatten)
    except BaseException:
        # The modules are back; re-sync the package namespace to them, then
        # let the caller report the failure.
        _flatten()
        raise


# Everything bound above this line is the package's own machinery, plus the
# builtins the machinery calls — reserved by construction, so adding a helper
# can never leave a gap.
_RESERVED = frozenset(globals()) | frozenset(dir(_builtins))
_flatten()
