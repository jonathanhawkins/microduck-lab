"""Behavior library — one module per trick family, split from the original
flat behaviors.py (3.4k lines) with ZERO surface change: every name the flat
module exposed (private helpers included — the tests import them) is
re-exported here, so `from microduck_local.behaviors import X` is untouched.

Layout: core.py (dataclasses, registry, catalog, shared reward helpers) →
poses / headstand / backflip / airflip / imitate / locomotion (each registers
its behaviors on import, same order as the flat file) → env.py (BehaviorEnv).
"""
import builtins as _builtins
import importlib as _importlib

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


# Names this module copied out of the submodules last time. Tracked so a
# reload can REMOVE what the submodules no longer define instead of leaving
# it behind (see reload_library).
_FLATTENED: set[str] = set()

# This package's OWN machinery. A submodule that happens to define one of
# these names must not overwrite it: clobbering _SUBMODULES (say) replaces the
# module list with a trick's constant, and the next reload_library() dies in
# vars() and stays dead for the life of the process — while the trainer
# subprocess, importing fresh, works fine. The submodules are trick recipes
# people are invited to edit, so the collision is a question of when.
# Computed, not hand-listed: everything bound above this line IS the
# machinery, by construction, so adding a submodule or a helper can never
# leave a gap for a trick constant to slip through.
# Filled in at the bottom of this module, once every name it must protect
# exists. Computing it here would capture only the names bound ABOVE, leaving
# any helper defined below silently unreserved — the gap this set exists to
# close. Builtins are included because _flatten and reload_library resolve
# `any`, `vars`, `open`, `compile`, `delattr`, `enumerate`, `zip` through THESE
# globals: a submodule defining `any = 1` would be flattened over the builtin,
# and reload_library would then die inside its own machinery and stay dead even
# after the offending line is removed.
_RESERVED: frozenset[str] = frozenset()


def _flatten():
    g = globals()
    for _k in _FLATTENED:
        g.pop(_k, None)
    _FLATTENED.clear()
    for _m in _SUBMODULES:
        for _k, _v in vars(_m).items():
            if not _k.startswith("__") and _k not in _RESERVED:
                g[_k] = _v
                _FLATTENED.add(_k)


def reload_library():
    """Hot-reload every submodule from disk, in registration order.

    The lab's /teach used importlib.reload on the flat module to pick up
    recipe edits without a server restart; reloading a PACKAGE only re-runs
    __init__ and would keep every submodule stale — so viz_server calls this
    instead.

    Two properties the naive version lacked, both of which let the lab serve
    a recipe the trainer subprocess (which always imports fresh) cannot run:

    1. NO GHOSTS. `importlib.reload` re-executes a module in its EXISTING
       dict, so a name deleted or renamed in the source survives — and each
       module's `__all__ = dir()` footer then re-exports the ghost down the
       whole star-import cascade. Every submodule namespace is emptied first,
       and `_flatten` drops package names the reload no longer produced.
    2. NO TORN STATE. Every source is compiled BEFORE anything is mutated, so
       the common failure (a syntax error in a recipe edit) leaves the whole
       library untouched — the property the single flat module had for free.
       A module that raises while EXECUTING has its previous namespace put
       back before the error is re-raised, so the library keeps serving what
       it served before and the caller can report the failure.

    Called via the `importlib` module (not a name bound at import) so tests
    that monkeypatch `importlib.reload` still neutralize it."""
    for m in _SUBMODULES:
        path = getattr(m, "__file__", None)
        if path:
            with open(path, encoding="utf-8") as f:
                compile(f.read(), path, "exec")  # SyntaxError mutates nothing
    # Snapshot EVERY module before touching any of them. Restoring only the
    # module that failed is not enough: core is reloaded first and rebinds
    # BEHAVIORS/CATALOG to fresh dicts, so if module i raises, modules i..n
    # never re-register into those new dicts — leaving `BEHAVIORS` full (the
    # package re-flattens an older module's view of it) while core's own dict,
    # which every helper closes over, holds only what modules 0..i-1
    # registered. `match_behavior` then answers "I don't know that trick" for
    # tricks the catalog still lists.
    snapshots = [{k: v for k, v in vars(m).items() if not k.startswith("__")}
                 for m in _SUBMODULES]
    try:
        for i, m in enumerate(_SUBMODULES):
            for k in snapshots[i]:
                delattr(m, k)
            new = _importlib.reload(m)
            # A no-op reload (tests monkeypatch importlib.reload to keep
            # classes stable across /teach calls) must not gut the module.
            if not any(not k.startswith("__") for k in vars(new)):
                vars(new).update(snapshots[i])
            _SUBMODULES[i] = new
    except BaseException:
        # Put the WHOLE library back the way it was, then let the caller
        # report the failure. Restoring in order matters: core's dicts must be
        # the pre-reload objects again before the modules that registered into
        # them are restored.
        for m, snap in zip(_SUBMODULES, snapshots):
            for k in [k for k in vars(m) if not k.startswith("__")]:
                delattr(m, k)
            vars(m).update(snap)
        _flatten()
        raise
    _flatten()


# Everything bound above this line is the package's own machinery, plus the
# builtins the machinery calls — reserved by construction, so adding a helper
# can never leave a gap.
_RESERVED = frozenset(globals()) | frozenset(dir(_builtins))
_flatten()
