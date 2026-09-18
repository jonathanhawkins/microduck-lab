"""Every robot answer the lab server gives, read off the registry.

`docs/mars-roadmap.md` §1 counted 45 hand-written `if robot == "g1"` sites,
15 of them in `viz_server.py`. Phase 1a moved the bodies behind
`robots/registry.py`; this file is the other half of §6.4 — the lab's robot
HTTP surface, extracted so that **adding a body never means opening the
4,500-line server**. What lives here:

* the palette's robot switch (`available_robots`) and the 🎬/🎓 editor list
  (`robot_entries`), both one entry per `registry.ids()`;
* the one-click download (`start_fetch` / `fetch_state`), per id, in a thread;
* `GET /scene?robot=<id>`'s body (`robot_scene`) and the body-name list the
  pose stream is indexed against (`scene_body_names`);
* the shipped-policy groups every body declares (`shipped_entries`,
  `shipped_groups`);
* which recipe a run practised (`run_trick`) and what the 🎓 panel offers
  (`teach_suggestions`);
* the refusal that used to be four copies of a width compare
  (`policy_refusal`, §6.3);
* and `KinematicIdle`, the "env" a level-0 body's roster slot runs.

It also owns the two things `robots/microduck.py` had to import OUT of
`viz_server` while this file did not exist (both were marked `PHASE 1B:`
there): `POLICIES_DIR` and `extract_scene`. The import direction is now the
right way round — a body reaches into the lab's own package, not into the
server module.

**Lazy bodies are never resolved by a LISTING.** A discovery namespace
(`robots/menagerie.py`) yields a proxy whose `id`, `title`, `noun`, `kind`
and `default_task` are real attributes and whose everything-else compiles the
model on first touch — ~155 ms, measured there. A listing runs per roster
change and per policy poll, so it must not pay that: `_is_lazy` recognises a
proxy by `_resolve` on its CLASS (a lookup that does not go through
`__getattr__`, which is `robots/registry.py`'s own idiom) and answers from
what the proxy carries. That is not a guess — a namespace only lists models
that are IN the cache, so `ready()` is true by construction, and a level-0
body has no walker fields and no env whatever its MJCF turns out to contain.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from .. import behaviors as behaviors_mod
from ..robots import registry as _registry
from ..robots.policy_contract import DEFAULT_ROBOT, recorded, resolve

#: Pollen's shipped reference policies, downloaded by `scripts/setup.sh` into
#: the sibling `microduck/` clone. Here rather than in `viz_server` because
#: `robots/microduck.py` reads it to list them and a body must not import the
#: server (`docs/mars-roadmap.md` §6.4).
POLICIES_DIR = Path(__file__).resolve().parents[4] / "microduck" / "policies"


# --------------------------------------------------------------- lazy bodies

def _is_lazy(body: Any) -> bool:
    """Is `body` a DISCOVERY NAMESPACE's un-built proxy?

    Asked of the ID, through `registry.namespace_of` — the one definition of
    "this id is a catalogue's" — and NOT by probing for `_resolve`. Two
    reasons, and the second one is a bug this went through:

    * probing the instance (`hasattr(body, "_resolve")`) goes through
      `__getattr__` and compiles the very model the question exists to avoid
      compiling;
    * probing the TYPE catches too much. `robots/g1._LazyG1Spec` is a proxy
      too, and it is a full `RobotSpec` behind a config read that
      `registry()` has already paid for — treating it as "do not touch"
      answered `animate: false` for the G1 and took the humanoid out of the
      🎬 editor it has always been in.
    """
    return bool(_registry.namespace_of(str(getattr(body, "id", ""))))


def _ready(body: Any) -> bool:
    """`body.ready()`, without resolving a proxy.

    A namespace's registry lists a model only once its files are in the cache
    (`robots/menagerie.registry` checks the manifest and the robot XML), so a
    proxy that EXISTS is ready and asking it costs a compile for an answer
    already known. A body whose `ready()` raises is treated as not ready:
    the palette's job is to offer the download, not to 500.
    """
    if _is_lazy(body):
        return True
    try:
        return bool(body.ready())
    except Exception:
        return False


def _animates(body: Any) -> bool:
    """Can the 🎬 pose editor open this body?

    `PoseScratch` is a WALKER's tool — effectors, soles, a base body — so the
    editor's list has to be filtered by CAPABILITY rather than by id: with a
    third body merely present, `pose.pose_scratch("mars")` raised
    `AttributeError: 'MarsBody' object has no attribute 'base_body'`
    (`docs/mars-roadmap.md` §2a). The capability is exactly "this body
    carries the walker's fields", which is what `RobotSpec` adds to `Body`,
    and `base_body` is the first name the scratch asks for.
    """
    return not _is_lazy(body) and hasattr(body, "base_body")


def _num_joints(body: Any) -> int:
    """`body.num_joints` for the editor, 0 for a body it cannot open.

    A proxy is not resolved for a number the panel will not use: a level-0
    body is never `animate`, and the editor is the only reader.
    """
    if _is_lazy(body):
        return 0
    try:
        return int(body.num_joints)
    except Exception:
        return 0


# ----------------------------------------------------------------- the nouns

def robot_noun(robot: str) -> str:
    """What a SENTENCE calls this body: "teach the {noun} a trick".

    `GET /policies` and `GET /robots` both send it, because the palette and
    the 🎓 panel each used to spell the same button their own way ("🤖 g1"
    beside "🤖 G1"). A body this install cannot load falls back to its id
    rather than to a table: `registry.ids()` lists a known-but-unfetched
    built-in, and the id is the only thing known about it.
    """
    body = _registry.registry().get(robot)
    if body is None:
        return robot
    return str(getattr(body, "noun", "") or getattr(body, "title", "") or robot)


def robot_title(robot: str) -> str:
    """The body's display title, or its id when it is not loadable."""
    body = _registry.registry().get(robot)
    if body is None:
        return robot
    return str(getattr(body, "title", "") or robot)


def robot_kind(robot: str) -> str:
    """"legged" | "wheeled" | "generic" — what SHAPE of robot this is."""
    body = _registry.registry().get(robot)
    return str(getattr(body, "kind", "") or "legged") if body is not None else "legged"


def available_animate(robot: str) -> bool:
    """Can the 🎬 pose editor open `robot`? — the `animate` flag, by id.

    The same question `available_robots()` answers per entry, reachable for
    one body so the editor's own endpoints can refuse rather than 500
    (`viz_server.scratch_for`).
    """
    body = _registry.registry().get(robot)
    return body is not None and _animates(body)


def default_task(robot: str) -> str | None:
    """Which `env_class` task a roster slot for `robot` runs by default.

    `None` means the body has no env at all — see `KinematicIdle`.
    """
    body = _registry.registry().get(robot)
    if body is None:
        return "walk"
    return getattr(body, "default_task", "walk")


# --------------------------------------------------------------- the 🎓 panel

def body_tasks(robot: str) -> tuple:
    """The `Behavior` recipes for `robot`, through the body that owns them.

    `Body.tasks()` is the declared seam and `BodyBase.tasks()` is
    `behaviors.for_robot(self.id)`, so for every body that exists this is the
    same answer the registry's own contract gives — reached through the body
    so that a plugin which ships its own recipes is listed too.

    A lazy proxy is answered `()` WITHOUT resolving: a level-0 body's
    `tasks()` is `()` by construction (`robots/mjcf_body.MjcfBody.tasks`
    says so in its own docstring), and compiling a Go2 to be told it has no
    recipes is 155 ms for a constant.
    """
    body = _registry.registry().get(robot)
    if body is None or _is_lazy(body):
        return ()
    try:
        return tuple(body.tasks())
    except Exception:
        return ()


def teach_suggestions(robot: str = DEFAULT_ROBOT) -> list[dict]:
    """The 🎓 panel's suggestion chips for `robot`.

    Every recipe of that body that names a `suggest` phrase, in registry
    order. A new robot's chips come from its own recipes — the viewer holds
    no list of its own, which is why a body with nothing to teach gets no
    chip at all rather than the duck's.
    """
    return [{"text": b.suggest, "behavior": b.id, "emoji": b.emoji,
             "title": b.title}
            for b in body_tasks(robot) if getattr(b, "suggest", "")]


def imitation_behavior(robot: str = DEFAULT_ROBOT):
    """The recipe that tracks a saved clip on `robot`: the duck's `imitate`,
    another body's `<robot>_imitate` task. None if that body has none."""
    bid = "imitate" if robot == DEFAULT_ROBOT else f"{robot}_imitate"
    return behaviors_mod.BEHAVIORS.get(bid)


def run_trick(run: Path, robot: str) -> str | None:
    """The RECIPE a run practised, as a behaviors id.

    What the palette groups "Our runs" by, and what a 🎓 suggestion chip
    looks its best run up under.

    Two trainers write two files: a duck trick run carries `behavior.json`
    (its `behavior` IS the id), a walk/task run carries `run.json` (a `task`,
    resolved through the recipes OF THAT BODY — g1 + "front_kick" is
    `g1_front_kick`, mars + "pick" is `mars_pick`). An imitation run whose
    CLIP is named after a recipe of the same robot is a run of that trick —
    the G1's measured-best front kick is `g1_imitate` tracking the
    "g1-front-kick" clip, and filing it under "copy the animation" would hide
    the pick from the trick it answers. None when neither file says.

    The recipe lookup goes through `body_tasks`, so a body's tasks are the
    body's — this used to call `behaviors.for_robot` directly, which is the
    same set today only because every body's `tasks()` delegates there.
    """
    import json

    beh_id = clip = None
    try:
        beh = json.loads((run / "behavior.json").read_text())
        if isinstance(beh, dict):
            beh_id, clip = beh.get("behavior"), beh.get("clip")
    except (OSError, ValueError):
        pass
    if not beh_id:
        try:
            rj = json.loads((run / "run.json").read_text())
        except (OSError, ValueError):
            return None
        task = rj.get("task") if isinstance(rj, dict) else None
        if not task:
            return None
        kw = rj.get("env_kwargs")
        clip = kw.get("clip_name") if isinstance(kw, dict) else None
        beh_id = next((b.id for b in body_tasks(robot)
                       if b.trainer and b.task == task), str(task))
    if clip:
        named = behaviors_mod.BEHAVIORS.get(str(clip).replace("-", "_"))
        if named is not None and named.robot == robot:
            return named.id
    return str(beh_id)


# ------------------------------------------------------------- the downloads

# One-click asset downloads (POST /robots/{id}/fetch), PER BODY: the G1's
# ~140 MB clone and MARS's 7.2 MB of STLs run in a thread so the 50 Hz loop
# keeps streaming, and the palette polls /policies to see them land. One at a
# time per robot — a second click on the same body is a no-op, while two
# different bodies may download at once.
_FETCH: dict[str, dict] = {}
_FETCH_LOCK = threading.Lock()


def fetch_state(robot: str) -> dict:
    """`{"state": "idle"|"fetching"|"error", "error": str}` for one body."""
    with _FETCH_LOCK:
        return dict(_FETCH.get(robot) or {"state": "idle", "error": ""})


def start_fetch(robot: str) -> dict:
    """Begin downloading `robot`'s assets. Returns at once.

    A namespaced id (`menagerie:unitree_go2`) routes to the NAMESPACE's own
    `fetch(name)` rather than to a body, because at this moment there is no
    body to ask: nothing is in the cache, so `registry.get` would raise.
    `fetch_robot.py`'s CLI has exactly the same branch, for the same reason.

    Raises `KeyError` for an id this install does not know at all — the
    caller turns that into a 404.
    """
    prefix = _registry.namespace_of(robot)
    if not prefix and robot not in _registry.ids():
        raise KeyError(f"nothing to fetch for robot {robot!r}")
    with _FETCH_LOCK:
        if (_FETCH.get(robot) or {}).get("state") == "fetching":
            return dict(_FETCH[robot])
        _FETCH[robot] = {"state": "fetching", "error": ""}

    def work() -> None:
        try:
            if prefix:
                _registry.namespace_module(prefix).fetch(
                    robot.split(":", 1)[1])
            else:
                _registry.get(robot).fetch()
            # `shipped_policies()` is lru_cached per body and was read while
            # the cache dir was empty — without this the palette lists no G1
            # policies until the lab restarts.
            _clear_shipped_cache(robot)
            with _FETCH_LOCK:
                _FETCH[robot] = {"state": "idle", "error": ""}
        except Exception as exc:                    # network, git, disk
            print(f"[lab] {robot} fetch failed: {type(exc).__name__}: {exc}",
                  flush=True)
            with _FETCH_LOCK:
                _FETCH[robot] = {"state": "error",
                                 "error": f"{type(exc).__name__}: {exc}"}

    threading.Thread(target=work, name=f"fetch-{robot}", daemon=True).start()
    return fetch_state(robot)


def _clear_shipped_cache(robot: str) -> None:
    """Drop whatever memo a body keeps of its own shipped drop.

    Generic on purpose: the G1's `shipped_policies()` is an `lru_cache` at
    module scope, and a body that memoises differently (or not at all) must
    not need a line here. Everything is best-effort — a body that cannot be
    imported yet simply has nothing cached.
    """
    try:
        body = _registry.registry().get(robot)
    except Exception:
        return
    fn = getattr(type(body), "shipped_policies", None)
    for target in (fn, getattr(body, "shipped_policies", None)):
        clear = getattr(target, "cache_clear", None)
        if callable(clear):
            clear()
    # The module-level cache the G1's body delegates to.
    module = getattr(type(body), "__module__", "")
    if module:
        import sys
        mod = sys.modules.get(module)
        clear = getattr(getattr(mod, "shipped_policies", None),
                        "cache_clear", None)
        if callable(clear):
            clear()


# ---------------------------------------------------------------- the switch

def available_robots() -> list[dict]:
    """The bodies this lab can load, for the palette's robot switch.

    One entry per `registry.ids()`, which deliberately includes a built-in
    whose assets are ABSENT: a switch that silently dropped the G1 would read
    as "no G1 runs", where `ready: false` plus `setup` reads as "press ⤓".

    * `id` / `label` / `noun` — the wire name, the chip and the sentence.
    * `kind` — "legged" | "wheeled" | "generic", so the viewer can pick an
      emoji and a verb without a table of ids.
    * `ready` — are the assets on disk (`Body.ready`).
    * `setup` — the command that fixes it (`registry.setup_hint`), absent
      when there is nothing to run.
    * `fetching` / `fetchError` — this body's own download state. PER ID
      since there are three downloadable bodies: one shared flag showed a
      MARS download as a G1 one.
    * `animate` — can the 🎬 pose editor open it (`_animates`).
    """
    reg = _registry.registry()
    out: list[dict] = []
    for rid in _registry.ids():
        body = reg.get(rid)
        ready = body is not None and _ready(body)
        state = fetch_state(rid)
        entry = {
            "id": rid,
            "label": robot_title(rid) if body is not None else rid,
            "noun": robot_noun(rid),
            "kind": robot_kind(rid),
            "ready": ready,
            "animate": body is not None and _animates(body),
        }
        if not ready:
            hint = _registry.setup_hint(rid)
            entry.update(setup=hint, fetching=state["state"] == "fetching",
                         fetchError=state["error"])
        out.append(entry)
    return out


def robot_entries() -> list[dict]:
    """`GET /robots`: every body, for the 🎬 editor and the 🎓 panel.

    `available_robots()` plus the two things only those panels want — the
    joint count and the teach suggestions. It replaced a loop over
    `spec.registry()` that answered `"ready": True if rid == "microduck" else
    bool(g1_ready())`, so a third body inherited the G1's download state, and
    then hand-patched an absent G1 back onto the end of its own list.
    """
    return [{**e, "title": e["label"], "numJoints": _num_joints(
        _registry.registry().get(e["id"])),
        "teach": teach_suggestions(e["id"])}
        for e in available_robots()]


# ----------------------------------------------------------------- the scene

def extract_scene() -> dict:
    """Visual geometry for Three.js, straight from the compiled DUCK model.

    (jenga-stacker's extract_visual_scene, deduplicated by mesh id.) Lives
    here rather than in `viz_server` so `robots/microduck.MicroduckBody`
    can call it without importing the lab server — the `PHASE 1B:` note on
    that method.

    NOT `from_xml_path(SCENE_WALK_XML)`: the viewer draws every duck from
    this one scene, and the world stream indexes its body list positionally,
    so this model has to carry the hinged `mouth` body too.
    """
    import mujoco

    from ..world.compose import MOUTH_GROUP, scene_model

    m = scene_model()
    mesh_ids: dict[int, int] = {}
    meshes: list[dict] = []
    geoms: list[dict] = []
    for i in range(m.ngeom):
        # Group 2 is where the upstream export puts the visual shells; the
        # hinged bill sits in MOUTH_GROUP so no range sensor sees it, and the
        # viewer has to ask for it by name.
        if (m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH
                or m.geom_group[i] not in (2, MOUTH_GROUP)):
            continue
        mid = int(m.geom_dataid[i])
        if mid not in mesh_ids:
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            mesh_ids[mid] = len(meshes)
            meshes.append({
                "v": np.round(m.mesh_vert[va:va + vn], 4).reshape(-1).tolist(),
                "f": m.mesh_face[fa:fa + fn].reshape(-1).tolist(),
            })
        # Material name + rgba ride along so the viewer can paint per-part
        # colors (eye ring, mouth, shells) instead of guessing per body.
        mat_id = int(m.geom_matid[i])
        rgba = m.mat_rgba[mat_id] if mat_id >= 0 else m.geom_rgba[i]
        geoms.append({
            "mesh": mesh_ids[mid],
            "body": int(m.geom_bodyid[i]),
            "pos": [round(float(x), 5) for x in m.geom_pos[i]],
            "quat": [round(float(x), 5) for x in m.geom_quat[i]],  # wxyz
            "mat": m.material(mat_id).name if mat_id >= 0 else "",
            "rgba": [round(float(x), 4) for x in rgba],
        })
    return {
        "bodies": [m.body(b).name for b in range(m.nbody)],
        "meshes": meshes,
        "geoms": geoms,
    }


def robot_scene(robot: str) -> dict:
    """`GET /scene?robot=<id>`'s payload — the body's own mesh dump.

    Every body answers `visual_scene()`, including one discovered in a
    Menagerie cache, so this is a registry lookup and a `ready()` gate rather
    than a branch per robot. `FileNotFoundError` carries the setup hint,
    which is what the 404 shows; `KeyError` means no such body at all.
    """
    body = _registry.registry().get(robot)
    if body is None:
        hint = _registry.setup_hint(robot)
        if hint:
            raise FileNotFoundError(
                f"{robot} assets missing — run `{hint}`")
        raise KeyError(f"unknown robot {robot!r}")
    if not _ready(body):
        hint = _registry.setup_hint(robot) or f"uv run fetch-robot {robot}"
        raise FileNotFoundError(f"{robot} assets missing — run `{hint}`")
    return body.visual_scene()


def scene_body_names(robot: str) -> tuple[str, ...]:
    """The body list `GET /scene?robot=<id>` serves, in its order.

    The viewer zips streamed poses against this positionally, so the lab
    resolves its own model's bodies BY NAME against it
    (`viz_server._robot_scene_to_env`).
    """
    return tuple(str(n) for n in robot_scene(robot)["bodies"])


# -------------------------------------------------------- shipped policies

def shipped_entries() -> list[dict]:
    """Every body's shipped drop, as palette entries.

    Was two hand-written blocks in `discover_policies` (a `pollen` glob and a
    `g1` try/except). A body that ships nothing contributes nothing and its
    group disappears from the palette, which is the normal case — MARS ships
    no policy this lab can run, because Innate's learned skills are ACT
    checkpoints and `innate-os` vendors none.

    A body whose drop cannot be read is an ABSENCE with a printed reason, not
    an exception: the palette must still list everything else.
    """
    out: list[dict] = []
    for rid, body in _registry.registry().items():
        if _is_lazy(body):
            continue                 # level 0 ships nothing; do not compile it
        try:
            entries = tuple(body.shipped_policies())
        except Exception as exc:     # assets missing / unreadable
            print(f"[lab] no {rid} policies in the palette: "
                  f"{type(exc).__name__}: {exc}")
            continue
        out.extend(dict(e) for e in entries)
    return out


def shipped_groups() -> list[dict]:
    """The palette's shipped SECTIONS: `{key, title, robot}`, in registry order.

    The viewer used to hold this list itself — `{ key: "g1", title: "Unitree
    G1 (shipped)" }` beside `pollen` — so a third body's section would have
    been a fourth literal in a TypeScript file. The key is the one each
    body's own entries carry (the duck's drop is POLLEN's, not ours, and the
    group is named for whoever shipped it), and the title rides on the entry
    beside it so that this function needs no table of ids either.
    """
    seen: dict[str, dict] = {}
    for e in shipped_entries():
        key = str(e.get("group") or e.get("robot") or "")
        if not key or key in seen:
            continue
        robot = str(e.get("robot") or "")
        seen[key] = {"key": key,
                     "title": str(e.get("groupTitle")
                                  or f"{robot_title(robot)} (shipped)"),
                     "robot": robot}
    return list(seen.values())


# ------------------------------------------------------- the policy refusal

def policy_robot(path: str | Path | None) -> str:
    """Which body a policy drives — `policy_contract.resolve(...).robot`.

    The palette shows every robot's runs; a chip that landed on the wrong
    body would load happily and behave like noise (99 obs read as 61). This
    was `export_onnx.run_robot(dir)`, which is the same call one rung down;
    naming the FILE lets a stamped `.onnx` answer for itself even when it has
    been moved away from its run (`robots/policy_contract.resolve`, rung 1).

    Never raises: `resolve` raises `KeyError` when a run names a body this
    install cannot load, and this is called once per run per palette poll —
    one unfetchable run must not empty the whole list.
    """
    if not path:
        return DEFAULT_ROBOT
    try:
        return resolve(Path(path)).robot
    except Exception as exc:
        print(f"[lab] {path}: {type(exc).__name__}: {exc} — listing it as a "
              f"{DEFAULT_ROBOT}")
        return DEFAULT_ROBOT


def policy_refusal(path: str | Path | None, robot: str,
                   want_obs: int | None, have_obs: int) -> str | None:
    """Why this policy must not be stepped on this body — None to go ahead.

    §6.3's four width guards (`do_assign`, `do_spawn_helper`, `do_spawn_duck`,
    `apply_snapshot`), as ONE comparison plus the last-ditch check. Two rungs,
    and the order is the whole point:

    1. **The contract, when the policy has one.** `policy_contract.recorded`
       is the SELF-DESCRIBING rungs only — the ONNX's own `metadata_props`,
       then `run.json`'s `"contract"` — never the two fallbacks that infer a
       contract from a body NAME or from the duck being the only body there
       used to be. Those fallbacks are what make `resolve` unusable here: a
       shipped G1 `walker.onnx` has neither metadata nor a run directory, so
       `resolve` would call it a duck and this would refuse a legitimate
       assign. A file that SAYS what it is gets judged on what it says.
    2. **The graph's width**, kept exactly as it was, as the last-ditch check
       on the FILE (§6.3's own words). It is what catches a policy with
       nothing to say, and what catches a file whose metadata lies about it.

    Width alone was a proxy: two bodies with the same observation width would
    have crossed silently, which the arrival of a 32-float body makes a
    question of when rather than whether.
    """
    want = None
    try:
        want = recorded(Path(path)) if path else None
    except Exception:
        want = None                  # unreadable protobuf / truncated json
    if want is not None:
        try:
            mine = _registry.get(robot).contract()
        except Exception:
            mine = None              # a body that declares none: fall through
        if mine is not None and not mine.matches(want):
            return (f"speaks {want.id}, a {robot} speaks {mine.id}")
    if want_obs is not None and int(want_obs) != int(have_obs):
        return f"wants {int(want_obs)} obs, a {robot} produces {int(have_obs)}"
    return None


# ----------------------------------------------------------- the level-0 env

class KinematicIdle:
    """A lab slot for a body with NO env: its model, held at its keyframe.

    Level 0 of `docs/mars-roadmap.md` §7.1 is "an MJCF and an id" — the
    palette chip, the meshes, a slot on the stage. There is no env, because
    choosing one would mean choosing a policy convention nobody declared
    (`robots/mjcf_body.MjcfBody.env_class` raises and names the level that
    lands it), and handing the slot another robot's env is the failure this
    whole split exists to prevent: it would arrive as a wrong picture and a
    wrong policy rather than as an error.

    So the slot IDLES. `step()` advances nothing — deliberately, and it is
    the honest answer rather than a shortcut: a Go2 stepped at its `home`
    keyframe with no controller folds onto the floor in about a second
    (`tests/test_body_conformance.py` measures exactly that for the duck and
    the G1, and holds them with their shipped idle instead), so a physics
    step here would draw every stranger's robot as a heap. What a level-0
    body promises is that you can SEE it, posed as its author keyframed it.

    It answers the small surface `viz_server.Duck` asks of an env — `model`,
    `data`, `observation_space`, `step_count`, `reset`, `step` — and
    deliberately does NOT answer `twist_cmd` or `heading_lin_vel`: the lab
    reads those behind `hasattr`, so a body with no drive channel is
    non-steerable and reports no speed instead of inventing a zero.
    """

    def __init__(self, body: Any, seed: int | None = None):
        import gymnasium as gym
        import mujoco

        self.body = body
        self.robot = str(body.id)
        self.model = mujoco.MjModel.from_xml_path(str(body.scene_fn()))
        self.data = mujoco.MjData(self.model)
        n = int(body.obs_dim)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (n,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (int(body.num_actions),), dtype=np.float32)
        self._obs = np.zeros(n, dtype=np.float32)
        self.step_count = 0
        self.reset(seed=seed)

    def _pose(self) -> None:
        import mujoco

        key = getattr(self.body, "stand_keyframe", "") or ""
        kid = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
               if key else -1)
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            # A scene whose keyframe is named something else: qpos0 is still
            # a pose, and a body that reaches here has already passed the
            # conformance suite's "its keyframe is in its scene" case.
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._pose()
        self.step_count = 0
        return self._obs, {}

    def step(self, action: Any):
        self.step_count += 1
        return self._obs, 0.0, False, False, {}


def slot_env(robot: str, seed: int, kwargs: dict | None = None):
    """The env one ROSTER SLOT of `robot` runs, or a `KinematicIdle`.

    The three answers, and each is the body's own:

    * a body with a `default_task` builds `env_class(task)` — the duck's
      `MicroduckWalkEnv`, the G1's `G1WalkEnv`, MARS's `MarsArmEnv` /
      `MarsPickEnv`. `task` comes from the caller when a teach job named one
      (the trainee mirrors what the trainer is practising) and from the body
      otherwise, because "walk" is not a question MARS can be asked.
    * a body with `default_task is None` has no env and idles.

    `viz_server.Duck._make_env` keeps the DUCK's own branch (behaviour envs,
    the shared-model scope, BAM) — that is the reference body's plumbing, not
    a generic seam, and moving it here would have been the "abstract the
    duck" trap of §7.3.
    """
    body = _registry.get(robot)
    kw = dict(kwargs or {})
    task = kw.pop("task", None) or getattr(body, "default_task", "walk")
    if task is None:
        # Every remaining kwarg describes an env this body does not have —
        # an actuator model, a randomiser, an episode length. DROPPED rather
        # than passed, because there is nothing here that could honour them
        # and a TypeError inside `_make_env` would be a slot that cannot be
        # created at all.
        return KinematicIdle(body, seed=seed)
    return body.env_class(task)(**kw)


__all__ = ["POLICIES_DIR", "KinematicIdle", "available_animate",
           "available_robots", "body_tasks",
           "default_task", "extract_scene", "fetch_state",
           "imitation_behavior", "policy_refusal", "policy_robot",
           "robot_entries", "robot_kind", "robot_noun", "robot_scene",
           "robot_title", "run_trick", "scene_body_names", "shipped_entries",
           "shipped_groups", "slot_env", "start_fetch", "teach_suggestions"]
