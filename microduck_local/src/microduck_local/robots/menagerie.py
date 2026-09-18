"""MuJoCo Menagerie as a catalogue: `fetch-robot menagerie:<name>`.

    uv run fetch-robot menagerie:unitree_go2     # a Go2 on the stage
    uv run fetch-robot menagerie:trs_so_arm100   # and a 6-DoF arm

[Menagerie](https://github.com/google-deepmind/mujoco_menagerie) holds 71
robot models — humanoids, quadrupeds, arms, hands, mobile manipulators — each
with a `home` keyframe and its own licence. `docs/mars-roadmap.md` §7 names it
as the test bench for "is this harness generic", with the settling number: a
Menagerie body reaches level 0 (on the stage, posable, in a room) with **zero
lines of its own code in this repo**. This file is the catalogue that makes
that true, and it contains no robot's name: `robots/mjcf_body.py` reads
whatever arrives, and what arrives is decided by the directory listing.

**Nothing is vendored.** The repo is 588 MB and its models are under
per-model licences (Apache, BSD, and several manufacturers' own terms — the
Go2's is Unitree's, not Apache), so a download is the only honest form. The
LICENCE file travels WITH the model into the cache and `fetch-robot` prints
its first line, because a licence that stayed on GitHub is a licence nobody
read.

**Discovered, not declared.** There is no `_BUILTINS` entry per model and
there could not be: 71 entries would be a table this repo has to maintain, and
the point is that it maintains none. So a fetched model is found by its
manifest — `registry()` scans `.cache/menagerie/*/body.json` — and an id this
install has never downloaded is still ANSWERED, by `registry.get` raising with
the command that fetches it (`robots/registry.py`'s namespace branch).

**The integrity story, in two halves.** The listing from GitHub's contents API
carries each file's git blob sha1, which is what `_verify_blob` checks a fresh
download against: it is the only thing that can tell the pinned tree apart
from a moved tag, a truncated transfer or a captive portal's sign-in page. The
manifest then records our own sha256 per file, which is what makes a SECOND
fetch free (a present file that hashes right is skipped) and what repairs a
file corrupted after it arrived. `MENAGERIE_SHA` is pinned for the same reason
`mars.INNATE_OS_SHA` is: a revision is a different robot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]  # microduck_local/

#: mujoco_menagerie at this revision (2026-09-18). Everything a manifest
#: records was read from this tree; a different one is a different robot.
MENAGERIE_SHA = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
REPO = "google-deepmind/mujoco_menagerie"
API_BASE = f"https://api.github.com/repos/{REPO}/contents"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{MENAGERIE_SHA}"

#: Where `fetch()` puts a model. `MICRODUCK_MENAGERIE_DIR` moves the cache the
#: way `MICRODUCK_G1_DIR` and `MICRODUCK_MARS_DIR` do, so a worktree can
#: verify without touching the main checkout's download.
CACHE_DIR = Path(os.environ.get("MICRODUCK_MENAGERIE_DIR")
                 or _ROOT / ".cache" / "menagerie")

#: The id namespace. `menagerie:unitree_go2` — a namespace rather than a bare
#: name because these ids are DISCOVERED, and a bare `unitree_go2` could
#: collide with a built-in or a pip-installed plugin's body, which is the one
#: thing `robots/registry.py` refuses outright.
NAMESPACE = "menagerie"

#: The manifest file, inside the model's own cache directory.
MANIFEST = "body.json"

DOWNLOAD_TIMEOUT_S = 120.0

#: Top-level files a model needs: its MJCFs, its licence and its notes.
#: Everything under `assets/` comes too, whatever the extension — that is
#: where meshes and textures live and a filter there would lose a body's
#: skin.
#:
#: An ALLOW-list and not a deny-list, MEASURED: a first draft carried both,
#: and a planted regression that turned the deny-list off changed nothing —
#: it was unreachable, because Menagerie's rendered previews (`go2.png`,
#: 1.9 MB; `go2_mjx.png`, 0.5 MB) are already excluded by not being on the
#: allow-list, and a TEXTURE is already included by living under `assets/`.
#: Two rules where one decides is a rule nobody can test.
_WANTED_SUFFIXES = (".xml", ".md", ".txt")
_WANTED_NAMES = ("LICENSE", "LICENCE", "COPYING")

#: `<include file="go2.xml"/>` — how a Menagerie scene names its robot. This
#: is how the robot MJCF is identified rather than by guessing `<dir>.xml`:
#: `trs_so_arm100/` holds `so_arm100.xml`, so the directory name is not the
#: model name, and several models ship an `_mjx` variant beside the real one.
_INCLUDE_RE = re.compile(r"""<include\s+file\s*=\s*["']([^"']+)["']""")


# ---------------------------------------------------------------- the cache


def cache_dir() -> Path:
    return CACHE_DIR


#: A Menagerie model directory's name: letters, digits, `_` and `-`. The
#: whole catalogue matches it.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def check_name(name: str) -> str:
    """`name` if it is a model directory's name, else a `ValueError`.

    The name reaches TWO places that must not be handed a path: a URL
    (`_listing`, `RAW_BASE/<name>/...`) and a directory under the cache. It
    arrives from a command line, so `menagerie:../../../etc` is a thing
    somebody can type — and the failure without this check is either a
    confusing listing error or a write outside the cache. Refusing by shape
    is cheaper than sanitising, and every one of Menagerie's 71 directories
    passes.
    """
    if not _NAME_RE.match(str(name)):
        raise ValueError(
            f"{name!r} is not a Menagerie model name — they are the "
            "directory names in github.com/google-deepmind/mujoco_menagerie "
            "(letters, digits, '_' and '-'), e.g. unitree_go2")
    return str(name)


def model_dir(name: str, dest: Path | None = None) -> Path:
    return (dest or CACHE_DIR) / check_name(name)


def manifest_path(name: str, dest: Path | None = None) -> Path:
    return model_dir(name, dest) / MANIFEST


def read_manifest(name: str, dest: Path | None = None) -> dict[str, Any] | None:
    """The manifest for `name`, or None if it has never been fetched.

    None rather than an exception, and also None for a manifest that will not
    parse: `registry()` calls this per scan and a half-written or hand-edited
    file must read as "not fetched" rather than take the whole roster down
    (`robots/registry.py`: resolution failures are absences).
    """
    p = manifest_path(name, dest)
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def ready(name: str, dest: Path | None = None) -> bool:
    """Is `name` fetched — a manifest, and every file it lists, present?

    Presence, not sha256, for `robots/mars.mars_ready`'s reason: this is asked
    per roster change and per policy load in the lab, and re-hashing 30 MB of
    meshes on each of those is a cost nobody asked for. `fetch()` verifies a
    file before it is put in place, so a file that is HERE was verified when
    it arrived — and the next fetch repairs one that was corrupted after.
    """
    man = read_manifest(name, dest)
    if not man:
        return False
    d = model_dir(name, dest)
    files = man.get("files") or ()
    return bool(files) and all((d / rel).is_file() for rel, _sha in files)


def names(dest: Path | None = None) -> tuple[str, ...]:
    """Every model fetched into the cache, sorted.

    A directory scan per call and deliberately not memoised: `fetch-robot`
    and the lab's ⤓ both add one inside a running process, and a cached
    listing would hide it until a restart. What IS memoised is the expensive
    part — `body()` compiles the model once (`_BODIES`).
    """
    root = dest or CACHE_DIR
    if not root.is_dir():
        return ()
    return tuple(sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / MANIFEST).is_file()))


# -------------------------------------------------------------- downloading


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    """git's own content hash: `sha1("blob <len>\\0" + data)`.

    Computed rather than trusted so the LISTING can verify the DOWNLOAD. The
    contents API reports this per file at the pinned revision, and the raw
    host serves the bytes; checking one against the other is what makes the
    pin mean something.
    """
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def _get(url: str) -> bytes:
    """One HTTP GET. The seam `tests/test_menagerie.py` replaces.

    Separated from everything else on purpose: the listing, the filter, the
    hash refusal, the atomic replace, the manifest and the idempotence are
    then all testable with no network at all — `robots/mars._download`'s
    pattern, widened to cover the API listing as well as the file bytes,
    because here the listing is part of the logic rather than a constant.

    `Accept` names the API version (GitHub's own advice) and a token is used
    when the environment has one, which is what keeps a CI run off the 60
    requests/hour unauthenticated limit. Two requests per model is not close
    to it, but a `setup.sh` that iterated the catalogue would be.
    """
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "microduck-local/fetch-robot",
    })
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith(API_BASE):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r:
        return r.read()


def _listing(path: str) -> list[dict[str, Any]]:
    """The contents API on one directory at the pinned sha, as a list."""
    url = f"{API_BASE}/{path}?ref={MENAGERIE_SHA}"
    data = json.loads(_get(url).decode())
    if not isinstance(data, list):
        raise RuntimeError(
            f"{url} did not list a directory — Menagerie has no model "
            f"directory {path!r} at {MENAGERIE_SHA[:12]}")
    return data


def _wanted(rel: str) -> bool:
    """Is this file part of the MODEL, rather than documentation of it?

    Everything under `assets/` is in (meshes, textures); at the top level it
    is the MJCFs, the licence and the notes — which leaves out the preview
    render Menagerie keeps beside every model, and that is 2.4 MB off a Go2.
    """
    p = Path(rel)
    if p.parts[:1] == ("assets",):
        return True
    return (p.suffix.lower() in _WANTED_SUFFIXES
            or p.stem.upper() in _WANTED_NAMES)


def _walk(name: str) -> list[tuple[str, str, int]]:
    """Every wanted file in the model as `(relative path, blob sha1, bytes)`.

    Recursive, but only into directories that survive `_wanted` — which is
    `assets/` and whatever a model nests inside it. Two API calls for a
    typical model, which is why the listing is not cached to disk: it is the
    cheap half of a download measured in tens of megabytes.

    A model that kept its meshes in a directory NOT called `assets` would
    lose them here, and that failure is loud in the right place: the download
    succeeds, and `MjcfBody`'s compile then says `Error opening file` and
    names the mesh. Menagerie is uniform on `assets/` across its 71 models,
    so the alternative — descending every directory — would buy nothing and
    cost a listing per subtree.
    """
    out: list[tuple[str, str, int]] = []
    todo = [str(name)]
    while todo:
        path = todo.pop(0)
        for entry in _listing(path):
            rel = str(entry["path"])[len(str(name)) + 1:]
            if entry["type"] == "dir":
                if _wanted(rel + "/x"):
                    todo.append(str(entry["path"]))
                continue
            if entry["type"] != "file" or not _wanted(rel):
                continue
            out.append((rel, str(entry["sha"]), int(entry["size"])))
    return sorted(out)


def _verify_blob(rel: str, data: bytes, want_sha1: str) -> None:
    """Refuse bytes that are not the pinned revision's.

    The one check that can tell a moved revision, a truncated transfer and a
    captive-portal HTML page apart from the robot — `robots/mars.fetch`'s
    sha256 refusal, against a hash the SERVER published rather than one this
    file pins, because a catalogue of 71 models cannot pin them all.
    """
    got = _git_blob_sha1(data)
    if got != want_sha1:
        raise RuntimeError(
            f"{rel} from Menagerie {MENAGERIE_SHA[:12]} hashes {got[:12]}, "
            f"the listing says {want_sha1[:12]} — refusing it")


def _pick_xmls(d: Path, files: list[str]) -> tuple[str, str | None]:
    """`(robot xml, scene xml or None)` for a fetched model. Data-driven.

    The robot MJCF is the file a SCENE includes, read out of the scene. Two
    guesses were rejected for being wrong on real models: `<dir>.xml` (the
    arm's directory is `trs_so_arm100` and its model is `so_arm100.xml`) and
    "the only non-scene xml" (several models ship an `_mjx` variant beside
    the real one, and the Go2 does).

    A model with no scene falls back to the single top-level non-scene MJCF,
    and `MjcfBody` then generates a scene for it. More than one candidate and
    no scene to disambiguate is a RuntimeError naming them, because picking
    one would be picking a robot.
    """
    tops = [f for f in files if "/" not in f and f.lower().endswith(".xml")]
    scenes = sorted(f for f in tops if Path(f).stem.lower().startswith("scene"))
    plain = sorted(f for f in tops if f not in scenes)
    for scene in scenes:
        if "mjx" in Path(scene).stem.lower():
            continue
        for inc in _INCLUDE_RE.findall((d / scene).read_text()):
            if inc in plain:
                return inc, scene
    if len(plain) == 1:
        return plain[0], (scenes[0] if scenes else None)
    raise RuntimeError(
        f"cannot tell which of {plain} is the robot in {d.name}: no scene "
        "includes any of them. Fetch it by hand and build the body with "
        "robots/mjcf_body.MjcfBody.from_mjcf(<the robot xml>, id=...)")


def _licence_line(d: Path, files: list[str]) -> str:
    """The licence's first non-empty line — it differs per model.

    Printed by `fetch-robot` and recorded in the manifest. The Go2's first
    line is Unitree's copyright, the arm's is "Apache License"; a single
    "Apache-2.0" written into this file would be a claim about 71 models that
    is false for some of them.
    """
    for f in files:
        if "/" in f or Path(f).stem.upper() not in _WANTED_NAMES:
            continue
        for line in (d / f).read_text(errors="replace").splitlines():
            if line.strip():
                return line.strip()
    return "unstated — read the model's own LICENSE"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic, for the rule the repo learned the hard way: a spawned worker
    can read a half-written file, and there is no safe window
    (`AGENTS.md`, "Atomic writes and live imports")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def fetch(name: str, dest: Path | None = None) -> Path:
    """Download the model `name` into the cache. Returns its directory.

    Idempotent: a file that is present and hashes to what the manifest
    recorded is skipped, so a second run costs one listing and no bytes. Every
    download lands on a temp name in the target directory and is
    `os.replace`d into place only after its git blob sha1 matches the pinned
    listing — the repo's atomic-write rule, and here it also means a
    half-written mesh can never be imported by a parallel worker.
    """
    name = str(name)
    root = dest or CACHE_DIR
    d = model_dir(name, root)
    entries = _walk(name)
    if not entries:
        raise RuntimeError(
            f"Menagerie {MENAGERIE_SHA[:12]} has no usable files under "
            f"{name!r} — check the model directory's name")
    known = {rel: sha for rel, sha in (read_manifest(name, root) or {}).get(
        "files", ())}
    fresh = bytes_new = 0
    recorded: list[list[str]] = []
    for rel, blob_sha1, _size in entries:
        out = d / rel
        want256 = known.get(rel)
        if out.is_file() and want256 and _sha256(out) == want256:
            recorded.append([rel, want256])
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{os.getpid()}.part")
        print(f"[menagerie] {name}/{rel}")
        try:
            data = _get(f"{RAW_BASE}/{name}/{rel}")
            _verify_blob(rel, data, blob_sha1)
            tmp.write_bytes(data)
            recorded.append([rel, _sha256(tmp)])
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        fresh += 1
        bytes_new += len(data)
    files = [f[0] for f in recorded]
    robot_xml, scene_xml = _pick_xmls(d, files)
    _write_json(manifest_path(name, root), {
        "name": name,
        "sha": MENAGERIE_SHA,
        "files": recorded,
        "robot_xml": robot_xml,
        "scene_xml": scene_xml,
        "license": _licence_line(d, files),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _BODIES.pop(name, None)
    total = sum(e[2] for e in entries)
    print(f"[menagerie] {name}: {fresh} file(s) downloaded "
          f"({bytes_new / 1e6:.1f} MB), {len(entries) - fresh} already "
          f"verified — {d} ({total / 1e6:.1f} MB total)")
    print(f"[menagerie] {name} licence: {_licence_line(d, files)}")
    return d


def setup_hint(name: str) -> str:
    return f"uv run fetch-robot {NAMESPACE}:{name}"


# ----------------------------------------------------------------- the body

#: Constructed bodies, by name. `body()` compiles the model (the Go2's 16
#: meshes cost ~0.3 s) and MEASURES its stage pitch, and `registry()` is
#: called per roster change and per policy load in the lab — so the
#: construction happens once and the manifest's mtime is the cache key, which
#: is what lets a re-fetch inside one process be seen.
_BODIES: dict[str, tuple[int, Any]] = {}


def body(name: str, dest: Path | None = None):
    """The `MjcfBody` for a fetched model. Raises if it is not fetched.

    Nothing here is per-robot: the manifest says which file is the robot and
    which is the scene, and `MjcfBody.from_mjcf` reads the rest off the
    compiled model. The `title` handed over is the directory name tidied up
    (`unitree_go2` -> "Unitree Go2"), and the licence rides along in `extra`
    so it reaches a palette tooltip rather than staying in the cache.

    The shipped `scene.xml` is used AS IT IS when it works — it lives in the
    cache beside `assets/`, so its relative `<include>` and its `meshdir`
    resolve with no rewriting, and it is the scene the model's author meant.
    A model whose scene will not serve (no scene at all, a scene MuJoCo
    refuses, a keyframe the two files disagree about) falls back to a
    generated one, and the body records which it got in
    `extra["scene_source"]`.
    """
    from .body import conforms
    from .mjcf_body import MjcfBody

    name = str(name)
    man = read_manifest(name, dest)
    if not man:
        raise FileNotFoundError(
            f"Menagerie model {name!r} is not fetched — run "
            f"`{setup_hint(name)}`")
    d = model_dir(name, dest)
    stamp = manifest_path(name, dest).stat().st_mtime_ns
    hit = _BODIES.get(name)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    robot = d / str(man["robot_xml"])
    scene = man.get("scene_xml")
    manifest = {"title": None, "license": man.get("license", ""),
                "menagerie_sha": man.get("sha", ""),
                "fetched_at": man.get("fetched_at", "")}
    manifest = {k: v for k, v in manifest.items() if v}
    body_id = f"{NAMESPACE}:{name}"
    made = None
    if scene:
        try:
            made = MjcfBody.from_mjcf(robot, id=body_id, scene_xml=d / scene,
                                      manifest=manifest, cache_dir=d)
        except Exception as exc:
            print(f"[menagerie] {name}: the shipped {scene} did not serve "
                  f"({type(exc).__name__}: {exc}) — generating one")
    if made is None:
        made = MjcfBody.from_mjcf(robot, id=body_id, manifest=manifest,
                                  cache_dir=d)
    # The conformance gate, moved here from `registry._load_namespaces`: it
    # asks `hasattr` for every `Body` name, which a lazy proxy answers by
    # RESOLVING — so running it at discovery would have compiled every model
    # and undone the laziness. Here it costs nothing (the body is already
    # built) and still refuses to hand out something that is not a Body.
    missing = conforms(made)
    if missing:
        raise TypeError(
            f"{body_id} is not a Body — missing {', '.join(missing)} "
            "(robots/body.py lists the contract)")
    _BODIES[name] = (stamp, made)
    return made


class LazyBody:
    """A fetched model that has not been compiled yet.

    `robots/registry.registry()` runs per roster change and per policy load in
    the lab, and building an `MjcfBody` COMPILES its MJCF twice (the robot and
    the scene) and measures its stage pitch. MEASURED at **~155 ms per
    model**, in a fresh interpreter: two in the cache put 310 ms on the first
    `registry()` of a process (650 ms against ~340 ms of built-ins), and
    twenty would put ~3.1 s there. A catalogue is meant to be cheap to HAVE,
    so what discovery yields is this — and `body()` runs when somebody asks
    this model a question only the model can answer.

    `robots/g1._LazyG1Spec` is the pattern and the reason it is a proxy rather
    than a subclass: the real thing is a frozen dataclass the conformance
    suite calls `dataclasses.replace` on, so it must not be wrapped, only
    deferred. `_resolve()` is named the same as the G1's because
    `tests/test_body_conformance._resolved()` already looks for it.

    **The five cheap fields are REAL attributes**, set from the manifest and
    the name: `id`, `title`, `noun`, `kind` and `default_task` are what a
    palette chip, a `--robot` listing, an `if kind ==` and the lab's roster
    builder ask, and none of them needs a compile. Everything else goes
    through `__getattr__` and resolves.
    """

    __slots__ = ("id", "title", "noun", "kind", "default_task", "_name",
                 "_dest")

    def __init__(self, name: str, dest: Path | None,
                 manifest: Mapping[str, Any]):
        from .mjcf_body import title_of

        self._name = str(name)
        self._dest = dest
        self.id = f"{NAMESPACE}:{self._name}"
        self.title = str(manifest.get("title") or title_of(self.id))
        self.noun = self.title
        # Every body read from an MJCF is level 0 (`robots/mjcf_body.py`), and
        # that is a property of HOW it was read, not of the model — so it is
        # knowable without opening the model, and the conformance suite's
        # kind rosters are built without compiling anything.
        self.kind = "generic"
        # Same reasoning: a body read from an MJCF has no env for any task
        # (`MjcfBody.env_class` raises), so the lab's roster builder can know
        # to idle a slot for one kinematically without opening the model.
        self.default_task = None

    def _resolve(self):
        return body(self._name, self._dest)

    def __getattr__(self, item):
        if item.startswith("__"):              # never resolve for dunders
            raise AttributeError(item)
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return (f"LazyBody(id={self.id!r}, "
                f"{'built' if self._name in _BODIES else 'not built yet'})")


def registry(dest: Path | None = None) -> dict[str, Any]:
    """Every fetched model as `{"menagerie:<name>": LazyBody}`.

    What `robots/registry.registry()` merges in, and nothing here compiles a
    model — see `LazyBody`.

    A model whose MANIFEST is unusable is an ABSENCE with a printed reason,
    which is the registry's own rule and covers the realistic failure: a
    half-finished download, a hand-edited or truncated `body.json`, files that
    were deleted from under it. What is NOT checked here is whether the MJCF
    compiles, because checking that is the ~155 ms this function exists to
    avoid — that one surfaces at first use, as a `ValueError` naming the file
    (`mjcf_body.compile_robot`). The split is deliberate: "is this model
    here" is cheap and is answered now; "is this model good" costs a compile
    and is answered when somebody asks the model something.
    """
    out: dict[str, Any] = {}
    for name in names(dest):
        man = read_manifest(name, dest)
        if not man or not ready(name, dest):
            print(f"[menagerie] {name} unavailable: its manifest is "
                  "unreadable or its files are missing — re-run "
                  f"`{setup_hint(name)}`")
            continue
        robot = model_dir(name, dest) / str(man.get("robot_xml") or "")
        if not robot.is_file():
            print(f"[menagerie] {name} unavailable: the manifest names "
                  f"{man.get('robot_xml')!r} and it is not there")
            continue
        out[f"{NAMESPACE}:{name}"] = LazyBody(name, dest, man)
    return out


def ids(dest: Path | None = None) -> tuple[str, ...]:
    """The ids of every fetched model, without building any of them.

    Separate from `registry()` because `--robot`'s choices must not pay for a
    model compile, the way `robots/registry.ids()` reads the built-in
    DECLARATION rather than the loaded body.
    """
    return tuple(f"{NAMESPACE}:{n}" for n in names(dest))


__all__ = ["CACHE_DIR", "MANIFEST", "MENAGERIE_SHA", "NAMESPACE", "LazyBody",
           "body", "cache_dir", "check_name", "fetch", "ids", "manifest_path",
           "model_dir", "names", "read_manifest", "ready", "registry",
           "setup_hint"]
