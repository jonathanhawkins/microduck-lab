"""The Menagerie catalogue: the download, the manifest, and the discovery.

Network: none, except in the two cases that skip unless a model is already in
the cache. `menagerie._get` is the seam — it serves both the contents-API
listing and the raw bytes, so a fake one covers the listing, the file filter,
the hash refusal, the atomic replace, the manifest, the idempotence and the
robot/scene detection with nothing on the wire. `robots/mars.py`'s
`_download` is the same pattern; what is widened here is that the LISTING is
part of the logic rather than a pinned constant, because a catalogue of 71
models cannot pin 71 manifests.

The fake model is a hand-written two-joint MJCF and a `scene.xml` that
includes it, laid out exactly as Menagerie lays a model out (`<name>/`,
`assets/`, `LICENSE`, a preview `.png` that must NOT be downloaded). That is
the whole point of testing it this way: the cases below are about the
CATALOGUE's rules, and a real 28 MB Go2 would only make them slow.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct

import pytest

from microduck_local.robots import menagerie as M
from microduck_local.robots import registry as R

# --------------------------------------------------------------- the fixture

_ROBOT_XML = b"""<mujoco model="fake">
  <compiler angle="radian" meshdir="assets"/>
  <asset><mesh name="blob" file="blob.stl"/></asset>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <geom name="shell" type="mesh" mesh="blob" contype="0" conaffinity="0"
            group="2" rgba="0.8 0.4 0.1 1"/>
      <geom name="hull" type="box" size="0.05 0.05 0.05" group="3"/>
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <body name="tip" pos="0 0 0.1">
        <geom name="tip_g" type="box" size="0.02 0.02 0.02" group="3"/>
        <joint name="j2" type="slide" axis="0 0 1" range="-0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="0.25 0.01"/></keyframe>
</mujoco>
"""

_SCENE_XML = b"""<mujoco model="fake scene">
  <include file="fake.xml"/>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane"/>
  </worldbody>
</mujoco>
"""

_LICENSE = b"Copyright (c) 2026 Somebody Else\nAll rights reserved.\n"


def _tetra_stl() -> bytes:
    """A tetrahedron as binary STL — the smallest mesh MuJoCo accepts.

    Both constraints are MEASURED: an ASCII STL is refused by the decoder,
    and a single triangle by "at least 4 vertices required".
    """
    v = [(0.0, 0.0, 0.0), (0.06, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, 0.0, 0.06)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    body = b"".join(struct.pack("<12fH", 0.0, 0.0, 0.0,
                                *v[a], *v[b], *v[c], 0)
                    for a, b, c in faces)
    return (b"fake".ljust(80, b"\0") + struct.pack("<I", len(faces)) + body)


def _model_files(name: str = "fake") -> dict[str, bytes]:
    """What the fake model's directory holds, `<name>/<rel>` -> bytes.

    Includes the two things a filter has to get right: a 1.9 MB-shaped
    preview render at the top level (Menagerie puts one beside every model)
    which must be skipped, and a mesh under `assets/` which must not be.
    """
    return {
        f"{name}/fake.xml": _ROBOT_XML,
        f"{name}/scene.xml": _SCENE_XML,
        f"{name}/LICENSE": _LICENSE,
        f"{name}/README.md": b"# fake\nA model for a test.\n",
        f"{name}/fake.png": b"\x89PNG\r\n\x1a\n" + b"\0" * 64,
        f"{name}/assets/blob.stl": _tetra_stl(),
    }


def _blob_sha1(data: bytes) -> str:
    """git's content hash, spelled out here so the module's copy is CHECKED
    against an independent expression rather than against itself."""
    return hashlib.sha1(b"blob " + str(len(data)).encode()
                        + b"\0" + data).hexdigest()


def _fake_get(files: dict[str, bytes], calls: list[str], *,
              payload: dict[str, bytes] | None = None):
    """A `_get` that answers the contents API from `files` and serves bytes.

    `payload` overrides what the RAW host returns without changing what the
    LISTING claims, which is how the hash refusal is driven: that is exactly
    the shape of a moved revision or a captive portal.
    """
    served = payload if payload is not None else files

    def get(url: str) -> bytes:
        calls.append(url)
        if url.startswith(M.API_BASE):
            path = url[len(M.API_BASE) + 1:].split("?")[0]
            assert f"ref={M.MENAGERIE_SHA}" in url, url
            out = []
            seen: set[str] = set()
            for rel in files:
                if not rel.startswith(path + "/"):
                    continue
                tail = rel[len(path) + 1:]
                if "/" in tail:
                    d = tail.split("/", 1)[0]
                    if d not in seen:
                        seen.add(d)
                        out.append({"path": f"{path}/{d}", "type": "dir",
                                    "sha": "0" * 40, "size": 0})
                    continue
                out.append({"path": rel, "type": "file",
                            "sha": _blob_sha1(files[rel]),
                            "size": len(files[rel])})
            return json.dumps(out).encode()
        assert url.startswith(M.RAW_BASE), url
        rel = url[len(M.RAW_BASE) + 1:]
        return served[rel]

    return get


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty cache of our own, and the body cache cleared either side.

    `menagerie._BODIES` is module state keyed by name, so a fake "fake" body
    built in one case would otherwise be served to the next — and the real
    cache's Go2 must not leak into a case about discovery.
    """
    monkeypatch.setattr(M, "CACHE_DIR", tmp_path / "menagerie")
    M._BODIES.clear()
    try:
        yield tmp_path / "menagerie"
    finally:
        M._BODIES.clear()


@pytest.fixture
def fetched(cache, monkeypatch):
    """The fake model, downloaded. Returns `(name, calls)`."""
    files = _model_files()
    calls: list[str] = []
    monkeypatch.setattr(M, "_get", _fake_get(files, calls))
    M.fetch("fake")
    return "fake", calls


# ------------------------------------------------------------- the hashing

def test_the_blob_hash_is_gits_own_and_is_pinned_against_it():
    """`_verify_blob` rests on this, so it is pinned against literals.

    These three came from `git hash-object --stdin`, not from the function
    under test — a hash helper checked against its own formula is a test of
    nothing, and this is the only thing standing between the pinned revision
    and a proxy's sign-in page.
    """
    assert M._git_blob_sha1(b"hello menagerie\n") == (
        "66cf13decff1bfc99edcc292ac3a68bb72d270f6")
    assert M._git_blob_sha1(b"") == (
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
    assert M._git_blob_sha1(b"x") == (
        "c1b0730e0133447badcfd47fd144e254807b06e1")


def test_a_wrong_byte_changes_the_hash():
    """The planted negative for the pins above."""
    assert M._git_blob_sha1(b"x") != M._git_blob_sha1(b"y")
    assert M._git_blob_sha1(b"x") != M._git_blob_sha1(b"x ")


# ------------------------------------------------------------ the download

def test_fetch_downloads_the_model_and_records_a_manifest(cache, monkeypatch,
                                                          capsys):
    """One download per wanted file, and a manifest that describes the lot.

    The manifest is what makes this catalogue work offline afterwards: it
    names the robot MJCF, the scene, the licence line and a sha256 per file,
    so `ready()`, `body()` and the next `fetch()` need neither the network
    nor a table in this repo.
    """
    files = _model_files()
    calls: list[str] = []
    monkeypatch.setattr(M, "_get", _fake_get(files, calls))

    d = M.fetch("fake")
    assert d == cache / "fake"
    man = M.read_manifest("fake")
    assert man["name"] == "fake"
    assert man["sha"] == M.MENAGERIE_SHA
    assert man["robot_xml"] == "fake.xml"
    assert man["scene_xml"] == "scene.xml"
    assert man["license"] == "Copyright (c) 2026 Somebody Else"
    assert man["fetched_at"].endswith("Z")
    got = {rel: sha for rel, sha in man["files"]}
    assert set(got) == {"fake.xml", "scene.xml", "LICENSE", "README.md",
                        "assets/blob.stl"}
    for rel, sha in got.items():
        assert (d / rel).is_file()
        assert hashlib.sha256((d / rel).read_bytes()).hexdigest() == sha
    assert M.ready("fake") is True
    assert "fake licence: Copyright (c) 2026" in capsys.readouterr().out


def test_the_preview_render_is_not_downloaded_and_the_mesh_is(fetched, cache):
    """The filter, both ways round.

    Menagerie keeps a rendered preview beside each model — `go2.png` is
    1.9 MB — and it is documentation of the robot, not part of it. A texture
    lives under `assets/` and must never be caught by the same rule, which is
    why the filter is by LOCATION first and extension second.

    The planted negative is inline and is what FOUND the rule that decides:
    the first draft also carried a deny-list of image extensions, and turning
    it off changed nothing because the allow-list had already excluded the
    preview. Only widening the allow-list moves this case, so that is what it
    plants (`AGENTS.md`: check a knob's reachable set before trusting it).
    """
    name, _calls = fetched
    assert not (cache / name / "fake.png").exists()
    assert (cache / name / "assets/blob.stl").is_file()
    assert M._wanted("assets/skin.png") is True
    assert M._wanted("fake.png") is False
    assert M._wanted("fake.xml") is M._wanted("LICENSE") is True
    assert M._wanted("notes.rst") is False
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "_WANTED_SUFFIXES", M._WANTED_SUFFIXES + (".png",))
        assert M._wanted("fake.png") is True, (
            "the allow-list is not what decides — this case cannot fail")


def test_a_second_fetch_downloads_nothing(cache, monkeypatch):
    """Idempotence: what makes `fetch-robot` safe to re-run, and what a
    `setup.sh` that iterated the catalogue would lean on. A present file whose
    sha256 matches the manifest is skipped, so the second call costs one
    listing and no bytes."""
    files = _model_files()
    calls: list[str] = []
    monkeypatch.setattr(M, "_get", _fake_get(files, calls))
    M.fetch("fake")
    first = [c for c in calls if c.startswith(M.RAW_BASE)]
    assert len(first) == 5
    M.fetch("fake")
    assert [c for c in calls if c.startswith(M.RAW_BASE)] == first, (
        "a second fetch re-downloaded files")


def test_fetch_refuses_bytes_that_are_not_the_pinned_revisions(cache,
                                                               monkeypatch):
    """The planted regression for the hash check: wrong bytes, refused.

    And refused WITHOUT leaving anything behind — no target file, no temp
    file, no manifest — so a failed fetch cannot be followed by a `ready()`
    that says yes.
    """
    files = _model_files()
    wrong = dict(files)
    wrong["fake/assets/blob.stl"] = b"<html>sign in to continue</html>"
    monkeypatch.setattr(M, "_get", _fake_get(files, [], payload=wrong))
    with pytest.raises(RuntimeError, match="the listing says"):
        M.fetch("fake")
    assert not (cache / "fake/assets/blob.stl").exists()
    assert M.ready("fake") is False
    assert not list((cache / "fake/assets").glob(".*")), "a temp file was left"


def test_fetch_repairs_a_file_that_was_corrupted_after_it_arrived(cache,
                                                                 monkeypatch):
    """`ready()` checks presence only, so this is what makes that cheap
    answer safe: the next fetch replaces what a truncated write, a
    half-finished copy or an edit left behind."""
    files = _model_files()
    calls: list[str] = []
    monkeypatch.setattr(M, "_get", _fake_get(files, calls))
    M.fetch("fake")
    victim = cache / "fake/assets/blob.stl"
    victim.write_bytes(b"truncated")
    M.fetch("fake")
    assert calls[-1].endswith("fake/assets/blob.stl")
    assert victim.read_bytes() == _tetra_stl()


def test_a_model_directory_that_is_not_there_is_refused(cache, monkeypatch):
    """A typo in a model name must say so, not produce an empty body."""
    monkeypatch.setattr(M, "_get", _fake_get({}, []))
    with pytest.raises(RuntimeError, match="no usable files under 'wombat'"):
        M.fetch("wombat")


def test_a_listing_that_is_not_a_directory_is_refused(cache, monkeypatch):
    """The contents API answers a FILE path with an object, not a list.

    Reachable with a perfectly well-shaped name — Menagerie has no model
    called `readme`, but the API will happily answer for a file that exists
    at the repo root — so the guard is not redundant with `check_name`.
    """
    monkeypatch.setattr(M, "_get", lambda url: b'{"type": "file"}')
    with pytest.raises(RuntimeError, match="did not list a directory"):
        M.fetch("readme")


@pytest.mark.parametrize("bad", [
    "../../etc", "unitree_go2/go2.xml", "", "a b", "/absolute", ".hidden",
])
def test_a_name_that_is_not_a_model_directory_is_refused(cache, bad):
    """A model name reaches a URL and a directory under the cache, and it
    arrives from a command line — so `menagerie:../../../etc` is a thing
    somebody can type. Refused by SHAPE, before either use.

    The complement is the half that matters: every real Menagerie directory
    must pass, which the two fetched models assert below.
    """
    with pytest.raises(ValueError, match="not a Menagerie model name"):
        M.check_name(bad)
    with pytest.raises(ValueError, match="not a Menagerie model name"):
        M.model_dir(bad)


@pytest.mark.parametrize("good", ["unitree_go2", "trs_so_arm100", "aloha",
                                  "franka_emika_panda", "ur5e", "shadow_hand"])
def test_a_real_model_directory_name_passes(good):
    """The planted complement: a guard that refused everything would make the
    case above pass and the catalogue unusable."""
    assert M.check_name(good) == good


# ------------------------------------------------ which file is the robot

def test_the_robot_mjcf_is_the_one_the_scene_includes(fetched, cache):
    """Read out of the scene, not guessed from the directory name.

    Two guesses were rejected against real models: `<dir>.xml`
    (`trs_so_arm100/` holds `so_arm100.xml`) and "the only non-scene xml"
    (the Go2 ships a `go2_mjx.xml` beside `go2.xml`). The scene's
    `<include>` is the model author's own statement of which file is the
    robot.
    """
    name, _ = fetched
    d = cache / name
    files = [rel for rel, _sha in M.read_manifest(name)["files"]]
    assert M._pick_xmls(d, files) == ("fake.xml", "scene.xml")

    # The Go2's shape: a real scene, an mjx scene, and two robots.
    (d / "fake_mjx.xml").write_bytes(_ROBOT_XML)
    (d / "scene_mjx.xml").write_bytes(
        _SCENE_XML.replace(b"fake.xml", b"fake_mjx.xml"))
    assert M._pick_xmls(d, files + ["fake_mjx.xml", "scene_mjx.xml"]) == (
        "fake.xml", "scene.xml")


def test_two_candidate_mjcfs_and_no_scene_is_refused_by_name(fetched, cache):
    """The planted negative: picking one would be picking a robot.

    A model this rule cannot read is a RuntimeError naming the candidates and
    the escape hatch (`MjcfBody.from_mjcf` by hand), which is a better answer
    than a body built on the wrong file.
    """
    name, _ = fetched
    d = cache / name
    (d / "other.xml").write_bytes(_ROBOT_XML)
    with pytest.raises(RuntimeError, match="cannot tell which of"):
        M._pick_xmls(d, ["fake.xml", "other.xml"])


def test_a_model_with_no_scene_still_becomes_a_body(cache, monkeypatch):
    """The fallback: no `scene.xml`, so `MjcfBody` generates one.

    Recorded as `scene_source == "generated"` rather than left to be inferred,
    because "whose floor is this body standing on" changes what the stage
    pitch was measured against.
    """
    files = {k: v for k, v in _model_files().items()
             if not k.endswith("scene.xml")}
    monkeypatch.setattr(M, "_get", _fake_get(files, []))
    M.fetch("fake")
    assert M.read_manifest("fake")["scene_xml"] is None
    body = M.body("fake")
    assert body.extra["scene_source"] == "generated"
    assert body.stand_keyframe == "home"       # the robot file still has one
    assert body.ready() is True


# ---------------------------------------------------------- the body it is

def test_the_fetched_model_becomes_a_level_zero_body(fetched):
    """Nothing per-robot: the joints, the pose, the pitch and the group all
    come off the file the manifest named."""
    body = M.body("fake")
    assert body.id == "menagerie:fake"
    assert body.kind == "generic"
    assert body.joint_names == ("j1", "j2")
    assert body.stand_keyframe == "home"
    assert body.default_pose.tolist() == pytest.approx([0.25, 0.01])
    assert body.obs_dim == 6 and body.num_actions == 2
    assert body.lab_spacing_m > 0
    assert body.visual_group == 2               # the one contype-0 mesh geom
    assert len(body.visual_scene()["meshes"]) == 1
    assert body.contract().tiles() == ()


def test_the_licence_travels_with_the_body(fetched):
    """A licence that stayed on GitHub is a licence nobody read.

    Menagerie's models are under per-model terms — the Go2's first line is
    Unitree's copyright, the SO-ARM100's is "Apache License" — so the line is
    in the manifest, on the body, and printed by `fetch-robot`.
    """
    body = M.body("fake")
    assert body.extra["license"] == "Copyright (c) 2026 Somebody Else"
    assert body.extra["menagerie_sha"] == M.MENAGERIE_SHA


def test_a_missing_licence_file_says_so_rather_than_claiming_apache(cache,
                                                                   monkeypatch):
    """The planted negative for the licence line: no LICENSE, no claim.

    Defaulting to "Apache-2.0" would be a statement about somebody else's
    model that this repo has no standing to make.
    """
    files = {k: v for k, v in _model_files().items()
             if not k.endswith("LICENSE")}
    monkeypatch.setattr(M, "_get", _fake_get(files, []))
    M.fetch("fake")
    assert M.read_manifest("fake")["license"].startswith("unstated")


def test_the_body_is_built_once_and_rebuilt_after_a_refetch(fetched,
                                                            monkeypatch,
                                                            cache):
    """`registry()` runs per roster change and per policy load in the lab, and
    building a body COMPILES its model. So it happens once — keyed on the
    manifest's MTIME, so a re-fetch is still seen.

    The mtime key is the load-bearing half and the obvious test misses it:
    `fetch()` also drops the cache entry itself, so driving this through
    `fetch()` passes with the key removed entirely (MEASURED — that plant was
    a MISS until this case was rewritten). What the key is actually for is a
    manifest written by ANOTHER process — a second `fetch-robot` in a second
    terminal while the lab is running — so the case writes the manifest
    directly and never calls `fetch()`.
    """
    first = M.body("fake")
    assert M.body("fake") is first, "the model was compiled twice"

    man_path = M.manifest_path("fake")
    man = M.read_manifest("fake")
    # Another process re-fetched: same files, a new manifest. `os.utime` makes
    # the mtime move on a filesystem whose resolution would otherwise put the
    # two writes in the same tick.
    M._write_json(man_path, man)
    os.utime(man_path, ns=(man_path.stat().st_atime_ns,
                           man_path.stat().st_mtime_ns + 1_000_000))
    again = M.body("fake")
    assert again is not first, (
        "a manifest rewritten by another process was not picked up")
    assert again.joint_names == first.joint_names

    # And `fetch()` drops the entry too, so the two paths agree.
    monkeypatch.setattr(M, "_get", _fake_get(_model_files(), []))
    M.fetch("fake")
    assert M.body("fake") is not again


# ----------------------------------------------------------- the discovery

def test_a_fetched_model_is_discovered_by_the_registry(fetched):
    """The claim: no entry was written anywhere for this body.

    `_BUILTINS` is untouched, no entry point was installed, nothing called
    `register()`. The body is in the registry because its manifest is in the
    cache — which is the only way a catalogue of 71 models can work.
    """
    assert "menagerie:fake" in R.ids()
    assert "menagerie:fake" in R.registry()
    entry = R.get("menagerie:fake")
    assert entry._resolve() is M.body("fake")
    assert entry.joint_names == ("j1", "j2")    # through the proxy
    # ...and it did not displace anything.
    assert R.ids()[:3] == ("microduck", "g1", "mars")
    assert R.get("microduck").id == "microduck"


def test_discovery_does_not_compile_a_single_model(fetched, monkeypatch):
    """The laziness, as a count rather than a stopwatch.

    `registry()` runs per roster change and per policy load in the lab, and
    building a discovered body compiles its MJCF twice and measures its stage
    pitch — MEASURED at ~155 ms each in a fresh interpreter, which is 310 ms
    for two and would be ~3.1 s for twenty on whichever call happened to be
    first. So discovery must build NOTHING, and the four questions a listing
    actually asks (`id`, `title`, `noun`, `kind`) must be answerable without
    a compile.

    Counted, not timed: a timing assertion on a shared machine is a flake,
    and "how many models did we build" is the property anyway.
    """
    from microduck_local.robots import mjcf_body as MB

    built: list[str] = []
    real = MB.MjcfBody.from_mjcf.__func__

    def counting(cls, robot_xml, **kw):
        built.append(kw.get("id", "?"))
        return real(cls, robot_xml, **kw)

    monkeypatch.setattr(MB.MjcfBody, "from_mjcf", classmethod(counting))
    M._BODIES.clear()

    for _ in range(5):
        reg = R.registry()
        R.ids()
        M.registry()
    assert built == [], f"discovery compiled {built}"
    entry = reg["menagerie:fake"]
    # The cheap four come off the manifest and the name.
    assert (entry.id, entry.title, entry.noun, entry.kind) == (
        "menagerie:fake", "Fake", "Fake", "generic")
    assert built == [], "one of the cheap fields resolved the body"

    # ...and asking something only the model knows builds it, ONCE.
    assert entry.joint_names == ("j1", "j2")
    assert entry.lab_spacing_m > 0
    assert entry.stand_keyframe == "home"
    assert built == ["menagerie:fake"], built


def test_a_proxy_that_resolved_at_construction_is_caught(fetched, monkeypatch):
    """The planted negative for the case above.

    Without it, `built == []` could be passing because the counter was never
    wired to the thing that builds — a renamed method, a `classmethod`
    wrapper that did not take. So this plants the regression the laziness
    exists to prevent (a proxy that resolves in `__init__`) and requires the
    same count to notice.
    """
    from microduck_local.robots import mjcf_body as MB

    built: list[str] = []
    real = MB.MjcfBody.from_mjcf.__func__

    def counting(cls, robot_xml, **kw):
        built.append(kw.get("id", "?"))
        return real(cls, robot_xml, **kw)

    monkeypatch.setattr(MB.MjcfBody, "from_mjcf", classmethod(counting))

    class EagerBody(M.LazyBody):
        __slots__ = ()

        def __init__(self, name, dest, manifest):
            super().__init__(name, dest, manifest)
            self._resolve()                    # the regression

    monkeypatch.setattr(M, "LazyBody", EagerBody)
    M._BODIES.clear()
    R.registry()
    assert built == ["menagerie:fake"], (
        "an eagerly-resolving proxy was not noticed — the counter is not "
        "wired to what builds a body, so the laziness case above is vacuous")


def test_an_empty_cache_discovers_nothing_and_breaks_nothing(cache):
    """The planted negative for discovery: the fresh-checkout case.

    An absent cache directory must be an empty answer, not an exception —
    `registry()` is called in the lab's 50 Hz loop's neighbourhood and a
    raise there takes every duck off the stage, not just the missing body.
    """
    assert M.names() == () and M.ids() == ()
    assert M.registry() == {}
    assert [i for i in R.ids() if i.startswith("menagerie:")] == []
    assert "microduck" in R.registry()


def test_a_model_that_does_not_build_a_body_is_refused_at_resolve(fetched,
                                                                  monkeypatch):
    """The conformance gate, which moved when discovery went lazy.

    `robots/registry._load_namespaces` used to run `conforms()` on every
    discovered body. It cannot any more: `conforms()` asks `hasattr`, which a
    lazy proxy answers by RESOLVING, so the gate would have compiled every
    model and undone the laziness (`registry._load_namespaces`' docstring
    states the measurement). So the check moved into `body()`, where the body
    is already built and it costs nothing — and this is the case that says it
    is still there.
    """
    from microduck_local.robots import mjcf_body as MB

    class NotABody:
        id = "menagerie:fake"

    monkeypatch.setattr(MB.MjcfBody, "from_mjcf",
                        classmethod(lambda cls, *a, **k: NotABody()))
    M._BODIES.clear()
    with pytest.raises(TypeError, match="not a Body — missing"):
        M.body("fake")
    # ...and the proxy surfaces it rather than swallowing it.
    with pytest.raises(TypeError, match="not a Body"):
        _ = R.get("menagerie:fake").joint_names


def test_a_manifest_that_will_not_parse_is_an_absence_with_a_reason(cache,
                                                                    capsys):
    """A half-written or hand-edited manifest must not take the roster down."""
    (cache / "broken").mkdir(parents=True)
    (cache / "broken" / M.MANIFEST).write_text("{not json")
    assert M.read_manifest("broken") is None
    assert M.ready("broken") is False
    # `names()` lists it (the file IS there) and `registry()` reports it.
    assert M.names() == ("broken",)
    assert M.registry() == {}
    assert "broken unavailable" in capsys.readouterr().out
    assert "microduck" in R.registry()


def test_an_unfetched_name_is_answered_with_the_command_that_fetches_it(cache):
    """The difference between "no such robot" and "you have not downloaded it
    yet" is the whole message — `robots/registry.get`'s rule, extended to a
    namespace where the catalogue cannot be enumerated without the network."""
    assert R.namespace_of("menagerie:unitree_go2") == "menagerie"
    assert R.setup_hint("menagerie:unitree_go2") == (
        "uv run fetch-robot menagerie:unitree_go2")
    with pytest.raises(KeyError, match="fetch-robot menagerie:unitree_go2"):
        R.get("menagerie:unitree_go2")
    # A prefix that is not a namespace is not given a fake command.
    assert R.namespace_of("wombat:x") == ""
    assert R.setup_hint("wombat:x") == ""


def test_the_namespace_never_collides_with_a_builtin_or_a_plugin():
    """Why the ids carry a prefix and a colon.

    A bare `unitree_go2` could be shadowed by, or shadow, a built-in or a
    pip-installed plugin's body — which is the one thing `register()` refuses
    outright, because the goldens and the deployment contract name specific
    bodies. The colon makes the two id spaces disjoint by construction.
    """
    assert all(":" not in b.id for b in R._BUILTINS)
    assert M.NAMESPACE in R._NAMESPACES
    assert R.namespace_module(M.NAMESPACE) is M


# ------------------------------------------------------------- fetch-robot

def test_fetch_robot_routes_a_namespaced_id_to_the_catalogue(cache,
                                                             monkeypatch,
                                                             capsys):
    """`fetch-robot menagerie:<name>` cannot go through `Body.fetch`.

    There is no body to ask: the body is READ from the files the download
    brings. So the CLI has a namespace branch, and what it prints afterwards
    is the body those files turned into — the proof that nothing per-robot was
    needed to read it.
    """
    from microduck_local import fetch_robot

    monkeypatch.setattr(M, "_get", _fake_get(_model_files(), []))
    fetch_robot.main(["menagerie:fake"])
    out = capsys.readouterr().out
    assert "menagerie:fake assets at" in out
    assert "2 joints, spawn keyframe 'home'" in out
    assert "stage pitch" in out
    assert "level 0" in out


def test_fetch_robot_on_an_already_fetched_model_is_the_no_op_download(
        fetched, capsys):
    """Once it is in the cache it is an ordinary body, and `Body.fetch()` on
    it returns the directory its files live in — so the palette's one
    download path needs no special case for it either."""
    from microduck_local import fetch_robot

    fetch_robot.main(["menagerie:fake"])
    assert "menagerie:fake assets at" in capsys.readouterr().out


def test_fetch_robot_with_no_argument_says_how_to_add_a_catalogue_model(cache,
                                                                        capsys):
    """The inventory cannot list 71 models — naming them here would be the
    per-robot table the whole seam exists to delete. So it lists what is
    FETCHED and says how to add one."""
    from microduck_local import fetch_robot

    fetch_robot.main([])
    out = capsys.readouterr().out
    assert "microduck" in out and "g1" in out and "mars" in out
    assert "uv run fetch-robot menagerie:<model>" in out
    assert "mujoco_menagerie" in out


def test_fetch_robot_refuses_a_namespaced_id_with_no_model_name(cache, capsys):
    from microduck_local import fetch_robot

    with pytest.raises(SystemExit) as e:
        fetch_robot.main(["menagerie:"])
    assert e.value.code == 1
    assert "names no model" in capsys.readouterr().err


# -------------------------------------------------- §7's settling number

def _code_mentions(needles: tuple[str, ...]) -> list[str]:
    """Every place in `src/` a robot's name reaches the CODE.

    `docs/mars-roadmap.md` §7.1's settling number is "0 lines of Go2-specific
    code in this repo", and a plain `grep -c` cannot state it: the module
    docstrings, the `#:` field comments and `fetch-robot`'s own help text all
    name `menagerie:unitree_go2` as the EXAMPLE somebody types, which is
    documentation and not a branch. 23 such lines exist and they are the
    point of the docs, not a leak.

    So this tokenises instead and drops every STRING and COMMENT token: what
    is left is identifiers, attributes, literals and operators — a name in
    there would be an `if name == "unitree_go2"`, a constant, a dict key or
    an import, which is exactly what must be zero. Enforced as a test rather
    than reported once, because the number is only worth anything if the next
    person cannot quietly break it.
    """
    import io
    import tokenize
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    hits: list[str] = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text()
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT,
                            tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE,
                            tokenize.FSTRING_END):
                continue
            low = tok.string.lower()
            if any(n in low for n in needles):
                hits.append(f"{path.relative_to(src)}:{tok.start[0]}: "
                            f"{tok.string}")
    return hits


def test_no_line_of_code_in_src_knows_what_a_go2_is():
    """§7's settling number, as an enforced test.

    Two real Menagerie robots reach the stage, are posable and pass the
    conformance suite's generic cases, and nothing in `src/` names either of
    them outside prose. Their names live in `.cache/menagerie/*/body.json`,
    which is data a download wrote.
    """
    assert _code_mentions(("go2", "so_arm100", "unitree_go2")) == []


def test_the_scan_would_catch_a_name_that_reached_the_code():
    """The planted negative: the scan must see a name it is given.

    Without this, a tokeniser that dropped the wrong token types (or a
    `rglob` that matched nothing) would report zero for every needle and the
    case above would be a test of the empty set. `menagerie` and `mjcf` DO
    appear as code in `src/` — module names, the namespace constant — so the
    scan finds them.
    """
    assert _code_mentions(("menagerie",)), "the scan sees no code at all"
    assert _code_mentions(("mjcf_body",))
    assert _code_mentions(("definitely_not_a_robot_name",)) == []


# ------------------------------------------- the real download, if it is here

def _real(name: str):
    return pytest.mark.skipif(
        not M.ready(name),
        reason=f"{name} not fetched — uv run fetch-robot menagerie:{name}")


@_real("unitree_go2")
def test_the_real_go2_is_a_twelve_joint_body_with_unitrees_licence():
    """MEASURED on the pinned revision, from the actual download.

    The numbers are the settling ones for `docs/mars-roadmap.md` §7: 12
    joints read off `go2.xml`, the `home` keyframe as the spawn pose, 0.7536 m
    across at that pose so a 2.653 m stage slot, and Unitree's own licence
    line — none of it written anywhere in `src/`.
    """
    body = R.get("menagerie:unitree_go2")
    assert body.kind == "generic" and body.num_joints == 12
    assert body.stand_keyframe == "home"
    assert body.extra["scene_source"] == "given"
    assert body.extra["measured_width_m"] == pytest.approx(0.7536, abs=0.002)
    assert body.lab_spacing_m == pytest.approx(2.653, abs=0.01)
    assert body.visual_group == 2
    assert "Unitree" in body.extra["license"]
    assert body.contract().id == "mjcf-menagerie:unitree_go2-36-v0"
    model = body.robot_model()
    assert len(body.visual_scene()["meshes"]) == int(model.nmesh) == 16


@_real("trs_so_arm100")
def test_the_real_arm_shows_why_nmesh_is_the_wrong_yardstick():
    """MEASURED, and it corrects the plan's check.

    §7's proof asked for "mesh counts == nmesh". It holds for the Go2, whose
    collision geoms are primitives, and it does NOT hold here: five of this
    arm's eighteen meshes are collision-only (`Fixed_Jaw_Collision_1`, three
    `Moving_Jaw_Collision_*` and one more), in group 3. The viewer must not
    draw those, so a dump of 13 against an `nmesh` of 18 is CORRECT, and the
    right count is how many distinct meshes the visual group references —
    which is what the conformance suite's generic branch computes.
    """
    body = R.get("menagerie:trs_so_arm100")
    assert body.num_joints == 6 and body.stand_keyframe == "home"
    assert body.lab_spacing_m == pytest.approx(1.163, abs=0.01)
    assert "Apache License" in body.extra["license"]
    model = body.robot_model()
    drawn = len(body.visual_scene()["meshes"])
    assert int(model.nmesh) == 18
    assert drawn == 13, drawn
    visual = {int(model.geom_dataid[g]) for g in range(model.ngeom)
              if int(model.geom_group[g]) == body.visual_group}
    assert drawn == len(visual)
