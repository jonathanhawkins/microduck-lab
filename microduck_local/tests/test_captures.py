"""🎥 screen-capture pipeline (viz_server captures endpoints' guts).

Function-level like test_lab.py (no httpx in the project, so no TestClient):
slug/filename hygiene, unique-stem collision handling, and one real
webm → mp4 + gif conversion through imageio-ffmpeg's bundled binary — the
same binary the lab server uses, so if this passes the endpoint's conversion
works too.
"""

import subprocess

from microduck_local import viz_server as V


def test_capture_slug_reduces_duck_names():
    assert V.capture_slug("backflip-e7f745 ✨ spotter-driven") == \
        "backflip-e7f745-spotter-driven"
    assert V.capture_slug("alpha_stand") == "alpha_stand"
    # emoji-only / empty names still make a usable filename
    assert V.capture_slug("🎓 !!") == "duck"
    assert V.capture_slug("") == "duck"
    # long names are trimmed and never end on a dangling dash
    assert len(V.capture_slug("x" * 200)) <= 40


def test_capture_file_re_blocks_traversal_and_uploads():
    ok = ["duck-20260831-120000.mp4", "a.gif", "teach-run.2.mp4"]
    bad = ["../x.mp4", ".hidden.mp4", "a/b.mp4", "a.webm", "a.upload", "a.MP4x"]
    for f in ok:
        assert V.CAPTURE_FILE_RE.match(f), f
    for f in bad:
        assert not V.CAPTURE_FILE_RE.match(f), f


def test_capture_base_uniquifies_same_second(tmp_path, monkeypatch):
    monkeypatch.setenv("MICRODUCK_CAPTURES_DIR", str(tmp_path))
    first = V.capture_base("duck")
    (tmp_path / f"{first}.mp4").write_bytes(b"x")
    second = V.capture_base("duck")
    assert second != first and second.startswith(first)


def test_convert_capture_round_trip(tmp_path, monkeypatch):
    """Odd-dimensioned webm (canvases often are) → playable mp4 + gif."""
    monkeypatch.setenv("MICRODUCK_CAPTURES_DIR", str(tmp_path))
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    src = tmp_path / "take.upload"
    subprocess.run(
        [exe, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=1.2:size=321x241:rate=15",
         "-c:v", "libvpx", "-f", "webm", str(src)],
        check=True)
    out = V.convert_capture(src, "take-test")
    assert out["mp4"] == "/captures/take-test.mp4"
    assert out["gif"] == "/captures/take-test.gif"
    mp4, gif = tmp_path / "take-test.mp4", tmp_path / "take-test.gif"
    assert mp4.stat().st_size > 1000 and gif.stat().st_size > 1000
    assert out["mp4Kb"] == mp4.stat().st_size // 1024
    # h264 demanded even dimensions — the crop guard must deliver them
    probe = subprocess.run(
        [exe, "-i", str(mp4)], capture_output=True, text=True)
    assert "320x240" in probe.stderr
