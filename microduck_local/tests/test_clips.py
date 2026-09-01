"""Keyframe-clip authoring endpoints: /joints, /pose, and the /clips CRUD.

The clip JSON is a CONTRACT shared with the imitation-RL side (it resamples a
saved clip at 50 Hz), so these tests lock the shape, the ordering rules, the
joint clamping, and the rootPitch SIGN — a silently flipped pitch would train
a backflip into a frontflip.

Handlers are pulled straight off the FastAPI app and called directly: the
project has no httpx, so starlette's TestClient is unavailable, and this still
exercises the registered endpoints (validation, HTTPExceptions and all).
"""

import json

import numpy as np
import pytest
from fastapi import HTTPException

from microduck_local import contract as C
from microduck_local import viz_server as V


def _endpoint(app, path: str, method: str):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ()):
            return r.endpoint
    raise AssertionError(f"no {method} {path} route")


@pytest.fixture
def app(monkeypatch, tmp_path):
    """A lab app with runs/, clips/ and lab-state.json all in tmp_path — a
    test must never write into the real workspace (or the live viewer's
    palette would sprout stray runs and clips)."""
    monkeypatch.setattr(V, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", str(tmp_path / "clips"))
    monkeypatch.setenv("LAB_STATE_PATH", str(tmp_path / "lab-state.json"))
    return V.make_app([])


def _pose_key(t=0.0, joints=None, root_pitch=0.0):
    return {"t": t,
            "joints": list(joints if joints is not None else C.DEFAULT_POSE),
            "rootPitch": root_pitch}


def _clip(name="hop", duration=1.0, keys=None, loop=False):
    return {"version": 1, "name": name, "duration": duration, "loop": loop,
            "keys": keys if keys is not None else [
                _pose_key(0.0), _pose_key(0.5, root_pitch=-0.8)]}


# --------------------------------------------------------------- /joints

def test_joints_meta_matches_the_model(app):
    meta = _endpoint(app, "/joints", "GET")()
    joints = meta["joints"]
    assert [j["name"] for j in joints] == list(C.JOINT_NAMES)
    assert [j["default"] for j in joints] == [
        round(float(v), 6) for v in C.DEFAULT_POSE]
    # Limits are the MJCF's, not hand-carried numbers.
    m = V.pose_scratch().model
    for j in joints:
        lo, hi = m.jnt_range[m.joint(j["name"]).id]
        assert (j["min"], j["max"]) == (round(float(lo), 6), round(float(hi), 6))
        assert j["min"] <= j["default"] <= j["max"]
    # Every joint names a distinct body inside the /scene body list — that map
    # is what turns a click on the 3D duck into a joint selection.
    bodies = [b["name"] if isinstance(b, dict) else b for b in meta["bodies"]]
    assert bodies == V.extract_scene()["bodies"]
    ids = [j["body"] for j in joints]
    assert len(set(ids)) == len(ids)
    assert all(0 < b < len(bodies) for b in ids)
    assert meta["trunkBody"] == bodies.index("trunk_base")


# ----------------------------------------------------------------- /pose

def test_pose_returns_one_body_pose_per_scene_body(app):
    out = _endpoint(app, "/pose", "POST")(
        V.PoseReq(joints=list(C.DEFAULT_POSE)))
    bodies = out["bodies"]
    assert len(bodies) == len(V.extract_scene()["bodies"])
    assert all(len(b) == 7 for b in bodies)          # x y z qw qx qy qz
    assert all(np.isfinite(b).all() for b in bodies)
    quats = np.array([b[3:] for b in bodies])
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-3)
    assert out["joints"] == [round(float(v), 6) for v in C.DEFAULT_POSE]
    # The standing pose stands on the floor, at the STAND keyframe's height.
    assert out["bodies"][1][2] == pytest.approx(0.12, abs=0.005)


def test_pose_clamps_to_joint_limits(app):
    out = _endpoint(app, "/pose", "POST")(V.PoseReq(joints=[9.0] * 14))
    meta = _endpoint(app, "/joints", "GET")()
    assert out["joints"] == [j["max"] for j in meta["joints"]]
    out = _endpoint(app, "/pose", "POST")(V.PoseReq(joints=[-9.0] * 14))
    assert out["joints"] == [j["min"] for j in meta["joints"]]


@pytest.mark.parametrize("bad", [
    [0.0] * 13,                    # too few
    [0.0] * 15,                    # too many
    [float("nan")] + [0.0] * 13,   # not finite
    [float("inf")] + [0.0] * 13,
])
def test_pose_rejects_malformed_joints(app, bad):
    with pytest.raises(HTTPException) as e:
        _endpoint(app, "/pose", "POST")(V.PoseReq(joints=bad))
    assert e.value.status_code == 422


def test_pose_root_pitch_sign_is_lean_back_negative(app):
    """THE sign lock: rootPitch < 0 must lean the trunk BACK, which the sim
    reads as projected gravity acquiring -x in the trunk frame."""
    post = _endpoint(app, "/pose", "POST")
    joints = list(C.DEFAULT_POSE)

    def gravity_x(pitch):
        trunk = post(V.PoseReq(joints=joints, rootPitch=pitch))["bodies"][1]
        return float(C.quat_rotate_inverse(
            np.array(trunk[3:]), np.array([0.0, 0.0, -1.0]))[0])

    assert gravity_x(0.0) == pytest.approx(0.0, abs=1e-6)
    assert gravity_x(-0.5) < -0.4      # lean back
    assert gravity_x(+0.5) > +0.4      # nose down


def test_pose_grounding_tracks_the_legs(app):
    """Grounding is what makes a crouch read as a crouch: bend the knees and
    the trunk must come DOWN, not float at the standing height."""
    post = _endpoint(app, "/pose", "POST")
    stand = list(C.DEFAULT_POSE)
    bent = list(C.DEFAULT_POSE)
    bent[3] += 0.5    # left_knee
    bent[12] -= 0.5   # right_knee (mirrored sign)
    z_stand = post(V.PoseReq(joints=stand))["bodies"][1][2]
    z_bent = post(V.PoseReq(joints=bent))["bodies"][1][2]
    assert z_bent < z_stand
    # ground=false pins the root at the keyframe height instead.
    assert post(V.PoseReq(joints=bent, ground=False))["bodies"][1][2] == \
        pytest.approx(z_stand, abs=1e-4)


def test_pose_does_not_disturb_a_live_duck(app):
    """The whole point of the scratch model: a lab duck mid-episode must not
    twitch because someone dragged a slider in the editor."""
    duck = V.Duck("d0", "probe", V._zero_infer, seed=3)
    for _ in range(5):
        duck.tick()
    before = duck.pose_payload()
    _endpoint(app, "/pose", "POST")(
        V.PoseReq(joints=[0.4] * 14, rootPitch=-1.0))
    assert duck.pose_payload() == before
    assert V.pose_scratch().data is not duck.env.data


# ---------------------------------------------------------------- /clips

def test_clip_save_load_list_delete_round_trip(app, tmp_path):
    put = _endpoint(app, "/clips/{name}", "PUT")
    get = _endpoint(app, "/clips/{name}", "GET")
    lst = _endpoint(app, "/clips", "GET")
    dele = _endpoint(app, "/clips/{name}", "DELETE")

    assert lst() == {"clips": []}
    saved = put("backflip", _clip(name="backflip", duration=1.6))
    assert saved["version"] == 1 and saved["name"] == "backflip"
    assert saved["duration"] == 1.6 and saved["loop"] is False
    assert [k["t"] for k in saved["keys"]] == [0.0, 0.5]
    assert saved["keys"][1]["rootPitch"] == -0.8
    assert len(saved["keys"][0]["joints"]) == C.NUM_JOINTS

    # On disk beside runs/, under the env-var-relocated clips dir.
    path = tmp_path / "clips" / "backflip.json"
    assert path.exists()
    assert json.loads(path.read_text())["keys"][1]["rootPitch"] == -0.8

    assert get("backflip")["keys"] == saved["keys"]
    put("wave", _clip(name="wave", duration=0.8, loop=True))
    names = [c["name"] for c in lst()["clips"]]
    assert sorted(names) == ["backflip", "wave"]
    assert all("modified" in c for c in lst()["clips"])

    assert dele("backflip") == {"deleted": "backflip"}
    assert [c["name"] for c in lst()["clips"]] == ["wave"]
    with pytest.raises(HTTPException) as e:
        get("backflip")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        dele("backflip")
    assert e.value.status_code == 404


def test_clip_name_from_the_url_wins(app):
    """The file and the clip's own `name` can never drift apart."""
    saved = _endpoint(app, "/clips/{name}", "PUT")(
        "real-name", _clip(name="something else"))
    assert saved["name"] == "real-name"
    assert _endpoint(app, "/clips/{name}", "GET")("real-name")["name"] == "real-name"


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden", "x" * 65,
                                  "hé", "semi;colon"])
def test_clip_rejects_unsafe_names(app, name):
    with pytest.raises(HTTPException) as e:
        _endpoint(app, "/clips/{name}", "PUT")(name, _clip())
    assert e.value.status_code == 422


@pytest.mark.parametrize("clip, why", [
    ({"duration": 1.0, "keys": []}, "no keys"),
    ({"duration": 1.0}, "keys missing"),
    ({"duration": 1.0, "keys": [_pose_key(0.25)]}, "first key not at t=0"),
    ({"duration": 1.0, "keys": [_pose_key(0.0), _pose_key(0.0)]}, "duplicate t"),
    ({"duration": 1.0, "keys": [_pose_key(0.0), _pose_key(0.6), _pose_key(0.3)]},
     "descending t"),
    ({"duration": 0.2, "keys": [_pose_key(0.0), _pose_key(0.5)]},
     "duration cuts the last key"),
    ({"duration": 0.0, "keys": [_pose_key(0.0)]}, "zero duration"),
    ({"duration": -1.0, "keys": [_pose_key(0.0)]}, "negative duration"),
    ({"duration": 1e6, "keys": [_pose_key(0.0)]}, "absurd duration"),
    ({"duration": 1.0, "keys": [{"t": 0.0, "joints": [0.0] * 13}]}, "13 joints"),
    ({"duration": 1.0, "keys": [{"t": 0.0, "joints": "nope"}]}, "joints not a list"),
    ({"duration": 1.0, "keys": [{"t": float("nan"), "joints": [0.0] * 14}]},
     "nan time"),
    ({"duration": 1.0, "keys": [{"t": 0.0, "joints": [float("inf")] * 14}]},
     "inf joint"),
    ({"duration": 1.0, "keys": [{"t": 0.0, "joints": [0.0] * 14,
                                 "rootPitch": float("nan")}]}, "nan pitch"),
    ({"duration": 1.0, "keys": "not a list"}, "keys not a list"),
])
def test_clip_rejects_contract_violations(app, clip, why):
    with pytest.raises(HTTPException) as e:
        _endpoint(app, "/clips/{name}", "PUT")("bad", clip)
    assert e.value.status_code == 422, why
    assert e.value.detail, why


def test_clip_joints_are_clamped_to_servo_limits(app):
    saved = _endpoint(app, "/clips/{name}", "PUT")(
        "wild", _clip(keys=[_pose_key(0.0, joints=[9.0] * 14)]))
    meta = _endpoint(app, "/joints", "GET")()
    assert saved["keys"][0]["joints"] == [j["max"] for j in meta["joints"]]


def test_clip_defaults_fill_in(app):
    """rootPitch and loop are optional in the contract — absent means 0/false,
    never a crash on the RL side."""
    saved = _endpoint(app, "/clips/{name}", "PUT")(
        "bare", {"duration": 0.5, "keys": [{"t": 0, "joints": [0.0] * 14}]})
    assert saved["keys"][0]["rootPitch"] == 0.0
    assert saved["loop"] is False
    assert saved["version"] == 1


def test_clips_dir_is_env_overridable(monkeypatch, tmp_path):
    """Same convention as MICRODUCK_RUNS_DIR / LAB_STATE_PATH."""
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", str(tmp_path / "elsewhere"))
    assert V.clips_dir() == tmp_path / "elsewhere"
    V.save_clip("x", V.clean_clip("x", _clip()))
    assert (tmp_path / "elsewhere" / "x.json").exists()
    monkeypatch.delenv("MICRODUCK_CLIPS_DIR")
    assert V.clips_dir() == V.CLIPS_DIR
    assert V.CLIPS_DIR.name == "clips"
    assert V.CLIPS_DIR.parent == V.RUNS_DIR.parent


def test_clip_listing_skips_unreadable_files(app, tmp_path):
    _endpoint(app, "/clips/{name}", "PUT")("good", _clip())
    (tmp_path / "clips" / "broken.json").write_text("{not json")
    assert [c["name"] for c in _endpoint(app, "/clips", "GET")()["clips"]] == ["good"]
