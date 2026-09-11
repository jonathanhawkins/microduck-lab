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

import mujoco
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


# ------------------------------------------------------- /pose balance

def _balance(app, joints=None, root_pitch=0.0):
    return _endpoint(app, "/pose", "POST")(V.PoseReq(
        joints=list(joints if joints is not None else C.DEFAULT_POSE),
        rootPitch=root_pitch))["balance"]


def _posed(**deltas):
    """DEFAULT_POSE with named joints turned by radians."""
    joints = list(C.DEFAULT_POSE)
    for name, rad in deltas.items():
        joints[C.JOINT_NAMES.index(name)] += rad
    return joints


def _swayed(side):
    """Both hip rolls turned together to their servo limit on `side`: the
    rig's "sway" control at the end of its travel, the furthest hip roll alone
    can carry the CoM toward that foot."""
    lo_hi = 0 if side == "left" else 1
    limits = V.pose_scratch().limits
    joints = list(C.DEFAULT_POSE)
    for name in ("left_hip_roll", "right_hip_roll"):
        joints[C.JOINT_NAMES.index(name)] = float(limits[C.JOINT_NAMES.index(name)][lo_hi])
    return joints


def _sole_outline_mm(side):
    """The flat of a sole as the MODEL has it: world xy of the mesh vertices
    within SOLE_TOL of the lowest point at STAND. The tests read the pad off
    the mesh rather than carrying its size as a number."""
    scratch = V.pose_scratch()
    m, d = scratch.model, scratch.data
    scratch.solve(np.array(C.DEFAULT_POSE))
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
    mesh = m.geom_dataid[g]
    verts = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
    world = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    flat = world[world[:, 2] <= world[:, 2].min() + V.SOLE_TOL]
    return flat[:, :2] * 1000


def test_convex_hull_and_signed_distance():
    """The two geometry helpers, on a shape whose answer is known: a 2 x 1
    box with a stray interior point."""
    pts = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [1, 0.5], [0, 0]], dtype=float)
    hull = V._convex_hull(pts)
    assert len(hull) == 4 and {tuple(h) for h in hull} == {(0, 0), (2, 0), (2, 1), (0, 1)}
    assert V._signed_distance(np.array([1.0, 0.5]), hull) == pytest.approx(0.5)   # centre
    assert V._signed_distance(np.array([0.2, 0.5]), hull) == pytest.approx(0.2)   # near an edge
    assert V._signed_distance(np.array([-1.0, 0.5]), hull) == pytest.approx(-1.0)  # outside an edge
    assert V._signed_distance(np.array([-3.0, -4.0]), hull) == pytest.approx(-5.0)  # outside a corner
    # Two points have no inside: everything is outside a line.
    assert V._signed_distance(np.array([1.0, 1.0]), V._convex_hull(pts[:2])) == pytest.approx(-1.0)


def test_over_names_a_grounded_foot_the_com_is_inside():
    foot = lambda margin, grounded=True: {"marginMm": margin, "grounded": grounded}
    assert V._over({"left": foot(3.0), "right": foot(-40.0)}) == "left"
    assert V._over({"left": foot(3.0), "right": foot(5.0)}) == "right"     # the deeper one
    assert V._over({"left": foot(-0.5), "right": foot(-25.0)}) is None      # edge is not inside
    assert V._over({"left": foot(3.0, grounded=False), "right": foot(-40.0)}) is None


def test_pose_reports_the_com_against_both_soles(app):
    b = _balance(app)
    assert len(b["com"]) == 3 and np.isfinite(b["com"]).all()
    assert set(b["feet"]) == {"left", "right"}
    assert b["over"] in ("left", "right", None)
    for foot in b["feet"].values():
        assert set(foot) == {"grounded", "marginMm"}
        assert isinstance(foot["grounded"], bool)


def test_balance_footprint_is_the_sole_flat_not_its_bounding_box(app):
    """The sole is a mesh with a ~5 mm fillet, and its geom_size is the
    bounding box, ~3 mm wider than the flat on every side. Standing square
    the CoM sits on the midline, so its margin is minus the distance from the
    midline to the flat's inner edge, read off the mesh itself."""
    b = _balance(app)
    for side in ("left", "right"):
        inner_edge = np.abs(_sole_outline_mm(side)[:, 1]).min()
        want = -(inner_edge - abs(b["com"][1] * 1000))
        assert b["feet"][side]["marginMm"] == pytest.approx(want, abs=0.3)
        assert b["feet"][side]["grounded"]
    assert b["over"] is None


@pytest.mark.parametrize("foot", ["left", "right"])
def test_balance_hip_sway_to_the_limit_reaches_the_edge_of_the_sole(app, foot):
    """The measurement this panel exists for, and the answer is marginal: hip
    roll at its servo limit brings the CoM from ~25 mm outside the stance
    sole to within a millimetre of its edge, and no further. The bounding-box
    draft of this called the same pose 9.5 mm INSIDE; the flat's real outline
    is what a one-legged move has to work with. Forward kinematics lifts the
    other foot on the way (there is no ankle roll to keep it down), and the
    readout says so."""
    square = _balance(app)["feet"][foot]["marginMm"]
    b = _balance(app, _swayed(foot))
    assert b["feet"][foot]["grounded"]
    assert b["feet"][foot]["marginMm"] > square + 20
    assert abs(b["feet"][foot]["marginMm"]) <= 1.0
    other = "right" if foot == "left" else "left"
    assert b["feet"][other]["marginMm"] < 0 and not b["feet"][other]["grounded"]


@pytest.mark.parametrize("rad", [0.3, 0.6, 0.9])
def test_balance_does_not_inflate_when_a_foot_turns(app, rad):
    """A yawed foot keeps its outline. The bounding-box draft of this grew
    the pad from 27 x 22 to 34 x 35 mm at 0.9 rad and flipped the margin from
    -18 to +6 mm with the mass barely moved; the true outline cannot report
    the CoM inside a sole that twisting alone never put under it."""
    square = _balance(app)
    turned = _balance(app, _posed(left_hip_yaw=rad, right_hip_yaw=rad))
    for side in ("left", "right"):
        assert turned["feet"][side]["marginMm"] < 0
        # The feet do swing a little under the trunk as the hips yaw, so the
        # margin moves, but by less than the box inflation was worth.
        assert abs(turned["feet"][side]["marginMm"] - square["feet"][side]["marginMm"]) < 10
    assert turned["over"] is None


def test_balance_reports_a_lifted_foot_as_airborne(app):
    b = _balance(app, _posed(right_hip_pitch=-0.8, right_knee=-0.9))
    assert b["feet"]["left"]["grounded"]
    assert not b["feet"]["right"]["grounded"]


def test_balance_follows_the_posed_state_not_the_default(app):
    """Read straight off the same posed mjData as `bodies`: a pose that moves
    the mass must move the CoM."""
    upright = _balance(app)
    leaning = _balance(app, root_pitch=-0.5)
    assert leaning["com"] != upright["com"]
    assert leaning["feet"]["left"]["marginMm"] != \
        upright["feet"]["left"]["marginMm"]


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
