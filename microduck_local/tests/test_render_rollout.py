"""render-rollout: the pure pieces the contact sheet's readability rests on.

The sheet exists so a model can READ what a policy did instead of guessing from
reward sums, so the things worth locking are the ones that would silently
corrupt that reading: which frames get sampled, whether a caption still carries
its standing reference, whether a caption can overflow its tile, and whether
the null control is actually limp.
"""

import types

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.render_rollout import (
    CAPTION_COLUMNS,
    FrameDiag,
    build_sheet,
    count_reversals,
    format_caption,
    grid_cols,
    handoff_due,
    load_driver,
    pack_names,
    parse_env_overrides,
    sheet_indices,
    short_label,
)


def _diag(**kw):
    base = dict(index=3, t=1.24, trunk_z=0.088, head_z=0.041, pitch_deg=-36.4,
                tilt_deg=36.4, rot_deg=319.0, contact_l=True, contact_r=False,
                ground_bodies=("jaw_soft",), driver="teach-backflip-402439-s5",
                handed_off=False)
    base.update(kw)
    return FrameDiag(**base)


def test_env_overrides_parse():
    got = parse_env_overrides(["MICRODUCK_BF_SPAWN_LO=1.4",
                               "MICRODUCK_SPAWN_FAMILY_PROBS=0.3,0.6"])
    assert got == {"MICRODUCK_BF_SPAWN_LO": "1.4",
                   "MICRODUCK_SPAWN_FAMILY_PROBS": "0.3,0.6"}
    # Values may contain '=' (only the first splits).
    assert parse_env_overrides(["K=a=b"]) == {"K": "a=b"}
    for bad in ("no_equals", "=novalue"):
        with pytest.raises(ValueError):
            parse_env_overrides([bad])


def test_sheet_indices_span_the_whole_episode():
    """The last frame is where a trick either landed or did not — it must never
    be sampled away, and the samples must not repeat."""
    idx = sheet_indices(201, 12)
    assert len(idx) == 12
    assert idx[0] == 0 and idx[-1] == 200
    assert idx == sorted(idx) and len(set(idx)) == 12
    # Roughly even spacing.
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1
    # Degenerate cases stay sane.
    assert sheet_indices(3, 12) == [0, 1, 2]   # never more frames than exist
    assert sheet_indices(1, 12) == [0]
    assert sheet_indices(0, 12) == []


def test_grid_cols_stays_near_square_and_capped():
    assert grid_cols(12) == 4
    assert grid_cols(9) == 3
    assert grid_cols(4) == 2
    assert grid_cols(30) == 4     # capped so tiles stay legible
    assert grid_cols(1) == 1


def test_caption_carries_the_standing_reference():
    """The whole point: a height without its reference is unreadable, and that
    is precisely how a folded crouch got scored as a stand."""
    lines = format_caption(_diag(), stand_z=0.120, head_ref_z=0.233)
    text = " ".join(lines)
    assert "trunk_z=0.088" in text and "stand 0.120" in text
    assert "head_z =0.041" in text and "stand 0.233" in text
    assert "pitch=-36" in text and "tilt=36" in text and "rot=+319" in text
    assert "feet L=1 R=0" in text
    assert "jaw_soft" in text            # non-foot body on the floor
    assert "t= 1.24s" in text


def test_caption_lines_fit_a_tile():
    """A clipped caption is a wrong caption. Worst case: long labels, three
    digits of rotation, several ground bodies, negative angles."""
    d = _diag(pitch_deg=-179.6, tilt_deg=179.6, rot_deg=-359.0, t=99.98,
              index=99, trunk_z=-0.123, head_z=1.234,
              ground_bodies=("trunk_base", "hip_left", "hip_right", "knee_left"),
              driver="teach-backflip-402439-s5")
    for line in format_caption(d, stand_z=0.120, head_ref_z=0.233):
        assert len(line) <= CAPTION_COLUMNS, f"{line!r} is {len(line)} columns"
    # Nothing is silently dropped: the overflowing bodies are counted.
    assert "+3" in format_caption(d, 0.120, 0.233)[4]


def test_pack_names_budgets_the_ground_list():
    assert pack_names((), 14) == "none"
    assert pack_names(("jaw_soft",), 14) == "jaw_soft"
    assert pack_names(("trunk_base", "hip_left", "hip_right"), 14) == "trunk_base,+2"
    # A single name wider than the budget is still trimmed to fit.
    assert len(pack_names(("a_very_long_body_name",), 14)) == 14


def test_caption_without_a_head_or_rotation():
    """Behaviors that track no trick rotation, and scenes with no jaw_soft
    body, must still produce a caption rather than 'nan'."""
    lines = format_caption(_diag(rot_deg=None, head_z=float("nan"),
                                 ground_bodies=()),
                           stand_z=0.120, head_ref_z=float("nan"))
    text = " ".join(lines)
    assert "rot=" not in text and "nan" not in text
    assert "n/a" in text and "floor:none" in text


def test_short_label_keeps_the_tail():
    """Run names disambiguate in their tail (…-402439-s5); truncating from the
    right would throw away the only part that identifies the stage."""
    assert short_label("alpha_stand") == "alpha_stand"
    out = short_label("teach-backflip-402439-s5")
    assert out.endswith("402439-s5") and len(out) == 13


def test_count_reversals_separates_a_hold_from_cycling():
    """A sustained pose and a policy flapping in and out of it can average to
    the same reward; they cannot have the same reversal count."""
    hold = [0.120 + 0.001 * np.sin(i) for i in range(200)]
    cycling = [0.120 + 0.040 * np.sin(i / 3.0) for i in range(200)]
    assert count_reversals(hold, 0.015) == 0
    assert count_reversals(cycling, 0.015) > 10
    # A single sustained move is one direction, not a reversal.
    assert count_reversals([0.05, 0.08, 0.11, 0.14], 0.015) == 0
    assert count_reversals([], 0.015) == 0


def test_handoff_due_mirrors_the_lab_rule():
    """viz_server.Duck._handoff_due: rotation past 5.2 rad, both feet down,
    AND the spin braked — an early handoff pivots on landing (the "lands then
    turns to the side" bug)."""
    import numpy as np

    def env(rot, left, right, gyro=(0.1, 0.1, 0.1)):
        e = types.SimpleNamespace(
            foot_contact_state={"left": left, "right": right},
            gyro_adr=slice(0, 3),
            data=types.SimpleNamespace(sensordata=np.array(gyro, dtype=float)))
        if rot is not None:
            e._bf_rot = rot
        return e

    assert handoff_due(env(5.3, True, True))
    assert not handoff_due(env(5.1, True, True))    # trick not finished
    assert not handoff_due(env(5.3, True, False))   # not on both feet
    assert not handoff_due(env(None, True, True))   # behavior tracks no rotation
    assert not handoff_due(env(5.3, True, True, gyro=(0.0, -2.0, 0.0)))  # still spinning


def test_null_controls_are_actually_null():
    """`limp` must leave every servo target where the joint already is — no
    restoring torque. If a maneuver still happens under this driver, the
    policy is not what caused it."""
    from microduck_local.walk_env import MicroduckWalkEnv

    env = MicroduckWalkEnv(obs_noise=False, domain_rand=False,
                           action_delay=False, random_yaw=False, seed=0)
    obs, _ = env.reset(seed=0)

    zero = load_driver("zero")
    np.testing.assert_array_equal(zero.fn(obs, env), np.zeros(14, np.float32))

    limp = load_driver("limp")
    assert limp.label == "limp"
    q_before = env.data.qpos[env.joint_qpos_adr].copy()
    env.step(limp.fn(obs, env))
    # ctrl = DEFAULT_POSE + action, so the applied target is the pose it was
    # already in.
    np.testing.assert_allclose(env.data.ctrl, q_before, atol=1e-6)

    with pytest.raises(SystemExit):
        load_driver("runs/definitely-not-a-real-run/policy.onnx")


def test_build_sheet_writes_a_grid_png(tmp_path):
    """Composition end to end (no MuJoCo): 12 tiles land in a 4x3 grid with
    room under each for its caption, and the header/footer bands are there."""
    from PIL import Image

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (360, 480, 3), dtype=np.uint8) for _ in range(12)]
    caps = [format_caption(_diag(index=i), 0.120, 0.233) for i in range(12)]
    out = tmp_path / "sheet.png"
    build_sheet(tiles, caps, ["header line"], ["footer line"],
                [i > 5 for i in range(12)], out)

    assert out.exists() and out.stat().st_size > 10_000
    w, h = Image.open(out).size
    assert w == 4 * 480 + 5 * 10          # 4 columns + padding
    assert h > 3 * 360                    # 3 rows plus caption/header/footer bands


def test_build_sheet_clips_overlong_lines(tmp_path):
    """A caption or footer longer than the sheet must be trimmed/wrapped, not
    silently painted past the edge."""
    tiles = [np.zeros((360, 480, 3), np.uint8)]
    long_line = "X" * 400
    out = tmp_path / "sheet.png"
    build_sheet(tiles, [[long_line]], [long_line], [long_line], [False], out)
    assert out.exists()


def test_default_pose_is_the_action_origin():
    """load_driver('limp') assumes the contract's target = DEFAULT_POSE +
    action; if that ever changes, the null control stops being null."""
    assert C.DEFAULT_POSE.shape == (C.NUM_JOINTS,)
