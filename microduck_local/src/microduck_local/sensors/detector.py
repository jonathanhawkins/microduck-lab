"""A geometric stand-in for the head camera + NPU detector (roadmap 1.3).

The real robot runs a YOLO11n on the RK3566's NPU (320×320 INT8, one class
today: duck; p50 25.7 ms, p95 58.4 ms per frame, upstream npu-bringup.md) and
plans to publish detections as state at a few Hz. Nothing here renders
pixels: a target is detected when it sits inside the camera's field of view,
is not occluded (one ray to its centre), and is large enough on the
imaginary sensor to be found — then it comes out as what the brain would
get: `{cls, bearing, elevation, width, conf}`, with bearing noise, a size
that under-reports at range, misses that grow as the target shrinks, the
odd false positive, and the measured latency between the frame and the
detection.

Classes that do NOT exist on the robot yet (`person`, `ball`, `marker`) are
still emitted here so brains can be written for them; the /sim inspector
labels them as simulated-only, as the roadmap asks (1.3, 4.3).

Frames: x-forward / y-left / z-up at the `head_camera` site (verified in
tests/test_sensors.py). Bearing is positive to the LEFT (+y), elevation
positive UP (+z), both in radians; `width` is the apparent angular width.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, fields, replace

import mujoco
import numpy as np

from .ray import UNSENSED_GROUP

DETECT_CLASSES = ("duck", "person", "ball", "marker", "toy", "basket", "post")

# THE CALIBRATION REFERENCE, NOT THE ROBOT'S CAMERA — the distinction matters
# and used to be blurred by the word "shipped".
#
# The two size thresholds below were sized on a 320 px YOLO11n input at 296
# px/rad, which happened to be a 62° lens. That pairing is now known NOT to be
# the robot's (docs/camera-hardware.md §1: the fitted module is 116° × 60°), but
# the thresholds are still *calibrated* at 296 px/rad and every other spec is
# read against it (`DetectorSpec.px_per_rad`). So these stay as they are: change
# them and you silently restate what "never found" and "always found" mean.
# They are a ruler, not a claim about the hardware.
SHIPPED_PX_H = 320
SHIPPED_FOV_H_DEG = 62.0
_SHIPPED_PX_PER_RAD = SHIPPED_PX_H / np.deg2rad(SHIPPED_FOV_H_DEG)


# THE CAMERA: see docs/camera-hardware.md, which is the single home for this
# and carries the workings. The three facts that bear on the numbers below:
#
# 1. `fov_h_deg` / `fov_v_deg` ARE THE FULL-ARRAY FIGURE FOR A CAMERA THE
#    ROBOT DOES NOT RUN THAT WAY. Upstream identifies an IMX219 (Pi Camera
#    v2, quoted 62.2 x 48.8) but `mediad` pins the SENSOR to 1920x1080, which
#    on that part is a centred CROP - libcamera: "(680, 692)/1920x1080 crop",
#    59% of the columns and 44% of the rows. With the stock f = 3.04 mm lens
#    that is 39.0 x 22.5 deg, so this spec may be ~23 deg too wide and ~25
#    deg TOO TALL. Unconfirmed (the lens is not in the repo, only the driver
#    overlay), so nothing here is re-baselined on it.
# 2. `px_h` IS THE NPU'S YOLO11n INPUT, NOT THE SENSOR. The sensor carries
#    1920 px across it, six times what the detector consumes, so a wider lens
#    is paid for by the INFERENCE INPUT and not by new hardware.
# 3. THE REPLACEMENT MODULE IS 116 x 60 deg (vendor: 1/2.9", 2.75 um BSI,
#    1920x1080 native, 90 fps, D 142.2 / H 116 / V 60, EFL ~2.9 mm). That is
#    essentially the lens sweep's 120 deg arm, which this repo has already
#    measured: at 320 px it tidied 0.632 against 0.889, worse on 24 of 24
#    seeds; at 640 px it recovered to 0.819. So the NPU input size decides
#    whether it helps or hurts. Its 60 deg VERTICAL is a clear win either way.
#
# One modelling gap it opens: that lens is NOT rectilinear (a pinhole EFL
# solved from H, V and D gives 1.65 / 2.57 / 1.04 mm - inconsistent; an
# equidistant model gives 2.61 / 2.84 / 2.44 mm, near the quoted 2.9). This
# detector maps pixel offset to bearing with a pinhole model and no
# distortion term, which is wrong exactly at the frame edges - where a 116
# deg lens keeps its extra view. Recorded, not built.


# The string camera fields and their legal values, for `from_env`.
CAMERA_CHOICES = {"projection": ("pinhole", "equidistant"), "site": ()}


@dataclass(frozen=True)
class DetectorSpec:
    # MEASURED, not assumed (2026-09-09): the module's own FOV table gives
    # D 142.2 deg, H 116 deg, V 60 deg, max DFOV 165 deg. The long-standing
    # "stock Pi Camera v2 or a wide M12 board?" question in
    # docs/camera-hardware.md section 1 is ANSWERED and it is the wide board.
    #
    # These replaced 62 x 48, which was the stock lens's full-array figure and
    # WRONG — every soccer and tidy number taken before this date is on a camera
    # this robot does not have. The correction runs opposite to what was
    # expected: the real camera is WIDER, and on the soccer ledger it beats the
    # old default by +3.7 s/min of possession (p = 0.011) at the same 320 px
    # detector input. It is also SOFTER per degree — 158 px/rad against 294 —
    # so the width wins on this task and may not on others.
    fov_h_deg: float = 116.0
    fov_v_deg: float = 60.0
    max_range_m: float = 4.0
    # AN ABLATION, NOT A ROBOT FEATURE. The Microduck has ONE camera; this
    # models a second one pitched down by this many degrees, sharing the
    # head's origin, lens and FOV, so the sim can ask one question: is the
    # blind radius really what the kick is waiting on? 0 = off = the robot.
    # Never a default, never a battery baseline; it exists so that
    # `MICRODUCK_CAMERA="bottom_pitch_deg=60"` can be a sensitivity test.
    #
    # Measured 2026-09-06 (roadmap Track 4 s6 A.1): with a second lens the
    # brain's fresh ball estimate at the swing goes from 34% of swings to
    # 97%, its correlation with the true side offset from r=0.48 to r=0.96,
    # and its median error from 2.3 cm to 1.2 cm. So yes - sensing is the
    # limit, and the closed list in item 7 stands. The useful part is the
    # geometry it forced out: the duck's camera is 25 cm up, half the
    # NAO's, so the NAO's 39.7 deg sees a floor ball at 20 cm but the 10 cm
    # kick spot needs about 60 deg - and nothing occludes it (the duck's
    # own body never blocks the ray). Which means the ONE real camera can
    # see the kick spot if the head pitches far enough; see `GAZE_MAX_BEARING`
    # in brain/controllers.py for why it currently refuses to try.
    #
    # A target the head camera cannot see is tested against the second
    # frustum, and if that one sees it the detection is reported in the
    # HEAD camera's frame - bearing is still the azimuth the brain steers
    # on, and `cam_pitch` on the frame is still the head's - so nothing
    # downstream needs to know which lens found it. Wide-lens pixel mapping
    # (`projection="equidistant"`) is applied in the head frame either way.
    bottom_pitch_deg: float = 0.0
    # The ROBOT's detector runs at 2 Hz — a thermal limit, not a taste
    # (microduck/robotd-params/src/lib.rs `DetectConfig.hz`, deploy/robotd.toml
    # [detect]; the 320 px INT8 YOLO11n on the NPU is p50 25.7 ms). 10 is the
    # lab default, kept on the owner's call (roadmap 12av) with the gap on
    # record: at 2 Hz the shipped kick pair whiffs 48 % in the per-swing gym
    # against 7 % here, because the duck arrives further from the ball. Do
    # not quote the sim's kick numbers as the robot's; `MICRODUCK_CAMERA=
    # "rate_hz=2.0"` is the honest arm, and 5 Hz the rung the brain's own
    # constants survive.
    rate_hz: float = 10.0
    site: str = "head_camera"
    # Detector input width in pixels.
    #
    # **ASSUMED, 2026-09-09, and it is the biggest assumption in this file.**
    # Upstream's npu-bringup.md has the NPU running YOLO11n on a 320×320 INT8
    # frame; 640 is a DECISION to model the robot as though the NPU sustains a
    # 640×640 input, taken because the alternative makes the fitted 116° lens
    # unusable and because nobody has measured the NPU's real p50/p95 at either
    # size (camera-hardware.md §5, open question 1 — still open).
    #
    # Why it matters this much: the size gate is a PIXEL fact, so a wide lens
    # without pixels costs detections outright. At 320 px the fitted lens is 158
    # px/rad and a duck at 3 m is 4.7 px wide — under the "sometimes found"
    # floor. At 640 px it is 316 px/rad, a duck at 3 m is 9.4 px, and the wide
    # lens costs nothing in reach while keeping nearly twice the view.
    #
    # If the NPU turns out NOT to sustain 640, set this back to 320 and every
    # number measured against this default is optimistic by roughly the ratio
    # above. That is the single number to check before quoting anything here.
    px_h: int = 640
    # Apparent-width thresholds: below `w_none` a target is never found,
    # above `w_full` always (before noise); linear in between. These two
    # fields are the angles AT THE SHIPPED 62° / 320 px frame; read them
    # through the `w_none` / `w_full` properties, which rescale them by this
    # spec's pixels-per-radian. Same pixels over 120° resolve half as finely,
    # so a small distant target must be found LESS often, not just as often.
    w_none_rad: float = np.deg2rad(1.0)     # ~5 px of a 320 px frame over 62°
    w_full_rad: float = np.deg2rad(4.0)     # ~21 px: always found (before noise)
    # MODELLING GAP, measured and named: `px_h` gates WHETHER a target is
    # found (the two thresholds above, read through `px_per_rad`) and does
    # NOT affect how precisely a found target is located. `DetectorNoise`
    # carries a fixed `bearing_sigma_rad` and a fixed RELATIVE
    # `width_sigma_frac`, and `range_est = radius / tan(width/2)`, so a 10%
    # width error is a ~10% range error whatever the frame size. Measured
    # over 3 seeds of 1v1, a floor ball inside 0.6 m: 320 -> 640 px moves
    # bearing error 3.35 -> 3.65 deg and range error 2.69 -> 2.62 cm, i.e.
    # nothing. On a real camera doubling the inference input halves the
    # pixel quantization of both the box centre and its width, so both
    # would genuinely improve. Anything this repo concludes about a bigger
    # inference input is therefore CONSERVATIVE: it captures the size-gate
    # half of the benefit and none of the precision half.
    # docs/camera-hardware.md 3c.
    # How the lens maps a ray to the frame, and so what a consumer that maps
    # a BOX BACK TO A BEARING gets wrong.
    #
    # "pinhole" (the default, and what every result in this repo was measured
    # on): bearings come back exact. "equidistant" models a wide lens whose
    # r = f*theta - which the replacement module is: a pinhole focal length
    # solved from its quoted H/V/D disagrees (1.65 / 2.57 / 1.04 mm) while an
    # equidistant one agrees near the quoted 2.9 mm EFL. See
    # docs/camera-hardware.md.
    #
    # The error modelled is the one a PINHOLE-CALIBRATED reader makes on such
    # a lens: it fits its focal length to the quoted FOV edge, so on-axis and
    # edge bearings come back right and everything between is pushed outward.
    # Systematic, not noise, and it grows fast with width - worst case
    # 1.2 deg at 62 deg but 9.7 deg at 116 deg, peaking near 28 deg off-axis.
    # That is larger than the chase brain's 3.4-6.9 deg aim tolerance, and it
    # is a cost of a wide lens the lens sweep never modelled because at
    # 62 deg it barely exists.
    #
    # NOTE the SIZE GATE is already equidistant-shaped: `px_per_rad` is
    # uniform across the field, which is exactly what r = f*theta gives and
    # is NOT what a pinhole lens does (a pinhole frame resolves more finely
    # toward its edges). So this field changes the BEARING only; the width
    # thresholds were always modelling the wide-lens case.
    # "pinhole" MODELS A CALIBRATED READER, and that is the baseline because
    # calibration is a one-off checkerboard, not a running cost — no one ships
    # a wide lens uncalibrated. "equidistant" is the ablation that asks what it
    # costs if you do.
    #
    # This default was "equidistant" from a9a4829 to 2026-09-09 and the flip is
    # MEASURED, because the docstring above undersells the error badly. It reads
    # as an edge-of-frame artefact; it is a GAIN. `seen_angle` is
    # atan(theta*tan(tmax)/tmax), and at tmax = 58 deg the near-axis slope is
    # tan(58 deg)/58 deg = **1.581** — a ball truly 7 deg off the nose is
    # reported at 11 deg, and EVERY bearing is inflated ~58%. Measured in play
    # (2 seeds x 60 s of 2v2, 9148 ticks with the ball visible): median bearing
    # error 3.81 deg, p90 8.08 deg, and 55% of ticks past the chase brain's
    # tightest aim tolerance (3.4 deg).
    #
    # What it costs, on `scripts/kick_gym.py`, 12 seeds x 40 episodes an arm:
    #
    #     arm                     swings   whiff   sweet spot   median |side|
    #     pinhole (calibrated)      390    17.9%      18.5%        0.063 m
    #     equidistant               402    31.1%       9.5%        0.081 m
    #
    # whiff -13.1 pp against an MDE(80%) of 8.6 pp — POWERED, p < 1e-4, and
    # pinhole is better on 11 of 12 seeds (sign test p = 0.0063). The side
    # offset IS the kick error, and it falls 0.081 -> 0.063 m.
    #
    # NEITHER ARM IS EXACT. A real calibrated fisheye keeps a residual
    # distortion error (sub-pixel RMS on a good fit) that this model has no
    # term for, so "pinhole" is the IDEALISED calibrated reader, not a promise.
    # docs/camera-hardware.md section 2 has the gap.
    projection: str = "pinhole"
    # PARTIAL VISIBILITY: the fraction of a target's angular extent that must
    # sit inside the frustum for it to be reported.
    #
    # 0.5 IS THE OLD CENTRE RULE. For a round target, centre-in-frame *is* the
    # 50% contour — measured, the two agree to the tick — so every number taken
    # before 2026-09-09 is a `partial_min` of 0.5 on the horizontal axis and on
    # the vertical axis of a point target. (Tall targets were looser still:
    # ANY vertical overlap counted. In practice a person's extent is so large
    # that it never falls under 0.25, so this is a no-op for them.)
    #
    # 0.25 models a detector that finds a truncated box, which a real YOLO does
    # — its training data is full of objects clipped by the frame edge. What the
    # centre rule was throwing away, over 72012 duck-ticks of 2v2: the vertical
    # gate rejected 12.1% of ticks, of which 27% had SOME of the ball in frame
    # and 14% had a quarter or more; the horizontal gate rejected 10.9%, of
    # which 17% and 7%.
    #
    # The cost is honest and worth stating: a box clipped at the LEFT or RIGHT
    # edge is narrower, so `width` shrinks and `range_est` (radius / tan(w/2))
    # inflates. That error is real on hardware and did not exist here before,
    # because the target was simply dropped. A ball clipped at the BOTTOM — the
    # common case, and the one that motivated this — keeps its full horizontal
    # width, so its ranging is untouched.
    partial_min: float = 0.25
    # Rays cast across the target's silhouette for the occlusion test.
    #
    # 1 IS THE OLD TEST: a single ray to the centre, which drops a target the
    # moment anything crosses its middle however much of it is plainly in view.
    # Over the same 72012 duck-ticks the occlusion gate rejected 4.8% of them,
    # and 25% of those had a QUARTER OR MORE of the ball's silhouette reachable.
    # All of it is duck-on-duck: self-occlusion by the viewer's own body fired
    # on 6 ticks in 72012, so the "nothing on the duck occludes it" note in
    # brain/controllers.py is measured and correct.
    #
    # MODELLING LIMIT: the fan is a disc of `Target.radius`, which is what that
    # field is for ("a radius that stands in for its silhouette"). For a TALL
    # target it therefore samples the waist and not the legs, so a person
    # occluded across the middle is under-reported. Nothing in the worlds this
    # repo runs occludes a person that way; recorded, not built.
    occl_rays: int = 13
    # ...and the fraction of those rays that must reach the target. At
    # `occl_rays` = 1 the fraction is 0 or 1 and any value in (0, 1] reproduces
    # the old test exactly.
    occl_min: float = 0.25
    # The visible fraction at which a target is AS EASY to find as a whole one.
    # `capture` scales the find probability by `visible / seen_full`, capped at
    # 1, so above this a partly-hidden target costs nothing and below it the
    # penalty ramps to zero at invisible.
    #
    # WHY IT IS NOT 1.0 (i.e. why the penalty is not just linear in the visible
    # fraction). Linear looked like the conservative choice and it is not: it
    # silently taxes TALL targets, whose extent overflows a 60 deg frustum by
    # design. Measured on a 1.6 m person, ideal noise, 20 captures a range:
    #
    #     range     0.4    0.6    0.8    1.0    1.5    2.0    3.0
    #     visible  1.000  0.897  0.789  0.725  0.727  0.854  1.000
    #     found     20     17     16     14     14     16     20
    #
    # A person at conversational range losing a quarter of its detections is a
    # regression in the follow brains, and it is not what a detector does: a
    # torso with the head out of frame is a plain detection. 0.5 makes that a
    # no-op (0.725 / 0.5 caps at 1) while still costing a ball clipped to a
    # quarter of itself half its chances.
    #
    # THE ONE NUMBER HERE NOBODY HAS MEASURED. It is a knob so a battery can
    # settle it; 1.0 restores the linear penalty this shipped with for an hour.
    seen_full: float = 0.5

    @staticmethod
    def from_env(spec: str | None = None) -> "DetectorSpec":
        """The defaults with `MICRODUCK_CAMERA` applied - the camera's
        `MICRODUCK_CHASE`, so a battery can say which sensor it measured
        from the same command line:

            MICRODUCK_CAMERA="bottom_pitch_deg=39.7" uv run python scripts/probe_shot_gate.py

        Numeric fields only; an unknown name or an unreadable value RAISES,
        because a typo that silently measured the shipped camera would be
        the expensive mistake here."""
        p = DetectorSpec()
        if spec is None:
            spec = os.environ.get("MICRODUCK_CAMERA", "")
        if not spec.strip():
            return p
        kinds = {f.name: getattr(p, f.name) for f in fields(p)}
        over: dict = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            k, sep, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if not sep or k not in kinds:
                raise ValueError(f"MICRODUCK_CAMERA: unknown camera field {k!r}")
            cur = kinds[k]
            if isinstance(cur, str):
                # The string fields are choices, and an unknown one RAISES for
                # the same reason a bad number does. `projection` was
                # unsettable here until 2026-09-09, so the one battery that
                # needs it - the wide lens read by a pinhole-calibrated
                # reader, i.e. the new module as it ships uncalibrated -
                # could not be run from the command line at all.
                allowed = CAMERA_CHOICES.get(k)
                if allowed is not None and v not in allowed:
                    raise ValueError(f"MICRODUCK_CAMERA: {k}={v!r} is not one of {allowed}")
                over[k] = v
                continue
            if isinstance(cur, bool) or not isinstance(cur, (int, float)):
                raise ValueError(f"MICRODUCK_CAMERA: {k!r} is not a settable field")
            try:
                over[k] = type(cur)(float(v)) if isinstance(cur, int) else float(v)
            except ValueError as e:
                raise ValueError(f"MICRODUCK_CAMERA: cannot read {k}={v!r}") from e
        return replace(p, **over)

    def seen_angle(self, true_rad: float, fov_deg: float) -> float:
        """The angle a pinhole-calibrated reader reports for a true one.

        Identity under "pinhole". Under "equidistant" the reader infers
        `atan(theta * tan(theta_max) / theta_max)`: right on axis, right at
        the edge it calibrated on, pushed outward in between - and, since
        2026-09-09, EXTRAPOLATED past that edge rather than clipped to it,
        because a partly-visible point target reports a true centre that can
        sit outside the frame. Clipping it there put the centre back on the
        frame edge; the comment below has the measurement."""
        if self.projection != "equidistant":
            return float(true_rad)
        tmax = np.deg2rad(fov_deg) / 2
        if tmax <= 1e-9:
            return float(true_rad)
        # NOT clipped to +/-tmax. It used to be, harmlessly, because every
        # angle reaching here was inside the frustum. Since 2026-09-09 a
        # partly-visible point target reports its TRUE centre, which can sit
        # just outside - and clipping snapped that centre back onto the frame
        # edge, quietly restoring on this arm the very box-centre migration
        # the partial-visibility work removed (a ball at -32.41 deg came back
        # as exactly -30.00 deg). The map is continuous and bounded past its
        # edge, so extrapolating is both defined and monotonic.
        t = float(true_rad)
        return float(np.sign(t) * np.arctan(abs(t) * np.tan(tmax) / tmax))

    @property
    def px_per_rad(self) -> float:
        """Angular resolution of this frame (296 px/rad as shipped)."""
        return float(self.px_h / np.deg2rad(self.fov_h_deg))

    @property
    def _px_coarseness(self) -> float:
        """How much coarser this frame is than the shipped one: exactly 1.0
        at 62°/320 px, 1.94 at 120°/320 px, 0.5 on a 640 px sensor."""
        return _SHIPPED_PX_PER_RAD / self.px_per_rad

    @property
    def w_none(self) -> float:
        """`w_none_rad` at this frame's resolution: an apparent width under
        this is never found.

        Note this scales an EXPLICIT `w_none_rad` too — the angle you pass is
        read through the frame like any other, not taken as final. That was
        invisible while the default spec was the calibration reference
        (coarseness 1.0) and a test asserted the identity without meaning to.
        """
        return float(self.w_none_rad * self._px_coarseness)

    @property
    def w_full(self) -> float:
        """`w_full_rad` at this frame's resolution: always found above it."""
        return float(self.w_full_rad * self._px_coarseness)


@dataclass(frozen=True)
class DetectorNoise:
    bearing_sigma_rad: float = 0.0
    width_sigma_frac: float = 0.0
    miss_p: float = 0.0            # extra P(miss) even when large and visible
    false_p: float = 0.0           # P(one spurious detection) per frame
    latency_s: float = 0.0         # frame → detection availability
    latency_jitter_s: float = 0.0
    conf_floor: float = 1.0        # confidence spread: conf ∈ [floor, 1] × visibility
    # Telling one duck's colorway from another is a colour classifier over
    # the box — the cheapest detector class anyone will add to this robot,
    # and the reason a team IS a colorway (a shell colour is the only thing
    # that distinguishes two of these robots to a camera OR to a person).
    # It fails the two ways a real one does: it gives up when the box is
    # small, and inside that range it is sometimes simply WRONG — reporting
    # another colorway, not "unknown", because a classifier with a softmax
    # always answers. A brain that trusts one frame of it deserves what it
    # gets; `Tracker` votes over the track's hits instead.
    color_range: float = math.inf  # beyond this the colour is not reported
    color_p: float = 1.0           # …inside it, P(the colour is right)

    @classmethod
    def ideal(cls) -> "DetectorNoise":
        return cls()

    @classmethod
    def datasheet(cls) -> "DetectorNoise":
        # Bearing to ~1°, width ±10 %, the measured p50/p95 latency spread,
        # a few misses on clean views, a rare ghost.
        return cls(bearing_sigma_rad=np.deg2rad(1.0), width_sigma_frac=0.10, miss_p=0.03,
                   false_p=0.005, latency_s=0.026, latency_jitter_s=0.02, conf_floor=0.6,
                   color_range=1.0, color_p=0.95)

    @classmethod
    def hostile(cls) -> "DetectorNoise":
        return cls(bearing_sigma_rad=np.deg2rad(3.0), width_sigma_frac=0.3, miss_p=0.25,
                   false_p=0.05, latency_s=0.06, latency_jitter_s=0.05, conf_floor=0.3,
                   color_range=0.6, color_p=0.75)

    @classmethod
    def preset(cls, name: str) -> "DetectorNoise":
        try:
            return {"ideal": cls.ideal, "datasheet": cls.datasheet,
                    "hostile": cls.hostile}[name]()
        except KeyError:
            raise ValueError(f"unknown detector noise preset {name!r}") from None


@dataclass(frozen=True)
class Target:
    """Something the detector can find: a body's position and a radius that
    stands in for its silhouette."""
    name: str
    cls: str
    body: int
    radius: float
    # What COLOUR this thing is, for the classes that have one: a duck's
    # colorway, i.e. its team (world/scenario.py TEAM_COLORWAYS). The
    # detector reports it, badly, at range (`DetectorNoise.color_range`).
    color: str | None = None
    # Vertical extent centred on the body (a person: its height). A detector
    # on a 24 cm-high head sees a person's legs long after its middle has
    # left the 48 deg vertical frustum (the capsule's centre leaves it at
    # 1.2 m); the part in view is what is reported. 0: a point-like thing.
    height: float = 0.0
    # A FIXED world position instead of a body: the goal posts of a pitch,
    # which are a line the World scores and not geometry in the model
    # (roadmap Track 4 s6 C.2). `body` is ignored (pass -1) and the target
    # is never "own"; occlusion is still ray-tested from the lens.
    pos: tuple[float, float, float] | None = None


@dataclass
class Detection:
    cls: str
    name: str            # which target (truth; the robot would not know) — "" for a ghost
    bearing: float       # rad, +left
    elevation: float     # rad, +up
    width: float         # rad, apparent angular width
    range_est: float     # m, from width and the class's nominal radius (what a brain may use)
    conf: float
    # The colorway the classifier read off the box, or None for "too far to
    # say" and for classes that have no colour. Unlike `name` this is NOT
    # privileged: a real camera can see a shell colour, which is the whole
    # argument for teams being colorways.
    color: str | None = None

    def as_payload(self) -> dict:
        return {"cls": self.cls, "name": self.name,
                "bearing": round(self.bearing, 4), "elevation": round(self.elevation, 4),
                "width": round(self.width, 4), "range": round(self.range_est, 3),
                "conf": round(self.conf, 3),
                **({"color": self.color} if self.color else {})}


@dataclass
class DetectionFrame:
    t: float             # sim time the frame was CAPTURED
    detections: list[Detection]
    # The camera pose the frame was taken from — height above the floor and
    # depression of the optical axis below horizontal (rad). On the robot
    # this is IMU pitch + the neck/head servo positions through the known
    # kinematics; a brain that ranges floor objects by elevation needs it,
    # because the walking gait swings the head by ±0.02 rad and holds it
    # 0.08 rad higher than the standing pose does (measured on the walker).
    cam_z: float = 0.0
    cam_pitch: float = 0.0
    cam_yaw: float = 0.0      # camera yaw relative to the BODY heading (rad): bearings are camera-frame, add this for body-frame
    # The camera's full world pose at capture (x, y, z, qw, qx, qy, qz; site
    # frame, x forward): what the /sim page renders the camera inset from,
    # so the boxes sit on what the detector saw - by the time a frame is
    # available the walking head has moved up to a fifth of the picture.
    cam_pose: tuple[float, ...] = ()


NOMINAL_RADIUS = {"duck": 0.10, "person": 0.20, "ball": 0.035, "marker": 0.05, "toy": 0.02, "basket": 0.12,
                  "post": 0.05}


class Detector:
    def __init__(self, model: mujoco.MjModel, site: str | None = None,
                 spec: DetectorSpec = DetectorSpec(), noise: DetectorNoise = DetectorNoise.ideal(),
                 targets: list[Target] | None = None, seed: int | None = 0):
        self.spec = spec if site is None else replace(spec, site=site)
        self.noise = noise
        self.model = model
        self.rng = np.random.default_rng(seed)
        # Landmarks (the `post` class) draw their find/noise from a SEPARATE
        # stream, so adding posts to a world leaves every ball and duck
        # detection - and so every soccer number ever measured - bit for bit
        # where it was. Seeded from the same seed, not drawn from `rng`.
        self.rng_land = np.random.default_rng(None if seed is None else seed + 10_007)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.spec.site)
        if self.site_id < 0:
            raise KeyError(f"site {self.spec.site!r} not in model")
        self.mount_body = int(model.site_bodyid[self.site_id])
        self.own_root = int(model.body_rootid[self.mount_body])
        self.targets: list[Target] = list(targets or [])
        self.period = 1.0 / self.spec.rate_hz
        self._next_t = 0.0
        self._pending: deque[tuple[float, DetectionFrame]] = deque()
        self.last: DetectionFrame | None = None
        # Everything occludes except toys (group 4, see world/compose.py):
        # a held toy sits right in front of the lens - and except the
        # render-only group, which is where the hinged bill lives. The camera
        # mounts on `jaw_soft` and cannot be blocked by its own beak.
        self._geomgroup = np.ones(6, dtype=np.uint8)
        self._geomgroup[4] = 0
        self._geomgroup[UNSENSED_GROUP] = 0

    # -- geometry ----------------------------------------------------------
    @staticmethod
    def _clip_arc(lo: float, hi: float, half_fov: float) -> tuple[float, float]:
        """(visible fraction, midpoint of the visible part) for the arc
        [lo, hi] clipped to +/-`half_fov`.

        The midpoint is what a clipped BOX's centre would be. ONLY the
        tall-target branch of `_visible` uses it; the point-target path
        discards it and reports the true centre on purpose, because a migrated
        centre that no consumer can tell is migrated is worse than none (see
        the note above `width`). Do not wire it into a new class without
        reading that.

        The fraction is over `min(span, 2*half_fov)` and NOT over the span,
        because the question is how much of what COULD be in frame is - a
        target bigger than the frame fills it, and a thing filling the frame is
        the easiest detection there is, not the hardest. Over the span alone a
        person at arm's length would score worse the closer it came."""
        a, b = max(lo, -half_fov), min(hi, half_fov)
        if a > b:
            return 0.0, 0.5 * (lo + hi)
        denom = min(hi - lo, 2.0 * half_fov)
        frac = 1.0 if denom <= 1e-12 else min((b - a) / denom, 1.0)
        return frac, 0.5 * (a + b)

    def _unoccluded(self, data: mujoco.MjData, tgt: Target, origin: np.ndarray,
                    p: np.ndarray, rng: float) -> float:
        """Fraction of `spec.occl_rays` cast across the target's silhouette
        that reach it. `occl_rays` = 1 is one ray to the centre, which is the
        test this detector shipped with; see `DetectorSpec.occl_rays`."""
        n = max(int(self.spec.occl_rays), 1)
        tgt_root = int(self.model.body_rootid[tgt.body]) if tgt.body >= 0 else -1
        d = p / rng
        # Any two axes spanning the plane normal to the line of sight. `d` is
        # parallel to world up only when looking straight down, hence the
        # fallback rather than a normalise of a zero vector.
        e1 = np.cross(d, np.array([0.0, 0.0, 1.0]))
        n1 = float(np.linalg.norm(e1))
        e1 = np.array([0.0, 1.0, 0.0]) if n1 < 1e-9 else e1 / n1
        e2 = np.cross(d, e1)
        pts = [p]
        for i in range(n - 1):
            a = 2.0 * np.pi * i / (n - 1)
            # Two rings, so the fan samples the middle of the disc as well as
            # its rim: a rim-only fan calls a target with a clear centre and a
            # blocked edge more occluded than it is.
            r = tgt.radius * (0.55 if i % 2 else 0.9)
            pts.append(p + r * (float(np.cos(a)) * e1 + float(np.sin(a)) * e2))
        geomid = np.zeros(1, dtype=np.int32)
        reached = 0
        for q in pts:
            qn = float(np.linalg.norm(q))
            if qn < 1e-9:
                reached += 1   # a sample AT the lens: nothing can be in front of it
                continue
            vec = np.ascontiguousarray(q / qn, dtype=np.float64)
            dist = mujoco.mj_ray(self.model, data, origin, vec, self._geomgroup, 1,
                                 self.mount_body, geomid)
            if dist >= 0 and geomid[0] >= 0:
                hit_root = int(self.model.body_rootid[self.model.geom_bodyid[geomid[0]]])
                if hit_root != tgt_root and dist < qn - tgt.radius:
                    continue
            reached += 1
        return reached / len(pts)

    def _visible(self, data: mujoco.MjData, tgt: Target,
                 origin: np.ndarray, R: np.ndarray,
                 R_frustum: np.ndarray | None = None) -> tuple[float, float, float, float, float] | None:
        """(bearing, elevation, width, range, visibility) if enough of the
        target is inside the frustum and unoccluded, else None.

        `visibility` is the part of the silhouette the frustum and the
        occluders left, on BOTH axes and after occlusion, and `capture` scales
        the find probability by it. Nothing is double-counted, because it is
        the ONLY channel partial visibility has - but WHAT IS REPORTED DIFFERS
        BY BRANCH, and the difference is the whole design (see the note above
        `width`):

        * a POINT target (`height` 0: ball, basket, toy, post, duck) reports
          its TRUE bearing, elevation and width however much is cut off. Its
          elevation may therefore lie OUTSIDE the frustum, which is the honest
          answer to "where is it" for a thing half below the frame.
        * a TALL target (`height` > 0: person) reports the midpoint of its
          VISIBLE extent, and `width` at that point's range. That is not the
          body's centre and is not meant to be - what you can see of a person
          at 0.5 m is its legs - and the gap is large: at 0.5 m the true centre
          elevation is +53.6 deg and this reports +0.6 deg. It predates the
          partial-visibility work and the follow brains are tuned on it.

        `brain/tidy.py::_locate` ranges floor objects BY elevation, so it is
        sound on the first kind and would be badly wrong on the second.

        `R` is the frame the detection is REPORTED in (the head camera's).
        `R_frustum`, when given, is the frame the frustum test is made in -
        the second, pitched-down camera of `bottom_pitch_deg` - so a target
        that lens can see comes back with the head's bearing and elevation,
        which is what every consumer already expects."""
        s = self.spec
        p = (np.asarray(tgt.pos, dtype=np.float64) if tgt.pos is not None else data.xpos[tgt.body]) - origin
        rng = float(np.linalg.norm(p))
        if rng < 1e-6 or rng > s.max_range_m:
            return None
        Rf = R if R_frustum is None else R_frustum
        local = Rf.T @ p                   # frustum camera frame: x fwd, y left, z up
        if local[0] <= 0:
            return None
        half_h = np.deg2rad(s.fov_h_deg) / 2
        half_v = np.deg2rad(s.fov_v_deg) / 2
        # Half the angular width of the silhouette. `atan`, NOT the sphere's
        # true `asin(r/d)`, so that this is the same angle `width` below is
        # built from: `frac_h` scales that width, and `range_est` inverts it as
        # `radius / tan(width/2)`, so a gate measured on a different angle
        # would disagree with the box it is gating. The two part company only
        # inside about three radii (0.10 m for a ball), where `atan` is the
        # smaller - so the error is toward finding the target LESS often.
        alpha = float(np.arctan(tgt.radius / rng))
        bearing = float(np.arctan2(local[1], local[0]))
        # NOTE the clipped midpoint is DISCARDED for a point target and the
        # true centre reported instead - see the long note above `width`.
        frac_h, _ = self._clip_arc(bearing - alpha, bearing + alpha, half_h)
        if frac_h < s.partial_min:
            return None
        if tgt.height > 0:
            # The part of a tall target inside the frustum: its top and bottom
            # in world z, through the camera's tilt, clipped to the frustum.
            up = Rf.T @ np.array([0.0, 0.0, 1.0])
            horiz = float(np.hypot(local[0], local[1]))
            lo, hi = local + up * (-tgt.height / 2), local + up * (tgt.height / 2)
            e_lo = float(np.arctan2(lo[2], horiz))
            e_hi = float(np.arctan2(hi[2], horiz))
            e_lo, e_hi = min(e_lo, e_hi), max(e_lo, e_hi)
            frac_v, elev = self._clip_arc(e_lo, e_hi, half_v)
            if frac_v < s.partial_min:
                return None
            # Range and the occlusion rays go to the point actually reported.
            p = Rf @ np.array([local[0], local[1], horiz * np.tan(elev)])
            rng = float(np.linalg.norm(p))
            # The horizontal extent belongs to the point actually REPORTED, not
            # to the body centre the gate above used - `p` has just moved and
            # `width` below is built from the new range. Re-gate on it, or a
            # tall target near the frame edge is scored on a box it does not
            # have (measured: a person at 0.5 m, frac_h 0.733 against a true
            # 0.683). `bearing` is unchanged by the reposition, which keeps
            # `local[0]` and `local[1]`, so only `alpha` moves.
            alpha = float(np.arctan(tgt.radius / rng))
            frac_h, _ = self._clip_arc(bearing - alpha, bearing + alpha, half_h)
            if frac_h < s.partial_min:
                return None
        else:
            elev = float(np.arctan2(local[2], np.hypot(local[0], local[1])))
            frac_v, _ = self._clip_arc(elev - alpha, elev + alpha, half_v)
            if frac_v < s.partial_min:
                return None
        if R_frustum is not None:
            # Seen by the second lens: report it where the HEAD would put it.
            # The clipped angles above are in the SECOND lens's frame and do
            # not transfer, so this is the centre direction, as it always was.
            rep = R.T @ p
            bearing = float(np.arctan2(rep[1], rep[0]))
            elev = float(np.arctan2(rep[2], np.hypot(rep[0], rep[1])))
        frac_occl = self._unoccluded(data, tgt, origin, p, rng)
        if frac_occl < s.occl_min:
            return None
        # THE TRUE ANGULAR SIZE AND CENTRE, DELIBERATELY NOT THE CLIPPED BOX'S.
        #
        # This applies to `bearing` and `elev` above as well as to `width`:
        # for a POINT target, partial visibility decides WHETHER it is seen,
        # never what you are told about where it is or how big it is. (A TALL
        # target still reports the midpoint of its visible part - that branch
        # predates this and is right on its own terms: what you can see of a
        # person at 0.4 m is its legs, and their centre is metres from the
        # body's.)
        #
        # A box clipped at the left or right edge really is narrower on
        # hardware, and reporting that was the first cut here. It is wrong to
        # model half of a mechanism: a real consumer KNOWS a box is truncated
        # (it touches the frame edge, and every detector API says so) and
        # distrusts its dimensions accordingly. This detector has no
        # truncation flag, so a clipped `width` hands the brain a confidently
        # WRONG `range_est` - `radius / tan(width/2)` - that nothing
        # downstream can discount.
        #
        # MEASURED, and it is not a small effect: with clipped widths the tidy
        # brain's median `range_est` to the basket went 0.346 -> 0.467 m, it
        # routed on the bad range and never picked the toy at all
        # (tests/test_tidy.py::test_tidy_picks_a_toy_behind_the_basket...).
        # Nothing else moved: a ball clipped at the BOTTOM, which is the case
        # this whole change exists for, has `frac_h` = 1 and never went
        # through here.
        #
        # The elevation is the same story and was measured the same way. A
        # clipped box's centre sits ~3 deg above the target's for a basket
        # 80% in frame, `brain/tidy.py::_locate` ranges floor objects BY
        # elevation, and reporting the migrated centre put its estimate ~2 cm
        # out and cost it the pick outright - the same test, at every value of
        # `seen_full` below 1.0.
        #
        # So the sim is optimistic by exactly one thing - it does not migrate
        # the centre or narrow the box of a truncated target - and the honest
        # fix is BOTH halves: a truncation flag on `Detection` (every real
        # detector API has one) and consumers that respect it, `_locate`
        # falling back to width-ranging when the box is cut. Modelling the
        # error without the mitigation is not the conservative half, it is
        # just wrong in a way no real system is. Recorded, not built;
        # docs/camera-hardware.md section 6.
        width = 2.0 * float(np.arctan(tgt.radius / rng))
        # The frustum tests above used the TRUE angles - what the lens can
        # physically see. What the brain receives is what a reader infers
        # from the box: the same under a pinhole model, pushed outward under
        # a wide one.
        return (s.seen_angle(bearing, s.fov_h_deg),
                s.seen_angle(elev, s.fov_v_deg), width, rng,
                frac_h * frac_v * frac_occl)

    def _color(self, tgt: "Target", range_est: float) -> str | None:
        """What the colour classifier says about this box: the truth inside
        `color_range` with probability `color_p`, another colorway otherwise,
        and nothing at all beyond the range."""
        nz = self.noise
        if not tgt.color or range_est > nz.color_range:
            return None
        if self.rng.random() < nz.color_p:
            return tgt.color
        from ..world.scenario import TEAM_COLORWAYS  # noqa: PLC0415  (cycle at import time)
        others = [c for c in TEAM_COLORWAYS if c != tgt.color]
        return str(self.rng.choice(others)) if others else None

    # -- measurement -------------------------------------------------------
    def capture(self, data: mujoco.MjData, t: float) -> DetectionFrame:
        """Run the detector on the world as it is now (no latency applied)."""
        s, nz = self.spec, self.noise
        origin = np.ascontiguousarray(data.site_xpos[self.site_id], dtype=np.float64)
        R = data.site_xmat[self.site_id].reshape(3, 3)
        R2 = None
        if s.bottom_pitch_deg > 0.0:
            # The second lens: the head frame pitched DOWN about its own left
            # axis (x fwd, y left, z up: +x goes toward -z).
            th = np.deg2rad(s.bottom_pitch_deg)
            c, sn = float(np.cos(th)), float(np.sin(th))
            R2 = R @ np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
        out: list[Detection] = []
        for tgt in self.targets:
            if tgt.body >= 0 and int(self.model.body_rootid[tgt.body]) == self.own_root:
                continue
            vis = self._visible(data, tgt, origin, R)
            if vis is None and R2 is not None:
                vis = self._visible(data, tgt, origin, R, R_frustum=R2)
            if vis is None:
                continue
            bearing, elev, width, rng, seen_frac = vis
            g = self.rng_land if tgt.cls == "post" else self.rng
            p_find = float(np.clip((width - s.w_none) / (s.w_full - s.w_none), 0.0, 1.0))
            # A partly-hidden target is harder to find. `seen_frac` is the
            # ONLY channel partial visibility has - the reported geometry is
            # the target's true geometry however much is cut off - so nothing
            # is charged twice and nothing is charged on one axis only. Full
            # marks from `seen_full` up (a half-visible thing is a plain
            # detection), then a ramp to zero. See `DetectorSpec.seen_full`.
            p_find *= min(seen_frac / s.seen_full, 1.0) if s.seen_full > 0 else 1.0
            p_find *= 1.0 - nz.miss_p
            if g.random() > p_find:
                continue
            if nz.bearing_sigma_rad:
                bearing += float(g.normal(0.0, nz.bearing_sigma_rad))
                elev += float(g.normal(0.0, nz.bearing_sigma_rad))
            if nz.width_sigma_frac:
                width *= float(np.clip(1.0 + g.normal(0.0, nz.width_sigma_frac), 0.3, 3.0))
            conf = p_find * float(g.uniform(nz.conf_floor, 1.0))
            rad = NOMINAL_RADIUS.get(tgt.cls, tgt.radius)
            range_est = rad / max(np.tan(width / 2), 1e-4)
            out.append(Detection(tgt.cls, tgt.name, bearing, elev, width, float(range_est), conf,
                                 self._color(tgt, float(range_est))))
        if nz.false_p and self.rng.random() < nz.false_p:
            cls = str(self.rng.choice(DETECT_CLASSES))
            width = float(self.rng.uniform(s.w_none, s.w_full))
            out.append(Detection(cls, "", float(self.rng.uniform(-0.5, 0.5)) * np.deg2rad(s.fov_h_deg),
                                 float(self.rng.uniform(-0.4, 0.4)) * np.deg2rad(s.fov_v_deg), width,
                                 NOMINAL_RADIUS[cls] / np.tan(width / 2), float(self.rng.uniform(0.2, 0.5))))
        Rb = data.xmat[self.own_root].reshape(3, 3)
        cam_yaw = float(np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(Rb[1, 0], Rb[0, 0]))
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R).reshape(-1))
        return DetectionFrame(t=float(t), detections=out, cam_z=float(origin[2]),
                              cam_pitch=float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0))),
                              cam_yaw=float(np.arctan2(np.sin(cam_yaw), np.cos(cam_yaw))),
                              cam_pose=tuple(float(v) for v in (*origin, *quat)))

    def sample(self, data: mujoco.MjData, t: float) -> DetectionFrame | None:
        """Rate-limited capture with latency: a frame captured at t becomes
        `last` at t + latency. Returns the frame that just became available."""
        if t + 1e-9 >= self._next_t:
            frame = self.capture(data, t)
            lat = self.noise.latency_s + abs(float(self.rng.normal(0.0, self.noise.latency_jitter_s))) \
                if self.noise.latency_jitter_s else self.noise.latency_s
            self._pending.append((t + lat, frame))
            self._next_t = (np.floor(t / self.period + 1e-9) + 1) * self.period
        got = None
        while self._pending and self._pending[0][0] <= t + 1e-9:
            got = self._pending.popleft()[1]
            self.last = got
        return got

    def age(self, t: float) -> float | None:
        return None if self.last is None else float(t - self.last.t)

    def reset(self) -> None:
        self._next_t = 0.0
        self._pending.clear()
        self.last = None
