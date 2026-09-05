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

from collections import deque
from dataclasses import dataclass, replace

import mujoco
import numpy as np

DETECT_CLASSES = ("duck", "person", "ball", "marker", "toy", "basket")

# The frame the two size thresholds below were sized on: the shipped 320 px
# YOLO11n input behind the 62° lens, 296 px/rad. A spec with another lens or
# another sensor is read against this (see `DetectorSpec.px_per_rad`).
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


@dataclass(frozen=True)
class DetectorSpec:
    fov_h_deg: float = 62.0      # ASSUMPTION: a Pi-camera-class module; the lens is still not specified
    fov_v_deg: float = 48.0      # ASSUMPTION, and 4:3-shaped: the sensor is 16:9 (see above)
    max_range_m: float = 4.0
    rate_hz: float = 10.0
    site: str = "head_camera"
    # Sensor width in pixels: the NPU runs YOLO11n on a 320×320 INT8 frame
    # (upstream npu-bringup.md). The size gate is a PIXEL fact, so widening
    # the lens without adding pixels has to cost detections — see `w_none`.
    px_h: int = SHIPPED_PX_H
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
    projection: str = "pinhole"

    def seen_angle(self, true_rad: float, fov_deg: float) -> float:
        """The angle a pinhole-calibrated reader reports for a true one.

        Identity under "pinhole". Under "equidistant" the reader infers
        `atan(theta * tan(theta_max) / theta_max)`: right on axis, right at
        the edge it calibrated on, pushed outward in between."""
        if self.projection != "equidistant":
            return float(true_rad)
        tmax = np.deg2rad(fov_deg) / 2
        if tmax <= 1e-9:
            return float(true_rad)
        t = float(np.clip(true_rad, -tmax, tmax))
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
        this is never found."""
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

    @classmethod
    def ideal(cls) -> "DetectorNoise":
        return cls()

    @classmethod
    def datasheet(cls) -> "DetectorNoise":
        # Bearing to ~1°, width ±10 %, the measured p50/p95 latency spread,
        # a few misses on clean views, a rare ghost.
        return cls(bearing_sigma_rad=np.deg2rad(1.0), width_sigma_frac=0.10, miss_p=0.03,
                   false_p=0.005, latency_s=0.026, latency_jitter_s=0.02, conf_floor=0.6)

    @classmethod
    def hostile(cls) -> "DetectorNoise":
        return cls(bearing_sigma_rad=np.deg2rad(3.0), width_sigma_frac=0.3, miss_p=0.25,
                   false_p=0.05, latency_s=0.06, latency_jitter_s=0.05, conf_floor=0.3)

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
    # Vertical extent centred on the body (a person: its height). A detector
    # on a 24 cm-high head sees a person's legs long after its middle has
    # left the 48 deg vertical frustum (the capsule's centre leaves it at
    # 1.2 m); the part in view is what is reported. 0: a point-like thing.
    height: float = 0.0


@dataclass
class Detection:
    cls: str
    name: str            # which target (truth; the robot would not know) — "" for a ghost
    bearing: float       # rad, +left
    elevation: float     # rad, +up
    width: float         # rad, apparent angular width
    range_est: float     # m, from width and the class's nominal radius (what a brain may use)
    conf: float

    def as_payload(self) -> dict:
        return {"cls": self.cls, "name": self.name,
                "bearing": round(self.bearing, 4), "elevation": round(self.elevation, 4),
                "width": round(self.width, 4), "range": round(self.range_est, 3),
                "conf": round(self.conf, 3)}


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


NOMINAL_RADIUS = {"duck": 0.10, "person": 0.20, "ball": 0.035, "marker": 0.05, "toy": 0.02, "basket": 0.12}


class Detector:
    def __init__(self, model: mujoco.MjModel, site: str | None = None,
                 spec: DetectorSpec = DetectorSpec(), noise: DetectorNoise = DetectorNoise.ideal(),
                 targets: list[Target] | None = None, seed: int | None = 0):
        self.spec = spec if site is None else replace(spec, site=site)
        self.noise = noise
        self.model = model
        self.rng = np.random.default_rng(seed)
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
        # a held toy sits right in front of the lens.
        self._geomgroup = np.ones(6, dtype=np.uint8)
        self._geomgroup[4] = 0

    # -- geometry ----------------------------------------------------------
    def _visible(self, data: mujoco.MjData, tgt: Target,
                 origin: np.ndarray, R: np.ndarray) -> tuple[float, float, float, float] | None:
        """(bearing, elevation, width, range) if inside the frustum and not
        occluded, else None."""
        p = data.xpos[tgt.body] - origin
        rng = float(np.linalg.norm(p))
        if rng < 1e-6 or rng > self.spec.max_range_m:
            return None
        local = R.T @ p                    # camera frame: x fwd, y left, z up
        if local[0] <= 0:
            return None
        bearing = float(np.arctan2(local[1], local[0]))
        if abs(bearing) > np.deg2rad(self.spec.fov_h_deg) / 2:
            return None
        half_v = np.deg2rad(self.spec.fov_v_deg) / 2
        if tgt.height > 0:
            # The part of a tall target inside the frustum: its top and bottom
            # in world z, through the camera's tilt, clipped to the frustum.
            up = R.T @ np.array([0.0, 0.0, 1.0])
            horiz = float(np.hypot(local[0], local[1]))
            lo, hi = local + up * (-tgt.height / 2), local + up * (tgt.height / 2)
            e_lo = float(np.arctan2(lo[2], horiz))
            e_hi = float(np.arctan2(hi[2], horiz))
            e_lo, e_hi = min(e_lo, e_hi), max(e_lo, e_hi)
            a, b = max(e_lo, -half_v), min(e_hi, half_v)
            if a > b:
                return None
            elev = 0.5 * (a + b)
            # Range and the occlusion ray go to the point actually reported.
            p = R @ np.array([local[0], local[1], horiz * np.tan(elev)])
            rng = float(np.linalg.norm(p))
        else:
            elev = float(np.arctan2(local[2], np.hypot(local[0], local[1])))
            if abs(elev) > half_v:
                return None
        # Occlusion: the first thing along the line of sight must be the
        # target itself (or nothing closer than its front face).
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.model, data, origin, p / rng, self._geomgroup, 1,
                             self.mount_body, geomid)
        if dist >= 0 and geomid[0] >= 0:
            hit_root = int(self.model.body_rootid[self.model.geom_bodyid[geomid[0]]])
            tgt_root = int(self.model.body_rootid[tgt.body])
            if hit_root != tgt_root and dist < rng - tgt.radius:
                return None
        width = 2.0 * float(np.arctan(tgt.radius / rng))
        # The frustum tests above used the TRUE angles - what the lens can
        # physically see. What the brain receives is what a reader infers
        # from the box: the same under a pinhole model, pushed outward under
        # a wide one.
        return (self.spec.seen_angle(bearing, self.spec.fov_h_deg),
                self.spec.seen_angle(elev, self.spec.fov_v_deg), width, rng)

    # -- measurement -------------------------------------------------------
    def capture(self, data: mujoco.MjData, t: float) -> DetectionFrame:
        """Run the detector on the world as it is now (no latency applied)."""
        s, nz = self.spec, self.noise
        origin = np.ascontiguousarray(data.site_xpos[self.site_id], dtype=np.float64)
        R = data.site_xmat[self.site_id].reshape(3, 3)
        out: list[Detection] = []
        for tgt in self.targets:
            if int(self.model.body_rootid[tgt.body]) == self.own_root:
                continue
            vis = self._visible(data, tgt, origin, R)
            if vis is None:
                continue
            bearing, elev, width, rng = vis
            p_find = float(np.clip((width - s.w_none) / (s.w_full - s.w_none), 0.0, 1.0))
            p_find *= 1.0 - nz.miss_p
            if self.rng.random() > p_find:
                continue
            if nz.bearing_sigma_rad:
                bearing += float(self.rng.normal(0.0, nz.bearing_sigma_rad))
                elev += float(self.rng.normal(0.0, nz.bearing_sigma_rad))
            if nz.width_sigma_frac:
                width *= float(np.clip(1.0 + self.rng.normal(0.0, nz.width_sigma_frac), 0.3, 3.0))
            conf = p_find * float(self.rng.uniform(nz.conf_floor, 1.0))
            rad = NOMINAL_RADIUS.get(tgt.cls, tgt.radius)
            range_est = rad / max(np.tan(width / 2), 1e-4)
            out.append(Detection(tgt.cls, tgt.name, bearing, elev, width, float(range_est), conf))
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
