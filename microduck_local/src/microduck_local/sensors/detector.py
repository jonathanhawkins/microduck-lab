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


# WHAT THE SENSOR ACTUALLY IS (2026-09, from the developer + its datasheet):
# 1/2.9", 2.75 um BSI pixels, 1920 x 1080 active, up to 90 fps. Derived:
# 5.28 x 2.97 mm active area, 6.06 mm diagonal, 16:9.
#
# Three things follow, and only the first two are answers.
#
# 1. THE SENSOR IS NOT THE PIXEL BUDGET. `px_h` below is the NPU's YOLO11n
#    INT8 input, and the sensor carries 1920 px across it - SIX TIMES what
#    the detector consumes. So "a wide lens is worth buying only with the
#    sensor to pay for it" (README) is loosely worded: what has to pay is
#    the INFERENCE INPUT, and this sensor can already feed a bigger one.
#    The 640 px arm measured in that sweep is an NPU question ("can it run
#    YOLO11n at 640, and at what rate"), not a new-hardware question.
# 2. FRAME RATE IS A NON-ISSUE. `rate_hz` is 10 and the brain decides at
#    10 Hz; the sensor offers 90. Nine times the headroom, so the "not sure
#    we have the compute for the full potential" caveat does not bite this
#    workload - frames can be dropped freely.
# 3. THE FOV IS STILL UNKNOWN for the new part, and it is the assumption
#    that matters. Sensor size does not give field of view without the
#    lens's focal length. On this active area: 62 deg needs f = 4.4 mm,
#    90 deg 2.6 mm, 120 deg 1.5 mm.
#
# AND THE CAMERA THE ROBOT HAS TODAY IS A DIFFERENT SENSOR, WITH A FOV
# THIS SPEC IS PROBABLY WRONG ABOUT. Upstream identifies it on hardware:
# `imx219 2-0010: Model ID 0x0219` (microduck/docs/project/media-bringup.md),
# i.e. a Raspberry Pi Camera v2 - 1/4", 1.12 um, 3280 x 2464, whose quoted
# 62.2 x 48.8 deg is where the 62/48 below came from. But `mediad`'s
# `pin_sensor_mode` pins the SENSOR to 1920x1080 (pipeline.rs), and on the
# IMX219 that mode is a CROP - libcamera reports it as
# "1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]", a centred window of
# the array, no binning and no scaling. It keeps 59% of the columns and
# 44% of the rows. With the stock f = 3.04 mm lens (confirmed by geometry:
# the full array gives 62.3 x 48.8, matching the published spec):
#
#     full array 3280x2464   62.3 x 48.8 deg   <- what this spec assumes
#     PINNED MODE 1920x1080  39.0 x 22.5 deg   <- what the robot may see
#
# So the frustum here may be ~23 deg too wide and ~25 deg TOO TALL. The
# vertical error is the one that bites: 22.5 deg of vertical view is half
# what the near-field reasoning assumes (a floor ball leaving the camera in
# the last 0.3 m, a person's middle leaving the frustum at 1.2 m).
#
# The same crop cuts the other way on resolution: 320 px over 39 deg is
# 471 px/rad against the 296 assumed here, because a narrow view
# concentrates pixels. Both errors push the same conclusion - the lens
# sweep's "the shipped camera is adequate" was measured on a baseline
# WIDER and BLURRIER than the robot's, so a wider lens is probably worth
# more than that battery said, not less.
#
# UNCONFIRMED, and it is one question: is the module the stock Pi Camera
# v2 (f = 3.04 mm), or a third-party IMX219 board? Those ship M12 lenses
# from 88 to 160 deg and would change every number above. The repo names
# the driver overlay, not the lens. Do not re-baseline the sim on 39 x
# 22.5 until someone confirms the lens - the point here is that 62 x 48 is
# the FULL-ARRAY figure and the robot does not run the full array.
#
# The new sensor's headline advantage falls out of this: 1920 x 1080 is
# its NATIVE array, so there is no crop to lose, and whatever the lens
# gives is what the robot sees.
#
# And it raises one: the frustum below is 62 x 48 deg, whose shape
# (tan24/tan31 = 0.74) is 4:3. THIS SENSOR IS 16:9 (0.56). At 62 deg
# horizontal a 16:9 frame gives 37 deg vertical, not 48 - so this spec may
# be ~11 deg too generous VERTICALLY, which is the axis the near-field
# blindness lives on (a floor ball leaves the camera in the last 0.3 m; a
# person's middle leaves the frustum at 1.2 m). If so the sim is optimistic
# about close-range vision. Unresolved: ask for the lens focal length or
# quoted FOV, and whether the 16:9 frame is letterboxed into the 320x320
# YOLO input (keeps the FOV, wastes pixels) or centre-cropped (keeps the
# pixels, loses horizontal FOV) - the two give different px/rad AND a
# different effective HFOV.


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
        return bearing, elev, width, rng

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
