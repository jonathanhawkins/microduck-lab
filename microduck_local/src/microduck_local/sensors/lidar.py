"""MARS's 360-degree planar LiDAR, simulated at the device's own rate.

The real robot carries a 2-D scanner on the chassis lid — 0.15-6 m, 6 Hz,
Apache-2.0 `innate-os` publishes it as a ROS `LaserScan` and
`mars_sim_driver/core.lidar_scan` simulates it with `mj_multiRay`
(`docs/mars-roadmap.md` §0). This module is that scan on this repo's sensor
stack: a `RayFan` of `planar_fan(n, 360, ccw=True)` directions cast from the
`base_laser` frame, through a noise model that goes from *ideal* to *hostile*
live, rate-limited the way `sensors/tof.TofSensor` is.

**The convention is Innate's**: ray k points `k * 360/n` degrees
COUNTER-CLOCKWISE from the mount frame's +x, which on this URDF is the
robot's own +x (MEASURED: `base_laser`'s rotation relative to `base_link` is
the identity). A frame therefore carries its `angles`, in the MOUNT frame,
and a consumer that needs world bearings composes the mount rotation — the
same contract `TofFrame` has with `mount_rot`.

**What it is blind to, measured.** `RayFan` excludes its mount body, and for
this sensor that is not enough: `base_laser` is a jointless frame that sits
INSIDE `base_turret`, the collision box modelling the scanner's own housing,
and that box is a geom of `base_link`. MEASURED on `mars.model()` at HOME —
excluding only `base_laser`, **all 360 rays return 0.039-0.101 m off
`base_turret`** and the scan is useless. So the exclusion defaults to the
mount's PARENT body, which is what Innate's own `lidar_scan` excludes
(`self._base_id`), and which is the honest model besides: a scanner is blind
to the platform it is bolted to, not to the 5 mm frame it is defined by.

The REST of the robot is still visible, and one part of it really is in the
way: at `ARM_HOME` the folded arm's `link5` blocks **4 of 360 rays, at 7-10
degrees, from 0.156 m** — the arm parked over the chassis, exactly as it
would occlude the real scanner. That shadow moves with the arm, so a brain
that treats "something at 0.16 m" as an obstacle will brake for MARS's own
elbow unless it ignores returns inside the footprint. It is reported here
rather than hidden because hiding it would make the sim kinder than the
robot: `tests/test_lidar.py` pins the count and the arc.

MEASURED on this Mac (`mars.model()`, one MARS, 360 rays, 6 m):

    one scan                    28.6-30.4 us
    polled every physics step   1.2-1.7 us a step at 6 Hz — 8-10 % of a
                                driven MARS's ~14 us step
    0 deg at a 2 m wall         2.0264 m, predicted 2.0264 (2.0 - 0.05 of
                                box - the laser's -0.0764 m x offset)
    the robot in its own scan   4 rays of 360, 7-10 deg, 0.1562 m off the
                                folded arm's `link5`; nothing else

**Noise is NOT measured.** `LidarNoise.datasheet()` is a placeholder until
the RPLIDAR datasheet is read (the class of scanner Innate ships is not
named in `innate-os`): a 1 cm sigma at 1 m rising with range, and a small
dropout probability, which is the shape of every planar scanner's error but
not this one's numbers. `ideal` is the default for that reason — a brain
tuned against invented noise is tuned against nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .ray import DEFAULT_GROUPS, RayFan, planar_fan

#: The device, from `docs/mars-roadmap.md` §0 (Innate's published spec).
DEFAULT_N_RAYS = 360
DEFAULT_FOV_DEG = 360.0
DEFAULT_MAX_RANGE_M = 6.0
DEFAULT_MIN_RANGE_M = 0.15
DEFAULT_RATE_HZ = 6.0
#: The URDF frame the scanner sits on — `robots/mars.LIDAR_SITE`, named there
#: as a site and kept as a BODY by the `fusestatic="false"` rewrite. Imported
#: by name rather than from `robots.mars` so this module stays a sensor: the
#: sensor package does not import the robot package anywhere else.
DEFAULT_MOUNT = "base_laser"


@dataclass(frozen=True)
class LidarNoise:
    """Per-ray measurement noise. All optional, all zero = ideal.

    NOT MEASURED — a placeholder until the RPLIDAR datasheet is read. The
    presets below are the right SHAPE (a range-proportional sigma and a
    dropout that worsens with distance, which is what a triangulation
    scanner does) with invented magnitudes.

    `sigma_*` and `dropout_*` apply to real returns only; `outlier_p` can
    also invent one where the beam hit nothing (`LidarSensor.scan` says why).
    """

    sigma_m: float = 0.0        # constant part of the Gaussian std, m
    sigma_frac: float = 0.0     # range-proportional part (0.005 = 0.5 % of d)
    dropout_near: float = 0.0   # P(ray invalid) at zero range …
    dropout_far: float = 0.0    # … rising quadratically to this at max range
    outlier_p: float = 0.0      # P(ray replaced by a uniform random range)

    @classmethod
    def ideal(cls) -> "LidarNoise":
        return cls()

    @classmethod
    def datasheet(cls) -> "LidarNoise":
        """~1 cm sigma at 1 m, rising to ~3.5 cm at 6 m, 1 % dropout far.

        NOT MEASURED — a placeholder until the RPLIDAR datasheet is read.
        `sigma_m + sigma_frac * d` is 0.010 m at 1 m and 0.035 m at 6 m.
        """
        return cls(sigma_m=0.005, sigma_frac=0.005, dropout_near=0.001,
                   dropout_far=0.01, outlier_p=0.0005)

    @classmethod
    def hostile(cls) -> "LidarNoise":
        """Three times the sigma and a tenth of the rays gone.

        NOT MEASURED either — this is the preset a brain should survive, not
        a claim about a device: dark carpet, a glass door and a sunlit window
        are all a dropout to a scanner, and none of them is modelled.
        """
        return cls(sigma_m=0.015, sigma_frac=0.015, dropout_near=0.01,
                   dropout_far=0.10, outlier_p=0.005)

    @classmethod
    def preset(cls, name: str) -> "LidarNoise":
        try:
            return {"ideal": cls.ideal, "datasheet": cls.datasheet,
                    "hostile": cls.hostile}[name]()
        except KeyError:
            raise ValueError(f"unknown lidar noise preset {name!r}") from None


@dataclass
class LidarFrame:
    """One scan.

    `ranges` is metres, `max_range` where a ray hit nothing — the LaserScan
    convention Innate's driver publishes, and the reason a miss is not a NaN:
    "nothing within 6 m" is information, and a consumer that averages a bin
    should get the far wall, not a hole.

    A return closer than `min_range` is CLIPPED to `min_range` and marked
    invalid: the device cannot resolve nearer than that, and clipping errs
    toward "there is something very close", which is the safe direction for
    anything that brakes. `valid` is False there and wherever the noise model
    dropped a ray.
    """

    t: float                    # sim time the scan was taken
    ranges: np.ndarray          # (n,) float32, m
    angles: np.ndarray          # (n,) float32, rad CCW from the mount's +x
    valid: np.ndarray           # (n,) bool
    truth_m: np.ndarray = field(repr=False, default=None)   # noise-free
    # Where the scanner is in the base body's HEADING frame (yaw only: the
    # frame a brain's odometry lives in), so a bin's hit can be placed at
    # odom (+) mount_pos (+) R(angle) * range. On MARS this is the 76 mm the
    # laser sits BEHIND the base origin, which is the difference between a
    # 2.03 m and a 1.95 m reading of the same wall. None without `base_body`.
    mount_pos: np.ndarray | None = field(repr=False, default=None)   # (3,)

    def as_payload(self) -> dict:
        """Wire shape for the lab's frame stream (small; the viewer draws it).

        Millimetre integers, `TofFrame.as_payload`'s units and its convention
        that **0 means no reading** — an invalid ray, not a zero-range one.
        `a0`/`da` are the first bearing and the increment in radians, so a
        consumer can rebuild every angle without shipping 360 floats it
        already knows.
        """
        mm = np.where(self.valid, np.round(self.ranges * 1000.0), 0)
        da = 0.0 if self.angles.size < 2 else float(self.angles[1] - self.angles[0])
        return {"t": round(self.t, 4),
                "a0": round(float(self.angles[0]), 5) if self.angles.size else 0.0,
                "da": round(da, 6),
                "mm": mm.astype(np.uint16).tolist()}


class LidarSensor:
    """A planar scanner on one body of a compiled model.

    `mount` is the body the rays leave from — `"base_laser"` for a
    standalone MARS, `"m0/base_laser"` for one attached in a room, which is
    all the prefixing a `/sim` world needs. `exclude_body` is what the beam
    cannot see; it defaults to the mount's PARENT for the reason in the
    module docstring, and passing a name overrides it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        mount: str = DEFAULT_MOUNT,
        *,
        n_rays: int = DEFAULT_N_RAYS,
        fov_deg: float = DEFAULT_FOV_DEG,
        max_range: float = DEFAULT_MAX_RANGE_M,
        min_range: float = DEFAULT_MIN_RANGE_M,
        rate_hz: float = DEFAULT_RATE_HZ,
        noise: LidarNoise = LidarNoise.ideal(),
        seed: int | None = 0,
        groups: tuple[int, ...] = DEFAULT_GROUPS,
        exclude_body: str | None = None,
        base_body: str | None = None,
    ):
        if n_rays < 1:
            raise ValueError("n_rays must be >= 1")
        if not 0.0 <= min_range < max_range:
            raise ValueError("need 0 <= min_range < max_range")
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        self.mount = mount
        self.n_rays = int(n_rays)
        self.fov_deg = float(fov_deg)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.rate_hz = float(rate_hz)
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._model = model
        if exclude_body is None:
            exclude_body = self._parent_body(model, mount)
        self.fan = RayFan(model, planar_fan(self.n_rays, self.fov_deg, ccw=True),
                          body=mount, max_range=self.max_range, groups=groups,
                          exclude_body=exclude_body)
        self.exclude_body = exclude_body
        #: (n,) bearings in the MOUNT frame, rad, CCW from +x. Fixed for the
        #: life of the sensor — a scanner's bins do not move, the mount does.
        self.angles = np.arctan2(self.fan.dirs_local[:, 1],
                                 self.fan.dirs_local[:, 0]).astype(np.float32)
        self.base_body = None
        if base_body is not None:
            self.base_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, base_body)
            if self.base_body < 0:
                raise KeyError(f"base_body {base_body!r} not in model")
        self.period = 1.0 / self.rate_hz
        self.last: LidarFrame | None = None
        self._next_t = 0.0
        self._last_hits = None

    @staticmethod
    def _parent_body(model: mujoco.MjModel, mount: str) -> str:
        """The mount's parent body name — the platform the scanner rides on.

        Resolved from the model rather than hard-coded to `base_link` so a
        prefixed MARS (`"m0/base_laser"` -> `"m0/base_link"`) and any later
        body that mounts a scanner both get the right answer with no id table.
        """
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mount)
        if bid < 0:
            raise KeyError(f"mount body {mount!r} not in model")
        parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   int(model.body_parentid[bid]))
        if not parent or parent == "world":
            # A scanner mounted straight on the world has no platform to be
            # blind to; exclude itself, which is `RayFan`'s own default.
            return mount
        return parent

    # -- measurement -------------------------------------------------------
    def scan(self, data: mujoco.MjData, t: float = 0.0) -> LidarFrame:
        """One scan, right now, regardless of rate. Applies the noise model.

        `TofSensor.measure`'s job, with a scanner's differences: no sub-rays
        to reduce (a beam is a line, not a zone patch) and a miss reports
        `max_range` rather than an invalid zone.
        """
        nz = self.noise
        hits = self.fan.cast(data)
        self._last_hits = hits
        got = hits.hit                      # a real return, not a max-range miss
        truth = np.where(hits.dist >= 0.0, hits.dist, self.max_range)
        meas = truth.astype(np.float64).copy()
        # Below the device's minimum is a reading it cannot make. Marked from
        # the TRUTH, before noise, so a sigma cannot argue a real 0.10 m
        # return up into the valid band.
        valid = truth >= self.min_range
        # Sigma and dropout apply to RETURNS only: a miss is not a
        # measurement, so it has no measurement error and nothing to drop —
        # it is the statement "nothing within max_range", and jittering it
        # would be inventing a wall to be uncertain about. An outlier is the
        # exception below, because a spurious return where there was none
        # (sunlight, a specular edge) is exactly what that term stands for.
        if nz.sigma_m or nz.sigma_frac:
            std = nz.sigma_m + nz.sigma_frac * truth
            meas = meas + got * self.rng.normal(0.0, 1.0, truth.shape) * std
        if nz.dropout_near or nz.dropout_far:
            frac = np.clip(truth / self.max_range, 0.0, 1.0)
            p = nz.dropout_near + (nz.dropout_far - nz.dropout_near) * frac ** 2
            valid &= (self.rng.random(truth.shape) >= p) | ~got
        if nz.outlier_p:
            out = self.rng.random(truth.shape) < nz.outlier_p
            meas = np.where(out, self.rng.uniform(self.min_range, self.max_range,
                                                  truth.shape), meas)
        meas = np.clip(meas, self.min_range, self.max_range)
        mount_pos = None
        if self.base_body is not None:
            bq = data.xquat[self.base_body]
            yaw = float(np.arctan2(2.0 * (bq[0] * bq[3] + bq[1] * bq[2]),
                                   1.0 - 2.0 * (bq[2] ** 2 + bq[3] ** 2)))
            c, sn = np.cos(yaw), np.sin(yaw)
            rh = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
            mount_pos = rh.T @ (self.fan.origin(data) - data.xpos[self.base_body])
        return LidarFrame(t=float(t), ranges=meas.astype(np.float32),
                          angles=self.angles, valid=valid,
                          truth_m=truth.astype(np.float32), mount_pos=mount_pos)

    def maybe_scan(self, data: mujoco.MjData, t: float) -> LidarFrame | None:
        """Rate-limited: a new scan when one is due at `rate_hz`, else None.

        `TofSensor.sample`, including the reason it schedules from the grid
        and not from `t`: a world loop polls every physics step, and adding a
        period to the poll time would let a 6 Hz sensor drift into a 5.9 Hz
        one over a long run. `self.last` always holds the newest scan;
        `age(t)` its staleness — at 6 Hz that is up to 167 ms, which is long
        enough to matter to anything that drives at 0.8 m/s (13 cm).
        """
        if t + 1e-9 < self._next_t:
            return None
        frame = self.scan(data, t)
        self.last = frame
        self._next_t = (np.floor(t / self.period + 1e-9) + 1) * self.period
        return frame

    def age(self, t: float) -> float | None:
        return None if self.last is None else float(t - self.last.t)

    def reset(self) -> None:
        self.last = None
        self._next_t = 0.0
        self._last_hits = None

    def hit_points(self, data: mujoco.MjData) -> np.ndarray | None:
        """World points of the last cast's rays (for the /sim overlay)."""
        if self._last_hits is None:
            return None
        return self.fan.hit_points(data, self._last_hits)


__all__ = ["DEFAULT_MAX_RANGE_M", "DEFAULT_MIN_RANGE_M", "DEFAULT_MOUNT",
           "DEFAULT_N_RAYS", "DEFAULT_RATE_HZ", "LidarFrame", "LidarNoise",
           "LidarSensor"]
