"""The head's 8×8 time-of-flight matrix, simulated at the device's own rate.

The robot carries one VL53L5CX/L8CX on the head HAT, served by `tofd` as a
`tof.stream` subscription of 8×8 distances (upstream architecture.md). This
module produces frames shaped the same way — `depth_mm` as uint16 with 0 for
"no target", plus a validity mask — from `RayFan` casts on the MJCF's `tof`
site, at 15 Hz (the device's 8×8 rate), through a noise model you can turn
from *ideal* to *hostile* live.

Geometry defaults are the datasheet's: ~45° square field of view (≈63–65°
diagonal), 4 m maximum range, a few cm minimum. Each zone integrates a small
patch of sub-rays and reports their median, which is close to how the device
reports the dominant target in a zone and keeps a zone that straddles an
edge from flickering.

What this deliberately does NOT model: multi-target returns, ambient light,
reflectivity. Those are what the *hostile* preset's dropouts stand in for,
until real `tof.stream` logs say otherwise (roadmap 11.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .ray import RayFan, tof_fan


@dataclass(frozen=True)
class TofSpec:
    rows: int = 8
    cols: int = 8
    fov_deg: float = 45.0
    max_range_m: float = 4.0
    min_range_m: float = 0.02
    rate_hz: float = 15.0
    subrays: int = 2            # per axis → 4 rays per zone, 256 per frame
    site: str = "tof"           # mount site name (prefixed per duck in a world)


@dataclass(frozen=True)
class TofNoise:
    """Per-zone measurement noise. All optional, all zero = ideal."""
    sigma_mm: float = 0.0       # constant part of the Gaussian std, mm
    sigma_frac: float = 0.0     # range-proportional part (0.01 = 1 % of d)
    dropout_near: float = 0.0   # P(invalid) at zero range …
    dropout_far: float = 0.0    # … rising quadratically to this at max range
    outlier_p: float = 0.0      # P(zone replaced by a uniform random range)
    warmup_s: float = 0.0       # frames before this are invalid (firmware upload)

    @classmethod
    def ideal(cls) -> "TofNoise":
        return cls()

    @classmethod
    def datasheet(cls) -> "TofNoise":
        # VL53L5CX-class figures at 8×8/15 Hz on a white target, roughly.
        return cls(sigma_mm=5.0, sigma_frac=0.01, dropout_near=0.005,
                   dropout_far=0.15, outlier_p=0.002, warmup_s=0.5)

    @classmethod
    def hostile(cls) -> "TofNoise":
        return cls(sigma_mm=15.0, sigma_frac=0.03, dropout_near=0.05,
                   dropout_far=0.6, outlier_p=0.02, warmup_s=1.0)

    @classmethod
    def preset(cls, name: str) -> "TofNoise":
        try:
            return {"ideal": cls.ideal, "datasheet": cls.datasheet,
                    "hostile": cls.hostile}[name]()
        except KeyError:
            raise ValueError(f"unknown ToF noise preset {name!r}") from None


@dataclass
class TofFrame:
    t: float                    # sim time the frame was taken
    depth_mm: np.ndarray        # (rows, cols) uint16, 0 = no target
    valid: np.ndarray           # (rows, cols) bool
    truth_m: np.ndarray = field(repr=False, default=None)  # noise-free, -1 = miss
    # Where the sensor was, in the base body's HEADING frame (yaw only: the
    # frame a brain's odometry lives in): origin and rotation, so a brain
    # can place a zone's hit at odom ⊕ mount_pos + mount_rot · dir · depth.
    # On the robot this is the neck/head servo positions + IMU pitch/roll
    # through the known kinematics. None when the sensor has no base body.
    mount_pos: np.ndarray | None = field(repr=False, default=None)     # (3,)
    mount_rot: np.ndarray | None = field(repr=False, default=None)     # (3, 3)
    dirs_local: np.ndarray | None = field(repr=False, default=None)    # (rows, cols, 3) zone directions, sensor frame

    def as_payload(self) -> dict:
        """Wire shape for the lab's frame stream (small; the viewer draws it)."""
        return {"t": round(self.t, 4),
                "mm": self.depth_mm.reshape(-1).tolist()}


class TofSensor:
    def __init__(
        self,
        model: mujoco.MjModel,
        site: str | None = None,
        spec: TofSpec = TofSpec(),
        noise: TofNoise = TofNoise.ideal(),
        seed: int | None = 0,
        base_body: int | None = None,
    ):
        self.spec = spec if site is None else replace(spec, site=site)
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.base_body = base_body          # frames are stamped with the mount pose relative to it
        self._model = model
        s = self.spec
        self.fan = RayFan(model, tof_fan(s.rows, s.cols, s.fov_deg, s.subrays),
                          site=s.site, max_range=s.max_range_m)
        self._zone_index = self._build_zone_index()
        self._dirs_cache = self.zone_dirs_local()
        self.period = 1.0 / s.rate_hz
        self.last: TofFrame | None = None
        self._next_t = 0.0
        self._t0 = None
        self._last_hits = None

    # -- geometry ----------------------------------------------------------
    def _build_zone_index(self) -> np.ndarray:
        """(rows, cols, subrays²) → ray index, from tof_fan's raster order."""
        s = self.spec
        S = s.subrays
        idx = np.empty((s.rows, s.cols, S * S), dtype=np.int64)
        for r in range(s.rows):
            for c in range(s.cols):
                k = 0
                for i in range(S):
                    for j in range(S):
                        idx[r, c, k] = (r * S + i) * (s.cols * S) + c * S + j
                        k += 1
        return idx

    def zone_dirs_local(self) -> np.ndarray:
        """(rows, cols, 3) centre direction per zone, mount frame."""
        d = self.fan.dirs_local[self._zone_index]      # (r, c, S², 3)
        m = d.mean(axis=2)
        return m / np.linalg.norm(m, axis=2, keepdims=True)

    # -- measurement -------------------------------------------------------
    def measure(self, data: mujoco.MjData, t: float = 0.0) -> TofFrame:
        """One frame, right now, regardless of rate. Applies the noise model."""
        s, nz = self.spec, self.noise
        hits = self.fan.cast(data)
        self._last_hits = hits
        d = hits.dist[self._zone_index]                # (r, c, S²), -1 = miss
        hit = d >= 0
        # A zone needs at least half its sub-rays to return to count.
        n_hit = hit.sum(axis=2)
        valid = n_hit * 2 >= d.shape[2]
        # Lower median of the sub-rays that hit: misses sort to the end and
        # the pick index only counts hits, so no NaN arithmetic is needed.
        srt = np.sort(np.where(hit, d, np.inf), axis=2)
        pick = np.maximum(n_hit - 1, 0) // 2
        med = np.take_along_axis(srt, pick[..., None], axis=2)[..., 0]
        truth = np.where(valid, med, -1.0)
        meas = truth.copy()
        # Range limits: too close saturates, too far is a miss.
        valid &= (truth >= s.min_range_m) & (truth <= s.max_range_m)
        if nz.sigma_mm or nz.sigma_frac:
            std = nz.sigma_mm / 1000.0 + nz.sigma_frac * np.maximum(truth, 0.0)
            meas = meas + self.rng.normal(0.0, 1.0, truth.shape) * std
        if nz.dropout_near or nz.dropout_far:
            frac = np.clip(truth / s.max_range_m, 0.0, 1.0)
            p = nz.dropout_near + (nz.dropout_far - nz.dropout_near) * frac ** 2
            valid &= self.rng.random(truth.shape) >= p
        if nz.outlier_p:
            out = self.rng.random(truth.shape) < nz.outlier_p
            meas = np.where(out, self.rng.uniform(s.min_range_m, s.max_range_m, truth.shape), meas)
        if nz.warmup_s:
            if self._t0 is None:
                self._t0 = t
            if t - self._t0 < nz.warmup_s:
                valid[:] = False
        meas = np.clip(meas, s.min_range_m, s.max_range_m)
        depth = np.where(valid, np.round(meas * 1000.0), 0).astype(np.uint16)
        mount_pos = mount_rot = None
        if self.base_body is not None:
            bp = data.xpos[self.base_body]
            bq = data.xquat[self.base_body]
            yaw = float(np.arctan2(2.0 * (bq[0] * bq[3] + bq[1] * bq[2]), 1.0 - 2.0 * (bq[2] ** 2 + bq[3] ** 2)))
            c, sn = np.cos(yaw), np.sin(yaw)
            Rh = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
            mount_pos = Rh.T @ (self.fan.origin(data) - bp)
            mount_rot = Rh.T @ self.fan.rotation(data)
        return TofFrame(t=float(t), depth_mm=depth, valid=valid, truth_m=truth,
                        mount_pos=mount_pos, mount_rot=mount_rot, dirs_local=self._dirs_cache)

    def sample(self, data: mujoco.MjData, t: float) -> TofFrame | None:
        """Rate-limited: a new frame when one is due at `rate_hz`, else None.
        `self.last` always holds the newest frame; `age(t)` its staleness."""
        if t + 1e-9 < self._next_t:
            return None
        frame = self.measure(data, t)
        self.last = frame
        # Schedule from the grid, not from `t`, so a late poll doesn't drift.
        self._next_t = (np.floor(t / self.period + 1e-9) + 1) * self.period
        return frame

    def age(self, t: float) -> float | None:
        return None if self.last is None else float(t - self.last.t)

    def reset(self) -> None:
        self.last = None
        self._next_t = 0.0
        self._t0 = None
        self._last_hits = None

    def hit_points(self, data: mujoco.MjData) -> np.ndarray | None:
        """World points of the last cast's rays (for the /sim overlay)."""
        if self._last_hits is None:
            return None
        return self.fan.hit_points(data, self._last_hits)

    def zone_points(self, data: mujoco.MjData, frame: TofFrame | None = None) -> np.ndarray:
        """(rows, cols, 3) world points at each zone's REPORTED depth along
        the zone's centre ray, NaN where the zone is invalid — what the
        sensor claims to see, drawn where it claims it is. Uses the mount's
        pose NOW; pair it with a fresh frame for the overlay."""
        frame = self.last if frame is None else frame
        s = self.spec
        if frame is None:
            return np.full((s.rows, s.cols, 3), np.nan)
        dirs = self.zone_dirs_local() @ self.fan.rotation(data).T
        depth = frame.depth_mm.astype(np.float64) / 1000.0
        pts = self.fan.origin(data)[None, None, :] + dirs * depth[..., None]
        pts[~frame.valid] = np.nan
        return pts
