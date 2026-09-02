"""Ray rigs on `mj_multiRay`: the primitive under every range sensor here.

A `RayFan` is a bundle of unit directions fixed in a mount frame (a site or a
body) and cast from that frame's origin — one aperture, many rays, which is
exactly what a time-of-flight matrix or a scanning LiDAR is. The 8×8 ToF on
the duck's head is one fan (`tof_fan`); a planar scan would be another
(`planar_fan`). Nothing else in the sensor stack touches MuJoCo's ray API.

Conventions (from the upstream MJCF, verified in the STAND pose): the `tof`
and `head_camera` sites use x-forward / y-left / z-up frames, so a fan is
authored around +x. The MJCF's own `<camera>` looks along -x, which is an
upstream quirk this module ignores.

The mount body is excluded from hits (a sensor cannot see its own housing);
the REST of the robot is not — a duck looking down at its feet sees its feet,
as the real sensor would.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Geom groups a range sensor "sees": 0 = scenery (floor, walls, objects),
# 2 = the robot's visual meshes (upstream puts the shells there), 3 = the
# collision pads, 4 = toys (`world.compose.PICKABLE_GROUP`). Group 1/5 are
# left for sensor-only proxies later.
DEFAULT_GROUPS: tuple[int, ...] = (0, 2, 3, 4)


@dataclass
class RayHits:
    """Result of one cast: per ray, the distance along it (m; -1 = no hit
    within `max_range`) and the geom id hit (-1 = none)."""

    dist: np.ndarray    # (N,) float64
    geomid: np.ndarray  # (N,) int32

    @property
    def hit(self) -> np.ndarray:
        return self.geomid >= 0


class RayFan:
    def __init__(
        self,
        model: mujoco.MjModel,
        dirs_local: np.ndarray,
        *,
        site: str | None = None,
        body: str | None = None,
        max_range: float = 4.0,
        groups: tuple[int, ...] = DEFAULT_GROUPS,
    ):
        if (site is None) == (body is None):
            raise ValueError("give exactly one of site= or body=")
        d = np.asarray(dirs_local, dtype=np.float64).reshape(-1, 3)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        if np.any(n == 0):
            raise ValueError("zero-length ray direction")
        self.dirs_local = d / n
        self.n = int(self.dirs_local.shape[0])
        self.max_range = float(max_range)
        self.model = model
        if site is not None:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            if self.site_id < 0:
                raise KeyError(f"site {site!r} not in model")
            self.body_id = int(model.site_bodyid[self.site_id])
        else:
            self.site_id = -1
            self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if self.body_id < 0:
                raise KeyError(f"body {body!r} not in model")
        gg = np.zeros(6, dtype=np.uint8)
        for g in groups:
            gg[int(g)] = 1
        self.geomgroup = gg
        # Scratch buffers: mj_multiRay writes in place, and a fan is cast
        # every sensor tick for the life of the world.
        self._vec = np.empty(self.n * 3, dtype=np.float64)
        self._geomid = np.empty(self.n, dtype=np.int32)
        self._dist = np.empty(self.n, dtype=np.float64)

    # -- geometry ----------------------------------------------------------
    def origin(self, data: mujoco.MjData) -> np.ndarray:
        if self.site_id >= 0:
            return data.site_xpos[self.site_id]
        return data.xpos[self.body_id]

    def rotation(self, data: mujoco.MjData) -> np.ndarray:
        if self.site_id >= 0:
            return data.site_xmat[self.site_id].reshape(3, 3)
        return data.xmat[self.body_id].reshape(3, 3)

    def dirs_world(self, data: mujoco.MjData) -> np.ndarray:
        return self.dirs_local @ self.rotation(data).T

    # -- casting -----------------------------------------------------------
    def cast(self, data: mujoco.MjData) -> RayHits:
        """Cast every ray from the mount origin. Distances beyond `max_range`
        are reported as misses (-1), matching a range-limited device."""
        np.matmul(self.dirs_local, self.rotation(data).T, out=self._vec.reshape(self.n, 3))
        pnt = np.ascontiguousarray(self.origin(data), dtype=np.float64)
        mujoco.mj_multiRay(
            self.model, data, pnt, self._vec, self.geomgroup,
            1,                      # flg_static: static geoms (floor, walls) count
            self.body_id,           # the mount body is invisible to itself
            self._geomid, self._dist, None,
            self.n, self.max_range,
        )
        dist = self._dist.copy()
        geomid = self._geomid.copy()
        miss = (geomid < 0) | (dist < 0) | (dist > self.max_range)
        dist[miss] = -1.0
        geomid[miss] = -1
        return RayHits(dist=dist, geomid=geomid)

    def hit_points(self, data: mujoco.MjData, hits: RayHits) -> np.ndarray:
        """World-frame hit points, (N, 3); NaN rows for misses. For overlays."""
        pts = self.origin(data)[None, :] + self.dirs_world(data) * hits.dist[:, None]
        pts[~hits.hit] = np.nan
        return pts


def tof_fan(rows: int, cols: int, fov_deg: float, subrays: int = 1) -> np.ndarray:
    """Directions for a rows×cols zone matrix spanning `fov_deg` on both axes,
    with `subrays`² rays per zone (a zone integrates a patch, not a line).

    Layout is image-like as seen by the sensor: row 0 is the TOP (rays tilted
    toward +z), column 0 is the LEFT (toward +y). Returned as
    (rows*subrays*cols*subrays, 3) in raster order, so zone (r, c) owns rays
    [(r*S + i)*cols*S + c*S + j] for i, j in range(S).
    """
    if rows < 1 or cols < 1 or subrays < 1:
        raise ValueError("rows, cols and subrays must be >= 1")
    half = np.tan(np.deg2rad(fov_deg) / 2.0)
    # Sub-ray centres inside each zone, on the tangent plane at x = 1.
    ny, nz = cols * subrays, rows * subrays
    ys = np.linspace(half, -half, 2 * ny + 1)[1::2]   # left (+y) → right
    zs = np.linspace(half, -half, 2 * nz + 1)[1::2]   # top (+z) → bottom
    d = np.array([[1.0, y, z] for z in zs for y in ys])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def planar_fan(n: int, fov_deg: float) -> np.ndarray:
    """Directions for an n-ray scan in the x-y plane centred on +x
    (left, +y, first) — what a 2D LiDAR would be, if the hardware ever grows one."""
    if n < 1:
        raise ValueError("n must be >= 1")
    a = np.deg2rad(np.linspace(fov_deg / 2, -fov_deg / 2, n))
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)
