"""Room mapping from the 8×8 ToF (roadmap 4.x's first step): an occupancy
grid in the brain's ODOMETRY frame, built from what the robot has — the
depth matrix, the sensor's mount pose on the body (neck/head servos + IMU
through the kinematics; the frame carries it) and its own dead-reckoned
pose. Nothing here reads the sim.

Log-odds cells at `res` metres. Every valid zone traces free space from
the sensor to its hit and marks the hit cell occupied — unless the hit is
the floor (below `floor_z`), which is free space seen from above, or the
ray left the room. Odometry drift (roadmap 1.7) smears the map exactly as
it would on the robot; that smear is the honest picture of what a duck
can know about a room without a loop closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sensors.tof import TofFrame


@dataclass(frozen=True)
class GridSpec:
    size: tuple[float, float] = (6.0, 6.0)   # metres, centred on the odom origin
    res: float = 0.05                          # cell edge, m
    floor_z: float = 0.03                      # hits lower than this are the floor
    ceiling_z: float = 0.6                     # …and higher than this are not walls a duck meets
    max_range: float = 3.0                     # ToF returns beyond this are too noisy to map
    hit: float = 0.85                          # log-odds step for an occupied hit
    miss: float = -0.4                         # …and for a cell the ray passed through
    clamp: float = 5.0


class OccupancyGrid:
    def __init__(self, spec: GridSpec = GridSpec()):
        self.spec = spec
        self.nx = int(round(spec.size[0] / spec.res))
        self.ny = int(round(spec.size[1] / spec.res))
        self.logodds = np.zeros((self.ny, self.nx), np.float32)
        self.frames = 0
        self._last_t: float | None = None

    def reset(self) -> None:
        self.logodds[:] = 0.0
        self.frames = 0
        self._last_t = None

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        s = self.spec
        i = int(math.floor((x + s.size[0] / 2) / s.res))
        j = int(math.floor((y + s.size[1] / 2) / s.res))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def update(self, frame: TofFrame | None, odom: tuple[float, float, float] | None) -> bool:
        """Fold one ToF frame taken at `odom`. Returns True if it was used."""
        if frame is None or odom is None or frame.t == self._last_t:
            return False
        if frame.mount_pos is None or frame.mount_rot is None or frame.dirs_local is None:
            return False
        self._last_t = frame.t
        s = self.spec
        x, y, yaw = odom
        c, sn = math.cos(yaw), math.sin(yaw)
        Rw = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([x, y, 0.0]) + Rw @ frame.mount_pos
        dirs = (Rw @ frame.mount_rot) @ frame.dirs_local.reshape(-1, 3).T    # (3, N) odom-frame unit vectors
        depth = frame.depth_mm.reshape(-1) / 1000.0
        valid = frame.valid.reshape(-1) & (depth > 0) & (depth <= s.max_range)
        o = self.cell(origin[0], origin[1])
        for k in np.nonzero(valid)[0]:
            end = origin + dirs[:, k] * depth[k]
            e = self.cell(end[0], end[1])
            if o is not None and e is not None:
                self._trace_free(o, e)
            if e is not None and s.floor_z <= end[2] <= s.ceiling_z:
                self.logodds[e[1], e[0]] = min(self.logodds[e[1], e[0]] + s.hit, s.clamp)
        self.frames += 1
        return True

    def _trace_free(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        """Bresenham from a to b, exclusive of b: the ray passed through."""
        (x0, y0), (x1, y1) = a, b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        s = self.spec
        lo = self.logodds
        while (x0, y0) != (x1, y1):
            lo[y0, x0] = max(lo[y0, x0] + s.miss, -s.clamp)
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def occupied(self, threshold: float = 1.0) -> np.ndarray:
        return self.logodds >= threshold

    def payload(self) -> dict:
        """Wire shape: a (ny, nx) grid of 0 (unknown) / 1 (free) / 2 (occupied)
        as a compact string, plus the geometry to draw it."""
        cls = np.where(self.logodds >= 1.0, 2, np.where(self.logodds <= -1.0, 1, 0)).astype(np.uint8)
        # One character per cell, row-major from -y: '0','1','2' — ASCII 48 + class,
        # straight from the array (a per-cell Python join cost ~3 ms per map).
        s = self.spec
        return {"nx": self.nx, "ny": self.ny, "res": s.res, "origin": [-s.size[0] / 2, -s.size[1] / 2],
                "frames": self.frames, "cells": (cls.reshape(-1) + 48).tobytes().decode("ascii")}


__all__ = ["GridSpec", "OccupancyGrid"]
