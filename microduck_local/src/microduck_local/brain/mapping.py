"""Room mapping from the 8×8 ToF (roadmap 4.x's first step): an occupancy
grid in the brain's ODOMETRY frame, built from what the robot has — the
depth matrix, the sensor's mount pose on the body (neck/head servos + IMU
through the kinematics; the frame carries it) and its own dead-reckoned
pose. Nothing here reads the sim.

Log-odds cells at `res` metres. Every valid zone traces free space from
the sensor to its hit and marks the hit cell occupied — unless the hit is
the floor (below `floor_z`), which is free space seen from above, or the
ray left the room.

Odometry drift (roadmap 1.7) would smear the map exactly as it does on the
robot, so the grid closes the loop with the one feature a 45° depth matrix
measures well: the WALL LINE. Before a frame is folded in, its wall hits
are fitted with a line; the map's occupied cells near where that line
lands are fitted with another; the angle between the two corrects the
heading and the perpendicular gap corrects the position. The correction is
a transform kept between odometry and the map frame, so a yaw bias is
caught as it grows, not after the map is a fan. One wall says nothing
about motion along it, and the matcher does not pretend otherwise (a
correlative search over cells was measured to trade a yaw error for a
sideways one and walk the map off — the field of view is too narrow to
tell the two apart by overlap). Frames without a single clean wall line
(corners, clutter, fewer than a handful of hits) are folded in unmatched.
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
    # Loop closure by wall-line matching (module docstring). `match_after`
    # frames are folded in on trust to seed the map; after that a frame whose
    # wall hits fit one line (rms under `match_line_m`) is matched against
    # the map's occupied cells within `match_radius` of it, and the pose
    # takes `match_gain` of the yaw and gap it finds beyond a DEADBAND
    # (`match_dead_deg`, `match_dead_m`: two line fits over a metre of wall
    # disagree by that much from noise alone, and chasing it turned good
    # odometry into a 12° random walk — measured) — if they are small
    # enough to be drift (`match_max_deg`, `match_max_shift`) and not a
    # different wall.
    match: bool = True
    match_after: int = 20
    match_min_hits: int = 6
    match_line_m: float = 0.05
    match_radius: float = 0.12
    match_max_deg: float = 8.0
    match_max_shift: float = 0.2
    match_dead_deg: float = 2.0
    match_dead_m: float = 0.05
    match_gain: float = 0.4


class OccupancyGrid:
    def __init__(self, spec: GridSpec = GridSpec()):
        self.spec = spec
        self.nx = int(round(spec.size[0] / spec.res))
        self.ny = int(round(spec.size[1] / spec.res))
        self.logodds = np.zeros((self.ny, self.nx), np.float32)
        self.frames = 0
        self._last_t: float | None = None
        # Odometry → map-frame correction (x, y, yaw) and the last corrected pose.
        self.offset = np.zeros(3)
        self.pose: tuple[float, float, float] | None = None
        self.corrections = 0
        self._field: np.ndarray | None = None      # blurred occupancy the matcher scores against
        self._field_frame = -1

    def reset(self) -> None:
        self.logodds[:] = 0.0
        self.frames = 0
        self._last_t = None
        self.offset[:] = 0.0
        self.pose = None
        self.corrections = 0
        self._field = None
        self._field_frame = -1

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
        depth = frame.depth_mm.reshape(-1) / 1000.0
        valid = frame.valid.reshape(-1) & (depth > 0) & (depth <= s.max_range)
        x, y, yaw = self.correct(odom)
        if s.match and self.frames >= s.match_after:
            x, y, yaw = self._match(frame, (x, y, yaw), depth, valid)
            self._refit(odom, (x, y, yaw))
        self.pose = (x, y, yaw)
        origin, dirs = self._rays(frame, (x, y, yaw))
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

    # -- scan matching --------------------------------------------------------
    def correct(self, odom: tuple[float, float, float]) -> tuple[float, float, float]:
        """The odometry pose in the map frame: offset ∘ odom."""
        ox, oy, oyaw = self.offset
        c, sn = math.cos(oyaw), math.sin(oyaw)
        x, y, yaw = odom
        return c * x - sn * y + ox, sn * x + c * y + oy, math.atan2(math.sin(yaw + oyaw), math.cos(yaw + oyaw))

    def _refit(self, odom: tuple[float, float, float], pose: tuple[float, float, float]) -> None:
        """Choose the offset that maps this odometry reading onto `pose`."""
        oyaw = math.atan2(math.sin(pose[2] - odom[2]), math.cos(pose[2] - odom[2]))
        c, sn = math.cos(oyaw), math.sin(oyaw)
        self.offset[:] = (pose[0] - (c * odom[0] - sn * odom[1]), pose[1] - (sn * odom[0] + c * odom[1]), oyaw)

    @staticmethod
    def _rays(frame: TofFrame, pose: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        x, y, yaw = pose
        c, sn = math.cos(yaw), math.sin(yaw)
        Rw = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([x, y, 0.0]) + Rw @ frame.mount_pos
        dirs = (Rw @ frame.mount_rot) @ frame.dirs_local.reshape(-1, 3).T    # (3, N) map-frame unit vectors
        return origin, dirs

    def _wall_hits(self, frame: TofFrame, pose, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """(N, 2) map-frame xy of this frame's wall hits at `pose`."""
        s = self.spec
        origin, dirs = self._rays(frame, pose)
        end = origin[:, None] + dirs * depth[None, :]
        wall = valid & (end[2] >= s.floor_z) & (end[2] <= s.ceiling_z)
        return end[:2, wall].T

    @staticmethod
    def _line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Total-least-squares line through (N, 2) points: (centroid, unit
        direction, rms distance of the points from the line)."""
        c = pts.mean(axis=0)
        q = pts - c
        w, v = np.linalg.eigh(q.T @ q / len(pts))
        return c, v[:, 1], float(math.sqrt(max(w[0], 0.0)))

    def _match(self, frame: TofFrame, pose, depth: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
        """The pose nudged onto the map's wall, or `pose` unchanged."""
        s = self.spec
        local = self._wall_hits(frame, (0.0, 0.0, 0.0), depth, valid)      # hits in the body frame
        if len(local) < s.match_min_hits:
            return pose
        c_l, d_l, rms = self._line(local)
        if rms > s.match_line_m:
            return pose                                   # a corner or clutter: no single wall to match
        placed = self._place(local, pose)
        js, is_ = np.nonzero(self.logodds >= 1.0)
        if len(is_) < s.match_min_hits:
            return pose
        cells = np.stack([(is_ + 0.5) * s.res - s.size[0] / 2, (js + 0.5) * s.res - s.size[1] / 2], axis=1)
        d2 = ((cells[:, None, :] - placed[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        near = cells[d2 <= s.match_radius ** 2]
        if len(near) < s.match_min_hits:
            return pose
        c_m, d_m, rms_m = self._line(near)
        if rms_m > 2.0 * s.match_line_m + s.res:
            return pose
        x, y, yaw = pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        d_p = np.array([cy * d_l[0] - sy * d_l[1], sy * d_l[0] + cy * d_l[1]])
        dth = math.atan2(d_p[0] * d_m[1] - d_p[1] * d_m[0], d_p @ d_m)
        if dth > math.pi / 2:                             # a line has no front: fold to ±90°
            dth -= math.pi
        elif dth < -math.pi / 2:
            dth += math.pi
        if abs(dth) > math.radians(s.match_max_deg):
            return pose
        dead = math.radians(s.match_dead_deg)
        dth = 0.0 if abs(dth) <= dead else dth - math.copysign(dead, dth)
        yaw2 = yaw + s.match_gain * dth
        c2, s2 = math.cos(yaw2), math.sin(yaw2)
        cp = np.array([c2 * c_l[0] - s2 * c_l[1] + x, s2 * c_l[0] + c2 * c_l[1] + y])
        n = np.array([-d_m[1], d_m[0]])                   # across the map wall
        gap = float((c_m - cp) @ n)
        if abs(gap) > s.match_max_shift:
            return pose
        gap = 0.0 if abs(gap) <= s.match_dead_m else gap - math.copysign(s.match_dead_m, gap)
        if dth == 0.0 and gap == 0.0:
            return pose
        self.corrections += 1
        g = s.match_gain * gap
        return x + g * n[0], y + g * n[1], math.atan2(math.sin(yaw2), math.cos(yaw2))

    @staticmethod
    def _place(local: np.ndarray, pose) -> np.ndarray:
        x, y, yaw = pose
        c, sn = math.cos(yaw), math.sin(yaw)
        return np.stack([c * local[:, 0] - sn * local[:, 1] + x, sn * local[:, 0] + c * local[:, 1] + y], axis=1)

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
                "frames": self.frames, "cells": (cls.reshape(-1) + 48).tobytes().decode("ascii"),
                "offset": [round(float(v), 4) for v in self.offset], "corrections": self.corrections,
                "pose": None if self.pose is None else [round(float(v), 4) for v in self.pose]}


__all__ = ["GridSpec", "OccupancyGrid"]
