"""A tracker over the detector's frames (roadmap 1.3's "tracker gives ids").

The detector returns per-frame detections; a brain that acts on the raw
list re-picks its target every frame (a ghost or a second person steals
it) and loses it the first frame it is missed. `Tracker` associates each
frame's detections with the tracks it already has — same class, nearest
bearing inside a gate, range not wildly different — smooths bearing and
range, counts hits, and COASTS a track through misses: with odometry, the
remembered bearing turns with the body, so a person the duck turns away
from is still "at −1.2 rad", not gone. Ghosts (no consistent position)
never reach the hit count a brain asks for.

In the sim the detector also hands out the true object name; the tracker
keeps it as `name` for the tools and tests, but its `id` is its own —
what the real robot will have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..sensors.detector import Detection, DetectionFrame


@dataclass
class Track:
    id: int
    cls: str
    bearing: float
    elevation: float
    width: float
    range: float
    conf: float
    born_t: float
    last_t: float               # last frame that HIT
    hits: int = 1
    misses: int = 0             # FRAMES since the last hit (no frame, no miss: use age() for staleness)
    name: str = ""
    names: dict = field(default_factory=dict)   # sim name → count, for `name`
    # Where it is and where it is going, in the ODOMETRY frame (only when
    # the brain passes its position in): the last hit's position, the time
    # of it, and a smoothed velocity from consecutive hits. A rolling ball
    # leaves the camera in a frame or two; this is what says where to look.
    xy: tuple[float, float] | None = None
    xy_t: float = 0.0
    vel: tuple[float, float] = (0.0, 0.0)
    vel_hits: int = 0                            # hits the velocity rests on (2+ before trusting it)

    def age(self, t: float) -> float:
        return t - self.last_t

    def predict(self, t: float, decel: float = 0.0) -> tuple[float, float] | None:
        """Position at t from the last hit and the velocity (a constant
        deceleration `decel` along it, to a stop). None without a position."""
        if self.xy is None:
            return None
        dt = max(0.0, t - self.xy_t)
        vx, vy = self.vel
        speed = math.hypot(vx, vy)
        if speed < 1e-6 or self.vel_hits < 2:
            return self.xy
        if decel > 0:
            t_stop = speed / decel
            dt = min(dt, t_stop)
            dist = speed * dt - 0.5 * decel * dt * dt
        else:
            dist = speed * dt
        return (self.xy[0] + vx / speed * dist, self.xy[1] + vy / speed * dist)


@dataclass(frozen=True)
class TrackerParams:
    gate_rad: float = 0.35         # a detection this close in bearing can update a track
    gate_range_frac: float = 0.6   # …if its range is within this fraction, too
    smooth: float = 0.6            # weight of the new measurement
    coast_s: float = 2.5           # a track survives this long without a hit
    confirm_hits: int = 2          # hits before a brain should trust it
    vel_smooth: float = 0.5        # weight of a new velocity sample (hits 0.05-1 s apart)
    vel_min_dt: float = 0.05
    vel_max_dt: float = 1.0


class Tracker:
    def __init__(self, p: TrackerParams = TrackerParams()):
        self.p = p
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_frame_t: float | None = None
        self._prev_yaw: float | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._last_frame_t = None
        self._prev_yaw = None

    def update(self, frame: DetectionFrame | None, t: float, yaw: float | None = None,
               pos: tuple[float, float] | None = None) -> list[Track]:
        """Fold one detection frame (or None) at time t. `yaw` is the body
        heading now: bearings of coasting tracks turn with the body. With
        `pos` (the body's odometry position) each hit also places the track
        in the odometry frame and feeds its velocity."""
        p = self.p
        if yaw is not None and self._prev_yaw is not None:
            dyaw = math.atan2(math.sin(yaw - self._prev_yaw), math.cos(yaw - self._prev_yaw))
            if dyaw:
                for tr in self.tracks:                  # a hit below replaces this with the measurement
                    tr.bearing = math.atan2(math.sin(tr.bearing - dyaw), math.cos(tr.bearing - dyaw))
        self._prev_yaw = yaw
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            hit = self._associate(frame.detections, frame.t, getattr(frame, "cam_yaw", 0.0))
            if pos is not None and yaw is not None:
                for tr in hit:
                    self._place(tr, frame.t, pos, yaw)
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= p.coast_s]
        return self.tracks

    def _place(self, tr: Track, t: float, pos: tuple[float, float], yaw: float) -> None:
        """A hit: the track's odometry-frame position, and a velocity sample
        against the previous hit when the two are usefully apart in time."""
        p = self.p
        a = yaw + tr.bearing
        xy = (pos[0] + tr.range * math.cos(a), pos[1] + tr.range * math.sin(a))
        if tr.xy is not None:
            dt = t - tr.xy_t
            if p.vel_min_dt <= dt <= p.vel_max_dt:
                sample = ((xy[0] - tr.xy[0]) / dt, (xy[1] - tr.xy[1]) / dt)
                k = p.vel_smooth if tr.vel_hits else 1.0
                tr.vel = (tr.vel[0] + k * (sample[0] - tr.vel[0]), tr.vel[1] + k * (sample[1] - tr.vel[1]))
                tr.vel_hits += 1
            elif dt > p.vel_max_dt:
                tr.vel, tr.vel_hits = (0.0, 0.0), 0        # too long ago to say
        tr.xy, tr.xy_t = xy, t

    def _associate(self, dets: list[Detection], t: float, cam_yaw: float = 0.0) -> list[Track]:
        """Detections come in the CAMERA's frame; tracks are kept in the
        BODY's (a brain steers the body), so `cam_yaw` — where the head
        was looking — is added on the way in."""
        p = self.p
        used: set[int] = set()
        hit: set[int] = set()
        body = [math.atan2(math.sin(d.bearing + cam_yaw), math.cos(d.bearing + cam_yaw)) for d in dets]
        # Greedy nearest-first: best pairs first, one detection per track.
        pairs = []
        for i, d in enumerate(dets):
            for tr in self.tracks:
                if tr.cls != d.cls:
                    continue
                db = abs(math.atan2(math.sin(body[i] - tr.bearing), math.cos(body[i] - tr.bearing)))
                if db > p.gate_rad:
                    continue
                if abs(d.range_est - tr.range) > p.gate_range_frac * max(tr.range, 0.3):
                    continue
                pairs.append((db, i, tr.id))
        pairs.sort()
        for db, i, tid in pairs:
            if i in used or tid in hit:
                continue
            tr = next(x for x in self.tracks if x.id == tid)
            d = dets[i]
            k = p.smooth
            tr.bearing = tr.bearing + k * math.atan2(math.sin(body[i] - tr.bearing), math.cos(body[i] - tr.bearing))
            tr.elevation += k * (d.elevation - tr.elevation)
            tr.width += k * (d.width - tr.width)
            tr.range += k * (d.range_est - tr.range)
            tr.conf = max(d.conf, 0.7 * tr.conf)
            tr.hits += 1
            tr.misses = 0
            tr.last_t = t
            if d.name:
                tr.names[d.name] = tr.names.get(d.name, 0) + 1
                tr.name = max(tr.names, key=tr.names.get)
            used.add(i)
            hit.add(tid)
        for tr in self.tracks:
            if tr.id not in hit:
                tr.misses += 1
        born = []
        for i, d in enumerate(dets):
            if i in used:
                continue
            tr = Track(self._next_id, d.cls, body[i], d.elevation, d.width, d.range_est, d.conf,
                       t, t, name=d.name, names={d.name: 1} if d.name else {})
            self.tracks.append(tr)
            born.append(tr)
            self._next_id += 1
        return [tr for tr in self.tracks if tr.id in hit] + born

    def best(self, cls: str, t: float, min_hits: int | None = None) -> Track | None:
        """The track of `cls` to act on: confirmed, freshest, then nearest."""
        need = self.p.confirm_hits if min_hits is None else min_hits
        cands = [tr for tr in self.tracks if tr.cls == cls and tr.hits >= need]
        if not cands:
            return None
        return min(cands, key=lambda tr: (round(tr.age(t), 1), tr.range))

    def payload(self, t: float) -> list[dict]:
        return [{"id": tr.id, "cls": tr.cls, "name": tr.name, "bearing": round(tr.bearing, 3),
                 "range": round(tr.range, 3), "hits": tr.hits, "age": round(tr.age(t), 2),
                 **({"xy": [round(tr.xy[0], 3), round(tr.xy[1], 3)],
                     "vel": [round(tr.vel[0], 2), round(tr.vel[1], 2)]} if tr.xy is not None else {})}
                for tr in self.tracks]


__all__ = ["Track", "Tracker", "TrackerParams"]
