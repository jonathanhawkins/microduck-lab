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

    def age(self, t: float) -> float:
        return t - self.last_t


@dataclass(frozen=True)
class TrackerParams:
    gate_rad: float = 0.35         # a detection this close in bearing can update a track
    gate_range_frac: float = 0.6   # …if its range is within this fraction, too
    smooth: float = 0.6            # weight of the new measurement
    coast_s: float = 2.5           # a track survives this long without a hit
    confirm_hits: int = 2          # hits before a brain should trust it


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

    def update(self, frame: DetectionFrame | None, t: float, yaw: float | None = None) -> list[Track]:
        """Fold one detection frame (or None) at time t. `yaw` is the body
        heading now: bearings of coasting tracks turn with the body."""
        p = self.p
        if yaw is not None and self._prev_yaw is not None:
            dyaw = math.atan2(math.sin(yaw - self._prev_yaw), math.cos(yaw - self._prev_yaw))
            if dyaw:
                for tr in self.tracks:                  # a hit below replaces this with the measurement
                    tr.bearing = math.atan2(math.sin(tr.bearing - dyaw), math.cos(tr.bearing - dyaw))
        self._prev_yaw = yaw
        if frame is not None and frame.t != self._last_frame_t:
            self._last_frame_t = frame.t
            self._associate(frame.detections, frame.t)
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= p.coast_s]
        return self.tracks

    def _associate(self, dets: list[Detection], t: float) -> None:
        p = self.p
        used: set[int] = set()
        hit: set[int] = set()
        # Greedy nearest-first: best pairs first, one detection per track.
        pairs = []
        for i, d in enumerate(dets):
            for tr in self.tracks:
                if tr.cls != d.cls:
                    continue
                db = abs(math.atan2(math.sin(d.bearing - tr.bearing), math.cos(d.bearing - tr.bearing)))
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
            tr.bearing = tr.bearing + k * math.atan2(math.sin(d.bearing - tr.bearing), math.cos(d.bearing - tr.bearing))
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
        for i, d in enumerate(dets):
            if i in used:
                continue
            self.tracks.append(Track(self._next_id, d.cls, d.bearing, d.elevation, d.width, d.range_est, d.conf,
                                     t, t, name=d.name, names={d.name: 1} if d.name else {}))
            self._next_id += 1

    def best(self, cls: str, t: float, min_hits: int | None = None) -> Track | None:
        """The track of `cls` to act on: confirmed, freshest, then nearest."""
        need = self.p.confirm_hits if min_hits is None else min_hits
        cands = [tr for tr in self.tracks if tr.cls == cls and tr.hits >= need]
        if not cands:
            return None
        return min(cands, key=lambda tr: (round(tr.age(t), 1), tr.range))

    def payload(self, t: float) -> list[dict]:
        return [{"id": tr.id, "cls": tr.cls, "name": tr.name, "bearing": round(tr.bearing, 3),
                 "range": round(tr.range, 3), "hits": tr.hits, "age": round(tr.age(t), 2)} for tr in self.tracks]


__all__ = ["Track", "Tracker", "TrackerParams"]
