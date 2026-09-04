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
    # Weight of the new measurement. MEASURED as already optimal, and the
    # sweep is worth keeping because the obvious intuition is wrong: heavier
    # averaging makes a STILL ball worse, not better. Reading `Track.xy`
    # against truth over 3 seeds of 1v1 (still ball inside 0.6 m / a ball
    # rolling above 0.3 m/s): 1.00 -> 3.42 / 9.75 cm, 0.60 -> 2.70 / 8.90,
    # 0.30 -> 4.36 / 14.26, 0.15 -> 8.89 / 14.40.
    #
    # The mechanism is the FRAME. This smooths bearing and range, in the
    # BODY frame, and the body walks and turns - so a ball stationary in the
    # WORLD still sweeps quickly in bearing, and averaging it lags. Past
    # ~0.6 the lag costs more than the noise it removes. "Still ball"
    # describes the world, not the measurement.
    #
    # The version the frame argument does NOT kill - smooth `xy` itself, in
    # the odometry frame where a stationary ball genuinely is stationary -
    # was measured too, as an EMA of the raw per-frame xy (still / rolling):
    # shipped 2.70 / 8.90 cm, xy-EMA a=0.60 2.41 / 10.37, a=0.30 2.19 /
    # 17.01, a=0.15 3.24 / 31.39. It works as predicted (19% off a still
    # ball, and unlike polar smoothing it keeps improving past a=0.6) and it
    # is still not worth building: it costs heavily on a rolling ball so it
    # needs gating on `vel`, and 0.5 cm off a 5.5 cm placement error is
    # below what the soccer benchmark can resolve (~200 seeds for +0.3
    # goals). Recorded so nobody re-derives it. docs/camera-hardware.md 3c.
    smooth: float = 0.6
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
        against the previous hit when the two are usefully apart in time.

        KNOWN BIAS, not yet fixed here. `pos`/`yaw` are the pose the caller
        has NOW, while `t` is the frame's timestamp - so a hit is anchored
        where the duck is rather than where it was when the picture was
        taken, and `xy` carries about (duck speed x detector period) of
        error in the direction of travel: ~3 cm at 10 Hz, ~6 cm at 5 Hz.
        `brain/tidy.py`'s `stale_fix` was the same bug in the same shape and
        has now been measured. THE RESULT IS NOT "FIX IT HERE TOO".
        Correcting the placement alone LOST 0.38 toys (p = 0.031) even
        though it cut the estimate's error from 5.4 cm to 3.7 cm, because
        the stop distance downstream had been hand-fitted against the
        biased estimate and absorbed it. Only correcting BOTH won
        (+0.44 toys, grasp 88% -> 93%). See AGENTS.md rule 7.

        MEASURED HERE, and the answer is DON'T. 12 348 ball sightings over
        2 seeds x 180 s of 1v1, each placement checked against truth and
        split into what moved between the frame and now:

            frame age          100 ms (median)
            placement error    7.7 cm   <- what the brain acts on
            duck moved since   1.8 cm   <- THIS bias
            ball moved since   0.7 cm   <- what predict() is for

        and in the line-up case that actually scores (ball inside 0.6 m and
        nearly still, 6 253 of those sightings): 5.5 cm of error, 1.7 cm of
        it from the stale pose. **The error is dominated by neither term -
        it is the detector's own bearing and range noise.** Fixing the pose
        removes at most a third of it, and by rule 7 it would ALSO move the
        line-up off the constants fitted around it. Bad trade; not done.

        (An earlier version of this note said the ball moves 1.4 m/s and so
        `predict()` dominates. That is the peak right after a kick, not the
        operating point: at the median the ball has moved 0.7 cm since the
        frame. Most sightings are of a nearly stationary ball.)

        VELOCITY is less affected - both samples carry a similar error, so
        it largely cancels in the difference - but the POSITION does not,
        and `Track.xy` is what a dead-reckoned approach steers by.
        """
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
