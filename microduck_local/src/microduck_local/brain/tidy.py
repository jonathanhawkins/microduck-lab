"""The tidy brain (roadmap 12.7): find toys, pick them up, drop them in the
basket, repeat — over the senses and intents the robot actually has.

    scan ──▶ approach ──▶ blind ──▶ settle ──▶ pick ──▶ verify ─┐
     ▲         (servo on     (dead-      (stand    (skill      │ held?
     │          detection)   reckon      still)    cycle)      ▼
     │                       the last                    carry ──▶ deliver ──▶ drop ──▶ backoff
     │                       half metre)                 (find the    (servo,    (open    (reverse)
     └─────────────────────────────────────────────────── basket)     blind end) beak)      │
                                                                                            └──▶ scan

Why the blind legs exist: the head camera looks straight ahead, so a toy
on the floor leaves its field of view about half a metre out, and the
basket rim about 0.4 m out. From there the brain walks a remembered point
in odometry — exactly the "operator aims the robot, the policy is blind"
pattern upstream ships, automated. Odometry is the truth here for now
(roadmap 1.7 adds drift), and the real robot's is contact-anchored and
drifts, which is why the basket is re-acquired visually every trip.

Per-toy retries and a give-up budget keep it from looping on a toy it
cannot grasp; two clean scans with nothing seen means "done".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..world.arena import PICK_REACH_AHEAD, PICK_REACH_LEFT
from .controllers import WanderParams, _column_clearance, wander_from_tof
from .runtime import REGISTRY, Intent, Senses, age_inputs


@dataclass(frozen=True)
class TidyParams:
    reach_ahead: float = PICK_REACH_AHEAD
    reach_left: float = PICK_REACH_LEFT
    approach_speed: float = 0.3        # the walker delivers ~half; below ~0.2 it stands still
    k_turn: float = 2.5
    head_down: float = 0.6             # head_pitch intent while hunting: camera ~37° down
    cam_ahead: float = 0.064           # camera ahead of the trunk origin (m)
    cam_z_down: float = 0.20           # camera height with the head down (m)
    cam_pitch_down: float = 0.655      # …and its depression (rad), measured on the walker
    toy_z: float = 0.015               # a toy's centre height (m), class-blind
    basket_z: float = 0.08             # the basket marker's height (m)
    basket_reach: float = 0.19         # trunk-to-basket-centre at release: feet outside a 0.3 m tray, beak over it
    settle_s: float = 0.6
    backoff_s: float = 2.0
    scan_wz: float = 1.0               # the shipped walker barely turns in place below 1.0
    explore_s: float = 6.0             # wander this long after an empty scan
    max_retries: int = 2               # pick attempts per toy before giving up on it
    done_after_scans: int = 4          # empty scans (each followed by a wander) before giving up


class Tidy:
    kind = "tidy"
    wants_head = True                  # the server applies this brain's head intents
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: TidyParams = TidyParams()):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self.state = "scan"
        self.est: tuple[float, float] | None = None      # odom-frame estimate of the target
        self.goal_kind: str | None = None                # "toy" | "basket"
        self.target_name: str | None = None
        self.scan_turned = 0.0
        self.scans_empty = 0
        self.t_state = 0.0
        self.t_seen = -9.0
        self.retries: dict[str, int] = {}
        self.given_up: set[str] = set()
        self.picked = 0
        self.delivered = 0
        self.last = (0.0, 0.0, 0.0)
        self._senses: Senses | None = None
        self._prev_yaw: float | None = None

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["target"] = None if self.est is None else {
            "bearing": 0.0, "range": None, "since": round(self._senses.t - self.t_seen, 2),
            "goal": [round(v, 3) for v in self.est], "kind": self.goal_kind, "name": self.target_name}
        out["tidy"] = {"picked": self.picked, "delivered": self.delivered,
                       "givenUp": sorted(self.given_up), "retries": dict(self.retries)}
        return out

    # -- helpers --------------------------------------------------------------
    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self.t_state = t

    def _nearest(self, senses: Senses, cls: str):
        det = senses.fresh_det(self.DET_MAX_AGE)
        if det is None:
            return None
        cands = [d for d in det.detections if d.cls == cls and d.name not in self.given_up]
        return min(cands, key=lambda d: d.range_est) if cands else None

    def _update_estimate(self, odom, det, target_z: float) -> float:
        """Fold one detection into the odom-frame target estimate using the
        elevation (a floor object seen from a known camera height and pitch
        gives range far better than its apparent width). Returns the
        horizontal range from the trunk."""
        p = self.p
        depression = p.cam_pitch_down - det.elevation          # rad below horizontal
        depression = max(depression, 0.05)
        horiz_cam = (p.cam_z_down - target_z) / math.tan(depression)
        horiz_cam = float(np.clip(horiz_cam, 0.0, 4.0))
        x, y, yaw = odom
        a = yaw + det.bearing
        tx = x + (p.cam_ahead + horiz_cam) * math.cos(a)
        ty = y + (p.cam_ahead + horiz_cam) * math.sin(a)
        if self.est is None:
            self.est = (tx, ty)
        else:
            k = 0.5
            self.est = (self.est[0] + k * (tx - self.est[0]), self.est[1] + k * (ty - self.est[1]))
        return math.hypot(self.est[0] - x, self.est[1] - y)

    def _servo(self, odom, stop_at: float, left: float = 0.0) -> tuple[tuple[float, float, float], float, float]:
        """Walk toward the estimate, turning in place first if it is far off
        the nose, and stop `stop_at` short of it. `left` biases the aim so
        an off-centre beak lands on the target."""
        x, y, yaw = odom
        dx, dy = self.est[0] - x, self.est[1] - y
        dist = math.hypot(dx, dy)
        want = math.atan2(left, max(dist, 0.05))
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw - want), math.cos(math.atan2(dy, dx) - yaw - want))
        if dist <= stop_at:
            return (0.0, 0.0, 0.0), dist, bearing
        if abs(bearing) > 0.35:
            return (0.0, 0.0, 1.0 if bearing > 0 else -1.0), dist, bearing
        return (self.p.approach_speed, 0.0, float(np.clip(self.p.k_turn * bearing, -1.0, 1.0))), dist, bearing

    def _bumper(self, senses: Senses) -> float:
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        if tof is None:
            return math.inf
        return float(_column_clearance(tof.depth_mm, tof.valid, WanderParams())[3:5].min())

    # -- the machine ---------------------------------------------------------
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p, t = self.p, senses.t
        odom = senses.odom or (0.0, 0.0, 0.0)
        twist = (0.0, 0.0, 0.0)
        head = (0.0, 0.0, 0.0, 0.0)
        beak = None
        skill = None
        note = self.state
        looking_down = self.state in ("scan", "explore", "approach", "blind", "carry", "deliver")
        if looking_down:
            head = (0.0, p.head_down, 0.0, 0.0)

        if self.state == "scan":
            toy = self._nearest(senses, "toy")
            if toy is not None:
                self.est, self.goal_kind, self.target_name = None, "toy", toy.name
                self._update_estimate(odom, toy, p.toy_z)
                self.t_seen = t
                self.scans_empty = 0
                self._enter("approach", t)
            else:
                twist = (0.0, 0.0, p.scan_wz)
                if self._prev_yaw is not None:
                    d = odom[2] - self._prev_yaw
                    self.scan_turned += abs(math.atan2(math.sin(d), math.cos(d)))
                if self.scan_turned >= 2 * math.pi:
                    self.scan_turned = 0.0
                    self.scans_empty += 1
                    self._enter("done" if self.scans_empty >= p.done_after_scans else "explore", t)
                note = f"scan {self.scans_empty}/{p.done_after_scans}"

        elif self.state == "explore":
            if self._nearest(senses, "toy") is not None:
                self._enter("scan", t)
            else:
                tof = senses.fresh_tof(self.TOF_MAX_AGE)
                # The ToF looks down with the head: only its top rows see ahead.
                twist = wander_from_tof(tof.depth_mm, tof.valid, WanderParams(rows=(0, 3))) if tof is not None else (0.0, 0.0, 0.0)
                if twist[0] == 0.0 and twist[2] != 0.0:
                    twist = (0.0, 0.0, 1.0 if twist[2] > 0 else -1.0)
                if t - self.t_state > p.explore_s:
                    self._enter("scan", t)

        elif self.state == "approach":
            toy = self._nearest(senses, "toy")
            if toy is not None and toy.name == self.target_name:
                self._update_estimate(odom, toy, p.toy_z)
                self.t_seen = t
            twist, dist, _ = self._servo(odom, p.reach_ahead + 0.01, p.reach_left)
            if dist <= p.reach_ahead + 0.01:
                self._enter("settle", t)
            elif t - self.t_seen > 1.0 and dist < 0.3:
                self._enter("blind", t)                  # it left the camera's view: dead-reckon the rest
            elif t - self.t_seen > 3.0 or t - self.t_state > 25.0:
                self._enter("scan", t)

        elif self.state == "blind":
            twist, dist, _ = self._servo(odom, p.reach_ahead + 0.01, p.reach_left)
            if dist <= p.reach_ahead + 0.01:
                self._enter("settle", t)
            elif t - self.t_state > 6.0:
                self._enter("scan", t)

        elif self.state == "settle":
            head = (0.0, 0.0, 0.0, 0.0)
            if t - self.t_state >= p.settle_s:
                skill = "ground_pick"
                self._enter("pick", t)

        elif self.state == "pick":
            if senses.skill is None and t - self.t_state > 0.5:
                self._enter("verify", t)

        elif self.state == "verify":
            if senses.holding:
                self.picked += 1
                self.est = None
                self._enter("carry", t)
            else:
                n = self.retries.get(self.target_name or "", 0) + 1
                self.retries[self.target_name or ""] = n
                if n > p.max_retries:
                    self.given_up.add(self.target_name or "")
                self.est = None
                self._enter("scan", t)

        elif self.state == "carry":
            basket = self._nearest(senses, "basket")
            if basket is not None:
                self.est, self.goal_kind, self.target_name = None, "basket", "basket"
                self._update_estimate(odom, basket, p.basket_z)
                self.t_seen = t
                self._enter("deliver", t)
            else:
                twist = (0.0, 0.0, p.scan_wz)
                if t - self.t_state > 14.0:              # a full turn and then some: walk somewhere else
                    self._enter("carry_explore", t)

        elif self.state == "carry_explore":
            head = (0.0, p.head_down, 0.0, 0.0)
            if self._nearest(senses, "basket") is not None:
                self._enter("carry", t)
            else:
                tof = senses.fresh_tof(self.TOF_MAX_AGE)
                twist = wander_from_tof(tof.depth_mm, tof.valid, WanderParams(rows=(0, 3))) if tof is not None else (0.0, 0.0, 0.0)
                if twist[0] == 0.0 and twist[2] != 0.0:
                    twist = (0.0, 0.0, 1.0 if twist[2] > 0 else -1.0)
                if t - self.t_state > p.explore_s:
                    self._enter("carry", t)

        elif self.state == "deliver":
            basket = self._nearest(senses, "basket")
            if basket is not None:
                self._update_estimate(odom, basket, p.basket_z)
                self.t_seen = t
            twist, dist, _ = self._servo(odom, p.basket_reach)
            if dist <= p.basket_reach:
                self._enter("drop", t)
            elif t - self.t_seen > 4.0 or t - self.t_state > 30.0:
                self._enter("carry", t)

        elif self.state == "drop":
            beak = "open"
            if t - self.t_state > 0.6:
                if not senses.holding:
                    self.delivered += 1
                self._enter("backoff", t)

        elif self.state == "backoff":
            twist = (-0.2, 0.0, 0.0)
            if t - self.t_state > p.backoff_s:
                self.est = None
                self.scan_turned = 0.0
                self._enter("scan", t)

        elif self.state == "done":
            twist = (0.0, 0.0, 0.0)

        # Never walk into whatever the ToF says is right there — read only its
        # top rows while the head is down, or they report the floor.
        if twist[0] > 0 and self.state in ("approach", "deliver", "explore", "carry_explore"):
            tof = senses.fresh_tof(self.TOF_MAX_AGE)
            if tof is not None:
                ahead = float(_column_clearance(tof.depth_mm, tof.valid, WanderParams(rows=(0, 2)))[3:5].min())
                if ahead < 0.25:
                    twist = (0.0, 0.0, twist[2] if twist[2] else 1.0)
                    note += " · blocked"

        self._prev_yaw = odom[2]
        self.last = twist
        return Intent(twist=twist, head=head, note=note, beak=beak, skill=skill)


REGISTRY.register("tidy", Tidy)
__all__ = ["Tidy", "TidyParams"]
