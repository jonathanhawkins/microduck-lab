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
on the floor leaves its field of view about half a metre out (the head
pitches down while walking in, which buys a few tens of centimetres), and
the basket's rim marker about 0.3 m out with the head level. From there
the brain walks a remembered point in odometry — exactly the "operator
aims the robot, the policy is blind" pattern upstream ships, automated.
Odometry is the truth here for now (roadmap 1.7 adds drift), and the real
robot's is contact-anchored and drifts, which is why the basket is
re-acquired visually every trip and never released on a long-range guess.

Per-toy retries and a give-up budget keep it from looping on a toy it
cannot grasp; two clean scans with nothing seen means "done".
"""

from __future__ import annotations

import collections
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..world.arena import PICK_REACH_AHEAD, PICK_REACH_LEFT
from .controllers import WanderParams, _column_clearance, wander_from_tof
from .gait import TURN_KICK, GaitWatch, back_up, max_wz, turn
from .mapping import GridSpec, OccupancyGrid
from .runtime import REGISTRY, Intent, Senses, age_inputs


@dataclass(frozen=True)
class TidyParams:
    reach_ahead: float = PICK_REACH_AHEAD
    reach_left: float = PICK_REACH_LEFT
    # Slack added to `reach_ahead` in the stop rule: the duck settles when
    # the toy's ESTIMATE is `reach_ahead + reach_pad` ahead. Not geometry
    # (`reach_ahead` is - it is where the beak tip lands) but a fitted
    # constant, and it was fitted against a BIASED estimate.
    #
    # It was +0.01 while `stale_fix` was off, and the two are a matched
    # pair: the old bias put the toy ahead along the travel direction, so
    # the duck walked past its own estimate and arrived at a TRUE 8.7 cm -
    # which is what the grasp wants. With the bias gone it stops honestly
    # and 1.7 cm short. -0.008 is that correction, taken from the measured
    # 8.7 -> 10.5 cm rather than fitted by search, so it is not tuned to
    # these 64 layouts. See `stale_fix` and AGENTS.md rule 7.
    #
    # It must NOT be changed without `stale_fix`: -0.008 with the fix OFF
    # is the worst cell of the 2x2 (-0.50 toys, p = 0.046, grasp 52%).
    reach_pad: float = -0.008
    approach_speed: float = 0.3        # the walker delivers ~half; below ~0.2 it stands still
    k_turn: float = 2.5
    head_down: float = 0.6             # head_pitch intent while walking in: camera ~37° down
    cam_ahead: float = 0.064           # camera ahead of the trunk origin (m)
    # Fallbacks for detection frames that do not carry the camera pose
    # (every frame from the lab's detector does): height / depression with
    # the head at rest, standing…
    cam_z_level: float = 0.234
    cam_pitch_level: float = 0.197     # (the stand pose already looks 11° down)
    cam_z_down: float = 0.202          # …and with head_pitch 0.6 (measured on the walker)
    cam_pitch_down: float = 0.647
    head_settle_s: float = 0.4         # the head takes ~0.3 s to get there: ignore detections meanwhile
    toy_z: float = 0.015               # a toy's centre height (m), class-blind
    basket_z: float = 0.08             # the basket marker's height (m)
    # Trunk-to-basket-centre at release. Measured on the walker: the feet
    # reach 0.034 m ahead of the trunk, the beak tip 0.080 m (head LEVEL —
    # pitching the head down pulls the tip back to 0.053), and a held toy
    # sits 0.005–0.023 m beyond the tip. A 0.3 m tray's rim outer face is
    # at 0.156, so 0.22 puts the feet 3 cm outside the rim and the toy
    # 2–4 cm inside it (releases at 0.23 landed in, at 0.25 on the rim).
    # That margin is what a brain tether spends: a stop decided 250 ms
    # late lands 4.7 cm further at a 0.3 command and 3.0 cm at 0.25
    # (`blind_speed`; below 0.25 the walker does not move at all), and
    # every tethered fall was that stride meeting the rim — the slower
    # leg took the tether row from 3.25 falls a run to 1.5. Pitching the
    # NECK back 0.6 rad pushes the tip out to 0.095 (`neck_reach`, so the
    # trunk could stop at 0.235) but the standing duck then creeps and
    # pitches forward into the rim during the drop: measured 0.75 falls a
    # run on ideal odometry against 0.50 without, so it ships OFF.
    # ASSUMPTION: a 0.3 m basket.
    basket_reach: float = 0.22
    neck_reach: float = 0.0            # neck_pitch intent on the blind end and the drop (measured, off)
    blind_speed: float = 0.25          # the last leg's command: slower, so a late stop lands nearer
    # A tethered brain's stop lands late by its round trip. The brain can
    # KNOW that: the floor of its ToF ages over the last second is the
    # link's one-way lag (0 onboard - the sensor runs at 15 Hz, so fresh
    # frames arrive with ages near zero), and the round trip is twice it.
    # The rim stop moves out by the measured speed times that
    # (`latency_gain` of it), so a 250 ms tether at ~0.16 m/s stops ~4 cm
    # earlier - the margin every traced tethered fall had spent.
    latency_gain: float = 1.0
    latency_max_s: float = 1.0
    basket_confirm_range: float = 0.6  # release only if the marker was seen from closer than this
    far_range: float = 1.2             # beyond this a sighting is a direction, not a range (see _locate)
    aim_range: float = 0.42            # stop here, square up, stand still and re-measure the basket before the blind end
    aim_settle_s: float = 0.6          # the walker needs ~0.5 s to come to rest
    aim_s: float = 1.4
    aim_align: float = 0.08            # squared up when the basket is within this of the nose (rad)
    # Toys near the basket. Inside `basket_inside` a toy is in the tray or
    # hugging its rim: leave it. Between that and `basket_zone` it lies where
    # every fall in 8 traced runs happened — an approach or back-off at the
    # rim after a toy 0.2–0.26 m out — so it is approached from the OUTSIDE:
    # walk to a staging point `stage_out` beyond it on the ray from the
    # basket, routed around the basket disc, then come in radially, feet
    # away from the rim (trunk 0.078 behind the toy, feet 0.034 ahead of
    # the trunk: 4 cm clear of a 0.156 rim face at 0.2 m out).
    basket_inside: float = 0.18
    basket_zone: float = 0.33
    stage_out: float = 0.45          # beyond the camera's blind range: the approach proper re-ranges the toy on the way in
    basket_avoid: float = 0.36         # a route that passes closer than this to the basket centre gets a via-point…
    basket_avoid_r: float = 0.45       # …this far out from it
    route_stop: float = 0.06           # a route point counts as reached inside this
    route_s: float = 14.0              # a route that takes longer than this is abandoned
    basket_keepout: float = 0.5        # exploring turns away from the basket inside this radius
    settle_s: float = 0.6
    # After a release: sidestep LEFT this long first (the feet stand 2–5 cm
    # from the rim after a drop, less under a brain tether or drift, and a
    # turn in place from there trips on it — measured standing 0.17–0.23 m
    # from the basket centre: the plain left turn fell 3/3 at 0.17, the
    # kicked one and a right turn 3/3 everywhere, a left sidestep first
    # 0/3 everywhere), then turn LEFT `backoff_turn` in place, then walk
    # straight `backoff_walk_s` before scanning again.
    backoff_side_s: float = 1.5
    backoff_side_vy: float = 0.3
    backoff_turn: float = 2.6
    backoff_walk_s: float = 1.5
    # ...OR back straight out for this long instead, and skip all three.
    #
    # The sequence above was justified by "the walker cannot walk
    # backwards", and that was a dead-band reading: -0.3 m/s moves 4 mm in
    # 6 s but -0.40 reverses at 0.228 m/s, steadily, cold or warm, with no
    # falls in any probe (`walker-facts`, `gait.back_up`). So the whole
    # ~7.3 s (1.5 sidestep + ~4.3 turning 2.6 rad at 0.6 rad/s + 1.5 walk)
    # buys a separation a 2 s reverse buys.
    #
    # The time is the smaller half of the argument. The sidestep exists
    # because a TURN IN PLACE at the rim falls (measured standing 0.17 m
    # from the basket centre: a plain left turn fell 3 of 3, a right turn
    # 3 of 3 everywhere, the sidestep-first 0 of 3) - and a reverse does
    # not turn at the rim at all, so it removes the fall mode rather than
    # dodging it. It leaves the duck FACING the basket, which `scan` is
    # happy with: it turns in place anyway, and by then the duck is outside
    # `basket_keepout`.
    #
    # MEASURED, and it is WORSE. 16 paired seeds x 300 s x 6 toys:
    # in the basket 5.31 -> 4.94 (0.885 -> 0.823 of the toys, -0.38 +/-
    # 0.18, p = 0.111, better on 2 of 16 and worse on 8, six ties), falls
    # 0.31 -> 0.44 (5 events against 7 - unresolvable either way at this
    # size, so the "it removes the fall mode" half is neither shown nor
    # refuted). Ships at 0.
    #
    # The saving was real and the diagnosis is the useful part. Traced over
    # 3 seeds, seconds a 300 s run:
    #
    #                backoff   scan   explore   approach   deliver
    #   sequence       39.5    60.1     20.5       64.0      72.2
    #   reverse 2 s     9.4    76.5     10.0      114.5      53.5
    #
    # The reverse does cut the back-off by 30 s a run, exactly as promised
    # - and `approach` grows by 50. The duck ends a reverse 0.56 m from the
    # basket centre (against 1.05 m after the sequence), which is on the
    # `basket_keepout` line, so it scans from beside the basket and then
    # walks the length of the room to whatever it finds.
    #
    # So the turn-and-walk is NOT overhead. "Leaving the rim" is what the
    # comment above called it; what it actually does is put the duck back
    # in the ROOM, where the toys are, pointed away from the basket. The
    # 30 s it costs buys shorter approaches for the rest of the cycle and
    # is paid back with interest. A faster back-off that skips the
    # repositioning is not faster.
    #
    # "A reverse that keeps the walk-out" was the first idea and it does
    # not exist: after backing out the duck FACES the basket, so there is
    # no walking clear without the 2.6 rad turn, which is the expensive
    # part. What the histogram actually indicts is the DISTANCE the duck
    # ends at, so the two coherent variants are both about reaching 1.05 m:
    #
    #   `backoff_back_s` 4-5   a longer straight reverse. Same distance,
    #                          about half the time. Ends facing the basket,
    #                          which `scan` turns out of anyway.
    #   `backoff_back_wz` != 0 reverse WHILE turning. The walker reverses at
    #                          -0.209 m/s along its own body axis with wz at
    #                          1.0 (against -0.219 straight) while turning at
    #                          0.30-0.40 rad/s, so the retreat and the
    #                          turn-around happen at once instead of in
    #                          series - see `gait.back_up`, which measured
    #                          the arc after a render caught the frame
    #                          mistake in reading it.
    #
    # Both ship at 0, and `backoff_back_s` 4.0 is now MEASURED at 80 seeds:
    #
    #                        in the basket            falls a run
    #   screen   0-15 (16)   5.31 -> 5.56  p=0.43   0.31 -> 0.62  p=0.40
    #   CONFIRM 16-79 (64)   5.27 -> 5.12  p=0.39   0.41 -> 0.77  p=0.003
    #   pooled        (80)   5.28 -> 5.21  p=0.69   0.39 -> 0.74  p=0.002
    #
    # The screen's +0.25 toys did not replicate - it reversed, and pooled
    # the tidied fraction is flat (0.879 -> 0.869, 25 seeds better, 25
    # worse, 30 tied). The FALLS did replicate and sharpened with n, which
    # is what a real effect does: +90% (31 events against 59), and not
    # exposure - work done is flat (picks +1%, deliveries +2%) so falls per
    # pick nearly doubled, 0.067 -> 0.128, p = 0.002.
    #
    # Where they come from, traced over 12 seeds: NOT the reverse itself
    # (1 of 11 falls happened while backing) and NOT the walls (median
    # clearance 1.25 m in both arms). Back-off falls actually went DOWN
    # (3 -> 2), so the "no turn at the rim" half was right. They moved to
    # `approach` (1 -> 4).
    #
    # The two arms are DISTANCE-MATCHED by construction (1.04 m against
    # 1.05 m), so the live variable left is orientation: the sequence ends
    # facing AWAY from the basket and a reverse ends facing it, and the
    # next route then sets out across the basket zone, whose 6 cm rim is
    # below the ToF guard until the last 0.26 m and trips the walker.
    #
    # So the 7.3 s buys two things, not one - distance AND heading - and
    # each variant bought a different half: 2 s gave neither (0.57 m) and
    # cost tidying; 4 s gives distance without heading, and costs falls.
    # That is the whole result, and it is why "make the back-off faster" is
    # the wrong frame.
    backoff_back_s: float = 0.0
    backoff_back_wz: float = 0.0
    # Place a detection with the odometry the duck had WHEN THE FRAME WAS
    # TAKEN, not the odometry it has now.
    #
    # `_locate` reads the camera's height and pitch off the frame ("the frame
    # says where the camera was") but takes x, y and yaw from the CURRENT
    # odometry - so a frame that is one detector period old is placed from a
    # pose the duck has already left. At `approach` speed that is 3 cm of
    # error at 10 Hz and 6 cm at 5 Hz, against a 3 cm toy.
    #
    # That is the shape of the measured detector-rate cliff: 10 -> 5 Hz costs
    # 0.55 of 6 toys (64 paired seeds, p = 0.0003), and it is the GRASP that
    # breaks - success 85% -> 67%, attempts per pick 1.19 -> 1.50 - while
    # picks barely move. The duck finds toys fine and fumbles them.
    #
    # The fix uses nothing the robot lacks: it keeps a second of its own
    # odometry and looks up the pose at the frame's timestamp. No ground
    # truth, no new sensor.
    #
    # MEASURED, and it only works PAIRED WITH `reach_pad` - alone it lost
    # 0.38 toys despite cutting the estimate's error from 5.4 to 3.7 cm,
    # because the stop distance had absorbed the bias. The 2x2, 64 paired
    # layouts, `reach_pad` moved with it:
    #
    #                       tidied   grasp        att/pick
    #   10 Hz, before        0.878   366/433 85%    1.19
    #   10 Hz, both          0.883   361/395 91%    1.09   +0.03, p = 0.88
    #    5 Hz, before        0.786   342/510 67%    1.50
    #    5 Hz, both          0.854   355/379 94%    1.07   +0.41, p = 0.0066
    #
    # So: NEUTRAL at 10 Hz and worth +0.41 toys at 5 Hz, with the grasp
    # better at both rates (it is the grasp this fixes; at 10 Hz the grasp
    # was not the binding constraint, which is why tidied does not move).
    #
    # That buys the thing it was built for. Halving the detector rate costs
    # -0.55 toys as it ships (p = 0.0003, worse on 35 of 64); with these
    # two the 5 Hz arm is -0.14 against 10 Hz and NOT RESOLVED (p = 0.32,
    # SE 0.12 - a real residual up to ~0.25 toys would not have been seen
    # here). The 10 Hz detector floor is no longer measurable, and frame
    # rate is what a bigger inference input is bought with
    # (docs/camera-hardware.md).
    stale_fix: bool = True
    turn_kick: float = TURN_KICK       # forward command that starts the gait for a cold turn (brain/gait.py)
    detour_s: float = 1.0              # after the ToF guard clears: walk straight this long before re-aiming
    hold_blind_m: float = 0.12         # ToF returns closer than this while holding are the toy in the beak
    scan_wz: float = 1.0               # below 1.0 a COLD walker does not turn at all (exactly 0, both ways)
    explore_s: float = 6.0             # wander this long after an empty scan
    min_conf: float = 0.1              # a tracked detection needs no more than this (see _trusted)
    # Odometry drift (roadmap 1.7): every estimate here lives in the odometry
    # frame, and a scale error or a gyro bias bends the trips. The brain
    # keeps its own room map (brain/mapping.py) and reads the LOOP-CLOSED
    # pose it gives back — the walls of the playroom are the anchor.
    loop_closure: bool = True
    map_size: tuple[float, float] = (6.0, 6.0)
    max_retries: int = 2               # pick attempts per toy before giving up on it
    done_after_scans: int = 6          # empty scans (each followed by a wander) before giving up


class Tidy:
    kind = "tidy"
    wants_head = True                  # the server applies this brain's head intents
    DET_MAX_AGE = 0.4
    TOF_MAX_AGE = 0.25

    def __init__(self, p: TidyParams = TidyParams()):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self._ages: deque = deque()
        self.latency = 0.0
        self.stop_margin = 0.0
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
        self._head_down = False
        self._head_since = -9.0
        self._cam: tuple[float, float] | None = None
        self._cam_yaw = 0.0
        self._cam_t: float | None = None
        self._odom_hist: collections.deque = collections.deque(maxlen=60)   # ~1.2 s at 50 Hz
        # Where toys were last seen (odom frame) — the work queue — and the
        # basket once it has been seen at all: it does not move.
        self.memory: dict[str, tuple[float, float, float]] = {}
        self.basket_mem: tuple[float, float] | None = None
        self.basket_confirmed = False    # seen from close enough that the estimate is trustworthy
        self.aimed = False               # this trip's standing re-measure of the basket is done
        self._aim_rounds = 0
        self._aim_turning = False
        self._backoff_t = 0.0
        self._blocked_t = -9.0
        self._gait = GaitWatch()
        self.map = OccupancyGrid(GridSpec(size=self.p.map_size)) if self.p.loop_closure else None
        self._aim_fixes: list[tuple[float, float]] = []
        self._aim_last_t = -9.0
        self._route: list[tuple[float, float]] = []      # where to walk before the approach proper
        self._route_t0 = -9.0

    def inputs(self) -> dict:
        if self._senses is None:
            return {}
        out = age_inputs(self._senses, self.TOF_MAX_AGE, self.DET_MAX_AGE)
        out["latency"] = round(self.latency, 3)
        out["target"] = None if self.est is None else {
            "bearing": 0.0, "range": None, "since": round(self._senses.t - self.t_seen, 2),
            "goal": [round(v, 3) for v in self.est], "kind": self.goal_kind, "name": self.target_name}
        out["tidy"] = {"picked": self.picked, "delivered": self.delivered,
                       "givenUp": sorted(self.given_up), "retries": dict(self.retries),
                       "loopClosure": None if self.map is None else
                       {"offset": [round(float(v), 3) for v in self.map.offset], "corrections": self.map.corrections}}
        return out

    # -- helpers --------------------------------------------------------------
    def _pose(self, senses: Senses) -> tuple[float, float, float]:
        """The (x, y, yaw) the brain steers by: odometry, loop-closed against
        the brain's own map when it has one (the map folds in each new
        ToF frame at its raw odometry and hands back the corrected pose)."""
        odom = senses.odom or (0.0, 0.0, 0.0)
        if self.map is None or senses.odom is None:
            return odom
        if senses.tof is not None:
            self.map.update(senses.tof, senses.odom)
        return self.map.correct(senses.odom)

    def _enter(self, state: str, t: float) -> None:
        self.state = state
        self.t_state = t

    def _nearest(self, senses: Senses, cls: str, odom=None):
        det = senses.fresh_det(self.DET_MAX_AGE)
        if det is None:
            return None
        cands = [d for d in det.detections if d.cls == cls and self._trusted(d) and d.name not in self.given_up]
        if cls == "toy" and odom is not None:
            cands = [d for d in cands if not self._in_basket_zone(odom, d, senses.t)]
        return min(cands, key=lambda d: d.range_est) if cands else None

    def _point_in_basket_zone(self, x: float, y: float) -> bool:
        return (self.basket_mem is not None and self.basket_confirmed
                and math.hypot(x - self.basket_mem[0], y - self.basket_mem[1]) < self.p.basket_inside)

    def _stage_point(self, x: float, y: float) -> tuple[float, float] | None:
        """For a toy at (x, y) in the rim zone of a confirmed basket: the
        point `stage_out` beyond it on the basket→toy ray, to come in from.
        None when the toy is not near the basket (or the basket is unknown)."""
        if self.basket_mem is None or not self.basket_confirmed:
            return None
        bx, by = self.basket_mem
        r = math.hypot(x - bx, y - by)
        if r < 1e-6 or r >= self.p.basket_zone:
            return None
        return x + self.p.stage_out * (x - bx) / r, y + self.p.stage_out * (y - by) / r

    def _via_point(self, start: tuple[float, float], goal: tuple[float, float]) -> tuple[float, float] | None:
        """A point to pass through so the straight walk from `start` to
        `goal` does not cross the basket disc: the closest point of the
        segment to the basket centre, pushed out to `basket_avoid_r`."""
        if self.basket_mem is None or not self.basket_confirmed:
            return None
        p = self.p
        bx, by = self.basket_mem
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return None
        u = ((bx - start[0]) * dx + (by - start[1]) * dy) / L2
        if u <= 0.0 or u >= 1.0:
            return None                              # the basket is not between the two
        cx, cy = start[0] + u * dx, start[1] + u * dy
        d = math.hypot(cx - bx, cy - by)
        if d >= p.basket_avoid:
            return None
        if d < 1e-6:
            ox, oy = -dy / math.sqrt(L2), dx / math.sqrt(L2)     # dead on: go round the left
        else:
            ox, oy = (cx - bx) / d, (cy - by) / d
        return bx + p.basket_avoid_r * ox, by + p.basket_avoid_r * oy

    def _plan_route(self, odom, t: float) -> None:
        """Set the route for the current toy estimate: nothing for a toy in
        the open; for one in the rim zone, the staging point, with a
        via-point first when the basket is in the way."""
        self._route = []
        stage = self._stage_point(*self.est) if self.est is not None else None
        if stage is None:
            return
        via = self._via_point((odom[0], odom[1]), stage)
        self._route = ([via] if via is not None else []) + [stage]
        self._route_t0 = t

    def _note_basket(self, senses: Senses, odom, t: float) -> None:
        """A close look at the basket while NOT carrying (scanning, walking
        at a toy) still fixes where it is — so the first trip's rim toys are
        staged too, not only those after a delivery."""
        if senses.holding or self.goal_kind == "basket":
            return
        det = self._nearest(senses, "basket")
        if det is None:
            return
        loc = self._locate(odom, det, self.p.basket_z, t)
        if loc is None or loc[2] >= self.p.basket_confirm_range:
            return
        if self.basket_mem is None or not self.basket_confirmed:
            self.basket_mem = (loc[0], loc[1])
        else:
            self.basket_mem = (self.basket_mem[0] + 0.5 * (loc[0] - self.basket_mem[0]),
                               self.basket_mem[1] + 0.5 * (loc[1] - self.basket_mem[1]))
        self.basket_confirmed = True

    def _trusted(self, det) -> bool:
        """A detection worth acting on is a TRACKED one — it has an id — not
        a confident one: a 2 cm toy at 1.5 m is 1.5° wide, found one frame
        in six with a confidence under 0.2, and that is still a toy worth
        walking toward (its range will be rough; the approach re-measures).
        Ghosts have no id. The sim hands out ids for free; on the robot they
        come from a tracker (roadmap 1.3), which is the honest boundary."""
        return bool(det.name) and det.conf >= self.p.min_conf

    def _in_basket_zone(self, odom, det, t: float) -> bool:
        """A toy that projects onto the basket is one already delivered (or
        lying against the rim): walking at it means walking into the rim —
        measured: every fall in a 300 s run was an approach to a toy sitting
        in the basket."""
        if self.basket_mem is None or not self.basket_confirmed:
            return False
        loc = self._locate(odom, det, self.p.toy_z, t)
        return loc is not None and self._point_in_basket_zone(loc[0], loc[1])

    def _locate(self, odom, det, target_z: float, t: float) -> tuple[float, float, float] | None:
        """One detection → (x, y, range) in the odom frame, from the
        elevation: a floor object seen from a known camera height and pitch
        gives range far better than its apparent width. None while the head
        is still moving."""
        p = self.p
        if t - self._head_since < p.head_settle_s:
            return None
        if self._cam is not None:
            cam_z, cam_pitch = self._cam                       # the frame says where the camera was
        else:
            cam_z, cam_pitch = (p.cam_z_down, p.cam_pitch_down) if self._head_down else (p.cam_z_level, p.cam_pitch_level)
        depression = max(cam_pitch - det.elevation, 0.05)     # rad below horizontal
        horiz_cam = float(np.clip((cam_z - target_z) / math.tan(depression), 0.0, 4.0))
        # The frame was taken up to one detector period ago; place it from
        # the pose of THEN, not of now (see `stale_fix`).
        x, y, yaw = (self._odom_at(self._cam_t, odom)
                     if (p.stale_fix and getattr(self, "_cam_t", None) is not None) else odom)
        a = yaw + det.bearing + getattr(self, "_cam_yaw", 0.0)      # camera-frame bearing → body → odom
        # Beyond `far_range` the elevation says almost nothing about range
        # (at 2.3 m the map is 34 m per radian: 0.02 rad of head bob is
        # 0.7 m — measured, an estimate that hopped between 1.3 and 3.4 m
        # and a duck that spun for four minutes). The bearing is still
        # good, so a far sighting means "walk that way, at least this far".
        rng = min(p.cam_ahead + horiz_cam, p.far_range)
        return x + rng * math.cos(a), y + rng * math.sin(a), rng

    def _odom_at(self, t: float, now):
        """The odometry the duck had at time `t` - the pose a frame stamped
        `t` should be placed from. Nearest sample within half a control step
        either side; `now` if the history cannot cover it (a fresh brain, or
        a frame older than the ring)."""
        best, best_dt = None, 1e9
        for ts, pose in self._odom_hist:
            dt = abs(ts - t)
            if dt < best_dt:
                best, best_dt = pose, dt
        return best if (best is not None and best_dt <= 0.5) else now

    def _set_head(self, down: bool, t: float) -> None:
        if down != self._head_down:
            self._head_down = down
            self._head_since = t

    def _update_estimate(self, odom, det, target_z: float, t: float, k: float | None = None) -> float | None:
        """Fold one detection into the odom-frame target estimate. Returns
        the MEASURED horizontal range from the trunk, which is also how much
        the measurement is trusted: the elevation-to-range map is steep at
        distance (at 1 m, 0.01 rad of body pitch is 6 cm of range; walking
        adds far more than that), so a far sighting mostly sets the
        direction and a close one the spot; `k` overrides (a standing,
        settled look is the best measurement there is)."""
        p = self.p
        loc = self._locate(odom, det, target_z, t)
        if loc is None:
            return None
        tx, ty, rng = loc
        if self.est is None or rng >= p.far_range:
            self.est = (tx, ty)                                  # a far fix is a direction: replace, never average
        else:
            if k is None:
                k = 0.7 if rng < p.basket_confirm_range else 0.3
            self.est = (self.est[0] + k * (tx - self.est[0]), self.est[1] + k * (ty - self.est[1]))
        if det.cls == "toy" and det.name:
            self.memory[det.name] = (self.est[0], self.est[1], t)
        elif det.cls == "basket":
            self.basket_mem = self.est
            if rng < p.basket_confirm_range:
                self.basket_confirmed = True
        return rng

    def _servo(self, odom, stop_at: float, left: float = 0.0,
               align: float = 0.35) -> tuple[tuple[float, float, float], float, float]:
        """Walk toward the estimate, turning in place first if it is more
        than `align` off the nose, and stop `stop_at` short of it. `left`
        biases the aim so an off-centre beak lands on the target."""
        x, y, yaw = odom
        dx, dy = self.est[0] - x, self.est[1] - y
        dist = math.hypot(dx, dy)
        want = math.atan2(left, max(dist, 0.05))
        bearing = math.atan2(math.sin(math.atan2(dy, dx) - yaw - want), math.cos(math.atan2(dy, dx) - yaw - want))
        if dist <= stop_at:
            return (0.0, 0.0, 0.0), dist, bearing
        if abs(bearing) > align:
            return self._turn(bearing), dist, bearing
        return (self.p.approach_speed, 0.0, float(np.clip(self.p.k_turn * bearing, -1.0, 1.0))), dist, bearing

    def _turn(self, sign: float) -> tuple[float, float, float]:
        """Turn in place (brain/gait.py: a cold gait needs the forward kick)."""
        return turn(sign, self._gait.cold, self.p.turn_kick)

    def _tof_view(self, senses: Senses):
        """(depth_mm, valid) for obstacle logic. A held toy sits 2–3 cm from
        the sensor, in the bottom rows of the centre columns (measured: ToF
        minimum 25 mm all through a carry, and a guard that read it as a
        wall — 3 minutes of "blocked" with a toy in the beak). While
        holding, returns closer than `hold_blind_m` are the toy, not the room."""
        tof = senses.fresh_tof(self.TOF_MAX_AGE)
        if tof is None:
            return None, None
        if senses.holding:
            near = tof.depth_mm < int(self.p.hold_blind_m * 1000)
            return tof.depth_mm, tof.valid & ~near
        return tof.depth_mm, tof.valid

    def _bumper(self, senses: Senses) -> float:
        depth, valid = self._tof_view(senses)
        if depth is None:
            return math.inf
        return float(_column_clearance(depth, valid, WanderParams())[3:5].min())

    # -- the machine ---------------------------------------------------------
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        p, t = self.p, senses.t
        # The link's lag, read off the sensor ages (see latency_gain).
        if senses.tof_age is not None:
            self._ages.append((t, float(senses.tof_age)))
        while self._ages and t - self._ages[0][0] > 1.0:
            self._ages.popleft()
        self.latency = min(p.latency_max_s, 2.0 * min((a for _, a in self._ages), default=0.0))
        self.stop_margin = p.latency_gain * max(0.0, float(senses.speed)) * self.latency
        odom = self._pose(senses)
        self._odom_hist.append((t, odom))              # for `stale_fix`: where the duck WAS
        fr = senses.fresh_det(self.DET_MAX_AGE)
        self._cam = (fr.cam_z, fr.cam_pitch) if (fr is not None and fr.cam_z > 0.0) else None
        self._cam_yaw = getattr(fr, "cam_yaw", 0.0) if fr is not None else 0.0
        # WHEN the frame was taken. A `Detection` carries no timestamp - only
        # the frame does - so `stale_fix` reads it from here, not from `det`.
        self._cam_t = fr.t if fr is not None else None
        self._gait.update(senses)                      # is the gait going? (see _turn)
        if self.state in ("scan", "explore", "approach"):
            self._note_basket(senses, odom, t)
        twist = (0.0, 0.0, 0.0)
        head = (0.0, 0.0, 0.0, 0.0)
        beak = None
        skill = None
        note = self.state
        # The walker cannot turn in place with its head down (measured: 0.2 rad
        # in 5 s vs 3.1 level), so the head drops only while walking straight
        # at a target; scanning and turning happen level.
        head_down = False

        if self.state == "scan":
            toy = self._nearest(senses, "toy", odom)
            if toy is not None and self._update_estimate(odom, toy, p.toy_z, t) is not None:
                self.goal_kind, self.target_name = "toy", toy.name
                self.t_seen = t
                self.scans_empty = 0
                self._plan_route(odom, t)
                self._enter("approach", t)
            else:
                # Remember toys seen while turning even when they are not the
                # target: a full turn builds the work queue.
                det = senses.fresh_det(self.DET_MAX_AGE)
                if det is not None and t - self._head_since >= p.head_settle_s:
                    for dd in det.detections:
                        if dd.cls == "toy" and self._trusted(dd) and dd.name not in self.given_up \
                                and not self._in_basket_zone(odom, dd, t):
                            saved = self.est
                            self.est = None
                            self._update_estimate(odom, dd, p.toy_z, t)
                            self.est = saved
                twist = (0.0, 0.0, max(p.scan_wz, max_wz()) if p.scan_wz >= 1.0 else p.scan_wz)
                if self._prev_yaw is not None:
                    d = odom[2] - self._prev_yaw
                    self.scan_turned += math.atan2(math.sin(d), math.cos(d))   # SIGNED: the gait wobbles ±0.02 rad a step
                if self.scan_turned >= 2 * math.pi:
                    self.scan_turned = 0.0
                    remembered = [(n, m) for n, m in self.memory.items()
                                  if n not in self.given_up and not self._point_in_basket_zone(m[0], m[1])]
                    if remembered:
                        # Nothing in view now, but the queue is not empty: walk
                        # to the nearest remembered toy and re-acquire it there.
                        x, y, _ = odom
                        name, (mx, my, _) = min(remembered, key=lambda nm: math.hypot(nm[1][0] - x, nm[1][1] - y))
                        self.est, self.goal_kind, self.target_name = (mx, my), "toy", name
                        self.t_seen = t
                        self.scans_empty = 0
                        self._plan_route(odom, t)
                        self._enter("approach", t)
                    else:
                        self.scans_empty += 1
                        self._enter("done" if self.scans_empty >= p.done_after_scans else "explore", t)
                note = f"scan {self.scans_empty}/{p.done_after_scans}"

        elif self.state == "explore":
            if self._nearest(senses, "toy", odom) is not None:
                self._enter("scan", t)
            else:
                depth, valid = self._tof_view(senses)
                twist = wander_from_tof(depth, valid) if depth is not None else (0.0, 0.0, 0.0)
                if twist[0] == 0.0 and twist[2] != 0.0:
                    twist = self._turn(twist[2])
                if t - self.t_state > p.explore_s:
                    self._enter("scan", t)

        elif self.state == "approach" and self._route:
            # A rim toy: walk the route (via-point, staging point) head LEVEL
            # first; the approach proper starts from the staging point, radial.
            toy = self._nearest(senses, "toy", odom)
            if toy is not None and toy.name == self.target_name:
                if self._update_estimate(odom, toy, p.toy_z, t) is not None:
                    stage = self._stage_point(*self.est)
                    if stage is not None:
                        self._route[-1] = stage
            saved = self.est
            self.est = self._route[0]
            twist, dist, _ = self._servo(odom, p.route_stop)
            self.est = saved
            self.t_seen = t                              # the toy is not expected in view on the way round
            note = f"approach · route {len(self._route)}"
            if dist <= p.route_stop:
                self._route.pop(0)
            elif t - self._route_t0 > p.route_s:
                self._route = []
                self.est = None
                self._enter("scan", t)

        elif self.state == "approach":
            toy = self._nearest(senses, "toy", odom)
            if toy is not None and toy.name == self.target_name:
                if self._update_estimate(odom, toy, p.toy_z, t) is not None:
                    self.t_seen = t
            # The stop lands late by the link (stop_margin, see latency_gain): a
            # tethered pick overshot the toy and came up empty - traced: 3
            # picks in 6 attempts after the first three toys, 80 s of scans.
            twist, dist, bearing = self._servo(odom, p.reach_ahead + p.reach_pad + self.stop_margin, p.reach_left)
            head_down = twist[0] > 0 or (dist < 0.5 and abs(bearing) <= 0.35)
            if dist <= p.reach_ahead + p.reach_pad + self.stop_margin:
                self._enter("settle", t)
            elif t - self.t_seen > 1.0 and dist < 0.3 and abs(bearing) <= 0.35:
                # It left the camera's view: dead-reckon the rest. Not while
                # still turning to face it (after a staged route the duck
                # arrives side-on and has not had a look yet — measured: a
                # blind leg on a 9 cm-stale estimate, and a miss).
                self._enter("blind", t)
            elif t - self.t_seen > 4.0 or t - self.t_state > 25.0:
                self.est = None
                self._enter("scan", t)

        elif self.state == "blind":
            twist, dist, _ = self._servo(odom, p.reach_ahead + p.reach_pad + self.stop_margin, p.reach_left)
            if self._point_in_basket_zone(*self.est):
                self.est = None                          # it is in (or against) the basket after all
                self._enter("scan", t)
            elif dist <= p.reach_ahead + p.reach_pad + self.stop_margin:
                self._enter("settle", t)
            elif t - self.t_state > 6.0:
                self.est = None
                self._enter("scan", t)

        elif self.state == "settle":
            if t - self.t_state >= p.settle_s:
                skill = "ground_pick"
                self._enter("pick", t)

        elif self.state == "pick":
            if senses.skill is None and t - self.t_state > 0.5:
                self._enter("verify", t)

        elif self.state == "verify":
            self.memory.pop(self.target_name or "", None)
            if senses.holding:
                self.picked += 1
                self.est = None
                self.goal_kind = None
                self._enter("carry", t)
            else:
                n = self.retries.get(self.target_name or "", 0) + 1
                self.retries[self.target_name or ""] = n
                if n > p.max_retries:
                    self.given_up.add(self.target_name or "")
                self.est = None
                self._enter("scan", t)

        elif self.state == "carry" and not senses.holding:
            self.est, self.goal_kind = None, None          # lost it (a fall releases the beak): start over
            self._enter("scan", t)

        elif self.state == "carry":
            self.aimed = False
            self._aim_rounds = 0
            # The basket does not move and odometry is the truth here, so a
            # basket seen up close on an earlier trip beats a fresh sighting
            # from across the room (measured: a far sighting after a fall
            # once put the estimate 0.8 m off and stayed there for five trips).
            basket = self._nearest(senses, "basket")
            if self.basket_mem is not None and self.basket_confirmed:
                self.est, self.goal_kind, self.target_name = self.basket_mem, "basket", "basket"
                self.t_seen = t
                self._enter("deliver", t)
            elif basket is not None:
                if self.goal_kind != "basket":
                    self.est, self.goal_kind, self.target_name = None, "basket", "basket"
                if self._update_estimate(odom, basket, p.basket_z, t) is not None:
                    self.t_seen = t
                    self._enter("deliver", t)
            elif self.basket_mem is not None:
                self.est, self.goal_kind, self.target_name = self.basket_mem, "basket", "basket"
                self.t_seen = t
                self._enter("deliver", t)
            else:
                twist = (0.0, 0.0, max(p.scan_wz, max_wz()) if p.scan_wz >= 1.0 else p.scan_wz)
                if t - self.t_state > 14.0:              # a full turn and then some: walk somewhere else
                    self._enter("carry_explore", t)

        elif self.state == "carry_explore":
            if self._nearest(senses, "basket") is not None:
                self._enter("carry", t)
            else:
                depth, valid = self._tof_view(senses)
                twist = wander_from_tof(depth, valid) if depth is not None else (0.0, 0.0, 0.0)
                if twist[0] == 0.0 and twist[2] != 0.0:
                    twist = self._turn(twist[2])
                if t - self.t_state > p.explore_s:
                    self._enter("carry", t)

        elif self.state in ("deliver", "aim") and not senses.holding:
            self.est, self.goal_kind = None, None
            self._enter("scan", t)

        elif self.state == "deliver":
            # Head LEVEL the whole way: the marker sits on the rim at 8 cm,
            # so a level camera keeps it until ~0.3 m out (a head-down camera
            # loses it at 0.8 m and the leg turns into dead reckoning on a
            # long-range estimate — measured drops 0.8–1.6 m from the basket).
            if not self.aimed:
                basket = self._nearest(senses, "basket")
                if basket is not None and self._update_estimate(odom, basket, p.basket_z, t) is not None:
                    self.t_seen = t
                twist, dist, bearing = self._servo(odom, p.basket_reach)
                if dist <= p.aim_range:
                    twist = (0.0, 0.0, 0.0)
                    self._aim_fixes = []
                    self._enter("aim", t)
            else:
                # The last 0.2 m after the standing look: walk with gentle
                # steering only. Alternating turn-in-place and walk here once
                # overran the stop by 6 cm and tripped on the rim (the plain
                # stop coasts 1 cm); a straight leg with NO steering once
                # walked off in whatever direction the ToF guard had left it.
                twist, dist, bearing = self._servo(odom, p.basket_reach + self.stop_margin)
                if twist[0] > p.turn_kick:
                    # …and none at all in the last few centimetres: a stop
                    # out of a steering step lunged 2–3 cm instead of the
                    # 1 cm coast, which is the whole margin at the rim.
                    wz = 0.0 if dist < p.basket_reach + self.stop_margin + 0.06 else float(np.clip(twist[2], -0.5, 0.5))
                    twist = (p.blind_speed, 0.0, wz)
            if dist <= p.basket_reach + self.stop_margin:
                if self.basket_confirmed:
                    self._enter("drop", t)
                else:
                    # Arrived on a long-range guess without ever seeing the
                    # basket up close: it is somewhere near — turn and look.
                    self.est, self.goal_kind, self.basket_mem = None, None, None
                    self._enter("carry", t)
            elif t - self.t_state > 30.0:
                self.est = None
                self.goal_kind = None
                self.basket_mem = None
                self.basket_confirmed = False
                self._enter("carry", t)

        elif self.state == "aim":
            # Stand still, square up, and look: walking rocks the head a few
            # hundredths of a radian, which at 0.4 m is centimetres of range —
            # more than the release geometry has to spare. First turn until
            # the basket is on the nose (so the blind leg needs no steering),
            # then a settled look (frames CAPTURED after the walker came to
            # rest) replaces the running estimate almost entirely.
            _, dist, bearing = self._servo(odom, p.basket_reach)
            if abs(bearing) > p.aim_align and self._aim_rounds < 3:
                twist = self._turn(bearing)
                self.t_state = t                            # the look starts once squared up
                self._aim_turning = True
            else:
                if self._aim_turning:
                    self._aim_turning = False
                    self._aim_rounds += 1
                settled = t - self.t_state >= p.aim_settle_s
                fr = senses.fresh_det(self.DET_MAX_AGE)
                if settled and fr is not None and fr.t >= self.t_state + p.aim_settle_s and fr.t > self._aim_last_t:
                    self._aim_last_t = fr.t
                    basket = self._nearest(senses, "basket")
                    if basket is not None:
                        loc = self._locate(odom, basket, p.basket_z, t)
                        if loc is not None:
                            self._aim_fixes.append((loc[0], loc[1]))
                            self.t_seen = t
                if t - self.t_state >= p.aim_s:
                    if self._aim_fixes:              # the mean of the standing looks, not the last one
                        self.est = (float(np.mean([f[0] for f in self._aim_fixes])),
                                    float(np.mean([f[1] for f in self._aim_fixes])))
                        self.basket_mem, self.basket_confirmed = self.est, True
                    self._aim_fixes = []
                    self.aimed = True
                    self._enter("deliver", t)

        elif self.state == "drop":
            beak = "open"
            if t - self.t_state > 0.6:
                if not senses.holding:
                    self.delivered += 1
                self.scan_turned = 0.0
                self._backoff_t = t
                self._enter("backoff", t)

        elif self.state == "backoff":
            # Leaving the rim is a left sidestep, a left turn-around on the
            # spot and a short straight walk. Scanning right at the rim once
            # put a foot on it.
            #
            # ...unless `backoff_back_s` is set, in which case the duck just
            # backs out (see the parameter: the walker reverses at 0.228 m/s,
            # and backing out never turns at the rim at all).
            def _done_backing_off() -> None:
                self.est = None
                self.scan_turned = 0.0
                self._enter("scan", t)

            if p.backoff_back_s > 0:
                if t - self.t_state < p.backoff_back_s:
                    twist = back_up(p.backoff_back_wz)
                    note += " · reverse" + ("" if not p.backoff_back_wz else " (arc)")
                else:
                    _done_backing_off()
            else:
                if self._prev_yaw is not None and self.scan_turned < p.backoff_turn:
                    d = odom[2] - self._prev_yaw
                    self.scan_turned += math.atan2(math.sin(d), math.cos(d))
                if t - self.t_state < p.backoff_side_s:
                    twist = (0.0, p.backoff_side_vy, 0.0)
                    self._backoff_t = t
                    note += " · sidestep"
                elif self.scan_turned < p.backoff_turn:
                    twist = self._turn(+1.0)
                    self._backoff_t = t
                elif t - self._backoff_t < p.backoff_walk_s:
                    twist = (p.approach_speed, 0.0, 0.0)
                else:
                    _done_backing_off()

        elif self.state == "done":
            twist = (0.0, 0.0, 0.0)

        # The rim is 6 cm high: too low for the ToF's guard until the last
        # 0.26 m, and the walker trips on it. Exploring never heads at a
        # confirmed basket (measured: most falls were explore legs ending
        # 0.2 m from its centre).
        if twist[0] > p.turn_kick and self.state in ("explore", "carry_explore") and self.basket_mem is not None \
                and self.basket_confirmed:
            bdx, bdy = self.basket_mem[0] - odom[0], self.basket_mem[1] - odom[1]
            bdist = math.hypot(bdx, bdy)
            bb = math.atan2(math.sin(math.atan2(bdy, bdx) - odom[2]), math.cos(math.atan2(bdy, bdx) - odom[2]))
            # A disc, not a cone: skirting the basket tangentially at 0.23 m
            # once clipped the rim's corner. Always a LEFT turn — a right
            # turn from a standstill does not happen (see _turn), and the
            # kicked one creeps forward, which is the wrong way here.
            if bdist < p.basket_keepout and abs(bb) < 1.2:
                twist = self._turn(+1.0)
                note += " · basket keep-out"

        self._set_head(head_down, t)
        if self._head_down:
            head = (0.0, p.head_down, 0.0, 0.0)
        elif self.aimed and self.state in ("deliver", "drop") and p.neck_reach:
            head = (p.neck_reach, 0.0, 0.0, 0.0)     # the beak out over the rim, the feet further from it
        # Never walk into whatever the ToF says is right there — read only its
        # top rows while the head is down, or they report the floor.
        guard = self.state in ("approach", "explore", "carry_explore") or (
            self.state == "deliver" and not (self.aimed and self.est is not None
                                             and math.hypot(self.est[0] - odom[0], self.est[1] - odom[1]) < 0.5))
        if guard:
            depth, valid = self._tof_view(senses)
            if depth is not None:
                rows = (0, 2) if self._head_down else (2, 7)
                ahead = float(_column_clearance(depth, valid, WanderParams(rows=rows))[3:5].min())
                if t - self._blocked_t < p.detour_s and ahead >= 0.25 and self.state in ("approach", "deliver"):
                    # Just cleared a blocker: one straight step past it
                    # WHATEVER the servo wants (it wanted to turn back into
                    # it — measured: a left/right ping-pong at a toy, 4 min).
                    twist = (p.approach_speed, 0.0, 0.0)
                    note += " · detour"
                elif twist[0] > p.turn_kick and ahead < 0.25:
                    # Turn LEFT at full rate (the servo's steering rate never
                    # turns the walker — measured: 200 s "blocked" on one spot).
                    twist = self._turn(+1.0)
                    self._blocked_t = t
                    note += " · blocked"

        self._prev_yaw = odom[2]
        self.last = twist
        return Intent(twist=twist, head=head, note=note, beak=beak, skill=skill)


REGISTRY.register("tidy", Tidy)
__all__ = ["Tidy", "TidyParams"]
