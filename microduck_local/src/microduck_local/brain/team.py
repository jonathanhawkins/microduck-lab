"""A team's shared blackboard (soccer, second form): who attacks the ball
and where the ball was last seen, for ducks that cannot tell a teammate
from an opponent by sight.

On the robot this is one small message a second over Wi-Fi between
teammates — id, what the ball is going to COST me, the ball's position and
my own pose in my odometry frame — which is exactly what `claim` carries.
Nothing here reads the sim. The poses are what a duck cannot get any other
way: a teammate beside or behind it is invisible to its camera and its
ToF, and most 3v3 falls were a supporter walking or turning into one
(`mates`, and the chase brain's `mate_keepout`).

Roles: the teammate that will REACH the ball soonest attacks (chase, line
up, kick or push); the others support, standing back toward their own
goal, spread sideways by rank.

The cost is a predicted time, not a distance. Distance was the first
form and it churned: over 3 seeds x 300 s of 3v3 the role changed hands
11.6 times a duck a run, a quarter of the spells lasted under a second,
and the designated attacker was the team's actually-nearest duck only
54% of the time. Three things were wrong with a straight line:

  * **It ignores the turn.** This walker turns in place at ~0.7 rad/s
    once the gait is kicked and barely at all cold (`walker-facts`), and
    walks at 0.45 m/s (`ChaseParams.speed`). A duck facing away at 0.4 m
    is four seconds from the ball; one facing it at 0.6 m is one. The
    line said the first was nearer, and the first turned round while the
    second was sent back to its support spot.
  * **Losing sight was a resignation.** The chase brain claims
    `inf` the moment its track goes cold (`lost_s`), and the level
    camera loses a floor ball inside ~0.3 m — exactly where an attacker
    lines up. So the duck ON the ball handed the role to one a metre
    away, walked off, and took it back when it saw the ball again.
    A duck that cannot see the ball now costs the board's freshest
    sighting plus `blind_s`: behind a duck that can see it, ahead of one
    that is genuinely further.
  * **A stale claim competed on equal terms.** A claim is worth its age:
    `age_rate` seconds of cost per second of age (on the robot, claims
    arrive a second apart and half of them are the older one). Past
    `stale_s` it stops counting at all.

And the hysteresis is a margin held for a WHILE, not a margin: a
challenger has to be `switch_s` better for `hold_s` continuously before
the role moves — unless it is `give_up_s` better, which is the incumbent
falling out of the play (fallen over, or the ball kicked past it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Claim:
    t: float
    dist: float                              # my distance to the ball (inf: not seen lately)
    ball: tuple[float, float] | None         # where I put the ball (odom frame; the pitch's frame at spawn)
    pos: tuple[float, float, float] | None = None   # where I am (x, y, yaw; the same frame)
    cost: float = math.inf                   # my predicted TIME to reach it (s), filled in by `Team.claim`
    ball_sigma: float = math.nan             # my own 1-sigma error of `ball` (Track.sigma; nan = not said)


# The thirds a static role owns, in ATTACK coordinates: a = +1 at the goal
# this team attacks, -1 at the one it defends (so both teams read the same
# numbers). A duck may take the ball wherever its own third allows; a duck
# with no role may take it anywhere, which is what every roster did before
# roles existed. `zones_for` replaces this with a halfway split when the
# roster has a defender and a striker and no midfielder — otherwise the
# middle third belongs to nobody and `candidates` falls back to everybody.
ROLE_ZONES: dict[str, tuple[float, float]] = {
    "defender": (-1.0, -1.0 / 3.0),
    "midfielder": (-1.0 / 3.0, 1.0 / 3.0),
    "striker": (1.0 / 3.0, 1.0),
    # THE KEEPER (roadmap Track 4 s6 B.2): the last fifth in front of its own
    # mouth. It takes the ball only there, and - unlike every other job - it
    # never covers for a teammate and is never the "everybody may" fallback:
    # a keeper up the pitch is an open goal.
    "keeper": (-1.0, -0.8),
}


def zones_for(jobs: dict[str, str]) -> dict[str, tuple[float, float]]:
    """Attack-axis intervals for the jobs that are actually on the roster.

    Thirds when a midfielder is present. A defender+striker pair (no mid)
    splits at halfway so a ball at midfield is inside someone's zone and
    `candidates` does not fall back to "everybody may"."""
    present = {j for j in jobs.values() if j in ROLE_ZONES}
    if "keeper" in present:
        # The keeper owns the box; the field players share the rest exactly
        # as they would without one, lifted off the box.
        field = present - {"keeper"}
        if field == {"striker"}:
            out = {"striker": (-0.8, 1.0)}
        elif field == {"defender", "striker"}:
            out = {"defender": (-0.8, 0.0), "striker": (0.0, 1.0)}
        else:
            out = {j: (max(lo, -0.8), hi) for j, (lo, hi) in ROLE_ZONES.items() if j != "keeper"}
        out["keeper"] = ROLE_ZONES["keeper"]
        return out
    if present == {"defender", "striker"}:
        return {"defender": (-1.0, 0.0), "striker": (0.0, 1.0)}
    return dict(ROLE_ZONES)


@dataclass
class Team:
    name: str
    stale_s: float = 1.0
    # --- the walker the cost is predicted for (measured, not assumed) ------
    # `walker-facts`: a turn in place runs at ~0.7 rad/s once the gait is
    # going (`turn` always kicks it), and 0.25 rad in the first cold second
    # - `cold_s` is that start-up. `speed` is `ChaseParams.speed`, the walk
    # at the ball, and `turn_free` is the bearing the chase's own steering
    # absorbs while walking (`_go` turns in place only beyond 0.5 rad), so
    # only the turn past it is time spent NOT closing. `reach` is how near
    # the trunk has to get: the kick spot is ~0.12 m from the ball's centre.
    speed: float = 0.45
    turn_rate: float = 0.7
    turn_free: float = 0.5
    cold_s: float = 0.4
    reach: float = 0.15
    # COST THE SIDE OF THE BALL, NOT JUST THE DISTANCE TO IT (roadmap F.2).
    # `_cost` answers "how many seconds to REACH the ball", and the team hands
    # the ball to whoever answers lowest. It says nothing about what that duck
    # can DO when it arrives: a duck 0.4 m away on the goal side of the ball
    # gets there first and can only knock it backwards, while a teammate 0.8 m
    # away behind it would arrive able to shoot.
    #
    # Every other attempt at this problem made the badly-placed duck walk
    # round, and every one of them cost about a quarter of the team's touches
    # (36 a run -> 24-30, the one delta that resolved in any arm). This costs
    # NOTHING: both ducks walk exactly as far as they were going to, the ball
    # is simply assigned to the better-placed one. It is also the only version
    # that can work without a walk-round at all.
    #
    # The penalty is the seconds the walk-round WOULD take if that duck took
    # the ball: the arc from where it stands to the far side, at `behind_r`
    # radius and `speed`. 0 = off, which is the pre-2026-09-12 board.
    side_s: float = 0.0
    behind_r: float = 0.30
    blind_s: float = 1.0           # not seeing the ball yourself is worth this much time
    age_rate: float = 1.0          # …and so is every second a claim has been sitting on the board
    # --- the hysteresis ----------------------------------------------------
    # Swept on the trace's own numbers (3 seeds x 300 s of 3v3, the same
    # window, the arms interleaved): against 0.35 s / 0.6 s, the pair below
    # takes handovers from 12.3 to 9.5 a duck a run, the median spell from
    # 5.8 s to 7.2 s and its 25th percentile from 2.1 s to 3.2 s, and the
    # spells under a second from 13% to 9% - while the board's attacker is
    # the team's truly nearest duck MORE often, not less (65% -> 68%). A
    # role that stops moving is a role a duck can act on.
    switch_s: float = 0.6          # a challenger must be this much quicker…
    hold_s: float = 1.2            # …for this long, without a break, before the role moves…
    give_up_s: float = 2.0         # …unless it is THIS much quicker: the incumbent is out of the play
    # --- where the ball will be --------------------------------------------
    # The board sees the ball only through the claims, so it keeps its own
    # velocity from consecutive fixes BY THE SAME DUCK (a fix is the duck's
    # own position plus its track's bearing and range, so two ducks' fixes
    # differ by centimetres and differencing across them is noise). A duck
    # that walks between two detector frames drags its fix with it — the
    # track coasts, the odometry does not — which is a spurious ball speed
    # of up to the walking speed, so a velocity is only ACTED on above
    # `vel_use`, where a rolling ball (a kick leaves at 1.4 m/s; the
    # `ball_decel` below is the same constant-deceleration stand-in as
    # `ChaseParams.ball_decel`, see its note) is what it must be.
    # MEASURED OFF (`lead_max_s` 0), and this is why the knob is here: over
    # the same 3 seeds x 300 s, aiming at the intercept made the churn
    # WORSE than the straight fix - 18.2 handovers a duck a run against
    # 12.3, a median spell of 3.0 s against 5.8, and 31% of spells under a
    # second against 13% - because the velocity is only as good as the
    # fixes it is differenced from, and a jittering aim point is a
    # jittering cost. The intercept is right for a ball that is genuinely
    # rolling; the board cannot yet tell one from a coasting track.
    ball_decel: float = 0.3
    lead_max_s: float = 0.0
    vel_smooth: float = 0.5
    vel_min_dt: float = 0.15
    vel_max_dt: float = 1.0
    vel_use: float = 0.7
    vel_max: float = 4.0           # a fix that says the ball moved faster than this is a bad fix
    # --- static roles (roadmap Track 4.3) ------------------------------------
    # Who plays what, and the pitch to read a zone against: the half-length in
    # metres and which way this team attacks (+1 = +x). Empty `jobs` is the
    # roster this repo has always had — the quickest duck attacks, wherever
    # the ball is.
    jobs: dict[str, str] = field(default_factory=dict)
    half_x: float = 0.0
    attack_sign: float = 1.0
    # --- the shared ball (roadmap Track 4 s6 C.3) -----------------------------
    # The board's ball was the FRESHEST sighting, whoever sent it and
    # however poor. With `fuse` it is the inverse-variance mean of every
    # live sighting - each weighed by its sender's own sigma (`Track.sigma`,
    # which the chase brain sends with the claim; `sigma_default` for a
    # sender that did not say) grown by the claim's age at `vel_prior` -
    # the SPL team-ball. Off until the odometry probe says it is closer.
    fuse: bool = False
    sigma_default: float = 0.10
    vel_prior: float = 0.06                  # the tracker's calibrated prior (TrackerParams.vel_prior)
    # Only claims this close in time to the FRESHEST one are fused with it.
    # MEASURED (3v3 with roles, 24 seeds, fused v freshest, forked on one
    # package copy): with the full 3 s window the board's ball lagged a
    # moving ball toward stale sightings - own goals 0 -> 6 (p=0.006); with
    # 0.5 s the own goals are gone (1 -> 2, p=0.57) and nothing else moves
    # (crowd, spread, possession flat; progress -0.056, p=0.10). So the
    # fusion, if anyone turns it on, is the 0.5 s one - and nobody should
    # yet: closer in the probe, nothing in the game.
    fuse_window: float = 0.5
    claims: dict[str, Claim] = field(default_factory=dict)
    _attacker: str | None = None
    # --- the game state (roadmap Track 4 s6 B.3) ------------------------------
    # What the World's GameController said at the last restart, stamped by
    # `kickoff_brains`: whether this team kicks off, until when the other
    # side must wait, and where the ball was put. Ours: play. Theirs: every
    # duck of ours is a supporter in its own half until the ball leaves the
    # spot (`waits`). A board nobody stamps never waits.
    _kick_ours: bool = True
    _kick_until: float = -1e9
    _kick_ball: tuple[float, float] | None = None
    _kick_moved: float = 0.1

    def __post_init__(self) -> None:
        self._reset_ball()
        self._pending: str | None = None
        self._pending_t0: float = 0.0

    def _reset_ball(self) -> None:
        self._fixes: dict[str, tuple[float, float, float]] = {}   # duck → (x, y, t) of its last ball fix
        self._fix: tuple[float, float] | None = None              # the freshest fix on the board…
        self._fix_t: float = -1e9                                 # …and when it was taken
        self._vel: tuple[float, float] = (0.0, 0.0)
        self._vel_hits: int = 0

    def reset(self) -> None:
        """Kickoff: nobody has seen the ball, nobody attacks yet."""
        self.claims.clear()
        self._attacker = None
        self._pending = None
        self._reset_ball()
        self._kick_ours, self._kick_until, self._kick_ball = True, -1e9, None

    # -- the game state --------------------------------------------------------
    def kickoff(self, ours: bool, until: float, ball: tuple[float, float] | None,
                moved_m: float = 0.1) -> None:
        """The controller's restart message: whether it is our ball, until
        when the other side waits, and where the ball was put."""
        self._kick_ours, self._kick_until = bool(ours), float(until)
        self._kick_ball = None if ball is None else (float(ball[0]), float(ball[1]))
        self._kick_moved = float(moved_m)

    def waits(self, t: float, ball: tuple[float, float] | None = None) -> bool:
        """Must this team stand off the kickoff? Not ours, inside the
        window, and the ball - a duck's own sighting, else the board's -
        still on the spot; a ball nobody sees is read as not yet in play."""
        if self._kick_ours or t >= self._kick_until or self._kick_ball is None:
            return False
        b = ball if ball is not None else self.ball(t)
        return b is None or math.dist(b, self._kick_ball) < self._kick_moved

    # -- what a duck sends ---------------------------------------------------
    def claim(self, duck_id: str, t: float, dist: float, ball: tuple[float, float] | None,
              pos: tuple[float, float, float] | None = None, ball_sigma: float = math.nan) -> None:
        """One duck's message: how far it puts the ball, where (and how
        surely: `ball_sigma`, roadmap C.3), and where it is. The cost it
        will be judged on is worked out here, from that message alone —
        every duck can do the same arithmetic on every message it
        receives, which is what keeps this a blackboard and not a
        coordinator."""
        if ball is not None:
            self._fold_ball(duck_id, t, ball)
        self.claims[duck_id] = Claim(t, dist, ball, pos, self._cost(t, dist, ball, pos), float(ball_sigma))

    def _fold_ball(self, duck_id: str, t: float, ball: tuple[float, float]) -> None:
        """A sighting: the board's freshest fix, and a velocity sample against
        this duck's own previous fix when the two are usefully apart in time
        (the claims come at the control rate; a 0.02 s baseline is all
        noise)."""
        prev = self._fixes.get(duck_id)
        dt = math.inf if prev is None else t - prev[2]
        if self.vel_min_dt <= dt <= self.vel_max_dt:
            s = ((ball[0] - prev[0]) / dt, (ball[1] - prev[1]) / dt)
            if math.hypot(*s) <= self.vel_max:
                k = self.vel_smooth if self._vel_hits else 1.0
                self._vel = (self._vel[0] + k * (s[0] - self._vel[0]),
                             self._vel[1] + k * (s[1] - self._vel[1]))
                self._vel_hits += 1
            self._fixes[duck_id] = (ball[0], ball[1], t)
        elif dt > self.vel_max_dt:
            if prev is not None:
                self._vel, self._vel_hits = (0.0, 0.0), 0        # too long ago to say
            self._fixes[duck_id] = (ball[0], ball[1], t)
        if t >= self._fix_t:
            self._fix, self._fix_t = (float(ball[0]), float(ball[1])), t

    def _aim(self, fix: tuple[float, float], fix_t: float, t: float, lead: float) -> tuple[float, float]:
        """Where the ball will be `lead` seconds from now: the fix, carried
        along the board's velocity (a constant deceleration, to a stop)."""
        vx, vy = self._vel
        sp = math.hypot(vx, vy)
        if self.lead_max_s <= 0 or self._vel_hits < 2 or sp < self.vel_use:
            return fix
        dt = max(0.0, t - fix_t) + lead
        if self.ball_decel > 0:
            dt = min(dt, sp / self.ball_decel)
            d = sp * dt - 0.5 * self.ball_decel * dt * dt
        else:
            d = sp * dt
        return (fix[0] + vx / sp * d, fix[1] + vy / sp * d)

    def _cost(self, t: float, dist: float, ball: tuple[float, float] | None,
              pos: tuple[float, float, float] | None) -> float:
        """Seconds to get to the ball: the turn this duck must make first,
        then the walk in — to where the ball will be by the time it arrives,
        which is the answer, so the aim point is iterated onto it."""
        seen = ball is not None
        fix, fix_t = (ball, t) if seen else (self._fix, self._fix_t)
        if pos is None or fix is None or (not seen and not math.isinf(dist)):
            # No pose to turn from, nothing on the board to walk to, or a
            # bare range with no place to put it: the straight line is all
            # there is.
            return math.inf if math.isinf(dist) else dist / self.speed
        x, y, yaw = pos
        cost, lead = math.inf, 0.0
        for _ in range(4):
            bx, by = self._aim(fix, fix_t, t, lead)
            bear = abs(_wrap(math.atan2(by - y, bx - x) - yaw))
            turn = max(0.0, bear - self.turn_free)
            cost = (turn / self.turn_rate + (self.cold_s if turn > 0 else 0.0)
                    + max(0.0, math.hypot(bx - x, by - y) - self.reach) / self.speed)
            # Halfway, not all the way: a ball rolling AT the duck arrives
            # sooner the longer you aim ahead of it, and the undamped
            # iteration swings from "it is at my feet" to "it is behind me".
            lead = min(0.5 * (lead + cost), self.lead_max_s)
        if self.side_s > 0.0 and self.half_x > 0.0:
            # ...plus what being on the WRONG SIDE would cost this duck: the
            # angle at the ball between it and the goal its team attacks is
            # pi when it is squarely behind (nothing to pay) and 0 when it is
            # between the ball and that goal (a half-circle to walk).
            gx = self.attack_sign * self.half_x
            ang = abs(_wrap(math.atan2(y - fix[1], x - fix[0])
                            - math.atan2(0.0 - fix[1], gx - fix[0])))
            cost += self.side_s * (math.pi - ang) / math.pi
        return cost + (0.0 if seen else self.blind_s)

    # -- what every duck reads the same way ----------------------------------
    def cost(self, duck_id: str, t: float) -> float:
        """A claim's cost as it stands NOW: what the sender predicted, plus
        what its age is worth. This is the only quantity roles are decided
        on, so a claim nobody has refreshed slides down the list by itself."""
        c = self.claims.get(duck_id)
        if c is None or t - c.t > self.stale_s:
            return math.inf
        return c.cost + self.age_rate * max(0.0, t - c.t)

    def mates(self, duck_id: str, t: float) -> list[tuple[str, tuple[float, float, float]]]:
        """Where my live teammates say they are (the ones that said)."""
        return [(k, c.pos) for k, c in sorted(self.claims.items())
                if k != duck_id and c.pos is not None and t - c.t <= self.stale_s]

    def members(self, t: float) -> list[str]:
        return sorted(k for k, c in self.claims.items() if t - c.t <= self.stale_s)

    def zone_of(self, duck_id: str) -> tuple[float, float] | None:
        """The attack-axis interval this duck may take the ball on, derived
        from the jobs that are actually present. None: no static job, so
        anywhere."""
        job = self.jobs.get(duck_id, "")
        return None if not job else zones_for(self.jobs).get(job)

    def zone_ok(self, duck_id: str, ball: tuple[float, float] | None) -> bool:
        """May this duck go for a ball THERE? A duck with no role always may.
        A duck with one may inside its own zone, measured along the pitch in
        attack coordinates — so a defender does not chase into the far corner
        and a striker does not come back to fetch.

        Intervals are half-open on the high side except the attacking end
        (hi == 1), so neighbouring zones cover [-1, 1] without overlap.
        Midfield a = 0 is the striker's when the roster splits at halfway."""
        z = self.zone_of(duck_id)
        if z is None or ball is None or self.half_x <= 0:
            return True
        a = self.attack_sign * ball[0] / self.half_x
        lo, hi = z
        if hi >= 1.0 - 1e-12:
            return lo <= a <= hi
        return lo <= a < hi

    def candidates(self, t: float) -> list[str]:
        """Who may attack the ball where it is. If nobody's zone covers it —
        it is on a third whose owner has gone missing, or the board has never
        seen it — everybody may, because a ball nobody is allowed to fetch is
        worse than a defender out of position.

        Cover: a live teammate outside the zone may also attack when they are
        `give_up_s` quicker than every zone owner (fallen, facing the wrong
        way, or the ball just skipped past). Static jobs do not change."""
        live = self.members(t)
        field = [k for k in live if self.jobs.get(k) != "keeper"]     # a keeper never leaves its box for a loose ball
        ball = self.ball(t)
        allowed = [k for k in live if self.zone_ok(k, ball)]
        if not allowed:
            return field or live
        owners_best = min(self.cost(k, t) for k in allowed)
        cover = [k for k in field if k not in allowed
                 and self.cost(k, t) < owners_best - self.give_up_s]
        # The cover that HOLDS the role stays a candidate until a zone owner
        # is quicker by the margin that let it in, so the handover back runs
        # through `attacker`'s hysteresis like every other one. Without this
        # the line it entered on (give_up_s quicker) was also the line it
        # left on, and a cost jittering across it moved the role every tick:
        # 300 handovers a run of 3v3, a median spell of 0.09 s, 77% of
        # spells under a second, the ball dead between the two of them
        # (roadmap Track 4 item 11: 46 / 9.7 s / 10% with this). A keeper is
        # never kept: it does not leave its box for a loose ball.
        cur = self._attacker
        if cur in live and cur not in allowed and cur not in cover \
                and self.jobs.get(cur) != "keeper" \
                and self.cost(cur, t) < owners_best + self.give_up_s:
            cover.append(cur)
        return allowed + cover

    def attacker(self, t: float) -> str | None:
        live = self.candidates(t)
        if not live:
            self._attacker = self._pending = None
            return None
        eff = {k: self.cost(k, t) for k in live}
        best = min(live, key=lambda k: (eff[k], k))
        cur = self._attacker
        if cur not in live:
            self._attacker, self._pending = best, None
            return best
        # The challenger is whoever has been pressing (while it is still
        # clearly quicker — otherwise two ducks taking turns at "best" would
        # keep resetting each other's clock and nobody would ever take over).
        cand = best
        if self._pending in live and self._pending != cur and eff[self._pending] < eff[cur] - self.switch_s:
            cand = self._pending
        if cand == cur or eff[cand] >= eff[cur] - self.switch_s:
            self._pending = None
        else:
            if self._pending != cand:
                self._pending, self._pending_t0 = cand, t
            if t - self._pending_t0 >= self.hold_s or eff[cand] < eff[cur] - self.give_up_s:
                self._attacker, self._pending = cand, None
        return self._attacker

    def role(self, duck_id: str, t: float) -> str:
        return "attack" if self.attacker(t) in (duck_id, None) else "support"

    def rank(self, duck_id: str, t: float) -> int:
        """0, 1, … among the supporters, by id: spreads them sideways."""
        att = self.attacker(t)
        sup = [k for k in self.members(t) if k != att]
        return sup.index(duck_id) if duck_id in sup else 0

    def job(self, duck_id: str) -> str | None:
        return self.jobs.get(duck_id)

    def ball(self, t: float) -> tuple[float, float] | None:
        """The board's ball: the freshest teammate sighting - or, with
        `fuse`, the inverse-variance mean of every live sighting, each
        weighed by its sender's sigma grown by its age (roadmap C.3)."""
        seen = [c for c in self.claims.values() if c.ball is not None and t - c.t <= 3 * self.stale_s]
        if not seen:
            return None
        if not self.fuse:
            return max(seen, key=lambda c: c.t).ball
        newest = max(c.t for c in seen)
        seen = [c for c in seen if newest - c.t <= self.fuse_window]
        wx = wy = wsum = 0.0
        for c in seen:
            s = c.ball_sigma if math.isfinite(c.ball_sigma) else self.sigma_default
            w = 1.0 / max(s * s + (self.vel_prior * max(0.0, t - c.t)) ** 2, 1e-6)
            wx += w * c.ball[0]
            wy += w * c.ball[1]
            wsum += w
        return (wx / wsum, wy / wsum)

    def ball_sigma(self, t: float) -> float | None:
        """The fused ball's own sigma (the same weights); None without a ball."""
        seen = [c for c in self.claims.values() if c.ball is not None and t - c.t <= 3 * self.stale_s]
        if not seen:
            return None
        newest = max(c.t for c in seen)
        inv = 0.0
        for c in seen:
            if newest - c.t > self.fuse_window:
                continue
            s = c.ball_sigma if math.isfinite(c.ball_sigma) else self.sigma_default
            inv += 1.0 / max(s * s + (self.vel_prior * max(0.0, t - c.t)) ** 2, 1e-6)
        return math.sqrt(1.0 / inv)

    def publish_kick(self, t: float, origin: tuple[float, float], heading: float, speed: float) -> None:
        """The kicker's known exit line: the ball leaves `origin` along
        `heading` at `speed`. Below `vel_use` this is a coasting track and
        is ignored (the intercept-on-claim measurement). Above it this is
        the board's ball velocity until the next teammate FIX folds a fresh
        sample over it in `_fold_ball` (a fix more than `vel_max_dt` after
        that duck's previous one zeroes it) - one frame in practice, which
        is why the exit line measured neutral in play (roadmap 4b)."""
        if speed < self.vel_use:
            return
        self._vel = (speed * math.cos(heading), speed * math.sin(heading))
        self._vel_hits = 2
        self._fix, self._fix_t = (float(origin[0]), float(origin[1])), t

    def ball_vel(self) -> tuple[float, float]:
        """The board's ball velocity: a published kick line when one is live,
        else the differenced-fixes estimate (usually near zero; `lead_max_s`
        stays 0 so ordinary claims do not intercept on it)."""
        return self._vel

    def led_ball(self, t: float, lead_s: float = 0.4) -> tuple[float, float] | None:
        """Where the ball will be shortly: the kick origin plus the published
        velocity when that speed is kick-like, else the freshest sighting."""
        vx, vy = self._vel
        sp = math.hypot(vx, vy)
        b = self._fix if (sp >= self.vel_use and self._fix is not None) else self.ball(t)
        if b is None:
            return None
        if sp < self.vel_use:
            return b
        dt = lead_s + max(0.0, t - self._fix_t)
        if self.ball_decel > 0:
            dt = min(dt, sp / self.ball_decel)
            d = sp * dt - 0.5 * self.ball_decel * dt * dt
        else:
            d = sp * dt
        return (b[0] + vx / sp * d, b[1] + vy / sp * d)

    def throw_in(self) -> None:
        """The referee moved the ball: drop the board's published VELOCITY and
        the fix it was differenced from, so an invented line stops propagating
        to teammates. Claims, roles, jobs and the kickoff state are untouched —
        a throw-in is not a restart (see `throw_in_brains`)."""
        self._vel, self._vel_hits = (0.0, 0.0), 0
        self._fixes.clear()

    def payload(self, t: float) -> dict:
        def num(v):
            return None if math.isinf(v) else round(v, 2)

        vx, vy = self._vel
        vel = [round(vx, 2), round(vy, 2)] if math.hypot(vx, vy) >= self.vel_use else None
        # THE SHARED BALL, so the viewer can draw what the TEAM believes
        # beside what each duck believes (2026-09-09, asked on /sim: "are the
        # ducks communicating where the ball is?"). This is the board's own
        # answer - the freshest teammate sighting, or the inverse-variance
        # fusion of all of them when `fuse` is on - and it is the value
        # `_support` steers by, so a supporter standing somewhere odd can be
        # read against the belief that put it there.
        shared = self.ball(t)
        return {"name": self.name, "attacker": self.attacker(t),
                **({"ball": [round(shared[0], 2), round(shared[1], 2)]} if shared is not None else {}),
                **({"jobs": dict(self.jobs)} if self.jobs else {}),
                **({"ballVel": vel} if vel is not None else {}),
                "claims": {k: {"dist": num(c.dist), "cost": num(self.cost(k, t)), "age": round(t - c.t, 2),
                               **({"pos": [round(v, 2) for v in c.pos]} if c.pos is not None else {})}
                           for k, c in self.claims.items()}}


def brain_kwargs(duck_spec, world, teams: dict[str, "Team"]) -> dict:
    """What a `chase` brain on a pitch is constructed with: the goal it
    attacks, its team's blackboard (created on first use) and its id.
    Anything else, on any other world: nothing."""
    kind = duck_spec.brain or ""
    if kind != "chase" or world is None or world.goal_width <= 0:
        return {}
    d = world.ducks[duck_spec.id]
    hx, hy = world.scenario.floor[0] / 2 - 0.25, world.scenario.floor[1] / 2 - 0.25   # the boards sit 0.25 m in
    goal = world.goal_for(d)
    team = None
    if duck_spec.team:
        team = teams.setdefault(duck_spec.team, Team(duck_spec.team))
        # The board needs the pitch to read a zone against, and the roster's
        # jobs so that ONE duck is chosen to attack: a zone gate applied per
        # duck instead would let every duck decide it is not allowed and
        # leave the ball to nobody. Filled from the scenario, which is where
        # a role lives (`world/scenario.py`), and idempotent — every teammate
        # writes the same values.
        team.half_x, team.attack_sign = hx, (1.0 if goal is None or goal[0] >= 0 else -1.0)
        from .controllers import ChaseParams  # noqa: PLC0415  (the env spec is one place)
        team.side_s = ChaseParams.from_env().team_side_s
        for x in world.scenario.ducks:
            if x.team == duck_spec.team and x.role:
                team.jobs[x.id] = x.role
    out = {"goal": goal, "team": team, "duck_id": duck_spec.id, "bounds": (hx, hy),
           "goal_w": world.goal_width, "role": duck_spec.role,
           "det_noise": duck_spec.detector}                  # the tracker's uncertainty model (C.1)
    # A roster with teammates plays in a crowd, so it gets the bump sense
    # (`ChaseParams.team_bump_stand_s`) where a lone attacker does not - in
    # 1v1 the rule measured worse on both goals and falls.
    #
    # The base is `from_env`, not `ChaseParams()`. It used to be the bare
    # defaults, and the comment beside it claimed a measurement sweep would
    # not be overridden back - it was. Every knob a battery set through
    # `MICRODUCK_CHASE` was silently discarded on ANY roster with two ducks a
    # side, so a 2v2 or 3v3 A/B would have run the shipped defaults in both
    # arms and reported the seed noise between them as the effect. That is
    # the playbook's rule 0 exactly ("a knob that changes NOTHING is broken,
    # not null"), and it is caught here by a test rather than by an arm that
    # comes back suspiciously flat.
    mates = sum(1 for x in world.scenario.ducks if duck_spec.team and x.team == duck_spec.team)
    from dataclasses import replace  # noqa: PLC0415

    from .controllers import ChaseParams  # noqa: PLC0415
    if mates > 1:
        base = ChaseParams.from_env()
        # …and the roster default applies only where the caller has not
        # already spoken: `bump_stand_s=0` on the command line means that
        # value, not the roster's. Asked by NAME, because the caller's
        # explicit 0 and the shipped default 0 are the same number.
        out["p"] = (base if "bump_stand_s" in ChaseParams.env_names()
                    else replace(base, bump_stand_s=base.team_bump_stand_s))
    # Localise whenever the duck's odometry is DECLARED to drift: the goal-
    # post particle filter (brain/localize.py) is what keeps its goal, its
    # spot and the board's ball in a frame that means the same thing to a
    # teammate (roadmap Track 4 s6 C.2 / item 10). At `ideal` the frames
    # already agree exactly and every soccer number was measured there, so
    # nothing here changes - bit for bit - unless a battery says otherwise
    # through MICRODUCK_CHASE. Same by-name rule as above.
    if duck_spec.odom != "ideal" and "localize" not in ChaseParams.env_names():
        out["p"] = replace(out.get("p") or ChaseParams.from_env(), localize=True)
    # The supporter FIELD (roadmap D.2) ships for the roster it measured a
    # win on and nowhere else: a side WITH A MIDFIELDER (3v3 with roles:
    # possession +2.1 s/min, +15%, pooled 48 seeds p=0.004, shape flat,
    # with the midfielder held behind the ball) - a defender + striker
    # pair lost ball progress under it (0.148 -> 0.061, p=0.002) and the
    # role-less roster measured null. Same by-name rule as above: a knob
    # named on the command line is the caller's.
    if (duck_spec.team and any(x.team == duck_spec.team and x.role == "midfielder" for x in world.scenario.ducks)
            and "support_field" not in ChaseParams.env_names()):
        p = out.get("p") or ChaseParams.from_env()
        out["p"] = replace(p, support_field=True,
                           field_mid_ahead=(p.field_mid_ahead if "field_mid_ahead" in ChaseParams.env_names() else -0.5))
    # The kicks the world will run may not be the shipped ones (a local
    # export under policies/kick, roadmap item 7): their exit angles come
    # from the sidecar beside the file, unless the command line names them.
    exits = getattr(world, "kick_exits", lambda: None)()
    if exits is not None:
        named = ChaseParams.env_names()                             # a named exit wins; the OTHER foot keeps its sidecar
        sidecar = {k: v for k, v in zip(("kick_exit_left", "kick_exit_right"), exits) if k not in named}
        if sidecar:
            out["p"] = replace(out.get("p") or ChaseParams.from_env(), **sidecar)
    # The shared ball (roadmap C.3): the board fuses sightings when the
    # brain's knob says so - every teammate writes the same value.
    if team is not None:
        pf = out.get("p") or ChaseParams.from_env()
        team.fuse, team.fuse_window = bool(pf.fuse_ball), float(pf.fuse_window)
    return out


def throw_in_brains(brains: dict, teams: dict[str, "Team"]) -> None:
    """The referee moved the ball (`World.ball_out_seq` moved): every brain's
    ball belief is stale by construction, so drop it — but NOTHING else.

    A throw-in is not a goal. `kickoff_brains` resets roles, plans, counters
    and the kickoff state, which is right after a goal and wrong here: play
    has not restarted, only the ball has been picked up and put down.

    Measured cost of not doing this (roadmap 12s, 2026-09-09): in the second
    after a throw-in the brain's predicted ball is more than 0.30 m from the
    truth on 49.1% of duck-ticks against 4.6% in a matched control window; the
    tracker reads the parked ball as MOVING at over 0.3 m/s on 33% against
    20%; and the board publishes an invented ball velocity on 15% against 10%.
    The World teleports up to `ball_out_in` 0.45 m and zeroes the ball's qvel,
    but a duck that saw the ball on both sides of that simply differences the
    two positions — and `Team.vel_max` 4.0 does not reject it, because 0.45 m
    over a 0.15-1.0 s baseline is 0.45-3.0 m/s.

    Two things have to go, and only one of them is `disturb`'s job:

    * the AT-REST flag, via `Tracker.disturb(cls)` with no `xy` — every ball
      track, since at a throw-in every belief is stale. (The selective form
      needs BOTH `xy` and a non-zero `radius`; passing `xy` with `radius` 0
      silently marks everything anyway.)
    * the VELOCITY, zeroed here rather than inside `disturb`, because
      `Track.predict` never consults `rest_block` and would coast the invented
      line regardless. It is not folded into `disturb` because `disturb` has
      three live callers on the shipped path — a duck near the ball, a push,
      a kick — and those are NUDGES, where the remembered velocity is
      plausibly still about right. A teleport is categorically different:
      position and velocity are both invalid, and only the throw-in knows it.
    """
    for b in brains.values():
        tr = getattr(b, "tracker", None)
        if tr is None:
            continue
        cls = getattr(getattr(b, "p", None), "target_cls", "ball")
        tr.disturb(cls)
        for t in tr.tracks:
            if t.cls == cls:
                t.vel, t.vel_hits, t.vel_sig = (0.0, 0.0), 0, 0.0
    for tm in teams.values():
        tm.throw_in()


def kickoff_brains(brains: dict, teams: dict[str, "Team"], world=None) -> None:
    """After a goal (World.goal_seq moved): every brain forgets its plan —
    the ball it was lining up on, the spot, the retreat it was in — through
    `kickoff()` where a brain has one (Chase keeps its kick count) and
    `reset()` otherwise; every team's blackboard is wiped. With the `world`,
    the boards also hear the GameController (roadmap B.3): whose kickoff it
    is and how long the other side stands off."""
    for b in brains.values():
        fn = getattr(b, "kickoff", None) or getattr(b, "reset", None)
        if fn is not None:
            fn()
    for tm in teams.values():
        tm.reset()
    kicker = getattr(world, "kickoff_team", None) if world is not None else None
    if kicker is not None:
        for tm in teams.values():
            tm.kickoff(tm.name == kicker, world.kickoff_until + world.kickoff_free_s,
                       world.kickoff_ball, world.kickoff_moved_m)


__all__ = ["Claim", "Team", "ROLE_ZONES", "zones_for", "brain_kwargs", "kickoff_brains",
           "throw_in_brains"]
