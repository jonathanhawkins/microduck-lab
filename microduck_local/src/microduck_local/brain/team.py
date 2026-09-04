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
    # `vel_use`, where a rolling ball (a kick leaves at 1.4 m/s and slows
    # at 0.04 m/s^2) is what it must be.
    # MEASURED OFF (`lead_max_s` 0), and this is why the knob is here: over
    # the same 3 seeds x 300 s, aiming at the intercept made the churn
    # WORSE than the straight fix - 18.2 handovers a duck a run against
    # 12.3, a median spell of 3.0 s against 5.8, and 31% of spells under a
    # second against 13% - because the velocity is only as good as the
    # fixes it is differenced from, and a jittering aim point is a
    # jittering cost. The intercept is right for a ball that is genuinely
    # rolling; the board cannot yet tell one from a coasting track.
    ball_decel: float = 0.04
    lead_max_s: float = 0.0
    vel_smooth: float = 0.5
    vel_min_dt: float = 0.15
    vel_max_dt: float = 1.0
    vel_use: float = 0.7
    vel_max: float = 4.0           # a fix that says the ball moved faster than this is a bad fix
    claims: dict[str, Claim] = field(default_factory=dict)
    _attacker: str | None = None

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

    # -- what a duck sends ---------------------------------------------------
    def claim(self, duck_id: str, t: float, dist: float, ball: tuple[float, float] | None,
              pos: tuple[float, float, float] | None = None) -> None:
        """One duck's message: how far it puts the ball, where, and where it
        is. The cost it will be judged on is worked out here, from that
        message alone — every duck can do the same arithmetic on every
        message it receives, which is what keeps this a blackboard and not
        a coordinator."""
        if ball is not None:
            self._fold_ball(duck_id, t, ball)
        self.claims[duck_id] = Claim(t, dist, ball, pos, self._cost(t, dist, ball, pos))

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

    def attacker(self, t: float) -> str | None:
        live = self.members(t)
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

    def ball(self, t: float) -> tuple[float, float] | None:
        """The freshest teammate sighting of the ball."""
        seen = [c for c in self.claims.values() if c.ball is not None and t - c.t <= 3 * self.stale_s]
        if not seen:
            return None
        return max(seen, key=lambda c: c.t).ball

    def payload(self, t: float) -> dict:
        def num(v):
            return None if math.isinf(v) else round(v, 2)

        return {"name": self.name, "attacker": self.attacker(t),
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
    team = None
    if duck_spec.team:
        team = teams.setdefault(duck_spec.team, Team(duck_spec.team))
    hx, hy = world.scenario.floor[0] / 2 - 0.25, world.scenario.floor[1] / 2 - 0.25   # the boards sit 0.25 m in
    out = {"goal": world.goal_for(d), "team": team, "duck_id": duck_spec.id, "bounds": (hx, hy),
           "goal_w": world.goal_width}
    # A roster with teammates plays in a crowd, so it gets the bump sense
    # (`ChaseParams.team_bump_stand_s`) where a lone attacker does not - in
    # 1v1 the rule measured worse on both goals and falls. Taken off the
    # LIVE defaults with `replace`, so a caller that overrides ChaseParams
    # (a measurement sweep) is not silently overridden back.
    mates = sum(1 for x in world.scenario.ducks if duck_spec.team and x.team == duck_spec.team)
    if mates > 1:
        from dataclasses import replace

        from .controllers import ChaseParams
        base = ChaseParams()
        out["p"] = replace(base, bump_stand_s=base.team_bump_stand_s)
    return out


def kickoff_brains(brains: dict, teams: dict[str, "Team"]) -> None:
    """After a goal (World.goal_seq moved): every brain forgets its plan —
    the ball it was lining up on, the spot, the retreat it was in — through
    `kickoff()` where a brain has one (Chase keeps its kick count) and
    `reset()` otherwise; every team's blackboard is wiped."""
    for b in brains.values():
        fn = getattr(b, "kickoff", None) or getattr(b, "reset", None)
        if fn is not None:
            fn()
    for tm in teams.values():
        tm.reset()


__all__ = ["Claim", "Team", "brain_kwargs", "kickoff_brains"]
