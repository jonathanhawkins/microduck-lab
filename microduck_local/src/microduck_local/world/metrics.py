"""Ball progress and possession: the two continuous soccer metrics.

They live here, not in `eval_pitch`, because they are properties of the
world the /sim page streams as much as of the benchmark - the page shows
them live, and a number on the page has to be the same number the battery
reports. See the class docstring for why these two and not the obvious
alternatives, and `eval_pitch`'s module docstring for what each costs in
seeds (goals 146, ballAdvance 43, possession 9, for a 25% shift).
"""

from __future__ import annotations

import math

from .. import contract as C
from .arena import World

# --- the continuous metrics ---------------------------------------------------
# Goals are a terrible ruler for this benchmark: ~2.5 a run against ~13 kicks,
# so the same brain measured twice gave 3 falls and 6, and four brain variants
# landed inside each other's noise on goals in one day. What the ducks actually
# do thousands of times a run is *reach the ball* and *move it*; these two
# accumulate that at the control tick.
#
# POSSESSION_R — a duck is "on the ball" inside this radius (trunk centre to
# ball centre, in the plane). The kick sweet spot is 6-10 cm ahead and 4-8 cm
# to the side of the trunk (`ChaseParams.kick_ahead/kick_side`), i.e. ~0.12 m
# out; 0.25 m is about twice that, so a duck lined up on the ball or one step
# short of it counts, while it stays well inside `duck_keepout` (0.40 m) — the
# range at which the chase brain starts avoiding the other duck — so the two
# contesting ducks are rarely both on the ball. Only the NEAREST duck can hold
# possession anyway, so the radius sets how much loitering counts, not who wins
# a contest.
POSSESSION_R = 0.25
# The same clock at the keepout radius, recorded only as a robustness check:
# it says whether a conclusion turns on the exact choice of POSSESSION_R.
#
# It did, and not in the direction that was expected: over 16 seeds an arm the
# WIDER radius came out both quieter (CV 0.11 vs 0.16) and far more sensitive
# to the lens contrast (p=0.0001 vs p=0.06). The primary stays 0.25 m anyway.
# 0.25 m was fixed before the battery and 0.40 m was not, and a radius chosen
# because it won on the one battery that could confirm it is exactly the kind
# of result this benchmark has already had to withdraw twice; 0.40 m also sits
# at the chase brain's own `duck_keepout` and is occupied 45 s in every 60 in
# 1v1, which is close enough to saturation to be measuring the pitch rather
# than the brain. Promote it when a battery run FOR that question says so.
POSSESSION_WIDE_R = 0.40
# After the last tick a team was on the ball it keeps CREDIT for the ball's
# motion for this long. A kick is exactly the case that matters: the ball
# leaves at speed and travels most of its distance with no duck within
# POSSESSION_R, so progress credited only while inside the radius would score
# dribbling and ignore the shot — the one thing the benchmark is about. 2.0 s
# is deliberately shorter than the World's own kick→goal attribution window
# (KICK_GOAL_S = 4.0 s): long enough for a kicked ball to run out, short
# enough that a ball rattling round the boards minutes later is nobody's.
CARRY_S = 2.0

# The per-team metric keys a row carries. A row written before these existed
# has none of them; `load_done` fills them with None rather than 0.0, and the
# summary reports how many seeds it could use (see `_mean_field`).
METRIC_FIELDS = ("ballProgress", "ballAdvance", "possession", "possessionWide")


class PitchMetrics:
    """Ball progress and possession, accumulated at the 50 Hz control tick.

    `tick()` is called once per control step, AFTER `World.step()`, and costs
    one distance per duck plus a few floats — nothing measurable against an
    `mj_step` of a 2-duck pitch.

    **Ball progress** is the ball's displacement toward the goal a team
    attacks (the pitch's long axis is x; `World.goal_for` gives the sign),
    summed over the ticks that team is in control. Why not the other two
    candidates:

    - *Signed per-step displacement over the whole run* telescopes. Sum every
      Δx and all that survives is the ball's net start-to-end position, and
      since every goal recentres the ball, that quantity is ~1.25 m × (goals
      for − goals against) plus a remainder — goals in disguise, with goals'
      variance. It cannot carry more signal than the thing it collapses to.
    - *Positive-only displacement, uncredited* measures how much the ball
      moved at all, which both teams get paid for equally; it is a ball-motion
      counter, not a per-side metric.

    Attributing each tick's displacement to the team in control keeps the
    telescoping LOCAL — inside one possession, where net start-to-end is
    exactly what "this team took the ball 40 cm toward the goal" means — and
    a run contains hundreds of possessions instead of two or three goals.
    `progress` is the signed sum (a team that shoves the ball back toward its
    own goal is charged for it); `advance` sums only the forward part, and is
    recorded alongside so the two can be compared on the same battery. They
    were, over 32 runs, and `advance` won on both counts: the signed sum has a
    CV of 4.2 as a run total, because in self-play the two sides' signs oppose
    and the run total is a difference of two similar numbers, while `advance`
    has a CV of 0.40 and is the only metric here whose correlation with goals
    is resolved away from zero (r 0.50, 95% CI [+0.19, +0.72]). Keep both:
    the signed one is the meaningful per-TEAM reading in an asymmetric
    matchup, where the cancellation that kills its run total is the point.

    **Possession** is the strict clock: seconds where the nearest duck to the
    ball is within POSSESSION_R and belongs to that team. No carry — the task
    it measures is "was one of mine on the ball", which is either true or not.
    It is by far the quietest number the benchmark produces (CV 0.16, against
    goals' 0.76) and therefore the cheapest way to tell that two variants
    differ AT ALL; it is not a measure of playing well, and the battery could
    not resolve its correlation with goals away from zero. Screen with it,
    judge with `advance`.
    """

    def __init__(self, world: World, team_of: dict[str, str]):
        self.w = world
        self.team_of = dict(team_of)
        self.teams = sorted(set(team_of.values()))
        # Which way is "forward" for each team, from the goal it attacks.
        self.sign: dict[str, float] = {}
        for did, tm in team_of.items():
            g = world.goal_for(world.ducks[did])
            s = 1.0 if g is None or g[0] >= 0 else -1.0
            if self.sign.setdefault(tm, s) != s:
                raise ValueError(f"team {tm!r} has ducks attacking opposite goals")
        self.progress = {t: 0.0 for t in self.teams}
        self.advance = {t: 0.0 for t in self.teams}
        self.possession = {t: 0.0 for t in self.teams}
        self.possession_wide = {t: 0.0 for t in self.teams}
        self._prev = world.ball_xy()
        self._holder: str | None = None       # team credited with the ball right now
        self._holder_t = -1e9                 # when it was last strictly on the ball
        self._goal_seq = world.goal_seq

    def nearest(self) -> tuple[str | None, float]:
        """(duck id, planar distance) of the duck closest to the ball."""
        ball = self.w.ball_xy()
        if ball is None:
            return None, math.inf
        best, best_d = None, math.inf
        for did, d in self.w.ducks.items():
            p = d.trunk_pos(self.w.data)
            r = math.hypot(float(p[0]) - ball[0], float(p[1]) - ball[1])
            if r < best_d:
                best, best_d = did, r
        return best, best_d

    def tick(self) -> None:
        w = self.w
        ball = w.ball_xy()
        if ball is None:
            return
        # The displacement first, and it is credited to whoever held the ball
        # THROUGH the step that just ended, not to whoever is standing over it
        # now: the tick a duck arrives to win the ball back is a tick of motion
        # its opponent caused, and charging a couple of centimetres to the
        # wrong side at every changeover — hundreds a run — is a real bias, not
        # a rounding one.
        if w.goal_seq != self._goal_seq:
            # A goal: the World teleported the ball back to the centre spot.
            # That jump is not anybody's progress (counted, it would cancel
            # almost exactly the goal it followed), and the next possession
            # starts clean.
            self._goal_seq = w.goal_seq
            self._prev, self._holder = ball, None
        elif self._prev is not None and self._holder is not None and w.t - self._holder_t <= CARRY_S:
            dx = self.sign[self._holder] * (ball[0] - self._prev[0])
            self.progress[self._holder] += dx
            self.advance[self._holder] += max(0.0, dx)
        self._prev = ball
        # Then who is on the ball at the end of the step, which is who the NEXT
        # step's motion belongs to.
        who, r = self.nearest()
        if who is not None and r <= POSSESSION_R:
            tm = self.team_of[who]
            self.possession[tm] += C.CTRL_DT
            self._holder, self._holder_t = tm, w.t          # control (and credit) passes
        if who is not None and r <= POSSESSION_WIDE_R:
            self.possession_wide[self.team_of[who]] += C.CTRL_DT

    def row(self) -> dict:
        """The four per-team metrics, as RATES per minute of play, so a row is
        comparable across `--seconds` (the totals divide out; `simSeconds` is
        in the row if anyone wants them back)."""
        per_min = 60.0 / max(self.w.t, 1e-9)
        return {"ballProgress": {t: round(v * per_min, 3) for t, v in self.progress.items()},
                "ballAdvance": {t: round(v * per_min, 3) for t, v in self.advance.items()},
                "possession": {t: round(v * per_min, 3) for t, v in self.possession.items()},
                "possessionWide": {t: round(v * per_min, 3) for t, v in self.possession_wide.items()}}


SPIN_FIELDS = ("spinFrac", "steerFrac", "spinYaw", "spinRate")


class SpinMetrics:
    """How much of a run is the robot rotating on the spot?

    A PER-TICK tally, which is the point: it resolves where goals and falls
    cannot. Measured with it, a 1v1 run is 47% in-place turning at the
    walker's 0.655 rad/s ceiling, and `ANG_VEL_Z_RANGE`'s +-1.0 is already
    asking for all of that - the largest measured lever in this repo
    (roadmap 3.7, docs/turn-rate-experiment.md).

    `spinYaw` integrates the yaw the body ACTUALLY swept while a turn was
    commanded, so a faster walker shows up as the same yaw demand in fewer
    ticks rather than as a different number. `spinRate` is the rate it
    managed, which is how you check a new walker is really delivering.

    Lives here rather than in either benchmark because `eval_pitch` and
    `eval_striker` run the same loop and their rows must stay identical -
    a test pins that, and it caught this class being added to only one.
    """

    def __init__(self, world):
        self.w = world
        self.spin = self.steer = self.ticks = 0
        self.yaw = 0.0
        self._prev = {d.id: world.odom(d)[2] for d in world.ducks.values()}

    def tick(self, duck, twist) -> None:
        """Once per DUCK per control step, with the twist the brain commanded."""
        from ..brain.gait import TURN_KICK
        vx, _, wz = twist
        self.ticks += 1
        y = self.w.odom(duck)[2]
        if wz != 0.0 and vx <= TURN_KICK:
            self.spin += 1
            self.yaw += abs(math.atan2(math.sin(y - self._prev[duck.id]),
                                       math.cos(y - self._prev[duck.id])))
        elif wz != 0.0:
            self.steer += 1
        self._prev[duck.id] = y

    def row(self) -> dict:
        n = max(self.ticks, 1)
        return {"spinFrac": round(self.spin / n, 4),
                "steerFrac": round(self.steer / n, 4),
                "spinYaw": round(self.yaw, 1),
                "spinRate": round(self.yaw / max(self.spin * C.CTRL_DT, 1e-9), 3)}
