"""A team's shared blackboard (soccer, second form): who attacks the ball
and where the ball was last seen, for ducks that cannot tell a teammate
from an opponent by sight.

On the robot this is one small message a second over Wi-Fi between
teammates — id, distance to the ball, the ball's position and my own pose
in my odometry frame — which is exactly what `claim` carries. Nothing
here reads the sim. The poses are what a duck cannot get any other way:
a teammate beside or behind it is invisible to its camera and its ToF,
and most 3v3 falls were a supporter walking or turning into one
(`mates`, and the chase brain's `mate_keepout`).

Roles: the teammate nearest the ball attacks (chase, line up, kick or
push); the others support, standing back toward their own goal, spread
sideways by rank. A claim older than `stale_s` no longer counts, and the
attacker keeps the role until a teammate is clearly nearer (`switch_m`),
so two ducks a centimetre apart in range do not swap every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Claim:
    t: float
    dist: float                              # my distance to the ball (inf: not seen lately)
    ball: tuple[float, float] | None         # where I put the ball (odom frame; the pitch's frame at spawn)
    pos: tuple[float, float, float] | None = None   # where I am (x, y, yaw; the same frame)


@dataclass
class Team:
    name: str
    stale_s: float = 1.0
    switch_m: float = 0.15
    claims: dict[str, Claim] = field(default_factory=dict)
    _attacker: str | None = None

    def reset(self) -> None:
        """Kickoff: nobody has seen the ball, nobody attacks yet."""
        self.claims.clear()
        self._attacker = None

    def claim(self, duck_id: str, t: float, dist: float, ball: tuple[float, float] | None,
              pos: tuple[float, float, float] | None = None) -> None:
        self.claims[duck_id] = Claim(t, dist, ball, pos)

    def mates(self, duck_id: str, t: float) -> list[tuple[str, tuple[float, float, float]]]:
        """Where my live teammates say they are (the ones that said)."""
        return [(k, c.pos) for k, c in sorted(self.claims.items())
                if k != duck_id and c.pos is not None and t - c.t <= self.stale_s]

    def members(self, t: float) -> list[str]:
        return sorted(k for k, c in self.claims.items() if t - c.t <= self.stale_s)

    def attacker(self, t: float) -> str | None:
        live = self.members(t)
        if not live:
            self._attacker = None
            return None
        best = min(live, key=lambda k: (self.claims[k].dist, k))
        cur = self._attacker
        if cur not in live:
            self._attacker = best
        elif self.claims[best].dist < self.claims[cur].dist - self.switch_m:
            self._attacker = best
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
        return {"name": self.name, "attacker": self.attacker(t),
                "claims": {k: {"dist": None if math.isinf(c.dist) else round(c.dist, 2), "age": round(t - c.t, 2),
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
    out = {"goal": world.goal_for(d), "team": team, "duck_id": duck_spec.id, "bounds": (hx, hy)}
    # A roster with teammates plays in a crowd: the bump sense stands the
    # duck for half a second after its feet touch another body (measured,
    # 4 seeds x 300 s: 3v3 falls 5.00 -> 1.75 a run, 2v2 4.00 -> 2.25, at
    # 0.75 and 1.25 goals against 1.50; in 1v1 the same rule measured
    # 1.50 goals and 1.00 falls against 2.38 / 0.38, so a lone attacker
    # keeps the default).
    mates = sum(1 for x in world.scenario.ducks if duck_spec.team and x.team == duck_spec.team)
    if mates > 1:
        from .controllers import ChaseParams
        out["p"] = ChaseParams(bump_stand_s=0.5)
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
