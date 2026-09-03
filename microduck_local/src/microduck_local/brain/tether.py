"""A brain tether (roadmap 12.10), modelled honestly: the brain runs somewhere
else — a laptop over Wi-Fi, a cloud model — so its senses reach it half a
round trip late and its intents land on the robot half a round trip later.
Both halves are what a real link does; the sim used to delay only the
intent, which let a brain see fresh senses it would never get, and hid
the one thing a brain CAN do about a tether: read its own sensor ages,
know its latency, and stop that much earlier.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from .runtime import Intent, Senses


class Tether:
    def __init__(self, delay_s: float):
        self.delay = max(0.0, float(delay_s))
        self._senses: deque[tuple[float, Senses]] = deque()
        self._intents: deque[tuple[float, Intent]] = deque()
        self._last_senses: Senses | None = None
        self._last_intent = Intent()

    @property
    def half(self) -> float:
        return self.delay / 2.0

    def senses_in(self, s: Senses) -> Senses:
        """The senses the brain gets at time s.t: the snapshot taken half a
        round trip ago, its frames aged to now, its odometry as stale as
        the link (0 delay: the snapshot itself)."""
        if self.delay <= 0:
            return s
        self._senses.append((s.t + self.half, s))
        while self._senses and self._senses[0][0] <= s.t + 1e-9:
            self._last_senses = self._senses.popleft()[1]
        old = self._last_senses
        if old is None:
            return replace(s, tof=None, tof_age=None, det=None, det_age=None)   # nothing has arrived yet
        return replace(old, t=s.t,
                       tof_age=None if old.tof is None else s.t - old.tof.t,
                       det_age=None if old.det is None else s.t - old.det.t)

    def intent_out(self, intent: Intent, t: float) -> Intent:
        """The intent landing on the robot at t: the one decided half a
        round trip ago (the previous one until it lands)."""
        if self.delay <= 0:
            return intent
        self._intents.append((t + self.half, intent))
        while self._intents and self._intents[0][0] <= t + 1e-9:
            self._last_intent = self._intents.popleft()[1]
        return self._last_intent

    def clear(self) -> None:
        self._senses.clear()
        self._intents.clear()
        self._last_senses = None
        self._last_intent = Intent()


__all__ = ["Tether"]
