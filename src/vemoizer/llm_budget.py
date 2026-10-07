"""TOTAL wall-clock budget for a single LLM stage (issue #148).

The per-call ``httpx`` timeout in :class:`~vemoizer.llm.LLMClient` is a
per-read bound: a stalled connection that keeps returning partial bytes
resets the per-read clock on every byte, so a single call can hold a run
forever. This module adds the *stage-level* bound the ticket requires:
a wall-clock allowance for the whole repair or notes stage, checked
*between* calls (the loop can only see the clock at iteration boundaries).

On expiry the owning stage fails open per invariant #5: the current
item's result is kept (un-repaired text / no notes), the remaining items
are skipped, and one warning is emitted.

The clock is ``time.monotonic`` by default but is injectable so tests
drive it deterministically (no real sleep, no network).
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["StageBudget"]


class StageBudget:
    """A wall-clock budget for one LLM stage, checked between calls.

    ``elapsed()`` and ``remaining()`` are pure arithmetic on the injected
    clock — never raise. ``exhausted()`` is the single gate the loop
    consults before each call; a budget of ``<= 0`` (or ``None``) is
    *never* exhausted (a degenerate budget disables the cap, which is the
    fail-open default).
    """

    def __init__(
        self,
        total_seconds: float | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._total = total_seconds
        self._clock = clock
        self._start = clock()

    @property
    def total_seconds(self) -> float | None:
        return self._total

    def elapsed(self) -> float:
        """Seconds of wall clock consumed so far (always >= 0).

        A pure clock read — the real elapsed time regardless of whether a
        budget cap is set. Only :meth:`exhausted` and :meth:`remaining`
        are budget-aware.
        """
        return max(0.0, self._clock() - self._start)

    def remaining(self) -> float:
        """Seconds of budget left; ``inf`` when no cap is set."""
        if self._total is None:
            return float("inf")
        return max(0.0, self._total - self.elapsed())

    def exhausted(self) -> bool:
        """True when the stage has used its full allowance (fail the stage).

        A ``None`` or ``<= 0`` budget is never exhausted — those are the
        "no cap" / "fail open to unbounded" cases, which the stage treats
        as no cap rather than an immediate abort.
        """
        if self._total is None or self._total <= 0:
            return False
        return self.elapsed() >= self._total
