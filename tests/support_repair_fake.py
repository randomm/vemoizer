"""The lock-protected fake repair client shared by the concurrency suites.

``unittest.mock.MagicMock`` is deliberately absent. Its call recording mutates
unlocked shared state (``call_count += 1``, ``call_args_list.append``), so
ordering and in-flight claims built on it would be testing the mock's own
races. Every fake here keeps its own state under a lock, drives overlap from
``threading.Event``s the test owns rather than from sleeps, and bounds every
wait — this suite has no ``pytest-timeout`` to catch a hang, so a broken
implementation must fail a test rather than wedge the run.

Imported by ``test_repair_concurrency.py`` (the pool's behaviour) and
``test_llm_tail_repair.py`` (how the tail wires the config through).
"""

from __future__ import annotations

import threading
import time
from typing import Any

#: How long any test-side wait may block before the test declares failure.
#: Generous for a loaded CI box, short enough to fail long before the suite
#: looks hung.
WAIT_S = 10.0


class _ConcurrencyFakeClock:
    """A controllable ``time.monotonic`` stand-in, safe under contention.

    The serial tests' ``_FakeClock`` is a bare float; here several workers
    read it while the caller advances it, so the number lives under a lock.
    """

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds

    def set_to(self, seconds: float) -> None:
        """Jump the clock to an absolute reading.

        Lets a test spend the budget at a moment it controls — while calls are
        parked, say — without depending on how much time the other calls in the
        window happened to burn.
        """
        with self._lock:
            self._now = seconds

    def charge(self, seconds: float) -> float:
        """Advance the clock by *seconds* and return the reading it reached.

        The monotonically increasing kind of advance: each call books its own
        duration exactly once, so the stage's view of elapsed time is a running
        total of the calls that actually ran. Advancing by a delta from every
        worker would let N concurrent calls each charge the same interval, and
        ``budget.remaining()`` handed to the next submission would then be a
        fiction — the budget would look spent on calls still queued behind two
        others.
        """
        with self._lock:
            self._now += seconds
            return self._now


class ThreadRecordingClient:
    """A fake ``LLMClient`` that proves what ran where, and when.

    Records the calling thread and arrival order of every ``complete`` call
    under a lock, keeps a high-water mark of concurrent in-flight calls, and
    releases each call on a caller-supplied gate so overlap is arranged by
    the test instead of inferred from timings.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        before: Any = None,
        replies: dict[str, str] | None = None,
        failures: dict[str, BaseException] | None = None,
        clock: _ConcurrencyFakeClock | None = None,
        burn_s: float = 0.0,
        honour_deadline: bool = False,
        arrived: Any = None,
        hold: Any = None,
        real_time: bool = False,
    ) -> None:
        # *config* is accepted (and ignored) because ``_run_repair`` builds
        # the client as ``llm_client_cls(llm_config)``.
        self.config = config
        #: Called with the paragraph text from inside ``complete``, on the
        #: worker thread, before the candidate is produced.
        self.before = before
        # *clock*/*burn_s* make each call consume a set amount of the fake
        # stage clock, so budget tests need no subclass of their own.
        self._clock = clock
        self._burn_s = burn_s
        #: When true, each call consumes *real* monotonic time as well as
        #: charging the fake stage clock. A budget is a wall-clock allowance, so
        #: a stage that decides "is there time left?" from a real clock must be
        #: measured against one; a fake clock the workers never actually spend
        #: makes every call finish at elapsed 0 and lets a stage that ignores
        #: the gate look identical to one that honours it.
        self._real_time = real_time
        self._replies = replies or {}
        self._failures = failures or {}
        #: When set, ``complete`` honours the ``deadline_s`` the stage hands it
        #: the way :class:`~vemoizer.llm.LLMClient` does: the call ends at its
        #: deadline instead of hanging. The drain contract depends on it — a
        #: fake that ignored the deadline would keep a worker alive past the
        #: stage and make the drain's behaviour a function of the fake.
        self._honour_deadline = honour_deadline
        #: Called with the paragraph text the instant ``complete`` is entered,
        #: before anything can block — the signal that a call is in flight.
        self._arrived = arrived
        #: Called with the paragraph text from inside ``complete`` and expected
        #: to block until the test releases *this* call. Per-paragraph, so one
        #: release cannot let every worker escape at once.
        self._hold = hold
        #: Call index -> how long that call actually occupied its worker,
        #: measured rather than nominalised.
        self.durations: dict[str, float] = {}
        #: Wall-clock entry instant per call, kept apart from
        #: :attr:`entered_at` (a fake-clock test reads that as a stage-clock
        #: reading) so a measured duration is always a real delta.
        self._entered_wall: dict[str, float] = {}
        #: Call index -> the stage-clock reading at which ``complete`` was
        #: *entered*. The submission-side evidence: a paragraph whose call
        #: entered after the budget was gone was started past the stage.
        self.entered_at: dict[str, float] = {}
        #: Call index -> the clock reading at which its call finished.
        self.finished_at: dict[str, float] = {}
        self._lock = threading.Lock()
        self._in_flight = 0
        self._peak = 0
        self.threads: list[str] = []
        self.calls: list[tuple[str, float | None]] = []
        self.closed = 0

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        *,
        deadline_s: float | None = None,
        **kw: Any,
    ) -> str | None:
        # Arrival is reported *before* anything can block, so a caller can
        # observe a call that is genuinely in flight. Had the park come first,
        # "arrived" would mean "this call is over", and a test waiting on
        # arrival while the stage waits on the call would deadlock.
        if self._arrived is not None:
            self._arrived(user)
        if self._clock is not None:
            # Booked under the same lock as the call record, so "entered at"
            # and "was called" are one atomic fact rather than two reads a
            # scheduling blip can separate.
            with self._lock:
                self.entered_at[user] = self._clock()
        else:
            # No fake stage clock in this test: record the real monotonic
            # reading instead, so "when did this call start" stays answerable
            # for a budget that runs on the wall clock.
            with self._lock:
                self.entered_at[user] = time.monotonic()
                self._entered_wall[user] = time.monotonic()
        with self._lock:
            self.threads.append(threading.current_thread().name)
            self.calls.append((user, deadline_s))
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        try:
            if self._hold is not None:
                self._hold(user)
            if self._real_time:
                # Spend the duration for real, unconditionally: a call that is
                # also parked on a gate must still cost its nominal duration,
                # or the budget it is being measured against never actually
                # elapses and a stage that ignores the gate looks identical to
                # one that honours it. Bounded by the deadline the stage handed
                # over, so a call can never outlive the bound it was given.
                threading.Event().wait(
                    min(self._burn_s, deadline_s)
                    if deadline_s is not None
                    else self._burn_s
                )
            if self._honour_deadline and deadline_s is not None:
                # Behave like the real client: the call cannot outlive the
                # bound the stage gave it. The bound is used *as given* — it is
                # the stage's own ``budget.remaining()``, so truncating it to
                # some fixed real-time cap would make every call finish in
                # about nil time and turn a budget test into a coin flip. A
                # test that passes a huge deadline is asking for a long call;
                # the tests that do this pass a small one.
                threading.Event().wait(deadline_s)
            if self.before is not None:
                self.before(user)
            failure = self._failures.get(user)
            if failure is not None:
                raise failure
            return self._replies.get(user, user + "x")
        finally:
            # The call's cost is booked as it *leaves*, not as it arrives: a
            # queue costs a paragraph its duration once, when it actually runs.
            # The ``finally`` keeps the accounting honest for a call that raises.
            entered = self._entered_wall.get(user)
            if entered is not None:
                with self._lock:
                    self.durations[user] = time.monotonic() - entered
            if self._clock is not None:
                at = self._clock.charge(self._burn_s)
                with self._lock:
                    self.finished_at[user] = at
            with self._lock:
                self._in_flight -= 1

    def close(self) -> None:
        self.closed += 1

    def elapsed_work_s(self) -> float:
        """Total time the calls actually occupied, summed over the pool.

        The denominator the budget bound needs: what a paragraph really costs is
        how long its call ran, not what the fixture asked it to pretend. Each
        call books its own duration on leaving, so this is a sum of real
        per-call spans rather than a clock delta that would double-count calls
        running side by side.
        """
        with self._lock:
            return sum(self.durations.values())

    def mean_call_s(self) -> float:
        """Average wall-clock duration of the calls that ran.

        The denominator the budget bound needs: what a paragraph really costs
        is how long its call ran, measured on the same wall clock the budget
        reads, not the figure the fixture nominalised. Grading the stage
        against a number the fixture invented would make a slightly slow call
        look like a gate bug.
        """
        with self._lock:
            n = len(self.durations)
            return (sum(self.durations.values()) / n) if n else 0.0

    @property
    def peak_in_flight(self) -> int:
        with self._lock:
            return self._peak

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)

    def order_of_arrival(self) -> list[int]:
        """Paragraph indices in the order their calls arrived."""
        with self._lock:
            return [int(user.rsplit(" ", 1)[1]) for user, _ in self.calls]


def _para(text: str, **extra: Any) -> dict[str, Any]:
    """One paragraph with the shape the stage expects."""
    return {"start": 0.0, "end": 5.0, "text": text, **extra}


def _paras(n: int) -> list[dict[str, Any]]:
    """*n* synthetic paragraphs. The text is a sentinel, never a transcript."""
    return [_para(f"lause {i}") for i in range(n)]


def _index(text: str) -> int:
    """The paragraph number a fake reply was made for (``"lause 7"`` -> 7)."""
    return int(text.rsplit(" ", 1)[1])


def _wait_until(predicate: Any) -> bool:
    """Poll *predicate* until it holds, bounded by :data:`WAIT_S`.

    For waiting on worker progress that no event stands in for. Bounded
    because this suite has no ``pytest-timeout`` to catch a hang.
    """
    deadline = time.monotonic() + WAIT_S
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        threading.Event().wait(0.005)
    return True
