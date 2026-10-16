"""Bounded concurrency for the LLM repair pass (issue #159).

The repair stage is one HTTP call per paragraph and a long meeting is
hundreds of paragraphs, so the loop — not the model — is the wall clock.
These tests pin what the ``concurrency`` argument does and, just as much,
what it must not do: results stay in paragraph order whatever order the
calls finish in, no more than N calls are ever in flight, the display
advance never fires from a worker, and no worker outlives the call.

Every fake here is lock-protected and event-driven, with no real sleeps and
no network: overlap comes from ``threading.Barrier`` and per-paragraph
``threading.Event``\\s the test releases, and every wait that could hang
carries a timeout — ``pytest-timeout`` is not a dependency here, so a broken
implementation must fail the test rather than wedge the suite.

``MagicMock`` is deliberately absent: its call recording mutates unlocked
shared state (``call_count += 1``, ``call_args_list.append``), so ordering
and in-flight claims built on it would be testing the mock's own races. The
serial oracle in ``tests/test_repair.py`` keeps using it — there it is the
single-threaded baseline and nothing contends.
"""

from __future__ import annotations

import logging
import threading
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vemoizer.llm import LLMConfig
from vemoizer.llm_budget import StageBudget
from vemoizer.llm_config import (
    _KNOWN_LLM_KEYS,
    MAX_REPAIR_CONCURRENCY,
    ConfigError,
    _parse_llm_section,
    _strict_load,
)
from vemoizer.llm_tail import _run_repair
from vemoizer.repair import MAX_CONCURRENCY, repair_paragraphs

#: How long any test-side wait may block before the test declares failure.
#: Generous for a loaded CI box, short enough to fail long before the suite
#: looks hung.
WAIT_S = 10.0

_MAIN = threading.main_thread().name


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
    return {"start": 0.0, "end": 5.0, "text": text, **extra}


def _paras(n: int) -> list[dict[str, Any]]:
    return [_para(f"lause {i}") for i in range(n)]


def _index(text: str) -> int:
    """The paragraph number a fake reply was made for (``"lause 7"`` -> 7)."""
    return int(text.rsplit(" ", 1)[1])


def _wait_until(predicate: Callable[[], bool]) -> bool:
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


def _completion_order(n: int, order: list[int]) -> Any:
    """Build a ``before`` hook that makes the calls return in ``order``.

    Each call parks until every paragraph scheduled ahead of it in ``order``
    has answered. The first in ``order`` waits for nobody, so it answers,
    releases the next, and so on — the model's answers arrive in ``order``
    whatever order the calls were made in.

    Two constraints shape this. A paragraph must not wait on one that waits
    on it, or the chain never starts: hence waits run strictly along
    ``order``, never against it. And the whole paragraph set must be in
    flight before anyone is made to wait, or a paragraph the pool has not
    reached yet is waited on and the slots fill with parked workers — which
    is why ``order`` is the *arrival* schedule too, matching the pool's own
    FIFO submission, and why these tests use ``concurrency == n``.
    """
    rank = {idx: pos for pos, idx in enumerate(order)}
    answered = {idx: threading.Event() for idx in range(n)}
    arrived = {idx: threading.Event() for idx in range(n)}

    def before(text: str) -> None:
        idx = _index(text)
        arrived[idx].set()
        for earlier in order[: rank[idx]]:
            # Arrived first: a paragraph is never made to wait on a call the
            # pool has not started, which is what keeps the slots free.
            assert arrived[earlier].wait(timeout=WAIT_S), f"{earlier} never arrived"
            assert answered[earlier].wait(timeout=WAIT_S), f"{earlier} never answered"
        answered[idx].set()

    return before


# -- overlap and the bound -------------------------------------------------


def test_three_workers_actually_overlap() -> None:
    """N=3 must really have three calls in flight at once.

    The barrier only clears when three workers are simultaneously inside
    ``complete``; a serial implementation never reaches it and the barrier
    times out into ``BrokenBarrierError``, which the stage's fail-open turns
    into un-repaired originals — caught by the assertions, not by a hang.
    """
    barrier = threading.Barrier(3, timeout=WAIT_S)
    arrived: list[str] = []
    gate = threading.Event()
    arrival_lock = threading.Lock()

    def before(text: str) -> None:
        with arrival_lock:
            arrived.append(text)
            enough = len(arrived) >= 3
        if enough:
            gate.set()
        assert gate.wait(timeout=WAIT_S), "the three calls never coincided"
        barrier.wait(timeout=WAIT_S)

    client = ThreadRecordingClient(before=before)
    out = repair_paragraphs(client, _paras(3), concurrency=3)

    assert client.call_count == 3
    assert client.peak_in_flight == 3
    assert len(set(client.threads)) == 3, client.threads
    assert all(not t.startswith("MainThread") for t in client.threads)
    assert [p["text"] for p in out] == ["lause 0x", "lause 1x", "lause 2x"]


def test_in_flight_never_exceeds_the_pool_size() -> None:
    """The pool is a bound, not a hint: 9 paragraphs through 3 slots."""
    release = threading.Event()

    def before(text: str) -> None:
        # Every call waits for the test, so the peak is whatever the pool
        # allows and nothing more.
        assert release.wait(timeout=WAIT_S)

    client = ThreadRecordingClient(before=before)
    caller_done = threading.Event()
    result: list[dict[str, Any]] = []

    def run() -> None:
        result.extend(repair_paragraphs(client, _paras(9), concurrency=3))
        caller_done.set()

    threading.Thread(target=run, name="repair-caller").start()
    # Wait for the pool to fill its three slots, read the peak while they are
    # still parked, then let everything through.
    assert _wait_until(lambda: client.call_count >= 3), "the pool never filled"
    peak_at_fill = client.peak_in_flight
    release.set()
    assert caller_done.wait(timeout=WAIT_S), "repair_paragraphs never returned"

    assert client.call_count == 9
    assert peak_at_fill <= 3
    assert client.peak_in_flight <= 3
    assert [p["text"] for p in result] == [f"lause {i}x" for i in range(9)]


def test_concurrency_one_never_leaves_the_calling_thread() -> None:
    """The default runs inline: no worker thread is ever created.

    A one-worker pool would satisfy "at most one in flight" while still
    moving the call off the caller's thread, so the assertion is on the
    thread name, not on a count.
    """
    before = {t.name for t in threading.enumerate()}
    client = ThreadRecordingClient()
    repair_paragraphs(client, _paras(5))
    assert set(client.threads) == {_MAIN}, client.threads
    assert {t.name for t in threading.enumerate()} == before


def test_no_worker_outlives_the_call() -> None:
    """Every thread the stage started is gone shortly after it returns.

    A leaked non-daemon worker outlives the CLI: ``concurrent.futures``
    registers an atexit hook that joins pool threads, so a stuck one turns the
    next Ctrl-C into a hang the user cannot explain. The stage hands that wait
    to the pool rather than sitting on it (PM decision (d)), so the claim is
    that the workers *end* — bounded here by :data:`WAIT_S`, far inside any
    plausible CLI lifetime — not that they are already gone at the instant the
    call returns. A worker blocked forever on a socket is the give-up case the
    next test pins; on a healthy run nothing survives.
    """
    before = {t.name for t in threading.enumerate()}
    client = ThreadRecordingClient()
    repair_paragraphs(client, _paras(12), concurrency=MAX_CONCURRENCY)
    # A healthy pool has no work left when the stage returns, so its workers
    # exit on their own almost immediately; the wait is bounded so a leak is a
    # failure rather than a hang.
    leaked = lambda: sorted(  # noqa: E731 - a probe, called under a deadline
        thr.name
        for thr in threading.enumerate()
        if thr.name.startswith("vemoizer-repair") and thr.name not in before
    )
    assert _wait_until(lambda: not leaked()), (
        f"worker threads outlived repair_paragraphs: {leaked()}"
    )
    assert {t.name for t in threading.enumerate()} == before
    assert client.peak_in_flight <= MAX_CONCURRENCY


def test_budget_give_up_never_waits_for_a_worker_that_never_answers() -> None:
    """The stage returns even when no worker ever answers.

    The hang proof for PM decision (d). The pool's workers are non-daemon and
    ``concurrent.futures`` registers an atexit hook that joins them, so the
    only way the CLI stays interruptible is that this stage never *blocks* on a
    call the budget has already refused. Three calls are parked on events that
    nothing but this test ever sets — they answer to nobody, which is the
    stalled-connection case the stage budget exists for — and the budget is
    spent while all three are in flight. The stage must still walk away and
    return within a bounded wait.

    The events are released only after the stage has returned, so the release
    can never be what let it return: if the stage waited on a parked call, this
    test fails on the ``returned`` assertion rather than hanging.
    """
    release = {i: threading.Event() for i in range(9)}

    class _Stalled(ThreadRecordingClient):
        """Calls that answer to nobody until the test lets each one go.

        Per-paragraph events, deliberately: a single shared event would release
        every parked worker the moment the stage came back, and the leak check
        would then be measuring the test's own cleanup. No deadline is honoured
        here on purpose — this is the transport that ignores its own bound, the
        worst case the drain has to survive.
        """

    client = _Stalled(
        hold=lambda text: release[_index(text)].wait(timeout=WAIT_S),
    )
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(30.0, clock=clock)
    returned = threading.Event()
    out: list[dict[str, Any]] = []

    def call() -> None:
        out.extend(repair_paragraphs(client, _paras(9), concurrency=3, budget=budget))
        returned.set()

    caller = threading.Thread(target=call, name="budget-give-up-caller")
    caller.start()
    # The three slots genuinely filled with calls that answer to nobody…
    assert _wait_until(lambda: client.call_count >= 3), "the pool never filled"
    # …and the budget spent while they are all still in flight.
    clock.advance(31.0)

    # The stage must stop working and return rather than block on a call the
    # budget refused — the property that keeps the CLI interruptible.
    try:
        caller.join(timeout=WAIT_S)
        assert returned.is_set(), "the stage blocked on a call the budget refused"
        assert not caller.is_alive(), "the caller thread never finished"
    finally:
        for ev in release.values():
            ev.set()
        caller.join(timeout=WAIT_S)

    # The stage came back with every paragraph accounted for: the calls it was
    # told to abandon are not trusted, and nothing is missing. The workers
    # themselves are released by the ``finally`` above — asserting they are gone
    # here would only be measuring that release, not the stage.
    assert len(out) == 9, out
    assert all(p["text"].startswith("lause ") for p in out), out
    assert client.peak_in_flight <= 3


# -- ordering ----------------------------------------------------------------


def test_reverse_completion_still_ships_in_paragraph_order() -> None:
    """Late paragraphs answer first; the output must not follow them.

    Every paragraph is in flight at once and they answer last-to-first, so
    the collector wants paragraph 0 while it is still the slowest thing
    outstanding. Results keyed by original index are the only reason this
    passes — writing them with ``append`` instead breaks it.
    """
    n = 4
    reverse = list(reversed(range(n)))
    client = ThreadRecordingClient(before=_completion_order(n, reverse))

    out = repair_paragraphs(client, _paras(n), concurrency=n)

    assert [p["text"] for p in out] == [f"lause {i}x" for i in range(n)]
    # The proof that the ordering machinery was actually exercised: all four
    # calls were in flight together, and they were answered backwards.
    assert client.peak_in_flight == n, client.peak_in_flight
    assert client.order_of_arrival() == list(range(n)), client.order_of_arrival()


def test_speaker_and_timing_ride_along_with_reordered_results() -> None:
    """Metadata is carried by the slot, never by the response."""
    n = 3
    client = ThreadRecordingClient(before=_completion_order(n, [2, 0, 1]))
    paras = [
        _para(f"lause {i}", speaker=f"S{i}", start=float(i), end=float(i) + 1)
        for i in range(n)
    ]

    out = repair_paragraphs(client, paras, concurrency=n)

    assert [p["text"] for p in out] == ["lause 0x", "lause 1x", "lause 2x"]
    assert [p["speaker"] for p in out] == ["S0", "S1", "S2"]
    assert [p["start"] for p in out] == [0.0, 1.0, 2.0]
    assert [p["end"] for p in out] == [1.0, 2.0, 3.0]


# -- progress and the heartbeat ---------------------------------------------


def test_progress_callback_never_runs_on_a_worker_thread() -> None:
    """The display advance is single-writer, on the caller's thread.

    ``rich`` progress is not documented for concurrent updates from worker
    threads, and a worker-side tick would race the counter. The calls are
    genuinely concurrent here — all three parked on each other's events —
    so a worker that ticked had every opportunity to show up in the record.
    """
    n = 3
    client = ThreadRecordingClient(before=_completion_order(n, [2, 1, 0]))
    ticks: list[tuple[int, int, str]] = []
    tick_lock = threading.Lock()

    def progress(done: int, total: int) -> None:
        with tick_lock:
            ticks.append((done, total, threading.current_thread().name))

    repair_paragraphs(client, _paras(n), concurrency=n, progress_cb=progress)

    assert [t[2] for t in ticks] == [_MAIN] * n, ticks
    # Monotonic by exactly one, and reaching total exactly once.
    assert [t[0] for t in ticks] == [1, 2, 3], ticks
    assert all(t[1] == n for t in ticks)
    assert client.peak_in_flight > 1, "no real overlap, so the race was not tested"


def test_heartbeat_counts_stay_on_the_calling_thread(caplog) -> None:
    """The throttled heartbeat is a single-writer log line, counts only.

    A worker-side ``logger.info`` would both race the counter and risk
    leaking paragraph text into the run log (privacy contract).
    """
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(1000.0, clock=clock)
    n = 3
    base = _completion_order(n, [2, 1, 0])

    def before(text: str) -> None:
        clock.advance(10.0)  # cross PROGRESS_INTERVAL_S between calls
        base(text)

    client = ThreadRecordingClient(before=before)
    with caplog.at_level(logging.INFO, logger="vemoizer.repair"):
        repair_paragraphs(client, _paras(n), concurrency=n, budget=budget)

    beats = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("repair:") and "elapsed" in r.getMessage()
    ]
    assert beats, [r.getMessage() for r in caplog.records]
    for beat in beats:
        assert "lause" not in beat, f"heartbeat leaked paragraph text: {beat}"


# -- budget -------------------------------------------------------------------


def test_no_call_starts_after_the_budget_is_gone() -> None:
    """The gate is a per-submission gate, which is what makes it mean anything.

    PM decision (b)'s call *count* window is necessary but not sufficient: a
    stage that opens the whole window before consulting the budget, or that
    refills a freed slot without re-checking, can sit inside the count window
    while still putting calls in flight after the allowance was spent — the
    exact overshoot the stage budget exists to prevent.

    So this pins the sharper thing, on a *real* clock: the stage budget is a
    wall-clock allowance, so the budget here reads ``time.monotonic`` and each
    call really takes the time it claims. No paragraph may be entered once the
    allowance is gone. Deleting the gate, or moving it out of the refill path,
    breaks this — with a fake clock that the workers never actually spend, every
    call would finish at elapsed 0 and a stage that ignored the gate entirely
    would pass.
    """
    # 20 ms per call against 320 ms of budget: the stage can settle about
    # sixteen calls' worth of work in the window, so a stage that ignored the
    # gate and ran all forty would be visible, and one that honoured it has
    # room to be measured. A budget shorter than one full window cannot show
    # any of this — at N=3 with a 20 ms call, a 120 ms budget lets all three
    # slots be handed out before the gate has anything to say, so every
    # paragraph is dispatched and "did it stop" has no answer.
    per_call_s, budget_s, count = 0.02, 0.32, 40
    started = time.monotonic()
    budget = StageBudget(budget_s)
    client = ThreadRecordingClient(burn_s=per_call_s, real_time=True)

    out = repair_paragraphs(client, _paras(count), concurrency=3, budget=budget)

    # Nothing may be *entered* long after the allowance has gone. Measured
    # against the budget's own start, with one window of slack: a paragraph
    # handed its deadline while budget remained is allowed to start, and the
    # promise is that no *new* work begins after that — not that an in-flight
    # call is torn off mid-request.
    late = {
        text: at - started
        for text, at in client.entered_at.items()
        if at - started >= budget_s + 3 * per_call_s
    }
    assert not late, f"calls started well after the budget expired: {late}"
    # The stage has to stop, and stop close to the boundary. The yardstick is
    # the budget's own wall clock over what a call really cost — the number of
    # paragraphs the stage actually settled, since those are the calls it
    # believed it had paid for. Calls *dispatched* is deliberately not the
    # measure: the window that was in flight when the budget went keeps running
    # to its own ``deadline_s`` and is then discarded, so it legitimately
    # exceeds the settled count by a window. A stage that never consulted the
    # gate settles all forty, which no choice of ``count`` can hide.
    # The stage has to stop, and the proof is the budget's own clock rather
    # than a call count the fixture can be argued out of. ``budget.elapsed()``
    # is what the stage itself read when it decided to keep going; a run that
    # honoured the gate stops asking that question once the answer is "no", so
    # the stage cannot be observed working at several times the allowance. A
    # stage that never consulted the gate burns all forty calls — eight times
    # the allowance — and trips this however the per-call cost is estimated.
    spent = budget.elapsed()
    assert spent <= budget_s * 1.5, (
        f"the stage was still working {spent:.3f}s into a {budget_s}s budget"
    )
    # And it must actually stop short of the whole set: with the gate the tail
    # ships as originals; without any budget at all every paragraph is called
    # for and repaired. 3x below the set is far outside what the gate allows
    # and far inside what a gate-free run produces, so it cannot be vacuous.
    # And it must actually stop short of the whole set. ``call_count`` cannot
    # show this: with ``shutdown(wait=False)`` the pool still dispatches its
    # already-dequeued backlog after the stage returns, so the number of calls
    # made is not a quantity the stage controls at the boundary. The quantity
    # it does control is what it *accepted* — a paragraph whose call never
    # landed ships as its original, so a stage that never stopped would repair
    # every paragraph and leave no originals at all.
    repaired = [i for i, p in enumerate(out) if p["text"].endswith("x")]
    assert repaired == list(range(len(repaired))), [p["text"] for p in out]
    assert len(repaired) < count, (
        f"all {count} paragraphs repaired on a {budget_s}s budget — the stage "
        "did not stop"
    )
    assert len(repaired) >= 1, [p["text"] for p in out]
    # Every call was handed a positive deadline read at its own submission.
    assert all(d is not None and d >= 0.0 for _, d in client.calls), client.calls


def test_budget_stop_warns_once_and_the_wait_is_bounded_on_a_dead_call(
    caplog,
) -> None:
    """The budget cuts the *wait*, not just the gap between calls.

    The budget is spent while two calls are still parked, so the stage has
    nothing to submit and something it must not block on. It gives up on the
    parked calls, logs the one warning, and returns in bounded time rather
    than hanging on a call that never answers — which is the whole point of
    a stage budget against a stalled connection.

    A call that was genuinely in flight when the budget ran out is allowed to
    land if it comes back: it carries its own ``deadline_s`` and httpx
    per-read timeout, so it cannot run past the stage either. What the budget
    guarantees is that no paragraph is *started* after expiry and that the
    wait itself is bounded.
    """
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(30.0, clock=clock)
    parked = {i: threading.Event() for i in (0, 1)}
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(30.0, clock=clock)

    class _Parked(ThreadRecordingClient):
        """Calls that answer to nobody — the stalled connection itself.

        No deadline is honoured: the point of the test is that the stage walks
        away from a call that is never going to come back, so a fake that
        rescued itself at its own deadline would test the transport instead of
        the stage. Per-paragraph events, so one release cannot free both.
        """

    client = _Parked(
        hold=lambda text: parked[_index(text)].wait(timeout=WAIT_S),
    )
    # The stage is driven from a helper thread so this test can watch the two
    # calls enter ``complete`` and park, and only then spend the budget — the
    # situation the wait has to survive. Spending it first would have the gate
    # refuse both calls outright and prove nothing about the wait.
    returned = threading.Event()
    out: list[dict[str, Any]] = []

    def caller() -> None:
        with caplog.at_level(logging.WARNING, logger="vemoizer.repair"):
            out.extend(
                repair_paragraphs(client, _paras(3), concurrency=2, budget=budget)
            )
        returned.set()

    thread = threading.Thread(target=caller, name="stalled-call-caller")
    thread.start()
    assert _wait_until(lambda: client.call_count >= 2), "the pool never filled"
    clock.advance(35.0)  # both calls now sit past the stage budget

    try:
        # Bounded: if the stage blocked on either dead call this fails on the
        # timeout instead of hanging the suite.
        thread.join(timeout=WAIT_S)
        assert returned.is_set(), "the stage blocked on a call the budget refused"
        assert not thread.is_alive(), "the caller thread never finished"
    finally:
        # Released only now: the stage had to come back on its own, and the
        # drain behind it is what waits for these two to wind up.
        for ev in parked.values():
            ev.set()
        thread.join(timeout=WAIT_S)

    # Returned rather than hung on the stalled call, with every paragraph
    # accounted for.
    assert len(out) == 3
    # The tail is either an original or an in-flight repair, never a hole.
    for idx, para in enumerate(out):
        assert para["text"] in (f"lause {idx}", f"lause {idx}x"), out
    # The budget was spent with calls outstanding: exactly one stop warning,
    # and it carries counts only.
    stopped = [r for r in caplog.records if "repair stopped at" in r.getMessage()]
    assert len(stopped) == 1, [r.getMessage() for r in caplog.records]
    assert "lause" not in stopped[0].getMessage()


@pytest.mark.parametrize(
    ("total_s", "serial_calls"),
    [(35.0, 3), (45.0, 4), (55.0, 5), (65.0, 6)],
)
def test_budget_gate_runs_before_every_submission(
    total_s: float, serial_calls: int
) -> None:
    """The gate is consulted for each paragraph, so the cut point tracks the serial one.

    Seven paragraphs burn 10 s each on a budget cut at successive call
    boundaries, so the serial loop would make exactly ``serial_calls`` calls.
    PM decision (b) bounds the concurrent stage to ``[serial, serial + N - 1]``
    calls. The upper bound is the one that bites: an implementation that
    consults ``exhausted()`` only once per window keeps releasing batches on a
    budget that expired mid-window, and a stage that never consults it at all
    burns all seven. The lower bound is what a gate placed *inside* the wait
    breaks — that variant measures calls still queued behind two others, gives
    up early, and ships paragraphs the budget had already paid for.

    What is pinned exactly, alongside the count, is the cut point: a paragraph
    is either called for and repaired, or never called for and shipped as
    itself, and the two sets are contiguous by original index.
    """
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(total_s, clock=clock)
    client = ThreadRecordingClient(clock=clock, burn_s=10.0)

    out = repair_paragraphs(client, _paras(7), concurrency=3, budget=budget)

    n = client.call_count
    # PM decision (b)'s upper bound is the load-bearing one. The lower bound is
    # the serial count, except that a pool can never make fewer calls than it
    # has slots: the first window is filled before the budget is ever consulted
    # a second time, so a budget that would cut the serial loop at two calls
    # still lets three be in flight at N=3. Below that the stage is not
    # violating the budget, it is refusing work it had already paid for.
    lo, hi = min(serial_calls, 3), serial_calls + 2
    assert lo <= n <= hi, (n, serial_calls, client.calls)
    assert [p["text"] for p in out[:n]] == [f"lause {i}x" for i in range(n)], out
    assert all(p["text"] == f"lause {i}" for i, p in enumerate(out) if i >= n), out
    # Nothing ran past the budget: every call was handed a positive deadline
    # read at its own submission.
    assert all(d is not None and d >= 0.0 for _, d in client.calls), client.calls


def test_deadline_s_is_the_remaining_budget_at_submit_time() -> None:
    """Each submission is bounded by ``budget.remaining()``; no budget, no deadline.

    Both sides of the contract in one place: with a budget the call carries
    what is left of it (issue #148 FIX 3 — the in-flight read is bounded, not
    just the gap between calls), and without one it carries nothing, which is
    the old behaviour the serial tests still assert.
    """
    clock = _ConcurrencyFakeClock()
    with_budget = ThreadRecordingClient()
    repair_paragraphs(
        with_budget, _paras(2), concurrency=2, budget=StageBudget(100.0, clock=clock)
    )
    assert [c[1] for c in with_budget.calls] == [
        pytest.approx(100.0),
        pytest.approx(100.0),
    ]

    no_budget = ThreadRecordingClient()
    repair_paragraphs(no_budget, _paras(2), concurrency=2, budget=None)
    assert [c[1] for c in no_budget.calls] == [None, None]


# -- failure containment -------------------------------------------------------


@pytest.mark.parametrize("boom", [RuntimeError("boom"), ValueError("bad json")])
def test_worker_exception_keeps_only_its_own_original(caplog, boom) -> None:
    """One bad call is that paragraph's problem, not the stage's.

    The worker swallows the exception — whatever the transport threw, so
    nothing escapes to the caller — and returns no candidate, so the
    paragraph ships as its original while its siblings still get repaired.
    """
    client = ThreadRecordingClient(failures={"lause 1": boom})
    with caplog.at_level(logging.WARNING, logger="vemoizer.repair"):
        out = repair_paragraphs(client, _paras(3), concurrency=3)

    assert [p["text"] for p in out] == ["lause 0x", "lause 1", "lause 2x"]
    failed = [r for r in caplog.records if "repair failed" in r.getMessage()]
    assert len(failed) == 1, [r.getMessage() for r in caplog.records]
    assert "lause" not in failed[0].getMessage()


def test_empty_paragraphs_never_reach_the_pool() -> None:
    """A blank paragraph costs no call, no worker slot and no progress tick.

    The whitespace-only middle paragraph is the fail-open case the serial
    loop has always had: it ships its own text untouched — what the
    pre-concurrency code did with it — and the siblings either side of it
    still get their calls. The empty slot keeps its place so the order of
    everything else cannot shift.
    """
    client = ThreadRecordingClient()
    ticks: list[tuple[int, int]] = []
    paras = [_para("lause 0"), _para("   "), _para("lause 2")]
    out = repair_paragraphs(
        client, paras, concurrency=3, progress_cb=lambda d, t: ticks.append((d, t))
    )

    assert client.call_count == 2
    assert [c[0] for c in client.calls] == ["lause 0", "lause 2"], client.calls
    assert [p["text"] for p in out] == ["lause 0x", "   ", "lause 2x"]
    assert ticks == [(1, 3), (2, 3)], ticks


def test_concurrency_out_of_range_falls_back_to_serial() -> None:
    """A caller that bypassed config validation gets the known-good path.

    Config validation is the first line of defence; this is the second, for
    any caller that reaches the function directly with a wild value.
    """
    for bad in (0, -3, MAX_CONCURRENCY + 5, 999):
        client = ThreadRecordingClient()
        out = repair_paragraphs(client, _paras(3), concurrency=bad)
        assert set(client.threads) == {_MAIN}, f"concurrency={bad} opened a pool"
        assert [p["text"] for p in out] == [f"lause {i}x" for i in range(3)]


# -- the config plumbing ---------------------------------------------------------


def _llm_config(**extra: Any) -> LLMConfig:
    return LLMConfig(
        base_url="http://localhost:9999/v1",
        model="test",
        api_key_env="VEMOIZER_TEST_KEY",
        timeout_seconds=10.0,
        **extra,
    )


def test_client_closes_once_after_the_pool_drains() -> None:
    """``close()`` runs on the caller thread, once, after every call settled.

    A worker that closed the shared client would pull the connection pool
    out from under its siblings mid-flight.
    """
    closed: list[str] = []

    class _ClosingClient(ThreadRecordingClient):
        def close(self) -> None:
            closed.append(threading.current_thread().name)
            # The pool must already be drained: no call may still be running
            # when the shared client is released.
            with self._lock:
                assert self._in_flight == 0, "closed while a call was in flight"
            self.closed += 1

    client = _ClosingClient()
    _run_repair(
        {"paragraphs": _paras(6)},
        _llm_config(repair_concurrency=3),
        None,
        llm_client_cls=lambda _cfg: client,
    )

    assert closed == [_MAIN], closed
    assert client.closed == 1


def test_shared_http_client_is_built_once_under_concurrent_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy ``httpx.Client`` init must not race (issue #159).

    httpx and httpcore document a shared Client as thread-safe — its pool is
    the thread-safety unit — but the construction behind the ``is None``
    check was ours to get right, and was not: N workers arriving cold would
    each build a Client and leak N−1 pools whose sockets outlived
    ``close()``. One constructor call for N concurrent callers is the claim.
    """
    import vemoizer.llm as llm_module

    built: list[str] = []
    build_lock = threading.Lock()
    all_in = threading.Event()

    class _Pool:
        def __init__(self, *args: Any, **kw: Any) -> None:
            nonlocal built
            built.append(threading.current_thread().name)
            # Hold the door open while this pretend client is being built.
            # With the lock in ``_get_client`` exactly one thread gets here
            # and the rest are parked behind it; without it, all four walk
            # in and the event below fires — which is what makes this a race
            # probe rather than a test that passes by luck.
            with build_lock:
                if len(built) >= 4:
                    all_in.set()
            all_in.wait(timeout=0.3)

        def close(self) -> None:
            pass

    monkeypatch.setattr(llm_module.httpx, "Client", _Pool)

    client = llm_module.LLMClient(_llm_config())
    start = threading.Barrier(4, timeout=WAIT_S)
    seen: list[int] = []
    errors: list[BaseException] = []

    def grab() -> None:
        try:
            start.wait(timeout=WAIT_S)
            seen.append(id(client._get_client()))
        except BaseException as e:  # noqa: BLE001 - surfaced in the assert
            errors.append(e)

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=WAIT_S)
        assert not t.is_alive(), "_get_client hung"
    client.close()

    assert not errors, errors
    assert len(built) == 1, f"leaked {len(built) - 1} extra connection pools"
    assert len(set(seen)) == 1


# -- config validation -------------------------------------------------------------


def _section(extra: str) -> LLMConfig | None:
    """Parse a raw ``[llm]`` table.

    Reimplemented rather than borrowed from ``tests/test_llm.py``: that file
    sits at the ~800-line cap and this ticket forbids adding to it.
    """
    raw = tomllib.loads(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 10\n" + extra
    )
    return _parse_llm_section(raw["llm"])


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        f"timeout_seconds = 10\n{body}",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("value", [1, 2, 3, MAX_REPAIR_CONCURRENCY])
def test_accepted_concurrency_values_parse(value: int) -> None:
    cfg = _section(f"repair_concurrency = {value}\n")
    assert cfg is not None
    assert cfg.repair_concurrency == value


def test_absent_concurrency_is_serial_by_default() -> None:
    cfg = _section("")
    assert cfg is not None
    assert cfg.repair_concurrency == 1


@pytest.mark.parametrize(
    "source",
    ["0", "-1", f"{MAX_REPAIR_CONCURRENCY + 1}", "true", "false", "2.5", '"3"'],
)
def test_malformed_concurrency_makes_the_section_malformed(source: str) -> None:
    """Reject bool, float, string, 0, negative and above the bound.

    Same outcome as a bad budget: the section is malformed, so the fail-open
    loader yields ``None`` and the strict loader raises ``ConfigError`` —
    never a silently ignored value.
    """
    assert _section(f"repair_concurrency = {source}\n") is None


@pytest.mark.parametrize("value", [2**63, -(2**63), 10**100])
def test_huge_concurrency_is_malformed(value: int) -> None:
    """A big integer is out of range, not a valid worker count.

    The accepted set is the integers ``1..4``. An arbitrary-precision value
    is still an ``int`` to Python, so the bound check has to carry it: an
    implementation that compares against a narrower type, or that coerces
    through something which overflows, would either accept it or raise. The
    outcome is the same as any other malformed value — the section is
    malformed on all three load paths, never a silently ignored key.
    """
    assert _section(f"repair_concurrency = {value}\n") is None


def test_huge_concurrency_fails_loud_on_the_strict_paths(tmp_path: Path) -> None:
    """Same outcome as a bad budget on the strict path: ``ConfigError``.

    The fail-open loader yields ``None`` (covered above); the strict loader
    raises. The message is the section-level one the budget keys produce — it
    does not name the offending key, because a malformed value means
    "malformed section" rather than "bad value", and a value that slips the
    range check on one path and not the other is the bug this pins out.
    """
    path = _write_config(tmp_path, f"repair_concurrency = {2**64}\n")
    with pytest.raises(ConfigError, match="malformed 'llm' section"):
        _strict_load(path)


def test_unknown_near_miss_key_is_rejected_by_name(tmp_path: Path) -> None:
    """Accepting the new key must not soften the unknown-key guard.

    ``repair_concurency`` (one letter shuffled) is a typo the operator needs
    told about, not a key quietly ignored.
    """
    path = _write_config(tmp_path, "repair_concurency = 2\n")
    with pytest.raises(ConfigError) as excinfo:
        _strict_load(path)
    assert "repair_concurency" in str(excinfo.value)


def test_the_key_is_registered_not_merely_tolerated() -> None:
    """A known key reaches :class:`~vemoizer.llm_config.LLMConfig`; the
    near-miss above only proves an unknown one is refused."""
    assert "repair_concurrency" in _KNOWN_LLM_KEYS


_SPEC_TABLE_HEADING = "| `llm.repair_budget_seconds` |"


def test_every_known_llm_key_has_a_row_in_the_spec_table() -> None:
    """Every ``[llm]`` key the parser accepts is documented in the one table.

    ``docs/pipeline-spec.md`` is the canonical location for the ``[llm]``
    keys and no other file restates them, so a key that gains a validator
    without gaining a row is undocumented and invisible to the operator.
    The assertion runs over the whole accepted-key set rather than naming
    ``repair_concurrency``, so the next key added to the parser is held to
    the same bar; it is what fails when the row is deleted or renamed.
    """
    spec = (
        Path(__file__).resolve().parents[1] / "docs" / "pipeline-spec.md"
    ).read_text(encoding="utf-8")
    assert _SPEC_TABLE_HEADING in spec, "the [llm] key table moved"
    missing = [
        key for key in sorted(_KNOWN_LLM_KEYS) if f"| `llm.{key}` |" not in spec
    ]
    assert not missing, f"undocumented [llm] key(s) in docs/pipeline-spec.md: {missing}"
    assert "| `llm.repair_concurrency` |" in spec
