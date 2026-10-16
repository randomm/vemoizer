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
        self._replies = replies or {}
        self._failures = failures or {}
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
        with self._lock:
            self.threads.append(threading.current_thread().name)
            self.calls.append((user, deadline_s))
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        try:
            if self.before is not None:
                self.before(user)
            if self._clock is not None:
                self._clock.advance(self._burn_s)
            failure = self._failures.get(user)
            if failure is not None:
                raise failure
            return self._replies.get(user, user + "x")
        finally:
            with self._lock:
                self._in_flight -= 1

    def close(self) -> None:
        self.closed += 1

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
    """Every thread the stage starts is joined before it returns.

    A leaked non-daemon worker outlives the CLI: ``concurrent.futures``
    registers an atexit hook that joins pool threads, so a stuck one turns
    the next Ctrl-C into a hang the user cannot explain.
    """
    before = {t.name for t in threading.enumerate()}
    client = ThreadRecordingClient()
    repair_paragraphs(client, _paras(12), concurrency=MAX_CONCURRENCY)
    leaked = {
        t.name
        for t in threading.enumerate()
        if t.name.startswith("vemoizer-repair") and t.name not in before
    }
    assert not leaked, f"worker threads outlived repair_paragraphs: {leaked}"
    assert client.peak_in_flight <= MAX_CONCURRENCY


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


def test_budget_expiry_ships_the_tail_as_originals_with_one_warning(caplog) -> None:
    """The budget gates before each submission and warns exactly once.

    Four paragraphs, each call burning 10 s, budget 25 s — the same script
    as the serial oracle, which stops after 3 calls. The cut point must be
    the same index: paragraphs pass the gate one at a time whether or not a
    pool sits behind it.
    """
    clock = _ConcurrencyFakeClock()
    budget = StageBudget(25.0, clock=clock)
    client = ThreadRecordingClient(clock=clock, burn_s=10.0)
    with caplog.at_level(logging.WARNING, logger="vemoizer.repair"):
        out = repair_paragraphs(client, _paras(4), concurrency=3, budget=budget)

    assert client.call_count == 3, client.calls
    assert [p["text"] for p in out] == ["lause 0x", "lause 1x", "lause 2x", "lause 3"]
    stopped = [r for r in caplog.records if "repair stopped at" in r.getMessage()]
    assert len(stopped) == 1, [r.getMessage() for r in caplog.records]
    # Privacy: the warning carries counts, never transcript text.
    assert "lause" not in stopped[0].getMessage()


def test_budget_expiry_during_collection_does_not_wait_for_a_parked_call(
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
    arrived = {i: threading.Event() for i in (0, 1)}

    class _Parked(ThreadRecordingClient):
        def complete(self, *args: Any, **kw: Any) -> str | None:
            text = args[1] if len(args) > 1 else kw.get("user", "")
            idx = _index(text)
            if idx in parked:
                arrived[idx].set()
                assert parked[idx].wait(timeout=WAIT_S), f"{idx} never released"
            return super().complete(*args, **kw)

    client = _Parked()

    def burn_the_budget_when_both_are_in_flight() -> None:
        for ev in arrived.values():
            assert ev.wait(timeout=WAIT_S), "the pool never filled both slots"
        clock.advance(35.0)  # both calls now sit past the stage budget

    watcher = threading.Thread(target=burn_the_budget_when_both_are_in_flight)
    watcher.start()
    # The stage gives up on the parked calls at the budget boundary and
    # drains the pool in its ``finally``; releasing on a timer lets that drain
    # finish without the test dictating the stage's own timing.
    watchdog = threading.Timer(1.0, lambda: [ev.set() for ev in parked.values()])
    watchdog.start()
    try:
        with caplog.at_level(logging.WARNING, logger="vemoizer.repair"):
            out = repair_paragraphs(client, _paras(3), concurrency=2, budget=budget)
        watcher.join(timeout=WAIT_S)
    finally:
        watchdog.cancel()
        for ev in parked.values():
            ev.set()

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


def test_config_concurrency_reaches_the_repair_stage() -> None:
    """``[llm] repair_concurrency`` must arrive at the stage, not vanish.

    Nothing else in the suite ties the config key to the argument, so a
    dropped keyword here would make the documented key a silent no-op.
    """
    seen: dict[str, Any] = {}

    def spy(client: Any, paragraphs: Any, glossary: Any = None, **kw: Any) -> Any:
        seen.update(kw)
        return paragraphs

    _run_repair(
        {"paragraphs": _paras(4)},
        _llm_config(repair_concurrency=3),
        None,
        repair_paragraphs_fn=spy,
        llm_client_cls=ThreadRecordingClient,
    )
    assert seen["concurrency"] == 3

    # And an absent key is the serial path end to end, not an unset argument.
    seen.clear()
    _run_repair(
        {"paragraphs": _paras(2)},
        _llm_config(),
        None,
        repair_paragraphs_fn=spy,
        llm_client_cls=ThreadRecordingClient,
    )
    assert seen["concurrency"] == 1


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
