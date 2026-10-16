"""The worker pool the concurrent repair pass runs its calls on (issue #159).

Split out of :mod:`vemoizer.repair` because the two concerns change for
different reasons. ``repair.py`` is the stage: the prompts, the no-invention
guard, the budget cut point, the progress contract. This module is the
lifecycle of the threads that carry the HTTP calls — what a pool is called,
what it is named, and, above all, what happens to calls it has been told to
abandon.

The rule that shapes everything here: **the stage never blocks on a call it
has given up on.** Pool workers are non-daemon and ``concurrent.futures``
registers an atexit hook that joins them, so the tempting move — a tidy
``shutdown(wait=True)`` so nothing leaks — puts the CLI straight back inside
the wait the stage budget exists to escape. A stalled connection that keeps
resetting its per-read timeout (issue #148) would turn "the repair stage is
done" into "the terminal is frozen". So the pool is released with
``wait=False``, stragglers are named in the log rather than joined, and the
call's own ``deadline_s`` plus the per-read httpx timeout are what actually
end it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

logger = logging.getLogger(__name__)

#: Prefix on every worker this module starts. It is the handle the drain uses
#: to spot a straggler, and the name a test asserts no thread of survives the
#: stage — so it is a constant, not a literal repeated at two call sites.
THREAD_PREFIX = "vemoizer-repair"

#: How the collector's wait on one in-flight call is chunked.
#: :func:`concurrent.futures.wait` cannot be woken early, so "the budget is
#: gone" can only be noticed at a chunk boundary: poll often enough that
#: giving up is prompt, since a CLI that looks frozen for five seconds reads
#: as broken.
AWAIT_POLL_S = 0.05

#: How long one call must still be outstanding before the wait is reported in
#: the run log, and how long the pool's workers stay in view after
#: ``shutdown(wait=False)``. Deliberately longer than the poll interval: a
#: single slow paragraph should not spam the log.
HEARTBEAT_S = 5.0

#: Returned by :func:`await_candidate` when the wait was given up on. Distinct
#: from ``None``, which the stage already uses for "the model offered nothing".
ABANDONED: Any = "_repair_pool_abandoned"


def await_candidate(
    future: Future,
    *,
    exhausted: Callable[[], bool],
    on_stalled: Callable[[], None],
) -> Any:
    """Return the future's result, giving up if *exhausted* ever says so.

    The collector's wait. The callable pair is the seam: this module must not
    know what a stage budget is, and the stage must not know how a future is
    polled. ``on_stalled`` fires at most once per wait, after
    :data:`HEARTBEAT_S`, so a genuinely slow call leaves one log line and a
    merely-queue-jittery run leaves none.
    """
    while not future.done():
        if exhausted():
            return ABANDONED
        waited = time.monotonic()
        reported = False
        while not future.done():
            if exhausted():
                return ABANDONED
            if futures_wait([future], timeout=AWAIT_POLL_S).done:
                break
            if not reported and time.monotonic() - waited >= HEARTBEAT_S:
                reported = True
                on_stalled()
    return future.result()


def open_pool(workers: int) -> ThreadPoolExecutor:
    """The bounded pool one concurrent repair pass runs its calls on."""
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=THREAD_PREFIX)


def drain(pool: ThreadPoolExecutor, futures: dict[int, Future]) -> None:
    """Release the pool without blocking on calls the stage has abandoned.

    ``shutdown(wait=False, cancel_futures=True)`` (issue #159, PM decision
    (d)): futures that never started are cancelled outright, and the ones
    already running are *not* joined here. Waiting for them would undo the
    point of the stage budget — a call the budget refused is a call the run is
    done with, and blocking on it is exactly the hang the budget exists to
    prevent. Each in-flight call carries its own bound in any case
    (``complete(deadline_s=...)`` plus the per-read httpx timeout), and
    :func:`concurrent.futures.thread._python_exit` joins any worker that
    outlives us at interpreter shutdown — the right place for that wait, past
    the CLI's own exit, rather than inside a stage that has already returned.

    A worker that ignores ``deadline_s`` and never returns is therefore not
    waited on, only counted, and said so in the log: a leaked non-daemon
    worker would otherwise turn the next Ctrl-C into a hang no one can
    explain.
    """
    for future in futures.values():
        future.cancel()
    # ``cancel_futures=True`` is what stops the tasks the pool has queued but
    # not started, and it is what the stage relies on to keep its promise: once
    # the budget is gone no further paragraph is called for. It is not redundant
    # with the loop above — the collector has already popped the futures it
    # settled out of *futures*, and with ``wait=False`` nothing else would stop
    # the pool's own backlog from being handed to a worker.
    #
    # One caveat the caller must not be allowed to forget: cancellation only
    # reaches work that has not *started*. A task already dequeued is running,
    # and no amount of cancelling turns it off — which is why the per-call
    # ``deadline_s`` is load-bearing rather than belt-and-braces.
    pool.shutdown(wait=False, cancel_futures=True)
    stragglers = [
        t
        for t in threading.enumerate()
        if t.name.startswith(THREAD_PREFIX) and t is not threading.current_thread()
    ]
    if stragglers:
        logger.warning(
            "repair: %d worker(s) left in flight at the stage boundary; each "
            "carries its own call deadline and the pool joins them at exit",
            len(stragglers),
        )
