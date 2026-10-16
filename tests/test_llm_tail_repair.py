"""How the tail wires ``repair_concurrency`` into the stage, and closes the client.

The tail (:mod:`vemoizer.llm_tail`) is the seam between a parsed ``[llm]``
section and the repair stage, and two things are easy to break here and
invisible everywhere else: the concurrency value can be dropped on its way
through, which turns a documented config key into a silent no-op; and the
shared client can be closed while calls are still in flight, or by a worker,
which strands a connection pool mid-request.

Split from ``tests/test_repair_concurrency.py`` (which pins what the pool does)
and kept apart from ``tests/test_llm.py`` (at its file cap, and the client's
own surface). Driven through the ``*_fn`` / ``llm_client_cls`` seams the tail
exists to expose: no pipeline, no network, no display beyond a stub.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from support_repair_fake import WAIT_S, ThreadRecordingClient, _paras

from vemoizer.llm import LLMConfig
from vemoizer.llm_tail import _run_repair

_MAIN = threading.main_thread().name


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
