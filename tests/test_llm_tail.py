"""The LLM tail's repair stage: how the config reaches the stage (issue #159).

The tail (:mod:`vemoizer.llm_tail`) is the seam between a parsed
``[llm]`` section and the repair stage itself. Two things are easy to
break here and invisible everywhere else:

- the ``repair_concurrency`` value can be dropped on its way through, which
  turns a documented config key into a silent no-op — no test would notice
  except one that watches the call;
- the shared client can be closed while calls are still in flight, or by a
  worker, which strands a connection pool mid-request.

Both are pinned here at the boundary that owns them, rather than in
``tests/test_repair.py`` (the serial oracle for the stage itself) or
``tests/test_repair_concurrency.py`` (the pool's behaviour). ``test_llm.py``
is at its file cap and is the client's own surface, not the tail's.

No pipeline, no network, no display beyond a stub: the tail is driven
directly through the ``*_fn`` / ``llm_client_cls`` seams it exists to
expose.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from vemoizer.llm import LLMConfig
from vemoizer.llm_budget import StageBudget
from vemoizer.llm_tail import _run_repair

_MAIN = threading.main_thread().name


def _config(**extra: Any) -> LLMConfig:
    return LLMConfig(
        base_url="http://localhost:9999/v1",
        model="test",
        api_key_env="VEMOIZER_TEST_KEY",
        timeout_seconds=10.0,
        **extra,
    )


def _paras(n: int) -> list[dict[str, Any]]:
    return [{"start": 0.0, "end": 5.0, "text": f"lause {i}"} for i in range(n)]


class _SpyClient:
    """Stands in for ``LLMClient`` and records how the tail uses it."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config
        self.closes: list[str] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        /,
        max_tokens: int = 2048,
        *,
        deadline_s: float | None = None,
    ) -> str | None:
        return user_prompt

    def close(self) -> None:
        self.closes.append(threading.current_thread().name)


class _StageSpy:
    """Captures the keyword arguments the tail hands the repair stage."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        client: Any,
        paragraphs: Any,
        glossary: Any = None,
        **kw: Any,
    ) -> Any:
        self.calls.append(
            {"client": client, "paragraphs": paragraphs, "glossary": glossary, **kw}
        )
        return paragraphs


def _run(
    *,
    concurrency: int | None = None,
    stage: _StageSpy | None = None,
    client: _SpyClient | None = None,
) -> _StageSpy:
    """Drive the real ``_run_repair`` through its injectable seams."""
    stage = stage or _StageSpy()
    client = client or _SpyClient()
    config = (
        _config() if concurrency is None else _config(repair_concurrency=concurrency)
    )
    _run_repair(
        {"paragraphs": _paras(4)},
        config,
        None,
        repair_paragraphs_fn=stage,
        llm_client_cls=lambda _cfg: client,
    )
    return stage


# -- the config reaches the stage -----------------------------------------


def test_configured_concurrency_reaches_the_repair_stage() -> None:
    """A configured worker count must arrive at the stage.

    Nothing else in the suite ties ``[llm] repair_concurrency`` to the
    ``concurrency`` argument, so a dropped keyword here would make the
    documented key a no-op that every other test still passes.
    """
    stage = _run(concurrency=3)

    assert len(stage.calls) == 1
    assert stage.calls[0]["concurrency"] == 3


def test_default_config_reaches_the_stage_as_serial() -> None:
    """A default config means the keyword is *not sent* — the serial call shape.

    This is the pin that keeps the four pipeline-namespace seams in
    ``test_batch_hardening`` / ``test_at_terms_e2e`` callable: their
    ``fake_repair`` stubs accept no ``concurrency`` keyword, and at the
    default they must keep being called exactly as before this feature
    existed. Sending it unconditionally would break them and would also
    claim a concurrency decision on a run that never made one.
    """
    stage = _run()

    assert "concurrency" not in stage.calls[0], stage.calls[0]


def test_the_stage_still_receives_its_budget_and_progress_contract() -> None:
    """Threading the new argument through did not disturb the old ones.

    The budget and ``progress_cb`` contracts predate concurrency (issue
    #148) and the tail is the only place they are assembled.
    """
    stage = _run(concurrency=2)
    call = stage.calls[0]

    assert isinstance(call["budget"], StageBudget)
    assert call["progress_cb"] is None  # no display passed
    assert call["glossary"] is None
    assert len(call["paragraphs"]) == 4


def test_the_client_the_stage_gets_is_the_client_the_tail_closes() -> None:
    """One client for the stage: the object handed in is the one released."""
    client = _SpyClient()
    stage = _run(concurrency=3, client=client)

    assert stage.calls[0]["client"] is client


# -- the client is closed exactly once, on the caller's thread --------------


def test_client_closed_once_by_the_calling_thread() -> None:
    """``close()`` runs once, on the caller's thread, in the tail's ``finally``.

    A worker closing the shared client would pull the connection pool out
    from under its siblings mid-flight; a second close would be a leak of
    intent if not of sockets.
    """
    client = _SpyClient()
    _run(concurrency=3, client=client)

    assert client.closes == [_MAIN], client.closes


def test_client_closed_once_when_the_stage_raises() -> None:
    """Fail-open means an exploding stage still releases the client."""
    client = _SpyClient()

    def explode(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("stage blew up")

    with pytest.raises(RuntimeError, match="stage blew up"):
        _run_repair(
            {"paragraphs": _paras(2)},
            _config(repair_concurrency=2),
            None,
            repair_paragraphs_fn=explode,
            llm_client_cls=lambda _cfg: client,
        )

    assert client.closes == [_MAIN], client.closes


def test_client_closed_after_the_last_call_settles() -> None:
    """No call may still be running when the shared client is released.

    The fake records whether a call was in flight at the moment of closing;
    the tail's ``finally`` runs only after the stage returns, and the stage
    drains its own pool before returning, so the count is zero. Asserting it
    here rather than only inside the stage pins the *ordering* across the
    two layers, which neither side can see alone.
    """
    inflight_at_close: list[int] = []
    inflight = [0]
    lock = threading.Lock()

    class _Counting(_SpyClient):
        def complete(self, *args: Any, **kw: Any) -> str | None:
            with lock:
                inflight[0] += 1
            try:
                return super().complete(*args, **kw)
            finally:
                with lock:
                    inflight[0] -= 1

        def close(self) -> None:
            with lock:
                inflight_at_close.append(inflight[0])
            super().close()

    def repair_all(
        client: Any, paragraphs: Any, glossary: Any = None, **kw: Any
    ) -> Any:
        # Exercise the real call path concurrently, then return.
        from vemoizer.repair import repair_paragraphs

        return repair_paragraphs(client, paragraphs, **kw)

    client = _Counting()
    _run_repair(
        {"paragraphs": _paras(6)},
        _config(repair_concurrency=3),
        None,
        repair_paragraphs_fn=repair_all,
        llm_client_cls=lambda _cfg: client,
    )

    assert inflight_at_close == [0], inflight_at_close


# -- the display contract survives concurrency ------------------------------


def test_progress_ticks_once_per_paragraph_from_the_caller_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tail wires the stage's tick to the display, once per paragraph.

    The callback the tail builds is the only thing moving the progress bar;
    under concurrency it must still fire exactly ``total`` times and only
    from the caller's thread, or the bar counts wrong for a whole meeting.

    The real :class:`~vemoizer.progress.ProgressDisplay` is driven with its
    output redirected, so this exercises the actual advance/update/finish
    calls rather than a stand-in that could drift from them.
    """
    import io
    import sys

    from vemoizer.progress import ProgressDisplay

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()  # isatty patched before construction

    ticks: list[tuple[int, int, str]] = []

    def stage(client: Any, paragraphs: Any, glossary: Any = None, **kw: Any) -> Any:
        cb = kw["progress_cb"]
        assert cb is not None
        for done in range(1, len(paragraphs) + 1):
            cb(done, len(paragraphs))
            ticks.append((done, len(paragraphs), threading.current_thread().name))
        return paragraphs

    _run_repair(
        {"paragraphs": _paras(4)},
        _config(repair_concurrency=3),
        None,
        repair_paragraphs_fn=stage,
        llm_client_cls=_SpyClient,
        display=display,
    )

    assert [t[0] for t in ticks] == [1, 2, 3, 4], ticks
    assert all(t[2] == _MAIN for t in ticks), ticks
    # The bar reached its total exactly once, and the stage task was closed
    # rather than left spinning (the #143 close-the-task contract).
    tasks = display._progress.tasks  # noqa: SLF001
    assert any(t.completed == 4 for t in tasks), [t.completed for t in tasks]
    assert all(t.description.startswith("[green]\u2713") for t in tasks), [
        t.description for t in tasks
    ]
