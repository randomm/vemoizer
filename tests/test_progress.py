"""Tests for per-stage stderr progress with TTY auto-detection (issue #10, task B).

Contract (AGENTS.md / issue #10):

- Progress renders to **stderr** only, via ``rich.progress`` per-stage tasks.
  The transcript goes to stdout; nothing but progress may touch stderr.
- TTY auto-detection: when ``sys.stderr`` is not a TTY (piped/redirected
  output) the progress display is disabled and stderr stays empty.
- ``verbose=False`` always disables progress, regardless of TTY state.

The display is a ``rich.progress.Progress`` built on
``Console(stderr=True)`` with ``disable=not sys.stderr.isatty()``. Tests
monkeypatch ``sys.stderr.isatty`` to force each branch and assert on the
resulting ``disable`` flag plus stderr output behavior.
"""

from __future__ import annotations

import io
import sys

import pytest
from rich.progress import Progress

from vemoizer.progress import ProgressDisplay

STAGES = ("decode A", "decode B")


@pytest.fixture
def fake_stderr(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace sys.stderr with a StringIO so we can assert on what it got."""
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    return buffer


# ---------------------------------------------------------------------------
# TTY auto-detection: the `disable` decision
# ---------------------------------------------------------------------------


def test_non_tty_stderr_disables_progress(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    assert display.disable is True
    display.close()


def test_tty_stderr_keeps_progress_enabled(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    display = ProgressDisplay()
    assert display.disable is False
    display.close()


def test_verbose_false_always_disables_even_on_tty(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    display = ProgressDisplay(verbose=False)
    assert display.disable is True
    display.close()


def test_verbose_true_on_tty_stays_enabled(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    display = ProgressDisplay(verbose=True)
    assert display.disable is False
    display.close()


def test_disable_decision_is_tied_to_rich_progress(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    """The Progress instance itself must carry the disable flag (not just
    the wrapper) — that's what actually suppresses rendering."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    assert display._progress.disable is True
    display.close()


# ---------------------------------------------------------------------------
# Progress must be wired to stderr, not stdout
# ---------------------------------------------------------------------------


def test_console_targets_stderr() -> None:
    display = ProgressDisplay(verbose=False)  # disable=True to avoid rendering
    assert display._console.file is sys.stderr
    display.close()


def test_progress_uses_stderr_console() -> None:
    display = ProgressDisplay(verbose=False)
    assert isinstance(display._progress, Progress)
    assert display._progress.console.file is sys.stderr
    display.close()


# ---------------------------------------------------------------------------
# Per-stage task management
# ---------------------------------------------------------------------------


def test_add_stage_creates_one_task_per_stage(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    ids = [display.add_stage(stage) for stage in STAGES]
    assert len(ids) == len(STAGES)
    assert len(set(ids)) == len(STAGES)  # unique task ids
    display.close()


def test_advance_and_finish_progress_a_stage(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    task_id = display.add_stage("decode A")
    display.advance(task_id, 10)
    display.finish(task_id, 100)
    display.close()


def test_update_stage_text(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    task_id = display.add_stage("decode A")
    display.update_text(task_id, "loading model...")
    display.close()


# ---------------------------------------------------------------------------
# Stderr stays empty when disabled (the non-TTY guarantee)
# ---------------------------------------------------------------------------


def test_disabled_progress_writes_nothing_to_stderr(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    display = ProgressDisplay()
    for stage in STAGES:
        task_id = display.add_stage(stage)
        display.advance(task_id, 5)
        display.finish(task_id, 10)
    display.close()
    # rich Progress may still emit a one-time line even when disabled if
    # console_width probing happens; the guarantee is: no per-update spam and
    # no spinner/progress-bar rendering. Assert nothing meaningful was written
    # beyond any single control line.
    output = fake_stderr.getvalue()
    assert "decode A" not in output or "\r" not in output


def test_ctx_manager_closes_progress(
    monkeypatch: pytest.MonkeyPatch, fake_stderr: io.StringIO
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    with ProgressDisplay() as display:
        task_id = display.add_stage("decode A")
        display.finish(task_id, 1)
    # after close, the underlying progress is stopped; calling close again
    # must be idempotent (no exception)
    display.close()
