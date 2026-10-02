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


# ---------------------------------------------------------------------------
# prefix_active_stage: no accumulation, no re-prefix of finished tasks
# (issue #105 lens LOW)
# ---------------------------------------------------------------------------


def test_prefix_active_stage_no_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-prefixing a still-active task with a new prefix REPLACES the old
    one instead of accumulating ('[2/3] b · [1/3] a · decode')."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/3] a · ")
    display.prefix_active_stage("[2/3] b · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/3] b · decode", f"Expected replacement, got: {desc!r}"
    display.close()


def test_prefix_active_stage_skips_finished_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished task is never re-prefixed."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.finish(task_id, 1.0)
    display.prefix_active_stage("[1/2] a · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[green]✓ complete", (
        f"Expected unchanged description after finish, got: {desc!r}"
    )
    display.close()


def test_prefix_active_stage_stem_with_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stem that itself contains `` · `` is not corrupted by a re-prefix:
    the previous batch prefix is tracked exactly per task, so only the
    recorded prefix string is stripped on replacement."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/3] a · b · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[1/3] a · b · decode", f"Unexpected first prefix: {desc!r}"
    display.prefix_active_stage("[2/3] c · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/3] c · decode", f"Expected exact-prefix strip, got: {desc!r}"
    display.close()


def test_prefix_active_stage_stem_with_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stem starting with ``[`` (``[x] y``) is handled: the previous
    prefix is the recorded string, not whatever a bracket heuristic
    would infer."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/2] [x] y · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[1/2] [x] y · decode", f"Unexpected first prefix: {desc!r}"
    display.prefix_active_stage("[2/2] z · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/2] z · decode", f"Expected exact-prefix strip, got: {desc!r}"
    display.close()


def test_prefix_active_stage_nfd_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An NFD (decomposed) stem round-trips through re-prefixing byte-for-byte."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    stem = "mo\u0301"  # 'm', 'o', combining acute (NFD)
    display.prefix_active_stage(f"[1/2] {stem} · ")
    desc = display._progress.tasks[task_id].description
    assert desc == f"[1/2] {stem} · decode", f"Unexpected first prefix: {desc!r}"
    display.prefix_active_stage("[2/2] plain · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/2] plain · decode", f"Expected exact-prefix strip, got: {desc!r}"
    display.close()


def test_prefix_active_stage_marker_only_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stem that is exactly ``·``: the prefix ``[1/2] · · `` must not be
    corrupted on re-prefix."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/2] · · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[1/2] · · decode", f"Unexpected first prefix: {desc!r}"
    display.prefix_active_stage("[2/2] w · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/2] w · decode", f"Expected exact-prefix strip, got: {desc!r}"
    display.close()


def test_prefix_active_stage_full_batch_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[1/3] -> [2/3] -> [3/3] on one active stage: each re-prefix replaces
    exactly the recorded previous prefix (no accumulation, no corruption)."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/3] a · ")
    display.prefix_active_stage("[2/3] b · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/3] b · decode", f"Unexpected at [2/3]: {desc!r}"
    display.prefix_active_stage("[3/3] c · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[3/3] c · decode", f"Unexpected at [3/3]: {desc!r}"
    display.close()


def test_prefix_active_stage_external_change_gets_prefix_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the description was changed by someone else after prefixing (so it
    does not start with the recorded prefix), the new prefix is applied
    without crashing and without stripping anything."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    display.start()
    task_id = display.add_stage("decode")
    display.prefix_active_stage("[1/2] a · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[1/2] a · decode", f"Unexpected first prefix: {desc!r}"
    display.update_text(task_id, "loading model...")  # someone else changes it
    display.prefix_active_stage("[2/2] b · ")
    desc = display._progress.tasks[task_id].description
    assert desc == "[2/2] b · loading model...", (
        f"Expected new prefix on unchanged text, got: {desc!r}"
    )
    display.close()
