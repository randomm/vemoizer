"""Ctrl-C handling for the ``meeting`` / ``memo`` preset commands (issue #148).

When the user interrupts a preset run, the command prints exactly one line
naming the stage the run was in and that no files were written, then exits
130 (the SIGINT convention) without a traceback.

A single module-level state pair (``_stage`` / ``_written_count``) is the
bookkeeping: ``begin_interrupt_tracking`` resets it at command start (the
command is the process boundary — one ``meeting``/``memo`` invocation at a
time), the run's own seam (``_set_interrupt_stage``) names the active
stage, and ``note_written_files`` tallies the files the run managed to
write. ``tests`` call ``reset_interrupt_tracking`` between invocations in
the same process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vemoizer.progress import ProgressDisplay

__all__ = [
    "begin_interrupt_tracking",
    "handle_interrupt",
    "note_written_files",
    "reset_interrupt_tracking",
]

#: The last stage named by ``_set_interrupt_stage``; ``None`` when the run
#: has not started any stage the bookkeeping knows about yet.
_stage: str | None = None
#: The number of output files written so far in the current invocation.
_written_count = 0


def begin_interrupt_tracking(display: ProgressDisplay | None) -> None:
    """Arm the Ctrl-C stage tracking for one ``meeting``/``memo`` invocation.

    Called at the top of the command, before any pipeline stage runs, so
    the first interrupt names the correct stage (the display's current
    task) and the "no files written" claim starts true. *display* is read
    only on interrupt (``handle_interrupt``), never here.
    """
    del display
    reset_interrupt_tracking()


def reset_interrupt_tracking() -> None:
    """Forget any previous invocation's bookkeeping (tests; re-entry)."""
    global _stage, _written_count
    _stage = None
    _written_count = 0


def _set_interrupt_stage(stage: str) -> None:
    """Name the stage the run is entering (called at the stage's seam)."""
    global _stage
    _stage = stage


def note_written_files(paths: list[Any]) -> None:
    """Tally files written this invocation; the line then reports the true
    "no files written" / "N files already written" state."""
    global _written_count
    _written_count += len(paths)


def handle_interrupt(display: ProgressDisplay | None) -> str:
    """Build the single Ctrl-C line for the current run.

    The stage is the display's active task description (the live decode /
    diarize / repair line) when one is running, else the last stage the
    bookkeeping named, else the generic "this run". The written-files
    claim is state-based, not hardcoded: an interrupted run that already
    wrote earlier files says so instead of lying.
    """
    stage = _active_display_stage(display) or _stage or "this run"
    if _written_count == 0:
        files_part = "no files written"
    else:
        files_part = f"{_written_count} file(s) already written"
    return f"interrupted during {stage} — {files_part}"


def _active_display_stage(display: ProgressDisplay | None) -> str | None:
    """The display's active (not-yet-finished) task description, or None.

    The description is read off the ``rich.progress.Task`` directly (no
    public accessor exists on ``ProgressDisplay``); the ``[green]`` check
    matches the ``prefix_active_stage`` convention for a finished stage.
    """
    if display is None:
        return None
    try:
        tasks = display._progress.tasks  # noqa: SLF001
    except AttributeError:  # noqa: SIM105 - display object without _progress
        return None
    if not tasks:
        return None
    task = tasks[-1]
    description = task.description  # noqa: SLF001
    if not description or description.startswith("[green]"):
        return None
    return description
