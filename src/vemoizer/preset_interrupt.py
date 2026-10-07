"""Ctrl-C handling for the ``meeting`` / ``memo`` preset commands (issue #148).

When the user interrupts a preset run, the command prints exactly one line
naming the stage the run was in and that no files were written, then exits
130 (the SIGINT convention) without a traceback.

One :class:`InterruptTracker` instance per invocation is the bookkeeping:
``begin_interrupt_tracking`` builds a fresh tracker at command start (the
command is the process boundary — one ``meeting``/``memo`` invocation at a
time), the run's own seam (``tracker.set_stage``) names the active stage,
and ``tracker.note_written_files`` tallies the files the run managed to
write. ``handle_interrupt`` builds the line from the tracker's stage and
the display's live task.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vemoizer.progress import ProgressDisplay

__all__ = [
    "InterruptTracker",
    "begin_interrupt_tracking",
    "handle_interrupt",
    "note_written_files",
    "set_interrupt_stage",
]

#: Matches a live decode task's batch prefix (``[1/3] stem · decode``);
#: the interrupt line strips it so the stage name never carries the file
#: stem (the "stage + files-written state" contract).
_BATCH_PREFIX_RE = re.compile(r"^\[\d+/\d+\][^·]*· ")


class InterruptTracker:
    """The per-invocation Ctrl-C stage bookkeeping (issue #148).

    One instance per ``meeting``/``memo`` call: the command creates it via
    :func:`begin_interrupt_tracking` and keeps it on the stack, so a second
    invocation in the same process (a test harness, a batch wrapper) never
    sees the first's state.
    """

    def __init__(self) -> None:
        self._stage: str | None = None
        self._written_count = 0

    def set_stage(self, stage: str) -> None:
        """Name the stage the run is entering (called at the stage's seam)."""
        self._stage = stage

    def note_written_files(self, paths: list[Any]) -> None:
        """Tally the files the run wrote this invocation."""
        self._written_count += len(paths)

    @property
    def stage(self) -> str | None:
        """The last stage named by :meth:`set_stage` (or ``None`)."""
        return self._stage

    @property
    def written_count(self) -> int:
        """The number of output files written so far this invocation."""
        return self._written_count


def begin_interrupt_tracking() -> InterruptTracker:
    """Build the Ctrl-C tracker for one ``meeting``/``memo`` invocation.

    Called at the top of the command, before any pipeline stage runs, so
    the first interrupt names the correct stage (the display's current
    task) and the "no files written" claim starts true. The command keeps
    the returned tracker on the stack until the run finishes.
    """
    return InterruptTracker()


def note_written_files(tracker: InterruptTracker | None, paths: list[Any]) -> None:
    """Tally the files *tracker* 's run wrote; the interrupt line then
    reports the true "no files written" / "N files already written"
    state. ``None`` (a run with no tracker) is a no-op."""
    if tracker is not None:
        tracker.note_written_files(paths)


def set_interrupt_stage(tracker: InterruptTracker | None, stage: str) -> None:
    """Name the stage the run is entering (called at the stage's seam).

    The public seam for batch_preset and other modules that need to name
    the active stage without reaching into the tracker's private API.
    ``None`` (a run with no tracker) is a no-op.
    """
    if tracker is not None:
        tracker.set_stage(stage)


def handle_interrupt(
    tracker: InterruptTracker | None, display: ProgressDisplay | None
) -> str:
    """Build the single Ctrl-C line for the current run.

    The stage is the display's active task's *base* stage name (the
    ``[i/N] <stem> · `` batch prefix stripped, so the line never carries a
    file name) when one is running, else the last stage the tracker named,
    else the generic "this run". The written-files claim is state-based,
    not hardcoded: an interrupted run that already wrote earlier files
    says so instead of lying.
    """
    stage = (
        _active_display_stage(display)
        or (tracker.stage if tracker is not None else None)
        or "this run"
    )
    count = tracker.written_count if tracker is not None else 0
    if count == 0:
        files_part = "no files written"
    else:
        files_part = f"{count} file(s) already written"
    return f"interrupted during {stage} — {files_part}"


def _active_display_stage(display: ProgressDisplay | None) -> str | None:
    """The display's active task's base stage name (prefix stripped), or None.

    The stage name is read off the display's own base-name registry (set
    at ``add_stage``, prefix-free), so the batch prefix's file stem cannot
    leak into the line; a live task the registry does not know falls back
    to its live description with the ``[i/N] <stem> · `` prefix stripped.
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
    if task.description.startswith("[green]"):
        # A completed task carries the completion marker; it is not the
        # active stage (the ``prefix_active_stage`` convention).
        return None
    # The base name per task id (set at add_stage); the prefix never
    # reaches this map, so the stem cannot leak.
    name = display._stage_names.get(task.id, "")  # noqa: SLF001
    if name and not name.startswith("[green]"):
        return name
    return _strip_batch_prefix(task.description)


def _strip_batch_prefix(description: str) -> str | None:
    """Strip a ``[i/N] <stem> · `` batch prefix, if present."""
    if not description or description.startswith("[green]"):
        return None
    return _BATCH_PREFIX_RE.sub("", description)
