"""Per-stage progress reporting: a rich stderr display and a logging heartbeat.

Two surfaces, one responsibility. :class:`ProgressDisplay` renders to a TTY;
:class:`StageProgress` emits throttled INFO logs and is what a piped or
redirected run sees, where the rich display is disabled.

Contract (issue #10): progress goes to **stderr** via ``rich.progress``
(one task per pipeline stage); the transcript goes to stdout. When
``sys.stderr`` is not a TTY — i.e. the user piped or redirected the output
— the display is disabled so progress spam never pollutes captured stderr.
``verbose=False`` forces the same off-switch regardless of TTY state.

The mlx-whisper progress shim (issue #105) lives in
:mod:`vemoizer.progress_shim`: a context manager that patches the ``tqdm``
referenced by ``mlx_whisper.transcribe`` for the duration of a decode so
its internal frame counter drives a :class:`ProgressDisplay` task in
*minutes* (``decode 38/56 min``) rather than raw frames.

This module only owns the reporting. The pipeline stages own *when* to
advance; CLI wiring that constructs the display belongs to the CLI task.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import IO, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

logger = logging.getLogger(__name__)


#: Minimum seconds between two per-item progress lines. The decode stages run
#: one model call per VAD slice (1000+ slices on an hour-long memo); logging
#: every slice would bury the stage lines, and logging none at all makes a
#: 20-minute stage indistinguishable from a hang.
PROGRESS_INTERVAL_S = 5.0


def format_duration(seconds: float) -> str:
    """Render *seconds* as ``1h02m``/``3m39s``/``9.7s`` for log lines."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def frames_to_minutes(
    frames: float, *, hop_length: int = 160, sample_rate: int = 16_000
) -> float:
    """Convert mel-spectrogram *frames* to minutes of audio.

    ``mlx_whisper``'s progress bar counts frames (one frame = ``HOP_LENGTH``
    samples), not seconds — see ``mlx_whisper.audio.HOP_LENGTH`` /
    ``SAMPLE_RATE``. The progress shim (``vemoizer.progress_shim``) applies
    this conversion before handing the counter to a :class:`ProgressDisplay`.
    Defaults match the installed 0.4.3 values (``HOP_LENGTH=160``,
    ``SAMPLE_RATE=16000``) so the function is usable without the dependency
    imported (the test suite mocks ``mlx_whisper`` throughout).
    """
    return frames * hop_length / sample_rate / 60.0


class StageProgress:
    """Throttled INFO progress for a stage that loops over many items.

    The decode and re-decode stages iterate over hundreds or thousands of
    items, each one model call, and previously emitted nothing between the
    stage's first and last line — so a slow stage looked exactly like a
    deadlock. This logs a heartbeat at most every ``PROGRESS_INTERVAL_S``
    seconds with a completion count, throughput and ETA, then one summary
    line on :meth:`done`.

    Progress reporting is never allowed to break a decode: the caller drives
    it from inside the loop it is measuring, so every method here is pure
    arithmetic and logging.
    """

    def __init__(
        self,
        label: str,
        total: int,
        audio_seconds: float = 0.0,
        unit: str = "slices",
    ) -> None:
        self.label = label
        self.total = total
        self.audio_seconds = audio_seconds
        self.unit = unit
        self.done_count = 0
        self.failed = 0
        self._start = time.monotonic()
        self._last_log = self._start
        detail = (
            f" ({format_duration(audio_seconds)} of audio)" if audio_seconds > 0 else ""
        )
        logger.info("%s: starting over %d %s%s", label, total, unit, detail)

    def advance(self, *, failed: bool = False) -> None:
        """Count one finished item and log a heartbeat if one is due."""
        self.done_count += 1
        if failed:
            self.failed += 1
        now = time.monotonic()
        if now - self._last_log < PROGRESS_INTERVAL_S:
            return
        self._last_log = now
        elapsed = now - self._start
        rate = self.done_count / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done_count) / rate if rate > 0 else 0.0
        logger.info(
            "%s: %d/%d %s (%.0f%%) %.1f/s elapsed %s eta %s%s",
            self.label,
            self.done_count,
            self.total,
            self.unit,
            100.0 * self.done_count / self.total if self.total else 100.0,
            rate,
            format_duration(elapsed),
            format_duration(remaining),
            f" ({self.failed} failed)" if self.failed else "",
        )

    def done(self) -> float:
        """Log the stage summary; return the stage's elapsed seconds."""
        elapsed = time.monotonic() - self._start
        speed = (
            f", {self.audio_seconds / elapsed:.1f}x realtime"
            if self.audio_seconds > 0 and elapsed > 0
            else ""
        )
        logger.info(
            "%s: finished %d/%d %s in %s%s%s",
            self.label,
            self.done_count - self.failed,
            self.total,
            self.unit,
            format_duration(elapsed),
            speed,
            f" ({self.failed} failed)" if self.failed else "",
        )
        return elapsed


#: Columns shown while a stage runs. The elapsed time keeps the display
#: useful for the slow model-load waits without implying a false total.
_COLUMNS: tuple[Any, ...] = (
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    TextColumn("({task.elapsed:.0f}s)"),
)


class ProgressDisplay:
    """Per-stage ``rich`` progress bound to stderr with TTY auto-detection.

    Usage (one task per pipeline stage)::

        display = ProgressDisplay()  # verbose defaults to True
        task_id = display.add_stage("decode A")
        display.advance(task_id, 10)   # e.g. chunks processed
        display.finish(task_id, 100)

    The constructor reads ``sys.stderr.isatty()`` once; when it is not a
    TTY (piped/redirected) or ``verbose`` is false, the underlying
    ``Progress`` is created with ``disable=True`` and every method becomes
    a no-op on stderr.
    """

    def __init__(self, verbose: bool = True) -> None:
        is_tty: bool = sys.stderr.isatty()
        self.disable: bool = not (verbose and is_tty)
        self._console = Console(
            stderr=True,
            no_color=not is_tty,
            file=_stderr_file(),
        )
        self._progress = Progress(
            *_COLUMNS,
            console=self._console,
            disable=self.disable,
            transient=False,
        )
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start rendering. Idempotent; a disabled progress is a no-op."""
        if not self._started:
            self._progress.start()
            self._started = True

    def close(self) -> None:
        """Stop rendering and release the display. Idempotent."""
        if self._started:
            self._progress.stop()
            self._started = False

    def __enter__(self) -> ProgressDisplay:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- per-stage task management ------------------------------------------

    def add_stage(self, description: str, total: float | None = None) -> TaskID:
        """Register a pipeline stage and return its task id."""
        self.start()
        return self._progress.add_task(description, total=total)

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        """Advance a stage's progress counter."""
        self._progress.advance(task_id, advance)

    def finish(self, task_id: TaskID, total: float | None = None) -> None:
        """Mark a stage complete (optional explicit total to end on)."""
        self._progress.update(task_id, completed=total)
        self._progress.update(task_id, description="[green]✓ complete")
        self._progress.stop_task(task_id)

    def update_text(self, task_id: TaskID, description: str) -> None:
        """Replace a running stage's status text (e.g. 'loading model...')."""
        self._progress.update(task_id, description=description)

    def prefix_active_stage(self, prefix: str) -> None:
        """Prepend *prefix* (e.g. ``"[1/3] memo · "``) to the active stage.

        Used by the batch layer (issue #105 M4b) to show ``[i/N] <stem> ·
        decode 38/56 min`` for multi-file runs.  The prefix is part of the
        description, not a separate echo line.  A re-prefix with a new
        value *replaces* any previous ``[i/N] stem · `` batch prefix (no
        accumulation), and a completed task is never re-prefixed.  A no-op
        when no stage is active or when the display is disabled.
        """
        if self.disable:
            return
        task = self._progress.tasks[-1] if self._progress.tasks else None
        if task is None:
            return
        # A completed task has its description replaced by the completion
        # marker; skip it so a re-prefix never resurrects a finished stage.
        if task.description.startswith("[green]"):
            return
        description = task.description
        if description.startswith(prefix):
            return
        # A previous batch prefix is a run of `[i/N] stem · ` (the marker
        # sits at the END of the prefix string). Find its closing `]` and
        # check the marker right after it, so a re-prefix replaces the old
        # prefix instead of accumulating ('[2/3] b · [1/3] a · ...').
        # A previous batch prefix is `[i/N] stem · ` (the `·` marker sits
        # inside the prefix string).  Find that marker and strip the old
        # prefix so a re-prefix replaces it instead of accumulating.
        if description.startswith("["):
            marker = " · "
            idx = description.find(marker)
            if idx != -1:
                description = description[idx + len(marker) :]
        self._progress.update(task.id, description=f"{prefix}{description}")


def _stderr_file() -> IO[str]:
    """Current sys.stderr at call time (tests monkeypatch it)."""
    return sys.stderr
