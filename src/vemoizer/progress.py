"""Per-stage progress display on stderr with TTY auto-detection.

Contract (issue #10): progress goes to **stderr** via ``rich.progress``
(one task per pipeline stage); the transcript goes to stdout. When
``sys.stderr`` is not a TTY — i.e. the user piped or redirected the output
— the display is disabled so progress spam never pollutes captured stderr.
``verbose=False`` forces the same off-switch regardless of TTY state.

This module only owns the display. The pipeline stages own *when* to
advance; CLI wiring that constructs the display belongs to the CLI task.
"""

from __future__ import annotations

import sys
from typing import IO, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn

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


def _stderr_file() -> IO[str]:
    """Current sys.stderr at call time (tests monkeypatch it)."""
    return sys.stderr
