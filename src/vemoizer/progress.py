"""Per-stage progress reporting: a rich stderr display and a logging heartbeat.

Two surfaces, one responsibility. :class:`ProgressDisplay` renders to a TTY;
:class:`StageProgress` emits throttled INFO logs and is what a piped or
redirected run sees, where the rich display is disabled.

Contract (issue #10): progress goes to **stderr** via ``rich.progress``
(one task per pipeline stage); the transcript goes to stdout. When
``sys.stderr`` is not a TTY — i.e. the user piped or redirected the output
— the display is disabled so progress spam never pollutes captured stderr.
``verbose=False`` forces the same off-switch regardless of TTY state.

The mlx-whisper progress shim (issue #105) also lives here: a context
manager that patches the ``tqdm`` referenced by ``mlx_whisper.transcribe``
for the duration of a decode so its internal frame counter drives a
:class:`ProgressDisplay` task in *minutes* (``decode 38/56 min``) rather
than raw frames. See :func:`with_whisper_progress`.

This module only owns the reporting. The pipeline stages own *when* to
advance; CLI wiring that constructs the display belongs to the CLI task.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager, suppress
from typing import IO, Any, cast

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
    ``SAMPLE_RATE``. The display wants ``decode 38/56 min``; this is the
    conversion the shim applies before handing the counter to a
    :class:`ProgressDisplay`. Defaults match the installed 0.4.3 values
    (``HOP_LENGTH=160``, ``SAMPLE_RATE=16000``) so the function is usable
    without the dependency imported (the test suite mocks ``mlx_whisper``
    throughout).
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


def _stderr_file() -> IO[str]:
    """Current sys.stderr at call time (tests monkeypatch it)."""
    return sys.stderr


# ---------------------------------------------------------------------------
# mlx-whisper tqdm shim (issue #105)
# ---------------------------------------------------------------------------
#
# mlx_whisper.transcribe counts its loop in mel frames via a module-level
# tqdm: ``with tqdm.tqdm(total=content_frames, unit="frames",
# disable=verbose is not False) as pbar:`` with ``pbar.update(delta)`` per
# decoded 30 s window. Because WhisperTranscriber.transcribe decodes the
# recording in 30 s windows (each its own transcribe() call, so the glossary
# initial_prompt re-seeds), the shim wraps the WHOLE transcribe() call: every
# window's internal bar hits the patched tqdm. The display's decode task has
# a FILE-level total (whole-recording minutes); each window advances it by
# window_offset_minutes + in-window frames converted to minutes.
#
# The patch is idempotent: a module-level sentinel marks the active shim, and
# the patched tqdm checks it, so a re-entrant short call (self-heal's
# re-decode lambdas) gets a bare tqdm instead of a second wrapped bar. The
# original module attribute is always restored in a finally, and the shim
# swallows its own errors — the decode result must be identical with or
# without a TTY.


@contextmanager
def with_whisper_progress(
    display: ProgressDisplay,
    *,
    window_offset_seconds: float = 0.0,
    window_seconds: float = 30.0,
    file_total_minutes: float | None = None,
) -> Any:
    """Wrap an ``mlx_whisper.transcribe`` call so its tqdm drives *display*.

    Patches the ``tqdm`` referenced by ``mlx_whisper.transcribe`` (a
    module-level attribute) for the duration of the ``with`` body and
    restores it in a ``finally``, so a decode that raises (or returns)
    leaves the module importable and untouched for the next window or file.

    The patched bar converts the raw frame counter to file-level minutes
    (``window_offset + in-window frames``) and forwards it to the display's
    decode task, whose total is ``file_total_minutes`` — the ``decode
    38/56 min`` status line.

    The shim never raises: on any internal error it degrades to the real
    tqdm (or a no-op) so the decode proceeds identically with or without a
    TTY. When the display is disabled (non-TTY), it is a pure pass-through
    and the module is never touched at all.
    """
    if display.disable:
        # Non-TTY / verbose=False: nothing to drive and the module must not
        # be touched (the decode result is identical in this case).
        yield
        return

    if file_total_minutes is None:
        file_total_minutes = (window_offset_seconds + window_seconds) / 60.0

    task_id = display.add_stage("decode", total=file_total_minutes)
    try:
        # The package __init__ re-exports the `transcribe` FUNCTION as
        # `mlx_whisper.transcribe`, shadowing the submodule of the same name.
        # `import mlx_whisper.transcribe as tr_mod` binds to the function
        # (which has no `tqdm` attribute); the MODULE that holds the
        # module-level `tqdm` is only reachable via sys.modules / importlib.
        import importlib

        tr_mod = importlib.import_module("mlx_whisper.transcribe")

        original_tqdm_module = getattr(tr_mod, "tqdm", None)
        if original_tqdm_module is None or not hasattr(original_tqdm_module, "tqdm"):
            # The contract test pins the module-level tqdm attribute; if an
            # upgrade removes it the shim has nothing to patch. Fail open:
            # decode without the shim (the display task is finished below).
            yield
            return

        real_factory = original_tqdm_module.tqdm
        shim = _ShimmedBar(real_factory, display, task_id, window_offset_seconds)
        shim._set_real_module(original_tqdm_module)
        previous_active = getattr(_state, "active", False)
        try:
            _state.active = True
            _state.window_seconds = window_seconds
            _state.file_total_minutes = file_total_minutes
            tr_mod.tqdm = cast(Any, shim)
            yield
        finally:
            _state.active = previous_active
            # Restore the original module attribute no matter how the body
            # exited (success, exception, or an error in the patch itself).
            with suppress(Exception):
                tr_mod.tqdm = original_tqdm_module
    finally:
        # Close the display task so it is never stranded on the status line
        # (e.g. when the meeting->dictation fallback takes over after this).
        with suppress(Exception):
            display.finish(task_id, file_total_minutes)


class _ShimmedBar:
    """A stand-in for the tqdm *module* whose ``tqdm(...)`` drives a display."""

    def __init__(
        self,
        real_factory: Any,
        display: ProgressDisplay,
        task_id: TaskID,
        window_offset_seconds: float,
    ) -> None:
        self._real_factory = real_factory
        self._display = display
        self._task_id = task_id
        self._window_offset = window_offset_seconds
        # The real tqdm factory as a callable attribute (mlx_whisper calls
        # ``tqdm.tqdm(...)``).
        self.tqdm = self._make_bar
        # The real tqdm module, captured before the patch, so attribute
        # forwarding (tqdm.auto, ...) stays a drop-in without re-importing.
        self._real_module: Any = None

    def _set_real_module(self, module: Any) -> None:
        self._real_module = module

    def __getattr__(self, name: str) -> Any:
        real_module = self._real_module
        if real_module is None:
            raise AttributeError(name)
        if hasattr(real_module, name):
            return getattr(real_module, name)
        raise AttributeError(name)

    def _make_bar(self, *args: Any, **kwargs: Any) -> Any:
        # Idempotent guard: while the shim is active, a re-entrant (short,
        # self-heal) call gets a no-op bar so it is not double-wrapped.
        # The outer window's bar keeps driving the display; the inner decode
        # finishes silently. We do NOT hand back the real tqdm (it would
        # render a real bar on stderr during the re-entrant call).
        if getattr(_state, "active", False):
            return _NoopBar()
        return _ShimmedProgress(
            self._real_factory,
            self._display,
            self._task_id,
            self._window_offset,
            args,
            kwargs,
        )


class _NoopBar:
    """A no-op tqdm bar for re-entrant (self-heal) calls.

    While the shim is active for the long whole-file decode, a re-entrant
    short call (self-heal's re-decode lambdas) gets this bar instead of the
    real tqdm: it does nothing, so the inner decode finishes silently and
    the outer window's bar keeps driving the display.
    """

    def __enter__(self) -> _NoopBar:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return 0.0


class _ShimmedProgress:
    """Drop-in for one ``tqdm`` bar instance that drives a rich task.

    Converts the raw frame counter to file-level minutes and forwards each
    ``update()`` to ``display._progress.update(task_id, completed=...)``.
    On any internal error it degrades to the real tqdm (or a no-op) so the
    decode result is identical with or without a TTY.
    """

    def __init__(
        self,
        real_factory: Any,
        display: ProgressDisplay,
        task_id: TaskID,
        window_offset_seconds: float,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._real: Any = None
        self._real_failed = False
        try:
            self._real = real_factory(*args, **kwargs)
        except Exception:  # noqa: BLE001 - shim must never break a decode
            logger.debug("shim: real tqdm failed; using no-op bar", exc_info=True)
            self._real_failed = True
        self._display = display
        self._task_id = task_id
        self._window_offset = window_offset_seconds
        self.n: float = 0.0
        self.total: float | None = None

    def __enter__(self) -> _ShimmedProgress:
        if self._real is not None and not self._real_failed:
            try:
                self._real.__enter__()
            except Exception:  # noqa: BLE001
                self._real_failed = True
        self.total = getattr(self._real, "total", None) if self._real else None
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._real is not None and not self._real_failed:
            with suppress(Exception):
                self._real.__exit__(*exc_info)

    def update(self, n: Any = 1, *a: Any, **kw: Any) -> None:
        with suppress(Exception):
            if self._real is not None and not self._real_failed:
                self._real.update(n, *a, **kw)
        total = getattr(self._real, "total", None) if self._real else None
        cur = getattr(self._real, "n", 0) if self._real else 0
        if total is None or total <= 0:
            return
        with suppress(Exception):
            window_minutes = getattr(_state, "window_seconds", 30.0) / 60.0
            minutes_cur = (cur / total) * window_minutes
            file_cur = self._window_offset / 60.0 + minutes_cur
            file_total = getattr(_state, "file_total_minutes", None)
            completed = min(max(0.0, file_cur), file_total if file_total else file_cur)
            self._display._progress.update(self._task_id, completed=completed)

    def __getattr__(self, name: str) -> Any:
        if self._real is not None and not self._real_failed:
            return getattr(self._real, name)
        # No usable real bar: return a harmless value so attribute access
        # (pbar.n, pbar.total, ...) never raises from the shim.
        return 0.0


class _ShimState:
    """Tiny mutable holder so the patched bar can see the active window."""

    active: bool = False
    window_seconds: float = 30.0
    file_total_minutes: float | None = None


_state = _ShimState()
