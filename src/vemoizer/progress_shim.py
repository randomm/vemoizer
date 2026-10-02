"""mlx-whisper tqdm shim: the frame counter drives a rich display task.

``mlx_whisper.transcribe`` counts its loop in mel frames via a module-level
``tqdm``. The shim patches that attribute for the duration of a decode so
the frame counter drives a :class:`~vemoizer.progress.ProgressDisplay`
task in *file-level minutes* (``decode 38/56 min``) rather than raw
frames; when the display is disabled (non-TTY) the shim is a pure
pass-through and the module is never touched.

:func:`with_whisper_progress` wraps ONE ``mlx_whisper.transcribe`` call —
in production the whole window loop (one decode stage, one patch and
restore) — and keeps the per-window offset in local state, so the display
advances monotonically in file-level minutes across windows.

The shim never raises: on any internal error it degrades to a no-op,
tqdm-compatible bar (or an unpatched decode) so the decode result is
identical with or without a TTY. The original attribute is always
restored in a ``finally``.

The real tqdm bar is never constructed on the happy path: mlx-whisper only
calls ``pbar.update(n)`` on it (pinned by the contract test), so progress
is computed from the frame counts the shim receives and the real bar is
skipped entirely — a real ``tqdm(total=..., disable=False)`` bar would
otherwise render its own stderr bar next to the rich display.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import contextmanager, suppress
from typing import Any

from rich.progress import TaskID

from vemoizer.progress import ProgressDisplay, frames_to_minutes

logger = logging.getLogger(__name__)


def _frames_to_window_minutes(frames: float, total_frames: float) -> float:
    """Convert *frames* of a *total_frames*-frame window to minutes of audio.

    Reads ``HOP_LENGTH``/``SAMPLE_RATE`` from the installed
    ``mlx_whisper.audio`` lazily and fails open to the 0.4.3 values the
    contract test pins, so a shim-internal error can never break a decode.
    (The ratio is independent of both constants, which is why the fallback
    still matches.)
    """
    try:
        from mlx_whisper.audio import HOP_LENGTH, SAMPLE_RATE

        return frames_to_minutes(frames, hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - shim must never break a decode
        return frames_to_minutes(frames)


@contextmanager
def with_whisper_progress(
    display: ProgressDisplay,
    *,
    window_seconds: float = 30.0,
    file_total_minutes: float | None = None,
) -> Any:
    """Wrap an ``mlx_whisper.transcribe`` call so its tqdm drives *display*.

    Patches the ``tqdm`` referenced by ``mlx_whisper.transcribe`` (a
    module-level attribute) for the duration of the ``with`` body and
    restores it in a ``finally``, so a decode that raises (or returns)
    leaves the module importable and untouched for the next call.

    In production the ``with`` body holds the whole 30 s window loop: one
    decode stage, one module patch/restore. Each window's bar reports a
    fresh total (its own content frames) and resets ``n`` at the end of
    the with-body; the shim maps every update to file-level minutes
    (window offset + in-window progress), so the display's completed
    value is monotonic across windows and the status line reads ``decode
    38/56 min`` rather than restarting at zero per window.

    The shim never raises: on any internal error it degrades to a no-op bar
    (or an unpatched decode) so the decode proceeds identically with or
    without a TTY. When the display is disabled (non-TTY), it is a pure
    pass-through and the module is never touched at all.
    """
    if display.disable:
        # Non-TTY / verbose=False: nothing to drive and the module must not
        # be touched (the decode result is identical in this case).
        yield
        return

    file_total_minutes = (
        window_seconds / 60.0 if file_total_minutes is None else file_total_minutes
    )
    try:
        task_id = display.add_stage("decode", total=file_total_minutes)
    except Exception:  # noqa: BLE001 - shim must never break a decode
        # A rich failure (e.g. a broken console mid-decode) must not
        # propagate: degrade to an unpatched decode, identical result.
        logger.debug("shim: add_stage failed; degrading", exc_info=True)
        yield
        return

    # The package __init__ re-exports the `transcribe` FUNCTION as
    # `mlx_whisper.transcribe`, shadowing the submodule of the same name.
    # `import mlx_whisper.transcribe as tr_mod` binds to the function
    # (which has no `tqdm` attribute); the MODULE that holds the
    # module-level `tqdm` is only reachable via sys.modules / importlib.
    tr_mod = importlib.import_module("mlx_whisper.transcribe")
    original_tqdm_module = getattr(tr_mod, "tqdm", None)
    if original_tqdm_module is None or not hasattr(original_tqdm_module, "tqdm"):
        # The contract test pins the module-level tqdm attribute; if an
        # upgrade removes it the shim has nothing to patch. Fail open:
        # decode without the shim (the display task is finished below).
        yield
        return

    # Per-shim state (not module-global): the factory advances the window
    # index here so the display's completed value is monotonic across the
    # window loop (one bar per full-window decode).
    state = {"window_index": 0}
    shim = _ShimmedBar(display, task_id, window_seconds, file_total_minutes, state)
    shim._set_real_module(original_tqdm_module)
    tr_mod_any: Any = tr_mod
    try:
        tr_mod_any.tqdm = shim
        yield
    finally:
        # Restore the original module attribute no matter how the body
        # exited (success, exception, or an error in the patch itself).
        with suppress(Exception):
            tr_mod_any.tqdm = original_tqdm_module
        # Close the display task so it is never stranded on the status
        # line (e.g. when the meeting->dictation fallback takes over
        # after this).
        with suppress(Exception):
            display.finish(task_id, file_total_minutes)


class _ShimmedBar:
    """A stand-in for the tqdm *module* whose ``tqdm(...)`` drives a display."""

    def __init__(
        self,
        display: ProgressDisplay,
        task_id: TaskID,
        window_seconds: float,
        file_total_minutes: float,
        state: dict[str, int],
    ) -> None:
        self._display = display
        self._task_id = task_id
        self._window_seconds = window_seconds
        self._file_total_minutes = file_total_minutes
        self._state = state
        # The real tqdm module, captured before the patch, so attribute
        # forwarding (tqdm.auto, ...) stays a drop-in without re-importing.
        self._real_module: Any = None
        # The shim itself plays the factory (mlx_whisper calls
        # ``tqdm.tqdm(...)``).
        self.tqdm = self._make_bar

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
        # A full window-length decode (audio near window_seconds) drives
        # the display; a short re-decode (self-heal, decision 1: outside
        # the shim) gets a no-op bar so it finishes silently. We do NOT
        # hand back the real tqdm (it would render a real bar on stderr).
        # The window index advances only for full-window decodes (one bar
        # per window), so the display's completed value is monotonic.
        audio_seconds = self._audio_seconds_of(args)
        if audio_seconds is not None and audio_seconds < self._window_seconds:
            # Short re-decode (self-heal): no-op, does not advance the
            # window index (the outer window's bar keeps driving).
            return _NoopBar()
        state = self._state
        window_index = state["window_index"]
        state["window_index"] = window_index + 1
        return _ShimmedProgress(
            self._display,
            self._task_id,
            float(window_index) * self._window_seconds,
            self._window_seconds,
            self._file_total_minutes,
            kwargs,
        )

    @staticmethod
    def _audio_seconds_of(args: tuple[Any, ...]) -> float | None:
        """Best-effort audio length in seconds from the factory args.

        mlx_whisper's transcribe() takes the audio as its first positional
        arg (a 16 kHz mono float32 numpy array). Returns None when the
        arg is not array-like (then the caller treats it as a full-window
        decode and drives the display).
        """
        if not args:
            return None
        try:
            return float(len(args[0])) / 16_000.0
        except Exception:  # noqa: BLE001 - shim must never break a decode
            return None


class _NoopBar:
    """A no-op tqdm bar for re-entrant (self-heal) calls.

    While the shim is active for the long whole-file decode, a re-entrant
    short call (self-heal's re-decode lambdas) gets this bar instead of
    the real tqdm: it does nothing, so the inner decode finishes silently
    and the outer window's bar keeps driving the display.
    """

    def __enter__(self) -> _NoopBar:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        # The numeric attributes mlx_whisper may read (n, total) must stay
        # numeric; everything else is a no-op callable so
        # pbar.close()/refresh()/set_description(...) never raises.
        if name in ("n", "total"):
            return 0.0
        return lambda *a: None


class _ShimmedProgress:
    """Drop-in for one ``tqdm`` bar instance that drives a rich task.

    Converts the raw frame counter to file-level minutes and forwards each
    ``update()`` to ``display._progress.update(task_id, completed=...)``.
    The real tqdm bar is deliberately NOT constructed on the happy path:
    mlx-whisper only calls ``pbar.update(n)`` on it (contract test), and a
    real ``tqdm(total=..., disable=False)`` bar would render its own bar on
    stderr next to the display. The frame counts the bar receives are
    tracked locally (``self.n`` / ``self.total``) as the single source of
    truth for the minutes conversion.
    """

    def __init__(
        self,
        display: ProgressDisplay,
        task_id: TaskID,
        window_offset_seconds: float,
        window_seconds: float,
        file_total_minutes: float,
        kwargs: dict[str, Any],
    ) -> None:
        self._display = display
        self._task_id = task_id
        self._window_offset = window_offset_seconds
        self._window_seconds = window_seconds
        self._file_total_minutes = file_total_minutes
        self.n: float = 0.0
        # The window's total frame count from the factory call; None until
        # __enter__ has seen the real total (and only then the conversion
        # is meaningful).
        self.total: float | None = None
        self._pending_total: float | None = kwargs.get("total")

    def __enter__(self) -> _ShimmedProgress:
        # mlx_whisper passes total=content_frames to the factory; record it
        # here (a real bar would do the same at construction time).
        if self.total is None and self._pending_total is not None:
            try:
                self.total = float(self._pending_total)
            except Exception:  # noqa: BLE001 - shim must never break a decode
                self.total = None
        return self

    def __exit__(self, *exc_info: Any) -> None:
        # A fresh window starts at n=0; the next bar's updates will map to
        # the next window offset (the factory advances the window index).
        self.n = 0.0

    def update(self, n: Any = 1, *a: Any, **kw: Any) -> None:
        try:
            self.n = float(n)
        except Exception:  # noqa: BLE001 - shim must never break a decode
            return
        total = self.total
        if total is None or total <= 0:
            return
        with suppress(Exception):
            minutes_cur = _frames_to_window_minutes(self.n, total)
            file_cur = self._window_offset / 60.0 + minutes_cur
            completed = min(
                max(0.0, file_cur),
                self._file_total_minutes if self._file_total_minutes else file_cur,
            )
            self._display._progress.update(self._task_id, completed=completed)

    def __getattr__(self, name: str) -> Any:
        # n and total are real attributes (initialised in __init__); every
        # other unknown attribute is a no-op callable so pbar.close()/
        # refresh()/set_description(...) never raises from the shim.
        return lambda *a: None
