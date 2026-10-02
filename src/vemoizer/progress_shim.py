"""mlx-whisper tqdm shim: the frame counter drives a rich display task.

``mlx_whisper.transcribe`` counts its loop in mel frames via a module-level
``tqdm``. The shim patches that attribute for the duration of a decode so
the frame counter drives a :class:`~vemoizer.progress.ProgressDisplay`
task in *file-level minutes* (``decode 38/56 min``) rather than raw
frames; when the display is disabled (non-TTY) the shim is a pure
pass-through and the module is never touched.

**Per-window protocol (no audio-length heuristics).** The window loop
(``WhisperTranscriber.transcribe``) calls :meth:`WhisperProgress.mark_window`
before each ``mlx_whisper.transcribe`` call, so the shim knows *which*
decode belongs to a main-loop window:

- the FIRST bar created after a mark (and before the next mark) is the
  window's decode and drives the display at that window's file offset;
- any further bar created before the next mark is a re-entrant call
  (nested decode) and gets a :class:`_NoopBar` — it cannot push the
  display backward (a 10 s tail window is a window, not a re-decode);
- if the loop never marks (unexpected), the first bar still drives the
  display and further bars are no-ops.

Self-heal re-decodes run AFTER the shim exits (they use the unpatched
module), so they never see the shim at all.

:func:`with_whisper_progress` yields a :class:`WhisperProgress` whose
``mark_window`` method the window loop drives. It wraps ONE decode stage
(the whole window loop: one patch and restore) and keeps the per-window
offset in local state, so the display advances monotonically in
file-level minutes across windows.

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
from contextlib import suppress
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


def with_whisper_progress(
    display: ProgressDisplay,
    *,
    window_seconds: float = 30.0,
    file_total_minutes: float | None = None,
) -> WhisperProgress:
    """Wrap the window loop so each window's tqdm drives *display*.

    Patches the ``tqdm`` referenced by ``mlx_whisper.transcribe`` (a
    module-level attribute) and restores it when the caller invokes
    :meth:`WhisperProgress.close` (or when the display is finished). The
    returned :class:`WhisperProgress` supports the context-manager protocol
    so the window loop can use ``with with_whisper_progress(...) as shim:``
    and the module attribute is automatically restored on exit.

    The window loop calls ``shim.mark_window(offset_seconds)`` before every
    main-loop ``mlx_whisper.transcribe`` call (see the per-window protocol
    in the module docstring).

    The shim never raises: on any internal error it degrades to a no-op bar
    (or an unpatched decode) so the decode proceeds identically with or
    without a TTY. When the display is disabled (non-TTY), it is a pure
    pass-through and the module is never touched at all.
    """
    if display.disable:
        return WhisperProgress()

    file_total_minutes = (
        window_seconds / 60.0 if file_total_minutes is None else file_total_minutes
    )

    # The package __init__ re-exports the `transcribe` FUNCTION as
    # `mlx_whisper.transcribe`, shadowing the submodule of the same name.
    # `import mlx_whisper.transcribe as tr_mod` binds to the function
    # (which has no `tqdm` attribute); the MODULE that holds the
    # module-level `tqdm` is only reachable via importlib.
    #
    # This check happens BEFORE add_stage: if there is nothing to patch,
    # no display task is created (nothing to strand).
    tr_mod = importlib.import_module("mlx_whisper.transcribe")
    original_tqdm_module = getattr(tr_mod, "tqdm", None)
    if original_tqdm_module is None or not hasattr(original_tqdm_module, "tqdm"):
        # The contract test pins the module-level tqdm attribute; if an
        # upgrade removes it the shim has nothing to patch. Fail open:
        # decode without the shim, display task never created.
        return WhisperProgress()

    try:
        task_id = display.add_stage("decode", total=file_total_minutes)
    except Exception:  # noqa: BLE001 - shim must never break a decode
        # A rich failure (e.g. a broken console mid-decode) must not
        # propagate: degrade to an unpatched decode, identical result.
        logger.debug("shim: add_stage failed; degrading", exc_info=True)
        return WhisperProgress()

    shim = _ShimmedBar(display, task_id, window_seconds, file_total_minutes)
    shim._set_real_module(original_tqdm_module)
    tr_mod_any: Any = tr_mod
    tr_mod_any.tqdm = shim
    progress = WhisperProgress(shim)
    progress._tr_mod = tr_mod_any
    progress._original_tqdm = original_tqdm_module
    progress._task_id = task_id
    progress._file_total_minutes = file_total_minutes
    return progress


class WhisperProgress:
    """The object :func:`with_whisper_progress` returns.

    ``mark_window`` is the per-window protocol: the window loop calls it
    before each main-loop ``mlx_whisper.transcribe`` call. When the shim
    is active it arms the next window's bar; when the shim was skipped
    (disabled display, unpatchable module) it is a no-op.

    Supports the context-manager protocol: entering the context does
    nothing (the module is already patched), exiting restores the original
    tqdm module attribute and finishes the display task.
    """

    def __init__(self, shim: _ShimmedBar | None = None) -> None:
        self._shim = shim
        self._tr_mod: Any = None
        self._original_tqdm: Any = None
        self._task_id: TaskID | None = None
        self._file_total_minutes: float = 0.0

    def mark_window(self, offset_seconds: float) -> None:
        shim = self._shim
        if shim is None:
            return
        with suppress(Exception):
            shim.mark_window(offset_seconds)

    def __enter__(self) -> WhisperProgress:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        tr_mod = self._tr_mod
        original = self._original_tqdm
        if tr_mod is not None and original is not None:
            with suppress(Exception):
                tr_mod.tqdm = original
        task_id = self._task_id
        if task_id is not None and self._shim is not None:
            with suppress(Exception):
                self._shim._display.finish(task_id, self._file_total_minutes)


class _ShimmedBar:
    """A stand-in for the tqdm *module* whose ``tqdm(...)`` drives a display."""

    def __init__(
        self,
        display: ProgressDisplay,
        task_id: TaskID,
        window_seconds: float,
        file_total_minutes: float,
    ) -> None:
        self._display = display
        self._task_id = task_id
        self._window_seconds = window_seconds
        self._file_total_minutes = file_total_minutes
        # The real tqdm module, captured before the patch, so attribute
        # forwarding (tqdm.auto, ...) stays a drop-in without re-importing.
        self._real_module: Any = None
        # Per-window protocol state: window_index counts marked windows
        # (fallback offset when the loop never marks); window_armed is
        # True until the marked window's bar is created.
        self._window_index = 0
        self._window_armed = True
        # The shim itself plays the factory (mlx_whisper calls
        # ``tqdm.tqdm(...)``).
        self.tqdm = self._make_bar

    def _set_real_module(self, module: Any) -> None:
        self._real_module = module

    def mark_window(self, offset_seconds: float) -> None:
        """The window loop declares the start of the next main-loop window."""
        self._window_index += 1
        self._window_armed = True

    def __getattr__(self, name: str) -> Any:
        real_module = self._real_module
        if real_module is None:
            raise AttributeError(name)
        if hasattr(real_module, name):
            return getattr(real_module, name)
        raise AttributeError(name)

    def _make_bar(self, *args: Any, **kwargs: Any) -> Any:
        # Per-window protocol (see module docstring): the FIRST bar
        # created after a mark (or before the first mark) drives the
        # display; any further bar before the next mark is a re-entrant
        # call and gets a no-op so it cannot push the display backward.
        # We do NOT hand back the real tqdm (it would render a real bar
        # on stderr).
        if self._window_armed:
            self._window_armed = False
            return _ShimmedProgress(
                self._display,
                self._task_id,
                float(self._window_index) * self._window_seconds,
                self._window_seconds,
                self._file_total_minutes,
                kwargs,
            )
        return _NoopBar(kwargs.get("total"))


class _NoopBar:
    """A no-op tqdm bar for re-entrant calls while the shim is active.

    A re-entrant short call (a nested ``mlx_whisper.transcribe`` inside a
    window) gets this bar instead of a driving one: it does nothing, so
    the inner decode finishes silently and the window's bar keeps driving
    the display. ``n``/``total`` remember the values a real bar would
    expose so a library reading them behaves sensibly.
    """

    def __init__(self, total: float | None = None) -> None:
        self.n = 0.0
        self.total = float(total) if total is not None else None

    def __enter__(self) -> _NoopBar:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        pass

    def update(self, n: Any = 1, *a: Any, **kw: Any) -> None:
        with suppress(Exception):
            self.n = float(n)

    def __getattr__(self, name: str) -> Any:
        # n and total are real attributes (initialised in __init__); every
        # other unknown attribute is a no-op callable so
        # pbar.close()/refresh()/set_description(...) never raises.
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
        # the next window offset (the window loop marks it first).
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
