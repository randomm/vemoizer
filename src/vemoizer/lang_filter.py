"""Scoped stdout filter for mlx-whisper's per-window 'Detected language' line.

Issue #147: mlx-whisper 0.4.3 prints ``Detected language: X`` to stdout once
per decode window (one ``print()`` per window, ~30 identical lines on a
15-minute memo), scrolling the rich progress display away.

``verbose=None`` suppresses this line but also disables the tqdm bar that
the :mod:`vemoizer.progress_shim` intercepts to drive the ``ProgressDisplay``.
``verbose=False`` enables the bar (so the shim works) but does NOT suppress
the line (``if verbose is not None:`` is True for ``False``).

This module provides :class:`filter_language_lines`, a context manager that
replaces ``sys.stdout`` with a line-filtering wrapper for the duration of a
single ``mlx_whisper.transcribe`` call. It:

- matches only the exact ``Detected language: X`` line pattern (one line,
  no trailing content) and drops it;
- passes every other line through unchanged;
- restores ``sys.stdout`` on every exit path (normal, exception,
  ``KeyboardInterrupt``);
- writes to stderr and interactive prompts are unaffected (the wrapper only
  wraps stdout);
- is thread-safe in the sense that it restores the original ``sys.stdout``
  reference, not a per-thread value (the decode loop is single-threaded).
"""

from __future__ import annotations

import contextlib
import re
import sys
from collections.abc import Generator
from typing import Any

__all__ = ["filter_language_lines"]

#: Matches the exact line mlx-whisper 0.4.3 prints: "Detected language: Finnish"
#: (one space after the colon, one word, no trailing content). The regex is
#: intentionally narrow: it must not match "Detected languages:" (plural) or
#: any other library output.
_LANGUAGE_LINE_RE = re.compile(r"^Detected language: \S+$")


class _FilteredStdout:
    """A line-buffered stdout wrapper that drops matching lines.

    Writes are buffered until a newline; a line that matches
    :data:`_LANGUAGE_LINE_RE` is dropped, everything else is written to the
    original stdout. The wrapper is transparent to all other ``io.TextIOBase``
    operations (``fileno``, ``encoding``, etc.).
    """

    def __init__(self, original: Any) -> None:
        self._original = original
        self._buffer = ""

    def write(self, data: str) -> int:
        self._buffer += data
        # Process complete lines; keep the trailing partial line in the buffer.
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not _LANGUAGE_LINE_RE.match(line):
                self._original.write(line + "\n")
        return len(data)

    def flush(self) -> None:
        # Flush any remaining partial line (shouldn't happen in practice,
        # but covers the case where the last line has no trailing newline).
        if self._buffer:
            if not _LANGUAGE_LINE_RE.match(self._buffer):
                self._original.write(self._buffer)
            self._buffer = ""
        self._original.flush()

    def isatty(self) -> bool:
        return self._original.isatty()

    def fileno(self) -> int:
        return self._original.fileno()

    def __getattr__(self, name: str) -> Any:
        # Forward all other attributes (encoding, errors, etc.) to the
        # original stdout.
        return getattr(self._original, name)

    def close(self) -> None:
        self.flush()
        self._original.close()


@contextlib.contextmanager
def filter_language_lines() -> Generator[None, None, None]:
    """Suppress ``Detected language: X`` lines on stdout for the duration of
    the ``with`` block.

    Replaces ``sys.stdout`` with a :class:`_FilteredStdout` that drops lines
    matching the exact ``Detected language: X`` pattern. Every other line is
    passed through unchanged. The original ``sys.stdout`` is restored on
    every exit path (normal return, exception, ``KeyboardInterrupt``).

    Usage::

        with filter_language_lines():
            raw = mlx_whisper.transcribe(...)

    The rich ``ProgressDisplay`` writes to stderr and is unaffected.
    Interactive prompts (``input()``) read from stdin and are unaffected.
    """
    original_stdout = sys.stdout
    try:
        sys.stdout = _FilteredStdout(original_stdout)
        yield None
    finally:
        # Flush any buffered content before restoring.
        wrapped = sys.stdout
        if isinstance(wrapped, _FilteredStdout):
            wrapped.flush()
        sys.stdout = original_stdout
