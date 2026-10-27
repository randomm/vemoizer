"""Live call draft for ``vemoizer listen`` (issue #179).

The draft is a single-decode preview of a call, never the deliverable: the
post-call pass always runs the full ``meeting`` pipeline. Lines are appended
as turns arrive; the draft is removed only once the final pair is written,
so an interrupted run keeps whatever was captured.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, TextIO

from vemoizer.preset_interrupt import InterruptTracker, note_written_files
from vemoizer.slice_align import slice_similarity
from vemoizer.textnorm import textnorm
from vemoizer.tracks import REMOTE_LABEL

Track = Literal["me", "them"]

#: A mic line and a remote line this close in time, with near-identical text,
#: is the remote voice leaking acoustically into the owner's microphone.
BLEED_WINDOW_S = 3.0

#: Char-level similarity at or above which two lines count as the same words.
BLEED_THRESHOLD = 0.8

#: ``(start_seconds, text)`` for one line of one track.
Line = tuple[float, str]


def _clock(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_line(start_s: float, label: str, text: str) -> str:
    """One draft line: ``[hh:mm:ss] Label: text``."""
    return f"[{_clock(start_s)}] {label}: {text}"


class LiveWriter:
    """Appends turn lines to the stream given at construction.

    The stream is captured here, never looked up from ``sys.stdout`` at write
    time, so a scoped stdout wrapper (``filter_language_lines``) cannot hold
    these lines back.
    """

    def __init__(self, out: TextIO, owner: str) -> None:
        self._out = out
        self._owner = owner

    def write_turn(self, start_s: float, track: Track, text: str) -> None:
        """Write one turn line; ``me`` is the owner, ``them`` the remote side."""
        label = self._owner if track == "me" else REMOTE_LABEL
        self._out.write(render_line(start_s, label, text) + "\n")
        self._out.flush()


def _is_bleed(mic: Line, them: Line, window_s: float, threshold: float) -> bool:
    (mic_start, mic_text), (them_start, them_text) = mic, them
    # Empty after normalization (punctuation-only, e.g. "..."), not just
    # whitespace-empty: slice_similarity scores two such texts as 1.0.
    if not textnorm(mic_text) or not textnorm(them_text):
        return False
    if abs(mic_start - them_start) > window_s:
        return False
    return slice_similarity(mic_text, them_text) >= threshold


def headphone_bleed(
    mic_lines: Sequence[Line],
    them_lines: Sequence[Line],
    window_s: float = BLEED_WINDOW_S,
    threshold: float = BLEED_THRESHOLD,
) -> bool:
    """True when any mic line echoes a nearby remote line (speakers, no headphones)."""
    return any(
        _is_bleed(mic, them, window_s, threshold)
        for mic in mic_lines
        for them in them_lines
    )


class BleedWarner:
    """Warns once, on the injected stream, the first time bleed is seen."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._warned = False

    def check(self, mic_lines: Sequence[Line], them_lines: Sequence[Line]) -> bool:
        """Return True only on the call that newly detects bleed."""
        if self._warned or not headphone_bleed(mic_lines, them_lines):
            return False
        self._warned = True
        self._out.write(
            "warning: possible headphone bleed (the mic is picking up the call); "
            "wear headphones for a clean Me track\n"
        )
        self._out.flush()
        return True


@contextlib.contextmanager
def live_draft(
    path: Path,
    *,
    owner: str,
    keep: bool,
    tracker: InterruptTracker | None = None,
) -> Iterator[LiveWriter]:
    """Append turns to *path* for the block; remove it on a clean exit.

    Run the final pair inside the block: the draft is deleted only when the
    block exits normally (unless *keep*), so Ctrl-C during capture or the
    final pass leaves the draft on disk.
    """
    with path.open("a", encoding="utf-8") as handle:
        note_written_files(tracker, [path])
        yield LiveWriter(handle, owner=owner)
    if not keep:
        path.unlink(missing_ok=True)
