"""M5b: speaker clip selection and playback helpers (issue #90).

Pure selectors over a rendered sidecar's ``paragraphs``/``segments``, plus
the audio helpers that extract the selected windows from the SOURCE files
recorded in the sidecar's ``source`` entries and play them with ``afplay``.

Design notes
------------

* A *turn* is one labelled, non-suspect paragraph that is not a
  backchannel and is at least 2 s long (see :func:`select_clips` rules).
  Because a speaker's contiguous same-speaker paragraphs are the only
  candidate turns, a turn's ``[start, end)`` never crosses a speaker
  boundary by construction.
* ``select_clips`` is pure and never touches audio; the window is the
  middle ``min(max_s, turn_duration)`` seconds of the turn, clamped to the
  audio duration (the end of the last paragraph).
* ``extract_clips`` maps each group-level window onto the source part that
  contains it via the sidecar ``source`` entries' ``part_offset_s`` (and
  optional ``duration_s``), decodes only the needed span with ffmpeg
  ``-ss``/``-t`` placed BEFORE ``-i`` (input seeking, O(window) decode),
  and returns ``None`` — never raises — for a window whose source file is
  missing, unreadable, or straddles a part boundary.
* Nothing is kept on disk: ``clip_session`` yields a fresh 0o700 temp dir
  and removes it on every exit path.

No ffprobe anywhere: durations come from the sidecar's measured
``duration_s`` entries (themselves from ``pcm_duration_seconds``) or from
the decoded windows.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .speaker_align import BACKCHANNEL_PHRASES, BACKCHANNELS

__all__ = [
    "ClipWindow",
    "clip_session",
    "extract_clips",
    "play",
    "select_clips",
    "talk_share",
]

#: A turn shorter than this is not a clip candidate.
MIN_TURN_S = 2.0

#: Quote text is bounded to roughly this many characters.
MAX_QUOTE_CHARS = 120

#: play() wall-clock bound for afplay, in seconds.
PLAY_TIMEOUT_S = 30.0

#: ffmpeg decode bound for a single window, in seconds.
FFMPEG_TIMEOUT_S = 60.0

#: Output sample rate and channel count (the 16 kHz mono audio contract).
_SAMPLE_RATE = "16000"
_CHANNELS = "1"


class ClipWindow(tuple):
    """A selected clip: ``(start_s, end_s, quote)``.

    Subclasses ``tuple`` so a window is a hashable, comparable 3-tuple the
    callers can use as a dict key (``extract_clips`` returns
    ``dict[ClipWindow, Path | None]``) while still exposing named fields.
    """

    __slots__ = ()

    def __new__(cls, start_s: float, end_s: float, quote: str = "") -> ClipWindow:
        return tuple.__new__(cls, (float(start_s), float(end_s), quote))

    @property
    def start_s(self) -> float:
        return self[0]

    @property
    def end_s(self) -> float:
        return self[1]

    @property
    def quote(self) -> str:
        return self[2]


def _labelled(p: dict[str, Any]) -> str | None:
    """The speaker label of *p*, or ``None`` when unlabelled.

    A paragraph is labelled iff ``speaker`` is a non-empty string; a
    missing key, ``None``, or ``""`` all count as unlabelled.
    """
    label = p.get("speaker")
    return label if isinstance(label, str) and label else None


def talk_share(paragraphs: list[dict[str, Any]]) -> dict[str, float]:
    """Per-speaker fraction of labelled talk time.

    Fractions sum to 1 over labelled paragraphs (within float error);
    unlabelled text is excluded. Empty or all-unlabelled input gives {}.
    """
    duration: dict[str, float] = {}
    for p in paragraphs:
        label = _labelled(p)
        if label is None:
            continue
        span = float(p.get("end", 0.0)) - float(p.get("start", 0.0))
        if span > 0:
            duration[label] = duration.get(label, 0.0) + span
    total = sum(duration.values())
    if total <= 0:
        return {}
    return {label: span / total for label, span in duration.items()}


def _is_backchannel(text: str) -> bool:
    """True when the whitespace-collapsed, casefolded turn text is a
    backchannel (membership in ``BACKCHANNELS ∪ BACKCHANNEL_PHRASES`` —
    no word-count condition, so "joo" inside a longer sentence is fine).
    """
    collapsed = " ".join(text.split()).casefold()
    return collapsed in BACKCHANNELS or collapsed in BACKCHANNEL_PHRASES


def _quote(paragraphs: list[dict[str, Any]], start: float, end: float) -> str:
    """Whitespace-collapsed text of the paragraphs overlapping [start, end),
    bounded to about :data:`MAX_QUOTE_CHARS` characters.
    """
    parts = [
        " ".join(str(p.get("text", "")).split())
        for p in paragraphs
        if float(p.get("start", 0.0)) < end and float(p.get("end", 0.0)) > start
    ]
    quote = " ".join(t for t in parts if t)
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[:MAX_QUOTE_CHARS].rstrip() + "…"
    return quote


def _third_for(start: float, audio_end: float) -> int:
    """Which third of the meeting [0, audio_end) *start* falls in (0..2)."""
    if audio_end <= 0:
        return 0
    third = audio_end / 3.0
    return min(int(start // third), 2)


def select_clips(
    paragraphs: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None = None,
    *,
    per_speaker: int = 3,
    min_s: float = 3.0,
    max_s: float = 5.0,
    total_duration: float | None = None,
) -> dict[str, list[ClipWindow]]:
    """Pick up to *per_speaker* short windows per speaker label.

    Rules (see module docstring): candidates are labelled, non-suspect,
    non-backchannel turns of one speaker lasting >= 2 s; the window is the
    middle ``min(max_s, D)`` seconds of the turn (so a turn of 3-5 s yields
    a window the full length of the turn, longer turns 5 s) centered on the
    turn midpoint and clamped to the audio duration. Picks for one speaker
    spread across the beginning, middle and end of the meeting: the longest
    candidate from each third where possible, otherwise the longest
    remaining candidate regardless of third.

    ``min_s`` is a display annotation only (the nominal lower bound for a
    clip's length); it never disqualifies a candidate — the disqualification
    threshold is the 2 s :data:`MIN_TURN_S` gate, and the window length is
    ``min(max_s, turn_duration)``.

    ``segments`` is accepted for interface symmetry with the sidecar (the
    selector's rules operate on paragraph fields) and is not read.
    *total_duration* clamps windows to the audio end; it defaults to the
    end of the last paragraph. Returns an ordered dict of label -> windows
    sorted by ``start_s``; a speaker with fewer clean turns gets fewer
    clips, without error.
    """
    candidates: dict[str, list[tuple[float, float]]] = {}
    audio_end = total_duration
    if audio_end is None:
        audio_end = max(
            (float(p.get("end", 0.0)) for p in paragraphs if "end" in p),
            default=0.0,
        )

    for p in paragraphs:
        label = _labelled(p)
        if label is None or p.get("suspect"):
            continue
        text = str(p.get("text", ""))
        if not text or _is_backchannel(text):
            continue
        start, end = float(p.get("start", 0.0)), float(p.get("end", 0.0))
        duration = end - start
        if duration < MIN_TURN_S:
            continue
        # Window: middle min(max_s, D) s centered on the turn midpoint,
        # clamped to [0, audio_end].
        length = min(max_s, duration)
        mid = (start + end) / 2.0
        w_start = mid - length / 2.0
        w_end = w_start + length
        if audio_end > 0:
            w_end = min(w_end, audio_end)
            w_start = max(0.0, max(w_start, w_end - length))
        candidates.setdefault(label, []).append((w_start, w_end))

    result: dict[str, list[ClipWindow]] = {}
    for label, turns in candidates.items():
        if not turns:
            continue
        picked: list[tuple[float, float]] = []
        # Spread rule: fill beginning -> middle -> end thirds with the
        # longest candidate in each, then top up from the longest
        # remaining regardless of third.
        for third in (0, 1, 2):
            best = max(
                (
                    t
                    for t in turns
                    if t not in picked and _third_for(t[0], audio_end) == third
                ),
                key=lambda t: t[1] - t[0],
                default=None,
            )
            if best is not None:
                picked.append(best)
        for t in sorted(
            (t for t in turns if t not in picked),
            key=lambda t: t[1] - t[0],
            reverse=True,
        ):
            if len(picked) >= per_speaker:
                break
            picked.append(t)
        picked.sort(key=lambda t: t[0])
        result[label] = [
            ClipWindow(w_start, w_end, _quote(paragraphs, w_start, w_end))
            for w_start, w_end in picked[:per_speaker]
        ]
    return result


def _map_window(
    source: list[dict[str, Any]], start: float, end: float
) -> tuple[Path, float, float] | None:
    """Map group-level [start, end) onto one source part.

    Returns ``(path, in_part_start, in_part_end)`` for the part whose
    ``[part_offset_s, part_offset_s + duration_s)`` fully contains the
    window, or ``None`` when the window crosses a part boundary (straddle)
    or no part covers it. A part without a measured ``duration_s`` extends
    to the next part's offset (or infinity for the last part).
    """
    chosen: tuple[str, float, float] | None = None
    for entry in source:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        offset = float(entry.get("part_offset_s", 0.0))
        duration = entry.get("duration_s")
        if isinstance(duration, (int, float)) and duration > 0:
            part_end = offset + float(duration)
        else:
            part_end = float("inf")
        if offset <= start and end <= part_end:
            chosen = (path, start - offset, end - offset)
            break
    if chosen is None:
        return None
    return Path(chosen[0]), chosen[1], chosen[2]


def extract_clips(
    source: list[dict[str, Any]],
    windows: list[ClipWindow],
    tmp_dir: Path | str,
) -> dict[ClipWindow, Path | None]:
    """Decode each *window* from its source part into *tmp_dir*.

    The source paths (and part offsets/durations) come from the sidecar's
    ``source`` entries. The window is mapped to the part containing it and
    its in-part span decoded with ffmpeg ``-ss``/``-t`` placed BEFORE
    ``-i`` (input seeking: O(window), not O(offset)). Output is 16 kHz
    mono wav. Returns ``{window: path or None}``; ``None`` (never an
    exception) when the window's source file is missing/unreadable or the
    window straddles two parts — the caller falls back to quotes only.
    *tmp_dir* is owned by the caller (see :func:`clip_session`).
    """
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    results: dict[ClipWindow, Path | None] = {}
    for i, window in enumerate(windows):
        mapping = _map_window(source, window.start_s, window.end_s)
        if mapping is None:
            results[window] = None
            continue
        src, in_start, in_end = mapping
        if not src.is_file():
            results[window] = None
            continue
        out = tmp / f"clip_{i:02d}.wav"
        argv = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{in_start:.3f}",
            "-t",
            f"{in_end - in_start:.3f}",
            "-i",
            str(src),
            "-ar",
            _SAMPLE_RATE,
            "-ac",
            _CHANNELS,
            str(out),
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            results[window] = None
            continue
        if proc.returncode != 0 or not out.is_file() or out.stat().st_size <= 78:
            results[window] = None
            continue
        results[window] = out
    return results


def _run_player(argv: list[str]) -> int:
    """Run the audio player for *argv*; fail-open.

    The single seam that performs the real ``afplay`` call. Returns the
    player's exit code, or ``-1`` when it cannot be spawned (missing
    afplay, OS error, timeout) — the fail-open handling lives here so tests
    can stub it via the ``system_effects`` autouse fixture without globally
    patching ``subprocess`` or ``sys``.
    """
    try:
        proc = subprocess.run(argv, timeout=PLAY_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return -1
    return proc.returncode


def play(path: Path | str) -> bool:
    """Play *path* with ``afplay``; fail-open.

    Returns True on success, False (no exception) on non-darwin, missing
    afplay, missing file, non-zero exit, or timeout.
    """
    if sys.platform != "darwin":
        return False
    p = Path(path)
    if not p.is_file():
        return False
    return _run_player(["afplay", str(p)]) == 0


@contextmanager
def clip_session() -> Iterator[Path]:
    """Yield a fresh 0o700 temp dir, removed on every exit path.

    Normal exit, exception, and KeyboardInterrupt all remove the directory
    (and its contents) — no clips are ever kept on disk.
    """
    tmp = Path(tempfile.mkdtemp(prefix="vemoizer-clips-"))
    with contextlib.suppress(OSError):
        tmp.chmod(0o700)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
