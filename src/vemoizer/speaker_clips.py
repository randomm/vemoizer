"""M5b: speaker clip extraction and playback helpers (issue #90).

The audio half of the speaker-clip feature: extract the windows chosen by
:func:`vemoizer.speaker_clip_select.select_clips` (re-exported here, with
:func:`talk_share`, as the module's public seam) from the SOURCE files
recorded in the sidecar's ``source`` entries, and play them with ``afplay``.

Design notes
------------

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

from .speaker_clip_select import ClipWindow, select_clips, talk_share

__all__ = [
    "ClipWindow",
    "clip_session",
    "extract_clips",
    "play",
    "select_clips",
    "source_duration",
    "talk_share",
]

#: play() wall-clock bound for afplay, in seconds.
PLAY_TIMEOUT_S = 30.0

#: ffmpeg decode bound for a single window, in seconds.
FFMPEG_TIMEOUT_S = 60.0

#: Output sample rate and channel count (the 16 kHz mono audio contract).
_SAMPLE_RATE = "16000"
_CHANNELS = "1"


def source_duration(
    source: list[dict[str, Any]], words: list[dict[str, Any]] | None = None
) -> float:
    """End of the last measured source part; falls back to the last word end.

    Clip windows must be clamped to something: the sidecar's ``duration_s``
    entries when present, the sidecar's ``words`` otherwise. *source* holds
    resolved entry dicts (see ``names_cli._resolve_source_entry``).
    """
    duration = 0.0
    for entry in source:
        offset = float(entry.get("part_offset_s", 0.0))
        part = entry.get("duration_s")
        if isinstance(part, (int, float)) and part > 0:
            duration = max(duration, offset + float(part))
    if duration <= 0:
        for w in words or []:
            duration = max(duration, float(w.get("end", w.get("start", 0.0))))
    return duration


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
    player's exit code, or ``-1`` for any failure to run the player — a
    missing binary (OSError) or a timeout (``subprocess.TimeoutExpired``,
    a ``SubprocessError`` subclass caught by the same clause). Callers treat
    any non-zero result as "not played". The fail-open handling lives here
    so tests can stub it via the ``system_effects`` autouse fixture without
    globally patching ``subprocess`` or ``sys``.
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
