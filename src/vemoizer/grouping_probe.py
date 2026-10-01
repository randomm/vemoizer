"""FFprobe helpers for the M3 split-recording grouping (issue #77).

``probe_duration_seconds`` returns the container duration via ffprobe —
advisory only, never load-bearing. It is used to BOUND a probe decode
(how many seconds to request from ffmpeg) — not the duration of any
transcribed audio. Part offsets and boundary positions come from decoded
PCM, never container metadata (edit lists in iOS Voice Memos make ffprobe
duration lie). An unreadable or corrupt file yields ``0.0`` ("no probe
evidence").

``_probe_stream`` returns the part's audio stream signature (codec /
sample rate / channels), used by :func:`grouping_concat.concat_groups`
to detect a codec mismatch before the ``-c copy`` concat.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _probe_stream(part: Path) -> str:
    """The part's audio stream signature (codec/sample_rate/channels)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "csv=p=0",
                str(part),
            ],
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except FileNotFoundError:
        from vemoizer.grouping import GroupingError

        raise GroupingError(
            "ffprobe not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        ) from None
    if proc.returncode != 0:
        from vemoizer.grouping import GroupingError

        # Truncate and collapse stderr to a single bounded paragraph that
        # still names the offending file (round 3 finding 8).
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stderr = " ".join(stderr.split())
        if len(stderr) > 200:
            stderr = stderr[:200] + "…"
        raise GroupingError(
            f"ffprobe failed for {part.name} (exit {proc.returncode}): {stderr}"
        )
    return proc.stdout.decode("utf-8", errors="replace").strip()


def probe_duration_seconds(path: Path) -> float:
    """Container duration via ffprobe — advisory only, never load-bearing.

    Used to BOUND a probe decode (how many seconds to request from
    ffmpeg) — not the duration of any transcribed audio. Part offsets
    and boundary positions come from decoded PCM, never container
    metadata (edit lists in iOS Voice Memos make ffprobe duration lie).
    An unreadable or corrupt file yields ``0.0`` ("no probe evidence").
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            check=False,
            timeout=30.0,
        )
    except FileNotFoundError:
        return 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    if proc.returncode != 0:
        return 0.0
    value = proc.stdout.decode("utf-8", errors="replace").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0
