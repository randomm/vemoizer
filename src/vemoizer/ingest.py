"""Audio ingest: decode .m4a (or any ffmpeg-readable container) to 16 kHz mono float32.

This is the first stage of the consensus pipeline. It runs a single ffmpeg
process that:
  - reads the input file (never trusts ffprobe for duration)
  - decodes to raw PCM on stdout
  - resamples to 16 kHz, mono, float32

iOS Voice Memos quirks (edit lists, HE-AAC in older exports) are handled
implicitly: ffmpeg's decoder does the right thing, and we read the raw PCM
byte count (not a container metadata field) to determine the sample count.

The stage is pure subprocess + numpy — no model loading, no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# ffmpeg argv contract (issue #2, AGENTS.md invariants):
#   -nostdin      : never block on stdin
#   -v error      : only surface real errors
#   -ac 1         : force mono
#   -ar 16000     : force 16 kHz
#   -c:a pcm_f32le: encode to raw little-endian float32 PCM
#   -f f32le      : raw format on stdout
#   -             : output to stdout (never write a temp file)
_FFMPEG_AUDIO_ARGS = (
    "-nostdin",
    "-v",
    "error",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "pcm_f32le",
    "-f",
    "f32le",
    "-",
)

SAMPLE_RATE = 16_000


class IngestError(RuntimeError):
    """Raised when ffmpeg fails to decode the input audio."""

    def __init__(self, message: str, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


def ingest_audio(path: Path | str) -> np.ndarray:
    """Decode *path* to a 16 kHz mono float32 numpy array.

    Args:
        path: Path to an audio file (typically .m4a from iOS Voice Memos).

    Returns:
        numpy array of shape ``(n,)`` with ``dtype=np.float32`` at 16 kHz.

    Raises:
        IngestError: ffmpeg is missing, the file is unreadable/corrupt, or
            ffmpeg exits non-zero for any other reason.
    """
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"audio file not found: {p}")

    argv = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        raise IngestError(
            "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        ) from None

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise IngestError(
            f"ffmpeg failed to decode {p} (exit {proc.returncode}): {stderr}",
            returncode=proc.returncode,
        )

    # Byte count → sample count. We never trust container metadata (edit
    # lists in iOS Voice Memos make ffprobe duration lie).
    raw = proc.stdout
    n = len(raw) // 4
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32, count=n).copy()


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Return the duration in seconds of a mono float32 array."""
    return len(audio) / sample_rate
