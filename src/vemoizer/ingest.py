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

from vemoizer.audio_contract import SAMPLE_RATE

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


def pcm_duration_seconds(path: Path | str, timeout: float = 300.0) -> float:
    """Duration in seconds of *path*'s decoded PCM, without materialising it.

    Same contract as :func:`ingest_audio` (the :data:`_FFMPEG_AUDIO_ARGS`
    argv, raw f32le on stdout — byte count, never ffprobe or container
    metadata), but the decoded stream is read in bounded chunks and only
    the byte count is kept: a 1-hour memo costs ~1 MB of transient memory
    instead of ~2.4 GB of float32 PCM (part offsets only need the length).
    A generous *timeout* bounds the decode.

    Returns exactly what ``duration_seconds(ingest_audio(path))`` would.

    Raises:
        IngestError: ffmpeg is missing, the file is unreadable/corrupt,
            ffmpeg exits non-zero, or the decode times out.
    """
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"audio file not found: {p}")

    argv = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]

    try:
        with subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            if proc.stdout is None or proc.stderr is None:
                # Unreachable: both pipes are set explicitly above. Guard
                # keeps ty quiet without changing behaviour.
                raise IngestError(
                    f"internal error: ffmpeg pipes not available for {p}",
                    returncode=None,
                )
            total = 0
            try:
                while chunk := proc.stdout.read(1 << 20):
                    total += len(chunk)
            finally:
                # proc.stdout is still open here, so proc has not been
                # reaped — enforce the bound before the join (a wedged
                # decode must not hang part offsets for hours).
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise IngestError(
                        f"ffmpeg timed out after {timeout:.0f}s decoding {p}",
                        returncode=None,
                    ) from None
                stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        raise IngestError(
            "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        ) from None

    if proc.returncode != 0:
        stderr = " ".join(stderr.split())
        raise IngestError(
            f"ffmpeg failed to decode {p} (exit {proc.returncode}): {stderr}",
            returncode=proc.returncode,
        )

    n = total // 4
    return n / SAMPLE_RATE if n else 0.0
