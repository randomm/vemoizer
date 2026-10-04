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

import contextlib
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
# The split into _FFMPEG_DECODE_ARGS (up to and including the codec spec)
# and _FFMPEG_OUTPUT_ARGS (the raw stream output spec) is structural, not
# arbitrary: the loudnorm filter is inserted BETWEEN them (after the
# resample/downmix, before the output spec) in the pass-2 argv builder.
_FFMPEG_DECODE_ARGS = (
    "-nostdin",
    "-v",
    "error",
    "-ac",
    "1",
    "-ar",
    "16000",
    "-c:a",
    "pcm_f32le",
)
_FFMPEG_OUTPUT_ARGS = (
    "-f",
    "f32le",
    "-",
)
_FFMPEG_AUDIO_ARGS = _FFMPEG_DECODE_ARGS + _FFMPEG_OUTPUT_ARGS


class IngestError(RuntimeError):
    """Raised when ffmpeg fails to decode the input audio."""

    def __init__(self, message: str, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


def ingest_audio(path: Path | str, *, preprocess: str | None = None) -> np.ndarray:
    """Decode *path* to a 16 kHz mono float32 numpy array.

    Args:
        path: Path to an audio file (typically .m4a from iOS Voice Memos).
        preprocess: ``"loudnorm"`` (issue #135) runs the two-pass
            ``loudnorm`` normalization first (fail open: any measurement
            failure falls back to the plain decode with one warning);
            ``None`` (the default) keeps the plain decode argv literally
            unchanged.

    Returns:
        numpy array of shape ``(n,)`` with ``dtype=np.float32`` at 16 kHz.

    Raises:
        IngestError: ffmpeg is missing, the file is unreadable/corrupt, or
            ffmpeg exits non-zero for any other reason.
    """
    from . import loudnorm as _loudnorm

    p = Path(path)
    if not p.is_file():
        raise IngestError(f"audio file not found: {p}")

    if preprocess == "loudnorm":
        return _loudnorm.preprocess_audio(p, preprocess="loudnorm")

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


# Upper bound (chars, whitespace-collapsed) on the stderr excerpt kept
# for error messages — consistent with grouping_concat / grouping_decode.
_STDERR_EXCERPT_CHARS = 200


@dataclass
class _DrainState:
    """The drain's reader-thread handoff (written only by the reader).

    The reader thread records how many bytes it saw on stdout
    (``total``) and the exception it hit, if any (``error``). The
    reader never raises into the main thread directly: a read failure
    (broken pipe, interrupt) is stored here so the caller thread —
    waiting on the reader join — can re-raise it instead of hanging.
    """

    total: int = 0
    error: BaseException | None = None


def _drain_ffmpeg_pcm(
    proc: Any,
    stderr_file: Any,
    timeout: float,
) -> tuple[int, str]:
    """Stream *proc*'s stdout in bounded chunks, enforcing *timeout* mid-drain.

    A daemon reader thread counts the bytes on stdout while the caller
    waits on that thread with the remaining wall-clock budget — the bound
    is enforced DURING the drain, not after EOF (a wedged ffmpeg that is
    alive but produces no output is killed, reaped, and an
    :class:`IngestError` (``"…timed out after N s…"``) is raised by the
    caller (``pcm_duration_seconds``) — the internal
    :class:`subprocess.TimeoutExpired` is converted before it propagates,
    never let hang the
    batch). *stderr_file* is a temp file the process already writes to
    (never an unread pipe, which could deadlock the drain once the pipe
    buffer fills); after the process is reaped it is read back and
    reduced to a whitespace-collapsed excerpt capped at
    :data:`_STDERR_EXCERPT_CHARS` chars. On timeout, on any exception,
    and on KeyboardInterrupt the process is killed and reaped before the
    caller is returned to, so no zombie or leaked fd survives.
    """
    deadline = time.monotonic() + timeout

    state = _DrainState()

    def _reader() -> None:
        assert proc.stdout is not None
        total = 0
        try:
            while chunk := proc.stdout.read(1 << 20):
                total += len(chunk)
        except BaseException as e:
            # The reader thread must not die silently: if the read fails
            # (e.g. a broken pipe or an interrupt), the caller thread
            # (stuck in reader.join) would never wake up. Store the
            # exception so the drain loop can handle it via its
            # except-BaseException handler.
            state.total = total
            state.error = e
            return
        state.total = total
        state.error = None

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            if reader.is_alive():
                reader.join(remaining)
            else:
                # The reader finished (or died): check for a stored
                # reader-side exception before waiting on the process.
                if state.error is not None:
                    raise state.error
                proc.wait(timeout=remaining)
                break
    except BaseException:
        # Timeout, a reader-side exception, or an interrupt: kill and
        # reap so nothing outlives this call (no zombie, no leaked fd).
        if proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()
        with contextlib.suppress(OSError):
            proc.wait(timeout=5.0)
        raise
    stderr_file.seek(0)
    stderr = stderr_file.read()
    return state.total, _bounded_stderr(stderr)


def _bounded_stderr(stderr: bytes) -> str:
    """Whitespace-collapse *stderr* and cap it for user-facing messages."""
    stderr_text = " ".join(stderr.decode("utf-8", errors="replace").split())
    if len(stderr_text) > _STDERR_EXCERPT_CHARS:
        stderr_text = stderr_text[:_STDERR_EXCERPT_CHARS] + "…"
    return stderr_text


def pcm_duration_seconds(
    path: Path | str, timeout: float = 300.0, *, preprocess: str | None = None
) -> float:
    """Duration in seconds of *path*'s decoded PCM, without materialising it.

    Same contract as :func:`ingest_audio` (the :data:`_FFMPEG_AUDIO_ARGS`
    argv, raw f32le on stdout — byte count, never ffprobe or container
    metadata), but the decoded stream is read in bounded chunks and only
    the byte count is kept: a 1-hour memo costs ~1 MB of transient memory
    instead of ~2.4 GB of float32 PCM (part offsets only need the length).

    *timeout* is an overall wall-clock bound on the decode, enforced
    DURING the stdout drain (a wedged ffmpeg that stops producing output
    is killed and reaped, then an :class:`IngestError` is raised) — not
    a post-EOF check that a stalled process could slip past. stderr is
    never an unread pipe: it drains to a temp file, so a chatty ffmpeg
    cannot deadlock the drain, and only a bounded excerpt of it is ever
    kept for error messages. The process is always killed and reaped on
    timeout, exception, and KeyboardInterrupt — no zombie, no leak.

    Returns exactly what ``duration_seconds(ingest_audio(path))`` would
    (the same decode argv, so part offsets and source durations match
    the processed decode when *preprocess* is ``"loudnorm"``).

    Raises:
        IngestError: ffmpeg is missing, the file is unreadable/corrupt,
            ffmpeg exits non-zero, or the decode times out.
    """
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"audio file not found: {p}")

    # The same decode as ingest_audio: the byte count of the decoded PCM
    # is the duration, and the decode must match what the transcript saw
    # (issue #135 consistency rule: part offsets and source durations
    # come from the processed decode, never the raw one). decode_argv
    # builds the loudnorm pass-2 argv from the same cached measurement
    # the transcript decode uses, so for an unchanged file the two
    # decodes share the measured values (one pass-1 per file).
    from . import loudnorm as _loudnorm

    argv = _loudnorm.decode_argv(p, preprocess)

    # stderr goes to a temp file (not a pipe) so a chatty ffmpeg cannot
    # fill the pipe buffer and deadlock the stdout drain; stdout is a
    # pipe.
    stderr_file = tempfile.TemporaryFile()  # noqa: SIM115
    try:
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr_file)
        except FileNotFoundError:
            raise IngestError(
                "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
            ) from None
        # stdout is a pipe above, so it is never None (the reader thread
        # would otherwise need a None guard per chunk); the assert keeps
        # the reader's contract explicit.
        assert proc.stdout is not None

        try:
            total, stderr = _drain_ffmpeg_pcm(proc, stderr_file, timeout)
        except subprocess.TimeoutExpired:
            raise IngestError(
                f"ffmpeg timed out after {timeout:.0f}s decoding {p}",
                returncode=None,
            ) from None
        except OSError as e:
            # A reader-side failure (broken pipe / interrupted read on
            # the stdout drain) is an ingest failure, not a crash.
            raise IngestError(
                f"ffmpeg stdout drain failed while decoding {p}: {e}",
                returncode=None,
            ) from None
    finally:
        # The process owns the fd itself; close only our handle so no
        # file object survives this frame (the temp file's name is
        # unlinked by the OS on close).
        stderr_file.close()

    if proc.returncode != 0:
        raise IngestError(
            f"ffmpeg failed to decode {p} (exit {proc.returncode}): {stderr}",
            returncode=proc.returncode,
        )

    n = total // 4
    return n / SAMPLE_RATE if n else 0.0
