"""Pure live-capture pieces (issue #172): chunks, track writer, silence, watchdog, skew.

Nothing here touches hardware. The Core Audio / ffmpeg seam lives in
``capture_sources.py``; every source hands this module 16 kHz mono float32
chunks and this module decides where they land in the FLAC stem.
"""

from __future__ import annotations

import contextlib
import logging
import queue
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile

from vemoizer.audio_contract import SAMPLE_RATE

log = logging.getLogger(__name__)

#: Flush the FLAC stream roughly every five seconds of audio.
FLUSH_FRAMES = 5 * SAMPLE_RATE

#: Seconds between measured me/them skew log lines.
SKEW_LOG_EVERY_S = 60.0


@dataclass(frozen=True, slots=True)
class Chunk:
    """16 kHz mono float32 samples plus where they sit on the session clock.

    ``first_sample_t`` is the capture time (seconds, same clock for both
    tracks) of ``samples[0]``. ``discontinuity_frames`` is a known gap in
    front of the samples, in frames; the writer inserts exactly that many zeros.
    """

    samples: np.ndarray
    first_sample_t: float
    discontinuity_frames: int = 0


class CaptureSource(Protocol):
    """A live source. ``chunks()`` yields until the source is stopped."""

    @property
    def failed(self) -> bool: ...

    def start(self) -> None: ...

    def chunks(self) -> Iterator[Chunk]: ...

    def stop(self) -> None: ...


def drain(out: queue.SimpleQueue[Chunk | None]) -> Iterator[Chunk]:
    """Yield chunks from a worker's output queue until its ``None`` sentinel."""
    while (item := out.get()) is not None:
        yield item


def is_digital_silence(samples: np.ndarray) -> bool:
    """True only for non-empty buffers of exact zeros (a dead input, not quiet)."""
    return samples.size > 0 and not bool(np.any(samples))


class SilenceWatchdog:
    """Fires once after ``seconds`` without a signal; a later signal re-arms it."""

    def __init__(self, seconds: float, clock: Callable[[], float]) -> None:
        self._seconds = seconds
        self._clock = clock
        self._since = clock()
        self._fired = False

    def signal(self) -> None:
        self._since = self._clock()
        self._fired = False

    def check(self) -> bool:
        if not self._fired and self._clock() - self._since >= self._seconds:
            self._fired = True
            return True
        return False


class SkewLogger:
    """Logs measured me/them skew (both directions). Never corrects anything."""

    def __init__(self, every_s: float = SKEW_LOG_EVERY_S) -> None:
        self._every = every_s
        self._next = every_s

    def observe(self, me_t: float, them_t: float) -> float | None:
        """Log when the session crosses the next interval; return the skew then."""
        if me_t < self._next:
            return None
        while self._next <= me_t:
            self._next += self._every
        return self.final(me_t, them_t)

    def final(self, me_t: float, them_t: float) -> float:
        """Log the skew now (used at stop)."""
        skew = them_t - me_t
        log.info(skew_message(skew))
        return skew


def skew_message(skew: float) -> str:
    """The one-line wording for a measured skew (``them`` minus ``me``, seconds)."""
    direction = "ahead of" if skew >= 0 else "behind"
    return f"track skew: them {direction} me by {abs(skew) * 1000:.0f} ms"


#: What soundfile 0.14 raises on a failed write or close: OSError from the OS,
#: LibsndfileError (a RuntimeError) from libsndfile, AssertionError on a short write.
_FILE_ERRORS = (OSError, RuntimeError, AssertionError)


def _open_flac(path: str | Path) -> Any:
    # soundfile >= 0.14 takes the container as a keyword, not positionally.
    return soundfile.SoundFile(str(path), "w", SAMPLE_RATE, 1, "PCM_24", format="FLAC")


class TrackWriter:
    """One track's FLAC stem. Both tracks share session index 0 via ``t0``.

    The first chunk is head-padded to ``first_sample_t - t0``; after that only
    ``discontinuity_frames`` add zeros. Arrival jitter never adds samples.
    """

    def __init__(
        self,
        path: str | Path,
        t0: float,
        *,
        open_file: Callable[[str | Path], Any] | None = None,
        flush_frames: int = FLUSH_FRAMES,
    ) -> None:
        self._path = path
        self._t0 = t0
        self._file = (open_file or _open_flac)(path)
        self._flush_frames = flush_frames
        self._frames = 0
        self._since_flush = 0
        self._started = False
        self._clipped = 0
        self._clip_warned = False
        self._usable = True
        self._closed = False

    @property
    def usable(self) -> bool:
        return self._usable

    @property
    def clipped(self) -> int:
        return self._clipped

    @property
    def position_s(self) -> float:
        """Seconds of audio (padding included) written since ``t0``."""
        return self._frames / SAMPLE_RATE

    def write(self, chunk: Chunk) -> bool:
        """Write one chunk; False once the track has stopped being usable."""
        if not self._usable:
            return False
        head = 0
        if not self._started:
            self._started = True
            head = max(0, round((chunk.first_sample_t - self._t0) * SAMPLE_RATE))
        pad = head + chunk.discontinuity_frames
        samples = np.asarray(chunk.samples, dtype=np.float32)
        try:
            if pad:
                self._file.write(np.zeros(pad, dtype=np.float32))
            if samples.size:
                self._file.write(samples)
            self._frames += pad + samples.size
            self._since_flush += pad + samples.size
            if self._since_flush >= self._flush_frames:
                self._file.flush()
                self._since_flush = 0
        except _FILE_ERRORS as exc:
            self._fail(exc)
            return False
        self._count_clips(samples)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._file.close()
        except _FILE_ERRORS as exc:
            log.error("closing %s failed: %s", self._path, exc)

    def _count_clips(self, samples: np.ndarray) -> None:
        over = int(np.count_nonzero(np.abs(samples) > 1.0))
        if not over:
            return
        self._clipped += over
        if not self._clip_warned:
            self._clip_warned = True
            log.warning("input exceeds +/-1.0 in %s; samples are clipped", self._path)

    def _fail(self, exc: BaseException) -> None:
        self._usable = False
        log.error("recording stopped: writing %s failed: %s", self._path, exc)
        # The write error above is the one line we report; a close error on the
        # same broken file adds nothing, so it is not logged a second time.
        self._closed = True
        with contextlib.suppress(*_FILE_ERRORS):
            self._file.close()
