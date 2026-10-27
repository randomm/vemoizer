"""Live decode worker for ``vemoizer listen --live`` (issue #178).

Utterances from both tracks are decoded in end-time order on ONE thread with
ONE transcriber. The live draft is a single-decode preview, never the
deliverable (invariant #2: the post-call pass always runs the full pipeline).

Why one thread: mlx-whisper's ``ModelHolder`` is an unlocked process-global,
and MLX streams are thread-local (F18). The worker therefore owns every MLX
call, and on exit it tears MLX down on its own thread so the main-thread
final pass reloads cleanly instead of silently taking the dictation path.
"""

from __future__ import annotations

import itertools
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.live_output import Track
from vemoizer.transcriber import Transcriber

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Utterance:
    """A closed utterance: 16 kHz mono float32 audio starting at ``start_s``.

    ``start_s`` is the capture-timeline second of ``audio[0]`` (the same
    clock as the capture stems).
    """

    start_s: float
    audio: np.ndarray

    @property
    def end_s(self) -> float:
        return self.start_s + len(self.audio) / SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class Decoded:
    """One decoded turn, with words and segments on the capture timeline."""

    track: Track
    start_s: float
    end_s: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)


def _shift(items: Sequence[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    return [
        {
            **item,
            "start": float(item["start"]) + offset_s,
            "end": float(item["end"]) + offset_s,
        }
        for item in items
    ]


def decode_utterance(transcriber: Transcriber, utt: Utterance, track: Track) -> Decoded:
    """Decode one utterance as a single window and shift it to the capture clock.

    The transcriber's own pipeline (glossary prompt, per-call language ID,
    echo filter, lost-window retry) runs unchanged; this only places its
    output on the timeline.
    """
    raw = transcriber.transcribe(utt.audio, vad_slices=[(0, len(utt.audio))])
    return Decoded(
        track=track,
        start_s=utt.start_s,
        end_s=utt.end_s,
        text=str(raw.get("text", "")).strip(),
        words=_shift(list(raw.get("words") or []), utt.start_s),
        segments=_shift(list(raw.get("segments") or []), utt.start_s),
    )


def nearest_rank(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile: the ``ceil(pct/100 * n)``-th smallest value."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class LagSummary:
    count: int
    p50: float
    p90: float
    max: float


def _summarize(values: Sequence[float]) -> LagSummary:
    if not values:
        return LagSummary(count=0, p50=0.0, p90=0.0, max=0.0)
    return LagSummary(
        count=len(values),
        p50=nearest_rank(values, 50),
        p90=nearest_rank(values, 90),
        max=max(values),
    )


class LagLog:
    """Lag = emit − (capture_t0 + utterance end). Per-word lag is informational."""

    def __init__(self) -> None:
        self._utterance: list[float] = []
        self._word: list[float] = []

    def record(
        self,
        *,
        utt_end_s: float,
        capture_t0: float,
        emit_t: float,
        word_end_s: Sequence[float],
    ) -> None:
        self._utterance.append(emit_t - (capture_t0 + utt_end_s))
        self._word.extend(emit_t - (capture_t0 + end) for end in word_end_s)

    def utterance_summary(self) -> LagSummary:
        return _summarize(self._utterance)

    def word_summary(self) -> LagSummary:
        return _summarize(self._word)


@dataclass(frozen=True, slots=True)
class LagReport:
    utterance: LagSummary
    word: LagSummary


#: Queue key that sorts after every real utterance end: stop() drains first.
_STOP_KEY = math.inf

_Job = tuple[Track, Utterance]


class DecodeWorker(threading.Thread):
    """Owns the transcriber and every MLX call; emits decoded turns in end order."""

    def __init__(
        self,
        transcriber: Transcriber,
        emit: Callable[[Decoded], None],
        *,
        capture_t0: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name="live-decode", daemon=True)
        self._transcriber = transcriber
        self._emit = emit
        self._capture_t0 = capture_t0
        self._clock = clock
        self._queue: queue.PriorityQueue[tuple[float, int, _Job | None]] = (
            queue.PriorityQueue()
        )
        self._seq = itertools.count()
        self._lag = LagLog()

    def submit(self, track: Track, utt: Utterance) -> None:
        """Queue one utterance; ordering is by its end time across tracks."""
        self._queue.put((utt.end_s, next(self._seq), (track, utt)))

    def stop(self) -> LagReport:
        """Drain the queue, tear MLX down on the worker thread, join, report lag."""
        self._queue.put((_STOP_KEY, next(self._seq), None))
        self.join()
        report = LagReport(
            utterance=self._lag.utterance_summary(), word=self._lag.word_summary()
        )
        u = report.utterance
        logger.info(
            "live decode lag: p50 %.2fs p90 %.2fs max %.2fs (n=%d)",
            u.p50,
            u.p90,
            u.max,
            u.count,
        )
        return report

    def run(self) -> None:
        try:
            while True:
                _, _, job = self._queue.get()
                if job is None:
                    return
                self._handle(*job)
        finally:
            _teardown(self._transcriber)

    def _handle(self, track: Track, utt: Utterance) -> None:
        # Fail-open per utterance: one bad decode never costs the rest of the draft.
        try:
            decoded = decode_utterance(self._transcriber, utt, track)
            emit_t = self._clock()
            self._lag.record(
                utt_end_s=utt.end_s,
                capture_t0=self._capture_t0,
                emit_t=emit_t,
                word_end_s=[float(w["end"]) for w in decoded.words],
            )
            self._emit(decoded)
        except Exception as exc:
            logger.warning(
                "live decode failed for %s utterance at %.1fs: %s",
                track,
                utt.start_s,
                exc,
            )


def _teardown(transcriber: Transcriber) -> None:
    """Release MLX state on the thread that created it (F18)."""
    from mlx_whisper.audio import mel_filters

    try:
        transcriber.cleanup()
    finally:
        mel_filters.cache_clear()
        mx.clear_cache()
