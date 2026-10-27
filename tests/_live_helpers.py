"""Replay a recorded stream through the live path at a chosen pace (issue #181).

The stream is fed in 20 ms blocks, the capture callback's block size. Each
block waits until its stream time has elapsed on ``clock`` (scaled by
``pace``), then goes through the segmenter; closed utterances go to the real
:class:`DecodeWorker`, so the lag numbers are the same ones ``listen --live``
would report.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.live_decode import Decoded, DecodeWorker, LagReport, Utterance
from vemoizer.live_output import Track
from vemoizer.transcriber import Transcriber
from vemoizer.vad import SpeechSegment

BLOCK_SAMPLES = 320  # 20 ms at 16 kHz

#: The replayed stream is one track; the label only matters for the live draft.
_TRACK: Track = "them"


class Segmenter(Protocol):
    """The part of :class:`StreamingVad` the replay drives."""

    def feed(self, chunk: np.ndarray) -> list[SpeechSegment]: ...

    def flush(self) -> list[SpeechSegment]: ...


@dataclass(frozen=True, slots=True)
class Replay:
    decoded: list[Decoded]
    lag: LagReport


def replay_lag(
    audio: np.ndarray,
    *,
    vad: Segmenter,
    transcriber: Transcriber,
    pace: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Replay:
    """Feed *audio* through *vad* at *pace* x real time and decode the utterances.

    ``pace=1.0`` is real time; stream second ``t`` is due at ``t0 + t / pace``.
    """
    if pace <= 0:
        raise ValueError(f"pace must be positive, got {pace}")
    decoded: list[Decoded] = []
    t0 = clock()
    worker = DecodeWorker(transcriber, decoded.append, capture_t0=t0, clock=clock)
    worker.start()
    fed = 0
    for start in range(0, len(audio), BLOCK_SAMPLES):
        chunk = audio[start : start + BLOCK_SAMPLES]
        fed += len(chunk)
        _wait_until(t0 + fed / SAMPLE_RATE / pace, clock, sleep)
        for seg in vad.feed(chunk):
            _submit(worker, audio, seg)
    for seg in vad.flush():
        _submit(worker, audio, seg)
    return Replay(decoded=decoded, lag=worker.stop())


def _wait_until(
    due: float, clock: Callable[[], float], sleep: Callable[[float], None]
) -> None:
    remaining = due - clock()
    if remaining > 0:
        sleep(remaining)


def _submit(worker: DecodeWorker, audio: np.ndarray, seg: SpeechSegment) -> None:
    worker.submit(
        _TRACK,
        Utterance(start_s=seg.start / SAMPLE_RATE, audio=audio[seg.start : seg.end]),
    )
