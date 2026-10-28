"""Decode backlog warning (issue #180): warn once past 30 s, never drop audio.

The transcriber blocks on an event so the queue can build up; no model loads.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.live_decode import BACKLOG_WARN_S, DecodeWorker, Utterance
from vemoizer.transcriber import TranscriptionResult


class _GatedTranscriber:
    def __init__(self) -> None:
        self.gate = threading.Event()
        self.decoded: list[float] = []

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> TranscriptionResult:
        self.gate.wait(timeout=5.0)
        self.decoded.append(len(audio) / SAMPLE_RATE)
        return {"text": "x", "words": [], "segments": []}

    def cleanup(self) -> None:
        return None


def test_backlog_warns_once_and_nothing_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    transcriber = _GatedTranscriber()
    worker = DecodeWorker(transcriber, lambda _d: None, capture_t0=0.0)
    worker.start()
    with caplog.at_level(logging.WARNING, logger="vemoizer.live_decode"):
        for i in range(4):  # 4 x 10 s of audio waiting behind the gate
            start = i * 10.0
            audio = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
            worker.submit("them", Utterance(start_s=start, audio=audio))
        transcriber.gate.set()
        report = worker.stop()

    assert BACKLOG_WARN_S == 30.0
    assert caplog.text.count("nothing is dropped") == 1
    assert len(transcriber.decoded) == 4
    assert report.utterance.count == 4


def test_no_warning_below_the_threshold(caplog: pytest.LogCaptureFixture) -> None:
    transcriber = _GatedTranscriber()
    transcriber.gate.set()
    worker = DecodeWorker(transcriber, lambda _d: None, capture_t0=0.0)
    worker.start()
    with caplog.at_level(logging.WARNING, logger="vemoizer.live_decode"):
        worker.submit(
            "me", Utterance(start_s=0.0, audio=np.zeros(SAMPLE_RATE, np.float32))
        )
        worker.stop()
    assert "nothing is dropped" not in caplog.text
