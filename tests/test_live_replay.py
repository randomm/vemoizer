"""Live replay helper: stream order, lag accounting, pacing (issue #181).

No models: a fake VAD segments the stream and a fake transcriber stands in for
whisper. The model-backed replay lives in ``test_live_replay_models.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _live_helpers import replay_lag

from vemoizer.transcriber import TranscriptionResult
from vemoizer.vad import SpeechSegment

SR = 16_000


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeVad:
    """Closes each segment once the fed sample count reaches its end."""

    def __init__(self, segments: list[SpeechSegment]) -> None:
        self._pending = sorted(segments, key=lambda s: s.end)
        self._fed = 0

    def feed(self, chunk: np.ndarray) -> list[SpeechSegment]:
        self._fed += len(chunk)
        return self._close_upto(float(self._fed))

    def flush(self) -> list[SpeechSegment]:
        return self._close_upto(float("inf"))

    def _close_upto(self, limit: float) -> list[SpeechSegment]:
        closed = [s for s in self._pending if s.end <= limit]
        self._pending = [s for s in self._pending if s.end > limit]
        return closed


class _FakeTranscriber:
    """Numbers its turns ``u1``, ``u2``...; each decode advances the clock."""

    def __init__(self, clock: _Clock | None = None, latency_s: float = 0.0) -> None:
        self._clock = clock
        self._latency_s = latency_s
        self._calls = 0

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> TranscriptionResult:
        self._calls += 1
        if self._clock is not None:
            self._clock.now += self._latency_s
        return {"text": f"u{self._calls}", "words": [], "segments": []}

    def cleanup(self) -> None:
        return None


def _stream_with_three_utterances() -> tuple[np.ndarray, _FakeVad]:
    audio = np.zeros(6 * SR, dtype=np.float32)
    vad = _FakeVad(
        [
            SpeechSegment(start=1 * SR, end=2 * SR),
            SpeechSegment(start=3 * SR, end=4 * SR),
            SpeechSegment(start=5 * SR, end=6 * SR),
        ]
    )
    return audio, vad


def test_utterances_are_decoded_in_stream_order_on_the_stream_clock() -> None:
    audio, vad = _stream_with_three_utterances()

    replay = replay_lag(
        audio, vad=vad, transcriber=_FakeTranscriber(), sleep=lambda _s: None
    )

    assert [d.text for d in replay.decoded] == ["u1", "u2", "u3"]
    assert [d.start_s for d in replay.decoded] == pytest.approx([1.0, 3.0, 5.0])
    assert [d.end_s for d in replay.decoded] == pytest.approx([2.0, 4.0, 6.0])


def test_lag_is_emit_minus_capture_start_minus_utterance_end() -> None:
    audio, vad = _stream_with_three_utterances()
    clock = _Clock()
    # Sleep is a no-op and only decodes move the clock, so emits land at
    # 0.25 s, 0.5 s, 0.75 s and the lags below are exact.
    replay = replay_lag(
        audio,
        vad=vad,
        transcriber=_FakeTranscriber(clock, latency_s=0.25),
        clock=clock,
        sleep=lambda _s: None,
    )

    assert replay.lag.utterance.count == 3
    assert replay.lag.utterance.max == pytest.approx(0.25 - 2.0)
    assert replay.lag.utterance.p50 == pytest.approx(0.5 - 4.0)


def test_pace_scales_the_wall_time_of_the_replay() -> None:
    clock = _Clock()

    def sleep(seconds: float) -> None:
        clock.now += seconds

    audio = np.zeros(4 * SR, dtype=np.float32)
    replay_lag(
        audio,
        vad=_FakeVad([]),
        transcriber=_FakeTranscriber(),
        pace=2.0,
        clock=clock,
        sleep=sleep,
    )

    assert clock.now == pytest.approx(2.0)


def test_non_positive_pace_is_rejected() -> None:
    audio, vad = _stream_with_three_utterances()
    with pytest.raises(ValueError, match="pace"):
        replay_lag(audio, vad=vad, transcriber=_FakeTranscriber(), pace=0.0)
