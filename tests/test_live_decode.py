"""Live decode worker: one thread, one transcriber, ordered turns (issue #178).

Every model call here is faked, except the echo test, which runs the real
``WhisperTranscriber`` with ``mlx_whisper`` mocked (no weights, no network).
The MLX stream-teardown test uses the real ``mlx`` / ``mlx_whisper.audio``
modules but loads no weights.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from _whisper_helpers import _raw, _seg

from vemoizer.live_decode import (
    Decoded,
    DecodeWorker,
    Utterance,
    decode_utterance,
)
from vemoizer.live_output import Track
from vemoizer.transcriber import TranscriptionResult
from vemoizer.whisper_transcriber import WhisperTranscriber


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeTranscriber:
    """Records calls; ``fail_on`` raises on the N-th call (1-based).

    Each decode advances the optional virtual clock by ``latency_s`` so lag
    is exact. The fake returns one word at 0.2–1.0 s of its own audio.
    """

    def __init__(
        self,
        *,
        fail_on: int | None = None,
        latency_s: float = 0.0,
        clock: _VirtualClock | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_on = fail_on
        self.latency_s = latency_s
        self.clock = clock

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> TranscriptionResult:
        n = len(self.calls) + 1
        self.calls.append(kwargs)
        if self.clock is not None:
            self.clock.now += self.latency_s
        if n == self.fail_on:
            raise RuntimeError(f"boom on call {n}")
        text = f"u{n}"
        return {
            "text": text,
            "words": [{"word": text, "start": 0.2, "end": 1.0}],
            "segments": [{"text": text, "start": 0.2, "end": 1.0, "words": []}],
        }

    def cleanup(self) -> None:
        return None


def _utt(start_s: float, seconds: float = 1.0) -> Utterance:
    return Utterance(start_s=start_s, audio=np.zeros(int(seconds * 16_000), np.float32))


# --- offset shift and call shape -------------------------------------------


def test_offset_shift_puts_words_on_capture_timeline() -> None:
    decoded = decode_utterance(_FakeTranscriber(), _utt(61.0, 1.0), "me")

    assert decoded.start_s == pytest.approx(61.0)
    assert decoded.end_s == pytest.approx(62.0)
    assert decoded.words[0]["start"] == pytest.approx(61.2)
    assert decoded.words[0]["end"] == pytest.approx(62.0)
    assert decoded.segments[0]["start"] == pytest.approx(61.2)
    assert decoded.segments[0]["end"] == pytest.approx(62.0)


def test_decode_passes_one_full_window_and_no_language_override() -> None:
    fake = _FakeTranscriber()
    decode_utterance(fake, _utt(5.0, seconds=2.0), "them")

    (kwargs,) = fake.calls
    assert kwargs["vad_slices"] == [(0, 32_000)]
    assert "language" not in kwargs


def test_decoded_track_is_carried_through() -> None:
    decoded = decode_utterance(_FakeTranscriber(), _utt(1.0), "them")
    assert decoded.track == "them"


# --- worker: serialization, ordering, fail-open ----------------------------


def test_two_producers_fifty_each_yield_hundred_serialized_lines() -> None:
    lock = threading.Lock()
    overlaps: list[bool] = []

    class _Locked(_FakeTranscriber):
        def transcribe(
            self, audio: np.ndarray, **kwargs: object
        ) -> TranscriptionResult:
            if not lock.acquire(blocking=False):
                overlaps.append(True)
                raise RuntimeError("concurrent decode")
            try:
                return super().transcribe(audio, **kwargs)
            finally:
                lock.release()

    lines: list[Decoded] = []
    worker = DecodeWorker(_Locked(), lines.append, capture_t0=0.0)
    worker.start()

    def produce(track: Track, base: float) -> None:
        for i in range(50):
            worker.submit(track, _utt(base + i * 2.0))

    threads = [
        threading.Thread(target=produce, args=("me", 0.0)),
        threading.Thread(target=produce, args=("them", 1.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    worker.stop()

    assert len(lines) == 100
    assert overlaps == []


def test_turns_emit_in_end_time_order_across_tracks() -> None:
    lines: list[Decoded] = []
    worker = DecodeWorker(_FakeTranscriber(), lines.append, capture_t0=0.0)
    # Queued before start so the whole queue is ordered by utterance end.
    worker.submit("them", _utt(4.0, 1.0))  # ends 5.0
    worker.submit("me", _utt(0.0, 1.0))  # ends 1.0
    worker.submit("them", _utt(2.0, 1.0))  # ends 3.0
    worker.start()
    worker.stop()

    assert [d.end_s for d in lines] == [1.0, 3.0, 5.0]


def test_error_on_second_utterance_keeps_first_and_third(
    caplog: pytest.LogCaptureFixture,
) -> None:
    lines: list[Decoded] = []
    worker = DecodeWorker(_FakeTranscriber(fail_on=2), lines.append, capture_t0=0.0)
    worker.start()
    for start in (0.0, 2.0, 4.0):
        worker.submit("me", _utt(start))
    with caplog.at_level(logging.WARNING, logger="vemoizer.live_decode"):
        worker.stop()

    assert [d.start_s for d in lines] == [0.0, 4.0]
    assert any("live decode failed" in r.getMessage() for r in caplog.records)


# --- echo filter through the real WhisperTranscriber -----------------------


def test_echo_segments_dropped_through_real_transcriber() -> None:
    raw = _raw(
        [
            _seg(
                "Sanasto, Pia, NG-TOPI",
                [{"word": " Sanasto", "start": 0.0, "end": 0.5}],
            ),
            _seg(
                "Pia soitti eilen Flagshipista",
                [{"word": " Pia", "start": 1.0, "end": 1.4}],
            ),
        ]
    )
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        transcriber = WhisperTranscriber(initial_prompt="Pia, NG-TOPI.")
        decoded = decode_utterance(transcriber, _utt(10.0, 3.0), "me")

    assert decoded.text == "Pia soitti eilen Flagshipista"
    assert [w["word"] for w in decoded.words] == ["Pia"]
    assert decoded.words[0]["start"] == pytest.approx(11.0)


# --- teardown: MLX streams are thread-local (F18) ---------------------------


def test_mel_filters_usable_on_main_thread_after_worker_teardown() -> None:
    import mlx.core as mx
    from mlx_whisper.audio import mel_filters

    class _MelTouching(_FakeTranscriber):
        def transcribe(
            self, audio: np.ndarray, **kwargs: object
        ) -> TranscriptionResult:
            mel_filters(128)  # lru-cached lazy array created on the worker
            return super().transcribe(audio, **kwargs)

    worker = DecodeWorker(_MelTouching(), lambda _d: None, capture_t0=0.0)
    worker.start()
    worker.submit("me", _utt(0.0))
    worker.stop()

    mx.eval(mel_filters(128))  # raises "no Stream(cpu, 0)" without the teardown


def test_worker_teardown_runs_on_worker_thread() -> None:
    seen: list[str] = []

    class _Recording(_FakeTranscriber):
        def cleanup(self) -> None:
            seen.append(threading.current_thread().name)

    worker = DecodeWorker(_Recording(), lambda _d: None, capture_t0=0.0)
    worker.start()
    worker.stop()

    assert seen == [worker.name]


# --- lag accounting ---------------------------------------------------------


def test_virtual_clock_lag_equals_decode_latency() -> None:
    clock = _VirtualClock()
    fake = _FakeTranscriber(latency_s=0.8, clock=clock)
    worker = DecodeWorker(fake, lambda _d: None, capture_t0=100.0, clock=clock)
    worker.start()
    clock.now = 105.0  # utterance ends at capture time 105 (t0 100 + end 5.0)
    worker.submit("me", _utt(4.0, 1.0))
    report = worker.stop()

    assert report.utterance.count == 1
    assert report.utterance.max == pytest.approx(0.8)


def test_queue_wait_is_counted_in_lag() -> None:
    clock = _VirtualClock()
    clock.now = 2.0  # both utterances have already ended when decoding starts
    fake = _FakeTranscriber(latency_s=0.8, clock=clock)
    worker = DecodeWorker(fake, lambda _d: None, capture_t0=0.0, clock=clock)
    worker.submit("me", _utt(0.0, 1.0))  # ends 1.0, emitted at 2.8
    worker.submit("them", _utt(0.5, 1.0))  # ends 1.5, emitted at 3.6 (waited)
    worker.start()
    report = worker.stop()

    assert report.utterance.count == 2
    assert report.utterance.max == pytest.approx(3.6 - 1.5)


def test_per_word_lag_is_reported_separately() -> None:
    clock = _VirtualClock()
    fake = _FakeTranscriber(latency_s=0.8, clock=clock)
    worker = DecodeWorker(fake, lambda _d: None, capture_t0=0.0, clock=clock)
    worker.start()
    worker.submit("me", _utt(0.0, 2.0))  # utterance ends 2.0; word ends 1.0
    report = worker.stop()

    assert report.utterance.max == pytest.approx(0.8 - 2.0)
    assert report.word.count == 1
    assert report.word.max == pytest.approx(0.8 - 1.0)


def test_stop_logs_lag_summary(caplog: pytest.LogCaptureFixture) -> None:
    worker = DecodeWorker(_FakeTranscriber(), lambda _d: None, capture_t0=0.0)
    worker.start()
    worker.submit("me", _utt(0.0))
    with caplog.at_level(logging.INFO, logger="vemoizer.live_decode"):
        worker.stop()

    assert any("live decode lag" in r.getMessage() for r in caplog.records)


def test_stop_without_turns_reports_empty_summary() -> None:
    worker = DecodeWorker(_FakeTranscriber(), lambda _d: None, capture_t0=0.0)
    worker.start()
    report = worker.stop()
    assert report.utterance.count == 0
    assert report.word.count == 0


# --- real model hand-off (opt-in) -------------------------------------------


@pytest.mark.models
def test_live_utterance_then_final_pass_does_not_fall_back() -> None:
    """One live utterance on the worker, then decode_meeting on main works.

    Without the worker-side teardown the final pass hits the F18 stream
    error and silently falls back to the dictation path (decode_meeting
    returns None).
    """
    import soundfile as sf

    from vemoizer.whisper_transcriber import decode_meeting

    audio, sr = sf.read(
        "tests/fixtures/corpus/fi_code_switch_short.wav", dtype="float32"
    )
    assert sr == 16_000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    lines: list[Decoded] = []
    worker = DecodeWorker(WhisperTranscriber(), lines.append, capture_t0=0.0)
    worker.start()
    worker.submit("me", Utterance(start_s=0.0, audio=audio))
    worker.stop()

    result = decode_meeting(audio, [(0, audio)], vad_slices=[(0, len(audio))])
    assert result is not None
    assert result["text"].strip()
