"""Tests for src/vemoizer/vad_stream.py (live streaming VAD, issue #177).

Pure-python: scripted fake VAD models supply deterministic per-window
speech probabilities (no onnxruntime, no network, no audio hardware).
The fakes record what they emitted so the streaming path can be compared
window-for-window with the reference batch path.
"""

from __future__ import annotations

import numpy as np
import pytest

from vemoizer.vad import SpeechSegment, _scan_chunk, vad_segments
from vemoizer.vad_stream import (
    ENDPOINT_SILENCE_MS,
    MAX_UTTERANCE_S,
    StreamingVad,
)

WINDOW = 512
SR = 16000


class ScriptedModel:
    """Fake VadModel: call *i* returns ``probs[i]`` (0.0 past the script)."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.calls = 0
        self.resets = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        p = self._probs[self.calls] if self.calls < len(self._probs) else 0.0
        self.calls += 1
        return np.array([p], dtype=np.float32)


class StatefulModel:
    """Fake whose output depends on the call counter and window content.

    A ``reset_states`` landing mid-stream zeroes the counter, so the output
    sequence diverges from an uninterrupted run. ``outputs`` records every
    probability emitted, in call order.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0
        self.outputs: list[float] = []

    def reset_states(self) -> None:
        self.resets += 1
        self.calls = 0

    def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        value = (self.calls * 0.137 + float(np.abs(window).mean()) * 7.0) % 1.0
        self.outputs.append(value)
        return np.array([value], dtype=np.float32)


def _speech(n_windows: int, p: float = 0.9) -> list[float]:
    return [p] * n_windows


def _silence(n_windows: int) -> list[float]:
    return [0.0] * n_windows


def _feed_all(vad: StreamingVad, audio: np.ndarray, chunk: int) -> list[SpeechSegment]:
    out: list[SpeechSegment] = []
    for i in range(0, len(audio), chunk):
        out.extend(vad.feed(audio[i : i + chunk]))
    out.extend(vad.flush())
    return out


def _as_pairs(utts: list[SpeechSegment]) -> list[tuple[int, int]]:
    return [(u.start, u.end) for u in utts]


# ---------------------------------------------------------------------------
# Public API contract
# ---------------------------------------------------------------------------


def test_max_utterance_is_shorter_than_whisper_window():
    from vemoizer.whisper_transcriber import WINDOW_SECONDS

    assert MAX_UTTERANCE_S < WINDOW_SECONDS


def test_endpoint_silence_is_500_ms():
    assert ENDPOINT_SILENCE_MS == 500


def test_rejects_float64_chunk():
    vad = StreamingVad(ScriptedModel([]))
    with pytest.raises(ValueError, match="float32"):
        vad.feed(np.zeros(WINDOW, dtype=np.float64))


def test_rejects_stereo_chunk():
    vad = StreamingVad(ScriptedModel([]))
    with pytest.raises(ValueError, match="mono"):
        vad.feed(np.zeros((WINDOW, 2), dtype=np.float32))


def test_constructor_resets_model_once_and_feed_never_resets():
    model = StatefulModel()
    vad = StreamingVad(model)
    assert model.resets == 1
    vad.feed(np.zeros(WINDOW * 3, dtype=np.float32))
    vad.feed(np.zeros(WINDOW * 3, dtype=np.float32))
    vad.flush()
    assert model.resets == 1


# ---------------------------------------------------------------------------
# Probability stream equals the batch scan for any chunking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 37, 511, 512, 513, 4000])
def test_probabilities_match_batch_scan_for_any_chunking(chunk: int):
    rng = np.random.default_rng(11)
    audio = rng.uniform(-0.5, 0.5, size=WINDOW * 9 + 123).astype(np.float32)

    reference = StatefulModel()
    expected = _scan_chunk(audio, reference, sample_rate=SR, window=WINDOW)

    streamed = StatefulModel()
    _feed_all(StreamingVad(streamed), audio, chunk=chunk)

    assert streamed.outputs == pytest.approx(expected)
    assert streamed.resets == 1


# ---------------------------------------------------------------------------
# Endpointing: 500 ms of silence closes an utterance; nothing before
# ---------------------------------------------------------------------------


def test_no_utterance_before_500_ms_silence_then_one_after():
    speech_windows = 32  # 1 s of speech
    probs = _speech(speech_windows) + _silence(200)
    audio = np.zeros(len(probs) * WINDOW, dtype=np.float32)
    vad = StreamingVad(ScriptedModel(probs))

    speech_end = speech_windows * WINDOW
    silence_needed = speech_end + SR * ENDPOINT_SILENCE_MS // 1000
    fed = 0
    emitted: list[tuple[int, int]] = []
    for i in range(0, len(audio), WINDOW):
        fed += WINDOW
        got = _as_pairs(vad.feed(audio[i : i + WINDOW]))
        if got:
            assert fed >= silence_needed
            emitted.extend(got)
    assert len(emitted) == 1
    start, end = emitted[0]
    assert start == 0
    assert end == speech_end + 480  # 30 ms speech pad


def test_flush_closes_open_utterance():
    probs = _speech(40)
    audio = np.zeros(len(probs) * WINDOW, dtype=np.float32)
    vad = StreamingVad(ScriptedModel(probs))
    assert vad.feed(audio) == []
    tail = vad.flush()
    assert len(tail) == 1
    assert tail[0].end == len(audio)


# ---------------------------------------------------------------------------
# Long speech: utterances capped at MAX_UTTERANCE_S, contiguous, none dropped
# ---------------------------------------------------------------------------


def test_forty_seconds_of_speech_is_capped_contiguous_and_complete():
    n = 40 * SR
    audio = np.zeros(n, dtype=np.float32)
    vad = StreamingVad(ScriptedModel(_speech(n // WINDOW + 1)))
    utts = _feed_all(vad, audio, chunk=4000)

    assert len(utts) >= 4
    cap = int(MAX_UTTERANCE_S * SR)
    for u in utts:
        assert u.end - u.start <= cap
    assert utts[0].start == 0
    # A trailing fragment shorter than min_speech (250 ms) is dropped by the
    # reference state machine too; everything longer is kept.
    assert n - utts[-1].end < SR * 250 // 1000
    for prev, nxt in zip(utts, utts[1:], strict=False):
        assert nxt.start <= prev.end


# ---------------------------------------------------------------------------
# Equivalence with the batch path on fixtures shorter than MAX_VAD_CHUNK_S
# ---------------------------------------------------------------------------


def _bursty_probs(seed: int, windows: int) -> list[float]:
    """Random speech/silence runs, including pauses around the 500 ms endpoint."""
    rng = np.random.default_rng(seed)
    probs: list[float] = []
    while len(probs) < windows:
        kind = int(rng.integers(0, 3))
        length = int(rng.integers(2, 60))
        if kind == 0:
            probs += [float(rng.uniform(0.0, 0.2))] * length
        elif kind == 1:
            probs += [float(rng.uniform(0.55, 1.0))] * length
        else:
            probs += rng.uniform(0.0, 1.0, size=length).tolist()
    return probs[:windows]


def _speech_with_short_pauses(seed: int, windows: int) -> list[float]:
    """Long speech runs separated by pauses shorter than the 500 ms endpoint.

    Every 10 s cap lands on a pause that the reference records as a possible
    split point, so the split path is the one under test.
    """
    rng = np.random.default_rng(seed)
    probs: list[float] = []
    while len(probs) < windows:
        length = int(rng.integers(5, 81))
        probs += [float(rng.uniform(0.6, 1.0))] * length
        pause = int(rng.integers(2, 15))  # 64-448 ms, under ENDPOINT_SILENCE_MS
        probs += [float(rng.uniform(0.0, 0.2))] * pause
    return probs[:windows]


def _assert_streaming_equals_batch(probs: list[float], chunk: int) -> None:
    audio = np.zeros(len(probs) * WINDOW, dtype=np.float32)
    batch = vad_segments(
        audio,
        ScriptedModel(probs),
        min_silence_ms=ENDPOINT_SILENCE_MS,
        max_speech_s=MAX_UTTERANCE_S,
    )
    streamed = _feed_all(StreamingVad(ScriptedModel(probs)), audio, chunk=chunk)

    assert _as_pairs(streamed) == [(s.start, s.end) for s in batch]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 6, 12, 19, 43, 61, 75, 88])
def test_streaming_segments_equal_batch_vad_segments(seed: int):
    # ~28 s, below MAX_VAD_CHUNK_S. Seeds 6..88 split at a sub-500 ms pause
    # and used to diverge once the buffer was trimmed after the split.
    _assert_streaming_equals_batch(_bursty_probs(seed, 900), chunk=4000)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5])
def test_long_speech_with_short_pauses_equals_batch(seed: int):
    _assert_streaming_equals_batch(_speech_with_short_pauses(seed, 900), chunk=4000)


@pytest.mark.parametrize("chunk", [512, 4000, 16000])
def test_short_pause_split_equal_for_any_chunking(chunk: int):
    _assert_streaming_equals_batch(_speech_with_short_pauses(1, 900), chunk=chunk)
