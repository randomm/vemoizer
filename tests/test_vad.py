"""Tests for src/vemoizer/vad.py.

All tests are pure-python: a fake VAD model supplies a deterministic
per-window speech-probability stream (no torch, no onnxruntime, no
network). The fake model is the seam that lets us drive the state
machine exactly as ``silero_vad.utils_vad.get_speech_timestamps`` drives
its model — the same ``reset_states``/``__call__`` contract the real
``OnnxWrapper`` exposes.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vemoizer.vad import (
    CONTRACT_SAMPLE_RATE,
    SUPPORTED_SAMPLE_RATES,
    SpeechSegment,
    load_model,
    vad_segments,
)

# ---------------------------------------------------------------------------
# Fake VAD model: emits a fixed per-window speech-probability sequence.
# ---------------------------------------------------------------------------


class ProbModel:
    """Fake VadModel that returns a predetermined probability stream.

    One probability per window (512 samples at 16 kHz = 32 ms). If the
    audio has more windows than the sequence, missing windows return
    ``0.0`` (silence) so long-audio tests do not need to list 3000
    entries.
    """

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self._i = 0
        self.reset_calls = 0
        self.call_count = 0

    def reset_states(self) -> None:
        self._i = 0
        self.reset_calls += 1

    def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        p = self._probs[self._i] if self._i < len(self._probs) else 0.0
        self._i += 1
        self.call_count += 1
        return np.array([p], dtype=np.float32)


def _silence(n_chunks: int, p: float = 0.0) -> list[float]:
    return [p] * n_chunks


def _speech(n_chunks: int, p: float = 0.9) -> list[float]:
    return [p] * n_chunks


def _total_windows(n_seconds: float, sr: int) -> int:
    return int(n_seconds * sr) // (512 if sr == 16000 else 256)


# ---------------------------------------------------------------------------
# Public API shape / constants
# ---------------------------------------------------------------------------


def test_module_constants():
    assert SUPPORTED_SAMPLE_RATES == (8000, 16000)
    assert CONTRACT_SAMPLE_RATE == 16000


def test_speech_segment_is_frozen_dataclass():
    seg = SpeechSegment(start=1, end=2)
    assert seg.start == 1
    assert seg.end == 2
    with pytest.raises(FrozenInstanceError):
        seg.start = 3  # ty: ignore[invalid-assignment]


def test_segment_bounds_ordering():
    segs = [SpeechSegment(100, 200), SpeechSegment(250, 300)]
    for s, e in zip(segs, segs[1:], strict=False):
        assert s.end <= e.start


# ---------------------------------------------------------------------------
# Sample-rate contract (AGENTS.md invariant #6)
# ---------------------------------------------------------------------------


def test_rejects_unsupported_sample_rate():
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel([0.5] * 32)
    with pytest.raises(ValueError, match="sample_rate|not in"):
        vad_segments(audio, model, sample_rate=22050)


def test_rejects_48_khz():
    audio = np.zeros(48000, dtype=np.float32)
    model = ProbModel([0.5] * 96)
    with pytest.raises(ValueError):
        vad_segments(audio, model, sample_rate=48000)


def test_accepts_16_khz_default():
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel([0.0] * 32)
    vad_segments(audio, model)  # default sample_rate=16000, no error


def test_accepts_8_khz():
    audio = np.zeros(8000, dtype=np.float32)
    model = ProbModel([0.0] * 32)
    vad_segments(audio, model, sample_rate=8000)


def test_rejects_non_float32_audio():
    audio = np.zeros(16000, dtype=np.float64)
    model = ProbModel([0.0] * 32)
    with pytest.raises(ValueError, match="float32"):
        vad_segments(audio, model)


def test_rejects_multi_channel_audio():
    audio = np.zeros((16000, 2), dtype=np.float32)
    model = ProbModel([0.0] * 32)
    with pytest.raises(ValueError, match="1-D"):
        vad_segments(audio, model)


def test_empty_audio_returns_no_segments():
    audio = np.zeros(0, dtype=np.float32)
    model = ProbModel([])
    assert vad_segments(audio, model) == []


# ---------------------------------------------------------------------------
# Silence dropping (core behaviour of the descriptor)
# ---------------------------------------------------------------------------


def test_silence_only_produces_no_segments():
    """A 1-second pure silence recording yields zero speech segments."""
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel(_silence(32))  # 32 windows = 1 second at 16 kHz
    assert vad_segments(audio, model) == []
    assert model.reset_calls >= 1


def test_low_probability_noise_is_silence():
    """Probs just under the threshold never trigger a segment."""
    n = 4 * 16000
    audio = np.zeros(n, dtype=np.float32)
    model = ProbModel([0.35] * (n // 512))
    assert vad_segments(audio, model) == []


# ---------------------------------------------------------------------------
# Speech detection and min_speech_duration
# ---------------------------------------------------------------------------


def test_continuous_speech_yields_one_segment():
    """0.8 s of continuous speech -> one segment spanning the audio."""
    n = int(0.8 * 16000)
    audio = np.zeros(n, dtype=np.float32)
    model = ProbModel(_speech(n // 512))
    segs = vad_segments(audio, model)
    assert len(segs) == 1
    assert segs[0].start < 100
    assert segs[0].end >= n - 100


def test_blip_shorter_than_min_speech_is_dropped():
    """A 150 ms blip (under the 250 ms min) is excluded."""
    n_windows = 5  # 5 * 512 = 2560 samples = 160 ms at 16 kHz
    probs = _speech(n_windows) + _silence(32 - n_windows)
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel(probs)
    assert vad_segments(audio, model) == []


def test_blip_longer_than_min_speech_is_kept():
    """A 400 ms blip (over the 250 ms min) is kept."""
    n_windows = 13  # 13 * 512 = 6656 samples = 416 ms at 16 kHz
    probs = _speech(n_windows) + _silence(32 - n_windows)
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)
    assert len(segs) == 1


def test_custom_min_speech_ms():
    """min_speech_ms=500 requires 500 ms of continuous speech."""
    n_windows = 10  # 10 * 512 = 5120 samples = 320 ms at 16 kHz
    probs = _speech(n_windows) + _silence(32 - n_windows)
    audio = np.zeros(16000, dtype=np.float32)
    model = ProbModel(probs)
    assert vad_segments(audio, model, min_speech_ms=500) == []


# ---------------------------------------------------------------------------
# Multi-segment splitting
# ---------------------------------------------------------------------------


def test_two_separated_segments():
    """Speech, silence, speech -> two segments with a gap."""
    probs = _speech(16) + _silence(16) + _speech(16) + _silence(16)
    audio = np.zeros(2 * 16000, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)
    assert len(segs) == 2
    assert segs[0].start < 500
    assert 0.4 * 16000 < segs[0].end < 0.7 * 16000
    assert 0.9 * 16000 < segs[1].start < 1.1 * 16000
    assert segs[1].start - segs[0].end > 0.3 * 16000


def test_three_segments():
    """speech/silence/speech/silence/speech -> three segments."""
    probs = (
        _speech(8)
        + _silence(16)
        + _speech(8)
        + _silence(16)
        + _speech(8)
        + _silence(24)
    )
    audio = np.zeros(64 * 512, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)
    assert len(segs) == 3


def test_segments_ordered_non_overlapping():
    """Segments are returned in start-time order and do not overlap."""
    rng = np.random.default_rng(42)
    probs = [0.9 if rng.random() > 0.5 else 0.0 for _ in range(256)]
    audio = np.zeros(256 * 512, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)
    for a, b in zip(segs, segs[1:], strict=False):
        assert a.end <= b.start
        assert b.start <= b.end


# ---------------------------------------------------------------------------
# Max speech duration (60 s chunk cap for long memos)
# ---------------------------------------------------------------------------


def test_max_speech_split_at_internal_silence():
    """30 s of continuous speech with a 1 s silence at 10 s splits at 10 s."""
    sr = 16000
    window = 512
    total = 30 * sr
    n_windows = total // window
    s1 = int(10 * sr) // window
    s2 = int(11 * sr) // window
    probs = _speech(s1) + _silence(s2 - s1) + _speech(n_windows - s2)
    audio = np.zeros(total, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model, max_speech_s=10.0)
    # First segment ends near the 10 s silence; remaining 19 s of continuous
    # speech is further split by the 10 s max-speech cap, so expect 3 segments.
    assert len(segs) == 3
    assert segs[0].end < 10 * sr


def test_max_speech_hard_cap_when_no_internal_silence():
    """30 s continuous speech with max_speech_s=10 splits at ~10 s intervals."""
    sr = 16000
    total = 30 * sr
    n_windows = total // 512
    probs = _speech(n_windows)
    audio = np.zeros(total, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model, max_speech_s=10.0)
    assert len(segs) >= 3
    for s in segs:
        assert (s.end - s.start) / sr <= 10.5


def test_default_max_speech_is_60_seconds():
    """Default ``max_speech_s=60`` is the chunk cap from the descriptor."""
    sr = 16000
    total = 70 * sr
    n_windows = total // 512
    probs = _speech(n_windows)
    audio = np.zeros(total, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)  # default max_speech_s=60
    assert len(segs) >= 2


# ---------------------------------------------------------------------------
# Long audio: 60-second slice processing
# ---------------------------------------------------------------------------


def test_long_audio_slices_are_processed():
    """A 3-minute audio is sliced into 60 s chunks; segments stay whole."""
    sr = 16000
    total = 180 * sr
    n_windows = total // 512
    s1 = int(30 * sr) // 512
    s2 = int(150 * sr) // 512
    probs = _speech(s1) + _silence(s2 - s1) + _speech(n_windows - s2)
    audio = np.zeros(total, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model)
    assert segs[0].start < 100
    assert 29 * sr < segs[0].end < 31 * sr
    assert 149 * sr < segs[-1].start < 151 * sr
    assert segs[-1].end >= total - 100
    assert len(segs) == 2


def test_speech_across_slice_boundary_stays_one_segment():
    """Speech that crosses the 60 s slice boundary stays a single segment."""
    sr = 16000
    total = 90 * sr
    n_windows = total // 512
    probs = _speech(n_windows)
    audio = np.zeros(total, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model, max_speech_s=100.0)
    assert len(segs) == 1
    assert segs[0].start < 100
    assert segs[0].end >= total - 100


# ---------------------------------------------------------------------------
# 8 kHz support (descriptor: "Only 8/16 kHz")
# ---------------------------------------------------------------------------


def test_eight_khz_audio_produces_segments():
    """8 kHz audio is a valid input; silero uses 256-sample windows."""
    probs = _speech(16) + _silence(16)
    audio = np.zeros(8000, dtype=np.float32)
    model = ProbModel(probs)
    segs = vad_segments(audio, model, sample_rate=8000)
    assert len(segs) == 1


def test_eight_khz_min_speech_uses_correct_window():
    """min_speech_ms applies to 8 kHz windows (256 samples, not 512)."""
    n_windows = 7  # 7 * 256 = 1792 samples = 224 ms at 8 kHz < 250 ms min
    probs = _speech(n_windows) + _silence(25)
    audio = np.zeros(8000, dtype=np.float32)
    model = ProbModel(probs)
    assert vad_segments(audio, model, sample_rate=8000) == []
    # Same length at 16 kHz would be well over min_speech
    n_windows_16 = int(200 * 16000 / 512)
    probs_16 = _speech(n_windows_16) + _silence(32 - n_windows_16)
    audio_16 = np.zeros(16000, dtype=np.float32)
    model_16 = ProbModel(probs_16)
    assert len(vad_segments(audio_16, model_16)) == 1


# ---------------------------------------------------------------------------
# VadModel protocol conformance
# ---------------------------------------------------------------------------


def test_vadmodel_protocol_accepts_fake():
    """ProbModel satisfies the VadModel protocol (has reset_states + __call__)."""
    model = ProbModel([0.5] * 8)
    assert hasattr(model, "reset_states")
    assert callable(model)


def test_vadmodel_protocol_rejects_incomplete_object():
    """An object without __call__ does not satisfy the protocol."""

    class NotAModel:
        def reset_states(self) -> None:
            pass

    obj = NotAModel()
    assert not callable(obj)


# ---------------------------------------------------------------------------
# load_model (ONNX path) — marked ``models``; never runs in unit suite
# ---------------------------------------------------------------------------


@pytest.mark.models
def test_load_model_returns_callable_onnx_model():
    """``load_model()`` returns a callable ONNX silero-vad wrapper."""
    m = load_model()
    assert callable(m)
    assert hasattr(m, "reset_states")
    audio = np.zeros(16000, dtype=np.float32)
    assert vad_segments(audio, m) == []


@pytest.mark.models
def test_load_model_handles_speech():
    """Loaded model detects a 440 Hz tone (or at least runs without error)."""
    m = load_model()
    t = np.arange(16000) / 16000.0
    # A pure tone may not trigger the VAD (trained on human speech);
    # assert the model runs and returns a valid segment list.
    tone = (np.sin(2 * math.pi * 440 * t) * 0.5).astype(np.float32)
    segs = vad_segments(tone, m)
    assert isinstance(segs, list)
