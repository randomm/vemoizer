"""Streaming voice activity detection for the live capture preview (#177).

``StreamingVad.feed`` takes audio chunks as they arrive and returns the
utterances that closed; ``flush`` ends the stream. Each live track owns one
instance and one model. The output is a single-decode preview, never the
deliverable: the post-call pass re-runs the full pipeline on the recording.

Exactness: the probability stream equals ``vad._scan_chunk`` over the whole
audio, and the utterances equal ``vad_segments(min_silence_ms=500,
max_speech_s=10)`` on fixtures shorter than ``MAX_VAD_CHUNK_S``. Both paths
drive the same resumable ``SpeechStateMachine`` one window at a time, so the
stream never re-runs or trims history.
"""

from __future__ import annotations

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.vad import SpeechSegment, SpeechStateMachine, VadModel

#: A closed live utterance, in sample coordinates of the stream.
Utterance = SpeechSegment

#: Silence that closes an utterance (live endpointing).
ENDPOINT_SILENCE_MS = 500

#: Longest utterance before a forced split; keeps each utterance well under
#: the whisper decode window (see tests/test_vad_stream.py).
MAX_UTTERANCE_S = 10.0

# silero-vad window at 16 kHz (32 ms); the model is called once per window.
_WINDOW = 512
_THRESHOLD = 0.5
_MIN_SPEECH_MS = 250
_SPEECH_PAD_MS = 30


class StreamingVad:
    """Incremental silero-vad segmentation with live endpointing."""

    def __init__(self, model: VadModel) -> None:
        self._model = model
        self._leftover = np.zeros(0, dtype=np.float32)
        self._machine = SpeechStateMachine(
            window=_WINDOW,
            sample_rate=SAMPLE_RATE,
            threshold=_THRESHOLD,
            min_speech_ms=_MIN_SPEECH_MS,
            max_speech_s=MAX_UTTERANCE_S,
            min_silence_ms=ENDPOINT_SILENCE_MS,
            speech_pad_ms=_SPEECH_PAD_MS,
        )
        model.reset_states()

    def feed(self, chunk: np.ndarray) -> list[Utterance]:
        """Consume *chunk* (16 kHz mono float32) and return closed utterances."""
        _check_chunk(chunk)
        samples = np.concatenate([self._leftover, chunk])
        full = len(samples) - len(samples) % _WINDOW
        out: list[Utterance] = []
        for i in range(0, full, _WINDOW):
            out.extend(self._push(self._score(samples[i : i + _WINDOW])))
        self._leftover = samples[full:]
        return out

    def flush(self) -> list[Utterance]:
        """End the stream: zero-pad the last partial window and close everything."""
        out: list[Utterance] = []
        if len(self._leftover):
            tail = np.zeros(_WINDOW, dtype=np.float32)
            tail[: len(self._leftover)] = self._leftover
            out.extend(self._push(self._score(tail)))
        self._leftover = np.zeros(0, dtype=np.float32)
        out.extend(Utterance(start=s, end=e) for s, e in self._machine.finish())
        return out

    def _score(self, window: np.ndarray) -> float:
        return float(np.asarray(self._model(window, SAMPLE_RATE)).ravel()[0])

    def _push(self, prob: float) -> list[Utterance]:
        return [Utterance(start=s, end=e) for s, e in self._machine.push(prob)]


def _check_chunk(chunk: np.ndarray) -> None:
    if chunk.ndim != 1:
        raise ValueError(f"expected 1-D mono audio, got shape {chunk.shape}")
    if chunk.dtype != np.float32:
        raise ValueError(
            f"expected float32 audio (16 kHz mono contract), got {chunk.dtype}"
        )
