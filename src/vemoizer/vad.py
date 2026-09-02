"""Voice activity detection: split 1-60 minute recordings into speech segments.

silero-vad in ONNX mode (``onnxruntime`` only, no torch) samples 16 kHz
audio in 512-sample (32 ms) windows and emits per-window speech
probabilities. The state machine below (``_state_machine``) turns that
probability stream into speech segments using silero's reference
algorithm. Long memos are fed in ``MAX_VAD_CHUNK_S``-second slices so
memory stays bounded.

The 16 kHz mono float32 audio contract (AGENTS.md invariant #6) is
enforced here: silero-vad natively supports 8/16 kHz, and vemoizer's
internal boundary is 16 kHz, so other rates are rejected.

Performance note: each 32 ms window requires its own ONNX inference call
(the LSTM state tensor is sequential and cannot be batched across
windows), so VAD wall-clock scales at ~31.25 inference calls per second
of audio. For a 60-minute memo this is an expected, budgeted cost — the
alternative is trading accuracy (fewer samples per call) or dropping the
window-level resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from vemoizer.audio_contract import SUPPORTED_SAMPLE_RATES

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

#: Internal audio contract (AGENTS.md invariant #6).
CONTRACT_SAMPLE_RATE = 16000

#: Max seconds fed to the VAD model in one pass (memory bound for 60-min memos).
MAX_VAD_CHUNK_S = 60.0

#: silero's reference: max seconds of audio per single inference call.
_MAX_CHUNK_S = 31.25

#: silero-vad ONNX model file shipped inside the ``silero_vad`` pip package.
_MODEL_FILENAME = "silero_vad.onnx"

# silero-vad reference defaults (silero_vad.utils_vad.get_speech_timestamps).
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_MIN_SPEECH_MS = 250
_DEFAULT_MAX_SPEECH_S = 60.0
_DEFAULT_MIN_SILENCE_MS = 100
_DEFAULT_SPEECH_PAD_MS = 30


class VadModel(Protocol):
    """A loaded silero-vad model callable with numpy windows.

    The real object is the ONNX wrapper (torch-free wrapper around an
    ``onnxruntime.InferenceSession``) returned by :func:`load_model`;
    unit tests use fakes implementing the same two members. Nothing else
    in this module imports silero-vad or torch directly.
    """

    def reset_states(self) -> None: ...

    def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeechSegment:
    """A speech slice in sample coordinates of the original recording."""

    start: int
    end: int


class VadModelError(RuntimeError):
    """Raised when the silero-vad ONNX model cannot be loaded."""


class _OnnxModel:
    """Torch-free equivalent of ``silero_vad.utils_vad.OnnxWrapper``.

    Mirrors the reference ONNX path exactly (512/256-sample windows,
    64/32-sample context carry-over, state tensor threaded through the
    graph) but carries state in numpy instead of torch. This keeps
    torch/torchaudio installed (silero-vad declares them as core deps)
    while never importing them at module scope.
    """

    def __init__(self, session: InferenceSession) -> None:
        self._session = session
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: np.ndarray = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def reset_states(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def _maybe_reset(self, batch: int, sample_rate: int) -> None:
        """Reset graph state when the batch size or sample rate changed."""
        if (
            self._last_batch_size == 0
            or self._last_sr != sample_rate
            or self._last_batch_size != batch
        ):
            self.reset_states()

    def _validate_input(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        """Validate and reshape *window* to a 2-D float32 batch."""
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            window = window[None, :]
        if window.ndim != 2:
            raise ValueError(f"Too many dimensions for input audio chunk {window.ndim}")
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Supported sampling rates: {SUPPORTED_SAMPLE_RATES} "
                "(or multiple of 16000)"
            )
        if len(window) / sample_rate > _MAX_CHUNK_S:
            raise ValueError("Input audio chunk is too short")
        return window

    def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
        window = self._validate_input(window, sample_rate)

        num_samples = 512 if sample_rate == 16000 else 256
        if window.shape[1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {window.shape[1]} "
                f"(supported: 256 for 8000 Hz, 512 for 16000 Hz)"
            )

        batch = window.shape[0]
        context_size = 64 if sample_rate == 16000 else 32
        self._maybe_reset(batch, sample_rate)
        if self._state.shape[1] != batch:
            self._state = np.zeros((2, batch, 128), dtype=np.float32)

        if not len(self._context):
            self._context = np.zeros((batch, context_size), dtype=np.float32)

        # Coalesce context + window into one float32 allocation (no extra
        # astype copy): both operands are already float32.
        padded = np.concatenate([self._context, window], axis=1)
        out, state = self._session.run(
            None,
            {
                "input": padded,
                "state": self._state,
                "sr": np.array(sample_rate, dtype=np.int64),
            },
        )
        # `out` has shape [batch, 1] (one probability per row); the slice
        # below drops the trailing probability axis, not the batch axis.
        self._state = np.asarray(state, dtype=np.float32)
        self._context = padded[..., -context_size:].copy()
        self._last_sr = sample_rate
        self._last_batch_size = batch
        return np.asarray(out)[..., 0]


def load_model() -> VadModel:
    """Load the ONNX silero-vad model bundled with the ``silero_vad`` package.

    Model loading is slow; this is the one place the slow imports happen.
    The ONNX graph file ships inside the ``silero_vad.data`` package, so
    no network access is needed once ``pip install silero-vad`` has run.

    Raises:
        VadModelError: if the model file is missing or the ONNX session
            fails to initialise.
    """
    from importlib import resources

    import onnxruntime

    try:
        model_path = str(resources.files("silero_vad.data").joinpath(_MODEL_FILENAME))
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"], sess_options=options
        )
    except Exception as e:
        raise VadModelError(
            f"failed to load ONNX silero-vad model {model_path!r}"
        ) from e
    return _OnnxModel(session)


def vad_segments(
    audio: np.ndarray,
    model: VadModel,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    sample_rate: int = CONTRACT_SAMPLE_RATE,
    min_speech_ms: int = _DEFAULT_MIN_SPEECH_MS,
    max_speech_s: float = _DEFAULT_MAX_SPEECH_S,
    min_silence_ms: int = _DEFAULT_MIN_SILENCE_MS,
    speech_pad_ms: int = _DEFAULT_SPEECH_PAD_MS,
) -> list[SpeechSegment]:
    """Run silero-vad on *audio* and return speech segments in sample units.

    Audio longer than ``MAX_VAD_CHUNK_S`` is processed in slices so memory
    stays bounded on 1-60 minute memos. A segment that starts at a slice
    boundary is merged with the previous slice's last segment (continuous
    speech across the boundary stays one segment), unless the merge would
    exceed the ``max_speech_s`` cap — in that case the previous segment is
    cut at the boundary and the continuation after the boundary is kept as
    a new segment (nothing is dropped).

    Raises:
        ValueError: if *audio* is not 1-D float32, or *sample_rate* is not
            8000/16000 (AGENTS.md invariant #6: internal audio is 16 kHz
            mono float32).
    """
    if audio.ndim != 1:
        raise ValueError(
            f"Expected 1-D mono audio, got shape {audio.shape} (multi-channel?)"
        )
    if audio.dtype != np.float32:
        raise ValueError(
            f"Expected float32 audio (16 kHz mono contract), got {audio.dtype}"
        )
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"sample_rate {sample_rate} Hz not in {SUPPORTED_SAMPLE_RATES}; "
            "resample at ingest (AGENTS.md audio contract)"
        )

    window = 512 if sample_rate == 16000 else 256
    chunk_samples = int(MAX_VAD_CHUNK_S * sample_rate)

    segments: list[list[int]] = []
    model.reset_states()
    base = 0
    while base < len(audio):
        chunk = audio[base : base + chunk_samples]
        probs = _scan_chunk(chunk, model, sample_rate=sample_rate, window=window)
        found = _state_machine(
            probs,
            window=window,
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            max_speech_s=max_speech_s,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        # If speech reached the start of this slice, merge with the
        # previous slice's last segment (continuous speech across the
        # boundary stays one segment), unless the merge would exceed the
        # max_speech_s cap — in that case the previous segment is cut
        # at the boundary and the tail is kept as a new segment, so the
        # speech after the boundary is not dropped.
        if found and found[0][0] <= 0 and segments:
            merged_end = found[0][1] + base
            max_speech_samples = int(sample_rate * max_speech_s)
            if merged_end - segments[-1][0] <= max_speech_samples:
                segments[-1][1] = merged_end
                found = found[1:]
            else:
                segments[-1][1] = base
                segments.append([base, merged_end])
                found = found[1:]
        for start, end in found:
            segments.append([start + base, end + base])
        base += chunk_samples

    return [SpeechSegment(start=int(s), end=int(e)) for s, e in segments]


def _scan_chunk(
    chunk: np.ndarray, model: VadModel, *, sample_rate: int, window: int
) -> list[float]:
    """Feed *chunk* to *model* window by window; return speech probabilities.

    The LSTM state is sequential, so each 32 ms window is its own ONNX
    call; this loop is the accepted cost for 60-min memos (see module
    docstring).
    """
    probs: list[float] = []
    for i in range(0, len(chunk), window):
        w = chunk[i : i + window]
        if len(w) < window:
            w = np.concatenate([w, np.zeros(window - len(w), dtype=np.float32)])
        out = model(w, sample_rate)
        probs.append(float(np.asarray(out).ravel()[0]))
    return probs


def _state_machine(
    probs: list[float],
    *,
    window: int,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    max_speech_s: float,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[tuple[int, int]]:
    """silero-vad 6.2.1 reference state machine, numpy-only.

    Mirrors ``silero_vad.utils_vad.get_speech_timestamps``: trigger at
    ``prob >= threshold``, exit at ``prob < neg_threshold`` held for
    ``min_silence_ms``, split segments longer than ``max_speech_s`` at the
    longest qualifying internal silence (else just before the cap), drop
    segments shorter than ``min_speech_ms``, and pad both ends by
    ``speech_pad_ms``.
    """
    min_speech = int(sample_rate * min_speech_ms / 1000)
    min_silence = int(sample_rate * min_silence_ms / 1000)
    speech_pad = int(sample_rate * speech_pad_ms / 1000)
    max_speech = (
        window * 10**9
        if max_speech_s == float("inf")
        else int(sample_rate * max_speech_s) - window - 2 * speech_pad
    )
    neg_threshold = max(threshold - 0.15, 0.01)
    min_silence_at_max = int(sample_rate * 98 / 1000)

    triggered = False
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []
    current_start = 0
    out: list[tuple[int, int]] = []

    def _clear_split_state() -> None:
        nonlocal prev_end, next_start, temp_end, possible_ends
        prev_end = next_start = temp_end = 0
        possible_ends = []

    for i, p in enumerate(probs):
        cur = window * i

        if p >= threshold and temp_end:
            sil = cur - temp_end
            if sil > min_silence_at_max:
                possible_ends.append((temp_end, sil))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur

        if p >= threshold and not triggered:
            triggered = True
            current_start = cur
            continue

        if triggered and (cur - current_start > max_speech):
            if possible_ends:
                prev_end, dur = max(possible_ends, key=lambda t: t[1])
                out.append((current_start, prev_end))
                next_start = prev_end + dur
                if next_start < prev_end + cur:
                    current_start = next_start
                else:
                    triggered = False
            elif prev_end:
                out.append((current_start, prev_end))
                if next_start < prev_end:
                    current_start = next_start
                else:
                    triggered = False
            else:
                out.append((current_start, cur))
                triggered = False
            _clear_split_state()
            if not triggered:
                continue

        if p < neg_threshold and triggered:
            if not temp_end:
                temp_end = cur
            if cur - temp_end >= min_silence:
                out.append((current_start, temp_end))
                triggered = False
                temp_end = 0
                possible_ends = []

    if triggered:
        out.append((current_start, len(probs) * window))

    total = len(probs) * window
    kept: list[tuple[int, int]] = []
    for start, end in out:
        if end - start < min_speech:
            continue
        start = max(0, start - speech_pad)
        end = min(total, end + speech_pad)
        kept.append((start, end))
    return kept
