"""Parakeet TDT 0.6B v3 transcriber for Apple Silicon (decode A).

Implements decode stage A of the consensus pipeline. The model is the
``mlx-community`` MLX port of ``nvidia/parakeet-tdt-0.6b-v3`` (25 languages,
including Finnish, with auto language ID). Model loading is lazy: nothing is
loaded at import time or construction time; the first call to :meth:`transcribe`
triggers a revision-pinned download + load, and that load is logged.

Language ID: Parakeet v3 detects the language internally, but the
``parakeet-mlx`` ``AlignedResult`` API does not surface it, so this module
cannot report per-utterance language. ``language`` is therefore only populated
when the model object happens to expose a ``language`` attribute; per-span
language identification is the job of a dedicated detection pass
(ticket 7), not of this decode stage.
"""

import logging
import threading
import time
from typing import Any

import mlx.core as mx
import numpy as np

from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

# nvidia/parakeet-tdt-0.6b-v3 — the MLX community port. We pin the exact commit
# SHA via snapshot_download (project invariant #4) so upstream pushes cannot
# change the weights we run.
MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"

# Audio contract (project invariant #6): 16 kHz mono float32.
SAMPLE_RATE = 16000


class ParakeetTranscriber:
    """Parakeet TDT 0.6B v3 speech-to-text transcriber using MLX."""

    def __init__(self) -> None:
        self.model: Any = None
        self._model_lock = threading.Lock()
        self._load_once = threading.Lock()

    def _load_model(self) -> None:
        """Download (revision-pinned) and load the Parakeet model.

        Lazy + idempotent: guarded by ``_load_once`` so repeated calls load
        exactly once. Loading is logged because model loads are slow. A load
        failure is logged and leaves ``self.model`` as ``None``; the next
        :meth:`transcribe` then raises ``RuntimeError`` rather than crashing
        the pipeline at import or construction time.
        """
        with self._load_once:
            if self.model is not None:
                return
            logger.info("Loading Parakeet model: %s@%s", MODEL_ID, MODEL_REVISION)
            start = time.time()
            try:
                from huggingface_hub import snapshot_download
                from parakeet_mlx import from_pretrained

                # Revision-pinned: never load from the bare repo ID (invariant #4).
                local_path = snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                )
                self.model = from_pretrained(local_path)
            except Exception as e:  # noqa: BLE001 - logged; model-load guard
                logger.error("Failed to load Parakeet model: %s", e)
                self.model = None
                return
            logger.info("Parakeet model loaded in %.2fs", time.time() - start)

    def _ensure_loaded(self) -> None:
        """Trigger the lazy load on first use."""
        self._load_model()

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> TranscriptionResult:
        """Transcribe audio (float32, mono, 16 kHz) to text with word timestamps."""
        self._ensure_loaded()
        if self.model is None:
            raise RuntimeError("Parakeet model failed to load")

        if len(audio) == 0:
            return {
                "text": "",
                "words": [],
                "segments": [],
                "transcribe_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }

        # Audio contract is float32; coerce any other dtype before the model sees it.
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.time()

        from parakeet_mlx.audio import get_logmel

        with self._model_lock:
            mel = get_logmel(mx.array(audio), self.model.preprocessor_config)
            alignments = self.model.generate(mel)

        transcribe_time = time.time() - start
        audio_duration = len(audio) / SAMPLE_RATE

        text = ""
        language = None
        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []

        if alignments:
            first = alignments[0]
            text = getattr(first, "text", "").strip()
            # The real parakeet-mlx AlignedResult has no ``language`` field
            # (the model does LID internally but does not expose it); report
            # it only if the backend happens to provide it.
            language = getattr(first, "language", None)
            words, segments = _extract_words_segments(alignments)

        result: TranscriptionResult = {
            "text": text,
            "words": words,
            "segments": segments,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }
        if language is not None:
            result["language"] = language
        return result

    def cleanup(self) -> None:
        """Release model resources.

        Takes both locks under ``_load_once`` (outer) and ``_model_lock``
        (inner) so the lock order matches :meth:`transcribe` (``_load_once``
        first inside :meth:`_load_model`, ``_model_lock`` on the inference
        path). New code must not acquire them in the opposite order.
        """
        with self._load_once, self._model_lock:
            if self.model is not None:
                self.model = None
        mx.clear_cache()


def _extract_words_segments(
    alignments: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract per-word timestamps from alignments (one AlignedResult per input).

    ``AlignedResult`` exposes ``tokens`` (a flat list of ``AlignedToken``
    with ``text`` / ``start`` / ``end``) and ``sentences`` (each an
    ``AlignedSentence`` with ``text`` / ``start`` / ``end``). We map them into
    the ``words`` / ``segments`` shape of :class:`TranscriptionResult` so the
    downstream alignment stage (ticket 6) sees one contiguous word stream.
    """
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for aligned in alignments:
        # The real AlignedResult exposes a flat ``tokens`` property (all
        # sentence tokens concatenated). If a backend only exposes
        # per-sentence tokens (no top-level ``tokens`` attribute), derive
        # the flat stream from ``sentences`` instead.
        tokens = getattr(aligned, "tokens", None)
        if tokens is None:
            sentences = getattr(aligned, "sentences", None) or []
            tokens = [
                token
                for sentence in sentences
                for token in (getattr(sentence, "tokens", None) or [])
            ]
        for token in tokens:
            words.append(
                {
                    "word": getattr(token, "text", ""),
                    "start": float(getattr(token, "start", 0.0)),
                    "end": float(getattr(token, "end", 0.0)),
                }
            )

        sentences = getattr(aligned, "sentences", None) or []
        for sentence in sentences:
            segments.append(
                {
                    "start": float(getattr(sentence, "start", 0.0)),
                    "end": float(getattr(sentence, "end", 0.0)),
                    "text": getattr(sentence, "text", ""),
                }
            )

    return words, segments
