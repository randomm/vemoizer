"""Canary-1b-v2 transcriber (decode B) using the MLX port.

Implements decode stage B of the consensus pipeline. The model is the
community MLX port ``Mediform/canary-1b-v2-mlx-q8`` — a direct-safetensors
q8 quantization of ``nvidia/canary-1b-v2``. It is **not** loaded via
``mlx-audio`` (the canonical package does not support Canary); instead the
architecture and weight loading live in :mod:`vemoizer.canary_mlx`.

Model loading is lazy and revision-pinned (project invariant #4): nothing is
loaded at import or construction time; the first call to :meth:`transcribe`
triggers a ``snapshot_download(repo_id, revision=<sha>)`` and loads from the
returned local path. Loading is logged because model loads are slow. A load
failure is logged and leaves ``self.model`` as ``None``; the next
:meth:`transcribe` then raises ``RuntimeError`` rather than crashing the
pipeline at import or construction time.

Language: Canary-1b-v2 transcribes in a fixed target language and does not
expose per-utterance language, so ``language`` is only populated when the
backend reports one (project invariant #3: language is a property of a span,
never a hardcoded file value). Word timestamps are not available from this
port, so the result is text-only (``words``/``segments`` are optional in
``TranscriptionResult``).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .canary_mlx import SAMPLE_RATE, compute_features, load_canary_weights
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

# The community MLX q8 port of nvidia/canary-1b-v2. We pin the exact commit
# SHA via snapshot_download (project invariant #4) so upstream pushes cannot
# change the weights we run.
MODEL_ID = "Mediform/canary-1b-v2-mlx-q8"
MODEL_REVISION = "0b6b32ee10f30c89e3ead7249bb636445e3019ee"


class CanaryTranscriber:
    """Canary-1b-v2 speech-to-text transcriber using the MLX port."""

    def __init__(self) -> None:
        self.model: Any = None
        self._load_once = threading.Lock()

    def _load_model(self) -> None:
        """Download (revision-pinned) and load the Canary model.

        Lazy + idempotent: guarded by ``_load_once`` so repeated calls load
        exactly once. Loading is logged because model loads are slow. A load
        failure is logged and leaves ``self.model`` as ``None``; the next
        :meth:`transcribe` then raises ``RuntimeError`` rather than crashing
        the pipeline at import or construction time.
        """
        with self._load_once:
            if self.model is not None:
                return
            logger.info("Loading Canary model: %s@%s", MODEL_ID, MODEL_REVISION)
            start = time.time()
            try:
                from huggingface_hub import snapshot_download

                # Revision-pinned: never load from the bare repo ID (invariant #4).
                local_path = snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                )
                self.model = load_canary_weights(local_path)
            except Exception as e:  # noqa: BLE001 - logged; model-load guard
                logger.error("Failed to load Canary model: %s", e)
                self.model = None
                return
            logger.info("Canary model loaded in %.2fs", time.time() - start)

    def _ensure_loaded(self) -> None:
        """Trigger the lazy load on first use."""
        self._load_model()

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> TranscriptionResult:
        """Transcribe audio (float32, mono, 16 kHz) to text (decode B).

        ``language`` is only reported when the backend provides it; Canary
        does not, so it is omitted. Word timestamps are not available from
        this port, so ``words``/``segments`` are empty.
        """
        self._ensure_loaded()
        if self.model is None:
            raise RuntimeError("Canary model failed to load")

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

        import mlx.core as mx

        mel = compute_features(audio, dtype=mx.bfloat16)
        prompt_ids = self.model.tokenizer.build_prompt("en", "en")
        text = self.model.generate(mel, prompt_ids)

        transcribe_time = time.time() - start
        audio_duration = len(audio) / SAMPLE_RATE

        result: TranscriptionResult = {
            "text": text,
            "words": [],
            "segments": [],
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }
        # Populate language only if the backend happens to report one.
        language = getattr(self.model, "language", None)
        if language is not None:
            result["language"] = language
        return result

    def cleanup(self) -> None:
        """Release model resources."""
        with self._load_once:
            if self.model is not None:
                self.model = None
