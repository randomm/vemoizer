"""Whisper re-decode of disputed spans (issue #8).

Re-decode stage of the consensus pipeline: takes the disputed
``Span``s flagged by :mod:`vemoizer.spans` and re-decodes only those
slices with ``Finnish-NLP/whisper-large-finnish-v3`` through
``mlx-whisper``. This is what keeps the third model affordable —
disputed spans are seconds long, not minutes.

Design decisions:

- **Targeted only (cost invariant).** ``redecode`` makes exactly one
  ``transcribe`` call per non-empty span and zero when given no spans or
  an empty recording. Never re-decodes the whole file.
- **Slice extraction is a pure function.** :func:`extract_slice` maps a
  ``Span`` to the sample range of a 16 kHz mono float32 buffer and
  returns a new array. No model, no side effects — trivially testable.
- **Revision-pinned load (invariant #4).** First ``transcribe`` triggers
  ``snapshot_download(MODEL_ID, revision=MODEL_REVISION)`` and passes the
  returned *local path* to ``mlx_whisper.transcribe`` — never the bare
  repo ID. ``mlx-whisper`` is imported lazily inside the load, so
  importing this module does not require it to be installed.
- **Fail-open per span.** A ``transcribe`` failure (load failure,
  decode error, timeout) logs the error and returns a *degraded* result
  for that span (empty ``text``) instead of raising. The adjudication
  stage still receives candidates A and B for the span; a re-decode
  failure must never abort the run.
- **Language.** Each slice is decoded with ``language="fi"``: the memo
  is Finnish with English seeping in, and Whisper detects English
  segments inside a Finnish utterance on its own — pinning the whole
  file to one language would be a bug (invariant #3), but the
  re-decode is a targeted third opinion on Finnish prose.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spans import Span
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

#: The Finnish fine-tune of Whisper-large, reached through mlx-whisper.
MODEL_ID = "Finnish-NLP/whisper-large-finnish-v3"
#: Pinned commit on the model repo so upstream pushes cannot change the
#: weights we run (project invariant #4).
MODEL_REVISION = "b23deb0b3855c829ffe04cb1c6709757ff16d49c"

#: Sample rate of the audio contract (project invariant #6).
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ReDecodeResult:
    """Outcome of re-decoding one disputed span.

    Attributes:
        span: The disputed span that was re-decoded.
        text: Whisper's transcription of the slice; ``""`` when the
            re-decode failed (fail-open).
        words: Word-level timestamp dicts, each with times *shifted back
            to the original recording timeline* (not slice-relative).
        ok: ``False`` when the re-decode failed for this span. The
            surrounding pipeline must still proceed (fail-open).
    """

    span: Span
    text: str
    words: list[dict[str, Any]]
    ok: bool


def extract_slice(
    audio: np.ndarray, span: Span, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Cut the ``span`` out of a 16 kHz mono float32 recording.

    The span ``[start, end)`` in seconds maps to the sample range
    ``[int(start * rate), min(int(end * rate), len(audio)))`` and a new
    float32 array is returned (a copy, not a view), so callers can keep
    the full recording intact.

    Out-of-bounds or empty ranges yield an empty array rather than an
    error: a span past the end of the buffer (e.g. word timestamps that
    run past ``len(audio)/rate``) simply has no audio to re-decode.
    """
    start_idx = int(np.clip(int(span.start * sample_rate), 0, len(audio)))
    end_idx = int(np.clip(int(span.end * sample_rate), 0, len(audio)))
    end_idx = max(end_idx, start_idx)
    return audio[start_idx:end_idx].copy()


def _to_result(span: Span, raw: dict[str, Any]) -> ReDecodeResult:
    """Convert an ``mlx_whisper.transcribe`` dict to a span-anchored result.

    Whisper reports word times relative to the *slice*; shift them back
    onto the original recording timeline so they line up with the
    disputed-span bookkeeping.
    """
    words: list[dict[str, Any]] = []
    for w in raw.get("words") or []:
        shifted = dict(w)
        if shifted.get("start") is not None:
            shifted["start"] = float(shifted["start"]) + span.start
        if shifted.get("end") is not None:
            shifted["end"] = float(shifted["end"]) + span.start
        words.append(shifted)
    return ReDecodeResult(
        span=span,
        text=str(raw.get("text", "")).strip(),
        words=words,
        ok=True,
    )


class WhisperReDecodeTranscriber:
    """Re-decodes disputed spans with whisper-large-finnish-v3 (mlx-whisper).

    Model loading is lazy and revision-pinned (invariant #4): nothing is
    downloaded at import or construction time; the first span triggers
    ``snapshot_download`` and ``mlx_whisper.load_model`` from the returned
    local path. A failed load leaves ``self.model`` ``None`` and every
    subsequent span fails open with an empty ``ReDecodeResult``.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self._model_path: str | None = None
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Download (revision-pinned) and load the Whisper model once."""
        if self._loaded:
            return
        self._loaded = True
        logger.info("Loading re-decode model: %s@%s", MODEL_ID, MODEL_REVISION)
        start = time.time()
        try:
            import mlx_whisper
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
            )
            self.model = mlx_whisper.load_model(local_path)
            self._model_path = local_path
        except Exception as e:  # noqa: BLE001 - logged; re-decode fails open
            logger.error("Failed to load re-decode model: %s", e)
            self.model = None
            return
        logger.info("Re-decode model loaded in %.2fs", time.time() - start)

    def transcribe_span(self, audio: np.ndarray, span: Span) -> ReDecodeResult:
        """Re-decode one disputed slice; fail open on any error.

        ``audio`` is the full 16 kHz mono float32 recording; only the
        ``span`` range is fed to the model. Returns a
        :class:`ReDecodeResult` with ``ok=False`` and empty ``text``
        when the model could not run (load or decode failure).
        """
        self._ensure_loaded()
        if self.model is None:
            logger.warning(
                "Re-decode unavailable; failing open for span [%0.2f, %0.2f)s",
                span.start,
                span.end,
            )
            return ReDecodeResult(span=span, text="", words=[], ok=False)

        if len(audio) == 0:
            return ReDecodeResult(span=span, text="", words=[], ok=True)

        slice_audio = extract_slice(audio, span)
        if slice_audio.size == 0:
            return ReDecodeResult(span=span, text="", words=[], ok=True)
        if slice_audio.dtype != np.float32:
            slice_audio = slice_audio.astype(np.float32)

        try:
            import mlx_whisper

            raw = mlx_whisper.transcribe(
                slice_audio,
                path_or_hf_repo=self._model_path,
                word_timestamps=True,
                language="fi",
                task="transcribe",
            )
        except Exception as e:  # noqa: BLE001 - logged; re-decode fails open
            logger.warning(
                "Re-decode failed for span [%0.2f, %0.2f)s: %s",
                span.start,
                span.end,
                e,
            )
            return ReDecodeResult(span=span, text="", words=[], ok=False)

        return _to_result(span, raw)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> TranscriptionResult:
        """``Transcriber``-protocol surface over the re-decode span set.

        ``spans`` (a sequence of :class:`vemoizer.spans.Span`) is passed
        via ``kwargs``. Concatenates the per-span re-decodes in time
        order. With no spans the model is never loaded and no
        ``transcribe`` call is made (targeted-only cost invariant).
        """
        spans: Sequence[Span] = kwargs.get("spans") or ()
        results = [self.transcribe_span(audio, span) for span in spans]
        ok = [r for r in results if r.ok]
        text = " ".join(r.text for r in ok if r.text).strip()
        words: list[dict[str, Any]] = []
        for r in ok:
            words.extend(r.words)
        words.sort(key=lambda w: (w.get("start", 0.0), w.get("end", 0.0)))
        return {
            "text": text,
            "words": words,
            "segments": [],
            "transcribe_time": 0.0,
            "audio_duration": 0.0,
            "rtf": 0.0,
        }

    def cleanup(self) -> None:
        """Release the loaded model."""
        self.model = None
        self._model_path = None
        self._loaded = False
