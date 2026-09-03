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
- **Revision-pinned load (invariant #4).** First ``transcribe_span``
  triggers ``snapshot_download(MODEL_ID, revision=MODEL_REVISION)`` and
  passes the returned *local path* to ``mlx_whisper.load_model`` — never
  the bare repo ID. ``mlx-whisper`` is imported lazily inside the load,
  so importing this module does not require it to be installed.
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

Adjudication: re-decode results are candidates for the LLM adjudication
stage; the LLM client itself is not part of this module (see
``docs/pipeline-spec.md`` for the stage contract).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from .spans import Span
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

#: The Finnish fine-tune of Whisper-large, as a community MLX conversion
#: (base_model: Finnish-NLP/whisper-large-finnish-v3). mlx-whisper cannot
#: consume the raw HF transformers checkpoint, and the pip package ships no
#: converter — the MLX port is the load repo, same pattern as decode B's
#: Canary port (see docs/pipeline-spec.md).
MODEL_ID = "FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx"
#: Pinned commit on the model repo so upstream pushes cannot change the
#: weights we run (project invariant #4).
MODEL_REVISION = "f51f0310c1b2a3e5acb16905c1a7245bb9476846"

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
    # ``transcribe`` returns ``{text, segments, language}`` — there is no
    # top-level ``words`` key. Word timestamps live inside each segment
    # (``seg["words"]``), opt-in via ``word_timestamps=True``.
    raw_words: list[dict[str, Any]] = []
    for seg in raw.get("segments") or []:
        raw_words.extend(seg.get("words") or [])
    words: list[dict[str, Any]] = []
    for w in raw_words:
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

    The public API is :meth:`transcribe_span` — one targeted decode of one
    disputed slice. This is *not* a drop-in :class:`Transcriber` replacement:
    the re-decode stage is a third opinion on seconds of audio, not a full
    transcription of the recording. The ``TranscriptionResult``-shaped
    :meth:`transcribe` is kept as a thin adapter for callers that want a
    sparse, per-span concatenation.

    Model loading is lazy and revision-pinned (invariant #4): nothing is
    downloaded at import or construction time; the first span triggers
    ``snapshot_download`` and ``mlx_whisper.load_models.load_model`` from
    the returned local path. A failed load leaves ``self.model`` ``None``
    and every subsequent span fails open with an empty ``ReDecodeResult``;
    a transient load failure does not permanently disable re-decode —
    ``_ensure_loaded`` retries on the next call until the load succeeds.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self._model_path: str | None = None
        self._loaded: bool = False
        self._load_failed: bool = False
        self._mlx_whisper: Any = None
        self._load_start: float | None = None

    def _ensure_loaded(self) -> None:
        """Resolve the revision-pinned model path once.

        ``mlx_whisper.transcribe`` loads (and caches, via its
        ``ModelHolder``) the model from the path we pass it, so this
        resolves the snapshot path and imports the library — it must NOT
        also call ``load_models.load_model``: that would hold a second
        ~3 GB copy of the weights that ``transcribe`` never uses.

        A failed resolve latches ``_load_failed`` so a broken cache is
        reported once, not retried for every one of hundreds of spans
        (the same latch decode A and B use).
        """
        if self._loaded or self._load_failed:
            return
        logger.info("Loading re-decode model: %s@%s", MODEL_ID, MODEL_REVISION)
        start = time.perf_counter()
        try:
            import mlx_whisper
            from huggingface_hub import snapshot_download

            local_path = snapshot_download(
                MODEL_ID,
                revision=MODEL_REVISION,
            )
            self._model_path = local_path
            self._mlx_whisper = mlx_whisper
            self._loaded = True
            # Marker object: the real model lives in mlx_whisper's
            # ModelHolder cache, keyed by path, once the first span runs.
            self.model = local_path
        except Exception as e:  # noqa: BLE001 - logged; re-decode fails open
            logger.error("Failed to load re-decode model (not retrying): %s", e)
            self.model = None
            self._load_failed = True
            return
        self._load_start = start
        logger.info("Re-decode model resolved in %.2fs", time.perf_counter() - start)

    def transcribe_span(
        self, audio: np.ndarray, span: Span, language: str | None = None
    ) -> ReDecodeResult:
        """Re-decode one disputed slice; fail open on any error.

        ``audio`` is the full 16 kHz mono float32 recording; only the
        ``span`` range is fed to the model. ``language`` is the span's
        detected language when the caller knows it (invariant #3:
        language is a span property); Finnish is the profile default.
        Returns a :class:`ReDecodeResult` with ``ok=False`` and empty
        ``text`` when the model could not run (load or decode failure).
        """
        self._ensure_loaded()
        if self.model is None:
            logger.warning(
                "Re-decode unavailable; failing open for span [%0.2f, %0.2f)s",
                span.start,
                span.end,
            )
            return ReDecodeResult(span=span, text="", words=[], ok=False)

        slice_audio = extract_slice(audio, span)
        if slice_audio.size == 0:
            return ReDecodeResult(span=span, text="", words=[], ok=True)
        if slice_audio.dtype != np.float32:
            slice_audio = slice_audio.astype(np.float32)

        model_path = self._model_path
        if model_path is None:
            logger.warning(
                "Re-decode unavailable (no model path); failing open for "
                "span [%0.2f, %0.2f)s",
                span.start,
                span.end,
            )
            return ReDecodeResult(span=span, text="", words=[], ok=False)

        try:
            raw = self._mlx_whisper.transcribe(
                slice_audio,
                path_or_hf_repo=model_path,
                word_timestamps=True,
                language=language or "fi",
                task="transcribe",
                # Deterministic third opinion: greedy, and never conditioned
                # on text from outside the disputed slice.
                temperature=0.0,
                condition_on_previous_text=False,
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
        """``TranscriptionResult``-shaped adapter over the re-decode span set.

        ``spans`` (a sequence of :class:`vemoizer.spans.Span`) is passed
        via ``kwargs``. Concatenates the per-span re-decodes in time
        order. With no spans the model is never loaded and no
        ``transcribe`` call is made (targeted-only cost invariant).

        The returned ``TranscriptionResult`` is *sparse*: ``text`` and
        ``words`` are the concatenation of the per-span re-decodes and
        ``segments`` is empty (there are no segment boundaries between
        disjoint spans). ``transcribe_time`` is wall-clock across the
        per-span decode calls only (model loading happens in
        :meth:`_ensure_loaded`, which is timed and logged separately),
        and ``rtf`` is the ratio of that to the total recording duration.
        """
        spans: Sequence[Span] = kwargs.get("spans") or ()
        started = time.perf_counter()
        results = [self.transcribe_span(audio, span) for span in spans]
        elapsed = time.perf_counter() - started
        ok = [r for r in results if r.ok]
        text = " ".join(r.text for r in ok if r.text).strip()
        words: list[dict[str, Any]] = []
        for r in ok:
            words.extend(r.words)
        words.sort(key=lambda w: (w.get("start", 0.0), w.get("end", 0.0)))
        audio_duration = float(len(audio)) / SAMPLE_RATE
        return {
            "text": text,
            "words": words,
            "segments": [],
            "transcribe_time": elapsed,
            "audio_duration": audio_duration,
            "rtf": (elapsed / audio_duration) if audio_duration > 0 else 0.0,
        }

    def cleanup(self) -> None:
        """Release the loaded model, including mlx-whisper's own cache.

        ``mlx_whisper.transcribe`` caches the loaded model in its module-
        level ``ModelHolder``; without clearing it the ~3 GB of weights
        outlive this transcriber and crowd out the next model load.
        """
        if self._mlx_whisper is not None:
            with suppress(Exception):  # best-effort cache release
                holder = self._mlx_whisper.transcribe.ModelHolder
                holder.model = None
                holder.model_path = None
        self.model = None
        self._model_path = None
        self._mlx_whisper = None
        self._loaded = False
        self._load_start = None
        mx.clear_cache()
