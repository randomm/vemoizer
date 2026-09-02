"""Whisper re-decode of disputed spans (issue #8).

Re-decode stage of the consensus pipeline: takes the disputed
``Span``s flagged by :mod:`vemoizer.spans` and re-decodes only those
slices with ``Finnish-NLP/whisper-large-finnish-v3`` through
``mlx-whisper``. This is what keeps the third model affordable —
disputed spans are seconds long, not minutes.

Design decisions:

- **Standalone, not a ``Transcriber``.** The re-decode is a targeted
  third opinion on seconds of audio, not a full transcription of the
  recording, so it is implemented directly on ``mlx_whisper`` rather
  than through the :class:`~vemoizer.transcriber.Transcriber` Protocol.
  The Protocol stays the backend seam for full-file decodes (Parakeet A,
  Canary B); squeezing a per-span slice re-decode into its
  ``transcribe(audio, **kwargs) -> TranscriptionResult`` shape was a
  misfit and has been removed. The public API is :func:`redecode`.
- **Targeted only (cost invariant).** ``redecode`` makes exactly one
  ``transcribe`` call per non-empty span and zero when given no spans or
  an empty recording. Never re-decodes the whole file.
- **Slice extraction is a pure function.** :func:`extract_slice` maps a
  ``Span`` to the sample range of a 16 kHz mono float32 buffer and
  returns a new array. No model, no side effects — trivially testable.
- **Revision-pinned load (invariant #4).** The first decode triggers
  ``snapshot_download(MODEL_ID, revision=MODEL_REVISION)`` and passes
  the returned *local path* (``str``, never the bare repo ID) to
  ``mlx_whisper.transcribe``. ``mlx-whisper`` is imported lazily
  inside the decode, so importing this module does not require it to
  be installed.
- **Fail-open per span.** A decode failure (load failure, decode error,
  timeout) logs the error and returns a *degraded* result for that span
  (empty ``text``) instead of raising. The adjudication stage still
  receives candidates A and B for the span; a re-decode failure must
  never abort the run.
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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spans import Span

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


def _load_model_path() -> str | None:
    """Download (revision-pinned) the model and return its local path.

    A ``None`` return means the download/load failed; the caller fails
    open for that span. The download is retried on the next call rather
    than permanently disabling re-decode for the run.
    """
    try:
        from huggingface_hub import snapshot_download

        return str(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    except Exception as e:  # noqa: BLE001 - logged; re-decode fails open
        logger.error("Failed to download re-decode model: %s", e)
        return None


def _transcribe_span(
    slice_audio: np.ndarray, span: Span, model_path: str
) -> ReDecodeResult:
    """Run one ``mlx_whisper.transcribe`` call; fail open on error."""
    try:
        import mlx_whisper

        raw = mlx_whisper.transcribe(
            slice_audio,
            path_or_hf_repo=model_path,
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


def redecode(audio: np.ndarray, spans: Sequence[Span]) -> list[ReDecodeResult]:
    """Re-decode disputed ``spans`` with whisper-large-finnish-v3.

    One targeted ``mlx_whisper.transcribe`` call per non-empty span, in
    order; zero calls (and no model download) when ``spans`` is empty or
    the recording is empty — the re-decode never decodes the whole file.

    Fails open per span: a download or decode failure yields a degraded
    :class:`ReDecodeResult`` (``ok=False``, empty ``text``) for that span
    instead of raising, so the run continues to adjudication with
    candidates A and B.

    Args:
        audio: The full 16 kHz mono float32 recording.
        spans: The disputed spans to re-decode (already merged by the
            alignment stage).

    Returns:
        One :class:`ReDecodeResult` per input span, in order.
    """
    if len(spans) == 0 or len(audio) == 0:
        return []

    results: list[ReDecodeResult] = []
    model_path: str | None = None
    for span in spans:
        slice_audio = extract_slice(audio, span)
        if slice_audio.size == 0:
            results.append(ReDecodeResult(span=span, text="", words=[], ok=True))
            continue
        if slice_audio.dtype != np.float32:
            slice_audio = slice_audio.astype(np.float32)

        if model_path is None:
            model_path = _load_model_path()
            if model_path is None:
                results.append(ReDecodeResult(span=span, text="", words=[], ok=False))
                continue

        results.append(_transcribe_span(slice_audio, span, model_path))
    return results
