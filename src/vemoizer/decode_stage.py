"""Decode stage: run one transcriber over the VAD slices and merge (issue #34).

Extracted from :mod:`vemoizer.pipeline` (500-line limit; single
responsibility: per-slice decoding and timeline merging). Both decode A
and decode B go through :func:`decode_all`; the per-slice records it
emits are what the slice-level A/B alignment pairs on.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import numpy as np

from .audio_contract import SAMPLE_RATE
from .progress import StageProgress

logger = logging.getLogger(__name__)


def _decode(transcriber: Any, audio: np.ndarray, label: str) -> dict[str, Any] | None:
    """Run one full decode; return its result or ``None`` (fail-open).

    Frees the Metal cache after every slice so GPU memory does not accumulate
    across VAD slices.
    """
    try:
        return transcriber.transcribe(audio)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("%s failed, using best available result: %s", label, e)
        return None
    finally:
        mx.clear_cache()


def decode_all(
    transcriber: Any, slices: list[tuple[int, np.ndarray]], label: str
) -> dict[str, Any] | None:
    """Decode every VAD slice through *transcriber* and merge the results.

    Word/segment times are shifted onto the full-recording timeline so the
    downstream alignment and re-decode stages work on one time base.
    """
    if not slices:
        return {"text": "", "words": [], "segments": []}
    merged_words: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    slice_records: list[dict[str, Any]] = []
    audio_seconds = sum(len(s) for _, s in slices) / SAMPLE_RATE
    progress = StageProgress(label, len(slices), audio_seconds)
    for index, (offset, slice_audio) in enumerate(slices):
        r = _decode(transcriber, slice_audio, label)
        progress.advance(failed=r is None)
        if r is None:
            continue
        text = str(r.get("text", "")).strip()
        texts.append(text)
        delta = offset / SAMPLE_RATE
        slice_words: list[dict[str, Any]] = []
        for w in r.get("words") or []:
            slice_words.append(
                {
                    **w,
                    "start": float(w.get("start", 0.0)) + delta,
                    "end": float(w.get("end", 0.0)) + delta,
                }
            )
        merged_words.extend(slice_words)
        for s in r.get("segments") or []:
            merged_segments.append(
                {
                    **s,
                    "start": float(s.get("start", 0.0)) + delta,
                    "end": float(s.get("end", 0.0)) + delta,
                }
            )
        # Per-slice record for the slice-level A/B alignment: both decodes
        # see the same slice index, so the records pair up even though only
        # decode A carries word timestamps.
        record: dict[str, Any] = {
            "index": index,
            "start_s": delta,
            "end_s": delta + len(slice_audio) / SAMPLE_RATE,
            "text": text,
            "words": slice_words,
        }
        if r.get("language") is not None:
            record["language"] = r["language"]
        slice_records.append(record)
    progress.done()
    merged = {
        "text": " ".join(t for t in texts if t),
        "words": merged_words,
        "segments": merged_segments,
        "slices": slice_records,
    }
    logger.info(
        "%s: %d chars, %d words, %d segments",
        label,
        len(merged["text"]),
        len(merged_words),
        len(merged_segments),
    )
    return merged
