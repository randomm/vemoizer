"""Per-span consensus helpers for the pipeline (issue #117).

Extracted from :mod:`vemoizer.pipeline` (issue #117, headroom): the pure,
self-contained span-level helpers that sit between the two decodes and the
assembler — detecting the disputed spans to re-decode (``_find_spans``),
scoping decode B's candidate to each span (``_b_text_in_span``),
and picking the final verdict fail-open down the candidate list
(``_adjudicate``).

The orchestrator :func:`vemoizer.pipeline._assemble` and the public
``transcribe_file`` entry point stay in ``pipeline.py``. ``pipeline`` imports
all three helpers (plus the ``Candidate`` alias) into its own module namespace,
so the ``monkeypatch.setattr(pipeline, "_find_spans" | "_b_text_in_span" |
"_adjudicate", ...)`` seams used by the tests still intercept the real call
path (the calls inside ``_assemble`` / ``transcribe_file`` resolve against
``pipeline``'s globals).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .llm import LLMClient
from .progress import format_duration
from .slice_align import find_disputed_slices, find_high_confidence_disagreement
from .spans import Span, apply_span_guardrails

logger = logging.getLogger(__name__)

Candidate = dict[str, str]  # {"source": str, "text": str}


def _find_spans(
    result_a: dict[str, Any] | None,
    result_b: dict[str, Any] | None,
    result_c: dict[str, Any] | None = None,
) -> list[Span]:
    """Disputed spans between the decodes, guardrailed; ``[]`` = no consensus.

    The dispute unit is the VAD slice: disputed when its normalized A/B
    texts diverge below the slice-similarity threshold. The third decode's
    per-slice text (*result_c*, issue #62) additionally flags the
    high-confidence A≈B slices the C text disagrees with — the
    agreement-on-wrong-answer case the two-way comparison is blind to.
    The ``VEMOIZER_DISABLE_CONSENSUS=1`` kill-switch and every failure path
    land on ``[]`` — the run ships decode A alone (fail-open).
    """
    if os.environ.get("VEMOIZER_DISABLE_CONSENSUS") == "1":
        logger.info("consensus disabled by VEMOIZER_DISABLE_CONSENSUS=1")
        return []
    if result_a is None or result_b is None:
        logger.info("disputed spans: 0 (a decode is missing)")
        return []
    slices_a = list(result_a.get("slices") or [])
    slices_b = list(result_b.get("slices") or [])
    spans = find_disputed_slices(slices_a, slices_b)
    if spans is None:
        logger.info("disputed spans: 0 (no comparable slices, re-decode skipped)")
        return []
    # Issue #62: the two-way A/B comparison is blind to slices both
    # decoders are confidently wrong on. The third decode is a *detector*
    # too: when it has produced per-slice text, sample the spans where
    # A≈B but the C text disagrees. No C result → no C text → no new
    # spans (zero-cost, no behaviour change for runs without it).
    slices_c = list((result_c or {}).get("slices") or [])
    c_records = [
        {"index": s["index"], "text": str(s.get("text", ""))} for s in slices_c
    ]
    if c_records:
        spans += find_high_confidence_disagreement(slices_a, slices_b, c_records)
    speech_seconds = sum(float(s["end_s"]) - float(s["start_s"]) for s in slices_a)
    guarded = apply_span_guardrails(spans, speech_seconds=speech_seconds)
    if guarded is None:
        logger.warning("disputed spans rejected by guardrails; shipping decode A")
        return []
    disputed_s = sum(s.end - s.start for s in guarded)
    fraction = 100.0 * disputed_s / speech_seconds if speech_seconds > 0 else 0.0
    logger.info(
        "disputed spans: %d (%s of audio, %.0f%%)",
        len(guarded),
        format_duration(disputed_s),
        fraction,
    )
    return guarded


def _b_text_in_span(result_b: dict[str, Any] | None, span: Span) -> str:
    """Decode B's slice text overlapping *span* (B has no word timestamps)."""
    if result_b is None:
        return ""
    parts: list[str] = []
    for s in result_b.get("slices") or []:
        if float(s["end_s"]) > span.start and float(s["start_s"]) < span.end:
            text = str(s.get("text", "")).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _adjudicate(
    span: Span,
    a_text: str,
    candidates: list[Candidate],
    client: LLMClient | None,
    context: str = "",
) -> str:
    """Final text for one disputed span, fail-open down the candidate list."""
    if client is not None:
        try:
            verdict = client.adjudicate(a_text, candidates, context)
            if verdict.strip():
                return verdict
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning(
                "adjudication failed for span [%0.2f, %0.2f): %s",
                span.start,
                span.end,
                e,
            )
    for candidate in reversed(candidates):  # re-decode > decode B > decode A
        if candidate["text"].strip():
            return candidate["text"]
    return a_text
