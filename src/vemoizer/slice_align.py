"""Slice-level A/B dispute detection (issue #55).

Decode B (Canary) emits no word timestamps, so word-onset DTW cannot run.
But both decodes share the VAD slices, and the slice (median ~2 s) is the
natural dispute unit for Finnish anyway: rich morphology makes the two
backends spell half the *words* differently while telling the same story,
so word-level similarity flags inflection and tokenization drift as
disputes. Measured on the real 64-min memo: 70 % of word pairs "dispute"
while only ~17 % of speech *time* has genuinely divergent slice text at
:data:`SLICE_DISPUTE_THRESHOLD`.

A slice is disputed when the char-level similarity of its normalized A and
B texts falls below the threshold. Span bounds are the slice's real VAD
bounds — no synthetic timestamps anywhere — and the span carries decode
B's per-slice detected language (invariant #3).

When the count cap trims the set, the *most severe* disputes (lowest
similarity) survive, not the earliest: re-decode effort goes where the
decoders disagree hardest.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from .spans import MAX_SPANS, Span, merge_spans
from .textnorm import textnorm

logger = logging.getLogger(__name__)

#: A slice whose normalized A/B texts are at least this similar is
#: undisputed. Calibrated on the 64-min reference memo: 0.55 puts ~17 % of
#: speech time in dispute (inside the 25 % guardrail) while catching the
#: slices where the decoders genuinely tell different stories; the naive
#: word-level unit flagged 53-87 % of the memo.
SLICE_DISPUTE_THRESHOLD = 0.55

#: A slice whose normalized A/B texts are at least this similar is
#: "high confidence": both decodes agree strongly, so a two-way
#: comparison sees the slice as clean — and cannot see the
#: agreement-on-wrong-answer case where both decoders are confidently
#: wrong in the same way (issue #62). The Whisper re-decode output is
#: what exposes those spans (see :func:`find_high_confidence_disagreement`).
SLICE_HIGH_CONFIDENCE_THRESHOLD = 0.75


def slice_similarity(text_a: str, text_b: str) -> float:
    """Char-level similarity of two normalized slice texts in ``[0, 1]``.

    Case, punctuation and whitespace never count as differences
    (:func:`vemoizer.textnorm.textnorm` runs first). Two empty texts are
    identical; one empty side is a total dispute.
    """
    norm_a, norm_b = textnorm(text_a), textnorm(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def find_disputed_slices(
    slices_a: list[dict[str, Any]],
    slices_b: list[dict[str, Any]],
    *,
    threshold: float = SLICE_DISPUTE_THRESHOLD,
) -> list[Span] | None:
    """Disputed spans between per-slice decode records; ``None`` = no basis.

    Records are ``{index, start_s, end_s, text, language?}`` (produced by
    ``decode_stage.decode_all``) and pair by ``index``. A slice missing
    from either side is undisputed (fail-open: a failed decode slice must
    not flag the other side). Overlapping/near-adjacent disputed slices
    merge via :func:`vemoizer.spans.merge_spans`; a count overflow keeps
    the lowest-similarity spans.

    Returns ``None`` when no slice could be compared at all, so callers
    can distinguish "no disputes" from "no alignment basis".
    """
    if not slices_a or not slices_b:
        return None
    b_by_index = {s["index"]: s for s in slices_b}
    compared = 0
    disputed: list[tuple[float, Span]] = []
    for a in slices_a:
        b = b_by_index.get(a["index"])
        if b is None:
            continue
        compared += 1
        sim = slice_similarity(str(a.get("text", "")), str(b.get("text", "")))
        if sim >= threshold:
            continue
        language = b.get("language") or a.get("language")
        span = Span(float(a["start_s"]), float(a["end_s"]), language)
        disputed.append((sim, span))
    if compared == 0:
        return None
    if len(disputed) > MAX_SPANS:
        disputed.sort(key=lambda pair: pair[0])  # most severe first
        dropped = len(disputed) - MAX_SPANS
        disputed = disputed[:MAX_SPANS]
        logger.warning(
            "disputed slices exceed cap %d; keeping the %d most severe (%d dropped)",
            MAX_SPANS,
            MAX_SPANS,
            dropped,
        )
    logger.info(
        "slice dispute: %d/%d slices disputed (threshold %.2f)",
        len(disputed),
        compared,
        threshold,
    )
    return merge_spans([span for _sim, span in disputed])


def find_high_confidence_disagreement(
    slices_a: list[dict[str, Any]],
    slices_b: list[dict[str, Any]],
    slices_c: list[dict[str, Any]],
    *,
    agreement_threshold: float = SLICE_HIGH_CONFIDENCE_THRESHOLD,
    threshold: float = SLICE_DISPUTE_THRESHOLD,
) -> list[Span]:
    """Spans where A and B agree but the third decode (C) disagrees (issue #62).

    This is the agreement-on-wrong-answer detector: the two-way A/B
    comparison is blind to slices both decoders are confidently wrong on,
    because their texts are similar. A slice qualifies when its A and B
    texts are similar at least *agreement_threshold* (the high-confidence,
    currently-clean case) while its text disagrees with the C text by
    more than the dispute *threshold*. The span bounds are the VAD slice's
    real bounds (the A record) and it carries the first reported language
    in the A → B → C order (invariant #3).

    C is the Whisper re-decode output. When the re-decode has not run for
    a slice (C record missing), the slice is silently skipped: the third
    detector simply has no opinion there, and a missing detector must not
    manufacture a dispute (fail-open). Records are ``{index, start_s,
    end_s, text, language?}``; slices missing from A or B do not qualify
    either (the high-confidence agreement cannot be established).
    """
    b_by_index = {s["index"]: s for s in slices_b}
    c_by_index: dict[Any, dict[str, Any]] = {s["index"]: s for s in slices_c}
    spans: list[Span] = []
    for a in slices_a:
        b = b_by_index.get(a["index"])
        c = c_by_index.get(a["index"])
        if b is None or c is None:
            continue
        a_text = str(a.get("text", ""))
        b_text = str(b.get("text", ""))
        c_text = str(c.get("text", ""))
        if not (a_text and b_text and c_text):
            continue  # an empty side cannot establish agreement or disagreement
        if slice_similarity(a_text, b_text) < agreement_threshold:
            continue  # not the high-confidence (currently-clean) case
        if slice_similarity(a_text, c_text) >= threshold:
            continue  # the third decode also agrees: nothing to sample
        language = a.get("language") or b.get("language") or c.get("language")
        spans.append(Span(float(a["start_s"]), float(a["end_s"]), language))
    if spans:
        logger.info(
            "high-confidence disagreement: %d slice(s) where A≈B but C differs",
            len(spans),
        )
    return merge_spans(spans)
