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
