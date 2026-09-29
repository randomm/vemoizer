"""Disputed-span flagging for the consensus pipeline (issue #7, task B).

Given the aligned word pairs produced by the alignment stage (issue #7
task A), decide which time ranges of the recording the models disagree on,
and merge those ranges into the slices that the re-decode stage will
target.

Design decisions (recorded here because the issue left them open; the
canonical spec in ``docs/pipeline-spec.md`` should pin these when it
lands):

- **Similarity metric.** Pairwise word similarity is character-level:
  :func:`similarity` returns the length of the longest common subsequence
  of the two normalized words (casefold + punctuation stripped) divided by
  the length of the *longer* word. Two identical words score ``1.0``; a
  one-sided insertion/deletion (a ``None`` side) scores ``0.0``. A near
  miss (``"malttas"`` vs ``"maltta"`` → 6/7 ≈ 0.857, passes) is low-risk
  decode drift, while a real disagreement (``"moottori"`` vs ``"piksel"``
  → 0, flagged) sits below :data:`DISPUTE_THRESHOLD`. A 3/4 partial match
  (e.g. ``"abde"`` vs ``"abce"`` → 3/4 = 0.75, exactly at the boundary)
  is *not* disputed under the strict-``<`` rule.
- **Threshold + boundary.** A pair is disputed when its similarity is
  strictly *below* :data:`DISPUTE_THRESHOLD` (default ``0.75``). The
  boundary value itself (``similarity == 0.75``) is *not* disputed, so a
  run whose local agreement sits exactly at the cutoff passes clean.
- **LID flip.** Per :ref:`invariant 3 <invariant-3>` (language is a
  property of a span, not of a file), two reported language tags that
  differ mark the pair disputed even when the texts match. A *missing*
  tag is not a reported language and never triggers a flip — a backend
  that doesn't report per-word language cannot be accused of flipping it.
- **Merge / overlap rule.** Disputed slices within
  :data:`SPAN_MERGE_GAP_S` seconds of each other (gap ``<=`` the limit)
  or that overlap are merged into one slice. Slightly over-merging is
  cheaper than under-merging: re-decoding a couple of extra seconds with
  the third model is cheap, while splitting one disagreement across two
  re-decode calls wastes a decode and risks an incoherent verdict.
- **Slice boundaries.** Each slice runs from the *start* of its first
  disputed word to the *end* of its last, so the re-decode stage always
  receives whole words, never a partial one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: A pair of aligned words; ``None`` on either side marks an insertion or
#: deletion (the other decode didn't produce a word there).
AlignedPair = tuple[dict[str, Any] | None, dict[str, Any] | None]

#: Binary threshold: a pair is disputed when its similarity is strictly
#: below this value. ``similarity == DISPUTE_THRESHOLD`` is *not* disputed.
DISPUTE_THRESHOLD: float = 0.75

#: Maximum seconds between two disputed slices before they stop merging.
#: Gaps ``<=`` this value (or overlapping slices) merge into one slice.
SPAN_MERGE_GAP_S: float = 2.0

_PUNCTUATION = frozenset(" \t\n\r,.!?;:()[]{}<>\"'`@#$%^&*+=/\\|~_-—–‘’“”«»")


@dataclass(frozen=True)
class Span:
    """A disputed time range to re-decode: ``[start, end)`` seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Span start ({self.start}) > end ({self.end})")


def _normalized(word: dict[str, Any] | None) -> str:
    """Normalize a word for comparison: casefold + strip punctuation.

    ``None`` (a missing side of the alignment) normalizes to the empty
    string, which never matches a non-empty word.
    """
    if word is None:
        return ""
    return "".join(ch for ch in str(word["word"]).casefold() if ch not in _PUNCTUATION)


def similarity(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    """Character-overlap similarity of one aligned word pair.

    Returns the fraction of the longer normalized word whose characters
    are also in the other (order-insensitive, so ``"maltta"``/``"malttas"``
    → 0.8). A missing side (insertion/deletion) scores ``0.0``.
    """
    na = _normalized(a)
    nb = _normalized(b)
    if not na or not nb:
        return 0.0
    return _lcs_length(na, nb) / max(len(na), len(nb))


def _lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence of two strings.

    Standard two-row dynamic program (``O(len(a) * len(b))`` time, ``O(min)``
    space). Word-level inputs are short, so this is cheap even for the
    worst case of a few hundred words per decode.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _languages_differ(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """True when both words report a language and the reports disagree.

    A missing or empty tag is not a *reported* language, so it can never
    participate in a flip — only two distinct reported codes trigger it.
    """
    if a is None or b is None:
        return False
    la = a.get("language")
    lb = b.get("language")
    if not la or not lb:
        return False
    return str(la).casefold() != str(lb).casefold()


def _is_disputed(pair: AlignedPair) -> bool:
    """Whether one aligned word pair is disputed (text or language)."""
    a, b = pair
    if _languages_differ(a, b):
        return True
    return similarity(a, b) < DISPUTE_THRESHOLD


def _word_span(a: dict[str, Any] | None, b: dict[str, Any] | None) -> Span | None:
    """The time span covering whichever side of the pair is present."""
    for w in (a, b):
        if w is None:
            continue
        start = float(w.get("start", 0.0))
        end = w.get("end", w.get("start", 0.0))
        end = float(end) if end is not None else start
        return Span(start, max(start, end))
    return None


def merge_spans(spans: Sequence[Span]) -> list[Span]:
    """Merge overlapping or near-adjacent spans into a sorted, disjoint list.

    Two spans merge when they overlap or the gap between them is
    ``<= SPAN_MERGE_GAP_S``. The merged span spans the full range of the
    merged members. Input order and duplicates do not matter.
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[Span] = []
    for s in ordered:
        if merged and s.start <= merged[-1].end + SPAN_MERGE_GAP_S:
            last = merged.pop()
            merged.append(Span(last.start, max(last.end, s.end)))
        else:
            merged.append(s)
    return merged


def find_disputed_spans(pairs: Sequence[AlignedPair]) -> list[Span]:
    """Find the disputed time ranges of an aligned decode pair.

    Args:
        pairs: Aligned word pairs in time order. Each side is a word of the
            ``TranscriptionResult["words"]`` contract (``word``/``start``/
            ``end``, optional ``language``) or ``None`` for an insertion /
            deletion on that side.

    Returns:
        Sorted, non-overlapping disputed spans (each ``[start, end)`` in
        seconds) covering every pair that is disputed by the similarity
        threshold or by a reported language flip, merged per
        :data:`SPAN_MERGE_GAP_S`. These are the slices the re-decode stage
        will target.
    """
    raw: list[Span] = []
    for a, b in pairs:
        if not _is_disputed((a, b)):
            continue
        span = _word_span(a, b)
        if span is not None:
            raw.append(span)
    return merge_spans(raw)


def words_in_span(words: list[dict[str, Any]], span: Span) -> str:
    """The *words* falling inside *span*, joined (A-side span text)."""
    return " ".join(
        str(w.get("word", ""))
        for w in words
        if span.start <= float(w.get("start", 0.0)) < span.end
    ).strip()


#: Seconds of decode-A words offered to the adjudicator around a span.
CONTEXT_WINDOW_S = 10.0


def span_context(words: list[dict[str, Any]], span: Span) -> str:
    """Decode-A words within ±``CONTEXT_WINDOW_S`` of *span* (span excluded).

    Surrounding context is what lets the LLM disambiguate a garbled span —
    a Finnish sentence about deployments makes "Kamal" more plausible than
    "kamala" — without ever seeing the whole transcript.
    """
    lo = span.start - CONTEXT_WINDOW_S
    hi = span.end + CONTEXT_WINDOW_S
    return " ".join(
        str(w.get("word", ""))
        for w in words
        if lo <= float(w.get("start", 0.0)) < hi
        and not (span.start <= float(w.get("start", 0.0)) < span.end)
    ).strip()
