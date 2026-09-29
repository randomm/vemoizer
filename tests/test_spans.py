"""Tests for disputed-span flagging (issue #7, task B).

Covers the pure span-level decisions of the consensus pipeline: the
word-level similarity metric, the dispute threshold and its boundary
semantics, the LID-flip trigger (invariant #3), the merge/overlap rule,
and the time boundaries of the resulting re-decode slices.

All inputs are synthetic aligned word pairs — no models, no network, no
audio fixtures needed (the alignment stage is issue #7 task A).
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from vemoizer.spans import (
    DISPUTE_THRESHOLD,
    MAX_DISPUTED_FRACTION,
    MAX_SPAN_S,
    MAX_SPANS,
    Span,
    apply_span_guardrails,
    find_disputed_spans,
    merge_spans,
    similarity,
)


def make_word(
    word: str,
    start: float,
    end: float,
    language: str | None = None,
) -> dict[str, Any]:
    """Build one word of the TranscriptionResult contract shape."""
    w: dict[str, Any] = {"word": word, "start": start, "end": end}
    if language is not None:
        w["language"] = language
    return w


# ---------------------------------------------------------------------------
# similarity()
# ---------------------------------------------------------------------------


def test_similarity_case_and_punctuation_insensitive() -> None:
    a = make_word("Hei, Maailma!", 0.0, 0.5)
    b = make_word("hei maailma", 0.0, 0.5)
    assert similarity(a, b) == 1.0


def test_similarity_exact_substitution_is_below_threshold() -> None:
    # "moottori" (8) vs "piksel" (6): LCS = "i" (1) -> 1/8 = 0.125.
    # A real disagreement is well below the threshold, even when the two
    # words happen to share a couple of letters.
    a = make_word("moottori", 0.0, 0.5)
    b = make_word("piksel", 0.0, 0.5)
    assert similarity(a, b) == pytest.approx(0.125)
    assert similarity(a, b) < DISPUTE_THRESHOLD


def test_similarity_substitution_scores_by_lcs_over_longer() -> None:
    # "malttas" (7) vs "maltta" (6): LCS = "maltta" (6) -> 6/7.
    a = make_word("malttas", 0.0, 0.5)
    b = make_word("maltta", 0.0, 0.5)
    assert similarity(a, b) == pytest.approx(6 / 7)


def test_similarity_order_matters() -> None:
    # Same multiset of chars, different order: LCS is 2 ("ab"), not 3.
    a = make_word("abc", 0.0, 0.1)
    b = make_word("bca", 0.0, 0.1)
    assert similarity(a, b) == pytest.approx(2 / 3)


def test_similarity_3_of_4_partial_match_is_exactly_threshold() -> None:
    # "abde" vs "abce": LCS = "abe" (3) over longer (4) -> 0.75 == threshold.
    a = make_word("abde", 0.0, 0.1)
    b = make_word("abce", 0.0, 0.1)
    assert similarity(a, b) == pytest.approx(DISPUTE_THRESHOLD)


def test_similarity_identical_texts_is_one() -> None:
    assert similarity(make_word("terve", 0.0, 0.3), make_word("terve", 0.0, 0.3)) == 1.0


def test_similarity_empty_side_is_zero() -> None:
    # One-side insertion/deletion: the empty side never matches.
    assert similarity(None, make_word("hei", 0.0, 0.3)) == 0.0
    assert similarity(make_word("hei", 0.0, 0.3), None) == 0.0
    assert similarity(None, None) == 0.0


# ---------------------------------------------------------------------------
# find_disputed_spans(): threshold + boundary
# ---------------------------------------------------------------------------


def test_identical_decodes_produce_no_disputed_spans() -> None:
    words = [
        make_word("hei", 0.0, 0.3),
        make_word("kaikki", 0.4, 0.8),
        make_word("hyva", 0.9, 1.2),
    ]
    pairs = [(w, copy.deepcopy(w)) for w in words]
    assert find_disputed_spans(pairs) == []


def test_single_word_substitution_is_flagged() -> None:
    pairs = [
        (make_word("hei", 0.0, 0.3), make_word("hei", 0.0, 0.3)),
        (make_word("parakeet", 0.4, 0.9), make_word("canary", 0.4, 0.9)),
        (make_word("hyva", 1.0, 1.3), make_word("hyva", 1.0, 1.3)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 0.4
    assert spans[0].end == 0.9


def test_disputes_below_threshold_are_flagged() -> None:
    # "malttas" vs "maltta" -> LCS 6 / longer 7 = 0.857 (near miss, passes);
    # "abcde" vs "abf" -> LCS 2 / longer 5 = 0.4 (real disagreement, flagged).
    # The two positions are > SPAN_MERGE_GAP_S apart, so they stay separate.
    a_words = ["malttas", "abcde"]
    b_words = ["maltta", "abf"]
    pairs = [
        (make_word(a_words[0], 0.0, 0.4), make_word(b_words[0], 0.0, 0.4)),
        (make_word(a_words[1], 5.0, 5.4), make_word(b_words[1], 5.0, 5.4)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 5.0
    assert spans[0].end == 5.4


def test_disputes_at_or_above_threshold_are_not_flagged() -> None:
    # "malttas"/"maltta" -> 6/7 = 0.857; "hyv"/"hyva" -> 3/4 = 0.75;
    # the rest are identical. All >= threshold: no disputed spans.
    a_words = ["malttas", "kestaa", "kaikki", "hyva", "olipa"]
    b_words = ["maltta", "kestaa", "kaikki", "hyv", "olipa"]
    pairs = [
        (make_word(a, 0.0 + i, 0.4 + i), make_word(b, 0.0 + i, 0.4 + i))
        for i, (a, b) in enumerate(zip(a_words, b_words, strict=True))
    ]
    assert find_disputed_spans(pairs) == []


def test_threshold_boundary_exactly_at_threshold_is_not_disputed() -> None:
    # "abde" vs "abce": LCS "abe" (3) / longer 4 = 0.75 == DISPUTE_THRESHOLD.
    # The rule is strict <, so exact equality is NOT disputed.
    pairs = [
        (make_word("abde", 0.0, 0.4), make_word("abce", 0.0, 0.4)),
    ]
    assert find_disputed_spans(pairs) == []


# ---------------------------------------------------------------------------
# find_disputed_spans(): one-sided insertions / deletions
# ---------------------------------------------------------------------------


def test_one_side_insertion_is_flagged() -> None:
    pairs = [
        (make_word("hei", 0.0, 0.3), make_word("hei", 0.0, 0.3)),
        (
            make_word("english", 0.5, 0.9, language="en"),
            make_word("englantia", 0.5, 0.9, language="fi"),
        ),
        (make_word("hyva", 1.0, 1.3), make_word("hyva", 1.0, 1.3)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 0.5
    assert spans[0].end == 0.9


def test_insertion_uses_only_side_timestamps() -> None:
    # B drops a word entirely; the disputed slice covers A's word bounds.
    a_words = [make_word("moottori", 0.0, 0.4), make_word("piksel", 0.5, 0.9)]
    pairs = [
        (a_words[0], make_word("moottori", 0.0, 0.4)),
        (a_words[1], None),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 0.5
    assert spans[0].end == 0.9


# ---------------------------------------------------------------------------
# find_disputed_spans(): LID-flip trigger (invariant #3)
# ---------------------------------------------------------------------------


def test_lid_flip_is_flagged_even_when_texts_match() -> None:
    # Same text, different language tag: the span is disputed because
    # language is a property of the span, not the file.
    pairs = [
        (
            make_word("moottori", 0.0, 0.4, language="fi"),
            make_word("moottori", 0.0, 0.4, language="en"),
        ),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 0.0
    assert spans[0].end == 0.4


def test_same_language_tag_is_not_a_lid_flip() -> None:
    pairs = [
        (
            make_word("moottori", 0.0, 0.4, language="fi"),
            make_word("moottori", 0.0, 0.4, language="fi"),
        ),
    ]
    assert find_disputed_spans(pairs) == []


def test_missing_language_tag_is_never_a_lid_flip() -> None:
    # A missing tag is not a reported language; only two *reported*
    # distinct languages trigger the flip.
    pairs = [
        (make_word("hei", 0.0, 0.3, language="fi"), make_word("hei", 0.0, 0.3)),
        (make_word("hei", 0.0, 0.3), make_word("hei", 0.0, 0.3, language="fi")),
    ]
    assert find_disputed_spans(pairs) == []


# ---------------------------------------------------------------------------
# merge / overlap rule
# ---------------------------------------------------------------------------


def _disputed_pair(start: float) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        make_word("aaa", start, start + 0.3),
        make_word("bbb", start, start + 0.3),
    )


def test_overlapping_disputed_words_merge_into_one_span() -> None:
    pairs = [
        (make_word("aa", 0.0, 0.5), make_word("bb", 0.0, 0.5)),
        (make_word("cc", 0.4, 0.9), make_word("dd", 0.4, 0.9)),
        (make_word("ee", 1.5, 1.8), make_word("ff", 1.5, 1.8)),
    ]
    spans = find_disputed_spans(pairs)
    # First two disputed words overlap (0.5 > 0.4) and merge; the third is
    # 0.6 s away, above SPAN_MERGE_GAP_S (2.0 s)? No — 1.5 - 0.9 = 0.6 < 2.0,
    # so it also merges. All three become one span.
    assert len(spans) == 1
    assert spans[0].start == 0.0
    assert spans[0].end == 1.8


def test_disputes_further_apart_than_merge_gap_stay_separate() -> None:
    # Gap between span 1 end (0.5) and span 2 start (3.0) is 2.5 s > 2.0 s.
    pairs = [
        (make_word("aa", 0.0, 0.5), make_word("bb", 0.0, 0.5)),
        (make_word("cc", 3.0, 3.4), make_word("dd", 3.0, 3.4)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 2
    assert spans[0].start == 0.0
    assert spans[0].end == 0.5
    assert spans[1].start == 3.0
    assert spans[1].end == 3.4


def test_merge_gap_exactly_at_limit_is_merged() -> None:
    # Gap of exactly SPAN_MERGE_GAP_S (2.0 s) is merged (<=, not <).
    pairs = [
        (make_word("aa", 0.0, 0.5), make_word("bb", 0.0, 0.5)),
        (make_word("cc", 2.5, 2.9), make_word("dd", 2.5, 2.9)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 1
    assert spans[0].start == 0.0
    assert spans[0].end == 2.9


def test_merge_gap_slightly_above_limit_is_not_merged() -> None:
    pairs = [
        (make_word("aa", 0.0, 0.5), make_word("bb", 0.0, 0.5)),
        (make_word("cc", 2.6, 3.0), make_word("dd", 2.6, 3.0)),
    ]
    spans = find_disputed_spans(pairs)
    assert len(spans) == 2


def test_merge_spans_is_sorted_and_deduplicated() -> None:
    spans = [
        Span(10.0, 11.0),
        Span(0.0, 1.0),
        Span(0.5, 1.5),
        Span(10.0, 11.0),
    ]
    merged = merge_spans(spans)
    assert merged == [Span(0.0, 1.5), Span(10.0, 11.0)]


def test_merge_spans_empty_input() -> None:
    assert merge_spans([]) == []


def test_merge_spans_zero_gap_adjacent_is_merged() -> None:
    spans = [Span(0.0, 1.0), Span(1.0, 2.0)]
    assert merge_spans(spans) == [Span(0.0, 2.0)]


# ---------------------------------------------------------------------------
# end-to-end shape: re-decode slices
# ---------------------------------------------------------------------------


def test_disputed_slice_time_boundaries_feed_redecode() -> None:
    # The full pipeline case: identical decodes except a code-switched
    # Finnish/English region where B drops the English word and one
    # substitution drifts. The re-decode slice must cover the union of
    # the word-level boundaries.
    pairs = [
        (make_word("moottori", 0.0, 0.4), make_word("moottori", 0.0, 0.4)),
        (
            make_word("piksel", 0.5, 0.9, language="en"),
            make_word("pikselli", 0.5, 0.9, language="fi"),
        ),
        (make_word("kone", 1.0, 1.3), make_word("kone", 1.0, 1.3)),
        (make_word("kaikki", 1.4, 1.8), None),
        (make_word("hyva", 5.0, 5.3), make_word("hyva", 5.0, 5.3)),
    ]
    spans = find_disputed_spans(pairs)
    # 0.5-0.9 and 1.4-1.8: gap = 0.5 s <= 2.0 s -> merged.
    assert len(spans) == 1
    assert spans[0].start == pytest.approx(0.5)
    assert spans[0].end == pytest.approx(1.8)


def test_empty_input_produces_no_spans() -> None:
    assert find_disputed_spans([]) == []


def test_single_word_pair_both_identical_no_spans() -> None:
    w = make_word("hei", 0.0, 0.3)
    assert find_disputed_spans([(w, copy.deepcopy(w))]) == []


# -- guardrails (issue #55) ----------------------------------------------
#
# Synthetic B word times can misalign pathologically; the guardrails keep
# a bad alignment from turning into hours of re-decode or a garbage
# transcript. Above the disputed-fraction cap the caller aborts consensus
# entirely and ships decode A (fail-open).


def test_guardrails_pass_a_sane_span_set_through() -> None:
    spans = [Span(0.0, 3.0), Span(10.0, 12.0)]
    assert apply_span_guardrails(spans, speech_seconds=600.0) == spans


def test_overlong_span_is_clipped_to_max() -> None:
    spans = [Span(0.0, MAX_SPAN_S * 3)]
    out = apply_span_guardrails(spans, speech_seconds=600.0)
    assert out is not None
    assert out[0].end - out[0].start == MAX_SPAN_S


def test_span_count_is_capped_keeping_earliest() -> None:
    spans = [Span(float(i), float(i) + 0.5) for i in range(MAX_SPANS + 50)]
    out = apply_span_guardrails(spans, speech_seconds=100000.0)
    assert out is not None
    assert len(out) == MAX_SPANS
    assert out[0].start == 0.0


def test_excessive_disputed_fraction_aborts_to_none() -> None:
    """> MAX_DISPUTED_FRACTION of the speech disputed = broken alignment."""
    speech = 100.0
    disputed = speech * MAX_DISPUTED_FRACTION + 10.0
    spans = [Span(0.0, disputed)]
    assert apply_span_guardrails(spans, speech_seconds=speech) is None


def test_zero_speech_never_divides() -> None:
    assert apply_span_guardrails([Span(0.0, 1.0)], speech_seconds=0.0) is None


def test_empty_spans_stay_empty() -> None:
    assert apply_span_guardrails([], speech_seconds=100.0) == []


def test_fraction_guard_skipped_for_short_recordings() -> None:
    """A 3-second clip disputing wholly is normal; re-decoding all of it
    is affordable. The fraction abort is a cost bound at scale only."""
    spans = [Span(0.0, 3.0)]
    assert apply_span_guardrails(spans, speech_seconds=3.0) == spans
