"""Slice-level A/B dispute detection (issue #55).

Pure functions — no models. Decode B has no word timestamps, but both
decodes share the VAD slices; a slice is disputed when its normalized A/B
texts diverge below the similarity threshold. Span bounds are the slice's
real VAD bounds; the span carries the slice's detected language.
"""

from __future__ import annotations

from vemoizer.slice_align import (
    SLICE_DISPUTE_THRESHOLD,
    find_disputed_slices,
    slice_similarity,
)
from vemoizer.spans import MAX_SPANS


def _slice(index, start_s, end_s, text, language=None) -> dict:
    d = {"index": index, "start_s": start_s, "end_s": end_s, "text": text}
    if language is not None:
        d["language"] = language
    return d


# -- slice_similarity ----------------------------------------------------


def test_identical_texts_are_fully_similar() -> None:
    assert slice_similarity("hei maailma", "hei maailma") == 1.0


def test_case_and_punctuation_never_count() -> None:
    assert slice_similarity("Hei, maailma!", "hei maailma") == 1.0


def test_inflection_drift_stays_above_the_threshold() -> None:
    """Finnish morphology: near-miss spellings are NOT disputes."""
    sim = slice_similarity(
        "ei osata vielä koneenistaa sitä",
        "ei osata vielä koneellistaa sitä",
    )
    assert sim >= SLICE_DISPUTE_THRESHOLD


def test_divergent_stories_fall_below_the_threshold() -> None:
    sim = slice_similarity(
        "valinnut vähän suuntimoja että täällä minä aivan",
        "puhutaan nyt ihan muusta asiasta kokonaan",
    )
    assert sim < SLICE_DISPUTE_THRESHOLD


def test_empty_versus_text_is_a_total_dispute() -> None:
    assert slice_similarity("", "jotain sanottiin") == 0.0
    assert slice_similarity("", "") == 1.0


# -- find_disputed_slices ------------------------------------------------


def test_agreeing_slices_produce_no_spans() -> None:
    a = [_slice(0, 0.0, 2.0, "hei maailma")]
    b = [_slice(0, 0.0, 2.0, "Hei maailma!", language="fi")]
    assert find_disputed_slices(a, b) == []


def test_disputed_slice_becomes_a_span_with_real_bounds() -> None:
    a = [
        _slice(0, 0.0, 2.0, "hei maailma"),
        _slice(1, 5.0, 8.5, "aivan eri tarina tässä kohtaa"),
    ]
    b = [
        _slice(0, 0.0, 2.0, "hei maailma", language="fi"),
        _slice(1, 5.0, 8.5, "nyt puhutaan lomasuunnitelmista", language="fi"),
    ]
    spans = find_disputed_slices(a, b)
    assert spans is not None
    assert len(spans) == 1
    assert spans[0].start == 5.0  # the slice's real VAD bounds
    assert spans[0].end == 8.5
    assert spans[0].language == "fi"


def test_missing_b_slice_is_undisputed_not_flagged() -> None:
    """A failed decode-B slice must not flag decode A's words (fail-open)."""
    a = [
        _slice(0, 0.0, 1.0, "eka juttu tässä"),
        _slice(1, 2.0, 3.0, "toka juttu tuossa"),
    ]
    b = [_slice(1, 2.0, 3.0, "toka juttu tuossa", language="fi")]
    assert find_disputed_slices(a, b) == []


def test_no_comparable_slices_returns_none() -> None:
    assert find_disputed_slices([], []) is None
    assert find_disputed_slices([_slice(0, 0.0, 1.0, "x")], []) is None
    # disjoint indices: nothing pairs up
    assert (
        find_disputed_slices([_slice(0, 0.0, 1.0, "x")], [_slice(7, 0.0, 1.0, "y")])
        is None
    )


def test_adjacent_disputed_slices_merge() -> None:
    a = [
        _slice(0, 0.0, 2.0, "yksi tarina alkaa näin"),
        _slice(1, 2.2, 4.0, "ja jatkuu tähän malliin"),
    ]
    b = [
        _slice(0, 0.0, 2.0, "kokonaan toinen aihe tässä", language="fi"),
        _slice(1, 2.2, 4.0, "lisää eri sisältöä tulossa", language="fi"),
    ]
    spans = find_disputed_slices(a, b)
    assert spans is not None
    assert len(spans) == 1  # 0.2s gap merges (SPAN_MERGE_GAP_S)
    assert spans[0].start == 0.0
    assert spans[0].end == 4.0


def test_overflow_keeps_the_most_severe_disputes() -> None:
    a, b = [], []
    for i in range(MAX_SPANS + 10):
        # Spread far apart so nothing merges; even i are mild disputes,
        # odd i are total disputes.
        start = i * 100.0
        a.append(_slice(i, start, start + 1.0, "sana toinen kolmas nelj"))
        if i % 2 == 0:
            b.append(_slice(i, start, start + 1.0, "sana toinen jotain muuta"))
        else:
            b.append(_slice(i, start, start + 1.0, "zzz qqq www rrr"))
    spans = find_disputed_slices(a, b)
    assert spans is not None
    assert len(spans) <= MAX_SPANS
    # The total disputes (odd starts) must all survive the trim.
    odd_survivors = [s for s in spans if (s.start // 100.0) % 2 == 1]
    assert len(odd_survivors) == (MAX_SPANS + 10) // 2
