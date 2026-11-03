"""Tests for the accept-benchmark metrics (issue #228).

All synthetic data: no models, no network, no real memos.
"""

from __future__ import annotations

from typing import Any

import pytest

from vemoizer import accept_metrics
from vemoizer.accept_metrics import (
    flag_quality,
    glossary_insertions,
    loop_seconds,
    score_excerpt,
    speaker_accuracy,
    split_merge_counts,
    title_hits,
    transcript_wer,
)

# --- synthetic builders ----------------------------------------------------


def ref_seg(start: float, end: float, speaker: str, text: str) -> dict[str, Any]:
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def hyp_seg(
    start: float,
    end: float,
    speaker: str | None,
    text: str,
    *,
    suspect: bool | None = None,
) -> dict[str, Any]:
    seg: dict[str, Any] = {"start": start, "end": end, "text": text}
    if speaker is not None:
        seg["speaker"] = speaker
    if suspect is not None:
        seg["suspect"] = suspect
    return seg


# --- speaker_accuracy ------------------------------------------------------


def test_speaker_accuracy_perfect_labels() -> None:
    ref = [ref_seg(0, 10, "Eero", "a"), ref_seg(10, 20, "Antti", "b")]
    hyp = [hyp_seg(0, 10, "Eero", "a"), hyp_seg(10, 20, "Antti", "b")]
    acc = speaker_accuracy(ref, hyp)
    assert acc["name_acc"] == pytest.approx(1.0)
    assert acc["mapped_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_is_case_insensitive() -> None:
    ref = [ref_seg(0, 10, "Eero", "a")]
    hyp = [hyp_seg(0, 10, "eero", "a")]
    assert speaker_accuracy(ref, hyp)["name_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_swapped_labels_map_to_one() -> None:
    ref = [ref_seg(0, 10, "Eero", "a"), ref_seg(10, 20, "Antti", "b")]
    hyp = [hyp_seg(0, 10, "SPEAKER_00", "a"), hyp_seg(10, 20, "SPEAKER_01", "b")]
    acc = speaker_accuracy(ref, hyp)
    assert acc["name_acc"] == pytest.approx(0.0)
    assert acc["mapped_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_time_weighted() -> None:
    # Eero right for 10 s of 40 s of scorable reference time.
    ref = [ref_seg(0, 10, "Eero", "a"), ref_seg(10, 20, "Eero", "b")]
    hyp = [hyp_seg(0, 10, "Eero", "a"), hyp_seg(10, 20, "Vale", "b")]
    assert speaker_accuracy(ref, hyp)["name_acc"] == pytest.approx(0.5)


def test_speaker_accuracy_ignores_unclear_reference() -> None:
    ref = [ref_seg(0, 10, "unclear", "a")]
    hyp: list[dict[str, Any]] = []
    acc = speaker_accuracy(ref, hyp)
    assert acc["name_acc"] == pytest.approx(1.0)
    assert acc["mapped_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_empty_inputs() -> None:
    acc = speaker_accuracy([], [])
    assert acc["name_acc"] == pytest.approx(1.0)
    assert acc["mapped_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_missing_hyp_speaker_never_matches() -> None:
    ref = [ref_seg(0, 10, "Eero", "a")]
    hyp = [hyp_seg(0, 10, None, "a")]
    assert speaker_accuracy(ref, hyp)["name_acc"] == pytest.approx(0.0)


def test_speaker_accuracy_overlapping_hyp_segments_not_double_counted() -> None:
    # PR #231 review: a duplicated hyp segment must not paper over the
    # wrong-speaker tail - only 8 of the 10 scorable seconds are Eero.
    ref = [ref_seg(0, 10, "Eero", "a")]
    hyp = [
        hyp_seg(0, 8, "Eero", "a"),
        hyp_seg(0, 8, "Eero", "a"),
        hyp_seg(8, 10, "Antti", "b"),
    ]
    assert speaker_accuracy(ref, hyp)["name_acc"] == pytest.approx(0.8)


def test_speaker_accuracy_unclear_is_case_insensitive() -> None:
    # Attribution status must not hinge on how the reviewer capitalised it.
    ref = [ref_seg(0, 10, "Unclear", "a"), ref_seg(10, 20, "Eero", "b")]
    hyp = [hyp_seg(10, 20, "Eero", "b")]
    assert speaker_accuracy(ref, hyp)["name_acc"] == pytest.approx(1.0)


def test_speaker_accuracy_mapped_never_below_name() -> None:
    # Greedy label swapping must never score raw labels below name_acc:
    # "alice"/"bob" are already correct, yet greedy maps "alice" -> "bob".
    ref = [ref_seg(0, 5, "Alice", "a"), ref_seg(5, 15, "Bob", "b")]
    hyp = [hyp_seg(0, 11, "alice", "a"), hyp_seg(11, 15, "bob", "b")]
    acc = speaker_accuracy(ref, hyp)
    assert acc["name_acc"] == pytest.approx(0.6)
    assert acc["mapped_acc"] >= acc["name_acc"]


# --- split_merge_counts ----------------------------------------------------


def test_split_one_speaker_over_two_labels() -> None:
    ref = [ref_seg(0, 60, "Eero", "x")]
    hyp = [hyp_seg(0, 30, "SPEAKER_00", "x"), hyp_seg(30, 60, "SPEAKER_01", "y")]
    assert split_merge_counts(ref, hyp) == (1, 0)


def test_merge_two_speakers_under_one_label() -> None:
    ref = [ref_seg(0, 30, "Antti", "a"), ref_seg(30, 60, "Jukka", "b")]
    hyp = [hyp_seg(0, 60, "SPEAKER_00", "ab")]
    assert split_merge_counts(ref, hyp) == (0, 1)


def test_split_merge_below_time_threshold_not_counted() -> None:
    # The 2-second second label is below both min_share and min_seconds.
    ref = [ref_seg(0, 60, "Eero", "x")]
    hyp = [hyp_seg(0, 58, "SPEAKER_00", "x"), hyp_seg(58, 60, "SPEAKER_01", "y")]
    assert split_merge_counts(ref, hyp) == (0, 0)


def test_split_merge_ignores_unclear_reference() -> None:
    ref = [ref_seg(0, 60, "unclear", "x")]
    hyp = [hyp_seg(0, 30, "SPEAKER_00", "x"), hyp_seg(30, 60, "SPEAKER_01", "y")]
    assert split_merge_counts(ref, hyp) == (0, 0)


def test_split_merge_empty_inputs() -> None:
    assert split_merge_counts([], []) == (0, 0)


def test_split_duplicate_label_does_not_inflate_split() -> None:
    # The duplicated SPEAKER_00 covers the same 30 s twice; only one
    # *other* label really holds the speaker, so the split count is 1.
    ref = [ref_seg(0, 60, "Eero", "x")]
    hyp = [
        hyp_seg(0, 30, "SPEAKER_00", "x"),
        hyp_seg(0, 30, "SPEAKER_00", "x"),
        hyp_seg(30, 60, "SPEAKER_01", "y"),
    ]
    assert split_merge_counts(ref, hyp) == (1, 0)


def test_split_unlabelled_hyp_segment_is_not_a_speaker() -> None:
    # "" (missing speaker) is wrong in the accuracy metrics, but it is
    # not a speaker, so it cannot split a reference speaker in two.
    ref = [ref_seg(0, 60, "A", "x")]
    hyp = [hyp_seg(0, 30, "SPEAKER_00", "x"), hyp_seg(30, 60, None, "y")]
    assert split_merge_counts(ref, hyp) == (0, 0)


def test_merge_unlabelled_hyp_segments_are_not_a_speaker() -> None:
    ref = [ref_seg(0, 30, "Antti", "a"), ref_seg(30, 60, "Jukka", "b")]
    hyp = [hyp_seg(0, 30, None, "a"), hyp_seg(30, 60, None, "b")]
    assert split_merge_counts(ref, hyp) == (0, 0)


# --- glossary_insertions ---------------------------------------------------


def test_glossary_insertion_counts_new_term_mention() -> None:
    n = glossary_insertions(
        "Kokouksessa mainittiin IWS.",
        "Kokouksessa mainittiin IBC-kontti.",
        ["IBC-kontti", "IWS"],
    )
    assert n == 1


def test_glossary_insertion_zero_when_reference_already_says_it() -> None:
    n = glossary_insertions(
        "Katso IBC-kontti.", "Katso IBC-kontti uudelleen.", ["IBC-kontti"]
    )
    assert n == 0


def test_glossary_insertion_is_case_insensitive() -> None:
    n = glossary_insertions("ei sanaa.", "Katso Ibc-Kontti.", ["ibc-kontti"])
    assert n == 1


def test_glossary_insertion_requires_whole_words() -> None:
    # "backlogi" contains "backlog" but must not count as a term mention.
    n = glossary_insertions("tyhja.", "paljon backlogi tavaraa.", ["backlog"])
    assert n == 0


def test_glossary_insertion_multiword_term() -> None:
    n = glossary_insertions(
        "mitaan.", "avoin lähdekoodi projekti", ["avoin lähdekoodi"]
    )
    assert n == 1


def test_glossary_insertion_empty_inputs() -> None:
    assert glossary_insertions("", "", ["IBC-kontti"]) == 0
    assert glossary_insertions("a", "IBC-kontti", []) == 0


def test_glossary_insertion_nested_terms_count_twice() -> None:
    # Documented behaviour: each listed term is scored on its own, so
    # "AI" and "AI agentti" both fire inside "AI agentti".
    assert glossary_insertions("mitaan.", "AI agentti", ["AI", "AI agentti"]) == 2


# --- title_hits ------------------------------------------------------------


def test_title_hits_clean_title_scores_zero() -> None:
    hits = title_hits("IWS ideointi", {"must_contain": ["IWS"], "forbid": ["kontti"]})
    assert hits["must_contain_missing"] == 0
    assert hits["forbid_hits"] == 0


def test_title_hits_missing_must_contain() -> None:
    hits = title_hits("Ideointi", {"must_contain": ["IWS", "roadmap"], "forbid": []})
    assert hits["must_contain_missing"] == 2


def test_title_hits_forbid_hit_case_insensitive() -> None:
    hits = title_hits("Kontti-ideo", {"must_contain": [], "forbid": ["KONTTI"]})
    assert hits["forbid_hits"] == 1


def test_title_hits_inflected_forbid_stem_hits() -> None:
    # 2026-11-03: reviewers write stems; "kont*" catches "kontti" and "kontin".
    hits = title_hits("IBC-kontin ideointi", {"must_contain": [], "forbid": ["kont*"]})
    assert hits["forbid_hits"] == 1


def test_title_hits_plain_term_is_whole_word() -> None:
    # A plain (non-stem) term matches whole words only: "sam" is a prefix
    # of "sama" but must not hit it; "AI" must not match "aina".
    hits = title_hits("Aina sama", {"must_contain": ["AI"], "forbid": ["sam"]})
    assert hits == {"must_contain_missing": 1, "forbid_hits": 0}
    exact = title_hits("Aina sam", {"must_contain": [], "forbid": ["sam"]})
    assert exact["forbid_hits"] == 1


def test_title_hits_term_that_textnorms_empty_is_ignored() -> None:
    # A pure-punctuation term can never match: neither missing nor a hit.
    hits = title_hits("mika tahansa", {"must_contain": ["***"], "forbid": ["!!!"]})
    assert hits == {"must_contain_missing": 0, "forbid_hits": 0}


def test_title_hits_multi_token_stem_in_order() -> None:
    hits = title_hits("IWS:n AI-työkalut", {"must_contain": ["ai työ*"], "forbid": []})
    assert hits["must_contain_missing"] == 0


def test_title_hits_term_split_by_textnorm() -> None:
    hits = title_hits("IWS:n ideointi", {"must_contain": ["IWS:n"], "forbid": []})
    assert hits["must_contain_missing"] == 0


def test_title_hits_empty_expectation() -> None:
    hits = title_hits("mika tahansa", {"must_contain": [], "forbid": []})
    assert hits == {"must_contain_missing": 0, "forbid_hits": 0}


# --- loop_seconds ----------------------------------------------------------


def test_loop_seconds_is_the_shared_detector() -> None:
    from vemoizer import loops

    assert accept_metrics.loop_seconds is loops.loop_seconds


def test_loop_seconds_measures_a_wall() -> None:
    wall = [hyp_seg(i, i + 1, "Eero", "Kiitos.") for i in range(6)]
    assert loop_seconds(wall) == pytest.approx(6.0)
    assert loop_seconds([]) == 0.0


# --- flag_quality ----------------------------------------------------------


def test_flag_quality_clean_segment_no_errors() -> None:
    ref = [ref_seg(0, 10, "Eero", "moro aami")]
    hyp = [hyp_seg(0, 10, "Eero", "moro aami")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == 0.0
    assert q["flag_precision"] is None  # nothing flagged
    assert q["flag_recall"] is None  # no errors


def test_flag_quality_unflagged_error_seconds() -> None:
    ref = [ref_seg(0, 10, "Eero", "moro aami")]
    hyp = [hyp_seg(0, 10, "Eero", "yksi kaksi kolme neli viisi")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(10.0)
    assert q["flag_recall"] == pytest.approx(0.0)


def test_flag_quality_flagged_error_gives_precision_and_recall() -> None:
    ref = [ref_seg(0, 10, "Eero", "moro aami")]
    hyp = [hyp_seg(0, 10, "Eero", "yksi kaksi kolme neli viisi", suspect=True)]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == 0.0
    assert q["flag_precision"] == pytest.approx(1.0)
    assert q["flag_recall"] == pytest.approx(1.0)


def test_flag_quality_flag_on_good_segment_kills_precision() -> None:
    ref = [ref_seg(0, 10, "Eero", "a b"), ref_seg(10, 20, "Eero", "c d e f g")]
    hyp = [
        hyp_seg(0, 10, "Eero", "a b", suspect=True),  # flagged but fine
        hyp_seg(10, 20, "Eero", "z z z z z", suspect=False),  # error, unflagged
    ]
    q = flag_quality(ref, hyp)
    assert q["flag_precision"] == pytest.approx(0.0)
    assert q["flag_recall"] == pytest.approx(0.0)
    assert q["unflagged_error_s"] == pytest.approx(10.0)


def test_flag_quality_orphan_segment_needs_three_words() -> None:
    hyp_only = [
        hyp_seg(30, 35, "Eero", "aika lyhyt"),
        hyp_seg(40, 45, "Eero", "yksi kaksi kolme"),
    ]
    q = flag_quality([], hyp_only)
    # Two-word orphan is not an error; three-word orphan is.
    assert q["flag_recall"] == pytest.approx(0.0)
    assert q["unflagged_error_s"] == pytest.approx(5.0)


def test_flag_quality_correct_resegmentation_not_punished() -> None:
    # A correct 30 s reference turn transcribed as five correct 6 s hyp
    # segments is clean: per-reference scoring must not punish that the
    # hypothesis chose different segment boundaries.
    words = [f"w{i}" for i in range(30)]
    ref = [ref_seg(0, 30, "Eero", " ".join(words))]
    hyp = [
        hyp_seg(i * 6, (i + 1) * 6, "Eero", " ".join(words[i * 6 : (i + 1) * 6]))
        for i in range(5)
    ]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == 0.0
    assert q["flag_recall"] is None


def test_flag_quality_hyp_spanning_turn_boundary_not_punished() -> None:
    # One correct hyp segment over two reference turns is clean: its words
    # belong to both turns, so neither may score as a deletion.
    ref = [ref_seg(0, 10, "Eero", "a b c"), ref_seg(10, 20, "Antti", "d e f")]
    hyp = [hyp_seg(0, 20, "Eero", "a b c d e f")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == 0.0
    assert q["flag_recall"] is None


def test_flag_quality_wrong_text_across_boundary_is_error() -> None:
    ref = [ref_seg(0, 10, "Eero", "a b c"), ref_seg(10, 20, "Antti", "d e f")]
    hyp = [hyp_seg(0, 20, "Eero", "x y z q r s")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(20.0)


def test_flag_quality_coarse_hyp_segment_keeps_local_error() -> None:
    # One hyp segment covering six 10 s turns, turn 2 entirely wrong: the
    # wrong turn must not be averaged away by its five correct neighbours.
    good = ["a b c", "d e f", "g h i", "j k l", "m n o", "p q r"]
    ref = [ref_seg(i * 10, (i + 1) * 10, "Eero", t) for i, t in enumerate(good)]
    said = list(good)
    said[1] = "x y z"
    hyp = [hyp_seg(0, 60, "Eero", " ".join(said))]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(10.0)


def test_flag_quality_jitter_overlap_does_not_chain_turns() -> None:
    # Hyp boundaries 0.01 s off the reference: the wrong middle turn stays
    # an error of its own.
    ref = [
        ref_seg(0, 10, "Eero", "a b c"),
        ref_seg(10, 20, "Antti", "d e f"),
        ref_seg(20, 30, "Eero", "g h i"),
    ]
    hyp = [
        hyp_seg(0, 10.01, "Eero", "a b c"),
        hyp_seg(9.99, 20.01, "Antti", "x y z"),
        hyp_seg(19.99, 30, "Eero", "g h i"),
    ]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(10.0)


def test_flag_quality_partly_wrong_across_boundary() -> None:
    # One hyp segment over two turns; only the second turn's words are wrong.
    ref = [ref_seg(0, 10, "Eero", "a b c"), ref_seg(10, 20, "Antti", "d e f")]
    hyp = [hyp_seg(0, 20, "Eero", "a b c x y z")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(10.0)


def test_flag_quality_segment_across_silence_not_invented() -> None:
    # A correct hyp segment spanning a 20 s pause must score like the same
    # text split at the pause: words never land in the silence.
    ref = [ref_seg(0, 10, "Eero", "a b c"), ref_seg(30, 40, "Antti", "d e f")]
    hyp = [hyp_seg(0, 40, "Eero", "a b c d e f")]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == 0.0
    assert q["flag_recall"] is None


def test_flag_quality_punctuation_only_orphan_is_not_invented() -> None:
    q = flag_quality([], [hyp_seg(0, 5, "Eero", "- ... , -")])
    assert q == {"unflagged_error_s": 0.0, "flag_precision": None, "flag_recall": None}


def test_flag_quality_invented_text_touching_a_turn_is_detected() -> None:
    # Known limit (segment-level timing): invented words glued to real
    # speech are weighted by the touched turn's seconds, not the silence
    # they really fill - but the turn is an unflagged error, never clean.
    ref = [ref_seg(0, 1, "Eero", "a")]
    hyp = [hyp_seg(0, 10, "Eero", "a " + " ".join(f"x{i}" for i in range(10)))]
    q = flag_quality(ref, hyp)
    assert q["unflagged_error_s"] == pytest.approx(1.0)
    assert q["flag_recall"] == pytest.approx(0.0)


def test_flag_quality_empty_inputs() -> None:
    q = flag_quality([], [])
    assert q == {"unflagged_error_s": 0.0, "flag_precision": None, "flag_recall": None}


# --- transcript_wer --------------------------------------------------------


def test_transcript_wer_joins_all_segment_texts() -> None:
    ref = [ref_seg(0, 5, "Eero", "moro aami"), ref_seg(5, 10, "Eero", "kiitos kiitos")]
    hyp = [hyp_seg(0, 10, "Eero", "moro aami kiitos kiitos")]
    assert transcript_wer(ref, hyp) == pytest.approx(0.0)


def test_transcript_wer_empty_against_reference() -> None:
    assert transcript_wer([ref_seg(0, 5, "Eero", "a b c")], []) == pytest.approx(1.0)


# --- score_excerpt ---------------------------------------------------------


def _regression_inputs() -> dict[str, Any]:
    """The 2026-11-03 failure: reference says IWS, hypothesis repairs to IBC-kontti."""
    return {
        "ref_segments": [ref_seg(0, 10, "Antti", "Kokouksessa mainittiin IWS.")],
        "hyp_segments": [hyp_seg(0, 10, "Antti", "Kokouksessa mainittiin IBC-kontti.")],
        "title": "IBC-kontin ideointi",
        "title_expect": {"must_contain": ["IWS"], "forbid": ["kont*"]},
        "glossary_terms": ["IBC-kontti", "IWS"],
    }


def test_score_excerpt_has_stable_flat_keys() -> None:
    result = score_excerpt([], [], title="", title_expect={}, glossary_terms=[])
    assert set(result) == {
        "speaker_name_acc",
        "speaker_mapped_acc",
        "split_count",
        "merge_count",
        "glossary_insertions",
        "loop_seconds",
        "unflagged_error_s",
        "flag_precision",
        "flag_recall",
        "wer",
        "title_must_contain_missing",
        "title_forbid_hits",
    }


def test_score_excerpt_2026_11_03_ibc_kontti_regression() -> None:
    bad = score_excerpt(**_regression_inputs())
    assert bad["glossary_insertions"] >= 1
    assert bad["title_forbid_hits"] >= 1

    clean_inputs = _regression_inputs()
    clean_inputs["hyp_segments"] = [
        hyp_seg(0, 10, "Antti", "Kokouksessa mainittiin IWS.")
    ]
    clean_inputs["title"] = "IWS ideointi"
    clean = score_excerpt(**clean_inputs)
    assert clean["glossary_insertions"] == 0
    assert clean["title_forbid_hits"] == 0
