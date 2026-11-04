"""Tests for the owner-quiz probe metrics of the accept bench (issue #243).

All synthetic segment dicts: no models, no network, no real memos.
"""

from __future__ import annotations

from typing import Any

import pytest

from vemoizer.accept_probes import (
    glossary_probe_insertions,
    probe_metrics,
    speaker_probe_metrics,
)


def hyp_seg(
    start: float, end: float, speaker: str | None, text: str = "x"
) -> dict[str, Any]:
    seg: dict[str, Any] = {"start": start, "end": end, "text": text}
    if speaker is not None:
        seg["speaker"] = speaker
    return seg


def probe(start: float, end: float, answer: str) -> dict[str, Any]:
    return {"id": f"p{start}", "start": start, "end": end, "answer": answer}


def term(start: float, end: float, options: list[str], answer: str) -> dict[str, Any]:
    return {
        "id": f"t{start}",
        "start": start,
        "end": end,
        "options": options,
        "answer": answer,
    }


# Four 5 s probes: Anna, Anna, Ben, Ben, each covered by its own hyp turn.
_PROBES = [
    probe(0, 5, "Anna"),
    probe(5, 10, "Anna"),
    probe(10, 15, "Ben"),
    probe(15, 20, "Ben"),
]
_PERFECT = [
    hyp_seg(0, 5, "Anna"),
    hyp_seg(5, 10, "Anna"),
    hyp_seg(10, 15, "Ben"),
    hyp_seg(15, 20, "Ben"),
]


# --- speaker metrics -------------------------------------------------------


def test_perfect_hypothesis_scores_one_with_no_splits_or_merges() -> None:
    got = speaker_probe_metrics(_PERFECT, _PROBES)
    assert got["speaker_name_acc"] == pytest.approx(1.0)
    assert got["speaker_mapped_acc"] == pytest.approx(1.0)
    assert got["split_count"] == 0
    assert got["merge_count"] == 0


def test_wrong_name_moves_name_accuracy() -> None:
    hyp = [*_PERFECT[:3], hyp_seg(15, 20, "Anna")]  # last Ben labelled Anna
    got = speaker_probe_metrics(hyp, _PROBES)
    assert got["speaker_name_acc"] == pytest.approx(0.75)


def test_split_moves_split_count_not_merge() -> None:
    # Anna's two probes carry two hypothesis labels: one name, two labels.
    hyp = [
        hyp_seg(0, 5, "Anna"),
        hyp_seg(5, 10, "Anna 2"),
        hyp_seg(10, 15, "Ben"),
        hyp_seg(15, 20, "Ben"),
    ]
    got = speaker_probe_metrics(hyp, _PROBES)
    assert got["split_count"] == 1
    assert got["merge_count"] == 0


def test_merge_moves_merge_count_and_mapped_accuracy() -> None:
    # One hypothesis label covers both Anna and Ben.
    hyp = [hyp_seg(0, 10, "S1"), hyp_seg(10, 20, "S1")]
    got = speaker_probe_metrics(hyp, _PROBES)
    assert got["merge_count"] == 1
    assert got["split_count"] == 0
    # Mapping keeps the first name; only the Anna probes can match it.
    assert got["speaker_mapped_acc"] == pytest.approx(0.5)


def test_mapped_accuracy_recovers_correct_clustering_under_raw_labels() -> None:
    hyp = [hyp_seg(0, 10, "S1"), hyp_seg(10, 20, "S2")]
    got = speaker_probe_metrics(hyp, _PROBES[:2] + [probe(10, 20, "Ben")])
    assert got["speaker_name_acc"] == pytest.approx(0.0)
    assert got["speaker_mapped_acc"] == pytest.approx(1.0)


def test_no_overlapping_segment_counts_as_wrong() -> None:
    probes = [probe(0, 5, "Anna"), probe(50, 55, "Anna")]
    got = speaker_probe_metrics([hyp_seg(0, 5, "Anna")], probes)
    assert got["speaker_name_acc"] == pytest.approx(0.5)


def test_hypothesis_speaker_is_the_largest_overlap() -> None:
    # Ben holds 3 s of the 5 s probe window, Anna only 2 s.
    hyp = [hyp_seg(0, 2, "Anna"), hyp_seg(2, 5, "Ben")]
    got = speaker_probe_metrics(hyp, [probe(0, 5, "Ben")])
    assert got["speaker_name_acc"] == pytest.approx(1.0)


def test_unscored_answers_are_ignored() -> None:
    probes = [
        probe(0, 5, "Anna"),
        probe(5, 10, "other"),
        probe(10, 15, "overlap"),
        probe(15, 20, "unclear"),
    ]
    # Every unscored probe is labelled wrongly on purpose: none may count.
    hyp = [hyp_seg(0, 5, "Anna"), hyp_seg(5, 20, "Ben")]
    got = speaker_probe_metrics(hyp, probes)
    assert got["speaker_name_acc"] == pytest.approx(1.0)
    assert got["split_count"] == 0
    assert got["merge_count"] == 0


def test_only_unscored_answers_give_no_signal() -> None:
    got = speaker_probe_metrics(_PERFECT, [probe(0, 5, "unclear")])
    assert got["speaker_name_acc"] is None
    assert got["speaker_mapped_acc"] is None
    assert got["split_count"] == 0


def test_name_comparison_is_case_insensitive_on_the_first_name() -> None:
    # Answer is a first name, the hypothesis label a full name, in other case.
    got = speaker_probe_metrics(
        [hyp_seg(0, 5, "eero virtanen"), hyp_seg(5, 10, "ANTTI")],
        [probe(0, 5, "Eero"), probe(5, 10, "antti")],
    )
    assert got["speaker_name_acc"] == pytest.approx(1.0)


def test_unlabelled_hypothesis_segment_is_wrong_not_a_split() -> None:
    hyp = [hyp_seg(0, 5, None), hyp_seg(5, 10, "Anna")]
    got = speaker_probe_metrics(hyp, [probe(0, 5, "Anna"), probe(5, 10, "Anna")])
    assert got["speaker_name_acc"] == pytest.approx(0.5)
    assert got["split_count"] == 0


# --- glossary insertions ---------------------------------------------------


def test_term_insertion_is_counted() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "meillä on IWS tiimi")]
    got = glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS", "IBC"], "IBS")])
    assert got == 1


def test_correct_term_is_not_counted() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "meillä on IBS tiimi")]
    got = glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS", "IBC"], "IBS")])
    assert got == 0


def test_term_match_is_case_insensitive_whole_token() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "ibs ja iwsx")]
    got = glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS"], "IWS")])
    assert got == 1  # "ibs" counts; "iwsx" is a different token


def test_term_outside_probe_window_is_not_counted() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "IWS"), hyp_seg(10, 14, "Anna", "IWS")]
    got = glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS"], "IBS")])
    assert got == 1


def test_shorter_option_nested_in_the_answer_is_not_counted() -> None:
    """PR #244 review: "kontti" inside a correct "IBC-kontti" is no insertion."""
    hyp = [hyp_seg(0, 4, "Anna", "se oli IBC-kontti")]
    probes = [term(0, 4, ["kontti", "IBC-kontti", "IWS"], "IBC-kontti")]
    assert glossary_probe_insertions(hyp, probes) == 0


def test_shorter_option_outside_the_answer_run_still_counts() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "AI agentti ja AI")]
    assert (
        glossary_probe_insertions(hyp, [term(0, 4, ["AI", "AI agentti"], "AI agentti")])
        == 1
    )


def test_long_paragraph_counts_only_words_timed_in_the_window() -> None:
    """PR #244 review: a 60 s paragraph is not one window; a term said at
    40 s is not an insertion in a 10-14 s probe."""
    hyp = [
        hyp_seg(
            0,
            60,
            "Anna",
            " ".join(["sana"] * 10 + ["IWS"] + ["sana"] * 5 + ["IBC"] + ["sana"] * 8),
        )
    ]
    # 25 words over 60 s: IWS sits at ~25 s, IBC at ~40 s.
    assert glossary_probe_insertions(hyp, [term(23, 27, ["IWS", "IBC"], "IWS")]) == 0
    assert glossary_probe_insertions(hyp, [term(23, 27, ["IWS", "IBC"], "IBC")]) == 1


def test_unclear_term_answer_is_unscored() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "IWS IBS")]
    assert glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS"], "unclear")]) == 0


def test_something_else_counts_every_listed_option() -> None:
    hyp = [hyp_seg(0, 4, "Anna", "IWS IBS")]
    got = glossary_probe_insertions(hyp, [term(0, 4, ["IWS", "IBS"], "other")])
    assert got == 2


def test_no_term_probes_gives_no_glossary_signal() -> None:
    assert glossary_probe_insertions(_PERFECT, []) is None


# --- combined ---------------------------------------------------------------


def test_probe_metrics_publishes_the_five_keys() -> None:
    got = probe_metrics(_PERFECT, _PROBES, [term(0, 5, ["IWS"], "IWS")])
    assert set(got) == {
        "speaker_name_acc",
        "speaker_mapped_acc",
        "split_count",
        "merge_count",
        "glossary_insertions",
    }
    assert got["glossary_insertions"] == 0
