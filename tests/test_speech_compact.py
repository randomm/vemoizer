"""Compacting the owner's speech spans (issue #209).

The mic stem is decoded from a dense buffer of its own speech, so bridging,
compacting and the timeline mapping are all pure functions tested here. No
models, no audio I/O, no network.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.speech_compact import (
    BRIDGE_GAP_S,
    CompactMap,
    bridge_spans,
    compact_spans,
    part_of,
    remap_result,
    to_original,
)

SR = SAMPLE_RATE


def _spans(*pairs: tuple[float, float]) -> list[dict[str, float]]:
    return [{"start": s, "end": e} for s, e in pairs]


# -- bridge_spans ---------------------------------------------------------


def test_bridge_spans_joins_gaps_shorter_than_the_maximum() -> None:
    bridged = bridge_spans(_spans((0.0, 1.0), (1.4, 2.0), (2.6, 3.0)), max_gap_s=1.0)
    assert bridged == [{"start": 0.0, "end": 3.0}]


def test_bridge_spans_keeps_a_leakage_stretch_split() -> None:
    """A gap of exactly the maximum is NOT bridged: it may be remote speech."""
    bridged = bridge_spans(_spans((0.0, 1.0), (2.0, 3.0)), max_gap_s=1.0)
    assert bridged == [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]


def test_bridge_spans_repairs_the_frame_gates_fragmentation() -> None:
    """The real shape: one 6 s sentence the 30 ms gate chopped into fragments."""
    fragments = [(60.0 + 0.5 * i, 60.3 + 0.5 * i) for i in range(12)]
    bridged = bridge_spans(_spans(*fragments))
    assert bridged == [{"start": 60.0, "end": 65.8}]


def test_bridge_spans_sorts_and_does_not_mutate_the_input() -> None:
    spans = _spans((5.0, 6.0), (0.0, 1.0))
    before = copy.deepcopy(spans)
    bridged = bridge_spans(spans, max_gap_s=0.5)
    assert bridged == [{"start": 0.0, "end": 1.0}, {"start": 5.0, "end": 6.0}]
    assert spans == before
    assert bridged[0] is not spans[0]


def test_bridge_spans_empty_input() -> None:
    assert bridge_spans([]) == []


def test_bridge_spans_default_gap_is_the_documented_one() -> None:
    assert bridge_spans(_spans((0.0, 1.0), (1.0 + BRIDGE_GAP_S, 2.0))) == _spans(
        (0.0, 1.0), (1.0 + BRIDGE_GAP_S, 2.0)
    )


# -- compact_spans --------------------------------------------------------


def test_compact_spans_pads_each_part_and_records_it() -> None:
    audio = np.full(SR, 0.5, dtype=np.float32)
    compact, cmap = compact_spans(audio, _spans((0.4, 0.6)), pad_s=0.25, gap_s=0.5)
    assert compact.dtype == audio.dtype
    assert cmap.gap == round(0.5 * SR)
    assert cmap.parts == ((0, round(0.15 * SR), round(0.7 * SR)),)
    assert len(compact) == round(0.7 * SR)
    assert compact == pytest.approx(0.5)
    assert audio == pytest.approx(0.5), "input must not be mutated"


def test_compact_spans_clamps_padding_to_the_recording() -> None:
    audio = np.ones(SR, dtype=np.float32)
    compact, cmap = compact_spans(audio, _spans((-5.0, 30.0)))
    assert cmap.parts == ((0, 0, SR),)
    assert len(compact) == SR


def test_compact_spans_merges_padded_spans_that_touch() -> None:
    audio = np.ones(SR, dtype=np.float32)
    compact, cmap = compact_spans(audio, _spans((0.2, 0.3), (0.4, 0.5)), pad_s=0.25)
    assert cmap.parts == ((0, 0, round(0.75 * SR)),)  # -0.05 clamped, 0.3+0.25 joins
    assert len(compact) == round(0.75 * SR)


def test_compact_spans_separates_parts_with_gap_silence() -> None:
    audio = np.ones(2 * SR, dtype=np.float32)
    compact, cmap = compact_spans(
        audio, _spans((0.1, 0.2), (0.7, 0.8)), pad_s=0.0, gap_s=0.5
    )
    part = round(0.1 * SR)
    gap = round(0.5 * SR)
    assert cmap.parts == ((0, part, part), (part + gap, round(0.7 * SR), part))
    assert len(compact) == 2 * part + gap
    assert compact[:part] == pytest.approx(1.0)
    assert not compact[part : part + gap].any(), "gap is silence"
    assert compact[part + gap :] == pytest.approx(1.0)


def test_compact_spans_keeps_the_input_dtype() -> None:
    for dtype in (np.float32, np.float64):
        audio = np.ones(SR, dtype=dtype)
        compact, _ = compact_spans(audio, _spans((0.1, 0.2), (0.7, 0.8)))
        assert compact.dtype == dtype


def test_compact_spans_without_a_gap_stores_parts_back_to_back() -> None:
    audio = np.ones(SR, dtype=np.float32)
    _compact, cmap = compact_spans(
        audio, _spans((0.1, 0.2), (0.7, 0.8)), pad_s=0.0, gap_s=0.0
    )
    assert [part[0] for part in cmap.parts] == [0, round(0.1 * SR)]


def test_compact_spans_drops_spans_outside_the_recording() -> None:
    audio = np.ones(SR, dtype=np.float32)
    compact, cmap = compact_spans(audio, _spans((30.0, 40.0)))
    assert cmap.parts == ()
    assert len(compact) == 0


def test_compact_spans_with_no_spans_is_empty() -> None:
    audio = np.ones(SR, dtype=np.float32)
    compact, cmap = compact_spans(audio, [])
    assert len(compact) == 0
    assert compact.dtype == audio.dtype
    assert cmap.parts == ()


# -- to_original / part_of ------------------------------------------------


# part 0: compact 0.0-1.0 s  -> original 100.0-101.0 s
# gap:    compact 1.0-1.5 s
# part 1: compact 1.5-2.5 s  -> original 200.0-201.0 s
_MAP = CompactMap(
    parts=((0, round(100.0 * SR), SR), (round(1.5 * SR), round(200.0 * SR), SR)),
    gap=round(0.5 * SR),
)


def test_to_original_inside_a_part_is_linear() -> None:
    assert to_original(0.0, _MAP) == 100.0
    assert to_original(0.25, _MAP) == 100.25
    assert to_original(2.0, _MAP) == 200.5


def test_to_original_in_a_gaps_clamps_to_the_preceding_part_end() -> None:
    assert to_original(1.0, _MAP) == 101.0
    assert to_original(1.2, _MAP) == 101.0  # first half of the gap: part before
    assert to_original(1.3, _MAP) == 200.0  # second half: start of the part after


def test_to_original_before_the_first_part_and_past_the_end() -> None:
    assert to_original(-1.0, _MAP) == 100.0
    assert to_original(9.0, _MAP) == 201.0


def test_to_original_with_no_parts_is_zero() -> None:
    assert to_original(3.0, CompactMap()) == 0.0


def test_part_of_names_the_part_or_the_one_before_a_gap() -> None:
    assert part_of(0.2, _MAP) == 0
    assert part_of(1.0, _MAP) == 0
    assert part_of(1.2, _MAP) == 0  # first half of the gap: the part before
    assert part_of(1.3, _MAP) == 1  # second half: the part after
    assert part_of(1.5, _MAP) == 1
    assert part_of(9.0, _MAP) == 1
    assert part_of(-1.0, _MAP) == 0


def test_part_of_with_no_parts_has_none() -> None:
    assert part_of(0.5, CompactMap()) == -1


# -- remap_result ---------------------------------------------------------


def _compact_result() -> dict[str, Any]:
    """A decode result shaped like ``decode_meeting``'s, on compact time."""
    words = [
        {"word": " a", "start": 0.1, "end": 0.3},
        {"word": " b", "start": 0.4, "end": 0.6},
        {"word": " c", "start": 1.6, "end": 1.8},
        {"word": " d", "start": 1.9, "end": 2.8},
    ]
    return {
        "text": "a b c d",
        "words": [dict(w) for w in words],
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "a b c d",
                "words": [dict(w) for w in words],
                "avg_logprob": -0.5,
            }
        ],
        "slices": [
            {"index": 0, "start_s": 0.0, "end_s": 1.0, "words": [dict(words[0])]},
            {"index": 1, "start_s": 1.5, "end_s": 2.5, "words": [dict(words[2])]},
        ],
        "lost_windows": [(1.1, 1.4)],
        "language": "fi",
    }


def test_remap_result_maps_word_times_and_clamps_end_to_its_part() -> None:
    remapped = remap_result(_compact_result(), _MAP)
    assert [(w["start"], w["end"]) for w in remapped["words"]] == [
        (100.1, 100.3),
        (100.4, 100.6),
        (200.1, 200.3),
        (200.4, 201.0),  # 201.4 would cross into the next speech span
    ]
    assert remapped["words"][0]["word"] == " a", "other keys are copied"


def test_remap_result_splits_a_segment_that_spans_two_parts() -> None:
    remapped = remap_result(_compact_result(), _MAP)
    segments = remapped["segments"]
    assert len(segments) == 2, "the two spans are far apart in the recording"
    first, second = segments
    assert (first["start"], first["end"], first["text"]) == (100.1, 100.6, "a b")
    assert (second["start"], second["end"], second["text"]) == (200.1, 201.0, "c d")
    assert first["avg_logprob"] == -0.5 == second["avg_logprob"], "keys are copied"
    assert [w["word"] for w in first["words"]] == [" a", " b"]
    assert [w["word"] for w in second["words"]] == [" c", " d"]


def test_remap_result_keeps_a_segment_without_words_on_the_timeline() -> None:
    result = _compact_result()
    result["segments"] = [{"start": 1.2, "end": 1.4, "text": "hallucinated"}]
    result["words"] = []
    remapped = remap_result(result, _MAP)
    assert remapped["segments"] == [
        {"start": 101.0, "end": 101.0, "text": "hallucinated"}
    ]


def test_remap_result_sorts_segments_by_start() -> None:
    result = _compact_result()
    result["segments"] = [
        {"start": 1.6, "end": 2.4, "text": "c d"},
        {"start": 0.1, "end": 0.6, "text": "a b"},
    ]
    remapped = remap_result(result, _MAP)
    assert [s["start"] for s in remapped["segments"]] == [100.1, 200.1]


def test_remap_result_maps_slices_and_lost_windows() -> None:
    remapped = remap_result(_compact_result(), _MAP)
    assert [(s["start_s"], s["end_s"]) for s in remapped["slices"]] == [
        (100.0, 101.0),
        (200.0, 201.0),
    ]
    assert remapped["slices"][0]["index"] == 0
    assert remapped["slices"][0]["words"] == [
        {"word": " a", "start": 100.1, "end": 100.3}
    ]
    assert remapped["slices"][1]["words"][0]["start"] == 200.1
    assert remapped["lost_windows"] == [(101.0, 101.0)]


def test_remap_result_keeps_unknown_keys_and_the_shape() -> None:
    result = _compact_result()
    result["audio_duration"] = 2.5
    remapped = remap_result(result, _MAP)
    assert remapped["audio_duration"] == 2.5
    assert remapped["language"] == "fi"
    assert remapped["text"] == "a b c d"


def test_remap_result_does_not_mutate_its_input() -> None:
    result = _compact_result()
    before = copy.deepcopy(result)
    remapped = remap_result(result, _MAP)
    assert result == before
    assert remapped is not result
    assert remapped["words"] is not result["words"]
    assert remapped["segments"][0] is not result["segments"][0]


def test_remap_result_tolerates_missing_optional_keys() -> None:
    remapped = remap_result({"words": [], "text": ""}, _MAP)
    assert remapped["words"] == []
    assert "segments" not in remapped
    assert "slices" not in remapped
    assert "lost_windows" not in remapped


def test_a_word_stamped_just_before_its_part_lands_in_that_part() -> None:
    """Whisper can stamp a part's first word a little early, inside the gap."""
    result = {"words": [{"word": "Hei", "start": 1.4, "end": 1.7}], "segments": []}
    word = remap_result(result, _MAP)["words"][0]
    assert (word["start"], word["end"]) == (200.0, pytest.approx(200.2))


def test_a_split_segment_joins_its_words_with_spaces() -> None:
    words = [
        {"word": "Hei", "start": 0.1, "end": 0.2},
        {"word": " mä", "start": 0.3, "end": 0.4},
        {"word": "joo", "start": 1.6, "end": 1.8},
    ]
    result = {"words": words, "segments": [{"start": 0.1, "end": 1.8, "text": "x"}]}
    texts = [s["text"] for s in remap_result(result, _MAP)["segments"]]
    assert texts == ["Hei mä", "joo"]


def test_a_segment_inside_one_part_keeps_whispers_text() -> None:
    words = [
        {"word": "AI", "start": 0.1, "end": 0.2},
        {"word": "-agentin", "start": 0.2, "end": 0.4},
    ]
    result = {
        "words": words,
        "segments": [{"start": 0.1, "end": 0.4, "text": " AI-agentin"}],
    }
    assert remap_result(result, _MAP)["segments"][0]["text"] == "AI-agentin"


def test_a_word_stamped_before_its_segment_still_lands_in_it() -> None:
    words = [
        {"word": " a", "start": 0.08, "end": 0.2},
        {"word": " b", "start": 0.4, "end": 0.6},
        {"word": " c", "start": 1.6, "end": 1.8},
    ]
    result = {
        "words": words,
        "segments": [{"start": 0.1, "end": 1.8, "text": " a b c"}],
    }
    texts = [s["text"] for s in remap_result(result, _MAP)["segments"]]
    assert texts == ["a b", "c"]


def test_a_word_between_segments_goes_to_the_nearer_one() -> None:
    words = [
        {"word": " a", "start": 0.1, "end": 0.2},
        {"word": " b", "start": 0.55, "end": 0.6},
        {"word": " c", "start": 0.7, "end": 0.8},
    ]
    result = {
        "words": words,
        "segments": [
            {"start": 0.1, "end": 0.2, "text": " a", "words": []},
            {"start": 0.6, "end": 0.8, "text": " b c", "words": []},
        ],
    }
    segments = remap_result(result, _MAP)["segments"]
    assert [[w["word"] for w in s["words"]] for s in segments] == [[" a"], [" b", " c"]]


def test_segment_words_are_not_the_top_level_word_objects() -> None:
    result = {
        "words": [{"word": " a", "start": 0.1, "end": 0.2}],
        "segments": [{"start": 0.1, "end": 0.2, "text": " a", "words": []}],
    }
    remapped = remap_result(result, _MAP)
    assert remapped["segments"][0]["words"][0] is not remapped["words"][0]
