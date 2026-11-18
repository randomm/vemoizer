"""Island rejection in speaker clip selection (issue #280).

A single short segment sandwiched inside another speaker's monologue is a
diarization mislabel: playing it plays that other person's voice. Islands are
judged among items of the same ``track`` only, so a genuine reply of the
remote side between two of the owner's segments is never an island. These
tests cover :func:`islands` itself and the rejection of such candidates from
both the segment pool and the paragraph pools, including the issue #202 last
resort for a label whose only candidates are islands.
"""

from __future__ import annotations

from typing import Any

from vemoizer.speaker_clip_islands import ISLAND_GAP_S, islands
from vemoizer.speaker_clip_select import select_clips


def _para(
    start: float,
    end: float,
    text: str = "lorem ipsum dolor sit amet consectetur",
    speaker: str | None = "A",
) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def _seg(
    start: float,
    end: float,
    text: str = "segment own words here",
    speaker: str | None = "A",
    track: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
    }
    if track is not None:
        item["track"] = track
    return item


# The real 7665 shape: SPEAKER_03's 2.4 s item wedged inside SPEAKER_02's
# monologue, touching both neighbours (one ends 7665.0, the next starts 7667.5).
ISLAND_START, ISLAND_END = 7665.1, 7667.5
# C's genuine turn: shorter than the island, so ranking alone would prefer
# the mislabel.
OTHER_START, OTHER_END = 6700.0, 6702.2


def _starts(clips: list[Any]) -> list[float]:
    return [c.start_s for c in clips]


class TestIslands:
    def test_sandwiched_item_is_an_island(self):
        items = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, 7700.0, speaker="B"),
        ]
        assert islands(items) == {"C": [(ISLAND_START, ISLAND_END)]}

    def test_gap_over_threshold_on_either_side_is_not_an_island(self):
        def items(gap_before: float, gap_after: float) -> list[dict[str, Any]]:
            return [
                _seg(7600.0, ISLAND_START - gap_before, speaker="B"),
                _seg(ISLAND_START, ISLAND_END, speaker="C"),
                _seg(ISLAND_END + gap_after, 7700.0, speaker="B"),
            ]

        assert islands(items(ISLAND_GAP_S, ISLAND_GAP_S)) == {
            "C": [(ISLAND_START, ISLAND_END)]
        }
        assert islands(items(ISLAND_GAP_S + 0.1, 0.0)) == {}
        assert islands(items(0.0, ISLAND_GAP_S + 0.1)) == {}

    def test_real_call_1_56_s_gap_is_an_island(self):
        # Regression (issue #280): the real mislabelled sentence, SPEAKER_04
        # 6782.43-6786.01 inside SPEAKER_02's monologue (ends 6780.87, next
        # starts 6786.01): a 1.56 s before-gap, missed at the old 1.5 s
        # threshold. Whisper segment boundaries leave 1-3 s gaps inside one
        # speaker's monologue, hence ISLAND_GAP_S = 3.0.
        items = [
            _seg(6770.0, 6780.87, speaker="SPEAKER_02"),
            _seg(6782.43, 6786.01, speaker="SPEAKER_04"),
            _seg(6786.01, 6795.0, speaker="SPEAKER_02"),
        ]
        assert islands(items) == {"SPEAKER_04": [(6782.43, 6786.01)]}

    def test_remote_reply_between_owner_segments_is_not_an_island(self):
        # Two-track call sidecar: the owner's mic-track ("me") segments are
        # interleaved in time with the remote side, so a genuine remote
        # reply between two owner segments has no same-track neighbours and
        # must not be judged as an island.
        items = [
            _seg(100.0, 103.0, speaker="Janni", track="me"),
            _seg(103.5, 106.0, speaker="SPEAKER_04", track="them"),
            _seg(106.5, 109.0, speaker="Janni", track="me"),
        ]
        assert islands(items) == {}

    def test_owner_segment_between_two_remote_ones_is_not_an_island(self):
        # The mirror case: an owner segment between two remote ones.
        items = [
            _seg(100.0, 103.0, speaker="SPEAKER_04", track="them"),
            _seg(103.5, 106.0, speaker="Janni", track="me"),
            _seg(106.5, 109.0, speaker="SPEAKER_04", track="them"),
        ]
        assert islands(items) == {}

    def test_island_detected_across_interleaved_other_track(self):
        # A them-track island is still found when a me-track segment sits
        # in between in time: neighbours are the same-track SPEAKER_02
        # items, and the me segment is ignored by their group.
        items = [
            _seg(100.0, 103.0, speaker="SPEAKER_02", track="them"),
            _seg(103.2, 105.0, speaker="Janni", track="me"),
            _seg(105.3, 108.0, speaker="SPEAKER_04", track="them"),
            _seg(108.2, 111.0, speaker="SPEAKER_02", track="them"),
        ]
        assert islands(items) == {"SPEAKER_04": [(105.3, 108.0)]}

    def test_different_neighbour_labels_is_not_an_island(self):
        items = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, 7700.0, speaker="D"),
        ]
        assert islands(items) == {}

    def test_first_and_last_item_are_never_islands(self):
        leading = [
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, ISLAND_END + 1.0, speaker="B"),
            _seg(ISLAND_END + 1.0, 7700.0, speaker="B"),
        ]
        trailing = [
            _seg(7600.0, ISLAND_START - 1.0, speaker="B"),
            _seg(ISLAND_START - 1.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
        ]
        assert islands(leading) == {}
        assert islands(trailing) == {}

    def test_unlabelled_items_are_not_islands(self):
        # The unlabelled item sits exactly where an island would sit: even
        # so, unlabelled items are never islands.
        items = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker=None),
            _seg(ISLAND_END, 7700.0, speaker="B"),
        ]
        assert islands(items) == {}

    def test_unlabelled_items_do_not_count_as_neighbours(self):
        # The unlabelled item sits between the C item and the trailing B
        # item: skipping it makes the two B items neighbours of C, so C is
        # an island (if unlabelled items counted as neighbours, C's
        # neighbours would differ and it would not be one).
        items = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, ISLAND_END + 1.0, speaker=None),
            _seg(ISLAND_END + 1.0, 7700.0, speaker="B"),
        ]
        assert islands(items) == {"C": [(ISLAND_START, ISLAND_END)]}


class TestSelectClipsRejectsIslands:
    @staticmethod
    def _sidecar() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Segments and paragraphs of the 7665 shape, with C elsewhere too."""
        paras = [
            _para(6600.0, 6690.0, "A pitka alkupuheenvuoro tássa", "A"),
            _para(OTHER_START, OTHER_END, "C aito lyhyt vuoro", "C"),
            _para(6800.0, 6900.0, "A jatkaa puhumista", "A"),
            _para(7600.0, ISLAND_START, "B pitka monologi jatkuu", "B"),
            _para(ISLAND_START, ISLAND_END, "C saareke kesken", "C"),
            _para(ISLAND_END, 7700.0, "B jatkaa monologia", "B"),
        ]
        segs = [
            _seg(6600.0, 6690.0, "A pitka alkupuheenvuoro tássa", "A"),
            _seg(OTHER_START, OTHER_END, "C aito lyhyt vuoro", "C"),
            _seg(6800.0, 6900.0, "A jatkaa puhumista", "A"),
            _seg(7600.0, ISLAND_START, "B pitka monologi jatkuu", "B"),
            _seg(ISLAND_START, ISLAND_END, "C saareke kesken", "C"),
            _seg(ISLAND_END, 7700.0, "B jatkaa monologia", "B"),
        ]
        return paras, segs

    def test_sandwiched_segment_never_picked_when_other_candidate_exists(self):
        paras, segs = self._sidecar()
        assert _starts(select_clips(paras, segs)["C"]) == [OTHER_START]
        assert _starts(select_clips(paras, segs, per_speaker=1)["C"]) == [OTHER_START]

    def test_the_island_is_a_clean_pickable_candidate(self):
        # Guards the test above against a vacuous pass: with C's real turn
        # removed the island is C's only candidate and gets picked, so what
        # kept it out is the island rule and not another gate.
        paras, segs = self._sidecar()
        paras_wo = [
            p for p in paras if not (p["speaker"] == "C" and p["start"] == OTHER_START)
        ]
        segs_wo = [
            s for s in segs if not (s["speaker"] == "C" and s["start"] == OTHER_START)
        ]
        assert _starts(select_clips(paras_wo, segs_wo)["C"]) == [ISLAND_START]

    def test_paragraph_fallback_also_skips_the_island(self):
        # C's only clean segment IS the island, but C has a long paragraph
        # elsewhere: the paragraph fallback must supply that clip instead.
        paras = [
            _para(6700.0, 6730.0, "C pitka puheenvuoro tássa lauseessa", "C"),
            _para(7600.0, ISLAND_START, "B pitka monologi jatkuu", "B"),
            _para(ISLAND_START, ISLAND_END, "C saareke kesken", "C"),
            _para(ISLAND_END, 7700.0, "B jatkaa monologia", "B"),
        ]
        segs = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, 7700.0, speaker="B"),
        ]
        assert _starts(select_clips(paras, segs, per_speaker=2)["C"]) == [6700.0]

    def test_label_whose_only_candidates_are_islands_still_gets_a_clip(self):
        paras = [
            _para(7600.0, ISLAND_START, "B pitka monologi jatkuu", "B"),
            _para(ISLAND_START, ISLAND_END, "C saareke kesken", "C"),
            _para(ISLAND_END, 7700.0, "B jatkaa monologia", "B"),
        ]
        segs = [
            _seg(7600.0, ISLAND_START, speaker="B"),
            _seg(ISLAND_START, ISLAND_END, speaker="C"),
            _seg(ISLAND_END, 7700.0, speaker="B"),
        ]
        assert _starts(select_clips(paras, segs)["C"]) == [ISLAND_START]
        assert _starts(select_clips(paras, [])["C"]) == [ISLAND_START]

    def test_old_sidecar_without_segments_detects_islands_in_paragraphs(self):
        paras = [
            _para(6700.0, 6730.0, "C pitka puheenvuoro tássa lauseessa", "C"),
            _para(7600.0, ISLAND_START, "B pitka monologi jatkuu", "B"),
            _para(ISLAND_START, ISLAND_END, "C saareke kesken", "C"),
            _para(ISLAND_END, 7700.0, "B jatkaa monologia", "B"),
        ]
        assert _starts(select_clips(paras, [])["C"]) == [6700.0]

        # Same shape, but the trailing B paragraph starts beyond
        # ISLAND_GAP_S later: no longer an island, so the pickable
        # candidate is available again.
        loose = [p for p in paras if p["start"] != ISLAND_END] + [
            _para(ISLAND_END + ISLAND_GAP_S + 0.5, 7700.0, "B jatkaa monologia", "B")
        ]
        assert _starts(select_clips(loose, [])["C"]) == [6700.0, ISLAND_START]
