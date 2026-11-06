"""Issue #268 rework: term-shaped word selection for the interactive check.

Synthetic sidecar shapes only — no models, no network. ``known_terms`` is
exercised with HOME and CWD pointed at a tmp dir (the real dev machine has a
live ``~/.vemoizer``), everything else is pure-function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

from vemoizer.term_review import (
    LIST_LIMIT,
    is_term_shaped,
    known_terms,
    normalize_token,
    select_terms,
)


def _word(word: str, start: float, p: float | None = None) -> dict[str, Any]:
    w: dict[str, Any] = {"word": word, "start": start, "end": start + 0.3}
    if p is not None:
        w["p"] = p
    return w


def _paragraphs(*texts: str) -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": t, "speaker": "SPEAKER_1"} for t in texts
    ]


class TestShape:
    def test_acronym_with_low_p_is_selected(self) -> None:
        words = [_word("AVS", 1.0, 0.21)]
        cands = select_terms(words, _paragraphs("Katsotaan AVS:t\u00e4."), set())
        assert [c.word for c in cands] == ["AVS"]
        assert cands[0].min_p == 0.21
        assert cands[0].score == pytest.approx((1.0 - 0.21) * 1.5)

    def test_colon_suffix_groups_with_the_stem(self) -> None:
        assert normalize_token("AVS:n,") == ("AVS:n", "AVS")
        words = [_word("AVS", 1.0, 0.4), _word("AVS:n", 5.0, 0.2)]
        cands = select_terms(words, _paragraphs("AVS ja AVS:n"), set())
        assert [(c.word, c.count, c.min_p) for c in cands] == [("AVS", 2, 0.2)]

    def test_plain_finnish_word_is_not_term_shaped(self) -> None:
        assert is_term_shaped("kokous", sentence_start=False) is False
        assert is_term_shaped("A", sentence_start=False) is False
        assert is_term_shaped("2024", sentence_start=False) is False
        assert is_term_shaped("AVS", sentence_start=True) is True

    def test_sentence_initial_capital_excluded_mid_sentence_included(self) -> None:
        words = [
            _word("Markku", 0.0, 0.1),
            _word("sanoi", 0.5, 0.9),
            _word("Klaudia", 1.2, 0.1),
            _word("sanoit?", 2.0, 0.9),
            _word("Matti", 2.6, 0.1),
            _word("hienoa!", 3.0, 0.9),
            _word("Liisa", 3.6, 0.1),
        ]
        text = "Markku sanoi Klaudia sanoit? Matti hienoa! Liisa"
        cands = select_terms(words, _paragraphs(text), set())
        # ? and ! end a sentence: Matti and Liisa are sentence-initial.
        assert [c.word for c in cands] == ["Klaudia"]

    def test_segment_start_word_is_sentence_initial(self) -> None:
        # A word at a VAD segment start is recognizer-capitalized prose,
        # not a mid-sentence term: "Joo" and "Päivää" must not be listed.
        # The word right before each sits inside the 1.5 s gap window, so
        # only the segment-start rule can spare them.
        words = [
            _word("kokous", 0.0, 0.9),
            _word("hyvä", 4.6, 0.9),
            _word("Joo", 5.0, 0.4),
            _word("kiitos", 9.7, 0.9),
            _word("Päivää", 10.02, 0.4),
        ]
        text = "kokous hyvä Joo kiitos Päivää"
        without = select_terms(words, _paragraphs(text), set())
        assert sorted(c.word for c in without) == ["Joo", "Päivää"]
        cands = select_terms(
            words, _paragraphs(text), set(), segment_starts=[5.0, 10.0]
        )
        assert cands == []

    def test_sentence_final_period_strips_from_stem(self) -> None:
        assert normalize_token("AVS.") == ("AVS", "AVS")
        assert normalize_token("S.R.") == ("S.R", "S.R")
        assert normalize_token("v1.2") == ("v1.2", "v1.2")
        words = [_word("AVS", 1.0, 0.4), _word("AVS.", 5.0, 0.2)]
        cands = select_terms(words, _paragraphs("AVS ja AVS."), set())
        assert [(c.word, c.count, c.min_p) for c in cands] == [("AVS", 2, 0.2)]
        # The dotted form still gets the acronym weight...
        only = select_terms([_word("AVS.", 1.0, 0.2)], _paragraphs("AVS."), set())
        assert only[0].score == pytest.approx((1.0 - 0.2) * 1.5)
        # ...and is excluded when the bare term is in the glossary.
        dotted = [_word("AVS.", 1.0, 0.2)]
        assert select_terms(dotted, _paragraphs("AVS."), {"avs"}) == []

    def test_time_gap_counts_as_sentence_start(self) -> None:
        words = [_word("kokous", 0.0, 0.5), _word("Markku", 6.0, 0.1)]
        assert select_terms(words, _paragraphs("kokous Markku"), set()) == []

    def test_punctuation_stripped_and_rare_letters(self) -> None:
        cands = select_terms(
            [_word("\u201cAVS\u201d,", 1.0, 0.2)], _paragraphs("AVS"), set()
        )
        assert [c.word for c in cands] == ["AVS"]
        assert (
            select_terms([_word("kokous", 1.0, 0.2)], _paragraphs("kokous"), set())
            == []
        )
        assert (
            len(select_terms([_word("Qiita", 1.0, 0.2)], _paragraphs("Qiita"), set()))
            == 1
        )


class TestFiltering:
    def test_known_words_excluded(self) -> None:
        words = [_word("AVS", 1.0, 0.1), _word("CDCI", 5.0, 0.1)]
        cands = select_terms(words, _paragraphs("AVS ja CDCI"), {"avs"})
        assert [c.word for c in cands] == ["CDCI"]

    def test_stem_absent_from_paragraphs_excluded(self) -> None:
        words = [_word("AVS", 1.0, 0.1)]
        assert select_terms(words, _paragraphs("Katsotaan juttu."), set()) == []

    def test_no_p_at_all_yields_nothing(self) -> None:
        words = [_word("AVS", 1.0), _word("CDCI", 5.0)]
        assert select_terms(words, _paragraphs("AVS CDCI"), set()) == []

    def test_list_mode_keeps_high_p_but_ranks_low_p_first(self) -> None:
        # Calibration rework: p alone does not find errors, so the list
        # keeps the confident-looking terms too — least sure just come
        # first (they get heard).
        words = [
            _word("AVS", 1.0, 0.96),
            _word("CDCI", 2.0, 0.1),
            _word("THS", 3.0, 0.75),
        ]
        cands = select_terms(words, _paragraphs("AVS CDCI THS"), set())
        assert [c.word for c in cands] == ["CDCI", "THS", "AVS"]

    def test_explicit_p_max_still_filters(self) -> None:
        words = [_word("AVS", 1.0, 0.95), _word("CDCI", 2.0, 0.1)]
        cands = select_terms(words, _paragraphs("AVS CDCI"), set(), p_max=0.6)
        assert [c.word for c in cands] == ["CDCI"]

    def test_limit_defaults_to_list_limit(self) -> None:
        words = [_word(f"X{i}Q", float(i), 0.1) for i in range(LIST_LIMIT + 5)]
        text = " ".join(f"X{i}Q" for i in range(LIST_LIMIT + 5))
        cands = select_terms(words, _paragraphs(text), set())
        assert len(cands) == LIST_LIMIT

    def test_lowest_p_occurrence_locates_the_playback_window(self) -> None:
        words = [
            _word("AVS", 1.0, 0.9),
            _word("AVS", 30.0, 0.05),
            _word("AVS", 60.0, 0.8),
        ]
        (cand,) = select_terms(words, _paragraphs("AVS AVS AVS"), set())
        assert (cand.start, cand.end) == (30.0, 30.3)
        assert cand.first_start == 1.0

    def test_known_terms_covers_glossary_people_speakers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        glossary = tmp_path / ".vemoizer" / "glossary.txt"
        glossary.parent.mkdir()
        glossary.write_text("@Codexia\nYki\nYksilo => Yksilo\n", encoding="utf-8")
        people = tmp_path / ".vemoizer" / "config.toml"
        people.write_text('people = ["Klaudia"]\n', encoding="utf-8")
        known = known_terms({"speaker_names": {"SPEAKER_1": "Matti"}})
        assert {"codexia", "yki", "yksilo", "klaudia", "matti"} <= known

    def test_known_terms_fail_open_on_unreadable_layers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        import vemoizer.glossary_layers as layers

        def boom(*a: Any, **k: Any) -> Any:
            raise OSError("unreadable")

        monkeypatch.setattr(layers, "load_layers", boom)
        assert known_terms({"speaker_names": {"S1": "Matti"}}) == {"matti"}

    def test_known_terms_fail_open_on_unreadable_people_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        import vemoizer.people_config as people

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("bad toml")

        monkeypatch.setattr(people, "read_people_list", boom)
        assert known_terms({"speaker_names": {"S1": "Matti"}}) == {"matti"}
