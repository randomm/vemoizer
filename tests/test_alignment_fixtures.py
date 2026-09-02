"""Conformance loader for the golden alignment fixtures.

Each fixture in ``tests/fixtures/alignment/*.json`` describes a pair of
decoded word lists (decode A, decode B) plus the expected set of disputed
spans that the alignment stage must produce. The fixtures are the
conformance contract for the DTW + disputed-span flagging logic
(``src/vemoizer/alignment.py`` / ``src/vemoizer/spans.py``, issues 7 task-a
and 7 task-b). They exist so that the three task streams (A: DTW core,
B: disputed-span flagging, C: golden fixtures) can be developed and
reviewed in parallel against the same reference data.

This module does NOT implement the alignment algorithm. It provides:

- :func:`load_fixture` — parse and validate one fixture file.
- :func:`load_all_fixtures` — return every fixture in the directory.
- :class:`AlignmentFixture` — a lightweight, frozen dataclass the tests
  and the future algorithm implementation both consume.

The fixture schema is documented inline in each JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "alignment"


@dataclass(frozen=True)
class Word:
    """A single word from a decode: text plus onset/offset in seconds."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class DisputedSpan:
    """A time region where decode A and decode B disagree."""

    start: float
    end: float
    a_words: tuple[str, ...]
    b_words: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentFixture:
    """One golden alignment fixture: a word-list pair plus expected spans."""

    name: str
    description: str
    a: tuple[Word, ...]
    b: tuple[Word, ...]
    language_a: str | None
    language_b: str | None
    expected_disputed_spans: tuple[DisputedSpan, ...]
    path: Path


def _parse_words(raw: list[dict[str, Any]]) -> tuple[Word, ...]:
    """Validate and convert raw JSON word dicts to :class:`Word` tuples.

    Enforces the invariant that each word has a positive duration
    (``end > start``) and that the word list is time-ordered
    (non-decreasing ``start``). These are the two properties the DTW
    algorithm relies on; catching them at load time rather than at
    alignment time makes fixture typos impossible to ship.
    """
    words: list[Word] = []
    for i, entry in enumerate(raw):
        word = str(entry.get("word", ""))
        if not word:
            raise ValueError(f"word at index {i} has empty 'word' field")
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", 0.0))
        if end <= start:
            raise ValueError(
                f"word at index {i} ('{word}') has non-positive duration: "
                f"start={start}, end={end}"
            )
        if words and start < words[-1].start:
            raise ValueError(
                f"word list is not time-ordered: word at index {i} "
                f"('{word}', start={start}) precedes the previous word's start"
            )
        words.append(Word(word=word, start=start, end=end))
    return tuple(words)


def _parse_span(raw: dict[str, Any]) -> DisputedSpan:
    """Validate and convert a raw expected-span dict to :class:`DisputedSpan`."""
    start = float(raw.get("start", 0.0))
    end = float(raw.get("end", 0.0))
    if end <= start:
        raise ValueError(
            f"expected span has non-positive duration: start={start}, end={end}"
        )
    a_words = tuple(str(w) for w in raw.get("a_words", []))
    b_words = tuple(str(w) for w in raw.get("b_words", []))
    return DisputedSpan(start=start, end=end, a_words=a_words, b_words=b_words)


def load_fixture(path: Path) -> AlignmentFixture:
    """Load and validate a single alignment fixture from *path*.

    Raises:
        ValueError: if the fixture violates the word-list invariants
            (empty word, non-positive duration, out-of-order onsets)
            or if an expected span has a non-positive duration.
        json.JSONDecodeError: if the file is not valid JSON.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    a_words = _parse_words(data.get("a", []))
    b_words = _parse_words(data.get("b", []))
    spans_raw = data.get("expected_disputed_spans", [])
    spans = tuple(_parse_span(s) for s in spans_raw)
    return AlignmentFixture(
        name=path.stem,
        description=data.get("description", ""),
        a=a_words,
        b=b_words,
        language_a=data.get("language_a"),
        language_b=data.get("language_b"),
        expected_disputed_spans=spans,
        path=path,
    )


def load_all_fixtures() -> list[AlignmentFixture]:
    """Load every ``*.json`` fixture in ``tests/fixtures/alignment/``.

    Returns:
        A list of :class:`AlignmentFixture` in sorted filename order.
        The list is empty if the directory does not exist (so the
        conformance test can be run in a worktree that has not yet
        merged the fixture commit without crashing the suite).
    """
    if not FIXTURES_DIR.is_dir():
        return []
    fixtures: list[AlignmentFixture] = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixtures.append(load_fixture(path))
    return fixtures


# ---------------------------------------------------------------------------
# Conformance tests
# ---------------------------------------------------------------------------


class TestFixtureConformance:
    """Schema and invariant conformance for the golden alignment fixtures."""

    def test_fixtures_directory_exists(self) -> None:
        """The fixture directory must exist and contain at least one file."""
        assert FIXTURES_DIR.is_dir(), f"missing fixture dir: {FIXTURES_DIR}"
        files = list(FIXTURES_DIR.glob("*.json"))
        assert len(files) >= 1, f"no .json fixtures in {FIXTURES_DIR}"

    def test_all_fixtures_load_without_error(self) -> None:
        """Every fixture must parse and validate cleanly."""
        fixtures = load_all_fixtures()
        assert len(fixtures) >= 1, "load_all_fixtures returned an empty list"
        for fixture in fixtures:
            assert fixture.name, f"fixture {fixture.path} has empty name"
            assert fixture.description, (
                f"fixture {fixture.path} is missing a description"
            )

    def test_word_lists_are_time_ordered(self) -> None:
        """Both decode A and decode B must be non-decreasing in ``start``."""
        for fixture in load_all_fixtures():
            for label, words in (("a", fixture.a), ("b", fixture.b)):
                for i in range(1, len(words)):
                    assert words[i].start >= words[i - 1].start, (
                        f"fixture {fixture.name}: decode {label} word "
                        f"at index {i} ({words[i].word!r}) starts before "
                        f"the previous word's start"
                    )

    def test_words_have_positive_duration(self) -> None:
        """Every word must satisfy ``end > start``."""
        for fixture in load_all_fixtures():
            for label, words in (("a", fixture.a), ("b", fixture.b)):
                for w in words:
                    assert w.end > w.start, (
                        f"fixture {fixture.name}: decode {label} word "
                        f"{w.word!r} has end <= start"
                    )

    def test_expected_spans_are_positive_duration(self) -> None:
        """Every expected disputed span must satisfy ``end > start``."""
        for fixture in load_all_fixtures():
            for span in fixture.expected_disputed_spans:
                assert span.end > span.start, (
                    f"fixture {fixture.name}: span [{span.start}, {span.end}] "
                    f"has non-positive duration"
                )

    def test_identical_fixture_has_zero_spans(self) -> None:
        """The 'identical' fixture must declare zero disputed spans."""
        fixture = next((f for f in load_all_fixtures() if f.name == "identical"), None)
        assert fixture is not None, "identical.json fixture missing"
        assert fixture.expected_disputed_spans == (), (
            f"identical fixture declares {len(fixture.expected_disputed_spans)} "
            f"disputed spans, expected 0"
        )

    def test_substitution_fixture_flags_exactly_one_span(self) -> None:
        """The 'substitution-drift' fixture must declare exactly one span."""
        fixture = next(
            (f for f in load_all_fixtures() if f.name == "substitution-drift"),
            None,
        )
        assert fixture is not None, "substitution-drift.json fixture missing"
        assert len(fixture.expected_disputed_spans) == 1, (
            f"substitution-drift fixture declares "
            f"{len(fixture.expected_disputed_spans)} spans, expected 1"
        )
        span = fixture.expected_disputed_spans[0]
        assert len(span.a_words) == 1 and len(span.b_words) == 2, (
            f"substitution-drift span should be 1 word in A vs 2 words in B, "
            f"got A={span.a_words}, B={span.b_words}"
        )

    def test_codeswitch_fixture_has_a_word_drop(self) -> None:
        """The 'codeswitch-en-drop' fixture must have a shorter B word list.

        The whole point of the codeswitch fixture is that one model drops
        the English acronym, so ``len(b) < len(a)`` is the invariant.
        """
        fixture = next(
            (f for f in load_all_fixtures() if f.name == "codeswitch-en-drop"),
            None,
        )
        assert fixture is not None, "codeswitch-en-drop.json fixture missing"
        assert len(fixture.b) < len(fixture.a), (
            f"codeswitch fixture: decode B ({len(fixture.b)} words) must be "
            f"shorter than decode A ({len(fixture.a)} words) — one model "
            f"must drop the English acronym"
        )
        assert len(fixture.expected_disputed_spans) >= 1, (
            "codeswitch fixture must declare at least one disputed span"
        )

    def test_insertion_fixture_has_a_word_drop(self) -> None:
        """The 'insertion-deletion' fixture must have a shorter B word list.

        Decode B is missing the word 'syy', so ``len(b) == len(a) - 1``.
        """
        fixture = next(
            (f for f in load_all_fixtures() if f.name == "insertion-deletion"),
            None,
        )
        assert fixture is not None, "insertion-deletion.json fixture missing"
        assert len(fixture.b) == len(fixture.a) - 1, (
            f"insertion-deletion fixture: decode B ({len(fixture.b)} words) "
            f"should be exactly 1 shorter than decode A ({len(fixture.a)})"
        )
        span = fixture.expected_disputed_spans[0]
        assert len(span.a_words) == 1 and len(span.b_words) == 0, (
            f"insertion-deletion span should be 1 word in A, 0 words in B, "
            f"got A={span.a_words}, B={span.b_words}"
        )

    def test_threshold_fixture_declares_zero_spans(self) -> None:
        """The 'threshold-boundary' fixture must declare zero spans.

        All words are identical, so the similarity is 1.0 — well above
        any reasonable threshold. The boundary note in the fixture
        documents the threshold-comparison contract for the
        implementation, but the expected outcome for this specific
        fixture (identical words) is zero disputed spans.
        """
        fixture = next(
            (f for f in load_all_fixtures() if f.name == "threshold-boundary"),
            None,
        )
        assert fixture is not None, "threshold-boundary.json fixture missing"
        assert fixture.expected_disputed_spans == (), (
            "threshold-boundary fixture (identical words) should declare "
            "zero disputed spans"
        )

    def test_fixture_files_match_expected_names(self) -> None:
        """The five canonical fixture names must all be present."""
        expected = {
            "identical",
            "substitution-drift",
            "codeswitch-en-drop",
            "insertion-deletion",
            "threshold-boundary",
        }
        actual = {f.name for f in load_all_fixtures()}
        missing = expected - actual
        assert not missing, f"missing fixture files: {sorted(missing)}"

    def test_span_words_are_substrings_of_their_decode(self) -> None:
        """Every word in a span's a_words / b_words must appear in the
        corresponding decode's word list. This catches fixture typos where
        the expected span references a word that was never transcribed."""
        for fixture in load_all_fixtures():
            a_word_set = {w.word for w in fixture.a}
            b_word_set = {w.word for w in fixture.b}
            for span in fixture.expected_disputed_spans:
                for w in span.a_words:
                    assert w in a_word_set, (
                        f"fixture {fixture.name}: span a_word {w!r} not found "
                        f"in decode A words {sorted(a_word_set)}"
                    )
                for w in span.b_words:
                    assert w in b_word_set, (
                        f"fixture {fixture.name}: span b_word {w!r} not found "
                        f"in decode B words {sorted(b_word_set)}"
                    )


class TestFixtureLoaderEdgeCases:
    """Edge cases for the loader itself (not the algorithm)."""

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An empty JSON file should raise (not silently produce a fixture)."""
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_fixture(p)

    def test_missing_word_field_raises(self, tmp_path: Path) -> None:
        """A word dict without a 'word' key should raise."""
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps({"a": [{"start": 0.0, "end": 0.1}], "b": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="empty 'word'"):
            load_fixture(p)

    def test_negative_duration_raises(self, tmp_path: Path) -> None:
        """A word with end <= start should raise."""
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "a": [{"word": "x", "start": 0.5, "end": 0.3}],
                    "b": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-positive duration"):
            load_fixture(p)

    def test_out_of_order_words_raises(self, tmp_path: Path) -> None:
        """A word list where starts decrease should raise."""
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps(
                {
                    "a": [
                        {"word": "x", "start": 0.5, "end": 0.6},
                        {"word": "y", "start": 0.3, "end": 0.4},
                    ],
                    "b": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not time-ordered"):
            load_fixture(p)

    def test_empty_decode_lists_are_valid(self, tmp_path: Path) -> None:
        """A fixture with empty word lists is schema-valid (the algorithm
        must handle empty input; the conformance test just verifies the
        loader doesn't reject it)."""
        p = tmp_path / "empty.json"
        p.write_text(
            json.dumps({"a": [], "b": [], "expected_disputed_spans": []}),
            encoding="utf-8",
        )
        fixture = load_fixture(p)
        assert fixture.a == ()
        assert fixture.b == ()
        assert fixture.expected_disputed_spans == ()
