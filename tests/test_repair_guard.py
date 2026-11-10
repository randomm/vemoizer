"""Token guard on accepted LLM repairs (issue #272).

The paragraph-level guard (length + whole similarity) accepts a long
paragraph even when one word was swapped for an unrelated one, when words
were inserted, or when a run of content was dropped. ``guard_tokens``
re-walks the accepted candidate and reverts exactly those edits. Pure
function over strings — no model, no network.
"""

from __future__ import annotations

from vemoizer.repair_guard import GuardStats, guard_tokens


def test_phonetic_fix_is_kept() -> None:
    text, stats = guard_tokens("me parastaa tätä", "me parantaa tätä")
    assert text == "me parantaa tätä"
    assert (stats.kept, stats.reverted_replace) == (1, 0)


def test_unrelated_swap_is_reverted() -> None:
    """Glossary over-steering: 'IBS' is not a garbled 'IBC-kontti'."""
    original = "meillä on IBS tänään"
    text, stats = guard_tokens(original, "meillä on IBC-kontti tänään")
    assert text == original
    assert (stats.kept, stats.reverted_replace) == (0, 1)


def test_unrelated_swap_cannot_hide_behind_adjacent_phonetic_fix() -> None:
    """SequenceMatcher merges the two edits into one replace run; the
    run's aggregate similarity must not smuggle the swap through."""
    text, stats = guard_tokens(
        "eteni rotkeasti IBS nousi", "eteni rohkeasti IBC-kontti nousi"
    )
    assert text == "eteni rohkeasti IBS nousi"
    assert (stats.kept, stats.reverted_replace) == (1, 1)


def test_equal_count_run_judges_each_pair_1to1() -> None:
    text, stats = guard_tokens(
        "meillä parastaa IBS tänään", "meillä parantaa IBC-kontti tänään"
    )
    assert text == "meillä parantaa IBS tänään"
    assert (stats.kept, stats.reverted_replace) == (1, 1)


def test_equal_count_run_with_two_phonetic_pairs_keeps_both() -> None:
    """Each kept pair is counted individually, not as one merged run."""
    text, stats = guard_tokens(
        "meillä parastaa rotkeasti tänään", "meillä parantaa rohkeasti tänään"
    )
    assert text == "meillä parantaa rohkeasti tänään"
    assert (stats.kept, stats.reverted_replace) == (2, 0)


def test_inserted_word_is_reverted() -> None:
    text, stats = guard_tokens("teemme uusia asioita", "teemme uusia lisää asioita")
    assert text == "teemme uusia asioita"
    assert (stats.kept, stats.reverted_insert) == (0, 1)


def test_short_filler_deletion_is_kept() -> None:
    """One or two dropped words are filler removal, not lost content."""
    text, stats = guard_tokens("mä sanoin sen että niinku", "mä sanoin sen")
    assert text == "mä sanoin sen"
    assert stats.reverted_delete == 0


def test_long_content_deletion_is_reverted() -> None:
    text, stats = guard_tokens("mä sanoin sen että niinku joo", "mä sanoin sen")
    assert text == "mä sanoin sen että niinku joo"
    assert stats.reverted_delete == 1


def test_unclear_marker_is_kept() -> None:
    """The prompt's [epäselvä] escape hatch is never guessed back into words."""
    text, stats = guard_tokens("sdkfj qwe", "[epäselvä]")
    assert text == "[epäselvä]"
    assert (stats.kept, stats.reverted_replace) == (1, 0)


def test_case_and_punctuation_changes_are_kept() -> None:
    """Alignment runs on normalized tokens, so casing/punctuation are equal."""
    text, stats = guard_tokens("hei maailma", "Hei maailma!")
    assert text == "Hei maailma!"
    assert stats == GuardStats()


def test_empty_inputs_return_the_original_untouched() -> None:
    empty = GuardStats(0, 0, 0, 0)
    for original, candidate in (("", "x"), ("   ", "x"), ("abc", ""), ("abc", "  ")):
        assert guard_tokens(original, candidate) == (original, empty)


def test_stats_count_every_kind_of_run() -> None:
    """One candidate exercising all four counters at once."""
    original = "minä sanoin rotkeasti että IBS on hankala juttu nyt"
    candidate = "todella minä sanoin rohkeasti että IBC-kontti on"
    text, stats = guard_tokens(original, candidate)
    # phonetic fix kept, swap and 3-word deletion restored, insert dropped
    assert text == "minä sanoin rohkeasti että IBS on hankala juttu nyt"
    assert stats == GuardStats(
        kept=1, reverted_replace=1, reverted_insert=1, reverted_delete=1
    )
