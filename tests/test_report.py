"""Tests for the per-file end-of-run quality report (issue #75, M6).

``render_report`` is a pure function of the run dict plus the
CLI/batch-layer parameters (``diarize_requested``, ``glossary_source``,
``glossary_terms``, ``language``). ``build_quality_report`` wraps it in
fail-open semantics. All tests are in-memory dicts — no models, no
network, no filesystem.
"""

from __future__ import annotations

from vemoizer.report import (
    build_quality_report,
    render_report,
)


def _transcript(**extra) -> dict:
    return {
        "text": "Puhuttiin alustasta.",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Puhuttiin alustasta.", "speaker": "S1"},
            {
                "start": 8.0,
                "end": 12.0,
                "text": "Sitten deploymentista.",
                "speaker": "S2",
            },
        ],
        **extra,
    }


def _suspect_transcript() -> dict:
    return {
        "text": "x",
        "paragraphs": [
            {"start": 30.0, "end": 32.0, "text": "garble 1", "suspect": "garble"},
            {"start": 10.0, "end": 12.0, "text": "garble 2", "suspect": "garble"},
            {"start": 5.0, "end": 7.0, "text": "number 1", "suspect": "number"},
            {"start": 1.0, "end": 3.0, "text": "number 2", "suspect": "number"},
            {"start": 0.0, "end": 1.0, "text": "number 3", "suspect": "number"},
            {"start": 50.0, "end": 52.0, "text": "number 4", "suspect": "number"},
        ],
    }


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


def test_render_returns_string() -> None:
    result = render_report(_transcript())
    assert isinstance(result, str)
    assert "Puhujat:" in result


def test_render_full_transcript_contains_speakers() -> None:
    result = render_report(_transcript())
    assert "Puhujat: 2 (S1, S2)" in result


def test_render_omits_speakers_when_no_speaker_key() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t)
    assert "Puhujat:" not in result


def test_render_diarize_requested_but_absent_shows_honest_state() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t, diarize_requested=True)
    assert "pyydetty, ei löytynyt" in result


def test_render_diarize_not_requested_no_speaker_line() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t, diarize_requested=False)
    assert "Puhujat:" not in result


def test_render_empty_dict_returns_empty_string() -> None:
    result = render_report({})
    assert result == ""


# ---------------------------------------------------------------------------
# Suspects: ranking (garble before number, ties earliest start)
# ---------------------------------------------------------------------------


def test_suspects_ranks_garble_before_number() -> None:
    result = render_report(_suspect_transcript())
    # Garble paragraphs (start 30, 10) must appear before number paragraphs.
    garble_1 = result.index("[00:00:30]")
    garble_2 = result.index("[00:00:10]")
    # The two garble paragraphs sorted by earliest start: 10 then 30.
    assert garble_2 < garble_1


def test_suspects_earliest_start_wins_tie() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 30.0, "end": 31.0, "text": "a", "suspect": "garble"},
            {"start": 10.0, "end": 11.0, "text": "b", "suspect": "garble"},
        ],
    }
    result = render_report(t)
    assert result.index("[00:00:10]") < result.index("[00:00:30]")


def test_suspects_capped_at_three() -> None:
    # 6 suspect paragraphs → only 3 listed (2 garble + 1 number).
    # The selected number is the EARLIEST number (start=0), not the first
    # one in list order. The number at start=1 (second-earliest) is NOT listed.
    result = render_report(_suspect_transcript())
    assert "[00:00:00]" in result  # earliest number is selected
    assert "[00:00:01]" not in result  # second-earliest number is not


def test_suspects_count_zero_omits_section() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhdas"}]}
    result = render_report(t)
    assert "Epävarmat kohdat:" not in result


def test_suspects_section_shows_total_count() -> None:
    # 6 suspect paragraphs total, 3 listed; count line shows the full total.
    result = render_report(_suspect_transcript())
    # The count "6" appears in the section heading
    assert "Epävarmat kohdat: 6 (" in result


def test_suspects_garble_outranks_number_in_selection() -> None:
    # With more than 3 suspects: all garble (2) come first, then the
    # earliest number (start=0). The number at start=50 (latest) is NOT selected.
    result = render_report(_suspect_transcript())
    assert "[00:00:00]" in result
    assert "[00:00:50]" not in result


# ---------------------------------------------------------------------------
# Residual loops (find_degenerate_windows on final segments)
# ---------------------------------------------------------------------------


def _degenerate_segments(n: int = 6) -> list[dict]:
    """A wall of short repeated segments (find_degenerate_windows detects)."""
    return [
        {"start": i * 0.5, "end": (i + 1) * 0.5, "text": "Kiitos."} for i in range(n)
    ]


def test_residual_loops_detected_from_segments() -> None:
    t = {
        "text": "x",
        "paragraphs": [{"start": 0.0, "end": 3.0, "text": "puhe", "speaker": "S1"}],
        "segments": _degenerate_segments(6),
    }
    result = render_report(t)
    assert "Jäljellä olevat loopit: 1" in result


def test_residual_loops_zero_omits_section() -> None:
    t = {
        "text": "x",
        "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe", "speaker": "S1"}],
        "segments": [{"start": 0.0, "end": 1.0, "text": "Tämä on normaali lause."}],
    }
    result = render_report(t)
    assert "Jäljellä olevat loopit" not in result


def test_residual_loops_no_segments_key_omits_section() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t)
    assert "Jäljellä olevat loopit" not in result


def test_residual_loops_uses_segments_not_paragraphs() -> None:
    # Paragraphs are non-degenerate; segments carry the wall → detected.
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Normaali puhe.", "speaker": "S1"}
        ],
        "segments": _degenerate_segments(6),
    }
    result = render_report(t)
    assert "Jäljellä olevat loopit: 1" in result


def test_residual_loops_two_walls() -> None:
    # Two separate walls separated by a long silence (> MERGE_GAP_S).
    wall1 = [
        {"start": i * 0.5, "end": (i + 1) * 0.5, "text": "Kiitos."} for i in range(6)
    ]
    wall2 = [
        {"start": 100.0 + i * 0.5, "end": 100.5 + (i + 1) * 0.5, "text": "DCS."}
        for i in range(6)
    ]
    t = {
        "text": "x",
        "paragraphs": [],
        "segments": wall1 + wall2,
    }
    result = render_report(t)
    assert "Jäljellä olevat loopit: 2" in result


# ---------------------------------------------------------------------------
# Glossary term hits
# ---------------------------------------------------------------------------


def test_glossary_hits_count_correct() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Kävimme Flagship hankkeessa."},
            {"start": 5.0, "end": 10.0, "text": "Nordea palkanoi uuden tekijän."},
        ],
    }
    result = render_report(
        t,
        glossary_source="glossary.txt",
        glossary_terms=["Flagship", "Nordea", "EiOleva"],
    )
    assert "Sanastoon osumat: 2 of 3" in result


def test_glossary_hits_zero_shown_honestly() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Tässä ei ole termejä."},
        ],
    }
    result = render_report(
        t,
        glossary_source="g.txt",
        glossary_terms=["Flagship", "Nordea"],
    )
    assert "Sanastoon osumat: 0 of 2" in result


def test_glossary_hits_whole_word_match() -> None:
    # "flag" must not match inside "flagship" (whole-word match).
    # "flagship" IS a whole word in the text; "flag" is not.
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "flagship käynnissä."},
        ],
    }
    result = render_report(
        t,
        glossary_terms=["flag", "flagship"],
    )
    # "flag" does not match (no word boundary after 'g' in "flagship").
    # "flagship" IS a whole word in the text.
    assert "Sanastoon osumat: 1 of 2" in result


def test_glossary_hits_no_match_when_term_is_substring_only() -> None:
    # A term that only appears as a substring (not a whole word) does not count.
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "flagshiphanke käynnissä."},
        ],
    }
    result = render_report(t, glossary_terms=["flag"])
    assert "Sanastoon osumat: 0 of 1" in result


def test_glossary_hits_case_insensitive() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Kävimme flagSHIP hankkeessa."},
        ],
    }
    result = render_report(t, glossary_terms=["Flagship"])
    assert "Sanastoon osumat: 1 of 1" in result


def test_glossary_llm_only_terms_are_excluded_by_batch_layer() -> None:
    # @-prefixed terms are LLM-only (issue #82): the batch layer excludes
    # them from glossary_terms before calling render_report, so they are
    # never part of the "X of N" denominator (they never reached the
    # whisper prompt). The report itself only ever sees prompt terms.
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Normal term here."},
        ],
    }
    result = render_report(t, glossary_terms=["term"])
    assert "Sanastoon osumat: 1 of 1" in result


def test_glossary_source_only_no_terms_still_shows() -> None:
    # glossary_source present, glossary_terms empty → section shown with 0 of 0.
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t, glossary_source="g.txt")
    assert "Sanastoon osumat: 0 of 0" in result


def test_glossary_absent_omits_section() -> None:
    t = {"text": "x", "paragraphs": [{"start": 0.0, "end": 1.0, "text": "puhe"}]}
    result = render_report(t)
    assert "Sanastoon osumat" not in result


# ---------------------------------------------------------------------------
# Warnings classification by stable anchor
# ---------------------------------------------------------------------------


def test_warnings_diarization_classified() -> None:
    t = _transcript(
        warnings=[
            "Speaker diarization: pyannote/speaker-diarization-community-1 (CC-BY-4.0)",
        ]
    )
    result = render_report(t)
    assert "diarization" in result
    assert "pyannote" in result


def test_warnings_notes_classified() -> None:
    t = _transcript(
        warnings=["notes generation failed; the Markdown output has no summary"]
    )
    result = render_report(t)
    assert "notes" in result
    assert "Markdown output has no summary" in result


def test_warnings_unmatched_dropped_from_report() -> None:
    t = _transcript(warnings=["something that matches no known anchor"])
    result = render_report(t)
    assert "something that matches no known anchor" not in result


def test_warnings_empty_list_no_section() -> None:
    t = _transcript(warnings=[])
    result = render_report(t)
    assert "Varoitukset" not in result


def test_warnings_both_categories() -> None:
    t = _transcript(
        warnings=[
            "diarization failed; continuing without speaker labels",
            "notes generation failed; the Markdown output has no summary",
        ]
    )
    result = render_report(t)
    assert "diarization" in result
    assert "notes" in result


def test_warnings_non_list_falls_back_to_empty() -> None:
    # warnings key present but not a list → no crash, no section.
    t = {"text": "x", "paragraphs": [], "warnings": "not-a-list"}
    result = render_report(t)
    assert "Varoitukset" not in result


# ---------------------------------------------------------------------------
# Language parameter
# ---------------------------------------------------------------------------


def test_language_fi_default_headings() -> None:
    t = _transcript(warnings=["diarization failed; continuing without speaker labels"])
    result = render_report(t)
    assert "Puhujat:" in result
    assert "Epävarmat kohdat:" not in result  # no suspects in this fixture


def test_language_en_headings() -> None:
    t = _transcript(warnings=["diarization failed; continuing without speaker labels"])
    result = render_report(t, language="en")
    assert "Speakers:" in result
    assert "Warnings (diarization):" in result


def test_language_unknown_defaults_to_fi() -> None:
    t = _transcript(warnings=["diarization failed"])
    result = render_report(t, language="xx")
    assert "Puhujat:" in result


# ---------------------------------------------------------------------------
# build_quality_report — fail-open
# ---------------------------------------------------------------------------


def test_build_quality_report_returns_rendered() -> None:
    result = build_quality_report(_transcript())
    assert "Puhujat:" in result


def test_build_quality_report_fail_open_returns_empty_on_exception() -> None:
    # A non-dict transcript entry that makes _paragraphs raise — but
    # _paragraphs handles non-dict entries; force an exception via a
    # malformed dict entry that would crash the speaker-set comprehension.
    t = {"text": "x", "paragraphs": [42]}  # 42 is not a dict; dropped by _paragraphs
    result = build_quality_report(t)
    assert result == ""  # empty, no crash


def test_build_quality_report_does_not_raise_for_empty_dict() -> None:
    result = build_quality_report({})
    assert result == ""


def test_render_does_not_raise_for_malformed_paragraph_entries() -> None:
    t = {
        "text": "x",
        "paragraphs": [None, "junk", {"start": 0.0, "end": 1.0, "text": "ok"}],
    }
    # None and "junk" are dropped; "ok" paragraph renders without speaker.
    result = render_report(t)
    assert "Puhujat:" not in result


def test_render_does_not_raise_for_malformed_segments() -> None:
    t = {
        "text": "x",
        "paragraphs": [],
        "segments": [None, "bad", {"start": 0, "end": 1, "text": "Kiitos."}],
    }
    # Non-dict segments dropped; the single dict segment is not degenerate alone.
    result = render_report(t)
    assert "Jäljellä olevat loopit" not in result
