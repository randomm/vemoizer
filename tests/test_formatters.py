"""Tests for the output formatters (issue #10, task A).

Pure ``TranscriptionResult -> str`` functions for the four transcript
formats: ``txt``, ``json``, ``srt``, ``vtt``. The golden fixtures in
``tests/fixtures/output/`` are the single fixed mock transcript (Finnish +
English code-switching, segments with word timestamps) rendered once — the
formatter tests diff actual output against them, which locks the two
non-negotiable format differences:

- SRT timestamps: ``HH:MM:SS,mmm`` (comma), blocks separated by blank
  lines, 1-based index.
- VTT: ``WEBVTT`` header, ``HH:MM:SS.mmm`` (dot), 0-based cue index,
  no ``NOTE`` or cue-identifier lines.

All tests are pure-stdlib + stdlib json: no model imports, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vemoizer.output.formatters import (
    FORMAT_EXTENSIONS,
    OUTPUT_FORMATS,
    format_json,
    format_srt,
    format_txt,
    format_vtt,
    srt_timestamp,
    vtt_timestamp,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "output"

#: The one fixed mock transcript (Finnish prose with English terms,
#: segments + word timestamps). The golden files under
#: ``tests/fixtures/output/`` are this dict rendered through each
#: formatter.
DIARIZED_TRANSCRIPT: dict = {
    "text": (
        "Käytimme aamupäivän debuggaamassa deploy pipeline issuea. "
        "Ei mitään outoa, mutta CI oli aivan flaky. "
        "Testataan uudestaan illalla, ehkä kun infra on ehtinyt "
        "vääntyä takaisin."
    ),
    "language": "fi",
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Käytimme aamupäivän debuggaamassa deploy pipeline issuea.",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 2.0,
            "end": 4.5,
            "text": "Ei mitään outoa, mutta CI oli aivan flaky.",
            "speaker": "SPEAKER_01",
        },
        {
            "start": 4.5,
            "end": 6.0,
            "text": (
                "Testataan uudestaan illalla, ehkä kun infra on ehtinyt "
                "vääntyä takaisin."
            ),
            "speaker": "SPEAKER_00",
        },
    ],
    "words": [
        {"word": "Käytimme", "start": 0.0, "end": 0.2},
        {"word": "Ei", "start": 2.0, "end": 2.1},
        {"word": "Testataan", "start": 4.5, "end": 4.7},
    ],
}

TRANSCRIPT: dict = {
    "text": (
        "Käytimme aamupäivän debuggaamassa deploy pipeline issuea. "
        "Ei mitään outoa, mutta CI oli aivan flaky. "
        "Testataan uudestaan illalla, ehkä kun infra on ehtinyt "
        "vääntyä takaisin."
    ),
    "language": "fi",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Käytimme aamupäivän debuggaamassa deploy pipeline issuea.",
        },
        {
            "start": 2.0,
            "end": 4.5,
            "text": "Ei mitään outoa, mutta CI oli aivan flaky.",
        },
        {
            "start": 4.5,
            "end": 6.0,
            "text": (
                "Testataan uudestaan illalla, ehkä kun infra on ehtinyt "
                "vääntyä takaisin."
            ),
        },
    ],
    "words": [
        {"word": "Käytimme", "start": 0.0, "end": 0.2},
        {"word": "aamupäivän", "start": 0.2, "end": 0.5},
        {"word": "debuggaamassa", "start": 0.5, "end": 0.8},
        {"word": "deploy", "start": 0.8, "end": 1.0},
        {"word": "pipeline", "start": 1.0, "end": 1.3},
        {"word": "issuea.", "start": 1.3, "end": 1.6},
        {"word": "Ei", "start": 2.0, "end": 2.1},
        {"word": "mitään", "start": 2.1, "end": 2.3},
        {"word": "outoa,", "start": 2.3, "end": 2.5},
        {"word": "mutta", "start": 2.5, "end": 2.7},
        {"word": "CI", "start": 2.7, "end": 2.9},
        {"word": "oli", "start": 2.9, "end": 3.0},
        {"word": "aivan", "start": 3.0, "end": 3.2},
        {"word": "flaky.", "start": 3.2, "end": 3.5},
        {"word": "Testataan", "start": 4.5, "end": 4.7},
        {"word": "uudestaan", "start": 4.7, "end": 5.0},
        {"word": "illalla,", "start": 5.0, "end": 5.2},
        {"word": "ehkä", "start": 5.2, "end": 5.3},
        {"word": "kun", "start": 5.3, "end": 5.4},
        {"word": "infra", "start": 5.4, "end": 5.6},
        {"word": "on", "start": 5.6, "end": 5.7},
        {"word": "ehtinyt", "start": 5.7, "end": 5.8},
        {"word": "vääntyä", "start": 5.8, "end": 5.9},
        {"word": "takaisin.", "start": 5.9, "end": 6.0},
    ],
}


def _expected(path: str) -> str:
    return (FIXTURE_DIR / path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# OUTPUT_FORMATS / FORMAT_EXTENSIONS surface
# ---------------------------------------------------------------------------


def test_output_formats_covers_all_five() -> None:
    assert sorted(OUTPUT_FORMATS) == ["json", "md", "srt", "txt", "vtt"]


def test_format_extensions_map_each_format() -> None:
    assert FORMAT_EXTENSIONS["txt"] == ".txt"
    assert FORMAT_EXTENSIONS["json"] == ".json"
    assert FORMAT_EXTENSIONS["srt"] == ".srt"
    assert FORMAT_EXTENSIONS["vtt"] == ".vtt"


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


def test_format_txt_matches_golden_fixture() -> None:
    assert format_txt(TRANSCRIPT) == _expected("expected.txt")


def test_format_txt_joins_segments_with_double_newline() -> None:
    result = format_txt(
        {
            "text": "a b c",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "a b"},
                {"start": 1.0, "end": 2.0, "text": "c"},
            ],
        }
    )
    assert result == "a b\nc\n"


def test_format_txt_uses_text_field_when_no_segments() -> None:
    assert format_txt({"text": "single line only"}) == "single line only\n"


def test_format_txt_no_segments_no_text_returns_empty() -> None:
    assert format_txt({}) == ""


def test_format_txt_segments_with_blank_text_yields_empty() -> None:
    # Segments present but all whitespace text: nothing to join.
    transcript = {"text": "x", "segments": [{"start": 0, "end": 1, "text": "   "}]}
    assert format_txt(transcript) == ""


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_format_json_matches_golden_fixture() -> None:
    assert format_json(TRANSCRIPT) == _expected("expected.json")


def test_format_json_roundtrips_through_json_loads() -> None:
    rendered = format_json(TRANSCRIPT)
    parsed = json.loads(rendered)
    assert parsed["text"] == TRANSCRIPT["text"]
    assert parsed["language"] == "fi"
    assert parsed["segments"] == TRANSCRIPT["segments"]
    assert parsed["words"] == TRANSCRIPT["words"]


def test_format_json_omits_language_when_absent() -> None:
    rendered = format_json({"text": "hei"})
    parsed = json.loads(rendered)
    assert "language" not in parsed
    assert parsed["text"] == "hei"


def test_format_json_omits_empty_segments_and_words() -> None:
    rendered = format_json({"text": "hei", "segments": [], "words": []})
    parsed = json.loads(rendered)
    # Empty optional collections are omitted, mirroring a backend that
    # reported nothing beyond the text.
    assert "segments" not in parsed
    assert "words" not in parsed


def test_format_json_ends_with_newline() -> None:
    assert format_json({"text": "x"}).endswith("\n")


# ---------------------------------------------------------------------------
# SRT timestamp formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (1.0, "00:00:01,000"),
        (59.0, "00:00:59,000"),
        (60.0, "00:01:00,000"),
        (60.5, "00:01:00,500"),
        (3661.25, "01:01:01,250"),
    ],
)
def test_srt_timestamp_format(seconds: float, expected: str) -> None:
    assert srt_timestamp(seconds) == expected


def test_srt_timestamp_uses_comma_separator() -> None:
    assert "," in srt_timestamp(1.5)
    assert "." not in srt_timestamp(1.5)


# ---------------------------------------------------------------------------
# VTT timestamp formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00.000"),
        (1.0, "00:00:01.000"),
        (59.0, "00:00:59.000"),
        (60.0, "00:01:00.000"),
        (60.5, "00:01:00.500"),
        (3661.25, "01:01:01.250"),
    ],
)
def test_vtt_timestamp_format(seconds: float, expected: str) -> None:
    assert vtt_timestamp(seconds) == expected


def test_vtt_timestamp_uses_dot_separator() -> None:
    assert "." in vtt_timestamp(1.5)
    assert "," not in vtt_timestamp(1.5)


# ---------------------------------------------------------------------------
# SRT / VTT full output
# ---------------------------------------------------------------------------


def test_format_srt_matches_golden_fixture() -> None:
    assert format_srt(TRANSCRIPT) == _expected("expected.srt")


def test_format_srt_index_is_1_based() -> None:
    first_line = format_srt(TRANSCRIPT).split("\n")[0]
    assert first_line == "1"


def test_format_srt_blocks_separated_by_blank_lines() -> None:
    # One blank line between each cue block, plus the terminating newline
    # after the last cue.
    blocks = format_srt(TRANSCRIPT).rstrip("\n").split("\n\n")
    assert len(blocks) == len(TRANSCRIPT["segments"])
    assert blocks[0].startswith("1\n00:00:00,000 --> 00:00:02,000\n")
    assert blocks[-1].startswith("3\n00:00:04,500 --> 00:00:06,000\n")
    assert blocks[-1].endswith(
        "Testataan uudestaan illalla, ehkä kun infra on ehtinyt vääntyä takaisin."
    )


def test_format_srt_no_segments_no_words_returns_empty() -> None:
    # No segments and no words: no cues to emit.
    assert format_srt({"text": "x"}) == ""


def test_format_srt_no_segments_falls_back_to_words() -> None:
    words = [
        {"word": "a", "start": 0.0, "end": 1.0},
        {"word": "b", "start": 1.0, "end": 2.0},
    ]
    out = format_srt({"text": "a b", "words": words})
    # 1-based index, comma-separated milliseconds
    assert "1\n00:00:00,000 --> 00:00:01,000\na\n" in out
    assert "2\n00:00:01,000 --> 00:00:02,000\nb\n" in out


def test_format_vtt_matches_golden_fixture() -> None:
    assert format_vtt(TRANSCRIPT) == _expected("expected.vtt")


def test_format_vtt_starts_with_webvtt_header() -> None:
    assert format_vtt(TRANSCRIPT).startswith("WEBVTT\n\n")


def test_format_vtt_index_is_0_based() -> None:
    lines = format_vtt(TRANSCRIPT).rstrip("\n").split("\n")
    assert lines[:4] == ["WEBVTT", "", "0", "00:00:00.000 --> 00:00:02.000"]
    # Second cue: index 1, dot-separated timestamps
    assert lines[6:8] == ["1", "00:00:02.000 --> 00:00:04.500"]
    # Third cue: index 2
    assert lines[10:12] == ["2", "00:00:04.500 --> 00:00:06.000"]


def test_format_vtt_no_segments_no_words_returns_header_only() -> None:
    # A VTT file with no cues still carries the WEBVTT header.
    assert format_vtt({"text": "x"}) == "WEBVTT\n"


def test_format_vtt_no_segments_falls_back_to_words() -> None:
    words = [
        {"word": "a", "start": 0.0, "end": 1.0},
        {"word": "b", "start": 1.0, "end": 2.0},
    ]
    out = format_vtt({"text": "a b", "words": words})
    assert out.startswith("WEBVTT\n\n0\n00:00:00.000 --> 00:00:01.000\na\n")
    assert "1\n00:00:01.000 --> 00:00:02.000\nb\n" in out


# ---------------------------------------------------------------------------
# Speaker labels (optional --diarize output)
# ---------------------------------------------------------------------------


def test_txt_with_speaker_labels_prefixes_segments() -> None:
    result = format_txt(DIARIZED_TRANSCRIPT)
    assert result == (
        "[SPEAKER_00] "
        "Käytimme aamupäivän debuggaamassa deploy pipeline issuea.\n"
        "[SPEAKER_01] "
        "Ei mitään outoa, mutta CI oli aivan flaky.\n"
        "[SPEAKER_00] "
        "Testataan uudestaan illalla, ehkä kun infra on ehtinyt vääntyä takaisin.\n"
    )


def test_txt_mixed_segments_only_labels_labelled_ones() -> None:
    transcript = {
        "text": "a b c",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "a b", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "c"},
        ],
    }
    assert format_txt(transcript) == "[SPEAKER_00] a b\nc\n"


def test_txt_without_speaker_key_unchanged() -> None:
    # No speaker anywhere: byte-identical to the pre-diarization output.
    assert format_txt(TRANSCRIPT) == _expected("expected.txt")


def test_txt_text_field_only_ignores_speaker_list() -> None:
    # No segments: the text fallback has no per-line speaker to attach.
    assert format_txt({"text": "solo", "speakers": ["SPEAKER_00"]}) == "solo\n"


def test_json_includes_speakers_when_present() -> None:
    parsed = json.loads(format_json(DIARIZED_TRANSCRIPT))
    assert parsed["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert parsed["segments"][0]["speaker"] == "SPEAKER_00"


def test_json_omits_speakers_when_absent_or_empty() -> None:
    parsed = json.loads(format_json({"text": "x", "speakers": []}))
    assert "speakers" not in parsed
    parsed = json.loads(format_json(TRANSCRIPT))
    assert "speakers" not in parsed


def test_srt_with_speaker_labels_prefixes_cues() -> None:
    out = format_srt(DIARIZED_TRANSCRIPT)
    blocks = out.rstrip("\n").split("\n\n")
    assert len(blocks) == 3
    assert blocks[0] == (
        "1\n00:00:00,000 --> 00:00:02,000\n"
        "[SPEAKER_00] Käytimme aamupäivän debuggaamassa deploy pipeline issuea."
    )
    assert blocks[1].endswith("[SPEAKER_01] Ei mitään outoa, mutta CI oli aivan flaky.")
    assert blocks[-1].endswith(
        "[SPEAKER_00] "
        "Testataan uudestaan illalla, ehkä kun infra on ehtinyt vääntyä takaisin."
    )


def test_srt_without_speaker_key_unchanged() -> None:
    assert format_srt(TRANSCRIPT) == _expected("expected.srt")


def test_vtt_with_speaker_labels_prefixes_cues() -> None:
    out = format_vtt(DIARIZED_TRANSCRIPT)
    assert out.startswith("WEBVTT\n\n0\n00:00:00.000 --> 00:00:02.000\n")
    assert (
        "[SPEAKER_00] Käytimme aamupäivän debuggaamassa deploy pipeline issuea.\n"
    ) in out
    assert "[SPEAKER_01] Ei mitään outoa, mutta CI oli aivan flaky.\n" in out
    assert "[SPEAKER_00] Testataan uudestaan illalla" in out


def test_vtt_without_speaker_key_unchanged() -> None:
    assert format_vtt(TRANSCRIPT) == _expected("expected.vtt")


def test_captions_skip_segments_with_blank_text() -> None:
    # Segments present but one carries whitespace-only text: no cue for it.
    segments = [
        {"start": 0.0, "end": 1.0, "text": "a b", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "   "},
    ]
    out = format_srt({"text": "a b", "segments": segments})
    blocks = out.rstrip("\n").split("\n\n")
    assert len(blocks) == 1
    assert blocks[0].startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    assert "[SPEAKER_00] a b" in blocks[0]
    assert "SPEAKER_01" not in out


def test_vtt_no_segments_key_falls_back_to_words() -> None:
    # No ``segments`` key at all: the caption builder uses the word-based
    # cues, which carry no speaker labels.
    words = [{"word": "a", "start": 0.0, "end": 1.0}]
    out = format_vtt({"text": "a", "words": words})
    assert out == "WEBVTT\n\n0\n00:00:00.000 --> 00:00:01.000\na\n"


def test_captions_from_words_without_speakers_have_no_labels() -> None:
    # The word-fallback path has no speaker data: labels never appear.
    words = [
        {"word": "a", "start": 0.0, "end": 1.0},
        {"word": "b", "start": 1.0, "end": 2.0},
    ]
    assert "[" not in format_srt({"text": "a b", "words": words})
    assert "[" not in format_vtt({"text": "a b", "words": words})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_format_transcript_dispatches_by_format() -> None:
    from vemoizer.output.formatters import format_transcript

    assert format_transcript(TRANSCRIPT, "txt") == format_txt(TRANSCRIPT)
    assert format_transcript(TRANSCRIPT, "json") == format_json(TRANSCRIPT)
    assert format_transcript(TRANSCRIPT, "srt") == format_srt(TRANSCRIPT)
    assert format_transcript(TRANSCRIPT, "vtt") == format_vtt(TRANSCRIPT)
    assert format_transcript(DIARIZED_TRANSCRIPT, "txt") == format_txt(
        DIARIZED_TRANSCRIPT
    )
    assert format_transcript(DIARIZED_TRANSCRIPT, "json") == format_json(
        DIARIZED_TRANSCRIPT
    )


def test_format_transcript_rejects_unknown_format() -> None:
    from vemoizer.output.formatters import format_transcript

    with pytest.raises(ValueError, match="Unknown output format"):
        format_transcript(TRANSCRIPT, "csv")


# -- paragraphs (issue #53) ----------------------------------------------


def test_txt_prefers_paragraphs_when_present() -> None:
    transcript = {
        "text": "eka lause toka lause",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "eka lause"},
            {"start": 4.0, "end": 5.0, "text": "toka lause"},
        ],
        "paragraphs": [
            {"start": 0.0, "end": 1.0, "text": "eka lause"},
            {"start": 4.0, "end": 5.0, "text": "toka lause", "speaker": "S1"},
        ],
    }
    rendered = format_txt(transcript)
    # Blank-line separated paragraph blocks, speaker prefix when known.
    assert rendered == "eka lause\n\n[S1] toka lause\n"


def test_txt_without_paragraphs_falls_back_to_segments() -> None:
    transcript = {
        "text": "x",
        "segments": [{"start": 0.0, "end": 1.0, "text": "vain segmentti"}],
    }
    assert format_txt(transcript) == "vain segmentti\n"


def test_json_carries_paragraphs_when_present() -> None:
    transcript = {
        "text": "x",
        "paragraphs": [{"start": 0.0, "end": 1.0, "text": "kappale"}],
    }
    data = json.loads(format_json(transcript))
    assert data["paragraphs"] == [{"start": 0.0, "end": 1.0, "text": "kappale"}]


def test_json_omits_empty_paragraphs() -> None:
    data = json.loads(format_json({"text": "x", "paragraphs": []}))
    assert "paragraphs" not in data
