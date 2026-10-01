"""Markdown notes output (issue #56).

Golden-style assertions over ``format_md`` and its registration in the
formatter registry. Pure string building — no models, no LLM.
"""

from __future__ import annotations

from vemoizer.output.formatters import (
    FORMAT_EXTENSIONS,
    OUTPUT_FORMATS,
    format_transcript,
)
from vemoizer.output.markdown import format_md


def _transcript(**extra) -> dict:
    return {
        "text": "Puhuttiin alustasta. Sitten deploymentista.",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "Puhuttiin alustasta."},
            {
                "start": 8.0,
                "end": 12.0,
                "text": "Sitten deploymentista.",
                "speaker": "S1",
            },
        ],
        **extra,
    }


def test_full_notes_render_all_sections() -> None:
    notes = {
        "title": "Viikkopalaveri",
        "summary": "Keskusteltiin alustan suunnasta.",
        "key_points": ["Alusta etenee"],
        "action_items": ["Kirjaa backlogiin", "Sovi demo"],
    }
    md = format_md(_transcript(notes=notes))
    # Header: speaker legend appears because paragraph 2 has a speaker.
    # Total speech: 5 + 4 = 9s; S1 speaks 4s → 44%.
    assert "Keskustelijat: S1 (44%)" in md
    # Title line is no longer the first line when a header is present.
    assert "# Viikkopalaveri" in md
    assert "## Tiivistelmä" in md
    assert "Keskusteltiin alustan suunnasta." in md
    assert "## Keskeisiä asioita" in md
    assert "- Alusta etenee" in md
    assert "## Toimet" in md
    assert "- [ ] Kirjaa backlogiin" in md
    assert "- [ ] Sovi demo" in md
    assert "## Ääniseloste" in md
    # Transcript renders as paragraph blocks with timestamps and speaker prefixes.
    assert "[00:00:00] Puhuttiin alustasta." in md
    assert "[00:00:08] [S1] Sitten deploymentista." in md


def test_without_notes_renders_a_clean_transcript_document() -> None:
    md = format_md(_transcript())
    # Speaker legend present (paragraph 2 has speaker S1).
    assert "Keskustelijat: S1 (44%)" in md
    assert "# Transcript" in md
    assert "## Tiivistelmä" not in md
    assert "## Toimet" not in md
    assert "[00:00:00] Puhuttiin alustasta." in md


def test_empty_sections_are_omitted() -> None:
    notes = {
        "title": "Otsikko",
        "summary": "Tiivistelmä.",
        "key_points": [],
        "action_items": [],
    }
    md = format_md(_transcript(notes=notes))
    assert "## Keskeisiä asioita" not in md
    assert "## Toimet" not in md
    assert "Tiivistelmä." in md


def test_without_paragraphs_falls_back_to_text() -> None:
    md = format_md({"text": "vain teksti tässä"})
    assert "vain teksti tässä" in md


def test_md_is_registered_as_an_output_format() -> None:
    assert "md" in OUTPUT_FORMATS
    assert FORMAT_EXTENSIONS["md"] == ".md"
    rendered = format_transcript(_transcript(), "md")
    # Header with speaker legend appears, so the title is not the first line.
    assert "Keskustelijat: S1 (44%)" in rendered
    assert "# Transcript" in rendered


def test_suspect_paragraphs_render_a_warning() -> None:
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "epävarma kohta",
                    "suspect": "garble",
                },
                {"start": 2.0, "end": 3.0, "text": "selvä kohta"},
            ],
        }
    )
    # garble → epäselvä
    assert "⚠ epäselvä" in md
    assert "⚠ epävarma kohta" not in md
    # The clear paragraph has no warning.
    assert "⚠ selvä kohta" not in md
    # Both paragraphs have timestamps.
    assert "[00:00:00] ⚠ epäselvä epävarma kohta" in md
    assert "[00:00:02] selvä kohta" in md


def test_suspect_number_renders_luku() -> None:
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "numero kohta",
                    "suspect": "number",
                },
            ],
        }
    )
    assert "⚠ luku" in md
    assert "[00:00:00] ⚠ luku numero kohta" in md


def test_suspect_garble_outranks_number() -> None:
    """When a paragraph has suspect='garble', it renders epäselvä, not luku."""
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "sekava kohta",
                    "suspect": "garble",
                },
            ],
        }
    )
    assert "⚠ epäselvä" in md
    assert "⚠ luku" not in md


def test_suspect_non_string_value_renders_raw() -> None:
    """A non-string (unhashable) suspect value renders raw, never crashes."""
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "outo kohta",
                    "suspect": ["weird", "value"],
                },
            ],
        }
    )
    assert "⚠ [" in md  # the list's repr, via the non-string fallback
    assert "outo kohta" in md


def test_paragraph_negative_start_omits_timestamp() -> None:
    """A negative ``start`` is omitted, not rendered as [00:00:00]."""
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {"start": -5.0, "end": 1.0, "text": "negatiivinen"},
            ],
        }
    )
    assert "[00:00:00]" not in md
    assert "negatiivinen" in md


def test_no_suspect_no_warning() -> None:
    md = format_md(
        {
            "text": "x",
            "paragraphs": [
                {"start": 0.0, "end": 1.0, "text": "puhdas kohta"},
            ],
        }
    )
    assert "⚠" not in md
    assert "[00:00:00] puhdas kohta" in md


def test_part_markers_interleaved_at_offsets() -> None:
    """Multi-part groups: a `— osa N (äänto X) —` marker renders as a
    standalone line, interleaved with the paragraph blocks at its offset
    (the issue #77 sidecar/MD contract)."""
    t = _transcript(
        part_markers=[
            {"offset": 0.0, "label": "— osa 1 (äänto Uusi äänto 425.m4a) —"},
            {"offset": 8.0, "label": "— osa 2 (äänto Uusi äänto 426.m4a) —"},
        ]
    )
    md = format_md(t)
    assert "— osa 1 (äänto Uusi äänto 425.m4a) —" in md
    assert "— osa 2 (äänto Uusi äänto 426.m4a) —" in md
    # Each marker is a standalone line, and the order is:
    # marker 1 (offset 0) < paragraph 1 < marker 2 (offset 8) < paragraph 2.
    i1 = md.index("— osa 1")
    i_p1 = md.index("Puhuttiin alustasta.")
    i2 = md.index("— osa 2")
    i_p2 = md.index("Sitten deploymentista.")
    assert i1 < i_p1 < i2 < i_p2


def test_part_markers_without_paragraphs_render_before_bare_text() -> None:
    """No timestamped structure: markers have no offset to anchor to, so
    they render before the bare text — no crash, no lost marker."""
    t = {
        "text": "vain teksti tässä",
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänto pair_a.m4a) —"},
            {"offset": 2.5, "label": "— osa 2 (äänto pair_b.m4a) —"},
        ],
    }
    md = format_md(t)
    i1 = md.index("— osa 1")
    i2 = md.index("— osa 2")
    i_t = md.index("vain teksti tässä")
    assert i1 < i2 < i_t


def test_non_dict_part_markers_are_dropped_not_crash() -> None:
    """A stray non-dict entry in `part_markers` cannot crash the render
    path with an `AttributeError` — it is dropped, the valid ones render."""
    t = _transcript(part_markers=[None, "junk", {"offset": 0.0, "label": "— osa 1 —"}])
    md = format_md(t)
    assert "— osa 1 —" in md


def test_transcript_without_part_markers_renders_unchanged() -> None:
    """The common single-file case: no `part_markers` key, no behaviour
    change — the marker path must be a no-op."""
    md = format_md(_transcript())
    assert "— osa" not in md
    assert "Puhuttiin alustasta." in md


def _no_para_markers() -> list[dict]:
    return [
        {"offset": 0.0, "label": "— osa 1 (äänto pair_a.m4a) —"},
        {"offset": 2.5, "label": "— osa 2 (äänto pair_b.m4a) —"},
    ]


def test_md_no_paragraphs_markers_only() -> None:
    """No timestamped structure and an empty body: the markers render
    (joined) and there is no trailing bare text to append."""
    md = format_md({"text": "", "part_markers": _no_para_markers()})
    # Both markers render; markers-only means no body appended.
    assert "— osa 1 (äänto pair_a.m4a) —" in md
    assert "— osa 2 (äänto pair_b.m4a) —" in md
    # Markers are the only content in the transcript section (no empty body).
    t_section = md.split("## Ääniseloste")[1]
    assert "\n\n" in t_section
    i1 = md.index("— osa 1")
    i2 = md.index("— osa 2")
    assert i1 < i2


def test_md_no_paragraphs_body_only() -> None:
    """No timestamped structure and no markers: the bare text renders
    unchanged (the common single-file case)."""
    md = format_md({"text": "vain teksti tässä"})
    assert "vain teksti tässä" in md
    assert "— osa" not in md


def test_md_no_paragraphs_markers_and_body() -> None:
    """No timestamped structure with markers AND a body: markers render
    first, then the body, joined by blank lines."""
    md = format_md({"text": "vain teksti tässä", "part_markers": _no_para_markers()})
    i1 = md.index("— osa 1")
    i2 = md.index("— osa 2")
    i_t = md.index("vain teksti tässä")
    assert i1 < i2 < i_t


def test_md_no_paragraphs_neither_markers_nor_body() -> None:
    """No timestamped structure, no markers, empty body: the transcript
    section renders as a clean empty line — no crash, no stray markers."""
    md = format_md({"text": ""})
    assert "## Ääniseloste" in md
    assert "— osa" not in md
    # The transcript section body is empty (no markers, no body text).
    t_section = md.split("## Ääniseloste")[1].strip()
    assert t_section == ""


# ---------------------------------------------------------------------------
# M6 header tests (issue #75)
# ---------------------------------------------------------------------------


def test_header_date_line() -> None:
    md = format_md({"text": "x", "date": "2025-01-15"})
    assert "_2025-01-15_" in md


def test_header_no_date_no_line() -> None:
    md = format_md({"text": "x"})
    assert "_2025" not in md


def test_header_duration_line() -> None:
    md = format_md({"text": "x", "duration_s": 3723.7})
    # 3723s = 1h 02m 03s
    assert "Kesto: [01:02:03]" in md


def test_header_no_duration_no_line() -> None:
    md = format_md({"text": "x"})
    assert "Kesto:" not in md


def test_header_parts_line_multifile() -> None:
    t = _transcript(
        part_markers=[
            {"offset": 0.0, "label": "— osa 1 —"},
            {"offset": 30.0, "label": "— osa 2 —"},
        ]
    )
    md = format_md(t)
    assert "Osia: 2" in md


def test_header_no_parts_line_single_file() -> None:
    md = format_md(_transcript())
    assert "Osia:" not in md


def test_header_speaker_legend_with_talk_share() -> None:
    # Two speakers: S1 speaks 0-5 (5s), S2 speaks 8-12 (4s). Total 9s.
    # S1: 56%, S2: 44%.
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "S1 puhe", "speaker": "S1"},
            {"start": 8.0, "end": 12.0, "text": "S2 puhe", "speaker": "S2"},
        ],
    }
    md = format_md(t)
    assert "Keskustelijat:" in md
    # S1 has the most talk, so it appears first.
    assert "S1 (56%)" in md
    assert "S2 (44%)" in md


def test_header_no_speaker_legend_when_no_speakers() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 0.0, "end": 5.0, "text": "puhe ilman puhujaa"},
        ],
    }
    md = format_md(t)
    assert "Keskustelijat:" not in md


def test_header_glossary_source_line() -> None:
    md = format_md({"text": "x", "glossary_source": "glossary.txt (12 terms)"})
    assert "Sanasto: glossary.txt (12 terms)" in md


def test_header_no_glossary_no_line() -> None:
    md = format_md({"text": "x"})
    assert "Sanasto:" not in md


def test_header_all_fields_present() -> None:
    t = {
        "text": "x",
        "date": "2025-03-10",
        "duration_s": 61,
        "glossary_source": "my_glossary.txt (5 terms)",
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 —"},
            {"offset": 30.0, "label": "— osa 2 —"},
        ],
        "paragraphs": [
            {"start": 0.0, "end": 30.0, "text": "S1 puhe", "speaker": "S1"},
            {"start": 30.0, "end": 61.0, "text": "S2 puhe", "speaker": "S2"},
        ],
    }
    md = format_md(t)
    lines = md.split("\n")
    # All header fields present, in order: date, duration, parts, legend, glossary.
    assert lines[0] == "_2025-03-10_"
    assert lines[1] == "Kesto: [00:01:01]"
    assert lines[2] == "Osia: 2"
    assert "Keskustelijat: S2 (51%), S1 (49%)" in lines[3]
    assert lines[4] == "Sanasto: my_glossary.txt (5 terms)"
    # Blank line separates header from title.
    assert lines[5] == ""
    assert lines[6] == "# Transcript"


def test_header_empty_for_bare_text() -> None:
    """No date, duration, parts, speakers, or glossary → no header,
    document starts at the title."""
    md = format_md({"text": "vain teksti"})
    assert md.startswith("# Transcript\n")


# ---------------------------------------------------------------------------
# [hh:mm:ss] timestamp tests (issue #75)
# ---------------------------------------------------------------------------


def test_paragraph_timestamp_prefix() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 192.0, "end": 195.0, "text": "a"},
            {"start": 3723.5, "end": 3725.0, "text": "b"},
        ],
    }
    md = format_md(t)
    # 192s = 0h 3m 12s
    assert "[00:03:12] a" in md
    # 3723s = 1h 2m 3s (floor)
    assert "[01:02:03] b" in md


def test_paragraph_without_start_omits_timestamp() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"text": "no start"},
            {"start": 5.0, "end": 10.0, "text": "with start"},
        ],
    }
    md = format_md(t)
    assert "no start" in md
    assert "[00:00:00] no start" not in md
    assert "[00:00:05] with start" in md


def test_timestamp_handles_hours() -> None:
    t = {
        "text": "x",
        "paragraphs": [
            {"start": 3661.0, "end": 3662.0, "text": "over an hour"},
        ],
    }
    md = format_md(t)
    # 3661s = 1h 1m 1s
    assert "[01:01:01] over an hour" in md


# ---------------------------------------------------------------------------
# Language parameter tests (issue #75)
# ---------------------------------------------------------------------------


def test_language_fi_default_headings() -> None:
    notes = {
        "title": "Palaveri",
        "summary": "Yhteenveto.",
        "key_points": ["A"],
        "action_items": ["B"],
    }
    md = format_md(_transcript(notes=notes))
    assert "## Tiivistelmä" in md
    assert "## Keskeisiä asioita" in md
    assert "## Toimet" in md
    assert "## Ääniseloste" in md


def test_language_en_headings() -> None:
    notes = {
        "title": "Meeting",
        "summary": "Summary.",
        "key_points": ["A"],
        "action_items": ["B"],
    }
    md = format_md(_transcript(notes=notes), language="en")
    assert "## Summary" in md
    assert "## Key points" in md
    assert "## Action items" in md
    assert "## Transcript" in md


def test_language_en_header_labels() -> None:
    t = {
        "text": "x",
        "date": "2025-01-01",
        "duration_s": 120,
        "glossary_source": "g.txt (3 terms)",
        "part_markers": [
            {"offset": 0.0, "label": "— part 1 —"},
            {"offset": 60.0, "label": "— part 2 —"},
        ],
        "paragraphs": [
            {"start": 0.0, "end": 60.0, "text": "a", "speaker": "S1"},
            {"start": 60.0, "end": 120.0, "text": "b", "speaker": "S2"},
        ],
    }
    md = format_md(t, language="en")
    assert "Duration: [00:02:00]" in md
    assert "Parts: 2" in md
    assert "Speakers: S1 (50%), S2 (50%)" in md
    assert "Glossary: g.txt (3 terms)" in md


def test_language_unknown_defaults_to_fi() -> None:
    md = format_md({"text": "x", "duration_s": 60}, language="xx")
    assert "Kesto: [00:01:00]" in md
