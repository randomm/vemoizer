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
    assert md.startswith("# Viikkopalaveri\n")
    assert "Keskusteltiin alustan suunnasta." in md
    assert "- Alusta etenee" in md
    assert "- [ ] Kirjaa backlogiin" in md
    assert "- [ ] Sovi demo" in md
    # transcript renders as paragraph blocks with speaker prefixes
    assert "Puhuttiin alustasta." in md
    assert "[S1] Sitten deploymentista." in md


def test_without_notes_renders_a_clean_transcript_document() -> None:
    md = format_md(_transcript())
    assert md.startswith("# Transcript\n")
    assert "## Summary" not in md
    assert "## Action items" not in md
    assert "Puhuttiin alustasta." in md


def test_empty_sections_are_omitted() -> None:
    notes = {
        "title": "Otsikko",
        "summary": "Tiivistelmä.",
        "key_points": [],
        "action_items": [],
    }
    md = format_md(_transcript(notes=notes))
    assert "## Key points" not in md
    assert "## Action items" not in md
    assert "Tiivistelmä." in md


def test_without_paragraphs_falls_back_to_text() -> None:
    md = format_md({"text": "vain teksti tässä"})
    assert "vain teksti tässä" in md


def test_md_is_registered_as_an_output_format() -> None:
    assert "md" in OUTPUT_FORMATS
    assert FORMAT_EXTENSIONS["md"] == ".md"
    rendered = format_transcript(_transcript(), "md")
    assert rendered.startswith("# Transcript\n")
