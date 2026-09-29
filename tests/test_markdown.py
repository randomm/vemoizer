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
    assert "⚠ epävarma kohta" in md
    assert "⚠ selvä kohta" not in md


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
