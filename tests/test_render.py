"""Pure render core (issue #89, M5a workstream ``render``).

``render_markdown`` re-applies correction pairs and speaker names to a
stored sidecar and re-renders Markdown via ``format_md`` — no model, no
LLM. Round-trip byte-identity against a sidecar assembled with
``vemoizer.sidecar.build_sidecar`` (the default workstream's writer)
holds iff the sidecar's paragraphs/notes/part_markers are faithful.
"""

from __future__ import annotations

import sys

from vemoizer import render
from vemoizer.output.markdown import format_md
from vemoizer.render import render_markdown
from vemoizer.sidecar import build_sidecar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _corrections() -> dict[str, str]:
    return {"Blacksit": "Flagship", "epit*": "EBITDA"}


def _result(**extra) -> dict:
    base = {
        "text": "Puhuttiin Blacksit-hankkeesta ja epittä.",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Puhuttiin Blacksit-hankkeesta.",
                "speaker": "SPEAKER_1",
            },
            {
                "start": 8.0,
                "end": 12.0,
                "text": "epittä selvitetään myöhemmin.",
                "speaker": "SPEAKER_2",
            },
        ],
        "notes": {
            "title": "Alustus",
            "summary": "Puhuttiin Blacksit-hankkeesta.",
            "key_points": ["Blacksit käynnistyy ensi kuussa"],
            "action_items": ["SPEAKER_1: Kirjaa epittä"],
        },
        **extra,
    }
    return base


def _sidecar(command: str = "meeting", **result_extra) -> dict:
    """A sidecar exactly as the default workstream's writer emits one."""
    result = _result(**result_extra)
    return build_sidecar(result, command=command, glossary_files=None)


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_render_imports_no_model_dependencies() -> None:
    """render.py must be model-free (issue #89 acceptance criterion).

    ``render_markdown`` is called through the module (already imported at
    the top of this file) and must not pull a model dependency into
    ``sys.modules``.
    """
    for name in ("mlx", "pyannote", "torch"):
        sys.modules.pop(name, None)

    render.render_markdown(
        {"text": "x", "paragraphs": []}, corrections={}, speaker_names={}
    )
    for name in ("mlx", "pyannote", "torch"):
        assert name not in sys.modules, f"{name} imported by the render path"


# ---------------------------------------------------------------------------
# Correction re-application
# ---------------------------------------------------------------------------


def test_render_applies_corrections_to_paragraphs_and_notes() -> None:
    md = render_markdown(_sidecar(), corrections=_corrections(), speaker_names={})

    assert "Flagship-hankkeesta" in md
    assert "Blacksit" not in md
    assert "EBITDA selvitetään myöhemmin" in md
    # Notes strings carry the corrections too.
    assert "# Alustus" in md
    assert "Puhuttiin Flagship-hankkeesta." in md
    assert "- Flagship käynnistyy ensi kuussa" in md


def test_render_without_corrections_is_verbatim() -> None:
    md = render_markdown(_sidecar(), corrections={}, speaker_names={})
    assert "Blacksit-hankkeesta" in md
    assert "epittä selvitetään myöhemmin" in md


# ---------------------------------------------------------------------------
# Speaker names: whole-word discipline
# ---------------------------------------------------------------------------


def test_render_names_are_whole_word_in_text_and_labels() -> None:
    sidecar = _sidecar()
    # A paragraph text containing 'Matskut' must survive naming 'Mats'.
    sidecar["paragraphs"].append(
        {
            "start": 20.0,
            "end": 22.0,
            "text": "Matskut lähetetty.",
            "speaker": "SPEAKER_1",
        }
    )

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Sanna"},
    )

    assert "[Mats] Puhuttiin" in md
    assert "[Sanna] epittä" in md
    assert "[Mats] Matskut lähetetty." in md  # text untouched; prefix renamed
    assert "[Matskut]" not in md


def test_render_names_do_not_rewrite_longer_labels() -> None:
    sidecar = _sidecar()
    sidecar["paragraphs"].append(
        {"start": 30.0, "end": 31.0, "text": "Kolmas ääni.", "speaker": "SPEAKER_12"}
    )

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_12": "Sanna"},
    )

    assert "[Mats] Puhuttiin" in md
    assert "[Sanna] Kolmas ääni." in md
    assert "[SPEAKER_12]" not in md
    assert "[Mats12]" not in md


def test_render_renames_action_item_owner_prefix() -> None:
    md = render_markdown(
        _sidecar(),
        corrections={},
        speaker_names={"SPEAKER_1": "Mats"},
    )

    assert "- [ ] Mats: Kirjaa epittä" in md


# ---------------------------------------------------------------------------
# Speaker-name merge rule
# ---------------------------------------------------------------------------


def test_render_same_name_labels_merge_to_earliest_label() -> None:
    """Two labels named the same person collapse to the earliest label."""
    sidecar = _sidecar()
    # SPEAKER_1 (start 0.0) is earlier than SPEAKER_2 (start 8.0), so
    # SPEAKER_1 is canonical: every other label in the group is rewritten
    # to it, and the canonical label itself carries the given name.
    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Mats"},
    )

    assert "[Mats] Puhuttiin" in md
    assert "[SPEAKER_1] epittä" in md
    assert "[Mats]" not in md.replace("[Mats] Puhuttiin", "", 1)
    assert "[SPEAKER_2]" not in md


def test_render_merge_picks_earliest_label_by_paragraph_start() -> None:
    """The label with the earliest first-seen start becomes canonical."""
    sidecar = _sidecar()
    # Make SPEAKER_2 the earliest: reorder starts so its first paragraph
    # comes before SPEAKER_1's.
    sidecar["paragraphs"] = [
        {"start": 0.0, "text": "Aloitus.", "speaker": "SPEAKER_2"},
        {"start": 5.0, "text": "Jatko.", "speaker": "SPEAKER_1"},
        {"start": 9.0, "text": "Lopetus.", "speaker": "SPEAKER_2"},
    ]

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Mats"},
    )

    # SPEAKER_2 starts at 0.0 → canonical; SPEAKER_1 is rewritten to the
    # canonical label, and SPEAKER_2 carries the given name.
    assert "[Mats] Aloitus." in md
    assert "[SPEAKER_2] Jatko." in md
    assert "[Mats] Lopetus." in md
    assert "[SPEAKER_1]" not in md


def test_render_merge_rewrites_notes_owner_prefixes() -> None:
    sidecar = _sidecar()
    sidecar["notes"]["action_items"] = [
        "SPEAKER_1: Kirjaa epittä",
        "SPEAKER_2: Lähetä päiväraja",
    ]
    sidecar["notes"]["key_points"] = ["SPEAKER_1 esitteli alustan"]

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Mats"},
    )

    # SPEAKER_1 (start 0.0) is canonical → the merged label (SPEAKER_2)
    # is rewritten to it, and the canonical label itself becomes "Mats".
    assert "- [ ] Mats: Kirjaa epittä" in md
    assert "- [ ] SPEAKER_1: Lähetä päiväraja" in md
    assert "SPEAKER_1 esitteli alustan" in md
    assert "Mats esitteli" not in md  # key_points carry no owner prefix rewrite


# ---------------------------------------------------------------------------
# Stored sidecar speaker_names (idempotent re-render)
# ---------------------------------------------------------------------------


def test_render_applies_stored_speaker_names_from_sidecar() -> None:
    sidecar = _sidecar()
    sidecar["speaker_names"] = {"SPEAKER_1": "Mats"}

    md = render_markdown(sidecar, corrections={}, speaker_names={})

    assert "[Mats] Puhuttiin" in md
    assert "[SPEAKER_2] epittä" in md


def test_render_cli_names_compose_with_stored_names() -> None:
    sidecar = _sidecar()
    sidecar["speaker_names"] = {"SPEAKER_1": "Mats"}

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_2": "Sanna"},
    )

    assert "[Mats] Puhuttiin" in md
    assert "[Sanna] epittä" in md


# ---------------------------------------------------------------------------
# Old sidecars / degraded shapes
# ---------------------------------------------------------------------------


def test_render_old_sidecar_without_m5a_keys() -> None:
    """A pre-M5 .json (no source/options/speaker_names) still renders."""
    old = {
        "text": "Vanha äänitys.",
        "paragraphs": [{"start": 0.0, "text": "Vanha äänitys."}],
        "notes": {"title": "Vanha", "action_items": []},
    }

    md = render_markdown(old, corrections={}, speaker_names={})

    assert "# Vanha" in md
    assert "Vanha äänitys." in md


def test_render_sidecar_without_notes() -> None:
    sidecar = _sidecar()
    sidecar.pop("notes")

    md = render_markdown(sidecar, corrections={}, speaker_names={})

    assert "# Transcript" in md
    assert "Puhuttiin" in md


def test_render_sidecar_without_paragraphs_falls_back_to_text() -> None:
    old = {"text": "Aika tosi vanha teksti.", "notes": {}}
    md = render_markdown(old, corrections={}, speaker_names={})
    assert "Aika tosi vanha teksti." in md


# ---------------------------------------------------------------------------
# Round trip: sidecar + same corrections → identical Markdown
# ---------------------------------------------------------------------------


def test_round_trip_meets_direct_format_md() -> None:
    """render(sidecar, same corrections) == format_md(result after run)."""
    sidecar = _sidecar()
    # What the run wrote: corrections already applied to the in-memory
    # result, then format_md.
    from vemoizer.glossary import apply_corrections, apply_corrections_to_notes

    run_result = _result()
    run_result["notes"] = apply_corrections_to_notes(
        run_result["notes"], _corrections()
    )
    run_result["paragraphs"] = apply_corrections(
        run_result["paragraphs"], _corrections()
    )
    run_md = format_md(run_result)

    render_md = render_markdown(sidecar, corrections=_corrections(), speaker_names={})

    assert render_md == run_md


def test_round_trip_without_notes_or_glossary() -> None:
    """Fail-open run (notes=None, no glossary): render matches the run md."""
    run_result = {
        "text": "Vanha teksti Blacksit.",
        "paragraphs": [{"start": 0.0, "text": "Vanha teksti Blacksit."}],
        "notes": None,
    }
    sidecar = build_sidecar(dict(run_result), command="meeting", glossary_files=None)
    run_md = format_md(run_result)

    assert render_markdown(sidecar, corrections={}, speaker_names={}) == run_md


def test_round_trip_holds_iff_glossary_hash_matches() -> None:
    """Different corrections (hash drift) → still renders, identity gone."""
    sidecar = _sidecar()
    other_corrections = {"Blacksit": "Erkä"}

    md = render_markdown(sidecar, corrections=other_corrections, speaker_names={})

    assert "Erkä-hankkeesta" in md
    assert "Flagship" not in md


# ---------------------------------------------------------------------------
# Corrupted sidecar: non-dict paragraphs
# ---------------------------------------------------------------------------


def test_render_sidecar_with_string_entry_in_paragraphs_does_not_crash() -> None:
    """A sidecar with a non-dict entry in paragraphs renders without crash.

    ``apply_corrections`` maps over dict entries (``para.get``); a
    corrupted sidecar can carry a string entry. ``_render_dict`` filters
    to dict entries first (consistent with ``_blocks``) and DROPS the
    non-dict entries.
    """
    sidecar = _sidecar()
    sidecar["paragraphs"].append("corrupted-string-entry")  # type: ignore[valid-type]
    sidecar["paragraphs"].append(None)  # type: ignore[valid-type]

    md = render_markdown(sidecar, corrections=_corrections(), speaker_names={})
    # The valid dict entries still render with corrections applied.
    assert "Flagship-hankkeesta" in md
    assert "EBITDA selvitetään" in md
