"""Pure render core (issue #89, M5a workstream ``render``).

``render_markdown`` re-applies correction pairs and speaker names to a
stored sidecar and re-renders Markdown via ``format_md`` — no model, no
LLM. Round-trip byte-identity against a sidecar assembled with
``vemoizer.sidecar.build_sidecar`` (the default workstream's writer)
holds iff the sidecar's paragraphs/notes/part_markers are faithful.
"""

from __future__ import annotations

import sys
from typing import cast

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
# call_corrections (issue #270): fixes kept for THIS call only
# ---------------------------------------------------------------------------


def test_render_applies_call_corrections_from_sidecar() -> None:
    sidecar = _sidecar()
    sidecar["call_corrections"] = {"Blacksit": "Vantaa", "epit*": "KPI"}
    md = render_markdown(sidecar, corrections={}, speaker_names={})
    assert "Vantaa-hankkeesta" in md
    assert "Blacksit" not in md
    assert "KPI selvitetään myöhemmin" in md


def test_render_call_corrections_override_the_glossary_pair() -> None:
    sidecar = _sidecar()
    sidecar["call_corrections"] = {"Blacksit": "Vantaa"}
    md = render_markdown(sidecar, corrections=_corrections(), speaker_names={})
    assert "Vantaa" in md and "Flagship" not in md


def test_render_call_correction_wins_case_insensitively() -> None:
    """The term check keys the map by the stem as typed and a glossary pair is
    written by hand, so ``Blacksit => Flagship`` and ``{blacksit: Vantaa}`` are
    the SAME key: the call-only fix wins, it does not lose to the glossary."""
    sidecar = _sidecar()
    sidecar["call_corrections"] = {"blacksit": "Vantaa"}
    md = render_markdown(sidecar, corrections=_corrections(), speaker_names={})
    assert "Vantaa" in md and "Flagship" not in md and "Blacksit" not in md


def test_render_call_correction_beats_a_glossary_wildcard() -> None:
    """Correction pairs are applied in dict order, so the call-only fix must
    come FIRST: a glossary wildcard that also covers the word (``epit*``)
    would otherwise rewrite it before the exact call fix can match."""
    sidecar = _sidecar()
    sidecar["call_corrections"] = {"epittä": "EBITDA"}
    md = render_markdown(sidecar, corrections={"epit*": "KPI"}, speaker_names={})
    assert "EBITDA selvitetään myöhemmin" in md
    assert "KPI" not in md


def test_render_call_fix_right_side_is_never_re_touched() -> None:
    """The call fix's OUTPUT is final: it is held as a token while the
    glossary pairs run, so ``AVS => AWS`` cannot undo it."""
    sidecar = _sidecar()
    sidecar["paragraphs"][0]["text"] = "Puhuttiin APS-hankkeesta."
    sidecar["call_corrections"] = {"APS": "AVS"}
    md = render_markdown(sidecar, corrections={"AVS": "AWS"}, speaker_names={})
    assert "AVS-hankkeesta" in md and "AWS" not in md


def test_render_chained_call_fixes_are_order_independent() -> None:
    """A call fix's output is never an input to another call fix: ``APS``
    renders as ``AVS`` and the real ``AVS`` as ``AWS``, whichever way the
    two pairs are ordered in the map."""
    for call in ({"APS": "AVS", "AVS": "AWS"}, {"AVS": "AWS", "APS": "AVS"}):
        sidecar = _sidecar()
        sidecar["paragraphs"][0]["text"] = "Puhuttiin APS ja AVS."
        sidecar["call_corrections"] = call
        md = render_markdown(sidecar, corrections={}, speaker_names={})
        assert "Puhuttiin AVS ja AWS." in md


def test_render_glossary_still_fixes_a_word_a_call_fix_also_produces() -> None:
    """The protection is per occurrence, not a blanket skip: the glossary
    pair still rewrites the ``AVS`` that was really spoken."""
    sidecar = _sidecar()
    sidecar["paragraphs"][0]["text"] = "Puhuttiin APS ja AVS."
    sidecar["call_corrections"] = {"APS": "AVS"}
    md = render_markdown(sidecar, corrections={"AVS": "AWS"}, speaker_names={})
    assert "Puhuttiin AVS ja AWS." in md


def test_render_call_fix_keeps_a_case_suffix() -> None:
    """``AVS:n`` (the form the term check is shown) keeps its case suffix."""
    sidecar = _sidecar()
    sidecar["paragraphs"][0]["text"] = "Puhuttiin AVS:n."
    sidecar["call_corrections"] = {"AVS": "AWS"}
    md = render_markdown(sidecar, corrections={}, speaker_names={})
    assert "Puhuttiin AWS:n." in md


def test_render_call_fix_right_side_survives_a_glossary_wildcard() -> None:
    """The same rule under the wildcard rule of ``_compile_corrections``:
    ``qwe*`` matches the stem of the word the call fix produced."""
    sidecar = _sidecar()
    sidecar["paragraphs"][0]["text"] = "Katsotaan QN2C demo."
    sidecar["call_corrections"] = {"QN2C": "Qwen 27B"}
    md = render_markdown(sidecar, corrections={"qwe*": "X"}, speaker_names={})
    assert "Qwen 27B demo" in md and "X" not in md


def test_render_unrelated_glossary_pair_still_applies_alongside_call_fixes() -> None:
    """The token protection covers only the words the call fixes rewrote:
    every other glossary pair applies in the very same render."""
    sidecar = _sidecar()
    sidecar["paragraphs"][0]["text"] = "Puhuttiin QN2C:stä Blacksit-hankkeesta."
    sidecar["call_corrections"] = {"QN2C": "Qwen 27B"}
    md = render_markdown(
        sidecar,
        corrections={"qwe*": "X", "Blacksit": "Flagship"},
        speaker_names={},
    )
    assert "Qwen 27B:stä" in md and "X" not in md
    assert "Flagship-hankkeesta" in md and "Blacksit" not in md


def test_render_ignores_malformed_call_correction_entries() -> None:
    control = render_markdown(_sidecar(), corrections={}, speaker_names={})
    sidecar = _sidecar()
    sidecar["call_corrections"] = {
        "Blacksit": 42,  # non-str value
        7: "Vantaa",  # non-str key
        "": "X",  # empty key
        "epit*": "",  # empty value
    }
    assert render_markdown(sidecar, corrections={}, speaker_names={}) == control


def test_render_ignores_non_dict_call_corrections() -> None:
    sidecar = _sidecar()
    sidecar["call_corrections"] = "Blacksit => Vantaa"
    md = render_markdown(sidecar, corrections={}, speaker_names={})
    assert "Blacksit-hankkeesta" in md and "Vantaa" not in md


def test_render_sidecar_without_call_corrections_key_is_stable() -> None:
    """No key (the pre-#270 world) and an empty map render byte-identically."""
    sidecar = _sidecar()
    assert "call_corrections" not in sidecar
    first = render_markdown(sidecar, corrections=_corrections(), speaker_names={})
    assert (
        render_markdown(_sidecar(), corrections=_corrections(), speaker_names={})
        == first
    )
    sidecar["call_corrections"] = {}
    assert (
        render_markdown(sidecar, corrections=_corrections(), speaker_names={}) == first
    )


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


def test_render_same_name_labels_render_as_one_named_speaker() -> None:
    """Two labels named the same person (one person split into two
    diarization clusters) both carry the name: no bare label survives, and
    the speaker legend has one entry with the combined share (issue #215)."""
    sidecar = _sidecar()
    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Mats"},
    )

    assert "[Mats] Puhuttiin" in md
    assert "[Mats] epittä" in md
    assert "SPEAKER_1" not in md
    assert "SPEAKER_2" not in md
    assert "Mats (100%)" in md


def test_render_same_name_labels_order_does_not_matter() -> None:
    """Whichever label speaks first, every paragraph carries the name."""
    sidecar = _sidecar()
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

    assert "[Mats] Aloitus." in md
    assert "[Mats] Jatko." in md
    assert "[Mats] Lopetus." in md
    assert "[SPEAKER_" not in md


def test_render_names_replace_raw_labels_in_notes_text() -> None:
    sidecar = _sidecar()
    sidecar["notes"]["title"] = "SPEAKER_2 ja alusta"
    sidecar["notes"]["summary"] = "SPEAKER_1 esitteli, SPEAKER_12 kommentoi."
    sidecar["notes"]["action_items"] = [
        "SPEAKER_1: Kirjaa epittä",
        "SPEAKER_2: Lähetä päiväraja SPEAKER_1:lle",
    ]
    sidecar["notes"]["key_points"] = ["SPEAKER_1 esitteli alustan"]

    md = render_markdown(
        sidecar,
        corrections={},
        speaker_names={"SPEAKER_1": "Mats", "SPEAKER_2": "Mats"},
    )

    assert "# Mats ja alusta" in md
    assert "Mats esitteli, SPEAKER_12 kommentoi." in md  # whole-word only
    assert "- [ ] Mats: Kirjaa epittä" in md
    assert "- [ ] Mats: Lähetä päiväraja Mats:lle" in md
    assert "Mats esitteli alustan" in md


def test_render_notes_text_keeps_non_label_words() -> None:
    """Only raw SPEAKER_N labels are replaced inside free notes text: a
    label like "Others" is an ordinary word there."""
    sidecar = _sidecar()
    sidecar["notes"]["summary"] = "Others joined late."
    md = render_markdown(sidecar, corrections={}, speaker_names={"Others": "Bot"})
    assert "Others joined late." in md


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
    paragraphs: list[dict | str | None] = cast(
        "list[dict | str | None]", sidecar["paragraphs"]
    )
    paragraphs.append("corrupted-string-entry")
    paragraphs.append(None)

    md = render_markdown(sidecar, corrections=_corrections(), speaker_names={})
    # The valid dict entries still render with corrections applied.
    assert "Flagship-hankkeesta" in md
    assert "EBITDA selvitetään" in md


# ---------------------------------------------------------------------------
# Section language (issue #247)
# ---------------------------------------------------------------------------


def test_render_keeps_english_headings_of_an_english_sidecar() -> None:
    """An ``en`` sidecar re-renders with the same headings the run wrote."""
    sidecar = _sidecar(language="en")

    out = render_markdown(sidecar, corrections={}, speaker_names={})

    assert out == format_md(_result(language="en"), language="en")
    assert "## Summary" in out
    assert "## Tiivistelmä" not in out


def test_render_defaults_to_finnish_without_a_language_key() -> None:
    """Old sidecars (no or non-string ``language``) keep the ``fi`` headings."""
    for sidecar in (_sidecar(), _sidecar(language=None)):
        out = render_markdown(sidecar, corrections={}, speaker_names={})
        assert "## Tiivistelmä" in out
