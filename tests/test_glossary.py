"""User glossary: domain terms fed to ASR and LLM stages (issue #71 QA).

QA on real meetings showed proper nouns garbling ("FLAG-sit" for
Flagship-hanke, "Riihimäärä" for Riihimäki, "Nurdea" for Nordea) — whisper
has no context for the user's vocabulary. A glossary file feeds whisper's
initial_prompt and the notes/repair prompts. Fail-open: a missing or
unreadable file is an empty glossary, never an error.
"""

from __future__ import annotations

from pathlib import Path

from vemoizer.glossary import (
    apply_corrections,
    glossary_prompt,
    load_corrections,
    load_glossary,
)


def test_loads_one_term_per_line_skipping_comments(tmp_path: Path) -> None:
    f = tmp_path / "glossary.txt"
    f.write_text(
        "# meeting vocabulary\nFlagship-hanke\nRiihimäki\n\nMovescount\n",
        encoding="utf-8",
    )
    assert load_glossary(f) == ["Flagship-hanke", "Riihimäki", "Movescount"]


def test_missing_file_is_empty_glossary(tmp_path: Path) -> None:
    assert load_glossary(tmp_path / "nope.txt") == []


def test_none_path_is_empty_glossary() -> None:
    assert load_glossary(None) == []


def test_prompt_joins_terms_for_whisper() -> None:
    prompt = glossary_prompt(["Flagship-hanke", "Nordea", "Movescount"])
    assert prompt is not None
    assert "Flagship-hanke" in prompt
    assert "Nordea" in prompt
    assert "Movescount" in prompt


def test_empty_terms_give_no_prompt() -> None:
    assert glossary_prompt([]) is None


def test_prompt_is_bounded() -> None:
    """Whisper's prompt window is ~224 tokens; a huge glossary must not
    push the actual instruction out of it."""
    terms = [f"Termi{i}" for i in range(500)]
    prompt = glossary_prompt(terms)
    assert prompt is not None
    assert len(prompt) < 1200


# -- correction pairs (issue #71 round 2) --------------------------------
#
# "Blacksit-hankkeiksi" and "Newport case" survived the LLM repair pass.
# Known garble->canonical pairs are deterministic, not a judgment call:
# the glossary format gains "wrong => right" lines applied mechanically.


def test_corrections_parse_from_arrow_lines(tmp_path: Path) -> None:
    f = tmp_path / "glossary.txt"
    f.write_text(
        "# terms\nFlagship-hanke\nBlacksit => Flagship\nNewport => Nyborg\n",
        encoding="utf-8",
    )
    assert load_corrections(f) == {"Blacksit": "Flagship", "Newport": "Nyborg"}
    # arrow lines are corrections, not prompt terms; right sides join terms
    assert "Blacksit" not in load_glossary(f)
    assert "Flagship" in load_glossary(f)


def test_corrections_apply_on_word_boundaries() -> None:
    paras = [
        {"start": 0.0, "end": 1.0, "text": "he kutsuvat Blacksit-hankkeiksi"},
        {"start": 1.0, "end": 2.0, "text": "se Newport case", "speaker": "S1"},
    ]
    out = apply_corrections(paras, {"Blacksit": "Flagship", "Newport": "Nyborg"})
    assert out[0]["text"] == "he kutsuvat Flagship-hankkeiksi"
    assert out[1]["text"] == "se Nyborg case"
    assert out[1]["speaker"] == "S1"


def test_corrections_do_not_touch_substrings_inside_words() -> None:
    paras = [{"start": 0.0, "end": 1.0, "text": "ANewportti ei muutu"}]
    out = apply_corrections(paras, {"Newport": "Nyborg"})
    assert out[0]["text"] == "ANewportti ei muutu"


def test_corrections_are_case_insensitive_on_match() -> None:
    paras = [{"start": 0.0, "end": 1.0, "text": "blacksit hanke"}]
    out = apply_corrections(paras, {"Blacksit": "Flagship"})
    assert out[0]["text"] == "Flagship hanke"


def test_no_corrections_is_identity() -> None:
    paras = [{"start": 0.0, "end": 1.0, "text": "sama teksti"}]
    assert apply_corrections(paras, {}) == paras
