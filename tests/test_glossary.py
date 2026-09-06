"""User glossary: domain terms fed to ASR and LLM stages (issue #71 QA).

QA on real meetings showed proper nouns garbling ("FLAG-sit" for
Flagship-hanke, "Riihimäärä" for Riihimäki, "Nurdea" for Nordea) — whisper
has no context for the user's vocabulary. A glossary file feeds whisper's
initial_prompt and the notes/repair prompts. Fail-open: a missing or
unreadable file is an empty glossary, never an error.
"""

from __future__ import annotations

from pathlib import Path

from vemoizer.glossary import glossary_prompt, load_glossary


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
