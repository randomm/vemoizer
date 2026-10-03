"""User glossary: domain terms fed to ASR and LLM stages (issue #71 QA).

QA on real meetings showed proper nouns garbling ("FLAG-sit" for
Flagship-hanke, "Riihimäärä" for Riihimäki, "Nurdea" for Nordea) — whisper
has no context for the user's vocabulary. A glossary file feeds whisper's
initial_prompt and the notes/repair prompts. Fail-open: a missing or
unreadable file is an empty glossary, never an error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vemoizer.glossary import (
    GLOSSARY_PROMPT_TOKEN_BUDGET,
    apply_corrections,
    glossary_prompt,
    load_corrections,
    load_glossary,
)


class FakeTokenizer:
    """A whisper-tokenizer stand-in with deterministic token counts.

    Each space-separated token costs 1 (the separator) plus 1 per non-space
    character in the token. Deliberately simple; the real tokenizer is
    exercised by the pipeline.
    """

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for i, tok in enumerate(text.split()):
            if i > 0:
                ids.append(0)  # separator token
            ids.append(len(tok))  # 1 per non-space char in the token
        return ids


def test_loads_one_term_per_line_skipping_comments(tmp_path: Path) -> None:
    f = tmp_path / "glossary.txt"
    f.write_text(
        "# meeting vocabulary\nFlagship-hanke\nRiihimäki\n\nMovescount\n",
        encoding="utf-8",
    )
    assert load_glossary(f) == ["Flagship-hanke", "Riihimäki", "Movescount"]


def test_non_utf8_explicit_glossary_fails_loud(tmp_path: Path) -> None:
    """A non-UTF-8 ``--glossary`` file is a user error: ValueError (fail-loud),
    not a raw UnicodeDecodeError and not a silent empty glossary."""
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_glossary(bad)
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_corrections(bad)


def test_missing_file_is_empty_glossary(tmp_path: Path) -> None:
    assert load_glossary(tmp_path / "nope.txt") == []


def test_none_path_is_empty_glossary() -> None:
    assert load_glossary(None) == []


def test_prompt_joins_terms_for_whisper() -> None:
    prompt = glossary_prompt(
        ["Flagship-hanke", "Nordea", "Movescount"], FakeTokenizer()
    )
    assert prompt is not None
    assert "Flagship-hanke" in prompt
    assert "Nordea" in prompt
    assert "Movescount" in prompt


def test_empty_terms_give_no_prompt() -> None:
    assert glossary_prompt([], FakeTokenizer()) is None


def test_prompt_is_bounded_by_token_budget() -> None:
    """Whisper's prompt window is 223 tokens; a huge glossary must not
    push the actual instruction out of it. The budget is in WHISPER
    TOKENS (issue #76), not characters."""
    terms = [f"Termi{i}" for i in range(500)]
    tok = FakeTokenizer()
    prompt = glossary_prompt(terms, tok)
    assert prompt is not None
    # The prompt string must fit in GLOSSARY_PROMPT_TOKEN_BUDGET tokens,
    # including the trailing period (the neutral form has no label prefix).
    assert len(tok.encode(prompt)) <= GLOSSARY_PROMPT_TOKEN_BUDGET


def test_prompt_tail_priority_and_drop_notice(caplog: pytest.LogCaptureFixture) -> None:
    """The LAST-listed term is highest priority and sits at the TAIL of
    the prompt string; earliest-listed (lowest-priority) terms are dropped
    first with a logged notice (never silently)."""
    terms = [f"Termi{i}" for i in range(200)]
    tok = FakeTokenizer()
    with caplog.at_level(logging.WARNING, logger="vemoizer.glossary"):
        prompt = glossary_prompt(terms, tok)
    assert prompt is not None
    # Highest-priority (last-listed) term is at the tail.
    assert prompt.endswith("Termi199.")
    # Lowest-priority (earliest-listed) terms are absent.
    assert "Termi0" not in prompt
    # A notice naming the dropped terms was emitted via logger.warning.
    assert any("dropped" in r.message.lower() for r in caplog.records), (
        "expected a drop notice via logger.warning"
    )


def test_prompt_at_prefixed_never_enter_whisper() -> None:
    """REGRESSION (issue #76): @-prefixed LLM-only names must never enter
    the whisper prompt, at any budget, and cannot occupy the tail via the
    @ path."""
    terms = ["@Jukka Loikkanen", "@Maija", "Flagship-hanke", "Nordea"]
    tok = FakeTokenizer()
    prompt = glossary_prompt(terms, tok)
    assert prompt is not None
    # No @ term (or its name) appears in the prompt.
    assert "Jukka" not in prompt
    assert "Loikkanen" not in prompt
    assert "Maija" not in prompt
    # The tail must be a non-@ term (Nordea, the last non-@-term).
    assert prompt.endswith("Nordea.")


def test_prompt_at_only_terms_give_no_prompt() -> None:
    """A glossary with only @-prefixed terms yields no whisper prompt —
    @-terms must not occupy the prompt tail via the @ path."""
    terms = ["@Jukka Loikkanen", "@Maija", "@Peltsi"]
    tok = FakeTokenizer()
    assert glossary_prompt(terms, tok) is None


# -- correction pairs (issue #71 round 2) --------------------------------
#
# "Blacksit-hankkeiksi" and "Newport case" survived the LLM repair pass.
# Known garble->canonical pairs are deterministic, not a judgment call:
# the glossary format gains "wrong => right" lines applied mechanically.


def test_corrections_parse_from_arrow_lines(tmp_path: Path) -> None:
    f = tmp_path / "glossary.txt"
    f.write_text(
        "# terms\n"
        "Flagship-hanke\n"
        "@Janni Peltola\n"
        "Blacksit => Flagship\n"
        "Newport => Nyborg\n",
        encoding="utf-8",
    )
    assert load_corrections(f) == {"Blacksit": "Flagship", "Newport": "Nyborg"}
    # arrow lines are corrections, never prompt terms — neither side:
    # prompt order is load-bearing (whisper echoes what leads it), so only
    # explicitly listed terms may seed recognition. A name deliberately
    # kept out of the prompt must not come back through a pair's right
    # side. @-lines ARE returned (LLM-only, stripped at the LLM boundary);
    # glossary_prompt is where they stay out of the whisper prompt.
    assert load_glossary(f) == ["Flagship-hanke", "@Janni Peltola"]
    prompt = glossary_prompt(load_glossary(f), FakeTokenizer())
    assert prompt is not None
    assert "Janni Peltola" not in prompt
    assert "Flagship-hanke" in prompt


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


def test_prefix_correction_covers_inflections(tmp_path: Path) -> None:
    """Finnish inflects: epittä/epitävaikutuksia must all land on EBITDA."""
    paras = [
        {"start": 0.0, "end": 1.0, "text": "katsotaan epittä ensin"},
        {"start": 1.0, "end": 2.0, "text": "ja epitävaikutuksia sitten"},
    ]
    out = apply_corrections(paras, {"epit*": "EBITDA"})
    assert out[0]["text"] == "katsotaan EBITDA ensin"
    assert out[1]["text"] == "ja EBITDA-vaikutuksia sitten"


def test_prefix_correction_never_fires_inside_words() -> None:
    paras = [{"start": 0.0, "end": 1.0, "text": "resepit ovat hyviä"}]
    out = apply_corrections(paras, {"epit*": "EBITDA"})
    assert out[0]["text"] == "resepit ovat hyviä"


def test_notes_strings_get_corrections() -> None:
    from vemoizer.glossary import apply_corrections_to_notes

    notes = {
        "title": "Click Sense -siirtymä",
        "summary": "Puhuttiin Click Sensestä.",
        "key_points": ["Click Sense korvataan"],
        "action_items": [],
    }
    out = apply_corrections_to_notes(notes, {"Click Sense": "Qlik Sense"})
    assert out["title"] == "Qlik Sense -siirtymä"
    assert out["key_points"] == ["Qlik Sense korvataan"]


def test_whole_word_backslash_in_right_side_is_literal() -> None:
    """REGRESSION (issue #79): a backslash in the right side must be
    literal in the output, not interpreted as a re.sub group reference."""
    paras = [{"start": 0.0, "end": 1.0, "text": "sanoi foo bar"}]
    out = apply_corrections(paras, {"foo": "bar\\baz"})
    assert out[0]["text"] == "sanoi bar\\baz bar"


def test_prefix_backslash_in_right_side_is_literal() -> None:
    """REGRESSION (issue #79): a backslash in the right side of a
    prefix correction must be literal in the output."""
    paras = [{"start": 0.0, "end": 1.0, "text": "testi epittä x"}]
    out = apply_corrections(paras, {"epit*": "EBITDA\\suffix"})
    assert out[0]["text"] == "testi EBITDA\\suffix x"


def test_whole_word_group_ref_in_right_side_is_literal() -> None:
    """REGRESSION (issue #79): a right side containing a backslash-digit
    (which re.sub would treat as a group reference) must be literal."""
    paras = [{"start": 0.0, "end": 1.0, "text": "sanoi foo"}]
    out = apply_corrections(paras, {"foo": "a\\1b"})
    assert out[0]["text"] == "sanoi a\\1b"
