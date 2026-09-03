"""LLM repair pass over paragraphs (issue #68 — measured working live).

A Finnish, directive prompt fixed real ASR garble in live probes
("parastaa" -> "parantaa", "ruumipalloilemaan" -> "lumipalloilemaan");
this stage productizes it with a no-invention guard: a "repair" that
rewrites too much is rejected and the original paragraph ships.
LLMClient is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vemoizer.repair import repair_paragraphs


def _client(replies):
    client = MagicMock()
    client.complete = MagicMock(side_effect=replies)
    return client


def _para(text, **extra):
    return {"start": 0.0, "end": 5.0, "text": text, **extra}


def test_repaired_text_replaces_the_paragraph() -> None:
    paras = [_para("teemme uusia asioita rotkeasti")]
    out = repair_paragraphs(_client(["teemme uusia asioita rohkeasti"]), paras)
    assert out[0]["text"] == "teemme uusia asioita rohkeasti"
    assert out[0]["start"] == 0.0  # timing/speaker metadata untouched


def test_speaker_metadata_survives_repair() -> None:
    paras = [_para("moi vaan kaikille", speaker="S1")]
    out = repair_paragraphs(_client(["moi vaan kaikille"]), paras)
    assert out[0]["speaker"] == "S1"


def test_overlong_rewrite_is_rejected() -> None:
    """A 'repair' that balloons the text is invention, not correction."""
    original = "lyhyt lause tässä"
    rewrite = "lyhyt lause tässä ja paljon uutta sisältöä jota kukaan ei sanonut " * 3
    out = repair_paragraphs(_client([rewrite]), [_para(original)])
    assert out[0]["text"] == original


def test_unrelated_rewrite_is_rejected() -> None:
    """Low similarity to the original means the model paraphrased."""
    original = "puhutaan alustan kehityksestä ja datan laadusta"
    out = repair_paragraphs(
        _client(["tänään on kaunis ilma ja aurinko paistaa"]), [_para(original)]
    )
    assert out[0]["text"] == original


def test_llm_failure_keeps_the_original() -> None:
    out = repair_paragraphs(_client([None]), [_para("alkuperäinen teksti")])
    assert out[0]["text"] == "alkuperäinen teksti"


def test_exception_keeps_all_originals() -> None:
    client = MagicMock()
    client.complete = MagicMock(side_effect=RuntimeError("boom"))
    paras = [_para("eka"), _para("toka")]
    out = repair_paragraphs(client, paras)
    assert [p["text"] for p in out] == ["eka", "toka"]


def test_empty_paragraph_is_skipped_without_a_call() -> None:
    client = _client([])
    out = repair_paragraphs(client, [_para("")])
    assert out[0]["text"] == ""
    assert client.complete.call_count == 0
