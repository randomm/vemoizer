"""LLM notes stage: chunking, JSON parsing, fail-open (issue #56).

``LLMClient.complete`` is mocked throughout — no network, no config files.
The stage must never raise and never lose the transcript: any failure
returns ``None`` and the caller ships the transcript without notes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from vemoizer.notes import _chunk_text, generate_notes


def _client(responses: list[str | None]) -> MagicMock:
    client = MagicMock()
    client.complete = MagicMock(side_effect=responses)
    return client


def _notes_json(**overrides) -> str:
    data = {
        "title": "Viikkopalaveri",
        "summary": "Keskusteltiin alustan suunnasta.",
        "key_points": ["Alusta etenee", "Deployment automatisoidaan"],
        "action_items": ["Kirjaa backlog-itemit"],
    }
    data.update(overrides)
    return json.dumps(data, ensure_ascii=False)


# -- _chunk_text ---------------------------------------------------------


def test_short_text_is_one_chunk() -> None:
    assert _chunk_text("moi " * 100, chunk_chars=12_000) == ["moi " * 100]


def test_long_text_splits_on_whitespace_within_budget() -> None:
    text = ("sana " * 5000).strip()  # 25K chars
    chunks = _chunk_text(text, chunk_chars=12_000)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 12_000
    # nothing lost, nothing duplicated
    assert " ".join(chunks).split() == text.split()


def test_chunking_never_splits_inside_a_word() -> None:
    text = ("pitkähkösana " * 2000).strip()
    for chunk in _chunk_text(text, chunk_chars=1000):
        assert not chunk.startswith("ana ")
        for word in chunk.split():
            assert word == "pitkähkösana"


# -- generate_notes ------------------------------------------------------


def test_single_call_returns_parsed_notes() -> None:
    client = _client([_notes_json()])
    notes = generate_notes(client, "lyhyt transkripti tästä")
    assert notes is not None
    assert notes["title"] == "Viikkopalaveri"
    assert notes["key_points"] == ["Alusta etenee", "Deployment automatisoidaan"]
    assert client.complete.call_count == 1


def test_json_inside_a_code_fence_is_parsed() -> None:
    fenced = f"```json\n{_notes_json()}\n```"
    notes = generate_notes(_client([fenced]), "transkripti")
    assert notes is not None
    assert notes["title"] == "Viikkopalaveri"


def test_long_transcript_map_reduces() -> None:
    long_text = ("sana " * 15_000).strip()  # ~75K chars -> several chunks

    def fake_complete(system, user):
        if "osayhteenveto" in user:
            return _notes_json(summary="koottu")  # reduce call sees the parts
        return "yhden osan tiivistelmä"  # map calls

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    notes = generate_notes(client, long_text)
    assert notes is not None
    assert notes["summary"] == "koottu"
    assert client.complete.call_count >= 3  # at least 2 maps + 1 reduce


def test_unparseable_response_returns_none() -> None:
    assert generate_notes(_client(["tässä ei ole jsonia"]), "teksti") is None


def test_client_failure_returns_none() -> None:
    assert generate_notes(_client([None]), "teksti") is None


def test_missing_fields_are_defaulted_not_fatal() -> None:
    partial = json.dumps({"title": "Vain otsikko"})
    notes = generate_notes(_client([partial]), "teksti")
    assert notes is not None
    assert notes["title"] == "Vain otsikko"
    assert notes["summary"] == ""
    assert notes["key_points"] == []
    assert notes["action_items"] == []


def test_non_string_items_are_coerced_or_dropped() -> None:
    weird = json.dumps(
        {"title": 42, "summary": None, "key_points": ["ok", 7], "action_items": "x"}
    )
    notes = generate_notes(_client([weird]), "teksti")
    assert notes is not None
    assert notes["title"] == "42"
    assert notes["summary"] == ""
    assert notes["key_points"] == ["ok", "7"]
    assert notes["action_items"] == []  # a bare string is not a list


def test_empty_transcript_returns_none_without_calling_the_llm() -> None:
    client = _client([])
    assert generate_notes(client, "   ") is None
    assert client.complete.call_count == 0


def test_generate_notes_never_raises() -> None:
    client = MagicMock()
    client.complete = MagicMock(side_effect=RuntimeError("provider exploded"))
    assert generate_notes(client, "teksti") is None
