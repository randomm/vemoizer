"""LLM notes stage: chunking, JSON parsing, fail-open (issue #56).

``LLMClient.complete`` is mocked throughout — no network, no config files.
The stage must never raise and never lose the transcript: any failure
returns ``None`` and the caller ships the transcript without notes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vemoizer.llm_budget import StageBudget
from vemoizer.notes import _chunk_text, generate_notes


class _FakeClock:
    """A controllable ``time.monotonic`` stand-in for the budget loop."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _client(responses: list[str | None]) -> MagicMock:
    client = MagicMock()
    client.complete = MagicMock(side_effect=responses)
    return client


def _client_with_kw(responses: list[str | None]) -> MagicMock:
    """A fake client that records the ``deadline_s`` kwarg per call."""
    calls: list[dict] = []
    client = MagicMock()

    def fake_complete(system, user, max_tokens=2048, deadline_s=None, **kw):
        calls.append({"deadline_s": deadline_s})
        return responses[0] if len(responses) == 1 else responses.pop(0)

    client.complete = MagicMock(side_effect=fake_complete)
    client.complete_calls = calls
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

    def fake_complete(system, user, **kw):
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


# -- speaker grounding + glossary (issue #71 QA) -------------------------
#
# QA on a real meeting: the notes stage received raw text with NO speaker
# information, so action items attributed tasks to invented names ("Mui")
# and to the wrong people. Notes must see the speaker-labelled paragraphs
# and be forbidden from inventing attributions.


def test_notes_prompt_carries_speaker_labels() -> None:
    paragraphs = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "minä teen matskut",
            "speaker": "SPEAKER_01",
        },
        {"start": 6.0, "end": 9.0, "text": "sovitaan niin", "speaker": "SPEAKER_00"},
    ]
    seen = {}

    def spy(system, user, max_tokens=2048, **kw):
        seen["system"], seen["user"] = system, user
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=spy)
    notes = generate_notes(
        client, "minä teen matskut sovitaan niin", paragraphs=paragraphs
    )
    assert notes is not None
    assert "[SPEAKER_01]" in seen["user"]
    assert "[SPEAKER_00]" in seen["user"]
    # attribution rules present
    assert "SPEAKER" in seen["system"]


def test_notes_prompt_forbids_inventing_names() -> None:
    seen = {}

    def spy(system, user, max_tokens=2048, **kw):
        seen["system"] = system
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=spy)
    generate_notes(client, "teksti tässä")
    lowered = seen["system"].lower()
    assert "älä keksi" in lowered or "never invent" in lowered


def test_notes_prompt_carries_glossary_terms() -> None:
    seen = {}

    def spy(system, user, max_tokens=2048, **kw):
        seen["system"] = system
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=spy)
    generate_notes(client, "teksti", glossary=["Flagship-hanke", "Riihimäki"])
    assert "Flagship-hanke" in seen["system"]
    assert "Riihimäki" in seen["system"]


def test_notes_prompt_carries_commitment_rules() -> None:
    """Declined proposals became action items; the prompt must rule on it."""
    from vemoizer.notes import _NOTES_SYSTEM_PROMPT

    assert "ACTION ITEM RULES" in _NOTES_SYSTEM_PROMPT
    assert "declined" in _NOTES_SYSTEM_PROMPT
    assert "name mentioned once is not an owner" in _NOTES_SYSTEM_PROMPT


def test_action_item_objects_ground_owner_via_evidence() -> None:
    """Owner survives only when the evidence quote exists in the input."""
    transcript = "[S1] Mä teen matskut valmiiksi torstaina. [S2] Hyvä juttu."
    payload = json.dumps(
        {
            "title": "t",
            "summary": "s",
            "key_points": [],
            "action_items": [
                {
                    "item": "Tekee matskut valmiiksi",
                    "owner": "S1",
                    "evidence": "Mä teen matskut valmiiksi torstaina",
                },
                {
                    "item": "Ostaa ponin",
                    "owner": "S2",
                    "evidence": "tätä ei sanottu missään kohtaa",
                },
            ],
        }
    )
    notes = generate_notes(_client([payload]), transcript)
    assert notes is not None
    assert notes["action_items"] == [
        "S1: Tekee matskut valmiiksi",
        "Ostaa ponin",
    ]


def test_suspect_paragraphs_are_marked_in_the_notes_prompt() -> None:
    seen: dict[str, str] = {}

    def spy(system: str, user: str, **kw) -> str:
        seen["system"], seen["user"] = system, user
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=spy)
    paragraphs = [
        {"text": "selvä kohta", "speaker": "S1"},
        {"text": "kolme miljoonaa euroa", "speaker": "S2", "suspect": "number"},
    ]
    generate_notes(client, "x", paragraphs=paragraphs)
    assert "⚠" in seen["user"]
    assert "epävarma" in seen["system"].lower()


def test_prompt_carries_grounding_rules() -> None:
    from vemoizer.notes import _NOTES_SYSTEM_PROMPT

    assert "evidence" in _NOTES_SYSTEM_PROMPT
    assert "Sovitaan" in _NOTES_SYSTEM_PROMPT


def test_owner_prefix_is_skipped_when_item_already_starts_with_owner() -> None:
    transcript = "[S1] Tuomas laittaa pyynnöt eteenpäin huomenna."
    payload = json.dumps(
        {
            "title": "t",
            "summary": "s",
            "key_points": [],
            "action_items": [
                {
                    "item": "Tuomas laittaa pyynnöt eteenpäin",
                    "owner": "Tuomas",
                    "evidence": "Tuomas laittaa pyynnöt eteenpäin huomenna",
                }
            ],
        }
    )
    notes = generate_notes(_client([payload]), transcript)
    assert notes is not None
    assert notes["action_items"] == ["Tuomas laittaa pyynnöt eteenpäin"]


# -- wall-clock budget (issue #148) ---------------------------------------
#
# A stalled connection that keeps resetting the per-call httpx timeout
# would otherwise hold the notes stage (which makes 1..N+1 sequential
# calls) forever. The budget bounds the whole call set: on expiry the
# stage returns None (fail-open) and the transcript ships without notes.


def test_notes_budget_exhausted_returns_none_before_reduce() -> None:
    """A long transcript map-reduces (N map calls + 1 reduce). A tiny
    budget that expires after the first map call makes the loop return
    None before the reduce — the transcript ships without notes."""
    clock = _FakeClock()
    budget = StageBudget(10.0, clock=clock)
    long_text = ("sana " * 5_000).strip()  # ~25K -> several chunks (map-reduce)

    def slow_complete(system, user, max_tokens=2048, **kw):
        clock.advance(10.0)  # each call burns the full remaining budget
        if "osayhteenveto" in user:
            return _notes_json(summary="koottu")  # reduce
        return "yhden osan tiivistelmä"  # map

    client = MagicMock()
    client.complete = MagicMock(side_effect=slow_complete)
    notes = generate_notes(client, long_text, budget=budget)
    # The budget expired mid-loop -> fail-open to None (no notes).
    assert notes is None
    # The reduce call must NOT have fired (the budget cut the loop off
    # before it could reach the reduce). Only map calls ran.
    for call in client.complete.call_args_list:
        assert "osayhteenveto" not in call.args[1]


def test_notes_budget_exhausted_before_reduce_returns_none_with_warning() -> None:
    """The map loop's last call exhausts the budget exactly, so the reduce
    call fires after the loop with an exhausted budget. The reduce must be
    gated: the stage returns None without spending the reduce call, with
    one warning (issue #148 FIX 4).

    Budget = N*10 (N = number of chunks); each map call burns 10 s, so the
    budget is exactly exhausted after the last map call. Without the reduce
    gate the reduce would be called; with it, the stage returns None."""
    from vemoizer.notes import _chunk_text

    clock = _FakeClock()
    long_text = ("sana " * 5_000).strip()  # ~25K -> several chunks (map-reduce)
    n_chunks = len(_chunk_text(long_text))
    budget = StageBudget(n_chunks * 10.0, clock=clock)
    reduce_fired = [False]

    def slow_complete(system, user, max_tokens=2048, **kw):
        if "osayhteenveto" in user:
            reduce_fired[0] = True  # the reduce call — must NOT fire
            return _notes_json(summary="koottu")
        clock.advance(10.0)  # each map call burns 10 s
        return "yhden osan tiivistelm\u00e4"  # map

    client = MagicMock()
    client.complete = MagicMock(side_effect=slow_complete)

    import vemoizer.notes as notes_module

    warnings: list[str] = []
    orig_warning = notes_module.logger.warning
    notes_module.logger.warning = lambda msg, *a, **kw: warnings.append(msg % a if a else msg)
    try:
        notes = generate_notes(client, long_text, budget=budget)
    finally:
        notes_module.logger.warning = orig_warning

    # The reduce call must NOT have fired (budget exhausted after last map).
    assert reduce_fired[0] is False, "reduce call must not fire after budget exhaustion"
    assert notes is None
    # Exactly one budget warning (from the reduce gate, not the loop).
    budget_warnings = [w for w in warnings if "budget" in w]
    assert len(budget_warnings) == 1, f"expected 1 budget warning, got {budget_warnings}"


def test_notes_budget_none_runs_to_completion() -> None:
    """No budget (None) -> the notes stage runs its full map-reduce and
    returns parsed notes."""
    long_text = ("sana " * 5_000).strip()

    def fake_complete(system, user, max_tokens=2048, **kw):
        if "osayhteenveto" in user:
            return _notes_json(summary="koottu")
        return "yhden osan tiivistelmä"

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    notes = generate_notes(client, long_text, budget=None)
    assert notes is not None
    assert notes["summary"] == "koottu"


def test_notes_budget_exhausted_on_single_call_path_returns_none() -> None:
    """Even the short single-call path is budget-gated: an already-expired
    budget returns None before any client.complete call.

    The budget is constructed at clock 0, then the clock advances past the
    budget (simulating time passing before the first ``exhausted()`` check).
    """
    clock = _FakeClock()
    budget = StageBudget(10.0, clock=clock)
    clock.advance(100.0)  # 100 s passes before the stage's first gate check
    assert budget.exhausted()  # 100s elapsed > 10s budget
    client = MagicMock()
    client.complete = MagicMock(side_effect=[_notes_json()])
    notes = generate_notes(client, "lyhyt transkripti", budget=budget)
    assert notes is None
    # The budget gate fired before any call was made.
    assert client.complete.call_count == 0


# -- in-flight deadline (issue #148 FIX 3) --------------------------------


def test_notes_passes_deadline_s_to_client_complete() -> None:
    """(d) generate_notes passes ``budget.remaining()`` as ``deadline_s``
    for each call (only when a budget exists); without a budget,
    ``deadline_s=None`` (old behaviour)."""
    clock = _FakeClock()
    budget = StageBudget(100.0, clock=clock)
    calls: list[float | None] = []

    def fake_complete(system, user, max_tokens=2048, deadline_s=None, **kw):
        calls.append(deadline_s)
        clock.advance(1.0)
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    notes = generate_notes(client, "lyhyt transkripti", budget=budget)
    assert notes is not None
    assert len(calls) == 1  # short transcript -> single call
    assert calls[0] == pytest.approx(100.0)  # remaining at call time


def test_notes_no_budget_deadline_s_is_none() -> None:
    """No budget -> ``deadline_s`` is None (old behaviour, fully read)."""
    calls: list[float | None] = []

    def fake_complete(system, user, max_tokens=2048, deadline_s=None, **kw):
        calls.append(deadline_s)
        return _notes_json()

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    notes = generate_notes(client, "lyhyt transkripti", budget=None)
    assert notes is not None
    assert calls == [None]
