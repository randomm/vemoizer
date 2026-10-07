"""LLM repair pass over paragraphs (issue #68 — measured working live).

A Finnish, directive prompt fixed real ASR garble in live probes
("parastaa" -> "parantaa", "ruumipalloilemaan" -> "lumipalloilemaan");
this stage productizes it with a no-invention guard: a "repair" that
rewrites too much is rejected and the original paragraph ships.
LLMClient is mocked throughout.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vemoizer.llm_budget import StageBudget
from vemoizer.repair import repair_paragraphs


class _FakeClock:
    """A controllable ``time.monotonic`` stand-in for the budget loop."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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


def test_repair_prompt_maps_glossary_near_misses() -> None:
    """The glossary must come with the mapping instruction, not just a list
    ('Flaksi' survived next to correct 'Flagship' without it)."""
    seen = {}

    def spy(system, user, max_tokens=2048, **kw):
        seen["system"] = system
        return user

    client = MagicMock()
    client.complete = MagicMock(side_effect=spy)
    repair_paragraphs(client, [_para("teksti")], glossary=["Flagship-hanke"])
    assert "foneettisesti lähellä" in seen["system"]
    assert "Flagship-hanke" in seen["system"]


# -- wall-clock budget (issue #148) ---------------------------------------
#
# A stalled connection that keeps resetting the per-call httpx timeout
# would otherwise hold the run forever. The stage-level budget bounds the
# whole loop: on expiry the stage stops calling, ships the remaining
# paragraphs un-repaired (fail-open, invariant #5), and logs one warning.
# The clock is a fake so the test is deterministic — no real sleep, no
# network.


def test_budget_exhausted_stops_loop_and_ships_originals() -> None:
    """Once the budget is spent, the loop stops and the remaining
    paragraphs ship with their ORIGINAL text (fail-open)."""
    clock = _FakeClock()
    # 4 paragraphs, each 'repair call' burns 10 s of wall time. Budget is
    # 25 s. The budget is checked at the TOP of the loop, before the call:
    #   para 1: elapsed 0  < 25 -> call  -> elapsed 10
    #   para 2: elapsed 10 < 25 -> call  -> elapsed 20
    #   para 3: elapsed 20 < 25 -> call  -> elapsed 30
    #   para 4: elapsed 30 >= 25 -> STOP (budget exhausted)
    # So 3 calls; the 4th paragraph ships un-repaired.
    clock.now = 0.0
    budget = StageBudget(25.0, clock=clock)

    # Each reply is a high-similarity fix (a single extra letter) that the
    # no-invention guard accepts — so the repaired text is visibly different
    # from the original, while the budget gate still cuts the loop off.
    def slow_complete(system, user, max_tokens=2048, **kw):
        clock.advance(10.0)  # each call burns 10 s of wall clock
        return user + "x"

    client = MagicMock()
    client.complete = MagicMock(side_effect=slow_complete)
    paras = [_para(f"lause {i}") for i in range(4)]

    out = repair_paragraphs(client, paras, budget=budget)

    assert len(out) == 4  # all paragraphs present in the output
    # Exactly 3 calls made; the 4th was cut off by the budget gate.
    assert client.complete.call_count == 3
    # The first three were repaired (high-similarity +x); the last kept
    # its original text verbatim.
    assert out[0]["text"] == "lause 0x"
    assert out[1]["text"] == "lause 1x"
    assert out[2]["text"] == "lause 2x"
    assert out[3]["text"] == "lause 3"  # untouched original
    assert out[3]["text"] != "lause 3x"


def test_budget_none_never_exhausted() -> None:
    """No budget (None) -> the stage runs to completion, every paragraph
    gets a call."""
    client = MagicMock()
    client.complete = MagicMock(side_effect=["a korjattu", "b korjattu"])
    out = repair_paragraphs(client, [_para("a"), _para("b")], budget=None)
    assert client.complete.call_count == 2
    assert len(out) == 2


def test_budget_zero_never_exhausted_disables_cap() -> None:
    """A 0-second budget is the 'no cap' fail-open case, not an immediate
    abort — the stage runs to completion."""
    clock = _FakeClock()
    budget = StageBudget(0.0, clock=clock)
    client = MagicMock()
    client.complete = MagicMock(side_effect=["a korjattu", "b korjattu"])
    out = repair_paragraphs(client, [_para("a"), _para("b")], budget=budget)
    assert client.complete.call_count == 2
    assert len(out) == 2


def test_repair_emits_throttled_heartbeat(caplog) -> None:
    """The loop logs a throttled INFO heartbeat (count + elapsed, no
    transcript text) so a hung stage is distinguishable from a slow one."""
    clock = _FakeClock()
    budget = StageBudget(1000.0, clock=clock)  # generous budget

    def slow_complete(system, user, max_tokens=2048, **kw):
        clock.advance(10.0)  # push past PROGRESS_INTERVAL_S (5.0)
        return user

    client = MagicMock()
    client.complete = MagicMock(side_effect=slow_complete)
    import logging

    with caplog.at_level(logging.INFO, logger="vemoizer.repair"):
        repair_paragraphs(client, [_para("eka"), _para("toka")], budget=budget)

    heartbeats = [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("repair:")
    ]
    # At least one heartbeat fired (elapsed crossed the 5 s interval).
    assert any("1/2 paragraphs" in h or "2/2 paragraphs" in h for h in heartbeats)
    # Privacy: the heartbeat carries only count + elapsed, never the text.
    for h in heartbeats:
        assert "eka" not in h
        assert "toka" not in h


# -- in-flight deadline (issue #148 FIX 3) --------------------------------


def test_repair_passes_deadline_s_to_client_complete() -> None:
    """(c) repair_paragraphs passes ``budget.remaining()`` as ``deadline_s``
    to each ``client.complete`` call (only when a budget exists). Without
    a budget, ``deadline_s`` is not passed (old behaviour)."""
    calls: list[dict[str, Any]] = []
    clock = _FakeClock()
    budget = StageBudget(100.0, clock=clock)

    def fake_complete(system, user, max_tokens=2048, deadline_s=None, **kw):
        calls.append({"deadline_s": deadline_s, "elapsed": clock()})
        clock.advance(1.0)
        return user  # no-op repair: passes the guard

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    paras = [_para("eka"), _para("toka")]

    repair_paragraphs(client, paras, budget=budget)
    # Two calls, each with a deadline_s (the remaining budget at call time):
    #   call 1: elapsed 0, remaining 100
    #   call 2: elapsed 1 (clock advanced by fake), remaining 99
    assert len(calls) == 2
    assert calls[0]["deadline_s"] == pytest.approx(100.0)
    assert calls[1]["deadline_s"] == pytest.approx(99.0)


def test_repair_no_budget_passes_no_deadline() -> None:
    """No budget -> ``deadline_s`` is not passed (None) — the old behaviour.
    (a) Without a deadline, the dribbling response is read fully."""
    calls: list[Any] = []

    def fake_complete(system, user, max_tokens=2048, deadline_s=None, **kw):
        calls.append(deadline_s)
        return user

    client = MagicMock()
    client.complete = MagicMock(side_effect=fake_complete)
    repair_paragraphs(client, [_para("eka")], budget=None)
    assert calls == [None]
