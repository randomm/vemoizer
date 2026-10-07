"""LLM tail of the consensus pipeline: repair pass and notes generation.

Extracted from :mod:`vemoizer.pipeline` (500-line limit; single
responsibility: the optional, fail-open LLM stages that run after the
transcript is already assembled). Both stages are presentation-layer: a
failure degrades to the un-repaired paragraphs / summary-less Markdown,
reported through the ``warnings`` channel, never to an error.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from rich.progress import TaskID

from .glossary import apply_corrections_to_notes
from .llm import LLMClient, LLMConfig
from .llm_budget import StageBudget
from .notes import generate_notes
from .progress import ProgressDisplay, format_duration
from .repair import repair_paragraphs

if TYPE_CHECKING:
    from .preset_interrupt import InterruptTracker

logger = logging.getLogger(__name__)

#: The .md file still renders as a clean transcript document when notes
#: fail; the warning tells the user why it has no summary.
NOTES_FAILURE_WARNING = "notes generation failed; the Markdown output has no summary"


def _run_repair(
    result: dict[str, Any],
    llm_config: LLMConfig,
    glossary: Any,
    repair_paragraphs_fn=None,
    llm_client_cls=None,
    display: ProgressDisplay | None = None,
    tracker: InterruptTracker | None = None,
) -> None:
    """LLM repair pass over the final paragraphs (fixes phonetic ASR garble;
    guarded against invention inside ``repair_paragraphs``).

    A wall-clock budget (``repair_budget_seconds``) bounds the whole loop:
    on expiry the stage fails open — the remaining paragraphs ship
    un-repaired and one warning is logged (invariant #5).
    """
    fn = repair_paragraphs_fn or repair_paragraphs
    client_cls = llm_client_cls or LLMClient
    budget = StageBudget(llm_config.repair_budget_seconds)
    repair_client = client_cls(llm_config)
    if tracker is not None:
        # Right before the repair work starts (issue #148 FIX 3): the
        # label a Ctrl-C reads must be the stage actually in flight.
        tracker.set_stage("repair")
    repair_task: TaskID | None = (
        display.add_stage("repair", total=len(result["paragraphs"]))
        if display is not None
        else None
    )

    def _report(done: int, total: int) -> None:
        assert display is not None and repair_task is not None
        display.update_text(repair_task, f"repair {done}/{total} paragraphs")
        display.advance(repair_task, 1)

    try:
        result["paragraphs"] = fn(
            repair_client,
            result["paragraphs"],
            glossary=glossary or None,
            budget=budget,
            progress_cb=_report if display is not None else None,
        )
    finally:
        repair_client.close()
        if repair_task is not None and display is not None:
            display.finish(repair_task)


def _run_notes(
    result: dict[str, Any],
    llm_config: LLMConfig,
    corrections: Any,
    glossary: Any,
    generate_notes_fn=None,
    llm_client_cls=None,
    display: ProgressDisplay | None = None,
    tracker: InterruptTracker | None = None,
) -> None:
    """Generate the Markdown summary; a failure warns, never aborts.

    A wall-clock budget (``notes_budget_seconds``) bounds the call set:
    on expiry the stage fails open — the transcript ships without notes
    and one warning is logged (invariant #5).
    """
    fn = generate_notes_fn or generate_notes
    client_cls = llm_client_cls or LLMClient
    notes_start = time.monotonic()
    budget = StageBudget(llm_config.notes_budget_seconds)
    client = client_cls(llm_config)
    if tracker is not None:
        # Right before the notes work starts (issue #148 FIX 3): a Ctrl-C
        # during repair must read 'repair', during notes 'notes' — the
        # stage is set where the work actually begins, not upstream.
        tracker.set_stage("notes")
    notes_task = display.add_stage("notes") if display is not None else None
    try:
        notes = fn(
            client,
            result["text"],
            paragraphs=result.get("paragraphs"),
            glossary=glossary or None,
            budget=budget,
        )
    finally:
        client.close()
        if notes_task is not None and display is not None:
            display.finish(notes_task)
    if notes is not None:
        if corrections:
            notes = apply_corrections_to_notes(notes, corrections)
        result["notes"] = notes
        logger.info(
            "notes: generated in %s",
            format_duration(time.monotonic() - notes_start),
        )
    else:
        result.setdefault("warnings", []).append(NOTES_FAILURE_WARNING)
        logger.warning("notes: generation failed (transcript unaffected)")


def apply_llm_tail(
    result: dict[str, Any],
    llm_config: LLMConfig | None,
    *,
    repair: bool,
    corrections: Any,
    glossary: Any,
    generate_notes_fn=None,
    repair_paragraphs_fn=None,
    llm_client_cls=None,
    display: ProgressDisplay | None = None,
    tracker: InterruptTracker | None = None,
) -> None:
    """Run the repair and notes stages in place on the assembled *result*.

    ``corrections`` is the deterministic glossary corrections table applied
    to the notes (never to the verbatim segments). Both stages are skipped
    without an LLM config (fail-open).

    ``glossary`` is the list ``load_glossary`` returns (bare prompt terms
    plus ``@``-prefixed LLM-only terms, one list for both consumers). The
    ``@`` is stripped at this boundary — the LLM stages — before it reaches
    the model (a leading ``@`` would read as a mention marker). ``@``-terms
    never enter the whisper prompt: ``glossary_prompt`` is the single
    enforcement point for that, so they reach repair and notes here, at no
    budget.

    The ``*_fn`` / ``llm_client_cls`` parameters let the caller (pipeline)
    pass its own namespace references so that test monkeypatching of the
    pipeline namespace propagates through to the tail stages.
    """
    # LLM-only glossary terms arrive ``@``-prefixed (issue #82); strip it
    # here, at the LLM boundary — the recognizer never sees them.
    if glossary is not None:
        glossary = [t[1:].lstrip() if t.startswith("@") else t for t in glossary]
    if repair and llm_config is not None and result.get("paragraphs"):
        _run_repair(
            result,
            llm_config,
            glossary,
            repair_paragraphs_fn,
            llm_client_cls,
            display,
            tracker,
        )
    if llm_config is not None and result.get("text"):
        _run_notes(
            result,
            llm_config,
            corrections,
            glossary,
            generate_notes_fn,
            llm_client_cls,
            display,
            tracker,
        )
