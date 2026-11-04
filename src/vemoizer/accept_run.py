"""The per-excerpt bench run and its scoring (issue #229, epic #220).

Single responsibility — what ONE excerpt's measurement is: run the current
meeting pipeline over the excerpt's bench audio through the same wrapper
``vemoizer meeting`` uses, then score the fresh hypothesis against the
owner's reviewed reference. Flags, the anonymised report and the gate live in
:mod:`vemoizer.accept_cli`; the baseline format lives in
:mod:`vemoizer.accept_baseline`.

The hypothesis is ALWAYS this fresh decode. The corrected sidecar an
excerpt's reference was drafted from is where the answer came from, so it is
never read here — a bench that scored the sidecar would measure the sidecar
writer, not the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from vemoizer import accept_baseline, accept_bundle, accept_metrics, accept_probes
from vemoizer.presets import RunOptions, replace
from vemoizer.tracks import TrackPaths

logger = logging.getLogger(__name__)


def run_excerpt(
    bench_dir: Path, excerpt: accept_bundle.Excerpt, options: RunOptions
) -> dict[str, Any] | None:
    """Run the REAL meeting path on one excerpt's bench audio.

    :func:`vemoizer.batch_guard._transcribe_one` is the wrapper
    ``vemoizer meeting`` itself uses — the fail-loud config check, then
    ``transcribe_file`` looked up on :mod:`vemoizer.pipeline` at call time
    (the preset CLI's patch seam) with the resolved :class:`RunOptions`: the
    meeting profile, diarization, REPAIR (the stage that inserts glossary
    terms), the layered config, the composed glossary, the speaker bounds and
    the language. A bench that spelled out its own kwarg subset measured a
    pipeline no user ever runs. Called with the cwd at the bench's project
    root, which is where the layered lookups resolve.

    The bench is fail-open by contract: one excerpt crashing is that
    excerpt's ``FAILED`` row (the exception type to stderr, never a path or
    transcript text), not an aborted bench.
    """
    # Deferred: batch_guard sits behind the batch/notify stack, which the
    # command imports only when it runs.
    from vemoizer.batch_guard import _transcribe_one

    paths = {slot: bench_dir / rel for slot, rel in excerpt.audio.items()}
    if excerpt.kind == "two-track":
        tracks: TrackPaths | None = TrackPaths(me=paths["me"], them=paths["them"])
        # A two-track run passes the caller's call file while the stems drive
        # the decode; the bench stores no mix for a two-track excerpt, so the
        # owner's mic stem is the ingest/fallback audio (both stems are still
        # decoded separately by the track decode stage).
        path = paths["me"]
    else:
        tracks, path = None, paths["mix"]
    # The per-excerpt fields of a meeting run, set the way the meeting CLI
    # sets them: the stems and the owner come from the excerpt, and a
    # meeting is always diarized.
    excerpt_options = replace(options, diarize=True, tracks=tracks, owner=excerpt.owner)
    try:
        return _transcribe_one(path, excerpt_options)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - a crashed excerpt is a FAILED row, never a traceback that aborts the bench
        typer.echo(f"error: excerpt decode raised {type(exc).__name__}", err=True)
        return None


def scoring_terms(options: RunOptions) -> list[str]:
    """The glossary terms scoring watches for invented insertions.

    The prompt terms plus the ``@``-LLM-only terms (``@`` already stripped
    by ``resolve_options``), deduplicated in order. The correction pairs'
    RIGHT sides are already in those lists when they are terms; the wrong
    sides are not terms and must not be counted as insertions. Feeding this
    to the scorer is what makes the repair stage's sprinkling show up
    (#228) — ``load_glossary(None)`` returned ``[]`` and hid it.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for term in [*options.whisper_prompt, *options.llm_terms]:
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def score_run(
    result: dict[str, Any] | None,
    reference: accept_bundle.Reference,
    terms: list[str],
) -> dict[str, Any] | None:
    """All metrics for one excerpt, or ``None`` when the run failed.

    The hypothesis is the RESULT's paragraphs — the post-repair text the user
    actually reads, with its speaker, suspect flags and times (glossary
    insertions are made by the repair stage, so scoring the raw segments
    would miss exactly the invention the metric is for).
    """
    if result is None:
        return None  # raised inside the pipeline; the type went to stderr
    if "error" in result:
        logger.error("excerpt failed: %s", result["error"])
        return None
    hyp = [dict(para) for para in result.get("paragraphs") or []]
    scored = accept_metrics.score_excerpt(
        [asdict(seg) for seg in reference.segments],
        hyp,
        title=str((result.get("notes") or {}).get("title") or ""),
        title_expect=reference.title_expect,
        glossary_terms=terms,
        text_reviewed=reference.reviewed_by is not None,
    )
    if reference.probes or reference.term_probes:
        # The owner's quiz is the ground truth for who spoke and which terms
        # were said, so it replaces the transcript-based speaker and glossary
        # keys. A key the quiz has no signal for (None: e.g. no term probes)
        # keeps the owner-reviewed transcript's value, so a reviewed text is
        # never silently ungated (PR #244 review); on draft text it stays None.
        quiz = accept_probes.probe_metrics(
            hyp,
            [asdict(p) for p in reference.probes],
            [asdict(t) for t in reference.term_probes],
        )
        text_reviewed = reference.reviewed_by is not None
        scored.update(
            {k: v for k, v in quiz.items() if v is not None or not text_reviewed}
        )
    # Override loop_seconds from the RAW segments: ``compression_ratio`` is a
    # whisper field that only survives on ``segments``, so scoring the
    # paragraphs would report 0.0 for every hallucination wall.
    scored["loop_seconds"] = accept_metrics.loop_seconds(
        list(result.get("segments") or [])
    )
    # ``sum([])`` is an int, so an unwallmarked / unflagged run would print
    # ``0`` while every other run prints ``0.0000``; seconds are always
    # seconds, in the report and in the numbers-only baseline alike. Keys an
    # unreviewed reference omits are simply absent (never a KeyError).
    for key in accept_baseline.SECOND_KEYS:
        if scored.get(key) is not None:
            scored[key] = float(scored[key])
    return scored
