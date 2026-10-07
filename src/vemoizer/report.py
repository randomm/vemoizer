"""Per-file end-of-run quality report (issue #75, M6).

``render_report`` is a pure function of the run dict (as returned by
``transcribe_file``) plus the parameters only the CLI/batch layer knows:
``diarize_requested`` (the flag the user passed — the run dict carries no
diarization-status key) and ``glossary_source`` (the resolved M2
glossary-path string + term count, report-only per the ticket — never
stored on the run dict). The language parameter selects the section
labels (``"fi"`` default, ``"en"``).

The report is NOT an output format: the CLI/batch layer renders it once
per file (BEFORE the destructive ``result.pop("warnings")``), prints it
to stdout (suppressed by ``--quiet``), and stores the string under the
``quality_report`` run-dict key so ``format_md`` can embed it as a
``<details>`` block. The report is a pure computation — it never raises
for any run-dict shape the pipeline can produce (missing keys, empty
lists, non-dict paragraphs): every section degrades to an omission,
never a crash.
"""

from __future__ import annotations

import re
from typing import Any

from .output.markdown import _clock_or_none
from .selfheal import find_degenerate_windows

#: Stable warning anchors (substring match) → report category. The
#: classification keys off the stable prefix of each warning, not the
#: full text, so a later wording change inside the sentence does not
#: un-categorise the warning.
_DIARIZATION_ANCHOR = "diarization"
_NOTES_ANCHOR = "notes"

#: The 3-worst-suspects cap (the ticket: "⚠ count with 3 worst
#: timestamps").
_MAX_SUSPECTS = 3


def _paragraphs(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """The run-dict paragraphs as a dict-only list (non-dicts dropped)."""
    paragraphs = transcript.get("paragraphs")
    if not isinstance(paragraphs, list):
        return []
    return [p for p in paragraphs if isinstance(p, dict)]


def _start_value(para: dict[str, Any]) -> float:
    """``para["start"]`` as a finite float; ``inf`` when missing/invalid.

    ``inf`` sorts last, so a suspect paragraph with a missing ``start``
    never wins the "earliest start" tiebreak over a valid one and never
    crashes the sort.
    """
    value = para.get("start")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("inf")
    value = float(value)
    return value if value == value else float("inf")


def _worst_suspects(paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Up to ``_MAX_SUSPECTS`` suspect paragraphs, ranked for the report.

    Rank: ``garble`` before ``number`` (garble is the stronger reader
    warning), ties broken by earliest ``start``. Suspect values other
    than the two known ones are excluded (the markdown fallback renders
    them, but the report only tracks the two labelled categories).
    """
    suspects = [p for p in paragraphs if p.get("suspect") in ("garble", "number")]
    suspects.sort(key=lambda p: (0 if p["suspect"] == "garble" else 1, _start_value(p)))
    return suspects[:_MAX_SUSPECTS]


def _residual_windows(transcript: dict[str, Any]) -> list[tuple[float, float]]:
    """``find_degenerate_windows`` over the FINAL segments, ``[]`` on absence.

    Runs the pure detector (``selfheal.find_degenerate_windows``) on
    ``transcript["segments"]`` — the post-heal list, so only windows the
    pipeline's own ``heal`` pass did NOT fix are reported. A missing or
    non-list ``segments`` key yields ``[]`` (section omitted), never a
    crash.
    """
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        return []
    return find_degenerate_windows([s for s in segments if isinstance(s, dict)])


def _matched_terms(terms: list[str], paragraphs: list[dict[str, Any]]) -> int:
    """Whole-word, case-insensitive hits of *terms* in the paragraph text.

    Only prompt terms are matched (``load_glossary`` already excludes
    ``=>`` correction pairs). The ``@`` prefix is stripped before the
    match (per M2 the prefix is a marker, not part of the term). A term
    matches once the text contains it as a whole word — the count is the
    number of distinct terms that matched at least once, shown honestly
    even when zero.
    """
    total_text = " ".join(str(p.get("text", "")) for p in paragraphs)
    if not total_text:
        return 0
    matched = 0
    for term in terms:
        term = term.strip()
        while term.startswith("@"):
            term = term[1:].lstrip()
        if not term:
            continue
        # \b/\b word boundaries; \. escapes the term literally (re.escape
        # would also escape non-ASCII letters, which is fine for \b in a
        # str pattern — Python treats \b as a unicode word boundary).
        if re.search(rf"\b{re.escape(term)}\b", total_text, re.IGNORECASE):
            matched += 1
    return matched


def _categorized_warnings(warnings: list[str]) -> dict[str, list[str]]:
    """Classify warnings by stable anchor; unmatched warnings dropped.

    ``diarization``-anchored warnings (attribution, "diarization failed")
    go under "diarization"; ``notes``-anchored warnings (the notes-failed
    warning) under "notes". A warning matching no anchor is NOT listed
    (it still reaches stderr via the CLI pop path). No repair producer
    exists in the codebase (repair.py logs internally only), so no
    repair category is rendered.
    """
    out: dict[str, list[str]] = {"diarization": [], "notes": []}
    for warning in warnings:
        text = str(warning)
        lowered = text.lower()
        if _DIARIZATION_ANCHOR in lowered:
            out["diarization"].append(text)
        elif _NOTES_ANCHOR in lowered:
            out["notes"].append(text)
        # No anchor match → not listed in the report.
    return out


def _normalize_language(language: str) -> str:
    """Coerce ``language`` to ``"fi"`` or ``"en"`` (default ``"fi"``)."""
    return "en" if str(language).strip().lower() == "en" else "fi"


def render_report(
    transcript: dict[str, Any],
    *,
    diarize_requested: bool = False,
    glossary_source: str | None = None,
    glossary_terms: list[str] | None = None,
    language: str = "fi",
) -> str:
    """Render the per-file quality report as a Markdown string.

    Pure function of the run dict plus the CLI/batch-layer parameters.
    Returns ``""`` when the report is empty (nothing to report); a
    non-empty string otherwise. Never raises for any dict shape the
    pipeline can produce — every section degrades to an omission.

    Sections (each omitted when its input is absent):

    - **Speakers**: found count from the run-dict paragraphs vs the
      requested state — ``diarize_requested`` with zero speaker-labeled
      paragraphs is the "requested but absent" state, rendered
      honestly (never a fabricated speaker list).
    - **Suspects**: ``⚠`` count plus the 3 worst timestamps (garble
      before number, ties earliest start). Count 0 → section omitted.
    - **Residual loops**: ``find_degenerate_windows`` over the FINAL
      segments; count and time windows. 0 → "none" rendered only when
      the section has something to anchor against; per the ticket, count
      0 omits the line.
    - **Glossary**: whole-word, case-insensitive hits of the prompt
      terms; shown honestly (``0 of N`` when nothing matched).
    - **Warnings**: classified by stable anchor (diarization / notes).
      Unmatched warnings dropped from the report (still on stderr).
    """
    lang = _normalize_language(language)
    paragraphs = _paragraphs(transcript)
    sections: list[str] = []

    # -- Speakers -------------------------------------------------------
    speakers = sorted(
        {
            p["speaker"]
            for p in paragraphs
            if isinstance(p.get("speaker"), str) and p["speaker"]
        }
    )
    if speakers:
        label = {"fi": "Puhujat", "en": "Speakers"}[lang]
        sections.append(f"{label}: {len(speakers)} ({', '.join(speakers)})")
    elif diarize_requested:
        label = {"fi": "Puhujat", "en": "Speakers"}[lang]
        note = {"fi": "pyydetty, ei löytynyt", "en": "requested, none found"}[lang]
        sections.append(f"{label}: {note}")

    # -- Suspects -------------------------------------------------------
    suspects = _worst_suspects(paragraphs)
    if suspects:
        label = {"fi": "Epävarmat kohdat", "en": "Suspect regions"}[lang]
        worst_lines: list[str] = []
        for para in suspects:
            stamp = _clock_or_none(para.get("start"))
            time_part = f" {stamp}" if stamp else ""
            worst_lines.append(f"- {para['suspect']}{time_part}")
        total_suspects = len([p for p in paragraphs if p.get("suspect")])
        suspect_labels = [
            "garble" if s["suspect"] == "garble" else "number" for s in suspects
        ]
        sections.append(
            f"{label}: {total_suspects} "
            f"({', '.join(suspect_labels)})"
            "\n" + "\n".join(worst_lines)
        )

    # -- Residual loops -------------------------------------------------
    windows = _residual_windows(transcript)
    if windows:
        label = {"fi": "Jäljellä olevat loopit", "en": "Residual loops"}[lang]
        window_lines = [
            f"- {_clock_or_none(s) or '?'} → {_clock_or_none(e) or '?'}"
            for s, e in windows
        ]
        sections.append(f"{label}: {len(windows)}" + "\n" + "\n".join(window_lines))

    # -- Glossary -------------------------------------------------------
    terms = glossary_terms or []
    if glossary_source or terms:
        hits = _matched_terms(terms, paragraphs)
        label = {"fi": "Sanastoon osumat", "en": "Glossary hits"}[lang]
        source_part = f" ({glossary_source})" if glossary_source else ""
        sections.append(f"{label}: {hits} of {len(terms)}{source_part}")

    # -- Languages (issue #147) ---------------------------------------
    # Per-window language distribution (display only, invariant #3):
    # e.g. "fi 29/30, en 1/30". Present only on meeting-profile runs
    # where the whisper decode detected per-window languages; absent
    # on dictation runs (no per-window detection) and empty decodes.
    lang_summary = transcript.get("language_summary")
    if isinstance(lang_summary, str) and lang_summary:
        label = {"fi": "Kielet", "en": "Languages"}[lang]
        sections.append(f"{label}: {lang_summary}")

    # -- Warnings -------------------------------------------------------
    warnings = transcript.get("warnings")
    warnings = [w for w in warnings] if isinstance(warnings, list) else []
    categorized = _categorized_warnings([str(w) for w in warnings])
    for category in ("diarization", "notes"):
        items = categorized[category]
        if items:
            label = {"fi": "Varoitukset", "en": "Warnings"}[lang]
            sections.append(
                f"{label} ({category}):\n" + "\n".join(f"- {w}" for w in items)
            )

    return "\n\n".join(sections)


def build_quality_report(
    transcript: dict[str, Any],
    *,
    diarize_requested: bool = False,
    glossary_source: str | None = None,
    glossary_terms: list[str] | None = None,
    language: str = "fi",
) -> str:
    """``render_report`` wrapped in fail-open semantics (invariant #5).

    A report generation failure (any exception) returns ``""`` — the
    caller omits the ``<details>`` block and the transcript document is
    written successfully. This is the seam the CLI/batch layer calls:
    store the result under the ``quality_report`` key on the run dict
    before the destructive ``result.pop("warnings")``.
    """
    try:
        return render_report(
            transcript,
            diarize_requested=diarize_requested,
            glossary_source=glossary_source,
            glossary_terms=glossary_terms,
            language=language,
        )
    except Exception:  # noqa: BLE001 - fail-open (block omitted, run continues)
        return ""
