"""Markdown notes output: title, summary, action items, transcript (issue #56).

The ``md`` format is the human-facing deliverable the spec promises
("LLM cleanup / summary -> text + Markdown"): a note you can read, not a
subtitle file. Sections render only when the notes stage produced them —
with no notes at all the document is a clean paragraphed transcript, so
the format degrades gracefully along the LLM's fail-open path.

M3 (issue #77): a multi-part group's ``transcript["part_markers"]`` (a
list of ``{"offset": float, "label": str}`` dicts injected by the batch
runner — no Markdown post-processing) is rendered as a standalone line
interleaved with the paragraph blocks at its offset; transcripts without
the key render unchanged.

M6 (issue #75): a reader-ready document. A header block (date, duration,
parts count, speaker legend with talk share, glossary provenance) renders
above the ``# {title}`` line — each line is omitted when its input is
absent, so the document degrades gracefully. Each timestamped paragraph
is prefixed with its start time as ``[hh:mm:ss]`` (omitted when the
paragraph has no ``start``), and a suspect paragraph renders a labelled
warning — ``suspect="garble"`` → ``⚠ epäselvä``, ``suspect="number"`` →
``⚠ luku`` (replacing the old bare ``⚠ `` prefix). The section headings
honour the ``language`` parameter (``"fi"`` default, ``"en"``), with the
section language threaded from the run dict / batch layer.
"""

from __future__ import annotations

from typing import Any

# Reader-warning labels for the two suspect values stamped by
# ``confidence.flag_suspect_segments``. Any other (future) suspect value
# renders as ``⚠ {value} `` via the fallback in ``_suspect_prefix``.
_SUSPECT_LABELS = {"garble": "⚠ epäselvä", "number": "⚠ luku"}


def _part_markers(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """``transcript["part_markers"]`` when present, else ``[]``.

    The dicts are built by ``run_batch`` from the ``PartOffset`` fields
    (``offset`` / ``label``) — non-dict entries (``None``, a stray
    string) cannot be rendered as markers and are dropped rather than
    crashing the render path with an ``AttributeError`` (``m.get`` on a
    non-dict).
    """
    markers = transcript.get("part_markers")
    if not isinstance(markers, list):
        return []
    return [m for m in markers if isinstance(m, dict)]


def _marker_labels(markers: list[dict[str, Any]]) -> list[str]:
    """The stripped, non-empty marker labels.

    One shared label-stripping helper for both render paths (the
    interleaved ``_render_blocks`` and the no-paragraphs branch of
    ``format_md``): the ``label`` of each marker, stripped, with blank
    labels dropped. ``markers`` is already the dict-only list from
    ``_part_markers`` (non-dict entries are dropped there), so this is
    tolerant of a missing/blank ``label`` key (``m.get("label", "")``).
    """
    labels: list[str] = []
    for m in markers:
        label = str(m.get("label", "")).strip()
        if label:
            labels.append(label)
    return labels


def _render_blocks(
    blocks: list[tuple[float, str]], markers: list[dict[str, Any]]
) -> list[str]:
    """Render (offset, text) blocks with marker lines interleaved.

    Each marker renders immediately before the first block whose start
    offset is >= the marker offset; markers past the last block's start
    go at the end. An empty marker list renders the blocks unchanged
    (the common single-file case — no part_markers key, no behaviour
    change).
    """
    lines: list[str] = []
    pending = sorted(markers, key=lambda m: float(m.get("offset", 0.0)))
    pi = 0
    for offset, text in blocks:
        while pi < len(pending) and float(pending[pi].get("offset", 0.0)) <= offset:
            # _marker_labels strips + drops blanks; a single-element call
            # shares the exact label semantics with the no-paragraphs branch.
            lines.extend(_marker_labels([pending[pi]]))
            pi += 1
        lines.append(text)
    while pi < len(pending):
        lines.extend(_marker_labels([pending[pi]]))
        pi += 1
    return lines


def _suspect_prefix(suspect: Any) -> str:
    """``suspect`` → reader warning label, or ``""`` when not suspect.

    ``"garble"`` → ``"⚠ epäselvä "`` and ``"number"`` → ``"⚠ luku "``;
    any other (future) suspect value renders as ``"⚠ {value} "`` rather
    than crashing or silently dropping the warning.
    """
    if suspect is None:
        return ""
    label = _SUSPECT_LABELS.get(suspect, f"⚠ {suspect}")
    return f"{label} "


def _clock_time(seconds: float) -> str:
    """Floor ``seconds`` to ``[hh:mm:ss]`` — handles hours, no millis."""
    total = int(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _clock_or_none(seconds: Any) -> str | None:
    """``_clock_time`` for a valid finite number of seconds, else ``None``.

    A missing/``None``/non-numeric ``start`` yields ``None`` (the timestamp
    is omitted) rather than a placeholder ``[00:00:00]``.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    value = float(seconds)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return _clock_time(value)


def _paragraph_timings(
    paragraphs: list[dict[str, Any]],
) -> tuple[dict[str, float], float]:
    """``(speaker → spoken seconds, total spoken seconds)`` from paragraphs.

    ``total`` is the sum of every timed paragraph's ``(end - start)``;
    a paragraph without a valid ``end`` (or with ``end <= start``) counts
    0 seconds. Only paragraphs with a ``speaker`` key contribute to the
    per-speaker totals, so a speakerless transcript never gets a
    fabricated share.
    """
    per_speaker: dict[str, float] = {}
    total = 0.0
    for para in paragraphs:
        start = para.get("start")
        end = para.get("end")
        duration = 0.0
        if (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
        ):
            duration = max(0.0, float(end) - float(start))
        if duration <= 0.0:
            continue
        total += duration
        speaker = para.get("speaker")
        if speaker:
            per_speaker[speaker] = per_speaker.get(speaker, 0.0) + duration
    return per_speaker, total


def _normalize_language(language: str) -> str:
    """Coerce ``language`` to ``"fi"`` or ``"en"`` (default ``"fi"``)."""
    return "en" if str(language).strip().lower() == "en" else "fi"


def _render_header(transcript: dict[str, Any], lang: str) -> list[str]:
    """The M6 reader header block, rendered above the ``# {title}`` line.

    Each line is added only when its input is present, so the header
    degrades gracefully: no date → no date line, no duration → no
    duration line, no glossary source → no provenance line, no
    diarized/timed paragraphs → no speaker legend. An empty block (none
    of the inputs present) renders as ``[]`` — the document then starts
    at the title, exactly as before M6.
    """
    lines: list[str] = []

    date = str(transcript.get("date", "")).strip()
    if date:
        lines.append(f"_{date}_")

    duration_s = transcript.get("duration_s")
    if (
        isinstance(duration_s, (int, float))
        and not isinstance(duration_s, bool)
        and duration_s >= 0
    ):
        label = {"fi": "Kesto", "en": "Duration"}[lang]
        lines.append(f"{label}: {_clock_time(float(duration_s))}")

    paragraphs = transcript.get("paragraphs")
    # Parts count: a multi-part group carries `part_markers`; a single
    # file has at most one implicit part. The line is omitted for the
    # common single-file case (N <= 1) to keep the header quiet.
    markers = _part_markers(transcript)
    n_parts = len(markers) if markers else 0
    if n_parts > 1:
        label = {"fi": "Osia", "en": "Parts"}[lang]
        lines.append(f"{label}: {n_parts}")

    if isinstance(paragraphs, list) and paragraphs:
        per_speaker, total = _paragraph_timings(paragraphs)
        if per_speaker:
            # Honest talk share per speaker; a legend is only rendered when
            # at least one paragraph carries a speaker key (no fabricated
            # "S1: 100%" for a dictation-style transcript).
            legend: list[str] = []
            for speaker in sorted(per_speaker, key=lambda s: (-per_speaker[s], s)):
                share = (per_speaker[speaker] / total * 100) if total > 0 else 0.0
                legend.append(f"{speaker} ({share:.0f}%)")
            label = {"fi": "Keskustelijat", "en": "Speakers"}[lang]
            lines.append(f"{label}: {', '.join(legend)}")

    glossary_source = str(transcript.get("glossary_source", "")).strip()
    if glossary_source:
        label = {"fi": "Sanasto", "en": "Glossary"}[lang]
        lines.append(f"{label}: {glossary_source}")

    return lines


def format_md(transcript: dict[str, Any], language: str = "fi") -> str:
    """Render the transcript (+ optional ``notes``) as a Markdown document.

    ``language`` selects the section-heading language (``"fi"`` default,
    ``"en"``); it is threaded from the run dict / batch layer rather than
    read from a global constant.
    """
    lang = _normalize_language(language)
    notes = transcript.get("notes") or {}
    lines: list[str] = []

    header = _render_header(transcript, lang)
    if header:
        lines.extend(header)
        lines.append("")

    title = str(notes.get("title", "")).strip() or "Transcript"
    lines.append(f"# {title}")
    lines.append("")

    summary = str(notes.get("summary", "")).strip()
    if summary:
        lines.append({"fi": "## Tiivistelmä", "en": "## Summary"}[lang])
        lines.append("")
        lines.append(summary)
        lines.append("")

    key_points = [
        str(p).strip() for p in notes.get("key_points") or [] if str(p).strip()
    ]
    if key_points:
        lines.append({"fi": "## Keskeisiä asioita", "en": "## Key points"}[lang])
        lines.append("")
        lines.extend(f"- {point}" for point in key_points)
        lines.append("")

    action_items = [
        str(item).strip()
        for item in notes.get("action_items") or []
        if str(item).strip()
    ]
    if action_items:
        lines.append({"fi": "## Toimet", "en": "## Action items"}[lang])
        lines.append("")
        lines.extend(f"- [ ] {item}" for item in action_items)
        lines.append("")

    lines.append({"fi": "## Ääniseloste", "en": "## Transcript"}[lang])
    lines.append("")
    markers = _part_markers(transcript)
    paragraphs = transcript.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        blocks: list[tuple[float, str]] = []
        for para in paragraphs:
            body = str(para.get("text", "")).strip()
            if not body:
                continue
            speaker = para.get("speaker")
            prefix = f"[{speaker}] " if speaker else ""
            suspect = para.get("suspect")
            if suspect is not None:
                # Low recognizer confidence: warn the reader with a labelled
                # marker (garble → epäselvä, number → luku) rather than the
                # old bare "⚠ " prefix, so the reader knows *why* the text is
                # suspect instead of silently shipping likely garble as fact.
                prefix = _suspect_prefix(suspect) + prefix
            stamp = _clock_or_none(para.get("start"))
            if stamp:
                prefix = f"{stamp} {prefix}"
            start = para.get("start")
            if isinstance(start, bool) or not isinstance(start, (int, float)):
                start = 0.0
            blocks.append((float(start), prefix + body))
        rendered = _render_blocks(blocks, markers)
        if rendered:
            lines.append("\n\n".join(rendered))
    else:
        # No timestamped paragraph structure: markers have no offset to
        # anchor to, so render any present markers first (the first part's
        # marker, offset 0.0, is the only meaningful position) then the
        # bare text.
        body = str(transcript.get("text", "")).strip()
        marker_lines = _marker_labels(markers)
        parts = marker_lines + ([body] if body else [])
        if parts:
            lines.append("\n\n".join(parts))

    # M6 (issue #75): the end-of-run quality report, embedded as a
    # ``<details>`` block after the transcript section. The string is
    # computed BEFORE this block (CLI/batch layer, before the warnings
    # pop) so a report failure degrades to an omitted block (fail-open,
    # invariant #5) — the transcript document always renders. Absent key
    # → no block, no behaviour change for pre-M6 run dicts.
    report = transcript.get("quality_report")
    if isinstance(report, str) and report.strip():
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Laadunseuranta</summary>")
        lines.append("")
        lines.append(report.strip())
        lines.append("")
        lines.append("</details>")

    lines.append("")

    return "\n".join(lines)
