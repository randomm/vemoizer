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
"""

from __future__ import annotations

from typing import Any


def _part_markers(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """``transcript["part_markers"]`` when present, else ``[]``.

    The sole producer is ``run_batch``, which always builds
    ``{"offset": float, "label": str}`` dicts from the typed
    ``PartOffset`` dataclass — non-dict entries (``None``, a stray
    string) cannot be rendered as markers and are dropped rather than
    crashing the render path with an ``AttributeError`` (``m.get`` on a
    non-dict).
    """
    markers = transcript.get("part_markers")
    if not isinstance(markers, list):
        return []
    return [m for m in markers if isinstance(m, dict)]


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
            label = str(pending[pi].get("label", "")).strip()
            if label:
                lines.append(label)
            pi += 1
        lines.append(text)
    while pi < len(pending):
        label = str(pending[pi].get("label", "")).strip()
        if label:
            lines.append(label)
        pi += 1
    return lines


def format_md(transcript: dict[str, Any]) -> str:
    """Render the transcript (+ optional ``notes``) as a Markdown document."""
    notes = transcript.get("notes") or {}
    lines: list[str] = []

    title = str(notes.get("title", "")).strip() or "Transcript"
    lines.append(f"# {title}")
    lines.append("")

    summary = str(notes.get("summary", "")).strip()
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")

    key_points = [
        str(p).strip() for p in notes.get("key_points") or [] if str(p).strip()
    ]
    if key_points:
        lines.append("## Key points")
        lines.append("")
        lines.extend(f"- {point}" for point in key_points)
        lines.append("")

    action_items = [
        str(item).strip()
        for item in notes.get("action_items") or []
        if str(item).strip()
    ]
    if action_items:
        lines.append("## Action items")
        lines.append("")
        lines.extend(f"- [ ] {item}" for item in action_items)
        lines.append("")

    lines.append("## Transcript")
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
            if para.get("suspect"):
                # Low recognizer confidence: warn the reader instead of
                # silently shipping likely garble as fact.
                prefix = "⚠ " + prefix
            start = float(para.get("start", 0.0))
            blocks.append((start, prefix + body))
        rendered = _render_blocks(blocks, markers)
        if rendered:
            lines.append("\n\n".join(rendered))
    else:
        # No timestamped paragraph structure: markers have no offset to
        # anchor to, so render any present markers first (the first part's
        # marker, offset 0.0, is the only meaningful position) then the
        # bare text.
        body = str(transcript.get("text", "")).strip()
        marker_lines = [
            str(m.get("label", "")).strip()
            for m in markers
            if str(m.get("label", "")).strip()
        ]
        if marker_lines and body:
            lines.append("\n\n".join(marker_lines + [body]))
        elif body:
            lines.append(body)
        elif marker_lines:
            lines.append("\n\n".join(marker_lines))
    lines.append("")

    return "\n".join(lines)
