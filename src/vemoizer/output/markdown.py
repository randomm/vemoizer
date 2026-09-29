"""Markdown notes output: title, summary, action items, transcript (issue #56).

The ``md`` format is the human-facing deliverable the spec promises
("LLM cleanup / summary -> text + Markdown"): a note you can read, not a
subtitle file. Sections render only when the notes stage produced them —
with no notes at all the document is a clean paragraphed transcript, so
the format degrades gracefully along the LLM's fail-open path.
"""

from __future__ import annotations

from typing import Any


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
    paragraphs = transcript.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        blocks: list[str] = []
        for para in paragraphs:
            body = str(para.get("text", "")).strip()
            if not body:
                continue
            speaker = para.get("speaker")
            prefix = f"[{speaker}] " if speaker else ""
            blocks.append(prefix + body)
        lines.append("\n\n".join(blocks))
    else:
        lines.append(str(transcript.get("text", "")).strip())
    lines.append("")

    return "\n".join(lines)
