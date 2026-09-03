"""Full-coverage transcript assembly: verdict splicing + paragraphs (issue #53).

Two pure functions between adjudication and formatting:

- :func:`splice_verdicts` patches adjudicated span verdicts INTO the
  decode-A sentence segments. The formatters render segments when present,
  so without this the transcript files would collapse to only the disputed
  fragments the moment consensus activates. With zero verdicts the input
  passes through byte-identical (today's behaviour is unchanged).
- :func:`paragraphs` groups consecutive segments into paragraphs at long
  silence gaps or speaker changes, giving the txt/Markdown outputs a
  readable shape instead of one line per sentence.

Both operate on plain word/segment dicts — no models, no I/O.
"""

from __future__ import annotations

from typing import Any

#: Silence between consecutive segments that starts a new paragraph.
PARAGRAPH_GAP_S = 1.5


def splice_verdicts(
    text: str,
    words: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Patch adjudicated *verdicts* into the decode-A *sentences*.

    Each verdict is ``{start, end, text[, speaker]}`` for one disputed span
    ``[start, end)``. Words of *words* falling inside any span are dropped;
    the verdict text is inserted once, in the sentence containing the
    span's start (the anchor), even when the span crosses a sentence
    boundary. Every sentence stays present — full coverage — with its
    original timing.

    With no verdicts, returns (*text*, *sentences*) unchanged, so a run
    without disputes is byte-identical to a run without this stage.

    Returns ``(spliced_text, spliced_segments)``; the text is the segment
    texts joined with single spaces (sentence punctuation from decode A is
    preserved inside each undisputed stretch).
    """
    if not verdicts:
        return text, list(sentences)

    ordered = sorted(verdicts, key=lambda v: float(v["start"]))

    def _span_of(t: float) -> dict[str, Any] | None:
        for v in ordered:
            if float(v["start"]) <= t < float(v["end"]):
                return v
        return None

    def _anchor_sentence(v: dict[str, Any]) -> int:
        """Index of the sentence the verdict lands in (contains its start)."""
        start = float(v["start"])
        for i, s in enumerate(sentences):
            if float(s["start"]) <= start < float(s["end"]):
                return i
        # Span starts in a silence gap: anchor to the first sentence that
        # begins after it, else the last sentence.
        for i, s in enumerate(sentences):
            if float(s["start"]) >= start:
                return i
        return len(sentences) - 1

    inserts: dict[int, list[dict[str, Any]]] = {}
    for v in ordered:
        inserts.setdefault(_anchor_sentence(v), []).append(v)

    spliced: list[dict[str, Any]] = []
    for i, s in enumerate(sentences):
        s_start, s_end = float(s["start"]), float(s["end"])
        pieces: list[tuple[float, str]] = []
        for w in words:
            w_start = float(w.get("start", 0.0))
            if not (s_start <= w_start < s_end):
                continue
            if _span_of(w_start) is not None:
                continue  # replaced by a verdict
            pieces.append((w_start, str(w.get("word", ""))))
        for v in inserts.get(i, []):
            v_text = str(v.get("text", "")).strip()
            if v_text:
                pieces.append((float(v["start"]), v_text))
        pieces.sort(key=lambda p: p[0])
        new_text = " ".join(p[1] for p in pieces if p[1]).strip()
        segment: dict[str, Any] = {**s, "text": new_text}
        for v in inserts.get(i, []):
            if v.get("speaker") is not None:
                segment["speaker"] = v["speaker"]
        spliced.append(segment)

    new_full_text = " ".join(s["text"] for s in spliced if s["text"]).strip()
    return new_full_text, spliced


def paragraphs(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive *segments* into paragraph dicts.

    A new paragraph starts when the silence gap to the previous segment is
    at least :data:`PARAGRAPH_GAP_S` seconds or the speaker label changes
    (an unlabelled segment continues the current speaker rather than
    breaking the paragraph). Each paragraph is ``{start, end, text}`` plus
    ``speaker`` when its segments carried one.
    """
    paras: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_speaker: str | None = None

    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        speaker = seg.get("speaker")
        gap_break = current is not None and start - float(current["end"]) >= (
            PARAGRAPH_GAP_S
        )
        speaker_break = (
            current is not None
            and speaker is not None
            and current_speaker is not None
            and speaker != current_speaker
        )
        if current is None or gap_break or speaker_break:
            current = {"start": start, "end": end, "text": seg_text}
            if speaker is not None:
                current["speaker"] = speaker
            current_speaker = speaker
            paras.append(current)
        else:
            current["text"] = f"{current['text']} {seg_text}"
            current["end"] = end
            if speaker is not None and current_speaker is None:
                current["speaker"] = speaker
                current_speaker = speaker
    return paras
