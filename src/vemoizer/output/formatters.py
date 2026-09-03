"""Transcript -> str formatters for txt, json, srt, vtt (issue #10, task A).

Pure string builders over the ``TranscriptionResult`` contract documented on
``vemoizer.transcriber.TranscriptionResult``:

- ``text`` (required): full transcript string.
- ``segments`` (optional): ``[{start, end, text}]`` in time order.
- ``words`` (optional): ``[{word, start, end}]`` in time order.
- ``language`` (optional): ISO 639-1 tag, present only when reported.

The four formats differ in exactly the ways their downstream consumers
expect:

- **txt**: plain text — segments joined by newlines (falls back to the
  single ``text`` field when there are no segments). A segment carrying a
  ``speaker`` label is prefixed with ``[SPEAKER]``; unlabelled segments
  are emitted bare, so output without diarization is byte-identical to
  the pre-diarization rendering.
- **json**: the transcript dict itself, re-serialised with UTF-8 and a
  trailing newline. Optional fields (``language``) are omitted when absent
  so the output mirrors what the backend actually reported.
- **srt**: SubRip — ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` (comma), 1-based cue
  index, blank-line separated blocks.
- **vtt**: WebVTT — ``WEBVTT`` header, ``HH:MM:SS.mmm --> HH:MM:SS.mmm``
  (dot), 0-based cue index, blank-line separated cues.

The SRT-vs-VTT millisecond separator is the single highest-risk formatting
difference between the two subtitle formats, so it is captured by golden
fixture files in ``tests/fixtures/output/`` and by explicit unit tests in
``tests/test_formatters.py``.
"""

from __future__ import annotations

import json
from typing import Any

#: Every output format the CLI accepts, in canonical order.
OUTPUT_FORMATS: tuple[str, ...] = ("txt", "json", "srt", "vtt")

#: File extension per format (no leading dot required; the CLI appends it
#: to the output base path).
FORMAT_EXTENSIONS: dict[str, str] = {
    "txt": ".txt",
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
}


def _timestamp(seconds: float, separator: str) -> str:
    """Format seconds as ``HH:MM:SS{separator}mmm``.

    ``separator`` is ``","`` for SRT and ``"."`` for VTT — the single
    difference between the two subtitle timestamp shapes. Milliseconds are
    truncated (not rounded) to keep SRT/VTT playback at the segment's start
    boundary; a segment that ends at 2.000 ms must not bleed into the
    next cue's 0.000 ms.
    """
    ms = int(seconds * 1000)
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def srt_timestamp(seconds: float) -> str:
    """SRT timestamp: ``HH:MM:SS,mmm`` (comma)."""
    return _timestamp(seconds, ",")


def vtt_timestamp(seconds: float) -> str:
    """VTT timestamp: ``HH:MM:SS.mmm`` (dot)."""
    return _timestamp(seconds, ".")


def _speaker_label(segment: dict[str, Any]) -> str:
    """The speaker label prefix ``[LABEL] `` for a segment, or ``""``."""
    label = segment.get("speaker")
    if label is None:
        return ""
    label = str(label).strip()
    return f"[{label}] " if label else ""


def _segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Segments from the transcript, or ``[]`` when absent."""
    segs = transcript.get("segments")
    return segs if isinstance(segs, list) else []


def _words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Words from the transcript, or ``[]`` when absent."""
    words = transcript.get("words")
    return words if isinstance(words, list) else []


def _caption_entries(
    transcript: dict[str, Any],
) -> list[tuple[float, float, str]]:
    """(start, end, text) tuples for srt/vtt cues.

    Segments are preferred — they are the sentence-level chunks the
    pipeline already produces and they read as proper subtitle cues. When a
    decode produced no segments (some backends only emit ``text`` +
    ``words``), fall back to one cue per word so the subtitles still line
    up with the audio. An empty transcript yields no cues.
    """
    entries: list[tuple[float, float, str]] = []
    segments = _segments(transcript)
    if segments:
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            entries.append((start, end, text))
        return entries
    for word in _words(transcript):
        start = float(word.get("start", 0.0))
        end = float(word.get("end", word.get("start", 0.0)))
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        entries.append((start, end, text))
    return entries


def format_txt(transcript: dict[str, Any]) -> str:
    """Plain text: paragraphs, else segments, else the ``text`` field.

    Paragraph blocks (from the readability stage) are blank-line separated
    and carry a speaker prefix when known — the shape a human actually
    reads. Without paragraphs the pre-existing renderings are unchanged.
    """
    paragraphs = transcript.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        blocks: list[str] = []
        for para in paragraphs:
            body = str(para.get("text", "")).strip()
            if not body:
                continue
            blocks.append(_speaker_label(para) + body)
        if blocks:
            return "\n\n".join(blocks) + "\n"
    segments = _segments(transcript)
    if segments:
        lines: list[str] = []
        for seg in segments:
            line = str(seg.get("text", "")).strip()
            if not line:
                continue
            lines.append(_speaker_label(seg) + line)
        if not lines:
            return ""
        return "\n".join(lines) + "\n"
    text = transcript.get("text", "")
    return text if not text else text + "\n"


def format_json(transcript: dict[str, Any]) -> str:
    """JSON: the transcript dict, UTF-8, indented, trailing newline.

    Optional fields (``language``, ``words``, ``segments``) are included
    only when present on the input, so the output mirrors what the backend
    actually reported rather than inventing empty collections.
    """
    out: dict[str, Any] = {"text": transcript.get("text", "")}
    if transcript.get("language") is not None:
        out["language"] = transcript["language"]
    if "speakers" in transcript and transcript["speakers"]:
        out["speakers"] = transcript["speakers"]
    if "paragraphs" in transcript and transcript["paragraphs"]:
        out["paragraphs"] = transcript["paragraphs"]
    if "segments" in transcript and transcript["segments"]:
        out["segments"] = transcript["segments"]
    if "words" in transcript and transcript["words"]:
        out["words"] = transcript["words"]
    return json.dumps(out, ensure_ascii=False, indent=2) + "\n"


def format_srt(transcript: dict[str, Any]) -> str:
    """SubRip: 1-based index, ``HH:MM:SS,mmm`` timestamps, blank-line blocks.

    Cues from speaker-labelled segments are prefixed with ``[SPEAKER]``;
    the word-fallback path has no speaker data and is emitted bare.
    """
    entries: list[tuple[float, float, str]] = []
    for seg in _segments(transcript):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        entries.append((start, end, _speaker_label(seg) + text))
    if not entries:
        entries = _caption_entries(transcript)
    if not entries:
        return ""
    blocks: list[str] = []
    for index, (start, end, text) in enumerate(entries, start=1):
        stamp = f"{srt_timestamp(start)} --> {srt_timestamp(end)}"
        blocks.append(f"{index}\n{stamp}\n{text}")
    return "\n\n".join(blocks) + "\n"


def format_vtt(transcript: dict[str, Any]) -> str:
    """WebVTT: ``WEBVTT`` header, 0-based index, ``HH:MM:SS.mmm`` timestamps.

    Cues from speaker-labelled segments are prefixed with ``[SPEAKER]``;
    the word-fallback path has no speaker data and is emitted bare.
    """
    entries: list[tuple[float, float, str]] = []
    for seg in _segments(transcript):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        entries.append((start, end, _speaker_label(seg) + text))
    if not entries:
        entries = _caption_entries(transcript)
    if not entries:
        return "WEBVTT\n"
    cues: list[str] = ["WEBVTT", ""]
    for index, (start, end, text) in enumerate(entries):
        cues.append(f"{index}\n{vtt_timestamp(start)} --> {vtt_timestamp(end)}\n{text}")
        cues.append("")
    return "\n".join(cues)


def format_transcript(transcript: dict[str, Any], format: str) -> str:
    """Render a transcript in the named output format.

    Args:
        transcript: A ``TranscriptionResult`` dict.
        format: One of ``txt`` / ``json`` / ``srt`` / ``vtt``.

    Raises:
        ValueError: when ``format`` is not a known output format.
    """
    dispatch = {
        "txt": format_txt,
        "json": format_json,
        "srt": format_srt,
        "vtt": format_vtt,
    }
    if format not in dispatch:
        known = ", ".join(OUTPUT_FORMATS)
        raise ValueError(f"Unknown output format: {format!r}. Expected one of: {known}")
    return dispatch[format](transcript)
