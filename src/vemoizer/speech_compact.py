"""Compacting the owner's speech spans into one dense decode buffer (issue #209).

The mic stem of a two-track call is gated to the owner's own speech before
its decode. That gate runs per 30 ms frame, so one 6 s sentence comes back as
~11 fragments with 0.1-0.8 s holes, and the stem keeps its full length while
the decode runs on fixed 30 s windows. Both halves of that hurt Whisper:
speech with holes decoded to garbled words, and a window that is mostly
digital silence with speech starting late (plus the glossary
``initial_prompt``) decoded to a repetition loop that swallowed the real
speech.

The fix is to decode a COMPACTED buffer: :func:`bridge_spans` rejoins the
fragments, :func:`compact_spans` concatenates them back to back with a short
silence between parts, and :func:`remap_result` puts the decode results of
that buffer back on the original timeline. Nothing here touches a model or
audio I/O — plain numpy and stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .whisper_windows import normalize_lost_windows

#: Consecutive kept spans closer than this are joined before compacting.
#: The owner gate keeps the 30 ms frames whose level says the owner spoke,
#: so the holes inside a sentence are dropped consonants and breaths
#: (0.1-0.8 s in practice), while a leakage stretch — the remote talking and
#: leaking into the mic — is longer than a second. 1.0 s sits between the
#: two: bridging a real leakage stretch would hand the decode remote speech.
BRIDGE_GAP_S = 1.0

#: Seconds of recording padded around a kept span before compacting, so
#: word boundaries at the span edges are not clipped.
COMPACT_PAD_S = 0.25

#: Seconds of silence inserted between two compacted parts. Speech with no
#: pause at all between turns reads as one continuous utterance to Whisper;
#: 0.5 s is a normal sentence pause.
COMPACT_GAP_S = 0.5

#: One compacted part: ``(compact_start, orig_start, length)``, samples.
Part = tuple[int, int, int]

#: Segment keys rebuilt from the run's words instead of mapped.
_SEGMENT_REBUILT_KEYS = ("start", "end", "text", "words")


@dataclass(frozen=True)
class CompactMap:
    """Where each part of a compacted buffer came from (all in samples).

    ``parts`` is ordered by ``compact_start`` and covers ``[0, total)`` of
    the compact buffer; ``gap`` is the silence length between two parts.
    """

    parts: tuple[Part, ...] = ()
    gap: int = 0


def bridge_spans(
    spans_s: Sequence[Mapping[str, float]],
    max_gap_s: float = BRIDGE_GAP_S,
) -> list[dict[str, float]]:
    """Join ``{"start", "end"}`` second spans separated by less than *max_gap_s*.

    The frame-level owner gate fragments a sentence into runs; decoding each
    run alone is what garbles the words, so the fragments are bridged first.
    """
    bridged: list[dict[str, float]] = []
    for span in sorted(spans_s, key=lambda s: float(s["start"])):
        start, end = float(span["start"]), float(span["end"])
        if bridged and start - float(bridged[-1]["end"]) < max_gap_s:
            bridged[-1]["end"] = max(bridged[-1]["end"], end)
        else:
            bridged.append({"start": start, "end": end})
    return bridged


def compact_spans(
    samples: np.ndarray,
    spans_s: Sequence[Mapping[str, float]],
    pad_s: float = COMPACT_PAD_S,
    gap_s: float = COMPACT_GAP_S,
) -> tuple[np.ndarray, CompactMap]:
    """Concatenate *spans_s* of *samples* into one dense buffer.

    Each span is widened by *pad_s* on both sides, clamped to the recording,
    and merged with any padded span it now touches; the parts are joined with
    *gap_s* of silence between them (none after the last one). Returns the
    compact buffer (same dtype as *samples*) and the :class:`CompactMap` that
    maps its times back. No spans means an empty buffer.
    """
    padded: list[tuple[int, int]] = []
    total = len(samples)
    for span in sorted(spans_s, key=lambda s: float(s["start"])):
        lo = max(0, round((float(span["start"]) - pad_s) * SAMPLE_RATE))
        hi = min(total, round((float(span["end"]) + pad_s) * SAMPLE_RATE))
        if hi <= lo:
            continue
        if padded and lo <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], hi))
        else:
            padded.append((lo, hi))
    gap = round(gap_s * SAMPLE_RATE)
    if not padded:
        return np.zeros(0, dtype=samples.dtype), CompactMap(gap=gap)
    parts: list[Part] = []
    chunks: list[np.ndarray] = []
    cursor = 0
    for lo, hi in padded:
        if chunks and gap > 0:
            chunks.append(np.zeros(gap, dtype=samples.dtype))
            cursor += gap
        parts.append((cursor, lo, hi - lo))
        chunks.append(samples[lo:hi])
        cursor += hi - lo
    return np.concatenate(chunks), CompactMap(parts=tuple(parts), gap=gap)


def _locate(t_s: float, cmap: CompactMap) -> tuple[int, float]:
    """``(part index, original seconds)`` of a compact-buffer time.

    A time inside a part maps linearly. Whisper's word timestamps drift a
    little, so a time in a gap goes to the nearer side: the first half to
    the end of the part before, the second half to the start of the part
    after. Before the first part clamps to its start, past the end to the
    end of the last part.
    """
    t = float(t_s) * SAMPLE_RATE
    previous: tuple[int, int, int] | None = None
    for index, (compact_start, orig_start, length) in enumerate(cmap.parts):
        if t <= compact_start + length:
            if t >= compact_start or previous is None:
                offset = max(0.0, t - compact_start)
                return index, (orig_start + offset) / SAMPLE_RATE
            before_index, before_end_c, before_end_o = previous
            if t - before_end_c < (compact_start - before_end_c) / 2:
                return before_index, before_end_o / SAMPLE_RATE
            return index, orig_start / SAMPLE_RATE
        previous = (index, compact_start + length, orig_start + length)
    _, last_start, last_length = cmap.parts[-1]
    return len(cmap.parts) - 1, (last_start + last_length) / SAMPLE_RATE


def part_of(t_s: float, cmap: CompactMap) -> int:
    """Index of the part a compact-buffer time belongs to; -1 if there is none."""
    return _locate(t_s, cmap)[0] if cmap.parts else -1


def to_original(t_s: float, cmap: CompactMap) -> float:
    """Compact-buffer seconds to original-recording seconds (see :func:`_locate`)."""
    return _locate(t_s, cmap)[1] if cmap.parts else 0.0


def _part_end_s(index: int, cmap: CompactMap) -> float:
    """Original-recording seconds at the end of part *index*."""
    _, orig_start, length = cmap.parts[index]
    return (orig_start + length) / SAMPLE_RATE


def _remap_span(start_c: float, end_c: float, cmap: CompactMap) -> tuple[float, float]:
    """A compact ``(start, end)`` on the original timeline, kept inside one part.

    The end never crosses the end of the part the start falls in: the parts
    are far apart in the recording, so nothing decoded can straddle two.
    """
    index, start = _locate(start_c, cmap)
    end = min(to_original(end_c, cmap), _part_end_s(index, cmap))
    return start, max(end, start)


def _remap_word(word: Mapping[str, Any], cmap: CompactMap) -> dict[str, Any]:
    """Copy of a word with its times on the original timeline."""
    remapped = dict(word)
    remapped["start"], remapped["end"] = _remap_span(
        float(word.get("start", 0.0)), float(word.get("end", 0.0)), cmap
    )
    return remapped


def _runs_by_part(
    entries: Sequence[tuple[float, dict[str, Any]]], cmap: CompactMap
) -> list[list[tuple[float, dict[str, Any]]]]:
    """Split ``[(compact_start, word)]`` entries into runs of one part."""
    runs: list[list[tuple[float, dict[str, Any]]]] = []
    previous = -2
    for start, word in entries:
        index = part_of(start, cmap)
        if index != previous or not runs:
            runs.append([])
            previous = index
        runs[-1].append((start, word))
    return runs


def _nearest_segment(start: float, segments: Sequence[Mapping[str, Any]]) -> int:
    """Index of the first segment containing *start*, else the nearest one.

    Whisper's word and segment timestamps come from different mechanisms, so
    a word can be stamped just outside its segment; it must still land in
    one, or it drops out of the transcript text.
    """
    best, best_distance = 0, float("inf")
    for index, segment in enumerate(segments):
        lo = float(segment.get("start", 0.0))
        hi = float(segment.get("end", 0.0))
        distance = max(lo - start, start - hi, 0.0)
        if distance < best_distance:
            best, best_distance = index, distance
            if distance == 0.0:
                break
    return best


def _remap_segments(
    segments: Sequence[Mapping[str, Any]],
    entries: Sequence[tuple[float, dict[str, Any]]],
    cmap: CompactMap,
) -> list[dict[str, Any]]:
    """Segment records on the original timeline, one segment per part.

    *entries* is ``[(compact_start, mapped_word)]`` in decode order. A
    compacted buffer joins speech from spans that are far apart in the
    recording, so one segment's text can cover two of them: it is split into
    one segment per part, its words are its own, and its times and text
    follow them. Words are attributed to a segment by their compact start,
    which is also what decides the split.
    """
    assigned: list[list[tuple[float, dict[str, Any]]]] = [[] for _ in segments]
    if segments:
        for start, word in entries:
            assigned[_nearest_segment(start, segments)].append((start, word))
    remapped: list[dict[str, Any]] = []
    for segment, segment_entries in zip(segments, assigned, strict=True):
        if not segment_entries:
            without_words = dict(segment)
            without_words["start"], without_words["end"] = _remap_span(
                float(segment.get("start", 0.0)), float(segment.get("end", 0.0)), cmap
            )
            remapped.append(without_words)
            continue
        had_words = "words" in segment
        runs = _runs_by_part(segment_entries, cmap)
        for run in runs:
            run_words = [dict(word) for _start, word in run]
            entry = {
                key: value
                for key, value in segment.items()
                if key not in _SEGMENT_REBUILT_KEYS
            }
            entry["start"] = float(run_words[0]["start"])
            entry["end"] = float(run_words[-1]["end"])
            # A segment inside one part keeps Whisper's own text (word joins
            # lose its spacing, e.g. "AI-agentin" -> "AI -agentin").
            entry["text"] = (
                str(segment.get("text", "")).strip()
                if len(runs) == 1
                else _join_words(run_words)
            )
            if had_words:
                entry["words"] = run_words
            remapped.append(entry)
    return sorted(remapped, key=lambda s: float(s["start"]))


def _join_words(words: Sequence[Mapping[str, Any]]) -> str:
    """Segment text from its words; decode words carry no leading space."""
    return " ".join(
        text for text in (str(w.get("word", "")).strip() for w in words) if text
    )


def _remap_slices(records: Sequence[Mapping[str, Any]], cmap: CompactMap) -> list[Any]:
    """Slice records with their times (and nested words) on the original timeline."""
    remapped: list[Any] = []
    for record in records:
        item = dict(record)
        for key in ("start_s", "end_s"):
            if key in item:
                item[key] = to_original(float(item[key]), cmap)
        if item.get("words") is not None:
            item["words"] = [_remap_word(w, cmap) for w in item["words"]]
        remapped.append(item)
    return remapped


def remap_result(result: Mapping[str, Any], cmap: CompactMap) -> dict[str, Any]:
    """New result dict with every compact-buffer time on the original timeline.

    *result* is shaped like :func:`whisper_transcriber.decode_meeting`
    returns (``words``, ``segments``, ``slices``, optionally ``lost_windows``);
    keys this module does not know are copied unchanged. The input is never
    mutated.
    """
    remapped = dict(result)
    words = list(result.get("words") or [])
    mapped_words = [_remap_word(w, cmap) for w in words]
    remapped["words"] = mapped_words
    entries = [
        (float(word.get("start", 0.0)), mapped)
        for word, mapped in zip(words, mapped_words, strict=True)
    ]
    if result.get("segments") is not None:
        remapped["segments"] = _remap_segments(result["segments"], entries, cmap)
    if result.get("slices") is not None:
        remapped["slices"] = _remap_slices(result["slices"], cmap)
    if "lost_windows" in result:
        remapped["lost_windows"] = [
            _remap_span(start, end, cmap)
            for start, end in normalize_lost_windows(result["lost_windows"])
        ]
    return remapped
