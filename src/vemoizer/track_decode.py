"""Per-track meeting decode (issue #207).

A two-track call records the owner on the mic stem and the remote side on
the system stem. Decoding the mix makes one language decision per window
for both speakers, so a Finnish owner with an English counterpart loses
one of the two (invariant #3: language is a property of a span, not of a
recording). Each stem is decoded on its own; fail-open, any problem
yields ``None`` and the caller decodes the mix.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .progress import ProgressDisplay
from .tracks import (
    TrackPaths,
    ingest_stems,
    owner_speech_segments,
    samples_to_seconds,
)
from .vad import load_model as load_vad_model
from .vad import vad_segments
from .whisper_transcriber import decode_meeting
from .whisper_windows import normalize_lost_windows

logger = logging.getLogger(__name__)

#: Silence padded around a kept span so word boundaries are not clipped.
MASK_PAD_S = 0.25

#: Sort key of a slice record (see ``slice_records_from_words``).
_SLICE_START_KEY = "start_s"

#: One "lang n/m" item of a ``language_summary`` string (see
#: ``whisper_windows.process_window_raws``, the producer).
_SUMMARY_ITEM = re.compile(r"(\S+) (\d+)/(\d+)")


def mask_outside(
    samples: np.ndarray,
    spans_s: Sequence[Mapping[str, float]],
    pad_s: float = MASK_PAD_S,
) -> np.ndarray:
    """Copy of *samples* with everything outside *spans_s* silenced.

    *spans_s* are ``{"start", "end"}`` second spans, widened by *pad_s* on
    both sides and clamped to the recording. The mic stem is masked to the
    owner's own speech so leakage cannot feed its language detection.
    """
    masked = np.zeros_like(samples)
    total = len(samples)
    for span in spans_s:
        lo = max(0, round((float(span["start"]) - pad_s) * SAMPLE_RATE))
        hi = min(total, round((float(span["end"]) + pad_s) * SAMPLE_RATE))
        if hi > lo:
            masked[lo:hi] = samples[lo:hi]
    return masked


def _sorted_sides(
    sides: Sequence[tuple[Sequence[Mapping[str, Any]], str | None]],
    key: str,
) -> list[dict[str, Any]]:
    """Copies of both sides' records on one timeline, tagged with their track.

    The ``track`` key (``me``/``them``, matching :class:`TrackPaths`) lets
    the assembly attribute words and segments per track instead of by
    time alone (issue #207); ``None`` adds no key.
    """
    records: list[dict[str, Any]] = []
    for side, track in sides:
        for record in side:
            copy = dict(record)
            if track is not None:
                copy["track"] = track
            records.append(copy)
    return sorted(records, key=lambda r: float(r[key]))


def _parse_summary(value: str) -> tuple[dict[str, int], int] | None:
    """Parse a ``"fi 29/30, en 1/30"`` summary into (per-language counts, total).

    ``None`` if any item does not have the producer's exact shape or the
    per-item totals disagree.
    """
    counts: dict[str, int] = {}
    total: int | None = None
    for item in value.split(","):
        match = _SUMMARY_ITEM.fullmatch(item.strip())
        if match is None:
            return None
        lang, count, denom = match.group(1), int(match.group(2)), int(match.group(3))
        if total is None:
            total = denom
        elif denom != total:
            return None
        counts[lang] = counts.get(lang, 0) + count
    return (counts, total) if total is not None else None


def _format_summary(counts: Mapping[str, int], total: int) -> str:
    """Format counts in the producer's shape.

    Sort is a stable one on count desc only: ties keep the existing order,
    i.e. the order :func:`whisper_windows.process_window_raws` put the
    items in (count desc, then language) as the summaries were read.
    """
    return ", ".join(
        f"{lang} {count}/{total}"
        for lang, count in sorted(counts.items(), key=lambda item: -item[1])
    )


def _combine_summaries(*sides: Mapping[str, Any] | None) -> str | None:
    """Add up the per-language window summaries of both sides, if either has one.

    ``decode_meeting`` reports ``language_summary`` as a string like
    ``"fi 29/30, en 1/30"`` (whisper_windows is the producer); the two
    sides' counts and totals are added and the result is formatted in the
    producer's shape. A side whose summary does not parse is dropped; if
    neither parses there is no summary.
    """
    texts: list[str] = []
    parsed: list[tuple[dict[str, int], int] | None] = []
    for side in sides:
        summary = side.get("language_summary") if side is not None else None
        if not summary:
            continue
        text = str(summary)
        texts.append(text)
        parsed.append(_parse_summary(text))
    both = [entry for entry in parsed if entry is not None]
    if len(texts) == 2 and len(both) == 2:
        counts: dict[str, int] = {}
        total = 0
        for side_counts, side_total in both:
            total += side_total
            for lang, count in side_counts.items():
                counts[lang] = counts.get(lang, 0) + count
        return _format_summary(counts, total)
    for text, entry in zip(texts, parsed, strict=True):
        if entry is not None:
            return text
    return None


def _tag_track(result: dict[str, Any], track: str) -> dict[str, Any]:
    """Copy of a one-sided decode result with its records tagged as *track*.

    A stem with no speech is legitimately skipped, but the surviving side
    must still carry its ``track`` key so speaker labelling attributes
    turns per track instead of falling back to all of them.
    """
    tagged = dict(result)
    for key in ("words", "segments"):
        tagged[key] = [dict(record, track=track) for record in result.get(key) or []]
    return tagged


def merge_track_results(
    me: dict[str, Any] | None,
    them: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Fuse two per-stem decode results into one decode-A-shaped result.

    Words, segments and slice records go on one timeline; the text follows
    the sorted segments; the language is the one of the side that spoke
    more; lost windows (issue #152) and the language summary are carried
    over from both sides. A ``None`` side had no speech (a failed stem
    decode is caught earlier, by :func:`decode_tracks`); both ``None`` is
    failure.
    """
    if me is None:
        return _tag_track(them, "them") if them is not None else None
    if them is None:
        return _tag_track(me, "me")
    segments = _sorted_sides(
        [(me.get("segments") or [], "me"), (them.get("segments") or [], "them")],
        "start",
    )
    merged: dict[str, Any] = {
        "words": _sorted_sides(
            [(me.get("words") or [], "me"), (them.get("words") or [], "them")],
            "start",
        ),
        "segments": segments,
        "slices": _sorted_sides(
            [(me.get("slices") or [], None), (them.get("slices") or [], None)],
            _SLICE_START_KEY,
        ),
        "text": " ".join(str(s.get("text", "")).strip() for s in segments).strip(),
        "language": _majority_language(me, them),
    }
    summary = _combine_summaries(me, them)
    if summary is not None:
        merged["language_summary"] = summary
    lost = normalize_lost_windows(me.get("lost_windows")) + normalize_lost_windows(
        them.get("lost_windows")
    )
    if lost:
        merged["lost_windows"] = sorted(lost, key=lambda window: window[0])
    return merged


def _majority_language(me: Mapping[str, Any], them: Mapping[str, Any]) -> str | None:
    """Language of whichever side produced more words."""
    sides = sorted((me, them), key=lambda s: len(s.get("words") or []), reverse=True)
    return sides[0].get("language")


def _stem_slices(
    stem: np.ndarray, spans_s: Sequence[Mapping[str, float]]
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, int]]]:
    """``(offset, audio)`` slices and ``(start, end)`` sample spans for one stem."""
    spans = [
        (
            max(0, round(float(span["start"]) * SAMPLE_RATE)),
            min(len(stem), round(float(span["end"]) * SAMPLE_RATE)),
        )
        for span in spans_s
    ]
    spans = [(lo, hi) for lo, hi in spans if hi > lo]
    return [(lo, stem[lo:hi]) for lo, hi in spans], spans


def decode_tracks(
    tracks: TrackPaths,
    mix_len: int,
    *,
    initial_prompt: str | None,
    language: str | None,
    display: ProgressDisplay | None = None,
) -> dict[str, Any] | None:
    """Decode the mic and the system stem separately; ``None`` to fail open.

    The mic stem is VAD'd, gated down to the owner's own speech (leakage
    removed) and masked, then both stems are sliced and decoded the way
    the mix would be, one ``decode_meeting`` call per stem with speech.
    """
    try:
        kwargs: dict[str, Any] = {
            "initial_prompt": initial_prompt,
            "language": language,
        }
        if display is not None:
            kwargs["display"] = display
        mic, them = ingest_stems(tracks, mix_len)
        vad_model = load_vad_model()
        mic_spans = owner_speech_segments(_speech_seconds(mic, vad_model), mic, them)
        them_spans = _speech_seconds(them, vad_model)
        mic_result = (
            _decode_stem(mask_outside(mic, mic_spans), mic_spans, kwargs)
            if mic_spans
            else None
        )
        them_result = _decode_stem(them, them_spans, kwargs) if them_spans else None
        for side, spans, result in (
            ("me", mic_spans, mic_result),
            ("them", them_spans, them_result),
        ):
            if spans and result is None:
                logger.warning(
                    "meeting: the %s stem had speech but its decode failed, "
                    "decoding the mix",
                    side,
                )
                return None
        decoded = sum(r is not None for r in (mic_result, them_result))
        logger.info("meeting: decoded the call per track (%d stems)", decoded)
        return merge_track_results(mic_result, them_result)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("per-track meeting decode failed, decoding the mix: %s", e)
        return None


def _decode_stem(
    stem: np.ndarray,
    spans_s: Sequence[Mapping[str, float]],
    kwargs: Mapping[str, Any],
) -> dict[str, Any] | None:
    """One ``decode_meeting`` over one stem's own speech spans."""
    slices, vad_slices = _stem_slices(stem, spans_s)
    if not slices:
        return None
    return decode_meeting(stem, slices, vad_slices=vad_slices, **dict(kwargs))


def _speech_seconds(stem: np.ndarray, model: Any) -> list[Mapping[str, float]]:
    """VAD speech spans of one stem as ``{"start", "end"}`` seconds."""
    return [
        {
            "start": samples_to_seconds(seg.start),
            "end": samples_to_seconds(seg.end),
        }
        for seg in vad_segments(stem, model)
    ]
