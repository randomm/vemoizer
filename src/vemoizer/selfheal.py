"""Self-healing re-decode of hallucination walls (meeting profile).

Whisper's rolling context (``condition_on_previous_text=True``, which the
glossary prompt needs) turns one bad window into a wall: the decoded loop
re-enters the context and feeds itself. Observed live: ~50 consecutive
"Kiitos." segments erasing a presentation opening, ~95 "DCS. DCS."
segments erasing a 43-minute demo.

This stage detects such walls after the whole-file decode and re-decodes
only the VAD slices under them with conditioning OFF — each slice gets a
fresh context (and the glossary ``initial_prompt`` again), so the loop
cannot propagate. A replacement is accepted only when it is not itself
degenerate; a failed or still-looping re-decode keeps the original slice
(fail-open, invariant #5). Detection and splicing are pure functions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .textnorm import textnorm

logger = logging.getLogger(__name__)

#: A wall segment carries at most this many normalized tokens ("Kiitos."
#: is one, "DCS. DCS." is two). Real sentences are longer and never join
#: a run.
MAX_TOKENS_PER_SEGMENT = 4

#: Consecutive qualifying segments needed before a run counts as a wall.
#: Below this, short segments are ordinary backchannels.
MIN_RUN = 5

#: A run is a wall only when its whole vocabulary collapses to at most
#: this many distinct tokens — "Kiitos." x50 has one, the "DCS."/"D-css."
#: drift has three, while five varied backchannels stay above it.
MAX_DISTINCT_TOKENS = 4

#: Windows closer than this merge: one real sentence surfacing inside a
#: wall does not end the wall.
MERGE_GAP_S = 10.0

#: A slice is re-decoded when it overlaps a degenerate window by more
#: than this.
MIN_OVERLAP_S = 0.2

#: Runaway backstop: never re-decode more slices than this per file.
MAX_HEAL_SLICES = 200

RedecodeFn = Callable[[np.ndarray], dict[str, Any]]


def find_degenerate_windows(
    segments: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Time windows ``[(start, end), ...]`` covered by hallucination walls.

    A wall is a run of at least :data:`MIN_RUN` consecutive segments,
    each at most :data:`MAX_TOKENS_PER_SEGMENT` normalized tokens, whose
    union of distinct tokens is at most :data:`MAX_DISTINCT_TOKENS`.
    Windows separated by less than :data:`MERGE_GAP_S` merge.
    """
    windows: list[tuple[float, float]] = []
    run: list[dict[str, Any]] = []
    run_tokens: set[str] = set()

    def _close() -> None:
        nonlocal run, run_tokens
        if len(run) >= MIN_RUN and len(run_tokens) <= MAX_DISTINCT_TOKENS:
            windows.append((float(run[0]["start"]), float(run[-1]["end"])))
        run, run_tokens = [], set()

    for seg in segments:
        tokens = textnorm(str(seg.get("text", ""))).split()
        if run and float(seg.get("start", 0.0)) - float(run[-1]["end"]) >= (
            MERGE_GAP_S
        ):
            # A long silence ends the wall even with nothing decoded in
            # between: two distant walls are two windows, not one.
            _close()
        if 0 < len(tokens) <= MAX_TOKENS_PER_SEGMENT:
            run.append(seg)
            run_tokens.update(tokens)
        else:
            _close()
    _close()

    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start - merged[-1][1] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _shift(items: list[dict[str, Any]], offset_s: float) -> list[dict[str, Any]]:
    """Copy *items* with start/end moved onto the full-recording timeline."""
    return [
        {
            **item,
            "start": float(item.get("start", 0.0)) + offset_s,
            "end": float(item.get("end", 0.0)) + offset_s,
        }
        for item in items
    ]


def heal(
    result: dict[str, Any],
    slices: list[tuple[int, np.ndarray]],
    redecode: RedecodeFn,
) -> dict[str, Any]:
    """Re-decode the VAD slices under hallucination walls; fail-open.

    *redecode* transcribes one slice with conditioning off and returns
    ``{"segments": [...], "words": [...]}`` on slice-local time. Returns
    *result* unchanged when nothing is degenerate or nothing could be
    healed; otherwise a new dict with segments/words/text rebuilt.
    """
    segments = list(result.get("segments") or [])
    windows = find_degenerate_windows(segments)
    if not windows:
        return result

    wall_s = sum(end - start for start, end in windows)
    logger.info(
        "self-heal: %d degenerate window(s), %.0fs of decoded audio suspect",
        len(windows),
        wall_s,
    )

    healed_bounds: list[tuple[float, float]] = []
    new_segments: list[dict[str, Any]] = []
    new_words: list[dict[str, Any]] = []
    for offset, chunk in slices:
        start_s = offset / SAMPLE_RATE
        end_s = start_s + len(chunk) / SAMPLE_RATE
        overlaps = any(
            min(end_s, w_end) - max(start_s, w_start) > MIN_OVERLAP_S
            for w_start, w_end in windows
        )
        if not overlaps:
            continue
        if len(healed_bounds) >= MAX_HEAL_SLICES:
            logger.warning("self-heal: slice cap reached, leaving the rest as-is")
            break
        try:
            replacement = redecode(chunk)
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning(
                "self-heal: re-decode failed for [%.0fs, %.0fs), keeping original: %s",
                start_s,
                end_s,
                e,
            )
            continue
        repl_segments = list(replacement.get("segments") or [])
        if find_degenerate_windows(repl_segments):
            logger.info(
                "self-heal: re-decode of [%.0fs, %.0fs) still loops, keeping original",
                start_s,
                end_s,
            )
            continue
        healed_bounds.append((start_s, end_s))
        new_segments.extend(_shift(repl_segments, start_s))
        new_words.extend(_shift(list(replacement.get("words") or []), start_s))

    if not healed_bounds:
        return result

    def _healed(t: float) -> bool:
        return any(lo <= t < hi for lo, hi in healed_bounds)

    kept_segments = [s for s in segments if not _healed(float(s.get("start", 0.0)))]
    kept_words = [
        w
        for w in (result.get("words") or [])
        if not _healed(float(w.get("start", 0.0)))
    ]
    out_segments = sorted(kept_segments + new_segments, key=lambda s: float(s["start"]))
    out_words = sorted(kept_words + new_words, key=lambda w: float(w["start"]))
    logger.info(
        "self-heal: re-decoded %d slice(s), %d -> %d segments",
        len(healed_bounds),
        len(segments),
        len(out_segments),
    )
    return {
        **result,
        "segments": out_segments,
        "words": out_words,
        "text": " ".join(str(s["text"]) for s in out_segments).strip(),
    }
