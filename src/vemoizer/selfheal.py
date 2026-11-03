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
degenerate. When the primary re-decode still loops or fails, an optional
*fallback* (the caller passes a prompt-free decode) gets one try: a
name-heavy glossary prompt can itself be what whisper echoes ("Janni."
walls), and re-sending that prompt reproduces the loop. If both fail
the original slice is kept (fail-open, invariant #5). Wall detection lives
in :mod:`vemoizer.loops`; the splicing here is a pure function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .loops import find_degenerate_windows

logger = logging.getLogger(__name__)

#: A slice is re-decoded when it overlaps a degenerate window by more
#: than this.
MIN_OVERLAP_S = 0.2

#: Runaway backstop: never re-decode more than this much audio per file.
#: Time-based, not a slice count: fine-grained VAD on dictaphone audio
#: puts 200+ slices under a 12-minute wall, and a count cap left the tail
#: of such walls unhealed.
MAX_HEAL_SECONDS = 1800.0

#: Flag stamp on a wall that survived the heal (issue #219).
SUSPECT_LOOP = "loop"

RedecodeFn = Callable[[np.ndarray], dict[str, Any]]


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


def _attempt(
    fn: RedecodeFn, chunk: np.ndarray, start_s: float, end_s: float
) -> dict[str, Any] | None:
    """One re-decode try; ``None`` when it raises or still loops."""
    try:
        replacement = fn(chunk)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning(
            "self-heal: re-decode failed for [%.0fs, %.0fs): %s", start_s, end_s, e
        )
        return None
    if find_degenerate_windows(list(replacement.get("segments") or [])):
        logger.info(
            "self-heal: re-decode of [%.0fs, %.0fs) still loops", start_s, end_s
        )
        return None
    return replacement


def _unhealed(
    windows: list[tuple[float, float]], healed: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """The parts of *windows* not covered by *healed* slice bounds.

    Sorted and non-overlapping. A healed slice far from a window clips to
    nothing and must not move the window's end.
    """
    out: list[tuple[float, float]] = []
    for w_start, w_end in windows:
        cursor = w_start
        cuts = [(max(lo, w_start), min(hi, w_end)) for lo, hi in healed]
        for lo, hi in sorted(cut for cut in cuts if cut[0] < cut[1]):
            if lo > cursor:
                out.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < w_end:
            out.append((cursor, w_end))
    return out


def _mark_residual_loops(
    segments: list[dict[str, Any]],
    originals: set[int],
    healed_bounds: list[tuple[float, float]],
) -> tuple[list[dict[str, Any]], float]:
    """Copy *segments*, stamping ``suspect: loop`` on the walls left over.

    Only the ORIGINAL segments the heal kept are candidates: *originals*
    carries their ``id()``. A replacement from a healed slice is never
    stamped — it came from a re-decode that was itself checked for
    degeneracy before being accepted, and calling it a loop would punish
    the one part that got fixed.

    The residual time is the detected window minus the healed slice bounds,
    so a half-healed wall is reported (and flagged) by the seconds still
    looping, not by the whole window. A real sentence that sits inside a
    merged window is flagged too — intended: the merge decided the whole
    stretch reads as one wall, and the reader should re-check all of it.

    Returns the (possibly new) segment list and the seconds of residual
    window. A flagged entry is always a copy, so the caller's dicts are never
    mutated.
    """
    windows = _unhealed(find_degenerate_windows(segments), healed_bounds)
    if not windows:
        return segments, 0.0
    residual_s = sum(w_end - w_start for w_start, w_end in windows)
    out: list[dict[str, Any]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        if id(seg) in originals and any(
            w_start <= start < w_end for w_start, w_end in windows
        ):
            out.append({**seg, "suspect": SUSPECT_LOOP})
        else:
            out.append(seg)
    return out, residual_s


def heal(
    result: dict[str, Any],
    slices: list[tuple[int, np.ndarray]],
    redecode: RedecodeFn,
    fallback: RedecodeFn | None = None,
) -> dict[str, Any]:
    """Re-decode the VAD slices under hallucination walls; fail-open.

    *redecode* transcribes one slice with conditioning off and returns
    ``{"segments": [...], "words": [...]}`` on slice-local time;
    *fallback* is tried when that still loops or fails. Returns
    *result* unchanged when nothing is degenerate; otherwise a new dict
    with segments/words/text rebuilt and any surviving wall flagged
    ``suspect: "loop"`` plus one warning naming its seconds.
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
    healed_s = 0.0
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
        if healed_s + (end_s - start_s) > MAX_HEAL_SECONDS:
            logger.warning("self-heal: time budget reached, leaving the rest as-is")
            break
        replacement = _attempt(redecode, chunk, start_s, end_s)
        if replacement is None and fallback is not None:
            replacement = _attempt(fallback, chunk, start_s, end_s)
        if replacement is None:
            continue
        healed_s += end_s - start_s
        repl_segments = list(replacement.get("segments") or [])
        healed_bounds.append((start_s, end_s))
        new_segments.extend(_shift(repl_segments, start_s))
        new_words.extend(_shift(list(replacement.get("words") or []), start_s))

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
    # Whatever is still a wall after the heal (unfixable by the fallback, or
    # never reached by the time budget) is labelled for the reader and the
    # quality report instead of shipping as fact: a wall is the loudest thing
    # in a transcript and must never be silent.
    out_segments, residual_s = _mark_residual_loops(
        out_segments, {id(s) for s in kept_segments}, healed_bounds
    )
    logger.info(
        "self-heal: re-decoded %d slice(s), %d -> %d segments, %.0fs still looping",
        len(healed_bounds),
        len(segments),
        len(out_segments),
        residual_s,
    )
    out: dict[str, Any] = {**result, "segments": out_segments}
    if healed_bounds:
        out["words"] = out_words
        out["text"] = " ".join(str(s["text"]) for s in out_segments).strip()
    if residual_s > 0.0:
        # The run-dict "warnings" list every stage appends to: the CLI/batch
        # layer builds the quality report from it (report.py renders the loop
        # category, seconds included) BEFORE batch_output._check_result pops
        # it to stderr.
        out["warnings"] = [
            *list(result.get("warnings") or []),
            f"self-heal: {residual_s:.0f}s of repetition loops could not be "
            "repaired and are flagged as suspect",
        ]
    return out
