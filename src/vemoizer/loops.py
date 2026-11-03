"""Hallucination-wall detection for whisper transcripts (issue #219).

Whisper's rolling context can turn one bad window into a wall of decoded
loops ("Kiitos." x50, "DCS. DCS." x95). This module is the single place
that decides what a wall is: :func:`vemoizer.selfheal.heal` re-decodes
the slices under the windows and the quality report renders them, so a
wall can only ever be counted one way. Pure functions of segment dicts.
"""

from __future__ import annotations

from typing import Any

from .textnorm import textnorm

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

#: Whisper's own hallucination cut: above this per-segment compression_ratio
#: its temperature fallback discards the segment. A single long segment that
#: compresses that well is a loop whisper fed itself — and being one segment
#: it never joins a run, so the run rules above cannot see it at all.
COMPRESSION_RATIO_CUT = 2.4

#: Token floor for the single-segment rules. Below it repetition is emphasis
#: ("se on just tosi tosi joo"), and short repetition is the run rules' job.
MIN_TOKENS_LONG_SEGMENT = 8


def _is_long_loop(seg: dict[str, Any]) -> bool:
    """True when one long segment is a repetition wall all by itself.

    The signal is the ``compression_ratio`` whisper stores per segment (its
    own discard cut): a segment that compresses that well is a loop whisper
    fed itself, and being one segment it never joins a run, so the run rules
    above cannot see it. The token floor keeps emphasis out ("se on just
    tosi tosi joo"); a word-repetition rule is NOT used — natural Finnish
    repeats itself ("no niin no niin …") and must not be called a wall.
    """
    tokens = textnorm(str(seg.get("text", ""))).split()
    if len(tokens) < MIN_TOKENS_LONG_SEGMENT:
        return False
    ratio = seg.get("compression_ratio")
    return isinstance(ratio, (int, float)) and ratio > COMPRESSION_RATIO_CUT


def find_degenerate_windows(
    segments: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Time windows ``[(start, end), ...]`` covered by hallucination walls.

    A wall is a run of at least :data:`MIN_RUN` consecutive segments,
    each at most :data:`MAX_TOKENS_PER_SEGMENT` normalized tokens, whose
    union of distinct tokens is at most :data:`MAX_DISTINCT_TOKENS`, or a
    single segment that loops on its own (:func:`_is_long_loop`).
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

    # A loop whisper emits as ONE segment is a wall too, even though nothing
    # repeats across segment boundaries.
    windows.extend(
        (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
        for seg in segments
        if _is_long_loop(seg)
    )

    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if merged and start - merged[-1][1] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def loop_seconds(segments: list[dict[str, Any]]) -> float:
    """Total decoded seconds under hallucination walls (``vemoizer accept``
    reuses this so a wall can only ever be counted one way)."""
    return sum(end - start for start, end in find_degenerate_windows(segments))
