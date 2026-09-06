"""Word-level speaker attribution: whisper words × pyannote turns (issue #71).

Speakers used to attach per whisper sentence segment by midpoint overlap,
so a question and its answer inside one segment fused under one label.
Both ingredients for doing better already exist — word timestamps from
whisper and speaker turns from pyannote — and their intersection is pure
geometry (the WhisperX recipe, no models):

1. :func:`assign_word_speakers` — each word takes the turn with maximum
   overlap of its (tolerance-padded) interval; a word touching no turn
   takes the nearest turn within a cap, else inherits its predecessor.
2. Single-word label islands are smoothed away as diarization flicker —
   unless the word is a backchannel ("joo", "niin") sitting mostly inside
   the other speaker's turn, which is a real interjection.
3. :func:`split_segments_at_speaker_changes` — whisper segments split at
   word-level speaker boundaries, so the paragraph stage sees true turns.

Fail-open: no words or no turns leaves segments untouched.
"""

from __future__ import annotations

from typing import Any

#: Timestamp jitter tolerance: whisper word times wobble ~0.1-0.2s.
BOUNDARY_TOLERANCE_S = 0.2

#: A word further than this from every turn inherits its predecessor.
NEAREST_TURN_CAP_S = 2.0

#: Legitimate one-word interjections that must survive smoothing.
BACKCHANNELS = frozenset(
    {"joo", "niin", "mm", "mmm", "okei", "aivan", "just", "kyllä", "ei", "no"}
)

Turn = tuple[float, float, str]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _norm(word: str) -> str:
    return "".join(ch for ch in word.casefold() if ch.isalpha())


def assign_word_speakers(
    words: list[dict[str, Any]],
    turns: list[Turn],
    tolerance: float = BOUNDARY_TOLERANCE_S,
) -> list[str | None]:
    """Per-word speaker labels by max turn overlap, smoothed."""
    if not words or not turns:
        return [None] * len(words)
    labels: list[str | None] = []
    for word in words:
        start = float(word.get("start", 0.0)) - tolerance
        end = float(word.get("end", 0.0)) + tolerance
        best: str | None = None
        best_overlap = 0.0
        for t_start, t_end, speaker in turns:
            ov = _overlap(start, end, t_start, t_end)
            if ov > best_overlap:
                best_overlap = ov
                best = speaker
        if best is None:
            mid = (start + end) / 2.0
            nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))
            distance = min(abs(mid - nearest[0]), abs(mid - nearest[1]))
            if distance <= NEAREST_TURN_CAP_S:
                best = nearest[2]
            elif labels:
                best = labels[-1]
        labels.append(best)
    return _smooth(words, labels, turns)


def _smooth(
    words: list[dict[str, Any]],
    labels: list[str | None],
    turns: list[Turn],
) -> list[str | None]:
    """Flip single-word A-B-A islands unless they are real backchannels."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i - 1] != out[i + 1] or out[i] == out[i - 1] or out[i] is None:
            continue
        word = words[i]
        if _norm(str(word.get("word", ""))) in BACKCHANNELS:
            start = float(word.get("start", 0.0))
            end = float(word.get("end", 0.0))
            duration = max(end - start, 1e-6)
            inside = sum(
                _overlap(start, end, t_start, t_end)
                for t_start, t_end, speaker in turns
                if speaker == out[i]
            )
            if inside / duration > 0.5:
                continue  # a real interjection in its own turn
        out[i] = out[i - 1]
    return out


def split_segments_at_speaker_changes(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    labels: list[str | None],
) -> list[dict[str, Any]]:
    """Split whisper *segments* wherever the word-level speaker changes.

    Segment metadata (``suspect`` etc.) is copied onto every piece; piece
    times come from the real word timestamps. Segments with no labelled
    words pass through untouched (fail-open).
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        s_start, s_end = float(seg["start"]), float(seg["end"])
        members = [
            (w, label)
            for w, label in zip(words, labels, strict=False)
            if s_start <= float(w.get("start", 0.0)) < s_end and label is not None
        ]
        if not members:
            out.append(seg)
            continue
        runs: list[tuple[str, list[dict[str, Any]]]] = []
        for w, label in members:
            if runs and runs[-1][0] == label:
                runs[-1][1].append(w)
            else:
                runs.append((label, [w]))
        if len(runs) == 1:
            out.append({**seg, "speaker": runs[0][0]})
            continue
        for label, run_words in runs:
            out.append(
                {
                    **seg,
                    "start": float(run_words[0]["start"]),
                    "end": float(run_words[-1]["end"]),
                    "text": " ".join(str(w.get("word", "")) for w in run_words),
                    "speaker": label,
                }
            )
    return out
