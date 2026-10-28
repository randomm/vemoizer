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

from .diarization import speaker_for_span

#: Timestamp jitter tolerance: whisper word times wobble ~0.1-0.2s.
BOUNDARY_TOLERANCE_S = 0.2

#: A word further than this from every turn inherits its predecessor.
NEAREST_TURN_CAP_S = 2.0

#: Legitimate one-word interjections that must survive smoothing.
BACKCHANNELS = frozenset(
    {"joo", "niin", "mm", "mmm", "okei", "aivan", "just", "kyllä", "ei", "no"}
)

#: Multi-word acknowledgements that are real turns even at 2-3 words.
BACKCHANNEL_PHRASES = frozenset(
    {
        "mä arvostan",
        "ahaa aivan",
        "joo joo",
        "kyllä kyllä",
        "hyvä kysymys",
        "ymmärrän",
        "hyvä",
        "totta",
        "niin on",
    }
)

#: A label run shorter than BOTH of these (and not a backchannel) is
#: diarization flicker that shreds a sentence across labels — it merges
#: into a neighbouring run ("turn shrapnel", issue #71 round 3).
MIN_RUN_WORDS = 4
MIN_RUN_SECONDS = 1.0

Turn = tuple[float, float, str]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _norm(word: str) -> str:
    return "".join(ch for ch in word.casefold() if ch.isalpha())


def _same_side(a: str | None, b: str | None, trusted: frozenset[str]) -> bool:
    """Smoothing never flips or merges across the trusted/untrusted line."""
    return (a in trusted) == (b in trusted)


def assign_word_speakers(
    words: list[dict[str, Any]],
    turns: list[Turn],
    tolerance: float = BOUNDARY_TOLERANCE_S,
    trusted: frozenset[str] = frozenset(),
) -> list[str | None]:
    """Per-word speaker labels by max turn overlap, smoothed.

    ``trusted`` labels (the owner's own track in a two-track call) are
    never flipped or merged into an untrusted label, nor the reverse.
    """
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
    labels = _requestion(words, labels, turns)
    labels = _smooth(words, labels, turns, trusted)
    return _merge_short_runs(words, labels, turns, trusted)


def turns_for_track(
    track: str | None,
    turns: list[Turn],
    trusted: frozenset[str],
) -> list[Turn]:
    """Diarization turns eligible for labelling one track's words.

    In a two-track call (issue #207) the owner's mic (``me``) words may
    only take a trusted (owner) label and the remote (``them``) words only
    a non-trusted one, so an overlapping turn of the other side can never
    steal a word or a segment. No track key (single-file runs) and an
    empty *trusted* set (no owner configured) keep the all-turns behaviour
    (fail-open).
    """
    if track not in ("me", "them") or not trusted:
        return turns
    want_trusted = track == "me"
    return [t for t in turns if (t[2] in trusted) == want_trusted]


def assign_track_labels(
    words: list[dict[str, Any]],
    turns: list[Turn],
    trusted: frozenset[str] = frozenset(),
) -> list[str | None]:
    """:func:`assign_word_speakers` restricted to each word's own track.

    Words carry the ``track`` key :func:`merge_track_results` tagged them
    with (``me`` for the mic stem, ``them`` for the system stem); each
    group is labelled only from the turns that can legitimately own it
    (:func:`turns_for_track`) and the labels are recombined in the
    original word order. Words without a track key (single-file runs) see
    every turn, so the output matches :func:`assign_word_speakers`.
    """
    labels: list[str | None] = [None] * len(words)
    groups: dict[str | None, list[int]] = {}
    for i, word in enumerate(words):
        groups.setdefault(word.get("track"), []).append(i)
    for track, indexes in groups.items():
        group_words = [words[i] for i in indexes]
        group_turns = turns_for_track(track, turns, trusted)
        group_labels = assign_word_speakers(group_words, group_turns, trusted=trusted)
        for i, label in zip(indexes, group_labels, strict=True):
            labels[i] = label
    return labels


def label_segments_by_track(
    segments: list[dict[str, Any]],
    turns: list[Turn],
    trusted: frozenset[str] = frozenset(),
) -> None:
    """Attach the best-overlap speaker to every segment from its own track.

    Mutates *segments*; see :func:`turns_for_track` for the per-track
    restriction. A segment labelled by none of its track's turns keeps the
    speaker a previous stage set (fail-open, as in ``speaker_for_span``).
    """
    for segment in segments:
        speaker = speaker_for_span(
            float(segment["start"]),
            float(segment["end"]),
            turns_for_track(segment.get("track"), turns, trusted),
        )
        if speaker is not None:
            segment["speaker"] = speaker


def _ends_question(word: dict[str, Any]) -> bool:
    return str(word.get("word", "")).rstrip().endswith("?")


def _requestion(
    words: list[dict[str, Any]],
    labels: list[str | None],
    turns: list[Turn],
) -> list[str | None]:
    """Re-assign the few words after a ``?`` without boundary padding.

    The asker's long turn plus the jitter tolerance swallows short answers
    ("Pääseekö Topiinkin? Pääsee."); at the question boundary the padded
    interval must not leak into the questioner's turn.
    """
    out = list(labels)
    for i, word in enumerate(words):
        if not _ends_question(word):
            continue
        for j in range(i + 1, min(i + 4, len(words))):
            w = words[j]
            w_start, w_end = float(w.get("start", 0.0)), float(w.get("end", 0.0))
            best: str | None = None
            best_overlap = 0.0
            for t_start, t_end, speaker in turns:
                ov = _overlap(w_start, w_end, t_start, t_end)
                if ov > best_overlap:
                    best_overlap = ov
                    best = speaker
            if best is not None:
                out[j] = best
    return out


def _smooth(
    words: list[dict[str, Any]],
    labels: list[str | None],
    turns: list[Turn],
    trusted: frozenset[str] = frozenset(),
) -> list[str | None]:
    """Flip single-word A-B-A islands unless they are real backchannels
    or answers right after a question (the ``?`` is a hard boundary)."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i - 1] != out[i + 1] or out[i] == out[i - 1] or out[i] is None:
            continue
        if not _same_side(out[i], out[i - 1], trusted):
            continue  # an owner word inside remote speech is not flicker
        if _ends_question(words[i - 1]):
            continue  # a one-word answer to a question is a real turn
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
        # Track matters as much as time: on a two-track call the owner's
        # and the remote's segments overlap, and time alone pulls the
        # other track's words in (issue #207). None == None keeps
        # single-file behaviour byte-identical.
        members = [
            (w, label)
            for w, label in zip(words, labels, strict=False)
            if s_start <= float(w.get("start", 0.0)) < s_end
            and label is not None
            and w.get("track") == seg.get("track")
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


def _merge_short_runs(
    words: list[dict[str, Any]],
    labels: list[str | None],
    turns: list[Turn],
    trusted: frozenset[str] = frozenset(),
) -> list[str | None]:
    """Merge flicker label runs into a neighbour ("turn shrapnel").

    A run merges only when ALL hold: it is interior (has neighbours on
    both sides), shorter than :data:`MIN_RUN_WORDS` words, its *backing
    diarization turn* is itself a micro-turn (< :data:`MIN_RUN_SECONDS` —
    a real answer rides a real turn, flicker rides a sliver), it is not a
    backchannel, and it does not directly follow or contain a question
    mark (answers are real turns). The target is the neighbour whose
    speaker's turns overlap the run's words most; previous wins ties.
    Neighbours on the other side of the ``trusted`` line are never targets.
    """
    if not words:
        return labels
    out = list(labels)
    runs: list[list[Any]] = []
    for i, label in enumerate(out):
        if runs and runs[-1][2] == label:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, label])

    def _words_turn_overlap(start: int, end: int, speaker: str) -> float:
        total = 0.0
        for i in range(start, end):
            w_start = float(words[i].get("start", 0.0))
            w_end = float(words[i].get("end", 0.0))
            for t_start, t_end, t_speaker in turns:
                if t_speaker == speaker:
                    total += _overlap(w_start, w_end, t_start, t_end)
        return total

    for r in range(1, len(runs) - 1):
        start, end, label = runs[r]
        if label is None or end - start >= MIN_RUN_WORDS:
            continue
        if _ends_question(words[start - 1]) or any(
            _ends_question(words[i]) for i in range(start, end)
        ):
            continue
        joined = " ".join(
            _norm(str(words[i].get("word", ""))) for i in range(start, end)
        ).strip()
        if joined in BACKCHANNEL_PHRASES or (
            end - start == 1 and joined in BACKCHANNELS
        ):
            continue
        run_start = float(words[start].get("start", 0.0))
        run_end = float(words[end - 1].get("end", 0.0))
        backing = max(
            (t for t in turns if t[2] == label),
            key=lambda t: _overlap(run_start, run_end, t[0], t[1]),
            default=None,
        )
        if backing is not None and (backing[1] - backing[0]) >= MIN_RUN_SECONDS:
            continue  # a real turn backs this run; not flicker
        prev_label, next_label = runs[r - 1][2], runs[r + 1][2]
        candidates = [
            c
            for c in (prev_label, next_label)
            if c is not None and _same_side(c, label, trusted)
        ]
        if not candidates:
            continue
        target = max(candidates, key=lambda c: _words_turn_overlap(start, end, c))
        for i in range(start, end):
            out[i] = target
        runs[r][2] = target
    return out
