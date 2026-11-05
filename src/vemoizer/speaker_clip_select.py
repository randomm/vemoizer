"""M5b: pure speaker clip selection over a rendered sidecar (issues #90, #253).

The selector half of the speaker-clip feature: :func:`select_clips` and
:func:`talk_share` read a sidecar's ``paragraphs``/``segments`` and never
touch audio or the filesystem. The audio half (extract, play, temp dir) is
in :mod:`vemoizer.speaker_clips`, which re-exports these names.

Design notes
------------

* A *turn* is one labelled, non-suspect paragraph that is not a
  backchannel and is at least 2 s long (see :func:`select_clips` rules).
  Because a speaker's contiguous same-speaker paragraphs are the only
  candidate turns, a turn's ``[start, end)`` never crosses a speaker
  boundary by construction.
* Windows come from the sidecar ``segments`` when available: a clean
  segment is :data:`MIN_TURN_S` to :data:`MAX_TURN_S` long and overlaps no
  other label's segment, the window IS that whole segment (start to end),
  and the quote is that segment's own text — every word printed is a word
  heard (issue #253). A label with no clean segment in a third of the
  meeting (or a sidecar with no segments at all) falls back to a window at
  its own paragraph's start lasting ``min(max_s, paragraph_duration)``
  seconds, quoting only the leading share of that paragraph's text the
  window actually plays. Either way a window overlapping another label's
  speech — another label's segment, or another label's paragraph for an old
  sidecar without segments — is rejected, and a third left with no
  acceptable window yields no clip.
* Picks for one label spread over the beginning, middle and end of the
  meeting, ranked by the turn's own duration (ties: earlier start), and no
  two chosen windows of a label overlap.

Nothing here is allowed to load a model, read a file, or play audio.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .speaker_align import BACKCHANNEL_PHRASES, BACKCHANNELS

__all__ = ["ClipWindow", "select_clips", "talk_share"]

#: A turn shorter than this is not a clip candidate.
MIN_TURN_S = 2.0

#: A segment longer than this is not a *clean* clip candidate (issue #253):
#: its window is the whole segment, so a longer one would play words that a
#: 120-char quote cannot cover.
MAX_TURN_S = 8.0

#: Quote text is bounded to roughly this many characters.
MAX_QUOTE_CHARS = 120


class ClipWindow(tuple):
    """A selected clip: ``(start_s, end_s, quote)``.

    Subclasses ``tuple`` so a window is a hashable, comparable 3-tuple the
    callers can use as a dict key (``extract_clips`` returns
    ``dict[ClipWindow, Path | None]``) while still exposing named fields.
    """

    __slots__ = ()

    def __new__(cls, start_s: float, end_s: float, quote: str = "") -> ClipWindow:
        return tuple.__new__(cls, (float(start_s), float(end_s), quote))

    @property
    def start_s(self) -> float:
        return self[0]

    @property
    def end_s(self) -> float:
        return self[1]

    @property
    def quote(self) -> str:
        return self[2]


def _labelled(p: dict[str, Any]) -> str | None:
    """The speaker label of *p*, or ``None`` when unlabelled.

    A paragraph is labelled iff ``speaker`` is a non-empty string; a
    missing key, ``None``, or ``""`` all count as unlabelled.
    """
    label = p.get("speaker")
    return label if isinstance(label, str) and label else None


def talk_share(paragraphs: list[dict[str, Any]]) -> dict[str, float]:
    """Per-speaker fraction of labelled talk time.

    Fractions sum to 1 over labelled paragraphs (within float error);
    unlabelled text is excluded. Empty or all-unlabelled input gives {}.
    """
    duration: dict[str, float] = {}
    for p in paragraphs:
        label = _labelled(p)
        if label is None:
            continue
        span = float(p.get("end", 0.0)) - float(p.get("start", 0.0))
        if span > 0:
            duration[label] = duration.get(label, 0.0) + span
    total = sum(duration.values())
    if total <= 0:
        return {}
    return {label: span / total for label, span in duration.items()}


def _is_backchannel(text: str) -> bool:
    """True when the whitespace-collapsed, casefolded turn text is a
    backchannel (membership in ``BACKCHANNELS ∪ BACKCHANNEL_PHRASES`` —
    no word-count condition, so "joo" inside a longer sentence is fine).
    """
    collapsed = " ".join(text.split()).casefold()
    return collapsed in BACKCHANNELS or collapsed in BACKCHANNEL_PHRASES


def _truncated(text: str) -> str:
    """Whitespace-collapsed *text* bounded to :data:`MAX_QUOTE_CHARS` chars."""
    collapsed = " ".join(text.split())
    if len(collapsed) > MAX_QUOTE_CHARS:
        return collapsed[:MAX_QUOTE_CHARS].rstrip() + "…"
    return collapsed


def _leading(text: str, share: float) -> str:
    """The leading *share* (0..1) of *text*, cut at a word boundary.

    Used by the paragraph fallback, whose window is only a prefix of the
    paragraph: quoting the whole paragraph would print words the clip never
    plays (issue #253). Dropped trailing words are signalled with "…"; one
    word is always kept, so a clip is never quoted by silence.
    """
    words = " ".join(text.split()).split(" ")
    n = round(len(words) * min(max(share, 0.0), 1.0))
    if n >= len(words):
        return " ".join(words)
    return " ".join(words[: max(n, 1)]) + "…"


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True when the half-open intervals [a_start, a_end) and [b_start, b_end) meet."""
    return a_start < b_end and b_start < a_end


def _overlaps_other_label(
    start: float, end: float, label: str, others: list[dict[str, Any]]
) -> bool:
    """True when any item in *others* NOT labelled *label* overlaps [start, end).

    An unlabelled item counts as another label: speech we cannot
    attribute may still bleed into the clip (issue #253). *others* is the
    sidecar's segments, or its paragraphs for an old sidecar without any.
    """
    for s in others:
        other = s.get("speaker")
        if isinstance(other, str) and other == label:
            continue
        if float(s.get("start", 0.0)) < end and float(s.get("end", 0.0)) > start:
            return True
    return False


def _third_for(start: float, audio_end: float) -> int:
    """Which third of the meeting [0, audio_end) *start* falls in (0..2)."""
    if audio_end <= 0:
        return 0
    third = audio_end / 3.0
    return min(int(start // third), 2)


class _Candidate(NamedTuple):
    """A clip candidate: window ``(start, end)``, its quote, and the duration
    of the turn it came from.

    Ranking uses the turn duration, not the window length: window length
    ties every long turn at ``max_s`` and silently degrades to "earliest
    wins".
    """

    start: float
    end: float
    quote: str
    duration: float


def _candidates(
    items: list[dict[str, Any]],
    audio_end: float,
    *,
    whole: bool,
    clean: bool,
    max_s: float,
    others: list[dict[str, Any]] | None,
) -> dict[str, list[_Candidate]]:
    """Candidate clip windows per label, anchored at each item's START.

    Candidates are labelled items of >= :data:`MIN_TURN_S` with text; ``clean``
    also drops suspect items and backchannels. With ``whole`` (the sidecar's
    segments) a candidate is clean only up to :data:`MAX_TURN_S` too, the
    window IS the item (start to end) and the quote is the item's own text;
    without it (the paragraph fallback) the window is the first
    ``min(max_s, duration)`` seconds and the quote is only that played share
    of the text. Either way an item overlapping another label's speech in
    *others* is rejected, and the window is clamped to ``[0, audio_end]``
    (issue #253).
    """
    windows: dict[str, list[_Candidate]] = {}
    for p in items:
        label = _labelled(p)
        if label is None:
            continue
        text = str(p.get("text", ""))
        if not text or (clean and (p.get("suspect") or _is_backchannel(text))):
            continue
        start, end = float(p.get("start", 0.0)), float(p.get("end", 0.0))
        duration = end - start
        if duration < MIN_TURN_S or (whole and duration > MAX_TURN_S):
            continue
        w_end = end if whole else start + min(max_s, duration)
        if audio_end > 0:
            w_end = min(w_end, audio_end)
        if w_end <= start:
            continue
        if others is not None and _overlaps_other_label(start, w_end, label, others):
            continue
        share = 1.0 if whole else (w_end - start) / duration
        windows.setdefault(label, []).append(
            _Candidate(start, w_end, _truncated(_leading(text, share)), duration)
        )
    return windows


def _fits(candidate: _Candidate, picked: list[_Candidate]) -> bool:
    """True when *candidate* overlaps none of the already-picked windows."""
    return not any(
        _overlaps(candidate.start, candidate.end, p.start, p.end) for p in picked
    )


def _longest_in_third(
    candidates: list[_Candidate],
    picked: list[_Candidate],
    third: int,
    audio_end: float,
) -> _Candidate | None:
    """The longest unpicked, non-overlapping candidate falling in *third*.

    Ranked by the turn's own duration, ties going to the earlier start, so a
    7 s turn beats a 3 s one instead of losing to list order.
    """
    in_third = [
        c
        for c in candidates
        if _fits(c, picked) and _third_for(c.start, audio_end) == third
    ]
    if not in_third:
        return None
    return max(in_third, key=lambda c: (c.duration, -c.start))


def select_clips(
    paragraphs: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None = None,
    *,
    per_speaker: int = 3,
    min_s: float = 3.0,
    max_s: float = 5.0,
    total_duration: float | None = None,
) -> dict[str, list[ClipWindow]]:
    """Pick up to *per_speaker* short windows per speaker label.

    Rules (see module docstring): with *segments* given, the primary
    candidates are that label's own clean segments — :data:`MIN_TURN_S` to
    :data:`MAX_TURN_S` long, non-suspect, non-backchannel, overlapping no
    other label's segment; the window IS the whole segment (start to end)
    and the quote is the segment's own text, so every quoted word is heard.
    A third of the meeting with no clean segment — or a sidecar without
    segments at all — falls back to the label's own paragraph candidates: a
    window starting at the paragraph start and lasting ``min(max_s,
    duration)`` seconds, quoting only the played leading share of that
    paragraph's text (never another speaker's). Both kinds reject a window
    overlapping another label's speech — another label's segment when the
    sidecar has segments, otherwise another label's paragraph, since old
    sidecars have no segments — and a third left with no acceptable window
    yields no clip. Paragraph candidates prefer non-suspect, non-backchannel
    turns; a label with only such turns still gets clips, so every label
    with speech gets a quote and a clip (issue #202). Picks spread across
    the beginning, middle and end of the meeting: the LONGEST candidate from
    each third where possible (ranked by its turn duration, ties going to
    the earlier start), otherwise the longest remaining candidate regardless
    of third. No two chosen windows of one label overlap (issue #253).

    ``min_s`` is a display annotation only (the nominal lower bound for a
    clip's length); it never disqualifies a candidate — the disqualification
    thresholds are the :data:`MIN_TURN_S` / :data:`MAX_TURN_S` gates for a
    segment candidate and the 2 s :data:`MIN_TURN_S` gate for a paragraph
    candidate, whose window length is ``min(max_s, item_duration)``.

    *total_duration* clamps windows to the audio end; it defaults to the
    end of the last paragraph or segment. Returns an ordered dict of label
    -> windows sorted by ``start_s``; a speaker with fewer clean turns gets
    fewer clips, without error.
    """
    segs = [s for s in segments or [] if isinstance(s, dict)]
    audio_end = total_duration
    if audio_end is None:
        audio_end = max(
            (float(p.get("end", 0.0)) for p in list(paragraphs) + segs if "end" in p),
            default=0.0,
        )
    # A label with a clean turn uses only clean turns; a label whose turns
    # are all suspect or backchannels still gets its longest turn (issue #202).
    # Both pools reject a window overlapping another label's speech: guarded
    # by the segments when there are any, by the paragraphs otherwise (#253).
    guards = segs if segs else list(paragraphs)
    clean_para = _candidates(
        paragraphs, audio_end, whole=False, clean=True, max_s=max_s, others=guards
    )
    any_para = _candidates(
        paragraphs, audio_end, whole=False, clean=False, max_s=max_s, others=guards
    )
    fallback = {
        label: clean_para.get(label) or turns for label, turns in any_para.items()
    }
    primary = (
        _candidates(segs, audio_end, whole=True, clean=True, max_s=max_s, others=segs)
        if segs
        else {}
    )

    labels = list(any_para) + [label for label in primary if label not in any_para]
    result: dict[str, list[ClipWindow]] = {}
    for label in labels:
        seg_turns = primary.get(label, [])
        para_turns = fallback.get(label, [])
        if not seg_turns and not para_turns:
            continue
        picked: list[_Candidate] = []
        # Spread rule: fill beginning -> middle -> end thirds with the
        # longest clean segment in each, falling back to the label's own
        # paragraph candidates when a third has none (a third with no
        # acceptable window stays empty), then top up from the longest
        # remaining regardless of third. _fits() keeps any window that
        # overlaps an already-picked one out, equal start or not.
        for third in (0, 1, 2):
            for pool in (seg_turns, para_turns):
                best = _longest_in_third(pool, picked, third, audio_end)
                if best is not None:
                    picked.append(best)
                    break
        leftovers = sorted(
            (c for c in seg_turns + para_turns if c not in picked),
            key=lambda c: (-c.duration, c.start),
        )
        for c in leftovers:
            if len(picked) >= per_speaker:
                break
            if _fits(c, picked):
                picked.append(c)
        picked.sort(key=lambda c: c.start)
        result[label] = [
            ClipWindow(c.start, c.end, c.quote) for c in picked[:per_speaker]
        ]
    return result
