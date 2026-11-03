"""Pure transcript-quality metrics for the accept benchmark (issue #228).

One hypothesis (pipeline output) scored against a human-reviewed
reference excerpt: pure functions of plain segment dicts — no IO, no
models, no CLI (#229 renders them). Walls and WER reuse
``vemoizer.loops.loop_seconds`` and ``vemoizer.eval_harness.wer`` so a
wall is counted one way. Seconds are relative to the excerpt start;
reference ``speaker`` is a name or ``"unclear"``, hypothesis segments
may lack a speaker and add ``suspect`` / ``compression_ratio``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .eval_harness import wer as _wer
from .loops import loop_seconds
from .textnorm import textnorm

Segment = dict[str, Any]

UNCLEAR = "unclear"  # reference time the owner could not attribute: unscored
MIN_WORDS_ORPHAN = 3  # words making an overlap-free segment invented content,
# not a backchannel (the repair stage talking to itself)
FRAME = 0.1  # seconds; resolution of the speaker-metric time grid


def _duration(seg: Segment) -> float:
    return max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))


def _overlap(a: Segment, b: Segment) -> float:
    lo = max(float(a.get("start", 0.0)), float(b.get("start", 0.0)))
    hi = min(float(a.get("end", 0.0)), float(b.get("end", 0.0)))
    return max(0.0, hi - lo)


def _label(seg: Segment) -> str:
    # A missing hypothesis speaker is its own label "", never a name match.
    return str(seg.get("speaker") or "").casefold()


def _ref_name(seg: Segment) -> str | None:
    # Casefolded reference name, or None when the segment is not scorable.
    # WHY casefold before the UNCLEAR test: reviewers type "Unclear" and
    # "UNCLEAR"; attribution status must not hinge on capitalisation.
    name = str(seg.get("speaker") or "").casefold()
    return None if not name or name == UNCLEAR else name


def _at(segments: list[Segment], t: float) -> Segment | None:
    """Segment covering time *t*: latest start wins, ties keep the segment
    listed first, so overlapping hypotheses yield one deterministic label."""
    best: Segment | None = None
    for seg in segments:
        start = float(seg.get("start", 0.0))
        if start <= t < float(seg.get("end", 0.0)) and (
            best is None or start > float(best.get("start", 0.0))
        ):
            best = seg
    return best


def _frame_samples(ref: list[Segment], hyp: list[Segment]) -> list[tuple[str, str]]:
    """(ref name, hyp label) per 0.1 s frame over scorable reference time.

    WHY a time grid instead of raw segment overlaps: duplicated or
    overlapping hyp segments must not count twice, and each frame carries
    exactly one label, chosen by :func:`_at`.
    """
    scorable = [r for r in ref if _ref_name(r) is not None]
    if not scorable:
        return []
    t0 = min(float(r.get("start", 0.0)) for r in scorable)
    t1 = max(float(r.get("end", 0.0)) for r in scorable)
    samples: list[tuple[str, str]] = []
    i = 0
    while t0 + (i + 0.5) * FRAME < t1:  # frame midpoint, global 0.1 s grid
        t = t0 + (i + 0.5) * FRAME
        r = _at(ref, t)
        if (name := _ref_name(r) if r else None) is not None:
            h = _at(hyp, t)
            samples.append((name, _label(h) if h else ""))
        i += 1
    return samples


def _frame_matrix(samples: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Seconds per (ref name, hyp label) cell: frame counts x FRAME."""
    return {cell: n * FRAME for cell, n in Counter(samples).items()}


def _greedy_mapping(cells: dict[tuple[str, str], float]) -> dict[str, str]:
    """One-to-one hyp label -> ref name maximizing overlap (greedy, largest
    cell first; exact enough for a handful of speakers)."""
    names = {name for name, _label_ in cells}
    # Seed identity first: a label that already IS a reference name stays
    # put, or greedy swapping could score a correct raw label below name_acc.
    # Trade-off: two real names swapped are not remapped (mapped_acc stays
    # low) - a swapped name IS a naming error the owner would see.
    mapping = {lb: lb for _n, lb in cells if lb and lb in names}
    used_names = set(mapping.values())
    for (name, label), sec in sorted(cells.items(), key=lambda c: (-c[1], c[0])):
        if sec > 0 and label not in mapping and name not in used_names:
            used_names.add(name)
            mapping[label] = name
    return mapping


def speaker_accuracy(ref: list[Segment], hyp: list[Segment]) -> dict[str, float]:
    """Time-weighted speaker accuracy over reference time not marked unclear.

    ``name_acc``: share of scorable reference time whose frame label equals
    the reference name (case-insensitive). ``mapped_acc``: same after the
    best one-to-one hyp-label -> name mapping, so correct clustering still
    scores 1.0 under raw labels; never below ``name_acc``. Vacuous truth:
    no scorable reference time means 1.0 — nothing was gotten wrong.
    """
    samples = _frame_samples(ref, hyp)
    total = len(samples)
    if total == 0:
        return {"name_acc": 1.0, "mapped_acc": 1.0}
    mapping = _greedy_mapping(_frame_matrix(samples))
    name_match = sum(1 for n, lb in samples if lb == n)
    mapped_match = sum(1 for n, lb in samples if mapping.get(lb) == n)
    return {
        "name_acc": name_match / total,
        # Backstop: the mapping is a convenience, never a penalty.
        "mapped_acc": max(mapped_match, name_match) / total,
    }


def _held_extra(secs: list[float], total: float, share: float, floor: float) -> int:
    """How many cells clear both thresholds, minus one (the extras)."""
    return max(0, sum(1 for s in secs if s >= floor and s >= share * total) - 1)


def split_merge_counts(
    ref: list[Segment],
    hyp: list[Segment],
    min_share: float = 0.1,
    min_seconds: float = 5.0,
) -> tuple[int, int]:
    """(split_count, merge_count) from the frame-derived ref x hyp-label matrix.

    Split: one reference speaker held by >= 2 hyp labels, each carrying
    at least ``min_share`` of the speaker's time and ``min_seconds`` —
    extra labels count. Merge: one hyp label covering >= 2 reference
    speakers — extra speakers count. Unclear reference time is ignored and
    the "" (unlabelled) label is not a speaker — unlabelled time is wrong
    in the accuracy metrics but splits/merges no one; the double floor
    keeps a 2-second diarization hiccup from reading as a split.
    """
    cells = _frame_matrix(_frame_samples(ref, hyp))
    held: dict[str, list[float]] = {}
    cover: dict[str, list[float]] = {}
    name_total: dict[str, float] = {}
    for (name, label), sec in cells.items():
        name_total[name] = name_total.get(name, 0.0) + sec
        if label:  # "" is missing speaker, not a speaker
            held.setdefault(name, []).append(sec)
            cover.setdefault(label, []).append(sec)
    split = sum(
        _held_extra(secs, name_total[name], min_share, min_seconds)
        for name, secs in held.items()
    )
    merge = sum(
        _held_extra(secs, sum(secs), min_share, min_seconds) for secs in cover.values()
    )
    return split, merge


def _count_token_run(tokens: list[str], term: list[str]) -> int:
    # Non-overlapping occurrences of the token run *term* in *tokens*.
    n, i = 0, 0
    while i + len(term) <= len(tokens):
        if tokens[i : i + len(term)] == term:
            n += 1
            i += len(term)
        else:
            i += 1
    return n


def glossary_insertions(ref_text: str, hyp_text: str, terms: list[str]) -> int:
    """Glossary terms the hypothesis invented relative to the reference.

    Per term: max(0, hyp occurrences - ref occurrences), summed; case-
    insensitive, whole-word (multi-word allowed) after textnorm. WHY: the
    repair stage sprinkles glossary terms never said — "IBC-kontti"
    appearing where the reference says "IWS" must lower the score (#228).
    Nested terms double count by design: each listed term is scored on its
    own, so "AI" and "AI agentti" both fire inside "AI agentti".
    """
    ref_tokens = textnorm(ref_text).split()
    hyp_tokens = textnorm(hyp_text).split()
    total = 0
    for term in terms:
        term_tokens = textnorm(term).split()
        if term_tokens:
            total += max(
                0,
                _count_token_run(hyp_tokens, term_tokens)
                - _count_token_run(ref_tokens, term_tokens),
            )
    return total


def _token_matches(token: str, pattern: str) -> bool:
    # "kont*" is a stem (prefix); anything else must equal the whole word.
    if pattern.endswith("*"):
        return token.startswith(pattern[:-1])
    return token == pattern


def _title_term_hit(term: str, title_tokens: list[str]) -> bool:
    """True when the words of *term* appear consecutively in the title.

    Matching is whole-word unless a word ends in ``*``, which makes it a
    stem: Finnish inflects word endings ("kontti" -> "kontin"), so the
    reviewer writes "kont*". Whole-word by default because a short
    acronym ("AI") is a prefix of ordinary words ("aina").
    """
    # textnorm drops "*" and may split a word ("IWS:n" -> "iws n"), so
    # normalise per raw word and re-attach the marker to its last piece.
    patterns: list[str] = []
    for word in term.split():
        pieces = textnorm(word.rstrip("*")).split()
        if pieces and word.endswith("*"):
            pieces[-1] += "*"
        patterns.extend(pieces)
    if not patterns:
        return False
    for i in range(len(title_tokens) - len(patterns) + 1):
        window = title_tokens[i : i + len(patterns)]
        if all(_token_matches(t, p) for t, p in zip(window, patterns, strict=True)):
            return True
    return False


def _count_hits(terms: list[str], toks: list[str], want: bool) -> int:
    # A term textnorm() empties (pure punctuation) can never match: ignore
    # it as neither missing nor a hit rather than emitting a false signal.
    return sum(1 for t in terms if textnorm(t) and _title_term_hit(t, toks) is want)


def title_hits(title: str, title_expect: dict[str, list[str]]) -> dict[str, int]:
    """Title checks: ``must_contain_missing`` and ``forbid_hits``, matched
    case-insensitively, whole-word or ``stem*`` (see :func:`_title_term_hit`)."""
    title_tokens = textnorm(title).split()
    must = title_expect.get("must_contain") or []
    forbid = title_expect.get("forbid") or []
    return {
        "must_contain_missing": _count_hits(must, title_tokens, False),
        "forbid_hits": _count_hits(forbid, title_tokens, True),
    }


def _speech_spans(seg: Segment, ref: list[Segment]) -> list[tuple[float, float]]:
    """The parts of *seg* that overlap some reference turn, merged, sorted."""
    s0, s1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    spans = sorted(
        (max(s0, float(r.get("start", 0.0))), min(s1, float(r.get("end", 0.0))))
        for r in ref
    )
    merged: list[tuple[float, float]] = []
    for lo, hi in spans:
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _word_times(words: list[str], spans: list[tuple[float, float]]) -> list[float]:
    """Word midpoints spread evenly over the speech *spans* only.

    Segment-level timing is all every backend shares. Spreading over the
    reference speech inside the segment (not its whole span) keeps words
    out of pauses, so a correct segment crossing a silence is not
    invented text; a word landing one turn off is absorbed by the WER
    threshold.
    """
    total = sum(hi - lo for lo, hi in spans)
    times: list[float] = []
    for k in range(len(words)):
        pos = (k + 0.5) * total / len(words)
        for lo, hi in spans:
            if pos <= hi - lo:
                times.append(lo + pos)
                break
            pos -= hi - lo
        else:  # float rounding past the last span
            times.append(spans[-1][1])
    return times


def _owner_turn(t: float, speaker: str, ref: list[Segment]) -> int | None:
    # The reference turn containing time t; under overlapping speech prefer
    # the turn of the same speaker, else the first one in reference order.
    hits = [
        i
        for i, r in enumerate(ref)
        if float(r.get("start", 0.0)) <= t <= float(r.get("end", 0.0))
    ]
    same = [i for i in hits if _label(ref[i]) == speaker]
    return (same or hits or [None])[0]


def flag_quality(
    ref: list[Segment], hyp: list[Segment], error_wer: float = 0.5
) -> dict[str, float | None]:
    """How well do ``suspect`` flags cover the wrong parts of the hypothesis?

    Scored per REFERENCE turn. Each hyp word is placed at an even share of
    the reference speech inside its segment (pauses excluded) and given to
    the turn containing that time, so the hypothesis choosing other
    boundaries (splitting a turn, one segment spanning several turns or a
    pause) is never an error by itself, and a local error is never
    averaged away by correct neighbours. A turn is an *error* when
    the WER of its text against its attributed words exceeds ``error_wer``
    (a turn with text and no words is dropped speech; "unclear" turns are
    scored like any other: the owner wrote what was said), and *flagged*
    when any segment contributing to it has a truthy ``suspect``; a turn
    with no reference words but attributed hyp words is an error of its
    full length. A hyp segment overlapping no turn is invented content:
    one error of its duration when it has at least
    :data:`MIN_WORDS_ORPHAN` normalized words (fewer is a backchannel).
    Known limit: with segment-level timing a correct segment spanning a
    pause and invented words spanning silence look alike in time, so
    invented text attached to real speech counts as an error of the
    touched turn's seconds; it is always detected, only under-weighted.
    precision = flagged errors / flagged units, recall = flagged errors /
    error units; both None (no signal), never 0/0.
    """
    words: list[list[str]] = [[] for _ in ref]
    flags = [False] * len(ref)
    units: list[tuple[bool, bool, float]] = []  # (is_error, is_flagged, seconds)
    for h in hyp:
        tokens = textnorm(str(h.get("text", ""))).split()
        spans = _speech_spans(h, ref)
        if not spans:  # no reference speech under it: invented content
            if len(tokens) >= MIN_WORDS_ORPHAN:
                units.append((True, bool(h.get("suspect")), _duration(h)))
            continue
        for t, w in zip(_word_times(tokens, spans), tokens, strict=True):
            i = _owner_turn(t, _label(h), ref)
            if i is not None:
                words[i].append(w)
                flags[i] = flags[i] or bool(h.get("suspect"))
    for r, ws, flagged in zip(ref, words, flags, strict=True):
        is_error = _wer(str(r.get("text", "")), " ".join(ws)) > error_wer
        units.append((is_error, flagged, _duration(r)))
    errors = sum(1 for e, _f, _d in units if e)
    flagged_n = sum(1 for _e, f, _d in units if f)
    flagged_errors = sum(1 for e, f, _d in units if e and f)
    unflagged_error_s = sum(d for e, f, d in units if e and not f)
    return {
        "unflagged_error_s": unflagged_error_s,
        "flag_precision": flagged_errors / flagged_n if flagged_n else None,
        "flag_recall": flagged_errors / errors if errors else None,
    }


def _join_text(segments: list[Segment]) -> str:
    return " ".join(str(s.get("text", "")) for s in segments)


def transcript_wer(ref: list[Segment], hyp: list[Segment]) -> float:
    """WER over the whole excerpt: joined reference vs joined hypothesis text."""
    return _wer(_join_text(ref), _join_text(hyp))


def score_excerpt(
    ref_segments: list[Segment],
    hyp_segments: list[Segment],
    *,
    title: str,
    title_expect: dict[str, list[str]],
    glossary_terms: list[str],
) -> dict[str, Any]:
    """All accept metrics for one excerpt as one flat dict of stable keys."""
    acc = speaker_accuracy(ref_segments, hyp_segments)
    split, merge = split_merge_counts(ref_segments, hyp_segments)
    quality = flag_quality(ref_segments, hyp_segments)
    titles = title_hits(title, title_expect)
    return {
        "speaker_name_acc": acc["name_acc"],
        "speaker_mapped_acc": acc["mapped_acc"],
        "split_count": split,
        "merge_count": merge,
        "glossary_insertions": glossary_insertions(
            _join_text(ref_segments), _join_text(hyp_segments), glossary_terms
        ),
        "loop_seconds": loop_seconds(hyp_segments),
        "unflagged_error_s": quality["unflagged_error_s"],
        "flag_precision": quality["flag_precision"],
        "flag_recall": quality["flag_recall"],
        "wer": transcript_wer(ref_segments, hyp_segments),
        "title_must_contain_missing": titles["must_contain_missing"],
        "title_forbid_hits": titles["forbid_hits"],
    }
