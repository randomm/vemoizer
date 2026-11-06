"""Term-shaped word selection for the interactive term check (issue #268).

The meeting sidecar (issue #263) carries ``words`` — Whisper per-word
probabilities for the words inside the kept segments — next to the final
``paragraphs``. A word the decoder was unsure of AND that looks like a term
(acronym, mixed alphanumeric, rare-letter spelling, mid-sentence capital) is
something only the operator can confirm, so ``vemoizer terms`` lists them
least-sure first and the operator hears and fixes them one by one
(:mod:`vemoizer.term_check_cli`). Pure functions here: no prompt, no writes,
``LIST_LIMIT`` the only knob.

Calibration note: a real 48-min call showed Whisper ``p`` alone does NOT
find the errors (several misrecognitions sat at p 0.7-0.96), so list mode
keeps every term-shaped word with a ``p`` and merely ranks by uncertainty.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

#: How many words the interactive check lists at most.
LIST_LIMIT = 30

#: A word starting this close to a segment start is sentence-initial: the
#: recognizer capitalizes real sentence starts, so ``Joo``/``Päivää`` at a
#: segment boundary are not term-shaped.
_SEGMENT_MATCH_S = 0.05

#: A pause this long between two words starts a new sentence.
_SENTENCE_GAP_S = 1.5

#: Punctuation stripped from token ends. ``-``, ``/`` and ``.`` stay: they are
#: part of term spellings (``Q3-report``, ``and/or``, ``S.R.``).
_EDGE_PUNCTUATION = (
    "".join(c for c in string.punctuation if c not in "-/.")
    + "\u201c\u201d\u2018\u2019\u2026"
)

#: Closing quotes/brackets that may trail a sentence terminator (``?"``).
_CLOSING_MARKS = "\"'\u201d\u2019)]}\u00bb"

#: Sentence terminators, checked after ``_CLOSING_MARKS`` are stripped.
_TERMINATORS = (".", "?", "!")

#: Letters a native Finnish word almost never contains.
_RARE_LETTERS = frozenset("cqxwz")

_ACRONYM = re.compile(r"[A-Z][A-Z0-9]{1,5}")


@dataclass(frozen=True)
class TermCandidate:
    """One word worth checking: *word* is its most frequent surface form and
    *stem* the original-case stem of the group (case suffix stripped, so
    ``AVS:n`` -> ``AVS``) — the glossary records key on it; ``start``/``end``
    locate its LOWEST-``p`` occurrence for playback."""

    word: str
    stem: str
    count: int
    min_p: float
    first_start: float
    score: float
    start: float
    end: float


def normalize_token(raw: str) -> tuple[str, str] | None:
    """``(surface, stem)`` of a token; ``None`` when nothing is left. ``:``
    splits a Finnish case suffix off, so ``"AVS:n,"`` -> ``("AVS:n", "AVS")``.
    """
    # A sentence-final "AVS." must stem to "AVS"; interior dots survive
    # ("S.R." -> "S.R", "v1.2" untouched).
    surface = raw.strip(_EDGE_PUNCTUATION).strip().rstrip(".")
    stem = surface.split(":", 1)[0].strip(_EDGE_PUNCTUATION).strip().rstrip(".")
    return (surface, stem) if stem else None


def _is_acronym(stem: str) -> bool:
    """2-6 letters, all upper case, digits allowed: ``AVS``, ``XOV1``."""
    letters = [c for c in stem if c.isalpha()]
    return (
        2 <= len(letters) <= 6
        and all(c.isupper() for c in letters)
        and _ACRONYM.fullmatch(stem) is not None
    )


def is_term_shaped(stem: str, *, sentence_start: bool) -> bool:
    """Whether *stem* reads as a term, not as ordinary Finnish prose: an
    acronym, a letters+digits mix, or a spelling with ``c q w x z``. A word
    that merely starts with a capital qualifies only when it is not
    sentence-initial: ``Klaudia`` mid-sentence means somebody heard a name.
    """
    letters = [c for c in stem if c.isalpha()]
    if not letters or len(stem) == 1:
        return False
    if _is_acronym(stem) or any(c.isdigit() for c in stem):
        return True
    if any(c.lower() in _RARE_LETTERS for c in letters):
        return True
    return stem[0].isupper() and not sentence_start


def _sentence_starts(
    words: list[dict[str, Any]], segment_starts: list[float] | None = None
) -> list[bool]:
    """Per-word "begins a sentence": previous token ended with ``. ? !``, or
    there is no previous word, or the gap to it exceeds ``_SENTENCE_GAP_S``,
    or the word starts within ``_SEGMENT_MATCH_S`` of a VAD segment start
    (the recognizer capitalizes those, so they are not mid-sentence).
    """
    seg_starts = sorted(segment_starts or [])
    flags: list[bool] = []
    prev_end: float | None = None
    prev_token = ""
    for w in words:
        start = float(w.get("start", 0.0))
        closed = prev_token.rstrip(_CLOSING_MARKS).endswith(_TERMINATORS)
        at_segment_start = any(abs(start - s) <= _SEGMENT_MATCH_S for s in seg_starts)
        flags.append(
            closed
            or prev_end is None
            or start - prev_end > _SENTENCE_GAP_S
            or at_segment_start
        )
        prev_end = float(w.get("end", start))
        prev_token = str(w.get("word", ""))
    return flags


def _appears_in_paragraphs(stem: str, paragraphs: list[dict[str, Any]]) -> bool:
    """Case-insensitive whole-word presence of *stem* in the final text."""
    pattern = re.compile(rf"(?<!\w){re.escape(stem)}(?!\w)", re.IGNORECASE)
    return any(
        pattern.search(str(p.get("text", "")))
        for p in paragraphs
        if isinstance(p, dict)
    )


def select_terms(
    words: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    known: set[str],
    *,
    limit: int = LIST_LIMIT,
    p_max: float | None = None,
    segment_starts: list[float] | None = None,
) -> list[TermCandidate]:
    """The term-shaped, still-present, unknown words, least sure first.

    ``p_max=None`` (the interactive list) keeps every term-shaped group that
    has at least one ``p`` and merely ranks by ``sum(1 - p)``: Whisper's ``p``
    alone does not find the errors, so nothing is filtered away on it. An
    explicit *p_max* drops groups whose best occurrence is at or above it.
    ``[]`` when no word carries a ``p``: nothing to rank, and shape alone
    would nag about every acronym in the memo.
    """
    ordered = sorted(words, key=lambda w: float(w.get("start", 0.0)))
    groups: dict[str, dict[str, Any]] = {}
    for w, sentence_start in zip(
        ordered, _sentence_starts(ordered, segment_starts), strict=True
    ):
        raw = str(w.get("word", ""))
        norm = normalize_token(raw)
        if norm is None or ({norm[1].lower(), raw.lower().strip()} & known):
            continue
        surface, stem = norm
        if not is_term_shaped(stem, sentence_start=sentence_start):
            continue
        entry = groups.setdefault(
            stem.lower(),
            {"stem": stem, "surfaces": Counter(), "occs": [], "start": float("inf")},
        )
        entry["surfaces"][surface] += 1
        if w.get("p") is not None:
            entry["occs"].append(
                (
                    float(w["p"]),
                    float(w.get("start", 0.0)),
                    float(w.get("end", w.get("start", 0.0))),
                )
            )
        entry["start"] = min(entry["start"], float(w.get("start", 0.0)))

    candidates: list[TermCandidate] = []
    for stem, entry in groups.items():
        occs: list[tuple[float, float, float]] = entry["occs"]
        surfaces: Counter = entry["surfaces"]
        if not occs:
            continue
        probs = [o[0] for o in occs]
        if p_max is not None and min(probs) >= p_max:
            continue
        if not _appears_in_paragraphs(stem, paragraphs):
            continue
        # The 1.5 weight is on the ORIGINAL spelling: the group key is lower
        # cased, and an acronym is defined by its upper case.
        weight = 1.5 if _is_acronym(str(entry["stem"])) else 1.0
        score = sum(1.0 - p for p in probs) * weight
        # Lowest-p occurrence wins the playback window (first on ties).
        _p, occ_start, occ_end = min(occs)
        candidates.append(
            TermCandidate(
                word=min(surfaces, key=lambda s: (-surfaces[s], s)),
                stem=str(entry["stem"]),
                count=sum(surfaces.values()),
                min_p=min(probs),
                first_start=entry["start"],
                score=score,
                start=occ_start,
                end=occ_end,
            )
        )
    candidates.sort(key=lambda c: (-c.score, c.first_start))
    return candidates[:limit]


def known_terms(sidecar: dict[str, Any]) -> set[str]:
    """Lower-cased words the run already knew: glossary, people, speakers.
    Each source is read fail-open so one bad file cannot hide the line.
    """
    from vemoizer import glossary_layers, people_config

    known: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            term = value.strip().lstrip("@").strip().lower()
            if term:
                known.add(term)

    try:
        home, home_c, project, project_c = glossary_layers.load_layers()
        terms, corrections, _dropped = glossary_layers.merge(
            project, project_c, home, home_c, tokenizer=None
        )
        for term in list(terms) + list(corrections.keys()) + list(corrections.values()):
            add(term)
    except (OSError, ValueError):
        pass
    try:
        people = people_config.read_people_list(people_config.find_people_config_path())
        for person in people:
            add(person)
    except (OSError, ValueError):
        pass
    names = sidecar.get("speaker_names")
    if isinstance(names, dict):
        for name in names.values():
            add(name)
    return known


# ---------------------------------------------------------------------------
# Pure display formatters for the interactive check (term_check_cli echoes
# their strings; no typer here).
# ---------------------------------------------------------------------------

_PER_ROW = 4


def terms_grid(cands: list[TermCandidate]) -> list[str]:
    """The numbered list, 4 aligned columns per row: ``  1 THS (2×)``."""
    entries = [
        f"{i} {c.word}" + ("" if c.count == 1 else f" ({c.count}\u00d7)")
        for i, c in enumerate(cands, 1)
    ]
    width = max(len(e) for e in entries) + 2
    return [
        "  " + "".join(e.ljust(width) for e in entries[i : i + _PER_ROW]).rstrip()
        for i in range(0, len(entries), _PER_ROW)
    ]


def _hhmmss(t: float) -> str:
    secs = int(t)
    return f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


def _occurrence_index(ordered: list[dict[str, Any]], cand: TermCandidate) -> int | None:
    """Index of the candidate's lowest-``p`` occurrence in the time-sorted
    words (same start, matching stem); ``None`` when not found."""
    key = cand.stem.lower()
    for i, w in enumerate(ordered):
        if abs(float(w.get("start", 0.0)) - cand.start) > 1e-6:
            continue
        norm = normalize_token(str(w.get("word", "")))
        if norm is None or norm[1].lower() == key:
            return i
    return None


def context_line(
    cand: TermCandidate,
    words: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    speaker_names: dict[str, str],
) -> str:
    """``  [HH:MM:SS] Matti: "…before [WORD] after…"`` from sidecar words."""
    ordered = sorted(words, key=lambda w: float(w.get("start", 0.0)))
    surfaces = [str(w.get("word", "")) for w in ordered]
    idx = _occurrence_index(ordered, cand)
    surface = surfaces[idx] if idx is not None else cand.word
    before = surfaces[max(0, idx - 6) : idx] if idx is not None else []
    after = surfaces[idx + 1 : idx + 7] if idx is not None else []
    lead = "\u2026" if idx is not None and idx > 6 else ""
    tail = "\u2026" if idx is not None and len(ordered) - idx - 1 > 6 else ""
    label = ""
    for p in paragraphs:
        if isinstance(p, dict) and float(p.get("start", 0.0)) <= cand.start:
            label = str(p.get("speaker", ""))
    name = speaker_names.get(label, label)
    return (
        f"  [{_hhmmss(cand.start)}] {name}: "
        f'"{lead}{" ".join(before)} [{surface}] {" ".join(after)}{tail}"'
    )
