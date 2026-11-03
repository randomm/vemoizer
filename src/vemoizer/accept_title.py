"""Title checks for the accept benchmark (issue #228).

A reviewed reference says what the pipeline's title must and must not
contain; this module turns that expectation into two counts. Pure string
work — no IO, no models — so it stays separate from the segment metrics in
:mod:`vemoizer.accept_metrics`, which are time-grid computations.
"""

from __future__ import annotations

from .textnorm import textnorm


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
