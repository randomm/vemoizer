"""Token-level guard on an accepted LLM repair (issue #272).

The paragraph-level guard in :mod:`vemoizer.repair` (length growth + whole
similarity) lets a long paragraph through even when single words were
swapped for unrelated ones — glossary over-steering ("IBS" ->
"IBC-kontti"), words inserted that nobody spoke, or a run of content
dropped. This module re-walks an accepted candidate token by token and
reverts exactly those edits, keeping every change that looks phonetic.

Pure function over two strings: no model, no network, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .slice_align import slice_similarity
from .textnorm import textnorm

#: A replace whose runs are less similar than this swapped in an unrelated
#: word. A phonetic fix ("parastaa" -> "parantaa") keeps most of its
#: characters; "IBS" -> "IBC-kontti" does not, and no model may decide the
#: meaning of a term it never heard spoken.
TOKEN_MIN_SIMILARITY = 0.5

#: Filler removal is one or two words ("niinku", "että niin"); a longer
#: deleted run is content the model dropped, so it ships back.
MAX_DELETE_RUN = 2

#: What the repair prompt is told to write for an unreadable passage — the
#: only edit the guard never questions, because guessing is worse.
_MARKER_TEXTNORM = frozenset({"epäselvä", "unclear"})
_MARKER_RAW = ("[epäselvä]", "[unclear]")


@dataclass(frozen=True)
class GuardStats:
    """What the token guard did to one paragraph (counts only, never text).

    ``kept`` counts the replaced runs whose candidate side survived —
    equal runs are not counted, they are not decisions.
    """

    kept: int = 0
    reverted_replace: int = 0
    reverted_insert: int = 0
    reverted_delete: int = 0


def _is_marker(candidate_run: list[str]) -> bool:
    """Whether the candidate run is the prompt's unclear-marker."""
    joined = " ".join(candidate_run)
    return textnorm(joined) in _MARKER_TEXTNORM or any(
        marker in joined for marker in _MARKER_RAW
    )


def _run_is_kept(orig_run: list[str], cand_run: list[str]) -> bool:
    """Whether a replaced run earns its place: the unclear marker or a
    run similar enough to be the same words, not a different term."""
    return _is_marker(cand_run) or (
        slice_similarity(" ".join(orig_run), " ".join(cand_run)) >= TOKEN_MIN_SIMILARITY
    )


def guard_tokens(original: str, candidate: str) -> tuple[str, GuardStats]:
    """Revert the un-phonetic edits of an accepted *candidate* over *original*.

    Both sides are tokenised on whitespace (punctuation stays attached) and
    aligned on their normalized forms, so a capitalization or punctuation
    change is an ``equal`` opcode and ships as the candidate wrote it. An
    empty side is not a repair at all: the *original* is returned untouched.
    """
    if not original.strip() or not candidate.strip():
        return original, GuardStats()
    orig_tokens = original.split()
    cand_tokens = candidate.split()
    matcher = SequenceMatcher(
        None,
        [textnorm(t) for t in orig_tokens],
        [textnorm(t) for t in cand_tokens],
        autojunk=False,
    )
    out: list[str] = []
    kept = 0
    reverted_replace = 0
    reverted_insert = 0
    reverted_delete = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(cand_tokens[j1:j2])
        elif tag == "insert":
            reverted_insert += 1  # words nobody spoke: emit nothing
        elif tag == "delete":
            if i2 - i1 > MAX_DELETE_RUN:
                out.extend(orig_tokens[i1:i2])  # content, not filler
                reverted_delete += 1
        else:  # replace
            orig_run = orig_tokens[i1:i2]
            cand_run = cand_tokens[j1:j2]
            if len(orig_run) == len(cand_run):
                # SequenceMatcher merges neighbouring edits into one run,
                # so an unrelated swap must not ride along on the run's
                # aggregate similarity: judge the pairs 1:1 with the same
                # rule and count each decision individually.
                for orig_tok, cand_tok in zip(orig_run, cand_run, strict=True):
                    if _run_is_kept([orig_tok], [cand_tok]):
                        out.append(cand_tok)
                        kept += 1
                    else:
                        out.append(orig_tok)
                        reverted_replace += 1
            elif _run_is_kept(orig_run, cand_run):
                out.extend(cand_run)
                kept += 1
            else:
                out.extend(orig_run)
                reverted_replace += 1
    return " ".join(out), GuardStats(
        kept=kept,
        reverted_replace=reverted_replace,
        reverted_insert=reverted_insert,
        reverted_delete=reverted_delete,
    )
