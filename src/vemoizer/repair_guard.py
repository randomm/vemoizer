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
from typing import Any

from .slice_align import slice_similarity
from .textnorm import textnorm

#: A replace whose runs are less similar than this swapped in an unrelated
#: word. A phonetic fix ("parastaa" -> "parantaa") keeps most of its
#: characters; "IBS" -> "IBC-kontti" does not, and no model may decide the
#: meaning of a term it never heard spoken.
TOKEN_MIN_SIMILARITY = 0.5

#: A kept replace that scores this low passed the gate, but only just: it
#: is likely a real word, just not a close rendering of the spoken one, so
#: the audit flags it for a human glance rather than trusting it blindly.
WEAK_SIMILARITY = 0.7

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


def _run_similarity(orig_run: list[str], cand_run: list[str]) -> float:
    """How much of the original a replaced run kept — its keep decision.

    The unclear marker scores 1.0 rather than its own character overlap:
    its letters are not a phonetic signal at all (the prompt writes it for
    a passage nobody can read), and the audit must never flag it as weak.
    """
    if _is_marker(cand_run):
        return 1.0
    return slice_similarity(" ".join(orig_run), " ".join(cand_run))


def guard_tokens(
    original: str, candidate: str
) -> tuple[str, GuardStats, list[dict[str, Any]]]:
    """Revert the un-phonetic edits of an accepted *candidate* over *original*.

    Both sides are tokenised on whitespace (punctuation stays attached) and
    aligned on their normalized forms, so a capitalization or punctuation
    change is an ``equal`` opcode and ships as the candidate wrote it. An
    empty side is not a repair at all: the *original* is returned untouched.

    Returns ``(text, stats, edits)``: *edits* are the changes the guard
    KEPT, in order, as ``{"from": orig run, "to": candidate run, "weak":
    bool}`` (a kept filler deletion has ``"to": ""``). Equal runs are not
    edits and reverted ones are not kept, so neither is recorded; the audit
    is what lets a reader check a repair after the fact (issue #274).
    """
    if not original.strip() or not candidate.strip():
        return original, GuardStats(), []
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
    edits: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(cand_tokens[j1:j2])
        elif tag == "insert":
            reverted_insert += 1  # words nobody spoke: emit nothing
        elif tag == "delete":
            if i2 - i1 > MAX_DELETE_RUN:
                out.extend(orig_tokens[i1:i2])  # content, not filler
                reverted_delete += 1
            else:
                edits.append(
                    {"from": " ".join(orig_tokens[i1:i2]), "to": "", "weak": False}
                )
        else:  # replace
            orig_run = orig_tokens[i1:i2]
            cand_run = cand_tokens[j1:j2]
            if len(orig_run) == len(cand_run):
                # SequenceMatcher merges neighbouring edits into one run,
                # so an unrelated swap must not ride along on the run's
                # aggregate similarity: judge the pairs 1:1 with the same
                # rule and count each decision individually.
                for orig_tok, cand_tok in zip(orig_run, cand_run, strict=True):
                    sim = _run_similarity([orig_tok], [cand_tok])
                    if sim >= TOKEN_MIN_SIMILARITY:
                        out.append(cand_tok)
                        kept += 1
                        edits.append(
                            {
                                "from": orig_tok,
                                "to": cand_tok,
                                "weak": sim < WEAK_SIMILARITY,
                            }
                        )
                    else:
                        out.append(orig_tok)
                        reverted_replace += 1
            else:
                sim = _run_similarity(orig_run, cand_run)
                if sim >= TOKEN_MIN_SIMILARITY:
                    out.extend(cand_run)
                    kept += 1
                    edits.append(
                        {
                            "from": " ".join(orig_run),
                            "to": " ".join(cand_run),
                            "weak": sim < WEAK_SIMILARITY,
                        }
                    )
                else:
                    out.extend(orig_run)
                    reverted_replace += 1
    return (
        " ".join(out),
        GuardStats(
            kept=kept,
            reverted_replace=reverted_replace,
            reverted_insert=reverted_insert,
            reverted_delete=reverted_delete,
        ),
        edits,
    )
