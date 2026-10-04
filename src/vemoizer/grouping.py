"""Split-recording grouping with confirmation (issue #77, M3).

The iPhone dictaphone splits long meetings into ``Uusi äänitys
425..429.m4a`` files. This module decides which consecutive files belong
to one logical recording and joins the accepted groups:

- **Natural sort**: NFC-normalised stem, trailing integer as the numeric
  key (never file metadata — a later ``creation_time`` must not change
  the order).
- **Advisory heuristic**: the last 20 s of file N and the first 20 s of
  file N+1 are decoded with Whisper (a separate boundary model) and
  :func:`propose_groups` maps each boundary to a :class:`GroupProposal`
  (continue vs. break) from closing-cue matching over
  ``textnorm``-normalised text. Silent, truncated, or failed boundaries
  degrade to "no evidence" (a break) — never a crash, never a false
  continuation.
- **Confirmation**: an interactive ``e``/Edit partition (any valid
  partition of the sorted stems), ``--yes`` (accept every proposal;
  boundary decodes still run), or ``--no-group`` (no grouping at all).
  The non-TTY guard lives in the CLI before any decode — a piped/CI
  invocation must never hang on ``input()``.
- **Concat**: the ffmpeg concat demuxer with ``-c copy``; the list file
  single-quote-escapes each path (apostrophes in filenames survive); a
  per-part ffprobe check turns a codec mismatch into a clear error
  naming the offending files. Part offsets come from decoded PCM
  (``pcm_duration_seconds``), never ffprobe or container metadata.

The heuristic is advisory: an explicit user partition always wins.

The impure halves of this module live in dedicated submodules:

- :mod:`vemoizer.grouping_decode` — the 20 s edge-window ffmpeg decode
  (``decode_boundaries``, ``_decode_edge_window``).
- :mod:`vemoizer.grouping_concat` — the ffmpeg concat demuxer
  (``concat_groups``, ``remove_concat_output``) and part offsets
  (``part_offsets``).
- :mod:`vemoizer.grouping_probe` — the ffprobe duration probe
  (``probe_duration_seconds``) and stream signature (``_probe_stream``).

``pcm_duration_seconds`` (the streaming decoded-PCM duration that
``part_offsets`` uses) lives in :mod:`vemoizer.ingest` and is
re-exported from here for the same monkeypatch-targeting reason as the
others.

All are re-exported from this module for backwards compatibility with
existing imports and ``monkeypatch.setattr("vemoizer.grouping.X", ...)``
targets.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from vemoizer.grouping_concat import (  # noqa: F401
    concat_groups,
    part_offsets,
    remove_concat_output,
)
from vemoizer.grouping_decode import (  # noqa: F401
    decode_boundaries,
)
from vemoizer.grouping_probe import (  # noqa: F401
    _probe_stream,
    probe_duration_seconds,
)
from vemoizer.ingest import (  # noqa: F401  (re-export)
    IngestError,
    duration_seconds,
    ingest_audio,
    pcm_duration_seconds,
)
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.textnorm import textnorm

# Re-export the shared data types (now in grouping_common.py).
from .grouping_common import PartMarker, PartOffset  # noqa: F401  (re-export)


def _decode_edge_window(
    path: Path, start: float, end: float, *, preprocess: str | None = None
) -> np.ndarray:
    """Indirection for :func:`grouping_decode._decode_edge_window`.

    Defined here (not imported) so that ``monkeypatch.setattr(grouping,
    "_decode_edge_window", fake)`` in tests patches the name that
    ``decode_boundaries`` actually calls (via this local name).
    """
    from .grouping_decode import _decode_edge_window as _dew

    return _dew(path, start, end, preprocess=preprocess)


#: Seconds of each file edge decoded for the continuation probe.
BOUNDARY_SECONDS = 20.0

#: Word budget per edge snippet in the evidence — the boundary text is
#: Whisper output, so an unbounded snippet could balloon the confirmation
#: prompt; the budget is on words, not characters.
EDGE_SNIPPET_WORDS = 8

#: Closing cues, seeded from the September corpus (owner-reviewable).
#: ``textnorm`` casefolds, so every entry is pre-casefolded and
#: punctuation-free.
FI_CLOSING_CUES: tuple[str, ...] = (
    "kiitos",
    "kiitoksia",
    "moi",
    "moro",
    "hei hei",
    "nähdään",
    "näkoon",
    "hyvästi",
    "loppu tässä",
    "tässä oli",
    "kahvia",
    "kahvihuoneeseen",
    "tilaan viereen",
    "huoneesta ulos",
)
EN_CLOSING_CUES: tuple[str, ...] = (
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "see you",
    "coffee break",
    "back in a few",
)

#: A cue must sit in the last N words of the edge to count as a closing
#: cue (a "kiitos" mid-sentence does not end a recording).
_CUE_TAIL_WINDOW = 3

#: The trailing integer of the stem ("Uusi äänitys 425" -> 425).
_STEM_TAIL_RE = re.compile(r"(\d+)\s*$")


class GroupingError(Exception):
    """User-visible grouping failure (malformed partition, concat error)."""


# PartOffset is re-exported from grouping_common for backwards compat.


@dataclass(frozen=True)
class GroupProposal:
    """Advisory verdict for one consecutive-file boundary.

    ``parts`` is the file pair the boundary sits between. ``evidence`` is
    the ``(tail_snippet, head_snippet)`` pair — "no evidence" (``"(no
    audio)"``) when the boundary degraded (silence, truncation, decode
    failure).
    """

    parts: tuple[str, str]
    is_continuation: bool
    evidence: tuple[str, str]


# ---------------------------------------------------------------------------
# Natural sort
# ---------------------------------------------------------------------------


def natural_sort(files: Sequence[Path | str]) -> list[Path]:
    """Sort files by the NFC-normalised trailing integer of the stem.

    ``Uusi äänitys 42.m4a`` comes before ``Uusi äänitys 425.m4a``
    (numeric, not lexicographic). Files without a trailing integer sort
    by their stem string and stay before the numbered ones. File
    metadata (``creation_time`` et al.) is never consulted, so a later
    ``creation_time`` on the higher-numbered file cannot reorder them.
    """
    return sorted(
        (Path(p) for p in files), key=lambda p: _sort_key(nfc_stem_and_suffix(p)[0])
    )


def _sort_key(stem: str) -> tuple[int, int | str, str]:
    """`(numbered?, key, stem)`: numeric beats string, stable on the stem.`"""
    match = _STEM_TAIL_RE.search(stem)
    if match:
        return (1, int(match.group(1)), stem)
    return (0, stem, stem)


def stems_of(files: Sequence[Path | str]) -> list[str]:
    """The NFC-normalised stems of *files*, in natural-sort order."""
    return [nfc_stem_and_suffix(p)[0] for p in natural_sort(files)]


# ---------------------------------------------------------------------------
# Pure heuristic
# ---------------------------------------------------------------------------


def edge_snippets(tail_text: str, head_text: str) -> tuple[str, str]:
    """Bounded evidence snippets for a boundary (pure, no I/O).

    Each side is trimmed to at most :data:`EDGE_SNIPPET_WORDS` words
    (word count, not character count). Either side empty (silent
    boundary or decode failure) yields ``"(no audio)"`` — not an
    exception, not a false continuation.
    """
    tail = textnorm(tail_text).split()
    head = textnorm(head_text).split()
    tail_s = " ".join(tail[-EDGE_SNIPPET_WORDS:])
    head_s = " ".join(head[-EDGE_SNIPPET_WORDS:])
    if not tail_s and not head_s:
        return ("(no audio)", "(no audio)")
    return (tail_s or "(no audio)", head_s or "(no audio)")


def _matching_cue(text: str, cues: tuple[str, ...]) -> str | None:
    """The first cue found in the last N words of *text*.

    *text* must already be ``textnorm``-normalised; matching is
    substring-based over the tail window, so "kiitos" inside the garbled
    "kiitost" still counts — a Whisper garble must not be missed.
    """
    tail_text = " ".join(text.split()[-_CUE_TAIL_WINDOW:])
    for cue in cues:
        if cue in tail_text:
            return cue
    return None


def propose_groups(
    files: Sequence[Path | str],
    tail_texts: Sequence[str],
    head_texts: Sequence[str],
) -> list[GroupProposal]:
    """Map each consecutive-file boundary to an advisory proposal (pure).

    ``tail_texts[i]`` is the decoded last 20 s of file i and
    ``head_texts[i]`` the decoded first 20 s of file i+1 — both already
    decoded, so this function does no I/O, loads no model, and never
    touches the clock. The heuristic, over ``textnorm``-normalised text:

    - an empty edge (``""`` — silence, truncation, or decode failure)
      -> no evidence -> break;
    - a closing cue in the last N words of either edge -> break,
      regardless of everything else;
    - a mid-sentence tail (no cue) -> continuation.
    """
    ordered = natural_sort(files)
    expected = len(ordered) - 1
    if len(tail_texts) != expected or len(head_texts) != expected:
        raise ValueError(
            "tail_texts and head_texts must have len(files) - 1 entries "
            f"(got {len(tail_texts)} tail / {len(head_texts)} head for "
            f"{len(ordered)} files)"
        )

    cues = FI_CLOSING_CUES + EN_CLOSING_CUES
    proposals: list[GroupProposal] = []
    for i, tail_text in enumerate(tail_texts):
        head_text = head_texts[i]
        tail_norm = textnorm(tail_text)
        head_norm = textnorm(head_text)
        tail_snip, head_snip = edge_snippets(tail_text, head_text)
        if not tail_norm or not head_norm:
            # No evidence: a break, never a false continuation.
            is_continuation = False
        else:
            is_continuation = (
                _matching_cue(tail_norm, cues) is None
                and _matching_cue(head_norm, cues) is None
            )
        proposals.append(
            GroupProposal(
                parts=(ordered[i].name, ordered[i + 1].name),
                is_continuation=is_continuation,
                evidence=(tail_snip, head_snip),
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# Edit-string partition parsing
# ---------------------------------------------------------------------------


def parse_partition(s: str, stems: Sequence[str]) -> list[list[Path]]:
    """Parse an ``e`` edit partition over the sorted stems.

    Groups are separated by whitespace and/or ``|``; ``+`` joins parts
    *within* a group. Every stem must appear exactly once, in natural
    order. The full stem is always a valid token; a trailing integer is
    accepted only when it matches exactly ONE stem — a token matching
    several stems raises an ``ambiguous part`` error (type the full
    stem). Other malformed tokens (unknown stem, duplicate, out-of-
    order, missing part) raise :class:`GroupingError` naming the
    offending token verbatim. The heuristic proposal is advisory — any
    valid partition wins.

    Ambiguity is detected over the full stem set *before* the
    duplicate check: a trailing integer matching two different stems is
    ``ambiguous``, not a duplicate (the duplicate check only fires when
    the *same* stem is referenced twice).
    """
    groups: list[list[str]] = []
    seen: set[str] = set()
    next_index = 0

    def _lookup(tok: str) -> str:
        """Resolve *tok* to a stem, raising a clear error on ambiguity."""
        if tok in stems:
            if tok in seen:
                raise GroupingError(f"duplicate part: {tok!r}")
            return tok
        matches = [
            st
            for st in stems
            if (m := _STEM_TAIL_RE.search(st)) is not None and m.group(1) == tok
        ]
        if len(matches) > 1:
            # Ambiguity beats the duplicate check; see parse_partition.
            # Each matching stem is rendered with repr and joined by ', '
            # — the raw list repr ("['a', 'b']") was a debugging artifact
            # that leaked into the user-facing message.
            rendered = ", ".join(repr(m) for m in matches)
            raise GroupingError(
                f"ambiguous part: {tok!r} matches {rendered}; type the full stem"
            )
        if len(matches) == 1:
            stem = matches[0]
            if stem in seen:
                raise GroupingError(f"duplicate part: {tok!r}")
            return stem
        raise GroupingError(f"unknown part: {tok!r}")

    for raw in re.split(r"[|\s]+", s.strip()):
        if not raw:
            continue
        group: list[str] = []
        for tok in (t.strip() for t in raw.split("+")):
            if not tok:
                raise GroupingError(f"empty part in token {raw!r}")
            stem = _lookup(tok)
            seen.add(stem)
            if stems.index(stem) != next_index:
                raise GroupingError(
                    f"out-of-order part: {tok!r} (expected {stems[next_index]!r})"
                )
            next_index += 1
            group.append(stem)
        groups.append(group)
    if next_index != len(stems):
        missing = [st for st in stems if st not in seen]
        raise GroupingError(f"missing part(s): {', '.join(missing)!r}")
    return [[Path(st) for st in group] for group in groups]


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


def _proposals_to_groups(
    ordered: list[Path], proposals: Sequence[GroupProposal]
) -> list[list[Path]]:
    """Fold accepted proposals into groups."""
    groups: list[list[Path]] = []
    current: list[Path] = [ordered[0]]
    for i, proposal in enumerate(proposals):
        if proposal.is_continuation:
            current.append(ordered[i + 1])
        else:
            groups.append(current)
            current = [ordered[i + 1]]
    groups.append(current)
    return groups


def confirm_groups(
    files: Sequence[Path | str],
    proposals: Sequence[GroupProposal],
    *,
    yes: bool = False,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[list[Path]]:
    """Confirm the grouping decision; returns groups in natural order.

    ``--yes``: every proposal is accepted as-is. Interactive: one prompt
    per boundary (Enter accept, ``e`` for a full partition edit, ``q``
    quit) — ``q`` aborts the run with a :class:`GroupingError`, and the
    edit path accepts ANY valid partition. The ``--no-group`` decision
    (no grouping at all, each file standalone) lives in
    :func:`vemoizer.batch.run_batch`, which handles it before any
    boundary decode and never reaches this function with it set. The
    non-TTY guard lives in the CLI, before any boundary decode, so a
    piped/CI invocation fails fast instead of hanging.
    """
    ordered = natural_sort(files)
    if yes:
        return _proposals_to_groups(ordered, proposals)

    groups: list[list[Path]] = []
    current: list[Path] = [ordered[0]]
    for i, proposal in enumerate(proposals):
        tail_snip, head_snip = proposal.evidence
        verdict = "continue" if proposal.is_continuation else "break"
        print_fn(
            f"{proposal.parts[0]} | {proposal.parts[1]}\n"
            f"  tail: {tail_snip}\n"
            f"  head: {head_snip}\n"
            f"  proposed: {verdict}"
        )
        answer = input_fn("[Enter] accept, (e)dit partition, (q)uit: ").strip().lower()
        if answer.startswith("q"):
            raise GroupingError("user aborted grouping (q)")
        if answer.startswith("e"):
            edit = input_fn("Partition (e.g. 425 | 426 | 427+428+429): ")
            return parse_partition(edit, stems_of(ordered))
        if proposal.is_continuation:
            current.append(ordered[i + 1])
        else:
            groups.append(current)
            current = [ordered[i + 1]]
    groups.append(current)
    return groups
