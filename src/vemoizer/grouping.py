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
  (``ingest_audio`` + ``duration_seconds``), never ffprobe or container
  metadata.

The heuristic is advisory: an explicit user partition always wins.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.ingest import duration_seconds, ingest_audio
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.redecode import extract_slice
from vemoizer.spans import Span
from vemoizer.textnorm import textnorm

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class PartOffset:
    """One part of a multi-part group: number, source file, start offset.

    ``start_offset`` is cumulative decoded-PCM seconds from the start of
    the concatenated group (part 1 starts at 0.0; part N starts at the
    sum of the decoded durations of parts 1..N-1) — never ffprobe or
    container metadata.
    """

    part_number: int
    source_filename: str
    start_offset: float


@dataclass(frozen=True)
class GroupProposal:
    """Advisory verdict for one consecutive-file boundary.

    ``parts`` is the file pair the boundary sits between. ``evidence`` is
    the ``(tail_snippet, head_snippet)`` pair — "no evidence" (``"(no
    audio)"``) when the boundary degraded (silence, truncation, decode
    failure).
    """

    parts: list[str]
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
                parts=[ordered[i].name, ordered[i + 1].name],
                is_continuation=is_continuation,
                evidence=(tail_snip, head_snip),
            )
        )
    return proposals


# ---------------------------------------------------------------------------
# Impure boundary decode
# ---------------------------------------------------------------------------


def decode_boundaries(
    files: Sequence[Path | str],
    transcribe_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Decode each file's 20 s tail and its next file's 20 s head.

    Returns ``(tail_texts, head_texts)`` for the naturally-sorted input.
    One shared *transcribe_fn* (defaults to a lazily created
    ``WhisperTranscriber(language=None)`` — boundary text can be Finnish
    or English, so no language pinning) is loaded once and cleaned up
    after the last boundary. Any per-slice decode failure degrades that
    edge to ``""`` (no evidence) — never a raise.
    """
    ordered = natural_sort(files)

    transcriber: Any | None = None
    try:
        if transcribe_fn is None:
            from vemoizer.whisper_transcriber import WhisperTranscriber

            transcriber = WhisperTranscriber(language=None)
            fn: Any = transcriber.transcribe
        else:
            fn = transcribe_fn

        def _edge_text(path: Path, start: float, end: float) -> str:
            """Decode ``[start, end)`` seconds of *path*; ``""`` on failure."""
            try:
                audio = ingest_audio(path)
                if len(audio) == 0:
                    return ""
                dur = duration_seconds(audio)
                if start >= dur:
                    return ""
                sliced = extract_slice(
                    audio,
                    Span(start=start, end=min(end, dur)),
                    sample_rate=SAMPLE_RATE,
                )
                if len(sliced) == 0:
                    return ""
                return str(fn(sliced).get("text", "")).strip()
            except Exception as e:  # noqa: BLE001 - fail-open boundary edge
                logger.warning("boundary decode failed for %s: %s", path, e)
                return ""

        tail_texts: list[str] = []
        head_texts: list[str] = []
        for i, path in enumerate(ordered):
            dur = duration_seconds(ingest_audio(path))
            tail_texts.append(_edge_text(path, max(0.0, dur - BOUNDARY_SECONDS), dur))
            if i + 1 < len(ordered):
                # The head is the first 20 s of the NEXT file (the one the
                # boundary leads into), not of the current file.
                head_texts.append(_edge_text(ordered[i + 1], 0.0, BOUNDARY_SECONDS))
        return tail_texts, head_texts
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def _escape_concat_path(p: Path) -> str:
    """Escape *p* for an ffmpeg concat list file line (``file '<path>'``).

    Single-quote quoting with ``'`` escaped as ``'\\''`` (POSIX style):
    paths with spaces, Unicode (``ä``/``ö``), or apostrophes
    (``Möös's memo.m4a``) all produce a valid list file.
    """
    return "file '" + str(p).replace("'", "'\\''") + "'"


def _probe_stream(part: Path) -> str:
    """The part's audio stream signature (codec/sample_rate/channels)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels",
                "-of",
                "csv=p=0",
                str(part),
            ],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        raise GroupingError(
            "ffprobe not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        ) from None
    if proc.returncode != 0:
        raise GroupingError(
            f"ffprobe failed for {part.name} "
            f"(exit {proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return proc.stdout.decode("utf-8", errors="replace").strip()


def concat_groups(group: Sequence[Path | str]) -> Path:
    """Join *group* into one temp .m4a with the ffmpeg concat demuxer.

    ``-c copy`` (no re-encode). The concat demuxer needs every part to
    share the same audio stream, so a per-part ffprobe check runs first
    and a mismatch raises :class:`GroupingError` naming the offending
    files — ``-c copy`` alone would silently corrupt the output. A
    single-file group is a passthrough: no ffmpeg call, no temp file,
    the input itself is returned.
    """
    parts = [Path(p) for p in group]
    if len(parts) == 1:
        return parts[0]

    for part in parts:
        if not part.is_file():
            raise GroupingError(f"concat: part file not found: {part}")

    probes = [(p, _probe_stream(p)) for p in parts]
    (first_part, first_sig) = probes[0]
    mismatched = [(p, sig) for p, sig in probes[1:] if sig != first_sig]
    if mismatched:
        names = ", ".join(p.name for p, _ in probes)
        raise GroupingError(
            f"concat: audio stream mismatch among {names}: "
            f"{first_part.name} has (codec, sample_rate, channels) "
            f"{first_sig!r} but {mismatched[0][0].name} has "
            f"{mismatched[0][1]!r}"
        )

    suffix = parts[0].suffix or ".m4a"
    fd, name = tempfile.mkstemp(prefix="vemoizer-concat-", suffix=suffix)
    os.close(fd)
    out_path = Path(name)
    list_path = Path(name + ".txt")
    try:
        list_path.write_text(
            "".join(_escape_concat_path(p) + "\n" for p in parts),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                "-y",
                str(out_path),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            names = ", ".join(p.name for p in parts)
            raise GroupingError(
                f"concat: ffmpeg failed for {names} (exit {proc.returncode}): {stderr}"
            )
        return out_path
    finally:
        list_path.unlink(missing_ok=True)


def part_offsets(group: Sequence[Path | str]) -> list[PartOffset]:
    """Cumulative decoded-PCM start offset per part of *group*.

    Part 1 starts at 0.0; part N starts at the sum of the decoded
    durations of parts 1..N-1. Durations come from ``ingest_audio``
    (decoded PCM byte count) — never ffprobe or container metadata (iOS
    Voice Memos edit lists make container duration lie).
    """
    parts = [Path(p) for p in group]
    offsets: list[PartOffset] = []
    total = 0.0
    for i, part in enumerate(parts, start=1):
        offsets.append(
            PartOffset(
                part_number=i,
                source_filename=part.name,
                start_offset=total,
            )
        )
        total += duration_seconds(ingest_audio(part))
    return offsets


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
            # Ambiguity check BEFORE the duplicate check: a trailing
            # integer matching two different stems is "ambiguous", not a
            # duplicate (the duplicate check only fires when the SAME
            # stem is referenced twice).
            raise GroupingError(
                f"ambiguous part: {tok!r} matches {', '.join(matches)!r}; "
                "type the full stem"
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
    no_group: bool = False,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[list[Path]]:
    """Confirm the grouping decision; returns groups in natural order.

    ``--no-group``: each file is its own group (the caller skips the
    boundary decodes entirely). ``--yes``: every proposal is accepted
    as-is. Interactive: one prompt per boundary (Enter accept, ``e`` for
    a full partition edit, ``q`` quit) — ``q`` aborts the run with a
    :class:`GroupingError`, and the edit path accepts ANY valid
    partition. The non-TTY guard lives in the CLI, before any boundary
    decode, so a piped/CI invocation fails fast instead of hanging.
    """
    ordered = natural_sort(files)
    if no_group:
        return [[p] for p in ordered]
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
