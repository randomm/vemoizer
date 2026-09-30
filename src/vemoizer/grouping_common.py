"""Shared data types and helpers for the M3 split-recording grouping.

Holds the :class:`PartOffset` dataclass (the sidecar contract for
multi-part groups) and the ``_escape_concat_path`` helper (ffmpeg concat
list file path escaping). Both are needed by :mod:`vemoizer.grouping`
(pure heuristic + dataclasses) and :mod:`vemoizer.grouping_concat`
(ffmpeg concat + part offsets) without creating a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class PartMarker(TypedDict):
    """One entry in a multi-part group's ``transcript["part_markers"]``.

    ``offset`` is the part's start offset in the concatenated group's
    decoded timeline (seconds); ``label`` is the ``— osa N (äänitys X) —``
    marker string. Present only for multi-part groups (issue #77).
    """

    offset: float
    label: str


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


def _escape_concat_path(p: Path) -> str:
    """Escape *p* for an ffmpeg concat list file line (``file '<path>'``).

    Single-quote quoting with ``'`` escaped as ``'\\''`` (POSIX style):
    paths with spaces, Unicode (``ä``/``ö``), or apostrophes
    (``Möös's memo.m4a``) all produce a valid list file.
    """
    return "file '" + str(p).replace("'", "'\\''") + "'"
