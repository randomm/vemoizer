"""NFC filename normalization for output paths (issue #10, task C).

macOS APFS stores filenames in NFD (decomposed) form while Python strings
are conventionally NFC (composed): ``"mó"`` in memory is ``"mo\\u0301"``
on disk. The OS folds the two spellings for lookup, so they name the same
file — but ``"mó" != "mo\\u0301"`` as Python strings. A batch run that
deduplicates or compares output paths will treat the two spellings as
different files and either collide or silently produce a double output
with one NFC and one NFD spelling of the same basename.

The fix: normalize every filename that enters or leaves this process to
NFC. NFC is stable (``normalize('NFC', normalize('NFC', x)) ==
normalize('NFC', x)``), it is the composed canonical form documented by
``unicodedata``, and it is the form the OS uses for *display* — so an NFC
string is the one a user actually typed or will recognize.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

#: Unicode normalization form every path in/out of this process is held in.
_NFC = "NFC"


def nfc(name: str) -> str:
    """Return *name* in NFC (composed) form.

    Idempotent: an already-NFC string passes through unchanged (the same
    normalized form, recomputed by ``unicodedata`` — no object identity
    guarantee is made or needed). This is a pure string function; it
    does not touch the filesystem, so it is safe on names that do not
    exist yet (the common case for an output path we are about to
    create).
    """
    return unicodedata.normalize(_NFC, name)


def nfc_path(path: Path | str) -> Path:
    """Return a copy of *path* with every component in NFC form.

    ``nfc_path(Path("m´emo/notes/memo.m4a"))`` yields a path whose
    directory components and filename are all NFC, so two spellings of
    the same on-disk file produce equal ``Path`` objects and hash
    identically — the property a batch dedup or output-collision check
    needs.

    An anchor (``/``, a Windows drive letter) is passed through
    unchanged: it is never subject to Unicode folding.
    """
    p = Path(path)
    parts = list(p.parts)
    if parts:
        first = parts[0]
        # ``Path.parts`` of an absolute POSIX path starts with ``/``;
        # Windows paths start with a drive anchor like ``C:\\``. Either
        # way the anchor is not a filename and must not be composed.
        if (len(first) == 1 and not first.isalnum()) or (
            len(first) == 2 and first[1] in "\\/"
        ):
            parts[1:] = [nfc(part) for part in parts[1:]]
        else:
            parts = [nfc(part) for part in parts]
    return Path(*parts) if parts else Path()


def nfc_stem_and_suffix(path: Path | str) -> tuple[str, str]:
    """Return the NFC-normalized ``(stem, suffix)`` of a single filename.

    ``nfc_stem_and_suffix("m´emo.m4a")`` → ``("mó", ".m4a")``. The suffix
    is normalized too because an extension can carry a combining mark
    (``.naïve`` is a legal filename). The two pieces rejoin to the full
    NFC name: ``stem + suffix == nfc(name)``.
    """
    p = Path(path)
    return nfc(p.stem), nfc(p.suffix)
