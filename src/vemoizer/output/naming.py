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

import re
import unicodedata
from datetime import date
from pathlib import Path

#: Unicode normalization form every path in/out of this process is held in.
_NFC = "NFC"

#: Maximum length of a dated-title base (everything after the date).
_MAX_BASE_CHARS = 80

#: ``"word  word"`` — multiple internal spaces collapse to one.
_WHITESPACE_RUN = re.compile(r"\s{2,}")


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


def sanitize_title(raw: str) -> str:
    """Sanitize an LLM title for use in a dated output filename.

    Path separators and filesystem-invalid characters are removed (the
    others are ``?*\"<>|`` plus the control characters, which include the
    zero-width joiners and the BOM); ``:`` — the legacy HFS path separator
    and invalid on Windows/SMB — is mapped to an en dash with surrounding
    spaces. Leading/trailing dots and spaces are removed, internal
    whitespace collapses to single spaces, and the result is NFC and
    capped at 80 characters. An empty result (blank or fully stripped
    input) returns ``""`` so the caller can fall back to a deterministic
    stem.
    """
    t = unicodedata.normalize("NFC", str(raw))
    # Path separators and control characters are dropped entirely, so
    # "a/b" becomes "ab" (not "a b"); zero-width joiners and the BOM
    # are the same — invisible characters that must not survive into a
    # filename.
    # ":" is the legacy HFS path separator on macOS (Finder renders it
    # as "/") and is invalid on Windows/SMB, so it becomes an en dash
    # with surrounding spaces ("Planning: X" -> "Planning – X"); the
    # other Windows-invalid characters are dropped (issue #110).
    t = re.sub(r":", " \u2013 ", t)
    t = re.sub(r"[/\\?*\"<>|\u0000-\u001f\u007f\u200b-\u200f\ufeff]", "", t)
    t = _WHITESPACE_RUN.sub(" ", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = t.strip(" .")
    return t[:_MAX_BASE_CHARS]


def dated_basename(
    title: str,
    *,
    date_str: str | None = None,
    fallback_stem: str | None = None,
) -> str:
    """Build the dated output base name ``YYYY-MM-DD <title>``.

    ``title`` is sanitized; if sanitising leaves nothing, *fallback_stem*
    (the first source file stem) is used, also sanitized. Returns the
    base without any suffix, so the caller appends ``.md`` / ``.json``.
    """
    d = date_str or date.today().isoformat()
    base = sanitize_title(title) or sanitize_title(fallback_stem or "")
    if not base:
        raise ValueError("dated_basename: title and fallback_stem both empty")
    return f"{d} {base}"


def collision_free_paths(
    directory: Path | str, base: str, suffixes: list[str]
) -> tuple[Path, ...]:
    """Return one non-colliding path per suffix, ALL sharing one stem.

    The *suffixes* are probed as a unit: any taken name (``X.md`` or
    ``X.json``) bumps the whole pair to `` (2)``, never one file at a
    time (an ``X.md`` + ``X (2).json`` pair would look like two different
    runs of the same meeting). Every candidate is NFC-normalised before
    probing, so an APFS NFD spelling of an existing file is a collision
    (never overwritten). ``collision_free_path`` is the single-suffix
    convenience wrapper over this.
    """
    dir_path = nfc_path(Path(directory))
    base = nfc(base)
    nfc_suffixes = [nfc(s) for s in suffixes]

    def _all_free(stem: str) -> bool:
        return all((dir_path / f"{stem}{s}").exists() is False for s in nfc_suffixes)

    if _all_free(base):
        n = 1
    else:
        n = 2
        while True:
            if _all_free(f"{base} ({n})"):
                break
            n += 1
    stem = base if n == 1 else f"{base} ({n})"
    return tuple(dir_path / f"{stem}{s}" for s in nfc_suffixes)


def collision_free_path(directory: Path | str, base: str, suffix: str) -> Path:
    """Return a non-colliding path in *directory* for *base* + *suffix*.

    The directory and every candidate name are NFC-normalised before
    probing, so an APFS NFD spelling of an existing file is treated as a
    collision (never overwritten). The first free name is returned;
    otherwise `` (2)``, `` (3)``, ... is inserted before the suffix.
    Candidates are checked against the real filesystem via
    ``Path.exists``. The returned path is NFC.
    """
    return collision_free_paths(directory, base, [suffix])[0]
