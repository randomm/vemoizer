"""Append operator-confirmed terms to the project glossary (issue #268).

The interactive term check saves what the operator confirmed or corrected
into the nearest project ``.vemoizer/glossary.txt`` — the layer the next
run in this folder reads first. The home glossary is never the target:
:func:`vemoizer.glossary_layers._nearest_project_glossary` never returns
it, and the fallback is ``<cwd>/.vemoizer/glossary.txt``.
"""

from __future__ import annotations

from pathlib import Path

from vemoizer.glossary_layers import _home_glossary_path, _nearest_project_glossary
from vemoizer.render_cli import _atomic_write_text


class HomeGlossaryWriteError(ValueError):
    """The project fallback resolves to the home glossary (cwd == $HOME).

    The term check must never write the home layer (issue #268 review):
    running it from ``$HOME`` with no project glossary would otherwise
    silently append to ``~/.vemoizer/glossary.txt``.
    """


def project_glossary_path(start: Path | None = None) -> Path:
    """Where :func:`append_to_project_glossary` would write (no creation)."""
    found = _nearest_project_glossary(start)
    if found is not None:
        return found
    return (start or Path.cwd()) / ".vemoizer" / "glossary.txt"


def _existing_keys(text: str) -> set[str]:
    """Case-folded identity of each line already in the glossary: bare
    terms by themselves, ``a => b`` pairs by their wrong side."""
    keys: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        wrong, arrow, _right = line.partition("=>")
        keys.add((wrong if arrow else line).strip().lower())
    return keys


def _acceptable(line: str) -> bool:
    """A saveable line: non-empty, and a pair only without ``*`` on the
    wrong side (wildcard pairs are hand-written policy, never machine
    output)."""
    if not line:
        return False
    wrong, arrow, right = line.partition("=>")
    if arrow:
        return bool(wrong.strip()) and bool(right.strip()) and "*" not in wrong
    return True


def append_to_project_glossary(lines: list[str], start: Path | None = None) -> Path:
    """Append *lines* to the project glossary; return its path.

    Duplicates are skipped: bare terms case-insensitively, pairs by their
    wrong side case-insensitively (against bare terms and wrong sides
    alike). The write is atomic (same-directory temp file + ``os.replace``
    via :func:`vemoizer.render_cli._atomic_write_text`). Nothing is written
    — and no ``.vemoizer/`` is created — when there is nothing new to add.

    Raises :class:`HomeGlossaryWriteError` when the target would be the
    home glossary (cwd is ``$HOME`` and no project glossary was found).
    """
    target = project_glossary_path(start)
    if target.resolve() == _home_glossary_path().resolve():
        raise HomeGlossaryWriteError(
            f"project glossary fallback resolves to the home layer: {target}"
        )
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    keys = _existing_keys(existing)
    new: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not _acceptable(line):
            continue
        wrong, arrow, _right = line.partition("=>")
        key = (wrong if arrow else line).strip().lower()
        if key in keys:
            continue
        keys.add(key)
        new.append(line)
    if not new:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    content = existing
    if content and not content.endswith("\n"):
        content += "\n"
    _atomic_write_text(target, content + "\n".join(new) + "\n")
    return target
