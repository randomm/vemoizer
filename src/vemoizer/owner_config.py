"""Owner name for two-track calls (issue #171, P5).

The owner is the one speaker the user never has to name by hand. Resolution
order: ``--me`` flag, ``[meeting] owner`` in the project config, the same key
in ``~/.vemoizer/config.toml`` (it falls through past a project config that
omits it), then — on an interactive run only — one question that is saved to
the home config. Non-interactive runs use the language fallback and say how
to name the owner.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vemoizer.llm_config import (
    _find_nearest_vemoizer_config,
    _parse_section_language,
    _read_toml,
)
from vemoizer.tracks import configured_owner, fallback_owner

#: The one question an interactive tracks run asks (trailing space included).
OWNER_PROMPT = "Your name for transcripts (shown instead of 'Me')? "

#: A ``[meeting]`` table header line, with an optional trailing comment.
_MEETING_HEADER = re.compile(r"^\[meeting\][ \t]*(#.*)?$", re.MULTILINE)

_HINT = (
    "vemoizer: no owner name set, labelling your speech as {name!r}; "
    "name it with --me NAME or [meeting] owner in ~/.vemoizer/config.toml"
)


def home_config_path() -> Path:
    """The home config the owner name is saved to (``~/.vemoizer/config.toml``)."""
    return Path.home() / ".vemoizer" / "config.toml"


def load_owner_inputs() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The parsed project and home config dicts (``None`` when absent/invalid)."""
    project = _find_nearest_vemoizer_config(Path.cwd())
    project_raw = _read_toml(project) if project is not None else None
    return project_raw, _read_toml(home_config_path())


def _with_owner_line(original: str, name: str) -> str:
    """*original* with ``owner = "<name>"`` under ``[meeting]``.

    Inserted right after an existing ``[meeting]`` header, else appended as a
    new table. Every other byte of *original* is kept as it was.
    """
    line = f"owner = {json.dumps(name, ensure_ascii=False)}\n"
    header = _MEETING_HEADER.search(original)
    if header is None:
        gap = "" if not original or original.endswith("\n\n") else "\n"
        if original and not original.endswith("\n"):
            gap = "\n\n"
        return f"{original}{gap}[meeting]\n{line}"
    end = header.end()
    if end < len(original) and original[end] == "\n":
        return original[: end + 1] + line + original[end + 1 :]
    return original[:end] + "\n" + line + original[end:]


def save_owner(path: Path, name: str) -> None:
    """Write the owner name into the config at *path*, atomically.

    The edited text must parse as TOML with ``meeting.owner == name``;
    otherwise nothing is written and ``ValueError`` is raised, so a
    hand-edited config is never corrupted by the first-run prompt.
    """
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    text = _with_owner_line(original, name)
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"config is not valid TOML after the edit: {e}") from e
    meeting = parsed.get("meeting")
    if not isinstance(meeting, dict) or meeting.get("owner") != name:
        raise ValueError("could not place the owner key under [meeting]")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _ask(input_fn: Callable[[str], str]) -> str:
    """The owner answer, stripped; end of input counts as a blank answer."""
    try:
        return input_fn(OWNER_PROMPT).strip()
    except EOFError:
        return ""


def _save_or_warn(name: str, echo: Callable[[str], None]) -> None:
    path = home_config_path()
    try:
        save_owner(path, name)
    except (OSError, ValueError) as e:
        echo(f"warning: could not save the owner name to {path}: {e}")


def resolve_run_owner(
    flag: str | None,
    *,
    prompt: bool,
    input_fn: Callable[[str], str],
    echo: Callable[[str], None],
) -> str:
    """The owner label for one two-track run.

    *prompt* is true only for an interactive run (no ``--yes``, a TTY); it
    gates the one question, so ``input_fn`` is never called otherwise.
    """
    project_raw, home_raw = load_owner_inputs()
    named = configured_owner(flag, project_raw, home_raw)
    if named is not None:
        return named
    fallback = fallback_owner(_parse_section_language(project_raw or home_raw))
    if not prompt:
        echo(_HINT.format(name=fallback))
        return fallback
    answer = _ask(input_fn)
    if not answer:
        return fallback
    _save_or_warn(answer, echo)
    return answer
