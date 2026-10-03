"""People-config read/write for ``vemoizer names`` (issue #93).

The ``people`` list in the layered ``.vemoizer/config.toml``: layered
path search, fail-open reads, and write-back that never destroys a
config it cannot parse or express. Write-back prefers a surgical text
edit (insert a ``people = [...]`` line at the top, or replace an
existing single-line top-level ``people`` array in place) so comments,
blank lines, array-of-tables, dotted keys and inline tables survive
untouched. Configs the surgical path cannot express (dotted keys,
array-of-tables, or nested-table layouts) are NOT adoptable by the
minimal-serializer fallback either: they are refused with a one-line
warning and left byte-identical — the file is never modified. Only
flat configs (top-level scalars plus simple tables) are re-emitted, and
only when the result re-parses to exactly the intended data.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

import typer

#: Matches a single-line top-level ``people = [...]`` array, optionally
#: followed by whitespace and a comment, on its own line.
_SINGLE_LINE_PEOPLE = re.compile(r"^people\s*=\s*\[[^\]]*\]\s*(#.*)?$", re.MULTILINE)


def find_people_config_path(
    home: Path | None = None,
    cwd: Path | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> Path | None:
    """Nearest layered config path for the ``people`` key.

    Same order as ``llm_config._default_search``: nearest ``./.vemoizer`` →
    ``~/.vemoizer`` → legacy. ``None`` when nothing exists. ``legacy_paths``
    is injectable so tests can isolate from a real dev-machine legacy config.
    """
    from vemoizer.llm_config import _LEGACY_CONFIG_PATHS, _find_nearest_vemoizer_config

    if cwd is None:
        cwd = Path.cwd()
    if home is None:
        home = Path.home()
    if legacy_paths is None:
        legacy_paths = _LEGACY_CONFIG_PATHS

    nearest = _find_nearest_vemoizer_config(cwd)
    if nearest is not None:
        return nearest
    home_config = home / ".vemoizer" / "config.toml"
    if home_config.is_file():
        return home_config
    for candidate in legacy_paths:
        if candidate.is_file():
            return candidate
    return None


def read_people_list(config_path: Path | None) -> list[str]:
    """Read the top-level ``people`` list; fail-open to ``[]``.

    A top-level ``people`` that is not a list is read as ``[]``; items
    in a list that are not strings are filtered out at read time (the
    key itself is not validated).
    """
    if config_path is None:
        return []
    raw, _ = _read_people_raw(config_path)
    if raw is None:
        return []
    return _top_level_people(raw) or []


def _top_level_people(raw: dict[str, Any]) -> list[str] | None:
    """The top-level ``people`` value, or ``None`` when it is not a list.

    In TOML, a bare ``people`` key placed after a ``[table]`` header belongs
    to that table, so only a top-level value is a valid ``people`` list.
    A ``people`` value nested inside a table is ignored here and stripped
    by :func:`write_people_list`.
    """
    value = raw.get("people")
    if not isinstance(value, list):
        return None
    return [s for s in value if isinstance(s, str)]


def _read_people_raw(config_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load an existing config, or report why it cannot be written.

    Returns ``(dict, None)`` for a parseable file, ``(None, exc_name)``
    when an EXISTING file could not be read or parsed (caller must leave
    it byte-identical), or ``(None, None)`` when no file exists yet.
    """
    if not config_path.is_file():
        return None, None
    try:
        with config_path.open("rb") as f:
            loaded = tomllib.load(f)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return None, type(exc).__name__
    if isinstance(loaded, dict):
        return loaded, None
    return {}, None


def write_people_list(config_path: Path, new_people: list[str]) -> None:
    """Atomically write *new_people* as the top-level ``people`` key.

    A valid config is updated in place. The surgical text path preserves
    every other byte of the file (comments, blank lines, array-of-tables,
    dotted keys, inline tables). Layouts it cannot express are re-emitted
    by the minimal TOML serializer only when the result re-parses to
    exactly the intended data (this is only ever true for flat configs —
    top-level scalars plus simple tables); anything else is left
    byte-identical with one warning line. Any pre-existing ``people`` key
    inside a table (e.g. under ``[llm]``) is dropped from the re-emitted
    text: a ``people`` key is only valid top-level, and a stale nested
    one would fail the next strict config load (issue #93).

    Two safety mechanisms keep the surgical path from writing a broken
    file: the surgical result must re-parse to exactly the intended data
    (a table-nested stale ``people`` key that would otherwise survive
    makes it fail), and on any failure the code falls through to the
    re-emit (or, if that cannot round-trip either, to the refusal path)
    rather than writing a partial edit.

    Best-effort: an existing config that cannot be read or parsed, or a
    layout that cannot be expressed without data loss, is left
    byte-identical with one warning line; a missing config is still
    created (issue #93).
    """
    raw, exc_name = _read_people_raw(config_path)
    if exc_name is not None:
        typer.echo(
            f"warning: could not update people in {config_path}: "
            f"{exc_name} (config left unchanged)",
            err=True,
        )
        return
    if raw is None:
        # Missing file: create it with just the people list.
        _atomic_write(config_path, _people_line(new_people) + "\n")
        return

    intended = dict(raw)
    for table in intended.values():
        if isinstance(table, dict):
            table.pop("people", None)
    intended["people"] = new_people

    lines = _surgical_people_edit(config_path, intended)
    if lines is None:
        emitted = _emit_toml(intended)
        lines = emitted if _parses_to(emitted, intended) else None
    if lines is None:
        typer.echo(
            f"warning: could not update people in {config_path}: "
            "unsupported config layout (config left unchanged)",
            err=True,
        )
        return
    _atomic_write(config_path, lines)


def _surgical_people_edit(config_path: Path, intended: dict[str, Any]) -> str | None:
    """Minimal text edit that changes only the top-level ``people`` key.

    Two shapes qualify, both verified by re-parsing the result and
    requiring it to equal *intended* exactly:

    - a single-line top-level ``people = [...]`` array on its own line:
      that line is replaced in place (a trailing ``# comment`` is kept);
    - no top-level ``people`` key: one ``people = [...]`` line is
      inserted at the very start of the file, before any table header.

    Returns the new full text, or ``None`` when no surgical edit applies
    (multi-line people array, ``[people]`` table, scalar people, a
    table-nested stale ``people`` that only the re-emit can drop, a
    re-parse mismatch, or the file disappearing/going unreadable between
    the parse and this read — both ``OSError`` and ``UnicodeDecodeError``
    on the second read route to the refusal path via ``None`` — the caller
    then falls through to the re-emit/refuse-with-warning path, never a
    traceback).
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    replacement = f"people = {_toml_value(intended['people'])}"

    match = _SINGLE_LINE_PEOPLE.search(text)
    if match is not None:
        lines = text.splitlines(keepends=True)
        line_no = text[: match.start()].count("\n")
        if line_no >= len(lines):
            return None
        kept_comment = match.group(1)
        new_line = (
            f"people = {_toml_value(intended['people'])}"
            + (f" {kept_comment}" if kept_comment else "")
            + ("\n" if lines[line_no].endswith("\n") else "")
        )
        lines[line_no] = new_line
        new_text = "".join(lines)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        new_text = f"{replacement}\n{text}"

    return new_text if _parses_to(new_text, intended) else None


def _parses_to(text: str, intended: dict[str, Any]) -> bool:
    """True when *text* parses as TOML and equals *intended* exactly."""
    try:
        parsed = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed == intended


def _atomic_write(config_path: Path, lines: str) -> None:
    """Write *lines* atomically: tmp file in the same directory + replace."""
    tmp = config_path.with_name(f"{config_path.name}.tmp-{os.getpid()}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(lines, encoding="utf-8")
    os.replace(str(tmp), str(config_path))


def _people_line(new_people: list[str]) -> str:
    return "people = " + _toml_value(new_people)


def _emit_toml(raw: dict[str, Any]) -> str:
    """Serialize *raw* as TOML: scalars as top-level, dicts as tables."""
    out: list[str] = []
    tables: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            out.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        out.append(f"[{name}]")
        out.extend(f"{k} = {_toml_value(v)}" for k, v in table.items())
    return "\n".join(out) + ("\n" if out else "")


def _escape_basic_string(value: str) -> str:
    """Escape *value* for a TOML basic string: backslash, quote, newline,
    tab, carriage return and every other control character (plus ``U+007F``
    as a TOML basic string forbids raw control characters)."""
    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return f'"{_escape_basic_string(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # ``tomllib`` only yields str/int/float/bool/list; unreachable via the
    # config round-trip. Kept as an explicit empty string, not invalid TOML.
    return '""'
