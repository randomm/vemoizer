"""Layered ``people`` config read/write for the ``names`` command (issue #93).

``people`` is a **top-level string key** in the layered
``.vemoizer/config.toml`` (the same layered search order as
``vemoizer.llm``: nearest ``./.vemoizer/config.toml`` walking up from
CWD → ``~/.vemoizer/config.toml`` → legacy). The value is a single
name string (this workstream's config seam: the most recently added
person; a full people list arrives with ``names_cli``).

The key is read fail-open (missing file, unparseable TOML, missing
``people`` key, or non-string value → empty string, never a warning).
Write-back sets the name in the project config if one exists, else in
the home config (created with parent dirs when absent); the file is
rewritten via a same-directory temp file + ``os.replace`` so the
``[llm]`` section and every other key are preserved verbatim.
"""

from __future__ import annotations

import contextlib
import os
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Top-level key carrying the people name in the layered config.
PEOPLE_KEY: str = "people"

#: Valid TOML bare key parts (letters, digits, ``-``, ``_``).
_BARE_KEY_RE = re.compile(r"[A-Za-z0-9_-]+")


def _key_part(key: str) -> str:
    """Render *key* as one dotted-key part (quoted when not bare-key safe,
    or when it embeds a dot)."""
    if _BARE_KEY_RE.fullmatch(key) and "." not in key:
        return key
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _dotted_key(key: str) -> str:
    """Render *key* as a dotted key, quoting/escaping each segment.

    A literal dot in *key* is treated as a path separator (matching how
    the emitter calls this per nesting level); any other non-bare-key
    character (or an embedded dot) forces a quoted, escaped part.
    """
    if "." in key:
        parts = [part for part in key.split(".") if part]
        if len(parts) > 1:
            return ".".join(_key_part(part) for part in parts)
    return _key_part(key)


def find_people_config_path(
    home: Callable[[], Path] | None = None,
    cwd: Callable[[], Path] | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> Path | None:
    """Nearest config file the ``people`` key is read from (or written to).

    Same layered order as ``vemoizer.llm``: nearest ``./.vemoizer``
    (walk up from CWD) → ``~/.vemoizer`` → the legacy ``~/.config`` /
    ``~/.vemoizer.toml`` locations. ``None`` when nothing exists.

    *home* and *legacy_paths* are injectable for tests (mirrors
    ``vemoizer.llm._default_search``); ``legacy_paths`` defaults to the
    real ``~/.config`` / ``~/.vemoizer.toml`` paths when not given.
    """
    from vemoizer.llm import _LEGACY_CONFIG_PATHS, _find_nearest_vemoizer_config

    if cwd is None:
        cwd = Path.cwd
    if home is None:
        home = Path.home
    if legacy_paths is None:
        legacy_paths = _LEGACY_CONFIG_PATHS
    nearest = _find_nearest_vemoizer_config(cwd())
    if nearest is not None:
        return nearest
    home_config = home() / ".vemoizer" / "config.toml"
    if home_config.is_file():
        return home_config
    for candidate in legacy_paths:
        if candidate.is_file():
            return candidate
    return None


def read_people(
    home: Callable[[], Path] | None = None,
    cwd: Callable[[], Path] | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> str:
    """Read the ``people`` value from the layered config; fail-open.

    Missing file, unparseable TOML, or missing/non-string ``people``
    key all yield an empty string (no warning).
    """
    path = find_people_config_path(home=home, cwd=cwd, legacy_paths=legacy_paths)
    if path is None:
        return ""
    try:
        if not path.is_file():
            return ""
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    value = raw.get(PEOPLE_KEY)
    return value if isinstance(value, str) else ""


def add_person(
    name: str,
    home: Callable[[], Path] | None = None,
    cwd: Callable[[], Path] | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> Path:
    """Set *name* as the ``people`` value atomically; return the config path.

    The project config is written if one exists, else the home config
    (created with parent dirs when absent). The rest of the file is
    preserved verbatim by round-tripping the parsed TOML.
    """
    if home is None:
        home = Path.home
    if cwd is None:
        cwd = Path.cwd
    existing = find_people_config_path(home=home, cwd=cwd, legacy_paths=legacy_paths)
    target = existing if existing is not None else home() / ".vemoizer" / "config.toml"

    raw: dict[str, Any] = {}
    if target.is_file():
        try:
            with target.open("rb") as f:
                loaded = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            raw = loaded

    raw[PEOPLE_KEY] = name

    lines = _emit_toml(raw)
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(lines, encoding="utf-8")
        os.replace(str(tmp), str(target))
    except OSError:
        with contextlib.suppress(OSError):  # cleanup is best-effort
            tmp.unlink(missing_ok=True)
        raise
    return target


def _emit_toml(raw: dict[str, Any]) -> str:
    """Serialize *raw* as valid TOML: top-level scalars/lists first, then
    every table (nested levels as ``[a.b]``) and array-of-tables.

    Only the scalar values ``tomllib`` can produce are emitted (str, int,
    float, bool, list); anything else is skipped rather than emitted as
    invalid TOML (``tomllib`` never yields such values).
    """
    out: list[str] = []
    for key, value in raw.items():
        if isinstance(value, dict) or _is_table_list(value):
            continue
        out.append(f"{_dotted_key(key)} = {_toml_value(value)}")
    for key, value in raw.items():
        if isinstance(value, dict):
            _emit_table(out, [_dotted_key(key)], value)
        elif _is_table_list(value):
            for table in value:
                out.append(f"[[{_dotted_key(key)}]]")
                _emit_table_body(out, table)
    return "\n".join(out) + ("\n" if out else "")


def _is_table_list(value: Any) -> bool:
    """True when *value* is a non-empty list of tables (array-of-tables)."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def _emit_table(out: list[str], quoted_path: list[str], table: dict[str, Any]) -> None:
    """Emit ``[path]`` (path parts already quoted/escaped) with its scalar
    members, then recurse into nested tables/arrays-of-tables."""
    header = quoted_path[0] if len(quoted_path) == 1 else ".".join(quoted_path)
    out.append(f"[{header}]")
    for key, value in table.items():
        if not isinstance(value, dict) and not _is_table_list(value):
            out.append(f"{_dotted_key(key)} = {_toml_value(value)}")
    for key, value in table.items():
        child = quoted_path + [_dotted_key(key)]
        if isinstance(value, dict):
            _emit_table(out, child, value)
        elif _is_table_list(value):
            for item in value:
                out.append(f"[[{'.'.join(child)}]]")
                _emit_table_body(out, item)


def _emit_table_body(out: list[str], table: dict[str, Any]) -> None:
    """Emit the scalar members of one array-of-tables entry."""
    for key, value in table.items():
        if not isinstance(value, dict) and not _is_table_list(value):
            out.append(f"{_dotted_key(key)} = {_toml_value(value)}")


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
    # ``tomllib`` only yields the types above; anything else is skipped
    # by the emitters, so this should be unreachable.
    return '""'


def _escape_basic_string(value: str) -> str:
    """Escape *value* for a TOML basic string (backslash, quote, all
    control characters, including ``U+007F``)."""
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
