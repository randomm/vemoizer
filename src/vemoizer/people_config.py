"""People-config read/write for ``vemoizer names`` (issue #93).

The ``people`` list-of-strings key in the layered ``.vemoizer/config.toml``:
layered path search, fail-open reads, and write-back that never destroys a
config it cannot parse. Serialisation is a minimal TOML emitter for the
round-tripped ``tomllib`` data (scalars top-level, dicts as tables).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import typer


def find_people_config_path(
    home: Path | None = None,
    cwd: Path | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> Path | None:
    """Nearest layered config path for the ``people`` key.

    Same order as ``llm._default_search``: nearest ``./.vemoizer`` →
    ``~/.vemoizer`` → legacy. ``None`` when nothing exists. ``legacy_paths``
    is injectable so tests can isolate from a real dev-machine legacy config.
    """
    from vemoizer.llm import _LEGACY_CONFIG_PATHS, _find_nearest_vemoizer_config

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
    """Read the ``people`` key as a list of strings; fail-open to []."""
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

    All other keys are preserved verbatim. Any pre-existing ``people`` key
    inside a table (e.g. under ``[llm]``) is dropped: a ``people`` key is
    only valid as a top-level list of strings, and leaving a stale one would
    make the next strict config load fail (issue #93).

    Write-back is best-effort: an existing config that cannot be read or
    parsed is left byte-identical with one warning line (rewriting it with
    only ``people`` would destroy unreadable keys); a missing config is
    still created (issue #93).
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
        raw = {}

    for table in raw.values():
        if isinstance(table, dict):
            table.pop("people", None)
    raw["people"] = new_people
    lines = _emit_toml(raw)
    tmp = config_path.with_name(f"{config_path.name}.tmp-{os.getpid()}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(lines, encoding="utf-8")
    os.replace(str(tmp), str(config_path))


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
