"""Layered ``people`` config read/write for the ``names`` command (issue #93).

``people`` is a **top-level string key** in the layered
``.vemoizer/config.toml`` (the same layered search order as
``vemoizer.llm``: nearest ``./.vemoizer/config.toml`` walking up from
CWD → ``~/.vemoizer/config.toml`` → legacy). The value is a single
name string — the most recently added person.

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
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Top-level key carrying the people name in the layered config.
PEOPLE_KEY: str = "people"


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
) -> Path | None:
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
    """Serialize *raw* as TOML: scalar values as top-level lines, dict
    values as ``[table]`` blocks."""
    out: list[str] = []
    tables: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            out.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        out.append(f"[{name}]")
        for key, value in table.items():
            out.append(f"{key} = {_toml_value(value)}")
    return "\n".join(out) + ("\n" if out else "")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return repr(value)
