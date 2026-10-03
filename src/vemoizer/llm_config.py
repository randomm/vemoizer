"""LLM config layer: section parsing, layered search, strict validation.

Extracted from ``llm.py`` (headroom, issue #117) — a pure move: the
``[llm]`` section parsing (both fail-open and strict), the layered config
search (project walk-up → ``~/.vemoizer`` → legacy), the section-language
read, and the ``os.devnull`` explicit-path short-circuit. The HTTP client
(:class:`vemoizer.llm.LLMClient`) stays in ``llm.py``.
"""

from __future__ import annotations

import math
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LLM_CONFIG_SECTION: str = "llm"


@dataclass(frozen=True)
class LLMConfig:
    """Parsed ``[llm]`` section. base_url, model, api_key_env, timeout_seconds."""

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float


def load_config(path: Path | str) -> LLMConfig | None:
    """Parse the user config file; ``None`` (fail-open) when malformed."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with p.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return None

    section = raw.get(LLM_CONFIG_SECTION)
    if not isinstance(section, dict):
        return None
    return _parse_llm_section(section)


_LEGACY_CONFIG_PATHS = (
    Path.home() / ".config" / "vemoizer" / "config.toml",
    Path.home() / ".vemoizer.toml",
)

#: One-line deprecation notice; printed only when the legacy file is used.
LEGACY_DEPRECATION_NOTICE: str = (
    "vemoizer: ~/.config/vemoizer/config.toml is deprecated; "
    "move it to ~/.vemoizer/config.toml"
)

#: Known top-level and [llm] keys for strict validation. ``people`` is a
#: top-level list (issue #93) that ``llm`` itself ignores; items that are
#: not strings are filtered at read time, not validated here.
_KNOWN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {LLM_CONFIG_SECTION, "people", "meeting"}
)
_KNOWN_LLM_KEYS: frozenset[str] = frozenset(
    {"base_url", "model", "api_key_env", "timeout_seconds"}
)
#: Known ``[meeting]`` keys for strict validation (issue #108): the single
#: ``language`` override — a typo (e.g. ``langugae``) is warned, not fatal.
_KNOWN_MEETING_KEYS: frozenset[str] = frozenset({"language"})
DEVNULL_SENTINEL: str = "os.devnull"


class ConfigError(Exception):
    """Invalid config file; message names the offending key."""


def _find_nearest_vemoizer_config(start: Path) -> Path | None:
    """Nearest ``./.vemoizer/config.toml`` walking up from *start* to root."""
    current = Path(os.path.realpath(start))
    while True:
        candidate = current / ".vemoizer" / "config.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _read_toml(path: Path) -> dict[str, Any] | None:
    """Parse *path* as TOML; ``None`` when absent or invalid."""
    try:
        if not path.is_file():
            return None
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _load_legacy_file(path: Path) -> tuple[LLMConfig | None, Path | None]:
    """Parse a legacy config file (fail-open); returns (config, used_path)."""
    raw = _read_toml(path)
    if raw is None:
        return None, path
    section = raw.get(LLM_CONFIG_SECTION)
    if not isinstance(section, dict):
        return None, path
    config = _parse_llm_section(section)
    if config is None:
        return None, path
    return config, path


def _parse_llm_section(section: dict[str, Any]) -> LLMConfig | None:
    """Parse and validate the ``[llm]`` section; ``None`` when malformed.

    Shared by the fail-open ``load_config`` and the strict ``_strict_load``;
    both must fail the same value the same way.
    """
    base_url = section.get("base_url")
    model = section.get("model")
    api_key_env = section.get("api_key_env")
    timeout = section.get("timeout_seconds")

    if not isinstance(base_url, str) or not base_url.strip():
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        return None
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        return None
    if timeout <= 0:
        return None
    if not math.isfinite(float(timeout)):
        # TOML's 1e400 parses to float("inf"); the strict path turns this
        # into a ConfigError (the section is malformed, not a 10^400-second
        # timeout) (issue #82 review).
        return None
    return LLMConfig(
        base_url=base_url.rstrip("/"),
        model=model.strip(),
        api_key_env=api_key_env,
        timeout_seconds=float(timeout),
    )


def _strict_load(path: Path) -> LLMConfig:
    """Load *path* under strict rules; raise :class:`ConfigError` on violation."""
    raw = _read_toml(path)
    if raw is None:
        raise ConfigError(f"config file not found or unreadable: {path}")

    _validate_sections(raw, path)

    for key, value in raw.items():
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            raise ConfigError(f"unknown top-level key or section {key!r} in {path}")
        if isinstance(value, dict):
            # ``[llm]`` and ``[meeting]`` (issue #108) are tables; ``people``
            # must be a top-level list (issue #93), not a table.
            if key not in (LLM_CONFIG_SECTION, "meeting"):
                raise ConfigError(
                    f"top-level key {key!r} must not be a table in {path}; "
                    "top-level 'people' must be a list"
                )
            continue

    section = raw.get(LLM_CONFIG_SECTION)
    if not isinstance(section, dict):
        raise ConfigError(
            f"missing or malformed {LLM_CONFIG_SECTION!r} section in {path}"
        )

    people = raw.get("people")
    if people is not None and not isinstance(people, list):
        # Covers both a ``[people]`` table (a dict) and a scalar value;
        # only a top-level list is valid (issue #93). Items are not
        # validated here: non-string items are filtered at read time.
        raise ConfigError(f"top-level 'people' must be a list in {path}")

    config = _parse_llm_section(section)
    if config is None:
        raise ConfigError(
            f"malformed {LLM_CONFIG_SECTION!r} section in {path} "
            f"(required: base_url, model, api_key_env, timeout_seconds>0)"
        )
    return config


def _validate_sections(raw: dict[str, Any], path: Path) -> None:
    """Validate table sections (``[llm]`` / ``[meeting]``) in *raw*.

    ``[llm]`` is strict (a malformed section must not silently disable the
    LLM); ``[meeting]`` is warned-only — its single ``language`` key is
    cosmetic (a typo just falls back to auto-detect), so a strict reject
    would trade a harmless typo for a failing run.
    """
    for key in raw:
        value = raw[key]
        if not isinstance(value, dict):
            continue
        known = _KNOWN_LLM_KEYS if key == LLM_CONFIG_SECTION else _KNOWN_MEETING_KEYS
        for sub in value:
            if sub not in known:
                if key == LLM_CONFIG_SECTION:
                    raise ConfigError(f"unknown key {key}.{sub} in {path}")
                print(
                    f"vemoizer: unknown key {key}.{sub} in {path} (ignored)",
                    file=sys.stderr,
                )


def _parse_meeting_language(section: dict[str, Any] | None) -> str:
    """The ``[meeting] language`` value of an already-parsed config.

    Fail-open: ``None`` (no section), a non-string value, or any value
    other than ``"fi"`` / ``"en"`` (case-insensitive) all yield
    ``"auto"`` (per-window detection).
    """
    if not isinstance(section, dict):
        return "auto"
    value = section.get("language")
    if not isinstance(value, str):
        return "auto"
    lowered = value.strip().lower()
    if lowered in ("fi", "en", "auto"):
        return lowered
    return "auto"


def _parse_section_language(raw: dict[str, Any] | None) -> str:
    """The top-level ``language`` value of an already-parsed config.

    Fail-open: an absent/non-string value yields ``"fi"``; only
    ``"en"`` (case-insensitive) selects the English heading language.
    """
    if not isinstance(raw, dict):
        return "fi"
    value = raw.get("language")
    if not isinstance(value, str):
        return "fi"
    return "en" if value.strip().lower() == "en" else "fi"


def _legacy_search(legacy_paths: tuple[Path, ...] | None = None) -> LLMConfig | None:
    """Probe legacy paths (fail-open); notice only when ``~/.config`` is used."""
    if legacy_paths is None:
        legacy_paths = _LEGACY_CONFIG_PATHS
    deprecated_first = legacy_paths[0]
    for candidate in legacy_paths:
        config, used = _load_legacy_file(candidate)
        if config is None:
            continue
        if used == deprecated_first:
            print(LEGACY_DEPRECATION_NOTICE, file=sys.stderr)
        return config
    return None


def _resolve_config_path(path: str | None) -> Path | None:
    """The config file *path* designates, or ``None``.

    Explicit path: as given. ``"os.devnull"`` sentinel: ``None`` (no
    config). Omitted: the documented layered search (project walk-up →
    home → legacy, issue #82) — the first EXISTING file wins, no per-key
    merging. Single source of "which config file wins" for the
    ``load_*`` readers.
    """
    if path is not None:
        if path == DEVNULL_SENTINEL:
            return None
        return Path(path)

    candidate = _find_nearest_vemoizer_config(Path.cwd())
    if candidate is None:
        home_candidate = Path.home() / ".vemoizer" / "config.toml"
        if home_candidate.is_file():
            return home_candidate
        candidate = next((p for p in _LEGACY_CONFIG_PATHS if p.is_file()), None)
    return candidate


def _default_search(
    home: Callable[[], Path] | None = None,
    cwd: Callable[[], Path] | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> LLMConfig | None:
    """Run the layered search (project walk-up → home → legacy); injectable
    hooks for tests only."""
    if home is None:
        home = Path.home
    if cwd is None:
        cwd = Path.cwd
    if legacy_paths is None:
        legacy_paths = _LEGACY_CONFIG_PATHS

    # Project layer first: the nearest ./.vemoizer/config.toml walking
    # up from CWD wins over the home layer (issue #82 precedence).
    project_config = _find_nearest_vemoizer_config(cwd())
    if project_config is not None:
        return _strict_load(project_config)

    home_config = home() / ".vemoizer" / "config.toml"
    if home_config.is_file():
        return _strict_load(home_config)

    return _legacy_search(legacy_paths)


def load_language(path: str | None = None) -> str:
    """The section-language setting (``"fi"`` / ``"en"``) for one run.

    M6 (issue #75): the Markdown header and the end-of-run report render
    their headings in the configured language. The value is a top-level
    ``language = "fi" | "en"`` key in the project/global config layer
    (M2), with ``"fi"`` as the default. Same seam as
    :func:`load_default_config`: an explicit path is read directly (the
    ``"os.devnull"`` sentinel yields the default); otherwise the nearest
    ``./.vemoizer/config.toml`` (walk-up) is read, falling back to the
    default. Missing files, unparseable TOML, and absent/non-string values
    all yield ``"fi"`` — the language is cosmetic, so the read is
    fail-open, never an error.
    """
    candidate = _resolve_config_path(path)
    raw = _read_toml(candidate) if candidate is not None else None
    if not isinstance(raw, dict):
        return "fi"
    return _parse_section_language(raw)


def load_meeting_language(path: str | None = None) -> str:
    """The ``[meeting] language`` recognition override for one run.

    Issue #108, option B: a run-level, explicit user choice — ``"fi"`` or
    ``"en"`` (case-insensitive; ``"auto"`` is the value for the key absent,
    meaning per-window detection). Same fail-open seam as :func:`load_language`
    (missing file, unparseable TOML, absent or non-string value all yield
    ``"auto"``) — a malformed config must never abort a transcription, and
    unlike the top-level ``language`` key this one controls RECOGNITION,
    not the Markdown heading language.
    """
    candidate = _resolve_config_path(path)
    raw = _read_toml(candidate) if candidate is not None else None
    if not isinstance(raw, dict):
        return "auto"
    return _parse_meeting_language(raw.get("meeting"))


def load_default_config(path: str | None = None) -> LLMConfig | None:
    """Load LLM config from *path* or the layered search.

    Explicit path short-circuit: ``"os.devnull"`` or a missing path →
    ``None``; a real path loads under legacy fail-open rules. With no
    path, the search runs: nearest ``./.vemoizer`` (walk up from CWD) →
    ``~/.vemoizer`` → legacy (fail-open, deprecation notice on
    ``~/.config`` only).
    """
    if path is not None:
        if path == DEVNULL_SENTINEL:
            return None
        return load_config(path)

    return _default_search()
