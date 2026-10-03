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

import typer

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


def _strict_load_raw(path: Path) -> tuple[LLMConfig, dict[str, Any]]:
    """Strict load that returns ``(config, raw)`` from a single read.

    Raises :class:`ConfigError` on any violation, exactly as
    :func:`_strict_load` does — the only difference is that the already
    parsed raw dict rides along, so the caller never re-reads the file
    (issue #108 review).
    """
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
    return config, raw


def _strict_load(path: Path) -> LLMConfig:
    """Load *path* under strict rules; raise :class:`ConfigError` on violation."""
    config, _raw = _strict_load_raw(path)
    return config


def _validate_sections(raw: dict[str, Any], path: Path) -> None:
    """Validate table sections (``[llm]`` / ``[meeting]``) in *raw*.

    ``[llm]`` is strict (a malformed section must not silently disable the
    LLM); ``[meeting]`` is warned-only — its single ``language`` key is
    cosmetic (a typo just falls back to auto-detect), so a strict reject
    would trade a harmless typo for a failing run. Only these two sections
    are tables: every other dict value keeps the pre-existing strict
    "must not be a table" check in ``_strict_load``.
    """
    for key in raw:
        value = raw[key]
        if not isinstance(value, dict):
            continue
        if key == LLM_CONFIG_SECTION:
            known = _KNOWN_LLM_KEYS
        elif key == "meeting":
            known = _KNOWN_MEETING_KEYS
        else:
            continue
        for sub in value:
            if sub not in known:
                if key == LLM_CONFIG_SECTION:
                    raise ConfigError(f"unknown key {key}.{sub} in {path}")
                typer.echo(
                    f"vemoizer: unknown key {key}.{sub} in {path} (ignored)",
                    err=True,
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


def _legacy_search_with_raw(
    legacy_paths: tuple[Path, ...],
) -> tuple[LLMConfig | None, dict[str, Any] | None]:
    """Probe legacy paths (fail-open) and return ``(config, raw)``.

    The legacy layer stays fail-open (a malformed or section-less file is
    skipped); the deprecation notice is printed at most once per run, and
    only when the ``~/.config`` path actually wins. The winning file is read
    exactly once here — the caller never re-reads it (issue #108 review).
    """
    deprecated_first = legacy_paths[0]
    for candidate in legacy_paths:
        raw = _read_toml(candidate)
        if raw is None:
            continue
        section = raw.get(LLM_CONFIG_SECTION)
        if not isinstance(section, dict):
            continue
        config = _parse_llm_section(section)
        if config is None:
            continue
        if candidate == deprecated_first:
            print(LEGACY_DEPRECATION_NOTICE, file=sys.stderr)
        return config, raw
    return None, None


def _resolve_config_path(path: str | None) -> Path | None:
    """The config file *path* designates, or ``None``.

    Explicit path: as given. ``"os.devnull"`` sentinel: ``None`` (no
    config). Omitted: the documented layered search (project walk-up →
    home → legacy, issue #82) — the first EXISTING file wins, no per-key
    merging.

    Used by :func:`load_language` (explicit-path and devnull handling
    included). The layered search in :func:`_default_search` resolves
    its own layers because it must interleave strict parsing with the
    search, not after it.
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
) -> tuple[LLMConfig | None, dict[str, Any] | None]:
    """Run the layered search (project walk-up → home → legacy); injectable
    hooks for tests only.

    Returns ``(llm_config, raw)`` — the parsed ``[llm]`` section and the
    raw dict of the single config file that won the search, so the caller
    gets both from exactly one read per run (issue #108 review). The legacy
    fail-open layer reads once per legacy candidate it probes (at most the
    two legacy paths), never the winning file twice.
    """
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
        return _strict_load_raw(project_config)

    home_config = home() / ".vemoizer" / "config.toml"
    if home_config.is_file():
        return _strict_load_raw(home_config)

    return _legacy_search_with_raw(legacy_paths)


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


def load_default_config(
    path: str | None = None,
) -> tuple[LLMConfig | None, dict[str, Any] | None]:
    """Load LLM config from *path* or the layered search, one read.

    Returns ``(llm_config, raw)``: the parsed ``[llm]`` section (``None``
    when absent or malformed) and the already-parsed config file dict the
    caller can read other keys from (``[meeting]``, top-level
    ``language``) without re-parsing the file (issue #108 review).

    Contracts: an explicit path stays fail-open — ``"os.devnull"`` or a
    missing/malformed file → ``(None, None)`` (issue #82). An omitted path
    runs the layered search (project walk-up → home → legacy, issue #82):
    the strict project/home layer fails loud on a malformed file (a
    ``ConfigError`` the batch pre-check turns into a clean error), and the
    legacy layer is fail-open with the deprecation notice on
    ``~/.config`` only.
    """
    if path is not None:
        if path == DEVNULL_SENTINEL:
            return None, None
        raw = _read_toml(Path(path))
        if raw is None:
            return None, None
        section_raw = raw.get(LLM_CONFIG_SECTION)
        return (
            _parse_llm_section(section_raw) if isinstance(section_raw, dict) else None,
            raw,
        )

    # The strict project/home layer is fail-LOUD by contract (issue #82);
    # the batch-layer pre-check (_resolve_llm_config) turns the ConfigError
    # into a clean error before transcribe_file ever runs. Each call reads
    # the winning file once (issue #108 review); a run makes one call in
    # the batch pre-check and one in transcribe_file (pre-existing on main).
    return _default_search()
