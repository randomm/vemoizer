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

#: Default TOTAL wall-clock allowance for the repair and notes stages
#: (issue #148). A stalled connection that keeps resetting the per-call
#: timeout would otherwise hold the run forever; these budgets bound the
#: stage as a whole (per invariant #5 the stage fails open on expiry).
DEFAULT_REPAIR_BUDGET_S: float = 600.0
DEFAULT_NOTES_BUDGET_S: float = 600.0


@dataclass(frozen=True)
class LLMConfig:
    """Parsed ``[llm]`` section.

    ``base_url``, ``model``, ``api_key_env`` and ``timeout_seconds`` are
    required (the per-call timeout is a per-read bound). The two stage
    budgets (``repair_budget_seconds``, ``notes_budget_seconds``) are the
    TOTAL wall-clock allowance for the repair and notes stages respectively
    (issue #148): they cap a stage whose per-call timeout keeps resetting on
    a stalled connection. Both default to 600 s (10 min) when absent and
    fail open — the stage just runs to its natural end.
    """

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float
    repair_budget_seconds: float = DEFAULT_REPAIR_BUDGET_S
    notes_budget_seconds: float = DEFAULT_NOTES_BUDGET_S


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
    {
        "base_url",
        "model",
        "api_key_env",
        "timeout_seconds",
        "repair_budget_seconds",
        "notes_budget_seconds",
    }
)
#: Known ``[meeting]`` keys for strict validation (issue #108): the ``language``
#: override and the two-track owner name (issue #171) — a typo is warned, not fatal.
_KNOWN_MEETING_KEYS: frozenset[str] = frozenset({"language", "owner"})
DEVNULL_SENTINEL: str = "os.devnull"


class ConfigError(Exception):
    """Invalid config file; message names the offending key."""


def _layer_raw(path: Path) -> tuple[bool, dict[str, Any] | None]:
    """``(is_layer, raw)`` for *path*, reading it at most once.

    A missing file is not a layer. Only the home file may be preferences-only
    (see :func:`_preferences_only`) and then it is skipped, so it cannot shadow
    a project, home or legacy layer (PR #193 review). Project files always count.
    """
    if not path.is_file():
        return False, None
    raw = _read_toml(path)
    home_config = Path.home() / ".vemoizer" / "config.toml"
    is_home = os.path.realpath(path) == os.path.realpath(home_config)
    return not (is_home and _preferences_only(raw)), raw


def _is_layer_file(path: Path) -> bool:
    """Whether *path* takes part in the layered lookups (see :func:`_layer_raw`)."""
    return _layer_raw(path)[0]


def _find_nearest_layer(start: Path) -> tuple[Path, dict[str, Any] | None] | None:
    """Nearest layer ``./.vemoizer/config.toml`` up from *start*, plus its raw dict."""
    current = Path(os.path.realpath(start))
    while True:
        candidate = current / ".vemoizer" / "config.toml"
        is_layer, raw = _layer_raw(candidate)
        if is_layer:
            return candidate, raw
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _find_nearest_vemoizer_config(start: Path) -> Path | None:
    """Nearest layer ``./.vemoizer/config.toml`` walking up from *start* to root."""
    found = _find_nearest_layer(start)
    return found[0] if found is not None else None


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

    # The stage budgets are optional (issue #148): absent -> the dataclass
    # field default (the safe 600 s); present -> must be a positive finite
    # number, else the section is malformed (return None). A typo'd
    # 0/negative budget is rejected here, not silently applied. The valid
    # values are collected as we validate (one pass over the same keys).
    budget_kwargs: dict[str, float] = {}
    for budget_key in ("repair_budget_seconds", "notes_budget_seconds"):
        if budget_key in section:
            budget = section[budget_key]
            if (
                not isinstance(budget, (int, float))
                or isinstance(budget, bool)
                or budget <= 0
                or not math.isfinite(float(budget))
            ):
                return None
            budget_kwargs[budget_key] = float(budget)

    return LLMConfig(
        base_url=base_url.rstrip("/"),
        model=model.strip(),
        api_key_env=api_key_env,
        timeout_seconds=float(timeout),
        **budget_kwargs,
    )


def _strict_load_raw(
    path: Path, raw: dict[str, Any] | None = None
) -> tuple[LLMConfig, dict[str, Any]]:
    """Strict load that returns ``(config, raw)`` from a single read.

    Raises :class:`ConfigError` on any violation, exactly as
    :func:`_strict_load` does — the only difference is that the already
    parsed raw dict rides along, so the caller never re-reads the file
    (issue #108 review). A caller that has already parsed *path* passes
    *raw* in, so the file is still read once.
    """
    if raw is None:
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
        if _is_layer_file(home_candidate):
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
    project = _find_nearest_layer(cwd())
    if project is not None:
        return _strict_load_raw(project[0], project[1])

    home_config = home() / ".vemoizer" / "config.toml"
    is_layer, home_raw = _layer_raw(home_config)
    if is_layer:
        return _strict_load_raw(home_config, home_raw)

    return _legacy_search_with_raw(legacy_paths)


def _preferences_only(raw: dict[str, Any] | None) -> bool:
    """A home file that holds ``[meeting]`` preferences and no ``[llm]`` table.

    The first-run owner prompt (issue #171) saves ``[meeting] owner`` into
    the home config; such a file is not an LLM config, so the layered search
    skips it instead of failing on the missing ``[llm]`` section.
    """
    return (
        raw is not None
        and LLM_CONFIG_SECTION not in raw
        and "meeting" in raw
        and "people" not in raw
        and "language" not in raw
    )


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
