"""The ``[llm] repair_concurrency`` config key: its accepted set and its docs.

Split from ``tests/test_repair_concurrency.py`` (the pool's behaviour) because
these cases are about the parser, not the threads: which values the ``[llm]``
section accepts, what a malformed value does on each of the three load paths,
and whether the key is actually documented. ``tests/test_llm.py`` is at its
file cap and is the client's own surface, so the matrix lives here.

The accepted set is the integers ``1..4``; bool, float, string, 0, negative,
above the bound and arbitrary-precision integers all make the section
malformed. That is the same three-outcome contract the stage-budget keys
follow, and the last test here is what keeps a future key from gaining a
validator without gaining a row in the spec table.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vemoizer.llm import LLMConfig
from vemoizer.llm_config import (
    _KNOWN_LLM_KEYS,
    MAX_REPAIR_CONCURRENCY,
    ConfigError,
    _parse_llm_section,
    _strict_load,
)


def _section(extra: str) -> LLMConfig | None:
    """Parse a raw ``[llm]`` table.

    Reimplemented rather than borrowed from ``tests/test_llm.py``: that file
    sits at the ~800-line cap and this ticket forbids adding to it.
    """
    raw = tomllib.loads(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 10\n" + extra
    )
    return _parse_llm_section(raw["llm"])


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        f"timeout_seconds = 10\n{body}",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("value", [1, 2, 3, MAX_REPAIR_CONCURRENCY])
def test_accepted_concurrency_values_parse(value: int) -> None:
    cfg = _section(f"repair_concurrency = {value}\n")
    assert cfg is not None
    assert cfg.repair_concurrency == value


def test_absent_concurrency_is_serial_by_default() -> None:
    cfg = _section("")
    assert cfg is not None
    assert cfg.repair_concurrency == 1


@pytest.mark.parametrize(
    "source",
    ["0", "-1", f"{MAX_REPAIR_CONCURRENCY + 1}", "true", "false", "2.5", '"3"'],
)
def test_malformed_concurrency_makes_the_section_malformed(source: str) -> None:
    """Reject bool, float, string, 0, negative and above the bound.

    Same outcome as a bad budget: the section is malformed, so the fail-open
    loader yields ``None`` and the strict loader raises ``ConfigError`` —
    never a silently ignored value.
    """
    assert _section(f"repair_concurrency = {source}\n") is None


@pytest.mark.parametrize("value", [2**63, -(2**63), 10**100])
def test_huge_concurrency_is_malformed(value: int) -> None:
    """A big integer is out of range, not a valid worker count.

    The accepted set is the integers ``1..4``. An arbitrary-precision value
    is still an ``int`` to Python, so the bound check has to carry it: an
    implementation that compares against a narrower type, or that coerces
    through something which overflows, would either accept it or raise. The
    outcome is the same as any other malformed value — the section is
    malformed on all three load paths, never a silently ignored key.
    """
    assert _section(f"repair_concurrency = {value}\n") is None


def test_huge_concurrency_fails_loud_on_the_strict_paths(tmp_path: Path) -> None:
    """Same outcome as a bad budget on the strict path: ``ConfigError``.

    The fail-open loader yields ``None`` (covered above); the strict loader
    raises. The message is the section-level one the budget keys produce — it
    does not name the offending key, because a malformed value means
    "malformed section" rather than "bad value", and a value that slips the
    range check on one path and not the other is the bug this pins out.
    """
    path = _write_config(tmp_path, f"repair_concurrency = {2**64}\n")
    with pytest.raises(ConfigError, match="malformed 'llm' section"):
        _strict_load(path)


def test_unknown_near_miss_key_is_rejected_by_name(tmp_path: Path) -> None:
    """Accepting the new key must not soften the unknown-key guard.

    ``repair_concurency`` (one letter shuffled) is a typo the operator needs
    told about, not a key quietly ignored.
    """
    path = _write_config(tmp_path, "repair_concurency = 2\n")
    with pytest.raises(ConfigError) as excinfo:
        _strict_load(path)
    assert "repair_concurency" in str(excinfo.value)


def test_the_key_is_registered_not_merely_tolerated() -> None:
    """A known key reaches :class:`~vemoizer.llm_config.LLMConfig`; the
    near-miss above only proves an unknown one is refused."""
    assert "repair_concurrency" in _KNOWN_LLM_KEYS


_SPEC_TABLE_HEADING = "| `llm.repair_budget_seconds` |"


def test_every_known_llm_key_has_a_row_in_the_spec_table() -> None:
    """Every ``[llm]`` key the parser accepts is documented in the one table.

    ``docs/pipeline-spec.md`` is the canonical location for the ``[llm]``
    keys and no other file restates them, so a key that gains a validator
    without gaining a row is undocumented and invisible to the operator.
    The assertion runs over the whole accepted-key set rather than naming
    ``repair_concurrency``, so the next key added to the parser is held to
    the same bar; it is what fails when the row is deleted or renamed.
    """
    spec = (
        Path(__file__).resolve().parents[1] / "docs" / "pipeline-spec.md"
    ).read_text(encoding="utf-8")
    assert _SPEC_TABLE_HEADING in spec, "the [llm] key table moved"
    missing = [key for key in sorted(_KNOWN_LLM_KEYS) if f"| `llm.{key}` |" not in spec]
    assert not missing, f"undocumented [llm] key(s) in docs/pipeline-spec.md: {missing}"
    assert "| `llm.repair_concurrency` |" in spec
