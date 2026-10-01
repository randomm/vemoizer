"""Tests for the layered ``people`` config read/write (issue #93, M5c-1).

``people`` is a **top-level string key** in the layered
``.vemoizer/config.toml``, read fail-open (missing file, invalid TOML,
non-string value → empty string). Written atomically to the project
config if one exists, else the home config; the ``[llm]`` section and
all other keys are preserved verbatim.

TOML note: ``people`` must appear BEFORE the ``[llm]`` section in the
file to be a top-level key (values after a section header belong to
that section).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from vemoizer.llm import ConfigError, _default_search, _strict_load
from vemoizer.people_config import (
    add_person,
    find_people_config_path,
    read_people,
)

#: A minimal valid ``[llm]`` section (same shape as test_config_search).
_VALID_SECTION = """\
[llm]
base_url = "https://example.invalid/v1"
model = "test-model"
api_key_env = "TEST_KEY"
timeout_seconds = 5.0
"""

#: ``people`` (top-level) followed by the [llm] section — correct TOML
#: ordering so that ``people`` is NOT a key under [llm].
_PEOPLE_PLUS_LLM = 'people = "Mikko"\n' + _VALID_SECTION


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _home_proj(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    return home, proj


def _legacy(home: Path) -> tuple[Path, ...]:
    """Legacy paths pinned to the fake home (dev machine isolation)."""
    return (home / ".config" / "vemoizer" / "config.toml", home / ".vemoizer.toml")


def _read(home: Path, cwd: Path, legacy: tuple[Path, ...]) -> str:
    return read_people(home=lambda: home, cwd=lambda: cwd, legacy_paths=legacy)


def _add(name: str, home: Path, cwd: Path, legacy: tuple[Path, ...]) -> Path:
    return add_person(name, home=lambda: home, cwd=lambda: cwd, legacy_paths=legacy)


def _find(home: Path, cwd: Path, legacy: tuple[Path, ...]) -> Path | None:
    return find_people_config_path(
        home=lambda: home, cwd=lambda: cwd, legacy_paths=legacy
    )


def _search(home: Path, proj: Path):
    return _default_search(
        home=lambda: home,
        cwd=lambda: proj,
        legacy_paths=_legacy(home),
    )


# ---------------------------------------------------------------------------
# llm strict load: people key is known
# ---------------------------------------------------------------------------


class TestStrictLoadPeople:
    def test_people_key_loads_without_config_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "config.toml", _PEOPLE_PLUS_LLM)
        config = _strict_load(path)
        assert config.model == "test-model"

    def test_project_layer_with_people_loads_via_search(self, tmp_path: Path) -> None:
        # The written ``people`` key (project layer, as written by the
        # names command write-back) must survive the next strict search.
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", _PEOPLE_PLUS_LLM)
        cfg = _search(home, proj)
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_unknown_scalar_key_still_rejected(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "config.toml", "nonsense = 1\n" + _VALID_SECTION)
        with pytest.raises(ConfigError, match=r"unknown top-level"):
            _strict_load(path)

    def test_people_as_table_is_rejected(self, tmp_path: Path) -> None:
        # A ``[people]`` section (table) is a structural error — strict
        # load must reject it, not skip-continue it (adversarial #3).
        path = _write(
            tmp_path / "config.toml",
            "[people]\nx = 1\n" + _VALID_SECTION,
        )
        with pytest.raises(ConfigError, match=r"must not be a table"):
            _strict_load(path)


# ---------------------------------------------------------------------------
# Layered read: precedence
# ---------------------------------------------------------------------------


class TestReadPeoplePrecedence:
    def test_no_config_gives_empty_string(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        assert _read(home, proj, _legacy(home)) == ""

    def test_reads_project_layer(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", 'people = "Mikko"\n')
        assert _read(home, proj, _legacy(home)) == "Mikko"

    def test_project_beats_home(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(home / ".vemoizer" / "config.toml", 'people = "Home"\n')
        _write(proj / ".vemoizer" / "config.toml", 'people = "Proj"\n')
        assert _read(home, proj, _legacy(home)) == "Proj"

    def test_home_layer_when_no_project(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(home / ".vemoizer" / "config.toml", 'people = "Home"\n')
        assert _read(home, proj, _legacy(home)) == "Home"

    def test_legacy_layer_last(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(home / ".config" / "vemoizer" / "config.toml", 'people = "Legacy"\n')
        assert _read(home, proj, _legacy(home)) == "Legacy"

    def test_walk_up_finds_nearest_project_layer(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        nested = proj / "a" / "b"
        nested.mkdir(parents=True)
        _write(proj / ".vemoizer" / "config.toml", 'people = "Parent"\n')
        _write(nested / ".vemoizer" / "config.toml", 'people = "Nested"\n')
        assert _read(home, nested, _legacy(home)) == "Nested"

    def test_people_alongside_llm_section_is_read(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", _PEOPLE_PLUS_LLM)
        assert _read(home, proj, _legacy(home)) == "Mikko"


# ---------------------------------------------------------------------------
# Fail-open matrix
# ---------------------------------------------------------------------------


class TestReadPeopleFailOpen:
    def test_missing_config_file(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        assert _read(home, proj, _legacy(home)) == ""

    def test_invalid_toml_gives_empty_string(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", "[llm\n  broken")
        assert _read(home, proj, _legacy(home)) == ""

    def test_people_not_a_string_gives_empty_string(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", "people = 42\n")
        assert _read(home, proj, _legacy(home)) == ""

    def test_people_not_in_config_gives_empty_string(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", "# nothing here\n")
        assert _read(home, proj, _legacy(home)) == ""

    def test_no_warning_on_any_fail_open_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home, proj = _home_proj(tmp_path)
        _read(home, proj, _legacy(home))
        assert capsys.readouterr().err == ""

    def test_no_warning_on_non_string_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", "people = 42\n")
        _read(home, proj, _legacy(home))
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


class TestAddPerson:
    def test_sets_people_in_project_config_when_it_exists(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, _PEOPLE_PLUS_LLM)

        used = _add("Jonna", home, proj, _legacy(home))
        assert used == cfg_path
        assert _read(home, proj, _legacy(home)) == "Jonna"
        raw = cfg_path.read_text(encoding="utf-8")
        # The [llm] section survives the round-trip.
        assert 'model = "test-model"' in raw

    def test_creates_home_config_when_no_project_layer(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)

        used = _add("Mikko", home, proj, _legacy(home))
        assert used == home / ".vemoizer" / "config.toml"
        assert used.is_file()
        assert _read(home, proj, _legacy(home)) == "Mikko"

    def test_preserves_other_keys_and_llm_section(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "A"\n' + _VALID_SECTION)

        _add("B", home, proj, _legacy(home))

        from vemoizer.llm import load_config

        cfg = load_config(cfg_path)
        assert cfg is not None
        assert cfg.model == "test-model"
        assert cfg.base_url == "https://example.invalid/v1"
        assert _read(home, proj, _legacy(home)) == "B"

    def test_written_file_round_trips_through_strict_load(self, tmp_path: Path) -> None:
        # The names command writes people; the next meeting run must load
        # the config without ConfigError (project layer, strict path).
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "A"\n' + _VALID_SECTION)

        _add("B", home, proj, _legacy(home))
        config = _strict_load(cfg_path)
        assert config.model == "test-model"

    def test_writes_atomically_no_tmp_left_behind(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = ""\n')

        _add("X", home, proj, _legacy(home))
        leftovers = [
            p.name for p in cfg_path.parent.iterdir() if p.name != "config.toml"
        ]
        assert leftovers == []

    def test_overwrites_existing_people_value(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "Old"\n')

        _add("New", home, proj, _legacy(home))
        assert _read(home, proj, _legacy(home)) == "New"

    def test_preserves_llm_section_in_written_output(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, _PEOPLE_PLUS_LLM)

        _add("New", home, proj, _legacy(home))
        raw = cfg_path.read_text(encoding="utf-8")
        # All four [llm] keys must survive the round-trip.
        assert 'base_url = "https://example.invalid/v1"' in raw
        assert 'model = "test-model"' in raw
        assert 'api_key_env = "TEST_KEY"' in raw
        assert "timeout_seconds = 5.0" in raw
        assert 'people = "New"' in raw


class TestAddPersonRoundTripStructures:
    """Write-back must be lossless for the TOML structures ``tomllib``
    can produce: nested tables, arrays of tables, inline tables, and
    non-ASCII or control-character string values (adversarial #1/#2).
    """

    def test_nested_table_round_trips(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(
            cfg_path,
            'people = "Mikko"\n[glossary.en-us]\nlabel = "english"\n' + _VALID_SECTION,
        )

        _add("Jonna", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == "Jonna"
        assert raw["glossary"]["en-us"]["label"] == "english"
        assert raw["llm"]["model"] == "test-model"

    def test_array_of_tables_round_trips(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(
            cfg_path,
            'people = "Mikko"\n[[clips]]\nlabel = "A"\n'
            '[[clips]]\nlabel = "B"\n' + _VALID_SECTION,
        )

        _add("Jonna", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["clips"] == [{"label": "A"}, {"label": "B"}]
        assert raw["people"] == "Jonna"
        assert raw["llm"]["model"] == "test-model"

    def test_inline_table_round_trips(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "Mikko"\ntags = {a = 1, b = "x"}\n' + _VALID_SECTION)

        _add("Jonna", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["tags"] == {"a": 1, "b": "x"}

    def test_name_with_newline_and_tab_escapes(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "Old"\n')

        _add("Jonna\nOjala\tJr", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == "Jonna\nOjala\tJr"

    def test_name_with_backslash_and_quote_escapes(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "Old"\n')

        _add('A\\B"C', home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == 'A\\B"C'

    def test_name_with_non_ascii_round_trips(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(cfg_path, 'people = "Old"\n')

        _add("Jönna Öljälä", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == "Jönna Öljälä"

    def test_rewritten_file_reloads_as_valid_toml(self, tmp_path: Path) -> None:
        # The rewritten file must be valid TOML that parses back to the
        # same structure (strict load rejects ``[glossary]`` as an
        # unknown top-level key, so parse directly instead).
        home, proj = _home_proj(tmp_path)
        cfg_path = proj / ".vemoizer" / "config.toml"
        _write(
            cfg_path,
            'people = "Mikko"\n[glossary.en-us]\nlabel = "english"\n' + _VALID_SECTION,
        )

        _add("Jonna", home, proj, _legacy(home))

        with cfg_path.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["glossary"]["en-us"]["label"] == "english"
        assert raw["people"] == "Jonna"
        assert raw["llm"]["model"] == "test-model"


# ---------------------------------------------------------------------------
# find_people_config_path
# ---------------------------------------------------------------------------


class TestFindPeopleConfigPath:
    def test_returns_project_when_present(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(proj / ".vemoizer" / "config.toml", "# x\n")
        assert _find(home, proj, _legacy(home)) == proj / ".vemoizer" / "config.toml"

    def test_returns_home_when_no_project(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(home / ".vemoizer" / "config.toml", "# x\n")
        assert _find(home, proj, _legacy(home)) == home / ".vemoizer" / "config.toml"

    def test_returns_legacy_when_only_legacy(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        _write(home / ".vemoizer.toml", "# x\n")
        assert _find(home, proj, _legacy(home)) == home / ".vemoizer.toml"

    def test_returns_none_when_nothing(self, tmp_path: Path) -> None:
        home, proj = _home_proj(tmp_path)
        assert _find(home, proj, _legacy(home)) is None
