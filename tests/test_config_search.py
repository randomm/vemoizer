"""Tests for the M2 layered LLM config search (issue #82).

Covers the search run by ``vemoizer.llm.load_default_config`` with no
explicit path (via the injectable ``_default_search`` so the dev machine's
live ``~/.config/vemoizer/config.toml`` never leaks in):

- Precedence: the nearest ``./.vemoizer/config.toml`` (walk-up from CWD,
  nearest wins) beats ``~/.vemoizer/config.toml``; the project walk-up
  beats the legacy locations.
- ``os.devnull`` and a missing explicit path short-circuit the search.
- Strict validation on the new paths: unknown top-level keys/sections and
  unknown ``[llm]`` keys raise ``ConfigError`` naming the key; other
  top-level sections (e.g. ``[output]``) are ignored.
- Legacy ``~/.config/vemoizer/config.toml`` still loads under the old
  fail-open rules, with the one-line deprecation notice printed ONLY when
  that file is the one actually used.
- The explicit-path contract (``load_config``) is unchanged: fail-open.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vemoizer.llm import (
    LEGACY_DEPRECATION_NOTICE,
    ConfigError,
    _default_search,
    _find_nearest_vemoizer_config,
    load_config,
    load_default_config,
)

#: A minimal valid ``[llm]`` section.
_VALID_SECTION = """\
[llm]
base_url = "https://example.invalid/v1"
model = "test-model"
api_key_env = "TEST_KEY"
timeout_seconds = 5.0
"""


def _write_valid(path: Path, base_url: str, model: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[llm]\nbase_url = "{base_url}"\nmodel = "{model}"\n'
        f'api_key_env = "K"\ntimeout_seconds = 5.0\n',
        encoding="utf-8",
    )
    return path


def _search_in(root: Path, *, chdir_to: Path | None = None):
    """Run the search with ``~`` = ``root/home`` and CWD = *chdir_to* (or
    ``root/proj``). The legacy probe is pinned to ``root/home`` so the dev
    machine's live config cannot leak in.
    """
    home = root / "home"
    home.mkdir(exist_ok=True)
    start = chdir_to if chdir_to is not None else root / "proj"
    start.mkdir(parents=True, exist_ok=True)
    return _default_search(
        home=lambda: home,
        cwd=lambda: start,
        legacy_paths=(
            home / ".config" / "vemoizer" / "config.toml",
            home / ".vemoizer.toml",
        ),
    )


def _write_section(path: Path, section: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(section, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_home_config_loads_when_no_project_layer(self, tmp_path: Path) -> None:
        _write_valid(tmp_path / "home" / ".vemoizer" / "config.toml", "h", "home-model")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "home-model"

    def test_project_layer_beats_home_layer(self, tmp_path: Path) -> None:
        # Precedence (issue #82): the nearest ./.vemoizer/config.toml
        # (walk up from CWD) beats ~/.vemoizer/config.toml.
        _write_valid(tmp_path / "home" / ".vemoizer" / "config.toml", "x", "home-model")
        _write_valid(tmp_path / "proj" / ".vemoizer" / "config.toml", "x", "proj-model")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "proj-model"

    def test_project_config_beats_legacy(self, tmp_path: Path) -> None:
        _write_valid(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "x",
            "legacy-model",
        )
        _write_valid(tmp_path / "proj" / ".vemoizer" / "config.toml", "x", "proj-model")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "proj-model"

    def test_nearest_project_config_wins_on_walk_up(self, tmp_path: Path) -> None:
        _write_valid(
            tmp_path / "proj" / ".vemoizer" / "config.toml", "x", "parent-model"
        )
        nested = tmp_path / "proj" / "a" / "b"
        nested.mkdir(parents=True)
        _write_valid(nested / ".vemoizer" / "config.toml", "x", "nested-model")
        cfg = _search_in(tmp_path, chdir_to=nested)
        assert cfg is not None
        assert cfg.model == "nested-model"

    def test_project_layer_wins_when_no_walk_up_hit(self, tmp_path: Path) -> None:
        # No project layer on the walk-up: the search falls through to
        # the home layer (project-first precedence, issue #82).
        _write_valid(tmp_path / "home" / ".vemoizer" / "config.toml", "x", "home-model")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "home-model"

    def test_home_config_used_when_no_project_layer(self, tmp_path: Path) -> None:
        # Walk-up finds nothing; the home .vemoizer config is the one
        # actually used (and the legacy file, if present, is not).
        _write_valid(tmp_path / "home" / ".vemoizer" / "config.toml", "x", "home-model")
        _write_valid(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "x",
            "legacy-model",
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "home-model"

    def test_no_config_anywhere_returns_none(self, tmp_path: Path) -> None:
        assert _search_in(tmp_path) is None

    def test_legacy_home_vemoizer_toml_still_probed(self, tmp_path: Path) -> None:
        _write_valid(tmp_path / "home" / ".vemoizer.toml", "x", "dotfile-model")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "dotfile-model"

    def test_legacy_config_wins_when_nothing_else(self, tmp_path: Path) -> None:
        _write_valid(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "x",
            "legacy-1",
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "legacy-1"


# ---------------------------------------------------------------------------
# Explicit path and sentinel short-circuit
# ---------------------------------------------------------------------------


class TestExplicitPath:
    def test_explicit_path_is_loaded(self, tmp_path: Path) -> None:
        f = tmp_path / "explicit.toml"
        f.write_text(_VALID_SECTION, encoding="utf-8")
        cfg = load_default_config(str(f))
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_os_devnull_short_circuits_the_search(self, tmp_path: Path) -> None:
        # A config exists in the tree — the sentinel must skip it entirely.
        _write_valid(tmp_path / "proj" / ".vemoizer" / "config.toml", "x", "proj-model")
        assert load_default_config("os.devnull") is None

    def test_missing_explicit_path_fails_open(self, tmp_path: Path) -> None:
        assert load_default_config(str(tmp_path / "nope.toml")) is None


# ---------------------------------------------------------------------------
# Walk-up helper in isolation
# ---------------------------------------------------------------------------


class TestWalkUp:
    def test_walk_up_finds_nearest(self, tmp_path: Path) -> None:
        parent = tmp_path / "proj"
        _write_valid(parent / ".vemoizer" / "config.toml", "x", "parent-model")
        nested = parent / "a" / "b"
        nested.mkdir(parents=True)
        found = _find_nearest_vemoizer_config(nested)
        assert found is not None
        # Compare by name — realpath may resolve symlinks on the path.
        assert found.name == "config.toml"
        assert found.parent.name == ".vemoizer"

    def test_walk_up_returns_none_when_absent(self, tmp_path: Path) -> None:
        proj = tmp_path / "proj"
        proj.mkdir()
        assert _find_nearest_vemoizer_config(proj) is None

    def test_walk_up_stops_at_filesystem_root(self, tmp_path: Path) -> None:
        # An isolated empty tree with no .vemoizer anywhere: the walk must
        # terminate (stop at the root) without hanging.
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _find_nearest_vemoizer_config(empty) is None


# ---------------------------------------------------------------------------
# Strict validation (new paths only)
# ---------------------------------------------------------------------------


class TestStrictValidation:
    def test_unknown_key_under_llm_raises_naming_the_key(self, tmp_path: Path) -> None:
        # A file with a valid [llm] section plus an unknown key under it.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            _VALID_SECTION + '\nextra = "oops"\n',
        )
        with pytest.raises(ConfigError, match=r"unknown key llm\.extra"):
            _search_in(tmp_path)

    def test_unknown_top_level_section_raises_naming_the_key(
        self, tmp_path: Path
    ) -> None:
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            _VALID_SECTION + "\n[nonsense]\nx = 1\n",
        )
        with pytest.raises(ConfigError, match=r"unknown top-level.*'nonsense'"):
            _search_in(tmp_path)

    def test_unknown_top_level_key_raises_naming_the_key(self, tmp_path: Path) -> None:
        # A top-level key (not a section): must be rejected.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            'stray = 1\n[llm]\nbase_url = "https://x"\n',
        )
        with pytest.raises(ConfigError, match=r"unknown top-level.*'stray'"):
            _search_in(tmp_path)

    def test_llm_section_loads_when_present(self, tmp_path: Path) -> None:
        # Pins the happy path: a well-formed [llm] section loads cleanly.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            _VALID_SECTION,
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_people_top_level_key_loads_without_error(self, tmp_path: Path) -> None:
        # issue #93: a config containing both [llm] and a top-level
        # ``people`` list-of-strings loads cleanly (``people`` is in
        # ``_KNOWN_TOP_LEVEL_KEYS``); the value is ignored by llm.
        # ``people`` must come BEFORE the [llm] section to be top-level.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            'people = ["Mikko", "Aino"]\n' + _VALID_SECTION,
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_people_as_table_or_scalar_raises(self, tmp_path: Path) -> None:
        # issue #93: ``people`` must be a top-level list. A
        # ``[people]`` table (a dict) and a scalar ``people`` are both
        # rejected by strict load.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            "[people]\nx = 1\n" + _VALID_SECTION,
        )
        with pytest.raises(ConfigError, match=r"top-level 'people' must be a list"):
            _search_in(tmp_path)
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            'people = "Mikko"\n' + _VALID_SECTION,
        )
        with pytest.raises(ConfigError, match=r"top-level 'people' must be a list"):
            _search_in(tmp_path)

    def test_bad_toml_on_strict_path_raises(self, tmp_path: Path) -> None:
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            "[llm\n  broken",
        )
        with pytest.raises(ConfigError):
            _search_in(tmp_path)

    def test_missing_llm_section_on_strict_path_raises(self, tmp_path: Path) -> None:
        # A file that exists but has no [llm] section: must be non-empty
        # and parseable so the "missing section" path is what fires.
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml",
            "# a valid TOML file with no [llm] section\n",
        )
        with pytest.raises(ConfigError, match=r"missing or malformed"):
            _search_in(tmp_path)

    def test_malformed_llm_section_on_strict_path_raises(self, tmp_path: Path) -> None:
        _write_section(
            tmp_path / "home" / ".vemoizer" / "config.toml", "[llm]\nbase_url = ''\n"
        )
        with pytest.raises(ConfigError, match=r"malformed"):
            _search_in(tmp_path)


# ---------------------------------------------------------------------------
# Legacy path: still read, deprecation notice only when actually used
# ---------------------------------------------------------------------------


class TestLegacyPath:
    def test_legacy_config_loads_with_deprecation_notice(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_valid(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "x",
            "legacy-1",
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "legacy-1"
        out = capsys.readouterr()
        assert LEGACY_DEPRECATION_NOTICE in out.err
        assert LEGACY_DEPRECATION_NOTICE not in out.out

    def test_no_notice_when_newer_layer_wins(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_valid(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "x",
            "legacy-1",
        )
        _write_valid(tmp_path / "proj" / ".vemoizer" / "config.toml", "x", "proj-1")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "proj-1"
        out = capsys.readouterr()
        assert LEGACY_DEPRECATION_NOTICE not in out.err

    def test_no_notice_for_legacy_dotfile_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Only the ~/.config location is deprecated; ~/.vemoizer.toml is not.
        _write_valid(tmp_path / "home" / ".vemoizer.toml", "x", "dot-1")
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "dot-1"
        out = capsys.readouterr()
        assert LEGACY_DEPRECATION_NOTICE not in out.err

    def test_malformed_legacy_file_fails_open_no_notice(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_section(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "[llm]\nbase_url = ''\n",
        )
        assert _search_in(tmp_path) is None
        out = capsys.readouterr()
        assert LEGACY_DEPRECATION_NOTICE not in out.err

    def test_legacy_unknown_key_still_fails_open(self, tmp_path: Path) -> None:
        # Strictness is new-path only: a legacy file with an extra [llm] key
        # loads under the old rules (the extra key is ignored).
        _write_section(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            _VALID_SECTION + '\nextra = "oops"\n',
        )
        cfg = _search_in(tmp_path)
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_legacy_bad_toml_fails_open(self, tmp_path: Path) -> None:
        _write_section(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "[llm\n  broken",
        )
        assert _search_in(tmp_path) is None

    def test_legacy_missing_llm_section_fails_open(self, tmp_path: Path) -> None:
        _write_section(
            tmp_path / "home" / ".config" / "vemoizer" / "config.toml",
            "[other]\nx = 1\n",
        )
        assert _search_in(tmp_path) is None


# ---------------------------------------------------------------------------
# Explicit-path contract unchanged (fail-open, legacy rules)
# ---------------------------------------------------------------------------


class TestExplicitPathContract:
    def test_explicit_unknown_key_fails_open_not_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "explicit.toml"
        f.write_text(_VALID_SECTION + '\nextra = "oops"\n', encoding="utf-8")
        cfg = load_default_config(str(f))
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_explicit_bad_section_fails_open(self, tmp_path: Path) -> None:
        f = tmp_path / "explicit.toml"
        f.write_text("[llm]\nbase_url = ''\n", encoding="utf-8")
        assert load_default_config(str(f)) is None

    def test_load_config_direct_still_fail_open(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.toml"
        f.write_text("[llm\nbroken", encoding="utf-8")
        assert load_config(str(f)) is None
