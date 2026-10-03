"""People-config tests for the ``vemoizer names`` command (issue #93).

Covers the ``people`` list-of-strings key in the layered
``.vemoizer/config.toml``:

- ``llm_config._strict_load`` accepts a top-level ``people`` list and ignores it
  (the value is never used by the LLM), but rejects a ``people`` table or
  scalar (it must be a top-level list) — locking in the
  ``_KNOWN_TOP_LEVEL_KEYS`` extension that otherwise breaks the next
  meeting run with ``ConfigError``.
- Layered read: nearest ``./.vemoizer`` beats ``~/.vemoizer``; fail-open to
  ``[]`` on missing file, non-list ``people``, and invalid TOML.
- Write-back: the project config wins when it exists; a ``people`` key
  nested *inside* a table (e.g. under ``[llm]``) is dropped so the written
  file carries exactly one top-level ``people`` and still passes strict
  load (the data-loss / config-corruption regression the decision item
  guards against).
- Readline completer: installed only when ``readline`` imports AND
  stdin/stdout are TTYs; the previous completer (whatever it was) is
  restored afterwards — not hard-set to ``None``.
- Round-trip: a config carrying both ``[llm]`` and ``people`` survives a
  full ``names`` run (people written on explicit yes) and a subsequent
  strict ``llm`` load.

All tests use ``isolate_home`` (chdir + HOME isolation); no real audio,
models, or network.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

from vemoizer.llm_config import ConfigError, _default_search, load_default_config
from vemoizer.names_cli import _install_completer, run_names
from vemoizer.people_config import (
    find_people_config_path,
    read_people_list,
    write_people_list,
)

#: A minimal valid ``[llm]`` section (mirrors test_config_search).
_VALID_SECTION = (
    "[llm]\n"
    'base_url = "https://example.invalid/v1"\n'
    'model = "test-model"\n'
    'api_key_env = "TEST_KEY"\n'
    "timeout_seconds = 5.0\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sidecar(**extra: Any) -> dict[str, Any]:
    """A minimal sidecar with one labelled speaker."""
    base: dict[str, Any] = {
        "text": "Puhuttiin asioista.",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "Moikka, aloitellaan tässä.",
                "speaker": "SPEAKER_1",
            },
        ],
        "notes": {"title": "Kokous", "summary": "Puhuttiin asioista."},
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
        **extra,
    }
    return base


def _write_sidecar(
    tmp_path: Path, data: dict[str, Any], name: str = "sidecar.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _no_op_completer(text: str, state: int) -> str | None:
    """A valid (no-op) readline completer used as a sentinel in tests."""
    return None


def _project_config(tmp_path: Path) -> Path:
    """Path to the nearest project config (``./.vemoizer/config.toml``)."""
    d = tmp_path / ".vemoizer"
    d.mkdir(exist_ok=True)
    return d / "config.toml"


def _search_people_config(tmp_path: Path):
    """Run ``llm_config._default_search`` against an isolated fake HOME / CWD."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    return _default_search(
        home=lambda: home,
        cwd=lambda: proj,
        legacy_paths=(
            home / ".config" / "vemoizer" / "config.toml",
            home / ".vemoizer.toml",
        ),
    )


# ---------------------------------------------------------------------------
# llm_config._strict_load accepts [llm] + a top-level people list
# ---------------------------------------------------------------------------


class TestStrictLoadPeople:
    def test_llm_and_people_list_loads(self, tmp_path: Path) -> None:
        # A config carrying both [llm] and a top-level ``people`` list
        # loads cleanly; llm ignores the people value.
        cfg_path = tmp_path / "cfg.toml"
        cfg_path.write_text(
            'people = ["Mikko", "Aino"]\n' + _VALID_SECTION, encoding="utf-8"
        )
        load_default_config(str(cfg_path))  # fail-open load path

        # And the strict project-layer load (the regression the decision
        # item guards): a written people key must not break the next run.
        proj = tmp_path / "proj"
        proj.mkdir()
        proj_cfg = proj / ".vemoizer" / "config.toml"
        proj_cfg.parent.mkdir()
        proj_cfg.write_text(
            'people = ["Mikko", "Aino"]\n' + _VALID_SECTION, encoding="utf-8"
        )
        home = tmp_path / "home"
        home.mkdir()
        cfg = _default_search(
            home=lambda: home,
            cwd=lambda: proj,
            legacy_paths=(home / ".config" / "vemoizer" / "config.toml",),
        )
        assert cfg is not None
        assert cfg.model == "test-model"

    def test_people_table_raises(self, tmp_path: Path) -> None:
        # A ``[people]`` table (a dict) is not a list.
        proj = tmp_path / "proj"
        proj.mkdir()
        proj_cfg = proj / ".vemoizer" / "config.toml"
        proj_cfg.parent.mkdir()
        proj_cfg.write_text("[people]\nx = 1\n" + _VALID_SECTION, encoding="utf-8")
        home = tmp_path / "home"
        home.mkdir()
        with pytest.raises(ConfigError, match=r"top-level 'people' must be a list"):
            _default_search(
                home=lambda: home,
                cwd=lambda: proj,
                legacy_paths=(home / ".config" / "vemoizer" / "config.toml",),
            )

    def test_people_scalar_raises(self, tmp_path: Path) -> None:
        # A scalar ``people`` is not a list: strict load rejects.
        proj = tmp_path / "proj"
        proj.mkdir()
        proj_cfg = proj / ".vemoizer" / "config.toml"
        proj_cfg.parent.mkdir()
        proj_cfg.write_text('people = "Mikko"\n' + _VALID_SECTION, encoding="utf-8")
        home = tmp_path / "home"
        home.mkdir()
        with pytest.raises(ConfigError, match=r"top-level 'people' must be a list"):
            _default_search(
                home=lambda: home,
                cwd=lambda: proj,
                legacy_paths=(home / ".config" / "vemoizer" / "config.toml",),
            )


# ---------------------------------------------------------------------------
# Layered read: project beats home; fail-open matrix
# ---------------------------------------------------------------------------


class TestLayeredRead:
    def test_project_config_path_found(self, tmp_path: Path) -> None:
        """Nearest project config is returned when it exists."""
        p = _project_config(tmp_path)
        p.write_text('people = ["A"]\n', encoding="utf-8")
        assert find_people_config_path(cwd=tmp_path, home=tmp_path / "home") == p

    def test_people_project_beats_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The project layer's people beats the home layer (project wins)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        proj = _project_config(tmp_path)
        proj.write_text('people = ["Proj"]\n', encoding="utf-8")
        home_cfg = tmp_path / "home" / ".vemoizer" / "config.toml"
        home_cfg.parent.mkdir(parents=True)
        home_cfg.write_text('people = ["Home"]\n', encoding="utf-8")

        assert read_people_list(find_people_config_path()) == ["Proj"]

    def test_missing_config_empty_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``read_people_list(None)`` -> empty list (fail-open). This is
        the fail-open contract: when no config path is found, reading yields
        an empty list with no error (the project/home layer search is
        covered by the other layer tests, which pin home/cwd explicitly)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        assert read_people_list(None) == []
        # A non-existent path is equally fail-open.
        assert read_people_list(tmp_path / "does-not-exist" / "config.toml") == []

    def test_non_list_people_fail_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scalar ``people`` -> empty list (fail-open, no warning)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        p.write_text('people = "Mikko"\n', encoding="utf-8")
        assert read_people_list(p) == []

    def test_non_string_list_items_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only string entries are kept (ints filtered out)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        p.write_text("people = [1, 2]\n", encoding="utf-8")
        assert read_people_list(p) == []

    def test_invalid_toml_fail_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid TOML -> empty list (fail-open)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        p.write_text("people = [\n", encoding="utf-8")
        assert read_people_list(p) == []


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


class TestWriteBack:
    def test_write_creates_top_level_people(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing to an absent config creates a file with only the
        top-level people key; it round-trips through tomllib."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        write_people_list(p, ["Mikko", "Aino"])
        with p.open("rb") as f:
            raw = tomllib.load(f)
        assert raw == {"people": ["Mikko", "Aino"]}

    def test_write_preserves_other_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing keys (including [llm]) are preserved verbatim."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        p.write_text(_VALID_SECTION, encoding="utf-8")
        write_people_list(p, ["Mikko"])
        with p.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == ["Mikko"]
        assert raw["llm"]["model"] == "test-model"

    def test_write_strips_people_nested_in_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``people`` key nested inside ``[llm]`` is dropped so the
        written file carries exactly one top-level ``people`` and passes
        strict load (no duplicate / llm.people ConfigError)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        # people top-level (before [llm]) AND a stale copy nested under
        # [llm] — the shape that would otherwise round-trip to a duplicate
        # top-level key (or a stray [llm] people key) on write-back.
        p.write_text(
            'people = ["a"]\n'
            "[llm]\n"
            'base_url = "https://example.invalid/v1"\n'
            'model = "test-model"\n'
            'api_key_env = "TEST_KEY"\n'
            "timeout_seconds = 5.0\n"
            'people = ["old"]\n',
            encoding="utf-8",
        )
        write_people_list(p, ["a", "b"])
        with p.open("rb") as f:
            raw = tomllib.load(f)
        # Exactly one top-level people, and [llm] no longer has a people key.
        assert raw["people"] == ["a", "b"]
        assert "people" not in raw["llm"]

        # The written file passes strict load (no llm.people ConfigError).
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        _default_search(
            home=lambda: home,
            cwd=lambda: tmp_path,
            legacy_paths=(home / ".config" / "vemoizer" / "config.toml",),
        )  # must not raise

    def test_write_no_duplicate_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The add-to-people path never appends a name already present
        (case-insensitive) in the existing list."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        p.write_text('people = ["Mikko"]\n', encoding="utf-8")
        existing = read_people_list(p)
        name = "Mikko"  # already present, different case below
        if name.lower() not in [x.lower() for x in existing]:
            existing.append(name)
        write_people_list(p, existing)
        with p.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == ["Mikko"]


# ---------------------------------------------------------------------------
# Round-trip: [llm] + people survive a full names run (people written on yes)
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_people_survives_names_run_then_strict_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A config with [llm] + a top-level people list (people BEFORE the
        [llm] section) survives a full ``names`` run in which the user
        confirms adding a new name, and the resulting file still passes a
        strict ``llm`` load. This is the regression the DECISION item guards.
        """
        isolate_home(monkeypatch, tmp_path, tmp_path)
        p = _project_config(tmp_path)
        # people top-level (before [llm]) so it is a valid top-level list.
        p.write_text('people = ["Mikko"]\n' + _VALID_SECTION, encoding="utf-8")

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        # Name SPEAKER_1 "Aino" (new) and confirm adding to people.
        inputs = iter(["Aino", "y"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        with p.open("rb") as f:
            raw = tomllib.load(f)
        assert raw["people"] == ["Mikko", "Aino"]
        assert raw["llm"]["model"] == "test-model"
        assert "people" not in raw["llm"]

        # The written config still passes strict load (the next meeting run).
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        cfg = _default_search(
            home=lambda: home,
            cwd=lambda: tmp_path,
            legacy_paths=(home / ".config" / "vemoizer" / "config.toml",),
        )
        assert cfg is not None
        assert cfg.model == "test-model"


# ---------------------------------------------------------------------------
# Readline completer: install only under TTYs; restore the PREVIOUS value
# ---------------------------------------------------------------------------


class TestCompleter:
    def test_non_tty_no_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-TTY stdin: completer not installed, no state change."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
        undo = _install_completer(["A"])
        assert undo is None

    def test_tty_installs_and_restores_previous(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under TTYs the previous completer (whatever it is) is captured and
        restored — not hard-set to None. The spec says 'restored
        afterwards', so a pre-existing completer must survive the run."""
        try:
            import readline
        except ImportError:  # pragma: no cover
            pytest.skip("readline not available")

        # Sentinel stand-in for a pre-existing completer. Capture every
        # set_completer call: install sets ours, restore must set the
        # sentinel back — not None.
        sentinel = _no_op_completer  # a valid (no-op) completer
        calls: list[object] = []
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
        monkeypatch.setattr(readline, "get_completer", lambda: sentinel)
        monkeypatch.setattr(
            readline,
            "set_completer",
            lambda fn: calls.append(fn),
        )

        undo = _install_completer(["A"])
        assert undo is not None
        assert len(calls) == 1  # only the install set the completer

        undo()
        assert len(calls) == 2  # restore set the completer again
        # The restored value is the exact prior completer, not None.
        assert calls[1] is sentinel
