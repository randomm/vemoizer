"""Write-back safety for ``vemoizer names`` people config (issue #93).

A people write-back must NEVER destroy a config it cannot parse. Covers
the unparseable-existing-config case (file left byte-identical, one
warning line, run still exits 0), the non-UTF-8 config case, the
missing-config case (still created), and the valid-config-with-other-keys
case (``[llm]`` preserved, name added).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

from vemoizer.names_cli import run_names
from vemoizer.people_config import write_people_list

# --- Helpers ---


def _sidecar(**extra: Any) -> dict[str, Any]:
    """A minimal sidecar with one labelled speaker."""
    return {
        "text": "Puhuttiin asioista.",
        "paragraphs": [
            {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"}
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


def _write_sidecar(
    tmp_path: Path, data: dict[str, Any], name: str = "sidecar.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _config_path(tmp_path: Path) -> Path:
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir(exist_ok=True)
    return vemoizer_dir / "config.toml"


def _run(sc: Path, inputs: list[str] | None = None, input_fn=None, **kw) -> int:
    if input_fn is None and inputs is not None:
        it = iter(inputs)

        def input_fn(p: str) -> str:
            return next(it)

    return run_names(sc, no_play=True, input_fn=input_fn, tty_isatty=lambda: True, **kw)


def _capture_stderr(fn, *args, **kwargs):
    import io
    import sys

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        result = fn(*args, **kwargs)
    finally:
        captured = sys.stderr
        sys.stderr = old_stderr
    return result, captured.getvalue()


def _warning_lines(stderr_text: str) -> list[str]:
    return [
        line
        for line in stderr_text.strip().splitlines()
        if line.startswith("warning: could not update people in")
    ]


# --- write_people_list direct: unparseable existing config ---


class TestWritePeopleListDirect:
    def test_unparseable_existing_config_left_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Unparseable TOML + existing file → file byte-identical, one warning,
        no exception raised."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        broken = "people = ['unclosed\n[llm\n   = \x00"
        config.write_text(broken, encoding="utf-8")
        before = config.read_bytes()

        rc, stderr = _capture_stderr(write_people_list, config, ["Mikko"])

        assert rc is None
        assert config.read_bytes() == before
        warnings = _warning_lines(stderr)
        assert len(warnings) == 1
        assert "TOMLDecodeError" in warnings[0]
        assert "(config left unchanged)" in warnings[0]

    def test_non_utf8_existing_config_left_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Existing config with invalid UTF-8 → byte-identical, one warning."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        raw = b"people = [\xff\xfe\x00broken\n"
        config.write_bytes(raw)

        rc, stderr = _capture_stderr(write_people_list, config, ["Mikko"])

        assert rc is None
        assert config.read_bytes() == raw
        warnings = _warning_lines(stderr)
        assert len(warnings) == 1
        assert "UnicodeDecodeError" in warnings[0]
        assert "(config left unchanged)" in warnings[0]

    def test_missing_config_created_with_people_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No config file → created with just the people list, no warning."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = tmp_path / "isolated" / "config.toml"
        assert not config.exists()

        rc, stderr = _capture_stderr(write_people_list, config, ["Mikko", "Aino"])

        assert rc is None
        assert config.is_file()
        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded == {"people": ["Mikko", "Aino"]}
        assert _warning_lines(stderr) == []

    def test_valid_config_preserves_other_tables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Valid config with an [llm] table → table preserved, people added."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text(
            'base_url = "http://llm.local/v1"\n\n[llm]\nmodel = "gpt-x"\n',
            encoding="utf-8",
        )

        write_people_list(config, ["Mikko"])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded == {
            "base_url": "http://llm.local/v1",
            "llm": {"model": "gpt-x"},
            "people": ["Mikko"],
        }


# --- run_names integration ---


def _unparseable_sidecar_setup(tmp_path: Path, config_content: str) -> Path:
    data = _sidecar()
    data.pop("source", None)
    config = _config_path(tmp_path)
    config.write_text(config_content, encoding="utf-8")
    return _write_sidecar(tmp_path, data)


class TestUnparseableConfigRun:
    def test_unparseable_config_yes_keeps_file_and_persists_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Unparseable existing config + answer yes → config byte-identical,
        one warning line, sidecar still persisted, exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        broken = "people = ['unclosed\n[llm\n"
        sc = _unparseable_sidecar_setup(tmp_path, broken)
        config = _config_path(tmp_path)
        before = config.read_bytes()

        rc, stderr = _capture_stderr(_run, sc, ["Mikko", "y"])

        assert rc == 0
        assert config.read_bytes() == before
        warnings = _warning_lines(stderr)
        assert len(warnings) == 1
        assert "TOMLDecodeError" in warnings[0]
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"

    def test_non_utf8_config_yes_keeps_file_and_persists_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-UTF-8 config + yes → byte-identical, warning, sidecar persisted,
        exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        config = _config_path(tmp_path)
        raw = b"people = [\xff\xfe\x00"
        config.write_bytes(raw)

        rc, stderr = _capture_stderr(_run, sc, ["Mikko", "y"])

        assert rc == 0
        assert config.read_bytes() == raw
        assert len(_warning_lines(stderr)) == 1
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"

    def test_missing_config_yes_creates_with_new_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No config + yes → config created with the new name, no warning."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        config = tmp_path / "isolated" / "config.toml"
        assert not config.exists()

        rc, stderr = _capture_stderr(_run, sc, ["Mikko", "y"], config_path=config)

        assert rc == 0
        assert config.is_file()
        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded["people"] == ["Mikko"]
        assert _warning_lines(stderr) == []
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"

    def test_valid_config_llm_table_preserved_on_yes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Valid config with [llm] + yes → [llm] preserved, name added."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        valid = 'base_url = "http://llm.local/v1"\n\n[llm]\nmodel = "gpt-x"\n'
        sc = _unparseable_sidecar_setup(tmp_path, valid)
        config = _config_path(tmp_path)

        rc, stderr = _capture_stderr(_run, sc, ["Mikko", "y"])

        assert rc == 0
        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded["llm"] == {"model": "gpt-x"}
        assert loaded["base_url"] == "http://llm.local/v1"
        assert loaded["people"] == ["Mikko"]
        assert _warning_lines(stderr) == []
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
