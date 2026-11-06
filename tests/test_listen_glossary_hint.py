"""Issue #268: ``listen`` hints at the folder glossary when there is none.

The hint prints once, before recording, only when the layered glossary has
0 terms; recording itself is stubbed (``run_listen`` replaced).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

import vemoizer.listen as listen
from vemoizer.cli import app

runner = CliRunner()

HINT = "no glossary for this folder yet"


@pytest.fixture
def stub_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(listen, "run_listen", lambda run, *, tracker: 0)


def test_hint_shown_with_an_empty_layered_glossary(
    stub_run: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["listen"])
    assert result.exit_code == 0, result.output
    assert HINT in result.output
    assert "./.vemoizer/glossary.txt" in result.output


def test_hint_hidden_when_home_has_terms(
    stub_run: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    home = Path.home() / ".vemoizer"
    home.mkdir(parents=True)
    (home / "glossary.txt").write_text("Yki\n", encoding="utf-8")
    result = runner.invoke(app, ["listen"])
    assert result.exit_code == 0, result.output
    assert HINT not in result.output


def test_hint_hidden_when_project_has_terms(
    stub_run: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    project = tmp_path / ".vemoizer"
    project.mkdir()
    (project / "glossary.txt").write_text("AVS\n", encoding="utf-8")
    result = runner.invoke(app, ["listen"])
    assert result.exit_code == 0, result.output
    assert HINT not in result.output
