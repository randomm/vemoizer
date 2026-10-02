"""CLI error-path tests for non-UTF-8 ``--glossary`` (issue #79).

A non-UTF-8 glossary file must degrade to one clean stderr line (never a
raw traceback) in both the ``memo`` and ``meeting`` preset commands.
"""

from __future__ import annotations

from pathlib import Path

from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


def test_memo_non_utf8_glossary_clean_error(tmp_path: Path, monkeypatch) -> None:
    """memo --glossary <non-UTF-8>: exit 1, one clean error line, no
    traceback, and no leaked temp file (issue #79)."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\xff\xfe\x00bad")
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--glossary", str(bad)])
    assert result.exit_code == 1
    assert result.stderr.count("error:") == 1
    assert "error: glossary" in result.stderr
    assert "Traceback" not in (result.stderr + result.stdout)
    # No leaked temp glossary file: run_preset's finally must have run.
    assert [p for p in tmp_path.glob("*.txt") if p.name != "bad.bin"] == []


def test_meeting_non_utf8_glossary_no_crash(tmp_path: Path, monkeypatch) -> None:
    """meeting --glossary <non-UTF-8> (single file): the explicit path is
    passed through (read only in the pipeline), so the preset must not
    crash — exit 0 with transcribe mocked, no traceback."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\xff\xfe\x00bad")
    a = tmp_path / "a.m4a"
    a.touch()
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", str(a), "--glossary", str(bad)])
    assert result.exit_code == 0
    assert "Traceback" not in (result.stderr + result.stdout)
