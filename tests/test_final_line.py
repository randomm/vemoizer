"""Tests for the final "complete" line of a meeting/memo run (issue #148).

The word "complete" appears exactly once per run, on the final line,
after the "wrote ..." lines. Per-stage markers use the stage name
("decode ✓", "diarize ✓", ...), so "complete" on the final line is
unambiguous: the run is done and its files are on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


def _two_label_paragraphs():
    return [
        {"start": 0.0, "end": 2.0, "text": "hei", "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "text": "moro", "speaker": "SPEAKER_01"},
    ]


def _fake_transcribe_rich(monkeypatch: pytest.MonkeyPatch, paragraphs) -> None:
    import vemoizer.pipeline as pipeline_module

    def fake(path, **kwargs):
        return {
            "text": " ".join(p["text"] for p in paragraphs),
            "segments": paragraphs,
            "paragraphs": paragraphs,
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake)


def _stub_run_names(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def fake(path, input_fn=None, tty_isatty=None):
        calls.append(path)

    import vemoizer.names_cli as names_cli

    monkeypatch.setattr(names_cli, "run_names", fake)
    return calls


class TestFinalLine:
    def test_success_prints_single_complete_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """On success, the word 'complete' appears exactly once, on the
        final line, after the wrote lines (issue #148 gate resolution)."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a", "--yes"], input="y\n")
        assert result.exit_code == 0
        # The word "complete" appears exactly once, on the final line.
        assert result.stdout.count("complete") == 1
        # It appears after the wrote lines.
        wrote_idx = result.stdout.index("wrote ")
        complete_idx = result.stdout.index("complete")
        assert complete_idx > wrote_idx

    def test_quiet_suppresses_final_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """--quiet suppresses the final 'complete' line."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(
            app, ["meeting", "a.m4a", "--yes", "--quiet"], input="y\n"
        )
        assert result.exit_code == 0
        assert "complete" not in result.stdout
