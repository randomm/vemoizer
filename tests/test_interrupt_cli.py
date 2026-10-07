"""CLI-level Ctrl-C tests for the ``meeting`` / ``memo`` preset commands
(issue #148).

A Ctrl-C raised inside the run's protected region (the fake transcribe
seam) is neither ``OSError`` nor ``ValueError``, so it propagates to the
command's ``except KeyboardInterrupt`` handler: one line naming the stage
and the written-files state, exit 130, no traceback.

These tests live in a separate file from ``test_meeting_memo_cli.py``
(which is at the 800-line test cap, AGENTS.md) so the new tests do not
push that file over the limit.
"""

from __future__ import annotations

from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.batch as batch
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


def test_meeting_keyboard_interrupt_prints_stage_and_exits_130(
    tmp_path, monkeypatch
) -> None:
    """A Ctrl-C raised inside the fake transcribe seam (the run's protected
    region) prints one line naming the stage and exits 130 with no
    traceback. The tracker is instance-based (created per invocation), so
    no module-level state to reset."""
    touch_files(["a.m4a"], tmp_path)

    def fake_transcribe(path, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 130
    # The single line names the stage and says no files were written.
    assert "interrupted during" in result.stderr
    assert "no files written" in result.stderr
    # No traceback escapes to stderr.
    assert "Traceback" not in result.stderr


def test_memo_keyboard_interrupt_prints_stage_and_exits_130(
    tmp_path, monkeypatch
) -> None:
    """memo: the same contract — one line naming the stage, exit 130, no
    traceback. The memo command has no naming hook, so the interrupt line
    is the only interactive output after the display closes."""
    touch_files(["a.m4a"], tmp_path)

    def fake_transcribe(path, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 130
    assert "interrupted during" in result.stderr
    assert "no files written" in result.stderr
    assert "Traceback" not in result.stderr
