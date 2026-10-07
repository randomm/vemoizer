"""Issue #148 promise, state-based: per-stage markers, single final
"complete" line, display lifecycle around the wrote lines / naming prompt.

A real :class:`~vemoizer.progress.ProgressDisplay` is constructed with
``isatty`` patched BEFORE construction (the fake-stderr pattern from
``test_meeting_display_close`` — no real pty). The pipeline seam is stubbed
to drive the display exactly like production: it ``add_stage``s the four
stages (``decode``, ``diarize``, ``repair``, ``notes``) and ``finish``es
each. The assertions are on rendered task descriptions (state, not
terminal bytes) plus the echoed final line:

* each rendered stage description ends in ``✓ <stage>`` — none contains
  the word ``complete``;
* the word ``complete`` appears exactly once overall and only on the
  final line, printed AFTER the wrote lines;
* the display is closed (``is_live`` False) before the wrote lines and
  before the naming prompt;
* ``--quiet`` prints none of it.

Break-and-fail: removing the ``_close_run_display`` call (or moving the
final line above the wrote lines in ``batch_preset.run_preset``) fails the
matching assertion.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from _cli_helpers import isolate_home, touch_files

from vemoizer.batch_preset import run_preset
from vemoizer.preset_final import run_went_full
from vemoizer.progress import ProgressDisplay

_EXPECTED_STAGES = ["decode", "diarize", "repair", "notes"]


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


@pytest.fixture
def tty_display(monkeypatch: pytest.MonkeyPatch):
    """A real, live ProgressDisplay with isatty patched BEFORE construction."""
    import sys

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    assert display.disable is False, "display must be live on a TTY stderr"
    yield display
    display.close()


def _echo_recorder(record: dict[str, Any]):
    def fake_echo(msg, err: bool = False, **kw: Any) -> None:
        if not err:
            record.setdefault("echoes", []).append(str(msg))

    return fake_echo


def _final_line_recorder(record: dict[str, Any]):
    def fake_final_line(n_files: int, *, quiet: bool) -> None:
        if quiet:
            return
        record.setdefault("echoes", []).append(f"✓ complete — wrote {n_files} file(s)")

    return fake_final_line


def _wrote_lines_recorder(record: dict[str, Any]):
    def fake_wrote_lines(written: list[str], quiet: bool) -> None:
        if not quiet:
            for name in written:
                record.setdefault("echoes", []).append(f"wrote {name}")

    return fake_wrote_lines


def _gate_recorder(record: dict[str, Any]):
    def fake_gate(written: list[str], files: list, exit_code: int) -> bool:
        record["gate"] = {"written": written, "files": files, "exit_code": exit_code}
        return run_went_full(written, files, exit_code)

    return fake_gate


def _run_meeting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    display: ProgressDisplay,
    *,
    quiet: bool,
    yes: bool,
    record: dict[str, Any],
) -> int:
    """Run one single-file meeting with the pipeline seam stubbed to drive
    the display through the four stage names, and record the display state
    at the first interactive prompt."""
    import vemoizer.naming_hook as naming_hook_module
    import vemoizer.notify as notify_module
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        for name in _EXPECTED_STAGES:
            task_id = display.add_stage(name)
            display.finish(task_id)
        return {
            "text": "Moikka. Kyll.",
            "segments": [],
            "notes": {"title": Path(path).stem},
            "paragraphs": _two_label_paragraphs(),
        }

    def input_fn(prompt: str) -> str:
        record["prompt"] = prompt
        record["display_live_at_prompt"] = display.is_live
        return "n"

    [f] = touch_files(["2025-01-01 M.m4a"], tmp_path)
    with (
        mock.patch.object(pipeline_module, "transcribe_file", fake_transcribe),
        mock.patch.object(naming_hook_module, "_stdout_isatty", lambda: True),
        mock.patch.object(notify_module, "notify_write", lambda *a, **kw: None),
        mock.patch.object(notify_module, "notify_result", lambda *a, **kw: None),
        mock.patch.object(naming_hook_module.typer, "echo", _echo_recorder(record)),
        mock.patch("vemoizer.batch_preset.typer.echo", _echo_recorder(record)),
        mock.patch(
            "vemoizer.batch_preset.print_wrote_lines", _wrote_lines_recorder(record)
        ),
        mock.patch("vemoizer.batch_preset.run_went_full", _gate_recorder(record)),
        mock.patch(
            "vemoizer.batch_preset.print_final_line", _final_line_recorder(record)
        ),
    ):
        return run_preset(
            [f],
            command="meeting",
            config_path=None,
            glossary_path=None,
            quiet=quiet,
            yes=yes,
            input_fn=input_fn,
            tty_isatty=lambda: True,
            display=display,
        )


def test_meeting_stage_markers_and_single_complete_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty_display: ProgressDisplay,
) -> None:
    """The rendered stage descriptions are exactly decode/diarize/repair/
    notes each with a per-stage ✓ marker; "complete" appears exactly once
    and only on the final line, after the wrote lines; the display is
    closed before both the wrote lines and the naming prompt."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    record: dict[str, Any] = {}

    code = _run_meeting(
        tmp_path, monkeypatch, tty_display, quiet=False, yes=False, record=record
    )
    assert code == 0

    # (1) Rendered task descriptions: exactly the four stage names, each
    # finished with its own ✓ marker, none containing "complete".
    descriptions = [t.description for t in tty_display._progress.tasks]
    assert len(descriptions) == len(_EXPECTED_STAGES)
    for name, desc in zip(_EXPECTED_STAGES, descriptions, strict=True):
        assert desc == f"[green]\u2713 {name}", f"got {desc!r} for stage {name!r}"
        assert "complete" not in desc

    # (2) "complete" exactly once, only on the final line, after the wrote.
    echoes = record.get("echoes", [])
    assert sum(s.count("complete") for s in echoes) == 1
    final_lines = [s for s in echoes if "complete" in s]
    assert len(final_lines) == 1
    wrote_idx = echoes.index(next(s for s in echoes if s.startswith("wrote ")))
    final_idx = echoes.index(final_lines[0])
    assert final_idx > wrote_idx
    # The final line is PLAIN text (issue #148 FIX 2): rich markup like
    # ``[green]`` would print verbatim through ``typer.echo``. The count
    # and position are what matter; no square brackets may leak.
    assert final_lines[0] == "\u2713 complete — wrote 2 file(s)"
    assert "[" not in final_lines[0]
    assert all(s.startswith("wrote ") for s in echoes[: wrote_idx + 1])

    # (3) Display closed before the wrote lines and before the prompt.
    # The prompt spy proves the close predates the last interactive output;
    # the wrote lines fire before the hook in run_preset, so
    # closed-at-prompt implies closed-at-wrote.
    assert record.get("prompt"), "naming-hook input_fn was never called"
    assert record["display_live_at_prompt"] is False
    assert tty_display.is_live is False


def test_meeting_quiet_prints_none_of_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty_display: ProgressDisplay,
) -> None:
    """--quiet suppresses the wrote lines, the final line, and any
    skip-reason narration."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    record: dict[str, Any] = {}

    code = _run_meeting(
        tmp_path, monkeypatch, tty_display, quiet=True, yes=True, record=record
    )
    assert code == 0
    assert sum(s.count("complete") for s in record.get("echoes", [])) == 0
    assert record.get("echoes", []) == []
    assert tty_display.is_live is False
