"""Unit tests for the Ctrl-C stage tracker (issue #148).

The tracker (``vemoizer.preset_interrupt``) is one
:class:`vemoizer.preset_interrupt.InterruptTracker` instance per
invocation: ``begin_interrupt_tracking`` builds it, ``set_stage`` names
the active stage, ``note_written_files`` tallies the run's output files,
and ``handle_interrupt`` builds the single line the command prints before
exiting 130. These tests exercise the state transitions directly — no
display, no CLI — so they are deterministic and fast.
"""

from __future__ import annotations

import pytest

from vemoizer import preset_interrupt as pi


@pytest.fixture
def tracker() -> pi.InterruptTracker:
    """A fresh tracker per test (no state leaks across invocations)."""
    return pi.begin_interrupt_tracking()


def test_handle_interrupt_fresh_tracker_names_this_run(
    tracker: pi.InterruptTracker,
) -> None:
    """A fresh tracker with no display: the line is the generic 'this run'
    form with the 'no files written' claim (the tracker's initial state)."""
    line = pi.handle_interrupt(tracker, None)
    assert line == "interrupted during this run — no files written"


def test_handle_interrupt_named_stage_is_used(
    tracker: pi.InterruptTracker,
) -> None:
    """A stage named by the run's seam is named in the line."""
    tracker.set_stage("decoding")
    line = pi.handle_interrupt(tracker, None)
    assert line == "interrupted during decoding — no files written"


def test_handle_interrupt_written_files_are_counted(
    tracker: pi.InterruptTracker,
) -> None:
    """A run that already wrote earlier files says so (state-based, not
    hardcoded 'no files written')."""
    tracker.set_stage("decoding")
    tracker.note_written_files(["/tmp/a.md", "/tmp/a.json"])
    line = pi.handle_interrupt(tracker, None)
    assert line == "interrupted during decoding — 2 file(s) already written"


def test_begin_interrupt_tracking_builds_a_fresh_tracker() -> None:
    """Each begin builds a clean tracker: a second invocation in the same
    process never sees the first invocation's state."""
    first = pi.begin_interrupt_tracking()
    first.set_stage("notes")
    first.note_written_files(["/tmp/z.md"])
    second = pi.begin_interrupt_tracking()
    assert second is not first
    line = pi.handle_interrupt(second, None)
    assert line == "interrupted during this run — no files written"
    # The first tracker's state is unaffected (it is instance state now).
    assert pi.handle_interrupt(first, None) == (
        "interrupted during notes — 1 file(s) already written"
    )


def test_note_written_files_helper_tallies_the_tracker() -> None:
    """The module-level helper tallies on the given tracker instance."""
    tracker = pi.begin_interrupt_tracking()
    pi.note_written_files(tracker, ["/tmp/a.md", "/tmp/a.json"])
    assert tracker.written_count == 2
    line = pi.handle_interrupt(tracker, None)
    assert "2 file(s) already written" in line


def test_none_tracker_falls_back_to_no_files_claim() -> None:
    """handle_interrupt tolerates a missing tracker (defensive: a None
    tracker reports 'no files written' and a bare stage name)."""
    line = pi.handle_interrupt(None, None)
    assert line == "interrupted during this run — no files written"


# -- Ctrl-C through the CLI (issue #148) ------------------------------------
#
# A Ctrl-C raised inside the run's protected region (the fake transcribe
# seam) is neither ``OSError`` nor ``ValueError``, so it propagates to the
# command's ``except KeyboardInterrupt`` handler: one line naming the stage
# and the written-files state, exit 130, no traceback. The tracker (set
# via ``set_interrupt_stage`` in ``batch_preset``) names the stage; the
# display's active task is the more specific source when available.


@pytest.mark.parametrize("command", ["meeting", "memo"])
def test_preset_keyboard_interrupt_prints_stage_and_exits_130(
    tmp_path, monkeypatch, command: str
) -> None:
    """Both preset commands: one line naming the stage, exit 130, no
    traceback, and the state-based 'no files written' claim (the interrupt
    fires in the first file's transcribe, before any output is written)."""
    from _cli_helpers import isolate_home, touch_files
    from typer.testing import CliRunner

    import vemoizer.batch as batch
    import vemoizer.batch_preset as batch_preset
    from vemoizer.cli import app

    touch_files(["a.m4a"], tmp_path)

    def fake_transcribe(file, options=None, glossary_path=None, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(batch_preset, "_transcribe_preset_file", fake_transcribe)
    monkeypatch.setattr(batch, "_resolve_llm_config", lambda p: None)
    # The preflight gate runs before the transcribe seam; pass it so the
    # run reaches the fake transcribe (and the display is live).
    import vemoizer.preflight as preflight

    monkeypatch.setattr(preflight, "ffmpeg_ok", lambda: True)
    monkeypatch.setattr(preflight, "config_parse_ok", lambda: True)
    monkeypatch.setattr(preflight, "models_cached", lambda: [])
    monkeypatch.setattr(preflight, "hf_token_present", lambda: True)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = CliRunner().invoke(app, [command, "a.m4a"])
    assert result.exit_code == 130
    assert "interrupted during" in result.stderr
    assert "no files written" in result.stderr
    # No traceback escapes to stderr.
    assert "Traceback" not in result.stderr
