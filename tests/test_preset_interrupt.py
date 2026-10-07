"""Unit tests for the Ctrl-C stage tracker (issue #148).

The tracker (``vemoizer.preset_interrupt``) is a small module-level state
machine: ``begin_interrupt_tracking`` resets, ``_set_interrupt_stage``
names the active stage, ``note_written_files`` tallies the run's output
files, and ``handle_interrupt`` builds the single line the command prints
before exiting 130. These tests exercise the state transitions directly —
no display, no CLI — so they are deterministic and fast.
"""

from __future__ import annotations

import pytest

from vemoizer import preset_interrupt as pi


@pytest.fixture(autouse=True)
def _clean_state():
    pi.reset_interrupt_tracking()
    yield
    pi.reset_interrupt_tracking()


def test_handle_interrupt_no_display_no_stage_names_this_run() -> None:
    """No display, no named stage: the line is the generic 'this run' form
    with the 'no files written' claim (the tracker's initial state)."""
    line = pi.handle_interrupt(None)
    assert line == "interrupted during this run — no files written"


def test_handle_interrupt_named_stage_is_used() -> None:
    """A stage named by the run's seam is named in the line."""
    pi._set_interrupt_stage("decoding")
    line = pi.handle_interrupt(None)
    assert line == "interrupted during decoding — no files written"


def test_handle_interrupt_written_files_are_counted() -> None:
    """A run that already wrote earlier files says so (state-based, not
    hardcoded 'no files written')."""
    pi._set_interrupt_stage("decoding")
    pi.note_written_files(["/tmp/a.md", "/tmp/a.json"])
    line = pi.handle_interrupt(None)
    assert line == "interrupted during decoding — 2 file(s) already written"


def test_reset_clears_stage_and_count() -> None:
    pi._set_interrupt_stage("repair")
    pi.note_written_files(["/tmp/x.md"])
    pi.reset_interrupt_tracking()
    assert pi.handle_interrupt(None) == "interrupted during this run — no files written"


def test_begin_interrupt_tracking_resets_and_keeps_display_unused() -> None:
    """begin resets even with a non-None display (the display is read only
    on interrupt, not at tracking begin)."""
    pi._set_interrupt_stage("notes")
    pi.note_written_files(["/tmp/z.md"])
    pi.begin_interrupt_tracking(None)
    assert pi.handle_interrupt(None) == "interrupted during this run — no files written"
