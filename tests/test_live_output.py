"""Tests for the live call draft (issue #179).

The draft is a single-decode preview: turn lines are appended as the call
runs, the file is removed only after the final pair is written, and a
headphone-bleed warning fires once when the mic track echoes the remote side.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from vemoizer.lang_filter import filter_language_lines
from vemoizer.live_output import (
    BleedWarner,
    LiveWriter,
    headphone_bleed,
    live_draft,
    render_line,
    render_terminal_line,
)
from vemoizer.preset_interrupt import InterruptTracker
from vemoizer.tracks import REMOTE_LABEL

# -- line format -----------------------------------------------------------------


def test_render_line_formats_timestamp_and_label() -> None:
    assert render_line(3725.0, "Janni", "moi") == "[01:02:05] Janni: moi"


def test_render_line_keeps_hours_beyond_a_day_digit() -> None:
    assert render_line(36000.0, "Janni", "x") == "[10:00:00] Janni: x"


def test_render_terminal_line_uses_mm_ss() -> None:
    """The streamed terminal line is shorter than the draft file's stamp."""
    assert render_terminal_line(3.0, "Others", "moi") == "[00:03] Others: moi"
    assert render_terminal_line(3599.0, "Janni", "x") == "[59:59] Janni: x"


def test_render_terminal_line_switches_to_hms_past_an_hour() -> None:
    assert render_terminal_line(3600.0, "Janni", "x") == "[01:00:00] Janni: x"


def test_writer_labels_mic_as_owner_and_system_audio_as_remote() -> None:
    out = io.StringIO()
    writer = LiveWriter(out, owner="Janni")

    writer.write_turn(1.0, "me", "hei")
    writer.write_turn(2.5, "them", "moi")

    assert out.getvalue() == (
        f"[00:00:01] Janni: hei\n[00:00:02] {REMOTE_LABEL}: moi\n"
    )


# -- writer keeps its stream -------------------------------------------------------


def test_status_line_not_buffered_by_language_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn reaches the stream captured at construction, not ``sys.stdout``.

    Inside ``filter_language_lines`` the global stdout is a line-holding
    wrapper. The writer must keep writing to the stream it was given; a
    write-time ``sys.stdout`` lookup would send the line into the wrapper.
    """
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    captured = io.StringIO()
    writer = LiveWriter(captured, owner="Janni")

    with filter_language_lines():
        assert sys.stdout is not fake_stdout
        writer.write_turn(1.0, "me", "kesken")
        assert captured.getvalue() == "[00:00:01] Janni: kesken\n"
        assert fake_stdout.getvalue() == ""


# -- draft lifecycle ---------------------------------------------------------------


def test_draft_is_appended_then_removed_after_final_pair(tmp_path: Path) -> None:
    draft = tmp_path / "2026-09-02 call (live).md"

    with live_draft(draft, owner="Janni", keep=False) as writer:
        writer.write_turn(1.0, "me", "hei")
        assert draft.read_text(encoding="utf-8") == "[00:00:01] Janni: hei\n"
        # the final pair is written inside the block; only then is it removed

    assert not draft.exists()


def test_draft_kept_with_keep_live(tmp_path: Path) -> None:
    draft = tmp_path / "call (live).md"

    with live_draft(draft, owner="Janni", keep=True) as writer:
        writer.write_turn(1.0, "them", "moi")

    assert draft.read_text(encoding="utf-8") == f"[00:00:01] {REMOTE_LABEL}: moi\n"


def test_draft_kept_on_interrupt_during_final_pass(tmp_path: Path) -> None:
    draft = tmp_path / "call (live).md"

    with (
        pytest.raises(KeyboardInterrupt),
        live_draft(draft, owner="Janni", keep=False) as writer,
    ):
        writer.write_turn(1.0, "me", "hei")
        raise KeyboardInterrupt

    assert draft.exists()


def test_draft_registered_with_interrupt_tracker(tmp_path: Path) -> None:
    tracker = InterruptTracker()
    draft = tmp_path / "call (live).md"

    with live_draft(draft, owner="Janni", keep=True, tracker=tracker):
        pass

    assert tracker.written_count == 1


# -- headphone bleed ----------------------------------------------------------------


def test_bleed_detected_for_similar_text_within_window() -> None:
    mic = [(10.0, "tämä on testi")]
    them = [(11.0, "tämä on testi")]

    assert headphone_bleed(mic, them) is True


def test_no_bleed_for_dissimilar_text() -> None:
    mic = [(10.0, "tämä on testi")]
    them = [(11.0, "aivan muuta puhetta kokonaan")]

    assert headphone_bleed(mic, them) is False


def test_no_bleed_when_similar_text_is_far_apart() -> None:
    mic = [(10.0, "tämä on testi")]
    them = [(30.0, "tämä on testi")]

    assert headphone_bleed(mic, them) is False


def test_empty_text_is_never_bleed() -> None:
    assert headphone_bleed([(1.0, "")], [(1.0, "")]) is False


def test_punctuation_only_lines_are_never_bleed() -> None:
    assert headphone_bleed([(1.0, "...")], [(1.5, "?")]) is False


def test_bleed_warner_warns_once() -> None:
    lines: list[str] = []
    warner = BleedWarner(lines.append)
    mic = [(10.0, "tämä on testi")]
    them = [(11.0, "tämä on testi")]

    assert warner.check(mic, them) is True
    assert warner.check(mic, them) is False

    assert sum("bleed" in line for line in lines) == 1
