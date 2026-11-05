"""The ``listen`` console voice (issue #236): quiet while recording, clear at stop.

A 38-minute call printed ~110 skew lines and left the owner staring at a
silent terminal after Ctrl-C. These are pure unit tests: the echo is a list
and the clock is a fake, so no terminal or hardware is involved.
"""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from vemoizer.listen_console import (
    ABNORMAL_SKEW_S,
    CallStatus,
    Notices,
    SkewConsole,
    clip_notice,
    drift_notice,
)


def _echo() -> tuple[list[str], Callable[[str], None]]:
    lines: list[str] = []
    return (lines, lines.append)


def test_skew_lines_stay_off_the_console_by_default() -> None:
    lines, echo = _echo()
    console = SkewConsole(echo=echo)
    for _ in range(200):
        console.interval(0.12)
        console.watch(0.12)
    assert lines == []


def test_verbose_echoes_every_interval_line() -> None:
    lines, echo = _echo()
    console = SkewConsole(verbose=True, echo=echo)
    console.interval(0.12)
    console.interval(-0.12)
    assert lines == [
        "track skew: them ahead of me by 120 ms",
        "track skew: them behind me by 120 ms",
    ]


def test_one_abnormal_line_on_the_first_threshold_crossing() -> None:
    lines, echo = _echo()
    console = SkewConsole(echo=echo)
    for _ in range(10):
        console.watch(ABNORMAL_SKEW_S + 0.2)
    assert len(lines) == 1
    assert "700 ms apart" in lines[0]
    assert "note:" in lines[0]
    assert "nothing to do" in lines[0]


def test_the_abnormal_skew_line_says_the_consequence_and_no_action() -> None:
    """The skew is never corrected, so the line must not imply a fix to make."""
    assert drift_notice(0.541) == (
        "note: the two tracks are 541 ms apart (often after an audio device "
        "change); speaker turns may be slightly out of order from here on — "
        "nothing to do"
    )
    assert "drifting" not in drift_notice(0.541)
    lines, echo = _echo()
    SkewConsole(echo=echo, elapsed=lambda: 307.0).watch(0.541)
    assert lines == [f"[00:05:07] {drift_notice(0.541)}"]


def test_abnormal_line_is_not_repeated_until_the_skew_recovers_and_crosses_again() -> (
    None
):
    lines, echo = _echo()
    console = SkewConsole(echo=echo)
    console.watch(1.0)  # crosses
    console.watch(1.1)  # still abnormal: no new line
    console.watch(0.01)  # recovered
    console.watch(0.9)  # crosses again
    assert len(lines) == 2


def test_abnormal_line_reports_the_magnitude_whichever_way_it_drifts() -> None:
    lines, echo = _echo()
    console = SkewConsole(echo=echo)
    console.watch(-(ABNORMAL_SKEW_S + 0.1))
    assert "600 ms apart" in lines[0]


def test_final_skew_line_needs_verbose() -> None:
    quiet_lines, quiet = _echo()
    SkewConsole(echo=quiet).final(0.9)
    assert quiet_lines == []

    lines, echo = _echo()
    SkewConsole(verbose=True, echo=echo).final(0.9)
    assert lines == ["track skew: them ahead of me by 900 ms"]


def test_clip_notice_names_the_source_and_the_fix() -> None:
    mic = clip_notice("mic")
    tap = clip_notice("tap")
    assert "microphone" in mic and "clipping" in mic and "gain" in mic
    assert "call audio" in tap and "clipping" in tap
    assert "call app" in tap and "Teams" in tap and "lower its volume" not in tap
    assert mic != tap


def _notices(elapsed: float = 307.0) -> tuple[list[str], Notices]:
    """The notices as the recorder wires them, ``elapsed`` seconds into the call."""
    lines: list[str] = []
    return lines, Notices(lambda: elapsed, 0.0, echo=lines.append)


def test_every_recording_notice_is_stamped_with_the_rec_clock() -> None:
    """The owner reads a notice against the ``● REC`` clock, so they share one."""
    lines, notices = _notices()
    notices.clip("mic")()
    notices.output_change()("AirPods Pro")
    SkewConsole(echo=lines.append, elapsed=notices.elapsed).watch(0.541)
    assert [line[:11] for line in lines] == ["[00:05:07] "] * 3


def test_the_output_change_notice_names_the_device_and_the_consequence() -> None:
    lines, notices = _notices()
    notices.output_change()("AirPods Pro")
    assert lines == [
        "[00:05:07] the call audio now plays through AirPods Pro; if the call "
        "is still on, use headphones or your mic will pick up the other side"
    ]


def test_a_notice_is_one_line_per_call_the_writer_owns_the_latch() -> None:
    """``TrackWriter`` latches a clip to once per recording and the skew
    console to once per excursion; a notice itself never swallows a call."""
    lines, notices = _notices()
    clip = notices.clip("tap")
    clip()
    clip()
    change = notices.output_change()
    change("AirPods Pro")
    change("MacBook Speakers")
    assert len(lines) == 4
    assert lines[0] == lines[1]  # a repeated clip is a repeated line: no dedupe here
    assert lines[2] != lines[3]  # each device change is announced on its own line


def test_a_notice_the_clock_cannot_answer_is_printed_bare() -> None:
    lines, echo = _echo()
    SkewConsole(echo=echo).watch(0.9)
    assert lines == [drift_notice(0.9)]


def test_a_notice_ends_the_rec_status_line_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The writer calls the notice from its pump thread while ``● REC`` is
    half-drawn; it must not be glued onto the status line."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    Notices(lambda: 307.0, 0.0).clip("mic")()
    assert capsys.readouterr().err == f"\r\033[K[00:05:07] {clip_notice('mic')}\n"


def test_the_output_change_notice_is_its_own_line_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    Notices(lambda: 5.0, 0.0).output_change()("MacBook Speakers")
    assert capsys.readouterr().err == (
        "\r\033[K[00:00:05] the call audio now plays through MacBook Speakers; "
        "if the call is still on, use headphones or your mic will pick up the "
        "other side\n"
    )


def test_call_status_lines_after_the_interrupt() -> None:
    lines, echo = _echo()
    status = CallStatus(echo=echo)
    status.stopped(37 * 60 + 54, Path("calls/2026-11-04_09-00-00"))
    with status.saving():
        pass
    status.mixing()
    status.transcribing(37 * 60 + 54)
    assert lines == [
        "■ Recording stopped at 00:37:54. The recording is saved in "
        "calls/2026-11-04_09-00-00.",
        "Pressing Ctrl-C again cancels the transcription; the recording is kept.",
        "Saving the tracks…",
        "Mixing the tracks…",
        "Transcribing 37 min of audio — this takes a while.",
    ]


def test_the_transcribing_line_names_the_audio_length() -> None:
    """The recorded duration is known before the models load, so the line can
    say what the wait is for instead of "several minutes for a long call"."""
    lines, echo = _echo()
    status = CallStatus(echo=echo)
    status.transcribing(12 * 60 + 4)
    status.transcribing((1 * 60 + 38) * 60)
    status.transcribing(3600)
    assert lines == [
        "Transcribing 12 min of audio — this takes a while.",
        "Transcribing 1 h 38 min of audio — this takes a while.",
        "Transcribing 1 h of audio — this takes a while.",
    ]
    assert not any("ETA" in line or "estimated" in line for line in lines)


def test_quiet_drops_the_phase_lines_but_keeps_the_kept_recording_notice() -> None:
    lines, echo = _echo()
    status = CallStatus(quiet=True, echo=echo)
    status.stopped(3.0, Path("calls/x"))
    with status.saving():
        pass
    status.mixing()
    status.transcribing(60.0)
    # --quiet silences the progress lines exactly as it silences the stage
    # display; what the recording was kept is not progress, it is the one
    # thing the owner needs to hear after Ctrl-C.
    assert len(lines) == 2
    assert "Recording stopped" in lines[0]
    assert all(
        phase not in line
        for line in lines
        for phase in ("Saving", "Mixing", "Transcribing")
    )


def test_saving_ignores_ctrl_c_until_the_stems_are_closed() -> None:
    """A second Ctrl-C inside the save would skip the FLAC writers' close and
    leave the stems unfinalised; it is held off and restored afterwards."""
    before = signal.getsignal(signal.SIGINT)
    lines, echo = _echo()
    with CallStatus(echo=echo).saving():
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) is before
    assert lines == ["Saving the tracks…"]


def test_saving_restores_ctrl_c_when_the_save_raises() -> None:
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(OSError), CallStatus(echo=_echo()[1]).saving():
        raise OSError("disk full")
    assert signal.getsignal(signal.SIGINT) is before


def test_saving_without_announce_is_silent_but_still_guards() -> None:
    lines, echo = _echo()
    with CallStatus(echo=echo).saving(announce=False):
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    assert lines == []


def test_saving_off_the_main_thread_leaves_signals_alone() -> None:
    seen: list[object] = []

    def run() -> None:
        with CallStatus(echo=_echo()[1]).saving():
            seen.append(signal.getsignal(signal.SIGINT))

    before = signal.getsignal(signal.SIGINT)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert seen == [before]
