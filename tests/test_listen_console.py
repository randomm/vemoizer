"""The ``listen`` console voice (issue #236): quiet while recording, clear at stop.

A 38-minute call printed ~110 skew lines and left the owner staring at a
silent terminal after Ctrl-C. These are pure unit tests: the echo is a list,
so no terminal, clock or hardware is involved.
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
    SkewConsole,
    clip_notice,
    clip_notifier,
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
    assert "700 ms" in lines[0]
    assert "warning:" in lines[0]


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


def test_abnormal_line_reports_both_directions() -> None:
    lines, echo = _echo()
    console = SkewConsole(echo=echo)
    console.watch(-(ABNORMAL_SKEW_S + 0.1))
    assert "behind" in lines[0]


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
    assert mic != tap


def test_clip_notice_ends_the_rec_status_line_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The writer calls the notice from its pump thread while ``● REC`` is
    half-drawn; it must not be glued onto the status line."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    clip_notifier("mic")()
    assert capsys.readouterr().err == f"\r\033[K{clip_notice('mic')}\n"


def test_call_status_lines_after_the_interrupt() -> None:
    lines, echo = _echo()
    status = CallStatus(echo=echo)
    status.stopped(37 * 60 + 54, Path("calls/2026-11-04_09-00-00"))
    with status.saving():
        pass
    status.mixing()
    status.transcribing()
    assert lines == [
        "■ Recording stopped at 00:37:54. The recording is saved in "
        "calls/2026-11-04_09-00-00.",
        "Pressing Ctrl-C again cancels the transcription; the recording is kept.",
        "Saving the tracks…",
        "Mixing the tracks…",
        "Transcribing — this can take several minutes for a long call.",
    ]


def test_quiet_drops_the_phase_lines_but_keeps_the_kept_recording_notice() -> None:
    lines, echo = _echo()
    status = CallStatus(quiet=True, echo=echo)
    status.stopped(3.0, Path("calls/x"))
    with status.saving():
        pass
    status.mixing()
    status.transcribing()
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
