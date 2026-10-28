"""Live draft session for ``listen --live`` (issue #180) with fake models.

The real StreamingVad, DecodeWorker and LiveWriter run; only the silero model
and whisper are fakes (a 1-means-speech window model and a scripted transcriber).
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from _live_helpers import EnergyVad, ScriptedWhisper

from vemoizer.live_output import Track
from vemoizer.live_session import LiveModels, LiveSession
from vemoizer.vad import VadModelError

SR = 16_000
BLOCK = 320  # 20 ms, the capture callback's block size


def _models(
    transcriber: ScriptedWhisper | None = None, vad: EnergyVad | None = None
) -> LiveModels:
    whisper = transcriber or ScriptedWhisper()
    return LiveModels(vad=lambda: vad or EnergyVad(), transcriber=lambda: whisper)


def _feed(session: LiveSession, track: Track, samples: np.ndarray) -> None:
    feed = session.subscriber(track)
    for start in range(0, len(samples), BLOCK):
        feed(samples[start : start + BLOCK])


def _speech(start_s: float, seconds: float, total_s: float) -> np.ndarray:
    audio = np.zeros(round(total_s * SR), dtype=np.float32)
    a = round(start_s * SR)
    audio[a : a + round(seconds * SR)] = 0.2
    return audio


def _open(
    tmp_path: Path,
    models: LiveModels,
    warn: io.StringIO | None = None,
    print_line: Callable[[str], None] | None = None,
):
    return LiveSession.open(
        tmp_path / "live.md",
        owner="Janni",
        models=models.load(),
        t0=0.0,
        warn=warn or io.StringIO(),
        print_line=print_line,
    )


def test_turns_land_in_the_draft_with_their_capture_times(tmp_path: Path) -> None:
    session = _open(tmp_path, _models())
    _feed(session, "them", _speech(3.0, 1.0, 8.0))
    _feed(session, "me", _speech(6.0, 1.0, 8.0))
    session.stop()
    session.close(remove_draft=False)

    lines = (tmp_path / "live.md").read_text(encoding="utf-8").splitlines()
    assert lines == ["[00:00:02] Others: hi", "[00:00:05] Janni: hi"]


def test_stop_flushes_open_speech_then_drains_before_teardown(tmp_path: Path) -> None:
    events: list[str] = []
    whisper = ScriptedWhisper(events=events)
    session = _open(tmp_path, _models(transcriber=whisper))
    # Speech runs to the end of the stream: only the VAD flush can close it.
    _feed(session, "them", _speech(1.0, 1.0, 2.0))
    report = session.stop()
    session.close(remove_draft=False)

    assert events == ["decode", "teardown"]
    assert report.utterance.count == 1
    assert "[00:00:00] Others: hi" in (tmp_path / "live.md").read_text("utf-8")


def test_stop_is_idempotent_and_late_blocks_are_ignored(tmp_path: Path) -> None:
    session = _open(tmp_path, _models())
    _feed(session, "me", _speech(0.5, 0.5, 1.0))
    first = session.stop()
    _feed(session, "me", _speech(0.5, 0.5, 1.0))
    assert session.stop() is first
    session.close(remove_draft=False)


def test_vad_failure_detaches_that_track_and_never_raises_into_feed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    session = _open(tmp_path, _models(vad=EnergyVad(fail_on_call=3)))
    with caplog.at_level(logging.WARNING, logger="vemoizer.live_session"):
        _feed(session, "them", _speech(0.5, 1.0, 4.0))  # must not raise
        session.stop()
    session.close(remove_draft=False)

    assert "stopped for the them track" in caplog.text
    assert "recording continues" in caplog.text


def test_headphone_bleed_warns_once_on_the_warn_stream(tmp_path: Path) -> None:
    warn = io.StringIO()
    session = _open(tmp_path, _models(), warn=warn)
    _feed(session, "me", _speech(2.0, 1.0, 4.0))
    _feed(session, "them", _speech(2.0, 1.0, 4.0))
    session.stop()
    session.close(remove_draft=False)

    assert warn.getvalue().count("possible headphone bleed") == 1


def test_headphone_bleed_goes_through_the_terminal_sink(tmp_path: Path) -> None:
    """The bleed warning shares the draft's locked sink, not a bare stream."""
    printed: list[str] = []
    warn = io.StringIO()
    session = _open(tmp_path, _models(), warn=warn, print_line=printed.append)
    _feed(session, "me", _speech(2.0, 1.0, 4.0))
    _feed(session, "them", _speech(2.0, 1.0, 4.0))
    session.stop()
    session.close(remove_draft=False)

    assert sum("possible headphone bleed" in line for line in printed) == 1
    assert warn.getvalue() == ""


def test_turns_are_also_streamed_to_the_terminal(tmp_path: Path) -> None:
    """issue #205: each draft line goes to the terminal sink AND the file."""
    printed: list[str] = []
    session = _open(tmp_path, _models(), print_line=printed.append)
    _feed(session, "them", _speech(3.0, 1.0, 8.0))
    _feed(session, "me", _speech(6.0, 1.0, 8.0))
    session.stop()
    session.close(remove_draft=False)

    assert printed == ["[00:02] Others: hi", "[00:05] Janni: hi"]
    # The draft file keeps its own hh:mm:ss rendering, unchanged.
    assert (tmp_path / "live.md").read_text(encoding="utf-8") == (
        "[00:00:02] Others: hi\n[00:00:05] Janni: hi\n"
    )


def test_empty_decodes_are_not_written(tmp_path: Path) -> None:
    session = _open(tmp_path, _models(transcriber=ScriptedWhisper(text="  ")))
    _feed(session, "them", _speech(0.5, 1.0, 3.0))
    session.stop()
    session.close(remove_draft=False)
    assert (tmp_path / "live.md").read_text(encoding="utf-8") == ""


def test_close_removes_the_draft_only_when_asked(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    kept = _open(tmp_path / "a", _models())
    kept.close(remove_draft=False)
    assert (tmp_path / "a" / "live.md").exists()

    removed = _open(tmp_path / "b", _models())
    removed.close(remove_draft=True)
    assert not (tmp_path / "b" / "live.md").exists()


def test_missing_vad_model_raises_before_any_draft_exists(tmp_path: Path) -> None:
    def no_model() -> EnergyVad:
        raise VadModelError("silero model missing")

    models = LiveModels(vad=no_model, transcriber=lambda: ScriptedWhisper())
    with pytest.raises(VadModelError, match="silero model missing"):
        _open(tmp_path, models)
    assert not (tmp_path / "live.md").exists()
