"""One run log per ``listen`` call (issue #254).

Recording and transcribing a call is one run of work, so it gets one log,
named after the call folder, from the first tick of the recorder to the last
line of the preset. The corollary is the console contract: with no root
handler during recording, ``logging.lastResort`` printed raw ``log.warning``
text glued onto the ``● REC`` line — a root handler now exists, so it does
not. Capture comes from the ``listen`` fakes; only the transcription test
reaches the real preset (with a faked ``transcribe_file``).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer
from _cli_helpers import fake_transcribe, isolate_home
from _listen_helpers import FakeClock, fake_hardware, tone
from _run_log_helpers import logs_dir

import vemoizer.listen as listen
from vemoizer.capture import Chunk
from vemoizer.listen import ListenRun, run_listen
from vemoizer.preset_interrupt import begin_interrupt_tracking
from vemoizer.run_log_io import QuietFileHandler

#: The raw clipping line ``capture.TrackWriter`` logs while recording.
CLIP_LINE = "input exceeds +/-1.0"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH"
)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate_home(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def stub_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_mix(argv: list[str]) -> None:
        Path(argv[-1]).write_bytes(b"mix")

    monkeypatch.setattr(listen, "_run_mix", fake_mix)


@pytest.fixture
def preset_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The preset is stubbed: these tests are about the log, not the decode."""
    import vemoizer.batch as batch

    record: list[dict[str, Any]] = []

    def fake_run_preset(files, **kwargs):
        record.append({"files": files, **kwargs})
        return 0

    monkeypatch.setattr(batch, "run_preset", fake_run_preset)
    return record


class _NamedDates:
    """A ``datetime`` stand-in: one frozen call-folder name per ``now()``."""

    def __init__(self, names: list[str]) -> None:
        self._names = list(names)
        self._taken = 0

    def now(self) -> _NamedNow:
        self._taken += 1
        return _NamedNow(self._names[self._taken - 1])


class _NamedNow:
    def __init__(self, name: str) -> None:
        self._name = name

    def strftime(self, _fmt: str) -> str:
        return self._name


def _chunk(samples: Any) -> Chunk:
    return Chunk(samples, 0.0)


def _tone_hw(value: float = 0.2) -> Any:
    """Hardware whose mic and system sources both carry 2.5 s of signal."""
    chunks = [_chunk(tone(2.5, value)), _chunk(tone(2.5, value))]
    return fake_hardware(chunks, list(chunks))


def _run(clock: FakeClock, hardware: Callable[..., Any], **fields: Any) -> int:
    """``run_listen`` with fakes; a Typer exit becomes its exit code."""
    fields.setdefault("duration", 3.0)
    fields.setdefault("repair", False)
    try:
        return run_listen(
            ListenRun(me="Janni", **fields),
            tracker=begin_interrupt_tracking(),
            hardware=hardware,
            clock=clock,
            sleep=clock.sleep,
            input_fn=lambda _prompt: pytest.fail("no prompt expected"),
            isatty=lambda: False,
            output_name=lambda: "MacBook Speakers",
        )
    except typer.Exit as exc:
        return exc.exit_code


def _only_log(home: Path) -> Path:
    logs = list(logs_dir(home).iterdir())
    assert len(logs) == 1, logs
    return logs[0]


def _only_call_dir(home: Path) -> Path:
    dirs = sorted((home / "calls").iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_a_recording_warning_goes_to_the_call_log_and_not_to_stderr(
    home: Path,
    stub_mix: None,
    preset_calls: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The clipping warning is logged on a pump thread while recording.

    With no root handler it reached stderr through ``lastResort``, welded to
    the ``● REC`` line; now it lands in the call's log and the console only
    gets the actionable line ``listen`` writes itself.
    """
    assert _run(FakeClock(), _tone_hw(2.0)) == 0
    call_dir = _only_call_dir(home)
    text = _only_log(home).read_text(encoding="utf-8")
    err = capsys.readouterr().err
    assert text.splitlines()[0].endswith(f"log started for {call_dir.name}")
    assert CLIP_LINE in text
    assert CLIP_LINE not in err
    assert "microphone is too loud" in err


def test_the_call_log_is_named_after_the_call_folder(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]]
) -> None:
    assert _run(FakeClock(), _tone_hw()) == 0
    call_dir = _only_call_dir(home)
    assert _only_log(home).name == f"{call_dir.name}.log"


def test_the_recording_runs_with_the_call_log_handler_on_the_root_logger(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]]
) -> None:
    """What keeps a raw ``log.warning`` off the terminal while recording.

    ``logging.lastResort`` prints a WARNING when the whole logger tree has no
    handler — which was the case for the whole recording before issue #254.
    The call's file handler is attached to the root logger for the duration,
    so the last-resort path cannot be reached. The probe runs at the capture
    seam, i.e. inside the recording phase.
    """
    seen: list[list[logging.Handler]] = []
    real = _tone_hw(2.0)

    def hardware(mic_name: str | None):
        seen.append(list(logging.getLogger().handlers))
        return real(mic_name)

    assert _run(FakeClock(), hardware) == 0
    (handlers,) = seen
    opened = [h for h in handlers if isinstance(h, QuietFileHandler)]
    assert len(opened) == 1
    call_dir = _only_call_dir(home)
    assert opened[0].baseFilename.endswith(f"{call_dir.name}.log")


def test_two_calls_write_two_logs(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_mix: None,
    preset_calls: list[dict[str, Any]],
) -> None:
    """The second call must not truncate the first call's log."""
    monkeypatch.setattr(
        listen,
        "datetime",
        _NamedDates(["2026-07-30_10-00-00", "2026-07-30_11-00-00"]),
    )
    assert _run(FakeClock(), _tone_hw()) == 0
    assert _run(FakeClock(), _tone_hw()) == 0
    names = sorted(p.name for p in logs_dir(home).iterdir())
    assert names == ["2026-07-30_10-00-00.log", "2026-07-30_11-00-00.log"]


@needs_ffmpeg
def test_the_transcription_writes_into_the_call_log_and_no_call_log_appears(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preset's per-file span for ``call.flac`` adopts the call's log."""
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    assert _run(FakeClock(), _tone_hw(), no_diarize=True) == 0
    call_dir = _only_call_dir(home)
    assert record == ["call.flac"]
    text = _only_log(home).read_text(encoding="utf-8")
    assert _only_log(home).name == f"{call_dir.name}.log"
    assert "log started for" in text
    # The stage line comes from the fake transcribe inside run_preset's span.
    assert "transcribe: call.flac" in text
    assert not (logs_dir(home) / "call.log").exists()
