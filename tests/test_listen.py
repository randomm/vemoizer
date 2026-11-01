"""``vemoizer listen`` orchestration with fake sources (issue #173).

No Core Audio or avfoundation: sources come from ``fake_hardware``. The mix
step is a stub unless the test needs real ffmpeg (``needs_ffmpeg``), and
``run_preset`` is patched except in the end-to-end cases, which run the real
preset with a fake ``transcribe_file``.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer
from _cli_helpers import isolate_home
from _listen_helpers import FakeClock, fake_hardware, tone, zeros
from typer.testing import CliRunner

import vemoizer.listen as listen
import vemoizer.pipeline as pipeline_module
import vemoizer.status_line as status_line
from vemoizer.capture import Chunk, SkewLogger
from vemoizer.cli import app
from vemoizer.listen import ListenRun, run_listen
from vemoizer.listen_capture import TrackState
from vemoizer.owner_config import RunOwner
from vemoizer.preset_interrupt import begin_interrupt_tracking
from vemoizer.tracks import fallback_owner

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH"
)
runner = CliRunner()


def _chunk(samples: Any) -> Chunk:
    return Chunk(samples, 0.0)


def _only_call_dir(root: Path) -> Path:
    dirs = sorted((root / "calls").iterdir())
    assert len(dirs) == 1, dirs
    return dirs[0]


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate_home(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def stub_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg is replaced by a stub that writes the output path it was given."""

    def fake_mix(argv: list[str]) -> None:
        Path(argv[-1]).write_bytes(b"mix")

    monkeypatch.setattr(listen, "_run_mix", fake_mix)


@pytest.fixture
def preset_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    import vemoizer.batch as batch

    record: list[dict[str, Any]] = []

    def fake_run_preset(files, **kwargs):
        record.append({"files": files, **kwargs})
        return 0

    monkeypatch.setattr(batch, "run_preset", fake_run_preset)
    return record


def _run(
    clock: FakeClock,
    hardware: Callable[..., Any] | None,
    *,
    sleep: Callable[[float], None] | None = None,
    isatty: bool = False,
    input_fn: Callable[[str], str] | None = None,
    **fields: Any,
) -> int:
    """Run ``run_listen`` with fakes; a Typer exit becomes its exit code."""
    fields.setdefault("duration", 3.0)
    fields.setdefault("repair", False)
    run = ListenRun(**fields)
    try:
        return run_listen(
            run,
            tracker=begin_interrupt_tracking(),
            hardware=hardware,
            clock=clock,
            sleep=sleep or clock.sleep,
            input_fn=input_fn or (lambda _prompt: pytest.fail("no prompt expected")),
            isatty=lambda: isatty,
            output_name=lambda: "MacBook Speakers",
        )
    except typer.Exit as exc:
        return exc.exit_code


def _stop_at(clock: FakeClock, t: float) -> Callable[[float], None]:
    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if clock.t >= t:
            raise KeyboardInterrupt

    return sleep


def _healthy_hw(opened: list[str | None] | None = None) -> Any:
    """Hardware whose mic and system sources both carry 2.5 s of signal."""
    return fake_hardware([_chunk(tone(2.5))], [_chunk(tone(2.5))], opened=opened)


def test_help_lists_listen_flags_and_no_files_argument() -> None:
    text = runner.invoke(app, ["listen", "--help"]).stdout
    for flag in ("--me", "--mic", "--duration", "--speakers", "--language"):
        assert flag in text
    assert "FILES" not in text
    assert "--keep-audio" not in text


def test_run_hands_mix_meeting_tracks_and_flags_to_the_preset(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]]
) -> None:
    opened: list[str | None] = []
    clock = FakeClock()
    code = _run(
        clock,
        _healthy_hw(opened),
        mic="USB Mic",
        me="Janni",
        language="fi",
        speakers="3",
        duration=2.0,
    )
    assert code == 0
    assert opened == ["USB Mic"]
    (call,) = preset_calls
    call_dir = _only_call_dir(home)
    assert call["files"] == [Path("calls") / call_dir.name / "call.flac"]
    assert call["command"] == "meeting"
    assert call["tracks"].me == Path("calls") / call_dir.name / "me.flac"
    assert call["tracks"].them == Path("calls") / call_dir.name / "them.flac"
    assert call["owner"] == RunOwner("Janni", named=True)
    assert call["language"] == "fi"
    assert call["speakers"] == 3
    assert call["yes"] is False


def test_consent_reminder_is_printed_before_any_source_starts(
    home: Path, stub_mix: None, preset_calls, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[str] = []
    open_real = _healthy_hw()

    def hardware(mic_name: str | None):
        seen.append(capsys.readouterr().err)
        return open_real(mic_name)

    _run(FakeClock(), hardware, me="Janni")
    assert seen and "consent" in seen[0].lower()


def test_owner_is_asked_before_recording_and_saved_home(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]]
) -> None:
    events: list[str] = []
    open_real = _healthy_hw()

    def hardware(mic_name: str | None):
        events.append("open")
        return open_real(mic_name)

    def ask(prompt: str) -> str:
        events.append("ask")
        return "Janni"

    code = _run(FakeClock(), hardware, isatty=True, input_fn=ask)
    assert code == 0
    assert events == ["ask", "open"]
    assert preset_calls[0]["owner"] == RunOwner("Janni", named=True)
    saved = (home / "home" / ".vemoizer" / "config.toml").read_text(encoding="utf-8")
    assert 'owner = "Janni"' in saved


def test_non_interactive_run_never_asks_and_uses_the_fallback(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]], capsys
) -> None:
    code = _run(FakeClock(), _healthy_hw(), yes=True)
    assert code == 0
    # No config: the Finnish default label, which is not a name (issue #202).
    assert preset_calls[0]["owner"] == RunOwner(fallback_owner("fi"), named=False)
    assert "--me" in capsys.readouterr().err


def test_catap_missing_exits_2_with_install_hint(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(listen, "catap_installed", lambda: False)
    code = _run(FakeClock(), None, me="Janni")
    err = capsys.readouterr().err
    assert code == 2
    assert "uv sync --extra listen" in err
    assert "Traceback" not in err


def test_silent_microphone_is_a_hard_fail_with_microphone_hint(
    home: Path, stub_mix: None, preset_calls, capsys
) -> None:
    hardware = fake_hardware([_chunk(zeros(2.5))], [_chunk(tone(2.5))])
    code = _run(FakeClock(), hardware, me="Janni")
    err = capsys.readouterr().err
    assert code == 1
    assert preset_calls == []
    assert "Microphone" in err


def test_microphone_that_never_starts_is_not_reported_as_permission(
    home: Path, stub_mix: None, preset_calls, capsys
) -> None:
    code = _run(FakeClock(), fake_hardware([], [_chunk(tone(2.5))]), me="Janni")
    err = capsys.readouterr().err
    assert code == 1
    assert preset_calls == []
    assert "did not start" in err
    assert "no audio within 4 s" in err
    assert "Privacy" not in err


def test_system_audio_that_never_starts_is_a_hard_fail_with_ten_seconds(
    home: Path, stub_mix: None, preset_calls, capsys
) -> None:
    """issue #205: the cold Core Audio tap gets 10 s, not the mic's 4 s."""
    code = _run(FakeClock(), fake_hardware([_chunk(tone(2.5))], []), me="Janni")
    err = capsys.readouterr().err
    assert code == 1
    assert preset_calls == []
    assert "system audio source did not start" in err
    assert "no audio within 10 s" in err


def test_silent_system_audio_warns_and_the_run_continues(
    home: Path, stub_mix: None, preset_calls, capsys
) -> None:
    hardware = fake_hardware([_chunk(tone(2.5))], [_chunk(zeros(2.5))])
    code = _run(FakeClock(), hardware, me="Janni")
    err = capsys.readouterr().err
    assert code == 0
    assert len(preset_calls) == 1
    # issue #205: silence before the call makes sound is normal, not a warning.
    assert "note: system audio is silent so far" in err
    assert "That is fine if nothing is playing yet" in err
    assert "System Audio Recording Only" in err
    assert "warning: system audio" not in err


def test_watchdog_warns_once_after_three_minutes_and_names_the_device(
    home: Path, stub_mix: None, preset_calls, capsys
) -> None:
    code = _run(
        FakeClock(),
        fake_hardware([_chunk(tone(2.5))], [_chunk(tone(2.5))]),
        me="Janni",
        duration=200.0,
    )
    err = capsys.readouterr().err
    assert code == 0
    assert err.count("no sound from the call") == 1
    assert "MacBook Speakers" in err


def test_duration_stops_the_recording(home: Path, stub_mix: None, preset_calls) -> None:
    clock = FakeClock()
    code = _run(clock, _healthy_hw(), me="Janni", duration=5.0)
    assert code == 0
    assert 5.0 <= clock.t <= 6.0
    assert len(preset_calls) == 1


def test_first_ctrl_c_finalizes_the_tracks_and_writes_the_pair(
    home: Path, stub_mix: None, preset_calls
) -> None:
    clock = FakeClock()
    code = _run(
        clock,
        _healthy_hw(),
        sleep=_stop_at(clock, 3.0),
        me="Janni",
        duration=None,
    )
    call_dir = _only_call_dir(home)
    assert code == 0
    assert (call_dir / "me.flac").is_file()
    assert (call_dir / "them.flac").is_file()
    assert len(preset_calls) == 1


def test_second_ctrl_c_during_mix_removes_partial_mix_and_keeps_stems(
    home: Path, monkeypatch: pytest.MonkeyPatch, preset_calls, capsys
) -> None:
    def interrupted_mix(argv: list[str]) -> None:
        Path(argv[-1]).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(listen, "_run_mix", interrupted_mix)
    clock = FakeClock()
    code = _run(clock, _healthy_hw(), sleep=_stop_at(clock, 3.0), me="Janni")
    call_dir = _only_call_dir(home)
    err = capsys.readouterr().err
    assert code == 130
    assert not (call_dir / "call.flac").exists()
    assert (call_dir / "me.flac").is_file()
    assert (call_dir / "them.flac").is_file()
    assert f"audio kept: calls/{call_dir.name}/" in err
    assert preset_calls == []


def test_recording_with_no_frames_skips_transcription_and_removes_folder(
    home: Path, monkeypatch: pytest.MonkeyPatch, preset_calls, capsys
) -> None:
    class DeadWriter:
        position_s = 0.0

        def __init__(self, path, t0, **_: Any) -> None:
            self.path = path

        def write(self, chunk: Chunk) -> bool:
            return False

        def close(self) -> None:
            return None

    monkeypatch.setattr(listen, "TrackWriter", DeadWriter)
    code = _run(FakeClock(), _healthy_hw(), me="Janni")
    assert code == 1
    assert preset_calls == []
    assert not (home / "calls").exists() or not any((home / "calls").iterdir())
    assert "nothing was recorded" in capsys.readouterr().err


def test_recording_runs_under_one_caffeinate_with_the_parent_pid(
    home: Path, stub_mix: None, preset_calls, system_effects
) -> None:
    code = _run(FakeClock(), _healthy_hw(), me="Janni")
    assert code == 0
    (argv,) = system_effects["caffeinate"]
    assert "-w" in argv


@needs_ffmpeg
def test_end_to_end_writes_dated_pair_with_track_sidecar_path(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_transcribe(path, **kwargs):
        seen.append({"path": Path(path), **kwargs})
        return {"text": "moikka", "segments": [], "notes": {"title": "Call"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    clock = FakeClock()
    code = _run(
        clock,
        _healthy_hw(),
        me="Janni",
        no_diarize=True,
    )
    call_dir = _only_call_dir(home)
    assert code == 0
    assert seen[0]["path"] == Path("calls") / call_dir.name / "call.flac"
    assert seen[0]["owner"] == "Janni"
    pair = sorted(p.name for p in home.iterdir() if p.suffix in {".md", ".json"})
    assert len(pair) == 2
    assert pair[0].startswith(call_dir.name[:10])
    sidecar = json.loads(
        (home / next(p for p in pair if p.endswith(".json"))).read_text()
    )
    assert sidecar["source"][0]["path"] == f"calls/{call_dir.name}/call.flac"


@needs_ffmpeg
def test_project_glossary_from_cwd_reaches_transcribe(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glossary_dir = home / ".vemoizer"
    glossary_dir.mkdir()
    (glossary_dir / "glossary.txt").write_text("Pia\n", encoding="utf-8")
    seen: dict[str, str] = {}

    def fake_transcribe(path, **kwargs):
        seen["glossary"] = Path(kwargs["glossary_path"]).read_text(encoding="utf-8")
        return {"text": "moikka", "segments": [], "notes": {"title": "Call"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    code = _run(FakeClock(), _healthy_hw(), me="Janni", no_diarize=True)
    assert code == 0
    assert "Pia" in seen["glossary"]


@needs_ffmpeg
def test_second_ctrl_c_during_decode_keeps_audio_and_names_the_stage(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def interrupted(path, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module, "transcribe_file", interrupted)
    clock = FakeClock()
    code = _run(
        clock,
        _healthy_hw(),
        sleep=_stop_at(clock, 3.0),
        me="Janni",
        no_diarize=True,
    )
    call_dir = _only_call_dir(home)
    err = capsys.readouterr().err
    assert code == 130
    assert "no files written" in err
    assert f"audio kept: calls/{call_dir.name}/" in err
    assert (call_dir / "me.flac").is_file()
    assert (call_dir / "them.flac").is_file()
    assert not list(home.glob("*.md"))


@pytest.fixture
def cli_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``run_listen`` so the command body is tested without recording."""
    seen: dict[str, Any] = {"exit": 0}

    def fake_run_listen(run, *, tracker):
        seen["run"] = run
        if seen["exit"] == "interrupt":
            raise KeyboardInterrupt
        return seen["exit"]

    monkeypatch.setattr(listen, "run_listen", fake_run_listen)
    return seen


def test_cli_maps_every_flag_into_the_run(cli_run: dict[str, Any], home: Path) -> None:
    args = [
        "listen",
        "--mic", "USB Mic",
        "--me", "Janni",
        "--duration", "90",
        "--no-diarize",
        "--no-repair",
        "--language", "FI",
        "--speakers", "3",
        "--yes",
        "--quiet",
    ]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    run = cli_run["run"]
    assert (run.mic, run.me, run.duration) == ("USB Mic", "Janni", 90.0)
    assert (run.no_diarize, run.repair, run.language) == (True, False, "FI")
    assert (run.speakers, run.yes, run.quiet) == ("3", True, True)


def test_cli_exit_code_is_the_run_code(cli_run: dict[str, Any], home: Path) -> None:
    cli_run["exit"] = 1
    assert runner.invoke(app, ["listen"]).exit_code == 1


def test_cli_ctrl_c_before_recording_is_one_clean_line_and_130(
    cli_run: dict[str, Any], home: Path
) -> None:
    cli_run["exit"] = "interrupt"
    result = runner.invoke(app, ["listen"])
    assert result.exit_code == 130
    assert "interrupted during" in result.output
    assert "Traceback" not in result.output


def test_bad_speakers_exit_2_before_any_folder_or_prompt(
    home: Path, stub_mix: None, preset_calls: list[dict[str, Any]]
) -> None:
    code = _run(FakeClock(), _healthy_hw(), speakers="nope")
    assert code == 2
    assert not (home / "calls").exists()
    assert preset_calls == []


def test_skew_is_echoed_at_stop_when_both_tracks_have_audio(
    home: Path, stub_mix: None, preset_calls, capsys: pytest.CaptureFixture[str]
) -> None:
    mic, tap = [Chunk(tone(2.5), 0.0)], [Chunk(tone(2.5), 0.2)]
    code = _run(FakeClock(), fake_hardware(mic, tap), me="Janni", duration=1.0)
    assert code == 0
    assert "track skew: them ahead of me by 200 ms" in capsys.readouterr().err


def test_skew_tick_echoes_once_per_interval_and_stays_quiet_between(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mic, tap = TrackState("mic"), TrackState("tap")
    mic.observe(Chunk(tone(61.0), 0.0))
    tap.observe(Chunk(tone(61.0), 0.3))
    skew = SkewLogger()
    listen._skew_tick(skew, mic, tap, t0=0.0)
    assert "them ahead of me by 300 ms" in capsys.readouterr().err
    listen._skew_tick(skew, mic, tap, t0=0.0)
    assert capsys.readouterr().err == ""


def test_skew_tick_is_silent_until_both_tracks_have_audio(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mic, tap = TrackState("mic"), TrackState("tap")
    tap.observe(Chunk(tone(61.0), 0.0))
    listen._skew_tick(SkewLogger(), mic, tap, t0=0.0)
    assert capsys.readouterr().err == ""


# --- live draft lines on stderr (issue #205) --------------------------------


def test_print_draft_clears_the_rec_line_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    status_line.say("[00:03] Others: moi")
    assert capsys.readouterr().err == "\r\033[K[00:03] Others: moi\n"


def test_print_draft_is_a_plain_line_without_a_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    status_line.say("[00:03] Others: moi")
    assert capsys.readouterr().err == "[00:03] Others: moi\n"


def test_print_draft_waits_while_the_rec_tick_holds_the_lock() -> None:
    """Draft lines (decode thread) and the REC tick (main) share one lock."""
    import threading

    printed = threading.Event()

    def draft() -> None:
        status_line.say("x")
        printed.set()

    with status_line.STATUS_LOCK:
        thread = threading.Thread(target=draft, daemon=True)
        thread.start()
        assert not printed.wait(timeout=0.2)
    assert printed.wait(timeout=5)
    thread.join(timeout=5)
