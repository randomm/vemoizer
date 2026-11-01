"""``listen --live`` wiring (issue #180): the draft never alters the recording.

Fake sources, fake VAD and whisper, ``_run_mix`` stubbed, ``run_preset``
replaced by a recorder. No Core Audio, no ffmpeg, no models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from _cli_helpers import isolate_home
from _listen_helpers import FakeClock, fake_hardware, tone
from _live_helpers import EnergyVad, ScriptedWhisper
from typer.testing import CliRunner

import vemoizer.listen as listen
from vemoizer.capture import Chunk
from vemoizer.cli import app
from vemoizer.listen import ListenRun, run_listen
from vemoizer.live_session import LiveModels
from vemoizer.preset_interrupt import begin_interrupt_tracking
from vemoizer.vad import VadModelError

runner = CliRunner()
DRAFT = "live.md"


def _healthy_hw() -> Any:
    chunks = [Chunk(tone(2.5), 0.0)]
    return fake_hardware(chunks, [Chunk(tone(2.5), 0.0)])


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate_home(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def stub_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_mix(argv: list[str]) -> None:
        Path(argv[-1]).write_bytes(b"mix")

    monkeypatch.setattr(listen, "_run_mix", fake_mix)


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def preset(
    monkeypatch: pytest.MonkeyPatch, events: list[str], capsys: pytest.CaptureFixture
) -> dict[str, Any]:
    """Records the final pass; snapshots stderr at the moment it starts."""
    import vemoizer.batch as batch

    record: dict[str, Any] = {"code": 0, "stderr_at_start": ""}

    def fake_run_preset(files, **kwargs):
        events.append("run_preset")
        record["stderr_at_start"] = capsys.readouterr().err
        return record["code"]

    monkeypatch.setattr(batch, "run_preset", fake_run_preset)
    return record


def _models(
    events: list[str] | None = None, *, decoder_error: BaseException | None = None
) -> LiveModels:
    whisper = ScriptedWhisper(events=events, error=decoder_error)
    return LiveModels(vad=EnergyVad, transcriber=lambda: whisper)


def _loading_models(clock: FakeClock, cost_s: float = 1.5) -> LiveModels:
    """Each model load costs ``cost_s`` of clock time, as a real load does."""

    def vad() -> EnergyVad:
        clock.t += cost_s
        return EnergyVad()

    def transcriber() -> ScriptedWhisper:
        clock.t += cost_s
        return ScriptedWhisper()

    return LiveModels(vad=vad, transcriber=transcriber)


def _audio_from_now(clock: FakeClock) -> Any:
    """Hardware whose first sample lands at the clock reading when sources open."""

    def open_sources(mic_name: str | None) -> Any:
        now = clock()
        return fake_hardware([Chunk(tone(2.5), now)], [Chunk(tone(2.5), now)])(mic_name)

    return open_sources


def _run(
    run_fields: dict[str, Any],
    *,
    hardware: Any | None = None,
    live_models: LiveModels | None = None,
    clock: FakeClock | None = None,
) -> int:
    clock = FakeClock() if clock is None else clock
    run = ListenRun(duration=2.0, repair=False, yes=True, **run_fields)
    try:
        return run_listen(
            run,
            tracker=begin_interrupt_tracking(),
            hardware=hardware or _healthy_hw(),
            clock=clock,
            sleep=clock.sleep,
            input_fn=lambda _p: "unused",
            isatty=lambda: False,
            output_name=lambda: "MacBook Speakers",
            live_models=live_models,
        )
    except typer.Exit as exc:
        return exc.exit_code


def _call_dir() -> Path:
    (only,) = sorted((Path.cwd() / "calls").iterdir())
    return only


def test_live_failure_never_touches_tracks(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: dict
) -> None:
    def stems(name: str, **kwargs: Any) -> tuple[bytes, bytes]:
        work = tmp_path / name
        work.mkdir()
        monkeypatch.chdir(work)
        assert _run({"me": "Janni", **kwargs.pop("fields", {})}, **kwargs) == 0
        call = _call_dir()
        return (call / "me.flac").read_bytes(), (call / "them.flac").read_bytes()

    off = stems("off", fields={"live": False})
    on = stems("on", fields={"live": True}, live_models=_models())
    broken = stems(
        "broken",
        fields={"live": True},
        live_models=_models(decoder_error=RuntimeError("decoder down")),
    )

    assert off[0] and off[1]
    assert on == off
    assert broken == off


def test_model_load_time_is_not_recording_time(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: dict
) -> None:
    """Loading the models before recording keeps the stems and the preflight clean.

    The loads advance the clock. If they ran after the recording clock started,
    the head pad would hold the load time as silence (the stems would differ from
    ``--live`` off) and preflight's start window would be spent before it began.
    """

    def stems(name: str, live: bool) -> tuple[bytes, bytes]:
        work = tmp_path / name
        work.mkdir()
        monkeypatch.chdir(work)
        clock = FakeClock()
        models = _loading_models(clock) if live else None
        code = _run(
            {"me": "Janni", "live": live},
            hardware=_audio_from_now(clock),
            live_models=models,
            clock=clock,
        )
        assert code == 0
        call = _call_dir()
        return (call / "me.flac").read_bytes(), (call / "them.flac").read_bytes()

    assert stems("on", live=True) == stems("off", live=False)


def test_draft_is_removed_on_success_unless_kept(
    home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, preset: dict
) -> None:
    for name, keep in (("gone", False), ("kept", True)):
        work = tmp_path / name
        work.mkdir()
        monkeypatch.chdir(work)
        code = _run(
            {"me": "Janni", "live": True, "keep_live": keep},
            live_models=_models(),
        )
        assert code == 0
        assert (_call_dir() / DRAFT).exists() is keep


def test_draft_is_kept_when_the_final_pass_fails(
    home: Path, preset: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    preset["code"] = 1
    code = _run({"me": "Janni", "live": True}, live_models=_models())
    assert code == 1
    assert (_call_dir() / DRAFT).exists()


def test_teardown_and_lag_summary_come_before_the_final_pass(
    home: Path, preset: dict, events: list[str], capsys: pytest.CaptureFixture
) -> None:
    code = _run({"me": "Janni", "live": True}, live_models=_models(events))
    assert code == 0
    # One utterance per track (both tracks carry a tone), then teardown, then the
    # final pass.
    assert events[-2:] == ["teardown", "run_preset"]
    assert set(events[:-2]) == {"decode"}
    assert "live draft lag" in preset["stderr_at_start"]


def test_live_without_the_listen_extra_prints_the_install_hint(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(listen, "catap_installed", lambda: False)
    code = run_listen(
        ListenRun(me="Janni", live=True, yes=True),
        tracker=begin_interrupt_tracking(),
        live_models=_models(),
    )
    assert code == 2
    assert "listen extra" in capsys.readouterr().err


def test_model_load_failure_warns_and_the_recording_goes_on(
    home: Path, preset: dict
) -> None:
    def no_model() -> EnergyVad:
        raise VadModelError("silero model missing")

    models = LiveModels(vad=no_model, transcriber=lambda: ScriptedWhisper())
    code = _run({"me": "Janni", "live": True}, live_models=models)

    assert code == 0
    err = preset["stderr_at_start"]
    assert "live draft unavailable: silero model missing" in err
    assert "live draft lag" not in err
    assert not (_call_dir() / DRAFT).exists()


def test_help_lists_the_live_flags() -> None:
    text = runner.invoke(app, ["listen", "--help"]).stdout
    assert "--live" in text
    assert "--keep-live" in text


def test_cli_hands_the_live_flags_to_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ListenRun] = []

    def fake_run_listen(run: ListenRun, **kwargs: Any) -> int:
        seen.append(run)
        return 0

    monkeypatch.setattr(listen, "run_listen", fake_run_listen)
    result = runner.invoke(app, ["listen", "--live", "--keep-live"])
    assert result.exit_code == 0, result.output
    (run,) = seen
    assert run.live is True
    assert run.keep_live is True
