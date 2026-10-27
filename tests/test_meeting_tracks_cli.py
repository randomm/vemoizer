"""CLI contract for the two-track meeting flags (issue #171).

``--mic-track`` and ``--system-track`` are given together on ``meeting``
only; a bad combination exits 2 before any decode. The fake transcribe
records the kwargs so the tests see what the pipeline would receive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, Any]]:
    """Isolated HOME and cwd; a fake ``transcribe_file`` that records calls."""
    isolate_home(monkeypatch, tmp_path)
    record: list[dict[str, Any]] = []

    def fake_transcribe(path, **kwargs):
        record.append({"path": Path(path).name, **kwargs})
        return {"text": "moikka", "segments": [], "notes": {"title": "Call"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    return record


def _meeting_args(tmp_path: Path, *extra: str, files: int = 1) -> list[str]:
    names = ["call.m4a", "call-me.wav", "call-them.wav"][: files + 2]
    paths = touch_files(names, tmp_path)
    args = ["meeting", str(paths[0])]
    if files == 2:
        args.append(str(touch_files(["second.m4a"], tmp_path)[0]))
    return args + list(extra)


def _track_flags(tmp_path: Path) -> list[str]:
    return [
        "--mic-track",
        str(tmp_path / "call-me.wav"),
        "--system-track",
        str(tmp_path / "call-them.wav"),
    ]


def test_meeting_help_lists_track_flags_and_memo_does_not(
    calls, tmp_path: Path
) -> None:
    meeting_help = runner.invoke(app, ["meeting", "--help"]).stdout
    memo_help = runner.invoke(app, ["memo", "--help"]).stdout
    for flag in ("--mic-track", "--system-track", "--me"):
        assert flag in meeting_help
        assert flag not in memo_help


def test_two_files_with_tracks_exit_2_before_any_decode(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path, *_track_flags(tmp_path), files=2)

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert calls == []


def test_one_track_flag_alone_exits_2(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path, "--mic-track", str(tmp_path / "call-me.wav"))

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "together" in result.output
    assert calls == []


def test_missing_track_file_exits_2_before_decode(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path, *_track_flags(tmp_path))
    (tmp_path / "call-them.wav").unlink()

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "call-them.wav" in result.output
    assert calls == []


def test_speakers_one_with_tracks_exits_2(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path, *_track_flags(tmp_path), "--speakers", "1")

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert calls == []


def test_tracks_and_owner_thread_through_to_transcribe(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path, *_track_flags(tmp_path), "--me", "Janni")

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    tracks = calls[0]["tracks"]
    assert tracks.me == tmp_path / "call-me.wav"
    assert tracks.them == tmp_path / "call-them.wav"
    assert calls[0]["owner"] == "Janni"


def test_without_tracks_transcribe_gets_no_track_stems(calls, tmp_path: Path) -> None:
    args = _meeting_args(tmp_path)

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert calls[0].get("tracks") is None
