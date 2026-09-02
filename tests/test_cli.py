"""CLI smoke tests for the Typer entry point (issue #10).

Unit tests only: no ffmpeg, no models, no network. The transcribe command
is a placeholder until the full pipeline lands; these tests pin the CLI
surface — flags, exit codes, stdout/stderr separation — so the placeholder
cannot silently regress it.
"""

from __future__ import annotations

from typer.testing import CliRunner

from vemoizer.cli import app, main

runner = CliRunner()


def test_help_exits_zero_and_shows_usage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # multi-command Typer app: --help shows the command list
    assert "Usage: vemoizer" in result.stdout
    assert "--help" in result.stdout
    # both subcommands are listed
    assert "transcribe" in result.stdout
    assert "models" in result.stdout


def test_transcribe_help_lists_flags() -> None:
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    for flag in ("--format", "--quiet", "--verbose"):
        assert flag in result.stdout
    # positional batch input is advertised
    assert "files" in result.stdout
    # documented defaults
    assert "all" in result.stdout
    assert "One or more audio files" in result.stdout


def test_transcribe_missing_file_fails_closed(tmp_path) -> None:
    missing = tmp_path / "no-such-memo.m4a"
    result = runner.invoke(app, ["transcribe", str(missing)])
    assert result.exit_code == 1
    # error/status goes to stderr, stdout stays clean for transcripts
    assert result.stdout == ""
    assert "not found" in result.stderr


def test_transcribe_with_all_flags(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe_file(path, **kwargs):
        return {"text": "moikka maailma", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "transcribe",
            "a.m4a",
            "b.m4a",
            "--format",
            "txt,srt",
            "--quiet",
            "--verbose",
            "--out",
            "-",
        ],
    )
    # flags are accepted and the pipeline result is emitted on stdout
    assert result.exit_code == 0
    assert result.stdout.count("moikka maailma") == 2


def test_main_is_callable_entry_point() -> None:
    # main() must wrap the same Typer app the console script uses
    assert callable(main)
