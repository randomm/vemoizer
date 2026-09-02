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


def test_transcribe_placeholder_fails_closed() -> None:
    result = runner.invoke(app, ["transcribe", "memo.m4a"])
    assert result.exit_code == 1
    # error/status goes to stderr, stdout stays clean for transcripts
    assert result.stdout == ""
    assert "not implemented" in result.stderr


def test_transcribe_placeholder_with_all_flags() -> None:
    result = runner.invoke(
        app,
        ["transcribe", "a.m4a", "b.m4a", "--format", "txt,srt", "--quiet", "--verbose"],
    )
    # flags are accepted (parsed before the placeholder body runs)
    assert result.exit_code == 1
    assert result.stdout == ""


def test_main_is_callable_entry_point() -> None:
    # main() must wrap the same Typer app the console script uses
    assert callable(main)
