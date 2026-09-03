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
    for flag in ("--format", "--quiet", "--verbose", "--diarize"):
        assert flag in result.stdout
    # positional batch input is advertised
    assert "files" in result.stdout
    # documented defaults
    assert "all" in result.stdout
    assert "One or more audio files" in result.stdout


def test_transcribe_missing_file_fails_closed(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe_file(path, **kwargs):
        return {"text": "", "segments": [], "error": f"{path} not found"}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
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
            "--diarize",
        ],
    )
    # flags are accepted and the pipeline result is emitted on stdout
    assert result.exit_code == 0
    assert result.stdout.count("moikka maailma") == 2


def test_transcribe_diarize_default_off(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe_file(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--out", "-"])
    assert result.exit_code == 0
    # --diarize defaults OFF: diarize is passed explicitly as False
    assert seen.get("diarize") is False


def test_transcribe_diarize_flag_passed_through(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe_file(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--diarize", "--out", "-"])
    assert result.exit_code == 0
    assert seen.get("diarize") is True


def test_main_is_callable_entry_point() -> None:
    # main() must wrap the same Typer app the console script uses
    assert callable(main)


# -- format handling (issue #49) -----------------------------------------
#
# The default invocation used to crash: --format defaults to "all", the
# format list was used verbatim, and FORMAT_EXTENSIONS["all"] raised
# KeyError -- after the full multi-minute transcription had completed.


def test_default_format_all_writes_every_extension(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {"text": "moikka", "segments": []},
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "memo.m4a"])
    assert result.exit_code == 0
    for ext in (".txt", ".json", ".srt", ".vtt", ".md"):
        assert (tmp_path / f"memo{ext}").is_file(), f"missing memo{ext}"


def test_unknown_format_rejected_before_transcription(tmp_path, monkeypatch) -> None:
    """An invalid --format must fail fast, not after minutes of decoding."""
    import vemoizer.pipeline as pipeline_module

    def _must_not_run(path, **kw):
        raise AssertionError("transcribe_file ran despite an invalid --format")

    monkeypatch.setattr(pipeline_module, "transcribe_file", _must_not_run)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "memo.m4a", "--format", "txt,docx"])
    assert result.exit_code == 2
    assert "docx" in result.stderr


def test_write_failure_exits_nonzero_without_success_line(
    tmp_path, monkeypatch
) -> None:
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {"text": "moikka", "segments": []},
    )
    monkeypatch.chdir(tmp_path)
    # An unwritable target: the stem collides with an existing directory.
    blocker = tmp_path / "memo.txt"
    blocker.mkdir()
    result = runner.invoke(app, ["transcribe", "memo.m4a", "--format", "txt"])
    assert result.exit_code != 0
    assert "wrote transcript" not in result.stdout


# -- polish (issue #59) --------------------------------------------------


def test_short_flags_q_and_v_are_accepted(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {"text": "moikka", "segments": []},
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["transcribe", "memo.m4a", "-q", "-v", "--format", "txt"]
    )
    assert result.exit_code == 0
    assert "wrote transcript" not in result.stdout  # -q suppressed it


def test_config_flag_is_forwarded_to_the_pipeline(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen = {}

    def fake_transcribe(path, **kw):
        seen.update(kw)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["transcribe", "memo.m4a", "--format", "txt", "--config", "/tmp/x.toml"],
    )
    assert result.exit_code == 0
    assert seen.get("config_path") == "/tmp/x.toml"


def test_out_with_multiple_formats_warns(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {"text": "moikka", "segments": []},
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["transcribe", "memo.m4a", "--format", "txt,json", "--out", "o.txt"],
    )
    assert result.exit_code == 0
    assert "only the first" in result.stderr
