"""CLI smoke tests for the Typer entry point (issue #10).

Unit tests only: no ffmpeg, no models, no network. The transcribe command
is a placeholder until the full pipeline lands; these tests pin the CLI
surface — flags, exit codes, stdout/stderr separation — so the placeholder
cannot silently regress it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vemoizer.cli import app, main

runner = CliRunner()


def test_help_exits_zero_and_shows_usage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # multi-command Typer app: --help shows the command list
    assert "Usage: vemoizer" in result.stdout
    assert "--help" in result.stdout
    # all preset subcommands are listed
    assert "transcribe" in result.stdout
    assert "meeting" in result.stdout
    assert "memo" in result.stdout
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


def test_profile_flag_is_forwarded(tmp_path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen = {}

    def fake_transcribe(path, **kw):
        seen.update(kw)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["transcribe", "memo.m4a", "--format", "txt", "--profile", "meeting"]
    )
    assert result.exit_code == 0
    assert seen.get("profile") == "meeting"


def _speakers_seen(tmp_path, monkeypatch, value: str):
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe_file(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["transcribe", "a.m4a", "--diarize", "--speakers", value, "--out", "-"]
    )
    return result, seen


def test_speakers_exact_count(tmp_path, monkeypatch) -> None:
    result, seen = _speakers_seen(tmp_path, monkeypatch, "4")
    assert result.exit_code == 0
    assert seen["speakers"] == 4


def test_speakers_range_for_people_joining_and_leaving(tmp_path, monkeypatch) -> None:
    """A meeting where people come and go has no single right count; a
    pinned count too high splits one voice, too low merges two."""
    result, seen = _speakers_seen(tmp_path, monkeypatch, "3-5")
    assert result.exit_code == 0
    assert seen["speakers"] == (3, 5)


def test_speakers_invalid_value_is_rejected_before_transcription(
    tmp_path, monkeypatch
) -> None:
    for bad in ("5-3", "0", "x", "2-"):
        result, seen = _speakers_seen(tmp_path, monkeypatch, bad)
        assert result.exit_code != 0, bad
        assert "speakers" not in seen, bad


# -- fail-loud exit rules (issue #78) ------------------------------------
#
# A total decode failure must not look like success. The CLI checks result
# for "error" and exits 1, but an empty transcript (no "error" key, no text,
# no segments) used to fall through to file writing and exit 0. The two
# rules below must fire before any output file is written, and they must not
# double-report on a single result.


def _invoke_transcribe(tmp_path, monkeypatch, fake, *args: str):
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake)
    monkeypatch.chdir(tmp_path)
    return runner.invoke(app, ["transcribe", *args])


def test_empty_transcript_exits_nonzero_and_writes_no_files(
    tmp_path, monkeypatch
) -> None:
    """An empty result (no text, no segments, no error key) is a failure.

    The check must run before _write_output so no output files are produced
    and the success line is never printed.
    """
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        lambda path, **kw: {"text": "", "segments": []},
        "memo.m4a",
    )
    assert result.exit_code == 1
    # No output files of any default format were written.
    for ext in (".txt", ".json", ".srt", ".vtt", ".md"):
        assert not (tmp_path / f"memo{ext}").exists(), f"unexpected memo{ext}"
    # The success line must not appear.
    assert "wrote transcript" not in result.stdout
    # The failure is reported on stderr.
    assert "no transcript" in result.stderr


def test_silent_audio_no_error_key_still_exits_nonzero(tmp_path, monkeypatch) -> None:
    """Legitimately-empty (silent) audio produces {text:'', segments:[]} with
    no "error" key. The committed simple rule treats that as a failure — it
    is not given a pass, since the pipeline has no silence marker."""
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        # No "error" key: this is the silent-audio shape, not an ingest error.
        lambda path, **kw: {"text": "", "segments": []},
        "silent.m4a",
    )
    assert result.exit_code == 1
    assert not (tmp_path / "silent.txt").exists()


def test_explicit_error_key_still_exits_nonzero_and_no_double_report(
    tmp_path, monkeypatch
) -> None:
    """A result with an "error" key is handled by the existing error branch;
    the empty-transcript rule must not also fire (single error path)."""
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        lambda path, **kw: {"text": "", "segments": [], "error": "boom"},
        "memo.m4a",
    )
    assert result.exit_code == 1
    assert "boom" in result.stderr
    # Exactly one error line: the empty-transcript rule is guarded by the
    # "error"-in-result check and does not double-report.
    assert "no transcript" not in result.stderr


def test_diarize_without_labels_exits_nonzero(tmp_path, monkeypatch) -> None:
    """--diarize requested, segments present, but no "speaker" key on any
    segment -> non-zero exit (labels were promised but never came back)."""
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        lambda path, **kw: {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
        },
        "memo.m4a",
        "--diarize",
    )
    assert result.exit_code == 1
    assert "no speaker labels" in result.stderr
    # No files written — the check fires before _write_output.
    assert not (tmp_path / "memo.txt").exists()


def test_diarize_with_labels_exits_zero(tmp_path, monkeypatch) -> None:
    """Control: --diarize with a "speaker" key on a segment exits 0 — the
    no-labels rule does not fire when labels are actually present."""
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        lambda path, **kw: {
            "text": "moikka",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "moikka", "speaker": "SPEAKER_0"}
            ],
        },
        "memo.m4a",
        "--diarize",
    )
    assert result.exit_code == 0
    assert (tmp_path / "memo.txt").is_file()


def test_diarize_without_labels_does_not_double_report_with_empty(
    tmp_path, monkeypatch
) -> None:
    """A result that is BOTH empty AND has no speaker labels must report
    once, not twice. Empty (no segments) means the empty-transcript rule
    fires; the diarize rule requires segments to be present, so it cannot
    also fire."""
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        lambda path, **kw: {"text": "", "segments": []},
        "memo.m4a",
        "--diarize",
    )
    assert result.exit_code == 1
    # The empty-transcript message fires; the no-labels message does not.
    assert "no transcript" in result.stderr
    assert "no speaker labels" not in result.stderr


def test_empty_then_healthy_batch_continues_and_exits_nonzero(
    tmp_path, monkeypatch
) -> None:
    """Batch continuation: the first file yields an empty result (exit_code
    set to 1 + continue), the second yields real text and IS written. The
    final exit code is 1, matching the existing exit_code=1 + continue
    pattern used for the "error" and write-failure paths."""
    calls: list = []

    def fake(path, **kw):
        calls.append(path.name)
        if path.name == "a.m4a":
            return {"text": "", "segments": []}
        return {"text": "moikka", "segments": []}

    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        fake,
        "a.m4a",
        "b.m4a",
    )
    assert result.exit_code == 1
    # Both files were attempted (the loop continued past the first failure).
    assert calls == ["a.m4a", "b.m4a"]
    # The empty first file wrote nothing.
    assert not (tmp_path / "a.txt").exists()
    # The healthy second file WAS written and its success line printed.
    assert (tmp_path / "b.txt").is_file()
    assert "wrote transcript for b.m4a" in result.stdout


# -- meeting and memo preset commands (issue #82) -----------------------
# The meeting and memo commands share the same batch orchestration; tests
# verify flag forwarding, profile selection, diarization default, and the
# "wrote <path>" output lines.


def test_meeting_help_lists_flags() -> None:
    result = runner.invoke(app, ["meeting", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--config",
        "--glossary",
        "--repair",
        "--no-repair",
        "--diarize",
        "--speakers",
    ):
        assert flag in result.stdout or "--no-diarize" in result.stdout
    assert "files" in result.stdout


def test_memo_help_lists_flags() -> None:
    result = runner.invoke(app, ["memo", "--help"])
    assert result.exit_code == 0
    for flag in ("--config", "--glossary", "--repair", "--no-repair"):
        assert flag in result.stdout
    assert "files" in result.stdout


def test_meeting_forwards_profile_meeting_and_diarize(tmp_path, monkeypatch) -> None:
    """meeting uses profile=meeting and diarize=True by default."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is True
    assert seen["repair"] is True
    assert seen["speakers"] == (2, 6)


def test_meeting_no_diarize_flag(tmp_path, monkeypatch) -> None:
    """--no-diarize disables diarization in the meeting preset."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--no-diarize"])
    assert result.exit_code == 0
    assert seen["diarize"] is False


def test_meeting_speakers_override(tmp_path, monkeypatch) -> None:
    """--speakers overrides the default (2, 6) range."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--speakers", "3-5"])
    assert result.exit_code == 0
    assert seen["speakers"] == (3, 5)


def test_meeting_writes_md_and_json_to_cwd(tmp_path, monkeypatch) -> None:
    """meeting writes .md and .json to the CWD with a dated title."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    # The "wrote" lines reference the dated title base
    assert "wrote" in result.stdout
    # Both .md and .json files exist
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 1, f"expected 1 .md, got {md_files}"
    assert len(json_files) == 1, f"expected 1 .json, got {json_files}"
    # The title appears in the filename
    assert "Team Sync" in md_files[0].name
    assert "Team Sync" in json_files[0].name


def test_meeting_title_fallback_to_first_stem(tmp_path, monkeypatch) -> None:
    """When notes is absent or title is empty, fall back to the first file's stem."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            # No notes key — LLM not configured
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "my-memo.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert "my-memo" in md_files[0].name


def test_memo_forwards_profile_meeting_no_diarize(tmp_path, monkeypatch) -> None:
    """memo uses profile=meeting but diarize=False."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Quick Note"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is False
    assert seen["repair"] is True


def test_memo_writes_md_and_json_to_cwd(tmp_path, monkeypatch) -> None:
    """memo writes .md and .json to the CWD."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Quick Note"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1


def test_meeting_glossary_forwarded(tmp_path, monkeypatch) -> None:
    """--glossary is forwarded to transcribe_file."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--glossary", "/tmp/g.txt"])
    assert result.exit_code == 0
    assert seen["glossary_path"] == "/tmp/g.txt"


def test_meeting_config_forwarded(tmp_path, monkeypatch) -> None:
    """--config is forwarded to transcribe_file."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--config", "/tmp/c.toml"])
    assert result.exit_code == 0
    assert seen["config_path"] == "/tmp/c.toml"


def test_meeting_empty_transcript_exits_nonzero(tmp_path, monkeypatch) -> None:
    """Empty transcript (no text, no segments) is a failure."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "no transcript" in result.stderr


def test_meeting_diarize_without_labels_exits_nonzero(tmp_path, monkeypatch) -> None:
    """meeting with diarize=True but no speaker labels exits non-zero."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "no speaker labels" in result.stderr


def test_meeting_collision_suffix(tmp_path, monkeypatch) -> None:
    """When the output file already exists, a collision suffix is added."""
    import vemoizer.pipeline as pipeline_module
    from vemoizer.output.naming import dated_basename

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    # Pre-create the .md file that the first run would write
    base = dated_basename("Team Sync", fallback_stem="a")
    (tmp_path / f"{base}.md").write_text("existing")
    (tmp_path / f"{base}.json").write_text("{}")
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    # A " (2)" suffix was added
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 2, f"expected 2 .md files, got {md_files}"
    assert any(" (2)" in f.name for f in md_files)


def test_memo_no_diarize_even_with_segments(tmp_path, monkeypatch) -> None:
    """memo never diarizes, even when segments have no speaker labels."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        assert kwargs["diarize"] is False
        return {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0


# -- layered glossary composition (issue #82, DESIGN DECISION) ----------
# run_preset calls glossary_layers.load_layers + merge, writes the
# composed glossary to a temp file (corrections-only for memo, merged
# terms+@+pairs for meeting), passes it through the existing glossary_path
# argument, and deletes the temp file after the run. The fake transcribe
# seam captures the glossary_path it actually received.


def _isolated_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Isolate HOME and CWD so the real .vemoizer layers never leak in."""
    monkeypatch.setenv("HOME", str(home))


def test_meeting_writes_merged_glossary_to_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """meeting: the temp glossary carries merged terms (@ lines) + pairs."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    glossary_content: dict = {}
    tmp_path.mkdir(exist_ok=True)
    # .vemoizer layer in tmp_path — the nearest project layer on the walk-up.
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    (vemoizer_dir / "glossary.txt").write_text(
        "Flagship-hanke\n@Nordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        gp = kwargs.get("glossary_path")
        if gp is not None:
            glossary_content["text"] = Path(gp).read_text(encoding="utf-8")
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    glossary_path = seen["glossary_path"]
    # A temp file was passed (not None, not the --glossary path).
    assert glossary_path is not None
    contents = glossary_content["text"]
    assert "Flagship-hanke" in contents
    assert "@Nordea" in contents
    assert "Blacksit => Flagship" in contents
    # The temp file is deleted after the run.
    assert not Path(glossary_path).exists()


def test_memo_writes_corrections_only_to_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """memo: the temp glossary contains ONLY correction pairs — no terms, no @ lines."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    glossary_content: dict = {}
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    (vemoizer_dir / "glossary.txt").write_text(
        "Flagship-hanke\n@Nordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        gp = kwargs.get("glossary_path")
        if gp is not None:
            glossary_content["text"] = Path(gp).read_text(encoding="utf-8")
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    glossary_path = seen["glossary_path"]
    assert glossary_path is not None
    contents = glossary_content["text"]
    # Corrections-only: the prompt terms must NOT be in the temp file.
    assert "Blacksit => Flagship" in contents
    assert "Flagship-hanke" not in contents
    assert "@Nordea" not in contents
    # The temp file is deleted after the run.
    assert not Path(glossary_path).exists()


def test_meeting_no_glossary_layers_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .vemoizer layers and no --glossary: glossary_path is None."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    # Clean tmp dir with no .vemoizer subdirectory and HOME pointed away.
    clean = tmp_path / "clean"
    clean.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(clean)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert seen["glossary_path"] is None


def test_memo_with_explicit_glossary_passes_path_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--glossary replaces the layers: the exact path is passed through."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    g = tmp_path / "override.txt"
    g.write_text("Nordea\nBlacksit => Flagship\n", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--glossary", str(g)])
    assert result.exit_code == 0
    assert seen["glossary_path"] == str(g)
