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
    """2 files + --yes: grouping is on by default for 2+ files. The
    mid-sentence boundary is a continuation, so both files land in ONE
    group: transcribe_file runs exactly once (on the concatenated group)
    and the combined output contains the transcript once. CliRunner's
    stdin is not a TTY by default — --yes is what keeps this from
    failing the TTY guard.
    """
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module

    calls: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        calls.append(str(path))
        return {"text": "moikka maailma", "segments": []}

    def fake_decode_boundaries(files, transcribe_fn=None):
        # Mid-sentence tail, no closing cue -> continuation -> one group.
        return ["x"], ["a"]

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr("vemoizer.batch.concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr("vemoizer.batch.part_offsets", lambda files: [])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "transcribe",
            "a.m4a",
            "b.m4a",
            "--format",
            "txt",
            "--quiet",
            "--verbose",
            "--out",
            "-",
            "--diarize",
            "--yes",
        ],
    )
    assert result.exit_code == 0
    # One combined group: exactly ONE decode, and the output contains the
    # transcript exactly once (not twice, not zero times).
    assert len(calls) == 1
    assert result.stdout.count("moikka maailma") == 1


def test_transcribe_no_group_produces_one_transcript_per_file(
    tmp_path, monkeypatch
) -> None:
    """2 files + --no-group: no grouping at all — one standalone transcript
    per file (transcribe_file called once per file), no boundary decode,
    no concat."""
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module

    calls: list[str] = []
    boundary_calls: list[int] = []

    def fake_transcribe_file(path, **kwargs):
        calls.append(str(path))
        return {"text": f"moikka {len(calls)}", "segments": []}

    def fake_decode_boundaries(files, transcribe_fn=None):
        boundary_calls.append(1)
        return [""], [""]

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["transcribe", "a.m4a", "b.m4a", "--out", "-", "--no-group"]
    )
    assert result.exit_code == 0
    # One transcript per file (2), and grouping was skipped entirely.
    assert len(calls) == 2
    assert boundary_calls == []
    assert "moikka 1" in result.stdout
    assert "moikka 2" in result.stdout
    assert result.stdout.count("moikka") == 2


def test_transcribe_non_tty_without_yes_or_no_group_fails_immediately(
    tmp_path, monkeypatch, capsys
) -> None:
    """2+ files, CliRunner's default non-TTY stdin, no --yes/--no-group:
    the CLI must fail IMMEDIATELY with exit 2 and a message naming --yes /
    --no-group, BEFORE any boundary decode, transcriber, or ingest. No real
    model or ffmpeg is touched (transcribe_file faked, decode_boundaries
    faked)."""
    import numpy as np

    import vemoizer.grouping as grouping
    import vemoizer.ingest as ingest
    import vemoizer.pipeline as pipeline_module

    touched: dict[str, int] = {"decode": 0, "transcribe_file": 0, "ingest": 0}

    def fake_decode_boundaries(files, transcribe_fn=None):
        touched["decode"] += 1
        return [""], [""]

    def fake_transcribe_file(path, **kwargs):
        touched["transcribe_file"] += 1
        return {"text": "hei", "segments": []}

    def fake_ingest_audio(path):
        touched["ingest"] += 1
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(ingest, "ingest_audio", fake_ingest_audio)
    monkeypatch.chdir(tmp_path)
    # CliRunner's stdin is not a TTY by default — the guard must fire.
    result = runner.invoke(app, ["transcribe", "a.m4a", "b.m4a", "--out", "-"])
    assert result.exit_code == 2
    assert "--yes" in result.stderr
    assert "--no-group" in result.stderr
    assert "TTY" in result.stderr
    # Nothing downstream ran: no decode, no transcribe, no ingest.
    assert touched == {"decode": 0, "transcribe_file": 0, "ingest": 0}


def test_transcribe_yes_and_no_group_mutually_exclusive(tmp_path, monkeypatch) -> None:
    """2 files with both --yes and --no-group: rejected immediately (exit 2,
    clear message) before any decode or boundary work."""
    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module

    touched: dict[str, int] = {"decode": 0, "transcribe_file": 0}

    def fake_decode_boundaries(files, transcribe_fn=None):
        touched["decode"] += 1
        return [""], [""]

    def fake_transcribe_file(path, **kwargs):
        touched["transcribe_file"] += 1
        return {"text": "hei", "segments": []}

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "b.m4a", "--yes", "--no-group"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.stderr
    assert touched == {"decode": 0, "transcribe_file": 0}


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
    import vemoizer.grouping as grouping

    calls: list = []

    def fake(path, **kw):
        calls.append(path.name)
        if path.name == "a.m4a":
            return {"text": "", "segments": []}
        return {"text": "moikka", "segments": []}

    def fake_decode(files, transcribe_fn=None):
        return ["kiitos ja moi"], ["a"]

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode)
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    result = _invoke_transcribe(
        tmp_path,
        monkeypatch,
        fake,
        "a.m4a",
        "b.m4a",
        "--yes",
    )
    assert result.exit_code == 1
    # Both files were attempted (the loop continued past the first failure).
    assert calls == ["a.m4a", "b.m4a"]
    # The empty first file wrote nothing.
    assert not (tmp_path / "a.txt").exists()
    # The healthy second file WAS written and its success line printed.
    assert (tmp_path / "b.txt").is_file()
    assert "wrote transcript for b.m4a" in result.stdout


# -- M6 quality report wire (issue #75) ----------------------------------
#
# The quality report is computed per file by batch_output BEFORE the
# warnings pop, printed to stdout (suppressed by --quiet), and still
# printed when --format excludes md.


def _report_result() -> dict:
    """A result with a speaker and a warning → a non-empty report."""
    return {
        "text": "moikka maailma",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "moikka maailma", "speaker": "S1"}
        ],
        "paragraphs": [
            {"start": 0.0, "end": 1.0, "text": "moikka maailma", "speaker": "S1"}
        ],
        "warnings": ["diarization failed; continuing without speaker labels"],
    }


def test_report_printed_per_file(tmp_path, monkeypatch) -> None:
    """The quality report is printed to stdout after the wrote line."""
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "transcribe_file", lambda path, **kw: _report_result()
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--format", "txt"])
    assert result.exit_code == 0
    # The report's speakers line is on stdout.
    assert "Puhujat: 1 (S1)" in result.stdout
    # And the warnings classification line too.
    assert "diarization" in result.stdout


def test_report_suppressed_by_quiet(tmp_path, monkeypatch) -> None:
    """--quiet suppresses the report (and the wrote line)."""
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "transcribe_file", lambda path, **kw: _report_result()
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--format", "txt", "--quiet"])
    assert result.exit_code == 0
    assert "Puhujat: 1 (S1)" not in result.stdout
    assert "wrote transcript" not in result.stdout


def test_report_printed_when_format_excludes_md(tmp_path, monkeypatch) -> None:
    """The report is printed even when --format excludes md (txt only)."""
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "transcribe_file", lambda path, **kw: _report_result()
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--format", "txt"])
    assert result.exit_code == 0
    assert "Puhujat: 1 (S1)" in result.stdout


def test_report_computed_before_warnings_pop(tmp_path, monkeypatch) -> None:
    """The report sees the warnings list BEFORE _check_result pops it.

    A result with a diarization warning must produce a report that
    classifies it — if the report were computed after the pop, the
    warnings section would be empty."""
    import vemoizer.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "transcribe_file", lambda path, **kw: _report_result()
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--format", "txt"])
    assert result.exit_code == 0
    # The report's warnings section must be populated (proves the report
    # ran before the pop consumed the list).
    assert "Varoitukset (diarization):" in result.stdout
