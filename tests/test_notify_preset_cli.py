"""CLI-level notification contract for the meeting preset paths (issue #100, M4a).

Proves, end-to-end through the Typer CLI (mocked ``pipeline.transcribe_file``
+ grouped-seam fakes; the notification poster ``vemoizer.notify.notify`` is
patched so no real ``osascript`` is ever invoked):

- exactly ONE notification per input file on success — ``meeting`` single
  file, ``meeting`` grouped (one per written group, named by the FIRST
  part's stem), ``meeting`` ``--no-group`` (one per file);
- a partial ``.md``/``.json`` pair (the ``.json`` write fails) is a FAILURE
  notification even though the ``.md`` was written;
- the meeting→dictation fallback (a ``warnings`` channel on the result)
  still produces exactly ONE success notification;
- the failure reason is the one-line reason already printed on stderr by
  the seam (seam (b) for the plain loop, seam (c) for the grouped seam);
- ``--quiet`` never suppresses the notification;
- a notify failure (the poster raises) never changes the run's exit code,
  and on a grouped run the notify call count equals the group count.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _cli_helpers import fake_transcribe, isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.grouping as grouping
import vemoizer.grouping_concat as grouping_concat
import vemoizer.grouping_probe as grouping_probe
import vemoizer.ingest as ingest_module
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


def _fake_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-byte touch-files are not real audio: skip the real ffmpeg
    probe/duration calls (the transcribe seam is faked anyway)."""
    monkeypatch.setattr(grouping_probe, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(
        ingest_module, "pcm_duration_seconds", lambda p, timeout=300.0, **kw: 1.0
    )
    monkeypatch.setattr(grouping_probe, "probe_duration_seconds", lambda p: 1.0)


def _fake_continuation_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files become ONE multi-part group."""
    from vemoizer.grouping import GroupProposal

    _fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None, **kw: (
            ["ja tässä ollaan nyt"],
            ["tässä jatketaan"],
        ),
    )
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files, **kw: [])
    monkeypatch.setattr(grouping, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(grouping_concat, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files, **kw: [])
    monkeypatch.setattr(grouping_concat, "part_offsets", lambda files, **kw: [])
    # One continuation proposal: fold the two files into a single group.
    monkeypatch.setattr(
        grouping,
        "propose_groups",
        lambda files, t, h: [
            GroupProposal(
                parts=(str(files[0].name), str(files[1].name)),
                is_continuation=True,
                evidence=("", ""),
            )
        ],
    )


def _notify_calls(mock_notify) -> list[str]:
    """The message string of each recorded notify call."""
    return [c.args[1] for c in mock_notify.call_args_list]


def _run_cli(monkeypatch, args: list[str]):
    """Invoke the CLI with the notify poster patched; return (mock, result).

    ``--yes`` is appended to meeting invocations with 2+ file args so the
    TTY guard passes under CliRunner (its stdin is not a TTY) — the
    grouped-seam fakes already patch the boundary decodes, so --yes just
    skips the confirmation prompt."""
    meeting_grouped = (
        args[0] == "meeting" and len([a for a in args if not a.startswith("-")]) > 1
    )
    if meeting_grouped and "--yes" not in args and "--no-group" not in args:
        args = list(args) + ["--yes"]
    with patch("vemoizer.notify.notify") as mock_notify:
        result = runner.invoke(app, args)
    return mock_notify, result


# --- meeting single file ------------------------------------------------------


def test_meeting_single_success_one_notify(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert len(record) == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


def test_meeting_single_transcribe_error_key_failure_notify(tmp_path, monkeypatch):
    """A transcribe_file result carrying an ``error`` key (the fail-loud
    config-check path) is a FAILURE notification with the check's reason
    (seam (b))."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": [], "error": "api_key_env is required"}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - api_key_env is required"


def test_meeting_single_empty_transcript_failure_notify(tmp_path, monkeypatch):
    """An empty result (no transcript) is a FAILURE notification; the
    reason is the one-line ``_check_result`` stderr line (seam (b))."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    msg = _notify_calls(mock_notify)[0]
    assert msg.startswith("a: failed")
    assert "no transcript produced" in msg


def test_meeting_single_transcribe_exception_failure_notify(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - a.m4a: boom"


# --- meeting grouped ----------------------------------------------------------


def test_meeting_grouped_one_group_one_notify_first_part_stem(tmp_path, monkeypatch):
    """A 2-part group notifies ONCE (not once per part), named by the
    FIRST part's NFC stem (seam (c) write_group)."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 1
    calls = _notify_calls(mock_notify)
    assert calls == ["a: done"], calls
    assert "a+b" not in calls[0]


def test_meeting_grouped_two_groups_call_count_equals_groups(tmp_path, monkeypatch):
    """A grouped run with 2 single-part groups: notify.call_count == 2
    (one per group) — the exactly-once invariant across the seams."""
    _fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None, **kw: ([""], [""]),
    )
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 2
    assert mock_notify.call_count == 2
    assert sorted(_notify_calls(mock_notify)) == ["a: done", "b: done"]


def test_meeting_grouped_partial_pair_is_failure_notify(tmp_path, monkeypatch):
    """A partial .md/.json pair (the .json write fails) in a grouped run
    is a FAILURE notification even though the .md was written (seam (c))."""
    from vemoizer.batch_output import PRESET_FORMATS, _write_output

    def partial_write(result, first_stem, out_dir, *, date_str=None):
        from vemoizer.batch_output import dated_basename
        from vemoizer.output.naming import collision_free_paths

        base = dated_basename(
            str(result.get("notes", {}).get("title", "")),
            fallback_stem=first_stem,
            date_str=date_str,
        )
        paths = collision_free_paths(
            out_dir, base, [f".{fmt}" for fmt in PRESET_FORMATS]
        )
        written = []
        (out_dir / "block_file").write_text("x", encoding="utf-8")
        for path, fmt in zip(paths, PRESET_FORMATS, strict=True):
            if fmt == "json":
                _write_output(out_dir / "block_file" / path.name, result, fmt)
                continue
            if _write_output(path, result, fmt):
                written.append(path.name)
        (out_dir / "block_file").unlink(missing_ok=True)
        return written

    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr("vemoizer.batch_preset._write_preset_output", partial_write)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - could not write output"


# --- meeting --no-group -------------------------------------------------------


def test_meeting_no_group_two_notifies(tmp_path, monkeypatch):
    """--no-group routes through the plain loop: one success notification
    per file (seam (b) write point)."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["meeting", "a.m4a", "b.m4a", "--no-group"]
    )
    assert result.exit_code == 0
    assert len(record) == 2
    assert sorted(_notify_calls(mock_notify)) == ["a: done", "b: done"]


def test_meeting_no_group_partial_pair_is_failure_notify(tmp_path, monkeypatch):
    """A partial .md/.json pair in the plain loop (--no-group / single) is
    a FAILURE notification (seam (b))."""
    from vemoizer.batch_output import PRESET_FORMATS, _write_output

    def partial_write(result, first_stem, out_dir, *, date_str=None):
        from vemoizer.batch_output import dated_basename
        from vemoizer.output.naming import collision_free_paths

        base = dated_basename(
            str(result.get("notes", {}).get("title", "")),
            fallback_stem=first_stem,
            date_str=date_str,
        )
        paths = collision_free_paths(
            out_dir, base, [f".{fmt}" for fmt in PRESET_FORMATS]
        )
        written = []
        (out_dir / "block_file").write_text("x", encoding="utf-8")
        for path, fmt in zip(paths, PRESET_FORMATS, strict=True):
            if fmt == "json":
                _write_output(out_dir / "block_file" / path.name, result, fmt)
                continue
            if _write_output(path, result, fmt):
                written.append(path.name)
        (out_dir / "block_file").unlink(missing_ok=True)
        return written

    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr("vemoizer.batch_preset._write_preset_output", partial_write)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - could not write output"


# --- meeting→dictation fallback ---------------------------------------------


def test_meeting_fallback_warnings_still_one_success(tmp_path, monkeypatch):
    """The meeting decode falling back to the dictation pipeline appends a
    warning to ``result["warnings"]`` but still produces exactly ONE
    success notification (not two, not a failure)."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Fallback"},
            "warnings": ["meeting decode unavailable, used dictation"],
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert mock_notify.call_count == 1
    msg = _notify_calls(mock_notify)[0]
    assert msg == "a: done"
    assert "failed" not in msg


# --- --quiet independence ------------------------------------------------------


def test_quiet_meeting_still_notifies(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert "wrote " not in result.output
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


def test_quiet_meeting_grouped_still_notifies(tmp_path, monkeypatch):
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes", "--quiet"]
    )
    assert result.exit_code == 0
    assert "wrote " not in result.output
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


# --- notify failure never changes the exit code ------------------------------


def test_notify_failure_grouped_run_exit_code_unchanged(tmp_path, monkeypatch):
    """A poster that raises must not change the run's exit code: a
    successful grouped run still exits 0 and writes its pair."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.notify.notify", side_effect=RuntimeError("osascript dead")):
        result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    mds = sorted(tmp_path.glob("*.md"))
    assert len(mds) == 1, mds
