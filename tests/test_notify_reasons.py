"""Pinned contract: a failed result check's notification reason equals the
stderr line ``_check_result`` prints (issue #100 decision 2: reason == the
stderr line, leading ``error: `` stripped, one line, capped).

End-to-end through the Typer CLI for each of the three failure conditions
(an ``error`` key, an empty transcript, ``--diarize`` without speaker
labels) in the real seams: the preset plain loop (``meeting`` single),
the grouped meeting seam (``run_batch``), and the expert ``transcribe``
loop. The ``capsys`` stderr line is the source of truth the notification
message must match (minus the ``error: `` prefix the notify helper
strips).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.grouping as grouping
import vemoizer.grouping_concat as grouping_concat
import vemoizer.grouping_probe as grouping_probe
import vemoizer.ingest as ingest_module
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


def _segments_result() -> dict[str, object]:
    """A transcribe result with one speakerless segment (no labels)."""
    return {
        "text": "moikka",
        "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
    }


def _fake_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-byte touch-files are not real audio: skip the real ffmpeg
    probe/duration calls (the transcribe seam is faked anyway)."""
    monkeypatch.setattr(grouping_probe, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(
        ingest_module, "pcm_duration_seconds", lambda p, timeout=300.0: 1.0
    )
    monkeypatch.setattr(grouping_probe, "probe_duration_seconds", lambda p: 1.0)


def _fake_continuation_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files become ONE multi-part group."""
    from vemoizer.grouping import GroupProposal

    _fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None: (["t1"], ["t2"]),
    )
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping_concat, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping_concat, "part_offsets", lambda files: [])
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


def _run_cli(monkeypatch, args: list[str]):
    """Invoke the CLI with the notify poster patched; return (mock, result)."""
    with patch("vemoizer.notify.notify") as mock_notify:
        result = runner.invoke(app, args)
    return mock_notify, result


def _stderr_error_lines(result) -> list[str]:
    """The ``error: `` lines the check printed on stderr."""
    lines = [line for line in result.output.splitlines() if line.startswith("error: ")]
    assert lines, f"no error: line in output: {result.output!r}"
    return lines


def _stderr_error_line(result) -> str:
    """The first ``error: `` line the check printed on stderr."""
    return _stderr_error_lines(result)[0]


def _assert_reason_matches_stderr(
    mock_notify, result, stem: str, index: int = 0
) -> None:
    """The notification carries the stderr line (at *index*) minus ``error: ``."""
    line = _stderr_error_lines(result)[index]
    expected_reason = line.removeprefix("error:").strip()
    calls = [c.args[1] for c in mock_notify.call_args_list]
    assert any(msg == f"{stem}: failed - {expected_reason}" for msg in calls), calls


# --- meeting plain loop (seam b) ---------------------------------------------


def test_meeting_plain_error_key_reason_matches_stderr(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": [], "error": "api_key_env is required"}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")


def test_meeting_plain_empty_transcript_reason_matches_stderr(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")


def test_meeting_plain_diarize_without_labels_reason_matches_stderr(
    tmp_path, monkeypatch
):
    """The `--diarize`-without-labels failure (the case that used to notify
    the generic ``check failed``) now carries the exact stderr line."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return _segments_result()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")
    # The pinned stderr line itself (not just equality with it).
    assert (
        "diarize requested but no speaker labels returned for a.m4a"
        in _stderr_error_line(result)
    )


# --- meeting grouped seam (seam c, via run_batch) -----------------------------


def test_meeting_grouped_diarize_without_labels_reason_matches_stderr(
    tmp_path, monkeypatch
):
    """A 2-part grouped meeting with ``--diarize`` and no speaker labels:
    the grouped seam's failure notification carries the exact stderr line
    (the group's label names its first part)."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return _segments_result()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")
    assert (
        "diarize requested but no speaker labels returned for a.m4a+b.m4a"
        in _stderr_error_line(result)
    )


def test_meeting_grouped_error_key_reason_matches_stderr(tmp_path, monkeypatch):
    _fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping, "decode_boundaries", lambda files, transcribe_fn=None: ([""], [""])
    )
    touch_files(["a.m4a", "b.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": [], "error": "api_key_env is required"}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")


# --- expert transcribe loop (seam a) ------------------------------------------


def test_transcribe_diarize_without_labels_reason_matches_stderr(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return _segments_result()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a", "--diarize"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")
    assert (
        "--diarize requested but no speaker labels returned for a.m4a"
        in _stderr_error_line(result)
    )


def test_transcribe_empty_transcript_reason_matches_stderr(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a"])
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a")


def test_transcribe_no_group_diarize_without_labels_reason_matches_stderr(
    tmp_path, monkeypatch
):
    """The ``--no-group`` multi-file path (``run_batch``'s plain loop,
    expert wording ``--diarize``): reason == stderr line per file."""
    _fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping, "decode_boundaries", lambda files, transcribe_fn=None: ([""], [""])
    )
    touch_files(["a.m4a", "b.m4a"], tmp_path)

    def fake_tf(path, **kw):
        return _segments_result()

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["transcribe", "a.m4a", "b.m4a", "--no-group", "--diarize"]
    )
    assert result.exit_code == 1
    _assert_reason_matches_stderr(mock_notify, result, "a", index=0)
    _assert_reason_matches_stderr(mock_notify, result, "b", index=1)
