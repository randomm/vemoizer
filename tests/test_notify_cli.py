"""CLI-level notification contract for the transcribe and memo paths (issue #100, M4a).

Proves, end-to-end through the Typer CLI (mocked ``pipeline.transcribe_file``;
the notification poster ``vemoizer.notify.notify`` is patched so no real
``osascript`` is ever invoked):

- exactly ONE notification per input file on success AND on failure —
  ``transcribe`` (single file via the expert loop, multi-file grouped and
  ``--no-group``), ``memo`` (single, multi);
- the notification message is the file's NFC-normalised stem + outcome,
  and on failure the one-line reason already printed on stderr (seam (a));
- ``--quiet`` never suppresses the notification;
- a notification error never changes the run's exit code;
- a ConfigError abort of the whole batch (before any per-file attempt)
  fires NO notification (decision 6).
"""

from __future__ import annotations

from pathlib import Path
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
        lambda files, transcribe_fn=None: (
            ["ja tässä ollaan nyt"],
            ["tässä jatketaan"],
        ),
    )
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping_concat, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping_concat, "part_offsets", lambda files: [])
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
    """Invoke the CLI with the notify poster patched; return (mock, result)."""
    with patch("vemoizer.notify.notify") as mock_notify:
        result = runner.invoke(app, args)
    return mock_notify, result


# --- transcribe (expert) ----------------------------------------------------


def test_transcribe_single_success_one_notify(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


def test_transcribe_single_failure_one_notify_with_reason(tmp_path, monkeypatch):
    """The failure reason is the one-line stderr line from the per-file
    guard (seam (a) transcribe path: ``error: <name>: <exc>``)."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - a.m4a: decoder exploded"


def test_transcribe_multi_no_group_two_notifies(tmp_path, monkeypatch):
    """Multi-file transcribe with --no-group: one success notification per
    file (the expert plain loop, seam (a))."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["transcribe", "a.m4a", "b.m4a", "--no-group"]
    )
    assert result.exit_code == 0
    assert len(record) == 2
    assert sorted(_notify_calls(mock_notify)) == ["a: done", "b: done"]


def test_transcribe_multi_grouped_one_group_one_notify(tmp_path, monkeypatch):
    """A grouped transcribe (2 parts merged) notifies once per group,
    naming the group's first part's stem (seam (a) group loop)."""
    _fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["transcribe", "a.m4a", "b.m4a", "--yes"]
    )
    assert result.exit_code == 0
    assert len(record) == 1  # one decode for the merged group
    calls = _notify_calls(mock_notify)
    assert calls == ["a: done"], calls
    assert "a+b" not in calls[0]


def test_transcribe_grouped_middle_failure_one_fail_notify(tmp_path, monkeypatch):
    """3 files, 3 single-part groups, middle decode raises: exactly one
    failure notification (seam (c) guard) + two successes, exit 1."""
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None: (["", ""], ["", ""]),
    )
    touch_files(["a.m4a", "b.m4a", "c.m4a"], tmp_path)

    def fake_tf(path, **kw):
        if Path(path).name == "b.m4a":
            raise RuntimeError("decoder exploded")
        return {"text": "hei", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(
        monkeypatch, ["transcribe", "a.m4a", "b.m4a", "c.m4a", "--yes"]
    )
    assert result.exit_code == 1
    calls = _notify_calls(mock_notify)
    assert sorted(calls) == [
        "a: done",
        "b: failed - b.m4a: decoder exploded",
        "c: done",
    ], calls


# --- memo -------------------------------------------------------------------


def test_memo_single_success_one_notify(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert len(record) == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


def test_memo_multi_two_notifies(tmp_path, monkeypatch):
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["memo", "a.m4a", "b.m4a"])
    assert result.exit_code == 0
    assert len(record) == 2
    assert sorted(_notify_calls(mock_notify)) == ["a: done", "b: done"]


def test_memo_transcribe_error_is_failure_notify(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["memo", "a.m4a"])
    assert result.exit_code == 1
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: failed - a.m4a: boom"


# --- NFC stems ----------------------------------------------------------------


def test_nfd_filename_notified_with_nfc_stem(tmp_path, monkeypatch):
    """iOS Voice Memos arrive NFD-composed; the notification carries the
    NFC-normalised stem (matching the on-disk naming convention)."""
    nfd_name = "caf\u0065\u0301.m4a"  # "café" decomposed
    touch_files([nfd_name], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", nfd_name])
    assert result.exit_code == 0
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "caf\u00e9: done"


# --- --quiet independence ------------------------------------------------------


def test_quiet_transcribe_still_notifies(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert "wrote transcript" not in result.output
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


def test_quiet_memo_still_notifies(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["memo", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert "wrote " not in result.output
    assert mock_notify.call_count == 1
    assert _notify_calls(mock_notify)[0] == "a: done"


# --- notify failure never changes the exit code ------------------------------


def test_notify_failure_does_not_change_exit_code(tmp_path, monkeypatch):
    """A poster that raises must not change the run's exit code: a
    successful run still exits 0."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.notify.notify", side_effect=RuntimeError("osascript dead")):
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    assert "wrote transcript" in result.output


# --- ConfigError abort fires no notification (decision 6) -------------------


def test_config_error_abort_fires_no_notification(tmp_path, monkeypatch):
    """A malformed project config aborts the whole batch BEFORE any
    per-file attempt — no notification (the notification fires only where
    a per-file attempt actually happened)."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost"\nmodel = "m"\n',
        encoding="utf-8",
    )  # missing api_key_env + timeout_seconds -> strict ConfigError

    def fake_tf(path, **kw):
        raise AssertionError("transcribe must not be attempted on abort")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    mock_notify, result = _run_cli(monkeypatch, ["transcribe", "a.m4a"])
    assert result.exit_code == 1
    assert "api_key_env" in result.output or "api_key_env" in (
        result.stderr if hasattr(result, "stderr") else ""
    )
    # The abort happens before any per-file attempt: no notification.
    assert mock_notify.call_count == 0
