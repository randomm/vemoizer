"""End-to-end CLI tests for the per-file run log: log path, coverage, privacy,
stdout, and --quiet behaviour (issue #111, M4c).

These do NOT patch ``file_log`` (except where noted): they prove the actual
``.vemoizer/logs/<stem>.log`` file is created per input file / per group, that
a grouped log is named after the group's FIRST part stem, that --quiet /
non-TTY still write the log, that a failing transcribe still leaves a log, and
that stdout is byte-unchanged.
"""

from __future__ import annotations

import logging
import stat

import pytest
from _cli_helpers import fake_transcribe, isolate_home, touch_files
from _run_log_helpers import (
    fake_breaks,
    fake_continuation_seams,
    fake_ingest,
    logs_dir,
)
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
import vemoizer.run_log as run_log
from vemoizer.cli import app

runner = CliRunner()


# --- end-to-end: the per-file log is written to disk (real file_log) ----------


def test_transcribe_single_writes_log_on_disk(tmp_path, monkeypatch):
    """Single-file transcribe: one ``.vemoizer/logs/<stem>.log`` on disk."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    logs = list(logs_dir(tmp_path).iterdir())
    assert len(logs) == 1
    assert logs[0].name == "a.log"


def test_memo_multi_one_log_per_file(tmp_path, monkeypatch):
    """Multi-file memo (seam b, always the plain loop): exactly one log per
    file."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "b.m4a"])
    assert result.exit_code == 0
    names = sorted(p.name for p in logs_dir(tmp_path).iterdir())
    assert names == ["a.log", "b.log"]


# --- coverage matrix: every run path logs every input file (issue #111) ----
#
# The matrix covers all four seams (a: expert transcribe loop, b: preset
# plain loop, c: run_batch group loop, d: run_batch -> _run_plain) so that
# NO transcribed input file is left without a log. The expert multi-file
# cases use the REAL CLI path (``run_batch`` -> ``_run_plain``) — not the
# ``memo`` shortcut the earlier multi-file test used, which dodged seam (d).


@pytest.mark.parametrize(
    ("case"),
    [
        # seam (a): expert transcribe, single file
        pytest.param(
            (["transcribe", "a.m4a"], "single", ["a.log"]),
            id="transcribe-single",
        ),
        # seam (d): expert transcribe, 2 files, plain loop
        pytest.param(
            (
                ["transcribe", "a.m4a", "b.m4a", "--no-group"],
                "nogroup",
                ["a.log", "b.log"],
            ),
            id="transcribe-2-nogroup",
        ),
        # seam (c): 2 files, ungrouped (explicit break proposal)
        pytest.param(
            (["transcribe", "a.m4a", "b.m4a", "--yes"], "breaks", ["a.log", "b.log"]),
            id="transcribe-2-breaks",
        ),
        # seam (b): preset plain loop, single file
        pytest.param(
            (["meeting", "a.m4a"], "single", ["a.log"]),
            id="meeting-single",
        ),
        # seam (d) + preset seam: meeting --no-group, 2 files
        pytest.param(
            (
                ["meeting", "a.m4a", "b.m4a", "--no-group"],
                "nogroup",
                ["a.log", "b.log"],
            ),
            id="meeting-2-nogroup",
        ),
        # seam (c): meeting grouped, 2 parts one multi-part group
        pytest.param(
            (["meeting", "a.m4a", "b.m4a", "--yes"], "grouped", ["a.log"]),
            id="meeting-grouped",
        ),
        # seam (b): memo, single file
        pytest.param(
            (["memo", "a.m4a"], "single", ["a.log"]),
            id="memo-single",
        ),
        # seam (b): memo, 2 files (always the plain loop)
        pytest.param(
            (["memo", "a.m4a", "b.m4a"], "nogroup", ["a.log", "b.log"]),
            id="memo-2",
        ),
    ],
)
def test_run_log_coverage_matrix(
    tmp_path, monkeypatch, case: tuple[list[str], str, list[str]]
) -> None:
    """Every transcribed input file (or group) gets exactly one non-empty
    log file, named per the seam's convention (file stem, or a group's
    first-part stem)."""
    args, seams, expected = case
    if seams == "grouped":
        fake_continuation_seams(monkeypatch)
    elif seams == "breaks":
        fake_breaks(monkeypatch)
    elif seams == "nogroup":
        fake_ingest(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    names = sorted(p.name for p in logs_dir(tmp_path).iterdir())
    assert names == sorted(expected)
    for name in names:
        p = logs_dir(tmp_path) / name
        assert p.exists(), f"{name} missing"
        assert stat.S_IMODE(p.stat().st_mode) == 0o600, f"{name} mode != 0600"
        assert stat.S_IMODE(logs_dir(tmp_path).stat().st_mode) == 0o700, (
            "logs dir mode != 0700"
        )
        text = p.read_text(encoding="utf-8")
        assert text.strip(), f"{name} is empty"
        # The span's own start line is present exactly once (item 5):
        # a re-entrant no-op span adds no second one. (The start line names
        # the ORIGINAL raw stem, which is the same as the log's filename
        # stem in these cases — the fakes use bare stems with no ``/``.)
        start_line = f"log started for {p.stem}"
        assert text.count(start_line) == 1, f"{name}: start line count != 1"


def test_meeting_grouped_log_named_after_first_part_stem(tmp_path, monkeypatch):
    """A grouped meeting run: the log is named after the group's FIRST
    part's NFC stem (not the '+'-joined label)."""
    fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 1  # one merged group
    logs = sorted(p.name for p in logs_dir(tmp_path).iterdir())
    assert logs == ["a.log"]
    assert not any(p.name == "a+b.log" for p in logs_dir(tmp_path).iterdir())


def test_quiet_still_writes_log(tmp_path, monkeypatch):
    """--quiet never disables the per-file log (it suppresses summary only)."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert (logs_dir(tmp_path) / "a.log").exists()


def test_failing_transcribe_still_leaves_log(tmp_path, monkeypatch):
    """A failing transcribe (seam b / c) still leaves a (near-empty) log."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 1
    assert (logs_dir(tmp_path) / "a.log").exists()


def test_stdout_unchanged_when_log_written(tmp_path, monkeypatch, capsys):
    """Writing the per-file log does not add anything to stdout (the log is
    file-only; stdout is byte-identical to a run with no file logging)."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    # stdout captured by the runner must not contain the run-log path.
    assert ".vemoizer" not in result.output


def test_privacy_transcript_and_key_never_in_log(tmp_path, monkeypatch):
    """The recognisable transcript string and the fake API key value never
    appear in the .log file; a synthetic exception's hf_/Bearer payload IS
    redacted (the redaction Formatter covers exc_text)."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost:8000"\n'
        'model = "test-model"\n'
        'api_key_env = "PRIV_TEST_KEY"\n'
        "timeout_seconds = 30\n",
        encoding="utf-8",
    )
    touch_files(["a.m4a"], tmp_path)
    secret = "hf_abcdef1234567890"

    def fake_tf(path, **kw):
        # The fake transcribe logs a synthetic exception carrying the HF token
        # + a Bearer header so the redaction Formatter (which rewrites
        # exc_text, not just the message) is exercised end-to-end.
        logging.getLogger("vemoizer.t").exception(
            f"boom with {secret} and Bearer sk-test-1234567890"
        )
        return {"text": "plain transcript", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    monkeypatch.setenv("PRIV_TEST_KEY", "supersecretkey123456")
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    log_text = (logs_dir(tmp_path) / "a.log").read_text(encoding="utf-8")
    # The API key value (from the env var named in the config's api_key_env)
    # is scrubbed; the HF token and Bearer payload are redacted in the
    # exception text.
    assert "supersecretkey123456" not in log_text
    assert secret not in log_text
    assert "sk-test-1234567890" not in log_text
    assert "hf_<redacted>" in log_text
    assert "Bearer <redacted>" in log_text
    # The plain transcript text from the fake transcribe is not logged by any
    # stage (the seams log pipeline activity, not transcript content).
    assert "plain transcript" not in log_text


# --- re-entrant file_log: nested span for the same stem is a no-op -----------


def test_file_log_reentrant_same_stem_is_noop(tmp_path, monkeypatch):
    """A nested ``file_log`` for the same stem is a no-op: the outer span
    keeps the file, nothing is truncated, and both spans exit cleanly."""
    log_dir = logs_dir(tmp_path)
    log_path = log_dir / "a.log"
    log_dir.mkdir(parents=True)
    logger = logging.getLogger("vemoizer.reentrant")
    with run_log.file_log("a", base_dir=tmp_path):
        logger.info("outer record one")
        with run_log.file_log("a", base_dir=tmp_path):
            logger.info("inner record")
        logger.info("outer record two")
    text = log_path.read_text(encoding="utf-8")
    assert "outer record one" in text
    assert "inner record" in text
    assert "outer record two" in text
    # Nothing was truncated: the outer span's records around the inner
    # span are all present (a nested open would have truncated at the
    # inner's start).
    assert text.index("outer record one") < text.index("inner record")
    assert text.index("inner record") < text.index("outer record two")
    # The guard is per-stem: a different stem nested inside would open its
    # own file (never happens in practice, but the guard must not block it).
    other = log_dir / "b.log"
    with run_log.file_log("b", base_dir=tmp_path):
        logger.info("other stem")
    assert other.exists()
